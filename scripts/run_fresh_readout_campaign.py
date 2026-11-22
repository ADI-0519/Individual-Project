"""Run the frozen fresh-readout campaign sequentially with no automatic retries.

Both models must already be prepared. This coordinator never changes scientific
inputs, selects methods, inspects partial test scores, or resumes an old attempt.
Use --detach to leave the same fixed worker running on native Linux storage.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from latent_probes.config import ModelSettings
from latent_probes.fresh_readout_capture import (
    BLOCKS,
    COUNTS,
    MODELS,
    _authorization,
    panel_sites,
    snapshot_files,
    verify_protocol,
)
from latent_probes.fresh_readout_review import validate_review_receipt
from latent_probes.storage import atomic_json, canonical_hash, file_sha256, read_json, utc_now

ROOT = Path(__file__).resolve().parents[1]
MODELS_ORDER = ("llama", "qwen")
FIT_FILES = ("settings.json", "heads.npz", "candidates.json", "manifest.json")
EXTRA_SOURCES = (
    "scripts/run_fresh_readout_campaign.py",
    "scripts/build_fresh_readout_bank.py",
    "scripts/analyze_fresh_readout.py",
    "src/latent_probes/fresh_readout_analysis.py",
    "docs/FRESH_READOUT_EXECUTION_PROTOCOL_V2.md",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def fresh_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def environment():
    return {
        **os.environ,
        "HF_HOME": str(ROOT / ".cache/huggingface-wsl"),
        "HF_HUB_CACHE": "/home/adirj/kcl-hf-cache",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "PYTHONUNBUFFERED": "1",
    }


def merge_bindings(*maps):
    result = {}
    for mapping in maps:
        for path, digest in mapping.items():
            path = str(Path(path).resolve())
            require(
                path not in result or result[path] == digest, "Conflicting source binding: " + path
            )
            result[path] = digest
    snapshot_files(result)
    return result


def preflight(output, review_receipt, source_bindings=None):
    output, review_receipt = Path(output).resolve(), Path(review_receipt).resolve()
    require(
        platform.system() == "Linux"
        and not str(output).startswith("/mnt/")
        and "onedrive" not in str(output).lower(),
        "Campaign requires native Linux storage",
    )
    require(
        output.is_dir() and not output.is_relative_to(ROOT), "Use an existing native prepared root"
    )
    protocols, sources, inventories = {}, {}, []
    for key in MODELS_ORDER:
        directory = output / key
        path = directory / "protocol.json"
        protocol = read_json(path)
        preparation = read_json(directory / "preparation.json")
        require(
            protocol.get("test_fixture") is False
            and protocol.get("model_key") == key
            and preparation.get("status") == "prepared_no_weights"
            and preparation.get("protocol_sha256") == file_sha256(path),
            "Need genuine unchanged prepared protocols for both models",
        )
        verify_protocol(protocol)
        expected = ModelSettings(
            model_id=MODELS[key]["model_id"],
            revision=MODELS[key]["revision"],
            dtype="bfloat16",
            quantization="4bit",
            device="cuda",
            capture_layers=list(BLOCKS[key]),
            max_input_tokens=2048,
            local_files_only=True,
        )
        require(
            ModelSettings(**protocol["model_settings"]) == expected
            and protocol["sites"] == panel_sites(key),
            "Pinned model/capture policy differs",
        )
        rows = protocol["planned_rows"]
        require(
            len(rows) == 7040 and Counter(r["partition"] for r in rows) == COUNTS,
            "Prepared inventory differs from the fixed 7040 rows",
        )
        require(protocol.get("engineering_row") is not None, "Missing excluded engineering row")
        inventories.append(
            canonical_hash(
                [
                    {
                        name: r[name]
                        for name in ("row_id", "partition", "messages_sha256", "state", "gold")
                    }
                    for r in rows
                ]
            )
        )
        for attempted in (*COUNTS, "source", "training", "engineering", "execution.lock"):
            require(
                not (directory / attempted).exists(),
                "Prepared model has a prior attempt: " + attempted,
            )
        protocols[key] = protocol
        sources = merge_bindings(
            sources,
            protocol["source_sha256"],
            {
                str(path): file_sha256(path),
                str(directory / "preparation.json"): file_sha256(directory / "preparation.json"),
                protocol["bank_path"]: protocol["bank_file_sha256"],
                protocol["forward_contract"]["source_path"]: protocol["forward_contract"][
                    "source_sha256"
                ],
            },
        )
    require(
        len(set(inventories)) == 1
        and len({p["bank_file_sha256"] for p in protocols.values()}) == 1,
        "Both model preparations must have the same exact ordered scientific inventory",
    )
    bank_path = Path(protocols["llama"]["bank_path"])
    review = validate_review_receipt(review_receipt, bank_path)
    review_files = {str(review_receipt): file_sha256(review_receipt)}
    for binding in review["source_files"].values():
        path = (ROOT / binding["path"]).resolve()
        review_files[str(path)] = binding["sha256"]
    sources = merge_bindings(sources, {str(ROOT / p): file_sha256(ROOT / p) for p in EXTRA_SOURCES})
    if source_bindings is not None:
        source_bindings = Path(source_bindings).resolve()
        sources = merge_bindings(
            sources,
            read_json(source_bindings),
            {str(source_bindings): file_sha256(source_bindings)},
        )
    snapshot_files(review_files)
    return {
        "output": str(output),
        "bank_path": str(bank_path),
        "protocols": protocols,
        "source_sha256": sources,
        "review_artifact_sha256": review_files,
        "inventory_sha256": inventories[0],
    }


def command(script, *args):
    return [sys.executable, str(ROOT / "scripts" / script), *(str(arg) for arg in args)]


def build_plan(config):
    output, bank = Path(config["output"]), config["bank_path"]
    stages = []
    for key in MODELS_ORDER:
        directory = output / key
        stages.append(
            {
                "name": key + "_training",
                "kind": "command",
                "command": command(
                    "run_fresh_readout_capture.py",
                    "execute",
                    "--output",
                    directory,
                    "--partition",
                    "training",
                    "--bindings",
                    output / (key + "_training_bindings.json"),
                ),
                "outputs": [
                    str(directory / "training" / n)
                    for n in ("features.npz", "rows.json", "manifest.json")
                ],
                "progress_manifest": str(directory / "training/manifest.json"),
                "cohort_rows": 2048,
            }
        )
    for key in MODELS_ORDER:
        stages.append(
            {
                "name": "fit_" + key,
                "kind": "command",
                "command": command(
                    "fit_fresh_readout.py",
                    "fit",
                    "--capture",
                    output / key / "training",
                    "--bank",
                    bank,
                    "--protocol",
                    output / key / "protocol.json",
                    "--output",
                    output / (key + "_fit"),
                ),
                "outputs": [str(output / (key + "_fit") / n) for n in FIT_FILES],
            }
        )
    stages.append(
        {
            "name": "freeze_both_fits",
            "kind": "command",
            "command": command(
                "fit_fresh_readout.py",
                "freeze",
                "--llama-fit",
                output / "llama_fit",
                "--qwen-fit",
                output / "qwen_fit",
                "--protocols",
                output / "llama/protocol.json",
                output / "qwen/protocol.json",
                "--out",
                output / "test_fit_freeze.json",
            ),
            "outputs": [str(output / "test_fit_freeze.json")],
        }
    )
    stages.append(
        {
            "name": "bind_both_tests",
            "kind": "bindings",
            "outputs": [str(output / (key + "_test_bindings.json")) for key in MODELS_ORDER],
        }
    )
    for key in MODELS_ORDER:
        directory = output / key
        stages.append(
            {
                "name": key + "_test",
                "kind": "command",
                "command": command(
                    "run_fresh_readout_capture.py",
                    "execute",
                    "--output",
                    directory,
                    "--partition",
                    "test",
                    "--bindings",
                    output / (key + "_test_bindings.json"),
                ),
                "outputs": [
                    str(directory / "test" / n)
                    for n in ("features.npz", "rows.json", "manifest.json")
                ],
                "progress_manifest": str(directory / "test/manifest.json"),
                "cohort_rows": 4992,
            }
        )
    for key in MODELS_ORDER:
        stages.append(
            {
                "name": "analyze_" + key,
                "kind": "command",
                "command": command(
                    "analyze_fresh_readout.py",
                    "--test-capture",
                    output / key / "test",
                    "--fit",
                    output / (key + "_fit"),
                    "--protocol",
                    output / key / "protocol.json",
                    "--bank",
                    bank,
                    "--seal",
                    output / "test_fit_freeze.json",
                    "--output",
                    output / (key + "_analysis"),
                ),
                "outputs": [
                    str(output / (key + "_analysis") / n)
                    for n in ("results.json", "predictions.npz", "manifest.json")
                ],
            }
        )
    return stages


def write_bindings(config, *, test=False):
    output = Path(config["output"])
    sources = config["source_sha256"]
    seal = None
    if test:
        seal_path = output / "test_fit_freeze.json"
        seal = read_json(seal_path)
        require(
            seal.get("schema") == "fresh-readout-fit-seal-v2"
            and seal.get("status") == "frozen"
            and seal.get("test_fixture") is False
            and len(seal.get("fitted_artifact_sha256", {})) == 8
            and seal["model_protocol_sha256"]
            == {key: file_sha256(output / key / "protocol.json") for key in MODELS_ORDER},
            "Need both genuine frozen fit bundles before test bindings",
        )
        snapshot_files(seal["fitted_artifact_sha256"])
        sources = merge_bindings(
            sources, seal["source_sha256"], {str(seal_path): file_sha256(seal_path)}
        )
    paths = {}
    for key in MODELS_ORDER:
        payload = {
            "schema": "fresh-readout-execution-bindings-v1",
            "status": "frozen",
            "research_status": "agent_audited_exploratory",
            "human_review_complete": False,
            "protocol_sha256": file_sha256(output / key / "protocol.json"),
            "source_sha256": sources,
            "review_artifact_sha256": config["review_artifact_sha256"],
        }
        if seal is not None:
            payload.update(
                {name: seal[name] for name in ("fitted_artifact_sha256", "model_protocol_sha256")}
            )
        path = output / f"{key}_{'test' if test else 'training'}_bindings.json"
        if path.exists():
            require(read_json(path) == payload, "Existing execution bindings differ; no overwrite")
        else:
            fresh_json(path, payload)
        _authorization(
            config["protocols"][key],
            payload["protocol_sha256"],
            path,
            "test" if test else "training",
        )
        paths[str(path)] = file_sha256(path)
    return paths


def archive_sources(output, bindings):
    destination = Path(output) / "campaign_source_archive"
    destination.mkdir(exist_ok=False)
    manifest = {}
    for index, (name, digest) in enumerate(sorted(bindings.items())):
        archived = destination / f"{index:04d}_{Path(name).name}"
        shutil.copyfile(name, archived)
        require(file_sha256(archived) == digest, "Source changed during archive: " + name)
        manifest[name] = {"sha256": digest, "archived_path": str(archived)}
    fresh_json(destination / "manifest.json", manifest)
    return str(destination / "manifest.json")


def run_child(arguments, stdout_path, stderr_path):
    with Path(stdout_path).open("xb") as stdout, Path(stderr_path).open("xb") as stderr:
        result = subprocess.run(
            arguments,
            cwd=ROOT,
            env=environment(),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            shell=False,
            check=False,
        )
    return result.returncode


def run_campaign(config, *, child=run_child, binder=write_bindings):
    output = Path(config["output"])
    lock, state_path = output / "campaign.lock", output / "campaign_state.json"
    require(not state_path.exists(), "Campaign already attempted; no retry or resume")
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(utc_now() + "\n")
    except FileExistsError as error:
        raise ValueError("Active or unknown campaign lock; preserve and inspect") from error
    state = {
        "schema": "fresh-readout-campaign-v1",
        "status": "starting",
        "created_at": utc_now(),
        "research_status": "agent_audited_exploratory",
        "human_review_complete": False,
        "cohort_prefills_planned": 14080,
        "excluded_engineering_prefills_planned": 2,
        "classifier_candidate_fits_per_model": 85,
        "coordinator_sha256": file_sha256(__file__),
        "inventory_sha256": config["inventory_sha256"],
        "stages": [],
    }
    try:
        fresh_json(state_path, state)
        plan = build_plan(config)
        fresh_json(output / "campaign_plan.json", plan)
        for stage in plan:
            for path in stage["outputs"]:
                require(
                    not Path(path).exists(),
                    "Prior stage output exists; no overwrite/retry: " + path,
                )
        snapshot_files(config["source_sha256"])
        snapshot_files(config["review_artifact_sha256"])
        training_bindings = binder(config, test=False)
        bindings = merge_bindings(
            config["source_sha256"], config["review_artifact_sha256"], training_bindings
        )
        state["source_archive_manifest"] = archive_sources(output, bindings)
        state["source_sha256"] = bindings
        logs = output / "campaign_logs"
        logs.mkdir(exist_ok=False)
        state["status"] = "running"
        atomic_json(state_path, state)
        for index, stage in enumerate(plan):
            snapshot_files(bindings)
            entry = {
                "name": stage["name"],
                "status": "running",
                "started_at": utc_now(),
                **(
                    {
                        "progress_manifest": stage["progress_manifest"],
                        "cohort_rows": stage["cohort_rows"],
                    }
                    if "progress_manifest" in stage
                    else {}
                ),
            }
            state["stages"].append(entry)
            atomic_json(state_path, state)
            print(f"[{utc_now()}] Stage {index + 1}/{len(plan)}: {stage['name']}", flush=True)
            if stage["kind"] == "bindings":
                new_bindings = binder(config, test=True)
                bindings = merge_bindings(bindings, new_bindings)
                bindings = merge_bindings(
                    bindings,
                    read_json(output / "test_fit_freeze.json")["fitted_artifact_sha256"],
                    {
                        str(output / "test_fit_freeze.json"): file_sha256(
                            output / "test_fit_freeze.json"
                        )
                    },
                )
            else:
                stdout, stderr = (
                    logs / (stage["name"] + ".stdout.log"),
                    logs / (stage["name"] + ".stderr.log"),
                )
                entry.update(
                    stdout_path=str(stdout), stderr_path=str(stderr), command=stage["command"]
                )
                atomic_json(state_path, state)
                code = child(stage["command"], stdout, stderr)
                entry.update(
                    returncode=code,
                    stdout_sha256=file_sha256(stdout),
                    stderr_sha256=file_sha256(stderr),
                )
                require(
                    code == 0,
                    f"Stage {stage['name']} failed with exit {code}; preserve logs and stop",
                )
            require(
                all(Path(path).is_file() for path in stage["outputs"]),
                "Stage returned without its complete output files: " + stage["name"],
            )
            entry.update(
                status="complete",
                finished_at=utc_now(),
                output_sha256={path: file_sha256(path) for path in stage["outputs"]},
            )
            atomic_json(state_path, state)
        snapshot_files(bindings)
        state.update(status="complete", finished_at=utc_now(), source_sha256=bindings)
        atomic_json(state_path, state)
        return state
    except BaseException as error:
        state.update(
            status="failed_preserved",
            finished_at=utc_now(),
            error_type=type(error).__name__,
            error=str(error),
        )
        if state["stages"] and state["stages"][-1]["status"] == "running":
            state["stages"][-1]["status"] = "failed_preserved"
        atomic_json(state_path, state)
        raise
    finally:
        lock.unlink()


def detach(output, review_receipt, source_bindings=None):
    """Launch one child worker; existing launch or campaign records forbid another."""
    config = preflight(output, review_receipt, source_bindings)
    output = Path(config["output"])
    require(
        not (output / "campaign_state.json").exists() and not (output / "campaign.lock").exists(),
        "Campaign already attempted; no detached retry",
    )
    launch_path = output / "campaign_launch.json"
    args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--output-root",
        str(output),
        "--review-receipt",
        str(Path(review_receipt).resolve()),
    ]
    if source_bindings is not None:
        args.extend(("--source-bindings", str(Path(source_bindings).resolve())))
    launch = {
        "schema": "fresh-readout-campaign-launch-v1",
        "status": "launching",
        "created_at": utc_now(),
        "command": args,
        "coordinator_sha256": file_sha256(__file__),
    }
    fresh_json(launch_path, launch)
    try:
        with (output / "campaign_worker.log").open("xb") as stream:
            process = subprocess.Popen(
                args,
                cwd=ROOT,
                env=environment(),
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                shell=False,
                start_new_session=True,
                close_fds=True,
            )
        launch.update(status="launched", pid=process.pid, launched_at=utc_now())
    except BaseException as error:
        launch.update(status="failed_preserved", error_type=type(error).__name__, error=str(error))
        atomic_json(launch_path, launch)
        raise
    atomic_json(launch_path, launch)
    return launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--review-receipt",
        type=Path,
        default=ROOT / "data/belief_main/fresh_readout_v2/compiled_v1/agent_review_receipt_v1.json",
    )
    parser.add_argument(
        "--source-bindings", type=Path, help="Additional frozen path-to-SHA256 JSON map"
    )
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    os.environ.update(environment())
    if args.detach:
        result = detach(args.output_root, args.review_receipt, args.source_bindings)
    else:
        result = run_campaign(
            preflight(args.output_root, args.review_receipt, args.source_bindings)
        )
        result = {"status": result["status"], "stages": len(result["stages"])}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
