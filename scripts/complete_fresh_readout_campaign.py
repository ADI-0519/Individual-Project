"""Wait for both completed analyses, independently audit, report, and copy exports.

This standalone helper imports only the standard library. It never launches or
retries model capture/fitting, never reads partial test results, and never edits
the existing campaign. Every helper and command is frozen before waiting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("llama", "qwen")
STAGES = (
    "llama_training",
    "qwen_training",
    "fit_llama",
    "fit_qwen",
    "freeze_both_fits",
    "bind_both_tests",
    "llama_test",
    "qwen_test",
    "analyze_llama",
    "analyze_qwen",
)
EXPORTS = {
    "preparation.json",
    "summary.json",
    "results.md",
    "fresh_readout.pdf",
    "fresh_readout.svg",
    "fresh_readout.png",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(UTC).isoformat()


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fresh_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def update_json(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def check_hashes(bindings):
    require(isinstance(bindings, dict) and bindings, "Need nonempty frozen hashes")
    for path, expected in bindings.items():
        require(Path(path).is_file() and digest(path) == expected, "Changed frozen file: " + path)


def commands(config):
    campaign = Path(config["campaign_root"])
    audit = [sys.executable, config["audit_script"], "--authorized-both-complete"]
    for model in MODELS:
        audit.extend(
            (
                "--model",
                model,
                str(campaign / (model + "_fit")),
                str(campaign / model / "test"),
                str(campaign / (model + "_analysis")),
            )
        )
    audit.extend(
        ("--seal", str(campaign / "test_fit_freeze.json"), "--output", config["audit_output"])
    )
    report = [
        sys.executable,
        config["reporter_script"],
        "--llama-analysis",
        str(campaign / "llama_analysis"),
        "--qwen-analysis",
        str(campaign / "qwen_analysis"),
        "--output",
        config["report_output"],
    ]
    return {"audit": audit, "report": report}


def completed_analysis_gate(campaign, state):
    """Read completion metadata and hash files, without parsing either result JSON."""
    campaign = Path(campaign)
    require(
        state.get("schema") == "fresh-readout-campaign-v1" and state.get("status") == "complete",
        "Campaign is not complete",
    )
    require(
        tuple(stage["name"] for stage in state["stages"]) == STAGES
        and all(stage.get("status") == "complete" for stage in state["stages"]),
        "All ten fixed campaign stages must complete in order",
    )
    stages = {stage["name"]: stage for stage in state["stages"]}
    manifests = {
        model: read(campaign / (model + "_analysis") / "manifest.json") for model in MODELS
    }
    bindings = {str(campaign / "campaign_state.json"): digest(campaign / "campaign_state.json")}
    # Validate BOTH metadata sets before hashing either saved result/prediction file.
    for model in MODELS:
        stage = stages.get("analyze_" + model, {})
        manifest = manifests[model]
        protocol_path = campaign / model / "protocol.json"
        protocol = read(protocol_path)
        capture_path = campaign / model / "test" / "manifest.json"
        capture = read(capture_path)
        require(
            stage.get("status") == "complete"
            and manifest.get("schema") == "fresh-readout-analysis-manifest-v2"
            and manifest.get("status", "complete") == "complete"
            and set(manifest.get("output_sha256", {})) == {"results.json", "predictions.npz"},
            "Both final analysis manifests/stages are required",
        )
        require(
            protocol.get("test_fixture") is False
            and protocol.get("model_key") == model
            and capture.get("status") == "complete"
            and capture.get("test_fixture") is False
            and capture.get("completed_rows") == 4992
            and capture.get("protocol_sha256") == digest(protocol_path),
            "Both analyses must come from completed native test captures",
        )
        require(
            manifest.get("source_sha256", {}).get(str(protocol_path)) == digest(protocol_path),
            "Analysis does not bind its native model protocol",
        )
        for path in (
            protocol_path,
            capture_path,
            campaign / (model + "_analysis") / "manifest.json",
        ):
            bindings[str(path)] = digest(path)
    for model in MODELS:
        directory = campaign / (model + "_analysis")
        stage_hashes = stages["analyze_" + model].get("output_sha256", {})
        for name in ("manifest.json", "results.json", "predictions.npz"):
            path = directory / name
            observed = digest(path)
            require(
                stage_hashes.get(str(path)) == observed,
                "Campaign analysis output hash differs: " + str(path),
            )
            if name != "manifest.json":
                require(
                    manifests[model]["output_sha256"][name] == observed,
                    "Completed analysis output changed: " + str(path),
                )
            bindings[str(path)] = observed
    return bindings


def audit_pass(directory, audit_script_sha256):
    directory = Path(directory)
    receipt = read(directory / "audit_result.json")
    require(
        receipt.get("schema") == "independent-fresh-readout-saved-data-audit-v1"
        and receipt.get("status") == "pass"
        and set(receipt.get("models", {})) == set(MODELS)
        and all(receipt["models"][model].get("all_metrics_match") is True for model in MODELS),
        "Independent audit did not pass for both models",
    )
    reconstruction = read(directory / "reconstruction_freeze.json")
    require(
        reconstruction.get("saved_results_or_predictions_parsed") is False
        and reconstruction.get("completed_output_bytes_hashed_before_reconstruction") is True
        and set(reconstruction.get("sha256", {})) == set(MODELS),
        "Missing both-model independent reconstruction freeze",
    )
    for model, hashes in reconstruction["sha256"].items():
        require(
            hashes and all(Path(name).name == name for name in hashes),
            "Invalid reconstruction file names",
        )
        check_hashes({str(directory / model / name): expected for name, expected in hashes.items()})
    with (directory / "access_log.jsonl").open(encoding="utf-8") as stream:
        first = json.loads(next(stream))
    require(
        first.get("kind") == "audit_started" and first.get("source_sha256") == audit_script_sha256,
        "Audit receipt does not bind the frozen independent checker",
    )
    return {str(path): digest(path) for path in directory.rglob("*") if path.is_file()}


def report_exports(directory, reporter_script, reporter_sha256, input_hashes):
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    require(
        manifest.get("schema") == "fresh-readout-report-manifest-v2"
        and manifest.get("status") == "complete"
        and manifest.get("test_fixture") is False
        and manifest.get("report_kind") == "completed_fresh_pipeline_comparison"
        and manifest.get("research_status") == "agent_audited_exploratory"
        and manifest.get("human_review_complete") is False
        and manifest.get("confirmatory_result") is False,
        "Need a complete native exploratory report",
    )
    require(
        manifest.get("reporter_source_sha256") == {str(reporter_script): reporter_sha256},
        "Report used a different reporter",
    )
    expected_inputs = {
        path: expected
        for path, expected in input_hashes.items()
        if Path(path).parent.name in {model + "_analysis" for model in MODELS}
    }
    require(
        manifest.get("input_sha256") == expected_inputs,
        "Report inputs differ from the two completed audited analyses",
    )
    hashes = manifest.get("output_sha256", {})
    require(
        set(hashes) == EXPORTS
        and {p.name for p in directory.iterdir()} == EXPORTS | {"manifest.json"},
        "Report export inventory differs",
    )
    require(
        all(
            (directory / name).is_file() and not (directory / name).is_symlink() for name in EXPORTS
        ),
        "Report exports must be ordinary files",
    )
    check_hashes({str(directory / name): expected for name, expected in hashes.items()})
    return {**hashes, "manifest.json": digest(directory / "manifest.json")}


def publish_exports(report, destination, hashes):
    report, destination = Path(report), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name, expected in hashes.items():
        require(
            Path(name).name == name and digest(report / name) == expected,
            "Report changed before publish",
        )
        shutil.copyfile(report / name, destination / name)
        require(digest(destination / name) == expected, "Copied export bytes differ: " + name)
    return {str(destination / name): expected for name, expected in hashes.items()}


def run_child(arguments, stdout_path, stderr_path):
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "PYTHONUNBUFFERED": "1",
    }
    with Path(stdout_path).open("xb") as stdout, Path(stderr_path).open("xb") as stderr:
        result = subprocess.run(
            arguments,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            shell=False,
            check=False,
        )
    return result.returncode


def watch(config, *, child=run_child, sleeper=time.sleep):
    require(0 < config["poll_seconds"] <= 45, "Polling interval must be within 45 seconds")
    require(
        len(
            {
                config[name]
                for name in ("report_output", "audit_output", "publish_dir", "watcher_output")
            }
        )
        == 4,
        "Watcher, audit, report and publish directories must be distinct",
    )
    output, campaign = Path(config["watcher_output"]), Path(config["campaign_root"])
    for name in ("report_output", "audit_output", "publish_dir", "watcher_output"):
        require(
            not Path(config[name]).exists(),
            "Use fresh exclusive outputs; no retry: " + config[name],
        )
    output.mkdir(parents=True, exist_ok=False)
    receipt_path = output / "completion_receipt.json"
    code = {
        str(Path(__file__).resolve()): digest(__file__),
        config["audit_script"]: digest(config["audit_script"]),
        config["reporter_script"]: digest(config["reporter_script"]),
    }
    frozen_commands = commands(config)
    state = {
        "schema": "fresh-readout-completion-receipt-v1",
        "status": "waiting",
        "created_at": now(),
        "config": config,
        "source_sha256": code,
        "commands": frozen_commands,
        "poll_count": 0,
        "stages": [],
        "new_language_model_calls": 0,
        "classifier_fits": 0,
    }
    fresh_json(receipt_path, state)
    try:
        archive = output / "code_archive"
        archive.mkdir()
        for index, source in enumerate(code):
            target = archive / f"{index}_{Path(source).name}"
            shutil.copyfile(source, target)
            require(digest(target) == code[source], "Helper changed during source freeze")
        while True:
            check_hashes(code)
            current = read(campaign / "campaign_state.json")
            status = current.get("status")
            state.update(
                poll_count=state["poll_count"] + 1, campaign_status=status, last_poll_at=now()
            )
            update_json(receipt_path, state)
            if status == "complete":
                inputs = completed_analysis_gate(campaign, current)
                state.update(status="both_analyses_verified", analysis_input_sha256=inputs)
                update_json(receipt_path, state)
                break
            require(
                status in {"starting", "running"},
                "Campaign stopped or failed; no audit/report retry",
            )
            sleeper(config["poll_seconds"])
        audit_hashes = {}
        for name in ("audit", "report"):
            check_hashes(code)
            check_hashes(inputs)
            if name == "report":
                check_hashes(audit_hashes)
            stdout, stderr = output / (name + ".stdout.log"), output / (name + ".stderr.log")
            stage = {
                "name": name,
                "status": "running",
                "started_at": now(),
                "command": frozen_commands[name],
                "stdout_path": str(stdout),
                "stderr_path": str(stderr),
            }
            state["stages"].append(stage)
            update_json(receipt_path, state)
            code_value = child(frozen_commands[name], stdout, stderr)
            stage.update(
                returncode=code_value, stdout_sha256=digest(stdout), stderr_sha256=digest(stderr)
            )
            require(code_value == 0, name + " exited unsuccessfully; preserve attempt and stop")
            check_hashes(code)
            check_hashes(inputs)
            if name == "audit":
                audit_hashes = audit_pass(config["audit_output"], code[config["audit_script"]])
                state["audit_artifact_sha256"] = audit_hashes
            stage.update(status="complete", finished_at=now())
            update_json(receipt_path, state)
        exports = report_exports(
            config["report_output"],
            config["reporter_script"],
            code[config["reporter_script"]],
            inputs,
        )
        check_hashes(code)
        check_hashes(inputs)
        check_hashes(audit_hashes)
        state["published_sha256"] = publish_exports(
            config["report_output"], config["publish_dir"], exports
        )
        state.update(status="complete", finished_at=now(), report_export_sha256=exports)
        update_json(receipt_path, state)
        return state
    except BaseException as error:
        state.update(
            status="failed_preserved",
            finished_at=now(),
            error_type=type(error).__name__,
            error=str(error),
        )
        if state["stages"] and state["stages"][-1]["status"] == "running":
            state["stages"][-1]["status"] = "failed_preserved"
        update_json(receipt_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--publish-dir", type=Path, required=True)
    parser.add_argument("--audit-script", type=Path, required=True)
    parser.add_argument(
        "--reporter-script", type=Path, default=ROOT / "scripts/report_fresh_readout.py"
    )
    parser.add_argument("--watcher-output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    report = args.report_output.resolve()
    watcher = (args.watcher_output or report.with_name(report.name + "_completion")).resolve()
    config = {
        name: str(getattr(args, name).resolve())
        for name in (
            "campaign_root",
            "report_output",
            "publish_dir",
            "audit_script",
            "reporter_script",
        )
    }
    config.update(
        watcher_output=str(watcher),
        audit_output=str((args.audit_output or watcher / "audit").resolve()),
        poll_seconds=args.poll_seconds,
    )
    require(platform.system() == "Linux", "Run the watcher with the native Linux environment")
    for name in ("report_output", "watcher_output", "audit_output"):
        require(
            not config[name].startswith("/mnt/") and "onedrive" not in config[name].lower(),
            "Watcher, audit and report outputs require native Linux storage",
        )
    result = watch(config)
    print(
        json.dumps(
            {"status": result["status"], "receipt": str(watcher / "completion_receipt.json")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
