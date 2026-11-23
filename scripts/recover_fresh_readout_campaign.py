"""Explicit zero-unknown recovery into a new root; frozen scientific code is reused.

The approved boundary is Llama 2048 complete and Qwen 900 complete. No row is
repeated. Preparation is model-free; a separately invoked run-prepared executes
once, adopting Llama and resuming the unchanged Qwen capture before the original
fixed fitting, sealing, test and analysis stages. Failed attempts are preserved.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
from pathlib import Path

from threadpoolctl import threadpool_limits

from latent_probes.fresh_readout_capture import _authorization, _completed, verify_protocol
from latent_probes.fresh_readout_fit import load_training_capture
from latent_probes.storage import atomic_json, file_sha256, read_json, utc_now

ROOT = Path(__file__).resolve().parents[1]
COMPLETE_FILES = {"started.json", "arrays.npz", "raw_forward.npz", "complete.json"}
PARTITIONS = ("source_fit", "source_dev", "correction_fit", "correction_dev")


def require(value, message):
    if not value:
        raise ValueError(message)


def launcher():
    path = ROOT / "scripts/run_fresh_readout_campaign.py"
    spec = importlib.util.spec_from_file_location("frozen_fresh_campaign_recovery", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def active_producers():
    names = {
        "run_fresh_readout_campaign.py",
        "run_fresh_readout_capture.py",
        "fit_fresh_readout.py",
        "analyze_fresh_readout.py",
    }
    found = []
    for process in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = process.read_bytes().split(b"\0")
            if any(Path(arg.decode(errors="replace")).name in names for arg in args if arg):
                found.append(int(process.parent.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return found


def inventory(directory):
    directory = Path(directory)
    result = {}
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), "Symlink in preserved tree: " + str(path))
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = file_sha256(path)
        else:
            require(path.is_dir(), "Nonregular preserved artifact: " + str(path))
    return result


def check_inventory(directory, hashes):
    require(inventory(directory) == hashes, "Preserved original tree changed")


def inspect_training(directory, protocol, *, complete_count, full, validator=_completed):
    directory = Path(directory)
    selected = [r for r in protocol["planned_rows"] if r["partition"] in PARTITIONS]
    digest = file_sha256(directory.parent / "protocol.json")
    manifest = read_json(directory / "manifest.json")
    require(
        manifest.get("protocol_sha256") == digest
        and manifest.get("partition") == "training"
        and manifest.get("completed_rows") == complete_count
        and manifest.get("status")
        in ({"complete"} if full else {"running", "interrupted_preserved"}),
        "Interrupted training boundary/status differs",
    )
    allowed = {"manifest.json", "attempts", "row_journal"} | (
        {"features.npz", "rows.json"} if full else set()
    )
    require(
        {p.name for p in directory.iterdir()} == allowed, "Unexpected or partial training artifact"
    )
    journal = directory / "row_journal"
    expected = [r["row_id"] for r in selected[:complete_count]]
    require(
        {p.name for p in journal.iterdir()} == set(expected),
        "Recovery requires exactly the completed chronological prefix; unknown rows forbid recovery",
    )
    require(
        all(p.is_file() and not p.is_symlink() for p in (directory / "attempts").iterdir()),
        "Invalid capture attempt journal",
    )
    for planned in selected[:complete_count]:
        row = journal / planned["row_id"]
        require(
            row.is_dir() and {p.name for p in row.iterdir()} == COMPLETE_FILES,
            "Only fully completed rows are recoverable; no unknown/error override",
        )
        require(
            validator(row, planned, protocol, digest) is not None,
            "Completed row failed frozen validation",
        )
    if full:
        require(
            complete_count == len(selected)
            and file_sha256(directory / "features.npz") == manifest["features_sha256"]
            and file_sha256(directory / "rows.json") == manifest["rows_sha256"],
            "Completed training aggregate changed",
        )
    else:
        require(0 < complete_count < len(selected), "Interrupted training has no pending rows")
    return {
        "retained_ids": expected,
        "remaining_ids": [r["row_id"] for r in selected[complete_count:]],
        "original_authorization": manifest["authorization"],
    }


def copy_file(original, destination, copied, recovery):
    original, destination = Path(original), Path(destination)
    require(
        original.is_file() and not original.is_symlink(), "Copy source must be an ordinary file"
    )
    require(not destination.exists(), "Recovery never overwrites artifacts")
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = file_sha256(original)
    shutil.copyfile(original, destination)
    require(
        file_sha256(original) == file_sha256(destination) == before, "Source/copy bytes changed"
    )
    copied[destination.relative_to(recovery).as_posix()] = before


def continuation_plan(config, original):
    helper = launcher()
    plan = helper.build_plan(config)
    require(
        [s["name"] for s in plan[:2]] == ["llama_training", "qwen_training"],
        "Frozen stage order changed",
    )
    plan[0]["kind"] = "adopt_completed"
    plan[0]["original_command_not_executed"] = plan[0].pop("command")
    command = plan[1]["command"]
    command[command.index("--bindings") + 1] = str(Path(original) / "qwen_training_bindings.json")
    command.append("--resume")
    return plan


@threadpool_limits.wrap(limits=4)
def prepare_recovery(
    original,
    recovery,
    *,
    attest_no_running_producers,
    inspection_receipt=None,
    process_check=active_producers,
    validator=_completed,
    aggregate_validator=load_training_capture,
    allow_test_fixture=False,
):
    helper = launcher()
    original, recovery = Path(original).resolve(), Path(recovery).resolve()
    require(
        attest_no_running_producers and not process_check(),
        "Require no-running-producers attestation and check",
    )
    require(
        original.is_dir()
        and not recovery.exists()
        and original != recovery
        and not recovery.is_relative_to(original)
        and not original.is_relative_to(recovery),
        "Use a fresh separate recovery root",
    )
    state = read_json(original / "campaign_state.json")
    require(
        state.get("schema") == "fresh-readout-campaign-v1"
        and state.get("status") == "running"
        and [s["name"] for s in state["stages"]] == ["llama_training", "qwen_training"]
        and state["stages"][0]["status"] == "complete"
        and state["stages"][1]["status"] == "running",
        "Recovery supports only the reviewed interrupted Qwen training stage",
    )
    for forbidden in (
        "llama_fit",
        "qwen_fit",
        "test_fit_freeze.json",
        "llama_analysis",
        "qwen_analysis",
        "llama_test_bindings.json",
        "qwen_test_bindings.json",
    ):
        require(
            not (original / forbidden).exists(),
            "Prior fit/test/analysis artifact forbids this recovery",
        )
    sources = helper.merge_bindings(state["source_sha256"])
    protocols, boundaries, authorizations = {}, {}, {}
    old_inventory = inventory(original)
    for model in ("llama", "qwen"):
        directory = original / model
        protocol = read_json(directory / "protocol.json")
        require(
            protocol.get("model_key") == model
            and (protocol.get("test_fixture") is False or allow_test_fixture),
            "Wrong model/fixture preparation",
        )
        require(
            {p.name for p in directory.iterdir()}
            <= {"protocol.json", "preparation.json", "engineering", "training", "execution.lock"},
            "Unexpected model capture partition/artifact",
        )
        verify_protocol(protocol)
        protocol_digest = file_sha256(directory / "protocol.json")
        require(
            read_json(directory / "preparation.json")["protocol_sha256"] == protocol_digest,
            "Prepared protocol changed",
        )
        selected = [r for r in protocol["planned_rows"] if r["partition"] in PARTITIONS]
        count = (
            len(selected)
            if model == "llama"
            else (len(selected) // 2 if allow_test_fixture else 900)
        )
        if not allow_test_fixture:
            require(len(selected) == 2048, "Need the exact native training inventory")
        engineering = directory / "engineering"
        require(
            {p.name for p in engineering.iterdir()} == COMPLETE_FILES
            and validator(engineering, protocol["engineering_row"], protocol, protocol_digest)
            is not None
            and read_json(engineering / "complete.json")["capture"]["hook_alignment_checked"]
            is True,
            "Engineering alignment record is incomplete",
        )
        boundary = inspect_training(
            directory / "training",
            protocol,
            complete_count=count,
            full=model == "llama",
            validator=validator,
        )
        binding_path = original / (model + "_training_bindings.json")
        authorization = _authorization(protocol, protocol_digest, binding_path, "training")
        require(
            authorization == boundary["original_authorization"],
            "Original resume authorization differs",
        )
        protocols[model], boundaries[model], authorizations[model] = (
            protocol,
            boundary,
            authorization,
        )
    require(
        [r["row_id"] for r in protocols["llama"]["planned_rows"]]
        == [r["row_id"] for r in protocols["qwen"]["planned_rows"]],
        "Both model row IDs differ",
    )
    aggregate_validator(
        original / "llama/training",
        protocols["llama"]["bank_path"],
        original / "llama/protocol.json",
        allow_test_fixture=allow_test_fixture,
    )
    helper.snapshot_files(sources)
    check_inventory(original, old_inventory)
    recovery.mkdir(parents=True, exist_ok=False)
    copied = {}
    try:
        for model in ("llama", "qwen"):
            for name in ("protocol.json", "preparation.json"):
                copy_file(original / model / name, recovery / model / name, copied, recovery)
            for area in ("engineering", "training"):
                for path in sorted((original / model / area).rglob("*")):
                    if path.is_file():
                        copy_file(
                            path,
                            recovery / model / area / path.relative_to(original / model / area),
                            copied,
                            recovery,
                        )
            copy_file(
                original / (model + "_training_bindings.json"),
                recovery / (model + "_training_bindings.json"),
                copied,
                recovery,
            )
            inspect_training(
                recovery / model / "training",
                protocols[model],
                complete_count=len(boundaries[model]["retained_ids"]),
                full=model == "llama",
                validator=validator,
            )
        aggregate_validator(
            recovery / "llama/training",
            protocols["llama"]["bank_path"],
            recovery / "llama/protocol.json",
            allow_test_fixture=allow_test_fixture,
        )
        check_inventory(original, old_inventory)
        helper.fresh_json(recovery / "original_inventory.json", old_inventory)
        helper.fresh_json(recovery / "copied_inventory.json", copied)
        receipt = {
            "schema": "fresh-readout-zero-unknown-recovery-v2",
            "created_at": utc_now(),
            "original_campaign": str(original),
            "recovery_campaign": str(recovery),
            "test_fixture": allow_test_fixture,
            "model_calls": 0,
            "repeat_count": 0,
            "unknown_rows": [],
            "selection_basis": "Durable completed recording status only; no outcome-based selection",
            "retained_completed_rows": sum(len(b["retained_ids"]) for b in boundaries.values()),
            "remaining_qwen_training_rows": len(boundaries["qwen"]["remaining_ids"]),
            "new_cohort_prefills_remaining": len(boundaries["qwen"]["remaining_ids"])
            + sum(r["partition"] == "test" for p in protocols.values() for r in p["planned_rows"]),
            "engineering_prefills_remaining": 0,
            "numerical_read_thread_limit": 4,
            "boundaries": boundaries,
            "original_inventory_sha256": file_sha256(recovery / "original_inventory.json"),
            "copied_inventory_sha256": file_sha256(recovery / "copied_inventory.json"),
            "mutable_copied_artifacts": ["qwen/training/manifest.json"],
            "unchanged_completed": True,
            "original_preserved": True,
            "stale_original_locks_preserved_not_copied": [
                name for name in old_inventory if name.endswith(".lock")
            ],
            "inspection_receipt": None
            if inspection_receipt is None
            else {
                "path": str(Path(inspection_receipt).resolve()),
                "sha256": file_sha256(inspection_receipt),
            },
        }
        helper.fresh_json(recovery / "recovery_receipt.json", receipt)
        sources = helper.merge_bindings(
            sources,
            {
                str(Path(__file__).resolve()): file_sha256(__file__),
                str(recovery / "recovery_receipt.json"): file_sha256(
                    recovery / "recovery_receipt.json"
                ),
                str(original / "campaign_state.json"): file_sha256(
                    original / "campaign_state.json"
                ),
            },
        )
        if inspection_receipt is not None:
            sources = helper.merge_bindings(
                sources, {str(Path(inspection_receipt).resolve()): file_sha256(inspection_receipt)}
            )
        review_files = read_json(original / "qwen_training_bindings.json")["review_artifact_sha256"]
        config = {
            "output": str(recovery),
            "original_campaign": str(original),
            "bank_path": protocols["llama"]["bank_path"],
            "protocols": protocols,
            "source_sha256": sources,
            "review_artifact_sha256": review_files,
            "inventory_sha256": state["inventory_sha256"],
            "test_fixture": allow_test_fixture,
        }
        helper.fresh_json(recovery / "recovery_config.json", config)
        helper.fresh_json(recovery / "campaign_plan.json", continuation_plan(config, original))
        archive = helper.archive_sources(recovery, sources)
        helper.fresh_json(
            recovery / "recovery_preparation.json",
            {
                "schema": "fresh-readout-recovery-preparation-v2",
                "model_calls": 0,
                "source_archive_manifest": archive,
                "sha256": {
                    name: file_sha256(recovery / name)
                    for name in (
                        "recovery_config.json",
                        "recovery_receipt.json",
                        "original_inventory.json",
                        "copied_inventory.json",
                        "campaign_plan.json",
                    )
                },
            },
        )
        helper.fresh_json(
            recovery / "campaign_state.json",
            {
                "schema": "fresh-readout-campaign-v1",
                "status": "prepared_recovery_no_model_calls",
                "created_at": utc_now(),
                "stages": [],
            },
        )
        return receipt
    except BaseException as error:
        helper.fresh_json(
            recovery / "preparation_failure.json",
            {"error_type": type(error).__name__, "error": str(error), "model_calls": 0},
        )
        raise


def verify_retained(recovery, copied, *, before_resume=False):
    for name, expected in copied.items():
        if name != "qwen/training/manifest.json" or before_resume:
            require(
                file_sha256(Path(recovery) / name) == expected,
                "Retained completed bytes changed: " + name,
            )


@threadpool_limits.wrap(limits=4)
def run_prepared(
    recovery,
    *,
    attest_no_running_producers,
    child=None,
    binder=None,
    process_check=active_producers,
    allow_test_fixture=False,
):
    helper, recovery = launcher(), Path(recovery).resolve()
    require(
        attest_no_running_producers and not process_check(),
        "A producer is running or absence was not attested",
    )
    seal = read_json(recovery / "recovery_preparation.json")
    require(seal.get("schema") == "fresh-readout-recovery-preparation-v2", "Wrong recovery seal")
    helper.snapshot_files(
        {str(recovery / name): expected for name, expected in seal["sha256"].items()}
    )
    config = read_json(recovery / "recovery_config.json")
    require(
        not config["test_fixture"] or allow_test_fixture,
        "Native execution rejects fixture recovery",
    )
    require(
        read_json(recovery / "campaign_state.json")["status"] == "prepared_recovery_no_model_calls",
        "Recovery already attempted; no automatic retry",
    )
    original = Path(config["original_campaign"])
    old_inventory, copied = (
        read_json(recovery / "original_inventory.json"),
        read_json(recovery / "copied_inventory.json"),
    )
    check_inventory(original, old_inventory)
    verify_retained(recovery, copied, before_resume=True)
    helper.snapshot_files(config["source_sha256"])
    helper.fresh_json(
        recovery / "execution_started.json",
        {
            "started_at": utc_now(),
            "repeat_count": 0,
            "recovery_config_sha256": file_sha256(recovery / "recovery_config.json"),
        },
    )
    state = {
        "schema": "fresh-readout-campaign-v1",
        "status": "running",
        "created_at": utc_now(),
        "research_status": "agent_audited_exploratory",
        "human_review_complete": False,
        "source_sha256": config["source_sha256"],
        "inventory_sha256": config["inventory_sha256"],
        "recovery_receipt_sha256": file_sha256(recovery / "recovery_receipt.json"),
        "repeat_count": 0,
        "stages": [],
    }
    atomic_json(recovery / "campaign_state.json", state)
    logs = recovery / "campaign_logs"
    logs.mkdir(exist_ok=False)
    plan = read_json(recovery / "campaign_plan.json")
    child, binder = child or helper.run_child, binder or helper.write_bindings
    bindings = config["source_sha256"]
    try:
        for stage in plan:
            helper.snapshot_files(bindings)
            entry = {"name": stage["name"], "status": "running", "started_at": utc_now()}
            state["stages"].append(entry)
            atomic_json(recovery / "campaign_state.json", state)
            print(f"[{utc_now()}] Recovery: {stage['name']}", flush=True)
            if stage["kind"] == "adopt_completed":
                entry.update(adopted_complete=True, new_model_forwards=0)
            elif stage["kind"] == "bindings":
                generated = binder(config, test=True)
                fit_seal = read_json(recovery / "test_fit_freeze.json")
                bindings = helper.merge_bindings(
                    bindings,
                    generated,
                    fit_seal["fitted_artifact_sha256"],
                    {
                        str(recovery / "test_fit_freeze.json"): file_sha256(
                            recovery / "test_fit_freeze.json"
                        )
                    },
                )
            else:
                if stage["name"] != "qwen_training":
                    require(
                        all(not Path(p).exists() for p in stage["outputs"]),
                        "Unexpected existing later-stage output",
                    )
                stdout, stderr = (
                    logs / (stage["name"] + ".stdout.log"),
                    logs / (stage["name"] + ".stderr.log"),
                )
                entry.update(
                    command=stage["command"], stdout_path=str(stdout), stderr_path=str(stderr)
                )
                atomic_json(recovery / "campaign_state.json", state)
                code = child(stage["command"], stdout, stderr)
                entry.update(
                    returncode=code,
                    stdout_sha256=file_sha256(stdout),
                    stderr_sha256=file_sha256(stderr),
                )
                require(code == 0, "Recovery stage failed; preserve and stop: " + stage["name"])
            require(
                all(Path(p).is_file() for p in stage["outputs"]), "Recovery stage output missing"
            )
            if stage["name"] == "qwen_training":
                verify_retained(recovery, copied)
            entry.update(
                status="complete",
                finished_at=utc_now(),
                output_sha256={p: file_sha256(p) for p in stage["outputs"]},
            )
            atomic_json(recovery / "campaign_state.json", state)
        check_inventory(original, old_inventory)
        verify_retained(recovery, copied)
        helper.snapshot_files(bindings)
        state.update(
            status="complete",
            finished_at=utc_now(),
            unchanged_completed=True,
            source_sha256=bindings,
        )
        atomic_json(recovery / "campaign_state.json", state)
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
        atomic_json(recovery / "campaign_state.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--inspection-receipt", type=Path)
    parser.add_argument("--attest-no-running-producers", action="store_true")
    parser.add_argument("--run-prepared", action="store_true")
    args = parser.parse_args()
    require(
        platform.system() == "Linux" and not str(args.recovery.resolve()).startswith("/mnt/"),
        "Use native Linux recovery storage",
    )
    os.environ.update(launcher().environment())
    if args.run_prepared:
        result = run_prepared(
            args.recovery, attest_no_running_producers=args.attest_no_running_producers
        )
        result = {"status": result["status"], "repeat_count": 0}
    else:
        require(args.original is not None, "--original is required for preparation")
        result = prepare_recovery(
            args.original,
            args.recovery,
            attest_no_running_producers=args.attest_no_running_producers,
            inspection_receipt=args.inspection_receipt,
        )
        result = {
            "retained_completed_rows": result["retained_completed_rows"],
            "remaining_qwen_training_rows": result["remaining_qwen_training_rows"],
            "repeat_count": 0,
            "model_calls": 0,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
