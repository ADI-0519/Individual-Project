"""Explicit recovery of one interrupted, unrecorded row into fresh native storage.

The original campaign, scientific sources, fits and freezes are never modified.
Preparation copies and validates completed measurements only. Execution is a
separate explicit action; no unknown model attempt is silently retried.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPLETE_FILES = {"started.json", "complete.json", "arrays.npz", "raw_forward.npz"}
PRODUCERS = {
    "run_main_study.py",
    "run_main_study_monitor.py",
    "run_main_study_campaign.py",
    "recover_main_study_campaign.py",
}


def launcher():
    spec = importlib.util.spec_from_file_location(
        "immutable_campaign_launcher", ROOT / "scripts/run_main_study_campaign.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise ValueError(message)


def active_producers(proc_root=Path("/proc")):
    """Record matching Python producers; the current recovery process is excluded."""
    found = []
    for directory in proc_root.iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            arguments = (directory / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        names = {Path(arg.decode("utf-8", errors="replace")).name for arg in arguments}
        if names & PRODUCERS:
            found.append(
                {
                    "pid": int(directory.name),
                    "arguments": [
                        arg.decode("utf-8", errors="replace") for arg in arguments if arg
                    ],
                }
            )
    return found


def copy_interrupted_capture(source, destination, *, validator):
    """Copy exact completed entries, preserve the one unknown entry at its source."""
    helper, source, destination = launcher(), Path(source), Path(destination)
    require(not destination.exists(), "Recovery capture must be fresh")
    protocol, manifest = (
        helper.read(source / "protocol.json"),
        helper.read(source / "manifest.json"),
    )
    require(
        manifest.get("status") in {"running", "failed_or_interrupted"},
        "Expected interrupted capture",
    )
    require(manifest.get("partition") == "test", "Only interrupted test capture may be recovered")
    require(
        manifest.get("protocol_sha256") == helper.digest(source / "protocol.json"),
        "Original protocol changed",
    )
    selected = [row for row in protocol["planned_rows"] if row["split"] == "test"]
    expected = {row["row_id"] for row in selected}
    require(len(expected) == len(selected), "Duplicate planned test identity")
    complete, unknown = [], []
    for directory in sorted((source / "row_journal").iterdir()):
        require(
            directory.is_dir() and not directory.is_symlink() and directory.name in expected,
            "Unknown journal entry",
        )
        files = {path.name for path in directory.iterdir()}
        require(
            all(path.is_file() and not path.is_symlink() for path in directory.iterdir()),
            "Nonregular journal artifact",
        )
        if files == COMPLETE_FILES:
            complete.append(directory)
        else:
            require(
                files == {"started.json"},
                "Only a started-only unknown row can be explicitly recovered",
            )
            started = helper.read(directory / "started.json")
            require(
                started.get("row_id") == directory.name
                and started.get("protocol_sha256") == manifest["protocol_sha256"],
                "Unknown row protocol differs",
            )
            unknown.append(directory)
    require(len(unknown) == 1, "Exactly one started-only unknown attempt is required")
    require(
        len(complete) == manifest.get("completed_rows"),
        "Completed journal count differs from interrupted manifest",
    )
    require(
        not (source / "capture_manifest.json").exists(), "Completed capture must not use recovery"
    )
    destination.mkdir(parents=True, exist_ok=False)
    copied = {}

    def copy_file(original, target):
        require(
            original.is_file() and not original.is_symlink(), "Copy source must be a regular file"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        require(not target.exists(), "Recovery never overwrites copied artifacts")
        before = helper.digest(original)
        shutil.copy2(original, target)
        require(
            helper.digest(original) == helper.digest(target) == before,
            "Copied artifact differs or source changed",
        )
        copied[target.relative_to(destination).as_posix()] = before

    for name in ("protocol.json", "manifest.json"):
        copy_file(source / name, destination / name)
    for attempt in sorted((source / "attempts").iterdir()):
        copy_file(attempt, destination / "attempts" / attempt.name)
    for directory in complete:
        for name in sorted(COMPLETE_FILES):
            copy_file(directory / name, destination / "row_journal" / directory.name / name)
    # Exact original checker, called on the byte-identical copied measurements.
    rows = validator(destination, protocol, selected)
    retained = [row["row_id"] for row in rows]
    require(
        set(retained) == {directory.name for directory in complete},
        "Original checker did not validate all retained rows",
    )
    require(len(retained) == len(complete), "Retained row inventory changed")
    remaining = [row["row_id"] for row in selected if row["row_id"] not in set(retained)]
    return {
        "original_capture": str(source.resolve()),
        "recovery_capture": str(destination.resolve()),
        "retained_completed_rows": len(retained),
        "retained_case_ids": retained,
        "remaining_rows": len(remaining),
        "remaining_case_ids": remaining,
        "copied_artifact_sha256": copied,
        "original_manifest_sha256": helper.digest(source / "manifest.json"),
        "unknown_attempt": {
            "case_id": unknown[0].name,
            "original_directory": str(unknown[0].resolve()),
            "artifact_sha256": {"started.json": helper.digest(unknown[0] / "started.json")},
            "observed_artifacts": ["started.json"],
            "raw_outcome_available": False,
            "planned_repeat_count": 1,
            "counts_as_new_independent_case": False,
            "interpretation": "The interrupted forward may have occurred; its outcome was not recorded. Repeat is explicitly disclosed, not outcome-selected.",
        },
        "validation": "Every retained copied row passed the unchanged original _complete_rows; original/copy bytes match",
    }


def continuation_plan(old, recovery):
    """Keep the old fixed policy; skip all completed capture/fitting/freezing stages."""
    helper, recovery = launcher(), Path(recovery)
    original = Path(old["campaign"])
    configured = dict(old, campaign=str(recovery))
    all_stages = helper.build_plan(configured)
    start = next(
        index for index, stage in enumerate(all_stages) if stage["name"] == "execute_llama_test"
    )
    stages = all_stages[start:]
    replacements = {str(recovery / "test_freeze.json"): str(original / "test_freeze.json")}
    for model in ("llama", "qwen"):
        replacements[str(recovery / f"{model}_fit")] = str(original / f"{model}_fit")
    for stage in stages:
        stage["command"] = [replacements.get(value, value) for value in stage["command"]]
    stages[0]["name"] = "explicit_resume_llama_test"
    stages[0]["command"].append("--resume")
    report = recovery / "report"
    stages.append(
        {
            "name": "report_frozen_results",
            "kind": "command",
            "command": helper.command(
                "report_main_study.py",
                "--llama-analysis",
                recovery / "llama_analysis_with_monitor.json",
                "--qwen-analysis",
                recovery / "qwen_analysis_with_monitor.json",
                "--output-directory",
                report,
            ),
            "fresh": [str(report)],
            "outputs": [
                str(report / name)
                for name in ("results.md", "results_summary.json", "report_manifest.json")
            ],
        }
    )
    return stages


def prepare_recovery(
    campaign,
    recovery,
    *,
    attest_no_running_producers,
    validator=None,
    process_check=active_producers,
):
    helper, original, recovery = launcher(), Path(campaign).resolve(), Path(recovery).resolve()
    require(attest_no_running_producers, "Explicit no-running-producers attestation is required")
    producers = process_check()
    require(not producers, f"A producer is still running: {producers}")
    require(not recovery.exists(), "Recovery directory must be fresh; preserve prior attempts")
    old = helper.read(original / "config.json")
    require(
        Path(old["campaign"]).resolve() == original, "Original campaign configuration path differs"
    )
    require(
        recovery.is_relative_to(Path(old["root_run"]))
        and recovery != original
        and not recovery.is_relative_to(original),
        "Recovery must use separate native run storage",
    )
    helper.verify_bindings(old["source_sha256"])
    freeze_path = original / "test_freeze.json"
    freeze = helper.read(freeze_path)
    require(
        freeze.get("stage") == "test" and freeze.get("status") == "frozen",
        "Original test freeze is absent",
    )
    helper.verify_bindings(freeze["source_sha256"])
    helper.verify_bindings(freeze["fitted_artifact_sha256"])
    require(
        not (original / "llama_analysis_without_monitor.json").exists()
        and not (original / "qwen_test").exists(),
        "This recovery supports only the documented interrupted Llama stage",
    )
    helper.validate_prepared(old["monitor_development"], old, monitor=True)
    if validator is None:
        from latent_probes.main_study_run import _complete_rows

        validator = _complete_rows
    recovery.mkdir(parents=True, exist_ok=False)
    (recovery / "logs").mkdir()
    try:
        inventory = copy_interrupted_capture(
            original / "llama_test", recovery / "llama_test", validator=validator
        )
        receipt = {
            "schema": "main-study-explicit-interruption-recovery-v1",
            "created_at": helper.now(),
            "reason": "Externally interrupted WSL session; no matching producer is currently running",
            "machine_producer_absence_attested": True,
            "observed_matching_producers": producers,
            "source_campaign_config_sha256": helper.digest(original / "config.json"),
            "original_fit_and_test_freeze_reused": True,
            "new_fit_or_selection": False,
            "research_status": helper.RESEARCH_STATUS,
            "human_review_complete": False,
            "confirmatory_result": False,
            **inventory,
        }
        receipt_path = recovery / "recovery_receipt.json"
        helper.fresh_json(receipt_path, receipt)
        bindings = {
            **old["source_sha256"],
            **freeze["source_sha256"],
            **freeze["fitted_artifact_sha256"],
            str(freeze_path): helper.digest(freeze_path),
            str(original / "config.json"): helper.digest(original / "config.json"),
            str(original / "llama_test" / "manifest.json"): inventory["original_manifest_sha256"],
            str(receipt_path): helper.digest(receipt_path),
        }
        for path in (Path(__file__).resolve(), ROOT / "scripts/report_main_study.py"):
            bindings[str(path)] = helper.digest(path)
        config = {
            **old,
            "schema": "main-study-explicit-recovery-campaign-v1",
            "campaign": str(recovery),
            "original_campaign": str(original),
            "test_freeze": str(freeze_path),
            "source_sha256": bindings,
            "stop_file": str(recovery / "STOP"),
            "created_at": helper.now(),
            "recovery_receipt_sha256": helper.digest(receipt_path),
            "stages": continuation_plan(old, recovery),
        }
        helper.fresh_json(recovery / "config.json", config)
        helper.fresh_json(
            recovery / "prepared_manifest.json",
            {
                "schema": "main-study-recovery-preparation-seal-v1",
                "created_at": helper.now(),
                "model_calls": 0,
                "sha256": {
                    "config.json": helper.digest(recovery / "config.json"),
                    "recovery_receipt.json": helper.digest(receipt_path),
                },
            },
        )
        helper.write_state(
            recovery / "state.json",
            {
                "schema": config["schema"],
                "status": "prepared_no_model_calls",
                "created_at": helper.now(),
                "stages": [],
            },
        )
        return receipt
    except BaseException as error:
        helper.fresh_json(
            recovery / "preparation_failure.json",
            {
                "failed_at": helper.now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "model_calls": 0,
            },
        )
        raise


def run_prepared(recovery, *, child=None, process_check=active_producers):
    helper, recovery = launcher(), Path(recovery).resolve()
    config = helper.read(recovery / "config.json")
    seal = helper.read(recovery / "prepared_manifest.json")
    require(
        seal.get("schema") == "main-study-recovery-preparation-seal-v1", "Unknown preparation seal"
    )
    helper.verify_bindings({str(recovery / name): value for name, value in seal["sha256"].items()})
    require(
        config.get("schema") == "main-study-explicit-recovery-campaign-v1"
        and Path(config["campaign"]) == recovery,
        "Unknown recovery configuration",
    )
    require(not process_check(), "A producer is already running; recovery will not duplicate it")
    require(
        helper.read(recovery / "state.json").get("status") == "prepared_no_model_calls",
        "Recovery is already attempted; never automatically retry",
    )
    helper.verify_bindings(config["source_sha256"])
    receipt = helper.read(recovery / "recovery_receipt.json")
    require(
        helper.digest(recovery / "recovery_receipt.json") == config["recovery_receipt_sha256"],
        "Recovery receipt changed",
    )
    copied_root = recovery / "llama_test"
    helper.verify_bindings(
        {
            str(copied_root / name): value
            for name, value in receipt["copied_artifact_sha256"].items()
        }
    )
    helper.fresh_json(
        recovery / "execution_started.json",
        {
            "started_at": helper.now(),
            "config_sha256": helper.digest(recovery / "config.json"),
            "explicit_unknown_attempt_repeat": receipt["unknown_attempt"],
        },
    )
    state = {
        "schema": config["schema"],
        "status": "running",
        "started_at": helper.now(),
        "stages": [],
    }
    helper.write_state(recovery / "state.json", state)
    child = child or helper.run_child
    try:
        for stage in config["stages"]:
            require(not (recovery / "STOP").exists(), "Recovery stop file observed between stages")
            helper.verify_bindings(config["source_sha256"])
            for path in stage.get("fresh", []):
                require(not Path(path).exists(), f"Unexpected existing recovery output: {path}")
            record = {"name": stage["name"], "status": "running", "started_at": helper.now()}
            state["stages"].append(record)
            helper.write_state(recovery / "state.json", state)
            print(f"[{helper.now()}] Recovery: {stage['name']}", flush=True)
            if stage.get("adopt_prepared_or_complete"):
                helper.validate_prepared(stage["monitor"]["directory"], config, monitor=True)
            result = child(stage["command"], recovery / "logs" / f"{stage['name']}.log")
            require(
                result["returncode"] == 0,
                f"Recovery child failed; no retry: {stage['name']}\n{result['output_tail']}",
            )
            for path in stage.get("outputs", []):
                require(Path(path).is_file(), f"Missing recovery output: {path}")
            if "capture" in stage:
                value = stage["capture"]
                helper.validate_capture(
                    value["directory"],
                    partition="test",
                    model_key=value["model"],
                    bank_sha256=config["bank_sha256"],
                    expected_freeze=config["test_freeze"],
                )
            if "monitor" in stage:
                value = stage["monitor"]
                helper.validate_monitor_complete(
                    value["directory"], partition=value["partition"], config=config
                )
            record.update(status="complete", finished_at=helper.now(), result=result)
            helper.write_state(recovery / "state.json", state)
        state.update(status="complete", finished_at=helper.now())
    except BaseException as error:
        state.update(
            status="failed_or_interrupted",
            stopped_at=helper.now(),
            error_type=type(error).__name__,
            error=str(error),
        )
        helper.write_state(recovery / "state.json", state)
        raise
    helper.write_state(recovery / "state.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign", type=Path, help="Original interrupted campaign; required for preparation"
    )
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--attest-no-running-producers", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Prepare then explicitly execute")
    mode.add_argument(
        "--run-prepared",
        action="store_true",
        help="Execute an already prepared recovery exactly once",
    )
    args = parser.parse_args()
    require(sys.platform == "linux", "Use the original WSL Python environment and native storage")
    require(
        not str(args.recovery.resolve()).startswith("/mnt/"),
        "Recovery outputs must use native storage",
    )
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        require(os.environ.get(key) == "1", f"{key}=1 is required")
    if args.run_prepared:
        run_prepared(args.recovery)
    else:
        if args.campaign is None:
            parser.error("--campaign is required for preparation")
        receipt = prepare_recovery(
            args.campaign,
            args.recovery,
            attest_no_running_producers=args.attest_no_running_producers,
        )
        print(
            json.dumps(
                {
                    "retained": receipt["retained_completed_rows"],
                    "remaining": receipt["remaining_rows"],
                    "planned_unknown_repeat": 1,
                    "model_calls": 0,
                }
            )
        )
        if args.execute:
            run_prepared(args.recovery)


if __name__ == "__main__":
    main()
