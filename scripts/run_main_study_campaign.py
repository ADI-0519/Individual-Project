"""Run the fixed agent-exploratory campaign sequentially, preserving every attempt.

This launcher is deliberately outside the frozen scientific policy. It never
changes cases, methods, selection, failures or output files, and never retries.
It adopts the externally running Llama capture instead of starting a second one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "run_main_study.py",
    "audit_main_study_capture.py",
    "fit_main_study.py",
    "prepare_main_study_execution.py",
    "analyse_main_study.py",
    "run_main_study_monitor.py",
    "analyse_main_study_monitor.py",
    "run_main_study_campaign.py",
)
SCHEMA = "main-study-sequential-campaign-v1"
RESEARCH_STATUS = "agent_audited_exploratory"
LOG_LIMIT = 64 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(UTC).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fresh_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_state(path, value):
    temporary = Path(path).with_suffix(".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def verify_bindings(bindings):
    require(isinstance(bindings, dict) and bindings, "Nonempty source bindings required")
    for path, expected in bindings.items():
        require(
            Path(path).is_file() and digest(path) == expected, f"Changed campaign input: {path}"
        )


def validate_capture(directory, *, partition, model_key, bank_sha256, expected_freeze=None):
    """Cheap identity checks; the separate standalone auditor checks raw arrays."""
    root = Path(directory)
    protocol, manifest = read(root / "protocol.json"), read(root / "capture_manifest.json")
    require(manifest.get("status") == "complete", "Capture is not complete")
    require(protocol.get("test_fixture") is False, "Campaign cannot adopt fixture captures")
    require(protocol.get("model_key") == model_key, "Adopted capture model differs")
    require(protocol.get("bank_sha256") == bank_sha256, "Adopted capture bank differs")
    require(manifest.get("partition") == partition, "Adopted capture partition differs")
    require(
        manifest.get("protocol_sha256") == digest(root / "protocol.json"),
        "Capture protocol changed",
    )
    provenance = manifest.get("provenance", {})
    require(
        provenance.get("research_status") == RESEARCH_STATUS
        and provenance.get("human_review_complete") is False,
        "Adopted capture must retain agent exploratory provenance",
    )
    if expected_freeze is not None:
        freeze = Path(expected_freeze).resolve()
        require(
            provenance.get("source_bindings", {}).get(str(freeze)) == digest(freeze),
            "Capture must bind the exact planned execution freeze",
        )
    for name in ("rows.json", "features.npz"):
        require(
            manifest.get("output_sha256", {}).get(name) == digest(root / name),
            f"Changed capture {name}",
        )
    return {
        "completed_rows": manifest["completed_rows"],
        "manifest_sha256": digest(root / "capture_manifest.json"),
    }


def wait_for_capture(
    directory,
    *,
    timeout,
    poll,
    stop_file,
    validator,
    sleeper=time.sleep,
    clock=time.monotonic,
    progress=print,
):
    """Observe an external run only; unknown/interrupted states never launch a retry."""
    started, last = clock(), None
    while True:
        if Path(stop_file).exists():
            raise InterruptedError("Stop file observed; external capture is left running")
        manifest = read(Path(directory) / "manifest.json")
        status = manifest.get("status")
        if status == "complete":
            return validator()
        require(status == "running", f"Cannot adopt external capture status {status!r}; no retry")
        current = (status, manifest.get("completed_rows"))
        if current != last:
            progress(
                f"[{now()}] Waiting for existing Llama: {current[1]} completed rows", flush=True
            )
            last = current
        if clock() - started >= timeout:
            raise TimeoutError("Existing Llama capture wait expired; no process was restarted")
        sleeper(min(poll, max(0.01, timeout - (clock() - started))))


def command(script, *arguments):
    return [sys.executable, str(ROOT / "scripts" / script), *(str(value) for value in arguments)]


def build_plan(config):
    """Return a fixed complete plan before any child process or outcome inspection."""
    folder = Path(config["campaign"])
    bank, audit, initial = config["bank"], config["agent_audit"], config["training_freeze"]
    llama, qwen = Path(config["llama_train"]), Path(config["qwen_train"])
    monitor_dev = Path(config["monitor_development"])
    fits = {model: folder / f"{model}_fit" for model in ("llama", "qwen")}
    tests = {model: folder / f"{model}_test" for model in ("llama", "qwen")}
    test_freeze, monitor_test = folder / "test_freeze.json", folder / "monitor_test"
    stages = [{"name": "adopt_llama_train", "kind": "wait_llama", "directory": str(llama)}]

    def add(name, script, *args, fresh=(), outputs=(), **metadata):
        stages.append(
            {
                "name": name,
                "kind": "command",
                "command": command(script, *args),
                "fresh": [str(path) for path in fresh],
                "outputs": [str(path) for path in outputs],
                **metadata,
            }
        )

    def audit_capture(model, split, directory):
        report = folder / f"{model}_{split}_audit.json"
        add(
            f"audit_{model}_{split}",
            "audit_main_study_capture.py",
            "--capture",
            directory,
            "--output",
            report,
            fresh=(report,),
            outputs=(report,),
        )

    def capture_execute(model, split, directory, freeze, *, adoption=False):
        args = [
            "execute",
            "--output",
            directory,
            "--partition",
            split,
            "--review-ledger",
            audit,
            "--execution-freeze",
            freeze,
        ]
        if split == "test":
            args.extend(("--fit-dir", fits[model]))
        add(
            f"execute_{model}_{split}",
            "run_main_study.py",
            *args,
            outputs=(directory / "capture_manifest.json",),
            capture={"directory": str(directory), "model": model, "partition": split},
            adopt_prepared_or_complete=adoption,
        )

    audit_capture("llama", "train", llama)
    add(
        "fit_llama",
        "fit_main_study.py",
        "--bank",
        bank,
        "--capture",
        llama,
        "--output",
        fits["llama"],
        fresh=(fits["llama"],),
        outputs=(fits["llama"] / "settings.json", fits["llama"] / "heads.npz"),
    )
    capture_execute("qwen", "train_dev", qwen, initial, adoption=True)
    audit_capture("qwen", "train", qwen)
    add(
        "fit_qwen",
        "fit_main_study.py",
        "--bank",
        bank,
        "--capture",
        qwen,
        "--output",
        fits["qwen"],
        fresh=(fits["qwen"],),
        outputs=(fits["qwen"] / "settings.json", fits["qwen"] / "heads.npz"),
    )
    add(
        "freeze_test",
        "prepare_main_study_execution.py",
        "--bank",
        bank,
        "--agent-audit",
        audit,
        "--stage",
        "test",
        "--initial-freeze",
        initial,
        "--fit-dir",
        fits["llama"],
        "--fit-dir",
        fits["qwen"],
        "--output",
        test_freeze,
        fresh=(test_freeze,),
        outputs=(test_freeze,),
    )
    for model in ("llama", "qwen"):
        directory = tests[model]
        add(
            f"prepare_{model}_test",
            "run_main_study.py",
            "prepare",
            "--bank",
            bank,
            "--model",
            model,
            "--output",
            directory,
            fresh=(directory,),
            outputs=(directory / "protocol.json",),
        )
        capture_execute(model, "test", directory, test_freeze)
        audit_capture(model, "test", directory)
        result = folder / f"{model}_analysis_without_monitor.json"
        add(
            f"analyse_{model}_without_monitor",
            "analyse_main_study.py",
            "--capture",
            directory,
            "--fit",
            fits[model],
            "--output",
            result,
            fresh=(result,),
            outputs=(result,),
        )
    add(
        "execute_monitor_development",
        "run_main_study_monitor.py",
        "execute",
        "--output",
        monitor_dev,
        "--partition",
        "development",
        "--review-ledger",
        audit,
        "--execution-freeze",
        initial,
        outputs=(monitor_dev / "records.json",),
        monitor={"directory": str(monitor_dev), "partition": "development"},
        adopt_prepared_or_complete=True,
    )
    report = folder / "monitor_development_analysis.json"
    add(
        "analyse_monitor_development",
        "analyse_main_study_monitor.py",
        "--bank",
        bank,
        "--run",
        monitor_dev,
        "--output",
        report,
        fresh=(report,),
        outputs=(report,),
    )
    add(
        "prepare_monitor_test",
        "run_main_study_monitor.py",
        "prepare",
        "--bank",
        bank,
        "--output",
        monitor_test,
        fresh=(monitor_test,),
        outputs=(monitor_test / "protocol.json",),
    )
    add(
        "execute_monitor_test",
        "run_main_study_monitor.py",
        "execute",
        "--output",
        monitor_test,
        "--partition",
        "test",
        "--review-ledger",
        audit,
        "--execution-freeze",
        test_freeze,
        "--fit-dir",
        fits["llama"],
        outputs=(monitor_test / "records.json",),
        monitor={"directory": str(monitor_test), "partition": "test"},
    )
    report = folder / "monitor_test_analysis.json"
    add(
        "analyse_monitor_test",
        "analyse_main_study_monitor.py",
        "--bank",
        bank,
        "--run",
        monitor_test,
        "--output",
        report,
        fresh=(report,),
        outputs=(report,),
    )
    for model in ("llama", "qwen"):
        report = folder / f"{model}_analysis_with_monitor.json"
        add(
            f"analyse_{model}_with_monitor",
            "analyse_main_study.py",
            "--capture",
            tests[model],
            "--fit",
            fits[model],
            "--bank",
            bank,
            "--monitor-run",
            monitor_test,
            "--output",
            report,
            fresh=(report,),
            outputs=(report,),
        )
    return stages


def run_child(arguments, log_path, *, limit=LOG_LIMIT):
    """Drain all child output, retain a bounded log, and never invoke a shell."""
    environment = dict(os.environ, PYTHONUNBUFFERED="1")
    tail, written, discarded = deque(maxlen=1), 0, 0
    with Path(log_path).open("xb") as stream:
        with subprocess.Popen(
            arguments,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=False,
        ) as process:
            assert process.stdout is not None
            for block in iter(lambda: process.stdout.read1(65536), b""):
                tail.append(((tail[-1] if tail else b"") + block)[-4096:])
                retained = block[: max(0, limit - written)]
                stream.write(retained)
                stream.flush()
                written += len(retained)
                discarded += len(block) - len(retained)
            code = process.wait()
    return {
        "returncode": code,
        "log_bytes": written,
        "discarded_stdout_bytes": discarded,
        "output_tail": (tail[-1] if tail else b"").decode("utf-8", errors="replace"),
    }


def validate_prepared(directory, config, *, monitor=False, model_key=None):
    root = Path(directory)
    protocol = read(root / "protocol.json")
    manifest = read(root / ("run_manifest.json" if monitor else "manifest.json"))
    require(protocol.get("test_fixture") is False, "Fixture preparation cannot enter campaign")
    require(protocol.get("bank_sha256") == config["bank_sha256"], "Prepared bank differs")
    if model_key is not None:
        require(protocol.get("model_key") == model_key, "Prepared capture model differs")
    require(
        manifest.get("status") == "prepared_no_weights",
        "Unknown partial capture; no automatic retry",
    )
    require(
        not (root / "attempts").exists() and not (root / "row_journal").exists(),
        "Preparation already attempted",
    )
    require(
        manifest.get("protocol_sha256") == digest(root / "protocol.json"),
        "Prepared protocol changed",
    )
    verify_bindings(protocol["policy_sha256"])


def validate_monitor_complete(directory, *, partition, config):
    manifest = read(Path(directory) / "run_manifest.json")
    payload = read(Path(directory) / "records.json")
    require(
        manifest.get("status") == "complete" and payload.get("partition") == partition,
        "Incomplete monitor",
    )
    require(
        payload.get("bank_sha256") == config["bank_sha256"]
        and payload.get("test_fixture") is False,
        "Monitor identity differs",
    )
    require(
        payload.get("research_status") == RESEARCH_STATUS
        and payload.get("human_review_complete") is False,
        "Monitor research status differs",
    )
    require(
        manifest.get("records_sha256") == digest(Path(directory) / "records.json"),
        "Monitor records changed",
    )


def execute_stage(stage, config, *, child=run_child, wait=wait_for_capture):
    """One stage, with no retry branch and no shell interpolation."""
    if stage["kind"] == "wait_llama":
        return wait(
            stage["directory"],
            timeout=config["wait_timeout_seconds"],
            poll=config["poll_seconds"],
            stop_file=config["stop_file"],
            validator=lambda: validate_capture(
                stage["directory"],
                partition="train_dev",
                model_key="llama",
                bank_sha256=config["bank_sha256"],
                expected_freeze=config["training_freeze"],
            ),
        )
    for path in stage.get("fresh", []):
        require(
            not Path(path).exists(), f"Expected a fresh output, found {path}; no overwrite/retry"
        )
    if stage.get("adopt_prepared_or_complete"):
        spec = stage.get("capture") or stage["monitor"]
        root = Path(spec["directory"])
        status = read(root / ("run_manifest.json" if "monitor" in stage else "manifest.json")).get(
            "status"
        )
        if status == "complete":
            if "capture" in stage:
                validate_capture(
                    root,
                    partition=spec["partition"],
                    model_key=spec["model"],
                    bank_sha256=config["bank_sha256"],
                    expected_freeze=config["training_freeze"],
                )
            else:
                validate_monitor_complete(root, partition=spec["partition"], config=config)
            return {"adopted_complete": True, "new_model_calls": 0}
        validate_prepared(root, config, monitor="monitor" in stage, model_key=spec.get("model"))
    result = child(stage["command"], Path(config["campaign"]) / "logs" / f"{stage['name']}.log")
    require(
        result["returncode"] == 0,
        f"Child failed in {stage['name']}: exit {result['returncode']}; see saved log.\n{result['output_tail']}",
    )
    for path in stage.get("outputs", []):
        require(Path(path).is_file(), f"Stage returned success but output is missing: {path}")
    if "capture" in stage:
        spec = stage["capture"]
        validate_capture(
            spec["directory"],
            partition=spec["partition"],
            model_key=spec["model"],
            bank_sha256=config["bank_sha256"],
            expected_freeze=(
                Path(config["campaign"]) / "test_freeze.json"
                if spec["partition"] == "test"
                else config["training_freeze"]
            ),
        )
    if "monitor" in stage:
        spec = stage["monitor"]
        validate_monitor_complete(spec["directory"], partition=spec["partition"], config=config)
    return {**result, "output_sha256": {path: digest(path) for path in stage.get("outputs", [])}}


def run_campaign(config, *, executor=execute_stage):
    folder = Path(config["campaign"])
    require(not folder.exists(), "Campaign folder must be fresh; prior campaigns are never resumed")
    verify_bindings(config["source_sha256"])
    stages = build_plan(config)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "logs").mkdir()
    fresh_json(folder / "config.json", {**config, "stages": stages})
    state = {"schema": SCHEMA, "status": "running", "created_at": now(), "stages": []}
    write_state(folder / "state.json", state)
    try:
        for number, stage in enumerate(stages):
            if Path(config["stop_file"]).exists():
                raise InterruptedError(
                    "Stop file observed between stages; no new subprocess started"
                )
            verify_bindings(config["source_sha256"])
            record = {"name": stage["name"], "status": "running", "started_at": now()}
            state["stages"].append(record)
            write_state(folder / "state.json", state)
            print(f"[{now()}] Stage {number + 1}/{len(stages)}: {stage['name']}", flush=True)
            result = executor(stage, config)
            record.update(status="complete", finished_at=now(), result=result)
            write_state(folder / "state.json", state)
        state.update(status="complete", finished_at=now())
    except BaseException as error:
        state.update(
            status="stopped" if isinstance(error, InterruptedError) else "failed_or_interrupted",
            stopped_at=now(),
            error_type=type(error).__name__,
            error=str(error),
        )
        if state["stages"] and state["stages"][-1]["status"] == "running":
            state["stages"][-1].update(status="failed_or_interrupted", stopped_at=now())
        write_state(folder / "state.json", state)
        raise
    write_state(folder / "state.json", state)
    print(f"[{now()}] Campaign complete; results remain agent-audited exploratory.", flush=True)
    return state


def make_config(args):
    require(
        sys.platform == "linux", "Launch this campaign with the existing WSL Python environment"
    )
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        require(os.environ.get(name) == "1", f"{name}=1 is required; no downloads during campaign")
    root_run, folder = args.root_run.resolve(), args.campaign.resolve()
    require(
        folder.is_relative_to(root_run) and folder != root_run,
        "Campaign must lie inside the declared native run root",
    )
    require(
        not str(root_run).startswith("/mnt/") and "OneDrive" not in root_run.parts,
        "Use native WSL storage",
    )
    paths = {
        name: str(getattr(args, name).resolve())
        for name in (
            "llama_train",
            "qwen_train",
            "monitor_development",
            "training_freeze",
            "agent_audit",
            "bank",
        )
    }
    for key in ("llama_train", "qwen_train", "monitor_development", "training_freeze"):
        require(
            Path(paths[key]).is_relative_to(root_run),
            f"{key} must use the declared native run root",
        )
    require(
        len({paths[key] for key in ("llama_train", "qwen_train", "monitor_development")}) == 3,
        "Initial run directories must be distinct",
    )
    freeze = read(paths["training_freeze"])
    require(
        freeze.get("schema") == "main-study-agent-exploratory-freeze-v1"
        and freeze.get("stage") == "train_dev"
        and freeze.get("status") == "frozen",
        "Expected initial agent exploratory freeze",
    )
    require(
        freeze.get("research_status") == RESEARCH_STATUS
        and freeze.get("human_review_complete") is False,
        "Initial freeze status differs",
    )
    require(
        freeze.get("agent_audit_path") == paths["agent_audit"]
        and freeze.get("agent_audit_sha256") == digest(paths["agent_audit"]),
        "Initial audit differs",
    )
    bindings = {
        **freeze["source_sha256"],
        paths["training_freeze"]: digest(paths["training_freeze"]),
    }
    require(
        bindings.get(paths["bank"]) == digest(paths["bank"]),
        "Campaign bank must be bound in the initial freeze",
    )
    for name in SCRIPTS:
        path = str(ROOT / "scripts" / name)
        bindings[path] = digest(path)
    for key in ("llama_train", "qwen_train", "monitor_development"):
        protocol = Path(paths[key]) / "protocol.json"
        bindings[str(protocol)] = digest(protocol)
    for key, expected in (("llama_train", "llama"), ("qwen_train", "qwen")):
        require(
            read(Path(paths[key]) / "protocol.json").get("model_key") == expected,
            f"{key} model differs",
        )
    require(
        args.wait_timeout_seconds > 0 and 0 < args.poll_seconds <= 60,
        "Positive bounded wait/poll times required",
    )
    return {
        "schema": SCHEMA,
        "created_at": now(),
        "root_run": str(root_run),
        "campaign": str(folder),
        **paths,
        "bank_sha256": freeze["bank_sha256"],
        "source_sha256": bindings,
        "wait_timeout_seconds": args.wait_timeout_seconds,
        "poll_seconds": args.poll_seconds,
        "stop_file": str(folder / "STOP"),
        "python": sys.executable,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "variables": {
                key: os.environ.get(key)
                for key in (
                    "HF_HOME",
                    "HF_HUB_CACHE",
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "TOKENIZERS_PARALLELISM",
                )
            },
        },
        "research_status": RESEARCH_STATUS,
        "human_review_complete": False,
        "policy": "Fixed full baseline campaign; sequential subprocesses; no retries or adaptive selection",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-run", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--llama-train", type=Path, required=True)
    parser.add_argument("--qwen-train", type=Path, required=True)
    parser.add_argument("--monitor-development", type=Path, required=True)
    parser.add_argument("--training-freeze", type=Path, required=True)
    parser.add_argument("--agent-audit", type=Path, required=True)
    parser.add_argument(
        "--bank", type=Path, default=ROOT / "data/belief_main/main_v1/main_bank_v1.json"
    )
    parser.add_argument("--wait-timeout-seconds", type=float, default=21600)
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args()
    run_campaign(make_config(args))


if __name__ == "__main__":
    main()
