"""Run one fixed new study sequentially; preserve failures, never retry a row."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from latent_probes.format_adaptation_capture import fresh_json, native, require, verify_test_seal
from latent_probes.format_adaptation_fit import validate_bundle
from latent_probes.main_study_run import ROOT, snapshot_files
from latent_probes.storage import atomic_json, file_sha256, read_json, utc_now


def commands(output):
    python = sys.executable
    stages = []
    for key in ("llama", "qwen"):
        stages.append(
            (
                key + "_fit_capture",
                [
                    python,
                    str(ROOT / "scripts/run_format_adaptation_capture.py"),
                    "capture",
                    "--output",
                    str(output / key),
                    "--partition",
                    "fit",
                ],
            )
        )
    for key in ("llama", "qwen"):
        protocol = read_json(output / key / "protocol.json")
        stages.append(
            (
                key + "_fit",
                [
                    python,
                    str(ROOT / "scripts/fit_format_adaptation.py"),
                    "--capture",
                    str(output / key / "fit"),
                    "--old-bundle",
                    protocol["old_bundle"],
                    "--model-key",
                    key,
                    "--output",
                    str(output / (key + "_fit")),
                ],
            )
        )
    stages.append(("seal_both_fits", None))
    for key in ("llama", "qwen"):
        stages.append(
            (
                key + "_test_capture",
                [
                    python,
                    str(ROOT / "scripts/run_format_adaptation_capture.py"),
                    "capture",
                    "--output",
                    str(output / key),
                    "--partition",
                    "test",
                    "--seal",
                    str(output / "test_seal.json"),
                ],
            )
        )
    for key in ("llama", "qwen"):
        stages.append(
            (
                key + "_analysis",
                [
                    python,
                    str(ROOT / "scripts/analyse_format_adaptation.py"),
                    "--capture",
                    str(output / key / "test"),
                    "--fit",
                    str(output / (key + "_fit")),
                    "--seal",
                    str(output / "test_seal.json"),
                    "--output",
                    str(output / (key + "_analysis")),
                ],
            )
        )
    stages.append(
        (
            "independent_audit",
            [
                python,
                str(ROOT / "scripts/audit_format_adaptation.py"),
                "--campaign",
                str(output),
                "--output",
                str(output) + "_audit",
            ],
        )
    )
    stages.append(
        (
            "report",
            [
                python,
                str(ROOT / "scripts/report_format_adaptation.py"),
                "--campaign",
                str(output),
                "--audit",
                str(output) + "_audit",
                "--output",
                str(output) + "_report",
            ],
        )
    )
    return stages


def seal_both(output, preparation):
    models = {}
    for key in ("llama", "qwen"):
        capture = read_json(output / key / "fit/manifest.json")
        require(
            capture["status"] == "complete" and capture["completed_rows"] == 768,
            "Both training captures must finish before sealing",
        )
        settings, _arrays = validate_bundle(output / (key + "_fit"))
        require(
            settings["model_key"] == key
            and settings["site"]
            == {"llama": "layer_16_query_location_last", "qwen": "layer_21_query_location_last"}[
                key
            ],
            "Validated fit bundle belongs to a different model slot/site",
        )
        bundle = output / (key + "_fit")
        models[key] = {
            "coefficients_sha256": file_sha256(bundle / "coefficients.npz"),
            "settings_sha256": file_sha256(bundle / "settings.json"),
            "fit_manifest_sha256": file_sha256(bundle / "fit_manifest.json"),
        }
        require(not (output / key / "test").exists(), "Test attempt predates both-model freeze")
    snapshot_files(preparation["source_sha256"])
    fresh_json(
        output / "test_seal.json",
        {
            "schema": "format-adaptation-test-seal-v1",
            "created_at": utc_now(),
            "models": models,
            "source_sha256": preparation["source_sha256"],
            "bank_sha256": preparation["bank_sha256"],
            "human_review_complete": False,
        },
    )
    for key in ("llama", "qwen"):
        verify_test_seal(output / key, output / "test_seal.json")


def run(output):
    output = native(output)
    preparation = read_json(output / "preparation.json")
    require(preparation["status"] == "prepared_no_weights", "Require complete preparation")
    snapshot_files(preparation["source_sha256"])
    for key in ("llama", "qwen"):
        require(
            file_sha256(output / key / "protocol.json")
            == preparation["model_protocol_sha256"][key],
            "Prepared model protocol changed",
        )
    fresh_json(
        output / "campaign_started.json",
        {
            "pid": os.getpid(),
            "started_at": utc_now(),
            "preparation_sha256": file_sha256(output / "preparation.json"),
        },
    )
    state = {
        "schema": "format-adaptation-campaign-state-v1",
        "status": "running",
        "started_at": utc_now(),
        "pid": os.getpid(),
        "stages": [],
    }
    atomic_json(output / "campaign_state.json", state)
    environment = {
        **os.environ,
        "HF_HOME": str(ROOT / ".cache/huggingface-wsl"),
        "HF_HUB_CACHE": "/home/adirj/kcl-hf-cache",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OPENBLAS_NUM_THREADS": "4",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4",
        "PYTHONUNBUFFERED": "1",
    }
    try:
        for name, command in commands(output):
            snapshot_files(preparation["source_sha256"])
            entry = {"name": name, "status": "running", "started_at": utc_now(), "command": command}
            state["stages"].append(entry)
            atomic_json(output / "campaign_state.json", state)
            if command is None:
                seal_both(output, preparation)
            else:
                with (
                    (output / (name + ".stdout.log")).open("xb") as stdout,
                    (output / (name + ".stderr.log")).open("xb") as stderr,
                ):
                    child = subprocess.Popen(
                        command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr
                    )
                    entry["child_pid"] = child.pid
                    atomic_json(output / "campaign_state.json", state)
                    while child.poll() is None:
                        atomic_json(
                            output / "worker_alive.json",
                            {
                                "at": utc_now(),
                                "pid": os.getpid(),
                                "child_pid": child.pid,
                                "stage": name,
                            },
                        )
                        time.sleep(5)
                    entry["returncode"] = child.returncode
                    require(child.returncode == 0, "Stage failed; preserve attempt: " + name)
            entry.update(status="complete", finished_at=utc_now())
            atomic_json(output / "campaign_state.json", state)
            print(name + " complete", flush=True)
        audit = read_json(Path(str(output) + "_audit") / "audit_receipt.json")
        require(audit["status"] == "pass", "Independent numerical audit did not pass")
        snapshot_files(preparation["source_sha256"])
        state.update(
            status="complete",
            finished_at=utc_now(),
            audit_receipt_sha256=file_sha256(Path(str(output) + "_audit") / "audit_receipt.json"),
        )
        atomic_json(output / "campaign_state.json", state)
    except BaseException as error:
        state.update(
            status="failed", failed_at=utc_now(), error_type=type(error).__name__, error=str(error)
        )
        if state["stages"]:
            state["stages"][-1]["status"] = "failed"
        atomic_json(output / "campaign_state.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
