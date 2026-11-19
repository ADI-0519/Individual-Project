"""Share one cached-model load across pre-report and sender-policy diagnostics."""

import argparse
import time
from dataclasses import replace
from pathlib import Path

from run_prereport_diagnostic import run_prereport_diagnostic
from sender_policy_ablation import run_sender_policy_ablation

from latent_probes.backends import HFBackend
from latent_probes.config import load_config
from latent_probes.storage import atomic_json, environment_info, file_sha256, utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.run, args.output = args.run.resolve(), args.output.resolve()
    if args.output.is_relative_to(args.run):
        raise ValueError("Diagnostic output must be outside the original frozen run.")
    if args.output.exists():
        raise FileExistsError("Use a new diagnostic output directory.")
    args.output.mkdir(parents=True)
    manifest = {
        "schema": "next_research_checks_v1",
        "status": "preparing",
        "created_at": utc_now(),
        "environment": environment_info(),
        "config_sha256": file_sha256(args.config),
        "source_manifest_sha256": file_sha256(args.run / "manifest.json"),
        "script_sha256": file_sha256(__file__),
        "interpretation": "exploratory development checks, not main-study completion",
    }
    atomic_json(args.output / "manifest.json", manifest)
    backend = None
    started = time.perf_counter()
    try:
        print("Loading cached pinned model for next research checks", flush=True)
        backend = HFBackend(replace(load_config(args.config).model, local_files_only=True))
        manifest.update(
            status="running", backend=backend.metadata, load_seconds=time.perf_counter() - started
        )
        atomic_json(args.output / "manifest.json", manifest)
        print("Capturing prefix-only pre-report features", flush=True)
        run_prereport_diagnostic(args.run, args.output / "prereport", backend)
        print("Running one-sentence sender-policy ablation on training cases", flush=True)
        run_sender_policy_ablation(args.run, args.output / "sender_policy", backend)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if backend is not None:
            backend.close()
        manifest.update(finished_at=utc_now(), elapsed_seconds=time.perf_counter() - started)
        atomic_json(args.output / "manifest.json", manifest)
    print(f"Completed exploratory checks: {args.output}", flush=True)


if __name__ == "__main__":
    main()
