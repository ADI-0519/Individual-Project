"""Run a genuine capture audit and frozen reader panel with one cached model load.

This is a local/CREATE research command, not an offline CI fixture. Original
capture/probe files are read only. Fresh outputs retain failures for inspection.
"""

from __future__ import annotations

import argparse
import time
import traceback
from dataclasses import replace
from pathlib import Path

from validate_full_baseline import run_validation

from latent_probes.backends import HFBackend
from latent_probes.config import load_config
from latent_probes.context_probe import run_context_probe
from latent_probes.reader_panel import load_panel, run_panel
from latent_probes.storage import atomic_json, environment_info, file_sha256, utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(
            "Choose a fresh output directory; existing research results stay frozen."
        )
    panel = load_panel(args.panel)
    config = load_config(args.capture / "config.json")
    settings = replace(config.model, local_files_only=True)
    args.output.mkdir(parents=True)
    manifest = {
        "schema": "combined_research_audit_v1",
        "status": "preparing",
        "created_at": utc_now(),
        "environment": environment_info(),
        "inputs": {
            str(path): file_sha256(path)
            for path in (
                args.capture / "manifest.json",
                args.probe / "manifest.json",
                args.panel,
                Path(__file__),
                Path(__file__).with_name("validate_full_baseline.py"),
            )
        },
        "limits": [
            "The instruction baseline has no audited behavioral-deception labels.",
            "Four authored development cases do not estimate population reader risk.",
            "The reader panel measures decisions, not sender intent or second-order beliefs.",
        ],
    }
    output_manifest = args.output / "manifest.json"
    atomic_json(output_manifest, manifest)
    backend = None
    start = time.perf_counter()
    try:
        print("Loading cached pinned model for genuine audit and reader panel", flush=True)
        backend = HFBackend(settings)
        manifest["load_seconds"] = time.perf_counter() - start
        manifest["backend"] = backend.metadata
        manifest["status"] = "running"
        atomic_json(output_manifest, manifest)
        print("Checking full-capture integrity and prespecified GPU replay subset", flush=True)
        run_validation(args.capture, args.probe, args.output / "gpu_validation", backend)
        print("Running frozen four-case reader panel (48 fresh conversations)", flush=True)
        run_panel(
            panel,
            args.output / "reader_panel",
            backend=backend,
            seed=panel["reader_seed"],
            tokenizer=backend.tokenizer,
        )
        print("Scoring 48 fixed reports under matched sender contexts", flush=True)
        run_context_probe(panel, args.output / "context_probe", args.probe, backend)
        manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "failed"
        manifest["failure"] = traceback.format_exc()
        raise
    finally:
        if backend is not None:
            backend.close()
        manifest["elapsed_seconds"] = time.perf_counter() - start
        manifest["finished_at"] = utc_now()
        atomic_json(output_manifest, manifest)
    print(f"Completed. Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
