"""Post-hoc development diagnostic of the frozen net-profit reader failure.

This is not another held-out experiment, a tuned replacement for the original
panel, or a test of sender intent. The plan is written before model generation.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from latent_probes.backends import HFBackend
from latent_probes.config import GenerationSettings, load_config
from latent_probes.reader_panel import build_reader_messages, load_panel, parse_decision
from latent_probes.storage import atomic_json, canonical_hash, environment_info, utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel = load_panel(args.panel)
    case = next(case for case in panel["cases"] if case["id"] == "net-profit")
    jobs = []
    for context in ("absent", "stale"):
        for budget in (8, 64):
            jobs.append(
                {
                    "id": f"unchanged_{context}_budget{budget}",
                    "messages": build_reader_messages(case, context, "restored"),
                    "max_new_tokens": budget,
                    "expected_decision": case["correct"],
                }
            )
        explicit = build_reader_messages(case, context, "restored")
        explicit[-1]["content"] += " Net profit is -15000."
        jobs.append(
            {
                "id": f"explicit_computed_net_{context}",
                "messages": explicit,
                "max_new_tokens": 8,
                "expected_decision": case["correct"],
            }
        )
    complete = build_reader_messages(case, "informed", "none")
    jobs.append(
        {
            "id": "repeat_informed_no_report",
            "messages": complete,
            "max_new_tokens": 8,
            "expected_decision": case["correct"],
        }
    )
    redistributed = build_reader_messages(case, "absent", "restored")
    redistributed[-1]["content"] = (
        "Decision rule:\n"
        + case["rule"]
        + "\n\nEvidence packet:\nQuarterly revenue:120000."
        + "\n\nReport:\n"
        + case["report_versions"]["restored"]
    )
    jobs.append(
        {
            "id": "remove_packet_no_cost_phrase",
            "messages": redistributed,
            "max_new_tokens": 8,
            "expected_decision": case["correct"],
        }
    )
    jobs.append(
        {
            "id": "standalone_subtraction",
            "messages": [
                {
                    "role": "user",
                    "content": "What is 120000 minus 135000? Reply with the number only.",
                }
            ],
            "max_new_tokens": 16,
            "expected_numeric_answer": -15000,
        }
    )
    if args.output.exists():
        raise FileExistsError("Choose a fresh diagnostic directory.")
    args.output.mkdir(parents=True)
    manifest = {
        "schema": "posthoc_reader_failure_v1",
        "status": "preparing",
        "created_at": utc_now(),
        "environment": environment_info(),
        "selected_after_observing_original_pilot": True,
        "original_panel_sha256": canonical_hash(panel),
        "plan_sha256": canonical_hash(jobs),
        "interpretation": "development diagnosis, not independent validation or definitive mechanism",
    }
    atomic_json(args.output / "plan.json", jobs)
    atomic_json(args.output / "manifest.json", manifest)
    backend = None
    records = []
    try:
        settings = replace(load_config(args.config).model, local_files_only=True)
        backend = HFBackend(settings)
        manifest.update(status="running", backend=backend.metadata)
        atomic_json(args.output / "manifest.json", manifest)
        for job in jobs:
            generation = backend.generate(
                job["messages"],
                GenerationSettings(max_new_tokens=job["max_new_tokens"], do_sample=False),
                42,
                capture=False,
            )
            record = {
                **job,
                "generation": generation.record(),
                "parsed_decision": parse_decision(generation.text),
            }
            record["record_sha256"] = canonical_hash(record)
            atomic_json(args.output / "records" / (job["id"] + ".json"), record)
            records.append(record)
            print(job["id"], repr(generation.text), flush=True)
        atomic_json(args.output / "results.json", records)
        manifest.update(
            status="complete", n_records=len(records), records_sha256=canonical_hash(records)
        )
    except BaseException:
        manifest["status"] = "failed"
        raise
    finally:
        if backend is not None:
            backend.close()
        manifest["finished_at"] = utc_now()
        atomic_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
