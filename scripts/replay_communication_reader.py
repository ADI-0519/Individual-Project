"""Frozen-report cross-reader replication with a prespecified capability gate."""

from __future__ import annotations

import argparse
import copy
import traceback
from pathlib import Path

from run_communication_transfer import READER_SETTINGS, strict_decision, summarize

from latent_probes.backends import HFBackend
from latent_probes.communication_panel import build_reader_messages
from latent_probes.config import load_config
from latent_probes.storage import (
    atomic_json,
    canonical_hash,
    environment_info,
    file_sha256,
    read_json,
    utc_now,
)

GATE = {"informed": 23, "absent": 20, "stale": 20, "max_invalid": 1}


def capability_gate(rows):
    result = {
        c: {
            "n": sum(r["context"] == c for r in rows),
            "correct": sum(
                r["context"] == c and r["decision"] == r["expected_decision"] for r in rows
            ),
        }
        for c in ("informed", "absent", "stale")
    }
    invalid = sum(r["decision"] == "INVALID" for r in rows)
    passed = (
        all(result[c]["n"] == 24 and result[c]["correct"] >= GATE[c] for c in result)
        and invalid <= GATE["max_invalid"]
    )
    return {"passed": passed, "by_context": result, "invalid": invalid, "requirements": GATE}


def run(args, backend_factory=HFBackend):
    source, output = args.source, args.output
    if output.exists():
        raise FileExistsError("Preserve all independent-reader attempts")
    original = read_json(source / "manifest.json")
    protocol = read_json(source / "protocol.json")
    if (
        original["status"] != "complete"
        or file_sha256(source / "protocol.json") != original["protocol_sha256"]
    ):
        raise ValueError("A complete immutable primary run is required")
    source_reports = [(p, read_json(p)) for p in sorted((source / "reports").glob("*.json"))]
    source_controls = read_json(source / "controls.json")
    source_replays = read_json(source / "replays.json")
    if len(source_reports) != 144 or len(source_controls) != 72 or len(source_replays) != 288:
        raise ValueError("Unexpected primary run dimensions")
    cases = {c["id"]: c for c in protocol["panel"]["cases"]}
    if {(r["case_id"], r["context"]) for r in source_controls} != {
        (c, x) for c in cases for x in ("informed", "absent", "stale")
    }:
        raise ValueError("Original controls must cover 72 unique case/context pairs")
    settings = load_config(args.config).model
    if (
        settings.model_id != "Qwen/Qwen2.5-7B-Instruct"
        or settings.revision != "a09a35458c702b33eeacc393d103063234e8bc28"
    ):
        raise ValueError("This predeclared independent checkpoint must remain fixed")
    snapshots = {
        str(p): file_sha256(p)
        for p in [
            source / "manifest.json",
            source / "protocol.json",
            source / "controls.json",
            source / "replays.json",
            *[p for p, _ in source_reports],
        ]
    }
    output.mkdir(parents=True)
    plan = {
        "schema": "independent_reader_communication_v1",
        "created_utc": utc_now(),
        "selection": "One independent checkpoint chosen after observing Llama reader-control failures, before Qwen outputs; no model search",
        "model": settings.__dict__,
        "reader_settings": READER_SETTINGS.__dict__,
        "seed": 42,
        "gate": GATE,
        "source_hashes": snapshots,
        "script_sha256": file_sha256(__file__),
        "scorer_source_sha256": file_sha256(
            Path(__file__).with_name("run_communication_transfer.py")
        ),
        "plan": "72 unchanged no-report controls; if gate passes,144 unchanged reports and288 same-report condition rows with exact prompt cache",
        "limits": [
            "Observed source bank and saved reports; independent reader behavior, not new untouched cases",
            "Generator, original forecaster and five readouts remain frozen and are not told the new reader identity",
            "No new activation capture, probe fit, threshold change or source report regeneration",
            "Cross-reader outcome transfer does not establish original sender knowledge of a later reader identity",
            "One selected reader; no further checkpoint search if gate fails",
        ],
    }
    atomic_json(output / "protocol.json", plan)
    manifest = {
        "status": "prepared",
        "created_utc": utc_now(),
        "protocol_sha256": file_sha256(output / "protocol.json"),
        "environment": environment_info(),
        "new_probe_fits": 0,
        "new_sender_reports": 0,
    }
    atomic_json(output / "manifest.json", manifest)
    backend = None
    cache, controls, rows, replays = {}, [], [], []
    try:
        backend = backend_factory(settings)
        if backend.metadata.get("is_test_fixture") is not False:
            raise ValueError("A genuine independent reader is required")
        if (
            backend.metadata["model_id"] != settings.model_id
            or backend.metadata["resolved_revision"] != settings.revision
        ):
            raise ValueError("Independent reader checkpoint identity changed")
        manifest.update(status="running", backend=backend.metadata)
        atomic_json(output / "manifest.json", manifest)

        def reader_call(case, report, context):
            messages = build_reader_messages(case, report, context)
            key = canonical_hash([messages, READER_SETTINGS.__dict__, 42])
            if key not in cache:
                generation = backend.generate(messages, READER_SETTINGS, 42).record()
                cache[key] = {
                    "messages": messages,
                    "generation": generation,
                    "decision": strict_decision(generation),
                    "prompt_sha256": canonical_hash(messages),
                }
                atomic_json(output / "reader_calls" / f"{key}.json", cache[key])
            return cache[key], key

        for c in source_controls:
            reader, key = reader_call(cases[c["case_id"]], "", c["context"])
            controls.append(
                {
                    **c,
                    "source_reader_decision": c["decision"],
                    "decision": reader["decision"],
                    "reader_call": key,
                }
            )
        gate = capability_gate(controls)
        atomic_json(output / "controls.json", controls)
        atomic_json(output / "gate.json", gate)
        print(f"Independent reader capability gate: {gate}", flush=True)
        if not gate["passed"]:
            if not all(file_sha256(Path(p)) == h for p, h in snapshots.items()):
                raise ValueError("Primary source changed during control validation")
            manifest.update(
                status="reader_controls_failed",
                reader_gate=gate,
                n_reader_calls=len(cache),
                source_unchanged=True,
            )
            return
        for i, (path, old) in enumerate(source_reports, 1):
            row = copy.deepcopy(old)
            reader, key = reader_call(cases[row["case_id"]], row["report_text"], row["context"])
            row.update(
                source_reader_decision=old["decision"],
                decision=reader["decision"],
                reader_call=key,
                source_report_sha256=file_sha256(path),
            )
            assert (
                row["predictions"] == old["predictions"]
                and row["probabilities"] == old["probabilities"]
            )
            row["predictions"]["source_reader_no_report"] = old["predictions"]["no_report"]
            row["predictions"]["no_report"] = next(
                c["decision"]
                for c in controls
                if c["case_id"] == row["case_id"] and c["context"] == row["context"]
            )
            row["record_sha256"] = canonical_hash(
                {k: v for k, v in row.items() if k != "record_sha256"}
            )
            atomic_json(output / "reports" / path.name, row)
            rows.append(row)
            print(
                f"Independent reader report {i}/144 {row['case_id']} {row['objective']} {row['context']} -> {row['decision']}",
                flush=True,
            )
        for replay in source_replays:
            reader, key = reader_call(cases[replay["case_id"]], replay["report"], replay["context"])
            replays.append(
                {
                    **replay,
                    "source_reader_decision": replay["decision"],
                    "decision": reader["decision"],
                    "reader_call": key,
                }
            )
        atomic_json(output / "replays.json", replays)
        atomic_json(output / "summary.json", summarize(rows, replays, controls))
        if not all(file_sha256(Path(p)) == h for p, h in snapshots.items()):
            raise ValueError("Primary source artifacts changed")
        manifest.update(
            status="complete",
            reader_gate=gate,
            n_reports=len(rows),
            n_replay_rows=len(replays),
            n_reader_calls=len(cache),
            source_unchanged=True,
        )
    except BaseException:
        manifest.update(status="failed", failure=traceback.format_exc())
        raise
    finally:
        if backend is not None:
            backend.close()
        manifest["finished_utc"] = utc_now()
        atomic_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args())
