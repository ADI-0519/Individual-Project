"""Create score-blind report review packets from a completed real communication run."""

from __future__ import annotations

import argparse
import random
import secrets
from pathlib import Path

from latent_probes.storage import atomic_json, canonical_hash, file_sha256, read_json, utc_now


def review_record(record, case, review_id):
    """Explicit whitelist: never export objectives, scores or actual reader answers."""
    return {
        "review_id": review_id,
        "rule": case["rule"],
        "source_facts": case["source"],
        "nominated_decisive_fact": case["decisive_fact"],
        "reader_starting_evidence": case["reader_packets"][record["context"]],
        "report": record["generation"]["text"],
        "finish_reason": record["generation"]["finish_reason"],
        "report_sha256": canonical_hash(record["generation"]["text"]),
        "human_review_required": True,
    }


def export_review(run_dir, output):
    run_dir, output = Path(run_dir), Path(output)
    if output.exists():
        raise FileExistsError("Preserve existing review exports and mappings")
    manifest = read_json(run_dir / "manifest.json")
    if manifest["status"] != "complete" or manifest["backend"]["is_test_fixture"]:
        raise ValueError("A completed genuine run is required")
    protocol = read_json(run_dir / "protocol.json")
    cases = {c["id"]: c for c in protocol["panel"]["cases"]}
    sources = [(p, read_json(p)) for p in sorted((run_dir / "reports").glob("*.json"))]
    if len(sources) != 144:
        raise ValueError("Expected all 144 natural reports")
    random.SystemRandom().shuffle(sources)
    groups = {"a": [], "b": [], "c": []}
    repeats = {"a": [], "b": [], "c": []}
    mapping = {}
    for i, (path, row) in enumerate(sources):
        review_id = "review_" + secrets.token_hex(10)
        group = "abc"[i % 3]
        record = review_record(row, cases[row["case_id"]], review_id)
        groups[group].append(record)
        mapping[review_id] = {
            "source_id": row["example_id"],
            "source_sha256": file_sha256(path),
            "group": group,
        }
        if row["objective"] == "favorable" and row["context"] == "stale":
            repeats["abc"[(i + 1) % 3]].append(record)
    output.mkdir(parents=True)
    atomic_json(output / "private_mapping.json", mapping)
    for group in groups:
        atomic_json(output / f"group_{group}.json", groups[group])
        atomic_json(output / f"independent_{group}.json", repeats[group])
    atomic_json(
        output / "manifest.json",
        {
            "schema": "score_blind_communication_review_v1",
            "created_utc": utc_now(),
            "source_manifest_sha256": file_sha256(run_dir / "manifest.json"),
            "source_protocol_sha256": file_sha256(run_dir / "protocol.json"),
            "export_script_sha256": file_sha256(__file__),
            "first_pass_counts": {g: len(v) for g, v in groups.items()},
            "independent_counts": {g: len(v) for g, v in repeats.items()},
            "human_review_required": True,
            "limits": "Text can reveal contextual clues; goal metadata and all scores/outcomes omitted. Agent labels are drafts.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_review(args.run, args.output)
