"""Verify that attaching a completed prose monitor preserves every other result."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def compare_sections(before: dict, after: dict) -> list[str]:
    if set(before) != set(after) or "prose_monitor" not in before:
        raise ValueError("Analysis field inventory changed")
    if before["prose_monitor"].get("status") != "not_provided":
        raise ValueError("Expected the original analysis without the prose monitor")
    if after["prose_monitor"].get("status") != "complete":
        raise ValueError("Integrated prose monitor is not complete")
    retained = sorted(set(before) - {"prose_monitor"})
    for key in retained:
        if before[key] != after[key]:
            raise ValueError(f"Previously reported analysis changed: {key}")
    return retained


def compare_files(before_path: Path, after_path: Path) -> dict:
    original, integrated = before_path.read_bytes(), after_path.read_bytes()
    fields = compare_sections(json.loads(original), json.loads(integrated))
    return {
        "before": {"path": str(before_path), "sha256": hashlib.sha256(original).hexdigest()},
        "after": {"path": str(after_path), "sha256": hashlib.sha256(integrated).hexdigest()},
        "identical_fields": fields,
        "only_changed_field": "prose_monitor",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing audits; output already exists")
    comparisons = {
        model: compare_files(
            args.campaign / f"{model}_analysis_without_monitor.json",
            args.campaign / f"{model}_analysis_with_monitor.json",
        )
        for model in ("llama", "qwen")
    }
    report = {
        "schema": "main-study-monitor-integration-audit-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "passed",
        "reviewer_type": "agent",
        "research_status": "agent_audited_exploratory",
        "human_review_complete": False,
        "confirmatory_result": False,
        "auditor_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "comparisons": comparisons,
        "new_model_calls": 0,
        "new_fits_or_selections": 0,
        "limitations": "Structural result equality, not new inference or human semantic review.",
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("Both integrations preserve every original field except prose_monitor.")


if __name__ == "__main__":
    main()
