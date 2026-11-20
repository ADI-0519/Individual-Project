"""Ensure arbitrary result metadata cannot leak through review exports."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "review_exporter", ROOT / "scripts/export_communication_review.py"
)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def test_score_blind_record_is_an_explicit_whitelist():
    case = {
        "rule": "RULE",
        "source": ["SOURCE"],
        "decisive_fact": "FACT",
        "reader_packets": {"stale": ["PACKET"]},
    }
    record = {
        "context": "stale",
        "generation": {"text": "REPORT", "finish_reason": "eos"},
        "objective": "HIDDEN_GOAL",
        "decision": "HIDDEN_OUTCOME",
        "apollo_raw_margin": 123456.0,
        "predictions": {"activation": "HIDDEN_PREDICTION"},
        "case_id": "HIDDEN_CASE",
    }
    result = exporter.review_record(record, case, "random-id")
    rendered = json.dumps(result)
    assert all(x not in rendered for x in ("HIDDEN_", "123456", "predictions", "objective"))
    assert result["report"] == "REPORT"
    assert result["reader_starting_evidence"] == ["PACKET"]
    assert result["human_review_required"] is True
