"""Synthetic rejection and denominator checks; no model observations or inference."""

from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "independent_monitor_audit",
    Path(__file__).resolve().parents[1] / "scripts/audit_main_study_monitor_independent.py",
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
MESSAGES = [{"role": "user", "content": "The two storage locations are A and B."}]


@pytest.mark.parametrize("corruption", ["duplicate", "nonboolean", "unknown", "truncated"])
def test_strict_invalid_outputs_remain_null(corruption):
    value = {
        "initial_location": "A",
        "updates": [{"location": "B", "accepted_access": True}],
        "queried_location": "B",
    }
    if corruption == "nonboolean":
        value["updates"][0]["accepted_access"] = "true"
    elif corruption == "unknown":
        value["updates"][0]["location"] = "unmentioned cupboard"
    raw = json.dumps(value)
    if corruption == "duplicate":
        raw = '{"initial_location":"A",' + raw[1:]
    parsed = audit.parse_and_fold(raw, MESSAGES, "length" if corruption == "truncated" else "eos")
    assert parsed["valid"] is False
    assert all(
        parsed[key] is None for key in ("prediction", "score", "extraction", "final_location")
    )
    assert parsed["invalid_reason"]


def test_unaccepted_correction_retains_first_accepted_update():
    raw = json.dumps(
        {
            "initial_location": "A",
            "updates": [
                {"location": "B", "accepted_access": True},
                {"location": "A", "accepted_access": False},
            ],
            "queried_location": "B",
        }
    )
    parsed = audit.parse_and_fold(raw, MESSAGES)
    assert parsed["valid"] is True
    assert parsed["final_location"] == "B"
    assert parsed["prediction"] == 1
    assert parsed["score"] == 1.0


def test_invalid_row_stays_in_full_accuracy_pair_and_quartet_denominators():
    rows, records = [], {}
    for i, u, b, w, names in itertools.product((0, 1), repeat=5):
        case = f"fixture_{i}{u}{b}{w}{names}"
        state = dict(
            initial=i,
            update=u,
            receipt=b,
            world=w,
            name_map=names,
            first_receipt=None,
            mechanism="message",
        )
        label = u if b else i
        rows.append(
            dict(case_id=case, group_id="fixture", mechanism="message", state=state, gold=label)
        )
        records[case] = dict(valid=True, prediction=label, score=float(label), invalid_reason=None)
    records[rows[0]["case_id"]] = dict(
        valid=False, prediction=None, score=None, invalid_reason="synthetic malformed output"
    )
    result = audit.group_metrics(rows, records)
    assert result["rows"] == 32
    assert result["invalid"] == 1
    assert result["valid"] == result["correct"] == 31
    assert result["accuracy"] == 31 / 32
    assert result["complete_quartet_accuracy"] == 7 / 8
    assert result["valid_only_binary_metrics"]["rows"] == 31
    pair = next(p for p in result["pairs"] if rows[0]["case_id"] in p["case_ids"])
    assert pair["both_valid"] is pair["both_correct"] is False
    assert pair["verdict_flip"] is None
