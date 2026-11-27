"""Analytic denominator checks for the independent saved-output auditor."""

import importlib.util
from itertools import product
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "adequacy_independent_audit",
    Path(__file__).resolve().parents[1] / "scripts/audit_cached_readout_adequacy_independent.py",
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def rows():
    result = []
    for i, u, b1, b2, w, m in product((0, 1), repeat=6):
        state = dict(initial=i, update=u, first_receipt=b1, receipt=b2, world=w, name_map=m)
        result.append(
            {
                "case_id": str((i, u, b1, b2, w, m)),
                "group_id": "fixture",
                "state": state,
                "gold": 1 - u if b2 else u if b1 else i,
            }
        )
    return result


def test_oracle_has_eight_complete_quartets_and_no_nuisance_flips():
    cohort = rows()
    scores = [
        {
            "case_id": r["case_id"],
            "prediction": r["gold"],
            "factor_predictions": [r["state"][n] for n in AUDIT.FACTORS],
        }
        for r in cohort
    ]
    result = AUDIT.metrics(cohort, scores, "fourfactor_lr")
    assert result["quartet_correct_cells_histogram"] == {"0": 0, "1": 0, "2": 0, "3": 0, "4": 8}
    assert result["factor_joint_accuracy"] == result["primary_quartet_accuracy"] == 1
    assert result["nuisance_verdict_flips"]["world"]["flips"] == 0
    assert result["receipt_pair_metrics"]["receipt"]["changing"]["pairs"] == 24
    assert result["receipt_pair_metrics"]["receipt"]["preserving"]["pairs"] == 8


def test_world_shortcut_has_expected_histogram_and_world_flips():
    cohort = rows()
    scores = [{"case_id": r["case_id"], "prediction": r["state"]["world"]} for r in cohort]
    result = AUDIT.metrics(cohort, scores, "direct_lr")
    assert result["row_accuracy"] == 0.5
    assert result["primary_quartet_accuracy"] == 0
    assert result["quartet_correct_cells_histogram"] == {"0": 0, "1": 4, "2": 0, "3": 4, "4": 0}
    assert result["nuisance_verdict_flips"]["world"]["flips"] == 32
    assert result["nuisance_verdict_flips"]["name_map"]["flips"] == 0
    with pytest.raises(ValueError, match="count/uniqueness"):
        AUDIT.metrics(cohort, scores[:-1], "direct_lr")
