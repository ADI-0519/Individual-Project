"""Synthetic mathematical fixtures exercise the independent auditor's rejection paths."""

from __future__ import annotations

import copy
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "standalone_heldout_auditor",
    Path(__file__).resolve().parents[1] / "scripts/audit_main_study_heldout_independent.py",
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture
def rows():
    result = []
    for initial, update, receipt, world, names in itertools.product((0, 1), repeat=5):
        state = {
            "group_id": "synthetic_fixture",
            "mechanism": "message",
            "initial": initial,
            "update": update,
            "receipt": receipt,
            "world": world,
            "name_map": names,
            "first_receipt": None,
        }
        result.append(
            {
                "case_id": f"fixture_{initial}{update}{receipt}{world}{names}",
                "group_id": state["group_id"],
                "mechanism": state["mechanism"],
                "state": state,
                "gold": update if receipt else initial,
            }
        )
    return result


def test_wrong_gold_label_is_rejected(rows):
    rows[0]["gold"] ^= 1
    with pytest.raises(ValueError, match="State/label disagreement"):
        audit.graph(rows)


def test_duplicate_factorial_state_is_rejected(rows):
    rows[-1] = copy.deepcopy(rows[0])
    with pytest.raises(ValueError, match="Duplicate factorial state"):
        audit.graph(rows)


def test_missing_factorial_state_is_rejected(rows):
    with pytest.raises(ValueError, match="Incomplete group"):
        audit.graph(rows[:-1])


@pytest.mark.parametrize("field", ["margin", "probability", "prediction", "case_id"])
def test_altered_saved_score_fails_closed(field):
    ids = ["negative_fixture", "positive_fixture"]
    record = {
        "scores": [
            {"case_id": ids[0], "margin": -1.0, "probability": 0.2689414213699951, "prediction": 0},
            {"case_id": ids[1], "margin": 1.0, "probability": 0.7310585786300049, "prediction": 1},
        ]
    }
    audit.verify_scores(ids, record, np.array([-1.0, 1.0]))
    if field == "case_id":
        record["scores"][0][field] = "unrecorded_case"
    else:
        record["scores"][0][field] += 0.1 if field != "prediction" else 1
    with pytest.raises(
        ValueError, match="Saved (score identity/order|margin|probability)|Raw prediction"
    ):
        audit.verify_scores(ids, record, np.array([-1.0, 1.0]))


def test_altered_accuracy_is_rejected(rows):
    verifier = audit.Verifier(rows)
    margins = 2 * verifier.labels - 1.0
    # For perfectly signed unit margins all classes are correct; the error
    # probability is 1/(1+e), so every squared residual is the same.
    saved = {
        "rows": 32,
        "correct": 32,
        "accuracy": 1.0,
        "recall_0": 1.0,
        "recall_1": 1.0,
        "auroc": 1.0,
        "brier": (1 / (1 + np.e)) ** 2,
    }
    verifier.binary(saved, list(range(32)), margins)
    saved["accuracy"] = 0.5
    with pytest.raises(ValueError, match="accuracy"):
        verifier.binary(saved, list(range(32)), margins)


@pytest.fixture
def constant_primary():
    seeds = [11, 23, 37, 53, 71]
    groups = [f"synthetic_{i:02d}" for i in range(30)]
    primary = {
        seed: {
            "restricted": dict.fromkeys(groups, 0.0),
            "full": dict.fromkeys(groups, 0.125),
        }
        for seed in seeds
    }
    # Every narrative has exactly the same difference, so every possible
    # whole-group bootstrap draw has mean 1/8, independently of the RNG.
    saved = {
        "restricted": 0.0,
        "full": 0.125,
        "full_minus_restricted": 0.125,
        "percentile_95_interval": [0.125, 0.125],
        "allocation_seeds": seeds,
        "bootstrap_draws": 10000,
        "bootstrap_seed": 20261121,
        "independent_scenario_groups": 30,
        "by_group": [
            {
                "group_id": group,
                "restricted_by_seed": [0.0] * 5,
                "full_by_seed": [0.125] * 5,
                "restricted_mean": 0.0,
                "full_mean": 0.125,
                "difference": 0.125,
            }
            for group in groups
        ],
        "by_seed": [
            {"seed": seed, "restricted": 0.0, "full": 0.125, "difference": 0.125} for seed in seeds
        ],
    }
    return primary, saved


def test_constant_group_effect_has_exact_bootstrap_interval(rows, constant_primary):
    primary, saved = constant_primary
    result, _, _, _ = audit.verify_primary(primary, saved, audit.Verifier(rows))
    assert result["percentile_95_interval"] == [0.125, 0.125]


@pytest.mark.parametrize(
    "field,value",
    [
        ("percentile_95_interval", [0.0, 0.25]),
        ("bootstrap_draws", 9999),
        ("bootstrap_seed", 0),
        ("independent_scenario_groups", 150),
        ("by_group", []),
    ],
)
def test_damaged_bootstrap_or_aggregation_is_rejected(rows, constant_primary, field, value):
    primary, saved = constant_primary
    saved[field] = value
    with pytest.raises(ValueError, match="Bootstrap|Primary aggregation inventory"):
        audit.verify_primary(primary, saved, audit.Verifier(rows))
