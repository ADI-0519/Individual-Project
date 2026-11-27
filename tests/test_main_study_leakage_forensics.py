"""Corruption checks for the independent leakage auditor; no model calls/refits."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "leakage_forensics",
    Path(__file__).resolve().parents[1] / "scripts/audit_main_study_leakage_forensics.py",
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def comparison():
    return {
        "training_n": 2,
        "development_ids": ["dev_1", "dev_2"],
        "arms": {
            arm: {
                "training_ids": ["train_1", "train_2"],
                "calibration": {"status": "complete", "source_ids": ["dev_1", "dev_2"]},
            }
            for arm in ("restricted", "full")
        },
    }


def check_memberships(value):
    AUDIT.check_memberships(
        value,
        {arm: ["train_1", "train_2"] for arm in ("restricted", "full")},
        {"train_1", "train_2"},
        ["dev_1", "dev_2"],
    )


def test_accepts_correct_memberships():
    check_memberships(comparison())


@pytest.mark.parametrize(
    "corruption",
    [
        "test_in_training",
        "dev_in_training",
        "test_in_calibration",
        "test_in_selection",
        "duplicate_training",
    ],
)
def test_rejects_membership_contamination(corruption):
    value = deepcopy(comparison())
    if corruption == "test_in_training":
        value["arms"]["full"]["training_ids"][0] = "test_1"
    elif corruption == "dev_in_training":
        value["arms"]["restricted"]["training_ids"][0] = "dev_1"
    elif corruption == "test_in_calibration":
        value["arms"]["full"]["calibration"]["source_ids"][0] = "test_1"
    elif corruption == "test_in_selection":
        value["development_ids"][0] = "test_1"
    else:
        value["arms"]["full"]["training_ids"][1] = "train_1"
    with pytest.raises(ValueError):
        check_memberships(value)


def test_rejects_scaler_contaminated_by_development():
    train = np.array([[1.0, 2.0], [3.0, 4.0]])
    AUDIT.check_scaler(train, train.mean(axis=0), train.std(axis=0))
    contaminated = np.vstack((train, [[100.0, -100.0]]))
    with pytest.raises(ValueError, match="selected-train-only"):
        AUDIT.check_scaler(train, contaminated.mean(axis=0), contaminated.std(axis=0))


def test_rejects_dev_only_vocabulary_and_idf():
    rows = [{"messages": [{"role": "user", "content": "alpha beta"}]}]
    vocabulary = {"user": 0, "alpha": 1, "beta": 2, "user alpha": 3, "alpha beta": 4}
    AUDIT.check_vocabulary(rows, vocabulary, np.ones(5))
    with pytest.raises(ValueError, match="Vocabulary"):
        AUDIT.check_vocabulary(rows, dict(vocabulary, devonly=5), np.ones(6))
    with pytest.raises(ValueError, match="IDF"):
        AUDIT.check_vocabulary(rows, vocabulary, np.full(5, 2.0))


def test_rejects_changed_source_bytes(tmp_path):
    source = tmp_path / "frozen.py"
    source.write_text("original", encoding="utf-8")
    bindings = {str(source): AUDIT.digest(source)}
    AUDIT.check_bindings(bindings)
    source.write_text("retuned", encoding="utf-8")
    with pytest.raises(ValueError, match="Changed bound input"):
        AUDIT.check_bindings(bindings)


def test_rejects_disguised_identical_context_across_splits():
    groups = [
        {
            "group_id": "train",
            "split": "train",
            "introduction": "With {sender} and {reader}.",
            "event_description": "A report.",
        },
        {
            "group_id": "test",
            "split": "test",
            "introduction": "With {sender} and {reader}.",
            "event_description": "A report.",
        },
    ]
    with pytest.raises(ValueError, match="Exact normalized"):
        AUDIT.contextual_overlap(groups)


def test_target_rule_ignores_world_name_map_and_supports_replacement():
    for world in (0, 1):
        for mapping in (0, 1):
            state = {
                "initial": 0,
                "update": 1,
                "first_receipt": 1,
                "receipt": 0,
                "world": world,
                "name_map": mapping,
                "mechanism": "correction",
            }
            assert AUDIT.gold(state) == 1
            state["receipt"] = 1
            assert AUDIT.gold(state) == 0
