"""Role/value factorial validity and fixed-reader-prefix checks; no model calls."""

import copy
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from latent_probes.calibrated_reader import worked_examples
from latent_probes.evidence_role_panel import authored_panel, jobs, load_panel, validate_panel
from latent_probes.forecast_panel import READER_INSTRUCTION

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def panel():
    return load_panel(ROOT / "data/communication/evidence_role_v1.json")


def test_exactly_96_fixed_role_value_conditions_and_balanced_controls(panel):
    assert panel == authored_panel()
    planned = jobs(panel)
    assert len(planned) == len({j["example_id"] for j in planned}) == 96
    assert Counter(j["family"] for j in planned) == {"date": 48, "score": 48}
    assert Counter(j["expected_decision"] for j in planned) == {
        "APPROVE": 16,
        "REJECT": 16,
        "DEFER": 64,
    }
    for case in panel["cases"]:
        members = [j for j in planned if j["case_id"] == case["id"]]
        assert len({(j["condition"], j["side"]) for j in members}) == 6


def test_original_nine_shot_prefix_and_source_visibility_are_exact(panel):
    expected = [
        {"role": "system", "content": READER_INSTRUCTION},
        *[m for e in worked_examples() for m in e["messages"]],
    ]
    for job in jobs(panel):
        assert job["messages"][:-1] == expected
        actual = job["messages"][-1]["content"]
        assert job["case_id"] not in actual
        assert "expected_decision" not in actual and "irrelevant" not in actual
        assert re.search(r"\b(low|high)\b", actual) is None
        assert actual.endswith("No report was supplied.")


def test_metadata_prompts_admit_two_opposite_completion_or_score_worlds(panel):
    # These are countermodels, not hidden labels supplied to a real reader.
    # Generic administrative registration can occur before OR after completion;
    # no such causal/temporal link or score/file-ID link is an authored rule.
    for case in panel["cases"]:
        for side in ("low", "high"):
            metadata = case["conditions"][f"metadata/{side}"]
            explicit = case["conditions"][f"explicit_missing/{side}"]
            assert metadata["expected_decision"] == explicit["expected_decision"] == "DEFER"
            assert explicit["evidence_packet"][:-1] == metadata["evidence_packet"]
            if case["family"] == "date":
                registration = date.fromisoformat(case[side])
                completion_a, completion_b = (
                    date.fromisoformat(case["low"]),
                    date.fromisoformat(case["high"]),
                )
                deadline = date.fromisoformat(case["boundary"])
                assert completion_a <= deadline < completion_b
                assert isinstance(registration, date)  # Same registration in both countermodels.
                assert "administrative file registration date" in metadata["evidence_packet"][0]
                assert "completion file" not in metadata["evidence_packet"][0]
            else:
                file_id = case[side]
                assert case["low"] < case["boundary"] <= case["high"]
                assert str(file_id) in metadata["evidence_packet"][0]
                assert "evaluation score" not in metadata["evidence_packet"][0]


def test_low_high_change_only_the_targeted_value_within_each_role(panel):
    for case in panel["cases"]:
        for condition in ("decisive", "metadata", "explicit_missing"):
            low = case["conditions"][f"{condition}/low"]
            high = case["conditions"][f"{condition}/high"]
            assert low["rule"] == high["rule"]
            assert low["evidence_packet"][0].replace(str(case["low"]), "<VALUE>") == high[
                "evidence_packet"
            ][0].replace(str(case["high"]), "<VALUE>")
            if condition == "decisive":
                assert {low["expected_decision"], high["expected_decision"]} == {
                    "APPROVE",
                    "REJECT",
                }
            else:
                assert low["expected_decision"] == high["expected_decision"] == "DEFER"


def test_new_entities_dates_and_integer_instantiations_are_disjoint_from_prior_banks(panel):
    entities = {c["entity"] for c in panel["cases"]}
    values = {v for c in panel["cases"] for k, v in c.items() if k in {"boundary", "low", "high"}}
    for name in ("fresh_transfer_v1", "fresh_calibration_v1", "fresh_recipient_test_v1"):
        previous = json.loads((ROOT / f"data/communication/{name}.json").read_text())
        assert not entities & {c["world"]["entity"] for c in previous["cases"]}
        assert not values & {v for c in previous["cases"] for v in c["world"].values()}
    old = json.loads((ROOT / "data/reader_forecast/development_v1.json").read_text())
    assert not entities & {c["world"]["entity"] for c in old["cases"]}
    assert not values & {v for c in old["cases"] for v in c["world"].values()}


@pytest.mark.parametrize("corruption", ["role", "label", "missing", "boundary", "policy"])
def test_role_fact_label_or_policy_changes_are_rejected(panel, corruption):
    changed = copy.deepcopy(panel)
    case = changed["cases"][0]
    if corruption == "role":
        case["conditions"]["metadata/low"]["evidence_packet"][0] = "The completion date is known."
    elif corruption == "label":
        case["conditions"]["metadata/low"]["expected_decision"] = "APPROVE"
    elif corruption == "missing":
        case["conditions"]["explicit_missing/low"]["evidence_packet"].pop()
    elif corruption == "boundary":
        case["high"] = case["low"]
    else:
        changed["reader_policy_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_panel(changed)
