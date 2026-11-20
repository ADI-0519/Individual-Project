"""Construct, gold, role, and information-boundary checks; no model calls."""

import copy
import itertools
import json
from collections import Counter
from pathlib import Path

import pytest

from latent_probes.belief_panel import TARGETS, build_panel, load_panel, validate_panel

DATA = Path(__file__).resolve().parents[1] / "data/belief_diagnostic/development_v1.json"


@pytest.fixture
def panel():
    return build_panel()


def test_size_template_splits_and_control_separation(panel):
    assert len(panel["cases"]) == 80
    assert Counter(case["split"] for case in panel["cases"]) == {
        "train": 32,
        "validation": 16,
        "test": 16,
        "control": 16,
    }
    prior = [case for case in panel["cases"] if case["phase"] == "prior"]
    assert len(prior) == 64
    by_split = {
        split: {case["template_id"] for case in prior if case["split"] == split}
        for split in ("train", "validation", "test")
    }
    assert [len(ids) for ids in by_split.values()] == [4, 2, 2]
    assert not any(a & b for a, b in itertools.combinations(by_split.values(), 2))
    assert len({case["id"] for case in panel["cases"]}) == 80


def test_exhaustive_gold_and_pairwise_independence(panel):
    prior = [case for case in panel["cases"] if case["phase"] == "prior"]
    cube = set(itertools.product((0, 1), repeat=3))
    for template_id in {case["template_id"] for case in prior}:
        cases = [case for case in prior if case["template_id"] == template_id]
        assert len(cases) == 8
        assert {
            (case["world_bit"], case["reader_observed"], case["sender_recorded"]) for case in cases
        } == cube
        for case in cases:
            world, actual, recorded = (
                case[key] for key in ("world_bit", "reader_observed", "sender_recorded")
            )
            assert case["labels"] == {
                "world": world,
                "reader": world ^ (1 - actual),
                "nested": world ^ (1 - recorded),
            }
        assert {tuple(case["labels"][key] for key in TARGETS) for case in cases} == cube
        for left, right in itertools.combinations(TARGETS, 2):
            assert Counter(
                (case["labels"][left], case["labels"][right]) for case in cases
            ) == dict.fromkeys(itertools.product((0, 1), repeat=2), 2)


def test_sender_cannot_see_private_actual_attendance(panel):
    prior = [case for case in panel["cases"] if case["phase"] == "prior"]
    for template_id, world, recorded in itertools.product(
        {case["template_id"] for case in prior}, (0, 1), (0, 1)
    ):
        pair = [
            case
            for case in prior
            if (case["template_id"], case["world_bit"], case["sender_recorded"])
            == (template_id, world, recorded)
        ]
        first, second = sorted(pair, key=lambda case: case["reader_observed"])
        assert first["labels"]["reader"] != second["labels"]["reader"]
        assert first["labels"]["nested"] == second["labels"]["nested"]
        assert json.dumps(first["messages"]["sender"], sort_keys=True) == json.dumps(
            second["messages"]["sender"], sort_keys=True
        )
        assert first["messages"]["omniscient"] != second["messages"]["omniscient"]
        assert "Narrator-only attendance" not in first["messages"]["sender"][-1]["content"]


def test_nested_visibility_intervention_holds_world_and_reader_fixed(panel):
    prior = [case for case in panel["cases"] if case["phase"] == "prior"]
    for template_id, world, actual in itertools.product(
        {case["template_id"] for case in prior}, (0, 1), (0, 1)
    ):
        pair = [
            case
            for case in prior
            if (case["template_id"], case["world_bit"], case["reader_observed"])
            == (template_id, world, actual)
        ]
        assert pair[0]["labels"]["world"] == pair[1]["labels"]["world"]
        assert pair[0]["labels"]["reader"] == pair[1]["labels"]["reader"]
        assert pair[0]["labels"]["nested"] != pair[1]["labels"]["nested"]


def test_update_controls_truthful_delivery_collapses_labels_and_cannot_fit(panel):
    controls = [case for case in panel["cases"] if case["phase"] == "delivered_current"]
    assert Counter(case["template_id"] for case in controls) == {"bd_t01": 8, "bd_t08": 8}
    for case in controls:
        assert case["split"] == "control"
        assert case["labels"] == dict.fromkeys(TARGETS, case["world_bit"])
        current = case["place_b"] if case["world_bit"] else case["place_a"]
        for messages in case["messages"].values():
            text = messages[-1]["content"]
            assert f'"The token is now in the {current}."' in text
            assert "receives and trusts" in text
            assert "delivery confirmation" in text
            assert "No message has been sent or planned" not in text


def test_roles_candidate_and_blind_messages(panel):
    frames = {}
    for case in panel["cases"]:
        reader, sender, place = case["reader_name"], case["sender_name"], case["place_b"]
        assert case["claim_place_bit"] == 1
        assert set(case["questions"]) == set(TARGETS)
        assert f"actually in the {place}" in case["questions"]["world"]
        assert f"does {reader} believe" in case["questions"]["reader"]
        assert f"does {sender} think that {reader} believes" in case["questions"]["nested"]
        assert all(
            question.endswith("Answer only YES or NO.") for question in case["questions"].values()
        )
        frames.setdefault(case["template_id"], (reader, sender, case["place_a"], place))
        for view, messages in case["messages"].items():
            assert [message["role"] for message in messages] == ["system", "user"]
            text = "\n".join(message["content"] for message in messages)
            assert "?" not in text
            assert "YES" not in text and "NO" not in text
            assert "world_bit" not in text and "reader_observed" not in text
            assert case["id"] not in text and case["template_id"] not in text
            assert f'"The token is in the {place}."' in text
            assert "not a new observation or message to either person" in text
            assert f"{reader} believes" not in text
            assert "unreliab" not in text and "always changes" not in text
            assert messages[-1]["content"].endswith("The scene is complete. Questions follow.")
            if view == "sender":
                assert "Narrator-only attendance" not in messages[-1]["content"]
    assert len(frames) == 8
    assert len({name for values in frames.values() for name in values[:2]}) == 16
    assert len({place for values in frames.values() for place in values[2:]}) == 16


@pytest.mark.parametrize(
    "corruption",
    (
        "gold",
        "private_attendance",
        "role_question",
        "control_split",
        "missing_cell",
        "duplicate_id",
        "boolean_bit",
        "candidate",
        "extra_gold_text",
    ),
)
def test_validator_rejects_corrupt_or_leaking_material(panel, corruption):
    changed = copy.deepcopy(panel)
    first = changed["cases"][0]
    if corruption == "gold":
        first["labels"]["nested"] ^= 1
    elif corruption == "private_attendance":
        first["messages"]["sender"][-1]["content"] += " Lee is outside throughout."
    elif corruption == "role_question":
        first["questions"]["nested"] = first["questions"]["reader"]
    elif corruption == "control_split":
        changed["cases"][-1]["split"] = "test"
    elif corruption == "missing_cell":
        changed["cases"].pop()
    elif corruption == "duplicate_id":
        changed["cases"][1] = copy.deepcopy(first)
    elif corruption == "boolean_bit":
        first["world_bit"] = False
    elif corruption == "candidate":
        first["claim_place_bit"] = 0
    elif corruption == "extra_gold_text":
        first["messages"]["sender"][-1]["content"] += " Gold nested label: 1."
    with pytest.raises(ValueError):
        validate_panel(changed)


def test_build_is_detached_deterministic_and_matches_frozen_json(panel):
    assert build_panel() == panel
    validated = validate_panel(panel)
    validated["cases"][0]["labels"]["world"] ^= 1
    assert validated != panel
    assert build_panel() == panel
    assert load_panel(DATA) == panel
