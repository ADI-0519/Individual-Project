"""Fresh authored content and mathematical/visibility checks; no model outputs."""

from __future__ import annotations

import copy
import itertools
import json
from collections import Counter
from pathlib import Path

import pytest

from latent_probes.belief_panel import SYSTEM, build_panel
from latent_probes.belief_query_panel import (
    TARGETS,
    build_fresh_panel,
    load_fresh_panel,
    validate_fresh_panel,
)

DATA = Path(__file__).resolve().parents[1] / "data/belief_diagnostic/fresh_query_validation_v1.json"


@pytest.fixture
def panel():
    return build_fresh_panel()


def test_fresh_groups_size_scope_and_legacy_schema_compatibility(panel):
    assert len(panel["cases"]) == 32
    assert len({case["id"] for case in panel["cases"]}) == 32
    assert Counter(case["template_id"] for case in panel["cases"]) == {
        f"bq_t{i:02d}": 8 for i in range(1, 5)
    }
    assert {case["split"] for case in panel["cases"]} == {"fresh"}
    assert {case["phase"] for case in panel["cases"]} == {"prior"}
    old = build_panel()
    assert set(panel) == set(old)
    assert all(set(case) == set(old["cases"][0]) for case in panel["cases"])
    assert all(
        messages[0] == {"role": "system", "content": SYSTEM}
        for case in panel["cases"]
        for messages in case["messages"].values()
    )
    reference = old["cases"][0]
    for case in panel["cases"]:
        assert case["questions"] == {
            target: question.replace(reference["reader_name"], case["reader_name"])
            .replace(reference["sender_name"], case["sender_name"])
            .replace(reference["place_b"], case["place_b"])
            for target, question in reference["questions"].items()
        }
    assert "needs_human_review=true" in " ".join(panel["assumptions"])
    assert "not been independently human validated" in " ".join(panel["assumptions"])
    assert "same logical family" in " ".join(panel["assumptions"])


def test_new_people_places_and_wording_groups_do_not_overlap_legacy(panel):
    old = build_panel()["cases"]
    names = {case[key] for case in panel["cases"] for key in ("reader_name", "sender_name")}
    old_names = {case[key] for case in old for key in ("reader_name", "sender_name")}
    places = {case[key] for case in panel["cases"] for key in ("place_a", "place_b")}
    old_places = {case[key] for case in old for key in ("place_a", "place_b")}
    assert len(names) == len(places) == 8
    assert not names & old_names
    assert not places & old_places
    assert not {case["template_id"] for case in panel["cases"]} & {
        case["template_id"] for case in old
    }
    old_messages = {case["messages"]["omniscient"][-1]["content"] for case in old}
    assert all(
        case["messages"]["omniscient"][-1]["content"] not in old_messages for case in panel["cases"]
    )


def test_each_new_group_exhausts_independent_source_and_target_cubes(panel):
    cube = set(itertools.product((0, 1), repeat=3))
    for template_id in {case["template_id"] for case in panel["cases"]}:
        cases = [case for case in panel["cases"] if case["template_id"] == template_id]
        assert {
            tuple(case[key] for key in ("world_bit", "reader_observed", "sender_recorded"))
            for case in cases
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


def test_exact_private_attendance_pairs_and_role_alignment(panel):
    for template_id, world, recorded in itertools.product(
        {case["template_id"] for case in panel["cases"]}, (0, 1), (0, 1)
    ):
        pair = [
            case
            for case in panel["cases"]
            if (case["template_id"], case["world_bit"], case["sender_recorded"])
            == (template_id, world, recorded)
        ]
        left, right = sorted(pair, key=lambda case: case["reader_observed"])
        assert left["labels"]["reader"] != right["labels"]["reader"]
        assert left["labels"]["nested"] == right["labels"]["nested"]
        assert (
            json.dumps(left["messages"]["sender"]).encode()
            == json.dumps(right["messages"]["sender"]).encode()
        )
        assert left["messages"]["omniscient"] != right["messages"]["omniscient"]
    for case in panel["cases"]:
        reader, sender, place = case["reader_name"], case["sender_name"], case["place_b"]
        assert f"actually in the {place}" in case["questions"]["world"]
        assert f"does {reader} believe" in case["questions"]["reader"]
        assert f"does {sender} think that {reader} believes" in case["questions"]["nested"]
        assert set(case["questions"]) == set(TARGETS)
        assert all(q.endswith("Answer only YES or NO.") for q in case["questions"].values())


def test_no_question_gold_or_authoring_rules_enter_base_messages(panel):
    for case in panel["cases"]:
        reader, sender = case["reader_name"], case["sender_name"]
        assert case["claim_place_bit"] == 1
        for view, messages in case["messages"].items():
            assert [message["role"] for message in messages] == ["system", "user"]
            text = "\n".join(message["content"] for message in messages)
            assert f'"The token is in the {case["place_b"]}."' in text
            assert "not new evidence or a message received" in text
            assert "?" not in text and "YES" not in text and "NO" not in text
            assert case["id"] not in text and case["template_id"] not in text
            assert "needs_human_review" not in text and "world_bit" not in text
            assert "always" not in text and "unreliab" not in text
            assert f"{reader} believes" not in text and f"{sender} believes" not in text
            assert "09:00" in text and "09:15" in text and "09:20" in text
            if case["template_id"] == "bq_t03":
                assert (
                    f"The camera provides no way for {sender} to check {reader}'s attendance."
                    in text
                )
                assert "there is no other way" not in text
            assert "watched" in text or "watch" in text
            if view == "sender":
                assert "Private attendance account" not in text
            else:
                assert f"not available to {sender}" in text
                if not case["reader_observed"]:
                    assert "does not learn that a transfer occurred" in text
            if not case["sender_recorded"]:
                assert "did not watch" in text


@pytest.mark.parametrize(
    "corruption",
    (
        "labels",
        "visibility",
        "roles",
        "split",
        "phase",
        "candidate",
        "bool_bit",
        "duplicate",
        "missing",
        "gold_in_message",
    ),
)
def test_reject_changed_or_leaking_panel(panel, corruption):
    changed = copy.deepcopy(panel)
    first = changed["cases"][0]
    if corruption == "labels":
        first["labels"]["nested"] ^= 1
    elif corruption == "visibility":
        first["messages"]["sender"][-1]["content"] += " Pavel did not watch."
    elif corruption == "roles":
        first["questions"]["nested"] = first["questions"]["reader"]
    elif corruption == "split":
        first["split"] = "test"
    elif corruption == "phase":
        first["phase"] = "delivered_current"
    elif corruption == "candidate":
        first["claim_place_bit"] = 0
    elif corruption == "bool_bit":
        first["reader_observed"] = False
    elif corruption == "duplicate":
        changed["cases"][1] = copy.deepcopy(first)
    elif corruption == "missing":
        changed["cases"].pop()
    else:
        first["messages"]["omniscient"][-1]["content"] += " Correct nested answer: YES."
    with pytest.raises(ValueError):
        validate_fresh_panel(changed)


def test_frozen_data_determinism_and_detached_results(panel):
    assert load_fresh_panel(DATA) == panel == build_fresh_panel()
    validated = validate_fresh_panel(panel)
    validated["cases"][0]["labels"]["world"] ^= 1
    assert validated != panel
    assert build_fresh_panel() == panel
