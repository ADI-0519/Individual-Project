"""Complement, channel, source-gold and private-input tests; no model calls."""

from __future__ import annotations

import copy
import itertools
import json
from collections import Counter
from pathlib import Path

import pytest

from latent_probes.belief_contrast_panel import (
    CHANNELS,
    TARGETS,
    build_contrast_panel,
    load_contrast_panel,
    validate_contrast_panel,
)
from latent_probes.belief_panel import SYSTEM, build_panel
from latent_probes.belief_query_panel import build_fresh_panel

DATA = Path(__file__).resolve().parents[1] / "data/belief_diagnostic/contrast_validation_v1.json"


@pytest.fixture
def panel():
    return build_contrast_panel()


def test_two_fresh_channels_have_eight_complete_groups_each(panel):
    assert set(panel) == {"id", "schema", "assumptions", "cases"}
    assert len(panel["cases"]) == 128
    assert len({case["id"] for case in panel["cases"]}) == 128
    assert Counter(case["channel"] for case in panel["cases"]) == dict.fromkeys(CHANNELS, 64)
    assert set(Counter(case["template_id"] for case in panel["cases"]).values()) == {8}
    for channel in CHANNELS:
        members = [case for case in panel["cases"] if case["channel"] == channel]
        assert len({case["template_id"] for case in members}) == 8
        assert {case["split"] for case in members} == {"fresh"}
        expected_phase = "prior" if channel == "observation_prior" else "post_message"
        assert {case["phase"] for case in members} == {expected_phase}
    declarations = " ".join(panel["assumptions"])
    assert "needs_human_review=true" in declarations
    assert "not broad reasoning-family OOD" in declarations


def test_source_gold_and_independent_target_cube(panel):
    cube = set(itertools.product((0, 1), repeat=3))
    for template_id in {case["template_id"] for case in panel["cases"]}:
        members = [case for case in panel["cases"] if case["template_id"] == template_id]
        assert len(members) == 8
        assert {
            (case["world_bit"], case["reader_updated"], case["sender_recorded"]) for case in members
        } == cube
        for case in members:
            world, actual, recorded = (
                case[key] for key in ("world_bit", "reader_updated", "sender_recorded")
            )
            assert case["labels"] == {
                "world": world,
                "reader": world ^ (1 - actual),
                "nested": world ^ (1 - recorded),
            }
        assert {tuple(case["labels"][target] for target in TARGETS) for case in members} == cube
        for left, right in itertools.combinations(TARGETS, 2):
            assert Counter(
                (case["labels"][left], case["labels"][right]) for case in members
            ) == dict.fromkeys(itertools.product((0, 1), repeat=2), 2)


def test_complementary_candidates_change_only_claim_and_negate_gold(panel):
    for case in panel["cases"]:
        first, second = case["candidates"]
        assert [first["place_bit"], second["place_bit"]] == [0, 1]
        assert first["claim"] == f"The token is in the {case['place_a']}."
        assert second["claim"] == f"The token is in the {case['place_b']}."
        assert second["labels"] == case["labels"]
        assert all(first["labels"][target] == 1 - second["labels"][target] for target in TARGETS)
        for view in ("omniscient", "sender"):
            a, b = first["messages"][view], second["messages"][view]
            assert a[0] == b[0] == {"role": "system", "content": SYSTEM}
            assert a[-1]["content"].count(first["claim"]) == 1
            assert b[-1]["content"].count(second["claim"]) == 1
            assert a[-1]["content"].replace(first["claim"], second["claim"]) == b[-1]["content"]


def test_hidden_access_pairs_are_identical_for_both_candidates(panel):
    all_prefixes = []
    sender_pairs = 0
    for template_id, world, recorded in itertools.product(
        {case["template_id"] for case in panel["cases"]}, (0, 1), (0, 1)
    ):
        pair = [
            case
            for case in panel["cases"]
            if (case["template_id"], case["world_bit"], case["sender_recorded"])
            == (template_id, world, recorded)
        ]
        a, b = sorted(pair, key=lambda case: case["reader_updated"])
        assert a["labels"]["reader"] != b["labels"]["reader"]
        assert a["labels"]["nested"] == b["labels"]["nested"]
        for index in (0, 1):
            assert (
                json.dumps(a["candidates"][index]["messages"]["sender"]).encode()
                == json.dumps(b["candidates"][index]["messages"]["sender"]).encode()
            )
            assert (
                a["candidates"][index]["messages"]["omniscient"]
                != b["candidates"][index]["messages"]["omniscient"]
            )
            sender_pairs += 1
    for case in panel["cases"]:
        for candidate in case["candidates"]:
            for messages in candidate["messages"].values():
                all_prefixes.append(json.dumps(messages, sort_keys=True))
    assert sender_pairs == 128
    assert len(all_prefixes) == 512
    assert len(set(all_prefixes)) == 384


def test_message_update_has_truthful_note_and_private_false_positive_receipt(panel):
    updates = [case for case in panel["cases"] if case["channel"] == "trusted_message_update"]
    for case in updates:
        current = case["place_b"] if case["world_bit"] else case["place_a"]
        for candidate in case["candidates"]:
            for view, messages in candidate["messages"].items():
                text = messages[-1]["content"]
                assert f'"The token is now in the {current}."' in text
                assert "16:05" in text and "16:12" in text and "16:15" in text
                assert f"{case['sender_name']} cannot see {case['reader_name']}" in text
                if view == "sender":
                    assert "Actual delivery, private" not in text
                    assert "false" not in text and "unreliab" not in text
                else:
                    assert "Actual delivery, private" in text
    false_positive = next(
        case
        for case in updates
        if case["world_bit"] == 1 and case["reader_updated"] == 0 and case["sender_recorded"] == 1
    )
    assert false_positive["labels"] == {"world": 1, "reader": 0, "nested": 1}
    full = false_positive["candidates"][1]["messages"]["omniscient"][-1]["content"]
    hidden = false_positive["candidates"][1]["messages"]["sender"][-1]["content"]
    assert "never reaches" in full
    assert "reached" in hidden and "was read and trusted" in hidden
    assert "never reaches" not in hidden


def test_question_roles_disclaimer_and_no_gold_in_capture_prefix(panel):
    reference = build_panel()["cases"][0]
    reference_parts = reference["messages"]["sender"][-1]["content"].split("\n\n")
    for case in panel["cases"]:
        for candidate in case["candidates"]:
            place = (case["place_a"], case["place_b"])[candidate["place_bit"]]
            assert candidate["questions"] == {
                target: question.replace(reference["reader_name"], case["reader_name"])
                .replace(reference["sender_name"], case["sender_name"])
                .replace(reference["place_b"], place)
                for target, question in reference["questions"].items()
            }
            assert case["reader_name"] not in candidate["claim"]
            assert case["sender_name"] not in candidate["claim"]
            for view, messages in candidate["messages"].items():
                text = "\n".join(message["content"] for message in messages)
                assert "?" not in text and "YES" not in text and "NO" not in text
                assert "not a new observation or message to either person" in text
                assert text.endswith(reference_parts[-1])
                assert messages[-1]["content"].split("\n\n")[-2] == reference_parts[-2].replace(
                    reference["place_b"], place
                )
                assert "reader_updated" not in text and "sender_recorded" not in text
                assert case["id"] not in text and case["template_id"] not in text
                assert "needs_human_review" not in text and "authoring" not in text
                assert f"{case['reader_name']} believes" not in text
                if view == "sender":
                    assert "Private attendance" not in text and "Actual delivery" not in text


def test_entities_are_new_and_role_order_does_not_depend_on_source_bits(panel):
    old = build_panel()["cases"] + build_fresh_panel()["cases"]
    old_names = {case[key] for case in old for key in ("reader_name", "sender_name")}
    old_places = {case[key] for case in old for key in ("place_a", "place_b")}
    names = {case[key] for case in panel["cases"] for key in ("reader_name", "sender_name")}
    places = {case[key] for case in panel["cases"] for key in ("place_a", "place_b")}
    assert len(names) == len(places) == 32
    assert not names & old_names and not places & old_places
    for channel in CHANNELS:
        order_counts = Counter()
        for template_id in {
            case["template_id"] for case in panel["cases"] if case["channel"] == channel
        }:
            members = [case for case in panel["cases"] if case["template_id"] == template_id]
            assert len({(case["reader_name"], case["sender_name"]) for case in members}) == 1
            first = members[0]
            opening = first["candidates"][0]["messages"]["sender"][-1]["content"].split("\n\n")[0]
            order_counts[
                opening.index(first["reader_name"]) < opening.index(first["sender_name"])
            ] += 1
        assert order_counts == {True: 4, False: 4}


@pytest.mark.parametrize(
    "corruption",
    (
        "base_gold",
        "candidate_gold",
        "candidate_order",
        "candidate_claim",
        "private_leak",
        "question",
        "phase",
        "channel",
        "bool_bit",
        "duplicate",
        "missing",
        "entity",
        "declarations",
    ),
)
def test_validator_rejects_semantic_or_visibility_corruption(panel, corruption):
    changed = copy.deepcopy(panel)
    case = changed["cases"][0]
    if corruption == "base_gold":
        case["labels"]["nested"] ^= 1
    elif corruption == "candidate_gold":
        case["candidates"][0]["labels"]["reader"] ^= 1
    elif corruption == "candidate_order":
        case["candidates"].reverse()
    elif corruption == "candidate_claim":
        case["candidates"][0]["claim"] = case["candidates"][1]["claim"]
    elif corruption == "private_leak":
        case["candidates"][0]["messages"]["sender"][-1]["content"] += " Aiden is absent."
    elif corruption == "question":
        case["candidates"][0]["questions"]["nested"] = case["candidates"][0]["questions"]["world"]
    elif corruption == "phase":
        case["phase"] = "post_message"
    elif corruption == "channel":
        case["channel"] = "trusted_message_update"
    elif corruption == "bool_bit":
        case["reader_updated"] = False
    elif corruption == "duplicate":
        changed["cases"][1] = copy.deepcopy(case)
    elif corruption == "missing":
        changed["cases"].pop()
    elif corruption == "entity":
        case["reader_name"] = "Lee"
    else:
        changed["assumptions"].pop()
    with pytest.raises(ValueError):
        validate_contrast_panel(changed)


def test_frozen_json_determinism_and_no_mutable_aliases(panel):
    assert load_contrast_panel(DATA) == panel == build_contrast_panel()
    changed = validate_contrast_panel(panel)
    changed["cases"][0]["candidates"][0]["labels"]["world"] ^= 1
    assert changed != panel
    assert build_contrast_panel() == panel
