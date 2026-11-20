"""Source-derived gold, visibility, prospective scope and corruption checks."""

from __future__ import annotations

import copy
import itertools
import json
from collections import Counter
from pathlib import Path

import pytest

from latent_probes.belief_panel import build_panel as build_legacy
from latent_probes.belief_query_panel import SYSTEM, build_fresh_panel
from latent_probes.message_belief_panel import (
    SOURCE_BITS,
    TARGETS,
    build_panel,
    labels,
    load_panel,
    render_messages,
    validate_panel,
)

DATA = (
    Path(__file__).resolve().parents[1] / "data/belief_diagnostic/known_message_validation_v1.json"
)


@pytest.fixture
def panel():
    return build_panel()


def test_exact_size_scope_and_reference_questions(panel):
    assert len(panel["cases"]) == len({case["id"] for case in panel["cases"]}) == 64
    assert Counter(case["template_id"] for case in panel["cases"]) == {
        f"km_t{i:02d}": 16 for i in range(1, 5)
    }
    assert {case["split"] for case in panel["cases"]} == {"fresh"}
    assert {case["phase"] for case in panel["cases"]} == {"known_message"}
    assert all(case["needs_human_review"] is True for case in panel["cases"])
    assert "not new probe training" in " ".join(panel["assumptions"])
    assert "No independent human validation" in " ".join(panel["assumptions"])
    reference = build_fresh_panel()["cases"][0]
    for case in panel["cases"]:
        assert case["questions"] == {
            target: question.replace(reference["reader_name"], case["reader_name"])
            .replace(reference["sender_name"], case["sender_name"])
            .replace(reference["place_b"], case["place_b"])
            for target, question in reference["questions"].items()
        }
        for view in ("sender", "omniscient"):
            assert render_messages(case, view)[0] == {"role": "system", "content": SYSTEM}


def test_entity_template_and_prompt_disjointness_from_both_seen_banks(panel):
    seen = build_legacy()["cases"] + build_fresh_panel()["cases"]
    for keys in (("reader_name", "sender_name"), ("place_a", "place_b"), ("template_id",)):
        old_values = {case[key] for case in seen for key in keys}
        values = {case[key] for case in panel["cases"] for key in keys}
        assert len(values) == (4 if keys == ("template_id",) else 8)
        assert not old_values & values
    old_prompts = {
        json.dumps(case["messages"][view], sort_keys=True)
        for case in seen
        for view in ("sender", "omniscient")
    }
    assert all(
        json.dumps(render_messages(case, view), sort_keys=True) not in old_prompts
        for case in panel["cases"]
        for view in ("sender", "omniscient")
    )


def test_all_sixteen_source_cells_and_independent_targets_per_group(panel):
    for template_id in {case["template_id"] for case in panel["cases"]}:
        cases = [case for case in panel["cases"] if case["template_id"] == template_id]
        assert {tuple(case[key] for key in SOURCE_BITS) for case in cases} == set(
            itertools.product((0, 1), repeat=4)
        )
        assert Counter(tuple(case["labels"][key] for key in TARGETS) for case in cases) == {
            triple: 2 for triple in itertools.product((0, 1), repeat=3)
        }
        for case in cases:
            world, message, actual, receipt = (case[key] for key in SOURCE_BITS)
            assert (
                labels(case)
                == case["labels"]
                == {
                    "world": world,
                    "reader": message ^ (1 - actual),
                    "nested": message ^ (1 - receipt),
                }
            )
        for left, right in itertools.combinations(TARGETS, 2):
            assert Counter(
                (case["labels"][left], case["labels"][right]) for case in cases
            ) == dict.fromkeys(itertools.product((0, 1), repeat=2), 4)


@pytest.mark.parametrize(
    ("world", "message", "actual", "receipt", "expected"),
    (
        (0, 1, 0, 1, {"world": 0, "reader": 0, "nested": 1}),
        (1, 0, 1, 0, {"world": 1, "reader": 0, "nested": 1}),
        (1, 1, 1, 0, {"world": 1, "reader": 1, "nested": 0}),
        (0, 0, 0, 1, {"world": 0, "reader": 1, "nested": 0}),
    ),
)
def test_gold_examples_where_world_reader_and_nested_separate(
    panel, world, message, actual, receipt, expected
):
    cases = [
        case
        for case in panel["cases"]
        if tuple(case[key] for key in SOURCE_BITS) == (world, message, actual, receipt)
    ]
    assert len(cases) == 4
    assert all(labels(case) == expected for case in cases)


def test_private_delivery_pairs_are_exact_with_opposing_reader_gold(panel):
    for template_id, world, message, receipt in itertools.product(
        {case["template_id"] for case in panel["cases"]}, (0, 1), (0, 1), (0, 1)
    ):
        pair = [
            case
            for case in panel["cases"]
            if (case["template_id"], case["world_bit"], case["message_bit"], case["receipt"])
            == (template_id, world, message, receipt)
        ]
        first, second = sorted(pair, key=lambda case: case["delivered"])
        assert first["labels"]["reader"] != second["labels"]["reader"]
        assert first["labels"]["nested"] == second["labels"]["nested"]
        assert (
            json.dumps(render_messages(first, "sender"), sort_keys=True).encode()
            == json.dumps(render_messages(second, "sender"), sort_keys=True).encode()
        )
        assert render_messages(first, "omniscient") != render_messages(second, "omniscient")
    assert len({json.dumps(render_messages(case, "sender")) for case in panel["cases"]}) == 32
    assert len({json.dumps(render_messages(case, "omniscient")) for case in panel["cases"]}) == 64


def test_stipulated_trust_known_message_and_hidden_world_evidence_are_explicit(panel):
    for case in panel["cases"]:
        for view in ("sender", "omniscient"):
            messages = render_messages(case, view)
            assert [message["role"] for message in messages] == ["system", "user"]
            text = messages[-1]["content"]
            initial = case["place_a"] if case["message_bit"] else case["place_b"]
            current = case["place_b"] if case["world_bit"] else case["place_a"]
            assertion = case["place_b"] if case["message_bit"] else case["place_a"]
            assert initial in text.split("\n\n")[0]
            assert f'"The token is now in the {assertion}."' in text
            assert current in text
            assert "even when the claim differs from reality" in text
            assert "no further view of the token" in text
            assert "no other evidence about actual delivery" in text
            assert "definitive" in text
            assert "known" not in text.lower().split("candidate proposition:")[-1]
            assert "?" not in text and "YES" not in text and "NO" not in text
            assert "world_bit" not in text and "message_bit" not in text
            for key in SOURCE_BITS:
                assert f'"{key}":' not in text and f"'{key}':" not in text
                assert all(f"{key}={bit}" not in text for bit in (0, 1))
            assert case["id"] not in text and "needs_human_review" not in text
            assert text.endswith("End of account. A question may follow.")
            assert ("Private delivery account for the 09:15 attempt, known to" in text) == (
                view == "omniscient"
            )


@pytest.mark.parametrize(
    "corruption",
    (
        "label",
        "bool_label",
        "bool_bit",
        "visibility",
        "receipt",
        "question",
        "review",
        "split",
        "phase",
        "candidate",
        "duplicate",
        "missing",
        "assumptions",
        "unknown_group",
    ),
)
def test_corrupt_or_relabelled_panels_are_rejected(panel, corruption):
    changed = copy.deepcopy(panel)
    first = changed["cases"][0]
    if corruption == "label":
        first["labels"]["nested"] ^= 1
    elif corruption == "bool_label":
        first["labels"]["reader"] = bool(first["labels"]["reader"])
    elif corruption == "bool_bit":
        first["delivered"] = False
    elif corruption == "visibility":
        first["messages"]["sender"][-1]["content"] += " Actual delivery failed."
    elif corruption == "receipt":
        first["receipt"] ^= 1
    elif corruption == "question":
        first["questions"]["nested"] = first["questions"]["reader"]
    elif corruption == "review":
        first["needs_human_review"] = False
    elif corruption == "split":
        first["split"] = "train"
    elif corruption == "phase":
        first["phase"] = "prior"
    elif corruption == "candidate":
        first["claim_place_bit"] = True
    elif corruption == "duplicate":
        changed["cases"][1] = copy.deepcopy(first)
    elif corruption == "missing":
        changed["cases"].pop()
    elif corruption == "assumptions":
        changed["assumptions"] = []
    else:
        first["template_id"] = "unknown"
    with pytest.raises(ValueError):
        validate_panel(changed)


@pytest.mark.parametrize("bad_view", ("reader", "", None))
def test_render_rejects_unknown_views(panel, bad_view):
    with pytest.raises(ValueError):
        render_messages(panel["cases"][0], bad_view)


def test_labels_derive_from_bits_and_return_detached_gold(panel):
    case = copy.deepcopy(panel["cases"][0])
    case["labels"] = {"world": 99}
    assert labels(case) == labels(panel["cases"][0])
    gold = labels(case)
    gold["reader"] ^= 1
    assert gold != labels(case)


def test_frozen_json_determinism_and_detached_results(panel):
    assert load_panel(DATA) == panel == build_panel()
    detached = validate_panel(panel)
    detached["cases"][0]["labels"]["reader"] ^= 1
    assert detached != panel
    rendered = render_messages(panel["cases"][0], "sender")
    rendered[-1]["content"] = "changed"
    assert rendered != panel["cases"][0]["messages"]["sender"]
