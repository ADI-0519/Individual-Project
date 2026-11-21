"""Factorial, semantic, privacy and fixed-target checks for the fresh bank."""

from __future__ import annotations

import copy
import itertools
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from latent_probes.belief_query_panel import SYSTEM, _questions
from latent_probes.crossed_belief_bank import (
    SOURCE_BITS,
    build_bank,
    build_messages,
    labels,
    load_bank,
    sender_rows,
    validate_bank,
)

DATA = (
    Path(__file__).resolve().parents[1] / "data/belief_diagnostic/crossed_belief_validation_v1.json"
)


@pytest.fixture(scope="module")
def bank():
    return build_bank()


def test_committed_bank_is_complete_deterministic_draft(bank):
    assert load_bank(DATA) == validate_bank(bank) == build_bank()
    assert bank["model_outputs_seen"] is False
    assert bank["needs_human_review"] is True
    assert len(bank["sender_inputs"]) == 128
    assert len(bank["hidden_delivery_rows"]) == 256
    assert len({r["id"] for r in bank["sender_inputs"]}) == 128
    assert len({r["id"] for r in bank["hidden_delivery_rows"]}) == 256


def test_every_wording_and_mapping_exhausts_independent_factors(bank):
    for template, mapping in itertools.product((f"cb_t{i:02d}" for i in range(1, 5)), (0, 1)):
        rows = [
            r
            for r in bank["sender_inputs"]
            if (r["template_id"], r["mapping_id"]) == (template, mapping)
        ]
        assert len(rows) == 16
        assert {tuple(r["source"][k] for k in SOURCE_BITS) for r in rows} == set(
            itertools.product((0, 1), repeat=4)
        )
        assert Counter(r["gold"] for r in rows) == {0: 8, 1: 8}
        assert Counter(r["stratum"] for r in rows) == {
            "receipt_sensitive": 8,
            "receipt_invariant": 8,
        }


def test_names_and_physical_places_are_disjoint_from_earlier_diagnostic_banks(bank):
    names_and_places = {
        row[key]
        for row in bank["sender_inputs"]
        for key in ("reader_name", "sender_name", "place_a", "place_b")
    }
    assert len(names_and_places) == 16
    earlier = [path for path in DATA.parent.glob("*.json") if path != DATA]
    assert earlier
    for path in earlier:
        text = path.read_text(encoding="utf-8")
        assert all(value not in text for value in names_and_places), path.name


@pytest.mark.parametrize(
    ("initial", "message", "receipt", "actual", "world", "expected"),
    (
        (0, 1, 1, 0, 0, {"nested": 1, "reader": 0, "world": 0}),
        (0, 1, 0, 1, 1, {"nested": 0, "reader": 1, "world": 1}),
        (1, 0, 1, 0, 1, {"nested": 0, "reader": 1, "world": 1}),
        (1, 0, 0, 1, 0, {"nested": 1, "reader": 0, "world": 0}),
        (1, 1, 0, 1, 0, {"nested": 1, "reader": 1, "world": 0}),
        (0, 0, 1, 0, 1, {"nested": 0, "reader": 0, "world": 1}),
    ),
)
def test_semantic_examples(initial, message, receipt, actual, world, expected):
    row = {"source": dict(zip(SOURCE_BITS, (initial, message, receipt, world), strict=True))}
    assert labels(row, actual) == expected
    assert labels(row) == {k: expected[k] for k in ("world", "nested")}


def test_primary_shortcut_baselines_and_actual_reader_information_limit(bank):
    visible = bank["sender_inputs"]
    for stratum, n, shortcut_correct in (
        (None, 128, 96),
        ("receipt_sensitive", 64, 32),
        ("receipt_invariant", 64, 64),
    ):
        rows = [r for r in visible if stratum is None or r["stratum"] == stratum]
        assert len(rows) == n
        for key in ("initial_slot", "message_slot"):
            assert sum(r["source"][key] == r["gold"] for r in rows) == shortcut_correct
        for key in ("world_slot", "receipt_reports_read"):
            assert sum(r["source"][key] == r["gold"] for r in rows) == n // 2
        # Best deterministic label given each sender input; A cannot be observed.
        paired_reader = [Counter(labels(r, a)["reader"] for a in (0, 1)) for r in rows]
        ceiling = sum(max(counts.values()) for counts in paired_reader)
        assert ceiling == {None: 192, "receipt_sensitive": 64, "receipt_invariant": 128}[stratum]


def test_private_actual_delivery_is_metadata_only_and_exactly_paired(bank):
    by_id = {r["id"]: r for r in bank["sender_inputs"]}
    for row_id, row in by_id.items():
        hidden = [r for r in bank["hidden_delivery_rows"] if r["sender_input_id"] == row_id]
        assert [r["actual_delivery"] for r in hidden] == [0, 1]
        assert all(r["labels"] == labels(row, r["actual_delivery"]) for r in hidden)
        assert "actual_delivery" not in row
        for queried in (False, True):
            base = build_messages(row, queried)
            # Even contaminated caller metadata must never change what sender sees.
            for actual in (0, 1):
                augmented = {**row, "actual_delivery": actual, "gold": 1 - row["gold"]}
                assert build_messages(augmented, queried) == base
        if row["stratum"] == "receipt_sensitive":
            assert hidden[0]["labels"]["reader"] != hidden[1]["labels"]["reader"]
        else:
            assert hidden[0]["labels"]["reader"] == hidden[1]["labels"]["reader"]


def test_source_gold_is_receipt_sensitive_only_when_initial_differs(bank):
    indexed = {
        (r["template_id"], r["mapping_id"], *(r["source"][k] for k in SOURCE_BITS)): r
        for r in bank["sender_inputs"]
    }
    for template, mapping, initial, message, world in itertools.product(
        (f"cb_t{i:02d}" for i in range(1, 5)), (0, 1), (0, 1), (0, 1), (0, 1)
    ):
        left = indexed[(template, mapping, initial, message, 0, world)]
        right = indexed[(template, mapping, initial, message, 1, world)]
        assert (left["gold"] != right["gold"]) is (initial != message)
        for receipt in (0, 1):
            base = indexed[(template, mapping, initial, message, receipt, 0)]
            world_swap = indexed[(template, mapping, initial, message, receipt, 1)]
            assert base["gold"] == world_swap["gold"]


def test_both_captures_use_exact_legacy_target_and_normal_neutral_system(bank):
    for row in bank["sender_inputs"]:
        old_question = _questions(
            SimpleNamespace(
                reader=row["reader_name"], sender=row["sender_name"], place_b=row["place_b"]
            )
        )["nested"]
        story = build_messages(row, False)
        queried = build_messages(row, True)
        assert story[0] == queried[0] == {"role": "system", "content": SYSTEM}
        assert [m["role"] for m in story] == ["system", "user"]
        assert queried[-1]["content"] == story[-1]["content"] + "\n\nQuestion: " + old_question
        assert story == row["story_messages"]
        assert queried == row["messages"]
        assert "?" not in story[-1]["content"]
        assert "YES" not in story[-1]["content"] and "NO" not in story[-1]["content"]
        assert story[-1]["content"].endswith("End of account. A question may follow.")
        assert row["id"] not in json.dumps(queried)
        assert "actual_delivery" not in json.dumps(queried)
        assert "gold" not in json.dumps(queried)


def test_all_templates_explicitly_order_belief_message_receipt_end(bank):
    for row in bank["sender_inputs"]:
        text = build_messages(row, False)[-1]["content"]
        # First occurrence of each timestamp locates its evidence paragraph.
        assert [text.index(t) for t in ("09:00", "09:03", "09:05", "09:10", "09:20")] == sorted(
            text.index(t) for t in ("09:00", "09:03", "09:05", "09:10", "09:20")
        )
        assert "initial location belief" in text
        assert "even if the message is false" in text
        assert "accepts this particular receipt as definitive" in text
        assert "no other delivery evidence" in text
        assert "No direct observation or independent delivery account" in text
        initial_place = (row["place_a"], row["place_b"])[row["source"]["initial_slot"]]
        assert initial_place in text.split("\n\n")[0]


def test_physical_mapping_is_exact_global_exchange_and_no_new_scenario(bank):
    indexed = {
        (r["template_id"], tuple(r["source"][k] for k in SOURCE_BITS), r["mapping_id"]): r
        for r in bank["sender_inputs"]
    }
    for template, factors in itertools.product(
        (f"cb_t{i:02d}" for i in range(1, 5)), itertools.product((0, 1), repeat=4)
    ):
        left, right = (indexed[(template, factors, mapping)] for mapping in (0, 1))
        assert left["gold"] == right["gold"]
        assert left["source"] == right["source"]
        assert [left["place_a"], left["place_b"]] == [right["place_b"], right["place_a"]]
        a, b = left["place_a"], left["place_b"]
        for queried in (False, True):
            for x, y in zip(
                build_messages(left, queried), build_messages(right, queried), strict=True
            ):
                swapped = x["content"].replace(a, "\x00").replace(b, a).replace("\x00", b)
                assert swapped == y["content"]
    for queried in (False, True):
        assert len({json.dumps(build_messages(r, queried)) for r in bank["sender_inputs"]}) == 128


@pytest.mark.parametrize("mutation", ("gold", "drop", "duplicate", "message", "private", "scope"))
def test_validator_rejects_corrupted_material(bank, mutation):
    altered = copy.deepcopy(bank)
    if mutation == "gold":
        altered["sender_inputs"][0]["gold"] ^= 1
    elif mutation == "drop":
        altered["sender_inputs"].pop()
    elif mutation == "duplicate":
        altered["sender_inputs"][1] = altered["sender_inputs"][0]
    elif mutation == "message":
        altered["sender_inputs"][0]["messages"][-1]["content"] += " Extra update."
    elif mutation == "private":
        altered["hidden_delivery_rows"][0]["labels"]["reader"] ^= 1
    else:
        altered["needs_human_review"] = False
    with pytest.raises(ValueError, match="authored full factorial"):
        validate_bank(altered)


@pytest.mark.parametrize("bad_bit", (True, False, -1, 2, 0.0, "1", None))
def test_invalid_factor_types_are_rejected(bank, bad_bit):
    row = copy.deepcopy(bank["sender_inputs"][0])
    row["source"]["initial_slot"] = bad_bit
    with pytest.raises(ValueError, match="integer bit"):
        build_messages(row, True)


def test_unique_sender_rows_are_isolated_from_bank_and_private_metadata(bank):
    rows = sender_rows(bank)
    rows[0]["gold"] ^= 1
    rows[0]["source"]["world_slot"] ^= 1
    assert rows[0] != bank["sender_inputs"][0]
    assert all("actual_delivery" not in r for r in rows)
