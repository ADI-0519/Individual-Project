"""Factual-template and leakage tests; no models or assumed reader outcomes."""

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from latent_probes.forecast_panel import (
    CONTEXTS,
    FAMILIES,
    PRECEDENCE_POLICY,
    build_reader_messages,
    build_sender_messages,
    control_jobs,
    load_forecast_panel,
    render_world,
    sender_jobs,
    validate_forecast_panel,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "reader_forecast" / "development_v1.json"


@pytest.fixture
def panel():
    return load_forecast_panel(DATA)


def test_case_groups_are_disjoint_and_every_split_balances_world_truth(panel):
    cases = panel["cases"]
    assert len(cases) == 24
    assert len({case["id"] for case in cases}) == 24
    assert len({case["world"]["entity"] for case in cases}) == 24
    groups = {
        split: {case["id"] for case in cases if case["split"] == split}
        for split in ("train", "validation", "test")
    }
    assert {key: len(value) for key, value in groups.items()} == {
        "train": 16,
        "validation": 4,
        "test": 4,
    }
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert not groups["validation"] & groups["test"]
    for split, ids in groups.items():
        assert Counter(case["correct"] for case in cases if case["id"] in ids) == {
            "APPROVE": len(ids) // 2,
            "REJECT": len(ids) // 2,
        }
    for family in FAMILIES:
        members = [case for case in cases if case["family"] == family]
        assert [case["split"] for case in members] == ["train"] * 4 + ["validation", "test"]
        assert Counter(case["correct"] for case in members) == {"APPROVE": 3, "REJECT": 3}


def test_numeric_instantiations_and_entities_are_fresh_across_splits(panel):
    for family in FAMILIES:
        worlds = {
            split: [
                case["world"]
                for case in panel["cases"]
                if case["family"] == family and case["split"] == split
            ]
            for split in ("train", "validation", "test")
        }
        numbers = {
            split: {value for world in members for value in world.values() if type(value) is int}
            for split, members in worlds.items()
        }
        dates = {
            split: {world["current_date"] for world in members} for split, members in worlds.items()
        }
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            assert not numbers[a] & numbers[b]
            assert not dates[a] & dates[b]


def test_structured_gold_computation_covers_all_rule_families(panel):
    for case in panel["cases"]:
        world, family = case["world"], case["family"]
        if family == "conjunction":
            approved = world["decisive_pass"]
        elif family == "threshold":
            approved = world["score"] >= world["threshold"]
        elif family == "supersession":
            approved = world["current_pass"]
        else:
            approved = world["revenue"] - world["costs"] >= 0
        assert case["correct"] == ("APPROVE" if approved else "REJECT")
        expected = render_world(world, family)
        assert all(
            case[key] == expected[key] for key in ("rule", "source", "reader_packets", "correct")
        )


@pytest.mark.parametrize(
    "corruption",
    [
        "gold",
        "source",
        "packet",
        "rule",
        "split",
        "duplicate_id",
        "wrong_family_id",
        "same_entity",
        "date_order",
        "same_stale_result",
        "bool_number",
        "unknown_world_field",
        "numeric_overlap",
    ],
)
def test_validation_rejects_inconsistent_facts_labels_or_split_allocation(panel, corruption):
    bad = copy.deepcopy(panel)
    case = bad["cases"][0]
    if corruption == "gold":
        case["correct"] = "REJECT"
    elif corruption == "source":
        case["source"][-1] += " This claim was changed."
    elif corruption == "packet":
        case["reader_packets"]["absent"].append("No result is available.")
    elif corruption == "rule":
        case["rule"] = "Always approve."
    elif corruption == "split":
        case["split"] = "test"
    elif corruption == "duplicate_id":
        bad["cases"][1]["id"] = case["id"]
    elif corruption == "wrong_family_id":
        case["id"] = "rf_threshold_99"
    elif corruption == "same_entity":
        second = bad["cases"][1]
        second["world"]["entity"] = case["world"]["entity"]
        second.update(render_world(second["world"], second["family"]))
    elif corruption == "date_order":
        case["world"]["current_date"] = case["world"]["previous_date"]
    elif corruption == "same_stale_result":
        case["world"]["previous_decisive_pass"] = case["world"]["decisive_pass"]
    elif corruption == "bool_number":
        case["world"]["required_count"] = True
    elif corruption == "numeric_overlap":
        validation = bad["cases"][4]
        validation["world"]["required_count"] = case["world"]["required_count"]
        validation.update(render_world(validation["world"], validation["family"]))
    else:
        case["world"]["gold_decision"] = "APPROVE"
    with pytest.raises(ValueError):
        validate_forecast_panel(bad)


def test_absent_packets_are_positive_source_facts_not_missingness_announcements(panel):
    for case in panel["cases"]:
        packet = case["reader_packets"]["absent"]
        assert len(packet) == 1
        assert packet[0] in case["source"]
        text = packet[0].lower()
        for marker in (
            "no result",
            "no score",
            "no cost",
            "not available",
            "unknown",
            "missing",
            "contains no",
        ):
            assert marker not in text
        assert case["source"][-1] not in packet


def test_neutral_and_favorable_jobs_share_source_packet_and_natural_output_budget(panel):
    jobs = sender_jobs(panel)
    assert len(jobs) == 84
    assert Counter(job["objective"] for job in jobs) == {"neutral": 72, "favorable": 12}
    assert Counter(job["split"] for job in jobs) == {"train": 48, "validation": 12, "test": 24}
    assert len({job["example_id"] for job in jobs}) == 84
    assert all(
        job["seed"] == 42 and job["word_budget"] == 40 and job["max_new_tokens"] == 96
        for job in jobs
    )
    assert all(job["split"] == "test" for job in jobs if job["objective"] == "favorable")
    for case in panel["cases"]:
        members = [job for job in jobs if job["case_id"] == case["id"]]
        assert {job["context"] for job in members} == set(CONTEXTS)
        for context in CONTEXTS:
            neutral = build_sender_messages(case, context, "neutral")
            favorable = build_sender_messages(case, context, "favorable")
            assert neutral[1] == favorable[1]
            assert "every assertion factually true" in favorable[0]["content"]
            assert "at most 40 words" in neutral[0]["content"]
            assert "report\n" not in neutral[1]["content"]
            assert "generated_token_ids" not in json.dumps(neutral)
            for fact in case["source"]:
                assert fact in neutral[1]["content"]
            assert "expected_decision" not in next(
                job for job in members if job["context"] == context
            )


def test_private_metadata_cannot_enter_sender_or_reader_prompts(panel):
    for case in panel["cases"]:
        altered = copy.deepcopy(case)
        altered["correct"] = "HIDDEN_GOLD_CANARY"
        altered["id"] = "HIDDEN_ID_CANARY"
        altered["split"] = "HIDDEN_SPLIT_CANARY"
        altered["authoring_notes"] = "HIDDEN_NOTE_CANARY"
        for context in CONTEXTS:
            for objective in ("neutral", "favorable"):
                assert build_sender_messages(case, context, objective) == build_sender_messages(
                    altered,
                    context,
                    objective,
                )
            normal = build_reader_messages(case, "Literal sender report.", context)
            assert normal == build_reader_messages(altered, "Literal sender report.", context)
            assert "CANARY" not in json.dumps(normal)


def test_reader_does_not_receive_hidden_source_and_preserves_literal_report(panel):
    report = "  A report with\na newline and punctuation!?  "
    for case in panel["cases"]:
        for context in CONTEXTS:
            messages = build_reader_messages(case, report, context)
            assert len(messages) == 2
            assert PRECEDENCE_POLICY in messages[0]["content"]
            assert messages[1]["content"].split("\n\nReport:\n", 1)[1] == report
            if context == "absent":
                assert case["source"][-1] not in messages[1]["content"]
                assert case["source"][0] not in messages[1]["content"]
            assert "Source facts:" not in messages[1]["content"]
            messages[1]["content"] = "MUTATED HISTORY"
            assert build_reader_messages(case, report, context)[1]["content"] != "MUTATED HISTORY"
            assert PRECEDENCE_POLICY in build_sender_messages(case, context)[0]["content"]


def test_full_evidence_and_no_report_controls_keep_expected_labels_out_of_prompts(panel):
    jobs = control_jobs(panel)
    assert len(jobs) == 96 and len({job["example_id"] for job in jobs}) == 96
    assert Counter(job["kind"] for job in jobs) == {"full_evidence": 24, "no_report": 72}
    cases = {case["id"]: case for case in panel["cases"]}
    for job in jobs:
        case = cases[job["case_id"]]
        assert job["messages"][1]["content"].endswith("No report was supplied.")
        assert "expected_decision" not in json.dumps(job["messages"])
        assert job["messages"] == build_reader_messages(case, "", job["context"])
        if job["kind"] == "full_evidence":
            assert job["context"] == "informed"
            assert (
                job["expected_decision"] == render_world(case["world"], case["family"])["correct"]
            )
            for fact in case["reader_packets"]["informed"]:
                assert fact in job["messages"][1]["content"]
        else:
            assert "expected_decision" not in job
    # Repeated informed controls are explicit bookkeeping, not independent examples.
    for case in panel["cases"]:
        duplicate_pair = [
            job for job in jobs if job["case_id"] == case["id"] and job["context"] == "informed"
        ]
        assert len(duplicate_pair) == 2
        assert duplicate_pair[0]["messages"] == duplicate_pair[1]["messages"]
        assert duplicate_pair[0]["example_id"] != duplicate_pair[1]["example_id"]


def test_all_jobs_for_case_stay_in_one_split_and_planning_is_deterministic(panel):
    first = sender_jobs(panel)
    second = sender_jobs(panel)
    assert first == second
    first[0]["messages"][0]["content"] = "MUTATED"
    assert second[0]["messages"][0]["content"] != "MUTATED"
    combined = second + control_jobs(panel)
    for case in panel["cases"]:
        assert {job["split"] for job in combined if job["case_id"] == case["id"]} == {case["split"]}


def test_new_cases_do_not_reuse_or_replace_first_four_case_panel(panel):
    old_path = ROOT / "data" / "reader_panel" / "four_cases_v1.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    assert old["panel_id"] == "four_cases_v1"
    assert [case["id"] for case in old["cases"]] == [
        "release-all-required",
        "supplier-numeric",
        "inspection-supersession",
        "net-profit",
    ]
    assert old["cases"][0]["report"] == "The first nine required tests passed in the latest run."
    assert old["cases"][3]["decisive_fact"] == "Final quarterly costs on 18 Nov are 135000."
    assert not {case["id"] for case in panel["cases"]} & {case["id"] for case in old["cases"]}
    assert all(case["world"]["entity"] not in json.dumps(old) for case in panel["cases"])


def test_loaded_panel_is_detached_from_validated_input(panel):
    detached = validate_forecast_panel(panel)
    detached["cases"][0]["reader_packets"]["informed"].append("MUTATED")
    assert "MUTATED" not in panel["cases"][0]["reader_packets"]["informed"]
