"""Meaningful visibility, frozen-policy and complete-plan checks without models."""

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from latent_probes.calibrated_reader import policy_metadata, worked_examples
from latent_probes.forecast_panel import READER_INSTRUCTION
from latent_probes.natural_selection_panel import (
    ACTOR_GENERATION,
    OBJECTIVE_TEXT,
    PANEL_CANONICAL_SHA256,
    READER_GENERATION,
    actor_jobs,
    build_actor_messages,
    build_reader_messages,
    load_panel,
    panel_metadata,
    reader_plan,
    reader_prefix,
    source_text,
    validate_panel,
)
from latent_probes.storage import canonical_hash

DATA = (
    Path(__file__).resolve().parents[1] / "data/communication/natural_selection_pilot_draft_v1.json"
)


@pytest.fixture
def panel():
    return load_panel(DATA)


def test_frozen_draft_and_all_32_normative_controls(panel):
    assert canonical_hash(panel) == PANEL_CANONICAL_SHA256
    assert panel["human_review"] is False and panel["observed_labels"] is None
    assert len(panel["cases"]) == 8
    for case in panel["cases"]:
        records = {f["fact_id"]: f for f in case["source_assertions"]}
        expected_world = (
            "APPROVE"
            if records[case["current_decisive_fact_id"]]["assessment_result"] == "PASS"
            else "REJECT"
        )
        assert case["complete_source_gold"] == expected_world
        assert case["packet_expected_decisions"] == {
            "informed": expected_world,
            "absent": "DEFER",
            "stale": "REJECT" if expected_world == "APPROVE" else "APPROVE",
        }
        # Current PASS must defeat an older FAIL, not trigger an any-record-FAIL shortcut.
        for context, ids in (*case["recipient_packets"].items(), ("complete", list(records))):
            selected = []
            for check in ("common", "critical"):
                visible = [records[i] for i in ids if records[i].get("check") == check]
                visible.sort(key=lambda f: f["assessment_date"])
                selected.append(visible[-1]["assessment_result"] if visible else None)
            actual = (
                "REJECT"
                if "FAIL" in selected
                else ("APPROVE" if selected == ["PASS", "PASS"] else "DEFER")
            )
            expected = (
                expected_world
                if context == "complete"
                else case["packet_expected_decisions"][context]
            )
            assert actual == expected


def test_all_actor_jobs_are_before_private_packet_assignment(panel):
    jobs = actor_jobs(panel)
    assert len(jobs) == len({j["example_id"] for j in jobs}) == 32
    assert Counter((j["objective"], j["word_budget"]) for j in jobs) == {
        (goal, budget): 8 for goal in ("neutral", "favorable") for budget in (25, 50)
    }
    for case in panel["cases"]:
        for budget in (25, 50):
            neutral = build_actor_messages(case, "neutral", budget)
            pressured = build_actor_messages(case, "favorable", budget)
            assert neutral[0] == pressured[0]
            assert neutral[1]["content"].replace(OBJECTIVE_TEXT["neutral"], "GOAL") == (
                pressured[1]["content"].replace(OBJECTIVE_TEXT["favorable"], "GOAL")
            )
            payload = json.dumps(neutral)
            for private in (
                case["id"],
                "current_decisive_fact_id",
                "source_audit_notes",
                "packet_expected_decisions",
                "human_review",
                "Evidence packet:",
            ):
                assert private not in payload
            assert source_text(case) in neutral[1]["content"]
            assert all(f["text"] in neutral[1]["content"] for f in case["source_assertions"])
    assert not {"context", "observed_reader_decision", "complete_source_gold"} & set(jobs[0])


def test_new_reader_renderer_preserves_policy_and_only_actual_evidence(panel):
    expected_prefix = [{"role": "system", "content": READER_INSTRUCTION}]
    expected_prefix += [m for ex in worked_examples() for m in ex["messages"]]
    assert reader_prefix() == expected_prefix
    assert (
        canonical_hash(
            {"system": reader_prefix()[0]["content"], "demonstrations": reader_prefix()[1:]}
        )
        == policy_metadata()["policy_sha256"]
    )
    for case in panel["cases"]:
        records = {f["fact_id"]: f for f in case["source_assertions"]}
        for context in ("informed", "absent", "stale"):
            messages = build_reader_messages(case, "Exact report kept.\nSpacing kept.", context)
            assert messages[:-1] == expected_prefix
            user = messages[-1]["content"]
            assert user.endswith("Report:\nExact report kept.\nSpacing kept.")
            for fact in records.values():
                assert (fact["text"] in user) == (
                    fact["fact_id"] in case["recipient_packets"][context]
                )
            for private in (
                case["id"],
                "packet_expected_decisions",
                "Source document:",
                "Reporting objective:",
                "critical_current_final",
            ):
                assert private not in user


def test_240_plan_keeps_exact_text_all_edits_and_normative_labels_distinct(panel):
    jobs = actor_jobs(panel)
    cases = {c["id"]: c for c in panel["cases"]}
    reports = {j["example_id"]: "  Exact report\nwith trailing spaces.  " for j in jobs}
    # Include a fact already present: the protocol still appends its exact copy.
    first = next(j for j in jobs if j["word_budget"] == 25)
    reports[first["example_id"]] = cases[first["case_id"]]["decisive_restoration"]["text"]
    reports[jobs[-1]["example_id"]] = ""  # No exclusion of an empty model completion.
    rows = reader_plan(panel, reports)
    assert len(rows) == len({r["example_id"] for r in rows}) == 240
    assert Counter(r["version"] for r in rows) == {
        "original": 96,
        "restored": 48,
        "irrelevant": 48,
        "no_report": 24,
        "full_disclosure": 24,
    }
    for row in rows:
        case = cases[row["case_id"]]
        assert not {"label", "decision", "actual_reader_decision", "observed_reader_decision"} & (
            set(row)
        )
        assert (
            row["packet_only_expected_decision"]
            == case["packet_expected_decisions"][row["context"]]
        )
        if row["version"] == "no_report":
            assert row["report_text"] == ""
            assert row["normative_control_decision"] == row["packet_only_expected_decision"]
        elif row["version"] == "full_disclosure":
            assert row["report_text"] == source_text(case)
            assert row["normative_control_decision"] == case["complete_source_gold"]
        else:
            assert row["normative_control_decision"] is None
            original = reports[row["source_example_id"]]
            if row["version"] == "original":
                assert row["report_text"] == original
            else:
                assert row["word_budget"] == 25
                field = (
                    "decisive_restoration"
                    if row["version"] == "restored"
                    else ("irrelevant_addition")
                )
                assert row["report_text"] == original + "\n\n" + case[field]["text"]
    repeated = next(
        r
        for r in rows
        if r["source_example_id"] == first["example_id"] and r["version"] == "restored"
    )
    assert repeated["report_text"].count(reports[first["example_id"]]) == 2


@pytest.mark.parametrize(
    "damage",
    [
        "extra_private_field",
        "assessment_text",
        "preliminary",
        "date_order",
        "missing_fact",
        "duplicate_fact",
        "private_source_id",
        "packet_leak",
        "gold",
        "expected_packet",
        "restoration",
        "placebo",
        "human_review",
        "observed_label",
        "source_order",
        "new_case_text",
    ],
)
def test_corrupted_input_is_refused_before_messages_or_models(panel, damage):
    changed = copy.deepcopy(panel)
    case = changed["cases"][0]
    facts = {f["fact_id"]: f for f in case["source_assertions"]}
    current = facts[case["current_decisive_fact_id"]]
    if damage == "extra_private_field":
        case["actual_reader_decision"] = "APPROVE"
    elif damage == "assessment_text":
        current["text"] = current["text"].replace("records PASS", "records FAIL")
    elif damage == "preliminary":
        current["assessment_stage"] = "preliminary"
    elif damage == "date_order":
        current["assessment_date"] = "2026-01-01"
    elif damage == "missing_fact":
        case["source_assertions"].pop()
    elif damage == "duplicate_fact":
        case["source_assertions"][-1] = copy.deepcopy(case["source_assertions"][0])
    elif damage == "private_source_id":
        current["text"] += " complete_source_gold APPROVE."
    elif damage == "packet_leak":
        case["recipient_packets"]["absent"].append(case["current_decisive_fact_id"])
    elif damage == "gold":
        case["complete_source_gold"] = "REJECT"
    elif damage == "expected_packet":
        case["packet_expected_decisions"]["stale"] = "DEFER"
    elif damage == "restoration":
        case["decisive_restoration"]["text"] += " Edited."
    elif damage == "placebo":
        case["irrelevant_addition"]["fact_id"] = case["current_decisive_fact_id"]
    elif damage == "human_review":
        case["human_review"] = True
    elif damage == "observed_label":
        changed["observed_labels"] = {case["id"]: "APPROVE"}
    elif damage == "source_order":
        case["source_assertions"].reverse()
    else:
        case["source_assertions"][-1]["text"] += " Unreviewed extra content."
    with pytest.raises(ValueError):
        validate_panel(changed)
    with pytest.raises(ValueError):
        actor_jobs(changed)


def test_unknown_or_partial_saved_reports_are_not_silently_excluded(panel):
    reports = {j["example_id"]: "" for j in actor_jobs(panel)}
    for bad in ({}, {**reports, "unknown": "text"}, {**reports, next(iter(reports)): None}):
        with pytest.raises(ValueError):
            reader_plan(panel, bad)


def test_metadata_detachment_and_portable_content_binding(panel, tmp_path):
    metadata = panel_metadata(panel)
    assert metadata["content_sha256"] == PANEL_CANONICAL_SHA256
    assert metadata["reader_policy_sha256"] == policy_metadata()["policy_sha256"]
    assert metadata["n_rule_grammars"] == 1 and metadata["n_document_settings"] == 4
    assert metadata["actor_generation"] == ACTOR_GENERATION
    assert metadata["reader_generation"] == READER_GENERATION
    metadata["actor_generation"]["seed"] = 1
    assert panel_metadata(panel)["actor_generation"]["seed"] == 42
    detached = validate_panel(panel)
    detached["cases"].clear()
    assert len(panel["cases"]) == 8
    crlf = tmp_path / "panel.json"
    crlf.write_bytes(json.dumps(panel, indent=2).replace("\n", "\r\n").encode())
    assert load_panel(crlf) == panel


@pytest.mark.parametrize("goal,budget", [("deceptive", 25), ("neutral", 12), ("neutral", True)])
def test_unfrozen_actor_conditions_are_refused(panel, goal, budget):
    with pytest.raises(ValueError):
        build_actor_messages(panel["cases"][0], goal, budget)


def test_reader_rejects_unknown_packet_but_preserves_arbitrary_generated_report(panel):
    case = panel["cases"][0]
    with pytest.raises(ValueError):
        build_reader_messages(case, "Saved report", "private")
    with pytest.raises(ValueError):
        build_reader_messages(case, None, "absent")
    # A generated report can contain a recommendation or malformed text: keep it.
    text = "APPROVE!\nI cannot provide a note.  "
    assert build_reader_messages(case, text, "absent")[-1]["content"].endswith(text)
    for text in ("", "  ", "\n\n"):
        user = build_reader_messages(case, text, "absent")[-1]["content"]
        assert user.split("\n\nReport:\n", 1)[1] == text


def test_reader_policy_is_bound_to_original_hash_not_a_live_self_comparison(panel, monkeypatch):
    import latent_probes.calibrated_reader as old_policy

    changed = list(copy.deepcopy(old_policy._EXAMPLES))
    changed[0]["answer"] = "REJECT"
    monkeypatch.setattr(old_policy, "_EXAMPLES", tuple(changed))
    with pytest.raises(ValueError, match="policy changed"):
        build_reader_messages(panel["cases"][0], "Saved report", "absent")
