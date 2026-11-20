"""Fresh communication inputs, evidence visibility and leakage guards, without models."""

import copy
import json
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from latent_probes.communication_panel import (
    CONTEXTS,
    FAMILIES,
    NO_RECOMMENDATION,
    audit_disjointness,
    authored_panel,
    build_reader_messages,
    load_panel,
    render_case,
    sender_jobs,
    validate_panel,
)
from latent_probes.forecast_panel import READER_INSTRUCTION

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "communication" / "fresh_transfer_v1.json"
REFERENCE = ROOT / "data" / "reader_forecast" / "development_v1.json"


@pytest.fixture
def panel():
    return load_panel(DATA)


def test_static_panel_matches_authored_inputs_and_is_fresh(panel):
    assert panel == authored_panel()
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    result = audit_disjointness(panel, reference)
    assert result["ids_entities_assertions_record_dates_and_family_numbers_disjoint"]
    assert len(panel["cases"]) == 24
    assert Counter(c["split"] for c in panel["cases"]) == {"test": 24}
    assert Counter(c["transfer_group"] for c in panel["cases"]) == {
        "fresh_case_seen_rule": 16,
        "heldout_rule_family": 8,
    }
    for family in FAMILIES:
        members = [c for c in panel["cases"] if c["family"] == family]
        assert Counter(c["correct"] for c in members) == {"APPROVE": 2, "REJECT": 2}


def test_world_decisions_are_independently_computed_and_absence_is_unknown(panel):
    for case in panel["cases"]:
        world, family = case["world"], case["family"]
        if family == "conjunction":
            approved = world["decisive_pass"]
        elif family == "threshold":
            approved = world["score"] >= world["threshold"]
        elif family == "supersession":
            approved = world["current_pass"]
        elif family == "arithmetic":
            approved = world["revenue"] >= world["costs"]
        elif family == "disjunction":
            approved = world["route_b_pass"]  # A is known to fail in both worlds.
        else:
            approved = date.fromisoformat(world["completion_date"]) <= date.fromisoformat(
                world["deadline"]
            )
        assert case["correct"] == ("APPROVE" if approved else "REJECT")
        assert case["decisive_fact"] in case["source"]
        assert case["irrelevant_fact"] in case["source"]
        assert case["decisive_fact"] not in case["reader_packets"]["absent"]
        assert case["packet_expected_decisions"] == {
            "informed": case["correct"],
            "absent": "DEFER",
            "stale": "REJECT" if approved else "APPROVE",
        }


def test_unknown_new_family_fact_can_support_either_world_decision(panel):
    # This proves no decisive conclusion follows from the common-only packet.
    for family in ("disjunction", "deadline"):
        case = next(c for c in panel["cases"] if c["family"] == family)
        alternative = copy.deepcopy(case["world"])
        if family == "disjunction":
            alternative["route_b_pass"] = not alternative["route_b_pass"]
            alternative["previous_route_b_pass"] = not alternative["previous_route_b_pass"]
        else:
            alternative["completion_date"], alternative["previous_completion_date"] = (
                alternative["previous_completion_date"],
                alternative["completion_date"],
            )
        other = render_case(alternative, family)
        assert other["rule"] == case["rule"]
        assert other["reader_packets"]["absent"] == case["reader_packets"]["absent"]
        assert other["correct"] != case["correct"]


def test_deadline_includes_the_boundary_and_observes_record_precedence(panel):
    case = next(c for c in panel["cases"] if c["family"] == "deadline")
    world = copy.deepcopy(case["world"])
    world["completion_date"] = world["deadline"]
    output = render_case(world, "deadline")
    assert output["correct"] == "APPROVE"
    assert output["packet_expected_decisions"]["stale"] == "REJECT"
    assert world["previous_completion_date"] in output["reader_packets"]["stale"][-1]


def test_all_144_natural_report_jobs_have_matched_goal_pairs(panel):
    jobs = sender_jobs(panel)
    assert len(jobs) == len({j["example_id"] for j in jobs}) == 144
    for case in panel["cases"]:
        for context in CONTEXTS:
            pair = [j for j in jobs if j["case_id"] == case["id"] and j["context"] == context]
            assert {j["objective"] for j in pair} == {"neutral", "favorable"}
            assert pair[0]["messages"][1] == pair[1]["messages"][1]
            for job in pair:
                assert NO_RECOMMENDATION in job["messages"][0]["content"]
                assert job["expected_world_decision"] == case["correct"]
                assert "transfer_group" not in json.dumps(job["messages"])
                assert case["id"] not in json.dumps(job["messages"])


def test_reader_gets_only_own_packet_and_literal_report(panel):
    for case in panel["cases"]:
        for context in CONTEXTS:
            report = "The submitted report is preserved exactly."
            messages = build_reader_messages(case, report, context)
            assert messages[0]["content"] == READER_INSTRUCTION
            assert messages[1]["content"] == (
                "Decision rule:\n"
                + case["rule"]
                + "\n\nEvidence packet:\n"
                + "\n".join(case["reader_packets"][context])
                + "\n\nReport:\n"
                + report
            )
            assert "strongly want" not in json.dumps(messages)
            assert case["irrelevant_fact"] not in json.dumps(messages)
        no_report = build_reader_messages(case, "", "absent")
        assert no_report[-1]["content"].endswith("No report was supplied.")


@pytest.mark.parametrize(
    "corruption", ["gold", "packet", "source", "group", "split", "id", "balance"]
)
def test_fresh_case_content_and_all_test_partition_cannot_be_changed(panel, corruption):
    changed = copy.deepcopy(panel)
    case = changed["cases"][0]
    if corruption == "gold":
        case["correct"] = "REJECT"
    elif corruption == "packet":
        case["reader_packets"]["absent"].append(case["decisive_fact"])
    elif corruption == "source":
        case["source"][-1] = "This file changes the rule."
    elif corruption == "group":
        case["transfer_group"] = "heldout_rule_family"
    elif corruption == "split":
        case["split"] = "train"
    elif corruption == "id":
        case["id"] = "rf_conjunction_01"
    else:
        case["world"]["decisive_pass"] = False
        case["world"]["previous_decisive_pass"] = True
        case.update(render_case(case["world"], case["family"]))
    with pytest.raises(ValueError):
        validate_panel(changed)


def test_overlap_audit_rejects_reused_instantiations_even_with_fresh_text(panel):
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    for key in ("entity", "current_date", "required_count"):
        changed = copy.deepcopy(panel)
        changed["cases"][0]["world"][key] = reference["cases"][0]["world"][key]
        with pytest.raises(ValueError):
            audit_disjointness(changed, reference)


@pytest.mark.parametrize("family", ["disjunction", "deadline"])
def test_new_family_stale_evidence_must_disagree_and_dates_are_ordered(panel, family):
    case = next(c for c in panel["cases"] if c["family"] == family)
    world = copy.deepcopy(case["world"])
    if family == "disjunction":
        world["previous_route_b_pass"] = world["route_b_pass"]
    else:
        world["previous_completion_date"] = world["completion_date"]
    with pytest.raises(ValueError, match="Stale"):
        render_case(world, family)
    world = copy.deepcopy(case["world"])
    world["current_date"] = world["previous_date"]
    with pytest.raises(ValueError, match="follow"):
        render_case(world, family)
