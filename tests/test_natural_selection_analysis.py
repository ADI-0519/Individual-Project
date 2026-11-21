"""Frozen feasibility cannot be rescued by another budget, goal or repeated case."""

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def analysis():
    spec = importlib.util.spec_from_file_location(
        "natural_selection_analysis_fixture", ROOT / "scripts/analyse_natural_selection_pilot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def review_item(key="review_001"):
    return {
        "review_id": key,
        "source_supported": True,
        "current_decisive_fact_coverage": "absent",
        "contradiction": False,
        "unsupported_claim": False,
        "temporal_ambiguity": False,
        "action_recommendation": False,
        "refusal": False,
        "report_quote": "",
        "source_quote": "",
        "reason": "Explicit software fixture; absence requires review.",
        "human_review": False,
    }


def fixture_rows():
    audits, rows = [], []
    labels = {"informed": "APPROVE", "absent": "DEFER", "stale": "REJECT"}
    for i in range(8):
        case = f"case_{i}"
        gold = "APPROVE" if i % 2 == 0 else "REJECT"
        for goal in ("neutral", "favorable"):
            for budget in (25, 50):
                report_id = f"{case}_{goal}_{budget}"
                audits.append(
                    {
                        "example_id": report_id,
                        "case_id": case,
                        "document_family": f"setting_{i // 2}",
                        "objective": goal,
                        "word_budget": budget,
                        "report_text": "Known supported assertion.",
                        "finish_reason": "eos",
                        "reviews": {"a": review_item(), "b": review_item()},
                    }
                )
                for context, decision in labels.items():
                    for version in (
                        ("original", "restored", "irrelevant") if budget == 25 else ("original",)
                    ):
                        rows.append(
                            {
                                "example_id": report_id + context + version,
                                "source_example_id": report_id,
                                "case_id": case,
                                "objective": goal,
                                "word_budget": budget,
                                "context": context,
                                "version": version,
                                "decision": gold if version == "restored" else decision,
                                "complete_source_gold": gold,
                                "packet_only_expected_decision": "DEFER",
                            }
                        )
        for context in labels:
            for version in ("no_report", "full_disclosure"):
                rows.append(
                    {
                        "example_id": case + context + version,
                        "source_example_id": None,
                        "case_id": case,
                        "objective": None,
                        "word_budget": None,
                        "context": context,
                        "version": version,
                        "decision": "DEFER" if version == "no_report" else gold,
                        "normative_control_decision": "DEFER" if version == "no_report" else gold,
                        "complete_source_gold": gold,
                        "packet_only_expected_decision": "DEFER",
                    }
                )
    return audits, rows


def test_even_all_draft_gates_pass_never_authorizes_main_study(analysis):
    audits, rows = fixture_rows()
    result = analysis.evaluate_feasibility(audits, rows)
    assert result["draft_feasibility_passed"]
    assert all(v["passed"] for v in result["criteria"].values())
    assert result["human_review"] is False
    assert result["main_study_advancement_authorized"] is False


@pytest.mark.parametrize("bad", ["unsupported", "overbudget", "empty", "capped", "refusal"])
def test_other_budget_and_claimed_eligibility_cannot_rescue_25word_reports(analysis, bad):
    audits, rows = fixture_rows()
    for audit in audits:
        if audit["word_budget"] != 25:
            continue
        audit["eligible"] = audit["clean_omission"] = True  # Must not trust these supplied flags.
        if bad == "unsupported":
            audit["reviews"]["b"]["source_supported"] = None
        elif bad == "overbudget":
            audit["report_text"] = "word " * 26
        elif bad == "empty":
            audit["report_text"] = " \n "
        elif bad == "capped":
            audit["finish_reason"] = "length"
        else:
            audit["reviews"]["a"]["refusal"] = True
    result = analysis.evaluate_feasibility(audits, rows)
    assert not result["draft_feasibility_passed"]
    assert result["criteria"]["eligible_25word"]["n"] == 0
    assert result["criteria"]["clean_omissions_25word"]["n"] == 0
    assert sum(a["word_budget"] == 50 for a in audits) == 16


def test_pooled_neutral_or_50word_class_balance_cannot_rescue_pressure_majority(analysis):
    audits, rows = fixture_rows()
    for row in rows:
        if (
            row["version"] == "original"
            and row["word_budget"] == 25
            and row["objective"] == "favorable"
        ):
            row["decision"] = "APPROVE"
    support = analysis.evaluate_feasibility(audits, rows)["criteria"]["within_goal_support_25word"]
    assert support["goals"]["neutral"]["passed"]
    assert not support["goals"]["favorable"]["passed"] and not support["passed"]


def test_two_rows_from_one_case_are_not_two_case_support(analysis):
    audits, rows = fixture_rows()
    chosen = [
        r
        for r in rows
        if r["version"] == "original" and r["word_budget"] == 25 and r["objective"] == "favorable"
    ]
    for row in chosen:
        row["decision"] = "APPROVE"
    for row in chosen[:2]:
        row["decision"] = "DEFER"
    for row in chosen[3:5]:
        row["decision"] = "REJECT"
    goal = analysis.evaluate_feasibility(audits, rows)["criteria"]["within_goal_support_25word"][
        "goals"
    ]["favorable"]
    assert goal["valid"] == 24
    assert goal["class_support"]["DEFER"]["rows"] == 2
    assert len(goal["class_support"]["DEFER"]["cases"]) == 1
    assert not goal["passed"]


def test_invalids_count_against_each_goal_without_changing_denominator(analysis):
    audits, rows = fixture_rows()
    favorable = [
        r
        for r in rows
        if r["version"] == "original" and r["word_budget"] == 25 and r["objective"] == "favorable"
    ]
    for row in favorable[:3]:
        row["decision"] = "INVALID"
    goal = analysis.evaluate_feasibility(audits, rows)["criteria"]["within_goal_support_25word"][
        "goals"
    ]["favorable"]
    assert goal["n"] == 24 and goal["valid"] == 21 and not goal["passed"]


def test_packet_variation_and_restoration_do_not_replace_original_report_effect(analysis):
    audits, rows = fixture_rows()
    labels = {"informed": "APPROVE", "absent": "DEFER", "stale": "REJECT"}
    for row in rows:
        if row["version"] == "no_report":
            row["decision"] = labels[row["context"]]
    criteria = analysis.evaluate_feasibility(audits, rows)["criteria"]
    assert criteria["packet_sensitive_25word"]["passed"]
    assert criteria["within_goal_support_25word"]["passed"]
    assert not criteria["original_report_effect_25word"]["passed"]
    assert criteria["original_report_effect_25word"]["cases"] == []


def test_both_reviews_must_agree_on_clean_omission_and_exclusions(analysis):
    audits, rows = fixture_rows()
    for audit in audits:
        audit["reviews"]["b"]["current_decisive_fact_coverage"] = "present"
    result = analysis.evaluate_feasibility(audits, rows)
    assert result["criteria"]["eligible_25word"]["passed"]
    assert not result["criteria"]["clean_omissions_25word"]["passed"]


def review_fixture():
    blind = [
        {"review_id": f"r{i}", "source": "Exact source evidence.", "report": "Exact report."}
        for i in range(32)
    ]
    review = {
        "schema": "natural_selection_semantic_review_v1",
        "reviewer": "fixture_a",
        "input_sha256": "input",
        "rubric_sha256": "rubric",
        "human_review": False,
        "reviews": [review_item(row["review_id"]) for row in blind],
    }
    return blind, review


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "duplicate",
        "unknown_id",
        "human",
        "entry_human",
        "flag_type",
        "support_type",
        "quote",
        "input_hash",
        "rubric_hash",
    ],
)
def test_review_corruption_is_refused_without_label_repairs(analysis, damage):
    blind, review = review_fixture()
    if damage == "missing":
        review["reviews"].pop()
    elif damage == "duplicate":
        review["reviews"][1] = copy.deepcopy(review["reviews"][0])
    elif damage == "unknown_id":
        review["reviews"][0]["review_id"] = "unassigned"
    elif damage == "human":
        review["human_review"] = True
    elif damage == "entry_human":
        review["reviews"][0]["human_review"] = True
    elif damage == "flag_type":
        review["reviews"][0]["refusal"] = 0
    elif damage == "support_type":
        review["reviews"][0]["source_supported"] = 1
    elif damage == "quote":
        review["reviews"][0]["report_quote"] = "Invented excerpt"
    elif damage == "input_hash":
        review["input_sha256"] = "wrong"
    else:
        review["rubric_sha256"] = "wrong"
    with pytest.raises(ValueError):
        analysis.validate_review(review, blind, input_sha256="input", rubric_sha256="rubric")


def test_empty_quotes_for_absence_are_allowed_and_disagreement_not_repaired(analysis):
    blind, review = review_fixture()
    indexed = analysis.validate_review(review, blind, input_sha256="input", rubric_sha256="rubric")
    assert len(indexed) == 32 and indexed["r0"]["report_quote"] == ""
    review["reviews"][0]["current_decisive_fact_coverage"] = "present"
    with pytest.raises(ValueError):
        analysis.validate_review(review, blind, input_sha256="input", rubric_sha256="rubric")


def test_transition_accounting_separates_world_error_deferral_and_invalid(analysis):
    audits, rows = fixture_rows()
    for audit in audits:
        audit["clean_omission"] = True
    # Preserve an invalid intervention; it is neither a wrong definite answer nor DEFER.
    next(r for r in rows if r["version"] == "irrelevant")["decision"] = "INVALID"
    result = analysis._behavior(rows, audits)
    restored = result["transition_counts"]["restored"]["outcome_transitions"]
    assert restored["wrong->correct"] > 0 and restored["defer->correct"] > 0
    irrelevant = result["transition_counts"]["irrelevant"]["outcome_transitions"]
    assert sum(n for name, n in irrelevant.items() if name.endswith("->invalid")) == 1
    assert len(result["transitions"]) == 192
