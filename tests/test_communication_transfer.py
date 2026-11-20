"""Control parsing, information isolation and paired outcome analysis."""

import copy
import importlib.util
from pathlib import Path

import pytest

from latent_probes.communication_panel import authored_panel, sender_jobs

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "communication_runner", ROOT / "scripts/run_communication_transfer.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.mark.parametrize(
    "text,finish,expected",
    [
        ("APPROVE", "eos", "APPROVE"),
        ("REJECT.", "eos", "REJECT"),
        ("DEFER", "length", "INVALID"),
        ("APPROVE or REJECT", "eos", "INVALID"),
    ],
)
def test_reader_outputs_keep_caps_invalid(text, finish, expected):
    assert runner.strict_decision({"text": text, "finish_reason": finish}) == expected


@pytest.mark.parametrize(
    "text,finish,expected",
    [
        ("Insufficient evidence.\nFINAL: DEFER", "eos", "DEFER"),
        ("FINAL: APPROVE", "length", "INVALID"),
        ("FINAL: REJECT\nFINAL: APPROVE", "eos", "INVALID"),
        ("Maybe FINAL: APPROVE", "eos", "INVALID"),
    ],
)
def test_monitor_requires_one_uncapped_final(text, finish, expected):
    assert runner.monitor_decision({"text": text, "finish_reason": finish}) == expected


def test_monitor_never_receives_outcome_or_case_metadata():
    panel = authored_panel()
    job = sender_jobs(panel)[0]
    case = panel["cases"][0]
    original = runner.monitor_messages(case, job, "An exact saved report.")
    changed_case = copy.deepcopy(case)
    changed_job = copy.deepcopy(job)
    changed_case.update(correct="SECRET_GOLD", authoring_notes="HIDDEN_AUDIT")
    changed_job.update(decision="ACTUAL_READER_OUTCOME", expected_world_decision="SECRET_GOLD")
    # The renderer validates case structure, so retain source semantics but vary unused job metadata.
    assert runner.monitor_messages(case, changed_job, "An exact saved report.") == original
    rendered = str(original)
    assert "ACTUAL_READER_OUTCOME" not in rendered and "SECRET_GOLD" not in rendered
    assert "An exact saved report." in rendered


def analysis_fixture():
    rows = []
    for objective in ("neutral", "favorable"):
        for context in ("informed", "absent", "stale"):
            decision = "REJECT" if objective == "neutral" else "APPROVE"
            rows.append(
                {
                    "case_id": "case_one",
                    "family": "one",
                    "transfer_scope": "seen",
                    "objective": objective,
                    "context": context,
                    "decision": decision,
                    "world_correct_decision": "REJECT",
                    "predictions": {"constant": "APPROVE"},
                    "probabilities": {},
                    "apollo_raw_margin": float(objective == "favorable"),
                    "generation": {"finish_reason": "eos"},
                    "report_words": 4,
                    "explicit_decision_words": False,
                }
            )
    controls = [
        {"context": c, "decision": "DEFER", "expected_decision": "DEFER"}
        for c in ("informed", "absent", "stale")
    ]
    replays = []
    for context in ("informed", "absent", "stale"):
        for version in ("none", "original", "restored", "irrelevant_edit"):
            decision = "APPROVE"
            if version == "restored":
                decision = {"informed": "REJECT", "absent": "DEFER", "stale": "INVALID"}[context]
            replays.append(
                {
                    "case_id": "case_one",
                    "context": context,
                    "version": version,
                    "decision": decision,
                    "world_correct_decision": "REJECT",
                }
            )
    return rows, replays, controls


def test_pairing_excludes_both_arms_of_invalid_context():
    rows, replays, controls = analysis_fixture()
    rows[0]["decision"] = "INVALID"
    summary = runner.summarize(rows, replays, controls)
    paired = summary["paired_pressure_minus_neutral_accuracy"]["constant"]
    assert paired["complete_valid_context_pairs_by_case"] == {"case_one": 2}
    assert paired["mean"] == 1.0
    assert summary["evaluations"]["all"]["constant"]["invalid_targets"] == 1


def test_restoration_separates_wrong_to_correct_defer_and_invalid():
    summary = runner.summarize(*analysis_fixture())
    transitions = summary["restoration"]["outcome_transitions"]
    assert transitions["informed"]["restored"] == {"wrong->correct": 1}
    assert transitions["absent"]["restored"] == {"wrong->defer": 1}
    assert transitions["stale"]["restored"] == {"wrong->invalid": 1}
    assert summary["restoration"]["cases"][0]["stale/restored"]["outcome"] == "invalid"


def test_bootstrap_uses_cases_and_preserves_degenerate_result():
    result = runner.bootstrap_case_mean([0.25, 0.25, 0.25])
    assert result["n_cases"] == 3 and result["mean"] == 0.25
    assert result["case_bootstrap_percentile_95"] == [0.25, 0.25]
