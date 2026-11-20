"""CPU checks for outcome-independent cap recovery and exact token replay."""

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

from latent_probes.communication_panel import authored_panel, sender_jobs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "communication_extension", ROOT / "scripts/extend_communication_monitor.py"
)
extension = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extension)


def generation(tokens=None, *, finish="length", text="FINAL: APPROVE"):
    return {
        "prompt_token_ids": [10, 20, 30],
        "generated_token_ids": list(range(160)) if tokens is None else tokens,
        "finish_reason": finish,
        "text": text,
        "elapsed_seconds": 1.25,
    }


def test_selector_uses_only_finish_flag_and_preserves_eos_forecasts():
    rows = [
        {
            "example_id": "capped",
            "decision": "APPROVE",
            "objective": "neutral",
            "monitor_generation": generation(),
        },
        {
            "example_id": "eos",
            "decision": "REJECT",
            "objective": "favorable",
            "monitor_generation": generation([91, 92], finish="eos"),
        },
    ]
    selected, cached = extension.select_recovery_rows(rows)
    assert [row["example_id"] for row in selected] == ["capped"]
    assert [row["example_id"] for row in cached] == ["eos"]
    changed = copy.deepcopy(rows)
    for row in changed:
        row.update(decision="INVALID", objective="opposite", apollo_raw_margin=999)
    assert [
        [r["example_id"] for r in group] for group in extension.select_recovery_rows(changed)
    ] == [["capped"], ["eos"]]
    changed[0]["monitor_generation"]["generated_token_ids"] = [1]
    with pytest.raises(ValueError, match="frozen cap"):
        extension.select_recovery_rows(changed)


def test_entire_prefix_and_prompt_are_required_even_when_final_answer_parses():
    original = generation()
    longer = generation(list(range(160)) + [999], finish="eos")
    assert extension.prefix_audit(original, longer)["entire_original_prefix_exact"]
    changed = copy.deepcopy(longer)
    changed["generated_token_ids"][159] = 777
    audit = extension.prefix_audit(original, changed)
    assert not audit["entire_original_prefix_exact"] and audit["first_prefix_mismatch"] == 159
    assert extension.monitor_decision(changed) == "APPROVE"
    assert not extension.prefix_audit(original, generation(list(range(159))))[
        "entire_original_prefix_exact"
    ]
    longer["prompt_token_ids"] = [10, 20, 31]
    assert not extension.prefix_audit(original, longer)["prompt_exact"]
    assert extension.monitor_decision(generation()) == "INVALID"


def test_original_monitor_prompt_excludes_saved_reader_answers_and_scores():
    panel = authored_panel()
    case, job = panel["cases"][0], sender_jobs(panel)[0]
    report = "The current source record is quoted exactly."
    baseline = extension.monitor_messages(case, job, report)
    changed = {**job, "decision": "SECRET_ACTUAL_READER", "probabilities": "SECRET_SCORE"}
    assert extension.monitor_messages(case, changed, report) == baseline
    assert "SECRET_ACTUAL_READER" not in str(baseline) and "SECRET_SCORE" not in str(baseline)


def test_recovery_summary_preserves_invalid_target_denominators_and_added_cost():
    original_rows, rows, attempts = [], [], []
    for index, target in enumerate(("APPROVE", "INVALID")):
        original = {
            "example_id": str(index),
            "case_id": "one_case",
            "family": "one_family",
            "objective": "neutral",
            "context": "absent",
            "transfer_scope": "seen",
            "world_correct_decision": "REJECT",
            "decision": target,
            "record_sha256": "fixture",
            "predictions": {"full_context_forecaster": "INVALID"},
            "monitor_generation": generation(),
        }
        new = generation(list(range(160)) + [999], finish="eos")
        audit = extension.prefix_audit(original["monitor_generation"], new)
        attempts.append({"generation": new, "replay_audit": audit})
        original_rows.append(original)
        rows.append(
            extension.forecast_row(original, new, provenance="extended_exact_prefix", audit=audit)
        )
    summary = extension.summarize_recovery(rows, original_rows, attempts)
    metric = summary["evaluations"]["objective/neutral"]["full_context_forecaster_512_recovery"]
    assert metric["n"] == 2 and metric["n_valid_targets"] == 1 and metric["invalid_targets"] == 1
    assert metric["correct"] == 1 and metric["n_cases"] == 1
    assert summary["cost"]["additional_from_scratch_recovery_calls"]["generated_tokens"] == 322
    assert summary["cost"]["original_prefix_tokens_regenerated"] == 320
    assert summary["cost"]["new_tokens_beyond_old_cap"] == 2
