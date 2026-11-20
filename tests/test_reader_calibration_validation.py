"""Fresh-bank integrity, input visibility and a control gate that genuinely stops."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from latent_probes.storage import read_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "calibration_validation", ROOT / "scripts/validate_calibrated_reader.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs():
    return read_json(ROOT / "data/communication/fresh_calibration_v1.json"), [
        read_json(ROOT / "data/reader_forecast/development_v1.json"),
        read_json(ROOT / "data/communication/fresh_transfer_v1.json"),
    ]


def test_fresh_bank_is_all_test_balanced_and_disjoint(runner):
    panel, references = inputs()
    assert runner.validate_bank(panel, references) == panel
    assert len({case["id"] for case in panel["cases"]}) == 24
    assert {case["split"] for case in panel["cases"]} == {"test"}
    for reference in references:
        changed = copy.deepcopy(panel)
        changed["cases"][0]["world"]["entity"] = reference["cases"][0]["world"]["entity"]
        with pytest.raises(ValueError):
            runner.validate_bank(changed, references)
    changed = copy.deepcopy(panel)
    changed["cases"][0]["correct"] = "REJECT"
    with pytest.raises(ValueError, match="normative label"):
        runner.validate_bank(changed, references)


def test_actual_reader_turn_has_only_assigned_packet_no_source_or_gold(runner):
    panel, _ = inputs()
    jobs = runner.control_jobs(panel, "fresh_validation")
    assert len(jobs) == 72
    for case in panel["cases"]:
        absent = next(
            job for job in jobs if job["case_id"] == case["id"] and job["context"] == "absent"
        )
        actual = absent["messages"][-1]["content"]
        assert case["rule"] in actual
        assert case["reader_packets"]["absent"][0] in actual
        assert case["decisive_fact"] not in actual
        assert case["source"][0] not in actual
        assert case["id"] not in actual and "expected_decision" not in actual
        assert actual.endswith("No report was supplied.")


def test_failed_fresh_gate_stops_before_any_old_case_diagnostic(runner, tmp_path):
    panel, references = inputs()
    fresh = runner.control_jobs(panel, "fresh_validation")
    old = runner.control_jobs(references[1], "secondary_observed_main")
    calls = []

    def generate(messages, settings, seed):
        calls.append(messages)
        assert settings.max_new_tokens == 8 and not settings.do_sample and seed == 42
        return SimpleNamespace(record=lambda: {"text": "APPROVE", "finish_reason": "eos"})

    summary = runner.run_validation_batches(
        SimpleNamespace(generate=generate), fresh, old, tmp_path
    )
    assert summary["status"] == "fresh_reader_controls_failed"
    assert len(calls) == summary["n_fresh_calls"] == 72
    assert summary["n_secondary_calls"] == 0
    assert not (tmp_path / "secondary_old_controls.json").exists()


def test_gate_preserves_fixed_thresholds_and_invalid_limit(runner):
    rows = [
        {"case_id": str(i), "context": context, "decision": "DEFER", "expected_decision": "DEFER"}
        for i in range(24)
        for context in runner.CONTEXTS
    ]
    assert runner.capability_gate(rows)["passed"]
    for row in [row for row in rows if row["context"] == "absent"][:5]:
        row["decision"] = "APPROVE"
    assert not runner.capability_gate(rows)["passed"]
    rows = [dict(row, decision=row["expected_decision"]) for row in rows]
    rows[0]["decision"] = rows[1]["decision"] = "INVALID"
    assert not runner.capability_gate(rows)["passed"]


def test_future_transfer_bank_is_disjoint_and_cannot_be_used_as_calibration(runner):
    calibration, references = inputs()
    untouched = read_json(ROOT / "data/communication/fresh_recipient_test_v1.json")
    assert (
        runner.validate_bank(untouched, [*references, calibration], purpose="recipient_test")
        == untouched
    )
    assert runner.validate_bank(calibration, [*references, untouched]) == calibration
    with pytest.raises(ValueError, match="fixed calibration-validation bank"):
        runner.validate_bank(untouched, references)
    assert {c["id"][:3] for c in untouched["cases"]} == {"ct_"}
