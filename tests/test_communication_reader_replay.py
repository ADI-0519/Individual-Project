"""Capability gates are fixed before any independent-reader outputs."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "independent_reader_replay", ROOT / "scripts/replay_communication_reader.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def controls():
    return [
        {
            "case_id": str(i),
            "context": context,
            "decision": "DEFER" if context == "absent" else "APPROVE",
            "expected_decision": "DEFER" if context == "absent" else "APPROVE",
        }
        for i in range(24)
        for context in ("informed", "absent", "stale")
    ]


def test_gate_passes_all_correct_controls(replay):
    result = replay.capability_gate(controls())
    assert result["passed"] is True
    assert result["by_context"]["absent"] == {"n": 24, "correct": 24}


def test_gate_requires_missing_evidence_deferral(replay):
    rows = controls()
    for row in [r for r in rows if r["context"] == "absent"][:5]:
        row["decision"] = "APPROVE"
    assert replay.capability_gate(rows)["passed"] is False


def test_gate_rejects_excess_invalids_even_when_accuracy_floors_pass(replay):
    rows = controls()
    rows[0]["decision"] = "INVALID"
    rows[1]["decision"] = "INVALID"
    result = replay.capability_gate(rows)
    assert result["by_context"]["informed"]["correct"] == 23
    assert result["by_context"]["absent"]["correct"] == 23
    assert result["passed"] is False
