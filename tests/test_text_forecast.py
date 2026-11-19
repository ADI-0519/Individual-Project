"""The elicited forecast sees original sender information, never outcome labels."""

import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "text_forecast", Path(__file__).resolve().parents[1] / "scripts/run_text_forecast.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_followup_ignores_outcome_metadata_and_preserves_original_prompt():
    record = {
        "messages": [{"role": "user", "content": "Only the source and attributed packet."}],
        "generation": {"text": "The observed quarterly costs were 90."},
        "decision": "CANARY_ACTUAL_ANSWER",
        "world_correct_decision": "CANARY_WORLD_GOLD",
    }
    original = copy.deepcopy(record)
    messages = module.build_followup(record)
    changed = copy.deepcopy(record)
    changed["decision"] = "DIFFERENT_OUTCOME"
    changed["world_correct_decision"] = "DIFFERENT_WORLD"
    assert messages == module.build_followup(changed)
    assert record == original
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert all("CANARY" not in m["content"] for m in messages)
    assert messages[1]["content"] == record["generation"]["text"]


def test_forecast_requires_a_real_saved_report():
    with pytest.raises(ValueError, match="nonempty report"):
        module.build_followup(
            {"messages": [{"role": "user", "content": "source"}], "generation": {"text": ""}}
        )
