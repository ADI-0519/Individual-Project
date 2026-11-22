"""Corruption fixtures for the independent before/after integration check."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_main_study_integration.py"
SPEC = importlib.util.spec_from_file_location("integration_auditor", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.fixture
def analyses():
    before = {"primary": {"score": 0.125}, "prose_monitor": {"status": "not_provided"}}
    after = {"primary": {"score": 0.125}, "prose_monitor": {"status": "complete"}}
    return before, after


def test_only_monitor_may_change(analyses):
    assert audit.compare_sections(*analyses) == ["primary"]


@pytest.mark.parametrize("corruption", ["score", "field", "incomplete_monitor"])
def test_changed_scientific_result_or_missing_monitor_is_rejected(analyses, corruption):
    before, after = copy.deepcopy(analyses)
    if corruption == "score":
        after["primary"]["score"] += 0.125
    elif corruption == "field":
        del after["primary"]
    else:
        after["prose_monitor"]["status"] = "running"
    with pytest.raises(ValueError):
        audit.compare_sections(before, after)
