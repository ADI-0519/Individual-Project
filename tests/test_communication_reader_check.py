"""The capability check changes policy while preserving the evidence payload."""

import importlib.util
from pathlib import Path

import pytest

from latent_probes.communication_panel import authored_panel, build_reader_messages

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def checker(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "reader_capability_check", ROOT / "scripts/check_communication_reader.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reasoned_policy_preserves_exact_user_evidence(checker):
    case = authored_panel()["cases"][0]
    original = build_reader_messages(case, "", "absent")
    new = checker.reasoned_messages(original)
    assert new[1] == original[1]
    assert original[0]["content"].endswith(checker.TAIL)
    assert new[0]["content"].endswith(checker.REASONED)
    assert case["decisive_fact"] not in new[1]["content"]


def test_reasoned_policy_refuses_unknown_source_prompt(checker):
    with pytest.raises(ValueError, match="Unexpected"):
        checker.reasoned_messages(
            [{"role": "system", "content": "different"}, {"role": "user", "content": "input"}]
        )
