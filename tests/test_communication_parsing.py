"""Secondary format tolerance must preserve ambiguity and truncation failures."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "communication_parsing",
    Path(__file__).resolve().parents[1] / "scripts/analyse_communication_parsing.py",
)
parser = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parser)


@pytest.mark.parametrize(
    "text,finish,expected",
    [
        ("Reasoning.\nFINAL: APPROVE.", "eos", "APPROVE"),
        ("FINAL: DEFER!", "eos", "DEFER"),
        ("FINAL: REJECT", "length", "INVALID"),
        ("FINAL: APPROVE.\nFINAL: REJECT", "eos", "INVALID"),
        ("FINAL: APPROVE because it passes", "eos", "INVALID"),
    ],
)
def test_only_unambiguous_uncapped_final_is_recovered(text, finish, expected):
    assert parser.punctuation_tolerant_final({"text": text, "finish_reason": finish}) == expected
