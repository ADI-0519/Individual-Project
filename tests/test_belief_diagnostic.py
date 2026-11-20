"""Guard query-free captures, unavailable information and retained failed runs."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from latent_probes.backends import GenerationResult
from latent_probes.belief_panel import build_panel
from latent_probes.storage import canonical_hash, read_json

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "belief_diagnostic", ROOT / "scripts/run_belief_diagnostic.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrefixTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        assert "Question:" not in messages[-1]["content"]
        digest = canonical_hash(messages)
        return {"input_ids": np.asarray([[int(digest[i : i + 2], 16) for i in range(0, 12, 2)]])}


class FixtureBackend:
    def __init__(self, fail=False):
        self.metadata = {"is_test_fixture": True, "model_id": "explicit-fixture"}
        self.settings = SimpleNamespace(capture_layers=[8, 16, 24], max_input_tokens=4096)
        self.tokenizer = PrefixTokenizer()
        self.torch = SimpleNamespace(tensor=lambda values, **_kwargs: np.asarray(values))
        self.input_device = "cpu"
        self.generation_calls = 0
        self.fail = fail

    def _capture_token_rows(self, sequence, positions):
        assert positions == [sequence.shape[1] - 1]
        return {
            f"layer_{layer}_tokens": sequence[:, :3].astype(np.float32) + layer
            for layer in MODULE.LAYERS
        }

    def generate(self, messages, settings, seed, capture=False):
        assert "Question:" in messages[-1]["content"]
        assert settings.max_new_tokens == 6 and not settings.do_sample and seed == 42
        assert not capture
        self.generation_calls += 1
        if self.fail:
            raise RuntimeError("fixture generation failure")
        return GenerationResult(text="YES", finish_reason="eos")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("YES", 1),
        ("no.", 0),
        ("Yes!", 1),
        ("YES because", None),
        ("UNKNOWN", None),
        ("NO YES", None),
    ],
)
def test_strict_yes_no_parser(text, expected):
    assert MODULE.parse_yes_no(text) == expected


def test_unavailable_private_reader_question_is_rejected():
    with pytest.raises(ValueError, match="private reader"):
        MODULE.query_messages(build_panel()["cases"][0], "sender", "reader")


def test_fixture_pipeline_separates_captures_questions_and_gold(tmp_path, monkeypatch):
    panel, backend = build_panel(), FixtureBackend()
    original = canonical_hash(panel)
    received = {}

    def evaluator(rows, features, **kwargs):
        received.update(rows=rows, features=features, kwargs=kwargs)
        return {"is_test_fixture": True}

    monkeypatch.setattr(MODULE, "evaluate_belief_readouts", evaluator)
    output = tmp_path / "run"
    manifest = MODULE.run_belief_diagnostic(panel, output, backend, progress=None)
    assert manifest["status"] == "complete" and manifest["is_test_fixture"]
    assert manifest["n_capture_rows"] == 160 and manifest["n_behavior_rows"] == 400
    assert manifest["n_unique_captures"] < 160
    assert backend.generation_calls == manifest["n_unique_generations"] < 400
    assert canonical_hash(panel) == original
    assert received["kwargs"] == {"primary_layer": "layer_16"}
    assert all("Question:" not in row["messages_text"] for row in received["rows"])
    assert {matrix.shape for matrix in received["features"].values()} == {(160, 3)}
    captures = read_json(output / "captures.json")
    assert all(
        not row["perspective_question_supplied"] and row["generated_answer_tokens_supplied"] == 0
        for row in captures
    )
    assert len(list((output / "behavior").glob("*.json"))) == 400
    assert manifest["layer_index_check"]["status"] == "not_run_test_fixture"


def test_generation_failure_preserves_prefix_artifacts(tmp_path):
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="fixture generation"):
        MODULE.run_belief_diagnostic(
            build_panel(), output, FixtureBackend(fail=True), progress=None
        )
    assert len(list((output / "prefix_features").glob("*.npz"))) == 1
    assert len(read_json(output / "captures.json")) == 1
    assert read_json(output / "manifest.json")["status"] == "failed"
