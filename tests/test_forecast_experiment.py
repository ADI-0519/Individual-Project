"""Natural-pilot scheduling/lineage checks using explicit model-free fixtures."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from latent_probes.backends import GenerationResult
from latent_probes.config import ExperimentConfig, GenerationSettings, ModelSettings
from latent_probes.forecast_experiment import run_forecast_experiment

PANEL = Path(__file__).resolve().parents[1] / "data/reader_forecast/development_v1.json"


def fixture_config():
    return ExperimentConfig(
        model=ModelSettings(model_id="fixture", capture_layers=[8]),
        sender=GenerationSettings(do_sample=False),
        seeds=[42],
    )


class ScriptedForecastBackend:
    """Fixture labels/vectors exercise the pipeline, not its research hypothesis."""

    def __init__(self, settings, fail_after=None):
        self.metadata = {
            "model_id": "explicit-test-fixture",
            "is_test_fixture": True,
            "capture_layers": [8],
            "decoder_blocks": 32,
        }
        self.natural = 0
        self.readers = 0
        self.closed = False
        self.fail_after = fail_after

    def generate(self, messages, settings, seed, capture=False):
        assert capture is False
        self.readers += 1
        return GenerationResult(
            text=("APPROVE", "REJECT", "DEFER")[self.readers % 3],
            prompt_token_ids=[1],
            generated_token_ids=[2],
            finish_reason="eos",
        )

    def generated_token_activations(self, messages, settings, seed):
        if self.fail_after is not None and self.natural >= self.fail_after:
            raise RuntimeError("explicit fixture interruption")
        self.natural += 1
        matrix = np.asarray([[self.natural, 0], [self.natural, 1]], dtype=np.float32)
        return {"layer_8_tokens": matrix}, {
            "text": f"SCRIPTED FIXTURE report {self.natural}.",
            "finish_reason": "eos",
            "prompt_token_ids": [1],
            "generated_token_ids": [2, 3],
            "selected_token_positions": [1, 2],
            "selected_token_ids": [2, 3],
        }

    def close(self):
        self.closed = True


def test_neutral_weights_are_frozen_before_any_pressure_generation(tmp_path):
    config = fixture_config()
    backend = ScriptedForecastBackend(config.model)
    events = []

    def fit(features, labels, output, **options):
        assert backend.natural == 72
        assert options["allow_two_class_diagnostic"] is True
        with labels.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 72 and {r["objective"] for r in rows} == {"neutral"}
        with np.load(features, allow_pickle=False) as data:
            assert data["X"].shape == (72, 2)
            assert all(data["X"][:, 1] == 1)  # last response row, not mean or prompt
        output.mkdir()
        (output / "frozen-weights.json").write_text('{"fixture":true}', encoding="utf-8")
        events.append("fit")
        return {"status": "complete"}

    def score(features, labels, fitted, output):
        assert backend.natural == 84 and events == ["fit"]
        with labels.open(newline="", encoding="utf-8") as handle:
            test_rows = list(csv.DictReader(handle))
        assert len(test_rows) == 24 and {row["split"] for row in test_rows} == {"test"}
        assert (fitted / "frozen-weights.json").read_text(encoding="utf-8") == '{"fixture":true}'
        output.mkdir()
        events.append("frozen_score")

    result = run_forecast_experiment(
        config,
        PANEL,
        tmp_path / "run",
        backend_factory=lambda _: backend,
        fit_function=fit,
        score_function=score,
        progress=None,
    )
    assert result["status"] == "complete" and result["source_artifacts_unchanged"]
    assert result["is_test_fixture"] is True
    assert events == ["fit", "frozen_score"] and backend.closed
    assert backend.natural == 84 and backend.readers == 180
    freeze = json.loads((tmp_path / "run/frozen_readout.json").read_text(encoding="utf-8"))
    assert freeze["favorable_reports_completed"] == 0
    assert freeze["neutral_reports_completed"] == 72


def test_failure_preserves_raw_reports_and_closes_backend(tmp_path):
    config = fixture_config()
    backend = ScriptedForecastBackend(config.model, fail_after=2)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="explicit fixture interruption"):
        run_forecast_experiment(
            config, PANEL, output, backend_factory=lambda _: backend, progress=None
        )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed" and manifest["completed_reports"] == 2
    assert len(list((output / "reports").glob("*.json"))) == 2
    assert backend.closed and not (output / "neutral_readout").exists()


def test_producer_and_real_frozen_evaluator_contract(tmp_path):
    config = fixture_config()
    backend = ScriptedForecastBackend(config.model)
    output = tmp_path / "integrated"
    manifest = run_forecast_experiment(
        config, PANEL, output, backend_factory=lambda _: backend, progress=None
    )
    transfer = json.loads((output / "frozen_transfer/report.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete" and backend.closed
    assert transfer["status"] != "not_trainable"
    assert manifest["source_artifacts_unchanged"] is True


def test_reader_failure_preserves_already_generated_sender(tmp_path):
    config = fixture_config()
    backend = ScriptedForecastBackend(config.model)
    original_generate = backend.generate

    def fail_reader(messages, settings, seed, capture=False):
        if backend.natural:
            raise RuntimeError("fixture reader failure")
        return original_generate(messages, settings, seed, capture)

    backend.generate = fail_reader
    output = tmp_path / "reader_failure"
    with pytest.raises(RuntimeError, match="fixture reader failure"):
        run_forecast_experiment(
            config, PANEL, output, backend_factory=lambda _: backend, progress=None
        )
    sender = list((output / "senders").glob("*.json"))
    assert len(sender) == 1 and not list((output / "reports").glob("*.json"))
    raw = json.loads(sender[0].read_text(encoding="utf-8"))
    assert raw["generation"]["text"] == "SCRIPTED FIXTURE report 1."
    assert (output / raw["feature_file"]).is_file() and backend.closed
