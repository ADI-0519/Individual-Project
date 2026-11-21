"""Synthetic CPU evidence for the separate posthoc, no-generation diagnostic."""

import copy
import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from latent_probes.storage import atomic_json, canonical_hash, file_sha256, read_json

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "answer_logits_runner", ROOT / "scripts/run_future_goal_answer_logits.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class Tokenizer:
    chat_template = "explicit-synthetic-posthoc-template"
    eos_token_id = 2
    all_special_ids = [1, 2]
    ambiguous = False
    wrong_no = False

    def encode(self, word, *, add_special_tokens):
        assert add_special_tokens is False
        return [3, 4] if self.ambiguous else [3 if word == "YES" else 5 if self.wrong_no else 4]

    def decode(self, ids, *, skip_special_tokens):
        return {3: "YES", 4: "NO", 5: "NO"}[ids[0]]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        return [1, 10 + int(messages[0]["content"])]


class Tensor(np.ndarray):
    def detach(self):
        return self

    def float(self):
        return self.astype(np.float32).view(Tensor)

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


class Layer:
    callback = None

    def register_forward_hook(self, callback):
        self.callback = callback
        return SimpleNamespace(remove=lambda: setattr(self, "callback", None))


class Model:
    training = True
    calls = 0
    mismatch = None

    def __init__(self):
        self.layer = Layer()

    def eval(self):
        self.training = False

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, input_ids, attention_mask, logits_to_keep, use_cache, return_dict):
        assert logits_to_keep == 1 and use_cache is False and return_dict is True
        assert np.array_equal(attention_mask, np.ones_like(input_ids))
        index = int(input_ids[0, -1] - 10)
        type(self).calls += 1
        feature = np.asarray([[[index, 1.0]]], np.float32).view(Tensor)
        if index == self.mismatch:
            feature[0, 0, 0] += 1
        self.layer.callback(None, None, feature)
        logits = np.arange(8, dtype=np.float32).reshape(1, 1, 8).view(Tensor)
        logits[0, 0, 3], logits[0, 0, 4] = index / 10, 2.0
        return SimpleNamespace(logits=logits)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    Model.calls, Model.mismatch = 0, None
    source, fit, output = (tmp_path / name for name in ("source", "fit", "diagnostic"))
    source.mkdir()
    fit.mkdir()
    atomic_json(fit / "fixture.json", {"no_fit_performed": True})
    config = tmp_path / "config.json"
    atomic_json(config, {"model": "synthetic"})
    tokenizer = Tokenizer()
    metadata = {
        "backend": "explicit_cpu_fixture",
        "is_test_fixture": True,
        "model_type": "llama",
        "resolved_revision": "a" * 40,
        "chat_template_sha256": canonical_hash(tokenizer.chat_template),
        "special_token_ids": {"eos": 2},
    }
    settings = {
        "model_id": "fixture",
        "revision": "a" * 40,
        "device": "cpu",
        "dtype": "float32",
        "capture_layers": [16],
        "local_files_only": True,
    }
    rows = []
    for i in range(32):
        feature = Path("features") / f"{i}.npz"
        (source / feature.parent).mkdir(exist_ok=True)
        np.savez(source / feature, layer_16=np.asarray([i, 1.0], np.float32))
        messages, ids = [{"role": "user", "content": str(i)}], [1, i + 10]
        row = {
            "id": f"fixture_{i}",
            "scene_id": f"scene_{i // 2}",
            "channel": "fixture",
            "wording_group": "fixture",
            "mapping_id": i % 2,
            "condition": "neutral",
            "gold": i % 2,
            "messages": messages,
            "prompt_token_ids": ids,
            "prompt_token_ids_sha256": canonical_hash(ids),
            "probe": {"margin": 0.0, "probability": 0.5, "prediction": 0},
            "capture": {
                "feature_path": feature.as_posix(),
                "feature_sha256": file_sha256(source / feature),
                "prompt_token_ids_sha256": canonical_hash(ids),
                "layer": 16,
                "hf_hidden_state_index": 17,
                "generated_answer_tokens_supplied": 0,
                "selected_position": 1,
                "selected_token_id": ids[-1],
            },
        }
        rows.append(row)
        atomic_json(
            source / "row_outputs" / (row["id"] + ".json"),
            {
                **row,
                "status": "complete",
                "generation": {
                    "text": "NO",
                    "generated_token_ids": [4, 2],
                    "prompt_token_ids": ids,
                },
            },
        )
    old = {
        "model_settings": settings,
        "planned_rows": rows,
        "policy_sha256": {},
        "bank_path": str(
            ROOT / "data/belief_diagnostic/future_goal_content_controls_draft_v1.json"
        ),
        "bank_sha256": runner.BANK_SHA256,
        "config_path": str(config),
        "config_sha256": file_sha256(config),
        "fit_dir": str(fit),
        "fitted_artifact_sha256": runner.snapshot(fit),
    }
    atomic_json(source / "protocol.json", old)
    atomic_json(
        source / "manifest.json",
        {
            "status": "neutral_failed_stopped",
            "n_completed_rows": 32,
            "backend": metadata,
            "protocol_sha256": file_sha256(source / "protocol.json"),
            "output_sha256": runner.snapshot(source),
        },
    )
    monkeypatch.setattr(runner, "native", lambda p: Path(p).resolve())

    class Backend:
        created = 0
        closed = False

        def __init__(self, model_settings):
            type(self).created += 1
            self.metadata = copy.deepcopy(metadata)
            self.model = Model()
            self.base = SimpleNamespace(layers=[self.model.layer] * 32)
            self.tokenizer = tokenizer
            self.input_device = "cpu"
            self.torch = SimpleNamespace(
                tensor=lambda x, **kw: np.asarray(x).view(Tensor),
                ones_like=lambda x: np.ones_like(x),
                inference_mode=nullcontext,
            )

        def close(self):
            type(self).closed = True

    return SimpleNamespace(
        source=source,
        output=output,
        tokenizer=tokenizer,
        backend=Backend,
        config=config,
        source_hashes=runner.snapshot(source),
    )


def test_prepare_loads_no_weights_and_freezes_all_sources(fixture):
    protocol = runner.prepare(
        fixture.source, fixture.output, tokenizer=fixture.tokenizer, forward=Model.forward
    )
    assert fixture.backend.created == Model.calls == 0
    assert len(protocol["planned_rows"]) == 32 and protocol["tokens"] == {"YES": 3, "NO": 4}
    assert protocol["source_sha256"] == fixture.source_hashes
    assert protocol["generated_answers"] == 0 and "Posthoc" in protocol["scope"]


def test_exact_replay_uses_one_forward_per_row_and_never_generates(fixture):
    runner.prepare(
        fixture.source, fixture.output, tokenizer=fixture.tokenizer, forward=Model.forward
    )
    manifest = runner.execute(fixture.output, backend_factory=fixture.backend)
    assert manifest["status"] == "complete_posthoc_development"
    assert manifest["n_rows"] == Model.calls == 32 and manifest["generated_answers"] == 0
    assert fixture.backend.closed and runner.snapshot(fixture.source) == fixture.source_hashes
    row = read_json(fixture.output / "row_outputs/fixture_10.json")
    assert row["feature_exact"] and row["yes_logit"] == 1.0 and row["no_logit"] == 2.0
    assert row["yes_minus_no"] == -1.0
    assert row["p_yes_given_yes_no"] == pytest.approx(1 / (1 + np.exp(1)))
    assert row["yes_logprob"] - row["no_logprob"] == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="No overwrite"):
        runner.execute(fixture.output, backend_factory=fixture.backend)
    assert Model.calls == 32


def test_feature_mismatch_stops_without_accepting_logits_or_rerun(fixture):
    runner.prepare(
        fixture.source, fixture.output, tokenizer=fixture.tokenizer, forward=Model.forward
    )
    Model.mismatch = 3
    with pytest.raises(ValueError, match="feature differs"):
        runner.execute(fixture.output, backend_factory=fixture.backend)
    manifest = read_json(fixture.output / "manifest.json")
    assert manifest["status"] == "failed_incomplete_no_retry" and manifest["n_rows"] == 3
    assert Model.calls == 4 and fixture.backend.closed
    failed = read_json(fixture.output / "row_outputs/fixture_3.json")
    assert failed["status"] == "feature_mismatch" and not failed["feature_exact"]
    assert "yes_logit" not in failed
    with pytest.raises(ValueError, match="No overwrite"):
        runner.execute(fixture.output, backend_factory=fixture.backend)
    assert Model.calls == 4 and runner.snapshot(fixture.source) == fixture.source_hashes


@pytest.mark.parametrize("problem", ["ambiguous", "wrong_no", "forward"])
def test_token_and_forward_ambiguity_stops_before_weights(fixture, problem):
    forward = Model.forward
    if problem == "forward":

        def forward(input_ids):
            return None
    else:
        setattr(fixture.tokenizer, problem, True)
    with pytest.raises(ValueError, match="Ambiguous|Saved NO|logits_to_keep"):
        runner.prepare(fixture.source, fixture.output, tokenizer=fixture.tokenizer, forward=forward)
    assert not fixture.output.exists() and fixture.backend.created == Model.calls == 0


def test_source_change_refused_before_backend_creation(fixture):
    runner.prepare(
        fixture.source, fixture.output, tokenizer=fixture.tokenizer, forward=Model.forward
    )
    fixture.config.write_text(fixture.config.read_text() + "\n")
    with pytest.raises(ValueError, match="Frozen config changed"):
        runner.execute(fixture.output, backend_factory=fixture.backend)
    assert fixture.backend.created == Model.calls == 0
