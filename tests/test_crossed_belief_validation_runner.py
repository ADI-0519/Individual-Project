"""Synthetic CPU checks of provenance, same-forward measurement and terminal attempts."""

import copy
import hashlib
import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from latent_probes.backends import GenerationResult
from latent_probes.crossed_belief_bank import build_bank
from latent_probes.crossed_belief_validation import (
    HEAD,
    answer_record,
    forward_measurement,
    summarize_logits,
)
from latent_probes.storage import atomic_json, canonical_hash, file_sha256, read_json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "crossed_validation_runner", ROOT / "scripts/run_crossed_belief_validation.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class Tokenizer:
    chat_template = "synthetic-crossed-assay-no-language-model"
    eos_token_id = 2
    pad_token_id = None
    all_special_ids = [2, 7]
    ambiguous = False

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        digest = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).digest()
        return [1, *digest[:8], 100]

    def encode(self, word, *, add_special_tokens):
        assert not add_special_tokens
        return [3, 4] if self.ambiguous else [3 if word == "YES" else 4]

    def decode(self, ids, *, skip_special_tokens):
        return "".join(
            {2: "" if skip_special_tokens else "<eos>", 3: "YES", 4: "NO"}[i] for i in ids
        )


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
        assert self.callback is None
        self.callback = callback
        return SimpleNamespace(remove=lambda: setattr(self, "callback", None))


class Model:
    calls = 0
    reference_mismatch = False
    fail_on_call = None

    def __init__(self):
        self.training = True
        self.config = SimpleNamespace(max_position_embeddings=4096)
        self.layer = Layer()

    def eval(self):
        self.training = False

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, input_ids, attention_mask, logits_to_keep, use_cache, return_dict):
        assert logits_to_keep == 1 and use_cache is False and return_dict is True
        assert np.array_equal(attention_mask, np.ones_like(input_ids))
        type(self).calls += 1
        if type(self).calls == self.fail_on_call:
            raise RuntimeError("synthetic terminal failure")
        feature = np.asarray([[[1.0, 0.0]]], np.float32).view(Tensor)
        if self.reference_mismatch:
            feature[0, 0, 0] = 2
        self.layer.callback(None, None, feature)
        logits = np.asarray([[[0, 0, 0, 5, 1, 0, 0, 0]]], np.float32).view(Tensor)
        return SimpleNamespace(logits=logits)


@pytest.fixture
def assay(tmp_path, monkeypatch):
    Model.calls, Model.reference_mismatch, Model.fail_on_call = 0, False, None
    source, known, fit = (tmp_path / name for name in ("source", "known", "fit"))
    for path in (source, known, fit):
        path.mkdir()
        atomic_json(path / "fixture.json", {"explicit_synthetic_fixture": True})
    bank_path, config_path, output = (
        tmp_path / name for name in ("bank.json", "config.json", "run")
    )
    atomic_json(bank_path, build_bank())
    atomic_json(config_path, {"synthetic": True})
    tokenizer = Tokenizer()
    revision = "0e9e39f249a16976918f6564b8830bc894c89659"
    model_settings = {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "revision": revision,
        "dtype": "bfloat16",
        "quantization": "4bit",
        "capture_layers": [16],
        "local_files_only": True,
        "max_input_tokens": 2048,
    }
    backend_metadata = {
        "backend": "synthetic_cpu_fixture",
        "is_test_fixture": True,
        "model_id": model_settings["model_id"],
        "resolved_revision": revision,
        "model_type": "llama",
        "dtype": "bfloat16",
        "quantization": "4bit",
        "decoder_blocks": 32,
        "hidden_size": 2,
        "capture_layers": [16],
        "chat_template_sha256": canonical_hash(tokenizer.chat_template),
        "generation_defaults": {"eos_token_id": [2, 8]},
        "special_token_ids": {"eos": 2, "pad": None},
        "attention_implementation": "sdpa",
    }
    metadata = {
        "feature_widths": {"layer_16": 2},
        "heads": {
            HEAD: {"layer": "layer_16", "activation_transform": "fixed", "parameter_prefix": HEAD}
        },
    }
    parameters = {
        "fixed__mean": np.zeros(2),
        "fixed__scale": np.ones(2),
        HEAD + "__coef": np.asarray([[2.0, 0.0]]),
        HEAD + "__intercept": np.asarray([0.0]),
    }
    monkeypatch.setattr(
        runner, "_load_fit", lambda p: (metadata, parameters, {}, runner._snapshot(p))
    )
    reference_messages = [{"role": "user", "content": "synthetic saved reference"}]
    vector = np.asarray([1.0, 0.0], np.float32)
    reference = {
        "example_id": "fixture_reference",
        "messages": reference_messages,
        "messages_sha256": canonical_hash(reference_messages),
        "prompt_token_ids": runner.token_ids(tokenizer, reference_messages),
        "feature": vector.tolist(),
        "feature_values_sha256": canonical_hash(vector.tolist()),
        "probe": runner.score_vector(vector, metadata, parameters, {}),
    }
    monkeypatch.setattr(runner, "_reference", lambda *args: copy.deepcopy(reference))
    old = {
        "known_run": str(known),
        "fit_dir": str(fit),
        "config_path": str(config_path),
        "known_artifact_sha256": runner._snapshot(known),
        "fitted_artifact_sha256": runner._snapshot(fit),
        "config_sha256": file_sha256(config_path),
        "primary_head": HEAD,
        "model_settings": model_settings,
    }
    atomic_json(source / "protocol.json", old)
    atomic_json(
        source / "manifest.json",
        {
            "status": "neutral_failed_stopped",
            "n_completed_rows": 32,
            "backend": backend_metadata,
            "protocol_sha256": file_sha256(source / "protocol.json"),
            "output_sha256": runner._snapshot(source),
        },
    )
    monkeypatch.setattr(runner, "native_output", lambda p: Path(p).resolve())
    monkeypatch.setattr(
        runner, "git_binding", lambda: {"commit": "fixture", "status_porcelain": ""}
    )
    monkeypatch.setattr(runner, "POLICY_FILES", ("scripts/run_crossed_belief_validation.py",))
    model_config = SimpleNamespace(
        _commit_hash=revision, model_type="llama", max_position_embeddings=4096
    )

    class Backend:
        created = 0
        generations = 0
        closed = False
        wrong_generation_prefix = False
        wrong_first_token = False

        def __init__(self, settings):
            type(self).created += 1
            self.settings = settings
            self.metadata = copy.deepcopy(backend_metadata)
            self.tokenizer = tokenizer
            self.model = Model()
            self.base = SimpleNamespace(layers=[self.model.layer] * 32)
            self.input_device = "synthetic_cpu"
            self.torch = SimpleNamespace(
                tensor=lambda x, **kw: np.asarray(x).view(Tensor),
                ones_like=lambda x: np.ones_like(x),
                inference_mode=nullcontext,
            )

        def generate(self, messages, settings, seed, *, capture):
            type(self).generations += 1
            assert not settings.do_sample and settings.max_new_tokens == 12 and not capture
            ids = runner.token_ids(self.tokenizer, messages)
            if self.wrong_generation_prefix:
                ids = ids + [0]
            return GenerationResult(
                text="NO" if self.wrong_first_token else "YES",
                prompt_token_ids=ids,
                generated_token_ids=[4 if self.wrong_first_token else 3, 2],
                finish_reason="eos",
            )

        def close(self):
            type(self).closed = True

    def prepare():
        return runner.prepare(
            bank_path,
            source,
            output,
            tokenizer=tokenizer,
            model_config=model_config,
            forward=Model.forward,
        )

    return SimpleNamespace(
        output=output,
        source=source,
        source_sha256=runner._snapshot(source),
        bank=bank_path,
        config=config_path,
        tokenizer=tokenizer,
        backend=Backend,
        prepare=prepare,
    )


def test_preflight_has_zero_forwards_deduplicates_actual_delivery_and_freezes_inputs(assay):
    protocol = assay.prepare()
    assert Model.calls == assay.backend.created == 0
    assert len(protocol["planned_rows"]) == 256
    assert len({r["sender_input_id"] for r in protocol["planned_rows"]}) == 128
    assert len({r["messages_sha256"] for r in protocol["planned_rows"]}) == 256
    assert protocol["n_queried_generations"] == 128
    assert all("actual_delivery" not in r for r in protocol["planned_rows"])
    assert protocol["preflight"]["model_forwards"] == 0
    assert protocol["answer_token_ids"] == {"YES": 3, "NO": 4}
    assert protocol["generation"]["eos_token_ids"] == [2, 8]
    with pytest.raises(ValueError, match="no overwrite"):
        assay.prepare()


def test_one_forward_binds_vector_logits_and_only_queried_answers(assay):
    protocol = assay.prepare()
    manifest = runner.execute(assay.output, backend_factory=assay.backend, progress=None)
    assert manifest["status"] == "complete_fresh_development_validation"
    assert manifest["n_completed_rows"] == manifest["model_measurement_forwards"] == 256
    assert manifest["generated_answers"] == assay.backend.generations == 128
    assert Model.calls == 257  # Plus one saved-source reference, no fixture layer-index forward.
    assert assay.backend.closed
    for planned in protocol["planned_rows"][:2]:
        row = read_json(assay.output / "row_outputs" / (planned["id"] + ".json"))
        with np.load(assay.output / row["capture"]["feature_path"], allow_pickle=False) as arrays:
            assert np.array_equal(arrays["layer_16"], [1, 0])
            assert (
                summarize_logits(arrays["next_token_logits"], protocol["answer_token_ids"])
                == row["output_logits"]
            )
        assert row["capture"]["same_forward_as_logits"]
        assert row["capture"]["generated_answer_tokens_supplied"] == 0
        if planned["condition"] == "story":
            assert row["generation"] is row["answer"] is None
        else:
            assert row["answer"]["prediction"] == 1
    assert runner._snapshot(assay.source) == assay.source_sha256
    with pytest.raises(ValueError, match="No overwrite"):
        runner.execute(assay.output, backend_factory=assay.backend)
    assert Model.calls == 257


def test_bad_reference_stops_before_any_new_input_or_generation(assay):
    assay.prepare()
    Model.reference_mismatch = True
    with pytest.raises(ValueError, match="full-vector/token/score"):
        runner.execute(assay.output, backend_factory=assay.backend, progress=None)
    assert Model.calls == 1 and assay.backend.generations == 0 and assay.backend.closed
    assert not (assay.output / "row_outputs").exists()
    assert read_json(assay.output / "manifest.json")["status"] == "failed_incomplete_no_retry"
    assert read_json(assay.output / "reference_check.json")["feature_exact"] is False


@pytest.mark.parametrize("problem", ["wrong_generation_prefix", "wrong_first_token"])
def test_generation_binding_failure_preserves_answer_and_refuses_retry(assay, problem):
    protocol = assay.prepare()
    setattr(assay.backend, problem, True)
    with pytest.raises(ValueError, match="prefix differs|first token differs"):
        runner.execute(assay.output, backend_factory=assay.backend, progress=None)
    queried = protocol["planned_rows"][1]
    row = read_json(assay.output / "row_outputs" / (queried["id"] + ".json"))
    assert row["generation"] is not None and row["answer"] is None
    assert read_json(assay.output / "manifest.json")["status"] == "failed_incomplete_no_retry"
    with pytest.raises(ValueError, match="No overwrite"):
        runner.execute(assay.output, backend_factory=assay.backend, progress=None)
    assert assay.backend.generations == 1


def test_bank_mutation_is_rejected_before_backend_creation(assay):
    assay.prepare()
    assay.bank.write_text(assay.bank.read_text() + "\n")
    with pytest.raises(ValueError, match="Frozen bank changed"):
        runner.execute(assay.output, backend_factory=assay.backend, progress=None)
    assert Model.calls == assay.backend.created == 0


def test_terminal_forward_failure_keeps_journal_and_does_not_generate(assay):
    protocol = assay.prepare()
    Model.fail_on_call = 2  # First new input; saved reference already checked.
    with pytest.raises(RuntimeError, match="terminal failure"):
        runner.execute(assay.output, backend_factory=assay.backend, progress=None)
    row = read_json(assay.output / "row_outputs" / (protocol["planned_rows"][0]["id"] + ".json"))
    assert row["status"] == "started" and assay.backend.generations == 0
    assert assay.backend.closed
    assert read_json(assay.output / "manifest.json")["status"] == "failed_incomplete_no_retry"


def test_ambiguous_tokens_stop_in_zero_model_preflight(assay):
    assay.tokenizer.ambiguous = True
    with pytest.raises(ValueError, match="single-token"):
        assay.prepare()
    assert not assay.output.exists() and Model.calls == assay.backend.created == 0


@pytest.mark.parametrize(
    "values", [np.array([1, 2]), np.array([[1, 2]], np.float32), np.array([np.nan, 0], np.float32)]
)
def test_logits_require_finite_complete_float32_vector(values):
    with pytest.raises(ValueError, match="float32 vocabulary"):
        summarize_logits(values, {"YES": 0, "NO": 1})


def test_logit_summary_is_stable_under_extreme_values_and_preserves_ties():
    values = np.asarray([1000, -1000, 1000], np.float32)
    result = summarize_logits(values, {"YES": 0, "NO": 1})
    assert result["argmax_token_id"] == 0
    assert result["p_yes_given_yes_no"] == 1
    assert result["full_vocab_logsumexp"] == pytest.approx(1000 + np.log(2))
    assert result["yes_logprob"] - result["no_logprob"] == 2000


@pytest.mark.parametrize(
    ("tokens", "text", "finish", "prediction", "capped", "unexpected"),
    [
        ([3, 2], "YES", "eos", 1, False, False),
        ([3, 8], "YES", "eos", 1, False, False),
        ([3, 2], "YES extra", "eos", None, False, False),
        ([3, 3], "YES", "length", None, True, False),
        ([3, 7, 2], "YES", "eos", None, False, True),
        ([3, 8, 2], "YES", "eos", None, False, True),
    ],
)
def test_strict_answer_parser_eos_and_invalids(
    tokens, text, finish, prediction, capped, unexpected
):
    result = answer_record(
        {
            "prompt_token_ids": [1],
            "generated_token_ids": tokens,
            "text": text,
            "finish_reason": finish,
        },
        [1],
        {"argmax_token_id": 3},
        max_new_tokens=12,
        eos_ids=[2, 8],
        special_ids=[2, 7],
    )
    assert result["prediction"] == prediction
    assert result["capped"] == capped and result["unexpected_special"] == unexpected


def test_capture_hook_is_removed_even_when_forward_raises(assay):
    backend = assay.backend(
        runner.ModelSettings(**read_json(assay.source / "protocol.json")["model_settings"])
    )
    Model.fail_on_call = 1
    with pytest.raises(RuntimeError):
        forward_measurement(backend, [1, 2], {"YES": 3, "NO": 4})
    assert backend.model.layer.callback is None
