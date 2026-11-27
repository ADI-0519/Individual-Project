"""Synthetic CPU checks for the dedicated, non-research single-forward profile."""

from __future__ import annotations

import copy
import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from latent_probes.storage import atomic_json, canonical_hash, file_sha256, read_json

SPEC = importlib.util.spec_from_file_location(
    "engineering_profile",
    Path(__file__).resolve().parents[1] / "scripts/profile_main_study_backend.py",
)
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


class Tokenizer:
    chat_template = "synthetic-engineering-fixture-only"
    eos_token_id = 2
    pad_token_id = None
    all_special_ids = [2]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert messages and tokenize and add_generation_prompt
        return [1, 6, 5, 7]

    def encode(self, word, *, add_special_tokens):
        assert not add_special_tokens
        return [{"YES": 3, "NO": 4}[word]]

    def decode(self, ids, *, skip_special_tokens):
        return "".join({3: "YES", 4: "NO"}[x] for x in ids)


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
    misaligned = False
    fail = False
    change_source = None

    def __init__(self, layers, layer):
        self.layers, self.selected = layers, layer
        self.training = True
        self.config = SimpleNamespace(vocab_size=8)

    def eval(self):
        self.training = False

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(
        self,
        input_ids,
        attention_mask,
        logits_to_keep,
        use_cache,
        return_dict,
        output_hidden_states=False,
    ):
        assert logits_to_keep == 1 and not use_cache and return_dict and output_hidden_states
        assert np.array_equal(attention_mask, np.ones_like(input_ids))
        type(self).calls += 1
        if type(self).fail:
            raise RuntimeError("synthetic forward failure preserved")
        hidden = np.asarray([[[1.0, 2.0]]], dtype=np.float32).view(Tensor)
        for layer in self.layers:
            if layer.callback:
                layer.callback(None, None, hidden)
        states = [hidden] * (len(self.layers) + 1)
        if type(self).misaligned:
            states[self.selected + 1] = (hidden + 1).view(Tensor)
        if type(self).change_source:
            type(self).change_source.write_text("changed source during fixture", encoding="utf-8")
        logits = np.asarray([[[0, 0, 0, 3, 1, 0, 0, 0]]], dtype=np.float32).view(Tensor)
        return SimpleNamespace(logits=logits, hidden_states=states)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    Model.calls, Model.misaligned, Model.fail, Model.change_source = 0, False, False, None
    bank = {
        "schema": "synthetic-fixture-not-main-study-bank",
        "groups": [{"sender": "Research Alpha", "reader": "Research Beta"}],
    }
    bank_path, excluded_path = tmp_path / "bank.json", tmp_path / "excluded.json"
    atomic_json(bank_path, bank)
    excluded = {
        "cases": [
            {
                "sender": "Legacy Gamma",
                "reader": "Legacy Delta",
                "messages": [{"role": "user", "content": "Synthetic excluded fixture"}],
            }
        ]
    }
    atomic_json(excluded_path, excluded)
    monkeypatch.setattr(profile, "load_bank", read_json)
    monkeypatch.setattr(
        profile,
        "iter_rows",
        lambda bank: iter([{"messages": [{"role": "user", "content": "Synthetic bank fixture"}]}]),
    )
    tokenizer = Tokenizer()

    class Backend:
        created = 0
        closed = 0

        def __init__(self, settings):
            type(self).created += 1
            self.settings, self.tokenizer = settings, tokenizer
            blocks = 32 if "Llama" in settings.model_id else 28
            self.base = SimpleNamespace(layers=[Layer() for _ in range(blocks)])
            self.model = Model(self.base.layers, settings.capture_layers[0])
            self.input_device = "synthetic-cpu-fixture"
            self.torch = SimpleNamespace(
                tensor=lambda x, **kwargs: np.asarray(x).view(Tensor),
                ones_like=np.ones_like,
                inference_mode=nullcontext,
            )
            self.metadata = {
                "is_test_fixture": True,
                "model_id": settings.model_id,
                "dtype": settings.dtype,
                "quantization": settings.quantization,
                "capture_layers": settings.capture_layers,
                "resolved_revision": settings.revision,
                "decoder_blocks": blocks,
                "hidden_size": 2,
            }

        def close(self):
            type(self).closed += 1

    def arguments(model_key="llama"):
        model = profile.MODELS[model_key]
        return {
            "bank_path": bank_path,
            "output": tmp_path / (model_key + "_profile"),
            "model_key": model_key,
            "excluded_path": excluded_path,
            "test_fixture": True,
            "backend_factory": Backend,
            "tokenizer": tokenizer,
            "model_config": SimpleNamespace(
                _commit_hash=model["revision"],
                model_type=model["model_type"],
                num_hidden_layers=32 if model_key == "llama" else 28,
                max_position_embeddings=4096,
                hidden_size=2,
                vocab_size=8,
            ),
            "forward": Model.forward,
        }

    return SimpleNamespace(
        arguments=arguments,
        backend=Backend,
        bank=bank,
        bank_path=bank_path,
        excluded=excluded,
        profile=read_json(profile.PROFILE_PATH),
    )


@pytest.mark.parametrize(("model_key", "expected_layer"), [("llama", 16), ("qwen", 14)])
def test_one_forward_exact_layer_and_full_vocabulary_preserved(prepared, model_key, expected_layer):
    args = prepared.arguments(model_key)
    result = profile.profile_backend(**args)
    output = args["output"]
    assert result["status"] == "complete_engineering_only"
    assert result["test_fixture"] is True and result["research_observation"] is False
    assert result["completed_forwards"] == Model.calls == prepared.backend.created == 1
    assert prepared.backend.closed == 1
    assert result["capture"]["layer"] == expected_layer
    assert result["capture"]["hf_hidden_state_index"] == expected_layer + 1
    assert result["capture"]["hook_alignment_checked"] is True
    assert result["capture"]["same_forward_as_full_vocabulary_logits"] is True
    assert result["capture"]["generated_answer_tokens"] == 0
    protocol = read_json(output / "protocol.json")
    assert protocol["main_bank_rows_passed_to_model"] == 0
    assert protocol["permitted_forwards"] == 1
    assert protocol["exclusion_checks"]["human_main_study_review_validated"] is False
    assert protocol["messages"] == prepared.profile["messages"]
    assert protocol["bank_sha256"] == profile.bank_digest(prepared.bank)
    assert result["arrays_sha256"] == file_sha256(output / "arrays.npz")
    for frozen in protocol["source_snapshots"].values():
        assert file_sha256(output / frozen["snapshot"]) == frozen["sha256"]
    with np.load(output / "raw_forward.npz", allow_pickle=False) as raw:
        assert raw["full_forward_logits"].shape == (1, 1, 8)
        assert raw["feature"].dtype == np.float32
    with np.load(output / "arrays.npz", allow_pickle=False) as data:
        assert data["feature"].tolist() == [1.0, 2.0]
        assert data["next_token_logits"].shape == (8,)
        assert canonical_hash(data["feature"].tolist()) == result["feature_values_sha256"]


def test_completed_directory_cannot_be_overwritten_or_retried(prepared):
    args = prepared.arguments()
    profile.profile_backend(**args)
    before = file_sha256(args["output"] / "manifest.json")
    with pytest.raises(ValueError, match="No overwrite or retry"):
        profile.profile_backend(**args)
    assert file_sha256(args["output"] / "manifest.json") == before
    assert Model.calls == 1


def test_alignment_failure_preserves_raw_forward_before_stopping(prepared):
    args = prepared.arguments()
    Model.misaligned = True
    with pytest.raises(ValueError, match="alignment differs"):
        profile.profile_backend(**args)
    result = read_json(args["output"] / "manifest.json")
    assert result["status"] == "failed_attempt_preserved"
    assert (args["output"] / "raw_forward.npz").is_file()
    assert not (args["output"] / "arrays.npz").exists()
    assert result["completed_forwards"] == 0
    assert Model.calls == prepared.backend.closed == 1


def test_failed_forward_is_journalled_and_never_implicitly_retried(prepared):
    args = prepared.arguments()
    Model.fail = True
    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        profile.profile_backend(**args)
    assert (args["output"] / "started.json").is_file()
    assert (args["output"] / "forward_started.json").is_file()
    assert (args["output"] / "failure.json").is_file()
    with pytest.raises(ValueError, match="No overwrite or retry"):
        profile.profile_backend(**args)
    assert Model.calls == 1


def test_changed_bound_source_preserves_tensors_and_fails(prepared):
    args = prepared.arguments()
    Model.change_source = prepared.bank_path
    with pytest.raises(ValueError, match="Changed input"):
        profile.profile_backend(**args)
    assert (args["output"] / "raw_forward.npz").is_file()
    assert (args["output"] / "arrays.npz").is_file()
    assert read_json(args["output"] / "manifest.json")["status"] == "failed_attempt_preserved"
    assert Model.calls == 1


def test_profile_rejects_first_name_reuse(prepared):
    bank = copy.deepcopy(prepared.bank)
    bank["groups"][0]["sender"] = "Hadley Otherwise"
    with pytest.raises(ValueError, match="actor component reuses"):
        profile.validate_profile(prepared.profile, bank, prepared.excluded)


def test_profile_rejects_main_prefix_reuse(prepared, monkeypatch):
    monkeypatch.setattr(profile, "iter_rows", lambda bank: iter([prepared.profile]))
    with pytest.raises(ValueError, match="duplicates a main-study prefix"):
        profile.validate_profile(prepared.profile, prepared.bank, prepared.excluded)


def test_profile_rejects_scientific_labels_and_larger_budget(prepared):
    draft = copy.deepcopy(prepared.profile)
    draft["gold"] = 1
    with pytest.raises(ValueError, match="No fit labels"):
        profile.validate_profile(draft, prepared.bank, prepared.excluded)
    draft.pop("gold")
    draft["permitted_forwards_per_model"] = 2
    with pytest.raises(ValueError, match="single-forward"):
        profile.validate_profile(draft, prepared.bank, prepared.excluded)


def test_fixture_cannot_use_real_backend(prepared):
    args = prepared.arguments()
    args["backend_factory"] = profile.HFBackend
    with pytest.raises(ValueError, match="cannot load real"):
        profile.profile_backend(**args)
    assert Model.calls == 0


def test_real_profile_cannot_use_injected_fixture_inputs(prepared):
    args = prepared.arguments()
    args["test_fixture"] = False
    with pytest.raises(ValueError, match="pinned cached tokenizer"):
        profile.profile_backend(**args)
    assert Model.calls == 0
