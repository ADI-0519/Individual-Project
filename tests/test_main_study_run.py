"""CPU fixtures exercise capture safeguards; they are never scientific observations."""

from __future__ import annotations

import copy
import hashlib
from contextlib import nullcontext
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from latent_probes import main_study_bank as bank_module
from latent_probes import main_study_run as run
from latent_probes.main_study_design import BeliefCase
from latent_probes.main_study_fit import load_feature_cache, save_fit_bundle
from latent_probes.storage import atomic_json, canonical_hash, file_sha256, read_json


class Tokenizer:
    chat_template = "explicit-synthetic-engineering-fixture"
    eos_token_id = 2
    pad_token_id = None
    all_special_ids = [2]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        return [1, *hashlib.sha256(str(messages).encode()).digest()[:8], 7]

    def encode(self, word, *, add_special_tokens):
        assert not add_special_tokens
        return [3 if word == "YES" else 4]

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
    fail_on_call = None
    misaligned = False

    def __init__(self, layers):
        self.layers = layers
        self.training = True
        self.config = SimpleNamespace(vocab_size=8, max_position_embeddings=4096)

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
        assert logits_to_keep == 1 and use_cache is False and return_dict
        assert np.array_equal(attention_mask, np.ones_like(input_ids))
        type(self).calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("preserved synthetic failure")
        hidden = np.asarray([[[float(input_ids[0, 1]), 1.0]]], np.float32).view(Tensor)
        for layer in self.layers:
            if layer.callback:
                layer.callback(None, None, hidden)
        values = np.asarray([[[0, 0, 0, 3, 1, 0, 0, 0]]], np.float32).view(Tensor)
        states = [hidden] * (len(self.layers) + 1)
        if self.misaligned:
            states[17] = (hidden + 1).view(Tensor)
        return SimpleNamespace(logits=values, hidden_states=states)


@pytest.fixture
def capture(tmp_path, monkeypatch):
    Model.calls, Model.fail_on_call, Model.misaligned = 0, None, False
    bank = {"schema": "explicit-engineering-fixture", "test_fixture": True}
    bank_path = tmp_path / "bank.json"
    atomic_json(bank_path, bank)
    monkeypatch.setattr(bank_module, "load_bank", lambda path: read_json(path))
    ledger = tmp_path / "ledger.json"
    atomic_json(ledger, {"schema": "synthetic-review-stub-not-human-validation"})
    review_calls = []

    def verify(bank, path):
        review_calls.append(str(path))
        assert read_json(path)["schema"] == "synthetic-review-stub-not-human-validation"
        return {"test_fixture": True, "review_gate_stub": "only inside pytest"}

    monkeypatch.setattr(bank_module, "verify_review_ledger", verify)
    tokenizer = Tokenizer()
    config = SimpleNamespace(
        _commit_hash=run.MODELS["llama"]["revision"],
        model_type="llama",
        num_hidden_layers=32,
        max_position_embeddings=4096,
        hidden_size=2,
        vocab_size=8,
    )
    rows = []
    for i, split in enumerate(("train", "train", "development", "test")):
        case = BeliefCase(f"fixture_group_{i}", "message", i % 2, 1 - i % 2, 1, i % 2, 0)
        rows.append(
            {
                "case_id": case.case_id,
                "group_id": case.group_id,
                "mechanism": case.mechanism,
                "split": split,
                "state": asdict(case),
                "gold": case.gold,
                "messages": [{"role": "user", "content": f"Synthetic fixture {i}"}],
            }
        )
    output = tmp_path / "capture"
    protocol = run.prepare(
        bank_path,
        output,
        tokenizer=tokenizer,
        model_config=config,
        forward=Model.forward,
        test_fixture=True,
        row_builder=lambda bank: copy.deepcopy(rows),
    )
    baseline = tmp_path / "baseline.json"
    atomic_json(baseline, {"schema": "explicit-fixture-baseline"})
    source = tmp_path / "bound_source.txt"
    source.write_text("explicit fixture", encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    atomic_json(
        freeze,
        {
            "schema": "main-study-execution-freeze-v1",
            "status": "frozen",
            "bank_sha256": protocol["bank_sha256"],
            "source_sha256": {**protocol["policy_sha256"], str(source): file_sha256(source)},
            "baseline_protocol_path": str(baseline),
            "baseline_protocol_sha256": file_sha256(baseline),
        },
    )

    class Backend:
        created = 0
        closed = 0

        def __init__(self, settings):
            type(self).created += 1
            self.settings, self.tokenizer = settings, tokenizer
            self.base = SimpleNamespace(layers=[Layer() for _ in range(32)])
            self.model = Model(self.base.layers)
            self.input_device = "cpu-fixture"
            self.torch = SimpleNamespace(
                tensor=lambda x, **kw: np.asarray(x).view(Tensor),
                ones_like=lambda x: np.ones_like(x),
                inference_mode=nullcontext,
            )
            self.metadata = {
                "is_test_fixture": True,
                "model_id": settings.model_id,
                "resolved_revision": settings.revision,
                "dtype": settings.dtype,
                "quantization": settings.quantization,
                "capture_layers": [16],
                "hidden_size": 2,
                "decoder_blocks": 32,
            }

        def close(self):
            type(self).closed += 1

    return SimpleNamespace(
        output=output,
        protocol=protocol,
        ledger=ledger,
        freeze=freeze,
        backend=Backend,
        baseline=baseline,
        source=source,
        bank=bank,
        bank_path=bank_path,
        rows=rows,
        review_calls=review_calls,
    )


def execute(capture, **kwargs):
    args = {
        "partition": "train_dev",
        "ledger_path": capture.ledger,
        "freeze_path": capture.freeze,
        "backend_factory": capture.backend,
        "progress": None,
    }
    args.update(kwargs)
    return run.execute(capture.output, **args)


def journal(capture, index):
    return capture.output / "row_journal" / capture.rows[index]["case_id"]


def test_prepare_loads_no_weights_and_binds_all_prefixes(capture):
    assert capture.backend.created == Model.calls == 0
    assert capture.protocol["preflight"]["model_weights_loaded"] is False
    assert len(capture.protocol["planned_rows"]) == 4
    assert capture.protocol["preflight"]["human_review_validated"] is False
    assert capture.protocol["capture_layer"] == 16


def test_same_forward_cache_is_complete_and_separates_test(capture):
    manifest = execute(capture)
    assert manifest["status"] == "complete"
    assert manifest["completed_rows"] == Model.calls == 3
    assert capture.backend.closed == 1
    rows = read_json(capture.output / "rows.json")
    assert rows["test_fixture"] is True
    assert rows["schema"] == "main-study-capture-v1"
    assert {row["split"] for row in rows["rows"]} == {"train", "development"}
    assert all(row["output_yes_no_logitdiff"] == 2 for row in rows["rows"])
    with np.load(capture.output / "features.npz", allow_pickle=False) as features:
        assert features["features"].shape == (3, 2)
        assert features["features"].dtype == np.float32
        assert features["row_ids"].tolist() == [r["case_id"] for r in capture.rows[:3]]
    fitted_cache = load_feature_cache(capture.output)
    assert fitted_cache.test_fixture is True
    assert fitted_cache.features.shape == (3, 2)
    assert fitted_cache.provenance["bank_sha256"] == capture.protocol["bank_sha256"]
    for row in rows["rows"]:
        journal = capture.output / "row_journal" / row["row_id"]
        assert (journal / "started.json").is_file()
        assert (journal / "complete.json").is_file()
        with np.load(journal / "arrays.npz", allow_pickle=False) as data:
            assert data["next_token_logits"].shape == (8,)
        assert row["capture"]["same_forward_as_logits"] is True
        assert row["capture"]["answer_tokens_supplied"] == 0


@pytest.mark.parametrize("target", ["bank_path", "baseline", "source"])
def test_changed_frozen_inputs_rejected_before_backend(capture, target):
    path = getattr(capture, target)
    path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError):
        execute(capture)
    assert capture.backend.created == 0


@pytest.mark.parametrize("missing", ["ledger_path", "freeze_path"])
def test_missing_review_or_freeze_rejected_before_weights(capture, missing):
    with pytest.raises(ValueError, match="human-review ledger"):
        execute(capture, **{missing: None})
    assert capture.backend.created == 0


def test_draft_freeze_rejected_before_weights(capture):
    freeze = read_json(capture.freeze)
    freeze["status"] = "draft"
    atomic_json(capture.freeze, freeze)
    with pytest.raises(ValueError, match="not frozen"):
        execute(capture)
    assert capture.backend.created == 0


def test_review_failure_rejected_before_weights(capture, monkeypatch):
    def reject(*args):
        raise ValueError("Missing actual human review")

    monkeypatch.setattr(bank_module, "verify_review_ledger", reject)
    with pytest.raises(ValueError, match="actual human review"):
        execute(capture)
    assert capture.backend.created == 0


def test_test_partition_requires_fitted_bundle_before_weights(capture):
    with pytest.raises(ValueError, match="frozen fitted head"):
        execute(capture, partition="test")
    assert capture.backend.created == 0


def test_test_partition_requires_fit_bytes_in_freeze(capture, tmp_path):
    fit = tmp_path / "fit"
    save_fit_bundle(
        fit,
        {
            "schema": "main-study-fit-v1",
            "test_fixture": True,
            "capture_provenance": {
                k: capture.protocol[k] for k in ("bank_sha256", "model_settings", "capture_layer")
            },
        },
        {"fixture_only_array": np.asarray([1.0, 2.0], np.float64)},
    )
    with pytest.raises(ValueError, match="fitted artifacts bound"):
        execute(capture, partition="test", fit_dir=fit)
    assert capture.backend.created == 0
    freeze = read_json(capture.freeze)
    freeze["fitted_artifact_sha256"] = {
        str(fit / name): file_sha256(fit / name) for name in ("settings.json", "heads.npz")
    }
    atomic_json(capture.freeze, freeze)
    result = execute(capture, partition="test", fit_dir=fit)
    assert result["completed_rows"] == Model.calls == 1
    assert read_json(capture.output / "rows.json")["rows"][0]["split"] == "test"


def test_failure_preserved_and_unknown_row_forbids_resume(capture):
    Model.fail_on_call = 2
    with pytest.raises(RuntimeError, match="preserved synthetic"):
        execute(capture)
    directory = journal(capture, 1)
    assert read_json(directory / "failure.json")["error_type"] == "RuntimeError"
    assert (directory / "started.json").is_file()
    assert not (capture.output / "features.npz").exists()
    calls = Model.calls
    with pytest.raises(ValueError, match="blocks resumption"):
        execute(capture, resume=True)
    assert Model.calls == calls


def test_explicit_resume_after_interruption_between_complete_rows(capture):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("synthetic stop after durable completion")

    with pytest.raises(KeyboardInterrupt):
        execute(capture, progress=interrupt)
    assert Model.calls == 1
    with pytest.raises(ValueError, match="explicit resume"):
        execute(capture)
    result = execute(capture, resume=True)
    assert result["completed_rows"] == 3 and Model.calls == 3
    assert len(list((capture.output / "attempts").iterdir())) == 2


def test_saved_completed_row_tampering_blocks_resume(capture):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute(capture, progress=interrupt)
    arrays = journal(capture, 0) / "arrays.npz"
    arrays.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="binding changed"):
        execute(capture, resume=True)
    assert Model.calls == 1


def test_hook_alignment_failure_preserves_observed_attempt(capture):
    Model.misaligned = True
    with pytest.raises(ValueError, match="alignment differs"):
        execute(capture)
    assert Model.calls == 1
    assert (journal(capture, 0) / "failure.json").is_file()
    assert (journal(capture, 0) / "raw_forward.npz").is_file()


def test_completed_capture_cannot_be_repeated(capture):
    execute(capture)
    with pytest.raises(ValueError, match="cannot be rerun"):
        execute(capture, resume=True)
    assert Model.calls == 3


def test_qwen_layer_selected_from_pinned_config_without_outputs(capture, tmp_path):
    config = SimpleNamespace(
        _commit_hash=run.MODELS["qwen"]["revision"],
        model_type="qwen2",
        num_hidden_layers=28,
        max_position_embeddings=4096,
        hidden_size=2,
        vocab_size=8,
    )
    protocol = run.prepare(
        capture.bank_path,
        tmp_path / "qwen",
        model_key="qwen",
        tokenizer=Tokenizer(),
        model_config=config,
        forward=Model.forward,
        test_fixture=True,
        row_builder=lambda bank: capture.rows,
    )
    assert protocol["capture_layer"] == 14
    assert Model.calls == 0


def test_fixture_protocol_cannot_load_real_backend(capture):
    with pytest.raises(ValueError, match="cannot load real model"):
        execute(capture, backend_factory=run.HFBackend)
    assert Model.calls == 0


def test_float32_full_vocabulary_summary_is_recomputable(capture):
    execute(capture)
    row = read_json(journal(capture, 0) / "complete.json")
    with np.load(journal(capture, 0) / "arrays.npz", allow_pickle=False) as data:
        assert (
            canonical_hash(data["next_token_logits"].tolist())
            == row["output_logits"]["logits_values_sha256"]
        )
        assert np.isclose(row["output_logits"]["p_yes_given_yes_no"], 1 / (1 + np.exp(-2)))
