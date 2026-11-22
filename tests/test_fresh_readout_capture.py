"""Synthetic exact-span/capture safeguards; no fixture is a scientific observation."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from latent_probes import fresh_readout_capture as capture
from latent_probes.storage import atomic_json, canonical_hash, file_sha256, read_json


class Tokenizer:
    is_fast = True
    chat_template = "explicit engineering character tokenizer"
    eos_token_id = 2
    pad_token_id = None
    all_special_ids = [2]
    mismatch = False
    leading_word = None

    def __init__(self):
        self.backend_tokenizer = SimpleNamespace(to_str=lambda: "engineering-character-tokenizer")

    def get_vocab(self):
        return {chr(i): i + 10 for i in range(128)}

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping, truncation):
        assert not add_special_tokens and return_offsets_mapping and not truncation
        ids, offsets = [], []
        index = 0
        while index < len(text):
            if self.leading_word and text.startswith(" " + self.leading_word, index):
                end = index + 1 + len(self.leading_word)
            else:
                end = index + 1
            ids.append(ord(text[index]) + 10)
            offsets.append((index, end))
            index = end
        return {"input_ids": ids, "offset_mapping": offsets}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert add_generation_prompt
        rendered = "".join(f"<{m['role']}>\n{m['content']}\n" for m in messages) + "<assistant>\n"
        if not tokenize:
            return rendered
        result = self(
            rendered, add_special_tokens=False, return_offsets_mapping=True, truncation=False
        )["input_ids"]
        return result + [7] if self.mismatch else result

    def encode(self, word, *, add_special_tokens):
        assert not add_special_tokens
        return [{"YES": 3, "NO": 4}[word]]

    def decode(self, ids, *, skip_special_tokens):
        assert not skip_special_tokens
        return "".join({3: "YES", 4: "NO"}[i] for i in ids)


def row(index=0, partition="source_fit"):
    account = f"Account {index}: North is named twice. North. End of account."
    question = "\nQuestion: Is the item at North?"
    content = account + question
    start = len(account) + len("\nQuestion: Is the item at ")
    return {
        "record_id": f"engineering_fixture_{index}",
        "partition": partition,
        "group_id": f"fixture_group_{index}",
        "gold": index % 2,
        "messages": [
            {"role": "system", "content": "Use the stated account."},
            {"role": "user", "content": content},
        ],
        "user_content_char_spans": {
            "account": [0, len(account)],
            "query_location": [start, start + len("North")],
        },
    }


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
    hidden_state_calls = []

    def __init__(self, layers):
        self.layers, self.training = layers, True
        self.config = SimpleNamespace(vocab_size=256, max_position_embeddings=4096)

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
        type(self).hidden_state_calls.append(output_hidden_states)
        if self.fail_on_call == self.calls:
            raise RuntimeError("preserved synthetic forward failure")
        states = [None]
        for index, layer in enumerate(self.layers):
            hidden = np.asarray(
                [[[index, position, token] for position, token in enumerate(input_ids[0])]],
                dtype=np.float32,
            ).view(Tensor)
            if layer.callback:
                layer.callback(None, None, (hidden,))
            states.append(hidden)
        if self.misaligned:
            states[17] = states[17] + 1
        values = np.zeros((1, 1, self.config.vocab_size), dtype=np.float32).view(Tensor)
        values[0, 0, 3], values[0, 0, 4] = 3, 1
        return SimpleNamespace(
            logits=values, hidden_states=states if output_hidden_states else None
        )


@pytest.fixture
def prepared(tmp_path):
    Model.calls, Model.fail_on_call, Model.misaligned, Model.hidden_state_calls = 0, None, False, []
    tokenizer = Tokenizer()
    config = SimpleNamespace(
        _commit_hash=capture.MODELS["llama"]["revision"],
        model_type="llama",
        num_hidden_layers=32,
        max_position_embeddings=4096,
        hidden_size=3,
        vocab_size=256,
    )
    bank = tmp_path / "capture_inputs.json"
    rows = [row(i, part) for i, part in enumerate(capture.COUNTS)]
    atomic_json(bank, {"rows": rows, "engineering_row": row(99, "engineering")})
    output = tmp_path / "capture"
    protocol = capture.prepare(
        bank,
        output,
        tokenizer=tokenizer,
        model_config=config,
        forward=Model.forward,
        test_fixture=True,
    )
    review = tmp_path / "agent_review.json"
    atomic_json(review, {"research_status": "engineering_fixture", "reviewed": True})
    freeze = tmp_path / "bindings.json"
    atomic_json(
        freeze,
        {
            "schema": "fresh-readout-execution-bindings-v1",
            "status": "frozen",
            "protocol_sha256": file_sha256(output / "protocol.json"),
            "source_sha256": protocol["source_sha256"],
            "review_artifact_sha256": {str(review): file_sha256(review)},
        },
    )

    class Backend:
        created = 0
        closed = 0

        def __init__(self, settings):
            type(self).created += 1
            self.settings, self.tokenizer = settings, tokenizer
            self.base = SimpleNamespace(layers=[Layer() for _ in range(config.num_hidden_layers)])
            self.model = Model(self.base.layers)
            self.input_device = "cpu-fixture"
            self.torch = SimpleNamespace(
                tensor=lambda x, **kw: np.asarray(x).view(Tensor),
                ones_like=np.ones_like,
                inference_mode=nullcontext,
            )
            self.metadata = {
                "is_test_fixture": True,
                "model_id": settings.model_id,
                "resolved_revision": settings.revision,
                "dtype": settings.dtype,
                "quantization": settings.quantization,
                "capture_layers": settings.capture_layers,
                "hidden_size": config.hidden_size,
                "decoder_blocks": config.num_hidden_layers,
            }

        def close(self):
            type(self).closed += 1

    return SimpleNamespace(
        output=output,
        bank=bank,
        protocol=protocol,
        config=config,
        tokenizer=tokenizer,
        freeze=freeze,
        review=review,
        backend=Backend,
        rows=rows,
    )


def execute(prepared, **kwargs):
    return capture.execute(
        prepared.output,
        partition=kwargs.pop("partition", "source_fit"),
        bindings_path=kwargs.pop("bindings_path", prepared.freeze),
        backend_factory=prepared.backend,
        progress=kwargs.pop("progress", None),
        **kwargs,
    )


def test_preparation_has_no_weights_and_exact_fixed_panel(prepared):
    assert prepared.backend.created == Model.calls == 0
    assert prepared.protocol["preflight"]["model_weights_loaded"] is False
    assert prepared.protocol["sites"] == capture.panel_sites("llama")
    assert prepared.protocol["model_settings"]["capture_layers"] == [8, 16, 24, 30]
    assert prepared.protocol["original_site"] == "layer_16_prompt_last"


def test_nested_input_provenance_preserves_complete_tokenizer_identity(
    prepared, tmp_path, monkeypatch
):
    """Native-shaped source bindings must never replace the tokenizer contract."""
    source_root = tmp_path / "source_root"
    source_root.mkdir()
    monkeypatch.setattr(capture, "ROOT", source_root)
    monkeypatch.setattr(capture, "POLICY_FILES", ("policy.py",))
    (source_root / "policy.py").write_text(
        "# Explicit engineering policy fixture\n", encoding="utf-8"
    )
    inputs = []
    for name in ("source_bank.json", "authored_train.json", "authored_test.json"):
        path = source_root / name
        atomic_json(path, {"schema": "explicit-nested-provenance-fixture", "name": name})
        inputs.append({"path": name, "sha256": file_sha256(path)})
    payload = read_json(prepared.bank)
    payload.update(source_bank=inputs[0], authored_inputs=inputs[1:])
    nested_bank = tmp_path / "nested_bank.json"
    atomic_json(nested_bank, payload)
    protocol = capture.prepare(
        nested_bank,
        tmp_path / "nested_capture",
        tokenizer=prepared.tokenizer,
        model_config=prepared.config,
        forward=Model.forward,
        test_fixture=True,
    )
    assert protocol["tokenizer"] == capture.tokenizer_binding(prepared.tokenizer)
    assert "answer_token_ids" in protocol["tokenizer"] and "backend_sha256" in protocol["tokenizer"]
    for binding in inputs:
        assert protocol["source_sha256"][str(source_root / binding["path"])] == binding["sha256"]
    # Exercise the complete native identity/config branch with synthetic tensors,
    # without constructing pretrained weights or invoking any model forward.
    backend = prepared.backend(capture.ModelSettings(**protocol["model_settings"]))
    native_contract = copy.deepcopy(protocol)
    native_contract["test_fixture"] = False
    backend.metadata["is_test_fixture"] = False
    for name, value in vars(prepared.config).items():
        setattr(backend.model.config, name, value)
    capture._validate_backend(backend, native_contract)
    assert not backend.model.training and Model.calls == 0
    backend.close()


@pytest.mark.parametrize("field", ["fields", "backend_sha256", "answer_token_ids"])
def test_tokenizer_identity_mismatch_names_exact_component(prepared, field):
    protocol = copy.deepcopy(prepared.protocol)
    if field == "fields":
        protocol["tokenizer"] = {"path": "unrelated-input.json", "sha256": "unrelated"}
    else:
        protocol["tokenizer"][field] = "changed"
    backend = prepared.backend(capture.ModelSettings(**protocol["model_settings"]))
    with pytest.raises(ValueError, match="Pinned tokenizer identity differs: " + field):
        capture._validate_backend(backend, protocol)
    assert Model.calls == 0
    backend.close()


def test_query_span_selects_last_subtoken_of_exact_question_occurrence():
    tokenizer, original = Tokenizer(), row()
    planned = capture.prepare_row(tokenizer, original)
    start, end = planned["rendered_char_spans"]["query_location"]
    assert planned["selected_tokens"]["query_location_last"]["rendered_char_offset"] == [
        end - 1,
        end,
    ]
    assert planned["selected_tokens"]["query_location_last"]["token_id"] == ord("h") + 10
    assert start > planned["rendered_char_spans"]["account"][1]
    assert (
        planned["token_positions"]["account_last"]
        < planned["token_positions"]["query_location_last"]
    )
    assert planned["token_positions"]["prompt_last"] == len(planned["prompt_token_ids"]) - 1


def test_crossing_boundary_rejected_without_explicit_alignment():
    tokenizer = Tokenizer()
    tokenizer.leading_word = "North"
    with pytest.raises(ValueError, match="crosses query_location"):
        capture.prepare_row(tokenizer, row())
    aligned = row()
    aligned["span_boundary_policy"] = {"query_location": "leading_whitespace"}
    planned = capture.prepare_row(tokenizer, aligned)
    assert planned["selected_tokens"]["query_location_last"]["rendered_char_offset"] == [
        planned["rendered_char_spans"]["query_location"][0] - 1,
        planned["rendered_char_spans"]["query_location"][1],
    ]


def test_right_boundary_crossing_is_never_allowed():
    tokenizer = Tokenizer()
    tokenizer.leading_word = "North"
    original = row()
    original["user_content_char_spans"]["query_location"][1] -= 1
    original["span_boundary_policy"] = {"query_location": "leading_whitespace"}
    with pytest.raises(ValueError, match="crosses query_location"):
        capture.prepare_row(tokenizer, original)


@pytest.mark.parametrize(
    "defect", ["missing_span", "duplicate_user", "ambiguous_content", "ids_differ", "slow"]
)
def test_tokenizer_and_renderer_defects_fail_closed(defect):
    tokenizer, original = Tokenizer(), row()
    if defect == "missing_span":
        del original["user_content_char_spans"]
    elif defect == "duplicate_user":
        original["messages"].append(original["messages"][-1])
    elif defect == "ambiguous_content":
        original["messages"][0]["content"] = original["messages"][-1]["content"]
    elif defect == "ids_differ":
        tokenizer.mismatch = True
    else:
        tokenizer.is_fast = False
        with pytest.raises(ValueError, match="fast tokenizer"):
            capture.tokenizer_binding(tokenizer)
        return
    with pytest.raises(ValueError):
        capture.prepare_row(tokenizer, original)


def test_no_truncation_or_unsafe_journal_ids():
    with pytest.raises(ValueError, match="no truncation"):
        capture.prepare_row(Tokenizer(), row(), max_tokens=10)
    original = row()
    original["record_id"] = "../unsafe"
    with pytest.raises(ValueError, match="journal basenames"):
        capture.prepare_row(Tokenizer(), original)


def test_all_twelve_vectors_same_forward_and_excluded_alignment(prepared):
    result = execute(prepared)
    assert result["status"] == "complete"
    assert Model.calls == 2 and Model.hidden_state_calls == [True, False]
    with np.load(prepared.output / "source_fit" / "features.npz", allow_pickle=False) as saved:
        assert saved["features"].shape == (1, 12, 3)
        assert saved["features"].dtype == np.float32
        assert saved["sites"].tolist() == prepared.protocol["sites"]
        planned = prepared.protocol["planned_rows"][0]
        expected = [
            [
                block,
                planned["token_positions"][position],
                planned["prompt_token_ids"][planned["token_positions"][position]],
            ]
            for block in capture.BLOCKS["llama"]
            for position in capture.POSITIONS
        ]
        np.testing.assert_array_equal(saved["features"][0], expected)
    journal = prepared.output / "source_fit" / "row_journal" / planned["row_id"]
    with np.load(journal / "arrays.npz", allow_pickle=False) as data:
        assert "next_token_logits" not in data
        np.testing.assert_array_equal(data["output_candidate_logits"], [3, 1])
    with np.load(journal / "raw_forward.npz", allow_pickle=False) as data:
        assert "full_forward_logits" not in data
    with np.load(prepared.output / "engineering" / "arrays.npz", allow_pickle=False) as data:
        assert data["next_token_logits"].shape == (256,)
    rows = read_json(prepared.output / "source_fit" / "rows.json")["rows"]
    assert rows[0]["output_yes_no_logitdiff"] == 2
    assert rows[0]["capture"]["same_forward_as_logits"]
    assert not rows[0]["capture"]["output_hidden_states"]
    assert prepared.backend.closed == 1


@pytest.mark.parametrize("defect", ["bank", "review", "protocol", "draft", "absent"])
def test_missing_or_changed_bindings_fail_before_weights(prepared, defect):
    if defect == "bank":
        prepared.bank.write_text("{}", encoding="utf-8")
    elif defect == "review":
        prepared.review.write_text("{}", encoding="utf-8")
    elif defect == "protocol":
        atomic_json(prepared.output / "protocol.json", {})
    elif defect == "draft":
        freeze = read_json(prepared.freeze)
        freeze["status"] = "draft"
        atomic_json(prepared.freeze, freeze)
    with pytest.raises((ValueError, KeyError)):
        execute(prepared, bindings_path=None if defect == "absent" else prepared.freeze)
    assert prepared.backend.created == Model.calls == 0


def fixture_fit_bindings(prepared, tmp_path, monkeypatch):
    """Stub only fit numerical validation; these tests isolate the capture gate."""
    monkeypatch.setattr(
        capture,
        "_load_fit_bundle",
        lambda directory, **kwargs: (read_json(directory / "settings.json"), {}),
    )
    freeze = read_json(prepared.freeze)
    freeze["model_protocol_sha256"] = {
        "llama": file_sha256(prepared.output / "protocol.json"),
        "qwen": "1" * 64,
    }
    fits = {}
    for key in capture.MODELS:
        directory = tmp_path / key
        directory.mkdir()
        settings = {
            "schema": "fresh-readout-fit-v2",
            "status": "frozen",
            "test_fixture": True,
            "research_status": "agent_audited_exploratory",
            "human_review_complete": False,
            "capture_provenance": {
                "bank_file_sha256": prepared.protocol["bank_file_sha256"],
                "model_key": key,
                "model_settings": {
                    **prepared.protocol["model_settings"],
                    "model_id": capture.MODELS[key]["model_id"],
                    "revision": capture.MODELS[key]["revision"],
                    "capture_layers": list(capture.BLOCKS[key]),
                },
                "sites": capture.panel_sites(key),
                "protocol_sha256": freeze["model_protocol_sha256"][key],
            },
            **{
                partition + "_ids": [
                    r["row_id"]
                    for r in prepared.protocol["planned_rows"]
                    if r["partition"] == partition
                ]
                for partition in capture.PARTITIONS["training"]
            },
        }
        atomic_json(directory / "settings.json", settings)
        np.savez(directory / "heads.npz", coefficients=np.zeros((1, 3), dtype=np.float32))
        atomic_json(directory / "candidates.json", {"schema": "isolated-capture-gate-fixture"})
        atomic_json(directory / "manifest.json", {"schema": "isolated-capture-gate-fixture"})
        fits.update(
            {
                str(directory / name): file_sha256(directory / name)
                for name in ("settings.json", "heads.npz", "manifest.json", "candidates.json")
            }
        )
    freeze["fitted_artifact_sha256"] = fits
    atomic_json(prepared.freeze, freeze)
    return freeze


def test_test_capture_requires_fit_bytes_before_any_forward(prepared, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="frozen fitted"):
        execute(prepared, partition="test")
    assert Model.calls == prepared.backend.created == 0
    fixture_fit_bindings(prepared, tmp_path, monkeypatch)
    assert execute(prepared, partition="test")["status"] == "complete"


@pytest.mark.parametrize("defect", ["bank", "ids", "sites", "protocol", "model", "status"])
def test_fitted_provenance_defects_block_all_test_forwards(prepared, tmp_path, monkeypatch, defect):
    freeze = fixture_fit_bindings(prepared, tmp_path, monkeypatch)
    path = tmp_path / "qwen" / "settings.json"
    settings = read_json(path)
    if defect == "bank":
        settings["capture_provenance"]["bank_file_sha256"] = "wrong"
    elif defect == "ids":
        settings["source_fit_ids"] = ["wrong"]
    elif defect == "sites":
        settings["capture_provenance"]["sites"] = capture.panel_sites("llama")
    elif defect == "protocol":
        settings["capture_provenance"]["protocol_sha256"] = "wrong"
    elif defect == "model":
        settings["capture_provenance"]["model_settings"]["revision"] = "mutable"
    else:
        settings["status"] = "draft"
    atomic_json(path, settings)
    freeze["fitted_artifact_sha256"][str(path)] = file_sha256(path)
    atomic_json(prepared.freeze, freeze)
    with pytest.raises(ValueError, match="Fitted bundle|Fit schema"):
        execute(prepared, partition="test")
    assert Model.calls == prepared.backend.created == 0


def test_active_or_unknown_output_lock_blocks_all_partitions(prepared):
    atomic_json(prepared.output / "execution.lock", {"status": "unknown prior attempt"})
    for partition in ("source_fit", "source", "training", "test"):
        with pytest.raises(ValueError, match="locked"):
            execute(prepared, partition=partition)
    assert Model.calls == prepared.backend.created == 0


def test_training_alias_captures_all_four_training_partitions_once(prepared):
    assert execute(prepared, partition="training")["completed_rows"] == 4
    assert Model.calls == 5 and prepared.backend.created == 1
    rows = read_json(prepared.output / "training" / "rows.json")["rows"]
    assert [r["partition"] for r in rows] == list(capture.PARTITIONS["training"])
    for partition in capture.PARTITIONS["training"]:
        with pytest.raises(ValueError, match="Overlapping partition"):
            execute(prepared, partition=partition)


def test_failed_unknown_row_is_preserved_and_never_retried(prepared):
    Model.fail_on_call = 2
    with pytest.raises(RuntimeError, match="preserved synthetic"):
        execute(prepared)
    planned = prepared.protocol["planned_rows"][0]
    journal = prepared.output / "source_fit" / "row_journal" / planned["row_id"]
    assert (journal / "started.json").is_file() and (journal / "failure.json").is_file()
    assert not (prepared.output / "source_fit" / "features.npz").exists()
    calls = Model.calls
    with pytest.raises(ValueError, match="blocks resumption"):
        execute(prepared, resume=True)
    assert Model.calls == calls


def test_engineering_mismatch_preserves_full_observation_and_blocks_cohort(prepared):
    Model.misaligned = True
    with pytest.raises(ValueError, match="alignment differs"):
        execute(prepared)
    assert Model.calls == 1
    assert (prepared.output / "engineering" / "raw_forward.npz").is_file()
    assert (prepared.output / "engineering" / "failure.json").is_file()
    assert not (prepared.output / "source_fit" / "row_journal").exists()
    with pytest.raises(ValueError, match="blocks resumption"):
        execute(prepared, resume=True)


def test_completed_partition_cannot_be_repeated_and_other_partitions_reuse_engineering(prepared):
    execute(prepared)
    with pytest.raises(ValueError, match="cannot be rerun"):
        execute(prepared, resume=True)
    execute(prepared, partition="source_dev")
    assert Model.calls == 3 and Model.hidden_state_calls == [True, False, False]


def test_explicit_resume_between_completed_rows_does_not_repeat_forward(prepared):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("between completed rows")

    with pytest.raises(KeyboardInterrupt):
        execute(prepared, progress=interrupt)
    assert Model.calls == 2
    assert execute(prepared, resume=True)["status"] == "complete"
    assert Model.calls == 2


def test_saved_candidate_summary_tampering_blocks_resume(prepared):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("between completed rows")

    with pytest.raises(KeyboardInterrupt):
        execute(prepared, progress=interrupt)
    planned = prepared.protocol["planned_rows"][0]
    path = prepared.output / "source_fit" / "row_journal" / planned["row_id"] / "complete.json"
    complete = read_json(path)
    complete["output_logits"]["full_vocab_logsumexp"] += 1
    atomic_json(path, complete)
    with pytest.raises(ValueError, match="Saved output measurement changed"):
        execute(prepared, resume=True)


def test_qwen_panel_and_pinned_config_checked(prepared, tmp_path):
    config = copy.copy(prepared.config)
    config.model_type, config.num_hidden_layers = "qwen2", 28
    config._commit_hash = capture.MODELS["qwen"]["revision"]
    result = capture.prepare(
        prepared.bank,
        tmp_path / "qwen",
        model_key="qwen",
        tokenizer=prepared.tokenizer,
        model_config=config,
        forward=Model.forward,
        test_fixture=True,
    )
    assert result["model_settings"]["capture_layers"] == [7, 14, 21, 26]
    assert result["original_site"] == "layer_14_prompt_last"
    config._commit_hash = "mutable"
    with pytest.raises(ValueError, match="Pinned model config differs"):
        capture.prepare(
            prepared.bank,
            tmp_path / "bad",
            model_key="qwen",
            tokenizer=prepared.tokenizer,
            model_config=config,
            forward=Model.forward,
            test_fixture=True,
        )


def test_fixture_cannot_construct_real_backend(prepared):
    with pytest.raises(ValueError, match="cannot load real"):
        capture.execute(
            prepared.output,
            partition="source_fit",
            bindings_path=prepared.freeze,
            backend_factory=capture.HFBackend,
            progress=None,
        )
    assert Model.calls == 0


def test_json_summary_hash_really_binds_stored_summary(prepared):
    execute(prepared)
    planned = prepared.protocol["planned_rows"][0]
    journal = prepared.output / "source_fit" / "row_journal" / planned["row_id"]
    complete = read_json(journal / "complete.json")
    with np.load(journal / "arrays.npz", allow_pickle=False) as data:
        assert data["output_summary_sha256"].item() == canonical_hash(complete["output_logits"])
