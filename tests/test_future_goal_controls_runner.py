"""CPU-only fake backend checks for the frozen, nonresumable feasibility runner."""

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from latent_probes.backends import GenerationResult
from latent_probes.future_goal_controls import BANK_SHA256, HEAD, load_material, qualify_neutral
from latent_probes.storage import atomic_json, canonical_hash, file_sha256, read_json

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data/belief_diagnostic/future_goal_content_controls_draft_v1.json"
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "future_controls_runner", ROOT / "scripts/run_future_goal_controls.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class Tokenizer:
    chat_template = "explicit-fixture-no-model"
    eos_token_id = 2
    pad_token_id = None
    all_special_ids = [2, 6]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        digest = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).digest()
        return [100 + int.from_bytes(digest[i : i + 4], "big") for i in range(0, 32, 4)]

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces=True):
        return "".join(
            {
                2: "" if skip_special_tokens else "<eos>",
                3: "YES",
                4: "NO",
                5: "extra",
                6: "" if skip_special_tokens else "<special>",
                7: " ",
            }[i]
            for i in ids
        )


@pytest.fixture
def attempt(tmp_path, monkeypatch):
    # Source artifacts are entirely synthetic; real frozen bank bytes are read only.
    source, known, fit = (tmp_path / p for p in ("source", "known", "fit"))
    for path in (source, known, fit):
        path.mkdir()
    tokenizer = Tokenizer()
    backend = {
        "backend": "explicit_cpu_fixture",
        "is_test_fixture": True,
        "model_id": "fixture",
        "resolved_revision": "a" * 40,
        "model_type": "llama",
        "dtype": "bfloat16",
        "quantization": "4bit",
        "decoder_blocks": 32,
        "hidden_size": 2,
        "capture_layers": [8, 16, 24],
        "chat_template_sha256": canonical_hash(tokenizer.chat_template),
        "generation_defaults": {"eos_token_id": [2], "pad_token_id": None},
        "special_token_ids": {"eos": 2, "pad": None},
        "attention_implementation": "sdpa",
    }
    metadata = {
        "feature_widths": {"layer_16": 2},
        "heads": {
            HEAD: {"layer": "layer_16", "activation_transform": "frozen", "parameter_prefix": HEAD}
        },
    }
    parameters = {
        "frozen__mean": np.zeros(2),
        "frozen__scale": np.ones(2),
        HEAD + "__coef": np.asarray([[2.0, 0.0]]),
        HEAD + "__intercept": np.asarray([0.0]),
    }
    atomic_json(fit / "frozen_fixture.json", {"description": "not a research fit"})
    monkeypatch.setattr(
        runner, "_load_fit", lambda path: (metadata, parameters, {}, runner._snapshot(path))
    )
    case = {
        "id": "reference",
        "messages": {
            "sender": [
                {"role": "system", "content": "fixture"},
                {"role": "user", "content": "reference"},
            ]
        },
        "questions": {"nested": "Is the fixture positive? Answer only YES or NO."},
    }
    messages = runner.query_messages(case, "sender", "nested")
    atomic_json(
        known / "protocol.json",
        {"panel": {"cases": [case]}, "fitted_artifact_sha256": runner._snapshot(fit)},
    )
    atomic_json(
        known / "rows.json",
        [
            {
                "case_id": "reference",
                "example_id": "reference__sender",
                "view": "sender",
                "condition": "nested_question",
                "messages_text": "\n".join(m["content"] for m in messages),
            }
        ],
    )
    atomic_json(
        known / "captures.json",
        [
            {
                "example_id": "reference__sender",
                "condition": "nested_question",
                "messages_sha256": canonical_hash(messages),
                "prompt_token_ids": runner.token_ids(tokenizer, messages),
                "generated_answer_tokens_supplied": 0,
            }
        ],
    )
    np.savez(known / "features.npz", layer_16=np.asarray([[1.0, 0.0]], np.float32))
    score = runner.score_vector(np.asarray([1.0, 0.0]), metadata, parameters, {})
    (known / "frozen_evaluation").mkdir()
    (known / "frozen_evaluation/predictions.csv").write_text(
        "head_id,example_id,evaluation_condition,p1,prediction\n"
        f"{HEAD},reference__sender,nested_question,{score['probability']},1\n"
    )
    atomic_json(
        known / "manifest.json",
        {"status": "complete", "backend": backend, "output_sha256": runner._snapshot(known)},
    )
    atomic_json(
        source / "protocol.json",
        {
            "fitted_artifact_sha256": runner._snapshot(fit),
            "source_artifact_sha256": runner._snapshot(known),
        },
    )
    atomic_json(
        source / "manifest.json",
        {"status": "complete", "backend": backend, "output_sha256": runner._snapshot(source)},
    )
    config = tmp_path / "config.json"
    atomic_json(
        config,
        {
            "model": {
                "model_id": "fixture",
                "revision": "a" * 40,
                "dtype": "bfloat16",
                "quantization": "4bit",
                "max_input_tokens": 2048,
            }
        },
    )
    monkeypatch.setattr(runner, "native_output", lambda p: Path(p).resolve())
    output = tmp_path / "attempt"
    # Freeze these exercised sources; a parallel author owns the analyzer sources.
    monkeypatch.setattr(
        runner,
        "POLICY_FILES",
        ("scripts/run_future_goal_controls.py", "src/latent_probes/future_goal_controls.py"),
    )
    prepared = runner.prepare_run(BANK, config, source, known, fit, output, tokenizer=tokenizer)

    class Backend:
        captures = 0
        generations = 0
        closed = False
        wrong_probe = False
        invalid = False
        fail_after = None
        prefix_mismatch = False
        reference_mismatch = False
        unexpected_special = False

        def __init__(self, settings):
            self.settings = settings
            self.metadata = {**copy.deepcopy(backend), "capture_layers": [16]}
            self.tokenizer = tokenizer
            self.torch = SimpleNamespace(tensor=lambda x, **kw: np.asarray(x))
            self.input_device = "fixture_cpu"
            self.by_ids = {tuple(r["prompt_token_ids"]): r for r in prepared["planned_rows"]}

        def _capture_token_rows(self, sequence, positions):
            type(self).captures += 1
            assert positions == [sequence.shape[1] - 1]
            row = self.by_ids.get(tuple(sequence[0]))
            positive = row is None or row["gold"] == 1
            if (row is None and self.reference_mismatch) or (row is not None and self.wrong_probe):
                positive = not positive
            return {"layer_16_tokens": np.asarray([[1.0 if positive else -1.0, 0.0]], np.float32)}

        def generate(self, messages, settings, seed, capture=False):
            assert settings.max_new_tokens == 12 and settings.do_sample is False
            assert seed == 20261121 and capture is False
            type(self).generations += 1
            if self.fail_after is not None and self.generations > self.fail_after:
                raise RuntimeError("fixture interruption")
            ids = runner.token_ids(tokenizer, messages)
            row = self.by_ids[tuple(ids)]
            token = 3 if row["gold"] else 4
            generated = [token] + [7] * 11 if self.invalid else [token, 2]
            if self.unexpected_special:
                generated.insert(0, 6)
            return GenerationResult(
                text=tokenizer.decode(generated, skip_special_tokens=True).strip(),
                prompt_token_ids=ids + [999] if self.prefix_mismatch else ids,
                generated_token_ids=generated,
                finish_reason="length" if self.invalid else "eos",
            )

        def close(self):
            type(self).closed = True

    return SimpleNamespace(
        output=output,
        protocol=prepared,
        backend=Backend,
        config=config,
        fit=fit,
        metadata=metadata,
        parameters=parameters,
        source=source,
        known=known,
        tokenizer=tokenizer,
    )


def test_reviewed_material_and_prepare_have_no_model_calls(attempt):
    bank = load_material(BANK)
    assert len(bank["scenes"]) == 16 and len(bank["planned_conditions"]) == 224
    assert file_sha256(BANK) == BANK_SHA256
    assert attempt.backend.captures == attempt.backend.generations == 0
    assert read_json(attempt.output / "manifest.json")["status"] == "prepared_no_model_calls"
    assert attempt.protocol["model_settings"]["local_files_only"] is True
    assert attempt.protocol["model_settings"]["capture_layers"] == [16]
    assert len(attempt.protocol["token_lengths"]["own_minus_archive"]) == 64
    assert len(attempt.protocol["token_lengths"]["map1_minus_map0"]) == 112


def test_passing_neutral_gate_collects_all_once_and_binds_tokens(attempt):
    manifest = runner.execute_run(attempt.output, backend_factory=attempt.backend, progress=None)
    assert manifest["status"] == "complete" and manifest["n_completed_rows"] == 224
    assert attempt.backend.captures == 225 and attempt.backend.generations == 224
    assert attempt.backend.closed
    assert read_json(attempt.output / "qualification.json")["passed"] is True
    rows = [read_json(p) for p in (attempt.output / "row_outputs").glob("*.json")]
    for row in rows:
        assert row["capture"]["generated_answer_tokens_supplied"] == 0
        assert row["capture"]["selected_position"] == len(row["prompt_token_ids"]) - 1
        assert row["prompt_token_ids"] == row["generation"]["prompt_token_ids"]
        assert row["capture"]["prompt_token_ids_sha256"] == row["prompt_token_ids_sha256"]
        assert (
            file_sha256(attempt.output / row["capture"]["feature_path"])
            == row["capture"]["feature_sha256"]
        )
    with pytest.raises(ValueError, match="No overwrite"):
        runner.execute_run(attempt.output, backend_factory=attempt.backend, progress=None)
    assert attempt.backend.generations == 224


def test_neutral_failure_stops_before_any_non_neutral(attempt):
    attempt.backend.wrong_probe = True
    manifest = runner.execute_run(attempt.output, backend_factory=attempt.backend, progress=None)
    assert manifest["status"] == "neutral_failed_stopped"
    assert attempt.backend.generations == 32 and attempt.backend.captures == 33
    rows = [read_json(p) for p in (attempt.output / "row_outputs").glob("*.json")]
    assert all(r["condition"] == "neutral" for r in rows)
    assert read_json(attempt.output / "qualification.json")["passed"] is False


def test_parseable_capped_answers_are_invalid_and_terminal(attempt):
    attempt.backend.invalid = True
    manifest = runner.execute_run(attempt.output, backend_factory=attempt.backend, progress=None)
    assert manifest["status"] == "neutral_failed_stopped"
    gate = read_json(attempt.output / "qualification.json")
    assert gate["overall"]["invalid"] == 32 and gate["overall"]["answer_correct"] == 0
    for path in (attempt.output / "row_outputs").glob("*.json"):
        assert read_json(path)["answer"]["prediction"] is None


def test_unexpected_special_cannot_be_stripped_into_valid_answer(attempt):
    attempt.backend.unexpected_special = True
    manifest = runner.execute_run(attempt.output, backend_factory=attempt.backend, progress=None)
    assert manifest["status"] == "neutral_failed_stopped"
    gate = read_json(attempt.output / "qualification.json")
    assert gate["overall"]["invalid"] == 32
    saved = read_json(next((attempt.output / "row_outputs").glob("*.json")))
    assert saved["generation"]["text"] in ("YES", "NO")
    assert saved["generation"]["unexpected_special_tokens"] == [6]
    assert saved["answer"]["prediction"] is None


@pytest.mark.parametrize("failure", ["prefix_mismatch", "reference_mismatch", "interruption"])
def test_incomplete_attempt_cannot_regenerate(attempt, failure):
    if failure == "interruption":
        attempt.backend.fail_after = 2
    else:
        setattr(attempt.backend, failure, True)
    with pytest.raises((ValueError, RuntimeError)):
        runner.execute_run(attempt.output, backend_factory=attempt.backend, progress=None)
    manifest = read_json(attempt.output / "manifest.json")
    assert manifest["status"] == "failed_incomplete_no_retry" and attempt.backend.closed
    generations = attempt.backend.generations
    with pytest.raises(ValueError, match="No overwrite"):
        runner.execute_run(attempt.output, backend_factory=attempt.backend, progress=None)
    assert attempt.backend.generations == generations
    if failure == "reference_mismatch":
        assert generations == 0
    elif failure == "prefix_mismatch":
        saved = next((attempt.output / "row_outputs").glob("*.json"))
        assert read_json(saved)["generation"]["prompt_token_ids"][-1] == 999
    else:
        assert manifest["n_completed_rows"] == 2
        assert len(list((attempt.output / "row_outputs").glob("*.json"))) == 3


def test_frozen_change_rejected_before_backend_creation(attempt):
    attempt.config.write_text(attempt.config.read_text() + "\n")
    with pytest.raises(ValueError, match="Frozen config changed"):
        runner.execute_run(attempt.output, backend_factory=attempt.backend, progress=None)
    assert attempt.backend.captures == attempt.backend.generations == 0


def test_wrong_fit_bundle_cannot_be_prepared(attempt, tmp_path):
    # Alter only synthetic fit bytes; neither source manifest authorizes this bundle.
    (attempt.fit / "frozen_fixture.json").write_text('{"different":true}')
    with pytest.raises(ValueError, match="exact original assay bindings"):
        runner.prepare_run(
            BANK,
            attempt.config,
            attempt.source,
            attempt.known,
            attempt.fit,
            tmp_path / "wrong_bundle_attempt",
            tokenizer=attempt.tokenizer,
        )
    assert not (tmp_path / "wrong_bundle_attempt").exists()
    assert attempt.backend.captures == attempt.backend.generations == 0


def test_gate_mapping_floor_and_exact_counts():
    rows = [
        {**r, "probe": {"prediction": r["gold"]}, "answer": {"prediction": r["gold"]}}
        for r in load_material(BANK)["planned_conditions"]
        if r["condition"] == "neutral"
    ]
    assert qualify_neutral(rows)["passed"]
    # Overall 27/32 still passes; one mapping's 11/16 must stop continuation.
    for row in [r for r in rows if r["mapping_id"] == 0][:5]:
        row["probe"]["prediction"] = 1 - row["gold"]
    gate = qualify_neutral(rows)
    assert gate["overall"]["probe_correct"] == 27 and not gate["passed"]
    assert gate["per_mapping"]["0"]["probe_correct"] == 11
    with pytest.raises(ValueError, match="exactly 32"):
        qualify_neutral(rows[:-1])


def test_margin_matches_saved_normalization_and_ties_zero(attempt):
    result = runner.score_vector(np.asarray([0.0, 0.0]), attempt.metadata, attempt.parameters, {})
    assert result["margin"] == 0 and result["probability"] == 0.5 and result["prediction"] == 0


def test_gate_channel_floor_with_overall_and_mapping_floors_met():
    rows = [
        {**r, "probe": {"prediction": r["gold"]}, "answer": {"prediction": r["gold"]}}
        for r in load_material(BANK)["planned_conditions"]
        if r["condition"] == "neutral"
    ]
    for row in [r for r in rows if r["channel"] == "observation"][:7]:
        row["answer"]["prediction"] = None
    gate = qualify_neutral(rows)
    assert gate["overall"]["answer_correct"] == 25
    assert all(c["answer_correct"] >= 12 for c in gate["per_mapping"].values())
    assert gate["per_channel"]["observation"]["answer_correct"] == 9
    assert not gate["passed"]


def test_bank_change_is_rejected_before_parsing(tmp_path):
    changed = tmp_path / "changed.json"
    changed.write_bytes(BANK.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA256"):
        load_material(changed)
