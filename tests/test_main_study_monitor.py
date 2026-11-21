"""Evidence-fold and execution safeguards using explicit CPU engineering fixtures."""

from __future__ import annotations

import copy
import hashlib
import json
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from latent_probes import main_study_monitor as monitor
from latent_probes.backends import GenerationResult
from latent_probes.main_study_bank import prompt_digest
from latent_probes.main_study_fit import save_fit_bundle
from latent_probes.storage import atomic_json, file_sha256, read_json

MESSAGES = [
    {"role": "system", "content": "Explicit synthetic engineering fixture, not a research case."},
    {
        "role": "user",
        "content": "The physical places are red cabinet and blue locker. "
        "Sender accepts the reading records. The private world check is not reader evidence.",
    },
]


def extraction(initial="red cabinet", updates=None, queried="blue locker"):
    return {
        "initial_location": initial,
        "updates": updates
        if updates is not None
        else [{"location": "blue locker", "accepted_access": True}],
        "queried_location": queried,
    }


@pytest.mark.parametrize("initial,update,receipt,query", tuple(product(range(2), repeat=4)))
def test_one_step_fold_matches_public_update_rule(initial, update, receipt, query):
    names = ["red cabinet", "blue locker"]
    data = extraction(
        names[initial],
        [{"location": names[update], "accepted_access": bool(receipt)}],
        names[query],
    )
    result = monitor.parse_monitor_output(json.dumps(data), MESSAGES)
    assert result["valid"] is True
    assert result["prediction"] == int((update if receipt else initial) == query)
    assert result["score"] == float(result["prediction"])


@pytest.mark.parametrize("initial,first,b1,b2,query", tuple(product(range(2), repeat=5)))
def test_correction_fold_preserves_order_and_unread_previous_belief(initial, first, b1, b2, query):
    names = ["red cabinet", "blue locker"]
    data = extraction(
        names[initial],
        [
            {"location": names[first], "accepted_access": bool(b1)},
            {"location": names[1 - first], "accepted_access": bool(b2)},
        ],
        names[query],
    )
    result = monitor.parse_monitor_output(json.dumps(data), MESSAGES)
    expected = (1 - first) if b2 else (first if b1 else initial)
    assert result["valid"] is True
    assert result["prediction"] == int(expected == query)
    assert result["final_location"] == names[expected]


@pytest.mark.parametrize(
    "raw",
    [
        "YES",
        "",
        "{}",
        "[]",
        "null",
        "```json\n{}\n```",
        "{} trailing",
        '{"initial_location":"red cabinet","initial_location":"blue locker"}',
        '{"initial_location":NaN}',
    ],
)
def test_malformed_outputs_remain_invalid_without_fake_scores(raw):
    result = monitor.parse_monitor_output(raw, MESSAGES)
    assert result["valid"] is False
    assert result["prediction"] is result["score"] is None
    assert result["extraction"] is None and result["invalid_reason"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(extra="YES"),
        lambda d: d.update(updates=[]),
        lambda d: d.update(updates=d["updates"] * 3),
        lambda d: d["updates"][0].update(accepted_access=1),
        lambda d: d["updates"][0].update(accepted_access="true"),
        lambda d: d["updates"][0].update(extra="leak"),
        lambda d: d.update(initial_location="RED CABINET"),
        lambda d: d.update(initial_location="invented place"),
        lambda d: d.update(initial_location=" red cabinet"),
        lambda d: d.update(initial_location=0),
        lambda d: d.update(queried_location=None),
    ],
)
def test_schema_does_not_coerce_repair_or_invent_locations(mutate):
    data = extraction()
    mutate(data)
    assert monitor.parse_monitor_output(json.dumps(data), MESSAGES)["valid"] is False


def test_outer_whitespace_is_only_normalization_and_cap_is_invalid():
    data = extraction()
    result = monitor.parse_monitor_output(" \n" + json.dumps(data) + "\t", MESSAGES)
    assert result["valid"] is True and result["extraction"] == data
    capped = monitor.parse_monitor_output(json.dumps(data), MESSAGES, finish_reason="length")
    assert capped["valid"] is False and capped["score"] is None


def test_model_payload_contains_prose_only_and_never_structured_oracle():
    policy = monitor.load_monitor_protocol()
    messages = monitor.monitor_messages(MESSAGES, policy)
    assert messages[0]["content"] == policy["system_prompt"]
    assert json.loads(messages[1]["content"][len(policy["user_prefix"]) :]) == MESSAGES
    assert policy["is_symbolic_oracle"] is False
    assert policy["max_new_tokens"] == 384 and policy["retries"] == 0
    with pytest.raises(ValueError, match="original prose"):
        monitor.monitor_messages({"gold": 1, "messages": MESSAGES}, policy)
    with pytest.raises(ValueError, match="only role"):
        monitor.monitor_messages([{**MESSAGES[0], "gold": 1}], policy)


class Tokenizer:
    chat_template = "explicit-synthetic-engineering-fixture-not-measured"
    eos_token_id = 2
    pad_token_id = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        return [1, *hashlib.sha256(json.dumps(messages).encode()).digest()[:8], 7]

    def decode(self, ids, *, skip_special_tokens):
        return "".join(
            "" if token == 2 and skip_special_tokens else "<EOS>" if token == 2 else chr(token - 10)
            for token in ids
        )


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    bank_path, ledger, freeze = [tmp_path / f"{name}.json" for name in ("bank", "ledger", "freeze")]
    atomic_json(bank_path, {"test_fixture": True, "schema": "explicit-synthetic-bank"})
    atomic_json(ledger, {"test_fixture": True, "human_validation": False})
    monkeypatch.setattr(monitor, "load_bank", read_json)
    monkeypatch.setattr(
        monitor,
        "verify_review_ledger",
        lambda bank, path: {
            "test_fixture": True,
            "review_stub_only_inside_pytest": file_sha256(path),
        },
    )
    rows = [
        {
            "case_id": f"fixture_{i}",
            "split": split,
            "messages": [
                MESSAGES[0],
                {"role": "user", "content": MESSAGES[1]["content"] + f" Fixture prose number {i}."},
            ],
            "state": {"POISON_STATE_MUST_NOT_ENTER_MODEL": 1},
            "gold": "POISON_GOLD",
        }
        for i, split in enumerate(("development", "development", "test"))
    ]
    rows.append(
        {
            "case_id": "fixture_train",
            "split": "train",
            "messages": [
                MESSAGES[0],
                {"role": "user", "content": "Excluded training fixture prose"},
            ],
        }
    )
    config = SimpleNamespace(
        _commit_hash=monitor.MODELS["qwen"]["revision"],
        model_type="qwen2",
        num_hidden_layers=28,
        max_position_embeddings=4096,
    )
    output = tmp_path / "monitor"
    protocol = monitor.prepare_monitor(
        bank_path,
        output,
        tokenizer=Tokenizer(),
        model_config=config,
        test_fixture=True,
        row_builder=lambda _: copy.deepcopy(rows),
    )
    baseline = tmp_path / "baseline.json"
    atomic_json(baseline, {"test_fixture": True})
    atomic_json(
        freeze,
        {
            "schema": "main-study-execution-freeze-v1",
            "status": "frozen",
            "bank_sha256": protocol["bank_sha256"],
            "source_sha256": protocol["policy_sha256"],
            "text_monitor_protocol_path": protocol["monitor_protocol_path"],
            "text_monitor_protocol_sha256": protocol["monitor_protocol_sha256"],
            "baseline_protocol_path": str(baseline),
            "baseline_protocol_sha256": file_sha256(baseline),
        },
    )

    class Backend:
        created = 0
        calls = 0
        closed = 0
        fail_on_call = None
        capped = False
        invalid = False
        bad_prompt = False
        received = []

        def __init__(self, settings):
            type(self).created += 1
            self.tokenizer = Tokenizer()
            self.metadata = {
                "is_test_fixture": True,
                "model_id": settings.model_id,
                "resolved_revision": settings.revision,
                "dtype": settings.dtype,
                "quantization": settings.quantization,
                "capture_layers": [0],
                "model_type": "qwen2",
                "decoder_blocks": 28,
            }
            self.model = SimpleNamespace(training=False, eval=lambda: None)

        def generate(self, messages, settings, seed, *, capture):
            assert settings.max_new_tokens == 384 and settings.do_sample is False
            assert seed == 20261121 and capture is False
            cls = type(self)
            cls.calls += 1
            cls.received.append(copy.deepcopy(messages))
            if cls.calls == cls.fail_on_call:
                raise RuntimeError("preserved synthetic generation failure")
            raw = "not-json" if cls.invalid else " \n" + json.dumps(extraction()) + "\n"
            generated = [ord(c) + 10 for c in raw] + ([] if cls.capped else [2])
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            return GenerationResult(
                text=raw.strip(),
                prompt_token_ids=prompt + [99] if cls.bad_prompt else prompt,
                generated_token_ids=generated,
                finish_reason="length" if cls.capped else "eos",
            )

        def close(self):
            type(self).closed += 1

    return SimpleNamespace(
        bank=bank_path,
        output=output,
        protocol=protocol,
        ledger=ledger,
        freeze=freeze,
        baseline=baseline,
        backend=Backend,
        rows=rows,
    )


def execute(prepared, **kwargs):
    options = {
        "partition": "development",
        "ledger_path": prepared.ledger,
        "freeze_path": prepared.freeze,
        "backend_factory": prepared.backend,
        "progress": None,
    }
    return monitor.execute_monitor(prepared.output, **{**options, **kwargs})


def test_tokenizer_preparation_loads_no_weights_and_removes_all_hidden_metadata(prepared):
    assert prepared.backend.created == prepared.backend.calls == 0
    assert prepared.protocol["preflight"]["model_weights_loaded"] is False
    assert prepared.protocol["preflight"]["human_review_validated"] is False
    assert prepared.protocol["preflight"]["rows"] == 3
    for row in prepared.protocol["planned_rows"]:
        assert "gold" not in row and "state" not in row
        assert "POISON" not in json.dumps(row["monitor_messages"])
        assert row["prompt_sha256"] == prompt_digest(row["messages"])


def test_fixed_generation_exports_analysis_schema_and_faithful_raw_text(prepared):
    result = execute(prepared)
    assert result["status"] == "complete" and prepared.backend.calls == 2
    assert prepared.backend.closed == 1
    payload = read_json(prepared.output / "records.json")
    assert payload["schema"] == "main-study-text-monitor-v1"
    assert payload["test_fixture"] is True and payload["partition"] == "development"
    assert payload["monitor_protocol_sha256"] == file_sha256(monitor.DEFAULT_PROTOCOL)
    assert payload["bank_sha256"] == prepared.protocol["bank_sha256"]
    assert [r["case_id"] for r in payload["records"]] == ["fixture_0", "fixture_1"]
    assert result["records_sha256"] == file_sha256(prepared.output / "records.json")
    for row in payload["records"]:
        assert row["valid"] is True and row["prediction"] == row["score"] == 1
        assert row["raw_response"].startswith(" \n") and row["raw_response"].endswith("\n")
        raw = read_json(prepared.output / "row_journal" / row["case_id"] / "raw_generation.json")
        assert raw["raw_response"] == row["raw_response"]
        assert raw["raw_response_with_special_tokens"].endswith("<EOS>")
    assert all("POISON" not in json.dumps(messages) for messages in prepared.backend.received)


@pytest.mark.parametrize("attribute", ("capped", "invalid"))
def test_invalid_or_capped_rows_are_completed_outcomes_without_retry(prepared, attribute):
    setattr(prepared.backend, attribute, True)
    execute(prepared)
    rows = read_json(prepared.output / "records.json")["records"]
    assert prepared.backend.calls == 2
    assert all(
        not row["valid"] and row["prediction"] is None and row["score"] is None for row in rows
    )


@pytest.mark.parametrize("target", ("bank", "baseline"))
def test_frozen_input_changes_fail_before_weights(prepared, target):
    getattr(prepared, target).write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError):
        execute(prepared)
    assert prepared.backend.created == 0


@pytest.mark.parametrize("missing", ("ledger_path", "freeze_path"))
def test_missing_review_or_freeze_is_not_bypassed(prepared, missing):
    with pytest.raises(ValueError, match="human-review"):
        execute(prepared, **{missing: None})
    assert prepared.backend.created == 0


def test_actual_review_failure_is_not_bypassed(prepared, monkeypatch):
    def reject(*args):
        raise ValueError("Missing actual human review")

    monkeypatch.setattr(monitor, "verify_review_ledger", reject)
    with pytest.raises(ValueError, match="actual human"):
        execute(prepared)
    assert prepared.backend.created == 0


def test_fixture_cannot_load_real_weights(prepared):
    with pytest.raises(ValueError, match="cannot load real"):
        execute(prepared, backend_factory=monitor.HFBackend)


def test_main_test_monitor_waits_for_frozen_fits(prepared, tmp_path):
    with pytest.raises(ValueError, match="development-selected"):
        execute(prepared, partition="test")
    assert prepared.backend.created == 0
    directory = tmp_path / "fit"
    save_fit_bundle(
        directory,
        {
            "schema": "main-study-fit-v1",
            "test_fixture": True,
            "capture_provenance": {"bank_sha256": prepared.protocol["bank_sha256"]},
        },
        {"fixture_only": np.asarray([1.0])},
    )
    with pytest.raises(ValueError, match="absent from execution freeze"):
        execute(prepared, partition="test", fit_dir=directory)
    freeze = read_json(prepared.freeze)
    freeze["fitted_artifact_sha256"] = {
        str(directory / name): file_sha256(directory / name)
        for name in ("settings.json", "heads.npz")
    }
    atomic_json(prepared.freeze, freeze)
    execute(prepared, partition="test", fit_dir=directory)
    assert prepared.backend.calls == 1
    assert read_json(prepared.output / "records.json")["records"][0]["case_id"] == "fixture_2"


def test_unknown_attempt_is_preserved_and_never_retried(prepared):
    prepared.backend.fail_on_call = 2
    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        execute(prepared)
    failed = prepared.output / "row_journal" / "fixture_1"
    assert (failed / "started.json").is_file() and (failed / "failure.json").is_file()
    assert not (prepared.output / "records.json").exists()
    with pytest.raises(ValueError, match="blocks resumption"):
        execute(prepared, resume=True)
    assert prepared.backend.calls == 2


def test_resume_only_after_durable_complete_row_without_duplicate_call(prepared):
    def interrupt(message):
        raise KeyboardInterrupt("explicit engineering interruption")

    with pytest.raises(KeyboardInterrupt):
        execute(prepared, progress=interrupt)
    assert prepared.backend.calls == 1
    with pytest.raises(ValueError, match="explicit resume"):
        execute(prepared)
    execute(prepared, resume=True)
    assert prepared.backend.calls == 2
    with pytest.raises(ValueError, match="cannot be rerun"):
        execute(prepared, resume=True)


def test_prefix_mismatch_preserves_generated_output_before_rejection(prepared):
    prepared.backend.bad_prompt = True
    with pytest.raises(ValueError, match="token IDs differ"):
        execute(prepared)
    directory = prepared.output / "row_journal" / "fixture_0"
    assert (directory / "raw_generation.json").is_file()
    assert (directory / "failure.json").is_file()
    assert prepared.backend.calls == 1


def test_completed_outcome_tampering_blocks_resume(prepared):
    def interrupt(message):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute(prepared, progress=interrupt)
    path = prepared.output / "row_journal" / "fixture_0" / "complete.json"
    row = read_json(path)
    row["prediction"] = 0
    atomic_json(path, row)
    with pytest.raises(ValueError, match="parsing changed"):
        execute(prepared, resume=True)
    assert prepared.backend.calls == 1
