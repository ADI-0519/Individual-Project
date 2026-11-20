"""Explicit CPU fixtures exercise provenance, staging and exact replay, not research."""

import copy
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from latent_probes.backends import GenerationResult
from latent_probes.storage import atomic_json, file_sha256, read_json

ROOT = Path(__file__).resolve().parents[1]
RAW_REPORT = "  First factual line.\nSecond factual line.  "


class FakeTokenizer:
    """Exact prompt chunks have stable IDs; generated characters decode exactly."""

    chat_template = "explicit-fixture-role-template-v1"
    all_special_ids = [1, 2]
    eos_token_id = 2

    def __init__(self, settings):
        self.settings = settings

    def get_vocab(self):
        return {"<bos>": 1, "<eos>": 2, **{chr(i): 1000 + i for i in range(128)}}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True and add_generation_prompt is True
        text = json.dumps(messages, ensure_ascii=False, separators=(",", ":")) + "<assistant>"
        # Eight-character chunks remain below the real input limit while retaining
        # every byte of whitespace and every role/message boundary in the identity.
        return [
            1000000 + int(hashlib.sha256(text[i : i + 8].encode()).hexdigest()[:12], 16)
            for i in range(0, len(text), 8)
        ]

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces=False):
        assert clean_up_tokenization_spaces is False
        pieces = []
        for token in ids:
            if token in self.all_special_ids:
                if not skip_special_tokens:
                    pieces.append("<eos>" if token == self.eos_token_id else "<bos>")
            else:
                pieces.append(chr(token - 1000))
        return "".join(pieces)


class FakeBackend:
    def __init__(self, settings, factory):
        self.settings, self.factory = settings, factory
        self.kind = "actor" if settings.model_id.startswith("meta-llama/") else "reader"
        self.closed = False
        self.tokenizer = factory.tokenizer_class(settings)
        self.metadata = {
            "is_test_fixture": factory.fixture_metadata,
            "model_id": settings.model_id,
            "resolved_revision": settings.revision,
            "dtype": settings.dtype,
            "quantization": settings.quantization,
            "capture_layers": list(settings.capture_layers),
            "hidden_size": 4,
            "generation_defaults": {"eos_token_id": 2},
        }
        self.input_device = "cpu"
        self.torch = SimpleNamespace(
            tensor=lambda values, *, device: np.asarray(values, dtype=np.int64),
            cuda=SimpleNamespace(is_available=lambda: False),
        )
        factory.events.append(("load", self.kind))

    def generate(self, messages, settings, seed, *, capture):
        assert not self.closed and capture is False and seed == 42
        assert settings.do_sample is False
        assert settings.max_new_tokens == (192 if self.kind == "actor" else 8)
        self.factory.calls[self.kind] += 1
        number = self.factory.calls[self.kind]
        self.factory.events.append(("generate", self.kind, number))
        if self.kind == "reader":
            # No first reader output may precede the complete actual-input plan.
            planned = read_json(self.factory.output / "reader_plan.json")
            assert len(planned) == 240 and all("prompt_token_ids" in row for row in planned)
            assert not any(b.kind == "actor" and not b.closed for b in self.factory.instances)
            if number == self.factory.fail_reader_at:
                self.factory.fail_reader_at = None
                raise RuntimeError("Explicit reader fixture failure")
            text = "DEFER"
        else:
            text = "" if number == 1 and self.factory.empty_first else RAW_REPORT
        generated = [1000 + ord(char) for char in text] + [2]
        if self.kind == "actor" and self.factory.internal_special and number == 2:
            generated.insert(2, 1)
        return GenerationResult(
            text=text.strip(),
            prompt_token_ids=self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            ),
            generated_token_ids=generated,
            finish_reason="eos",
            elapsed_seconds=0.0,
        )

    def _capture_token_rows(self, sequence, positions):
        self.factory.capture_calls += 1
        if self.factory.capture_calls == self.factory.fail_capture_at:
            self.factory.fail_capture_at = None
            raise RuntimeError("Explicit saved-sequence capture fixture failure")
        assert sequence.ndim == 2 and sequence.shape[0] == 1
        values = np.asarray([[float(position + j) for j in range(4)] for position in positions])
        return {"layer_8_tokens": values}

    def close(self):
        assert not self.closed
        self.closed = True
        self.factory.events.append(("close", self.kind))


class BackendFactory:
    def __init__(
        self,
        output,
        *,
        empty_first=False,
        fail_capture_at=None,
        fail_reader_at=None,
        internal_special=False,
    ):
        self.output = output
        self.empty_first = empty_first
        self.fail_capture_at = fail_capture_at
        self.fail_reader_at = fail_reader_at
        self.internal_special = internal_special
        self.fixture_metadata = True
        self.tokenizer_class = FakeTokenizer
        self.calls = Counter(actor=0, reader=0)
        self.capture_calls = 0
        self.events, self.instances = [], []

    def __call__(self, settings):
        if settings.model_id.startswith("Qwen/"):
            assert not any(b.kind == "actor" and not b.closed for b in self.instances)
        backend = FakeBackend(settings, self)
        self.instances.append(backend)
        return backend


@pytest.fixture
def runner(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "natural_selection_runner_fixture", ROOT / "scripts/run_natural_selection_pilot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Test fixtures bind a real input bank plus a dedicated immutable source, so
    # concurrent authoring of the runner cannot masquerade as a runtime failure.
    source = tmp_path / "explicit_fixture_source.txt"
    source.write_text("Frozen software fixture source, not a model result.\n")
    module.fixture_source = source
    monkeypatch.setattr(module, "source_paths", lambda panel: [Path(panel), source])
    monkeypatch.setattr(
        module,
        "environment_info",
        lambda: {
            "python": "explicit-fixture-3.11",
            "platform": "CPU fixture",
            "packages": {"fixture": "1"},
        },
    )
    return module


def prepared(runner, tmp_path, name="attempt"):
    output = tmp_path / name
    result = runner.prepare(output, tokenizer_factory=FakeTokenizer)
    assert result["status"] == "prepared" and result["is_test_fixture"] is True
    return output


def test_all_conditions_exact_wire_content_cache_and_fixture_identity(runner, tmp_path):
    output = prepared(runner, tmp_path)
    factory = BackendFactory(output, empty_first=True, internal_special=True)
    manifest = runner.execute(output, backend_factory=factory)
    assert manifest["status"] == "complete" and manifest["is_test_fixture"] is True
    assert all(b.closed for b in factory.instances)
    assert factory.events.index(("close", "actor")) < factory.events.index(("load", "reader"))
    sender = read_json(output / "sender_rows.json")
    reader = read_json(output / "reader_rows.json")
    summary = read_json(output / "summary.json")
    assert len(sender) == len({r["job"]["example_id"] for r in sender}) == 32
    assert len(reader) == len({r["example_id"] for r in reader}) == 240
    assert Counter(row["version"] for row in reader) == {
        "original": 96,
        "restored": 48,
        "irrelevant": 48,
        "no_report": 24,
        "full_disclosure": 24,
    }
    assert Counter((r["job"]["objective"], r["job"]["word_budget"]) for r in sender) == {
        (goal, budget): 8 for goal in ("neutral", "favorable") for budget in (25, 50)
    }
    assert sender[0]["report_text"] == "" and sender[0]["generation"]["generated_token_ids"] == [2]
    assert sender[0]["capture"]["status"] == "unavailable_no_content"
    assert "feature_file" not in sender[0]
    assert factory.calls["actor"] == 32 and factory.capture_calls == 31
    assert sender[1]["report_text"] == RAW_REPORT
    assert sender[1]["generation"]["text"] == RAW_REPORT.strip()
    assert sender[1]["generation"]["unexpected_special_tokens"] == [{"index": 2, "id": 1}]
    assert "<bos>" in sender[1]["generation"]["decoded_including_special_tokens"]
    for row in sender:
        expected = hashlib.sha256(row["report_text"].encode("utf-8")).hexdigest()
        assert row["report_sha256"] == row["generation"]["raw_text_sha256"] == expected
    by_sender = {row["job"]["example_id"]: row for row in sender}
    for row in reader:
        if row["version"] == "original":
            exact = by_sender[row["source_example_id"]]["report_text"]
            assert row["report_text"] == exact
            assert row["messages"][-1]["content"].split("\n\nReport:\n", 1)[1] == exact
    unique = len({r["cache_key"] for r in reader})
    assert unique < 240 and any(r["cache_hit"] for r in reader)
    assert unique == factory.calls["reader"] == summary["unique_reader_calls"]
    assert manifest["generation_attempts"] == {"actor": 32, "reader": unique}
    assert factory.calls.total() < 272
    assert summary["is_test_fixture"] is True and summary["new_probe_fits"] == 0
    assert summary["advancement_status"] == "not_determined_without_semantic_review"
    # Actual fixture answers are not replaced by the authored complete-source gold.
    assert {r["decision"] for r in reader} == {"DEFER"}
    assert summary["controls"]["full_disclosure"]["normative_correct"] == 0
    old_calls = factory.calls.copy()
    with pytest.raises(FileExistsError):
        runner.execute(output, resume=True, backend_factory=factory)
    assert factory.calls == old_calls
    with pytest.raises(FileExistsError):
        runner.prepare(output, tokenizer_factory=FakeTokenizer)


def test_capture_failure_resume_reuses_saved_generation_without_regeneration(runner, tmp_path):
    output = prepared(runner, tmp_path)
    factory = BackendFactory(output, fail_capture_at=1)
    with pytest.raises(RuntimeError, match="capture fixture failure"):
        runner.execute(output, stage="reports", backend_factory=factory)
    saved = read_json(output / "sender_rows.json")
    assert len(saved) == 1 and saved[0]["capture"]["status"] == "pending"
    frozen_generation = copy.deepcopy(saved[0]["generation"])
    assert factory.calls["actor"] == 1 and all(b.closed for b in factory.instances)
    with pytest.raises(ValueError, match="resume"):
        runner.execute(output, stage="reports", backend_factory=factory)
    result = runner.execute(output, stage="reports", resume=True, backend_factory=factory)
    assert result["status"] == "reports_complete"
    assert factory.calls == {"actor": 32, "reader": 0}
    assert factory.capture_calls == 33  # Failed capture retried; generation was not.
    final = read_json(output / "sender_rows.json")
    assert final[0]["generation"] == frozen_generation
    assert all(row["capture"]["status"] == "complete" for row in final)
    assert len(list((output / "features").glob("*.npz"))) == 32
    for row in final:
        with np.load(output / row["feature_file"], allow_pickle=False) as arrays:
            assert arrays["layer_8_tokens"].shape == (len(RAW_REPORT), 4)
    report_bank_sha = file_sha256(output / "sender_rows.json")
    feature_shas = {p.name: file_sha256(p) for p in (output / "features").glob("*.npz")}
    factory.fail_reader_at = 2
    with pytest.raises(RuntimeError, match="reader fixture failure"):
        runner.execute(output, stage="readers", backend_factory=factory)
    first_cache = read_json(output / "reader_cache.json")
    assert len(first_cache) == 1
    key, frozen_reader_call = next(iter(first_cache.items()))
    result = runner.execute(output, stage="readers", resume=True, backend_factory=factory)
    assert result["status"] == "complete"
    assert read_json(output / "reader_cache.json")[key] == frozen_reader_call
    assert read_json(output / "reader_rows.json")[0]["cache_hit"] is True
    assert factory.calls["reader"] == result["unique_reader_calls"] + 1  # One failed attempt.
    assert factory.calls["actor"] == 32 and factory.capture_calls == 33
    assert file_sha256(output / "sender_rows.json") == report_bank_sha
    assert {p.name: file_sha256(p) for p in (output / "features").glob("*.npz")} == feature_shas


@pytest.mark.parametrize("damage", ["source", "protocol", "preflight", "forged_protocol_hash"])
def test_prepared_source_or_protocol_corruption_stops_before_backend(runner, tmp_path, damage):
    output = prepared(runner, tmp_path)
    factory = BackendFactory(output)
    if damage == "source":
        runner.fixture_source.write_text("Changed after freeze.\n")
    elif damage in {"protocol", "forged_protocol_hash"}:
        protocol = read_json(output / "protocol.json")
        protocol["seed"] = 43
        atomic_json(output / "protocol.json", protocol)
        if damage == "forged_protocol_hash":
            manifest = read_json(output / "manifest.json")
            manifest["protocol_sha256"] = file_sha256(output / "protocol.json")
            atomic_json(output / "manifest.json", manifest)
    else:
        preflight = read_json(output / "preflight.json")
        next(iter(preflight["actor_inputs"].values())).append(123)
        atomic_json(output / "preflight.json", preflight)
    with pytest.raises(ValueError):
        runner.execute(output, backend_factory=factory)
    assert not factory.instances and factory.calls.total() == 0


def test_tokenizer_identity_change_stops_before_any_generation(runner, tmp_path):
    output = prepared(runner, tmp_path)
    factory = BackendFactory(output)

    class ChangedTokenizer(FakeTokenizer):
        chat_template = "changed-fixture-template"

    factory.tokenizer_class = ChangedTokenizer
    with pytest.raises(ValueError, match="tokenizer changed"):
        runner.execute(output, backend_factory=factory)
    assert factory.calls.total() == 0 and all(b.closed for b in factory.instances)


def test_preparation_refuses_overflow_without_truncation_or_generation(runner, tmp_path):
    class OverflowTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            return [999] * 4097

    output = tmp_path / "overflow_attempt"
    with pytest.raises(ValueError, match="no truncation"):
        runner.prepare(output, tokenizer_factory=OverflowTokenizer)
    manifest = read_json(output / "manifest.json")
    assert manifest["status"] == "failed_preparation"
    assert not (output / "sender_rows.json").exists()
    assert not (output / "reader_cache.json").exists()


def test_fixture_attempt_cannot_run_as_production_or_lie_about_backend(runner, tmp_path):
    output = prepared(runner, tmp_path)
    # This fails before constructing the genuine HFBackend or loading weights.
    with pytest.raises(ValueError, match="real model run"):
        runner.execute(output, backend_factory=runner.HFBackend)
    factory = BackendFactory(output)
    factory.fixture_metadata = False
    with pytest.raises(ValueError, match="fixture identity"):
        runner.execute(output, backend_factory=factory)
    assert factory.calls.total() == 0 and all(b.closed for b in factory.instances)


def test_saved_generation_tampering_is_not_accepted_during_capture_resume(runner, tmp_path):
    output = prepared(runner, tmp_path)
    factory = BackendFactory(output, fail_capture_at=1)
    with pytest.raises(RuntimeError):
        runner.execute(output, stage="reports", backend_factory=factory)
    rows = read_json(output / "sender_rows.json")
    rows[0]["generation"]["finish_reason"] = "length"
    atomic_json(output / "sender_rows.json", rows)
    with pytest.raises(ValueError):
        runner.execute(output, stage="reports", resume=True, backend_factory=factory)
    assert factory.calls["actor"] == 1


def test_partial_reader_cache_tampering_stops_resume_without_new_calls(runner, tmp_path):
    output = prepared(runner, tmp_path)
    factory = BackendFactory(output, fail_reader_at=2)
    with pytest.raises(RuntimeError, match="reader fixture failure"):
        runner.execute(output, backend_factory=factory)
    cache_path = output / "reader_cache.json"
    cache = read_json(cache_path)
    assert len(cache) == 1
    next(iter(cache.values()))["generation"]["finish_reason"] = "length"
    atomic_json(cache_path, cache)
    calls = factory.calls.copy()
    with pytest.raises(ValueError):
        runner.execute(output, stage="readers", resume=True, backend_factory=factory)
    assert factory.calls == calls


def test_saved_feature_tampering_stops_reader_only_stage(runner, tmp_path):
    output = prepared(runner, tmp_path)
    factory = BackendFactory(output)
    runner.execute(output, stage="reports", backend_factory=factory)
    first = read_json(output / "sender_rows.json")[0]
    with (output / first["feature_file"]).open("ab") as stream:
        stream.write(b"changed fixture feature bytes")
    with pytest.raises(ValueError):
        runner.execute(output, stage="readers", backend_factory=factory)
    assert factory.calls["reader"] == 0


def test_changed_host_runtime_stops_resume_before_reusing_saved_coordinates(
    runner, tmp_path, monkeypatch
):
    output = prepared(runner, tmp_path)
    factory = BackendFactory(output, fail_capture_at=1)
    with pytest.raises(RuntimeError):
        runner.execute(output, stage="reports", backend_factory=factory)
    monkeypatch.setattr(
        runner,
        "environment_info",
        lambda: {
            "python": "explicit-fixture-3.11",
            "platform": "different CPU fixture host",
            "packages": {"fixture": "1"},
        },
    )
    with pytest.raises(ValueError, match="runtime changed on resume"):
        runner.execute(output, stage="reports", resume=True, backend_factory=factory)
    assert factory.calls == {"actor": 1, "reader": 0}
    assert factory.capture_calls == 1
    assert all(b.closed for b in factory.instances)
