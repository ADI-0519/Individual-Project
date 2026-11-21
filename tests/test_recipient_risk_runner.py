"""CPU fixtures prove staging and provenance, never experimental feasibility."""

import copy
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from latent_probes.backends import GenerationResult
from latent_probes.storage import atomic_json, canonical_hash, file_sha256, read_json

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "risk_runner_shared_cpu_fixtures", ROOT / "tests/test_natural_selection_runner.py"
)
shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shared)
RAW_REPORT = shared.RAW_REPORT


class FakeTokenizer(shared.FakeTokenizer):
    """Hash every input byte into 128-character chunks; these are not model tokens."""

    chat_template = "explicit-risk-fixture-role-template-128-v1"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True and add_generation_prompt is True
        text = json.dumps(messages, ensure_ascii=False, separators=(",", ":")) + "<assistant>"
        return [
            1000000 + int(canonical_hash(text[start : start + 128])[:12], 16)
            for start in range(0, len(text), 128)
        ]


class RiskBackend(shared.FakeBackend):
    def generate(self, messages, settings, seed, *, capture):
        assert not self.closed and not capture and seed == 42
        assert not settings.do_sample
        self.factory.calls[self.kind] += 1
        number = self.factory.calls[self.kind]
        self.factory.events.append(("generate", self.kind, number))
        if self.kind == "actor":
            assert read_json(self.factory.output / "qualification.json")["status"] == "passed"
            text = "" if self.factory.empty_first and number == 1 else RAW_REPORT
        else:
            assert not any(b.kind == "actor" and not b.closed for b in self.factory.instances)
            plan_path = self.factory.output / "reader_plan.json"
            if not plan_path.exists():
                plan_path = self.factory.output / "control_plan.json"
            if plan_path not in self.factory.plans:
                planned = read_json(plan_path)
                expected_size = 504 if plan_path.stem == "reader_plan" else 72
                assert len(planned) == expected_size
                assert all("prompt_token_ids" in row for row in planned)
                self.factory.plans[plan_path] = {
                    canonical_hash(row["messages"]): row for row in planned
                }
            match = self.factory.plans[plan_path].get(canonical_hash(messages))
            assert match, "No answer may precede a full preflight of every actual payload"
            if number == self.factory.fail_reader_at:
                self.factory.fail_reader_at = None
                raise RuntimeError("Explicit reader fixture failure")
            if self.factory.controls_pass and match.get("normative_control_decision") in {
                "APPROVE",
                "REJECT",
                "DEFER",
            }:
                text = match["normative_control_decision"]
            else:
                text = "DEFER"
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


class RiskFactory(shared.BackendFactory):
    def __init__(self, output, *, controls_pass=True, **kwargs):
        super().__init__(output, **kwargs)
        self.tokenizer_class = FakeTokenizer
        self.controls_pass = controls_pass
        self.plans = {}

    def __call__(self, settings):
        backend = RiskBackend(settings, self)
        self.instances.append(backend)
        return backend


@pytest.fixture
def runner(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "recipient_risk_runner_fixture", ROOT / "scripts/run_recipient_risk_validation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "immutable_fixture_source.txt"
    source.write_text("Explicit CPU fixture source, not model evidence.\n")
    module.fixture_source = source
    monkeypatch.setattr(module, "source_paths", lambda panel: [Path(panel), source])
    fixed_environment = lambda: {  # noqa: E731
        "python": "explicit-fixture-3.11",
        "platform": "CPU fixture",
        "packages": {"fixture": "1"},
    }
    monkeypatch.setattr(module, "environment_info", fixed_environment)
    monkeypatch.setattr(module.core, "environment_info", fixed_environment)
    return module


def prepare(runner, tmp_path):
    output = tmp_path / "attempt"
    result = runner.prepare(output, tokenizer_factory=FakeTokenizer)
    assert result["status"] == "prepared" and result["is_test_fixture"] is True
    return output


def qualify_fixture(runner, tmp_path, **factory_options):
    output = prepare(runner, tmp_path)
    factory = RiskFactory(output, **factory_options)
    result = runner.execute(output, stage="controls", backend_factory=factory)
    assert result["status"] == "controls_passed"
    return output, factory


def test_all_504_conditions_use_saved_reports_and_exact_qualified_control_cache(runner, tmp_path):
    output, factory = qualify_fixture(runner, tmp_path, empty_first=True, internal_special=True)
    assert factory.calls["actor"] == 0
    assert not any(event == ("load", "actor") for event in factory.events)
    frozen_controls = copy.deepcopy(read_json(output / "reader_cache.json"))
    control_calls = factory.calls["reader"]
    protocol = read_json(output / "protocol.json")
    assert protocol["n_reader_conditions"] == 504
    assert protocol["new_probe_fits"] == 0
    runner.execute(output, stage="reports", backend_factory=factory)
    sender = read_json(output / "sender_rows.json")
    assert len(sender) == 48 and factory.calls["actor"] == 48
    assert sender[0]["report_text"] == ""
    assert sender[0]["generation"]["generated_token_ids"] == [2]
    assert sender[0]["capture"]["status"] == "unavailable_no_content"
    assert "feature_file" not in sender[0]
    assert sender[1]["report_text"] == RAW_REPORT
    assert sender[1]["generation"]["text"] == RAW_REPORT.strip()
    assert sender[1]["generation"]["unexpected_special_tokens"] == [{"index": 2, "id": 1}]
    assert factory.capture_calls == 47
    result = runner.execute(output, stage="readers", backend_factory=factory)
    assert result["status"] == "complete"
    rows = read_json(output / "reader_rows.json")
    assert len(rows) == len({r["example_id"] for r in rows}) == 504
    assert Counter(r["version"] for r in rows) == {
        "original": 144,
        "restored": 144,
        "irrelevant": 144,
        "no_report": 36,
        "full_disclosure": 36,
    }
    assert Counter((r["job"]["objective"], r["job"]["word_budget"]) for r in sender) == {
        (goal, budget): 12 for goal in ("neutral", "favorable") for budget in (25, 50)
    }
    cache = read_json(output / "reader_cache.json")
    assert all(cache[key] == saved for key, saved in frozen_controls.items())
    assert len(cache) == factory.calls["reader"] == len({r["cache_key"] for r in rows})
    assert (
        sum(r["cache_hit"] for r in rows if r["version"] in {"no_report", "full_disclosure"}) == 72
    )
    assert factory.calls["reader"] > control_calls
    sources = {r["job"]["example_id"]: r for r in sender}
    for row in rows:
        if row["version"] == "original":
            assert row["report_text"] == sources[row["source_example_id"]]["report_text"]
            assert row["report_text"] in row["messages"][-1]["content"]
    assert all(backend.closed for backend in factory.instances)
    assert read_json(output / "summary.json")["new_probe_fits"] == 0
    old_calls = factory.calls.copy()
    with pytest.raises(FileExistsError):
        runner.execute(output, stage="controls", resume=True, backend_factory=factory)
    assert factory.calls == old_calls


def test_failed_controls_irrevocably_stop_actor_and_policy_rescue(runner, tmp_path):
    output = prepare(runner, tmp_path)
    factory = RiskFactory(output, controls_pass=False)
    result = runner.execute(output, stage="controls", backend_factory=factory)
    qualification = read_json(output / "qualification.json")
    assert result["status"] == "qualification_failed"
    assert qualification["status"] == "failed"
    assert not qualification["dependent_natural_generation_permitted"]
    assert not qualification["new_probe_fits_permitted"]
    assert qualification["old_qualification_remains_failed"]
    old_calls = factory.calls.copy()
    factory.controls_pass = True
    for stage in ("controls", "reports", "readers"):
        with pytest.raises(FileExistsError):
            runner.execute(output, stage=stage, resume=True, backend_factory=factory)
    assert factory.calls == old_calls and factory.calls["actor"] == 0
    assert not any(b.kind == "actor" for b in factory.instances)


@pytest.mark.parametrize("stage", ["reports", "readers"])
def test_reports_and_replay_cannot_skip_qualification(runner, tmp_path, stage):
    output = prepare(runner, tmp_path)
    factory = RiskFactory(output)
    with pytest.raises(ValueError):
        runner.execute(output, stage=stage, resume=True, backend_factory=factory)
    assert not factory.instances and factory.calls.total() == 0


def test_saved_generation_survives_capture_failure_and_explicit_resume(runner, tmp_path):
    output, factory = qualify_fixture(runner, tmp_path, fail_capture_at=1)
    with pytest.raises(RuntimeError, match="capture fixture failure"):
        runner.execute(output, stage="reports", backend_factory=factory)
    first = copy.deepcopy(read_json(output / "sender_rows.json")[0])
    assert first["capture"]["status"] == "pending"
    assert factory.calls["actor"] == 1
    with pytest.raises(ValueError, match="resume"):
        runner.execute(output, stage="reports", backend_factory=factory)
    result = runner.execute(output, stage="reports", resume=True, backend_factory=factory)
    assert result["status"] == "reports_complete"
    saved = read_json(output / "sender_rows.json")
    assert saved[0]["generation"] == first["generation"]
    assert factory.calls["actor"] == 48 and factory.capture_calls == 49
    assert len(list((output / "features").glob("*.npz"))) == 48


def test_interrupted_control_calls_are_reused_without_new_actor_generation(runner, tmp_path):
    output = prepare(runner, tmp_path)
    factory = RiskFactory(output, fail_reader_at=2)
    with pytest.raises(RuntimeError, match="reader fixture failure"):
        runner.execute(output, stage="controls", backend_factory=factory)
    frozen = copy.deepcopy(read_json(output / "reader_cache.json"))
    assert len(frozen) == 1
    result = runner.execute(output, stage="controls", resume=True, backend_factory=factory)
    assert result["status"] == "controls_passed"
    cache = read_json(output / "reader_cache.json")
    assert all(cache[key] == saved for key, saved in frozen.items())
    assert factory.calls["reader"] == len(cache) + 1
    assert factory.calls["actor"] == 0


def test_passed_controls_cannot_substitute_for_saved_natural_reports(runner, tmp_path):
    output, factory = qualify_fixture(runner, tmp_path)
    old_calls = factory.calls.copy()
    with pytest.raises(ValueError):
        runner.execute(output, stage="readers", backend_factory=factory)
    assert factory.calls == old_calls and factory.calls["actor"] == 0


def test_source_drift_during_generation_preserves_one_attempt_and_cannot_complete(
    runner, tmp_path, monkeypatch
):
    output, factory = qualify_fixture(runner, tmp_path)
    original_generate = RiskBackend.generate

    def drifting_generate(backend, *args, **kwargs):
        result = original_generate(backend, *args, **kwargs)
        if backend.kind == "actor":
            runner.fixture_source.write_text("Source drift after actual output.\n")
        return result

    monkeypatch.setattr(RiskBackend, "generate", drifting_generate)
    with pytest.raises(ValueError, match="source"):
        runner.execute(output, stage="reports", backend_factory=factory)
    manifest = read_json(output / "manifest.json")
    assert manifest["status"] == "failed_source_binding"
    assert manifest["source_unchanged"] is False
    assert factory.calls["actor"] == 1
    assert len(read_json(output / "sender_rows.json")) == 1
    old_calls = factory.calls.copy()
    with pytest.raises(ValueError):
        runner.execute(output, stage="reports", resume=True, backend_factory=factory)
    assert factory.calls == old_calls


@pytest.mark.parametrize("damage", ["source", "protocol", "preflight", "forged_settings_hash"])
def test_prepared_provenance_corruption_stops_before_backend(runner, tmp_path, damage):
    output = prepare(runner, tmp_path)
    factory = RiskFactory(output)
    if damage == "source":
        runner.fixture_source.write_text("Changed source after preparation.\n")
    elif damage == "preflight":
        preflight = read_json(output / "preflight.json")
        next(iter(preflight["control_inputs"].values())).append(123)
        atomic_json(output / "preflight.json", preflight)
    else:
        protocol = read_json(output / "protocol.json")
        protocol["seed"] = 43
        atomic_json(output / "protocol.json", protocol)
        if damage == "forged_settings_hash":
            manifest = read_json(output / "manifest.json")
            manifest["protocol_sha256"] = file_sha256(output / "protocol.json")
            atomic_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError):
        runner.execute(output, stage="controls", backend_factory=factory)
    assert not factory.instances and factory.calls.total() == 0


@pytest.mark.parametrize("journal", ["reader_cache", "control_rows", "qualification"])
@pytest.mark.parametrize("damage", ["delete", "empty"])
def test_qualified_journal_damage_cannot_trigger_policy_retries(runner, tmp_path, journal, damage):
    output, factory = qualify_fixture(runner, tmp_path)
    path = output / f"{journal}.json"
    if damage == "delete":
        path.unlink()
    else:
        atomic_json(path, [] if journal == "control_rows" else {})
    old_calls = factory.calls.copy()
    with pytest.raises(ValueError):
        runner.execute(output, stage="reports", resume=True, backend_factory=factory)
    assert factory.calls == old_calls and factory.calls["actor"] == 0


def test_saved_feature_damage_stops_recipient_replay(runner, tmp_path):
    output, factory = qualify_fixture(runner, tmp_path)
    runner.execute(output, stage="reports", backend_factory=factory)
    first = read_json(output / "sender_rows.json")[0]
    with (output / first["feature_file"]).open("ab") as stream:
        stream.write(b"Changed feature bytes")
    old_calls = factory.calls.copy()
    with pytest.raises(ValueError):
        runner.execute(output, stage="readers", backend_factory=factory)
    assert factory.calls == old_calls


def test_actual_report_overflow_preflights_every_payload_before_new_reader_calls(runner, tmp_path):
    output, factory = qualify_fixture(runner, tmp_path)
    runner.execute(output, stage="reports", backend_factory=factory)

    class ActualOverflowTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            if any(RAW_REPORT in message["content"] for message in messages):
                return [999] * 4097
            return super().apply_chat_template(
                messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
            )

    factory.tokenizer_class = ActualOverflowTokenizer
    old_calls = factory.calls.copy()
    with pytest.raises(ValueError, match="no truncation"):
        runner.execute(output, stage="readers", backend_factory=factory)
    assert factory.calls == old_calls


@pytest.mark.parametrize("damage", ["delete", "modify"])
def test_partial_reader_rows_are_immutable_on_resume(runner, tmp_path, damage):
    output, factory = qualify_fixture(runner, tmp_path)
    runner.execute(output, stage="reports", backend_factory=factory)
    factory.fail_reader_at = factory.calls["reader"] + 2
    with pytest.raises(RuntimeError, match="reader fixture failure"):
        runner.execute(output, stage="readers", backend_factory=factory)
    path = output / "reader_rows.json"
    assert path.is_file()
    if damage == "delete":
        path.unlink()
    else:
        rows = read_json(path)
        rows[0]["decision"] = "INVALID"
        atomic_json(path, rows)
    old_calls = factory.calls.copy()
    with pytest.raises(ValueError):
        runner.execute(output, stage="readers", resume=True, backend_factory=factory)
    assert factory.calls == old_calls


def test_fixture_preparation_cannot_load_a_production_backend(runner, tmp_path):
    output = prepare(runner, tmp_path)
    with pytest.raises(ValueError, match="real model run"):
        runner.execute(output, stage="controls", backend_factory=runner.HFBackend)


def test_qualification_checks_balanced_packets_not_only_minimum_correct_counts(runner):
    rows = []
    for rule in range(3):
        for case in range(4):
            for context in ("informed", "absent", "stale"):
                for version in ("no_report", "full_disclosure"):
                    rows.append(
                        {
                            "example_id": f"{rule}/{case}/{context}/{version}",
                            "case_id": f"{rule}/{case}",
                            "document_family": f"domain_{case}",
                            "rule_family": f"rule_{rule}",
                            "context": context,
                            "version": version,
                            "decision": "DEFER",
                            "normative_control_decision": "DEFER",
                        }
                    )
    assert runner.qualify(rows)["status"] == "passed"
    malformed = copy.deepcopy(rows)
    absent = next(r for r in malformed if r["version"] == "no_report" and r["context"] == "absent")
    absent["context"] = "informed"
    with pytest.raises(ValueError):
        runner.qualify(malformed)


def test_cached_reader_token_hash_damage_stops_without_replacing_calls(runner, tmp_path):
    output, factory = qualify_fixture(runner, tmp_path)
    cache = read_json(output / "reader_cache.json")
    first = next(iter(cache.values()))
    first["generation"]["raw_decoded_text"] = "Wrong unbound output"
    first["generation_sha256"] = canonical_hash(first["generation"])
    atomic_json(output / "reader_cache.json", cache)
    old_calls = factory.calls.copy()
    with pytest.raises(ValueError):
        runner.execute(output, stage="reports", backend_factory=factory)
    assert factory.calls == old_calls


def test_transient_windows_sharing_violation_retries_only_identical_write(
    runner, tmp_path, monkeypatch
):
    target = tmp_path / "transient_journal.json"
    content = {"exact_text": "  Untouched report.\n", "new_probe_fits": 0}
    real_write = runner.storage_atomic_json
    calls, delays = [], []

    def transient_write(path, value):
        calls.append((path, copy.deepcopy(value)))
        if len(calls) < 3:
            error = PermissionError("Explicit transient sharing violation")
            error.winerror = 32
            raise error
        real_write(path, value)

    monkeypatch.setattr(runner, "storage_atomic_json", transient_write)
    monkeypatch.setattr(runner.time, "sleep", delays.append)
    runner.atomic_json(target, content)
    assert calls == [(target, content)] * 3
    assert delays == [0.01, 0.05]
    assert read_json(target) == content


def test_persistent_windows_sharing_violation_exhausts_six_writes_and_propagates(
    runner, tmp_path, monkeypatch
):
    calls, delays = [], []
    failure = PermissionError("Explicit persistent sharing violation")
    failure.winerror = 32

    def blocked_write(path, value):
        calls.append((path, value))
        raise failure

    monkeypatch.setattr(runner, "storage_atomic_json", blocked_write)
    monkeypatch.setattr(runner.time, "sleep", delays.append)
    target = tmp_path / "never_written.json"
    with pytest.raises(PermissionError) as caught:
        runner.atomic_json(target, {"preserve_failure": True})
    assert caught.value is failure and len(calls) == 6
    assert delays == [0.01, 0.05, 0.1, 0.2, 0.4]
    assert not target.exists()


def test_other_permission_errors_propagate_without_sleep_or_write_retry(
    runner, tmp_path, monkeypatch
):
    calls, delays = [], []
    failure = PermissionError("Explicit access denied, not sharing violation")
    failure.winerror = 5

    def denied_write(path, value):
        calls.append((path, value))
        raise failure

    monkeypatch.setattr(runner, "storage_atomic_json", denied_write)
    monkeypatch.setattr(runner.time, "sleep", delays.append)
    with pytest.raises(PermissionError) as caught:
        runner.atomic_json(tmp_path / "denied.json", {"no_retry": True})
    assert caught.value is failure and len(calls) == 1 and delays == []
