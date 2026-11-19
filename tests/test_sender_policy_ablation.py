"""The policy diagnostic removes one sentence and never selects test cases."""

import copy
import importlib.util
from pathlib import Path

import pytest

from latent_probes.backends import GenerationResult
from latent_probes.forecast_panel import load_forecast_panel, sender_jobs
from latent_probes.storage import atomic_json, canonical_hash, read_json

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "sender_policy_ablation", ROOT / "scripts/sender_policy_ablation.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_single_sentence_change_preserves_facts_goal_reader_and_budget():
    panel = load_forecast_panel(ROOT / "data/reader_forecast/development_v1.json")
    source = sender_jobs(panel)[0]["messages"]
    untouched = copy.deepcopy(source)
    changed = module.ablate_sender_sentence(source)
    assert source == untouched and changed[1:] == source[1:]
    assert changed[0]["content"] == source[0]["content"].replace(module.REMOVED_SENTENCE, "", 1)
    assert "Use at most 40 words." in changed[0]["content"]
    assert "Do not assume an unmentioned fact." in changed[0]["content"]
    assert "Write a concise and accurate report" in changed[0]["content"]
    with pytest.raises(ValueError, match="exactly once"):
        module.ablate_sender_sentence(changed)


def test_ablation_uses_only_first_two_training_cases_per_family():
    panel = load_forecast_panel(ROOT / "data/reader_forecast/development_v1.json")
    jobs = module.select_ablation_jobs(panel)
    assert len(jobs) == 24 and len({j["case_id"] for j in jobs}) == 8
    assert {j["split"] for j in jobs} == {"train"}
    assert {j["objective"] for j in jobs} == {"neutral"}
    assert {j["context"] for j in jobs} == {"informed", "absent", "stale"}
    assert all(j["case_id"].endswith(("_01", "_02")) for j in jobs)
    assert len({j["example_id"] for j in jobs}) == 24


def make_source_fixture(directory):
    panel = load_forecast_panel(ROOT / "data/reader_forecast/development_v1.json")
    metadata = {"model_id": "scripted-test-fixture", "is_test_fixture": True}
    atomic_json(directory / "manifest.json", {"status": "complete", "backend": metadata})
    atomic_json(directory / "plan.json", {"panel": panel})
    selected = {job["source_example_id"] for job in module.select_ablation_jobs(panel)}
    for job in sender_jobs(panel):
        if job["example_id"] not in selected:
            continue
        record = {
            **job,
            "generation": GenerationResult(
                text="SCRIPTED OLD REPORT", finish_reason="eos"
            ).record(),
            "is_test_fixture": True,
        }
        record["record_sha256"] = canonical_hash(record)
        atomic_json(directory / "senders" / (job["example_id"] + ".json"), record)
        reader = {**record, "decision": "DEFER"}
        reader["record_sha256"] = canonical_hash(
            {k: v for k, v in reader.items() if k != "record_sha256"}
        )
        atomic_json(directory / "reports" / (job["example_id"] + ".json"), reader)
    return metadata


class ScriptedAblationBackend:
    def __init__(self, metadata, fail_reader=False):
        self.metadata = metadata
        self.calls = []
        self.fail_reader = fail_reader

    def generate(self, messages, settings, seed, capture=False):
        self.calls.append({"messages": messages, "settings": settings, "seed": seed})
        if settings.max_new_tokens == 8:
            if self.fail_reader:
                raise RuntimeError("scripted reader failure")
            return GenerationResult(text="APPROVE", finish_reason="eos")
        return GenerationResult(text="SCRIPTED NEW REPORT", finish_reason="eos")


def test_ablation_api_preserves_source_and_reader_policy(tmp_path):
    source = tmp_path / "source"
    metadata = make_source_fixture(source)
    backend = ScriptedAblationBackend(metadata)
    output = tmp_path / "ablation"
    manifest = module.run_sender_policy_ablation(source, output, backend, progress=None)
    assert manifest["status"] == "complete" and manifest["source_inputs_unchanged"]
    assert len(backend.calls) == 48
    summary = read_json(output / "summary.json")
    assert summary["n_cases"] == 8 and summary["n_report_pairs"] == 24
    for index, call in enumerate(backend.calls):
        system = call["messages"][0]["content"]
        assert (module.REMOVED_SENTENCE in system) == (index % 2 == 1)
    assert summary["is_test_fixture"] is True


def test_failed_reader_retains_sender_record(tmp_path):
    source = tmp_path / "source"
    backend = ScriptedAblationBackend(make_source_fixture(source), fail_reader=True)
    output = tmp_path / "ablation"
    with pytest.raises(RuntimeError, match="scripted reader failure"):
        module.run_sender_policy_ablation(source, output, backend, progress=None)
    assert len(list((output / "senders").glob("*.json"))) == 1
    manifest = read_json(output / "manifest.json")
    assert manifest["status"] == "failed" and manifest["source_inputs_unchanged"]
