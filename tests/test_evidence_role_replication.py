"""Model-free discovery binding and replication stop-condition checks."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from latent_probes.calibrated_reader import policy_metadata
from latent_probes.evidence_role_panel import authored_panel, jobs
from latent_probes.storage import atomic_json, file_sha256, read_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "role_replication_fixture", ROOT / "scripts/replicate_evidence_role.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def discovery(tmp_path, runner):
    """All synthetic records are software fixtures, never research evidence."""
    source = tmp_path / "explicit_fixture_discovery"
    planned = jobs(authored_panel())
    rows = [
        {
            **job,
            "generation": {
                "text": job["expected_decision"],
                "finish_reason": "eos",
                "prompt_token_ids": [1],
                "generated_token_ids": [2],
            },
            "decision": job["expected_decision"],
        }
        for job in planned
    ]
    dependency = tmp_path / "fixture_frozen_source.txt"
    dependency.write_text("Frozen fixture input\n")
    protocol = {
        "jobs": planned,
        "policy": policy_metadata(),
        "source_sha256": {str(dependency): file_sha256(dependency)},
        "is_test_fixture": True,
    }
    atomic_json(source / "protocol.json", protocol)
    atomic_json(source / "rows.json", rows)
    atomic_json(source / "summary.json", runner.summarise(rows))
    for row in rows:
        atomic_json(source / "records" / f"{row['example_id']}.json", row)
    atomic_json(
        source / "manifest.json",
        {
            "status": "complete",
            "source_unchanged": True,
            "protocol_sha256": file_sha256(source / "protocol.json"),
            "rows_sha256": file_sha256(source / "rows.json"),
            "summary_sha256": file_sha256(source / "summary.json"),
            "backend": {"model_id": "Qwen/Qwen2.5-7B-Instruct", "is_test_fixture": False},
            "fixture_record_notice": "Synthetic software-test fixture, not real inference evidence.",
        },
    )
    return source, dependency


def test_all96_ordered_frozen_messages_are_reused_without_output_selection(runner, discovery):
    source, _ = discovery
    planned, provenance = runner.prepare(source)
    assert len(planned) == 96
    assert planned == read_json(source / "protocol.json")["jobs"]
    assert provenance["reused_text_policy_sha256"] == policy_metadata()["policy_sha256"]
    planned[0]["messages"][0]["content"] = "changed detached result"
    assert runner.prepare(source)[0][0]["messages"][0]["content"] != "changed detached result"


@pytest.mark.parametrize(
    "corruption", ["incomplete", "dependency", "rows", "call", "prefix", "order"]
)
def test_source_or_frozen_prefix_changes_stop_replication(runner, discovery, corruption):
    source, dependency = discovery
    if corruption == "incomplete":
        manifest = read_json(source / "manifest.json")
        manifest["status"] = "running"
        atomic_json(source / "manifest.json", manifest)
    elif corruption == "dependency":
        dependency.write_text("Changed frozen fixture input\n")
    elif corruption == "rows":
        rows = read_json(source / "rows.json")
        rows.pop()
        atomic_json(source / "rows.json", rows)
    elif corruption == "call":
        row = read_json(source / "rows.json")[0]
        row["decision"] = "INVALID"
        atomic_json(source / "records" / f"{row['example_id']}.json", row)
    else:
        protocol = read_json(source / "protocol.json")
        if corruption == "prefix":
            protocol["jobs"][0]["messages"][0]["content"] = "Changed system"
        else:
            protocol["jobs"][0], protocol["jobs"][1] = protocol["jobs"][1], protocol["jobs"][0]
        atomic_json(source / "protocol.json", protocol)
        manifest = read_json(source / "manifest.json")
        manifest["protocol_sha256"] = file_sha256(source / "protocol.json")
        atomic_json(source / "manifest.json", manifest)
    with pytest.raises(ValueError):
        runner.prepare(source)


def test_failed_source_and_nested_output_prevent_any_backend_call(runner, discovery):
    source, _ = discovery

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid source/output must prevent all backend loading")

    with pytest.raises(ValueError, match="separate"):
        runner.run(
            SimpleNamespace(source=source, output=source / "nested"), backend_factory=forbidden
        )
    manifest = read_json(source / "manifest.json")
    manifest["status"] = "failed"
    atomic_json(source / "manifest.json", manifest)
    output = source.parent / "never_created"
    with pytest.raises(ValueError, match="completed"):
        runner.run(SimpleNamespace(source=source, output=output), backend_factory=forbidden)
    assert not output.exists()


def test_only_fixed_independent_checkpoint_precision_and_budget_are_exposed(runner):
    assert runner.SETTINGS.model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert runner.SETTINGS.revision == "0e9e39f249a16976918f6564b8830bc894c89659"
    assert runner.SETTINGS.quantization == "4bit" and runner.SETTINGS.dtype == "bfloat16"
    assert runner.SETTINGS.local_files_only and runner.SETTINGS.device == "cuda"
    assert runner.SETTINGS.max_input_tokens == 2048
    assert runner.GENERATION.max_new_tokens == 8 and not runner.GENERATION.do_sample
    assert runner.SEED == 42
