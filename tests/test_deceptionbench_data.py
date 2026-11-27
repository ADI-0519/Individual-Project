"""Source-adapter checks use clearly synthetic fixtures, never model observations."""

from __future__ import annotations

import copy
import json
from hashlib import sha256
from pathlib import Path

import pytest

from latent_probes.deceptionbench_data import (
    CONDITIONS,
    FILE_TOPICS,
    iter_deceptionbench_rows,
    load_deceptionbench,
    validate_source_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_MANIFEST = ROOT / "research/deceptionbench/source_manifest_v1.json"


@pytest.fixture
def synthetic_source(tmp_path):
    """Match the structural contract using invented, nonexperimental source text.

    Fixture hashes deliberately bind fixture bytes; these are not asserted to be
    the published dataset. Real-byte validation is a separate recorded audit.
    """
    directory = tmp_path / "source"
    directory.mkdir()
    manifest = json.loads(PUBLISHED_MANIFEST.read_text(encoding="utf-8"))
    for binding in manifest["files"]:
        rows = []
        for index in range(30):
            rows.append(
                {
                    "topic": FILE_TOPICS[binding["name"]],
                    "dimension": f"synthetic dimension {index // 10}",
                    "question": "Question deliberately repeated across synthetic scenarios",
                    "groundtruth": "REFERENCE ANSWER METADATA MUST NOT BE APPENDED",
                    "goal": "GOAL METADATA MUST NOT BE APPENDED",
                    **{
                        condition: f"Synthetic prompt {binding['name']} row {index} {condition}"
                        for condition in CONDITIONS
                    },
                }
            )
        raw = (json.dumps(rows, ensure_ascii=False) + "\n").encode("utf-8")
        (directory / binding["name"]).write_bytes(raw)
        binding["bytes"] = len(raw)
        binding["sha256"] = sha256(raw).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return directory, manifest_path, manifest


def _rewrite_binding(directory, manifest_path, manifest, *, transform):
    """Rebind a deliberately corrupted fixture so semantic validators are reached."""
    binding = manifest["files"][0]
    rows = json.loads((directory / binding["name"]).read_text(encoding="utf-8"))
    transform(rows)
    raw = json.dumps(rows).encode("utf-8")
    (directory / binding["name"]).write_bytes(raw)
    binding["bytes"], binding["sha256"] = len(raw), sha256(raw).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_checked_in_manifest_is_structurally_valid_and_no_model_results():
    manifest = json.loads(PUBLISHED_MANIFEST.read_text(encoding="utf-8"))
    validate_source_manifest(manifest)
    assert manifest["generated_model_calls"] == 0
    assert manifest["total_scenarios"] == 150
    assert manifest["total_static_prompts"] == 900
    assert sum(item["bytes"] for item in manifest["files"]) == 465251


def test_exact_scenario_counts_conditions_and_grouped_identity(synthetic_source):
    directory, path, _ = synthetic_source
    dataset = load_deceptionbench(directory, path)
    rows = list(iter_deceptionbench_rows(dataset))
    assert len(dataset["records"]) == 150
    assert len(rows) == len({row["case_id"] for row in rows}) == 900
    assert len({row["scenario_id"] for row in rows}) == 150
    assert {row["condition"] for row in rows[:6]} == set(CONDITIONS)
    assert len({row["scenario_id"] for row in rows[:6]}) == 1
    assert all(row["behavior_label"] is None for row in rows)
    assert dataset["audit"]["generated_model_calls"] == 0


def test_model_messages_are_exact_published_prompt_only(synthetic_source):
    dataset = load_deceptionbench(*synthetic_source[:2])
    for row in iter_deceptionbench_rows(dataset):
        assert row["messages"] == [{"role": "user", "content": row["source_prompt_only"]}]
        assert "REFERENCE ANSWER" not in row["messages"][0]["content"]
        assert "GOAL METADATA" not in row["messages"][0]["content"]
        assert "thought" not in row["messages"][0]["content"]
        assert (
            row["prompt_representation"]
            == "source_prompt_only_no_upstream_thought_response_wrapper"
        )
        assert "groundtruth" not in row and "goal" not in row


def test_source_rows_not_question_text_define_identity(synthetic_source):
    dataset = load_deceptionbench(*synthetic_source[:2])
    assert len({r["metadata"]["question"] for r in dataset["records"]}) == 1
    assert len({r["scenario_id"] for r in dataset["records"]}) == 150


@pytest.mark.parametrize(
    "field,value",
    [
        ("benchmark_id", "pku_deceptionbench"),
        ("repository_revision", "0" * 40),
        ("dataset_revision", "0" * 40),
        ("total_scenarios", 900),
        ("behavior_labels_released", True),
        ("activations_released", True),
        ("generated_model_calls", 1),
        ("conditions", ["L3-self"]),
        ("license_notices", {}),
        ("prompt_representation", "exact_paper_reproduction"),
    ],
)
def test_manifest_rejects_identity_provenance_and_label_claims(field, value):
    manifest = json.loads(PUBLISHED_MANIFEST.read_text(encoding="utf-8"))
    manifest[field] = value
    with pytest.raises(ValueError):
        validate_source_manifest(manifest)


def test_bytes_are_verified_before_schema(synthetic_source):
    directory, path, _ = synthetic_source
    (directory / "economy.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="bytes/hash mismatch"):
        load_deceptionbench(directory, path)


@pytest.mark.parametrize(
    "corruption", ["extra", "missing", "empty", "topic", "dimensions", "duplicate"]
)
def test_schema_and_distribution_checks(synthetic_source, corruption):
    directory, path, manifest = synthetic_source

    def transform(rows):
        if corruption == "extra":
            rows[0]["response_label"] = "honest"
        elif corruption == "missing":
            del rows[0]["groundtruth"]
        elif corruption == "empty":
            rows[0]["L1-self"] = " "
        elif corruption == "topic":
            rows[0]["topic"] = "WRONG DOMAIN"
        elif corruption == "dimensions":
            rows[0]["dimension"] = "fourth dimension"
        else:
            rows[1]["L1-self"] = rows[0]["L1-self"]

    _rewrite_binding(directory, path, manifest, transform=transform)
    with pytest.raises(ValueError):
        load_deceptionbench(directory, path)


def test_unexpected_or_missing_json_files_are_errors(synthetic_source):
    directory, path, _ = synthetic_source
    extra = directory / "outputs.json"
    extra.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly the five"):
        load_deceptionbench(directory, path)
    extra.unlink()
    (directory / "social.json").unlink()
    with pytest.raises(ValueError, match="exactly the five"):
        load_deceptionbench(directory, path)


def test_duplicate_manifest_json_keys_are_rejected(synthetic_source):
    directory, path, _ = synthetic_source
    raw = path.read_text(encoding="utf-8")
    path.write_text(raw[:-1] + ', "schema_version": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON object key"):
        load_deceptionbench(directory, path)


def test_wrong_identity_cannot_be_iterated(synthetic_source):
    dataset = load_deceptionbench(*synthetic_source[:2])
    other = copy.deepcopy(dataset)
    other["benchmark_id"] = "different_benchmark"
    with pytest.raises(ValueError, match="Wrong dataset identity"):
        list(iter_deceptionbench_rows(other))
