"""Data/manifest integrity checks use explicit numerical fixtures, never LLM data."""

import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from latent_probes.apollo import (
    extract_token_examples,
    load_token_examples,
    prepare_pairs,
)
from latent_probes.config import ExperimentConfig, GenerationSettings, ModelSettings
from latent_probes.storage import atomic_json, canonical_hash, read_json


def facts(path, count=8):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["statement", "label"])
        writer.writeheader()
        writer.writerow(
            {"statement": "This deliberately false fixture must never be selected.", "label": 0}
        )
        for i in range(count):
            writer.writerow(
                {
                    "statement": f"Fixture fact {i} contains enough separate words for a five word suffix.",
                    "label": 1,
                }
            )


class TokenFixture:
    """Marked artificial backend injected only through the Python testing API."""

    def __init__(self, settings):
        self.settings = settings
        self.metadata = {
            "model_id": settings.model_id,
            "resolved_revision": "fixture-v1",
            "dtype": settings.dtype,
            "quantization": "none",
            "capture_layers": [0],
            "chat_template_sha256": "fixture",
            "is_test_fixture": True,
        }

    def forced_token_activations(self, messages, response, trim_last_words):
        value = -1.0 if "an honest" in messages[0]["content"] else 1.0
        return {"layer_0_tokens": np.full((3, 4), value)}, {
            "mode": "off_policy_fixed_response",
            "text": response,
            "selected_token_ids": [1, 2, 3],
            "trim_last_words": trim_last_words,
        }

    def close(self):
        pass


def prepared(tmp_path):
    source = tmp_path / "facts.csv"
    output = tmp_path / "pairs.jsonl"
    facts(source)
    prepare_pairs(source, output, limit=8)
    return output


def configuration():
    return ExperimentConfig(
        model=ModelSettings(model_id="fixture", capture_layers=[0]),
        sender=GenerationSettings(do_sample=False),
        seeds=[42],
    )


def test_pairs_use_only_true_rows_and_keep_ordered_facts_together(tmp_path):
    path = prepared(tmp_path)
    rows = load_token_examples(path)
    assert len(rows) == 16
    assert [r["label"] for r in rows] == [0, 1] * 8
    assert [r["split"] for r in rows] == ["train"] * 14 + ["validation"] * 2
    assert not any("deliberately false" in r["response"] for r in rows)
    for a, b in zip(rows[::2], rows[1::2], strict=True):
        assert a["response"] == b["response"]
        assert a["pair_id"] == b["pair_id"]
        assert a["trim_last_words"] == 5
    with pytest.raises(FileExistsError):
        prepare_pairs(tmp_path / "facts.csv", path, limit=8)


def test_custom_csv_does_not_inherit_pinned_upstream_data_identity(tmp_path):
    source = tmp_path / "facts.csv"
    facts(source)
    manifest = prepare_pairs(source, tmp_path / "pairs.jsonl", limit=8)
    assert manifest["source_revision"] is None
    assert manifest["source_verified"] is False


def test_preparation_rejects_short_or_duplicate_facts(tmp_path):
    source = tmp_path / "facts.csv"
    source.write_text("statement,label\nshort fact,1\nshort fact,1\nshort fact,1\n")
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_pairs(source, tmp_path / "out.jsonl", limit=3)
    source.write_text("statement,label\nshort fact,1\nvery short fact,1\nanother short fact,1\n")
    with pytest.raises(ValueError, match="no prefix"):
        prepare_pairs(source, tmp_path / "out.jsonl", limit=3)


def test_short_fact_policy_is_explicit_and_split_precedes_exclusions(tmp_path):
    source = tmp_path / "facts.csv"
    facts(source, count=8)
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[2]["statement"] = "Short excluded training fact."
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["statement", "label"])
        writer.writeheader()
        writer.writerows(rows)
    manifest = prepare_pairs(source, tmp_path / "out.jsonl", limit=512, short_fact_policy="exclude")
    assert manifest["selected_source_pairs"] == 8
    assert manifest["fact_pairs"] == 7
    assert manifest["training_pairs"] == 6
    assert manifest["validation_pairs"] == 1
    assert manifest["excluded_pairs"][0]["split"] == "train"


def test_duplicate_policy_preserves_original_split_and_exposes_exclusion(tmp_path):
    source = tmp_path / "facts.csv"
    facts(source, count=8)
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[7]["statement"] = rows[2]["statement"]
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["statement", "label"])
        writer.writeheader()
        writer.writerows(rows)
    manifest = prepare_pairs(
        source, tmp_path / "out.jsonl", limit=8, duplicate_fact_policy="exclude"
    )
    assert manifest["fact_pairs"] == 7
    assert manifest["validation_pairs"] == 1
    assert manifest["excluded_pairs"][0]["source_index"] == 6
    assert "duplicate" in manifest["excluded_pairs"][0]["reason"]


def test_token_exports_preserve_ids_labels_and_fixture_provenance(tmp_path):
    path = prepared(tmp_path)
    run = tmp_path / "run"
    result = extract_token_examples(configuration(), path, run, backend_factory=TokenFixture)
    assert result == {"output": str(run), "examples": 16, "token_rows": 48, "is_test_fixture": True}
    with np.load(run / "features.npz", allow_pickle=False) as saved:
        assert saved["X"].shape == (48, 4)
        assert len(set(saved["example_ids"])) == 16
        assert json.loads(str(saved["feature_signature"]))["is_test_fixture"]
    with (run / "metadata.csv").open(newline="") as stream:
        metadata = list(csv.DictReader(stream))
    assert {r["label_target"] for r in metadata} == {"instructed_honesty_condition"}
    assert all(r["label_audited"] == "" for r in metadata)
    repeated = extract_token_examples(
        configuration(), path, run, resume=True, backend_factory=TokenFixture
    )
    assert repeated == result


def test_resume_rejects_changed_completed_export(tmp_path):
    path = prepared(tmp_path)
    run = tmp_path / "run"
    extract_token_examples(configuration(), path, run, backend_factory=TokenFixture)
    with (run / "metadata.csv").open("a") as stream:
        stream.write("tampered\n")
    with pytest.raises(ValueError, match="artifacts changed"):
        extract_token_examples(
            configuration(), path, run, resume=True, backend_factory=TokenFixture
        )


def test_prepared_example_edits_cannot_keep_the_original_source_attribution(tmp_path):
    path = prepared(tmp_path)
    rows = load_token_examples(path)
    rows[0]["response"] += " Changed."
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="differ from their provenance"):
        extract_token_examples(
            configuration(), path, tmp_path / "run", backend_factory=TokenFixture
        )


def test_resume_rejects_repointed_cached_feature_even_with_rehashed_record(tmp_path):
    path = prepared(tmp_path)
    run = tmp_path / "run"
    extract_token_examples(configuration(), path, run, backend_factory=TokenFixture)
    manifest = read_json(run / "manifest.json")
    manifest["status"] = "failed"
    atomic_json(run / "manifest.json", manifest)
    records = sorted((run / "records").glob("*.json"))
    record = read_json(records[0])
    record["feature_file"] = read_json(records[1])["feature_file"]
    record["record_sha256"] = canonical_hash(
        {k: v for k, v in record.items() if k != "record_sha256"}
    )
    atomic_json(records[0], record)
    with pytest.raises(ValueError, match="Cached token record"):
        extract_token_examples(
            configuration(), path, run, resume=True, backend_factory=TokenFixture
        )


def test_multiple_capture_layers_are_separate_declared_experiments(tmp_path):
    config = configuration()
    config = replace(config, model=replace(config.model, capture_layers=[0, 1]))
    with pytest.raises(ValueError, match="one declared layer"):
        extract_token_examples(
            config, prepared(tmp_path), tmp_path / "run", backend_factory=TokenFixture
        )


def test_training_labels_cannot_silently_become_behavior_labels(tmp_path):
    path = prepared(tmp_path)
    rows = load_token_examples(path)
    rows[0]["label_target"] = "observed_deception"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="declared fixed instruction pairs"):
        load_token_examples(path)
