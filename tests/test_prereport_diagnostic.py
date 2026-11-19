"""Prefix-only, leakage and lineage checks with an injected CPU backend."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from latent_probes.outcome_readout import CLASSES
from latent_probes.storage import atomic_json, atomic_npz, canonical_hash, file_sha256

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_prereport_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("prereport_diagnostic", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TokenizerStub:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
        }
        return {"input_ids": np.asarray([[4, 5, int(messages[1]["content"])]])}


class BackendStub:
    def __init__(self, *, full_offset=0.0):
        self.metadata = {
            "model_id": "fixture",
            "resolved_revision": "fixture-pin",
            "dtype": "bfloat16",
            "quantization": "none",
            "chat_template_sha256": "fixture-template",
            "attention_implementation": "fixture",
            "is_test_fixture": True,
            "decoder_blocks": 10,
        }
        self.settings = SimpleNamespace(capture_layers=[8], max_input_tokens=100)
        self.model = SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=100, vocab_size=64)
        )
        self.torch = SimpleNamespace(tensor=lambda values, **kwargs: np.asarray(values))
        self.input_device = "cpu"
        self.tokenizer = TokenizerStub()
        self.full_offset = full_offset
        self.calls = []
        self.closed = False

    def _capture_token_rows(self, sequence, positions):
        self.calls.append((sequence.tolist(), list(positions)))
        position = positions[0]
        # Causal state depends on prefix content only, plus optional shape drift.
        value = float(sequence[0, position])
        if sequence.shape[1] > position + 1:
            value += self.full_offset
        return {"layer_8_tokens": np.asarray([[value, 3.0, 1.0]], dtype=np.float32)}

    def close(self):
        self.closed = True


def persist(path, record):
    record = dict(record)
    record["record_sha256"] = canonical_hash(record)
    atomic_json(path, record)
    return record


def source_fixture(root):
    source = root / "source"
    source.mkdir()
    backend = BackendStub()
    jobs = []
    for split, repeats in (("train", 2), ("validation", 1), ("test", 1)):
        for repeat in range(repeats):
            for label, decision in enumerate(CLASSES):
                case = f"family-{label}-{split}-{repeat}"
                for objective in ("neutral", "favorable") if split == "test" else ("neutral",):
                    identifier = case + "-" + objective
                    messages = [
                        {"role": "system", "content": "Report source evidence"},
                        {"role": "user", "content": str(8 + label)},
                    ]
                    job = {
                        "example_id": identifier,
                        "case_id": case,
                        "family": "family-" + str(label),
                        "split": split,
                        "objective": objective,
                        "context": "absent",
                        "messages": messages,
                        "seed": 42,
                    }
                    jobs.append(job)
                    generation = {
                        "prompt_token_ids": [4, 5, 8 + label],
                        "generated_token_ids": [20, 21, 22],
                        "text": "PRIVATE FUTURE REPORT " + decision,
                    }
                    feature = "report_features/" + identifier + ".npz"
                    atomic_npz(source / feature, X=np.asarray([[9999.0, 9999.0, 9999.0]]))
                    base = {
                        **job,
                        "generation": generation,
                        "feature_file": feature,
                        "feature_sha256": file_sha256(source / feature),
                        "is_test_fixture": True,
                    }
                    persist(source / "senders" / (identifier + ".json"), base)
                    persist(
                        source / "reports" / (identifier + ".json"),
                        {**base, "reader_generation": {"text": decision}, "decision": decision},
                    )
    plan = {"sender_jobs": jobs}
    atomic_json(source / "plan.json", plan)
    atomic_json(
        source / "manifest.json",
        {"status": "complete", "plan_sha256": canonical_hash(plan), "backend": backend.metadata},
    )
    atomic_npz(
        source / "neutral_readout" / "after_report_original.npz", weights=np.asarray([123.0])
    )
    atomic_json(
        source / "frozen_readout.json",
        {
            "artifacts": {
                "after_report_original.npz": file_sha256(
                    source / "neutral_readout" / "after_report_original.npz"
                )
            }
        },
    )
    return source, backend


def test_prefix_vector_cannot_receive_response_or_outcome_and_matches_saved_ids():
    backend = BackendStub()
    messages = [{"role": "system", "content": "source"}, {"role": "user", "content": "8"}]
    vector, details = MODULE.prefix_vector(backend, messages, [4, 5, 8])
    np.testing.assert_array_equal(vector, [[8, 3, 1]])
    assert backend.calls == [([[4, 5, 8]], [2])]
    assert details["generated_response_tokens_supplied"] == 0
    assert details["supplied_token_ids_sha256"] == canonical_hash([4, 5, 8])
    with pytest.raises(ValueError, match="differs from saved prompt"):
        MODULE.prefix_vector(backend, messages, [4, 5, 9])
    assert len(backend.calls) == 1  # Rejected before any model forward.


def test_entire_pipeline_uses_new_prefix_weights_and_constant_pre_report_text(tmp_path):
    source, backend = source_fixture(tmp_path)
    before = MODULE._snapshot(source)
    output = tmp_path / "diagnostic"
    result = MODULE.run_prereport_diagnostic(source, output, backend, progress=None)
    assert result["status"] == "complete"
    assert result["n_captured"] == 15
    assert result["numerical_status"] == "passed"
    assert result["source_artifacts_unchanged"] is True
    assert not backend.closed
    assert MODULE._snapshot(source) == before
    # All primary feature forwards happen before any full-sequence diagnostic.
    assert all(len(call[0][0]) == 3 and call[1] == [2] for call in backend.calls[:15])
    captures = json.loads((output / "captures.json").read_text())
    assert all(c["generated_response_tokens_supplied"] == 0 for c in captures)
    with np.load(output / "neutral_dataset" / "features.npz", allow_pickle=False) as saved:
        assert saved["X"].shape == (12, 3)
        assert not np.any(saved["X"] == 9999)  # Never reuse post-report vectors.
        signature = json.loads(saved["feature_signature"].item())
        assert signature["pooling"] == "last_prompt_token"
        assert signature["prediction_time"] == "before_report"
    for directory in ("neutral_dataset", "test_dataset"):
        with (output / directory / "labels.csv").open(newline="", encoding="utf-8") as stream:
            labels = list(csv.DictReader(stream))
        assert all(row["report_text"] == MODULE.REPORT_PLACEHOLDER for row in labels)
        assert all(
            "PRIVATE FUTURE REPORT" not in row["prompt_text"] + row["report_text"] for row in labels
        )
    fit = json.loads((output / "neutral_readout" / "report.json").read_text())
    assert fit["status"] == "completed_development_evaluation"
    assert fit["methods"]["report_only"]["n_training_rows"] == 6
    frozen = json.loads((output / "frozen_transfer" / "report.json").read_text())
    assert frozen["no_refit"] is True
    assert result["analysis_status"] == "exploratory_replay_of_already_observed_outcomes"


def test_failed_shape_parity_is_retained_without_relaxing_bound_or_changing_features(tmp_path):
    source, backend = source_fixture(tmp_path)
    backend.full_offset = 0.25
    output = tmp_path / "diagnostic"
    result = MODULE.run_prereport_diagnostic(source, output, backend, progress=None)
    assert result["status"] == "complete"
    assert result["numerical_status"] == "numerical_inconclusive"
    controls = json.loads((output / "numerical_controls.json").read_text())
    assert all(not c["prefix_vs_full_one_neighbor"]["passed"] for c in controls)
    assert all(
        c["prefix_vs_full_one_neighbor"]["allowed_bf16_neighbor_steps"] == 1 for c in controls
    )
    assert all(c["full_sequence_repeat_exact"] for c in controls)
    assert all(all(m["passed"] for m in c["future_mutations"]) for c in controls)
    with np.load(output / "neutral_dataset" / "features.npz", allow_pickle=False) as saved:
        assert set(saved["X"][:, 0]) == {8.0, 9.0, 10.0}


def test_numerical_control_error_preserves_valid_prefix_features(tmp_path, monkeypatch):
    source, backend = source_fixture(tmp_path)

    def unavailable_control(*args, **kwargs):
        raise RuntimeError("diagnostic control unavailable")

    monkeypatch.setattr(MODULE, "numerical_controls", unavailable_control)
    output = tmp_path / "diagnostic"
    result = MODULE.run_prereport_diagnostic(source, output, backend, progress=None)
    assert result["status"] == "complete"
    assert result["numerical_status"] == "numerical_inconclusive"
    controls = json.loads((output / "numerical_controls.json").read_text())
    assert all(c["error"] == "diagnostic control unavailable" for c in controls)
    assert (output / "frozen_transfer" / "report.json").exists()


def test_selection_is_metadata_only_and_preserves_case_clusters():
    rows = [
        {
            "example_id": f"{family}-{case}-{context}",
            "family": family,
            "case_id": f"case-{case}",
            "objective": objective,
            "context": context,
            "decision": "INVALID",
        }
        for family in ("a", "b", "c", "d")
        for case in range(1, 7)
        for context in ("stale", "absent", "informed")
        for objective in ("neutral", "favorable")
    ]
    selected = MODULE.select_numerical_examples(rows)
    assert len(selected) == 8
    assert {row["case_id"] for row in selected} == {"case-1", "case-6"}
    assert all(row["objective"] == "neutral" and row["context"] == "absent" for row in selected)
    assert [r["example_id"] for r in selected] == [
        r["example_id"] for r in MODULE.select_numerical_examples(list(reversed(rows)))
    ]


def test_bf16_neighbor_bound_is_not_relative_epsilon():
    reference = np.asarray([1.0, -1.0, 0.0], dtype=np.float32)
    assert MODULE.bf16_neighbor_check(reference, [1.0078125, -1.0078125, 0.0])["passed"]
    assert not MODULE.bf16_neighbor_check(reference, [1.015625, -1.0, 0.0])["passed"]
    with pytest.raises(ValueError, match="BF16 grid"):
        MODULE.bf16_neighbor_check(np.asarray([1.001]), np.asarray([1.001]))


@pytest.mark.parametrize("corruption", ["record", "feature", "old_weights", "plan_bound_prompt"])
def test_source_tampering_is_rejected_before_forward_or_output(tmp_path, corruption):
    source, backend = source_fixture(tmp_path)
    record_path = next((source / "reports").glob("*.json"))
    if corruption == "record":
        row = json.loads(record_path.read_text())
        row["decision"] = "APPROVE" if row["decision"] != "APPROVE" else "REJECT"
        atomic_json(record_path, row)
        pattern = "record hash mismatch"
    elif corruption == "feature":
        with next((source / "report_features").glob("*.npz")).open("ab") as stream:
            stream.write(b"corrupt")
        pattern = "feature artifact hash mismatch"
    elif corruption == "old_weights":
        with (source / "neutral_readout" / "after_report_original.npz").open("ab") as stream:
            stream.write(b"corrupt")
        pattern = "Original frozen artifact hash mismatch"
    else:
        row = json.loads(record_path.read_text())
        row.pop("record_sha256")
        row["messages"][0]["content"] += " hidden observed outcome"
        persist(record_path, row)
        pattern = "Source job/record mismatch"
    with pytest.raises(ValueError, match=pattern):
        MODULE.run_prereport_diagnostic(source, tmp_path / "diagnostic", backend, progress=None)
    assert not backend.calls
    assert not (tmp_path / "diagnostic").exists()


def test_invalid_target_preserved_and_excluded_without_regeneration(tmp_path):
    source, backend = source_fixture(tmp_path)
    row_path = next((source / "reports").glob("*train-0-neutral.json"))
    row = json.loads(row_path.read_text())
    row.pop("record_sha256")
    row["decision"] = "INVALID"
    row["reader_generation"]["text"] = "unclear result"
    persist(row_path, row)
    output = tmp_path / "diagnostic"
    MODULE.run_prereport_diagnostic(source, output, backend, progress=None)
    with (output / "neutral_readout" / "exclusions.csv").open(newline="") as stream:
        ledger = list(csv.DictReader(stream))
    assert len(ledger) == 1 and ledger[0]["decision"] == "INVALID"


def test_mid_run_source_change_fails_and_keeps_partial_features(tmp_path):
    source, backend = source_fixture(tmp_path)

    def change_source(features_path, labels_path, output_dir, **kwargs):
        output_dir.mkdir()
        atomic_json(output_dir / "report.json", {"status": "not_trainable"})
        atomic_json(source / "changed_after_capture.json", {"changed": True})
        return {"status": "not_trainable"}

    output = tmp_path / "diagnostic"
    with pytest.raises(ValueError, match="Original source run changed"):
        MODULE.run_prereport_diagnostic(
            source, output, backend, fit_function=change_source, progress=None
        )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["n_captured"] == 15
    assert (output / "neutral_dataset" / "features.npz").exists()
    assert not backend.closed


def test_output_cannot_modify_original_run_and_mismatched_backend_rejected(tmp_path):
    source, backend = source_fixture(tmp_path)
    with pytest.raises(ValueError, match="separate from source"):
        MODULE.run_prereport_diagnostic(source, source / "new", backend, progress=None)
    backend.metadata["resolved_revision"] = "different"
    with pytest.raises(ValueError, match="coordinate mismatch"):
        MODULE.run_prereport_diagnostic(source, tmp_path / "out", backend, progress=None)
    assert not backend.calls
