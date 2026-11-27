"""The independent auditor checks raw values as well as recorded file hashes."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import test_main_study_run as capture_support

from latent_probes import main_study_run as run
from latent_probes.crossed_belief_validation import summarize_logits
from latent_probes.storage import atomic_json, file_sha256, read_json

capture = capture_support.capture

spec = importlib.util.spec_from_file_location(
    "independent_capture_auditor",
    Path(__file__).resolve().parents[1] / "scripts/audit_main_study_capture.py",
)
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


def execute_fixture(capture):
    run.execute(
        capture.output,
        partition="train_dev",
        ledger_path=capture.ledger,
        freeze_path=capture.freeze,
        backend_factory=capture.backend,
        progress=None,
    )


def rewrite_record(capture, transform):
    """Rebind both exports so corruption tests exercise reconstruction, not stale hashes."""
    payload_path = capture.output / "rows.json"
    payload = read_json(payload_path)
    row = payload["rows"][0]
    record_path = capture.output / "row_journal" / row["row_id"] / "complete.json"
    record = read_json(record_path)
    transform(record)
    atomic_json(record_path, record)
    payload["rows"][0] = {
        **record,
        "output_yes_no_logitdiff": record["output_logits"]["yes_minus_no"],
    }
    atomic_json(payload_path, payload)
    rebind_export(capture)


def rebind_export(capture):
    path = capture.output / "capture_manifest.json"
    manifest = read_json(path)
    manifest["output_sha256"]["rows.json"] = file_sha256(capture.output / "rows.json")
    atomic_json(path, manifest)


def test_independent_audit_reads_every_raw_forward(capture):
    execute_fixture(capture)
    result = auditor.audit_capture(capture.output)
    assert result["rows_verified"] == 3
    assert result["raw_export_array_identity"] is True
    assert result["full_vocabulary_summary_reconstructed"] is True
    assert result["test_fixture"] is True


def test_rehashed_export_cannot_hide_wrong_feature_values(capture):
    execute_fixture(capture)
    path = capture.output / "features.npz"
    with np.load(path, allow_pickle=False) as data:
        values, ids = data["features"].copy(), data["row_ids"].copy()
    values[0, 0] += 1
    np.savez_compressed(path, features=values, row_ids=ids)
    manifest_path = capture.output / "capture_manifest.json"
    manifest = read_json(manifest_path)
    manifest["output_sha256"]["features.npz"] = file_sha256(path)
    atomic_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="Exported feature differs"):
        auditor.audit_capture(capture.output)


def test_raw_journal_modification_is_detected(capture):
    execute_fixture(capture)
    row = read_json(capture.output / "rows.json")["rows"][0]
    path = capture.output / "row_journal" / row["row_id"] / "raw_forward.npz"
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Raw arrays changed"):
        auditor.audit_capture(capture.output)


@pytest.mark.parametrize(
    "field",
    [
        "yes_logit",
        "no_logit",
        "yes_logprob",
        "no_logprob",
        "p_yes_given_yes_no",
        "full_vocab_logsumexp",
        "vocab_size",
        "argmax_token_id",
        "logits_values_sha256",
    ],
)
def test_rehashed_wrong_output_summary_is_rejected(capture, field):
    execute_fixture(capture)

    def corrupt(record):
        summary = record["output_logits"]
        summary[field] = "wrong" if field.endswith("sha256") else summary[field] + 1

    rewrite_record(capture, corrupt)
    with pytest.raises(ValueError, match="Output summary differs"):
        auditor.audit_capture(capture.output)


def test_margin_subtracts_after_float64_promotion(capture):
    execute_fixture(capture)
    row = read_json(capture.output / "rows.json")["rows"][0]
    item = capture.output / "row_journal" / row["row_id"]
    with np.load(item / "arrays.npz", allow_pickle=False) as saved:
        vector, logits = saved["feature"].copy(), saved["next_token_logits"].copy()
    tokens = capture.protocol["tokenizer"]["answer_token_ids"]
    logits[tokens["YES"]], logits[tokens["NO"]] = 1.0, 2**-25
    summary = summarize_logits(logits, tokens)
    assert summary["yes_minus_no"] != float(logits[tokens["YES"]] - logits[tokens["NO"]])
    np.savez_compressed(item / "arrays.npz", feature=vector, next_token_logits=logits)
    np.savez_compressed(
        item / "raw_forward.npz", feature=vector, full_forward_logits=logits.reshape(1, 1, -1)
    )

    def update(record):
        record["arrays_sha256"] = file_sha256(item / "arrays.npz")
        record["raw_forward_sha256"] = file_sha256(item / "raw_forward.npz")
        record["output_logits"] = summary

    rewrite_record(capture, update)
    assert auditor.audit_capture(capture.output)["rows_verified"] == 3


def test_numerically_equal_float64_raw_arrays_are_rejected(capture):
    execute_fixture(capture)
    row = read_json(capture.output / "rows.json")["rows"][0]
    item = capture.output / "row_journal" / row["row_id"]
    path = item / "raw_forward.npz"
    with np.load(path, allow_pickle=False) as raw:
        arrays = {name: raw[name].astype(np.float64) for name in raw.files}
    np.savez_compressed(path, **arrays)
    rewrite_record(capture, lambda record: record.update(raw_forward_sha256=file_sha256(path)))
    with pytest.raises(ValueError, match="Wrong raw dtype"):
        auditor.audit_capture(capture.output)


@pytest.mark.parametrize("key", ["bank_sha256", "capture_layer", "partition", "model_settings"])
def test_consistent_exports_must_still_match_protocol_provenance(capture, key):
    execute_fixture(capture)
    payload_path = capture.output / "rows.json"
    manifest_path = capture.output / "capture_manifest.json"
    payload, manifest = read_json(payload_path), read_json(manifest_path)
    payload["provenance"][key] = "wrong"
    manifest["provenance"][key] = "wrong"
    atomic_json(payload_path, payload)
    atomic_json(manifest_path, manifest)
    rebind_export(capture)
    with pytest.raises(ValueError, match="[Pp]rovenance.*differs"):
        auditor.audit_capture(capture.output)


def test_export_cannot_add_unjournalled_fields(capture):
    execute_fixture(capture)
    path = capture.output / "rows.json"
    payload = read_json(path)
    payload["rows"][0]["unrecorded_measurement"] = 1
    atomic_json(path, payload)
    rebind_export(capture)
    with pytest.raises(ValueError, match="Export differs"):
        auditor.audit_capture(capture.output)


def test_backend_model_claim_must_match_protocol(capture):
    execute_fixture(capture)
    path = capture.output / "capture_manifest.json"
    manifest = read_json(path)
    manifest["backend"]["resolved_revision"] = "different-model-revision"
    atomic_json(path, manifest)
    with pytest.raises(ValueError, match="Backend identity differs"):
        auditor.audit_capture(capture.output)


def test_source_bindings_must_match_recorded_authorization(capture):
    execute_fixture(capture)
    manifest_path, payload_path = (
        capture.output / "capture_manifest.json",
        capture.output / "rows.json",
    )
    manifest, payload = read_json(manifest_path), read_json(payload_path)
    manifest["provenance"]["source_bindings"] = {}
    payload["provenance"]["source_bindings"] = {}
    atomic_json(manifest_path, manifest)
    atomic_json(payload_path, payload)
    rebind_export(capture)
    with pytest.raises(ValueError, match="Provenance source bindings differ"):
        auditor.audit_capture(capture.output)


def test_standalone_auditor_imports_no_project_modules_or_runs_models(capture, tmp_path):
    execute_fixture(capture)
    before = capture_support.Model.calls
    output = tmp_path / "independent_audit.json"
    result = subprocess.run(
        [sys.executable, spec.origin, "--capture", str(capture.output), "--output", str(output)],
        check=True,
        text=True,
        capture_output=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(result.stdout) == {"rows_verified": 3}
    assert report["project_modules_imported"] is False
    assert report["new_model_calls"] == 0
    assert capture_support.Model.calls == before
    assert "internal hash consistency" in " ".join(report["limitations"])
