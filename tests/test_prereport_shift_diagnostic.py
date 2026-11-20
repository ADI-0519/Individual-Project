"""Saved-array diagnosis without language models or test-guided selection."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from latent_probes.outcome_readout import CLASSES, REQUIRED_COLUMNS, fit_evaluate_readout
from latent_probes.storage import atomic_json, atomic_npz, file_sha256

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_prereport_shift.py"
SPEC = importlib.util.spec_from_file_location("prereport_shift_diagnostic", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture(root, *, rotate_test_labels=False):
    root.mkdir()
    signature = {
        "prediction_time": "before_report",
        "pooling": "last_prompt_token",
        "generated_response_tokens_supplied": 0,
        "is_test_fixture": True,
    }
    rows, vectors = [], []
    for split, repeats in (("train", 2), ("validation", 1), ("test", 1)):
        for repeat in range(repeats):
            for label in range(3):
                case = f"{split}-{repeat}-{label}"
                for objective in ("neutral", "favorable") if split == "test" else ("neutral",):
                    decision = label
                    if split == "validation" and label == 1:
                        decision = 0  # Deliberately absent REJECT validation target.
                    if split == "test" and rotate_test_labels:
                        decision = (label + 1) % 3
                    rows.append(
                        {
                            "example_id": case + "-" + objective,
                            "case_id": case,
                            "family": "family-" + str(label),
                            "objective": objective,
                            "split": split,
                            "decision": CLASSES[decision],
                            "prompt_text": "evidence for " + case,
                            "report_text": "[REPORT NOT YET GENERATED]",
                        }
                    )
                    vector = np.zeros(4)
                    vector[label] = 2.0 + repeat * 0.2
                    vector[3] = 0.25
                    if objective == "favorable":
                        vector[:3] = [0, 50, 0]
                    vectors.append(vector)
    vectors = np.asarray(vectors)
    for name, indices in (
        ("neutral_dataset", [i for i, r in enumerate(rows) if r["objective"] == "neutral"]),
        ("test_dataset", [i for i, r in enumerate(rows) if r["split"] == "test"]),
    ):
        atomic_npz(
            root / name / "features.npz",
            X=vectors[indices],
            example_ids=np.asarray([rows[i]["example_id"] for i in indices]),
            feature_signature=np.asarray(json.dumps(signature)),
            provenance_json=np.asarray(json.dumps({"is_test_fixture": True})),
        )
        with (root / name / "labels.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerows([rows[i] for i in indices])
    fit_evaluate_readout(
        root / "neutral_dataset/features.npz",
        root / "neutral_dataset/labels.csv",
        root / "neutral_readout",
    )
    frozen = {p.name: file_sha256(p) for p in (root / "neutral_readout").iterdir()}
    atomic_json(root / "frozen_readout.json", {"artifacts": frozen})
    hashes = {
        p.relative_to(root).as_posix(): file_sha256(p) for p in root.rglob("*") if p.is_file()
    }
    atomic_json(
        root / "manifest.json",
        {
            "status": "complete",
            "numerical_status": "numerical_inconclusive",
            "output_sha256": hashes,
        },
    )
    return root


def test_original_train_separation_pressure_shift_and_source_immutability(tmp_path):
    source = fixture(tmp_path / "source")
    before = {
        p.relative_to(source).as_posix(): file_sha256(p) for p in source.rglob("*") if p.is_file()
    }
    report = MODULE.diagnose_shift(source, tmp_path / "diagnostic.json", c_grid=(0.01, 1.0))
    assert report["valid_training_rows"] == 6
    assert report["training_feature_width"] == 4
    assert report["original_frozen_C1"]["train/neutral"]["metrics"]["accuracy"] == 1
    pressure = report["original_frozen_C1"]["test/favorable"]
    assert pressure["standardized_shift"]["predicted_class_counts"]["REJECT"] == 3
    assert pressure["metrics"]["accuracy"] == pytest.approx(1 / 3)
    assert report["mean_favorable_minus_neutral_test"]["reject_minus_approve_margin_change"] > 0
    assert report["original_numerical_status"] == "numerical_inconclusive"
    assert report["original_frozen_C1"]["validation/neutral"]["metrics"][
        "missing_evaluation_classes"
    ] == ["REJECT"]
    assert report["posthoc_validation_grid"]["no_test_guided_selection"] is True
    after = {
        p.relative_to(source).as_posix(): file_sha256(p) for p in source.rglob("*") if p.is_file()
    }
    assert before == after


def test_validation_selection_does_not_change_with_test_labels(tmp_path):
    first = fixture(tmp_path / "first")
    second = fixture(tmp_path / "second", rotate_test_labels=True)
    a = MODULE.diagnose_shift(first, tmp_path / "a.json", c_grid=(0.001, 1.0))
    b = MODULE.diagnose_shift(second, tmp_path / "b.json", c_grid=(0.001, 1.0))
    assert (
        a["posthoc_validation_grid"]["selected_settings"]
        == b["posthoc_validation_grid"]["selected_settings"]
    )
    assert a["posthoc_validation_grid"]["candidates"] == b["posthoc_validation_grid"]["candidates"]
    assert (
        a["posthoc_validation_grid"]["selected_test_only"]
        != b["posthoc_validation_grid"]["selected_test_only"]
    )


def test_grid_api_has_no_test_inputs_and_rejects_missing_training_classes():
    train = np.asarray([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=float)
    results, selected, _ = MODULE.fit_validation_grid(
        train, np.asarray([0, 1, 2]), train[[0, 2]], np.asarray([0, 2]), (0.001, 1.0)
    )
    assert len(results) == 6
    assert selected == min(
        range(6), key=lambda i: (results[i]["validation"]["multiclass_log_loss"], i)
    )
    assert all("test" not in result for result in results)
    with pytest.raises(ValueError, match="three training classes"):
        MODULE.fit_validation_grid(
            train[:2], np.asarray([0, 1]), train[[0]], np.asarray([0]), (1.0,)
        )


def test_corruption_and_source_output_overwrite_are_rejected(tmp_path):
    source = fixture(tmp_path / "source")
    with pytest.raises(ValueError, match="outside the original run"):
        MODULE.diagnose_shift(source, source / "bad.json", c_grid=(1.0,))
    with (source / "neutral_readout/activation.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="Input hash mismatch"):
        MODULE.diagnose_shift(source, tmp_path / "bad.json", c_grid=(1.0,))
    assert not (tmp_path / "bad.json").exists()


def test_pre_report_placeholder_cannot_contain_real_report(tmp_path):
    source = fixture(tmp_path / "source")
    path = source / "test_dataset/labels.csv"
    text = path.read_text(encoding="utf-8").replace(
        "[REPORT NOT YET GENERATED]", "leaked future report", 1
    )
    path.write_text(text, encoding="utf-8")
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["output_sha256"]["test_dataset/labels.csv"] = file_sha256(path)
    atomic_json(source / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Report leakage"):
        MODULE.diagnose_shift(source, tmp_path / "bad.json", c_grid=(1.0,))
