"""Small offline, grouped fixtures for the neutral-trained outcome diagnostic."""

from __future__ import annotations

import csv
import hashlib
import json

import numpy as np
import pytest

from latent_probes.outcome_readout import (
    CLASSES,
    METHODS,
    REQUIRED_COLUMNS,
    decision_metrics,
    evaluate_frozen_readout,
    fit_evaluate_readout,
    main,
)


def write_inputs(root, rows=None, X=None, *, suffix="", signature=None):
    root.mkdir(parents=True, exist_ok=True)
    if rows is None:
        rows, vectors = [], []
        for split, repeats in (("train", 2), ("validation", 1), ("test", 1)):
            for repeat in range(repeats):
                for label, name in enumerate(CLASSES):
                    case = f"{split}-{repeat}-{label}"
                    objectives = ("neutral", "favorable") if split == "test" else ("neutral",)
                    for objective in objectives:
                        row = {
                            "example_id": case + "-" + objective,
                            "case_id": case,
                            "family": "family-" + str(label),
                            "objective": objective,
                            "split": split,
                            "decision": name,
                            "prompt_text": f"decision request {name.lower()} case {case}",
                            "report_text": f"report evidence {name.lower()} revision {repeat}",
                        }
                        rows.append(row)
                        vector = np.zeros(4)
                        vector[label] = 3.0 + repeat
                        vector[3] = 7.0  # Constant train column must get scale one.
                        if split != "train":
                            vector[:3] *= 1000.0  # Detect accidentally pooled normalization.
                        vectors.append(vector)
        X = np.asarray(vectors)
    features, labels = root / f"features{suffix}.npz", root / f"labels{suffix}.csv"
    np.savez_compressed(
        features,
        X=np.asarray(X, dtype=np.float64),
        example_ids=np.asarray([row["example_id"] for row in rows]),
        provenance_json=np.asarray(json.dumps({"is_test_fixture": True, "capture": "synthetic"})),
        feature_signature=np.asarray(
            json.dumps(
                signature or {"is_test_fixture": True, "position": "response_last", "width": 4}
            )
        ),
    )
    with labels.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return features, labels, rows, np.asarray(X)


def read_predictions(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_train_only_normalization_and_proper_multiclass_artifacts(tmp_path):
    features, labels, rows, X = write_inputs(tmp_path)
    before = file_hash(features), file_hash(labels)
    output = tmp_path / "readout"
    report = fit_evaluate_readout(features, labels, output)
    assert report["status"] == "completed_development_evaluation"
    assert report["learning_classes"] == list(CLASSES)
    assert not report["unsupported_classes"]
    assert report["families_shared_across_partitions"] == ["family-0", "family-1", "family-2"]
    train = [i for i, row in enumerate(rows) if row["split"] == "train"]
    with np.load(output / "activation.npz", allow_pickle=False) as saved:
        np.testing.assert_allclose(saved["mean"], X[train].mean(axis=0), rtol=0, atol=0)
        expected = X[train].std(axis=0)
        expected[expected == 0] = 1
        np.testing.assert_allclose(saved["scale"], expected, rtol=0, atol=0)
        assert not np.allclose(saved["mean"], X.mean(axis=0))
        assert saved["coefficients"].shape == (3, 4)
    assert (file_hash(features), file_hash(labels)) == before
    predictions = read_predictions(output / "predictions.csv")
    assert len(predictions) == 9 * len(METHODS)
    for row in predictions:
        probabilities = np.asarray([float(row[f"p_{name}"]) for name in CLASSES])
        assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
        assert probabilities.sum() == pytest.approx(1.0, abs=1e-12)
    for method in METHODS:
        with np.load(output / (method + ".npz"), allow_pickle=False) as saved:
            for key in saved.files:
                assert saved[key].dtype.kind != "O"
            if "vocabulary_terms" in saved.files:
                assert not any(
                    "validation" in word or "favorable" in word
                    for word in saved["vocabulary_terms"]
                )
        result = report["methods"][method]
        assert result["n_training_rows"] == 6
        assert result["evaluations"]["test/favorable"]["n_valid_rows"] == 3


@pytest.mark.parametrize("split", ["train", "validation"])
def test_pressure_rows_cannot_enter_fitting_or_validation(tmp_path, split):
    _, _, rows, X = write_inputs(tmp_path)
    next(row for row in rows if row["split"] == split)["objective"] = "favorable"
    features, labels, _, _ = write_inputs(tmp_path, rows, X)
    with pytest.raises(ValueError, match="Only neutral"):
        fit_evaluate_readout(features, labels, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_case_leakage_rejected_even_when_leaking_row_invalid(tmp_path):
    _, _, rows, X = write_inputs(tmp_path)
    first_train = next(row for row in rows if row["split"] == "train")
    first_test = next(row for row in rows if row["split"] == "test")
    first_test["case_id"] = first_train["case_id"]
    first_test["decision"] = "INVALID"
    features, labels, _, _ = write_inputs(tmp_path, rows, X)
    with pytest.raises(ValueError, match="Case leakage"):
        fit_evaluate_readout(features, labels, tmp_path / "out")


def test_missing_training_class_not_trainable_and_invalid_ledger(tmp_path):
    _, _, rows, X = write_inputs(tmp_path)
    for row in rows:
        if row["split"] == "train" and row["decision"] == "DEFER":
            row["decision"] = "INVALID"
    next(row for row in rows if row["split"] == "test")["decision"] = "INVALID"
    features, labels, _, _ = write_inputs(tmp_path, rows, X)
    output = tmp_path / "out"
    report = fit_evaluate_readout(features, labels, output)
    assert report["status"] == "not_trainable"
    assert report["reason"] == "requires_all_three_training_classes"
    assert report["missing_training_classes"] == ["DEFER"]
    assert report["counts"]["train/neutral"]["n_invalid_excluded"] == 2
    assert report["counts"]["train/neutral"]["class_counts"]["DEFER"] == 0
    assert not list(output.glob("*.npz"))
    exclusions = read_predictions(output / "exclusions.csv")
    assert len(exclusions) == 3
    assert all(row["decision"] == "INVALID" for row in exclusions)
    assert not read_predictions(output / "predictions.csv")
    assert (output / "manifest.json").exists()


def test_optional_two_class_diagnostic_explicitly_keeps_unsupported_defer(tmp_path):
    _, _, rows, X = write_inputs(tmp_path)
    for row in rows:
        if row["split"] == "train" and row["decision"] == "DEFER":
            row["decision"] = "INVALID"
    features, labels, _, _ = write_inputs(tmp_path, rows, X)
    output = tmp_path / "out"
    report = fit_evaluate_readout(features, labels, output, allow_two_class_diagnostic=True)
    assert report["status"] == "diagnostic_only_missing_training_class"
    assert report["primary_three_class_status"] == "not_trainable"
    assert report["unsupported_classes"] == ["DEFER"]
    assert report["learning_classes"] == ["APPROVE", "REJECT"]
    for row in read_predictions(output / "predictions.csv"):
        assert float(row["p_DEFER"]) == 0.0
        assert float(row["p_APPROVE"]) + float(row["p_REJECT"]) == pytest.approx(1)
    assert (
        report["methods"]["activation"]["evaluations"]["test/neutral"]["multiclass_log_loss"] > 10
    )


def test_one_training_class_never_fits_even_when_diagnostic_enabled(tmp_path):
    _, _, rows, X = write_inputs(tmp_path)
    for row in rows:
        if row["split"] == "train":
            row["decision"] = "APPROVE"
    features, labels, _, _ = write_inputs(tmp_path, rows, X)
    report = fit_evaluate_readout(
        features, labels, tmp_path / "out", allow_two_class_diagnostic=True
    )
    assert report["status"] == "not_trainable"
    assert report["missing_training_classes"] == ["REJECT", "DEFER"]


def test_test_labels_never_change_learned_parameters(tmp_path):
    features, labels, rows, X = write_inputs(tmp_path)
    output_a, output_b = tmp_path / "a", tmp_path / "b"
    fit_evaluate_readout(features, labels, output_a)
    for row in rows:
        if row["split"] in {"validation", "test"}:
            row["decision"] = CLASSES[(CLASSES.index(row["decision"]) + 1) % 3]
    features_b, labels_b, _, _ = write_inputs(tmp_path, rows, X, suffix="_b")
    fit_evaluate_readout(features_b, labels_b, output_b)
    for method in METHODS:
        with (
            np.load(output_a / (method + ".npz"), allow_pickle=False) as first,
            np.load(output_b / (method + ".npz"), allow_pickle=False) as second,
        ):
            assert first.files == second.files
            for key in first.files:
                np.testing.assert_array_equal(first[key], second[key])


def test_metrics_fixed_class_universe_ties_and_log_loss():
    metrics = decision_metrics([0, 1, 2], np.eye(3))
    assert metrics["balanced_accuracy_fixed_three_class_universe"] == 1.0
    assert metrics["macro_f1_fixed_three_class_universe"] == 1.0
    assert metrics["multiclass_log_loss"] == 0.0
    absent = decision_metrics([0], [[0.5, 0.5, 0.0]])
    assert absent["accuracy"] == 1
    assert absent["balanced_accuracy_fixed_three_class_universe"] == pytest.approx(1 / 3)
    assert absent["macro_f1_fixed_three_class_universe"] == pytest.approx(1 / 3)
    assert absent["missing_evaluation_classes"] == ["REJECT", "DEFER"]
    assert absent["multiclass_log_loss"] == pytest.approx(np.log(2))
    with pytest.raises(ValueError, match="Invalid class probabilities"):
        decision_metrics([0], [[0.8, 0.8, -0.6]])


def test_frozen_scoring_matches_fitted_predictions_without_refit(tmp_path, monkeypatch):
    features, labels, rows, X = write_inputs(tmp_path)
    fitted = tmp_path / "fitted"
    fit_evaluate_readout(features, labels, fitted)
    test_indices = [i for i, row in enumerate(rows) if row["split"] == "test"]
    test_rows = [rows[i] for i in test_indices]
    features_test, labels_test, _, _ = write_inputs(
        tmp_path, test_rows, X[test_indices], suffix="_test"
    )
    before = {path.name: file_hash(path) for path in fitted.iterdir()}

    def no_fit(*args, **kwargs):
        raise AssertionError("Frozen scoring must not call fit")

    monkeypatch.setattr("latent_probes.outcome_readout.LogisticRegression.fit", no_fit)
    monkeypatch.setattr("latent_probes.outcome_readout.TfidfVectorizer.fit", no_fit)
    monkeypatch.setattr("latent_probes.outcome_readout.StandardScaler.fit", no_fit)
    frozen = tmp_path / "frozen"
    report = evaluate_frozen_readout(features_test, labels_test, fitted, frozen)
    assert report["no_refit"] is True
    assert report["fitted_manifest_sha256"] == before["manifest.json"]
    original = {
        (row["method"], row["example_id"]): row
        for row in read_predictions(fitted / "predictions.csv")
        if row["split"] == "test"
    }
    for row in read_predictions(frozen / "predictions.csv"):
        matched = original[row["method"], row["example_id"]]
        for name in CLASSES:
            assert float(row[f"p_{name}"]) == pytest.approx(float(matched[f"p_{name}"]), abs=1e-14)
    assert {path.name: file_hash(path) for path in fitted.iterdir()} == before


def test_frozen_two_class_manual_sigmoid_matches_sklearn(tmp_path):
    _, _, rows, X = write_inputs(tmp_path)
    for row in rows:
        if row["split"] == "train" and row["decision"] == "DEFER":
            row["decision"] = "INVALID"
    features, labels, _, _ = write_inputs(tmp_path, rows, X)
    fitted = tmp_path / "fitted"
    fit_evaluate_readout(features, labels, fitted, allow_two_class_diagnostic=True)
    test_indices = [i for i, row in enumerate(rows) if row["split"] == "test"]
    ft, lt, _, _ = write_inputs(
        tmp_path, [rows[i] for i in test_indices], X[test_indices], suffix="_test"
    )
    evaluate_frozen_readout(ft, lt, fitted, tmp_path / "frozen")
    original = {
        (row["method"], row["example_id"]): row
        for row in read_predictions(fitted / "predictions.csv")
    }
    for row in read_predictions(tmp_path / "frozen" / "predictions.csv"):
        matched = original[row["method"], row["example_id"]]
        for name in CLASSES:
            assert float(row[f"p_{name}"]) == pytest.approx(float(matched[f"p_{name}"]), abs=1e-14)


@pytest.mark.parametrize("corruption", ["weights", "signature", "test_case"])
def test_frozen_rejects_tampering_signature_drift_and_undeclared_cases(tmp_path, corruption):
    features, labels, rows, X = write_inputs(tmp_path)
    fitted = tmp_path / "fitted"
    fit_evaluate_readout(features, labels, fitted)
    test_indices = [i for i, row in enumerate(rows) if row["split"] == "test"]
    test_rows = [dict(rows[i]) for i in test_indices]
    signature = None
    if corruption == "weights":
        with (fitted / "activation.npz").open("ab") as stream:
            stream.write(b"corruption")
        message = "artifact hash mismatch"
    elif corruption == "signature":
        signature = {"position": "prompt_last", "width": 4}
        message = "Feature signature mismatch"
    else:
        test_rows[0]["case_id"] = "train-0-0"
        message = "Undeclared test case"
    ft, lt, _, _ = write_inputs(
        tmp_path, test_rows, X[test_indices], suffix="_test", signature=signature
    )
    with pytest.raises(ValueError, match=message):
        evaluate_frozen_readout(ft, lt, fitted, tmp_path / "frozen")


def test_reject_misaligned_ids_nonfinite_features_and_output_reuse(tmp_path):
    features, labels, rows, X = write_inputs(tmp_path)
    rows[0], rows[1] = rows[1], rows[0]
    _, altered_labels, _, _ = write_inputs(tmp_path, rows, X, suffix="_wrong")
    with pytest.raises(ValueError, match="ordered IDs differ"):
        fit_evaluate_readout(features, altered_labels, tmp_path / "out")
    X[0, 0] = np.nan
    broken_features, broken_labels, _, _ = write_inputs(tmp_path, rows, X, suffix="_nan")
    with pytest.raises(ValueError, match="finite floats"):
        fit_evaluate_readout(broken_features, broken_labels, tmp_path / "out")
    features, labels, _, _ = write_inputs(tmp_path)
    fit_evaluate_readout(features, labels, tmp_path / "out")
    with pytest.raises(ValueError, match="absent or empty"):
        fit_evaluate_readout(features, labels, tmp_path / "out")


def test_cli_guard_returns_honest_not_trainable_status(tmp_path, capsys):
    _, _, rows, X = write_inputs(tmp_path)
    for row in rows:
        if row["split"] == "train":
            row["decision"] = "APPROVE"
    features, labels, _, _ = write_inputs(tmp_path, rows, X)
    assert (
        main(
            [
                "--features",
                str(features),
                "--labels",
                str(labels),
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "not_trainable"
