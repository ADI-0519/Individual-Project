"""Post-hoc CPU diagnosis of a saved before-report readout and pressure shift.

No language model is loaded. Candidate settings are selected solely by neutral
validation log loss; only the selected candidate is evaluated on test outcomes.
All outcomes were already observed before this diagnostic was designed.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize

from latent_probes.outcome_readout import CLASSES, decision_metrics
from latent_probes.storage import atomic_json, environment_info, file_sha256, read_json, utc_now

C_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
SCALINGS = ("standard", "standard_row_l2", "raw_row_l2")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _dataset(directory):
    with np.load(directory / "features.npz", allow_pickle=False) as archive:
        X = archive["X"].astype(np.float64)
        ids = archive["example_ids"].tolist()
        signature = json.loads(archive["feature_signature"].item())
    with (directory / "labels.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    _require(X.ndim == 2 and np.isfinite(X).all(), "Invalid feature matrix")
    _require(ids == [r["example_id"] for r in rows] and len(set(ids)) == len(ids), "Misaligned IDs")
    _require(len(rows) == len(X) and len(rows) > 0, "Dataset shape mismatch")
    _require(all(r["decision"] in {*CLASSES, "INVALID"} for r in rows), "Unknown decision")
    _require(all(r["report_text"] == "[REPORT NOT YET GENERATED]" for r in rows), "Report leakage")
    return X, rows, signature


def _probabilities(X, weights, intercept):
    logits = X @ weights.T + intercept
    _require(np.isfinite(logits).all(), "Nonfinite logits")
    probabilities = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    return logits, probabilities


def _indices(rows, split=None, objective=None):
    return np.asarray(
        [
            i
            for i, r in enumerate(rows)
            if (split is None or r["split"] == split)
            and (objective is None or r["objective"] == objective)
            and r["decision"] != "INVALID"
        ],
        dtype=np.int64,
    )


def _labels(rows, indices):
    return np.asarray([CLASSES.index(rows[i]["decision"]) for i in indices])


def _scale_fit(X, mode):
    if mode == "raw_row_l2":
        return None, normalize(X, norm="l2")
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    return scaler, normalize(Z, norm="l2") if mode == "standard_row_l2" else Z


def _scale_transform(X, scaler, mode):
    if mode == "raw_row_l2":
        return normalize(X, norm="l2")
    Z = scaler.transform(X)
    return normalize(Z, norm="l2") if mode == "standard_row_l2" else Z


def fit_validation_grid(X_train, y_train, X_validation, y_validation, c_grid=C_GRID):
    """Test arrays/labels are deliberately absent from this selection API."""
    _require(np.array_equal(np.unique(y_train), np.arange(3)), "Need all three training classes")
    _require(len(y_validation) > 0, "Need valid neutral validation rows")
    _require(bool(c_grid) and all(np.isfinite(c) and c > 0 for c in c_grid), "Invalid C grid")
    results, candidates = [], []
    for mode in SCALINGS:
        scaler, train = _scale_fit(X_train, mode)
        validation = _scale_transform(X_validation, scaler, mode)
        for c in c_grid:
            model = LogisticRegression(
                C=float(c), solver="lbfgs", max_iter=2000, random_state=20261119
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(train, y_train)
            _require(np.array_equal(model.classes_, np.arange(3)), "Fitted class order mismatch")
            converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            result = {
                "scaling": mode,
                "C": float(c),
                "converged": converged,
                "n_iterations": model.n_iter_.tolist(),
                "train": decision_metrics(y_train, model.predict_proba(train)),
                "validation": decision_metrics(y_validation, model.predict_proba(validation)),
            }
            results.append(result)
            candidates.append((model, scaler, mode))
    eligible = [i for i, r in enumerate(results) if r["converged"]]
    _require(bool(eligible), "No converged candidate")
    selected = min(eligible, key=lambda i: (results[i]["validation"]["multiclass_log_loss"], i))
    return results, selected, candidates[selected]


def standardized_summary(Z, logits, probabilities):
    mean = Z.mean(axis=0)
    return {
        "n": len(Z),
        "mean_vector_l2": float(np.linalg.norm(mean)),
        "mean_vector_rms_per_coordinate": float(np.sqrt(np.mean(mean**2))),
        "median_row_l2": float(np.median(np.linalg.norm(Z, axis=1))),
        "max_abs_coordinate": float(np.max(np.abs(Z))),
        "fraction_coordinates_abs_z_gt3": float(np.mean(np.abs(Z) > 3)),
        "fraction_feature_means_abs_gt1": float(np.mean(np.abs(mean) > 1)),
        "mean_logits": dict(zip(CLASSES, logits.mean(axis=0).tolist(), strict=True)),
        "mean_probabilities": dict(zip(CLASSES, probabilities.mean(axis=0).tolist(), strict=True)),
        "predicted_class_counts": {
            name: int(np.sum(probabilities.argmax(axis=1) == i)) for i, name in enumerate(CLASSES)
        },
    }


def diagnose_shift(prereport_dir, output_path, *, c_grid=C_GRID):
    root, output_path = Path(prereport_dir).resolve(), Path(output_path).resolve()
    _require(not output_path.exists(), "Use a fresh diagnostic output")
    _require(not output_path.is_relative_to(root), "Output must be outside the original run")
    manifest = read_json(root / "manifest.json")
    _require(manifest["status"] == "complete", "Prereport source must be complete")
    required = (
        "neutral_dataset/features.npz",
        "neutral_dataset/labels.csv",
        "test_dataset/features.npz",
        "test_dataset/labels.csv",
        "neutral_readout/activation.npz",
        "neutral_readout/report.json",
        "neutral_readout/manifest.json",
        "frozen_readout.json",
    )
    hashes = {"manifest.json": file_sha256(root / "manifest.json")}
    for name in required:
        actual = file_sha256(root / name)
        _require(actual == manifest["output_sha256"][name], f"Input hash mismatch: {name}")
        hashes[name] = actual
    frozen = read_json(root / "frozen_readout.json")
    _require(
        frozen["artifacts"]["activation.npz"] == hashes["neutral_readout/activation.npz"],
        "Frozen weights changed",
    )
    X, rows, signature = _dataset(root / "neutral_dataset")
    X_test, test_rows, test_signature = _dataset(root / "test_dataset")
    _require(
        signature == test_signature and signature["prediction_time"] == "before_report",
        "Wrong feature time/signature",
    )
    _require(
        signature["generated_response_tokens_supplied"] == 0, "Features include response tokens"
    )
    _require(all(r["objective"] == "neutral" for r in rows), "Neutral dataset includes pressure")
    _require(all(r["split"] == "test" for r in test_rows), "Test dataset has non-test cases")
    train_i, val_i = _indices(rows, "train"), _indices(rows, "validation")
    _require(len(train_i) > 0 and len(val_i) > 0, "Missing valid train/validation rows")
    case_sets = {
        split: {r["case_id"] for r in rows if r["split"] == split}
        for split in ("train", "validation", "test")
    }
    _require(
        not (
            case_sets["train"] & case_sets["validation"]
            or case_sets["train"] & case_sets["test"]
            or case_sets["validation"] & case_sets["test"]
        ),
        "Case leakage",
    )
    _require(all(r["case_id"] in case_sets["test"] for r in test_rows), "Undeclared test case")
    with np.load(root / "neutral_readout/activation.npz", allow_pickle=False) as saved:
        mean, scale, weights, intercept = (
            saved[k].copy() for k in ("mean", "scale", "coefficients", "intercept")
        )
        _require(
            np.array_equal(saved["class_indices"], np.arange(3)), "Expected three frozen classes"
        )
    scaler_check = StandardScaler().fit(X[train_i])
    _require(
        np.array_equal(mean, scaler_check.mean_) and np.array_equal(scale, scaler_check.scale_),
        "Frozen scaler is not neutral-train-only",
    )
    Z, Z_test = (X - mean) / scale, (X_test - mean) / scale
    logits, probabilities = _probabilities(Z, weights, intercept)
    test_logits, test_probabilities = _probabilities(Z_test, weights, intercept)
    groups = {
        "train/neutral": (rows, train_i, Z, logits, probabilities),
        "validation/neutral": (rows, val_i, Z, logits, probabilities),
        "test/neutral": (
            test_rows,
            _indices(test_rows, "test", "neutral"),
            Z_test,
            test_logits,
            test_probabilities,
        ),
        "test/favorable": (
            test_rows,
            _indices(test_rows, "test", "favorable"),
            Z_test,
            test_logits,
            test_probabilities,
        ),
    }
    original = {}
    for name, (r, indices, values, scores, probs) in groups.items():
        _require(len(indices) > 0, f"No rows for {name}")
        original[name] = {
            "metrics": decision_metrics(_labels(r, indices), probs[indices]),
            "standardized_shift": standardized_summary(
                values[indices], scores[indices], probs[indices]
            ),
        }
    neutral_test = _indices(test_rows, "test", "neutral")
    favorable_test = _indices(test_rows, "test", "favorable")
    mean_delta = Z_test[favorable_test].mean(axis=0) - Z_test[neutral_test].mean(axis=0)
    logit_delta = weights @ mean_delta
    centered_train = Z[train_i] - Z[train_i].mean(axis=0)
    singular = np.linalg.svd(centered_train, compute_uv=False)
    tolerance = singular.max() * max(centered_train.shape) * np.finfo(np.float64).eps
    candidates, selected, (selected_model, selected_scaler, selected_mode) = fit_validation_grid(
        X[train_i], _labels(rows, train_i), X[val_i], _labels(rows, val_i), c_grid
    )
    # This is the first candidate use of test data. No other setting is evaluated here.
    selected_test = {}
    for objective in ("neutral", "favorable"):
        indices = _indices(test_rows, "test", objective)
        values = _scale_transform(X_test[indices], selected_scaler, selected_mode)
        selected_test["test/" + objective] = decision_metrics(
            _labels(test_rows, indices), selected_model.predict_proba(values)
        )
    report = {
        "schema": "prereport_shift_diagnostic_v1",
        "created_at": utc_now(),
        "analysis_status": "posthoc_exploratory_validation_selected_diagnostic",
        "source_input_sha256": hashes,
        "script_sha256": file_sha256(Path(__file__)),
        "environment": environment_info(),
        "feature_signature": signature,
        "original_numerical_status": manifest["numerical_status"],
        "valid_training_rows": len(train_i),
        "excluded_invalid_neutral_rows": sum(r["decision"] == "INVALID" for r in rows),
        "training_feature_width": X.shape[1],
        "unique_training_vectors": len(np.unique(X[train_i], axis=0)),
        "centered_training_rank": int(np.sum(singular > tolerance)),
        "rank_tolerance": float(tolerance),
        "original_frozen_C1": original,
        "mean_favorable_minus_neutral_test": {
            "standardized_vector_l2": float(np.linalg.norm(mean_delta)),
            "standardized_vector_rms_per_coordinate": float(np.sqrt(np.mean(mean_delta**2))),
            "logit_change": dict(zip(CLASSES, logit_delta.tolist(), strict=True)),
            "reject_minus_approve_margin_change": float(logit_delta[1] - logit_delta[0]),
            "reject_minus_defer_margin_change": float(logit_delta[1] - logit_delta[2]),
            "interpretation": "Descriptive aggregate over the same test-case/context panel; no new causal mechanism claim.",
        },
        "posthoc_validation_grid": {
            "selection": "minimum neutral validation log loss among converged candidates; ties use listed order",
            "C_grid": list(c_grid),
            "scalings": list(SCALINGS),
            "no_test_guided_selection": True,
            "candidates": candidates,
            "selected_index": selected,
            "selected_settings": {
                "scaling": candidates[selected]["scaling"],
                "C": candidates[selected]["C"],
            },
            "selected_test_only": selected_test,
        },
        "limitations": [
            "All test outcomes were known before this diagnostic was proposed; this remains post-hoc even though settings are selected only by validation labels.",
            "Validation class counts are reported explicitly; an absent class cannot constrain its validation recall or log loss.",
            "One layer, one position, one small case bank and 47 labels cannot establish absence of anticipatory information or second-order beliefs.",
            "Perfect in-sample separation in a high-dimensional feature space is not evidence of generalization.",
            "Case-held-out development with repeated contexts and shared rule families; no population confidence claim.",
            "Original numerical parity failures, source factual errors and reader-control failures remain unresolved and are not edited here.",
            "No after-report weights, features, reports, labels or earlier results are changed or overwritten.",
        ],
    }
    _require(
        all(file_sha256(root / name) == digest for name, digest in hashes.items()),
        "Source changed during diagnosis",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = diagnose_shift(args.run, args.output)
    print(
        json.dumps(
            {
                "status": report["analysis_status"],
                "original": {
                    key: value["metrics"]["accuracy"]
                    for key, value in report["original_frozen_C1"].items()
                },
                "selected": report["posthoc_validation_grid"]["selected_settings"],
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
