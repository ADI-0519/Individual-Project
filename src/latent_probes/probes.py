"""A small, auditable linear-probe baseline with group-disjoint evaluation.

This module does not infer intention or create deception labels. The caller
must supply an explicitly defined binary target and independent training data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_features(path: str | Path) -> tuple[np.ndarray, np.ndarray, str | None]:
    with np.load(path, allow_pickle=False) as data:
        if not {"X", "example_ids"}.issubset(data.files):
            raise ValueError("Feature NPZ must contain X and example_ids.")
        X = np.asarray(data["X"], dtype=np.float64)
        ids = np.asarray(data["example_ids"])
        signature = None
        if "feature_signature" in data.files:
            raw = np.asarray(data["feature_signature"])
            if raw.ndim != 0 or raw.dtype.kind != "U" or not str(raw.item()).strip():
                raise ValueError("feature_signature must be a nonempty scalar Unicode string.")
            signature = str(raw.item())
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a nonempty two-dimensional feature matrix.")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values.")
    if ids.ndim != 1 or ids.dtype.kind != "U" or len(ids) != len(X):
        raise ValueError("example_ids must be a Unicode vector with one ID per row of X.")
    if any(not value.strip() for value in ids.tolist()):
        raise ValueError("Example IDs must not be empty.")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("Duplicate example IDs in the feature file.")
    return X, ids, signature


def _load_labels(
    path: str | Path, ids: np.ndarray, target: str, group_column: str
) -> tuple[np.ndarray, np.ndarray, list[str] | None, dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields):
            raise ValueError("The labels CSV has duplicate column names.")
        required = {"example_id", target, group_column}
        if not required.issubset(fields):
            raise ValueError(
                f"The labels CSV is missing columns: {sorted(required - set(fields))}."
            )
        rows: dict[str, dict[str, str]] = {}
        for line, row in enumerate(reader, start=2):
            identifier = (row.get("example_id") or "").strip()
            if not identifier or identifier in rows:
                raise ValueError(f"Empty or duplicate example_id at CSV line {line}.")
            if (row.get(target) or "").strip() not in {"0", "1"}:
                raise ValueError(f"Target {target!r} for {identifier!r} must be explicitly 0 or 1.")
            if not (row.get(group_column) or "").strip():
                raise ValueError(f"Missing {group_column!r} for {identifier!r}.")
            rows[identifier] = row
    feature_ids = set(ids.tolist())
    if set(rows) != feature_ids:
        missing = sorted(feature_ids - set(rows))
        extra = sorted(set(rows) - feature_ids)
        raise ValueError(
            "Features and labels must have an exact one-to-one ID match. "
            f"Missing labels: {missing[:5]}; extra label rows: {extra[:5]}. "
            "Export an explicit matching feature/label subset rather than dropping rows silently."
        )
    ordered = [rows[identifier] for identifier in ids.tolist()]
    semantics: dict[str, Any] = {
        "definition": None,
        "declaration_source": "undeclared_legacy",
        "consistency_checked": False,
        "independently_validated": False,
    }
    if target == "label" and "label_target" in fields:
        meanings = []
        for row in ordered:
            meaning = (row.get("label_target") or "").strip()
            if not meaning or meaning == "manual_definition_required":
                raise ValueError(
                    f"label_target for {row['example_id']!r} must declare a nonempty target meaning; "
                    "replace blank values and manual-definition placeholders before fitting."
                )
            meanings.append(meaning)
        if len(set(meanings)) != 1:
            raise ValueError(
                "The selected label column has multiple label_target meanings. "
                "Train separate probes for separate targets rather than pooling their binary codes."
            )
        semantics.update(
            definition=meanings[0],
            declaration_source="label_target_column",
            consistency_checked=True,
        )
    elif target != "label":
        named_meanings = {
            "objective_pressure": (
                "Assigned favourable objective (1) versus neutral objective (0); "
                "an instruction condition, not deception."
            ),
            "critical_fact_disclosed": (
                "User-supplied annotation that the designated fact is disclosed (1) or omitted (0); "
                "not a label of misleadingness or intent."
            ),
        }
        semantics.update(
            definition=named_meanings.get(target),
            declaration_source="named_target_column",
            consistency_checked=target in named_meanings,
        )
    groups = np.asarray([row[group_column].strip() for row in ordered], dtype=str)
    # Related scenarios/families must not be distributed across chosen groups.
    # In particular, choosing scenario_id must not split a known template family.
    for related_column in ("scenario_id", "family_id"):
        if related_column not in fields or related_column == group_column:
            continue
        mapping: dict[str, str] = {}
        for row, group in zip(ordered, groups, strict=True):
            key = (row.get(related_column) or "").strip()
            if not key:
                raise ValueError(
                    f"Missing {related_column!r}; cannot verify related-example isolation."
                )
            if key in mapping and mapping[key] != group:
                raise ValueError(
                    f"{related_column} {key!r} spans multiple {group_column} groups. "
                    "Use the coarsest related scenario/template family as the split unit."
                )
            mapping[key] = str(group)
    texts = None
    if "text" in fields:
        texts = [(row.get("text") or "").strip() for row in ordered]
        if not all(texts):
            raise ValueError("The optional text baseline requires nonempty text in every row.")
    labels = np.asarray([int(row[target].strip()) for row in ordered], dtype=np.int64)
    return labels, groups, texts, semantics


def split_groups(
    groups: Sequence[str],
    seed: int = 42,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> dict[str, np.ndarray]:
    """Return row indices for one reproducible group split, without label peeking.

    Fractions apply to group counts, not row counts. The labels are deliberately
    absent from this function. There is no search for a more favourable split.
    """
    if not all(math.isfinite(f) and 0 < f < 1 for f in (train_fraction, validation_fraction)):
        raise ValueError("Split fractions must be finite and strictly between zero and one.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Training and validation fractions must leave a test partition.")
    values = np.asarray(groups, dtype=str)
    if values.ndim != 1 or any(not group.strip() for group in values.tolist()):
        raise ValueError("groups must be a one-dimensional sequence of nonempty strings.")
    unique = np.asarray(sorted(set(values.tolist())), dtype=str)
    if len(unique) < 3:
        raise ValueError(
            "At least three independent groups are required for train/validation/test."
        )
    shuffled = np.random.default_rng(seed).permutation(unique)
    n_train = max(1, math.floor(len(unique) * train_fraction))
    n_validation = max(1, math.floor(len(unique) * validation_fraction))
    if n_train + n_validation >= len(unique):
        raise ValueError(
            "These fractions leave no test group; add groups or change the planned fractions."
        )
    partition_groups = {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train : n_train + n_validation],
        "test": shuffled[n_train + n_validation :],
    }
    return {
        name: np.flatnonzero(np.isin(values, selected))
        for name, selected in partition_groups.items()
    }


def binary_metrics(y: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Report explicit denominators; do not invent a rate for an absent class."""
    predictions = scores >= threshold
    tn, fp, fn, tp = (
        int(value) for value in confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    )
    two_classes = len(np.unique(y)) == 2
    return {
        "n_examples": int(len(y)),
        "n_positive": int(np.sum(y)),
        "positive_prevalence": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, scores)) if two_classes else None,
        "average_precision": float(average_precision_score(y, scores)) if two_classes else None,
        "brier_score": float(brier_score_loss(y, scores)),
        "accuracy": float(accuracy_score(y, predictions)),
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "true_positive_rate": tp / (tp + fn) if tp + fn else None,
        "threshold": float(threshold),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _bootstrap(
    y: np.ndarray, scores: np.ndarray, groups: np.ndarray, samples: int, seed: int
) -> dict[str, Any]:
    unique = np.asarray(sorted(set(groups.tolist())), dtype=str)
    result: dict[str, Any] = {
        "method": "percentile bootstrap resampling entire test groups with replacement",
        "n_groups": int(len(unique)),
        "requested_draws": int(samples),
        "valid_draws": 0,
        "confidence_level": 0.95,
        "intervals": {},
        "caution": "Few test groups give unstable intervals; this is not a power analysis.",
    }
    if samples == 0:
        result["status"] = "disabled"
        return result
    if len(unique) < 2:
        result["status"] = "insufficient_groups"
        return result
    rng = np.random.default_rng(seed)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    values: dict[str, list[float]] = {
        key: [] for key in ("roc_auc", "average_precision", "brier_score")
    }
    for _ in range(samples):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([indices[group] for group in selected])
        if len(np.unique(y[rows])) < 2:
            continue
        metrics = binary_metrics(y[rows], scores[rows])
        for key in values:
            values[key].append(metrics[key])
    result["valid_draws"] = len(values["roc_auc"])
    result["status"] = "ok" if result["valid_draws"] >= 20 else "insufficient_valid_draws"
    if result["status"] == "ok":
        result["intervals"] = {
            key: {
                "lower": float(np.quantile(draws, 0.025)),
                "upper": float(np.quantile(draws, 0.975)),
            }
            for key, draws in values.items()
        }
    return result


def _fit_checked(estimator: Any, X: Any, y: np.ndarray) -> Any:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            return estimator.fit(X, y)
    except ConvergenceWarning as exc:
        raise ValueError(
            "Probe fitting did not converge; inspect features and regularisation before interpreting scores."
        ) from exc


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def train_probe(
    features_path: str | Path,
    labels_path: str | Path,
    output_dir: str | Path,
    target: str = "label",
    group_column: str = "family_id",
    seed: int = 42,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    C_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    bootstrap_samples: int = 300,
) -> dict[str, Any]:
    """Fit on train groups, select C on validation groups, evaluate test once.

    A nonempty `text` column enables a separate TF-IDF baseline using exactly the
    same examples/splits. The caller must ensure its information access is fair.
    Saved weights are numeric NPZ/JSON, never pickle or executable objects.
    """
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(
            "Output directory must be new or empty; existing results are never overwritten."
        )
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples < 0
    ):
        raise ValueError("bootstrap_samples must be a nonnegative integer.")
    candidates = sorted(set(float(value) for value in C_grid))
    if not candidates or any(not math.isfinite(value) or value <= 0 for value in candidates):
        raise ValueError("C_grid must contain finite positive regularisation values.")
    X, ids, signature = _load_features(features_path)
    y, groups, texts, target_semantics = _load_labels(labels_path, ids, target, group_column)
    splits = split_groups(groups, seed, train_fraction, validation_fraction)
    for name, rows in splits.items():
        if len(np.unique(y[rows])) != 2:
            raise ValueError(
                f"The preselected {name} partition contains only one class. "
                "Collect more independent groups or revise the study design; do not hunt for a favourable random seed."
            )
    train, validation, test = (splits[name] for name in ("train", "validation", "test"))
    best, best_auc, best_C = None, -math.inf, None
    validation_results = []
    for value in candidates:
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=value, solver="lbfgs", max_iter=3000, random_state=seed),
        )
        _fit_checked(estimator, X[train], y[train])
        auc = float(roc_auc_score(y[validation], estimator.predict_proba(X[validation])[:, 1]))
        validation_results.append({"C": value, "roc_auc": auc})
        if auc > best_auc:  # Candidates are ascending: ties retain stronger regularisation.
            best, best_auc, best_C = estimator, auc, value
    assert best is not None
    scores = best.predict_proba(X[test])[:, 1]
    text_info: dict[str, Any] = {"enabled": False}
    text_best, text_scores = None, None
    if texts is not None:
        text_best_auc, text_best_C = -math.inf, None
        text_validation = []
        for value in candidates:
            candidate = make_pipeline(
                TfidfVectorizer(
                    ngram_range=(1, 2), min_df=1, max_features=20000, sublinear_tf=True
                ),
                LogisticRegression(C=value, solver="lbfgs", max_iter=3000, random_state=seed),
            )
            _fit_checked(candidate, [texts[index] for index in train], y[train])
            auc = float(
                roc_auc_score(
                    y[validation],
                    candidate.predict_proba([texts[index] for index in validation])[:, 1],
                )
            )
            text_validation.append({"C": value, "roc_auc": auc})
            if auc > text_best_auc:
                text_best, text_best_auc, text_best_C = candidate, auc, value
        text_scores = text_best.predict_proba([texts[index] for index in test])[:, 1]
        text_info = {
            "enabled": True,
            "input_column": "text",
            "information_parity_verified": False,
            "selected_C": text_best_C,
            "validation_selection": text_validation,
            "test": binary_metrics(y[test], text_scores),
            "bootstrap": _bootstrap(
                y[test], text_scores, groups[test], bootstrap_samples, seed + 1
            ),
            "interpretation": "A bounded TF-IDF baseline; superiority does not establish information unavailable in text.",
        }
    scaler = best.named_steps["standardscaler"]
    classifier = best.named_steps["logisticregression"]
    manifest = {
        "format_version": 1,
        "method": "train-only StandardScaler followed by L2 logistic regression",
        "target_column": target,
        "target_semantics": target_semantics,
        "group_column": group_column,
        "n_features": int(X.shape[1]),
        "n_examples": int(len(X)),
        "feature_signature": signature,
        "capture_signature_present": signature is not None,
        "feature_provenance_verified": False,
        "provenance_note": "The declared capture signature is checked; external model-weight provenance is not independently authenticated.",
        "features_sha256": _sha256(Path(features_path)),
        "labels_sha256": _sha256(Path(labels_path)),
        "seed": int(seed),
        "selected_C": best_C,
        "validation_selection": validation_results,
        "selection_metric": "validation ROC AUC; ties choose smaller C",
        "threshold": 0.5,
        "threshold_rule": "fixed before fitting; not selected on test and not a 1% FPR threshold",
        "refit_on_train_plus_validation": False,
        "calibration": "No post-hoc calibration; Brier score evaluates the uncalibrated logistic score.",
        "splits": {
            name: {
                "groups": sorted(set(groups[rows].tolist())),
                "example_ids": ids[rows].tolist(),
                "n_examples": int(len(rows)),
                "n_positive": int(np.sum(y[rows])),
            }
            for name, rows in splits.items()
        },
        "test": binary_metrics(y[test], scores),
        "bootstrap": _bootstrap(y[test], scores, groups[test], bootstrap_samples, seed + 1),
        "text_baseline": text_info,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "limitations": [
            "A supplied target is not independently validated by this utility.",
            "The 24-scenario development pilot is not a final confirmatory test set.",
            "Pressure, disclosure, false statements, elicited forecasts and recipient outcomes are different targets.",
            "This implementation does not establish an exact reproduction of the Apollo paper.",
            "One selected split and few test families cannot support broad generalisation claims.",
        ],
    }
    if target_semantics["definition"] is None:
        manifest["limitations"].append(
            "No applicable label_target declaration defines this target; its semantics are unverified."
        )
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "probe.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        coefficient=classifier.coef_[0],
        intercept=classifier.intercept_,
        classes=classifier.classes_,
    )
    if text_best is not None:
        vectorizer = text_best.named_steps["tfidfvectorizer"]
        text_classifier = text_best.named_steps["logisticregression"]
        np.savez_compressed(
            output / "text_probe.npz",
            idf=vectorizer.idf_,
            coefficient=text_classifier.coef_[0],
            intercept=text_classifier.intercept_,
            classes=text_classifier.classes_,
        )
        _write_json(
            output / "text_vocabulary.json",
            {
                "vocabulary": {word: int(index) for word, index in vectorizer.vocabulary_.items()},
                "settings": {
                    "ngram_range": [1, 2],
                    "min_df": 1,
                    "max_features": 20000,
                    "sublinear_tf": True,
                },
            },
        )
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["example_id", group_column, "target", "score", "prediction"]
        if text_scores is not None:
            fields.append("text_score")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for offset, index in enumerate(test):
            row = {
                "example_id": ids[index],
                group_column: groups[index],
                "target": int(y[index]),
                "score": float(scores[offset]),
                "prediction": int(scores[offset] >= 0.5),
            }
            if text_scores is not None:
                row["text_score"] = float(text_scores[offset])
            writer.writerow(row)
    _write_json(output / "manifest.json", manifest)
    return manifest


def predict_probe(
    features_path: str | Path, probe_dir: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Score new feature rows, checking the capture signature when recorded."""
    X, ids, signature = _load_features(features_path)
    directory, destination = Path(probe_dir), Path(output_path)
    if destination.exists():
        raise ValueError("Prediction output already exists; use a new path.")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported saved probe format.")
    if X.shape[1] != manifest["n_features"]:
        raise ValueError("Feature dimension differs from the trained probe.")
    expected_signature = manifest.get("feature_signature")
    if expected_signature is not None and signature != expected_signature:
        raise ValueError(
            "Feature capture signature differs or is missing; probes cannot be transferred blindly between models or capture settings."
        )
    with np.load(directory / "probe.npz", allow_pickle=False) as data:
        mean, scale, coefficient, intercept = (
            data[key] for key in ("mean", "scale", "coefficient", "intercept")
        )
        expected_shape = (X.shape[1],)
        if any(
            value.shape != expected_shape for value in (mean, scale, coefficient)
        ) or intercept.shape != (1,):
            raise ValueError("Malformed saved probe parameter dimensions.")
        if not all(
            np.isfinite(value).all() for value in (mean, scale, coefficient, intercept)
        ) or np.any(scale <= 0):
            raise ValueError("Invalid saved probe parameter values.")
        if not np.array_equal(data["classes"], np.array([0, 1])):
            raise ValueError("Saved probe must have classes [0, 1].")
        logits = ((X - mean) / scale) @ coefficient + float(intercept[0])
    # Stable logistic transform, including large negative logits.
    scores = np.exp(-np.logaddexp(0.0, -logits))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["example_id", "score", "prediction"])
        writer.writeheader()
        for identifier, score in zip(ids, scores, strict=True):
            writer.writerow(
                {
                    "example_id": identifier,
                    "score": float(score),
                    "prediction": int(score >= manifest["threshold"]),
                }
            )
    return {
        "n_examples": len(ids),
        "output_path": str(destination),
        "feature_signature_verified": expected_signature is not None,
        "target_column": manifest["target_column"],
        "threshold": manifest["threshold"],
    }
