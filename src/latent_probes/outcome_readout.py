"""Neutral-trained, case-held-out readouts of observed reader decisions.

This development diagnostic predicts APPROVE/REJECT/DEFER under the packet
attributed to a sender. It supplies neither deception nor second-order-belief
labels. No model is loaded and no test labels affect any fitted parameter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.sparse import csr_matrix, hstack
from scipy.special import expit, logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CLASSES = ("APPROVE", "REJECT", "DEFER")
REQUIRED_COLUMNS = (
    "example_id",
    "case_id",
    "family",
    "objective",
    "split",
    "decision",
    "prompt_text",
    "report_text",
)
METHODS = ("activation", "prompt_only", "report_only", "full_text", "full_text_activation")
C = 1.0
SEED = 20261119
MAX_ITER = 2000
TOKEN_PATTERN = r"(?u)\b\w+\b"
TARGET = "observed_reader_decision_under_sender_attributed_packet"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _csv(path, fields, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _json_scalar(archive, key):
    _require(key in archive.files, f"Missing NPZ field: {key}")
    value = archive[key]
    _require(value.shape == () and value.dtype.kind == "U", f"{key} must be scalar Unicode")
    parsed = json.loads(str(value.item()))
    _require(isinstance(parsed, dict) and bool(parsed), f"{key} must encode a nonempty object")
    _canonical(parsed)  # Reject NaN/Infinity in supplied provenance.
    return parsed


def _read_inputs(features_path, labels_path, *, require_all_splits=True):
    hashes = {"features_sha256": _hash(features_path), "labels_sha256": _hash(labels_path)}
    with np.load(features_path, allow_pickle=False) as archive:
        _require("X" in archive.files and "example_ids" in archive.files, "Missing X/example_ids")
        X = archive["X"]
        ids = archive["example_ids"]
        _require(X.ndim == 2 and all(X.shape), "X must be a nonempty two-dimensional matrix")
        _require(X.dtype.kind == "f" and np.isfinite(X).all(), "X must contain finite floats")
        _require(ids.ndim == 1 and ids.dtype.kind == "U", "example_ids must be Unicode strings")
        _require(len(ids) == len(X), "Feature/ID row count mismatch")
        example_ids = ids.tolist()
        _require(
            all(example_ids) and len(set(example_ids)) == len(ids), "Duplicate/empty example ID"
        )
        provenance = _json_scalar(archive, "provenance_json")
        signature = _json_scalar(archive, "feature_signature")
        X = X.astype(np.float64, copy=True)
    with labels_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames is not None, "Labels CSV has no header")
        _require(len(reader.fieldnames) == len(set(reader.fieldnames)), "Duplicate CSV columns")
        _require(set(REQUIRED_COLUMNS) <= set(reader.fieldnames), "Missing required label columns")
        rows = list(reader)
    _require(len(rows) == len(X), "Feature/CSV row count mismatch")
    _require([row["example_id"] for row in rows] == example_ids, "Feature/CSV ordered IDs differ")
    case_splits, case_families = {}, {}
    for row in rows:
        _require(
            None not in row and all(row[k] is not None for k in REQUIRED_COLUMNS), "Bad CSV row"
        )
        for key in ("example_id", "case_id", "family", "objective", "split", "decision"):
            _require(bool(row[key]) and row[key] == row[key].strip(), f"Empty/untrimmed {key}")
        _require(row["split"] in {"train", "validation", "test"}, "Unknown split")
        _require(row["objective"] in {"neutral", "favorable"}, "Unknown objective")
        _require(
            row["decision"] in {*CLASSES, "INVALID"}, "Unknown decision; use INVALID explicitly"
        )
        _require(
            row["objective"] == "neutral" or row["split"] == "test",
            "Only neutral rows may enter training or validation",
        )
        case = row["case_id"]
        _require(
            case not in case_splits or case_splits[case] == row["split"],
            f"Case leakage across splits: {case}",
        )
        _require(
            case not in case_families or case_families[case] == row["family"],
            f"Inconsistent family for case: {case}",
        )
        case_splits[case], case_families[case] = row["split"], row["family"]
    if require_all_splits:
        _require(
            set(case_splits.values()) == {"train", "validation", "test"}, "All splits required"
        )
    else:
        _require(set(case_splits.values()) == {"test"}, "Frozen evaluation accepts test rows only")
    _require(
        hashes == {"features_sha256": _hash(features_path), "labels_sha256": _hash(labels_path)},
        "Input changed during loading",
    )
    return X, rows, provenance, signature, hashes


def decision_metrics(labels, probabilities):
    """Three-class universe, including zero recall/F1 for absent target classes."""
    y = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    _require(y.ndim == 1 and len(y) > 0, "Metrics require at least one valid label")
    _require(np.isin(y, range(3)).all(), "Metrics label outside declared classes")
    _require(probabilities.shape == (len(y), 3), "Expected three probability columns")
    _require(
        np.isfinite(probabilities).all()
        and (probabilities >= 0).all()
        and (probabilities <= 1).all()
        and np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-10, atol=1e-12),
        "Invalid class probabilities",
    )
    predictions = probabilities.argmax(axis=1)
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (y, predictions), 1)
    counts = confusion.sum(axis=1)
    recalls, f1 = [], []
    for index in range(3):
        tp = int(confusion[index, index])
        fn = int(confusion[index].sum()) - tp
        fp = int(confusion[:, index].sum()) - tp
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {
        "n_valid_rows": len(y),
        "class_counts": {name: int(counts[i]) for i, name in enumerate(CLASSES)},
        "missing_evaluation_classes": [name for i, name in enumerate(CLASSES) if not counts[i]],
        "confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
        "accuracy": float(np.mean(predictions == y)),
        "balanced_accuracy_fixed_three_class_universe": float(np.mean(recalls)),
        "macro_f1_fixed_three_class_universe": float(np.mean(f1)),
        "multiclass_log_loss": float(
            -np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-15, 1.0)).mean()
        ),
        "per_class_recall": dict(zip(CLASSES, recalls, strict=True)),
        "per_class_f1": dict(zip(CLASSES, f1, strict=True)),
        "absent_class_metric_policy": "recall and F1 are zero; fixed universe is always three classes",
        "argmax_tie_policy": "first class in declared class order",
    }


def _counts(rows):
    result = {}
    for split in ("train", "validation", "test"):
        for objective in ("neutral", "favorable"):
            selected = [r for r in rows if r["split"] == split and r["objective"] == objective]
            if not selected:
                continue
            result[f"{split}/{objective}"] = {
                "n_rows_including_invalid": len(selected),
                "n_cases_including_invalid": len({r["case_id"] for r in selected}),
                "class_counts": {
                    name: sum(r["decision"] == name for r in selected) for name in CLASSES
                },
                "n_invalid_excluded": sum(r["decision"] == "INVALID" for r in selected),
            }
    return result


def _text(rows, method):
    if method == "prompt_only":
        return [r["prompt_text"] for r in rows]
    if method == "report_only":
        return [r["report_text"] for r in rows]
    return [r["prompt_text"] + "\n[REPORT]\n" + r["report_text"] for r in rows]


def _fit(matrix, labels):
    model = LogisticRegression(C=C, solver="lbfgs", max_iter=MAX_ITER, random_state=SEED)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(matrix, labels)
    return model, not any(issubclass(item.category, ConvergenceWarning) for item in caught)


def _save_model(path, model, signature, scaler=None, vectorizer=None):
    arrays = {
        "coefficients": model.coef_.astype(np.float64),
        "intercept": model.intercept_.astype(np.float64),
        "class_indices": model.classes_.astype(np.int64),
        "class_names": np.asarray(CLASSES),
        "settings_json": np.asarray(
            _canonical(
                {
                    **signature,
                    "coefficient_interpretation": (
                        "binary sigmoid gives second fitted class probability"
                        if len(model.classes_) == 2
                        else "multinomial softmax in fitted class order"
                    ),
                }
            )
        ),
    }
    if scaler is not None:
        arrays.update(mean=scaler.mean_, scale=scaler.scale_, variance=scaler.var_)
    if vectorizer is not None:
        arrays.update(
            vocabulary_terms=np.asarray(vectorizer.get_feature_names_out(), dtype=np.str_),
            idf=vectorizer.idf_.astype(np.float64),
        )
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _expand_probabilities(probabilities, classes):
    expanded = np.zeros((len(probabilities), 3), dtype=np.float64)
    expanded[:, classes] = probabilities
    return expanded


def fit_evaluate_readout(
    features_path, labels_path, output_dir, *, allow_two_class_diagnostic=False
):
    """Fit once on neutral training cases; evaluate frozen validation and test rows.

    Required NPZ: X, ordered Unicode example_ids, scalar Unicode JSON objects
    provenance_json and feature_signature. CSV fields are REQUIRED_COLUMNS.
    A nonempty output directory is rejected to avoid mixing experiment versions.
    """
    features_path, labels_path, output_dir = map(Path, (features_path, labels_path, output_dir))
    _require(
        not output_dir.exists() or (output_dir.is_dir() and not any(output_dir.iterdir())),
        "Output directory must be absent or empty",
    )
    X, rows, provenance, signature, hashes = _read_inputs(features_path, labels_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = np.asarray(
        [i for i, r in enumerate(rows) if r["split"] == "train" and r["decision"] != "INVALID"],
        dtype=np.int64,
    )
    label_indices = np.asarray(
        [CLASSES.index(r["decision"]) if r["decision"] in CLASSES else -1 for r in rows]
    )
    row_manifest = [
        {
            **{key: row[key] for key in REQUIRED_COLUMNS[:6]},
            "excluded": row["decision"] == "INVALID",
            "exclusion_reason": "INVALID_reader_output" if row["decision"] == "INVALID" else "",
            "used_for_training": i in train,
        }
        for i, row in enumerate(rows)
    ]
    ledger_fields = [*REQUIRED_COLUMNS[:6], "excluded", "exclusion_reason", "used_for_training"]
    _csv(output_dir / "row_manifest.csv", ledger_fields, row_manifest)
    _csv(output_dir / "exclusions.csv", ledger_fields, [r for r in row_manifest if r["excluded"]])
    settings = {
        "schema": "neutral_outcome_readout_settings_v1",
        "target": TARGET,
        "classes": list(CLASSES),
        "C": C,
        "solver": "lbfgs",
        "penalty": "l2",
        "fit_intercept": True,
        "class_weight": None,
        "max_iter": MAX_ITER,
        "seed": SEED,
        "hyperparameter_selection": "none; fixed C=1 before reading evaluation labels",
        "allow_two_class_diagnostic": bool(allow_two_class_diagnostic),
        "unsupported_class_probability": 0.0,
        "log_loss_probability_clip": 1e-15,
        "activation_normalization": "StandardScaler population mean/variance from valid neutral train rows only",
        "feature_order_combined": "TF-IDF columns then standardized activation columns",
        "tfidf": {
            "ngram_range": [1, 2],
            "max_features": 20000,
            "min_df": 1,
            "lowercase": True,
            "token_pattern": TOKEN_PATTERN,
            "sublinear_tf": True,
            "norm": "l2",
            "smooth_idf": True,
            "fit_partition": "valid neutral train rows only",
            "full_text_separator": "\n[REPORT]\n",
        },
    }
    cases = {
        split: sorted({r["case_id"] for r in rows if r["split"] == split})
        for split in ("train", "validation", "test")
    }
    families = {
        split: sorted({r["family"] for r in rows if r["split"] == split})
        for split in ("train", "validation", "test")
    }
    report = {
        "schema": "neutral_outcome_readout_report_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "not_trainable",
        "target": TARGET,
        "settings": settings,
        "input_sha256": hashes,
        "feature_signature": signature,
        "provenance": provenance,
        "features_shape": list(X.shape),
        "counts": _counts(rows),
        "case_partitions": cases,
        "family_partitions": families,
        "families_shared_across_partitions": sorted(
            (set(families["train"]) & set(families["validation"]))
            | (set(families["train"]) & set(families["test"]))
            | (set(families["validation"]) & set(families["test"]))
        ),
        "split_scope": "case-held-out development; family OOD is not established",
        "methods": {},
        "limitations": [
            "Observed reader decisions under sender-attributed packets, not deception or second-order belief labels.",
            "A sender activation can encode source content or instructions; predictive accuracy alone does not identify a mechanism.",
            "TF-IDF methods are lexical controls, not capable LLM monitors.",
            "Repeated contexts within a case are clustered; descriptive row metrics supply no population confidence estimate.",
            "INVALID outputs remain excluded from fitting/metrics and visible in every partition count and the ledger.",
            "Case partitions are checked by declared case_id; semantic duplicates require a separate case-bank audit.",
            "The source features are supplied artifacts; this CPU readout does not independently replay the model.",
        ],
    }
    missing = [CLASSES[i] for i in range(3) if i not in label_indices[train]]
    report["learning_classes"] = [name for name in CLASSES if name not in missing]
    report["unsupported_classes"] = missing
    report["primary_three_class_status"] = "not_trainable" if missing else "trainable"
    prediction_rows = []
    diagnostic = bool(allow_two_class_diagnostic and len(missing) == 1)
    if missing and not diagnostic:
        report["reason"] = "requires_all_three_training_classes"
        report["missing_training_classes"] = missing
    else:
        report["status"] = (
            "diagnostic_only_missing_training_class"
            if diagnostic
            else "completed_development_evaluation"
        )
        if diagnostic:
            report["reason"] = (
                "primary_requires_all_three_classes; optional two-class feasibility only"
            )
            report["limitations"].append(
                "The missing training class is unsupported and assigned probability zero; clipped log loss does not remedy open-set failure."
            )
        scaler = StandardScaler().fit(X[train])
        standardized = scaler.transform(X)
        groups = sorted({(r["split"], r["objective"]) for r in rows if r["split"] != "train"})
        for method in METHODS:
            vectorizer = None
            if method == "activation":
                matrix = standardized
            else:
                texts = _text(rows, method)
                vectorizer = TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=20000,
                    min_df=1,
                    lowercase=True,
                    token_pattern=TOKEN_PATTERN,
                    sublinear_tf=True,
                    norm="l2",
                    smooth_idf=True,
                    dtype=np.float64,
                )
                try:
                    vectorizer.fit([texts[i] for i in train])
                except ValueError as error:
                    if "empty vocabulary" not in str(error):
                        raise
                    report["methods"][method] = {"status": "unavailable_empty_training_vocabulary"}
                    continue
                matrix = vectorizer.transform(texts)
                if method == "full_text_activation":
                    matrix = hstack([matrix, csr_matrix(standardized)], format="csr")
            model, converged = _fit(matrix[train], label_indices[train])
            expected_classes = np.unique(label_indices[train])
            _require(
                np.array_equal(model.classes_, expected_classes), "Unexpected fitted class order"
            )
            artifact = method + ".npz"
            _save_model(
                output_dir / artifact,
                model,
                {**settings, "method": method},
                scaler if method in {"activation", "full_text_activation"} else None,
                vectorizer,
            )
            result = {
                "status": "fitted" if converged else "fitted_with_convergence_warning",
                "converged": converged,
                "n_iterations": model.n_iter_.tolist(),
                "n_training_rows": len(train),
                "n_training_cases": len({rows[i]["case_id"] for i in train}),
                "learning_classes": [CLASSES[i] for i in model.classes_],
                "unsupported_classes": missing,
                "artifact": artifact,
                "artifact_sha256": _hash(output_dir / artifact),
                "evaluations": {},
            }
            for split, objective in groups:
                indices = np.asarray(
                    [
                        i
                        for i, r in enumerate(rows)
                        if r["split"] == split
                        and r["objective"] == objective
                        and r["decision"] != "INVALID"
                    ],
                    dtype=np.int64,
                )
                group = split + "/" + objective
                if not len(indices):
                    result["evaluations"][group] = {"status": "no_valid_labels", "n_valid_rows": 0}
                    continue
                probabilities = _expand_probabilities(
                    model.predict_proba(matrix[indices]), model.classes_
                )
                result["evaluations"][group] = decision_metrics(
                    label_indices[indices], probabilities
                )
                for i, probabilities_row in zip(indices, probabilities, strict=True):
                    prediction_rows.append(
                        {
                            "method": method,
                            **{key: rows[i][key] for key in REQUIRED_COLUMNS[:6]},
                            "predicted_decision": CLASSES[int(probabilities_row.argmax())],
                            **{
                                f"p_{name}": float(probabilities_row[j])
                                for j, name in enumerate(CLASSES)
                            },
                        }
                    )
            report["methods"][method] = result
    _require(
        hashes == {"features_sha256": _hash(features_path), "labels_sha256": _hash(labels_path)},
        "Input changed during fitting/evaluation; output is incomplete",
    )
    prediction_fields = [
        "method",
        *REQUIRED_COLUMNS[:6],
        "predicted_decision",
        *[f"p_{name}" for name in CLASSES],
    ]
    _csv(output_dir / "predictions.csv", prediction_fields, prediction_rows)
    _json(output_dir / "report.json", report)
    outputs = {
        path.name: _hash(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    _json(
        output_dir / "manifest.json",
        {
            "schema": "neutral_outcome_readout_manifest_v1",
            "status": report["status"],
            "input_sha256": hashes,
            "source_code_sha256": _hash(Path(__file__)),
            "settings_sha256": hashlib.sha256(_canonical(settings).encode("utf-8")).hexdigest(),
            "output_sha256": outputs,
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
            },
        },
    )
    return report


def _frozen_probabilities(path, X, rows):
    """Reconstruct only transforms and numeric decision functions; never fit."""
    with np.load(path, allow_pickle=False) as archive:
        settings = _json_scalar(archive, "settings_json")
        method = settings["method"]
        _require(method in METHODS, "Unknown frozen method")
        classes = archive["class_indices"]
        weights = archive["coefficients"]
        intercept = archive["intercept"]
        _require(
            classes.ndim == 1
            and classes.dtype.kind in "iu"
            and len(classes) in {2, 3}
            and np.array_equal(classes, np.unique(classes))
            and np.isin(classes, range(3)).all(),
            "Invalid frozen class indices",
        )
        n_outputs = 1 if len(classes) == 2 else 3
        _require(
            weights.ndim == 2
            and weights.shape[0] == n_outputs
            and intercept.shape == (n_outputs,)
            and np.isfinite(weights).all()
            and np.isfinite(intercept).all(),
            "Invalid frozen coefficients",
        )
        if method in {"activation", "full_text_activation"}:
            mean, scale = archive["mean"], archive["scale"]
            _require(
                mean.shape == scale.shape == (X.shape[1],)
                and np.isfinite(mean).all()
                and np.isfinite(scale).all()
                and (scale > 0).all(),
                "Frozen scaler/feature width mismatch",
            )
            standardized = (X - mean) / scale
        if method == "activation":
            matrix = standardized
        else:
            terms, idf = archive["vocabulary_terms"], archive["idf"]
            _require(
                terms.ndim == 1
                and terms.dtype.kind == "U"
                and len(terms) > 0
                and len(set(terms.tolist())) == len(terms)
                and idf.shape == terms.shape
                and np.isfinite(idf).all()
                and (idf > 0).all(),
                "Invalid frozen TF-IDF vocabulary/IDF",
            )
            vectorizer = TfidfVectorizer(
                vocabulary={term: index for index, term in enumerate(terms.tolist())},
                ngram_range=(1, 2),
                max_features=20000,
                min_df=1,
                lowercase=True,
                token_pattern=TOKEN_PATTERN,
                sublinear_tf=True,
                norm="l2",
                smooth_idf=True,
                dtype=np.float64,
            )
            vectorizer.idf_ = idf
            matrix = vectorizer.transform(_text(rows, method))
            if method == "full_text_activation":
                matrix = hstack([matrix, csr_matrix(standardized)], format="csr")
        _require(matrix.shape[1] == weights.shape[1], "Frozen classifier feature width mismatch")
        logits = np.asarray(matrix @ weights.T) + intercept
        _require(np.isfinite(logits).all(), "Nonfinite frozen decision function")
        if len(classes) == 2:
            positive = expit(logits[:, 0])
            probabilities = np.column_stack([1.0 - positive, positive])
        else:
            probabilities = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        return method, _expand_probabilities(probabilities, classes)


def evaluate_frozen_readout(features_path, labels_path, fitted_dir, output_dir):
    """Score declared test cases with verified saved transforms and fixed weights.

    No estimator.fit call occurs. The source readout's weights, output manifest,
    and feature signature are checked before and after evaluation. Source and
    destination directories must differ, and source artifacts remain untouched.
    """
    features_path, labels_path, fitted_dir, output_dir = map(
        Path, (features_path, labels_path, fitted_dir, output_dir)
    )
    _require(fitted_dir.resolve() != output_dir.resolve(), "Frozen output must use a new directory")
    _require(
        not output_dir.exists() or (output_dir.is_dir() and not any(output_dir.iterdir())),
        "Output directory must be absent or empty",
    )
    manifest_path = fitted_dir / "manifest.json"
    manifest_hash = _hash(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest["schema"] == "neutral_outcome_readout_manifest_v1", "Unknown fitted manifest")
    artifact_hashes = manifest["output_sha256"]
    _require(
        isinstance(artifact_hashes, dict) and "report.json" in artifact_hashes,
        "Missing output hashes",
    )
    for name, digest in artifact_hashes.items():
        _require(
            Path(name).name == name and name not in {"", ".", ".."}, "Unsafe frozen artifact path"
        )
        _require(_hash(fitted_dir / name) == digest, f"Frozen artifact hash mismatch: {name}")
    fitted_report = json.loads((fitted_dir / "report.json").read_text(encoding="utf-8"))
    X, rows, provenance, signature, hashes = _read_inputs(
        features_path, labels_path, require_all_splits=False
    )
    _require(
        _canonical(signature) == _canonical(fitted_report["feature_signature"]),
        "Feature signature mismatch",
    )
    _require(X.shape[1] == fitted_report["features_shape"][1], "Feature width mismatch")
    declared_test = set(fitted_report["case_partitions"]["test"])
    _require(
        all(r["case_id"] in declared_test for r in rows),
        "Undeclared test case or training/validation leakage",
    )
    with (fitted_dir / "row_manifest.csv").open(encoding="utf-8", newline="") as stream:
        known_families = {row["case_id"]: row["family"] for row in csv.DictReader(stream)}
    _require(all(known_families[r["case_id"]] == r["family"] for r in rows), "Case family changed")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "frozen_outcome_readout_evaluation_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": fitted_report["status"],
        "target": TARGET,
        "no_refit": True,
        "input_sha256": hashes,
        "provenance": provenance,
        "feature_signature": signature,
        "fitted_manifest_sha256": manifest_hash,
        "frozen_artifact_sha256": artifact_hashes,
        "settings": fitted_report["settings"],
        "learning_classes": fitted_report["learning_classes"],
        "unsupported_classes": fitted_report["unsupported_classes"],
        "primary_three_class_status": fitted_report["primary_three_class_status"],
        "counts": _counts(rows),
        "methods": {},
        "limitations": fitted_report["limitations"],
    }
    ledger = [
        {
            **{key: r[key] for key in REQUIRED_COLUMNS[:6]},
            "excluded": r["decision"] == "INVALID",
            "exclusion_reason": "INVALID_reader_output" if r["decision"] == "INVALID" else "",
            "used_for_training": False,
        }
        for r in rows
    ]
    ledger_fields = [*REQUIRED_COLUMNS[:6], "excluded", "exclusion_reason", "used_for_training"]
    _csv(output_dir / "row_manifest.csv", ledger_fields, ledger)
    _csv(output_dir / "exclusions.csv", ledger_fields, [r for r in ledger if r["excluded"]])
    predictions = []
    groups = sorted({r["objective"] for r in rows})
    for method, fitted in fitted_report["methods"].items():
        if "artifact" not in fitted:
            report["methods"][method] = dict(fitted)
            continue
        artifact = fitted["artifact"]
        _require(artifact in artifact_hashes, "Unmanifested frozen model")
        actual_method, probabilities = _frozen_probabilities(fitted_dir / artifact, X, rows)
        _require(actual_method == method, "Frozen method identity mismatch")
        result = {
            "status": "scored_frozen",
            "source_artifact_sha256": artifact_hashes[artifact],
            "evaluations": {},
        }
        for objective in groups:
            indices = np.asarray(
                [
                    i
                    for i, r in enumerate(rows)
                    if r["objective"] == objective and r["decision"] != "INVALID"
                ],
                dtype=np.int64,
            )
            group = "test/" + objective
            if not len(indices):
                result["evaluations"][group] = {"status": "no_valid_labels", "n_valid_rows": 0}
                continue
            labels = [CLASSES.index(rows[i]["decision"]) for i in indices]
            result["evaluations"][group] = decision_metrics(labels, probabilities[indices])
            for index in indices:
                predictions.append(
                    {
                        "method": method,
                        **{key: rows[index][key] for key in REQUIRED_COLUMNS[:6]},
                        "predicted_decision": CLASSES[int(probabilities[index].argmax())],
                        **{
                            f"p_{name}": float(probabilities[index, j])
                            for j, name in enumerate(CLASSES)
                        },
                    }
                )
        report["methods"][method] = result
    if fitted_report["status"] == "not_trainable":
        report["reason"] = fitted_report["reason"]
    _require(
        hashes == {"features_sha256": _hash(features_path), "labels_sha256": _hash(labels_path)},
        "Frozen evaluation inputs changed during scoring",
    )
    _require(_hash(manifest_path) == manifest_hash, "Fitted manifest changed during frozen scoring")
    _require(
        all(_hash(fitted_dir / name) == digest for name, digest in artifact_hashes.items()),
        "Frozen source artifact changed during scoring",
    )
    _csv(
        output_dir / "predictions.csv",
        ["method", *REQUIRED_COLUMNS[:6], "predicted_decision", *[f"p_{name}" for name in CLASSES]],
        predictions,
    )
    _json(output_dir / "report.json", report)
    _json(
        output_dir / "manifest.json",
        {
            "schema": "frozen_outcome_readout_evaluation_manifest_v1",
            "status": report["status"],
            "no_refit": True,
            "input_sha256": hashes,
            "fitted_manifest_sha256": manifest_hash,
            "frozen_artifact_sha256": artifact_hashes,
            "source_code_sha256": _hash(Path(__file__)),
            "output_sha256": {
                path.name: _hash(path) for path in sorted(output_dir.iterdir()) if path.is_file()
            },
        },
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-two-class-diagnostic", action="store_true")
    parser.add_argument(
        "--fitted", type=Path, help="Score test cases with saved weights, without fitting"
    )
    args = parser.parse_args(argv)
    if args.fitted is not None:
        _require(
            not args.allow_two_class_diagnostic, "Diagnostic policy is fixed by the fitted readout"
        )
        report = evaluate_frozen_readout(args.features, args.labels, args.fitted, args.output)
    else:
        report = fit_evaluate_readout(
            args.features,
            args.labels,
            args.output,
            allow_two_class_diagnostic=args.allow_two_class_diagnostic,
        )
    print(json.dumps({"status": report["status"], "target": TARGET, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
