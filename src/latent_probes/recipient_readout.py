"""Isolated, validation-selected forecasts of a fixed recipient policy's decisions.

No model inference occurs here. Source excerpts and fresh natural reports retain
different provenance. The original outcome_readout evaluator is unchanged.
This is a conditional software asset: research-data collection and fitting require
the separate fresh recipient capability gate to pass. Unit-test fixtures are not
evidence that this gate or the proposed research experiment has been executed.
"""

from __future__ import annotations

import json
import platform
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .outcome_readout import (
    CLASSES,
    REQUIRED_COLUMNS,
    _csv,
    _hash,
    _json,
    _json_scalar,
    _read_inputs,
    _save_model,
    decision_metrics,
)
from .storage import canonical_hash

METHODS = ("activation", "full_text", "goal_stripped_text")
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
SEED = 42
MAX_ITER = 2000
LABEL_COLUMNS = (*REQUIRED_COLUMNS, "goal_stripped_prompt_text")
TARGET = "actual_calibrated_recipient_decision_after_known_report"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def label_row_sha256(row):
    """Bind ordered feature IDs to the exact CSV strings supplied by the exporter."""
    _require(
        all(isinstance(row.get(key), str) for key in LABEL_COLUMNS), "Label fields must be strings"
    )
    return canonical_hash({key: row[key] for key in LABEL_COLUMNS})


def _inputs(features_path, labels_path, *, source):
    X, rows, provenance, signature, hashes = _read_inputs(
        Path(features_path), Path(labels_path), require_all_splits=source
    )
    _require(
        all(set(LABEL_COLUMNS) <= set(row) for row in rows), "Missing goal-stripped prompt column"
    )
    _require(
        all(row["goal_stripped_prompt_text"].strip() for row in rows),
        "Goal-stripped prompts cannot be empty",
    )
    policy = provenance.get("reader_policy_sha256")
    _require(
        isinstance(policy, str) and re.fullmatch(r"[0-9a-f]{64}", policy) is not None,
        "A frozen recipient-policy SHA256 is required",
    )
    bindings = provenance.get("label_row_sha256")
    _require(
        isinstance(bindings, dict) and set(bindings) == {row["example_id"] for row in rows},
        "Feature/label binding IDs differ",
    )
    _require(
        all(bindings[row["example_id"]] == label_row_sha256(row) for row in rows),
        "Feature/label row binding mismatch",
    )
    return X, rows, provenance, signature, hashes


def _partition_check(rows):
    _require(
        all(row["objective"] == "neutral" for row in rows),
        "Source fitting accepts neutral rows only",
    )
    counts = Counter(row["split"] for row in rows)
    _require(
        counts == {"train": 144, "validation": 36, "test": 36},
        "Expected source row split 144/36/36",
    )
    cases = {
        split: sorted({row["case_id"] for row in rows if row["split"] == split})
        for split in ("train", "validation", "test")
    }
    _require(
        [len(cases[split]) for split in cases] == [16, 4, 4], "Expected source case split 16/4/4"
    )
    _require(
        set(Counter(row["case_id"] for row in rows).values()) == {9},
        "Every source case must retain nine states",
    )
    return cases


def _texts(rows, method):
    field = "goal_stripped_prompt_text" if method == "goal_stripped_text" else "prompt_text"
    return [row[field] + "\n[REPORT]\n" + row["report_text"] for row in rows]


def _vectorizer(*, vocabulary=None):
    return TfidfVectorizer(
        vocabulary=vocabulary,
        ngram_range=(1, 2),
        max_features=20000,
        min_df=1,
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=True,
        norm="l2",
        smooth_idf=True,
        dtype=np.float64,
    )


def _valid(rows, *, split=None, objective=None):
    return np.asarray(
        [
            i
            for i, row in enumerate(rows)
            if row["decision"] != "INVALID"
            and (split is None or row["split"] == split)
            and (objective is None or row["objective"] == objective)
        ],
        dtype=np.int64,
    )


def _metrics(rows, probabilities, indices):
    if not len(indices):
        return {"status": "no_valid_targets", "n_valid_rows": 0}
    return decision_metrics(
        [CLASSES.index(rows[i]["decision"]) for i in indices], probabilities[indices]
    )


def _empty_output(path):
    path = Path(path)
    _require(
        not path.exists() or (path.is_dir() and not any(path.iterdir())),
        "Output must be absent or empty",
    )
    return path


def _input_unchanged(features_path, labels_path, hashes):
    _require(
        hashes
        == {
            "features_sha256": _hash(Path(features_path)),
            "labels_sha256": _hash(Path(labels_path)),
        },
        "Readout inputs changed during execution",
    )


def _save_predictions(path, rows, predictions):
    values = [
        {
            "method": method,
            **{key: row[key] for key in REQUIRED_COLUMNS[:6]},
            "predicted_decision": CLASSES[int(probabilities.argmax())],
            **{f"p_{name}": float(probabilities[i]) for i, name in enumerate(CLASSES)},
        }
        for method, matrix in predictions.items()
        for row, probabilities in zip(rows, matrix, strict=True)
    ]
    _csv(
        path,
        ["method", *REQUIRED_COLUMNS[:6], "predicted_decision", *[f"p_{name}" for name in CLASSES]],
        values,
    )


def fit_recipient_readouts(features_path, labels_path, output_dir):
    """Fit on 144 source rows; select each C on 36 validation rows only.

    NPZ uses the original X/example_ids/provenance_json/feature_signature schema.
    Provenance requires reader_policy_sha256 and per-ID label_row_sha256. CSV adds
    goal_stripped_prompt_text to the original required outcome label columns.
    """
    output = _empty_output(output_dir)
    X, rows, provenance, signature, hashes = _inputs(features_path, labels_path, source=True)
    cases = _partition_check(rows)
    train, validation = _valid(rows, split="train"), _valid(rows, split="validation")
    train_labels = np.asarray([CLASSES.index(rows[i]["decision"]) for i in train])
    missing = [name for name in CLASSES if name not in {rows[i]["decision"] for i in train}]
    _require(not missing, "Missing training classes: " + ", ".join(missing))
    _require(len(validation) > 0, "No valid neutral validation labels")
    settings = {
        "target": TARGET,
        "classes": list(CLASSES),
        "C_grid": list(C_GRID),
        "selection": "validation multiclass log loss, then smaller C on exact tie",
        "solver": "lbfgs",
        "max_iter": MAX_ITER,
        "seed": SEED,
        "fit_partition": "valid neutral training rows only; no train-plus-validation refit",
        "normalization": "train-only StandardScaler or train-only TF-IDF",
        "text_separator": "\n[REPORT]\n",
        "methods": list(METHODS),
    }
    report = {
        "schema": "recipient_readout_report_v1",
        "status": "complete",
        "target": TARGET,
        "settings": settings,
        "feature_signature": signature,
        "input_sha256": hashes,
        "provenance": provenance,
        "source_case_partitions": cases,
        "reader_policy_sha256": provenance["reader_policy_sha256"],
        "features_shape": list(X.shape),
        "methods": {},
        "counts": {
            split: {
                "n_rows": sum(row["split"] == split for row in rows),
                "n_valid_rows": len(_valid(rows, split=split)),
                "class_counts": dict(
                    Counter(rows[i]["decision"] for i in _valid(rows, split=split))
                ),
            }
            for split in cases
        },
        "missing_validation_classes": [
            name for name in CLASSES if name not in {rows[i]["decision"] for i in validation}
        ],
        "limits": [
            "Source cases are clustered development data; source test is a diagnostic, not fresh confirmation.",
            "Fixed source-supported continuations are off-policy; recipient outcomes are measured labels, not normative replacements.",
            "TF-IDF controls do not establish superiority over capable text monitors or exact recipient simulation.",
            "INVALID targets remain in the ledger and are excluded from fitting and target metrics.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    predictions = {}
    for method in METHODS:
        scaler, vectorizer = None, None
        if method == "activation":
            scaler = StandardScaler().fit(X[train])
            matrix = scaler.transform(X)
        else:
            texts = _texts(rows, method)
            vectorizer = _vectorizer()
            vectorizer.fit([texts[i] for i in train])
            matrix = vectorizer.transform(texts)
        candidates, fits = [], []
        for c_value in C_GRID:
            model = LogisticRegression(
                C=c_value, solver="lbfgs", max_iter=MAX_ITER, random_state=SEED
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(matrix[train], train_labels)
            _require(
                np.array_equal(model.classes_, np.arange(3)), "Unexpected fitted class ordering"
            )
            probabilities = model.predict_proba(matrix[validation])
            metrics = decision_metrics(
                [CLASSES.index(rows[i]["decision"]) for i in validation], probabilities
            )
            candidates.append(
                {
                    "C": c_value,
                    "validation": metrics,
                    "converged": not any(
                        issubclass(item.category, ConvergenceWarning) for item in caught
                    ),
                    "n_iterations": model.n_iter_.tolist(),
                }
            )
            fits.append(model)
        selected = min(
            range(len(candidates)),
            key=lambda i: (candidates[i]["validation"]["multiclass_log_loss"], candidates[i]["C"]),
        )
        model, selection = fits[selected], candidates[selected]
        artifact = method + ".npz"
        _save_model(
            output / artifact,
            model,
            {**settings, "method": method, "selected_C": selection["C"]},
            scaler,
            vectorizer,
        )
        probabilities = model.predict_proba(matrix)
        predictions[method] = probabilities
        report["methods"][method] = {
            "artifact": artifact,
            "artifact_sha256": _hash(output / artifact),
            "selected_C": selection["C"],
            "selection_converged": selection["converged"],
            "validation_candidates": candidates,
            "evaluations": {
                split: _metrics(rows, probabilities, _valid(rows, split=split)) for split in cases
            },
        }
    _input_unchanged(features_path, labels_path, hashes)
    _save_predictions(output / "predictions.csv", rows, predictions)
    _csv(
        output / "row_manifest.csv",
        [*REQUIRED_COLUMNS[:6], "excluded", "exclusion_reason"],
        [
            {
                **{key: row[key] for key in REQUIRED_COLUMNS[:6]},
                "excluded": row["decision"] == "INVALID",
                "exclusion_reason": "INVALID_reader_output" if row["decision"] == "INVALID" else "",
            }
            for row in rows
        ],
    )
    _json(output / "report.json", report)
    _json(
        output / "manifest.json",
        {
            "schema": "recipient_readout_manifest_v1",
            "status": "complete",
            "input_sha256": hashes,
            "source_code_sha256": _hash(Path(__file__)),
            "output_sha256": {
                path.name: _hash(path) for path in output.iterdir() if path.is_file()
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
            },
        },
    )
    return report


def _score_artifact(path, X, rows):
    """Portable numeric reconstruction; no estimator fitting or pickle loading."""
    with np.load(path, allow_pickle=False) as archive:
        settings = _json_scalar(archive, "settings_json")
        method = settings.get("method")
        _require(method in METHODS, "Unknown recipient readout method")
        weights, intercept = archive["coefficients"], archive["intercept"]
        _require(
            np.array_equal(archive["class_indices"], np.arange(3)), "Frozen class order differs"
        )
        _require(
            weights.ndim == 2
            and weights.shape[0] == 3
            and intercept.shape == (3,)
            and np.isfinite(weights).all()
            and np.isfinite(intercept).all(),
            "Invalid frozen coefficients",
        )
        if method == "activation":
            mean, scale = archive["mean"], archive["scale"]
            _require(
                mean.shape == scale.shape == (X.shape[1],)
                and np.isfinite(mean).all()
                and np.isfinite(scale).all()
                and (scale > 0).all(),
                "Frozen scaler mismatch",
            )
            matrix = (X - mean) / scale
        else:
            terms, idf = archive["vocabulary_terms"], archive["idf"]
            _require(
                terms.ndim == 1
                and terms.dtype.kind == "U"
                and len(set(terms.tolist())) == len(terms)
                and idf.shape == terms.shape
                and np.isfinite(idf).all()
                and (idf > 0).all(),
                "Invalid frozen vocabulary/IDF",
            )
            vectorizer = _vectorizer(vocabulary={term: i for i, term in enumerate(terms.tolist())})
            vectorizer.idf_ = idf
            matrix = vectorizer.transform(_texts(rows, method))
        _require(matrix.shape[1] == weights.shape[1], "Frozen feature width differs")
        logits = np.asarray(matrix @ weights.T) + intercept
        _require(np.isfinite(logits).all(), "Nonfinite frozen scores")
        return method, np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def evaluate_recipient_readouts(
    features_path, labels_path, fitted_dir, output_dir, *, expected_case_ids, forbidden_case_ids=()
):
    """Score declared fresh ct cases, rejecting overlap with every source-fit case."""
    fitted_dir, output = Path(fitted_dir), _empty_output(output_dir)
    _require(
        fitted_dir.resolve() != output.resolve(), "Fresh output cannot replace fitted artifacts"
    )
    manifest_hash = _hash(fitted_dir / "manifest.json")
    manifest = json.loads((fitted_dir / "manifest.json").read_text(encoding="utf-8"))
    _require(
        manifest.get("schema") == "recipient_readout_manifest_v1"
        and manifest.get("status") == "complete",
        "Unrecognized completed recipient fit",
    )
    bindings = manifest.get("output_sha256")
    _require(
        isinstance(bindings, dict) and "report.json" in bindings, "Missing frozen artifact hashes"
    )
    for name, digest in bindings.items():
        _require(
            Path(name).name == name and name not in {"", ".", ".."}, "Unsafe frozen artifact path"
        )
        _require(_hash(fitted_dir / name) == digest, f"Frozen artifact hash mismatch: {name}")
    source = json.loads((fitted_dir / "report.json").read_text(encoding="utf-8"))
    X, rows, provenance, signature, hashes = _inputs(features_path, labels_path, source=False)
    _require(
        canonical_hash(signature) == canonical_hash(source["feature_signature"]),
        "Feature signature mismatch",
    )
    _require(X.shape[1] == source["features_shape"][1], "Feature width mismatch")
    _require(
        provenance["reader_policy_sha256"] == source["reader_policy_sha256"],
        "Recipient policy mismatch",
    )
    expected = set(expected_case_ids)
    actual = {row["case_id"] for row in rows}
    forbidden = set(forbidden_case_ids).union(*map(set, source["source_case_partitions"].values()))
    _require(
        len(expected) == 24 and actual == expected,
        "Fresh ct evaluation requires exactly the declared 24 cases",
    )
    _require(not actual & forbidden, "Fresh cases overlap source or forbidden capability cases")
    _require(
        len(rows) == 144 and set(Counter(row["case_id"] for row in rows).values()) == {6},
        "Fresh ct requires six rows per case and 144 rows total",
    )
    for case in actual:
        _require(
            Counter(row["objective"] for row in rows if row["case_id"] == case)
            == {"neutral": 3, "favorable": 3},
            "Each fresh case requires matched neutral/favorable contexts",
        )
    report = {
        "schema": "fresh_recipient_readout_evaluation_v1",
        "status": "complete",
        "no_refit": True,
        "target": TARGET,
        "source_manifest_sha256": manifest_hash,
        "input_sha256": hashes,
        "feature_signature": signature,
        "reader_policy_sha256": source["reader_policy_sha256"],
        "provenance": provenance,
        "expected_case_ids": sorted(expected),
        "methods": {},
    }
    predictions = {}
    for method in METHODS:
        artifact = source["methods"][method]["artifact"]
        _require(artifact in bindings, "Unmanifested selected fit")
        actual_method, probabilities = _score_artifact(fitted_dir / artifact, X, rows)
        _require(actual_method == method, "Frozen method identity mismatch")
        predictions[method] = probabilities
        report["methods"][method] = {
            "selected_C": source["methods"][method]["selected_C"],
            "artifact_sha256": bindings[artifact],
            "evaluations": {
                goal: _metrics(rows, probabilities, _valid(rows, objective=goal))
                for goal in ("neutral", "favorable")
            },
        }
    _input_unchanged(features_path, labels_path, hashes)
    _require(_hash(fitted_dir / "manifest.json") == manifest_hash, "Fitted manifest changed")
    _require(
        all(_hash(fitted_dir / name) == digest for name, digest in bindings.items()),
        "Fitted artifacts changed",
    )
    output.mkdir(parents=True, exist_ok=True)
    _save_predictions(output / "predictions.csv", rows, predictions)
    _json(output / "report.json", report)
    _json(
        output / "manifest.json",
        {
            "schema": "fresh_recipient_readout_manifest_v1",
            "status": "complete",
            "no_refit": True,
            "source_manifest_sha256": manifest_hash,
            "input_sha256": hashes,
            "output_sha256": {
                path.name: _hash(path) for path in output.iterdir() if path.is_file()
            },
        },
    )
    return report
