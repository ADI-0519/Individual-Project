"""Numerical Apollo instruction-pair baseline, without model loading or pickle.

Training labels describe the honest/dishonest *instruction condition*. They are
not observations of deceptive behavior. Training uses individual token rows;
response scores are means of raw standardized linear margins.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

INSTRUCTION_TARGET = "instructed_honesty_condition"
UPSTREAM_COMMIT = "f8ec4010e74927394709dffa22b97bdf8cd5a62f"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signature(value: str) -> str:
    try:
        decoded = json.loads(value)
        if not isinstance(decoded, dict) or not decoded:
            raise ValueError("expected a nonempty JSON object")
        return json.dumps(decoded, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("feature_signature must describe a nonempty finite JSON object.") from exc


def _load_features(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    with np.load(path, allow_pickle=False) as archive:
        if not {"X", "example_ids", "feature_signature"}.issubset(archive.files):
            raise ValueError("Token NPZ requires X, example_ids and feature_signature.")
        original = np.asarray(archive["X"])
        if original.dtype.kind not in "fiu":
            raise ValueError("X must have a numeric dtype.")
        X = np.asarray(original, dtype=np.float64)
        ids = np.asarray(archive["example_ids"])
        raw_signature = np.asarray(archive["feature_signature"])
    if X.ndim != 2 or not all(X.shape) or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty finite two-dimensional token matrix.")
    if ids.ndim != 1 or ids.dtype.kind != "U" or len(ids) != len(X):
        raise ValueError("example_ids must be a Unicode vector with one ID per token row.")
    if any(not identifier or identifier != identifier.strip() for identifier in ids.tolist()):
        raise ValueError("Example IDs must be nonempty and have no surrounding whitespace.")
    seen: set[str] = set()
    previous: str | None = None
    for identifier in ids.tolist():
        if identifier != previous:
            if identifier in seen:
                raise ValueError("All token rows for an example must form one contiguous group.")
            seen.add(identifier)
            previous = identifier
    if raw_signature.ndim != 0 or raw_signature.dtype.kind != "U":
        raise ValueError("feature_signature must be a scalar Unicode JSON string.")
    return X, ids, _signature(str(raw_signature.item()))


def _load_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {"example_id", "pair_id", "split", "label", "label_target"}
        if len(set(fields)) != len(fields) or not required.issubset(fields):
            raise ValueError(f"Metadata requires unique columns including {sorted(required)}.")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            if None in row:
                raise ValueError("Metadata CSV contains extra fields in a row.")
            cleaned = {key: (value or "").strip() for key, value in row.items()}
            identifier = cleaned["example_id"]
            if not identifier or identifier in rows:
                raise ValueError("Metadata example_id values must be nonempty and unique.")
            if not cleaned["pair_id"]:
                raise ValueError(f"Missing pair_id for {identifier!r}.")
            if cleaned["split"] not in {"train", "validation", "evaluation", "control"}:
                raise ValueError(f"Invalid explicit split for {identifier!r}.")
            rows[identifier] = cleaned
    if not rows:
        raise ValueError("Metadata CSV is empty.")
    return rows


def _match_ids(ids: set[str], rows: dict[str, dict[str, str]]) -> None:
    if ids != set(rows):
        raise ValueError(
            "Features/scores and metadata require an exact ID match; "
            f"missing={sorted(ids - set(rows))[:5]}, extra={sorted(set(rows) - ids)[:5]}."
        )


def _binary_label(row: dict[str, str]) -> int:
    if row["label"] not in {"0", "1"}:
        raise ValueError(f"An explicit binary label is required for {row['example_id']!r}.")
    return int(row["label"])


def _aggregate(ids: np.ndarray, margins: np.ndarray) -> list[dict[str, Any]]:
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(ids)]
    return [
        {
            "example_id": str(ids[start]),
            "score": float(margins[start:end].mean()),
            "n_tokens": int(end - start),
        }
        for start, end in zip(starts, ends, strict=True)
    ]


def _auc(labels: list[int], scores: list[float]) -> float | None:
    return float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None


def _json_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def _read_sidecar(path: Path) -> dict[str, Any]:
    companion = _sidecar(path)
    if not companion.exists():
        return {}
    payload = json.loads(companion.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Score manifest must be a JSON object.")
    if payload.get("scores_sha256") != _sha256(path):
        raise ValueError("Score CSV does not match its recorded manifest hash.")
    # A CSV checksum alone cannot verify probe identity or the overlap ledger.
    if payload.get("schema") != "apollo_scores_v1":
        raise ValueError("Score manifest has an incomplete or unsupported schema.")
    for key in ("features_sha256", "probe_sha256", "scores_sha256"):
        value = payload.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError(f"Score manifest is incomplete: invalid {key}.")
    _signature(payload.get("feature_signature", ""))
    if payload.get("score_aggregation") != "mean_raw_margin":
        raise ValueError("Score manifest has an unsupported aggregation.")
    for key in (
        "train_example_ids",
        "validation_example_ids",
        "train_pair_ids",
        "validation_pair_ids",
    ):
        values = payload.get(key)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"Score manifest is incomplete: invalid {key}.")
    for key in ("n_examples", "n_tokens"):
        if type(payload.get(key)) is not int or payload[key] <= 0:
            raise ValueError(f"Score manifest is incomplete: invalid {key}.")
    if len(_load_scores(path)) != payload["n_examples"]:
        raise ValueError("Score manifest example count differs from its CSV.")
    return payload


def _verify_capture(features: Path, signature: str, labels: Path | None = None) -> dict[str, Any]:
    """Check a supplied Apollo capture ledger; generic numeric inputs stay explicit."""
    companion = features.parent / "manifest.json"
    if not companion.is_file():
        return {"verified": False, "reason": "No adjacent Apollo capture manifest."}
    payload = json.loads(companion.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("stage") != "apollo_tokens":
        return {"verified": False, "reason": "Adjacent manifest is not an Apollo capture."}
    if payload.get("status") != "complete":
        raise ValueError("Apollo capture must have status complete before training or scoring.")
    if payload.get("features_sha256") != _sha256(features):
        raise ValueError("Features differ from their Apollo capture hash.")
    if labels is not None and payload.get("metadata_sha256") != _sha256(labels):
        raise ValueError("Labels differ from their Apollo capture hash.")
    if _signature(payload.get("feature_signature", "")) != signature:
        raise ValueError("Feature signature differs from the Apollo capture manifest.")
    return {
        "verified": True,
        "manifest_sha256": _sha256(companion),
        "scope": "completed capture, artifact hashes and feature identity; not behavior labels",
    }


def train_apollo_probe(
    features_path: str | Path, labels_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Fit the source-recipe numeric readout on explicit training token rows."""
    features, labels, destination = Path(features_path), Path(labels_path), Path(output_dir)
    X, ids, signature = _load_features(features)
    capture_provenance = _verify_capture(features, signature, labels)
    metadata = _load_metadata(labels)
    _match_ids(set(ids.tolist()), metadata)
    if any(row["split"] not in {"train", "validation"} for row in metadata.values()):
        raise ValueError("Training input must be an explicit train/validation-only export.")
    if any(row["label_target"] != INSTRUCTION_TARGET for row in metadata.values()):
        raise ValueError(f"Training labels must declare label_target={INSTRUCTION_TARGET!r}.")
    by_split = {
        split: [identifier for identifier, row in metadata.items() if row["split"] == split]
        for split in ("train", "validation")
    }
    if not all(by_split.values()):
        raise ValueError("Explicit nonempty train and validation partitions are required.")
    pairs = {
        split: {metadata[identifier]["pair_id"] for identifier in selected}
        for split, selected in by_split.items()
    }
    if pairs["train"] & pairs["validation"]:
        raise ValueError("pair_id leakage between train and validation partitions.")
    for split, selected_pairs in pairs.items():
        for pair in selected_pairs:
            observed = [
                _binary_label(metadata[identifier])
                for identifier in by_split[split]
                if metadata[identifier]["pair_id"] == pair
            ]
            if sorted(observed) != [0, 1]:
                raise ValueError(
                    f"{split.capitalize()} pair {pair!r} must contain labels 0 and 1 "
                    "exactly once each."
                )
    for row in metadata.values():
        _binary_label(row)
    token_labels = np.asarray([_binary_label(metadata[identifier]) for identifier in ids])
    train_mask = np.asarray([metadata[identifier]["split"] == "train" for identifier in ids])
    scaler = StandardScaler().fit(X[train_mask])
    estimator = LogisticRegression(
        C=0.1, fit_intercept=False, random_state=42, solver="lbfgs", max_iter=1000
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            estimator.fit(scaler.transform(X[train_mask]), token_labels[train_mask])
    except ConvergenceWarning as exc:
        raise ValueError("Apollo logistic fitting did not converge; no probe was saved.") from exc
    coefficient = np.asarray(estimator.coef_[0], dtype=np.float64)
    margins = scaler.transform(X) @ coefficient
    aggregate = _aggregate(ids, margins)
    validation = [row for row in aggregate if row["example_id"] in set(by_split["validation"])]
    manifest: dict[str, Any] = {
        "schema": "apollo_probe_v1",
        "capture_provenance": capture_provenance,
        "upstream_commit": UPSTREAM_COMMIT,
        "label_target": INSTRUCTION_TARGET,
        "feature_signature": signature,
        "feature_dimension": int(X.shape[1]),
        "features_sha256": _sha256(features),
        "labels_sha256": _sha256(labels),
        "train_example_ids": sorted(by_split["train"]),
        "validation_example_ids": sorted(by_split["validation"]),
        "train_pair_ids": sorted(pairs["train"]),
        "validation_pair_ids": sorted(pairs["validation"]),
        "n_training_token_rows": int(train_mask.sum()),
        "n_validation_token_rows": int((~train_mask).sum()),
        "settings": {
            "C": 0.1,
            "fit_intercept": False,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": 42,
            "numeric_dtype": "float64",
            "normalization": "StandardScaler fitted on training token rows only",
            "training_unit": "token",
            "score_aggregation": "mean_raw_margin",
        },
        "validation_diagnostic": {
            "n_examples": len(validation),
            "n_positive": sum(_binary_label(metadata[row["example_id"]]) for row in validation),
            "n_negative": sum(1 - _binary_label(metadata[row["example_id"]]) for row in validation),
            "roc_auc": _auc(
                [_binary_label(metadata[row["example_id"]]) for row in validation],
                [row["score"] for row in validation],
            ),
            "target": "held-out instruction condition, not observed deceptive behavior",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "limitations": [
            "This numerical recipe does not reproduce the original 70B model results.",
            "Checkpoint, tokenizer, precision, layer and masking adaptations are in feature_signature.",
            "Raw margins are not calibrated probabilities of deception or recipient belief.",
            "max_iter=1000 changes the solver iteration allowance, not the fitted objective.",
            "Fitting/scoring uses float64 arithmetic, not bitwise parity with upstream float32.",
        ],
    }
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError("Probe output must be a new or empty directory.")
    destination.mkdir(parents=True, exist_ok=True)
    probe_path = destination / "probe.npz"
    np.savez_compressed(
        probe_path,
        mean=np.asarray(scaler.mean_),
        scale=np.asarray(scaler.scale_),
        coefficient=coefficient,
        intercept=np.asarray(0.0),
    )
    manifest["probe_sha256"] = _sha256(probe_path)
    _json_output(destination / "manifest.json", manifest)
    return manifest


def score_apollo_probe(
    features_path: str | Path, probe_dir: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Average raw standardized margins over each contiguous example span."""
    features, directory, output = Path(features_path), Path(probe_dir), Path(output_path)
    X, ids, signature = _load_features(features)
    capture_provenance = _verify_capture(features, signature)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if signature != _signature(manifest.get("feature_signature", "")):
        raise ValueError("Feature identity differs from the fitted probe; scoring refused.")
    probe_path = directory / "probe.npz"
    if _sha256(probe_path) != manifest.get("probe_sha256"):
        raise ValueError("Probe NPZ does not match its manifest hash.")
    with np.load(probe_path, allow_pickle=False) as probe:
        mean = np.asarray(probe["mean"], dtype=np.float64)
        scale = np.asarray(probe["scale"], dtype=np.float64)
        coefficient = np.asarray(probe["coefficient"], dtype=np.float64)
        intercept = np.asarray(probe["intercept"], dtype=np.float64)
    if any(value.shape != (X.shape[1],) for value in (mean, scale, coefficient)):
        raise ValueError("Probe feature dimensions differ from the token matrix.")
    if not all(np.isfinite(value).all() for value in (mean, scale, coefficient, intercept)):
        raise ValueError("Probe contains nonfinite numeric values.")
    if np.any(scale <= 0) or intercept.shape != () or float(intercept) != 0.0:
        raise ValueError("Probe must have positive scales and a zero scalar intercept.")
    rows = _aggregate(ids, ((X - mean) / scale) @ coefficient)
    if not all(math.isfinite(row["score"]) for row in rows):
        raise ValueError("Scoring produced nonfinite response scores.")
    if output.exists() or _sidecar(output).exists():
        raise FileExistsError("Score output or its manifest already exists.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["example_id", "score", "n_tokens"])
        writer.writeheader()
        writer.writerows(rows)
    score_manifest = {
        "schema": "apollo_scores_v1",
        "capture_provenance": capture_provenance,
        "n_examples": len(rows),
        "n_tokens": len(ids),
        "feature_signature": signature,
        "features_sha256": _sha256(features),
        "probe_sha256": manifest["probe_sha256"],
        "scores_sha256": _sha256(output),
        "score_aggregation": "mean_raw_margin",
        **{
            key: manifest[key]
            for key in (
                "train_example_ids",
                "validation_example_ids",
                "train_pair_ids",
                "validation_pair_ids",
            )
        },
    }
    _json_output(_sidecar(output), score_manifest)
    return score_manifest


def _load_scores(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields) or not {"example_id", "score"}.issubset(fields):
            raise ValueError("Scores CSV requires unique example_id and score columns.")
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in reader:
            identifier = (row.get("example_id") or "").strip()
            if not identifier or identifier in seen:
                raise ValueError("Score example_id values must be nonempty and unique.")
            try:
                score = float(row["score"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric score for {identifier!r}.") from exc
            if not math.isfinite(score):
                raise ValueError("Score values must be finite.")
            rows.append({"example_id": identifier, "score": score})
            seen.add(identifier)
    if not rows:
        raise ValueError("Scores CSV is empty.")
    return rows


def calibrate_control_threshold(
    scores_path: str | Path, output_path: str | Path, quantile: float = 0.99
) -> dict[str, Any]:
    """Fix a neutral-control quantile; this does not guarantee shifted FPR."""
    if not math.isfinite(quantile) or not 0 < quantile < 1:
        raise ValueError("quantile must be finite and strictly between zero and one.")
    path, output = Path(scores_path), Path(output_path)
    rows = _load_scores(path)
    provenance = _read_sidecar(path)
    ids = {row["example_id"] for row in rows}
    fitted_ids = set(provenance.get("train_example_ids", [])) | set(
        provenance.get("validation_example_ids", [])
    )
    if ids & fitted_ids:
        raise ValueError("Control calibration overlaps probe training/validation example IDs.")
    scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
    threshold = float(np.quantile(scores, quantile, method="linear"))
    n_above = int(np.sum(scores > threshold))
    report: dict[str, Any] = {
        "schema": "apollo_control_threshold_v1",
        "threshold": threshold,
        "quantile": float(quantile),
        "quantile_method": "linear",
        "comparison": "score > threshold",
        "n_control": len(rows),
        "n_control_strictly_above": n_above,
        "empirical_calibration_fpr": n_above / len(rows),
        "control_example_ids": sorted(ids),
        "scores_sha256": _sha256(path),
        "feature_signature": provenance.get("feature_signature"),
        "probe_sha256": provenance.get("probe_sha256"),
        "provenance_verified": bool(provenance),
        "warnings": ["The calibration quantile does not guarantee FPR on new or shifted data."],
    }
    if len(rows) * (1 - quantile) < 10:
        report["warnings"].append(
            "Fewer than ten expected observations lie above the requested quantile; "
            "this tail estimate is an unreliable pilot calibration."
        )
    if not provenance:
        report["warnings"].append(
            "No calibration score sidecar: feature/probe identity and training overlap are unverified."
        )
    _json_output(output, report)
    return report


def evaluate_apollo_scores(
    scores_path: str | Path,
    labels_path: str | Path,
    output_path: str | Path,
    threshold_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate audited behavior labels, never instruction labels as deception."""
    path, labels, output = Path(scores_path), Path(labels_path), Path(output_path)
    rows, metadata = _load_scores(path), _load_metadata(labels)
    ids = {row["example_id"] for row in rows}
    _match_ids(ids, metadata)
    targets = {row["label_target"] for row in metadata.values()}
    if len(targets) != 1 or not next(iter(targets)) or INSTRUCTION_TARGET in targets:
        raise ValueError(
            "Evaluation requires one explicit behavior target, not instruction labels."
        )
    if any(row["label_target"] == "manual_definition_required" for row in metadata.values()):
        raise ValueError("Replace manual label-target placeholders before evaluation.")
    if any(row["split"] != "evaluation" for row in metadata.values()):
        raise ValueError("Empirical evaluation requires an explicit evaluation-only export.")
    if any(row.get("label_audited", "").lower() not in {"true", "1"} for row in metadata.values()):
        raise ValueError("Each evaluation label requires explicit label_audited=true or 1.")
    y = np.asarray([_binary_label(metadata[row["example_id"]]) for row in rows])
    scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
    provenance = _read_sidecar(path)
    fitted_ids = set(provenance.get("train_example_ids", [])) | set(
        provenance.get("validation_example_ids", [])
    )
    fitted_pairs = set(provenance.get("train_pair_ids", [])) | set(
        provenance.get("validation_pair_ids", [])
    )
    if ids & fitted_ids or {row["pair_id"] for row in metadata.values()} & fitted_pairs:
        raise ValueError("Evaluation overlaps probe training/validation examples or pair groups.")
    report: dict[str, Any] = {
        "schema": "apollo_evaluation_v1",
        "label_target": next(iter(targets)),
        "label_audit": "explicit user declaration; not independently verified by this software",
        "n_examples": len(rows),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "n_groups": len({row["pair_id"] for row in metadata.values()}),
        "roc_auc": _auc(y.tolist(), scores.tolist()),
        "scores_sha256": _sha256(path),
        "labels_sha256": _sha256(labels),
        "score_aggregation": "mean_raw_margin",
        "warnings": [
            "Raw margins are not calibrated probabilities; no Brier score is computed.",
            "Results apply only to the audited sample, model, target and feature profile.",
        ],
    }
    if not provenance:
        report["warnings"].append(
            "No score provenance sidecar: training-feature identity and overlap cannot be verified."
        )
    if threshold_path is not None:
        threshold_file = Path(threshold_path)
        calibration = json.loads(threshold_file.read_text(encoding="utf-8"))
        if not calibration.get("feature_signature") or not calibration.get("probe_sha256"):
            report["warnings"].append(
                "Calibration provenance is incomplete; threshold identity compatibility is unverified."
            )
        if calibration.get("comparison") != "score > threshold":
            raise ValueError("Threshold must declare strict score > threshold comparison.")
        control_ids = calibration.get("control_example_ids")
        if (
            not isinstance(control_ids, list)
            or not control_ids
            or len(set(control_ids)) != len(control_ids)
        ):
            raise ValueError("Threshold requires unique nonempty control example IDs.")
        if ids & set(control_ids):
            raise ValueError("Evaluation example IDs overlap control calibration IDs.")
        for key in ("feature_signature", "probe_sha256"):
            if calibration.get(key) is not None and provenance.get(key) != calibration[key]:
                raise ValueError("Threshold and evaluation feature/probe identity differ.")
        threshold = float(calibration["threshold"])
        if not math.isfinite(threshold):
            raise ValueError("Threshold must be finite.")
        predictions = scores > threshold
        tp, fp = int(np.sum(predictions & (y == 1))), int(np.sum(predictions & (y == 0)))
        fn, tn = int(np.sum(~predictions & (y == 1))), int(np.sum(~predictions & (y == 0)))
        report.update(
            threshold=threshold,
            comparison="score > threshold",
            true_positive_rate=tp / (tp + fn) if tp + fn else None,
            false_positive_rate=fp / (fp + tn) if fp + tn else None,
            confusion={"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            threshold_sha256=_sha256(threshold_file),
            n_calibration_control=len(control_ids),
        )
    _json_output(output, report)
    return report
