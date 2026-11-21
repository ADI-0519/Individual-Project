"""Frozen, train-only matched probe fitting for the main belief study.

This CPU module never calls a language model. Test rows are rejected before
selection, scaling, vocabulary building, calibration or model fitting. Safe
JSON/NPZ bundles retain every selected source ID and candidate failure.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize

from .main_study_design import (
    ALLOCATION_SEEDS,
    TRAINING_SIZES,
    BeliefCase,
    TrainingGroup,
    matched_training_allocation,
    one_step_states,
)
from .storage import atomic_json, atomic_npz, canonical_hash, environment_info, file_sha256

C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
NUMERICAL_SEED = 20261121
TFIDF_SETTINGS = {
    "lowercase": True,
    "analyzer": "word",
    "token_pattern": r"(?u)\b\w+\b",
    "ngram_range": (1, 2),
    "min_df": 1,
    "max_df": 1.0,
    "max_features": 50000,
    "sublinear_tf": True,
    "norm": "l2",
    "dtype": np.float64,
}


def _array_hash(values: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class FeatureCache:
    """An exact ordered capture table; no rows are silently aligned or dropped."""

    rows: tuple[dict, ...]
    features: np.ndarray
    provenance: dict
    test_fixture: bool
    source_hashes: dict

    def __post_init__(self) -> None:
        x = self.features
        if x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
            raise ValueError("features must be a nonempty finite 2-D matrix")
        if len(self.rows) != len(x):
            raise ValueError("One feature vector is required per row")
        if type(self.test_fixture) is not bool:
            raise ValueError("Explicit boolean test_fixture provenance is required")
        ids, groups = set(), {}
        for row, feature in zip(self.rows, x, strict=True):
            case = BeliefCase(**row["state"])
            if row.get("case_id") != case.case_id or row.get("row_id") != case.case_id:
                raise ValueError("row_id/case_id do not match the declared state")
            if case.case_id in ids:
                raise ValueError("Duplicate captured case ID")
            ids.add(case.case_id)
            if row.get("group_id") != case.group_id or row.get("mechanism") != case.mechanism:
                raise ValueError("State and row grouping disagree")
            if type(row.get("gold")) is not int or row["gold"] != case.gold:
                raise ValueError("Gold does not match the independent state rule")
            split = row.get("split")
            if split not in {"train", "development", "test"}:
                raise ValueError("Unknown row split")
            if groups.setdefault(case.group_id, split) != split:
                raise ValueError("A scenario group spans multiple splits")
            recorded = row.get("feature_sha256")
            if recorded is not None and recorded != _array_hash(feature):
                raise ValueError("Feature hash mismatch")
            values_hash = row.get("feature_values_sha256")
            if values_hash is not None and values_hash != canonical_hash(feature.tolist()):
                raise ValueError("Captured feature-value hash mismatch")
            if "test_fixture" in row and row["test_fixture"] != self.test_fixture:
                raise ValueError("Row and cache fixture status disagree")
            if "messages_sha256" in row and row["messages_sha256"] != canonical_hash(
                row.get("messages")
            ):
                raise ValueError("Captured messages hash mismatch")
            _nonempty(row.get("group_id"), "group_id")


def load_feature_cache(directory: str | Path) -> FeatureCache:
    """Read a complete, hash-bound capture with pickle explicitly disabled."""
    root = Path(directory)
    paths = {name: root / name for name in ("rows.json", "features.npz")}
    manifest = json.loads((root / "capture_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Capture is not complete")
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    recorded = manifest.get("output_sha256", {})
    for name, digest in hashes.items():
        bound = recorded.get(name)
        if isinstance(bound, dict):
            bound = bound.get("sha256")
        if bound != digest:
            raise ValueError(f"Capture manifest does not bind {name}")
    payload = json.loads(paths["rows.json"].read_text(encoding="utf-8"))
    if payload.get("schema") != "main-study-capture-v1":
        raise ValueError("Unsupported capture schema")
    rows = tuple(payload["rows"])
    with np.load(paths["features.npz"], allow_pickle=False) as data:
        if set(data.files) != {"features", "row_ids"}:
            raise ValueError("Expected features and row_ids only")
        features = np.array(data["features"], copy=True)
        row_ids = data["row_ids"]
        if row_ids.ndim != 1 or row_ids.dtype.kind != "U":
            raise ValueError("row_ids must be a Unicode vector")
        if row_ids.tolist() != [row.get("row_id") for row in rows]:
            raise ValueError("Feature/record ID order is not exact")
    return FeatureCache(rows, features, payload["provenance"], payload.get("test_fixture"), hashes)


def group_weights(rows: tuple[dict, ...] | list[dict]) -> np.ndarray:
    """Weights sum to one, assigning each scenario equal total influence."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["group_id"]] = counts.get(row["group_id"], 0) + 1
    if not counts:
        raise ValueError("At least one scenario group is required")
    return np.array([1 / (len(counts) * counts[r["group_id"]]) for r in rows])


def grouped_log_loss(labels: np.ndarray, margins: np.ndarray, rows: tuple[dict, ...]) -> float:
    if len(labels) != len(margins) or len(rows) != len(labels):
        raise ValueError("Loss arrays and rows must align")
    return float(np.dot(group_weights(rows), np.logaddexp(0, margins) - labels * margins))


def fit_monotone_calibration(
    scores: np.ndarray, labels: np.ndarray, rows: tuple[dict, ...]
) -> dict:
    """Development-only weighted logistic calibration with slope >= 0.

    Optimize two parameters from fixed (1, 0), no penalty, L-BFGS-B bounds,
    maxiter=2000, ftol=1e-12, gtol=1e-8. A zero slope is permitted. A failed
    optimization is terminal for that calibrated comparator, never a retry.
    """
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)
    if scores.ndim != 1 or labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Calibration needs aligned finite score and label vectors")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Calibration development data must contain both classes")
    weights = group_weights(rows)

    def objective(theta):
        margin = theta[0] * scores + theta[1]
        residual = weights * (expit(margin) - labels)
        loss = np.dot(weights, np.logaddexp(0, margin) - labels * margin)
        return float(loss), np.array([np.dot(residual, scores), residual.sum()])

    result = minimize(
        objective,
        np.array([1.0, 0.0]),
        method="L-BFGS-B",
        jac=True,
        bounds=((0, None), (None, None)),
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success or not np.isfinite(result.x).all():
        return {"status": "failed", "message": str(result.message)}
    return {
        "status": "complete",
        "slope": float(result.x[0]),
        "intercept": float(result.x[1]),
        "weighted_log_loss": float(result.fun),
        "iterations": int(result.nit),
        "source_ids": [row["case_id"] for row in rows],
    }


def row_text(row: dict) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("TF-IDF requires the exact nonempty message list")
    parts = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("Invalid text baseline messages")
        parts.append(f"{message.get('role', '')}: {message['content']}")
    return "\n".join(parts)


def _fit_transform(cache: FeatureCache, indices: list[int], kind: str):
    if kind == "activation":
        selected = np.asarray(cache.features[indices], dtype=np.float64)
        scaler = StandardScaler().fit(selected)
        x = normalize(scaler.transform(selected), norm="l2")
        return x, {
            "kind": kind,
            "mean": scaler.mean_,
            "scale": scaler.scale_,
        }
    if kind != "tfidf":
        raise ValueError("Feature kind must be activation or tfidf")
    vectorizer = TfidfVectorizer(**TFIDF_SETTINGS)
    x = vectorizer.fit_transform([row_text(cache.rows[i]) for i in indices])
    return x, {
        "kind": kind,
        "vectorizer": vectorizer,
        "vocabulary": vectorizer.vocabulary_,
        "idf": vectorizer.idf_,
    }


def _transform(cache: FeatureCache, indices: list[int], transform: dict):
    if transform["kind"] == "activation":
        return normalize(
            (np.asarray(cache.features[indices], dtype=np.float64) - transform["mean"])
            / transform["scale"],
            norm="l2",
        )
    return transform["vectorizer"].transform([row_text(cache.rows[i]) for i in indices])


def _fit_candidate(x, labels: np.ndarray, c_value: float):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = LogisticRegression(
            C=c_value, solver="lbfgs", max_iter=2000, random_state=NUMERICAL_SEED
        ).fit(x, labels)
    failed = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    if failed or not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        return None
    return model


def fit_comparison(
    cache: FeatureCache,
    groups: tuple[TrainingGroup, ...],
    *,
    training_n: int,
    seed: int,
    kind: str = "activation",
) -> tuple[dict, dict[str, np.ndarray]]:
    """Fit one fixed-budget allocation; never accept a cache containing test rows."""
    if any(row["split"] == "test" for row in cache.rows):
        raise ValueError("Test rows are forbidden to the fitting API")
    if any(row["mechanism"] == "correction" for row in cache.rows):
        raise ValueError("Correction cases are forbidden during fitting and selection")
    allocation = matched_training_allocation(groups, training_n=training_n, seed=seed)
    lookup = {row["case_id"]: i for i, row in enumerate(cache.rows)}
    development = [i for i, row in enumerate(cache.rows) if row["split"] == "development"]
    if not development:
        raise ValueError("A disjoint development partition is required")
    dev_rows = tuple(cache.rows[i] for i in development)
    dev_labels = np.array([r["gold"] for r in dev_rows])
    if set(dev_labels.tolist()) != {0, 1}:
        raise ValueError("Development requires both target classes")
    prepared = {}
    for arm in ("restricted", "full"):
        cases = getattr(allocation, arm)
        try:
            selected = [lookup[case.case_id] for case in cases]
        except KeyError as error:
            raise ValueError(f"Missing selected training case {error}") from error
        if len(set(selected)) != training_n or any(
            cache.rows[i]["split"] != "train" for i in selected
        ):
            raise ValueError("Selected inputs must be unique, training-only, and exactly N")
        x, transform = _fit_transform(cache, selected, kind)
        prepared[arm] = {
            "indices": selected,
            "x": x,
            "labels": np.array([cache.rows[i]["gold"] for i in selected]),
            "transform": transform,
            "dev_x": _transform(cache, development, transform),
        }
    candidates, successful = [], []
    for c_value in C_GRID:
        models, losses, failures = {}, {}, []
        for arm, data in prepared.items():
            model = _fit_candidate(data["x"], data["labels"], c_value)
            if model is None:
                failures.append(arm)
            else:
                models[arm] = model
                losses[arm] = grouped_log_loss(
                    dev_labels, model.decision_function(data["dev_x"]), dev_rows
                )
        mean_loss = None if failures else float(np.mean(list(losses.values())))
        candidates.append(
            {"C": c_value, "excluded_arms": failures, "losses": losses, "mean_loss": mean_loss}
        )
        if not failures:
            successful.append((mean_loss, c_value, models))
    metadata = {
        "training_n": training_n,
        "allocation_seed": seed,
        "feature_kind": kind,
        "candidates": candidates,
        "development_ids": [r["case_id"] for r in dev_rows],
        "selected_groups": [g.group_id for g in allocation.selected_groups],
        "arms": {},
    }
    if not successful:
        metadata["status"] = "no_converged_shared_candidate"
        return metadata, {}
    _, selected_c, models = min(successful, key=lambda value: (value[0], value[1]))
    metadata.update(status="complete", selected_C=selected_c)
    arrays = {}
    for arm, data in prepared.items():
        key = f"{kind}_n{training_n}_seed{seed}_{arm}"
        model, transform = models[arm], data["transform"]
        arrays[f"{key}__coef"] = model.coef_[0].copy()
        arrays[f"{key}__intercept"] = model.intercept_.copy()
        source_rows = [cache.rows[i] for i in data["indices"]]
        arm_meta = {
            "array_prefix": key,
            "training_ids": [r["case_id"] for r in source_rows],
            "training_rows_sha256": canonical_hash(source_rows),
            "calibration": fit_monotone_calibration(
                model.decision_function(data["dev_x"]), dev_labels, dev_rows
            ),
        }
        if kind == "activation":
            arrays[f"{key}__mean"] = transform["mean"].copy()
            arrays[f"{key}__scale"] = transform["scale"].copy()
        else:
            arrays[f"{key}__idf"] = transform["idf"].copy()
            arm_meta["vocabulary"] = {
                term: int(index) for term, index in transform["vocabulary"].items()
            }
        metadata["arms"][arm] = arm_meta
    return metadata, arrays


def fit_study(
    cache: FeatureCache,
    groups: tuple[TrainingGroup, ...],
    *,
    kinds: tuple[str, ...] = ("activation", "tfidf"),
    sizes: tuple[int, ...] = TRAINING_SIZES,
    seeds: tuple[int, ...] = ALLOCATION_SEEDS,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Run prespecified comparisons only; fixture outputs remain labelled fixtures."""
    if any(row["split"] == "test" for row in cache.rows):
        raise ValueError("Test rows are forbidden to the fitting API")
    if not cache.test_fixture:
        expected_training = {
            case.case_id
            for group in groups
            for case in one_step_states(group.mechanism, group.group_id)
        }
        training_ids = {row["case_id"] for row in cache.rows if row["split"] == "train"}
        if training_ids != expected_training:
            raise ValueError("Real fitting requires the complete frozen training pool")
        dev_groups: dict[str, list[dict]] = {}
        for row in cache.rows:
            if row["split"] == "development":
                dev_groups.setdefault(row["group_id"], []).append(row)
        mechanisms = [values[0]["mechanism"] for values in dev_groups.values()]
        if (
            len(dev_groups) != 16
            or mechanisms.count("observation") != 8
            or mechanisms.count("message") != 8
            or any(len(values) != 32 for values in dev_groups.values())
        ):
            raise ValueError("Real selection requires all sixteen complete development scenarios")
    if not kinds or len(set(kinds)) != len(kinds) or set(kinds) - {"activation", "tfidf"}:
        raise ValueError("Unknown or duplicate feature kinds")
    if not sizes or len(set(sizes)) != len(sizes) or set(sizes) - set(TRAINING_SIZES):
        raise ValueError("Use only prespecified training sizes")
    if not seeds or len(set(seeds)) != len(seeds) or set(seeds) - set(ALLOCATION_SEEDS):
        raise ValueError("Use only prespecified allocation seeds")
    if not cache.test_fixture and (
        set(kinds) != {"activation", "tfidf"}
        or set(sizes) != set(TRAINING_SIZES)
        or set(seeds) != set(ALLOCATION_SEEDS)
    ):
        raise ValueError("Real fitting requires the complete frozen comparison inventory")
    comparisons, arrays = [], {}
    for kind in kinds:
        for size in sizes:
            for seed in seeds:
                comparison, fitted = fit_comparison(
                    cache, groups, training_n=size, seed=seed, kind=kind
                )
                comparisons.append(comparison)
                arrays.update(fitted)
    development = tuple(row for row in cache.rows if row["split"] == "development")
    output = {}
    if development and all("output_yes_no_logitdiff" in row for row in development):
        output = fit_monotone_calibration(
            np.array([row["output_yes_no_logitdiff"] for row in development]),
            np.array([row["gold"] for row in development]),
            development,
        )
    settings = {
        "schema": "main-study-fit-v1",
        "test_fixture": cache.test_fixture,
        "scientific_result": False,
        "capture_provenance": cache.provenance,
        "capture_source_hashes": cache.source_hashes,
        "fitting_group_splits": {row["group_id"]: row["split"] for row in cache.rows},
        "environment": environment_info(),
        "policy": {
            "C_grid": list(C_GRID),
            "max_iter": 2000,
            "solver": "lbfgs",
            "numerical_seed": NUMERICAL_SEED,
            "activation_pipeline": "float64; train-only StandardScaler then row L2 then logistic",
            "shared_C": "mean of equal-scenario development log losses; lower C breaks ties",
            "raw_threshold": 0.5,
            "tie_class": 0,
            "tfidf": {k: ("float64" if k == "dtype" else v) for k, v in TFIDF_SETTINGS.items()},
            "calibration": {
                "loss": "equal-scenario development logistic",
                "slope_min": 0,
                "initial_slope_intercept": [1.0, 0.0],
                "method": "L-BFGS-B",
                "maxiter": 2000,
                "ftol": 1e-12,
                "gtol": 1e-8,
                "penalty": None,
            },
            "pipeline_effect": "Each arm fits its own scaler/vocabulary and coefficients",
        },
        "comparisons": comparisons,
        "output_calibration": output,
    }
    return settings, arrays


def save_fit_bundle(directory: str | Path, settings: dict, arrays: dict) -> None:
    """Create an immutable, pickle-free fit bundle; an existing output is an error."""
    root = Path(directory)
    if root.exists():
        raise FileExistsError("Fit output already exists; preserve the earlier attempt")
    root.mkdir(parents=True)
    atomic_npz(root / "heads.npz", **arrays)
    payload = dict(settings, heads_sha256=file_sha256(root / "heads.npz"))
    payload["array_sha256"] = {name: _array_hash(value) for name, value in arrays.items()}
    atomic_json(root / "settings.json", payload)
