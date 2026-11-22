"""Descriptive group cross-fitting on an exposed correction activation bank.

Original case splits remain test. This estimates post-hoc readout adequacy,
not fresh generalization or a causal comparison with the old zero-shot heads.
No language model, pickle or joblib is used.
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC

from .main_study_fit import load_feature_cache

STATUS = "posthoc_exposed_bank_group_crossfit"
SEED = 20261122
LR_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
RBF_C_GRID = (0.1, 1.0, 10.0, 100.0)
RBF_GAMMA_GRID = (0.1, 1.0, 10.0)
FACTORS = ("initial", "update", "first_receipt", "receipt")
METHODS = ("direct_lr", "rbf_svc", "fourfactor_lr")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    value = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def policy():
    return {
        "research_status": STATUS,
        "seed": SEED,
        "outer_folds": 5,
        "inner_train_groups": 18,
        "inner_development_groups": 6,
        "transform": "float64; train-only StandardScaler then row L2; common across methods",
        "direct_lr": {
            "C": list(LR_GRID),
            "solver": "lbfgs",
            "max_iter": 2000,
            "tol": 1e-4,
            "class_weight": None,
        },
        "rbf_svc": {
            "C": list(RBF_C_GRID),
            "gamma": list(RBF_GAMMA_GRID),
            "probability": False,
            "tol": 1e-3,
            "class_weight": None,
            "max_iter": 200000,
            "cache_size_mb": 512,
        },
        "fourfactor_lr": {
            "factors": list(FACTORS),
            "C": list(LR_GRID),
            "shared_C": True,
            "selection": "mean four-factor development logistic loss",
        },
        "target_selection": "primary complete-quartet accuracy; overall row accuracy; lower C; lower gamma",
        "threshold": "margin > 0; tie class 0",
        "model_calls": 0,
        "thread_limit": 4,
        "inference": "descriptive only; no bootstrap, confidence intervals or p-values",
    }


def fold_target(bits):
    values = np.asarray(bits)
    require(
        values.ndim == 2 and values.shape[1] == 4 and np.isin(values, (0, 1)).all(),
        "Factor matrix must have four binary columns",
    )
    initial, update, first, final = values.T
    return np.where(final, 1 - update, np.where(first, update, initial)).astype(int)


@dataclass(frozen=True)
class CorrectionData:
    rows: tuple[dict, ...]
    features: np.ndarray
    source: dict


def factor_labels(rows):
    return np.array([[r["state"][name] for name in FACTORS] for r in rows], dtype=int)


def validate_data(data):
    require(
        data.features.ndim == 2
        and len(data.features) == len(data.rows)
        and np.isfinite(data.features).all(),
        "Invalid feature matrix",
    )
    grouped = defaultdict(list)
    ids = set()
    for row in data.rows:
        require(
            row["split"] == "test" and row["mechanism"] == "correction",
            "Original split must remain test and mechanism correction",
        )
        require(
            row["row_id"] == row["case_id"] and row["case_id"] not in ids,
            "Duplicate/misaligned IDs",
        )
        ids.add(row["case_id"])
        state = row["state"]
        require(
            state["group_id"] == row["group_id"] and state["mechanism"] == "correction",
            "State identity differs",
        )
        key = tuple(
            state[n] for n in ("initial", "update", "first_receipt", "receipt", "world", "name_map")
        )
        require(
            all(type(v) is int and v in (0, 1) for v in key), "State factors must be integer bits"
        )
        grouped[row["group_id"]].append(key)
        i, u, b1, b2, w, m = key
        require(
            row["case_id"] == f"{row['group_id']}__i{i}_u{u}_b{b2}_b1{b1}_w{w}_map{m}",
            "State/case ID differs",
        )
    expected = set(product((0, 1), repeat=6))
    require(
        len(grouped) == 30 and len(data.rows) == 1920, "Need all thirty complete correction groups"
    )
    require(
        all(len(v) == 64 and set(v) == expected for v in grouped.values()),
        "Incomplete/duplicate state factorial",
    )
    require(
        np.array_equal(fold_target(factor_labels(data.rows)), [r["gold"] for r in data.rows]),
        "Gold differs from correction rule",
    )


def load_data(model_spec):
    root = Path(model_spec["capture_directory"])
    hashes = model_spec["capture_sha256"]
    require(
        set(hashes) == {"rows.json", "features.npz", "capture_manifest.json"},
        "Need exact capture bindings",
    )
    require(
        all(file_hash(root / name) == value for name, value in hashes.items()),
        "Capture source changed",
    )
    cache = load_feature_cache(root)
    require(
        cache.test_fixture is False and all(r["split"] == "test" for r in cache.rows),
        "Need retained real test capture",
    )
    provenance = cache.provenance
    require(provenance["bank_sha256"] == model_spec["bank_sha256"], "Bank identity differs")
    require(
        provenance["model_settings"]["model_id"] == model_spec["model_id"]
        and provenance["model_settings"]["revision"] == model_spec["revision"]
        and provenance["capture_layer"] == model_spec["capture_layer"],
        "Model/revision/layer differs",
    )
    indices = [i for i, row in enumerate(cache.rows) if row["mechanism"] == "correction"]
    data = CorrectionData(
        tuple(cache.rows[i] for i in indices),
        cache.features[indices].copy(),
        {"model_key": model_spec["key"], "capture_sha256": hashes, "provenance": provenance},
    )
    validate_data(data)
    return data


def make_folds(groups):
    ordered = sorted(
        set(groups),
        key=lambda g: (sha256(f"readout-adequacy-cf-v1|{SEED}|{g}".encode()).digest(), g),
    )
    require(len(ordered) == 30, "Need thirty distinct groups")
    folds = []
    for number in range(5):
        heldout = ordered[number::5]
        train = [g for g in ordered if g not in heldout]
        inner = sorted(
            train,
            key=lambda g: (
                sha256(f"readout-adequacy-cf-v1|inner|{number}|{g}".encode()).digest(),
                g,
            ),
        )
        folds.append(
            {
                "fold": number,
                "outer_test_groups": heldout,
                "outer_train_groups": train,
                "inner_train_groups": inner[:18],
                "inner_dev_groups": inner[18:],
            }
        )
    return folds


def validate_folds(folds, groups):
    expected = make_folds(groups)
    require(len(folds) == 5, "Need five folds")
    for actual, reference in zip(folds, expected, strict=True):
        require(actual["fold"] == reference["fold"], "Fold order differs")
        for key in (
            "outer_test_groups",
            "outer_train_groups",
            "inner_train_groups",
            "inner_dev_groups",
        ):
            require(
                len(actual[key]) == len(reference[key]) and set(actual[key]) == set(reference[key]),
                "Fold membership differs: " + key,
            )


def indices_for(rows, groups):
    wanted = set(groups)
    return np.array([i for i, row in enumerate(rows) if row["group_id"] in wanted], dtype=int)


def fit_transform(features, train_indices):
    scaler = StandardScaler().fit(np.asarray(features[train_indices], dtype=np.float64))
    require(
        np.isfinite(scaler.mean_).all()
        and np.isfinite(scaler.scale_).all()
        and (scaler.scale_ > 0).all(),
        "Invalid fitted transform",
    )
    return {"mean": scaler.mean_.copy(), "scale": scaler.scale_.copy()}


def transform(features, parameters):
    values = normalize(
        (np.asarray(features, dtype=np.float64) - parameters["mean"]) / parameters["scale"],
        norm="l2",
    )
    require(np.isfinite(values).all(), "Nonfinite transformed features")
    return values


class CandidateFailure(ValueError):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details


def fit_candidate(kind, x, labels, parameters):
    estimator = (
        SVC(
            kernel="rbf",
            C=parameters["C"],
            gamma=parameters["gamma"],
            probability=False,
            max_iter=200000,
            cache_size=512,
            tol=1e-3,
            class_weight=None,
        )
        if kind == "rbf_svc"
        else LogisticRegression(
            C=parameters["C"],
            solver="lbfgs",
            max_iter=2000,
            tol=1e-4,
            class_weight=None,
            random_state=SEED,
        )
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(x, labels)
    require(np.array_equal(estimator.classes_, [0, 1]), "Both ordered binary classes required")
    if (
        any(issubclass(w.category, ConvergenceWarning) for w in caught)
        or getattr(estimator, "fit_status_", 0) != 0
    ):
        raise CandidateFailure("Candidate did not converge")
    arrays = (
        [estimator.dual_coef_, estimator.support_vectors_, estimator.intercept_]
        if kind == "rbf_svc"
        else [estimator.coef_, estimator.intercept_]
    )
    if not all(np.isfinite(a).all() for a in arrays):
        raise CandidateFailure("Nonfinite fitted parameter")
    return estimator


def selection_metrics(rows, predicted):
    predicted = np.asarray(predicted)
    require(
        predicted.shape == (len(rows),) and np.isin(predicted, (0, 1)).all(),
        "Invalid binary predictions",
    )
    blocks = defaultdict(list)
    for row, value in zip(rows, predicted, strict=True):
        s = row["state"]
        if s["initial"] == 1 - s["update"]:
            blocks[(row["group_id"], s["update"], s["world"], s["name_map"])].append(
                value == row["gold"]
            )
    require(blocks and all(len(v) == 4 for v in blocks.values()), "Incomplete primary quartets")
    return {
        "primary_quartet_accuracy": float(np.mean([all(v) for v in blocks.values()])),
        "row_accuracy": float(np.mean(predicted == [r["gold"] for r in rows])),
    }


def select_method(method, x, labels, dev_x, dev_rows, factor_y, on_event=None):
    candidates, successful = [], []
    grid = (
        [{"C": c, "gamma": g} for c in RBF_C_GRID for g in RBF_GAMMA_GRID]
        if method == "rbf_svc"
        else [{"C": c} for c in LR_GRID]
    )
    dev_factor_y = factor_labels(dev_rows)
    for parameters in grid:
        record = {"parameters": parameters, "status": "failed"}
        try:
            if method == "fourfactor_lr":
                models, factor_fits = [], []
                for i, factor in enumerate(FACTORS):
                    try:
                        models.append(fit_candidate(method, x, factor_y[:, i], parameters))
                        factor_fits.append({"factor": factor, "status": "complete"})
                    except Exception as error:
                        factor_fits.append(
                            {
                                "factor": factor,
                                "status": "failed",
                                "error": str(error),
                                "error_type": type(error).__name__,
                            }
                        )
                record["factor_fits"] = factor_fits
                if len(models) != 4:
                    raise CandidateFailure("One or more factor candidates failed")
                margins = np.column_stack([m.decision_function(dev_x) for m in models])
                require(np.isfinite(margins).all(), "Nonfinite development factor margin")
                loss = float(np.mean(np.logaddexp(0, margins) - dev_factor_y * margins))
                require(np.isfinite(loss), "Nonfinite development factor log loss")
                record.update(
                    status="complete",
                    mean_factor_log_loss=loss,
                    folded_target_metrics=selection_metrics(
                        dev_rows, fold_target((margins > 0).astype(int))
                    ),
                )
                key = (loss, parameters["C"])
            else:
                model = fit_candidate(method, x, labels, parameters)
                margins = model.decision_function(dev_x)
                require(np.isfinite(margins).all(), "Nonfinite development margin")
                scores = selection_metrics(dev_rows, margins > 0)
                record.update(status="complete", **scores)
                key = (
                    -scores["primary_quartet_accuracy"],
                    -scores["row_accuracy"],
                    parameters["C"],
                    parameters.get("gamma", 0),
                )
            require(np.isfinite(key).all(), "Nonfinite candidate selection key")
            successful.append((key, parameters))
        except Exception as error:
            record.update(status="failed", error_type=type(error).__name__, error=str(error))
        candidates.append(record)
        if on_event:
            on_event({"method": method, **record})
    if not successful:
        raise CandidateFailure("No converged candidate for " + method, candidates)
    return dict(min(successful, key=lambda item: item[0])[1]), candidates


def serialize_model(method, models, parameters, fit_ids):
    metadata = {"method": method, "parameters": parameters, "fit_source_ids": fit_ids}
    if method == "rbf_svc":
        model = models[0]
        metadata["support_source_ids"] = [fit_ids[i] for i in model.support_]
        arrays = {
            "support_vectors": model.support_vectors_.copy(),
            "dual_coef": model.dual_coef_.copy(),
            "intercept": model.intercept_.copy(),
            "support_indices": model.support_.copy(),
            "n_support": model.n_support_.copy(),
            "classes": model.classes_.copy(),
            "gamma": np.array([model._gamma]),
        }
    else:
        arrays = {
            "coef": np.vstack([model.coef_[0] for model in models]),
            "intercept": np.array([model.intercept_[0] for model in models]),
        }
    return metadata, arrays


def serialized_margins(method, x, metadata, arrays):
    if method != "rbf_svc":
        return x @ arrays["coef"].T + arrays["intercept"]
    support = arrays["support_vectors"]
    distances = np.maximum(
        np.sum(x * x, axis=1)[:, None]
        + np.sum(support * support, axis=1)[None, :]
        - 2 * x @ support.T,
        0,
    )
    require(arrays["gamma"][0] == metadata["parameters"]["gamma"], "Serialized gamma differs")
    return (
        np.exp(-arrays["gamma"][0] * distances) @ arrays["dual_coef"][0] + arrays["intercept"][0]
    )[:, None]


def run_fold(data, fold, on_event=None):
    # No held-out labels/features enter preprocessing, candidate fitting or selection.
    inner_train = indices_for(data.rows, fold["inner_train_groups"])
    inner_dev = indices_for(data.rows, fold["inner_dev_groups"])
    outer_train = indices_for(data.rows, fold["outer_train_groups"])
    outer_test = indices_for(data.rows, fold["outer_test_groups"])
    require(
        len(inner_train) == 1152
        and len(inner_dev) == 384
        and len(outer_train) == 1536
        and len(outer_test) == 384,
        "Fold row counts differ",
    )
    require(
        not set(outer_train) & set(outer_test)
        and set(inner_train) | set(inner_dev) == set(outer_train),
        "Fold rows overlap",
    )
    inner_transform = fit_transform(data.features, inner_train)
    outer_transform = fit_transform(data.features, outer_train)
    train_x = transform(data.features[inner_train], inner_transform)
    dev_x = transform(data.features[inner_dev], inner_transform)
    outer_x = transform(data.features[outer_train], outer_transform)
    test_x = transform(data.features[outer_test], outer_transform)
    inner_rows, dev_rows = [data.rows[i] for i in inner_train], [data.rows[i] for i in inner_dev]
    outer_rows, test_rows = [data.rows[i] for i in outer_train], [data.rows[i] for i in outer_test]
    train_y, outer_y = (
        np.array([r["gold"] for r in inner_rows]),
        np.array([r["gold"] for r in outer_rows]),
    )
    factor_y, outer_factor_y = factor_labels(inner_rows), factor_labels(outer_rows)
    result = {
        "fold": fold["fold"],
        "memberships": fold,
        "research_status": STATUS,
        "human_review_complete": False,
        "confirmatory_result": False,
        "inner_train_ids": [r["case_id"] for r in inner_rows],
        "inner_development_ids": [r["case_id"] for r in dev_rows],
        "outer_train_ids": [r["case_id"] for r in outer_rows],
        "outer_test_ids": [r["case_id"] for r in test_rows],
        "methods": {},
    }
    arrays = {
        "inner_mean": inner_transform["mean"],
        "inner_scale": inner_transform["scale"],
        "outer_mean": outer_transform["mean"],
        "outer_scale": outer_transform["scale"],
    }
    for method in METHODS:
        parameters, candidates = select_method(
            method, train_x, train_y, dev_x, dev_rows, factor_y, on_event
        )
        try:
            models = (
                [fit_candidate(method, outer_x, outer_factor_y[:, i], parameters) for i in range(4)]
                if method == "fourfactor_lr"
                else [fit_candidate(method, outer_x, outer_y, parameters)]
            )
        except Exception as error:
            raise CandidateFailure(
                "Selected outer refit failed for " + method,
                {"candidates": candidates, "error": str(error)},
            ) from error
        metadata, head_arrays = serialize_model(
            method, models, parameters, result["outer_train_ids"]
        )
        margins = serialized_margins(method, test_x, metadata, head_arrays)
        reference = np.column_stack([m.decision_function(test_x) for m in models])
        require(
            np.allclose(margins, reference, atol=1e-9, rtol=1e-9), "Serialized prediction differs"
        )
        if method == "fourfactor_lr":
            bits = (margins > 0).astype(int)
            predictions = fold_target(bits)
        else:
            predictions = (margins[:, 0] > 0).astype(int)
        scores = [
            {
                "case_id": row["case_id"],
                "row_id": row["row_id"],
                "group_id": row["group_id"],
                "split": row["split"],
                "original_split": row["split"],
                "state": row["state"],
                "gold": row["gold"],
                "fold": fold["fold"],
                "prediction": int(pred),
                "margin": float(2 * pred - 1),
            }
            for row, pred in zip(test_rows, predictions, strict=True)
        ]
        for score, margin in zip(scores, margins, strict=True):
            if method == "fourfactor_lr":
                score.update(
                    margin_kind="synthetic_hard_fold",
                    folded_binary_score=score["prediction"],
                    factor_margins=margin.tolist(),
                    factor_predictions=(margin > 0).astype(int).tolist(),
                )
            else:
                score["margin"] = float(margin[0])
                score["margin_kind"] = (
                    "svc_decision" if method == "rbf_svc" else "logistic_decision"
                )
        result["methods"][method] = {
            "selected_parameters": parameters,
            "candidates": candidates,
            "head": metadata,
            "heldout_scores": scores,
            "heldout_metrics": summarize(test_rows, scores, method),
        }
        arrays.update({method + "__" + name: value for name, value in head_arrays.items()})
    return result, arrays


def summarize(rows, scores, method):
    by_id = {s["case_id"]: s for s in scores}
    require(
        len(by_id) == len(rows) and set(by_id) == {r["case_id"] for r in rows},
        "Prediction coverage differs",
    )
    predictions = np.array([by_id[r["case_id"]]["prediction"] for r in rows])
    labels = np.array([r["gold"] for r in rows])
    primary = np.array([r["state"]["initial"] == 1 - r["state"]["update"] for r in rows])
    result = {
        **selection_metrics(rows, predictions),
        "rows": len(rows),
        "primary_row_accuracy": float(np.mean(predictions[primary] == labels[primary])),
        "secondary_row_accuracy": float(np.mean(predictions[~primary] == labels[~primary])),
    }
    for label, mask in (
        ("all", np.ones(len(rows), dtype=bool)),
        ("primary", primary),
        ("secondary", ~primary),
    ):
        result[label + "_class_recalls"] = {
            str(value): float(np.mean(predictions[mask & (labels == value)] == value))
            for value in (0, 1)
        }
    quartets, nuisance = defaultdict(list), {"world": {}, "name_map": {}}
    receipt_pairs = {"first_receipt": {}, "receipt": {}}
    for row, prediction in zip(rows, predictions, strict=True):
        state = row["state"]
        if state["initial"] == 1 - state["update"]:
            quartets[(row["group_id"], state["update"], state["world"], state["name_map"])].append(
                bool(prediction == row["gold"])
            )
        for factor in nuisance:
            key = (
                row["group_id"],
                *(
                    state[n]
                    for n in ("initial", "update", "first_receipt", "receipt", "world", "name_map")
                    if n != factor
                ),
            )
            nuisance[factor].setdefault(key, {})[state[factor]] = int(prediction)
        for factor in receipt_pairs:
            key = (
                row["group_id"],
                *(
                    state[n]
                    for n in ("initial", "update", "first_receipt", "receipt", "world", "name_map")
                    if n != factor
                ),
            )
            receipt_pairs[factor].setdefault(key, {})[state[factor]] = (
                int(prediction),
                row["gold"],
            )
    result["quartets"] = [
        {
            "group_id": k[0],
            "update": k[1],
            "world": k[2],
            "name_map": k[3],
            "correct_cells": sum(v),
            "all_correct": all(v),
        }
        for k, v in sorted(quartets.items())
    ]
    result["quartet_correct_cells_histogram"] = {
        str(n): sum(sum(v) == n for v in quartets.values()) for n in range(5)
    }
    result["receipt_pair_metrics"] = {}
    for name, pairs in receipt_pairs.items():
        require(all(set(v) == {0, 1} for v in pairs.values()), "Incomplete receipt counterparts")
        result["receipt_pair_metrics"][name] = {}
        for changes in (False, True):
            selected = [v for v in pairs.values() if (v[0][1] != v[1][1]) == changes]
            result["receipt_pair_metrics"][name]["changing" if changes else "preserving"] = {
                "pairs": len(selected),
                "both_correct": sum(v[0][0] == v[0][1] and v[1][0] == v[1][1] for v in selected),
                "verdict_flips": sum(v[0][0] != v[1][0] for v in selected),
                "both_correct_fraction": float(
                    np.mean([v[0][0] == v[0][1] and v[1][0] == v[1][1] for v in selected])
                ),
            }
    result["nuisance_verdict_flips"] = {}
    for name, pairs in nuisance.items():
        require(all(set(v) == {0, 1} for v in pairs.values()), "Incomplete nuisance counterparts")
        result["nuisance_verdict_flips"][name] = {
            "pairs": len(pairs),
            "flips": sum(v[0] != v[1] for v in pairs.values()),
            "fraction": float(np.mean([v[0] != v[1] for v in pairs.values()])),
        }
    if method == "fourfactor_lr":
        bits = np.array([by_id[r["case_id"]]["factor_predictions"] for r in rows])
        correct = bits == factor_labels(rows)
        result["factor_joint_accuracy"] = float(np.mean(correct.all(axis=1)))
        result["factor_accuracies"] = {
            name: float(np.mean(correct[:, i])) for i, name in enumerate(FACTORS)
        }
    return result


def aggregate(data, fold_results):
    expected_ids = {r["case_id"] for r in data.rows}
    result = {
        "research_status": STATUS,
        "confirmatory_result": False,
        "fresh_test": False,
        "original_split": "test",
        "human_review_complete": False,
        "model_calls": 0,
        "inference": "descriptive only",
        "source": data.source,
        "methods": {},
    }
    for method in METHODS:
        all_scores = [s for fold in fold_results for s in fold["methods"][method]["heldout_scores"]]
        require(
            len(all_scores) == 1920 and {s["case_id"] for s in all_scores} == expected_ids,
            "OOF coverage differs",
        )
        by_id = {s["case_id"]: s for s in all_scores}
        ordered = [by_id[r["case_id"]] for r in data.rows]
        groups = []
        for group in sorted({r["group_id"] for r in data.rows}):
            rows = [r for r in data.rows if r["group_id"] == group]
            groups.append(
                {
                    "group_id": group,
                    "fold": by_id[rows[0]["case_id"]]["fold"],
                    **summarize(rows, [by_id[r["case_id"]] for r in rows], method),
                }
            )
        result["methods"][method] = {
            "summary": summarize(data.rows, ordered, method),
            "groups": groups,
            "oof_scores": ordered,
            "fold_values": [
                {"fold": f["fold"], **f["methods"][method]["heldout_metrics"]} for f in fold_results
            ],
        }
    return result


def verify_protocol(protocol_path):
    protocol = read_json(protocol_path)
    require(
        protocol["schema"] == "cached-readout-adequacy-protocol-v1"
        and protocol["research_status"] == STATUS
        and protocol["seed"] == SEED,
        "Protocol identity differs",
    )
    bindings = protocol["source_sha256"]
    required = required_source_paths()
    require(
        required <= {Path(p).resolve() for p in bindings},
        "Protocol must bind new module and runner",
    )
    require(all(file_hash(p) == value for p, value in bindings.items()), "Frozen source changed")
    if "policy" in protocol:
        require(protocol["policy"] == policy(), "Protocol policy differs from implementation")
    manifest = protocol["fold_manifest"]
    require(file_hash(manifest["path"]) == manifest["sha256"], "Fold manifest changed")
    fold_payload = read_json(manifest["path"])
    require(fold_payload["schema"] == "cached-readout-adequacy-folds-v1", "Fold schema differs")
    require(
        len(protocol["models"]) == 2
        and {m["key"] for m in protocol["models"]} == {"llama", "qwen"},
        "Need both pinned models",
    )
    for model in protocol["models"]:
        require(
            all(
                file_hash(Path(model["capture_directory"]) / name) == value
                for name, value in model["capture_sha256"].items()
            ),
            "Frozen capture changed",
        )
    return protocol, fold_payload["folds"]


def required_source_paths():
    root = Path(__file__).resolve().parents[2]
    return {
        root / path
        for path in (
            "src/latent_probes/cached_readout_adequacy.py",
            "scripts/run_cached_readout_adequacy.py",
            "docs/CACHED_READOUT_ADEQUACY_PROTOCOL_V1.md",
            "src/latent_probes/main_study_fit.py",
            "src/latent_probes/main_study_design.py",
            "src/latent_probes/storage.py",
            "src/latent_probes/__init__.py",
        )
    }
