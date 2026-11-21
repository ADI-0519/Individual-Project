"""Held-out scoring and scenario-level estimands for the matched belief study.

The binary target here is fictional sender-attributed reader belief, never a
deception label. Quartet metrics are deliberately separate from generic score
metrics. Fixture caches cannot be reported as measured scientific outcomes.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import normalize

from .main_study_design import ALLOCATION_SEEDS, TRAINING_SIZES, BeliefCase, primary_quartets
from .main_study_fit import (
    NUMERICAL_SEED,
    TFIDF_SETTINGS,
    FeatureCache,
    _array_hash,
    row_text,
)
from .storage import file_sha256


def decisions(probabilities: np.ndarray) -> np.ndarray:
    """The frozen source convention uses strict >.5, so an exact tie is zero."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilities must be a finite vector in [0,1]")
    return (values > 0.5).astype(int)


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    """Generic binary metrics; no claim about the meaning of either class."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or set(labels.tolist()) - {0, 1} or not len(labels):
        raise ValueError("Labels must be a nonempty binary vector")
    predicted = decisions(probabilities)
    if labels.shape != predicted.shape:
        raise ValueError("Label/score length mismatch")
    return {
        "rows": len(labels),
        "correct": int((labels == predicted).sum()),
        "accuracy": float(np.mean(labels == predicted)),
        "recall_0": float(np.mean(predicted[labels == 0] == 0)) if (labels == 0).any() else None,
        "recall_1": float(np.mean(predicted[labels == 1] == 1)) if (labels == 1).any() else None,
        "auroc": float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else None,
        "brier": float(brier_score_loss(labels, probabilities)),
    }


def belief_group_metrics(rows: tuple[dict, ...], margins: np.ndarray) -> dict:
    """Score exact counterfactual pairs and complete primary belief quartets."""
    if len(rows) != len(margins) or not np.isfinite(margins).all():
        raise ValueError("Aligned finite margins are required")
    if len({row["group_id"] for row in rows}) != 1:
        raise ValueError("One scenario group is required")
    cases = tuple(BeliefCase(**row["state"]) for row in rows)
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Duplicate counterfactual case")
    by_id = {
        case.case_id: (case, float(margin)) for case, margin in zip(cases, margins, strict=True)
    }
    quartets = primary_quartets(cases)
    if len(quartets) != 8:
        raise ValueError("A complete scenario must have eight primary quartets")
    expected = 64 if cases[0].mechanism == "correction" else 32
    if len(cases) != expected:
        raise ValueError("All primary and secondary scenario cells must be retained")
    records = []
    for quartet in quartets:
        correct = [int(by_id[c.case_id][1] > 0) == c.gold for c in quartet.cases]
        records.append(
            {
                "quartet_id": quartet.quartet_id,
                "case_ids": [c.case_id for c in quartet.cases],
                "all_four_correct": bool(all(correct)),
            }
        )
    pairs: dict[tuple, list] = defaultdict(list)
    for case, margin in by_id.values():
        key = (case.initial, case.update, case.world, case.name_map, case.first_receipt)
        pairs[key].append((case, margin))
    pair_records = []
    for key in sorted(pairs, key=str):
        pair = sorted(pairs[key], key=lambda value: value[0].receipt)
        if len(pair) != 2 or [c.receipt for c, _ in pair] != [0, 1]:
            raise ValueError("A counterfactual receipt pair is incomplete")
        (before, before_margin), (after, after_margin) = pair
        change = before.changes_if_received
        pair_records.append(
            {
                "case_ids": [before.case_id, after.case_id],
                "changes_if_received": change,
                "both_correct": (
                    int(before_margin > 0) == before.gold and int(after_margin > 0) == after.gold
                ),
                "verdict_flip": bool((before_margin > 0) != (after_margin > 0)),
                "margin_difference": after_margin - before_margin,
                "oriented_margin_difference": (
                    (2 * after.gold - 1) * (after_margin - before_margin) if change else None
                ),
            }
        )
    labels = np.array([case.gold for case in cases])
    result = {
        "group_id": cases[0].group_id,
        "mechanism": cases[0].mechanism,
        "all_rows": binary_metrics(labels, expit(margins)),
        "complete_quartet_accuracy": float(np.mean([r["all_four_correct"] for r in records])),
        "quartets": records,
        "pairs": pair_records,
    }
    for label, changes in (("changing", True), ("preserving", False)):
        selected = [pair for pair in pair_records if pair["changes_if_received"] == changes]
        result[label] = {
            "pairs": len(selected),
            "both_correct_fraction": float(np.mean([p["both_correct"] for p in selected])),
            "verdict_flip_fraction": float(np.mean([p["verdict_flip"] for p in selected])),
            "mean_absolute_margin_difference": float(
                np.mean([abs(p["margin_difference"]) for p in selected])
            ),
        }
        if changes:
            result[label]["mean_oriented_margin_difference"] = float(
                np.mean([p["oriented_margin_difference"] for p in selected])
            )
    if cases[0].mechanism == "correction":
        secondary = np.array([not case.primary_eligible for case in cases])
        result["secondary_correction_cells"] = binary_metrics(
            labels[secondary], expit(margins[secondary])
        )
    return result


def cluster_primary_summary(
    groups_by_seed: dict[int, dict[str, dict]], *, bootstrap_draws: int = 10000
) -> dict:
    """Average paired allocations within group, then bootstrap whole groups.

    ``groups_by_seed[seed][group]`` supplies restricted/full quartet accuracies.
    Factorial rows and training allocations never become extra narrative units.
    """
    if set(groups_by_seed) != set(ALLOCATION_SEEDS):
        raise ValueError("Primary analysis needs all five prespecified allocations")
    group_sets = [set(values) for values in groups_by_seed.values()]
    if not group_sets[0] or any(values != group_sets[0] for values in group_sets):
        raise ValueError("All allocations must contain the same scenario groups")
    group_ids = sorted(group_sets[0])
    seeds = sorted(groups_by_seed)
    restricted = np.array([[groups_by_seed[s][g]["restricted"] for s in seeds] for g in group_ids])
    full = np.array([[groups_by_seed[s][g]["full"] for s in seeds] for g in group_ids])
    if not np.isfinite(restricted).all() or not np.isfinite(full).all():
        raise ValueError("Nonfinite primary scores")
    if ((restricted < 0) | (restricted > 1) | (full < 0) | (full > 1)).any():
        raise ValueError("Quartet accuracy must be in [0,1]")
    if type(bootstrap_draws) is not int or bootstrap_draws <= 0:
        raise ValueError("Positive bootstrap draw count required")
    differences = (full - restricted).mean(axis=1)
    rng = np.random.default_rng(NUMERICAL_SEED)
    indices = rng.integers(0, len(group_ids), size=(bootstrap_draws, len(group_ids)))
    boot = differences[indices].mean(axis=1)
    return {
        "independent_scenario_groups": len(group_ids),
        "allocation_seeds": seeds,
        "restricted": float(restricted.mean()),
        "full": float(full.mean()),
        "full_minus_restricted": float(differences.mean()),
        "percentile_95_interval": np.percentile(boot, [2.5, 97.5]).tolist(),
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": NUMERICAL_SEED,
        "sampling_unit": "whole correction scenario; five paired allocations retained",
        "interval_scope": "conditional on authored groups and fixed training-allocation design",
        "by_group": [
            {
                "group_id": g,
                "restricted_mean": float(restricted[i].mean()),
                "full_mean": float(full[i].mean()),
                "difference": float(differences[i]),
                "restricted_by_seed": restricted[i].tolist(),
                "full_by_seed": full[i].tolist(),
            }
            for i, g in enumerate(group_ids)
        ],
        "by_seed": [
            {
                "seed": s,
                "restricted": float(restricted[:, j].mean()),
                "full": float(full[:, j].mean()),
                "difference": float((full[:, j] - restricted[:, j]).mean()),
            }
            for j, s in enumerate(seeds)
        ],
    }


def load_fit_bundle(directory: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    root = Path(directory)
    settings = json.loads((root / "settings.json").read_text(encoding="utf-8"))
    if settings.get("schema") != "main-study-fit-v1":
        raise ValueError("Unsupported fit bundle schema")
    if file_sha256(root / "heads.npz") != settings.get("heads_sha256"):
        raise ValueError("Fit NPZ file hash mismatch")
    with np.load(root / "heads.npz", allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    if set(arrays) != set(settings.get("array_sha256", {})):
        raise ValueError("Fit array inventory mismatch")
    for name, values in arrays.items():
        if not np.isfinite(values).all() or _array_hash(values) != settings["array_sha256"][name]:
            raise ValueError("Fit array hash mismatch or nonfinite array")
    settings["_artifact_sha256"] = {
        "settings.json": file_sha256(root / "settings.json"),
        "heads.npz": file_sha256(root / "heads.npz"),
    }
    return settings, arrays


def _head_margins(cache: FeatureCache, comparison: dict, arm: str, arrays: dict) -> np.ndarray:
    head = comparison["arms"][arm]
    key = head["array_prefix"]
    if comparison["feature_kind"] == "activation":
        mean, scale = arrays[f"{key}__mean"], arrays[f"{key}__scale"]
        if (
            mean.shape != (cache.features.shape[1],)
            or scale.shape != mean.shape
            or (scale <= 0).any()
        ):
            raise ValueError("Head feature dimension or scaler is invalid")
        x = normalize((cache.features - mean) / scale, norm="l2")
    else:
        vectorizer = TfidfVectorizer(**TFIDF_SETTINGS, vocabulary=head["vocabulary"])
        vectorizer.idf_ = arrays[f"{key}__idf"]
        x = vectorizer.transform([row_text(row) for row in cache.rows])
    coef, intercept = arrays[f"{key}__coef"], arrays[f"{key}__intercept"]
    if coef.shape != (x.shape[1],) or intercept.shape != (1,):
        raise ValueError("Head coefficients do not match feature dimensions")
    return np.asarray(x @ coef).ravel() + intercept[0]


def analyse_study(
    cache: FeatureCache, settings: dict, arrays: dict, *, allow_fixture: bool = False
) -> dict:
    """Score retained test rows without refitting or changing any threshold."""
    if cache.test_fixture != settings.get("test_fixture"):
        raise ValueError("Fit and evaluation fixture status must agree")
    if cache.test_fixture and not allow_fixture:
        raise ValueError("Fixture outputs cannot be accepted as scientific analysis")
    if any(row["split"] != "test" for row in cache.rows):
        raise ValueError("Analysis accepts an exact held-out test cache only")
    if not cache.test_fixture:
        artifact_hashes = settings.get("_artifact_sha256", {})
        bindings = cache.provenance.get("source_bindings", {})
        by_name: dict[str, set[str]] = defaultdict(set)
        for path, digest in bindings.items():
            by_name[path.replace("\\", "/").rsplit("/", 1)[-1]].add(digest)
        if set(artifact_hashes) != {"settings.json", "heads.npz"} or any(
            digest not in by_name[name] for name, digest in artifact_hashes.items()
        ):
            raise ValueError("Analysis heads were not bound before held-out capture")
    for key in ("bank_sha256", "model_settings", "capture_layer"):
        if cache.provenance.get(key) != settings["capture_provenance"].get(key):
            raise ValueError(f"Fit/evaluation capture provenance mismatch: {key}")
        if not cache.test_fixture and key not in cache.provenance:
            raise ValueError(f"Missing capture provenance: {key}")
    group_indices: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(cache.rows):
        group_indices[row["group_id"]].append(i)
    if set(group_indices) & set(settings.get("fitting_group_splits", {})):
        raise ValueError("A fitted/development scenario group overlaps held-out evaluation")
    if not cache.test_fixture:
        mechanisms = [cache.rows[indices[0]]["mechanism"] for indices in group_indices.values()]
        if len(group_indices) != 90 or any(
            mechanisms.count(name) != 30 for name in ("observation", "message", "correction")
        ):
            raise ValueError("Real analysis requires every frozen held-out scenario")
        expected_inventory = {
            (kind, n, seed)
            for kind in ("activation", "tfidf")
            for n in TRAINING_SIZES
            for seed in ALLOCATION_SEEDS
        }
        actual_inventory = [
            (item["feature_kind"], item["training_n"], item["allocation_seed"])
            for item in settings["comparisons"]
        ]
        if (
            len(actual_inventory) != len(expected_inventory)
            or set(actual_inventory) != expected_inventory
        ):
            raise ValueError("Real analysis requires the complete frozen comparison inventory")
    comparisons = []
    primary = {}
    for comparison in settings["comparisons"]:
        record = {
            key: comparison[key]
            for key in ("training_n", "allocation_seed", "feature_kind", "status")
        }
        if comparison["status"] != "complete":
            comparisons.append(record)
            continue
        record["arms"] = {}
        for arm in ("restricted", "full"):
            source_ids = comparison["arms"][arm]["training_ids"] + comparison["development_ids"]
            if set(source_ids) & {row["case_id"] for row in cache.rows}:
                raise ValueError("Fit/selection source overlaps held-out evaluation")
            margins = _head_margins(cache, comparison, arm, arrays)
            group_scores = [
                belief_group_metrics(tuple(cache.rows[i] for i in indices), margins[indices])
                for indices in group_indices.values()
            ]
            arm_record = {
                "groups": group_scores,
                "scores": [
                    {
                        "case_id": row["case_id"],
                        "margin": float(margin),
                        "probability": float(expit(margin)),
                        "prediction": int(margin > 0),
                    }
                    for row, margin in zip(cache.rows, margins, strict=True)
                ],
            }
            calibration = comparison["arms"][arm]["calibration"]
            if calibration.get("status") == "complete":
                calibrated = calibration["slope"] * margins + calibration["intercept"]
                arm_record["calibrated_groups"] = [
                    belief_group_metrics(tuple(cache.rows[i] for i in indices), calibrated[indices])
                    for indices in group_indices.values()
                ]
            record["arms"][arm] = arm_record
        if comparison["feature_kind"] == "activation" and comparison["training_n"] == 512:
            seed = comparison["allocation_seed"]
            if seed in primary:
                raise ValueError("Duplicate primary comparison")
            primary[seed] = {}
            for arm in ("restricted", "full"):
                for group in record["arms"][arm]["groups"]:
                    if group["mechanism"] == "correction":
                        primary[seed].setdefault(group["group_id"], {})[arm] = group[
                            "complete_quartet_accuracy"
                        ]
        comparisons.append(record)
    primary_result = None
    primary_declared = any(
        item["feature_kind"] == "activation" and item["training_n"] == 512
        for item in settings["comparisons"]
    )
    if primary_declared:
        if set(primary) != set(ALLOCATION_SEEDS):
            primary_result = {
                "status": "unavailable_missing_or_failed_primary_comparisons",
                "available_allocation_seeds": sorted(primary),
                "missing_allocation_seeds": sorted(set(ALLOCATION_SEEDS) - set(primary)),
            }
        else:
            primary_result = dict(status="complete", **cluster_primary_summary(primary))
            if not cache.test_fixture and primary_result["independent_scenario_groups"] != 30:
                raise ValueError(
                    "The frozen primary study requires all thirty correction scenarios"
                )
    baselines = {}
    for name in ("constant_0", "initial", "final_update", "world", "symbolic_oracle"):
        margins = []
        for row in cache.rows:
            case = BeliefCase(**row["state"])
            value = {
                "constant_0": 0,
                "initial": case.initial,
                "final_update": case.final_update,
                "world": case.world,
                "symbolic_oracle": case.gold,
            }[name]
            margins.append(1.0 if value else -1.0)
        baseline_margins = np.array(margins)
        baselines[name] = [
            belief_group_metrics(tuple(cache.rows[i] for i in indices), baseline_margins[indices])
            for indices in group_indices.values()
        ]
    if all("output_yes_no_logitdiff" in row for row in cache.rows):
        margins = np.array([row["output_yes_no_logitdiff"] for row in cache.rows], dtype=float)
        baselines["output_raw"] = [
            belief_group_metrics(tuple(cache.rows[i] for i in indices), margins[indices])
            for indices in group_indices.values()
        ]
        calibration = settings["output_calibration"]
        if calibration.get("status") == "complete":
            margins = calibration["slope"] * margins + calibration["intercept"]
            baselines["output_calibrated"] = [
                belief_group_metrics(tuple(cache.rows[i] for i in indices), margins[indices])
                for indices in group_indices.values()
            ]
    return {
        "schema": "main-study-analysis-v1",
        "test_fixture": cache.test_fixture,
        "scientific_result": not cache.test_fixture,
        "target": "fictional sender-attributed reader belief content, not deception",
        "capture_provenance": cache.provenance,
        "capture_source_hashes": cache.source_hashes,
        "primary": primary_result,
        "comparisons": comparisons,
        "baselines": baselines,
    }
