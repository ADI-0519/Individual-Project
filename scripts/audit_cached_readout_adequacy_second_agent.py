"""Independently reconstruct saved readouts without importing project code or fitting.

The reconstruction phase does not read saved scores, saved metrics, candidate
outcomes, implementation code, or reports.  It freezes its predictions and
derived metrics in a native scratch directory before comparison is allowed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

METHODS = ("direct_lr", "rbf_svc", "fourfactor_lr")
FACTORS = ("initial", "update", "first_receipt", "receipt")
REPO = Path(__file__).resolve().parents[1]
NATIVE = Path("//wsl.localhost/Ubuntu/home/adirj/kcl-latent-probes-runs")
PROTOCOL = NATIVE / "readout_adequacy_cf_v1_20261122_protocol.json"
RESULTS = NATIVE / "readout_adequacy_cf_v1_20261122"
SCRATCH = NATIVE / "readout_adequacy_cf_v1_20261122_second_agent_independent"
FINAL = REPO / "docs/validation/2026-11-22-cached-readout-adequacy-second-agent.json"


def now() -> str:
    return datetime.now(UTC).isoformat()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def mapped(path: str) -> Path:
    if path.startswith("/home/"):
        return Path("//wsl.localhost/Ubuntu" + path)
    if path.startswith("/mnt/c/"):
        return Path("C:/" + path.removeprefix("/mnt/c/"))
    return Path(path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def rate(correct: int, total: int) -> dict:
    return {
        "correct": int(correct),
        "total": int(total),
        "accuracy": correct / total if total else None,
    }


def metric(rows: list[dict], predictions: np.ndarray, factors: np.ndarray | None = None) -> dict:
    """Evaluate by original state, independently of the saved evaluation code."""
    gold = np.array([r["gold"] for r in rows], dtype=np.int64)
    primary = np.array([r["state"]["initial"] != r["state"]["update"] for r in rows])
    right = predictions == gold
    result = {"all_rows": rate(int(right.sum()), len(rows))}
    for label, mask in (("primary_rows", primary), ("secondary_rows", ~primary)):
        result[label] = rate(int(right[mask].sum()), int(mask.sum()))
    result["class_recalls"] = {}
    for label, mask in (
        ("all", np.ones(len(rows), dtype=bool)),
        ("primary", primary),
        ("secondary", ~primary),
    ):
        result["class_recalls"][label] = {
            str(c): rate(int(right[mask & (gold == c)].sum()), int((mask & (gold == c)).sum()))
            for c in (0, 1)
        }
    quartets = defaultdict(list)
    for i, row in enumerate(rows):
        if primary[i]:
            s = row["state"]
            quartets[(row["group_id"], s["update"], s["world"], s["name_map"])].append(i)
    histogram = Counter()
    for members in quartets.values():
        require(len(members) == 4, "Incomplete primary quartet")
        require(
            {(rows[i]["state"]["first_receipt"], rows[i]["state"]["receipt"]) for i in members}
            == {(0, 0), (0, 1), (1, 0), (1, 1)},
            "Wrong receipt cells",
        )
        histogram[int(right[members].sum())] += 1
    result["primary_quartets"] = rate(histogram[4], len(quartets))
    result["quartet_correct_cells_histogram"] = {str(i): histogram[i] for i in range(5)}

    def pairs(axis: str, selected: np.ndarray) -> dict:
        grouped = defaultdict(list)
        keys = ("initial", "update", "first_receipt", "receipt", "world", "name_map")
        for i, row in enumerate(rows):
            if selected[i]:
                s = row["state"]
                grouped[(row["group_id"], *(s[k] for k in keys if k != axis))].append(i)
        count = flips = both_correct = transition_correct = gold_flips = gold_stable = 0
        stable_flips = changed_gold_both_correct = 0
        for members in grouped.values():
            require(len(members) == 2, f"Incomplete {axis} pair")
            a, b = members
            require(rows[a]["state"][axis] != rows[b]["state"][axis], "Duplicate axis value")
            change = predictions[a] != predictions[b]
            gold_change = gold[a] != gold[b]
            count += 1
            flips += int(change)
            both_correct += int(right[a] and right[b])
            transition_correct += int(change == gold_change)
            gold_flips += int(gold_change)
            gold_stable += int(not gold_change)
            stable_flips += int(not gold_change and change)
            changed_gold_both_correct += int(gold_change and right[a] and right[b])
        return {
            "pairs": count,
            "prediction_flips": flips,
            "flip_rate": flips / count,
            "gold_flipping_pairs": gold_flips,
            "gold_preserving_pairs": gold_stable,
            "both_correct": rate(both_correct, count),
            "transition_consistency": rate(transition_correct, count),
            "gold_preserving_prediction_flips": stable_flips,
            "gold_flipping_both_correct": rate(changed_gold_both_correct, gold_flips),
        }

    result["receipt_pairs"] = {
        subset: {axis: pairs(axis, mask) for axis in ("first_receipt", "receipt")}
        for subset, mask in (("all", np.ones(len(rows), dtype=bool)), ("primary", primary))
    }
    result["nuisance_pairs"] = {
        axis: pairs(axis, np.ones(len(rows), dtype=bool)) for axis in ("world", "name_map")
    }
    if factors is not None:
        target = np.array([[r["state"][name] for name in FACTORS] for r in rows])
        hits = factors == target
        result["factors"] = {
            name: rate(int(hits[:, j].sum()), len(rows)) for j, name in enumerate(FACTORS)
        }
        result["factor_joint"] = rate(int(hits.all(axis=1).sum()), len(rows))
    return result


def derived_metrics(
    rows: list[dict], folds: np.ndarray, predictions: np.ndarray, factors=None
) -> dict:
    def subset(indices):
        indices = np.array(indices, dtype=np.int64)
        return metric(
            [rows[i] for i in indices],
            predictions[indices],
            None if factors is None else factors[indices],
        )

    return {
        "overall": metric(rows, predictions, factors),
        "by_fold": {str(f): subset(np.flatnonzero(folds == f)) for f in range(5)},
        "by_group": {
            g: subset([i for i, r in enumerate(rows) if r["group_id"] == g])
            for g in sorted({r["group_id"] for r in rows})
        },
    }


def reconstruct() -> None:
    require(not SCRATCH.exists(), "Refusing to overwrite an existing independent freeze")
    started = now()
    protocol = read_json(PROTOCOL)
    fold_path = mapped(protocol["fold_manifest"]["path"])
    require(sha(fold_path) == protocol["fold_manifest"]["sha256"], "Fold seal mismatch")
    folds = read_json(fold_path)["folds"]
    all_groups = {f"test_correction_{i:02d}" for i in range(1, 31)}
    ordered = sorted(
        all_groups,
        key=lambda g: (
            hashlib.sha256(f"readout-adequacy-cf-v1|20261122|{g}".encode()).hexdigest(),
            g,
        ),
    )
    for f in folds:
        number = f["fold"]
        test_groups = ordered[number::5]
        train_groups = [g for g in ordered if g not in test_groups]
        inner = sorted(
            train_groups,
            key=lambda g: (
                hashlib.sha256(f"readout-adequacy-cf-v1|inner|{number}|{g}".encode()).hexdigest(),
                g,
            ),
        )
        require(f["outer_test_groups"] == test_groups, "Outer fold rule mismatch")
        require(f["outer_train_groups"] == train_groups, "Outer train rule mismatch")
        require(f["inner_train_groups"] == inner[:18], "Inner train rule mismatch")
        require(f["inner_dev_groups"] == inner[18:], "Inner development rule mismatch")
    require(
        Counter(g for f in folds for g in f["outer_test_groups"]) == Counter(all_groups),
        "Nonpartitioned outer tests",
    )
    source_hashes = {str(PROTOCOL): sha(PROTOCOL), str(fold_path): sha(fold_path)}
    independent = {}
    arrays_to_save = {}
    chronology = [{"event": "reconstruction_started", "time": started}]
    for spec in protocol["models"]:
        key = spec["key"]
        capture = mapped(spec["capture_directory"])
        for filename, expected in spec["capture_sha256"].items():
            path = capture / filename
            source_hashes[str(path)] = sha(path)
            require(
                source_hashes[str(path)] == expected, f"Capture seal mismatch: {key}/{filename}"
            )
        source = read_json(capture / "rows.json")
        originals = source["rows"]
        with np.load(capture / "features.npz", allow_pickle=False) as archive:
            features = np.array(archive["features"], dtype=np.float64)
            feature_ids = archive["row_ids"].tolist()
        require(
            feature_ids == [r["row_id"] for r in originals], "Original feature row order mismatch"
        )
        require(np.isfinite(features).all(), "Nonfinite original features")
        selected = [i for i, r in enumerate(originals) if r["mechanism"] == "correction"]
        rows = [
            {k: originals[i][k] for k in ("row_id", "group_id", "split", "state", "gold")}
            for i in selected
        ]
        x = features[selected]
        require(
            len(rows) == 1920 and len({r["row_id"] for r in rows}) == 1920,
            "Correction coverage mismatch",
        )
        require(
            Counter(r["group_id"] for r in rows) == Counter({g: 64 for g in all_groups}),
            "Group coverage mismatch",
        )
        require(all(r["split"] == "test" for r in rows), "Original split changed")
        for row in rows:
            s = row["state"]
            require(all(s[k] in (0, 1) for k in (*FACTORS, "world", "name_map")), "Nonbinary state")
            gold = (
                1 - s["update"]
                if s["receipt"]
                else s["update"]
                if s["first_receipt"]
                else s["initial"]
            )
            require(row["gold"] == gold, "Original gold violates declared fold")
        predictions = {m: np.full(1920, -1, dtype=np.int64) for m in METHODS}
        margins = {m: np.full(1920, np.nan) for m in METHODS}
        factor_margins = np.full((1920, 4), np.nan)
        factor_predictions = np.full((1920, 4), -1, dtype=np.int64)
        assignments = np.full(1920, -1, dtype=np.int64)
        fold_checks = []
        for fold in folds:
            number = fold["fold"]
            train = np.array(
                [i for i, r in enumerate(rows) if r["group_id"] in fold["outer_train_groups"]]
            )
            test = np.array(
                [i for i, r in enumerate(rows) if r["group_id"] in fold["outer_test_groups"]]
            )
            require(len(train) == 1536 and len(test) == 384, "Outer row membership mismatch")
            require(
                set(train).isdisjoint(test) and (assignments[test] == -1).all(),
                "Outer leakage/duplication",
            )
            assignments[test] = number
            folder = RESULTS / key / f"fold_{number}"
            heads_path = folder / "heads.npz"
            metadata_path = folder / "results.json"
            source_hashes[str(heads_path)] = sha(heads_path)
            source_hashes[str(metadata_path)] = sha(metadata_path)
            # Parse metadata, but access only membership and head provenance.
            metadata = read_json(metadata_path)
            require(metadata["memberships"] == fold, "Fold metadata membership mismatch")
            train_ids = [rows[i]["row_id"] for i in train]
            test_ids = [rows[i]["row_id"] for i in test]
            require(metadata["outer_train_ids"] == train_ids, "Saved outer training order mismatch")
            require(metadata["outer_test_ids"] == test_ids, "Saved outer test order mismatch")
            mean = x[train].mean(axis=0)
            scale = x[train].std(axis=0, ddof=0)
            scale[scale == 0] = 1

            def transform(values):
                standardized = (values - mean) / scale
                lengths = np.linalg.norm(standardized, axis=1, keepdims=True)
                lengths[lengths == 0] = 1
                return standardized / lengths

            heldout = transform(x[test])
            training = transform(x[train])
            with np.load(heads_path, allow_pickle=False) as archive:
                h = {k: np.array(archive[k]) for k in archive.files}
            require(all(np.isfinite(v).all() for v in h.values()), "Nonfinite safe head")
            mean_error = float(np.max(np.abs(mean - h["outer_mean"])))
            scale_error = float(np.max(np.abs(scale - h["outer_scale"])))
            require(
                mean_error < 1e-12 and scale_error < 1e-12,
                "Saved outer transform differs from train-only recomputation",
            )
            for method in METHODS:
                require(
                    metadata["methods"][method]["head"]["fit_source_ids"] == train_ids,
                    "Head fit IDs differ from outer train",
                )
            direct = (heldout @ h["direct_lr__coef"].T + h["direct_lr__intercept"]).reshape(-1)
            require(h["rbf_svc__classes"].tolist() == [0, 1], "Unverified SVC positive class")
            support_indices = h["rbf_svc__support_indices"]
            support_ids = [train_ids[i] for i in support_indices]
            require(
                metadata["methods"]["rbf_svc"]["head"]["support_source_ids"] == support_ids,
                "Support provenance mismatch",
            )
            support = h["rbf_svc__support_vectors"]
            support_error = float(np.max(np.abs(training[support_indices] - support)))
            require(support_error < 1e-12, "SVC supports differ from transformed training rows")
            # Explicit squared Euclidean differences, avoiding a library kernel.
            nonlinear = np.empty(len(test))
            gamma = float(h["rbf_svc__gamma"][0])
            require(gamma > 0, "Invalid kernel gamma")
            for start in range(0, len(test), 16):
                distance = ((heldout[start : start + 16, None, :] - support[None, :, :]) ** 2).sum(
                    axis=2
                )
                nonlinear[start : start + 16] = np.exp(-gamma * distance) @ h[
                    "rbf_svc__dual_coef"
                ].reshape(-1) + float(h["rbf_svc__intercept"][0])
            fm = heldout @ h["fourfactor_lr__coef"].T + h["fourfactor_lr__intercept"]
            fp = (fm > 0).astype(np.int64)
            folded = np.where(fp[:, 3], 1 - fp[:, 1], np.where(fp[:, 2], fp[:, 1], fp[:, 0]))
            margins["direct_lr"][test] = direct
            margins["rbf_svc"][test] = nonlinear
            margins["fourfactor_lr"][test] = 2 * folded - 1
            predictions["direct_lr"][test] = (direct > 0).astype(np.int64)
            predictions["rbf_svc"][test] = (nonlinear > 0).astype(np.int64)
            predictions["fourfactor_lr"][test] = folded
            factor_margins[test] = fm
            factor_predictions[test] = fp
            fold_checks.append(
                {
                    "fold": number,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "max_saved_mean_error": mean_error,
                    "max_saved_scale_error": scale_error,
                    "support_vectors": len(support),
                    "max_support_vector_error": support_error,
                    "classes": h["rbf_svc__classes"].tolist(),
                    "gamma": gamma,
                }
            )
            print(f"reconstructed {key} fold {number}", flush=True)
        require((assignments >= 0).all(), "Missing outer-fold assignments")
        require(all(np.isfinite(v).all() for v in margins.values()), "Missing/nonfinite scores")
        require(
            all(np.isin(v, [0, 1]).all() for v in predictions.values()),
            "Missing/nonbinary predictions",
        )
        result = {"rows": 1920, "fold_checks": fold_checks, "methods": {}}
        save = {
            "row_ids": np.array([r["row_id"] for r in rows]),
            "folds": assignments,
            "gold": np.array([r["gold"] for r in rows]),
            "factor_margins": factor_margins,
            "factor_predictions": factor_predictions,
        }
        for method in METHODS:
            result["methods"][method] = derived_metrics(
                rows,
                assignments,
                predictions[method],
                factor_predictions if method == "fourfactor_lr" else None,
            )
            save[method + "__prediction"] = predictions[method]
            save[method + "__margin"] = margins[method]
        result["row_metadata"] = rows
        independent[key] = result
        arrays_to_save[key] = save
    SCRATCH.mkdir()
    checker = Path(__file__).resolve()
    # Preserve exact phase-one source, allowing later comparison-only changes to be disclosed.
    (SCRATCH / "reconstruction_checker.py").write_bytes(checker.read_bytes())
    artifacts = {}
    for key, data in arrays_to_save.items():
        path = SCRATCH / f"{key}_independent_predictions.npz"
        np.savez_compressed(path, **data)
        artifacts[str(path)] = sha(path)
    metrics_path = SCRATCH / "independent_metrics.json"
    write_json(metrics_path, independent)
    artifacts[str(metrics_path)] = sha(metrics_path)
    chronology.append({"event": "independent_predictions_and_metrics_frozen", "time": now()})
    freeze = {
        "schema": "cached-readout-second-agent-phase1-v1",
        "research_status": "posthoc_exposed_bank_group_crossfit",
        "human_review_complete": False,
        "confirmatory_result": False,
        "chronology": chronology,
        "prior_exposure": {
            "cached_readout_implementation": False,
            "cached_readout_runner": False,
            "previous_auditor": False,
            "research_summaries_or_interpretations": False,
            "report_renderer_or_tests": False,
            "saved_readout_score_or_metric_values_before_freeze": False,
            "allowed_material": [
                "task mathematical specification",
                "protocol Markdown and JSON",
                "fold manifest",
                "original capture rows/features",
                "safe heads NPZ",
                "fold membership/training/support IDs",
            ],
            "schema_inspection_before_freeze": "Fold JSON key names, including heldout metric field names, were printed, without their values. One original capture row was printed to inspect state schema and included original language-model logits; these were not diagnostic readout outcomes and were not used.",
        },
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "configuration": "Root configured shared workspace with configurePythonEnvironment and getPythonExecutableCommand; exact existing .venv interpreter supplied to this agent. No installs.",
        },
        "reconstruction_checker_sha256": sha(checker),
        "preserved_reconstruction_checker_sha256": sha(SCRATCH / "reconstruction_checker.py"),
        "source_sha256": source_hashes,
        "artifact_sha256": artifacts,
        "method": "float64 original correction features; train-only population mean/std; zero std to one; row L2; explicit LR dot products and RBF squared-distance sums; >0 threshold; independent four-factor Boolean fold and metrics",
        "model_calls": 0,
        "fits": 0,
        "project_code_imports": 0,
        "pickle_loads": 0,
        "limitations": [
            "Reconstructs predictions conditional on saved fitted heads; does not independently redo fitting, inner selection or optimization.",
            "Checks saved fit/support IDs and train-only outer transforms, but IDs and safe head arrays alone cannot prove the historical fitting procedure or absence of unrecorded leakage.",
            "Descriptive exposed-bank crossfit with overlapping outer training sets; no confirmatory or causal inference.",
            "Human semantic acceptance remains incomplete.",
        ],
    }
    write_json(SCRATCH / "phase1_freeze.json", freeze)
    print(
        json.dumps(
            {
                "frozen": str(SCRATCH / "phase1_freeze.json"),
                "freeze_sha256": sha(SCRATCH / "phase1_freeze.json"),
            }
        ),
        flush=True,
    )


def saved_schema_metrics(own: dict, rows: list[dict], prediction: np.ndarray) -> dict:
    """Map frozen metric counts to saved field names after the blind freeze.

    The mapping changes no predictions or outcome definitions.  Receipt pairs
    are partitioned by whether their independently checked gold values change.
    """
    result = {
        "primary_quartet_accuracy": own["primary_quartets"]["accuracy"],
        "row_accuracy": own["all_rows"]["accuracy"],
        "rows": own["all_rows"]["total"],
        "primary_row_accuracy": own["primary_rows"]["accuracy"],
        "secondary_row_accuracy": own["secondary_rows"]["accuracy"],
        "quartet_correct_cells_histogram": own["quartet_correct_cells_histogram"],
    }
    for subset in ("all", "primary", "secondary"):
        result[subset + "_class_recalls"] = {
            c: value["accuracy"] for c, value in own["class_recalls"][subset].items()
        }
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        s = row["state"]
        if s["initial"] != s["update"]:
            grouped[(row["group_id"], s["update"], s["world"], s["name_map"])].append(i)
    quartets = []
    for (group, update, world, name_map), indices in sorted(grouped.items()):
        count = sum(int(prediction[i]) == rows[i]["gold"] for i in indices)
        quartets.append(
            {
                "group_id": group,
                "update": update,
                "world": world,
                "name_map": name_map,
                "correct_cells": count,
                "all_correct": count == 4,
            }
        )
    result["quartets"] = quartets
    receipt = {}
    for axis, values in own["receipt_pairs"]["all"].items():
        changing_correct = values["gold_flipping_both_correct"]["correct"]
        both = values["both_correct"]["correct"]
        receipt[axis] = {}
        for name, pairs, correct, flips in (
            (
                "preserving",
                values["gold_preserving_pairs"],
                both - changing_correct,
                values["gold_preserving_prediction_flips"],
            ),
            (
                "changing",
                values["gold_flipping_pairs"],
                changing_correct,
                values["prediction_flips"] - values["gold_preserving_prediction_flips"],
            ),
        ):
            receipt[axis][name] = {
                "pairs": pairs,
                "both_correct": correct,
                "verdict_flips": flips,
                "both_correct_fraction": correct / pairs if pairs else None,
            }
    result["receipt_pair_metrics"] = receipt
    result["nuisance_verdict_flips"] = {}
    for axis, values in own["nuisance_pairs"].items():
        require(values["gold_flipping_pairs"] == 0, "Nuisance pair changed gold")
        result["nuisance_verdict_flips"][axis] = {
            "pairs": values["pairs"],
            "flips": values["prediction_flips"],
            "fraction": values["flip_rate"],
        }
    if "factors" in own:
        result["factor_accuracies"] = {
            name: value["accuracy"] for name, value in own["factors"].items()
        }
        result["factor_joint_accuracy"] = own["factor_joint"]["accuracy"]
    return result


def assert_nested(actual, expected, context: str, counts: dict) -> None:
    if isinstance(expected, dict):
        require(
            isinstance(actual, dict) and set(actual) == set(expected),
            f"Metric key mismatch: {context}",
        )
        for key in expected:
            assert_nested(actual[key], expected[key], context + "/" + str(key), counts)
    elif isinstance(expected, list):
        require(
            isinstance(actual, list) and len(actual) == len(expected),
            f"Metric list mismatch: {context}",
        )
        for i, value in enumerate(expected):
            assert_nested(actual[i], value, context + f"/{i}", counts)
    else:
        require(actual == expected, f"Metric mismatch: {context}: {actual!r} != {expected!r}")
        counts["scalar_fields"] += 1


def compare_rows(
    saved: list[dict], indices: np.ndarray, rows: list[dict], data: dict, method: str
) -> dict:
    require(len(saved) == len(indices), "Saved score count mismatch")
    by_id = {r["row_id"]: r for r in saved}
    require(len(by_id) == len(saved), "Duplicate saved score IDs")
    require(set(by_id) == {rows[i]["row_id"] for i in indices}, "Saved score coverage mismatch")
    score_error = factor_error = 0.0
    margin_kind = {
        "direct_lr": "logistic_decision",
        "rbf_svc": "svc_decision",
        "fourfactor_lr": "synthetic_hard_fold",
    }[method]
    for i in indices:
        row = rows[i]
        value = by_id[row["row_id"]]
        for field in ("row_id", "group_id", "state", "gold", "split"):
            require(value[field] == row[field], f"Original metadata mismatch: {field}")
        require(value["case_id"] == row["row_id"], "Saved case ID mismatch")
        require(value["original_split"] == row["split"], "Original split mismatch")
        require(value["fold"] == int(data["folds"][i]), "Saved fold mismatch")
        require(
            value["prediction"] == int(data[method + "__prediction"][i]),
            "Saved verdict differs from independent reconstruction",
        )
        require(value["margin_kind"] == margin_kind, "Mislabelled decision margin")
        score_error = max(score_error, abs(value["margin"] - float(data[method + "__margin"][i])))
        if method == "fourfactor_lr":
            require(
                value["folded_binary_score"] == int(data[method + "__prediction"][i]),
                "Synthetic folded score mismatch",
            )
            require(
                value["factor_predictions"] == data["factor_predictions"][i].tolist(),
                "Factor predictions mismatch",
            )
            factor_error = max(
                factor_error,
                float(
                    np.max(np.abs(np.array(value["factor_margins"]) - data["factor_margins"][i]))
                ),
            )
    require(
        score_error < 1e-10 and factor_error < 1e-10,
        "Saved numeric margins differ beyond tolerance",
    )
    return {
        "rows": len(saved),
        "verdict_mismatches": 0,
        "factor_prediction_mismatches": 0 if method == "fourfactor_lr" else None,
        "max_absolute_margin_error": score_error,
        "max_absolute_factor_margin_error": factor_error if method == "fourfactor_lr" else None,
        "margin_tolerance_absolute": 1e-10,
    }


def compare() -> None:
    started = now()
    freeze_path = SCRATCH / "phase1_freeze.json"
    freeze_sha = sha(freeze_path)
    freeze = read_json(freeze_path)
    for path, expected in freeze["source_sha256"].items():
        require(sha(Path(path)) == expected, "Phase-one source changed: " + path)
    for path, expected in freeze["artifact_sha256"].items():
        require(sha(Path(path)) == expected, "Frozen independent output changed: " + path)
    require(
        sha(SCRATCH / "reconstruction_checker.py") == freeze["reconstruction_checker_sha256"],
        "Reconstruction checker changed",
    )

    def function_asts(path: Path) -> dict:
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef) and node.name != "main"
        }

    original_functions = function_asts(SCRATCH / "reconstruction_checker.py")
    current_functions = function_asts(Path(__file__).resolve())
    require(
        all(current_functions[name] == value for name, value in original_functions.items()),
        "Reconstruction logic changed after blind freeze",
    )

    def compact_metrics(value: dict) -> dict:
        result = {
            "all_row_counts": [value["all_rows"]["correct"], value["all_rows"]["total"]],
            "primary_row_counts": [
                value["primary_rows"]["correct"],
                value["primary_rows"]["total"],
            ],
            "secondary_row_counts": [
                value["secondary_rows"]["correct"],
                value["secondary_rows"]["total"],
            ],
            "primary_quartet_counts": [
                value["primary_quartets"]["correct"],
                value["primary_quartets"]["total"],
            ],
            "quartet_histogram_zero_through_four": [
                value["quartet_correct_cells_histogram"][str(i)] for i in range(5)
            ],
        }
        if "factors" in value:
            result["factor_correct_counts_I_U_B1_B2"] = [
                value["factors"][name]["correct"] for name in FACTORS
            ]
            result["factor_joint_correct_count"] = value["factor_joint"]["correct"]
        return result

    own = read_json(SCRATCH / "independent_metrics.json")
    protocol = read_json(PROTOCOL)
    protocol_source_checks = {}
    for name, expected in protocol["source_sha256"].items():
        observed = sha(
            mapped(name)
        )  # Hash bytes only; never inspect forbidden implementation contents.
        require(observed == expected, "Protocol source seal mismatch: " + name)
        protocol_source_checks[name] = observed
    completed_path = RESULTS / "completed_manifest.json"
    completed = read_json(completed_path)
    require(completed["status"] == "complete", "Execution not complete")
    require(
        completed["confirmatory_result"] is False and completed["model_calls"] == 0,
        "Execution scope changed",
    )
    completed_output_checks = {}
    for relative, expected in completed["output_sha256"].items():
        observed = sha(RESULTS / relative)
        require(observed == expected, "Completed output seal mismatch: " + relative)
        completed_output_checks[relative] = observed
    prepared_path = RESULTS / "prepared_manifest.json"
    prepared = read_json(prepared_path)
    require(
        sha(prepared_path) == completed["prepared_manifest_sha256"],
        "Prepared execution seal mismatch",
    )
    require(prepared["protocol_sha256"] == sha(PROTOCOL), "Prepared protocol seal mismatch")
    checked = {}
    models = {}
    count = {"scalar_fields": 0}
    for model, independent in own.items():
        rows = independent["row_metadata"]
        archive_path = SCRATCH / f"{model}_independent_predictions.npz"
        with np.load(archive_path, allow_pickle=False) as archive:
            data = {k: np.array(archive[k]) for k in archive.files}
        require(
            data["row_ids"].tolist() == [r["row_id"] for r in rows],
            "Frozen metadata order mismatch",
        )
        saved_path = RESULTS / model / "oof_results.json"
        saved = read_json(saved_path)
        checked[str(saved_path)] = sha(saved_path)
        require(
            saved["research_status"] == "posthoc_exposed_bank_group_crossfit"
            and saved["confirmatory_result"] is False
            and saved["fresh_test"] is False
            and saved["human_review_complete"] is False,
            "OOF epistemic status mismatch",
        )
        models[model] = {
            "rows": len(rows),
            "fold_checks": independent["fold_checks"],
            "methods": {},
        }
        for method in METHODS:
            predicted = data[method + "__prediction"]
            derived = independent["methods"][method]
            source_method = saved["methods"][method]
            comparison = compare_rows(
                source_method["oof_scores"], np.arange(len(rows)), rows, data, method
            )
            expected = saved_schema_metrics(derived["overall"], rows, predicted)
            assert_nested(source_method["summary"], expected, f"{model}/{method}/overall", count)
            by_fold = {}
            require(len(source_method["fold_values"]) == 5, "Wrong saved fold metric count")
            for entry in source_method["fold_values"]:
                number = entry["fold"]
                require(str(number) not in by_fold, "Duplicate fold metric")
                indices = np.flatnonzero(data["folds"] == number)
                expected_fold = saved_schema_metrics(
                    derived["by_fold"][str(number)], [rows[i] for i in indices], predicted[indices]
                )
                assert_nested(
                    {k: v for k, v in entry.items() if k != "fold"},
                    expected_fold,
                    f"{model}/{method}/fold{number}",
                    count,
                )
                fold_path = RESULTS / model / f"fold_{number}/results.json"
                fold_result = read_json(fold_path)["methods"][method]
                fold_rows = compare_rows(fold_result["heldout_scores"], indices, rows, data, method)
                assert_nested(
                    fold_result["heldout_metrics"],
                    expected_fold,
                    f"{model}/{method}/heldout{number}",
                    count,
                )
                by_fold[str(number)] = {
                    "score_comparison": fold_rows,
                    "independent_metric_counts": compact_metrics(derived["by_fold"][str(number)]),
                }
            require(
                set(by_fold) == {str(i) for i in range(5)}, "Incomplete saved fold metric coverage"
            )
            require(len(source_method["groups"]) == 30, "Wrong saved group metric count")
            groups = {}
            for entry in source_method["groups"]:
                group = entry["group_id"]
                require(group not in groups, "Duplicate saved group metric")
                indices = np.array([i for i, row in enumerate(rows) if row["group_id"] == group])
                require(
                    len(indices) == 64 and (data["folds"][indices] == entry["fold"]).all(),
                    "Group fold mismatch",
                )
                expected_group = saved_schema_metrics(
                    derived["by_group"][group], [rows[i] for i in indices], predicted[indices]
                )
                assert_nested(
                    {k: v for k, v in entry.items() if k not in ("group_id", "fold")},
                    expected_group,
                    f"{model}/{method}/{group}",
                    count,
                )
                gm = derived["by_group"][group]
                groups[group] = {
                    "fold": entry["fold"],
                    "independent_metric_counts": compact_metrics(gm),
                }
            models[model]["methods"][method] = {
                "oof_score_comparison": comparison,
                "all_metric_comparisons_exact": True,
                "overall_independent_metrics": derived["overall"],
                "by_fold": by_fold,
                "by_group": groups,
            }
    # Recheck hashes after all outcome comparisons, then emit the compact checked report.
    require(sha(freeze_path) == freeze_sha, "Independent freeze changed during comparison")
    for path, expected in freeze["source_sha256"].items():
        require(sha(Path(path)) == expected, "Phase-one source changed during comparison")
    report = {
        "schema": "cached-readout-adequacy-second-agent-v1",
        "created_at": now(),
        "status": "passed",
        "conclusion": "Independent reconstruction of every held-out readout verdict and factor prediction agrees with the saved outputs; all compared aggregate, fold and group metric fields agree exactly. This validates arithmetic conditional on saved heads, not the fitting history or a confirmatory scientific claim.",
        "research_status": "posthoc_exposed_bank_group_crossfit",
        "human_review_complete": False,
        "confirmatory_result": False,
        "chronology": freeze["chronology"]
        + [
            {
                "event": "saved_diagnostic_values_first_inspected",
                "time": "after phase-one freeze; command-level record in agent transcript",
            },
            {"event": "comparison_started", "time": started},
            {"event": "comparison_completed", "time": now()},
        ],
        "independence": freeze["prior_exposure"],
        "checker_provenance": {
            "current_checker": str(Path(__file__).resolve()),
            "current_checker_sha256": sha(Path(__file__).resolve()),
            "phase1_checker": str(SCRATCH / "reconstruction_checker.py"),
            "phase1_checker_sha256": freeze["reconstruction_checker_sha256"],
            "phase1_checker_preserved": True,
            "comparison_phase_added_after_freeze": True,
            "reconstruction_outputs_recomputed_after_saved_outcome_exposure": False,
            "reconstruction_function_asts_unchanged_since_blind_freeze": True,
            "format_and_lint": "ruff check and ruff format passed before each checker hash was bound",
        },
        "phase1_freeze": {
            "path": str(freeze_path),
            "sha256": freeze_sha,
            "independent_artifact_sha256": freeze["artifact_sha256"],
        },
        "environment": freeze["environment"],
        "source_sha256": freeze["source_sha256"],
        "protocol_source_seal_checks": protocol_source_checks,
        "completed_manifest": {
            "path": str(completed_path),
            "sha256": sha(completed_path),
            "status": completed["status"],
            "sealed_output_files": len(completed_output_checks),
            "output_sha256": completed_output_checks,
        },
        "saved_oof_sha256": checked,
        "comparison_totals": {
            "models": 2,
            "model_folds": 10,
            "methods": 3,
            "oof_verdicts": 11520,
            "oof_factor_predictions": 15360,
            "fold_metric_objects": 30,
            "group_metric_objects": 180,
            "overall_metric_objects": 6,
            "saved_heldout_metric_objects": 30,
            "metric_scalar_fields_compared_exactly": count["scalar_fields"],
            "verdict_mismatches": 0,
            "factor_prediction_mismatches": 0,
        },
        "models": models,
        "model_calls": 0,
        "fits": 0,
        "project_code_imports": 0,
        "pickle_loads": 0,
        "limitations": freeze["limitations"],
    }
    write_json(FINAL, report)
    print(
        json.dumps(
            {
                "report": str(FINAL),
                "report_sha256": sha(FINAL),
                "status": report["status"],
                "comparison_totals": report["comparison_totals"],
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("reconstruct", "compare"))
    args = parser.parse_args()
    if args.phase == "reconstruct":
        reconstruct()
    else:
        compare()


if __name__ == "__main__":
    main()
