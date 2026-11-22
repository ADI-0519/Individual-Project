"""Recompute sealed adequacy artifacts without importing or fitting their pipeline."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

FACTORS = ("initial", "update", "first_receipt", "receipt")
METHODS = ("direct_lr", "rbf_svc", "fourfactor_lr")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    value = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def native(path):
    if Path.cwd().drive and path.startswith("/mnt/c/"):
        return Path("C:/" + path[len("/mnt/c/") :])
    if Path.cwd().drive and path.startswith("/"):
        return Path("//wsl.localhost/Ubuntu" + path)
    return Path(path)


def recorded_path(path):
    """Use reproducible WSL names even when this auditor runs on Windows."""
    value = Path(path).resolve().as_posix()
    prefix = "//wsl.localhost/Ubuntu"
    if value.casefold().startswith(prefix.casefold() + "/"):
        return value[len(prefix) :]
    if len(value) > 2 and value[1:3] == ":/":
        return "/mnt/" + value[0].lower() + "/" + value[3:]
    return value


def fold_factors(bits):
    i, u, b1, b2 = np.asarray(bits).T
    return np.where(b2, 1 - u, np.where(b1, u, i)).astype(int)


def metrics(rows, scores, method):
    by_id = {s["case_id"]: s for s in scores}
    require(len(scores) == len(by_id) == len(rows), "Prediction count/uniqueness differs")
    correct, primary, labels, predictions = [], [], [], []
    quartets, pair_maps = (
        defaultdict(list),
        {n: defaultdict(dict) for n in ("world", "name_map", "first_receipt", "receipt")},
    )
    for row in rows:
        score = by_id[row["case_id"]]
        state, prediction, label = row["state"], score["prediction"], row["gold"]
        is_primary = state["initial"] == 1 - state["update"]
        correct.append(prediction == label)
        primary.append(is_primary)
        labels.append(label)
        predictions.append(prediction)
        if is_primary:
            quartets[(row["group_id"], state["update"], state["world"], state["name_map"])].append(
                prediction == label
            )
        for name, pairs in pair_maps.items():
            key = (
                row["group_id"],
                *(state[n] for n in (*FACTORS, "world", "name_map") if n != name),
            )
            pairs[key][state[name]] = (prediction, label)
    require(all(len(v) == 4 for v in quartets.values()), "Incomplete primary quartet")
    correct, primary, labels, predictions = map(np.asarray, (correct, primary, labels, predictions))
    result = {
        "rows": len(rows),
        "primary_quartet_accuracy": float(np.mean([all(v) for v in quartets.values()])),
        "row_accuracy": float(np.mean(correct)),
        "primary_row_accuracy": float(np.mean(correct[primary])),
        "secondary_row_accuracy": float(np.mean(correct[~primary])),
        "quartet_correct_cells_histogram": {
            str(n): sum(sum(v) == n for v in quartets.values()) for n in range(5)
        },
        "quartets": [
            {
                "group_id": k[0],
                "update": k[1],
                "world": k[2],
                "name_map": k[3],
                "correct_cells": sum(v),
                "all_correct": all(v),
            }
            for k, v in sorted(quartets.items())
        ],
        "nuisance_verdict_flips": {},
        "receipt_pair_metrics": {},
    }
    for name, mask in (
        ("all", np.ones(len(rows), dtype=bool)),
        ("primary", primary),
        ("secondary", ~primary),
    ):
        result[name + "_class_recalls"] = {
            str(v): float(np.mean(predictions[mask & (labels == v)] == v)) for v in (0, 1)
        }
    for name, pairs in pair_maps.items():
        require(all(set(v) == {0, 1} for v in pairs.values()), "Incomplete counterpart pair")
        if name in ("world", "name_map"):
            flips = [v[0][0] != v[1][0] for v in pairs.values()]
            result["nuisance_verdict_flips"][name] = {
                "pairs": len(pairs),
                "flips": sum(flips),
                "fraction": float(np.mean(flips)),
            }
        else:
            result["receipt_pair_metrics"][name] = {}
            for changes in (False, True):
                values = [v for v in pairs.values() if (v[0][1] != v[1][1]) == changes]
                both = [v[0][0] == v[0][1] and v[1][0] == v[1][1] for v in values]
                result["receipt_pair_metrics"][name]["changing" if changes else "preserving"] = {
                    "pairs": len(values),
                    "both_correct": sum(both),
                    "verdict_flips": sum(v[0][0] != v[1][0] for v in values),
                    "both_correct_fraction": float(np.mean(both)),
                }
    if method == "fourfactor_lr":
        bits = np.array([by_id[r["case_id"]]["factor_predictions"] for r in rows])
        actual = np.array([[r["state"][n] for n in FACTORS] for r in rows])
        result["factor_joint_accuracy"] = float(np.mean((bits == actual).all(axis=1)))
        result["factor_accuracies"] = {
            n: float(np.mean(bits[:, i] == actual[:, i])) for i, n in enumerate(FACTORS)
        }
    return result


def equal(actual, expected, label):
    require(set(actual) == set(expected), label + ": key inventory differs")
    for key in expected:
        if isinstance(expected[key], float):
            require(abs(actual[key] - expected[key]) < 1e-12, label + ": " + key)
        else:
            require(actual[key] == expected[key], label + ": " + key)


def audit(output):
    complete, prepared = (
        read(output / "completed_manifest.json"),
        read(output / "prepared_manifest.json"),
    )
    require(
        complete["status"] == "complete" and complete["model_calls"] == 0, "Run did not complete"
    )
    require(
        digest(output / "prepared_manifest.json") == complete["prepared_manifest_sha256"],
        "Preparation seal changed",
    )
    for name, value in complete["output_sha256"].items():
        require(digest(output / name) == value, "Output changed: " + name)
    protocol_path = native(prepared["protocol_path"])
    require(digest(protocol_path) == prepared["protocol_sha256"], "Protocol changed")
    protocol = read(protocol_path)
    for name, value in protocol["source_sha256"].items():
        require(digest(native(name)) == value, "Frozen source changed: " + name)
    fold_path = native(protocol["fold_manifest"]["path"])
    require(digest(fold_path) == protocol["fold_manifest"]["sha256"], "Fold manifest changed")
    folds = read(fold_path)["folds"]
    result = {
        "schema": "cached-readout-adequacy-independent-audit-v1",
        "model_calls": 0,
        "classifier_refits": 0,
        "pipeline_imported": False,
        "research_status": "posthoc_exposed_bank_group_crossfit",
        "models": {},
    }
    auditor = Path(__file__).resolve()
    audit_test = auditor.parents[1] / "tests/test_cached_readout_adequacy_independent.py"
    bound_files = [
        output / "completed_manifest.json",
        output / "prepared_manifest.json",
        protocol_path,
        fold_path,
        auditor,
        audit_test,
    ]
    for model in protocol["models"]:
        capture = native(model["capture_directory"])
        for name, value in model["capture_sha256"].items():
            require(digest(capture / name) == value, "Capture changed")
        payload = read(capture / "rows.json")
        all_rows = payload["rows"]
        with np.load(capture / "features.npz", allow_pickle=False) as saved:
            require(
                saved["row_ids"].tolist() == [r["case_id"] for r in all_rows],
                "Source feature IDs differ",
            )
            features = saved["features"].copy()
        chosen = [i for i, row in enumerate(all_rows) if row["mechanism"] == "correction"]
        rows, features = [all_rows[i] for i in chosen], features[chosen].astype(np.float64)
        require(
            len(rows) == 1920 and all(r["split"] == "test" for r in rows),
            "Source cohort/split differs",
        )
        groups = {r["group_id"] for r in rows}
        ordered = sorted(
            groups,
            key=lambda g: (sha256(f"readout-adequacy-cf-v1|20261122|{g}".encode()).hexdigest(), g),
        )
        require(len(groups) == 30, "Need thirty groups")
        oof = read(output / model["key"] / "oof_results.json")
        bound_files.append(output / model["key"] / "oof_results.json")
        require(
            oof["research_status"] == result["research_status"]
            and oof["fresh_test"] is False
            and oof["confirmatory_result"] is False
            and oof["human_review_complete"] is False,
            "Research status differs",
        )
        maximum_margin_error, all_scores = 0.0, {n: [] for n in METHODS}
        for fold in folds:
            number = fold["fold"]
            expected_test = ordered[number::5]
            expected_train = [g for g in ordered if g not in expected_test]
            inner = sorted(
                expected_train,
                key=lambda g: (
                    sha256(f"readout-adequacy-cf-v1|inner|{number}|{g}".encode()).hexdigest(),
                    g,
                ),
            )
            require(
                set(fold["outer_test_groups"]) == set(expected_test)
                and set(fold["outer_train_groups"]) == set(expected_train)
                and set(fold["inner_train_groups"]) == set(inner[:18])
                and set(fold["inner_dev_groups"]) == set(inner[18:]),
                "Group assignment differs",
            )
            fold_dir = output / model["key"] / f"fold_{number}"
            record = read(fold_dir / "results.json")
            require(digest(fold_dir / "heads.npz") == record["heads_sha256"], "Head NPZ changed")
            with np.load(fold_dir / "heads.npz", allow_pickle=False) as saved:
                arrays = {k: saved[k].copy() for k in saved.files}
            require(set(arrays) == set(record["array_sha256"]), "Array inventory differs")
            for name, values in arrays.items():
                require(
                    sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
                    == record["array_sha256"][name],
                    "Array changed",
                )
            for role, members in (("inner", inner[:18]), ("outer", expected_train)):
                indices = [i for i, r in enumerate(rows) if r["group_id"] in members]
                x = features[indices]
                scale = x.std(axis=0)
                scale[scale == 0] = 1
                require(
                    np.allclose(arrays[role + "_mean"], x.mean(axis=0), atol=1e-12, rtol=1e-12)
                    and np.allclose(arrays[role + "_scale"], scale, atol=1e-12, rtol=1e-12),
                    "Training-only transform differs",
                )
            test_indices = [i for i, r in enumerate(rows) if r["group_id"] in expected_test]
            test_rows = [rows[i] for i in test_indices]
            train_ids = [r["case_id"] for r in rows if r["group_id"] in expected_train]
            require(
                record["outer_train_ids"] == train_ids
                and record["outer_test_ids"] == [r["case_id"] for r in test_rows],
                "Recorded memberships differ",
            )
            require(
                record["inner_train_ids"]
                == [r["case_id"] for r in rows if r["group_id"] in inner[:18]]
                and record["inner_development_ids"]
                == [r["case_id"] for r in rows if r["group_id"] in inner[18:]],
                "Inner memberships differ",
            )
            x = (features[test_indices] - arrays["outer_mean"]) / arrays["outer_scale"]
            norms = np.linalg.norm(x, axis=1)
            x /= np.where(norms == 0, 1, norms)[:, None]
            for method in METHODS:
                m = record["methods"][method]
                require(m["head"]["fit_source_ids"] == train_ids, "Outer refit sources differ")
                successful = [c for c in m["candidates"] if c["status"] == "complete"]
                if method == "fourfactor_lr":
                    best = min(
                        successful, key=lambda c: (c["mean_factor_log_loss"], c["parameters"]["C"])
                    )
                else:
                    best = min(
                        successful,
                        key=lambda c: (
                            -c["primary_quartet_accuracy"],
                            -c["row_accuracy"],
                            c["parameters"]["C"],
                            c["parameters"].get("gamma", 0),
                        ),
                    )
                require(
                    m["selected_parameters"] == best["parameters"],
                    "Inner candidate selection differs",
                )
                prefix = method + "__"
                if method == "rbf_svc":
                    support = arrays[prefix + "support_vectors"]
                    require(
                        arrays[prefix + "classes"].tolist() == [0, 1], "SVC orientation differs"
                    )
                    require(
                        m["head"]["support_source_ids"]
                        == [train_ids[i] for i in arrays[prefix + "support_indices"]],
                        "SVC support source differs",
                    )
                    distances = np.maximum(
                        (x * x).sum(axis=1)[:, None]
                        + (support * support).sum(axis=1)[None, :]
                        - 2 * x @ support.T,
                        0,
                    )
                    margin = (
                        np.exp(-arrays[prefix + "gamma"][0] * distances)
                        @ arrays[prefix + "dual_coef"][0]
                        + arrays[prefix + "intercept"][0]
                    )[:, None]
                else:
                    margin = x @ arrays[prefix + "coef"].T + arrays[prefix + "intercept"]
                expected = np.array(
                    [
                        s["factor_margins"] if method == "fourfactor_lr" else [s["margin"]]
                        for s in m["heldout_scores"]
                    ]
                )
                maximum_margin_error = max(
                    maximum_margin_error, float(np.max(np.abs(margin - expected)))
                )
                require(
                    np.allclose(margin, expected, atol=1e-9, rtol=1e-9),
                    "Serialized held-out margin differs",
                )
                predicted = (
                    fold_factors((margin > 0).astype(int))
                    if method == "fourfactor_lr"
                    else (margin[:, 0] > 0).astype(int)
                )
                require(
                    predicted.tolist() == [s["prediction"] for s in m["heldout_scores"]],
                    "Serialized verdict differs",
                )
                for row, score in zip(test_rows, m["heldout_scores"], strict=True):
                    require(
                        score["case_id"] == row["case_id"]
                        and score["state"] == row["state"]
                        and score["gold"] == row["gold"]
                        and score["split"] == score["original_split"] == "test"
                        and score["fold"] == number,
                        "OOF provenance differs",
                    )
                    if method == "fourfactor_lr":
                        require(
                            score["margin_kind"] == "synthetic_hard_fold"
                            and score["folded_binary_score"] == score["prediction"],
                            "Folded score semantics differ",
                        )
                equal(
                    m["heldout_metrics"],
                    metrics(test_rows, m["heldout_scores"], method),
                    "Fold metrics",
                )
                expected_fold = {"fold": number, **m["heldout_metrics"]}
                require(
                    oof["methods"][method]["fold_values"][number] == expected_fold,
                    "Aggregate fold metrics differ",
                )
                all_scores[method].extend(m["heldout_scores"])
        for method in METHODS:
            lookup = {s["case_id"]: s for s in all_scores[method]}
            require(
                len(all_scores[method]) == len(lookup) == 1920, "OOF coverage/uniqueness differs"
            )
            ordered_scores = [lookup[r["case_id"]] for r in rows]
            m = oof["methods"][method]
            require(m["oof_scores"] == ordered_scores, "OOF assembly/order differs")
            equal(m["summary"], metrics(rows, ordered_scores, method), "OOF metrics")
            require(
                len(m["groups"]) == 30 and {g["group_id"] for g in m["groups"]} == groups,
                "Group report coverage differs",
            )
            for g in m["groups"]:
                group_rows = [r for r in rows if r["group_id"] == g["group_id"]]
                expected = metrics(group_rows, [lookup[r["case_id"]] for r in group_rows], method)
                equal(
                    {k: v for k, v in g.items() if k not in ("group_id", "fold")},
                    expected,
                    "Group metrics",
                )
        result["models"][model["key"]] = {
            "rows": len(rows),
            "groups": len(groups),
            "outer_folds": len(folds),
            "reconstructed_outer_heads": 15,
            "train_only_transforms_checked": 10,
            "reconstructed_oof_predictions": 5760,
            "maximum_margin_absolute_error": maximum_margin_error,
            "all_group_fold_and_aggregate_metrics_match": True,
        }
    result["source_sha256"] = {recorded_path(path): digest(path) for path in bound_files}
    result["status"] = "all_independent_checks_passed"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Use a fresh audit output")
    with threadpool_limits(limits=4):
        result = audit(args.run)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
