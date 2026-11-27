"""Sealed-scope, old-data-only relevance and privileged external-fold diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = (
    "//wsl.localhost/Ubuntu/home/adirj/kcl-latent-probes-runs/fresh_readout_v2_20261123_recovery_v2"
)
FACTORS = ("I", "U", "B1", "B2")
FIELDS = ("initial", "update", "first_receipt", "receipt")
STRATA = ("union", "correction_easy", "correction_hard")
SPEC = ROOT / "docs/LOGICAL_RELEVANCE_OLD_DATA_SPEC_2026-11-27.md"
OUTPUT = ROOT / "docs/results/logical-relevance-old-data-2026-11-27"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fold(bits):
    bits = np.asarray(bits)
    require(bits.ndim == 2 and bits.shape[1] == 4 and np.isin(bits, (0, 1)).all(), "Invalid bits")
    initial, update, first, final = bits.T
    return np.where(final, 1 - update, np.where(first, update, initial)).astype(np.int64)


def relevance(truth):
    truth = np.asarray(truth)
    fold(truth)
    answer = np.empty(truth.shape, dtype=bool)
    for column in range(4):
        low, high = truth.copy(), truth.copy()
        low[:, column], high[:, column] = 0, 1
        answer[:, column] = fold(low) != fold(high)
    return answer


def substitute(predictions, truth, mask):
    require(type(mask) is int and 0 <= mask < 16, "Invalid oracle mask")
    predictions, truth = np.asarray(predictions), np.asarray(truth)
    require(predictions.shape == truth.shape, "Oracle shape mismatch")
    fold(predictions)
    fold(truth)
    result = predictions.copy()
    for column in range(4):
        if mask & (1 << column):
            result[:, column] = truth[:, column]
    return result


def metrics(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    require(labels.ndim == 1 and labels.shape == scores.shape, "Metric shape mismatch")
    require(np.isin(labels, (0, 1)).all() and np.isfinite(scores).all(), "Invalid metric values")
    n = len(labels)
    n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())
    prediction = scores > 0
    auc = None
    if n0 and n1:
        order = np.argsort(scores, kind="stable")
        ranks = np.empty(n, dtype=np.float64)
        start = 0
        while start < n:
            end = start + 1
            while end < n and scores[order[end]] == scores[order[start]]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2
            start = end
        auc = float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))
    return {
        "n": n,
        "class0_n": n0,
        "class1_n": n1,
        "raw_oriented_auc": auc,
        "auc_null_reason": None if auc is not None else "empty" if not n else "one_class",
        "zero_accuracy": float((prediction == labels).mean()) if n else None,
        "accuracy_null_reason": None if n else "empty",
        "class0_recall": float((prediction[labels == 0] == 0).mean()) if n0 else None,
        "class0_recall_null_reason": None if n0 else "class_absent",
        "class1_recall": float((prediction[labels == 1] == 1).mean()) if n1 else None,
        "class1_recall_null_reason": None if n1 else "class_absent",
    }


def align_rows(rows, ids):
    require(len(ids) == len(set(ids)), "Duplicate prediction row identity")
    indexed = {row["row_id"]: row for row in rows}
    require(len(indexed) == len(rows), "Duplicate capture row identity")
    require(set(indexed) == set(ids), "Capture/prediction identity mismatch")
    return [indexed[row_id] for row_id in ids]


def validate_rows(rows):
    require(len(rows) == 4992, "Need all 4992 old test rows")
    require(
        {r["group_id"] for r in rows} == {f"fresh_test_{i:02d}" for i in range(24)},
        "Need all 24 contexts",
    )
    cells = defaultdict(set)
    for row in rows:
        require(row["partition"] == "test", "Unexpected non-test row")
        state, condition = row["state"], row["condition"]
        require(row["renderer"] in ("A", "B"), "Unknown renderer")
        if condition.startswith("correction_"):
            bits = tuple(state[field] for field in (*FIELDS, "world", "name_map"))
            require(np.isin(bits, (0, 1)).all(), "Invalid correction state")
            require(
                condition == ("correction_easy" if bits[0] == bits[1] else "correction_hard"),
                "Wrong stratum",
            )
            gold = int(fold(np.asarray([bits[:4]]))[0])
        elif condition == "one_step":
            bits = tuple(state[k] for k in ("initial", "update", "receipt", "world", "name_map"))
            gold = state["update"] if state["receipt"] else state["initial"]
        elif condition == "asserted":
            bits = tuple(state[k] for k in ("asserted", "world", "name_map"))
            gold = state["asserted"]
        else:
            raise ValueError("Unexpected condition")
        require(np.isin(bits, (0, 1)).all() and row["gold"] == gold, "Formal target disagreement")
        key = (row["group_id"], row["renderer"], condition)
        require(bits not in cells[key], "Duplicate factorial cell")
        cells[key].add(bits)
    for group in sorted({r["group_id"] for r in rows}):
        for renderer in ("A", "B"):
            for condition in ("correction_easy", "correction_hard", "one_step", "asserted"):
                width = (
                    6
                    if condition.startswith("correction_")
                    else 5
                    if condition == "one_step"
                    else 3
                )
                expected = set(itertools.product((0, 1), repeat=width))
                if condition.startswith("correction_"):
                    expected = {
                        b for b in expected if (b[0] == b[1]) == (condition == "correction_easy")
                    }
                require(
                    cells[(group, renderer, condition)] == expected,
                    "Incomplete factorial inventory",
                )


def quartet_indices(rows, indices):
    quartets = defaultdict(dict)
    for index in indices:
        row = rows[index]
        state = row["state"]
        key = (
            row["group_id"],
            row["renderer"],
            *(state[k] for k in ("initial", "update", "world", "name_map")),
        )
        pair = (state["first_receipt"], state["receipt"])
        require(pair not in quartets[key], "Duplicate receipt cell")
        quartets[key][pair] = index
    required = set(itertools.product((0, 1), repeat=2))
    require(all(set(q) == required for q in quartets.values()), "Incomplete receipt quartet")
    return np.asarray(
        [[q[pair] for pair in sorted(required)] for q in quartets.values()], dtype=np.int64
    ).reshape(-1, 4)


def load_model(root, model, hashes):
    directory = root / (model + "_analysis")
    paths = {
        "manifest": directory / "manifest.json",
        "predictions": directory / "predictions.npz",
        "rows": root / model / "test/rows.json",
        "settings": root / (model + "_fit") / "settings.json",
        "heads": root / (model + "_fit") / "heads.npz",
    }
    for path in paths.values():
        hashes[str(path)] = digest(path)
    manifest = read(paths["manifest"])
    require(manifest["schema"] == "fresh-readout-analysis-manifest-v2", "Wrong old manifest")
    require(
        hashes[str(paths["predictions"])] == manifest["output_sha256"]["predictions.npz"],
        "Changed predictions",
    )
    for field in ("rows", "settings", "heads"):
        tail = (
            "/".join(paths[field].parts[-3:])
            if field == "rows"
            else "/".join(paths[field].parts[-2:])
        )
        expected = [
            v
            for p, v in manifest["source_sha256"].items()
            if p.replace("\\", "/").endswith("/" + tail)
        ]
        require(expected == [hashes[str(paths[field])]], "Changed old bound " + field)
    settings = read(paths["settings"])
    old_names = ("I", "M1", "B1", "B2")
    head_map = settings["pipelines"]["C2"]["factor_heads"]
    require(set(head_map) == set(old_names), "Wrong factor head inventory")
    for factor in old_names:
        head = settings["heads"][head_map[factor]]
        require(head["target"] == factor, "Wrong factor target mapping")
        require(
            head["site"]
            == {"llama": "layer_16_query_location_last", "qwen": "layer_21_query_location_last"}[
                model
            ],
            "Wrong frozen site",
        )
    payload = read(paths["rows"])
    require(payload["test_fixture"] is False, "Scientific fixture forbidden")
    with np.load(paths["predictions"], allow_pickle=False) as saved:
        arrays = {
            k: saved[k].copy()
            for k in ("row_ids", "factor_margins", "factor_predictions", "methods", "verdicts")
        }
    rows = align_rows(payload["rows"], arrays["row_ids"].tolist())
    validate_rows(rows)
    scores, prediction = arrays["factor_margins"], arrays["factor_predictions"]
    require(
        scores.shape == prediction.shape == (4992, 4) and np.isfinite(scores).all(),
        "Invalid factor margins",
    )
    require(np.array_equal(scores > 0, prediction), "Frozen zero-threshold mismatch")
    selected = np.asarray([i for i, r in enumerate(rows) if r["condition"] in STRATA[1:]])
    methods = arrays["methods"].tolist()
    require(len(set(methods)) == len(methods) and "C2" in methods, "Invalid method inventory")
    require(
        np.array_equal(
            fold(prediction[selected]), arrays["verdicts"][selected, methods.index("C2")]
        ),
        "Saved fold discrepancy",
    )
    return (
        rows,
        scores,
        prediction,
        {
            "factor_heads": head_map,
            "site": settings["heads"][head_map["I"]]["site"],
            "selected_rows": len(selected),
        },
    )


def analyze_model(model, rows, scores, prediction):
    selected = np.asarray([i for i, r in enumerate(rows) if r["condition"] in STRATA[1:]])
    truth = np.zeros((len(rows), 4), dtype=np.int64)
    truth[selected] = [[rows[i]["state"][field] for field in FIELDS] for i in selected]
    relevant = relevance(truth)
    gold = np.asarray([r["gold"] for r in rows])
    groups = sorted({r["group_id"] for r in rows})
    factors, comparisons, oracles = [], [], []
    predictions = [fold(substitute(prediction, truth, mask)) for mask in range(16)]
    require(np.array_equal(predictions[15][selected], gold[selected]), "Oracle all-truth failed")
    for renderer in ("A", "B"):
        for stratum in STRATA:
            pool = np.asarray(
                [
                    i
                    for i in selected
                    if rows[i]["renderer"] == renderer
                    and (stratum == "union" or rows[i]["condition"] == stratum)
                ]
            )
            for context in ("ALL", *groups):
                indices = (
                    pool
                    if context == "ALL"
                    else np.asarray([i for i in pool if rows[i]["group_id"] == context])
                )
                base = {
                    "model": model,
                    "renderer": renderer,
                    "stratum": stratum,
                    "context": context,
                }
                for column, factor in enumerate(FACTORS):
                    populations = {}
                    for status, flag in (("relevant", True), ("irrelevant", False)):
                        ix = indices[relevant[indices, column] == flag]
                        populations[status] = metrics(truth[ix, column], scores[ix, column])
                        factors.append(
                            {**base, "factor": factor, "relevance": status, **populations[status]}
                        )
                    difference = {}
                    for metric in (
                        "raw_oriented_auc",
                        "zero_accuracy",
                        "class0_recall",
                        "class1_recall",
                    ):
                        a, b = (
                            populations[status][metric] for status in ("relevant", "irrelevant")
                        )
                        difference[metric + "_relevant_minus_irrelevant"] = (
                            a - b if a is not None and b is not None else None
                        )
                    comparisons.append({**base, "factor": factor, **difference})
                quartets = quartet_indices(rows, indices)
                baseline = predictions[0] == gold
                for mask, verdict in enumerate(predictions):
                    correct = verdict == gold
                    counts = correct[quartets].sum(axis=1)
                    oracle = {
                        **base,
                        "mask": mask,
                        "truth_factors": "+".join(
                            f for j, f in enumerate(FACTORS) if mask & (1 << j)
                        )
                        or "none",
                        "rows": len(indices),
                        "row_accuracy": float(correct[indices].mean()),
                        "quartets": len(quartets),
                        "quartet_all_correct": float((counts == 4).mean()),
                        "row_accuracy_change_from_mask0": float(
                            correct[indices].mean() - baseline[indices].mean()
                        ),
                        "quartet_all_correct_change_from_mask0": float(
                            (counts == 4).mean() - baseline[quartets].all(axis=1).mean()
                        ),
                    }
                    oracle.update(
                        {f"quartets_with_{k}_correct": int((counts == k).sum()) for k in range(5)}
                    )
                    oracles.append(oracle)
    return factors, comparisons, oracles


def write_csv(path, records):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, default=Path(OLD_ROOT))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    require(
        args.old_root.resolve() == Path(OLD_ROOT).resolve(), "Only the declared old root is allowed"
    )
    seal = read(args.output / "spec_seal.json")
    require(
        digest(SPEC) == seal["spec_sha256"]
        and seal["old_raw_margins_accessed_before_seal"] is False,
        "Specification seal mismatch",
    )
    names = (
        "factor_metrics.csv",
        "factor_comparisons.csv",
        "oracle_metrics.csv",
        "summary.json",
        "analysis_receipt.json",
    )
    require(
        not any((args.output / name).exists() for name in names),
        "Preserve previous diagnostic outputs",
    )
    hashes, factors, comparisons, oracles, metadata = {}, [], [], [], {}
    first_read = datetime.now(UTC).isoformat()
    for model in ("llama", "qwen"):
        rows, scores, prediction, metadata[model] = load_model(args.old_root, model, hashes)
        a, b, c = analyze_model(model, rows, scores, prediction)
        factors.extend(a)
        comparisons.extend(b)
        oracles.extend(c)
    context_means = []
    for model, renderer, stratum, factor in itertools.product(
        ("llama", "qwen"), ("A", "B"), STRATA, FACTORS
    ):
        entries = [
            r
            for r in comparisons
            if (r["model"], r["renderer"], r["stratum"], r["factor"])
            == (model, renderer, stratum, factor)
            and r["context"] != "ALL"
        ]
        record = {"model": model, "renderer": renderer, "stratum": stratum, "factor": factor}
        for metric in ("raw_oriented_auc", "zero_accuracy", "class0_recall", "class1_recall"):
            key = metric + "_relevant_minus_irrelevant"
            values = [r[key] for r in entries if r[key] is not None]
            record[key] = {
                "mean": float(np.mean(values)) if values else None,
                "defined_contexts": len(values),
                "total_contexts": 24,
            }
        context_means.append(record)
    summary = {
        "schema": "old-logical-relevance-summary-v1",
        "scope": "all four factors, both relevance populations, both models/renderers, union plus easy/hard; all 16 oracle masks; all 24 contexts",
        "factor_order": list(FACTORS),
        "old_head_alias": {"U": "M1"},
        "metadata": metadata,
        "excluded_conditions": ["one_step", "asserted"],
        "zero_prediction_rule": "margin > 0",
        "pooled_factor_metrics": [r for r in factors if r["context"] == "ALL"],
        "within_context_relevance_differences": context_means,
        "pooled_oracle_metrics": [r for r in oracles if r["context"] == "ALL"],
        "relevance_definition": "Intervene on one true factor and hold all other true factors fixed; different from predicted-other-factor relevance.",
        "oracle_limit": "Privileged counterfactual of the supplied external Boolean algorithm; not deployable repair or language-model causal evidence.",
    }
    for name, records in ((names[0], factors), (names[1], comparisons), (names[2], oracles)):
        write_csv(args.output / name, records)
    write_json(args.output / "summary.json", summary)
    for path, value in hashes.items():
        require(digest(path) == value, "Old source changed during analysis")
    sources = [
        Path(__file__),
        ROOT / "tests/test_logical_relevance_old.py",
        SPEC,
        args.output / "spec_seal.json",
    ]
    write_json(
        args.output / "analysis_receipt.json",
        {
            "schema": "logical-relevance-old-analysis-receipt-v1",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "spec_sha256": seal["spec_sha256"],
            "spec_sealed_at_utc": seal["sealed_at_utc"],
            "old_data_first_load_at_utc": first_read,
            "old_input_sha256": hashes,
            "source_sha256": {str(p): digest(p) for p in sources},
            "output_sha256": {name: digest(args.output / name) for name in names[:-1]},
            "model_calls": 0,
            "fitting": False,
            "threshold_selection": False,
            "new_campaign_artifacts_accessed": False,
            "table_rows": {
                "factor_metrics": len(factors),
                "factor_comparisons": len(comparisons),
                "oracle_metrics": len(oracles),
            },
        },
    )
    print(
        json.dumps(
            {"output": str(args.output), "factor_rows": len(factors), "oracle_rows": len(oracles)}
        )
    )


if __name__ == "__main__":
    main()
