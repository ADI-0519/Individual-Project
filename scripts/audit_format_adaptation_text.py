"""Independent metric reconstruction for the posthoc lexical controls.

Only saved margins and formal test metadata are read. This does not refit the
lexical controls, verify their vocabulary, or make the exposed bank held out.
Production fitting and metric modules are deliberately not imported.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ARMS = ("A512", "Mixed512", "Ahalf256", "Offset")
VIEWS = ("full_messages", "system_only")
FACTORS = ("I", "M1", "B1", "B2")
FIELDS = ("initial", "update", "first_receipt", "receipt")
CONDITIONS = ("one_step", "correction_easy", "correction_hard", "asserted")
UNCERTAINTY = (
    "Paired whole contexts conditional on exposed authored bank, fixed lexical policies "
    "and fitted heads; excludes fitting-cohort uncertainty; posthoc diagnostic"
)


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def validate_rows(rows):
    """Reconstruct every expected cell rather than trusting saved gold labels."""
    require(isinstance(rows, list) and len(rows) == 4992, "Need 4992 test rows")
    require(
        all(isinstance(r, dict) and "messages" not in r for r in rows),
        "Rows must be metadata only",
    )
    ids = [r.get("row_id") for r in rows]
    require(
        all(isinstance(i, str) and i for i in ids) and len(set(ids)) == 4992,
        "Missing or duplicate row identity",
    )
    group_ids = {r["group_id"] for r in rows}
    require(len(group_ids) == 24, "Need 24 contexts")
    expected_fields = {
        "group_id",
        "mechanism",
        "initial",
        "update",
        "first_receipt",
        "receipt",
        "world",
        "name_map",
    }
    rosters = defaultdict(set)
    for row in rows:
        group, fmt, condition = row["group_id"], row["renderer"], row["condition"]
        require(fmt in ("A", "B") and condition in CONDITIONS, "Unknown test scope")
        require(row.get("partition") == row.get("split", "test") == "test", "Non-test row")
        state = row["state"]
        require(state.get("group_id") == group, "State/context disagreement")
        for field in ("world", "name_map"):
            require(type(state.get(field)) is int and state[field] in (0, 1), "Invalid nuisance")
        if condition == "asserted":
            require(
                set(state) == {"group_id", "mechanism", "asserted", "world", "name_map"}
                and state["mechanism"] == "asserted",
                "Invalid asserted state",
            )
            require(type(state.get("asserted")) is int and state["asserted"] in (0, 1), "Bad bit")
            target, family = state["asserted"], "asserted"
            cell = tuple(state[k] for k in ("asserted", "world", "name_map"))
        else:
            require(set(state) == expected_fields, "Invalid update-state fields")
            for field in ("initial", "update", "receipt"):
                require(type(state[field]) is int and state[field] in (0, 1), "Bad factor bit")
            if condition == "one_step":
                require(
                    state["mechanism"] == "message" and state["first_receipt"] is None,
                    "Invalid one-step state",
                )
                target = state["update"] if state["receipt"] else state["initial"]
                family = "one_step"
                cell = tuple(
                    state[k] for k in ("initial", "update", "receipt", "world", "name_map")
                )
            else:
                require(
                    state["mechanism"] == "correction"
                    and type(state["first_receipt"]) is int
                    and state["first_receipt"] in (0, 1),
                    "Invalid correction state",
                )
                require(
                    (state["initial"] == state["update"]) == (condition == "correction_easy"),
                    "Wrong correction stratum",
                )
                # Walk the accepted updates chronologically, rather than calling
                # the production nested-conditional fold.
                target = state["initial"]
                if state["first_receipt"]:
                    target = state["update"]
                if state["receipt"]:
                    target = 1 - state["update"]
                family = "correction"
                cell = tuple(state[k] for k in (*FIELDS, "world", "name_map"))
        require(type(row.get("gold")) is int and row["gold"] == target, "Gold/rule disagreement")
        require(cell not in rosters[(group, fmt, family)], "Duplicate formal state")
        rosters[(group, fmt, family)].add(cell)
    require(len(rosters) == 24 * 2 * 3, "Missing test family")
    for group, fmt, family in product(
        sorted(group_ids), ("A", "B"), ("one_step", "correction", "asserted")
    ):
        width = {"one_step": 5, "correction": 6, "asserted": 3}[family]
        require(
            rosters[(group, fmt, family)] == set(product((0, 1), repeat=width)), "Incomplete roster"
        )
    return sorted(group_ids)


def binary(truth, predicted):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    require(
        truth.ndim == 1
        and truth.shape == predicted.shape
        and truth.size > 0
        and np.isin(truth, (0, 1)).all()
        and np.isin(predicted, (0, 1)).all(),
        "Invalid binary arrays",
    )
    counts = {str(k): int(np.count_nonzero(truth == k)) for k in (0, 1)}
    correct = int(np.count_nonzero(truth == predicted))
    return {
        "rows": len(truth),
        "correct": correct,
        "row_accuracy": correct / len(truth),
        "class_counts": counts,
        "class_recalls": {
            str(k): float(np.count_nonzero((truth == k) & (predicted == k)) / counts[str(k)])
            if counts[str(k)]
            else None
            for k in (0, 1)
        },
    }


def verdict(rows, predicted):
    predicted = np.asarray(predicted)
    result = binary([r["gold"] for r in rows], predicted)
    panels = defaultdict(dict)
    pairs = {name: defaultdict(dict) for name in ("world", "name_map")}
    for row, value in zip(rows, predicted, strict=True):
        state, scope = row["state"], (row["group_id"], row["renderer"], row["condition"])
        if row["condition"] != "asserted":
            panel_id = (*scope, state["update"], state["world"], state["name_map"])
            bit_pair = (
                (state["initial"], state["receipt"])
                if row["condition"] == "one_step"
                else (state["first_receipt"], state["receipt"])
            )
            require(bit_pair not in panels[panel_id], "Duplicate quartet judgment")
            panels[panel_id][bit_pair] = int(value == row["gold"])
        for nuisance in pairs:
            key = (*scope, tuple(sorted((k, v) for k, v in state.items() if k != nuisance)))
            require(state[nuisance] not in pairs[nuisance][key], "Duplicate nuisance judgment")
            pairs[nuisance][key][state[nuisance]] = (int(value), row["gold"])
    require(
        all(set(p) == set(product((0, 1), repeat=2)) for p in panels.values()), "Incomplete quartet"
    )
    histogram = Counter(sum(p.values()) for p in panels.values())
    result.update(
        quartets=len(panels),
        complete_quartets=int(histogram[4]),
        quartet_accuracy=histogram[4] / len(panels) if panels else None,
        quartet_correct_cells_histogram={str(k): int(histogram[k]) for k in range(5)},
        nuisance_pairs={},
    )
    for nuisance, values in pairs.items():
        require(
            all(set(pair) == {0, 1} and pair[0][1] == pair[1][1] for pair in values.values()),
            "Incomplete or target-changing nuisance pair",
        )
        result["nuisance_pairs"][nuisance] = {
            "pairs": len(values),
            "verdict_flips": sum(p[0][0] != p[1][0] for p in values.values()),
            "both_correct": sum(p[0][0] == p[0][1] and p[1][0] == p[1][1] for p in values.values()),
        }
    return result


def factors(rows, margins):
    require(all(r["condition"].startswith("correction_") for r in rows), "No control factor truth")
    truth = np.array([[r["state"][field] for field in FIELDS] for r in rows])
    values = np.asarray(margins)
    require(values.shape == truth.shape and np.isfinite(values).all(), "Invalid factor margins")
    bits = values > 0
    result = {
        "rows": len(rows),
        "all_four_correct": float(np.mean(np.all(bits == truth, axis=1))),
        "factor_order": list(FACTORS),
        "factors": {},
    }
    for column, name in enumerate(FACTORS):
        entry = binary(truth[:, column], bits[:, column])
        entry["raw_auroc"] = float(roc_auc_score(truth[:, column], values[:, column]))
        entry["class_margin_quantiles_10_50_90"] = {
            str(label): np.quantile(
                values[truth[:, column] == label, column], (0.1, 0.5, 0.9)
            ).tolist()
            for label in (0, 1)
        }
        entry["positive_minus_negative_mean_margin"] = float(
            values[truth[:, column] == 1, column].mean()
            - values[truth[:, column] == 0, column].mean()
        )
        result["factors"][name] = entry
    return result


def contrast(context_values, reference):
    require(reference in ("A512", "Offset") and len(context_values) == 24, "Invalid contrast")
    names = sorted(context_values)
    left = np.array([context_values[g]["Mixed512"] for g in names], dtype=np.float64)
    right = np.array([context_values[g][reference] for g in names], dtype=np.float64)
    require(
        np.isin(left * 8, np.arange(9)).all() and np.isin(right * 8, np.arange(9)).all(),
        "Bad context panels",
    )
    delta = left - right
    sampled = np.random.default_rng(20261127).integers(0, 24, size=(10000, 24))
    interval = np.quantile(np.mean(delta[sampled], axis=1), (0.025, 0.975))
    return {
        "contrast": "Mixed512_minus_" + reference,
        "renderer": "B",
        "condition": "correction_hard",
        "endpoint": "complete_four_case_quartet_accuracy",
        "groups": 24,
        "quartets": 192,
        "Mixed512": float(left.mean()),
        reference: float(right.mean()),
        "difference": float(delta.mean()),
        "conditional_percentile_interval_95": interval.tolist(),
        "bootstrap_samples": 10000,
        "bootstrap_seed": 20261127,
        "uncertainty_scope": UNCERTAINTY,
        "role": "posthoc_descriptive_comparison",
        "group_values": [
            {"group_id": g, "Mixed512": float(a), reference: float(b), "difference": float(a - b)}
            for g, a, b in zip(names, left, right, strict=True)
        ],
    }


def reconstruct(rows, margin_arrays):
    groups = validate_rows(rows)
    result = {"arms": {}, "factor_diagnostics": {}, "comparisons": {}}
    context_values = {group: {} for group in groups}
    for arm in ARMS:
        values = margin_arrays[arm]
        require(
            np.asarray(values).shape == (4992, 4) and np.isfinite(values).all(),
            "Invalid margin shape/value",
        )
        bits = values > 0
        predictions = np.empty(len(rows), dtype=np.int64)
        # This rowwise update implementation is intentionally separate from the
        # production vectorized fold and handles exact-zero scores as false.
        for index, (initial, message, first, last) in enumerate(bits):
            current = int(initial)
            if first:
                current = int(message)
            if last:
                current = 1 - int(message)
            predictions[index] = current
        result["arms"][arm], result["factor_diagnostics"][arm] = {}, {}
        for fmt, condition in product(("A", "B"), CONDITIONS):
            selected = [
                i
                for i, r in enumerate(rows)
                if r["renderer"] == fmt and r["condition"] == condition
            ]
            entries = []
            for group in groups:
                within = [i for i in selected if rows[i]["group_id"] == group]
                metrics = verdict([rows[i] for i in within], predictions[within])
                entries.append({"group_id": group, **metrics})
                if fmt == "B" and condition == "correction_hard":
                    require(metrics["quartets"] == 8, "Wrong context hard-panel count")
                    context_values[group][arm] = metrics["quartet_accuracy"]
            result["arms"][arm][f"{fmt}/{condition}"] = {
                "aggregate": verdict([rows[i] for i in selected], predictions[selected]),
                "groups": entries,
                "scope": "correction-trained four-factor external fold"
                if condition.startswith("correction_")
                else "out_of_training_construct_control; no factor-truth claim",
            }
        for fmt, stratum in product(
            ("A", "B"), ("correction_all", "correction_easy", "correction_hard")
        ):
            selected = [
                i
                for i, r in enumerate(rows)
                if r["renderer"] == fmt
                and (
                    r["condition"].startswith("correction_")
                    if stratum == "correction_all"
                    else r["condition"] == stratum
                )
            ]
            entries = []
            for group in groups:
                within = [i for i in selected if rows[i]["group_id"] == group]
                entries.append(
                    {"group_id": group, **factors([rows[i] for i in within], values[within])}
                )
            result["factor_diagnostics"][arm][f"{fmt}/{stratum}"] = {
                "aggregate": factors([rows[i] for i in selected], values[selected]),
                "groups": entries,
            }
    for reference in ("A512", "Offset"):
        result["comparisons"]["Mixed512_minus_" + reference] = contrast(context_values, reference)
    return result


def compare_tree(expected, actual, path="", stats=None):
    if stats is None:
        stats = {"numeric": 0, "exact": 0, "maximum_numeric_difference": 0.0}
    if isinstance(expected, dict):
        require(
            isinstance(actual, dict) and set(expected) == set(actual), "Metric keys differ: " + path
        )
        for key, value in expected.items():
            compare_tree(value, actual[key], path + "/" + key, stats)
    elif isinstance(expected, list):
        require(
            isinstance(actual, list) and len(expected) == len(actual),
            "Metric list differs: " + path,
        )
        for index, value in enumerate(expected):
            compare_tree(value, actual[index], path + "/" + str(index), stats)
    elif isinstance(expected, float):
        require(
            type(actual) in (int, float) and np.isfinite(actual),
            "Nonfinite numeric metric: " + path,
        )
        difference = abs(expected - actual)
        require(
            np.isclose(expected, actual, rtol=1e-12, atol=1e-12), "Numeric metric differs: " + path
        )
        stats["numeric"] += 1
        stats["maximum_numeric_difference"] = max(stats["maximum_numeric_difference"], difference)
    else:
        require(
            type(actual) is type(expected) and actual == expected, "Exact metric differs: " + path
        )
        stats["exact"] += 1
    return stats


def audit(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    require(source.is_dir() and not output.exists(), "Use existing input and fresh audit directory")
    require(
        source != output and not output.is_relative_to(source),
        "Audit must be outside input directory",
    )
    names = {}
    for stem in ("rows", "summary"):
        candidates = [source / (stem + suffix) for suffix in (".json", ".json.gz")]
        present = [p for p in candidates if p.is_file()]
        require(len(present) == 1, "Need exactly one JSON or gzip input: " + stem)
        names[stem] = present[0]
    files = {path.name: digest(path) for path in (*names.values(), source / "predictions.npz")}
    output.mkdir(parents=True)
    receipt = {
        "schema": "format-adaptation-text-independent-metric-audit-v1",
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "source": str(source),
        "input_sha256": files,
        "auditor_sha256": digest(__file__),
        "model_calls": 0,
        "classifier_fits": 0,
        "scope": "Independent reconstruction from saved margins and formal test metadata; no feature or fit audit",
    }
    try:
        payload = read(names["rows"])
        rows = payload["rows"] if isinstance(payload, dict) else payload
        validate_rows(rows)
        with np.load(source / "predictions.npz", allow_pickle=False) as saved:
            expected_keys = {f"{view}__{arm}__factor_margins" for view in VIEWS for arm in ARMS} | {
                "row_ids"
            }
            require(set(saved.files) == expected_keys, "Unexpected prediction-array inventory")
            row_ids = saved["row_ids"]
            require(
                row_ids.dtype.kind == "U"
                and row_ids.shape == (4992,)
                and row_ids.tolist() == [r["row_id"] for r in rows],
                "Prediction-row ordering differs",
            )
            arrays = {name: saved[name].copy() for name in expected_keys - {"row_ids"}}
        reconstructions = {
            view: reconstruct(rows, {arm: arrays[f"{view}__{arm}__factor_margins"] for arm in ARMS})
            for view in VIEWS
        }
        # Finish both views and freeze their exact bytes before opening summary.
        for view, values in reconstructions.items():
            write(output / (view + "_reconstruction.json"), values)
        write(
            output / "reconstruction_seal.json",
            {
                "schema": "format-adaptation-text-metric-reconstruction-seal-v1",
                "saved_summary_opened": False,
                "views_reconstructed": list(VIEWS),
                "output_sha256": {
                    view + "_reconstruction.json": digest(output / (view + "_reconstruction.json"))
                    for view in VIEWS
                },
            },
        )
        summary = read(names["summary"])
        require(set(summary["views"]) == set(VIEWS), "Missing/extra saved view")
        receipt["views"] = {
            view: compare_tree(reconstructions[view], summary["views"][view]) for view in VIEWS
        }
        require(
            all(digest(source / name) == sha for name, sha in files.items()),
            "Input changed during audit",
        )
        receipt["status"] = "pass"
        write(output / "audit_receipt.json", receipt)
        return receipt
    except BaseException as error:
        receipt.update(status="failed_preserved", error_type=type(error).__name__, error=str(error))
        write(output / "audit_receipt.json", receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    print(json.dumps(audit(options.source, options.output), indent=2))


if __name__ == "__main__":
    main()
