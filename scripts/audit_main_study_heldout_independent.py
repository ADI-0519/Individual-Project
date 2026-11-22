"""Reconstruct saved held-out results with NumPy; never fit or select a model."""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_array
from scipy.special import expit


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def native(path):
    """Read Linux absolute paths from either native Linux or Windows UNC."""
    if sys.platform == "win32" and str(path).startswith("/mnt/"):
        parts = str(path).split("/", 3)
        return Path(parts[2].upper() + ":/" + parts[3])
    if sys.platform == "win32" and str(path).startswith("/"):
        return Path("//wsl.localhost/Ubuntu" + str(path))
    return Path(path)


def sigmoid(margins):
    return expit(margins)


def normalize(values):
    norms = np.sqrt(np.einsum("ij,ij->i", values, values))
    norms[norms < 10 * np.finfo(np.float64).eps] = 1
    return values / norms[:, None]


def belief(state):
    before = state["initial"]
    if state["mechanism"] == "correction":
        before = state["update"] if state["first_receipt"] else before
        final = 1 - state["update"]
    else:
        final = state["update"]
    return final if state["receipt"] else before


def graph(rows):
    """Build quartets and receipt pairs directly from factorial states."""
    indices = collections.defaultdict(list)
    for index, row in enumerate(rows):
        require(belief(row["state"]) == row["gold"], "State/label disagreement")
        indices[row["group_id"]].append(index)
    result = {}
    for group, selected in indices.items():
        mechanism = rows[selected[0]]["mechanism"]
        require(len(selected) == (64 if mechanism == "correction" else 32), "Incomplete group")
        by_state = {}
        pairs = collections.defaultdict(dict)
        for index in selected:
            state = rows[index]["state"]
            key = tuple(
                state[k]
                for k in ("initial", "update", "receipt", "world", "name_map", "first_receipt")
            )
            require(key not in by_state, "Duplicate factorial state")
            by_state[key] = index
            pairs[key[:2] + key[3:]][state["receipt"]] = index
        quartets = []
        for update in (0, 1):
            for world in (0, 1):
                for names in (0, 1):
                    if mechanism == "correction":
                        keys = [
                            (1 - update, update, receipt, world, names, first)
                            for first in (1, 0)
                            for receipt in (0, 1)
                        ]
                    else:
                        keys = [
                            (initial, update, receipt, world, names, None)
                            for initial in (1 - update, update)
                            for receipt in (0, 1)
                        ]
                    quartets.append([by_state[key] for key in keys])
        require(len(quartets) == 8, "Quartet count")
        pair_list = []
        for pair in pairs.values():
            require(set(pair) == {0, 1}, "Incomplete receipt pair")
            pair_list.append((pair[0], pair[1]))
        result[group] = {
            "indices": selected,
            "mechanism": mechanism,
            "quartets": quartets,
            "pairs": pair_list,
            "secondary": [
                i
                for i in selected
                if mechanism == "correction"
                and rows[i]["state"]["initial"] == rows[i]["state"]["update"]
            ],
        }
    return result


class Verifier:
    def __init__(self, rows):
        self.rows = rows
        self.labels = np.array([belief(row["state"]) for row in rows])
        self.groups = graph(rows)
        self.numeric_fields = 0
        self.quartets = 0
        self.pairs = 0
        self.max_metric_error = 0.0

    def number(self, actual, expected, name, tolerance=5e-10):
        require(actual is not None and np.isfinite(actual), "Nonfinite " + name)
        error = abs(float(actual) - float(expected))
        require(error <= tolerance, f"{name}: {actual} != {expected}")
        self.max_metric_error = max(self.max_metric_error, error)
        self.numeric_fields += 1

    def binary(self, saved, selected, margins):
        labels = self.labels[selected]
        probabilities = sigmoid(margins[selected])
        predicted = margins[selected] > 0
        expected = {
            "rows": len(selected),
            "correct": int((predicted == labels).sum()),
            "accuracy": float((predicted == labels).mean()),
            "recall_0": float((predicted[labels == 0] == 0).mean()),
            "recall_1": float((predicted[labels == 1] == 1).mean()),
            "brier": float(((probabilities - labels) ** 2).mean()),
        }
        positive = probabilities[labels == 1]
        negative = probabilities[labels == 0]
        expected["auroc"] = float(
            (
                (positive[:, None] > negative).astype(float) + 0.5 * (positive[:, None] == negative)
            ).mean()
        )
        require(set(saved) == set(expected), "Binary metric schema")
        for name, value in expected.items():
            self.number(saved[name], value, name)

    def group_metrics(self, saved_groups, margins):
        require(len(saved_groups) == len(self.groups), "Metric group count")
        saved = {item["group_id"]: item for item in saved_groups}
        require(set(saved) == set(self.groups), "Metric group inventory")
        strict = {}
        correct = (margins > 0) == self.labels
        for group, layout in self.groups.items():
            record = saved[group]
            require(record["mechanism"] == layout["mechanism"], "Metric mechanism")
            self.binary(record["all_rows"], layout["indices"], margins)
            quartets = {frozenset(item["case_ids"]): item for item in record["quartets"]}
            expected_quartets = {}
            for quartet in layout["quartets"]:
                ids = frozenset(self.rows[i]["case_id"] for i in quartet)
                expected_quartets[ids] = bool(correct[quartet].all())
            require(set(quartets) == set(expected_quartets), "Quartet cell membership")
            for ids, expected in expected_quartets.items():
                require(quartets[ids]["all_four_correct"] == expected, "Strict quartet verdict")
                self.quartets += 1
            strict[group] = float(np.mean(list(expected_quartets.values())))
            self.number(record["complete_quartet_accuracy"], strict[group], "Quartet accuracy")
            pairs = {tuple(item["case_ids"]): item for item in record["pairs"]}
            require(len(pairs) == len(layout["pairs"]), "Pair count")
            buckets = {True: [], False: []}
            for before, after in layout["pairs"]:
                ids = (self.rows[before]["case_id"], self.rows[after]["case_id"])
                require(ids in pairs, "Pair membership")
                item = pairs[ids]
                changes = bool(self.labels[before] != self.labels[after])
                both = bool(correct[before] and correct[after])
                flip = bool((margins[before] > 0) != (margins[after] > 0))
                difference = float(margins[after] - margins[before])
                require(
                    item["changes_if_received"] == changes
                    and item["both_correct"] == both
                    and item["verdict_flip"] == flip,
                    "Pair verdict",
                )
                self.number(item["margin_difference"], difference, "Pair margin")
                oriented = (2 * self.labels[after] - 1) * difference if changes else None
                if oriented is None:
                    require(item["oriented_margin_difference"] is None, "Preserving orientation")
                else:
                    self.number(item["oriented_margin_difference"], oriented, "Pair orientation")
                buckets[changes].append((both, flip, difference, oriented))
                self.pairs += 1
            for changes, name in ((True, "changing"), (False, "preserving")):
                values = buckets[changes]
                require(record[name]["pairs"] == len(values), "Pair category count")
                for field, expected in {
                    "both_correct_fraction": np.mean([v[0] for v in values]),
                    "verdict_flip_fraction": np.mean([v[1] for v in values]),
                    "mean_absolute_margin_difference": np.mean([abs(v[2]) for v in values]),
                }.items():
                    self.number(record[name][field], expected, field)
                if changes:
                    self.number(
                        record[name]["mean_oriented_margin_difference"],
                        np.mean([v[3] for v in values]),
                        "Mean pair orientation",
                    )
            if layout["mechanism"] == "correction":
                self.binary(record["secondary_correction_cells"], layout["secondary"], margins)
        return strict


def verify_primary(primary, saved, verifier):
    groups = sorted(next(iter(primary.values()))["restricted"])
    seeds = sorted(primary)
    require(len(groups) == 30 and seeds == [11, 23, 37, 53, 71], "Primary input inventory")
    require(
        len(saved["by_group"]) == 30 and len(saved["by_seed"]) == 5,
        "Primary aggregation inventory",
    )
    restricted = np.array(
        [[primary[seed]["restricted"][group] for seed in seeds] for group in groups]
    )
    full = np.array([[primary[seed]["full"][group] for seed in seeds] for group in groups])
    differences = (full - restricted).mean(axis=1)
    bootstrap = differences[np.random.default_rng(20261121).integers(0, 30, size=(10000, 30))].mean(
        axis=1
    )
    reconstructed_primary = {
        "restricted": float(restricted.mean()),
        "full": float(full.mean()),
        "full_minus_restricted": float(differences.mean()),
        "percentile_95_interval": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
    }
    for key in ("restricted", "full", "full_minus_restricted"):
        verifier.number(saved[key], reconstructed_primary[key], "Primary " + key, 1e-15)
    require(
        saved["allocation_seeds"] == seeds == [11, 23, 37, 53, 71],
        "Primary allocations",
    )
    require(
        saved["bootstrap_draws"] == 10000
        and saved["bootstrap_seed"] == 20261121
        and saved["independent_scenario_groups"] == 30,
        "Bootstrap specification",
    )
    require(
        np.array_equal(
            saved["percentile_95_interval"],
            reconstructed_primary["percentile_95_interval"],
        ),
        "Bootstrap interval differs",
    )
    for i, item in enumerate(saved["by_group"]):
        require(
            item["group_id"] == groups[i]
            and item["restricted_by_seed"] == restricted[i].tolist()
            and item["full_by_seed"] == full[i].tolist(),
            "Primary group-seed aggregation",
        )
        for key, value in {
            "restricted_mean": restricted[i].mean(),
            "full_mean": full[i].mean(),
            "difference": differences[i],
        }.items():
            verifier.number(item[key], value, "Primary group mean", 1e-15)
    for j, item in enumerate(saved["by_seed"]):
        require(item["seed"] == seeds[j], "Primary seed order")
        for key, value in {
            "restricted": restricted[:, j].mean(),
            "full": full[:, j].mean(),
            "difference": (full[:, j] - restricted[:, j]).mean(),
        }.items():
            verifier.number(item[key], value, "Primary seed mean", 1e-15)
    return reconstructed_primary, restricted, full, seeds


def verify_scores(ids, record, margins):
    require([s["case_id"] for s in record["scores"]] == ids, "Saved score identity/order")
    old_margins = np.array([s["margin"] for s in record["scores"]])
    old_probabilities = np.array([s["probability"] for s in record["scores"]])
    max_margin_error = float(np.max(np.abs(old_margins - margins)))
    max_probability_error = float(np.max(np.abs(old_probabilities - sigmoid(margins))))
    require(np.allclose(old_margins, margins, rtol=1e-12, atol=5e-10), "Saved margin differs")
    require(max_probability_error <= 5e-12, "Saved probability differs")
    require(
        np.array_equal([s["prediction"] for s in record["scores"]], margins > 0),
        "Raw prediction differs",
    )
    return max_margin_error, max_probability_error


def verify_recovery(args, receipt, ids, model):
    if model == "qwen":
        require(
            str(native(receipt["recovery_capture"])) != str(args.capture),
            "Qwen cannot reuse Llama cache",
        )
        return {
            "retained_measurements": 0,
            "new_measurements": 3840,
            "unique_analysis_row_per_case": True,
            "possible_duplicate_computational_attempts": 0,
            "duplicate_retained_measurements": 0,
            "missing_cases": 0,
            "campaign_lost_attempt_applies_to": "llama",
            "campaign_recovery_receipt_sha256": digest(args.recovery / "recovery_receipt.json"),
        }
    require(
        receipt["retained_case_ids"] + receipt["remaining_case_ids"] == ids,
        "Recovery retained/new inventory",
    )
    require(
        receipt["retained_completed_rows"] == 2758 and receipt["remaining_rows"] == 1082,
        "Recovery counts",
    )
    original = native(receipt["original_capture"])
    copied = receipt["copied_artifact_sha256"]
    for relative, expected in copied.items():
        require(digest(original / relative) == expected, "Original cached measurement changed")
        if relative != "manifest.json":
            require(digest(args.capture / relative) == expected, "Copied measurement changed")
    unknown = receipt["unknown_attempt"]
    old_unknown = native(unknown["original_directory"])
    require(
        {p.name for p in old_unknown.iterdir()} == {"started.json"},
        "Original unknown attempt changed",
    )
    require(
        digest(old_unknown / "started.json") == unknown["artifact_sha256"]["started.json"],
        "Original unknown start changed",
    )
    require(
        ids.count(unknown["case_id"]) == 1
        and unknown["planned_repeat_count"] == 1
        and not unknown["counts_as_new_independent_case"],
        "Duplicate recovery measurement",
    )
    return {
        "retained_measurements": 2758,
        "new_measurements": 1082,
        "all_original_and_copied_retained_hashes_match": True,
        "original_manifest_preserved_recovery_manifest_legitimately_finalized": True,
        "original_unknown_still_started_only": True,
        "unique_analysis_row_per_case": True,
        "lost_attempt_case_id": unknown["case_id"],
        "possible_duplicate_computational_attempts": 1,
        "duplicate_retained_measurements": 0,
        "missing_cases": 0,
        "receipt_sha256": digest(args.recovery / "recovery_receipt.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "fit", "analysis", "recovery", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve prior audit; use a fresh output")
    started = time.perf_counter()
    paths = {
        "analysis": args.analysis,
        "rows": args.capture / "rows.json",
        "features": args.capture / "features.npz",
        "fit_settings": args.fit / "settings.json",
        "fit_heads": args.fit / "heads.npz",
        "recovery_receipt": args.recovery / "recovery_receipt.json",
    }
    hashes = {name: digest(path) for name, path in paths.items()}
    analysis, payload, settings, receipt = (
        read(paths[key]) for key in ("analysis", "rows", "fit_settings", "recovery_receipt")
    )
    rows = payload["rows"]
    protocol = read(args.capture / "protocol.json")
    model = protocol["model_key"]
    planned = [row["case_id"] for row in protocol["planned_rows"] if row["split"] == "test"]
    ids = [row["case_id"] for row in rows]
    require(ids == planned and len(ids) == len(set(ids)) == 3840, "Exact unique held-out inventory")
    require(all(row["split"] == "test" for row in rows), "Non-test row")
    require(
        analysis["capture_source_hashes"]
        == {"rows.json": hashes["rows"], "features.npz": hashes["features"]}
        and analysis["capture_provenance"] == payload["provenance"],
        "Analysis capture bindings",
    )
    require(
        analysis["fit_settings_sha256"] == hashes["fit_settings"]
        and analysis["fit_heads_sha256"] == hashes["fit_heads"] == settings["heads_sha256"],
        "Original fit binding",
    )
    bindings = payload["provenance"]["source_bindings"]
    for path, expected in bindings.items():
        require(digest(native(path)) == expected, "Frozen source/fit binding changed: " + path)
    original_fits = {
        "llama": (
            "d8bcaba0573de37632fe13aa4e812d95e0b44f88c9a6c1bc7ac64543236c0fd1",
            "45fbd2d149540247ae2c928f57ea9a1ef37befcedba4dc006d6e4d8238fe173e",
        ),
        "qwen": (
            "a56e81928347be7f8854f4f9a7047a0c0afbe2ca589988e3291c69b3d79dd032",
            "c86d77f8ccd050a4a51365b1740434247bb8d6eec509806113eead5801bae609",
        ),
    }
    require(
        model in original_fits
        and (hashes["fit_settings"], hashes["fit_heads"]) == original_fits[model],
        "Original frozen heads changed",
    )
    with np.load(paths["features"], allow_pickle=False) as data:
        features = data["features"].astype(np.float64)
        require(data["row_ids"].tolist() == ids, "Feature row order")
    with np.load(paths["fit_heads"], allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    require(features.shape == (3840, protocol["hidden_size"]), "Feature dimensions")
    for key, value in arrays.items():
        require(
            np.isfinite(value).all()
            and hashlib.sha256(value.tobytes()).hexdigest() == settings["array_sha256"][key],
            "Saved head array integrity",
        )
    verifier = Verifier(rows)
    require(len(verifier.groups) == 90, "Held-out scenario count")
    require(
        collections.Counter(g["mechanism"] for g in verifier.groups.values())
        == {"message": 30, "observation": 30, "correction": 30},
        "Mechanism group balance",
    )
    require(not (set(verifier.groups) & set(settings["fitting_group_splits"])), "Group leakage")
    require(
        settings["policy"]["raw_threshold"] == 0.5 and settings["policy"]["tie_class"] == 0,
        "Threshold",
    )
    expected_comparisons = {
        (kind, n, seed)
        for kind in ("activation", "tfidf")
        for n in (128, 256, 512)
        for seed in (11, 23, 37, 53, 71)
    }
    actual_comparisons = {
        (c["feature_kind"], c["training_n"], c["allocation_seed"]): c
        for c in analysis["comparisons"]
    }
    require(
        set(actual_comparisons) == expected_comparisons and len(analysis["comparisons"]) == 30,
        "Comparison inventory",
    )
    pattern = re.compile(r"(?u)\b\w+\b")
    documents = []
    for row in rows:
        text = "\n".join(m.get("role", "") + ": " + m["content"] for m in row["messages"]).lower()
        tokens = pattern.findall(text)
        documents.append(
            collections.Counter(tokens + [" ".join(pair) for pair in zip(tokens, tokens[1:])])
        )
    primary = {}
    arms = []
    for comparison in settings["comparisons"]:
        kind, n, seed = (comparison[k] for k in ("feature_kind", "training_n", "allocation_seed"))
        saved = actual_comparisons[kind, n, seed]
        require(saved["status"] == comparison["status"] == "complete", "Failed comparison")
        for arm in ("restricted", "full"):
            head, record = comparison["arms"][arm], saved["arms"][arm]
            require(
                not (set(head["training_ids"] + comparison["development_ids"]) & set(ids)),
                "Row leakage",
            )
            prefix = head["array_prefix"]
            if kind == "activation":
                x = normalize((features - arrays[prefix + "__mean"]) / arrays[prefix + "__scale"])
            else:
                vocabulary, idf = head["vocabulary"], arrays[prefix + "__idf"]
                columns, counts_data, offsets = [], [], [0]
                for counts in documents:
                    row_counts = []
                    for term, count in counts.items():
                        j = vocabulary.get(term)
                        if j is not None:
                            row_counts.append((j, count))
                    row_counts.sort()
                    columns.extend(j for j, _ in row_counts)
                    counts_data.extend(count for _, count in row_counts)
                    offsets.append(len(columns))
                columns = np.asarray(columns, dtype=np.int32)
                values = np.log(np.asarray(counts_data, dtype=np.float64))
                values += 1.0
                values *= idf[columns]
                for before, after in zip(offsets, offsets[1:]):
                    squared_norm = 0.0
                    for value in values[before:after]:
                        squared_norm += value * value
                    if squared_norm:
                        values[before:after] /= math.sqrt(squared_norm)
                x = csr_array(
                    (values, columns, np.asarray(offsets, dtype=np.int32)),
                    shape=(len(rows), len(vocabulary)),
                )
            margins = x @ arrays[prefix + "__coef"] + arrays[prefix + "__intercept"][0]
            del x
            max_margin_error, max_probability_error = verify_scores(ids, record, margins)
            strict = verifier.group_metrics(record["groups"], margins)
            calibration = head["calibration"]
            require(calibration["status"] == "complete", "Missing calibration")
            verifier.group_metrics(
                record["calibrated_groups"],
                calibration["slope"] * margins + calibration["intercept"],
            )
            if kind == "activation" and n == 512:
                primary.setdefault(seed, {})[arm] = {
                    group: value
                    for group, value in strict.items()
                    if verifier.groups[group]["mechanism"] == "correction"
                }
            arms.append(
                {
                    "kind": kind,
                    "training_n": n,
                    "seed": seed,
                    "arm": arm,
                    "rows": len(rows),
                    "raw_and_calibrated_group_metrics_verified": 90,
                    "max_saved_margin_absolute_error": max_margin_error,
                    "max_saved_probability_absolute_error": max_probability_error,
                    "independent_margin_sha256": canonical(margins.tolist()),
                }
            )
        print(json.dumps({"verified_comparison": f"{kind}_n{n}_seed{seed}"}), flush=True)
    reconstructed_primary, restricted, full, seeds = verify_primary(
        primary, analysis["primary"], verifier
    )
    baseline_margins = {
        "constant_0": np.full(len(rows), -1.0),
        "initial": np.array([2 * r["state"]["initial"] - 1 for r in rows], dtype=float),
        "final_update": np.array(
            [
                2
                * (
                    (1 - r["state"]["update"])
                    if r["mechanism"] == "correction"
                    else r["state"]["update"]
                )
                - 1
                for r in rows
            ],
            dtype=float,
        ),
        "world": np.array([2 * r["state"]["world"] - 1 for r in rows], dtype=float),
        "symbolic_oracle": 2 * verifier.labels - 1.0,
        "output_raw": np.array([r["output_yes_no_logitdiff"] for r in rows]),
    }
    baseline_margins["output_calibrated"] = (
        settings["output_calibration"]["slope"] * baseline_margins["output_raw"]
        + settings["output_calibration"]["intercept"]
    )
    require(set(analysis["baselines"]) == set(baseline_margins), "Baseline inventory")
    for name, margins in baseline_margins.items():
        verifier.group_metrics(analysis["baselines"][name], margins)
    recovery_summary = verify_recovery(args, receipt, ids, model)
    tokens = protocol["tokenizer"]["answer_token_ids"]
    for i, row in enumerate(rows):
        item = args.capture / "row_journal" / row["row_id"]
        record = read(item / "complete.json")
        require(all(row.get(k) == v for k, v in record.items()), "Journal/export row differs")
        require(
            digest(item / "arrays.npz") == row["arrays_sha256"], "Saved measurement hash differs"
        )
        with np.load(item / "arrays.npz", allow_pickle=False) as data:
            require(np.array_equal(data["feature"], features[i]), "Feature export differs")
            logits = data["next_token_logits"]
            difference = float(logits[tokens["YES"]]) - float(logits[tokens["NO"]])
            require(
                difference == baseline_margins["output_raw"][i],
                "Output baseline differs from saved logits",
            )
    require(
        all(digest(path) == hashes[name] for name, path in paths.items()),
        "Inputs changed during audit",
    )
    report = {
        "schema": "main-study-heldout-independent-audit-v1",
        "reviewer_type": "agent",
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "status": "passed",
        "model": model,
        "research_status": "agent_audited_exploratory",
        "human_review_complete": False,
        "confirmatory_result": False,
        "source_hashes": hashes,
        "auditor_source_sha256": digest(__file__),
        "coverage": {
            "test_rows": 3840,
            "groups": 90,
            "groups_per_mechanism": 30,
            "activation_allocations": 15,
            "tfidf_allocations": 15,
            "heads": len(arms),
            "raw_and_calibrated_group_records": 60 * 90 * 2,
            "strict_quartet_records_checked_including_baselines": verifier.quartets,
            "receipt_pair_records_checked_including_baselines": verifier.pairs,
            "numeric_metric_fields_checked": verifier.numeric_fields,
            "saved_logit_output_baseline_rows_verified": 3840,
        },
        "primary_reconstructed": reconstructed_primary,
        "primary_sampling": {
            "correction_groups": 30,
            "paired_allocation_seeds": seeds,
            "quartets_per_group_per_seed": 8,
            "quartet_evaluations_per_arm": 1200,
            "restricted_successful_quartets": int(restricted.sum() * 8),
            "full_successful_quartets": int(full.sum() * 8),
            "bootstrap_draws": 10000,
            "bootstrap_seed": 20261121,
            "bootstrap_unit": "whole narrative group; five paired seeds retained",
            "interval_exactly_matches_saved_analysis": True,
        },
        "recovery": recovery_summary,
        "arms": arms,
        "baselines_verified": list(baseline_margins),
        "maximum_numeric_metric_absolute_error": verifier.max_metric_error,
        "arithmetic_note": "TF-IDF term counts, IDF weighting and row normalization were reconstructed with sorted sparse accumulation, and scipy.special.expit reproduces probability rounding. A dense alternative can perturb near-tied TF-IDF AUROC ranks at floating-point precision even when margins match within tolerance.",
        "frozen_sources_fits_and_test_freeze_unchanged": True,
        "new_model_calls": 0,
        "new_fits_or_selections": 0,
        "project_fit_analysis_modules_imported": any(
            n.startswith("latent_probes") for n in sys.modules
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "Independent NumPy reconstruction, not new model observations or human semantic validation.",
            "The 30 correction narrative groups, not 1200 quartet evaluations or five training seeds, are bootstrap units.",
            "Intervals condition on the authored bank, shared renderer and fixed training allocations; no independent-template or causal-mechanism claim follows.",
            (
                "Llama recovery retained one measurement per case but repeats one computational attempt whose original outcome was lost; this remains disclosed exploratory evidence."
                if model == "llama"
                else "Qwen was captured afresh after campaign recovery; the earlier lost Llama attempt is not an additional Qwen case."
            ),
            "The prior full raw-capture audit remains responsible for raw_forward tensor identity; this audit additionally verifies all saved-logit baseline values, exported features, and retained original/copy hashes.",
            "Only this model's finalized held-out analysis and shared frozen inputs were read; no prose-monitor results, fitting, threshold changes or outcome-driven exclusions were used.",
        ],
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": digest(args.output),
                "primary": reconstructed_primary,
                "elapsed_seconds": report["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
