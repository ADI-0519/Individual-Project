#!/usr/bin/env python3
"""POSTHOC fixed-orientation diagnostics of saved frozen readout scores.

Independent of scientific production modules: stdlib and NumPy only, no model
calls, feature tensors, fitting, threshold selection, or output overwrite.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

CONDITIONS = ("correction_easy", "correction_hard", "one_step", "asserted")
FACTORS = {"I": "initial", "M1": "update", "B1": "first_receipt", "B2": "receipt"}
METHODS = ("S0", "S1", "C1", "output")
PROMPT_HASHES = ("messages_sha256", "rendered_chat_sha256", "prompt_token_ids_sha256")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    check(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(), "Bad vector")
    q = np.quantile(values, [0, 0.25, 0.5, 0.75, 1], method="linear")
    return {
        "n": len(values), "mean": float(values.mean()), "sd_population": float(values.std()),
        "min": float(q[0]), "q25": float(q[1]), "median": float(q[2]),
        "q75": float(q[3]), "max": float(q[4]), "fraction_gt_zero": float((values > 0).mean()),
    }


def average_ranks(values):
    values = np.asarray(values)
    ranks = np.empty(len(values), dtype=np.float64)
    order = np.argsort(values, kind="stable")
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def raw_auc(labels, scores):
    """Positive-is-target-1 AUROC, exact score ties receive half credit."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    check(labels.shape == scores.shape and labels.ndim == 1, "AUROC shape mismatch")
    check(np.isin(labels, (0, 1)).all() and np.isfinite(scores).all(), "Invalid AUROC inputs")
    n1, n0 = int((labels == 1).sum()), int((labels == 0).sum())
    if not n1 or not n0:
        return None
    return float((average_ranks(scores)[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def correlation(a, b, *, rank=False):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if rank:
        a, b = average_ranks(a), average_ranks(b)
    a, b = a - a.mean(), b - b.mean()
    divisor = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return None if divisor == 0 else float(np.clip(np.sum(a * b) / divisor, -1, 1))


def score_metrics(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    check(labels.shape == scores.shape and len(scores) > 0, "Invalid score-metric shape")
    predicted = scores > 0
    classes = {}
    for label in (0, 1):
        selected = labels == label
        classes[str(label)] = {
            "n": int(selected.sum()),
            "recall_at_zero": float((predicted[selected] == label).mean()) if selected.any() else None,
            "margin": distribution(scores[selected]) if selected.any() else None,
        }
    separation = None
    if classes["0"]["n"] and classes["1"]["n"]:
        separation = float(scores[labels == 1].mean() - scores[labels == 0].mean())
    return {
        "n": len(scores), "raw_auroc": raw_auc(labels, scores),
        "accuracy_at_zero": float((predicted == labels).mean()),
        "predicted_positive_rate": float(predicted.mean()),
        "positive_minus_negative_mean_margin": separation, "classes": classes,
    }


def canonical_state(row, *, exclude=()):
    return json.dumps(
        {k: v for k, v in row["state"].items() if k not in exclude},
        sort_keys=True, separators=(",", ":"),
    )


def formal_key(row):
    return row["group_id"], row["condition"], canonical_state(row)


def align_rows(rows, row_ids):
    """Index by explicit IDs, never by incidental sort or incoming order."""
    index = {}
    for row in rows:
        identity = row["row_id"]
        check(identity not in index, "Duplicate metadata row ID: " + identity)
        index[identity] = row
    ids = list(row_ids)
    check(len(ids) == len(set(ids)), "Duplicate array row IDs")
    check(set(ids) == set(index), "Missing/extra array or metadata row IDs")
    return [index[identity] for identity in ids]


def ab_pairs(rows):
    joined = defaultdict(dict)
    for index, row in enumerate(rows):
        renderer = row["renderer"]
        check(renderer in ("A", "B"), "Unexpected renderer")
        check(row["state"]["group_id"] == row["group_id"], "State/context ID mismatch")
        pair = joined[formal_key(row)]
        check(renderer not in pair, "Duplicate formal A/B state")
        pair[renderer] = index
    pairs = []
    for key, pair in sorted(joined.items()):
        check(set(pair) == {"A", "B"}, "Missing formal A/B counterpart")
        a, b = rows[pair["A"]], rows[pair["B"]]
        check(a["gold"] == b["gold"], "A/B formal target mismatch")
        check(a["row_id"] != b["row_id"], "A/B prompt identities coincide")
        for name in PROMPT_HASHES:
            if name in a or name in b:
                check(a.get(name) and b.get(name) and a[name] != b[name], "A/B prompt hash invalid")
        pairs.append((key, pair["A"], pair["B"]))
    return pairs


def paired_metrics(labels, a, b):
    labels, a, b = np.asarray(labels), np.asarray(a), np.asarray(b)
    pa, pb = a > 0, b > 0
    ca, cb = pa == labels, pb == labels
    return {
        "n": len(a), "pearson": correlation(a, b), "spearman": correlation(a, b, rank=True),
        "verdict_agreement": float((pa == pb).mean()),
        "correct_A_correct_B": int((ca & cb).sum()),
        "correct_A_wrong_B": int((ca & ~cb).sum()),
        "wrong_A_correct_B": int((~ca & cb).sum()),
        "wrong_A_wrong_B": int((~ca & ~cb).sum()),
        "B_minus_A": distribution(b - a), "absolute_B_minus_A": distribution(abs(b - a)),
        "classes": {str(label): distribution((b - a)[labels == label])
                    for label in (0, 1) if (labels == label).any()},
    }


def order_counts(labels, a, b):
    """All positive-negative comparisons in one context; no cross-context comparisons."""
    labels, a, b = np.asarray(labels), np.asarray(a), np.asarray(b)
    da = np.sign(a[labels == 1, None] - a[labels == 0][None, :])
    db = np.sign(b[labels == 1, None] - b[labels == 0][None, :])
    return {
        "n_pairs": int(da.size),
        "both_correct_order": int(((da > 0) & (db > 0)).sum()),
        "A_only_correct_order": int(((da > 0) & (db < 0)).sum()),
        "B_only_correct_order": int(((da < 0) & (db > 0)).sum()),
        "both_wrong_order": int(((da < 0) & (db < 0)).sum()),
        "either_tied": int(((da == 0) | (db == 0)).sum()),
        "A_wins": int((da > 0).sum()), "A_ties": int((da == 0).sum()),
        "B_wins": int((db > 0).sum()), "B_ties": int((db == 0).sum()),
    }


def summarize_order(counts):
    result = dict(counts)
    n = counts["n_pairs"]
    for name in ("both_correct_order", "A_only_correct_order", "B_only_correct_order",
                 "both_wrong_order", "either_tied"):
        result[name + "_fraction"] = counts[name] / n if n else None
    for renderer in ("A", "B"):
        result[renderer + "_within_context_pair_auroc"] = (
            (counts[renderer + "_wins"] + 0.5 * counts[renderer + "_ties"]) / n if n else None
        )
    non_tied = n - counts["either_tied"]
    result["non_tied_order_retention"] = (
        (counts["both_correct_order"] + counts["both_wrong_order"]) / non_tied
        if non_tied else None
    )
    return result


def nuisance_pairs(rows, indices, factor):
    joined = defaultdict(dict)
    for index in indices:
        row = rows[index]
        state = row["state"]
        value = state[factor]
        check(value in (0, 1), "Invalid nuisance bit")
        key = row["group_id"], row["condition"], row["renderer"], canonical_state(
            row, exclude=(factor,)
        )
        check(value not in joined[key], "Duplicate nuisance counterpart")
        joined[key][value] = index
    result = []
    for pair in joined.values():
        check(set(pair) == {0, 1}, "Missing nuisance counterpart")
        result.append((pair[0], pair[1]))
    return result


def nuisance_metrics(pairs, labels, scores):
    i0 = np.array([a for a, _ in pairs], dtype=np.int64)
    i1 = np.array([b for _, b in pairs], dtype=np.int64)
    check(np.array_equal(labels[i0], labels[i1]), "Nuisance changes target")
    gap = scores[i1] - scores[i0]
    return {
        "n_pairs": len(pairs), "signed_state1_minus_state0_mean": float(gap.mean()),
        "absolute_gap": distribution(abs(gap)),
        "verdict_flip_fraction": float(((scores[i0] > 0) != (scores[i1] > 0)).mean()),
    }


def factor_fold(bits):
    bits = np.asarray(bits)
    i, m, b1, b2 = bits.T
    return np.where(b2 == 1, 1 - m, np.where(b1 == 1, m, i))


def verify_formal_rows(rows):
    check(len(rows) == 4992, "Need all 4,992 test rows/model")
    groups = sorted({r["group_id"] for r in rows})
    check(len(groups) == 24, "Need all 24 contexts")
    expected = {"correction_easy": 32, "correction_hard": 32, "one_step": 32, "asserted": 8}
    for renderer in ("A", "B"):
        for condition in CONDITIONS:
            subset = [r for r in rows if r["renderer"] == renderer and r["condition"] == condition]
            counts = {g: sum(r["group_id"] == g for r in subset) for g in groups}
            check(set(counts.values()) == {expected[condition]}, "Incomplete condition/context")
    for row in rows:
        check(row["partition"] == "test", "Non-test row")
        state = row["state"]
        condition = row["condition"]
        if condition in ("correction_easy", "correction_hard"):
            bits = np.array([[state[field] for field in FACTORS.values()]])
            check(np.isin(bits, (0, 1)).all(), "Invalid factor targets")
            gold = int(factor_fold(bits)[0])
            hard = state["initial"] == 1 - state["update"]
            check(hard == (condition == "correction_hard"), "Wrong correction condition")
        elif condition == "one_step":
            gold = state["update"] if state["receipt"] else state["initial"]
        elif condition == "asserted":
            gold = state["asserted"]
        else:
            raise ValueError("Unexpected condition")
        check(row["gold"] == gold, "Formal gold disagreement")
    return groups


def analyze_head(rows, pairs, labels, scores, condition, groups):
    indices = [i for i, row in enumerate(rows) if row["condition"] == condition]
    result = {"renderers": {}, "paired": {}, "order": {}}
    for renderer in ("A", "B"):
        selected = [i for i in indices if rows[i]["renderer"] == renderer]
        contexts = {g: score_metrics(labels[gi], scores[gi]) for g in groups
                    if (gi := [i for i in selected if rows[i]["group_id"] == g])}
        context_aucs = [v["raw_auroc"] for v in contexts.values() if v["raw_auroc"] is not None]
        result["renderers"][renderer] = {
            "pooled": score_metrics(labels[selected], scores[selected]), "contexts": contexts,
            "context_mean_raw_auroc": float(np.mean(context_aucs)) if context_aucs else None,
            "context_min_raw_auroc": min(context_aucs) if context_aucs else None,
            "context_max_raw_auroc": max(context_aucs) if context_aucs else None,
            "contexts_with_defined_auroc": len(context_aucs),
            "context_mean_accuracy": float(np.mean([v["accuracy_at_zero"] for v in contexts.values()])),
            "context_min_accuracy": min(v["accuracy_at_zero"] for v in contexts.values()),
            "context_max_accuracy": max(v["accuracy_at_zero"] for v in contexts.values()),
            "nuisance": {},
        }
        for nuisance in ("world", "name_map"):
            npairs = nuisance_pairs(rows, selected, nuisance)
            result["renderers"][renderer]["nuisance"][nuisance] = {
                "pooled": nuisance_metrics(npairs, labels, scores),
                "contexts": {
                    g: nuisance_metrics([(a, b) for a, b in npairs if rows[a]["group_id"] == g],
                                        labels, scores) for g in groups
                },
            }
    relevant = [(key, a, b) for key, a, b in pairs if key[1] == condition]
    result["paired"]["contexts"] = {}
    result["order"]["contexts"] = {}
    total_counts = None
    for group in [None, *groups]:
        selected = [(a, b) for key, a, b in relevant if group is None or key[0] == group]
        ai, bi = [a for a, _ in selected], [b for _, b in selected]
        check(np.array_equal(labels[ai], labels[bi]), "A/B head target mismatch")
        measurement = paired_metrics(labels[ai], scores[ai], scores[bi])
        if group is None:
            result["paired"]["pooled"] = measurement
        else:
            result["paired"]["contexts"][group] = measurement
            counts = order_counts(labels[ai], scores[ai], scores[bi])
            result["order"]["contexts"][group] = summarize_order(counts)
            if total_counts is None:
                total_counts = dict.fromkeys(counts, 0)
            for key, value in counts.items():
                total_counts[key] += value
    result["order"]["aggregate_within_context_pairs"] = summarize_order(total_counts)
    return result


def joint_factors(rows, arrays, labels, methods, groups, *, conditions=CONDITIONS[:2]):
    result = {}
    bits = arrays["factor_predictions"]
    c2 = arrays["verdicts"][:, methods.index("C2")]
    check(np.array_equal(factor_fold(bits), c2), "Saved C2 verdict differs from frozen fold")
    for condition in conditions:
        result[condition] = {}
        for renderer in ("A", "B"):
            target = result[condition][renderer] = {"contexts": {}}
            for group in [None, *groups]:
                ix = [i for i, r in enumerate(rows) if r["condition"] == condition
                      and r["renderer"] == renderer and (group is None or r["group_id"] == group)]
                truth = np.array([[rows[i]["state"][f] for f in FACTORS.values()] for i in ix])
                values = {
                    "n": len(ix), "all_four_correct": float((bits[ix] == truth).all(axis=1).mean()),
                    "C2_fold_accuracy": float((c2[ix] == labels[ix]).mean()),
                }
                if group is None:
                    target["pooled"] = values
                else:
                    target["contexts"][group] = values
    return result


def load_model(root, model, input_hashes):
    analysis = root / (model + "_analysis")
    manifest_path = analysis / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    check(manifest["schema"] == "fresh-readout-analysis-manifest-v2", "Wrong completed manifest")
    prediction_path = analysis / "predictions.npz"
    rows_path = root / model / "test" / "rows.json"
    for path in (manifest_path, prediction_path, rows_path):
        input_hashes[str(path)] = digest(path)
    check(input_hashes[str(prediction_path)] == manifest["output_sha256"]["predictions.npz"],
          "Prediction bytes differ from completed manifest")
    expected_rows_hash = [v for k, v in manifest["source_sha256"].items()
                          if k.endswith("/" + model + "/test/rows.json")]
    check(expected_rows_hash == [input_hashes[str(rows_path)]], "Rows differ from completed manifest")
    rows_payload = json.loads(rows_path.read_text(encoding="utf-8"))
    check(not rows_payload["test_fixture"], "Fixture data forbidden")
    with np.load(prediction_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    methods = arrays["methods"].tolist()
    check(len(methods) == len(set(methods)), "Duplicate method names")
    check(set(METHODS + ("C2",)).issubset(methods), "Missing frozen methods")
    rows = align_rows(rows_payload["rows"], arrays["row_ids"].tolist())
    groups = verify_formal_rows(rows)
    check(arrays["margins"].shape == arrays["verdicts"].shape == (4992, len(methods)),
          "Wrong verdict array dimensions")
    check(arrays["factor_margins"].shape == arrays["factor_predictions"].shape == (4992, 4),
          "Wrong factor dimensions")
    check(np.isfinite(arrays["margins"]).all() and np.isfinite(arrays["factor_margins"]).all(),
          "Nonfinite saved score")
    for name in METHODS:
        column = methods.index(name)
        check(np.array_equal(arrays["margins"][:, column] > 0, arrays["verdicts"][:, column]),
              "Frozen threshold discrepancy: " + name)
    check(np.array_equal(arrays["factor_margins"] > 0, arrays["factor_predictions"]),
          "Frozen factor threshold discrepancy")
    return rows, arrays, methods, groups


def write_csv(path, records):
    check(bool(records), "Empty CSV output")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def flat_tables(results):
    scores, pairs, orders, nuisances = [], [], [], []
    for model, analysis in results.items():
        for condition, heads in analysis["conditions"].items():
            for head, data in heads.items():
                base = {"model": model, "condition": condition, "head": head}
                for renderer, values in data["renderers"].items():
                    for context, m in [("ALL", values["pooled"]), *values["contexts"].items()]:
                        row = {**base, "renderer": renderer, "context": context,
                               **{k: v for k, v in m.items() if k != "classes"}}
                        for label, cl in m["classes"].items():
                            row[f"class{label}_n"] = cl["n"]
                            row[f"class{label}_recall_at_zero"] = cl["recall_at_zero"]
                            for key in ("mean", "sd_population", "min", "q25", "median", "q75", "max",
                                        "fraction_gt_zero"):
                                row[f"class{label}_margin_{key}"] = (
                                    cl["margin"][key] if cl["margin"] else None
                                )
                        scores.append(row)
                    for nuisance, ndata in values["nuisance"].items():
                        for context, m in [("ALL", ndata["pooled"]), *ndata["contexts"].items()]:
                            nuisances.append({**base, "renderer": renderer, "nuisance": nuisance,
                                              "context": context,
                                              **{k: v for k, v in m.items() if k != "absolute_gap"},
                                              **{"absolute_gap_" + k: v
                                                 for k, v in m["absolute_gap"].items()}})
                for context, m in [("ALL", data["paired"]["pooled"]),
                                   *data["paired"]["contexts"].items()]:
                    pairs.append({**base, "context": context,
                                  **{k: v for k, v in m.items()
                                     if k not in ("B_minus_A", "absolute_B_minus_A", "classes")},
                                  **{"B_minus_A_" + k: v for k, v in m["B_minus_A"].items()},
                                  **{"class" + label + "_B_minus_A_" + k: v
                                     for label, distribution_ in m["classes"].items()
                                     for k, v in distribution_.items()}})
                for context, m in [("ALL_WITHIN_CONTEXT", data["order"]["aggregate_within_context_pairs"]),
                                   *data["order"]["contexts"].items()]:
                    orders.append({**base, "context": context, **m})
    return scores, pairs, orders, nuisances


def receipt(summary, output):
    pooled = summary["analysis_scope"] == "correction_all_addendum"
    scope = (
        "All 3,072 correction rows/model (easy+hard), both renderers, all 24 paired contexts; "
        if pooled else "All 4,992 saved rows/model, both renderers, all 24 paired contexts; "
    )
    lines = [
        "# POSTHOC saved-score format diagnostic", "",
        "26 November 2026. " + scope +
        "frozen orientation and margin > 0. No model calls, fitting, threshold search, p-values, "
        "or new confidence intervals. The exposed bank cannot support confirmatory claims.", "",
        "Raw AUROC measures target ordering; accuracy uses the unchanged zero boundary. "
        "WC AUROC compares positive/negative rows within the same context. Order retention "
        "counts agreement of A/B positive-negative order including jointly wrong orders, "
        "excluding ties. Correlation alone cannot establish useful ordering.", "",
        "| Model | Condition | Head | Accuracy A / B | Raw AUROC A / B | WC AUROC A / B | "
        "A/B Spearman | Order retention |", "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    def fmt(value):
        return "undefined" if value is None else f"{value:.4f}"
    for model, result in summary["results"].items():
        for condition, heads in result["conditions"].items():
            for head, data in heads.items():
                a, b = [data["renderers"][r]["pooled"] for r in ("A", "B")]
                order = data["order"]["aggregate_within_context_pairs"]
                lines.append(f"| {model} | {condition} | {head} | "
                             f"{fmt(a['accuracy_at_zero'])} / {fmt(b['accuracy_at_zero'])} | "
                             f"{fmt(a['raw_auroc'])} / {fmt(b['raw_auroc'])} | "
                             f"{fmt(order['A_within_context_pair_auroc'])} / "
                             f"{fmt(order['B_within_context_pair_auroc'])} | "
                             f"{fmt(data['paired']['pooled']['spearman'])} | "
                             f"{fmt(order['non_tied_order_retention'])} |")
    lines += ["", "All-context details and class-conditional margin distributions are in "
              "[summary.json](./summary.json), [scores.csv](./scores.csv), "
              "[A/B pair shifts](./paired_scores.csv), [within-context order transitions](./order_transitions.csv), "
              "and [nuisance margin pairs](./nuisance_pairs.csv). The "
              "[pairing inventory](./pairing_inventory.csv) retains formal state, exact row IDs and prompt hashes. "
              "The [manifest](./manifest.json) binds inputs, code, specification, and output bytes.", "",
              "Joint all-four ingredient correctness and C2 folded row accuracy are in the JSON "
              "for every analyzed correction population, both renderers and all contexts. "
              "Within hard rows I=1-M1 and within easy rows I=M1; each stratum's I/M1 results "
              "alone do not establish separate accessibility. Their full easy+hard union varies "
              "both independently. Factor diagnostics are undefined for one_step/asserted.", "",
              "Pylance, configurePythonEnvironment and getPythonExecutableCommand were not available "
              "in the tool catalog; the supplied .venv runtime was verified directly. "
              "The separately sealed pooling addendum was written after v1 stratum outcomes "
              "were seen and before this pooled population was computed; its formal-algebra "
              "rationale came from independent design review. Original v1 outputs remain unchanged."
              if pooled else "No analysis-specification deviation was required."]
    (output / "receipt.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(root, output, specification, *, pooled_corrections=False):
    root, output, specification = Path(root), Path(output), Path(specification)
    allowed = {"specification_seal.json"}
    if specification.parent.resolve() == output.resolve():
        allowed.add(specification.name)
    check(output.is_dir() and {p.name for p in output.iterdir()} == allowed,
          "Output must contain only a prior specification seal; refuse overwrite")
    seal_path = output / "specification_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8-sig"))
    check(seal["specification_sha256"] == digest(specification), "Specification changed after seal")
    if pooled_corrections:
        check(seal.get("new_scope_computed_before_seal") is False
              and seal.get("initial_stratum_outcomes_seen") is True,
              "Missing transparently timed pooling-addendum seal")
    else:
        check(seal["raw_margin_arrays_parsed_before_seal"] is False,
              "Missing pre-array specification seal")
    started = datetime.now(UTC).isoformat()
    input_hashes = {str(specification): digest(specification), str(seal_path): digest(seal_path)}
    results, inventory, reference = {}, [], None
    for model in ("llama", "qwen"):
        rows, arrays, methods, groups = load_model(root, model, input_hashes)
        if pooled_corrections:
            rows = [{**r, "original_condition": r["condition"], "condition": "correction_all"}
                    if r["condition"] in CONDITIONS[:2] else r for r in rows]
        pairs = ab_pairs(rows)
        if pooled_corrections:
            pairs = [(key, a, b) for key, a, b in pairs if key[1] == "correction_all"]
            check(len(pairs) == 1536, "Need every pooled correction A/B pair")
        identities = {(r["row_id"], formal_key(r), r["gold"], r["messages_sha256"]) for r in rows}
        if reference is not None:
            check(identities == reference, "Cross-model formal or prompt-message inventory differs")
        reference = identities
        for key, a, b in pairs:
            item = {"model": model, "group_id": key[0], "condition": key[1], "state": key[2],
                    "A_row_id": rows[a]["row_id"], "B_row_id": rows[b]["row_id"], "gold": rows[a]["gold"]}
            for renderer, index in (("A", a), ("B", b)):
                item.update({renderer + "_" + field: rows[index].get(field) for field in PROMPT_HASHES})
            inventory.append(item)
        labels = np.array([r["gold"] for r in rows])
        model_result = {"rows": 3072 if pooled_corrections else len(rows), "loaded_rows": len(rows),
                        "contexts": groups, "A_B_pairs": len(pairs), "conditions": {}}
        conditions = ("correction_all",) if pooled_corrections else CONDITIONS
        for condition in conditions:
            heads = model_result["conditions"][condition] = {}
            for head in METHODS:
                scores = arrays["margins"][:, methods.index(head)]
                heads[head] = analyze_head(rows, pairs, labels, scores, condition, groups)
            if condition.startswith("correction_"):
                for column, (head, field) in enumerate(FACTORS.items()):
                    targets = np.array([r["state"].get(field, -1) for r in rows])
                    heads[head] = analyze_head(rows, pairs, targets,
                                               arrays["factor_margins"][:, column], condition, groups)
        model_result["joint_factors"] = joint_factors(
            rows, arrays, labels, methods, groups,
            conditions=("correction_all",) if pooled_corrections else CONDITIONS[:2],
        )
        results[model] = model_result
    summary = {
        "schema": "posthoc-format-score-shift-v1", "started_at_utc": started,
        "completed_at_utc": datetime.now(UTC).isoformat(), "threshold": "margin > 0; zero predicts 0",
        "orientation": "positive_is_formal_target_1; raw AUROC, never absolute/inverted",
        "analysis_scope": "correction_all_addendum" if pooled_corrections else "original_v1_strata",
        "status": "POSTHOC descriptive; exposed test; no fitting or model calls",
        "specification_sha256": digest(specification), "results": results,
    }
    write_json(output / "summary.json", summary)
    for name, table in zip(("scores.csv", "paired_scores.csv", "order_transitions.csv", "nuisance_pairs.csv"),
                           flat_tables(results), strict=True):
        write_csv(output / name, table)
    write_csv(output / "pairing_inventory.csv", inventory)
    receipt(summary, output)
    manifest = {
        "schema": "posthoc-format-score-shift-manifest-v1", "completed_at_utc": datetime.now(UTC).isoformat(),
        "input_sha256": input_hashes, "source_sha256": digest(__file__),
        "interpreter": sys.executable, "python_version": platform.python_version(),
        "numpy_version": np.__version__, "specification_sealed_at_utc": seal["sealed_at_utc"],
        "output_sha256": {p.name: digest(p) for p in output.iterdir() if p.name != "manifest.json"},
        "integrity_checks": "Completed prediction/row manifest hashes; exact ID joins; full formal A/B "
        "states; prompt identities; 24 complete contexts; cross-model inventories; formal gold and frozen thresholds/fold",
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "source_sha256": manifest["source_sha256"],
                      "models": {m: {"rows": r["rows"], "contexts": len(r["contexts"]),
                                       "A_B_pairs": r["A_B_pairs"]} for m, r in results.items()}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", required=True)
    parser.add_argument("--output", required=True, help="Unique folder containing only pre-array spec seal")
    parser.add_argument("--specification", required=True)
    parser.add_argument("--pooled-corrections", action="store_true",
                        help="Separately sealed posthoc full easy+hard correction supplement")
    args = parser.parse_args()
    run(args.native_root, args.output, args.specification, pooled_corrections=args.pooled_corrections)


if __name__ == "__main__":
    main()
