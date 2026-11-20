"""Two-claim relative scores and prespecified linear contrast readouts.

One observation is a physical case/view/condition with candidate B as endpoint.
Candidate A is its logical complement, not an independent success. The old
normalized head is retained exactly; its intercept cancels in a difference,
but raw contextual offsets need not cancel through its per-row normalization.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
from scipy.sparse import vstack
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize

from . import belief_query_readout as query
from .belief_readout import binary_metrics
from .storage import (
    atomic_json,
    atomic_npz,
    canonical_hash,
    environment_info,
    file_sha256,
    read_json,
    utc_now,
)

CONDITIONS = ("story", "nested_question")
VIEWS = ("omniscient", "sender")
TARGETS = ("world", "reader", "nested")
METHODS = (
    "original_single",
    "original_relative",
    "new_single",
    "paired_linear",
    "tfidf_single",
    "tfidf_paired",
)
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
LAYER = "layer_16"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _validate(rows, features, *, fresh=False):
    _require(
        isinstance(rows, list) and rows and set(features) == set(query.LAYERS),
        "Missing rows or archived layers",
    )
    arrays = {}
    for layer, value in features.items():
        value = np.asarray(value)
        _require(
            value.ndim == 2
            and value.shape[0] == len(rows)
            and value.shape[1] > 0
            and value.dtype.kind == "f"
            and np.isfinite(value).all(),
            "Feature alignment/finiteness error",
        )
        arrays[layer] = value.astype(np.float64)
    by_pair, ids, cases, template_splits = {}, set(), {}, {}
    for i, row in enumerate(rows):
        _require(
            all(
                k in row
                for k in (
                    "example_id",
                    "case_id",
                    "template_id",
                    "split",
                    "channel",
                    "phase",
                    "view",
                    "condition",
                    "claim_bit",
                    "entity_ids",
                    "messages_text",
                    "labels",
                )
            ),
            "Missing contrast row fields",
        )
        _require(
            row["view"] in VIEWS
            and row["condition"] in CONDITIONS
            and type(row["claim_bit"]) is int
            and row["claim_bit"] in (0, 1),
            "Invalid view/condition/claim",
        )
        expected = "__".join((row["case_id"], row["view"], row["condition"], str(row["claim_bit"])))
        _require(
            row["example_id"] == expected and expected not in ids,
            "Duplicate or mismatched example identity",
        )
        ids.add(expected)
        _require(
            isinstance(row["messages_text"], str) and row["messages_text"].strip(),
            "Missing message",
        )
        _require(
            "candidate_text" not in row
            or (isinstance(row["candidate_text"], str) and row["candidate_text"].strip()),
            "Candidate text must be a nonempty string when supplied",
        )
        _require(
            isinstance(row["entity_ids"], list)
            and row["entity_ids"]
            and all(isinstance(e, str) and e for e in row["entity_ids"]),
            "Missing entity IDs",
        )
        _require(
            set(row["labels"]) == set(TARGETS)
            and all(type(v) is int and v in (0, 1) for v in row["labels"].values()),
            "Binary per-claim labels required",
        )
        _require(
            all(isinstance(row[k], str) and row[k] for k in ("case_id", "template_id", "channel")),
            "Missing group identities",
        )
        if fresh:
            _require(
                row["split"] == "fresh" and row["phase"] in {"prior", "post_message"},
                "Fresh scoring accepts prior or post_message fresh rows only",
            )
        elif row["split"] == "control":
            _require(
                row["phase"] == "delivered_current" and len(set(row["labels"].values())) == 1,
                "Malformed delivered-current control",
            )
        else:
            _require(
                row["split"] in {"train", "validation", "test", "development"}
                and (
                    row["phase"] == "prior"
                    or (row["split"] == "development" and row["phase"] == "post_message")
                ),
                "Malformed fitting/development split",
            )
        metadata = {k: row[k] for k in ("template_id", "split", "channel", "phase", "entity_ids")}
        _require(
            row["case_id"] not in cases or cases[row["case_id"]] == metadata,
            "Case metadata differs across claims/views/conditions",
        )
        cases[row["case_id"]] = metadata
        if row["split"] != "control":
            group = (row["channel"], row["template_id"])
            _require(
                group not in template_splits or template_splits[group] == row["split"],
                "Template leakage across splits",
            )
            template_splits[group] = row["split"]
        key = (row["case_id"], row["view"], row["condition"])
        by_pair.setdefault(key, {})[row["claim_bit"]] = i
    _require(
        all(set(pair) == {0, 1} for pair in by_pair.values()),
        "Each case/view/condition requires both claims",
    )
    pairs = []
    for key, pair in by_pair.items():
        a, b = pair[0], pair[1]
        _require(
            all(rows[a]["labels"][t] == 1 - rows[b]["labels"][t] for t in TARGETS),
            "Claim labels are not complements",
        )
        pairs.append(
            {
                "a": a,
                "b": b,
                **{
                    k: rows[b][k]
                    for k in (
                        "case_id",
                        "template_id",
                        "split",
                        "channel",
                        "phase",
                        "view",
                        "condition",
                    )
                },
                "labels": rows[b]["labels"],
            }
        )
    for case in cases:
        _require(
            {(p["view"], p["condition"]) for p in pairs if p["case_id"] == case}
            == {(v, c) for v in VIEWS for c in CONDITIONS},
            "Each case requires both views/conditions",
        )
    required = {"fresh"} if fresh else {"train", "validation", "test"}
    _require(required <= {p["split"] for p in pairs}, "Missing required partitions")
    hidden = {}
    for pair in pairs:
        if pair["view"] == "sender" and pair["phase"] in {"prior", "post_message"}:
            label = pair["labels"]
            key = (
                pair["channel"],
                pair["template_id"],
                pair["condition"],
                label["world"],
                label["nested"],
            )
            hidden.setdefault(key, []).append(pair)
    for group in hidden.values():
        _require(
            len(group) == 2 and {p["labels"]["reader"] for p in group} == {0, 1},
            "Need balanced private-reader counterfactual pairs",
        )
        for claim in ("a", "b"):
            a, b = group[0][claim], group[1][claim]
            _require(
                rows[a]["messages_text"] == rows[b]["messages_text"]
                and all(np.array_equal(arrays[layer][a], arrays[layer][b]) for layer in arrays),
                "Sender input exposes private reader label",
            )
    return arrays, pairs


def _select(X_train, y_train, X_val, y_val, *, intercept):
    candidates, models = [], []
    _require(np.array_equal(np.unique(y_train), [0, 1]), "Both classes required")
    for c in C_GRID:
        model = LogisticRegression(
            C=c, fit_intercept=intercept, solver="lbfgs", max_iter=2000, random_state=20261120
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_train, y_train)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        loss = binary_metrics(y_val, model.predict_proba(X_val))["binary_log_loss"]
        candidates.append({"C": c, "validation_NLL": loss, "converged": converged})
        models.append(model)
    eligible = [i for i, item in enumerate(candidates) if item["converged"]]
    _require(eligible, "No converged C candidate")
    i = min(eligible, key=lambda j: (candidates[j]["validation_NLL"], candidates[j]["C"]))
    return models[i], {"selected_C": candidates[i]["C"], "candidates": candidates}


def _metrics(y, probability):
    result = binary_metrics(y, probability)
    y, score = np.asarray(y), probability[:, 1]
    positive, negative = score[y == 1], score[y == 0]
    result["AUROC_fixed_orientation"] = (
        float(np.mean((positive[:, None] > negative) + 0.5 * (positive[:, None] == negative)))
        if len(positive) and len(negative)
        else None
    )
    return result


def _text_matrices(transform, texts_a, texts_b):
    vectorizer = query._vectorizer({term: i for i, term in enumerate(transform["terms"])})
    vectorizer.idf_ = np.asarray(transform["idf"], dtype=np.float64)
    return vectorizer.transform(texts_a), vectorizer.transform(texts_b)


def _predict(head, params, texts, XA, XB, messages_a, messages_b):
    method, key = head["method"], head["parameter_prefix"]
    if method.startswith("tfidf"):
        A, B = _text_matrices(texts[head["text_transform"]], messages_a, messages_b)
        norm_a = np.sqrt(np.asarray(A.multiply(A).sum(axis=1))).ravel()
        norm_b = np.sqrt(np.asarray(B.multiply(B).sum(axis=1))).ravel()
        difference = B - A
        norm_d = np.sqrt(np.asarray(difference.multiply(difference).sum(axis=1))).ravel()
    else:
        transform = head["transform"]
        A = (XA - params[transform + "__mean"]) / params[transform + "__scale"]
        B = (XB - params[transform + "__mean"]) / params[transform + "__scale"]
        norm_a, norm_b = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
        difference = B - A
        norm_d = np.linalg.norm(difference, axis=1)
        if method in {"original_single", "original_relative", "new_single"}:
            A, B = normalize(A, norm="l2"), normalize(B, norm="l2")
    coef, bias = params[key + "__coef"], params[key + "__intercept"][0]
    if method in {"paired_linear", "tfidf_paired"}:
        _require(bias == 0, "Paired model must have zero intercept")
        margin = np.asarray(difference @ coef.T).ravel()
        a, b = -margin, margin
    else:
        a, b = np.asarray(A @ coef.T).ravel() + bias, np.asarray(B @ coef.T).ravel() + bias
        margin = b - a if method == "original_relative" else b
    _require(np.isfinite(margin).all(), "Nonfinite contrast margins")
    p = expit(margin)
    return np.column_stack([1 - p, p]), (margin, a, b, norm_a, norm_b, norm_d)


def _group_metrics(pairs, probability, target, rows):
    result = {}
    for channel in sorted({p["channel"] for p in pairs}):
        result[channel] = {}
        for split in sorted({p["split"] for p in pairs if p["channel"] == channel}):
            indices = np.asarray(
                [i for i, p in enumerate(pairs) if p["channel"] == channel and p["split"] == split],
                dtype=np.int64,
            )
            labels = np.asarray([pairs[i]["labels"][target] for i in indices])
            metric = _metrics(labels, probability[indices])
            metric["n_templates"] = len({pairs[i]["template_id"] for i in indices})
            metric["n_unique_candidate_B_messages"] = len(
                {rows[pairs[i]["b"]]["messages_text"] for i in indices}
            )
            metric["by_template"] = {}
            for template in sorted({pairs[i]["template_id"] for i in indices}):
                sub = np.asarray([i for i in indices if pairs[i]["template_id"] == template])
                metric["by_template"][template] = _metrics(
                    [pairs[i]["labels"][target] for i in sub], probability[sub]
                )
            metric["strata"] = {}
            for a, b in (("world", "nested"), ("reader", "nested"), ("world", "reader")):
                for differs in (False, True):
                    sub = np.asarray(
                        [
                            i
                            for i in indices
                            if (pairs[i]["labels"][a] != pairs[i]["labels"][b]) == differs
                        ],
                        dtype=np.int64,
                    )
                    name = a + ("_NE_" if differs else "_EQ_") + b
                    metric["strata"][name] = (
                        _metrics([pairs[i]["labels"][target] for i in sub], probability[sub])
                        if len(sub)
                        else {"n_rows": 0}
                    )
            if (
                target == "reader"
                and pairs[indices[0]]["view"] == "sender"
                and all(pairs[i]["phase"] in {"prior", "post_message"} for i in indices)
            ):
                _require(metric["accuracy"] == 0.5, "Private reader negative control must be 50%")
            world = np.asarray([pairs[i]["labels"]["world"] for i in indices])
            metric["oracle_world_copy_accuracy"] = float(np.mean(world == labels))
            result[channel][split] = metric
    return result


def _candidate_checks(rows):
    groups = {}
    for row in rows:
        literal = row.get(
            "candidate_text", "metadata_proxy:" + row["template_id"] + ":" + str(row["claim_bit"])
        )
        key = (row["channel"], row["split"], row["phase"], row["view"], row["condition"], literal)
        groups.setdefault(key, []).append(row)
    checks = []
    for key, group in groups.items():
        counts = {
            t: {str(label): sum(r["labels"][t] == label for r in group) for label in (0, 1)}
            for t in TARGETS
        }
        checks.append(
            {
                "channel": key[0],
                "split": key[1],
                "phase": key[2],
                "view": key[3],
                "condition": key[4],
                "literal": key[5],
                "n_rows": len(group),
                "target_class_counts": counts,
                "candidate_only_best_deterministic_accuracy_bound": {
                    t: max(counts[t].values()) / len(group) for t in TARGETS
                },
            }
        )
    return {
        "literal_text_supplied_for_all_rows": all("candidate_text" in row for row in rows),
        "checks": checks,
        "interpretation": "Structural label-balance control; not an additionally trained head. World-copy metrics are authored-metadata oracle diagnostics.",
    }


def _score(rows, X, pairs, heads, params, texts):
    equal, cross, records = {}, [], []
    for head_id, head in heads.items():
        destinations = [head["condition"]] + (
            ["story"] if head["condition"] == "nested_question" else []
        )
        for condition in destinations:
            selected = [
                p for p in pairs if p["view"] == head["view"] and p["condition"] == condition
            ]
            a, b = np.asarray([p["a"] for p in selected]), np.asarray([p["b"] for p in selected])
            probability, details = _predict(
                head,
                params,
                texts,
                X[a],
                X[b],
                [rows[i]["messages_text"] for i in a],
                [rows[i]["messages_text"] for i in b],
            )
            metrics = _group_metrics(selected, probability, head["target"], rows)
            item = {"head_id": head_id, "selected_C": head.get("selected_C"), "metrics": metrics}
            if condition == head["condition"]:
                equal.setdefault(condition, {}).setdefault(head["view"], {}).setdefault(
                    head["target"], {}
                )[head["method"]] = item
            else:
                cross.append(
                    {
                        "source_condition": head["condition"],
                        "evaluation_condition": condition,
                        "view": head["view"],
                        "target": head["target"],
                        "method": head["method"],
                        **item,
                    }
                )
            for i, pair in enumerate(selected):
                margin, ma, mb, na, nb, nd = (float(value[i]) for value in details)
                records.append(
                    {
                        "head_id": head_id,
                        "source_condition": head["condition"],
                        "evaluation_condition": condition,
                        "view": head["view"],
                        "target": head["target"],
                        "method": head["method"],
                        **{
                            k: pair[k]
                            for k in ("channel", "split", "phase", "case_id", "template_id")
                        },
                        "label_B": pair["labels"][head["target"]],
                        "p_B": float(probability[i, 1]),
                        "prediction_B": int(probability[i].argmax()),
                        "margin": margin,
                        "candidate_margin_A": ma,
                        "candidate_margin_B": mb,
                        "candidate_pre_normalization_norm_A": na,
                        "candidate_pre_normalization_norm_B": nb,
                        "raw_standardized_or_tfidf_difference_norm": nd,
                    }
                )
    return equal, cross, records


def _fresh_path(path):
    path = Path(path).resolve()
    _require(not path.exists(), "Use a fresh output directory")
    return path


def _finish(output, report, records, stage, extra=None):
    atomic_json(output / "report.json", report)
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    atomic_json(
        output / "manifest.json",
        {
            "schema": "belief_contrast_artifacts_v1",
            "stage": stage,
            "status": "complete",
            "created_at": utc_now(),
            "source_code_sha256": file_sha256(__file__),
            "environment": environment_info(),
            "output_sha256": query._snapshot(output),
            **(extra or {}),
        },
    )
    return report


def _settings():
    return {
        "primary_view": "sender",
        "primary_target": "nested",
        "primary_condition": "story",
        "primary_layer": LAYER,
        "primary_endpoints": ["original_relative", "paired_linear"],
        "methods": list(METHODS),
        "C_grid": list(C_GRID),
        "paired_transform": "train-claim StandardScaler; raw zB-zA; NO per-row or difference L2",
        "single_transform": "train-claim StandardScaler; per-row L2",
        "paired_fit": "D,-D with yB,1-yB; no intercept",
        "single_fit": "both claims and their complementary labels; intercept",
        "single_validation": "both claims",
        "paired_validation": "candidate B; complementary A has the same loss",
        "probability": "expit(raw margin), temperature1; calibration not assumed",
        "unit_of_evaluation": "one candidate-B result per physical case/view/condition; A is not an independent success",
        "cross_condition": "nested_question-trained to story only",
        "text_transform": query.TEXT_SETTINGS,
    }


def fit_contrast_readouts(rows, features, outdir, original_fit_dir):
    output, original = _fresh_path(outdir), Path(original_fit_dir).resolve()
    _require(
        not output.is_relative_to(original) and not original.is_relative_to(output),
        "Output overlaps original fits",
    )
    arrays, pairs = _validate(rows, features)
    old_meta, old_params, _old_text, old_hashes = query._load_fit(original)
    _require(
        arrays[LAYER].shape[1] == old_meta["feature_widths"][LAYER],
        "Original feature width mismatch",
    )
    original_rows = {
        (r["case_id"], r["view"], r["condition"]): r for r in read_json(original / "rows.json")
    }
    for pair in pairs:
        if pair["split"] in {"train", "validation", "test", "control"}:
            expected = original_rows.get((pair["case_id"], pair["view"], pair["condition"]))
            _require(
                expected is not None
                and rows[pair["b"]]["messages_text"] == expected["messages_text"]
                and rows[pair["b"]]["labels"] == expected["labels"]
                and pair["split"] == expected["split"],
                "Candidate B differs from original legacy input/labels",
            )
    source_hash = canonical_hash(rows)
    feature_hashes = {layer: query._array_hash(features[layer]) for layer in query.LAYERS}
    X = arrays[LAYER]
    params, texts, heads = {"classes": np.asarray([0, 1])}, {}, {}
    for condition in CONDITIONS:
        for view in VIEWS:
            selected = [p for p in pairs if p["condition"] == condition and p["view"] == view]
            a, b = np.asarray([p["a"] for p in selected]), np.asarray([p["b"] for p in selected])
            train = np.asarray([i for i, p in enumerate(selected) if p["split"] == "train"])
            val = np.asarray([i for i, p in enumerate(selected) if p["split"] == "validation"])
            _require(len(train) and len(val), "Missing fitting/validation pair group")
            transform = "new__" + condition + "__" + view
            scaler = StandardScaler().fit(np.concatenate([X[a[train]], X[b[train]]]))
            params[transform + "__mean"], params[transform + "__scale"] = (
                scaler.mean_.copy(),
                scaler.scale_.copy(),
            )
            ZA, ZB = scaler.transform(X[a]), scaler.transform(X[b])
            SA, SB, D = normalize(ZA), normalize(ZB), ZB - ZA
            text_key = condition + "__" + view
            ta, tb = [rows[i]["messages_text"] for i in a], [rows[i]["messages_text"] for i in b]
            vectorizer = query._vectorizer().fit([ta[i] for i in train] + [tb[i] for i in train])
            texts[text_key] = {
                "terms": vectorizer.get_feature_names_out().tolist(),
                "idf": vectorizer.idf_.tolist(),
            }
            TA, TB = vectorizer.transform(ta), vectorizer.transform(tb)
            TD = TB - TA
            for target in TARGETS:
                y = np.asarray([p["labels"][target] for p in selected])
                prefix = "original__" + condition + "__" + view + "__" + target
                old = old_meta["heads"]["__".join((condition, view, target, LAYER))]
                old_transform = "old__" + condition + "__" + view
                for suffix in ("mean", "scale"):
                    params[old_transform + "__" + suffix] = old_params[
                        old["activation_transform"] + "__" + suffix
                    ].copy()
                for suffix in ("coef", "intercept"):
                    params[prefix + "__" + suffix] = old_params[
                        old["parameter_prefix"] + "__" + suffix
                    ].copy()
                for method in ("original_single", "original_relative"):
                    head_id = "__".join((condition, view, target, method))
                    heads[head_id] = {
                        "condition": condition,
                        "view": view,
                        "target": target,
                        "method": method,
                        "selected_C": old["selected_C"],
                        "transform": old_transform,
                        "parameter_prefix": prefix,
                    }
                for method, AA, BB, DD, sparse in (
                    ("new_single", SA, SB, D, False),
                    ("paired_linear", ZA, ZB, D, False),
                    ("tfidf_single", TA, TB, TD, True),
                    ("tfidf_paired", TA, TB, TD, True),
                ):
                    paired = method in {"paired_linear", "tfidf_paired"}
                    stack = vstack if sparse else np.concatenate
                    xt = stack([DD[train], -DD[train]]) if paired else stack([AA[train], BB[train]])
                    yt = np.r_[y[train], 1 - y[train]] if paired else np.r_[1 - y[train], y[train]]
                    xv = DD[val] if paired else stack([AA[val], BB[val]])
                    yv = y[val] if paired else np.r_[1 - y[val], y[val]]
                    model, selection = _select(xt, yt, xv, yv, intercept=not paired)
                    head_id = "__".join((condition, view, target, method))
                    params[head_id + "__coef"], params[head_id + "__intercept"] = (
                        model.coef_.copy(),
                        model.intercept_.copy(),
                    )
                    head = {
                        "condition": condition,
                        "view": view,
                        "target": target,
                        "method": method,
                        "transform": transform,
                        "text_transform": text_key,
                        "parameter_prefix": head_id,
                        "n_training_physical_cases": len(train),
                        "n_training_templates": len({selected[i]["template_id"] for i in train}),
                        **selection,
                    }
                    probability, _details = _predict(head, params, texts, X[a], X[b], ta, tb)
                    reference = model.predict_proba(DD if paired else BB)
                    _require(
                        np.allclose(probability, reference, rtol=0, atol=1e-12),
                        "Saved contrast scorer differs from estimator",
                    )
                    heads[head_id] = head
    _require(
        canonical_hash(rows) == source_hash
        and all(
            query._array_hash(features[layer]) == feature_hashes[layer] for layer in query.LAYERS
        ),
        "Fitting inputs changed",
    )
    _require(query._snapshot(original) == old_hashes, "Original fits changed")
    equal, cross, records = _score(rows, X, pairs, heads, params, texts)
    metadata = {
        "settings": _settings(),
        "heads": heads,
        "feature_width": X.shape[1],
        "known_case_ids": sorted({r["case_id"] for r in rows}),
        "known_template_ids": sorted({r["template_id"] for r in rows}),
        "known_entity_ids": sorted({e for r in rows for e in r["entity_ids"]}),
        "known_messages": sorted({canonical_hash(r["messages_text"]) for r in rows}),
        "original_fit_hashes": old_hashes,
        "input_rows_sha256": source_hash,
        "input_feature_sha256": feature_hashes,
    }
    report = {
        "schema": "belief_contrast_fit_report_v1",
        "status": "complete",
        "settings": _settings(),
        "equal_condition": equal,
        "cross_condition": cross,
        "candidate_only_control": _candidate_checks(rows),
        "n_prediction_records": len(records),
        "limitations": _limitations(),
    }
    output.mkdir(parents=True)
    atomic_npz(output / "parameters.npz", **params)
    atomic_json(output / "text_transforms.json", texts)
    atomic_json(output / "fit_metadata.json", metadata)
    atomic_json(output / "rows.json", rows)
    return _finish(output, report, records, "fit")


def _limitations():
    return [
        "Two model forwards and candidate identity are required at inference; gold labels are only used for supervision/evaluation.",
        "Paired relative scores do not establish belief mechanisms, spontaneity, causal use or deception.",
        "The original head retains per-row L2: intercept cancels, but shared raw context need not cancel after different row norms.",
        "The new paired head is linear in the raw standardized difference, with no difference L2 and no intercept.",
        "Probabilities are fixed-temperature sigmoid scores; calibration is not assumed.",
        "Candidate A is complementary and never counted as an independent success; template/channel dependence remains.",
        "Capture/checkpoint/tokenization identity and literal-only prompt edits must be established by the calling runner.",
    ]


def evaluate_contrast_readouts(rows, features, fit_dir, outdir):
    fit_dir, output = Path(fit_dir).resolve(), _fresh_path(outdir)
    _require(
        not output.is_relative_to(fit_dir) and not fit_dir.is_relative_to(output),
        "Fresh output overlaps fitted artifacts",
    )
    manifest = read_json(fit_dir / "manifest.json")
    _require(
        manifest.get("schema") == "belief_contrast_artifacts_v1"
        and manifest.get("stage") == "fit"
        and manifest.get("status") == "complete",
        "Need complete contrast fit",
    )
    hashes = query._snapshot(fit_dir)
    _require(
        set(hashes) == set(manifest["output_sha256"]) | {"manifest.json"}
        and all(hashes[k] == v for k, v in manifest["output_sha256"].items()),
        "Fitted artifacts changed",
    )
    metadata, texts = (
        read_json(fit_dir / "fit_metadata.json"),
        read_json(fit_dir / "text_transforms.json"),
    )
    _require(metadata["settings"] == _settings(), "Frozen contrast settings changed")
    with np.load(fit_dir / "parameters.npz", allow_pickle=False) as archive:
        params = {key: archive[key].copy() for key in archive.files}
    _require(
        all(value.dtype.kind in "fi" and np.isfinite(value).all() for value in params.values())
        and np.array_equal(params.get("classes"), [0, 1]),
        "Invalid frozen numeric parameters or classes",
    )
    arrays, pairs = _validate(rows, features, fresh=True)
    _require(arrays[LAYER].shape[1] == metadata["feature_width"], "Fresh feature width changed")
    for field, known in (("case_id", "known_case_ids"), ("template_id", "known_template_ids")):
        _require(
            not ({r[field] for r in rows} & set(metadata[known])),
            "Fresh " + field + " overlaps fitting/development bank",
        )
    _require(
        not ({e for r in rows for e in r["entity_ids"]} & set(metadata["known_entity_ids"])),
        "Fresh entity IDs overlap fitting/development bank",
    )
    _require(
        not ({canonical_hash(r["messages_text"]) for r in rows} & set(metadata["known_messages"])),
        "Fresh message was already observed",
    )
    row_hash = canonical_hash(rows)
    feature_hashes = {layer: query._array_hash(features[layer]) for layer in query.LAYERS}
    equal, cross, records = _score(rows, arrays[LAYER], pairs, metadata["heads"], params, texts)
    _require(
        query._snapshot(fit_dir) == hashes
        and canonical_hash(rows) == row_hash
        and all(
            query._array_hash(features[layer]) == feature_hashes[layer] for layer in query.LAYERS
        ),
        "Frozen artifacts or fresh inputs changed during scoring",
    )
    report = {
        "schema": "belief_contrast_fresh_report_v1",
        "status": "complete",
        "no_refit": True,
        "settings": _settings(),
        "equal_condition": equal,
        "cross_condition": cross,
        "candidate_only_control": _candidate_checks(rows),
        "fitted_manifest_sha256": hashes["manifest.json"],
        "fitted_artifacts_unchanged": True,
        "input_rows_sha256": row_hash,
        "input_feature_sha256": feature_hashes,
        "limitations": _limitations(),
    }
    output.mkdir(parents=True)
    atomic_json(output / "rows.json", rows)
    return _finish(output, report, records, "fresh", {"fitted_input_sha256": hashes})
