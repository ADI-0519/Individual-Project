"""Grouped query-condition comparisons and unchanged fresh-bank inference.

Supervised labels are never appended to message text. The primary comparison is
separately fitted layer-16 nested heads with matched validation selection. Fixed
cross-condition transfers measure coordinate stability as a different question.
"""

from __future__ import annotations

import csv
import hashlib
import json
import warnings
from itertools import product
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize

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

CONDITIONS = ("story", "nested_question", "world_question")
VIEWS = ("omniscient", "sender")
TARGETS = ("world", "reader", "nested")
LAYERS = ("layer_8", "layer_16", "layer_24")
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
CROSS_PAIRS = (
    ("story", "nested_question"),
    ("story", "world_question"),
    ("nested_question", "story"),
    ("world_question", "story"),
)
TEXT_SETTINGS = {
    "ngram_range": [1, 2],
    "max_features": 20000,
    "token_pattern": r"(?u)\b\w+\b",
    "sublinear_tf": True,
    "norm": "l2",
    "lowercase": True,
    "smooth_idf": True,
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(value.dtype).encode() + json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _snapshot(root):
    return {
        p.relative_to(root).as_posix(): file_sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _validate(rows, features, *, fresh=False):
    _require(isinstance(rows, list) and bool(rows), "Need query rows")
    _require(set(features) == set(LAYERS), "Exactly three prespecified feature layers required")
    arrays = {}
    for layer, value in features.items():
        value = np.asarray(value)
        _require(
            value.ndim == 2
            and value.shape[0] == len(rows)
            and value.shape[1] > 0
            and value.dtype.kind == "f"
            and np.isfinite(value).all(),
            "Feature shape/alignment/finiteness error",
        )
        arrays[layer] = value.astype(np.float64)
    identities, condition_sets, case_metadata, template_splits, entities, cells = (
        set(),
        {},
        {},
        {},
        {},
        {},
    )
    for i, row in enumerate(rows):
        _require(
            all(
                k in row
                for k in (
                    "example_id",
                    "case_id",
                    "template_id",
                    "split",
                    "phase",
                    "view",
                    "condition",
                    "messages_text",
                    "labels",
                    "entity_ids",
                )
            ),
            "Missing required row metadata/entity IDs",
        )
        _require(
            row["condition"] in CONDITIONS and row["view"] in VIEWS, "Unknown view or condition"
        )
        _require(
            isinstance(row["messages_text"], str) and bool(row["messages_text"].strip()),
            "Missing message text",
        )
        _require(
            all(
                isinstance(row[k], str) and row[k] for k in ("example_id", "case_id", "template_id")
            ),
            "Missing row identity",
        )
        _require(
            row["example_id"] == row["case_id"] + "__" + row["view"],
            "Example/case/view identity mismatch",
        )
        _require(
            isinstance(row["entity_ids"], list)
            and row["entity_ids"]
            and all(isinstance(e, str) and e for e in row["entity_ids"])
            and len(set(row["entity_ids"])) == len(row["entity_ids"]),
            "Entity IDs must be a nonempty unique string list",
        )
        _require(
            set(row["labels"]) == set(TARGETS)
            and all(type(value) is int and value in (0, 1) for value in row["labels"].values()),
            "Labels must be world/reader/nested binary integers",
        )
        identity = (row["example_id"], row["condition"])
        _require(identity not in identities, "Duplicate example/condition")
        identities.add(identity)
        condition_sets.setdefault(row["example_id"], set()).add(row["condition"])
        meta = {
            k: row[k] for k in ("case_id", "template_id", "split", "phase", "labels", "entity_ids")
        }
        _require(
            row["case_id"] not in case_metadata or case_metadata[row["case_id"]] == meta,
            "Case metadata changes across conditions/views",
        )
        case_metadata[row["case_id"]] = meta
        template = row["template_id"]
        entity_tuple = tuple(sorted(row["entity_ids"]))
        _require(
            template not in entities or entities[template] == entity_tuple,
            "Template entity identities changed",
        )
        entities[template] = entity_tuple
        if fresh:
            _require(
                row["split"] == "fresh" and row["phase"] == "prior",
                "Fresh evaluator accepts prior/fresh rows only",
            )
        elif row["split"] == "control":
            _require(
                row["phase"] == "delivered_current" and len(set(row["labels"].values())) == 1,
                "Malformed delivered-current control",
            )
            continue
        else:
            _require(
                row["split"] in {"train", "validation", "test"} and row["phase"] == "prior",
                "Malformed prior split/phase",
            )
        _require(
            template not in template_splits or template_splits[template] == row["split"],
            "Template leakage across splits",
        )
        template_splits[template] = row["split"]
        cell = (template, row["condition"], row["view"], *(row["labels"][t] for t in TARGETS))
        _require(cell not in cells, "Duplicate factorial cell")
        cells[cell] = i
    _require(
        all(values == set(CONDITIONS) for values in condition_sets.values()),
        "Each case/view requires all three conditions",
    )
    _require(
        all(
            {r["view"] for r in rows if r["case_id"] == case} == set(VIEWS)
            for case in case_metadata
        ),
        "Each case requires both views",
    )
    _require(
        set(template_splits.values()) == ({"fresh"} if fresh else {"train", "validation", "test"}),
        "Missing split",
    )
    factorial = set(product((0, 1), repeat=3))
    pairs = 0
    for template in template_splits:
        for condition in CONDITIONS:
            for view in VIEWS:
                observed = {key[3:] for key in cells if key[:3] == (template, condition, view)}
                _require(
                    observed == factorial,
                    "Every template/condition/view needs the full balanced factorial",
                )
            for world, nested in product((0, 1), repeat=2):
                a, b = (
                    cells[(template, condition, "sender", world, reader, nested)]
                    for reader in (0, 1)
                )
                _require(
                    rows[a]["messages_text"].encode() == rows[b]["messages_text"].encode(),
                    "Sender text exposes hidden actual-reader state",
                )
                _require(
                    all(np.array_equal(arrays[layer][a], arrays[layer][b]) for layer in LAYERS),
                    "Identical sender inputs have different feature rows",
                )
                pairs += 1
    return arrays, {
        "template_partitions": {
            split: sorted(t for t, s in template_splits.items() if s == split)
            for split in sorted(set(template_splits.values()))
        },
        "n_cases": len(case_metadata),
        "n_rows": len(rows),
        "hidden_reader_pairs": pairs,
        "independence_unit": "template; conditions, views and factorial cells are repeated measurements",
    }


def _vectorizer(vocabulary=None):
    return TfidfVectorizer(
        vocabulary=vocabulary,
        ngram_range=(1, 2),
        max_features=20000,
        token_pattern=TEXT_SETTINGS["token_pattern"],
        sublinear_tf=True,
        norm="l2",
        lowercase=True,
        smooth_idf=True,
        dtype=np.float64,
    )


def _select_model(X_train, y_train, X_validation, y_validation):
    """Test and control data are absent from this selection API."""
    _require(np.array_equal(np.unique(y_train), [0, 1]), "Both training classes required")
    candidates, models = [], []
    for c in C_GRID:
        model = LogisticRegression(C=c, solver="lbfgs", max_iter=2000, random_state=20261120)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_train, y_train)
        _require(np.array_equal(model.classes_, [0, 1]), "Class order changed")
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        loss = binary_metrics(y_validation, model.predict_proba(X_validation))["binary_log_loss"]
        candidates.append({"C": c, "validation_binary_log_loss": loss, "converged": converged})
        models.append(model)
    eligible = [i for i, item in enumerate(candidates) if item["converged"]]
    _require(eligible, "No converged C candidate")
    selected = min(
        eligible, key=lambda i: (candidates[i]["validation_binary_log_loss"], candidates[i]["C"])
    )
    return models[selected], {
        "selected_C": candidates[selected]["C"],
        "validation_selection_loss": candidates[selected]["validation_binary_log_loss"],
        "candidates": candidates,
    }


def _subsets(rows, condition, view):
    return np.asarray(
        [i for i, row in enumerate(rows) if row["condition"] == condition and row["view"] == view],
        dtype=np.int64,
    )


def _row_normalize(values):
    norms = np.sqrt(np.einsum("ij,ij->i", values, values))
    norms[norms < 10 * np.finfo(np.float64).eps] = 1
    return values / norms[:, None]


def _predict(head, parameters, text_transforms, X, texts):
    if head["layer"] == "text":
        transform = text_transforms[head["text_transform"]]
        vectorizer = _vectorizer({word: i for i, word in enumerate(transform["terms"])})
        vectorizer.idf_ = np.asarray(transform["idf"], dtype=np.float64)
        values = vectorizer.transform(texts)
    else:
        prefix = head["activation_transform"]
        values = _row_normalize(
            (X - parameters[prefix + "__mean"]) / parameters[prefix + "__scale"]
        )
    logits = (
        np.asarray(values @ parameters[head["parameter_prefix"] + "__coef"].T)[:, 0]
        + parameters[head["parameter_prefix"] + "__intercept"][0]
    )
    _require(np.isfinite(logits).all(), "Nonfinite frozen logits")
    p = expit(logits)
    return np.column_stack([1 - p, p])


def _score(rows, arrays, heads, parameters, text_transforms):
    equal, cross, predictions = {}, [], []
    for head_id, head in heads.items():
        source_condition, view, target, layer = (
            head[k] for k in ("condition", "view", "target", "layer")
        )
        destinations = [source_condition] + [b for a, b in CROSS_PAIRS if a == source_condition]
        for destination in destinations:
            indices = _subsets(rows, destination, view)
            X = arrays[layer][indices] if layer != "text" else None
            probability = _predict(
                head, parameters, text_transforms, X, [rows[i]["messages_text"] for i in indices]
            )
            metrics = {}
            for split in sorted({rows[i]["split"] for i in indices}):
                local = np.asarray(
                    [j for j, i in enumerate(indices) if rows[i]["split"] == split], dtype=np.int64
                )
                y = [rows[indices[j]]["labels"][target] for j in local]
                metric = binary_metrics(y, probability[local])
                metric["n_templates"] = len({rows[indices[j]]["template_id"] for j in local})
                if (
                    view == "sender"
                    and target == "reader"
                    and all(rows[indices[j]]["phase"] == "prior" for j in local)
                ):
                    _require(
                        metric["accuracy"] == 0.5,
                        "Identical-input reader control must be exactly 50%",
                    )
                metrics[split] = metric
            item = {
                "head_id": head_id,
                "selected_C_from_validation": head["selected_C"],
                "metrics": metrics,
            }
            if destination == source_condition:
                equal.setdefault(destination, {}).setdefault(view, {}).setdefault(target, {})[
                    layer
                ] = item
            else:
                cross.append(
                    {
                        "source_condition": source_condition,
                        "evaluation_condition": destination,
                        "view": view,
                        "target": target,
                        "layer": layer,
                        **item,
                    }
                )
            for j, i in enumerate(indices):
                row = rows[i]
                predictions.append(
                    {
                        "head_id": head_id,
                        "source_condition": source_condition,
                        "evaluation_condition": destination,
                        "view": view,
                        "target": target,
                        "layer": layer,
                        "split": row["split"],
                        "phase": row["phase"],
                        "example_id": row["example_id"],
                        "case_id": row["case_id"],
                        "template_id": row["template_id"],
                        "label": row["labels"][target],
                        "p0": float(probability[j, 0]),
                        "p1": float(probability[j, 1]),
                        "prediction": int(probability[j].argmax()),
                    }
                )
    return equal, cross, predictions


def _fresh_output(output):
    output = Path(output).resolve()
    _require(not output.exists(), "Use a fresh output directory")
    return output


def _write_predictions(path, rows):
    fields = (
        "head_id",
        "source_condition",
        "evaluation_condition",
        "view",
        "target",
        "layer",
        "split",
        "phase",
        "example_id",
        "case_id",
        "template_id",
        "label",
        "p0",
        "p1",
        "prediction",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _settings():
    return {
        "primary_target": "nested",
        "primary_layer": "layer_16",
        "primary_comparison": "equal-condition refits with matched validation C selection",
        "C_grid": list(C_GRID),
        "solver": "lbfgs",
        "max_iter": 2000,
        "seed": 20261120,
        "class_order": [0, 1],
        "activation_transform": "prior-train-only StandardScaler, then per-row L2",
        "text_settings": TEXT_SETTINGS,
        "text_fields": ["messages_text"],
        "cross_condition_pairs": [list(pair) for pair in CROSS_PAIRS],
        "test_or_control_used_for_selection": False,
        "layer_selection_from_test": False,
    }


def _finish(output, report, stage, extra=None):
    atomic_json(output / "report.json", report)
    manifest = {
        "schema": "belief_query_readout_artifacts_v1",
        "stage": stage,
        "status": "complete",
        "created_at": utc_now(),
        "source_code_sha256": file_sha256(__file__),
        "environment": environment_info(),
        "output_sha256": _snapshot(output),
        **(extra or {}),
    }
    atomic_json(output / "manifest.json", manifest)
    return report


def fit_query_comparison(rows, features, output_dir):
    """Fit legacy heads; persist numeric coefficients and safe text transforms."""
    output = _fresh_output(output_dir)
    arrays, design = _validate(rows, features)
    input_rows_hash = canonical_hash(rows)
    input_feature_hashes = {layer: _array_hash(features[layer]) for layer in LAYERS}
    parameters, heads, text_transforms = {"classes": np.asarray([0, 1], dtype=np.int64)}, {}, {}
    for condition in CONDITIONS:
        for view in VIEWS:
            indices = _subsets(rows, condition, view)
            selected_rows = [rows[i] for i in indices]
            train = np.asarray(
                [j for j, row in enumerate(selected_rows) if row["split"] == "train"],
                dtype=np.int64,
            )
            validation = np.asarray(
                [j for j, row in enumerate(selected_rows) if row["split"] == "validation"],
                dtype=np.int64,
            )
            _require(
                all(
                    selected_rows[j]["phase"] == "prior"
                    for j in np.concatenate([train, validation])
                ),
                "Controls cannot fit or select",
            )
            matrices = {}
            for layer in LAYERS:
                X = arrays[layer][indices]
                scaler = StandardScaler().fit(X[train])
                matrices[layer] = normalize(scaler.transform(X), norm="l2")
                transform = "transform__" + condition + "__" + view + "__" + layer
                parameters[transform + "__mean"] = scaler.mean_.copy()
                parameters[transform + "__scale"] = scaler.scale_.copy()
            text_key = condition + "__" + view
            vectorizer = _vectorizer()
            try:
                vectorizer.fit([selected_rows[j]["messages_text"] for j in train])
            except ValueError as error:
                raise ValueError(
                    "Query text control requires a nonempty training vocabulary"
                ) from error
            matrices["text"] = vectorizer.transform([row["messages_text"] for row in selected_rows])
            text_transforms[text_key] = {
                "terms": vectorizer.get_feature_names_out().tolist(),
                "idf": vectorizer.idf_.tolist(),
                "settings": TEXT_SETTINGS,
            }
            for target in TARGETS:
                y = np.asarray([row["labels"][target] for row in selected_rows], dtype=np.int64)
                for layer in (*LAYERS, "text"):
                    model, selection = _select_model(
                        matrices[layer][train], y[train], matrices[layer][validation], y[validation]
                    )
                    head_id = "__".join((condition, view, target, layer))
                    parameters[head_id + "__coef"] = model.coef_.copy()
                    parameters[head_id + "__intercept"] = model.intercept_.copy()
                    head = {
                        "condition": condition,
                        "view": view,
                        "target": target,
                        "layer": layer,
                        "parameter_prefix": head_id,
                        **selection,
                        "n_training_rows": len(train),
                        "n_training_templates": len(
                            {selected_rows[j]["template_id"] for j in train}
                        ),
                        "training_class_counts": {
                            str(label): int(np.sum(y[train] == label)) for label in (0, 1)
                        },
                    }
                    if layer == "text":
                        head["text_transform"] = text_key
                    else:
                        head["activation_transform"] = (
                            "transform__" + condition + "__" + view + "__" + layer
                        )
                    independent = _predict(
                        head,
                        parameters,
                        text_transforms,
                        arrays[layer][indices] if layer != "text" else None,
                        [row["messages_text"] for row in selected_rows],
                    )
                    _require(
                        np.allclose(
                            independent, model.predict_proba(matrices[layer]), rtol=0, atol=1e-12
                        ),
                        "Saved parameter scorer disagrees with fitted model",
                    )
                    heads[head_id] = head
    _require(
        canonical_hash(rows) == input_rows_hash
        and all(_array_hash(features[layer]) == input_feature_hashes[layer] for layer in LAYERS),
        "Inputs changed during fitting",
    )
    equal, cross, predictions = _score(rows, arrays, heads, parameters, text_transforms)
    metadata = {
        "schema": "belief_query_fitted_metadata_v1",
        "settings": _settings(),
        "design": design,
        "heads": heads,
        "input_rows_sha256": input_rows_hash,
        "input_feature_sha256": input_feature_hashes,
        "feature_widths": {layer: arrays[layer].shape[1] for layer in LAYERS},
        "known_case_ids": sorted({row["case_id"] for row in rows}),
        "known_example_ids": sorted({row["example_id"] for row in rows}),
        "known_template_ids": sorted({row["template_id"] for row in rows}),
        "known_entity_ids": sorted({entity for row in rows for entity in row["entity_ids"]}),
        "known_message_hashes": sorted({canonical_hash(row["messages_text"]) for row in rows}),
    }
    report = {
        "schema": "belief_query_comparison_report_v1",
        "status": "complete",
        "analysis_scope": "exploratory comparison on legacy cases; no prospective claim for those cases",
        "settings": _settings(),
        "design": design,
        "equal_condition": equal,
        "cross_condition": cross,
        "n_prediction_rows": len(predictions),
        "limitations": [
            "Explicit fictional-state decodability, not spontaneous mental state, causal use or unknown-message anticipation.",
            "Equal-condition refits concern accessibility; cross-condition scoring concerns coordinate invariance and retains source-condition scaling and weights.",
            "TF-IDF controls have the same messages as activation captures but are lexical models, not capable semantic judges.",
            "Checkpoint/revision/pooling identity must be bound by the calling capture runner; this array API validates layer names and widths.",
            "Template groups, not individual rows or repeated conditions, determine independence.",
        ],
    }
    output.mkdir(parents=True)
    atomic_npz(output / "parameters.npz", **parameters)
    atomic_json(output / "text_transforms.json", text_transforms)
    atomic_json(output / "fit_metadata.json", metadata)
    atomic_json(output / "rows.json", rows)
    _write_predictions(output / "predictions.csv", predictions)
    return _finish(output, report, "fit_legacy")


def _load_fit(fit_dir):
    manifest = read_json(fit_dir / "manifest.json")
    _require(
        manifest.get("schema") == "belief_query_readout_artifacts_v1"
        and manifest.get("stage") == "fit_legacy"
        and manifest.get("status") == "complete",
        "Need a complete fitted comparison",
    )
    hashes = _snapshot(fit_dir)
    _require(
        set(hashes) == set(manifest["output_sha256"]) | {"manifest.json"},
        "Fitted artifact set changed",
    )
    _require(
        all(hashes[name] == digest for name, digest in manifest["output_sha256"].items()),
        "Fitted artifact hash mismatch",
    )
    metadata, text = (
        read_json(fit_dir / "fit_metadata.json"),
        read_json(fit_dir / "text_transforms.json"),
    )
    _require(metadata["settings"] == _settings(), "Fitted readout settings/code contract changed")
    with np.load(fit_dir / "parameters.npz", allow_pickle=False) as archive:
        parameters = {key: archive[key].copy() for key in archive.files}
    _require(
        all(
            value.dtype.kind in "fiu" and np.isfinite(value).all() for value in parameters.values()
        ),
        "Invalid numeric model parameters",
    )
    _require(np.array_equal(parameters.get("classes"), [0, 1]), "Frozen class order changed")
    return metadata, parameters, text, hashes


def evaluate_fresh_query_comparison(rows, features, fit_dir, output_dir):
    """Apply saved scalers, weights and text transforms; never fit or choose C."""
    fit_dir, output = Path(fit_dir).resolve(), _fresh_output(output_dir)
    _require(
        not output.is_relative_to(fit_dir) and not fit_dir.is_relative_to(output),
        "Fresh output must be separate from fitted artifacts",
    )
    metadata, parameters, text, hashes = _load_fit(fit_dir)
    arrays, design = _validate(rows, features, fresh=True)
    for layer in LAYERS:
        _require(
            arrays[layer].shape[1] == metadata["feature_widths"][layer],
            "Fresh feature width changed",
        )
    for field, protected in (
        ("case_id", "known_case_ids"),
        ("example_id", "known_example_ids"),
        ("template_id", "known_template_ids"),
    ):
        _require(
            not ({row[field] for row in rows} & set(metadata[protected])),
            f"Fresh {field} overlaps original bank",
        )
    _require(
        not (
            {entity for row in rows for entity in row["entity_ids"]}
            & set(metadata["known_entity_ids"])
        ),
        "Fresh entity IDs overlap original bank",
    )
    _require(
        not (
            {canonical_hash(row["messages_text"]) for row in rows}
            & set(metadata["known_message_hashes"])
        ),
        "Fresh messages duplicate original bank",
    )
    row_hash = canonical_hash(rows)
    feature_hashes = {layer: _array_hash(features[layer]) for layer in LAYERS}
    equal, cross, predictions = _score(rows, arrays, metadata["heads"], parameters, text)
    _require(_snapshot(fit_dir) == hashes, "Fitted artifacts changed during fresh scoring")
    _require(
        canonical_hash(rows) == row_hash
        and all(_array_hash(features[layer]) == feature_hashes[layer] for layer in LAYERS),
        "Fresh inputs changed during scoring",
    )
    report = {
        "schema": "belief_query_fresh_report_v1",
        "status": "complete",
        "no_refit": True,
        "no_hyperparameter_selection": True,
        "settings": metadata["settings"],
        "design": design,
        "fitted_manifest_sha256": hashes["manifest.json"],
        "fitted_artifacts_unchanged": True,
        "input_rows_sha256": row_hash,
        "input_feature_sha256": feature_hashes,
        "equal_condition": equal,
        "cross_condition": cross,
        "n_prediction_rows": len(predictions),
        "fresh_entity_ids_disjoint": True,
        "limitations": [
            "Fresh authored wording templates share the logical task; they are not new story-family or natural-deception evidence.",
            "Prospective bank timing and checkpoint/tokenization identity must be established by the calling runner.",
            "Equal-condition accessibility and cross-condition coordinate stability remain distinct endpoints.",
            "All weights, C values, training scalers and text vocabularies remain those of the original fit.",
        ],
    }
    output.mkdir(parents=True)
    atomic_json(output / "rows.json", rows)
    _write_predictions(output / "predictions.csv", predictions)
    return _finish(output, report, "score_fresh", {"fitted_input_sha256": hashes})
