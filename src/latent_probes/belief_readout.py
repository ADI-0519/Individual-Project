"""Grouped binary readouts for a neutral, explicitly authored belief-state gate.

Caller-provided activations and messages are the only predictors. Label metadata
is never appended to text. This CPU API does not itself audit the LLM forwards
or establish spontaneous beliefs, anticipation, intention, or deception.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from itertools import product

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize

LAYERS = ("layer_8", "layer_16", "layer_24")
TARGETS = ("world", "reader", "nested")
VIEWS = ("omniscient", "sender")
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
SPLITS = ("train", "validation", "test", "control")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array_hash(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def binary_metrics(labels, probabilities):
    """Fixed {0,1} universe; invalid labels are rejected, never recoded."""
    y, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=np.float64)
    _require(y.ndim == 1 and len(y) > 0 and np.isin(y, [0, 1]).all(), "Need binary labels")
    _require(
        probabilities.shape == (len(y), 2)
        and np.isfinite(probabilities).all()
        and (probabilities >= 0).all()
        and (probabilities <= 1).all()
        and np.allclose(probabilities.sum(axis=1), 1, rtol=1e-10, atol=1e-12),
        "Invalid binary probabilities",
    )
    prediction = probabilities.argmax(axis=1)
    confusion = np.zeros((2, 2), dtype=np.int64)
    np.add.at(confusion, (y.astype(np.int64), prediction), 1)
    recall, f1 = [], []
    for index in range(2):
        tp = int(confusion[index, index])
        fn, fp = int(confusion[index].sum()) - tp, int(confusion[:, index].sum()) - tp
        recall.append(tp / (tp + fn) if tp + fn else 0.0)
        f1.append(2 * tp / (2 * tp + fn + fp) if 2 * tp + fn + fp else 0.0)
    return {
        "n_rows": len(y),
        "class_counts": {str(i): int(np.sum(y == i)) for i in range(2)},
        "accuracy": float(np.mean(prediction == y)),
        "binary_log_loss": float(
            -np.log(np.clip(probabilities[np.arange(len(y)), y.astype(int)], 1e-15, 1)).mean()
        ),
        "macro_f1": float(np.mean(f1)),
        "balanced_accuracy": float(np.mean(recall)),
        "class_recall": {str(i): recall[i] for i in range(2)},
        "class_f1": {str(i): f1[i] for i in range(2)},
        "confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
        "absent_class_policy": "fixed binary universe; undefined recall/F1 equals zero",
        "probability_clip": 1e-15,
    }


def _validate(rows, features, primary_layer):
    _require(isinstance(rows, list) and bool(rows), "Need a nonempty row list")
    _require(primary_layer in LAYERS, "Primary layer must be declared from prespecified layers")
    _require(set(features) == set(LAYERS), "Exactly layer_8/layer_16/layer_24 features required")
    arrays = {}
    for layer, value in features.items():
        array = np.asarray(value)
        _require(
            array.ndim == 2 and array.shape[0] == len(rows) and array.shape[1] > 0,
            "Feature alignment/shape mismatch",
        )
        _require(
            array.dtype.kind == "f" and np.isfinite(array).all(), "Features must be finite floats"
        )
        arrays[layer] = array.astype(np.float64)
    identities, template_splits, cells = set(), {}, {}
    for i, row in enumerate(rows):
        _require(
            all(
                key in row
                for key in (
                    "example_id",
                    "template_id",
                    "split",
                    "phase",
                    "view",
                    "messages_text",
                    "labels",
                )
            ),
            "Missing row fields",
        )
        _require(row["view"] in VIEWS and row["split"] in SPLITS, "Unknown view/split")
        _require(
            isinstance(row["messages_text"], str) and row["messages_text"].strip(),
            "Missing message text",
        )
        _require(
            all(isinstance(row[k], str) and row[k] for k in ("example_id", "template_id")),
            "Missing identity",
        )
        identity = (row["example_id"], row["view"])
        _require(identity not in identities, "Duplicate example/view identity")
        identities.add(identity)
        _require(set(row["labels"]) == set(TARGETS), "Exactly world/reader/nested labels required")
        _require(
            all(
                isinstance(v, (int, np.integer))
                and not isinstance(v, (bool, np.bool_))
                and v in (0, 1)
                for v in row["labels"].values()
            ),
            "Labels must be binary integers",
        )
        if row["split"] == "control":
            _require(
                row["phase"] == "delivered_current", "Controls must be delivered-current phase"
            )
            _require(
                len(set(row["labels"].values())) == 1,
                "Delivered-current control labels must all equal world",
            )
            continue
        _require(row["phase"] == "prior", "Only prior rows may train/validate/test")
        template = row["template_id"]
        _require(
            template not in template_splits or template_splits[template] == row["split"],
            "Template leakage across prior splits",
        )
        template_splits[template] = row["split"]
        cell = (template, row["view"], *(int(row["labels"][t]) for t in TARGETS))
        _require(cell not in cells, "Duplicate factorial cell")
        cells[cell] = i
    _require(
        set(template_splits.values()) == {"train", "validation", "test"},
        "All three prior splits required",
    )
    factorial = set(product((0, 1), repeat=3))
    for template in template_splits:
        for view in VIEWS:
            observed = {cell[2:] for cell in cells if cell[:2] == (template, view)}
            _require(
                observed == factorial,
                "Every template/view requires the full balanced 2x2x2 factorial",
            )
    pairs = {split: 0 for split in ("train", "validation", "test")}
    for template, split in template_splits.items():
        for world, nested in product((0, 1), repeat=2):
            a, b = (
                cells[(template, "sender", world, 0, nested)],
                cells[(template, "sender", world, 1, nested)],
            )
            _require(
                rows[a]["messages_text"].encode() == rows[b]["messages_text"].encode(),
                "Sender reader-pair messages expose the hidden reader state",
            )
            _require(
                all(np.array_equal(arrays[layer][a], arrays[layer][b]) for layer in LAYERS),
                "Sender reader-pair features differ despite identical input",
            )
            pairs[split] += 1
    return arrays, template_splits, pairs


def _masks(rows, view):
    return {
        split: np.asarray(
            [i for i, row in enumerate(rows) if row["view"] == view and row["split"] == split],
            dtype=np.int64,
        )
        for split in SPLITS
    }


def _fit_select(X, y, masks):
    """Fit only train rows and choose C only from neutral validation labels."""
    candidates, models = [], []
    for c in C_GRID:
        model = LogisticRegression(C=c, solver="lbfgs", max_iter=2000, random_state=20261120)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X[masks["train"]], y[masks["train"]])
        _require(np.array_equal(model.classes_, [0, 1]), "Both training classes required")
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        validation = binary_metrics(
            y[masks["validation"]], model.predict_proba(X[masks["validation"]])
        )
        candidates.append(
            {"C": c, "validation_log_loss": validation["binary_log_loss"], "converged": converged}
        )
        models.append(model)
    eligible = [i for i, item in enumerate(candidates) if item["converged"]]
    _require(bool(eligible), "No converged readout candidate")
    selected = min(
        eligible, key=lambda i: (candidates[i]["validation_log_loss"], candidates[i]["C"])
    )
    model = models[selected]
    return {
        "selected_C": candidates[selected]["C"],
        "validation_selection_loss": candidates[selected]["validation_log_loss"],
        "candidates": candidates,
        "fit_fingerprint": {
            "coefficients_sha256": _array_hash(model.coef_),
            "intercept_sha256": _array_hash(model.intercept_),
        },
        "metrics": {
            split: binary_metrics(y[indices], model.predict_proba(X[indices]))
            if len(indices)
            else {"status": "no_rows", "n_rows": 0}
            for split, indices in masks.items()
        },
    }


def _group_stats(rows, masks, array=None):
    output = {}
    for split, indices in masks.items():
        output[split] = {
            "n_rows": len(indices),
            "n_unique_groups": len({rows[i]["template_id"] for i in indices}),
            "n_unique_messages": len({rows[i]["messages_text"] for i in indices}),
            "template_ids": sorted({rows[i]["template_id"] for i in indices}),
        }
        if array is not None and len(indices):
            output[split]["n_unique_feature_rows"] = len(np.unique(array[indices], axis=0))
    return output


def _baseline(rows, masks, target, kind):
    labels = np.asarray([int(r["labels"][target]) for r in rows])
    world = np.asarray([int(r["labels"]["world"]) for r in rows])
    prediction = np.zeros(len(rows), dtype=np.int64) if kind == "constant_zero" else world
    if kind == "inverted_source_world_copy":
        prediction = 1 - world
    probabilities = np.column_stack([1 - prediction, prediction])
    return {
        split: binary_metrics(labels[indices], probabilities[indices])
        if len(indices)
        else {"status": "no_rows", "n_rows": 0}
        for split, indices in masks.items()
    }


def evaluate_belief_readouts(rows, features, *, primary_layer="layer_16"):
    """Return JSON-safe diagnostics; perform no file writes or language-model calls.

    Features must contain aligned [N,D] arrays for layers 8,16,24. Every prior
    template/view must contain exactly one of each independent world/reader/
    nested bit combination. Delivered-current controls never fit any transform
    or classifier and never participate in regularization selection.
    """
    arrays, template_splits, pairs = _validate(rows, features, primary_layer)
    canonical_rows = [
        {**row, "labels": {target: int(row["labels"][target]) for target in TARGETS}}
        for row in rows
    ]
    row_digest = hashlib.sha256(
        json.dumps(
            canonical_rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()
    result = {
        "schema": "neutral_belief_readout_v1",
        "primary_layer": primary_layer,
        "primary_target": "nested",
        "layers": list(LAYERS),
        "settings": {
            "C_grid": list(C_GRID),
            "selection": "minimum prior-validation binary log loss among converged candidates; smaller C breaks ties",
            "solver": "lbfgs",
            "max_iter": 2000,
            "fit_intercept": True,
            "class_weight": None,
            "seed": 20261120,
            "activation_transform": "train-only StandardScaler, then per-row L2 normalization",
            "text_transform": "TF-IDF unigrams/bigrams fitted on prior training messages_text only",
            "text_predictor_fields": ["messages_text"],
            "controls_used_for_fitting_or_selection": False,
            "layer_selection_from_test": False,
            "secondary_layers": [layer for layer in LAYERS if layer != primary_layer],
        },
        "row_sha256": row_digest,
        "feature_sha256": {layer: _array_hash(features[layer]) for layer in LAYERS},
        "design": {
            "n_prior_factorial_cases": 8 * len(template_splits),
            "n_prior_view_rows": sum(r["phase"] == "prior" for r in rows),
            "n_control_view_rows": sum(r["split"] == "control" for r in rows),
            "template_partitions": {
                split: sorted(t for t, s in template_splits.items() if s == split)
                for split in ("train", "validation", "test")
            },
            "independence_unit": "template; repeated factorial cells and views are not independent templates",
        },
        "sender_reader_impossibility_control": {
            "paired_identical_messages_and_features_verified": True,
            "n_opposite_reader_pairs": pairs,
            "maximum_deterministic_prior_accuracy": 0.5,
            "interpretation": "Each identical sender input occurs with both actual reader labels; a deterministic classifier cannot distinguish them.",
            "observed_accuracy": {},
        },
        "readouts": {},
        "baselines": {},
        "limitations": [
            "Authored explicit belief-state labels do not establish spontaneous hidden beliefs, anticipation, intention or deceptive behavior.",
            "Primary layer/target must be fixed by the caller before outputs; this API does not choose a layer from test performance.",
            "Training-label fitting and prior validation selection are separate from test/control evaluation; repeated rows have template-level dependence.",
            "The caller must ensure label metadata was absent from the LLM input; this CPU API does not replay or inspect LLM forwards.",
            "World-copy baselines use the authored source-world bit as an oracle diagnostic, not a trained monitor.",
            "TF-IDF is a lexical control, not a capable semantic LLM monitor.",
            "Delivered-current controls may reuse training templates and are reported as controls, not held-out-template estimates.",
        ],
    }
    for view in VIEWS:
        masks = _masks(rows, view)
        _require(
            all(len(masks[split]) for split in ("train", "validation", "test")),
            "Missing view partition",
        )
        result["readouts"][view], result["baselines"][view] = {}, {}
        texts = [r["messages_text"] for r in rows]
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20000,
            token_pattern=r"(?u)\b\w+\b",
            sublinear_tf=True,
            dtype=np.float64,
        )
        try:
            vectorizer.fit([texts[i] for i in masks["train"]])
            text_matrix = vectorizer.transform(texts)
        except ValueError as error:
            if "empty vocabulary" not in str(error):
                raise
            text_matrix = None
        scaled = {}
        scale_stats = {}
        for layer in LAYERS:
            scaler = StandardScaler().fit(arrays[layer][masks["train"]])
            scaled[layer] = normalize(scaler.transform(arrays[layer]), norm="l2")
            scale_stats[layer] = {
                "mean_sha256": _array_hash(scaler.mean_),
                "scale_sha256": _array_hash(scaler.scale_),
                "n_training_rows": len(masks["train"]),
                "group_stats": _group_stats(rows, masks, arrays[layer]),
            }
        for target in TARGETS:
            y = np.asarray([int(r["labels"][target]) for r in rows])
            layers = {}
            for layer in LAYERS:
                fitted = _fit_select(scaled[layer], y, masks)
                fitted["train_statistics"] = scale_stats[layer]
                fitted["role"] = (
                    "primary" if layer == primary_layer and target == "nested" else "diagnostic"
                )
                layers[layer] = fitted
            if text_matrix is None:
                text = {"status": "unavailable_empty_training_vocabulary"}
            else:
                text = _fit_select(text_matrix, y, masks)
                text["group_stats"] = _group_stats(rows, masks)
                text["vocabulary_size"] = len(vectorizer.vocabulary_)
                text["vocabulary_sha256"] = _array_hash(
                    np.asarray(vectorizer.get_feature_names_out(), dtype=np.str_)
                )
                text["idf_sha256"] = _array_hash(vectorizer.idf_)
            result["readouts"][view][target] = {"layers": layers, "text": text}
            result["baselines"][view][target] = {
                kind: _baseline(rows, masks, target, kind)
                for kind in ("constant_zero", "source_world_copy", "inverted_source_world_copy")
            }
            if view == "sender" and target == "reader":
                methods = {**layers, "text": text}
                for method, fitted in methods.items():
                    if "metrics" not in fitted:
                        continue
                    observed = {
                        split: fitted["metrics"][split]["accuracy"]
                        for split in ("train", "validation", "test")
                    }
                    _require(
                        all(value == 0.5 for value in observed.values()),
                        "Identical-input reader negative control exceeded its exact 50% limit",
                    )
                    result["sender_reader_impossibility_control"]["observed_accuracy"][method] = (
                        observed
                    )
    result["primary_results"] = {
        view: result["readouts"][view]["nested"]["layers"][primary_layer] for view in VIEWS
    }
    return result
