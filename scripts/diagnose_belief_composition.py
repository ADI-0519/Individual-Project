"""Post-hoc factual readouts plus an external Boolean nested-state rule.

This uses saved query-free story features only. It tests recoverable ingredients
and classifier form, not spontaneous nested representations, intent, or causal
use of an internally anticipated belief. No model is loaded or queried.
"""

from __future__ import annotations

import argparse
import importlib.util
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize

from latent_probes.storage import atomic_json, atomic_npz, environment_info, file_sha256, utc_now

HELPER_PATH = Path(__file__).with_name("export_belief_readout.py")
SPEC = importlib.util.spec_from_file_location("_composition_export_verification", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load the fixed source-verification helper")
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def hard_composition(world_predictions, recorded_predictions):
    world, recorded = np.asarray(world_predictions), np.asarray(recorded_predictions)
    _require(world.shape == recorded.shape and world.ndim == 1, "Fact prediction shape mismatch")
    _require(
        np.isin(world, [0, 1]).all() and np.isin(recorded, [0, 1]).all(),
        "Fact predictions must be binary",
    )
    return (world == recorded).astype(np.int64)


def classification_metrics(labels, predictions):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    _require(
        labels.ndim == predictions.ndim == 1
        and len(labels) > 0
        and labels.shape == predictions.shape,
        "Classification shape mismatch",
    )
    _require(
        np.isin(labels, [0, 1]).all() and np.isin(predictions, [0, 1]).all(),
        "Classification labels must be binary",
    )
    confusion = np.asarray(
        [[np.sum((labels == a) & (predictions == b)) for b in (0, 1)] for a in (0, 1)]
    )
    recall, f1 = [], []
    for index in (0, 1):
        tp = int(confusion[index, index])
        fn, fp = int(confusion[index].sum()) - tp, int(confusion[:, index].sum()) - tp
        recall.append(tp / (tp + fn) if tp + fn else 0.0)
        f1.append(2 * tp / (2 * tp + fn + fp) if 2 * tp + fn + fp else 0.0)
    return {
        "n_rows": len(labels),
        "correct": int(np.sum(labels == predictions)),
        "accuracy": float(np.mean(labels == predictions)),
        "macro_f1": float(np.mean(f1)),
        "class_recall": {str(i): recall[i] for i in (0, 1)},
        "confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
    }


def _fit(values, labels, indices, c):
    model = LogisticRegression(C=float(c), solver="lbfgs", max_iter=2000, random_state=20261120)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(values[indices], labels[indices])
    _require(
        not any(issubclass(w.category, ConvergenceWarning) for w in caught),
        "Factual reconstruction did not converge",
    )
    return model


def _recorded_head(values, labels, masks, head):
    model = _fit(values, labels, masks["train"], head["selected_C"])
    _require(
        HELPER._array_hash(model.coef_) == head["fit_fingerprint"]["coefficients_sha256"]
        and HELPER._array_hash(model.intercept_) == head["fit_fingerprint"]["intercept_sha256"],
        "Recorded head fingerprint mismatch",
    )
    probabilities = model.predict_proba(values)
    for split, indices in masks.items():
        if len(indices):
            HELPER._verify_metrics(
                HELPER._metrics(labels[indices], probabilities[indices]), head["metrics"][split]
            )
    return model, probabilities.argmax(axis=1)


def fit_receipt_head(X_train, y_train, X_validation, y_validation):
    """C selection API receives no test labels or indices."""
    _require(np.array_equal(np.unique(y_train), [0, 1]), "Need both factual-record classes")
    candidates, models = [], []
    for c in C_GRID:
        model = _fit(X_train, y_train, np.arange(len(y_train)), c)
        loss = HELPER._metrics(y_validation, model.predict_proba(X_validation))["binary_log_loss"]
        candidates.append({"C": c, "validation_binary_log_loss": loss})
        models.append(model)
    selected = min(
        range(len(candidates)),
        key=lambda i: (candidates[i]["validation_binary_log_loss"], candidates[i]["C"]),
    )
    return models[selected], {
        "selected_C": candidates[selected]["C"],
        "selection": "minimum prior validation binary log loss; smaller C breaks ties",
        "candidates": candidates,
    }


def diagnose_composition(source_run, output_path):
    source, output = Path(source_run).resolve(), Path(output_path).resolve()
    sidecar = output.with_suffix(".npz")
    _require(
        output.suffix == ".json"
        and not output.exists()
        and not sidecar.exists()
        and not output.is_relative_to(source),
        "Use fresh JSON and NPZ outputs outside the source run",
    )
    manifest, protocol, rows, report, X, hashes = HELPER._source(source)
    cases = {case["id"]: case for case in protocol["panel"]["cases"]}
    b = np.asarray(
        [cases[row["example_id"].rsplit("__", 1)[0]]["sender_recorded"] for row in rows],
        dtype=np.int64,
    )
    world = np.asarray([row["labels"]["world"] for row in rows], dtype=np.int64)
    nested = np.asarray([row["labels"]["nested"] for row in rows], dtype=np.int64)
    prior = np.asarray([i for i, row in enumerate(rows) if row["phase"] == "prior"], dtype=np.int64)
    _require(np.isin(b, [0, 1]).all(), "Bad factual-record labels")
    _require(
        np.array_equal(hard_composition(world[prior], b[prior]), nested[prior]),
        "Authored prior labels do not implement the specified Boolean rule",
    )
    output_report = {
        "schema": "belief_composition_diagnostic_v2",
        "created_at": utc_now(),
        "analysis_status": "posthoc_exploratory_same_known_cases",
        "layer": "layer_16",
        "is_test_fixture": manifest["is_test_fixture"],
        "source_input_sha256": hashes,
        "script_sha256": file_sha256(__file__),
        "verification_helper_sha256": file_sha256(HELPER_PATH),
        "environment": environment_info(),
        "C_grid": list(C_GRID),
        "rule": "nested = world if sender_recorded == 1 else 1-world; hard XNOR of decoded facts",
        "gold_rule_agreement_on_prior_rows": 1.0,
        "composed_probabilities": "not defined; no calibration or conditional-independence assumption",
        "views": {},
        "prior_predictions": [],
        "limitations": [
            "Post-hoc diagnostic on already known cases; no fresh confirmation or layer selection.",
            "The composed classifier decodes two narrated facts and applies an external rule; it does not establish spontaneous nested representation, intention or causal use.",
            "Sender-recorded attendance is an explicit display fact, not a label added to the language-model input.",
            "Both factual heads fit prior training rows only; only factual-record C is newly selected, using prior validation labels.",
            "Delivered-current messages override the prior nested rule; controls are excluded from composition metrics and all fitting/selection.",
            "Repeated factorial cells share wording templates; row accuracies are descriptive and do not yield population confidence claims.",
        ],
    }
    parameters = {}
    for view in ("omniscient", "sender"):
        masks = {
            split: np.asarray(
                [i for i, row in enumerate(rows) if row["view"] == view and row["split"] == split],
                dtype=np.int64,
            )
            for split in ("train", "validation", "test", "control")
        }
        _require(
            all(
                rows[i]["phase"] == "prior"
                for i in np.concatenate([masks[s] for s in ("train", "validation", "test")])
            ),
            "Prior fit/evaluation includes controls",
        )
        scaler = StandardScaler().fit(X[masks["train"]])
        values = normalize(scaler.transform(X), norm="l2")
        for target in ("world", "nested"):
            stats = report["readouts"][view][target]["layers"]["layer_16"]["train_statistics"]
            _require(
                HELPER._array_hash(scaler.mean_) == stats["mean_sha256"]
                and HELPER._array_hash(scaler.scale_) == stats["scale_sha256"],
                "Recorded scaler fingerprint mismatch",
            )
        world_head = report["readouts"][view]["world"]["layers"]["layer_16"]
        nested_head = report["readouts"][view]["nested"]["layers"]["layer_16"]
        world_model, world_prediction = _recorded_head(values, world, masks, world_head)
        nested_model, nested_prediction = _recorded_head(values, nested, masks, nested_head)
        receipt_model, receipt_selection = fit_receipt_head(
            values[masks["train"]],
            b[masks["train"]],
            values[masks["validation"]],
            b[masks["validation"]],
        )
        receipt_prediction = receipt_model.predict(values)
        composed = hard_composition(world_prediction, receipt_prediction)
        _require(
            np.array_equal(world_model.classes_, [0, 1])
            and np.array_equal(receipt_model.classes_, world_model.classes_)
            and np.array_equal(nested_model.classes_, world_model.classes_),
            "Factual and direct-head classes differ",
        )
        parameters.update(
            {
                view + "_mean": scaler.mean_.copy(),
                view + "_scale": scaler.scale_.copy(),
                view + "_world_coef": world_model.coef_.copy(),
                view + "_world_intercept": world_model.intercept_.copy(),
                view + "_receipt_coef": receipt_model.coef_.copy(),
                view + "_receipt_intercept": receipt_model.intercept_.copy(),
                view + "_direct_nested_coef": nested_model.coef_.copy(),
                view + "_direct_nested_intercept": nested_model.intercept_.copy(),
                view + "_classes": world_model.classes_.copy(),
            }
        )
        for i in prior:
            if rows[i]["view"] == view:
                output_report["prior_predictions"].append(
                    {
                        "example_id": rows[i]["example_id"],
                        "template_id": rows[i]["template_id"],
                        "split": rows[i]["split"],
                        "view": view,
                        "phase": "prior",
                        "gold": {
                            "world": int(world[i]),
                            "sender_recorded": int(b[i]),
                            "nested": int(nested[i]),
                        },
                        "predictions": {
                            "world": int(world_prediction[i]),
                            "sender_recorded": int(receipt_prediction[i]),
                            "direct_nested": int(nested_prediction[i]),
                            "composed_nested": int(composed[i]),
                        },
                    }
                )
        groups = {}
        for split, indices in masks.items():
            if not len(indices):
                groups[split] = {"status": "no_rows"}
                continue
            group = {
                "n_templates": len({rows[i]["template_id"] for i in indices}),
                "world": classification_metrics(world[indices], world_prediction[indices]),
                "sender_recorded": classification_metrics(b[indices], receipt_prediction[indices]),
                "original_direct_nested": classification_metrics(
                    nested[indices], nested_prediction[indices]
                ),
            }
            if split == "control":
                group["composed_nested"] = {
                    "status": "out_of_scope_delivered_message_overrides_prior_rule"
                }
            else:
                group["composed_nested"] = classification_metrics(
                    nested[indices], composed[indices]
                )
                world_ok, b_ok = (
                    world_prediction[indices] == world[indices],
                    receipt_prediction[indices] == b[indices],
                )
                group["joint_fact_errors"] = {
                    "both_facts_correct": int(np.sum(world_ok & b_ok)),
                    "both_facts_wrong_error_cancellation": int(np.sum(~world_ok & ~b_ok)),
                    "exactly_one_fact_wrong": int(np.sum(world_ok != b_ok)),
                }
                _require(
                    group["composed_nested"]["correct"]
                    == group["joint_fact_errors"]["both_facts_correct"]
                    + group["joint_fact_errors"]["both_facts_wrong_error_cancellation"],
                    "Composition error accounting mismatch",
                )
            groups[split] = group
        output_report["views"][view] = {
            "recorded_world_C": world_head["selected_C"],
            "recorded_direct_nested_C": nested_head["selected_C"],
            "recorded_heads_and_metrics_exactly_reproduced": True,
            "factual_record_selection": receipt_selection,
            "factual_record_coefficients_sha256": HELPER._array_hash(receipt_model.coef_),
            "factual_record_intercept_sha256": HELPER._array_hash(receipt_model.intercept_),
            "groups": groups,
        }
    _require(
        HELPER._snapshot(source) == hashes, "Original source changed during composition diagnosis"
    )
    output_report["source_artifacts_unchanged"] = True
    _require(
        len(output_report["prior_predictions"]) == len(prior)
        and len({r["example_id"] for r in output_report["prior_predictions"]}) == len(prior),
        "Prior prediction ledger is incomplete",
    )
    _require(
        not output.exists() and not sidecar.exists(), "JSON or NPZ output appeared during diagnosis"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_npz(sidecar, **parameters)
    output_report["parameter_sidecar"] = {
        "filename": sidecar.name,
        "sha256": file_sha256(sidecar),
        "arrays": {key: list(value.shape) for key, value in parameters.items()},
        "feature_layer": "layer_16",
        "transform": "saved train-only mean/scale, then per-row L2; near-zero row norm replaced by one",
        "world_and_direct_nested_decision": "sigmoid, then argmax over [1-p,p]; ties choose class 0",
        "receipt_decision": "decision logit > 0; zero ties choose class 0",
        "composition": "hard XNOR of world and receipt predictions",
        "contains_pickle": False,
    }
    atomic_json(output, output_report)
    return output_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = diagnose_composition(args.run, args.output)
    for view, result in report["views"].items():
        test = result["groups"]["test"]
        print(
            view,
            {
                key: test[key]["accuracy"]
                for key in ("world", "sender_recorded", "original_direct_nested", "composed_nested")
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
