"""Verify and export the two predeclared layer-16 nested-belief readouts.

Reconstruction uses only prior training rows and the already recorded C. No
language model is loaded, no test-based setting is chosen, and source artifacts
remain unchanged. Numeric files contain no pickled estimator objects.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize

from latent_probes import belief_readout
from latent_probes.storage import (
    atomic_json,
    atomic_npz,
    canonical_hash,
    environment_info,
    file_sha256,
    read_json,
    utc_now,
)

LAYER = "layer_16"
TARGET = "nested"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _snapshot(root):
    return {
        p.relative_to(root).as_posix(): file_sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _load_probe(probe):
    if isinstance(probe, (str, Path)):
        with np.load(probe, allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}
    return probe


def apply_probe(X, probe_or_path):
    """Apply saved train-only scaling, row L2 and a binary sigmoid in NumPy.

    The caller must check model/revision/layer/timing/candidate compatibility
    against signature_json. Shape agreement alone cannot establish that match.
    """
    probe = _load_probe(probe_or_path)
    X = np.asarray(X, dtype=np.float64)
    mean, scale, coef, intercept, classes = (
        np.asarray(probe[key]) for key in ("mean", "scale", "coef", "intercept", "classes")
    )
    _require(X.ndim == 2 and len(X) > 0 and np.isfinite(X).all(), "Invalid probe input matrix")
    _require(
        mean.shape == scale.shape == (X.shape[1],)
        and coef.shape == (1, X.shape[1])
        and intercept.shape == (1,),
        "Probe feature/parameter shape mismatch",
    )
    _require(np.array_equal(classes, [0, 1]), "Unsupported probe classes")
    _require(
        all(np.isfinite(a).all() for a in (mean, scale, coef, intercept)) and (scale > 0).all(),
        "Invalid probe parameters",
    )
    values = (X - mean) / scale
    norms = np.sqrt(np.einsum("ij,ij->i", values, values))
    norms[norms < 10 * np.finfo(np.float64).eps] = 1.0
    values = values / norms[:, None]
    logits = (values @ coef.T + intercept)[:, 0]
    _require(np.isfinite(logits).all(), "Nonfinite probe logits")
    probability = expit(logits)
    return np.column_stack([1 - probability, probability])


def _metrics(labels, probabilities):
    """Independent metric calculation; no production metric helper is called."""
    labels = np.asarray(labels, dtype=int)
    predictions = probabilities.argmax(axis=1)
    confusion = np.asarray(
        [[np.sum((labels == a) & (predictions == b)) for b in (0, 1)] for a in (0, 1)]
    )
    recalls, f1 = [], []
    for index in (0, 1):
        tp = int(confusion[index, index])
        fn, fp = int(confusion[index].sum()) - tp, int(confusion[:, index].sum()) - tp
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)
        f1.append(2 * tp / (2 * tp + fn + fp) if 2 * tp + fn + fp else 0.0)
    return {
        "n_rows": len(labels),
        "accuracy": float(np.mean(labels == predictions)),
        "binary_log_loss": float(
            -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-15, 1)).mean()
        ),
        "macro_f1": float(np.mean(f1)),
        "balanced_accuracy": float(np.mean(recalls)),
        "class_recall": {str(i): recalls[i] for i in (0, 1)},
        "confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
    }


def _verify_metrics(actual, expected):
    _require(actual["n_rows"] == expected["n_rows"], "Verification row count mismatch")
    for key in ("accuracy", "binary_log_loss", "macro_f1", "balanced_accuracy"):
        _require(abs(actual[key] - expected[key]) <= 1e-12, f"Source metric mismatch: {key}")
    _require(
        actual["confusion_matrix_true_rows_predicted_columns"]
        == expected["confusion_matrix_true_rows_predicted_columns"],
        "Source confusion matrix mismatch",
    )
    _require(
        all(
            abs(actual["class_recall"][str(i)] - expected["class_recall"][str(i)]) <= 1e-12
            for i in (0, 1)
        ),
        "Source recall mismatch",
    )


def _source(source):
    manifest = read_json(source / "manifest.json")
    _require(
        manifest.get("schema") == "belief_construct_development_run_v1"
        and manifest.get("status") == "complete",
        "Source must be a complete belief diagnostic",
    )
    fixture = manifest.get("is_test_fixture")
    backend = manifest.get("backend", {})
    _require(
        type(fixture) is bool and fixture is backend.get("is_test_fixture"),
        "Fixture/genuine backend flags mismatch",
    )
    if not fixture:
        _require(
            backend.get("backend") == "huggingface" and bool(backend.get("resolved_revision")),
            "Genuine source lacks pinned HF identity",
        )
        _require(
            manifest.get("layer_index_check", {}).get("status") == "passed",
            "Genuine layer-index validation did not pass",
        )
    outputs = manifest.get("output_sha256", {})
    required = {"features.npz", "rows.json", "readouts.json", "protocol.json", "captures.json"}
    _require(required <= set(outputs), "Missing manifested source artifacts")
    actual = _snapshot(source)
    _require(
        set(actual) == set(outputs) | {"manifest.json"}, "Source artifact set differs from manifest"
    )
    for name, digest in outputs.items():
        path = (source / name).resolve()
        _require(path.is_relative_to(source), "Unsafe source artifact path")
        _require(actual[name] == digest, f"Source artifact hash mismatch: {name}")
    protocol, rows, readouts = (
        read_json(source / name) for name in ("protocol.json", "rows.json", "readouts.json")
    )
    _require(manifest["protocol_sha256"] == actual["protocol.json"], "Protocol identity mismatch")
    _require(
        manifest["panel_sha256"] == canonical_hash(protocol["panel"]), "Panel identity mismatch"
    )
    _require(
        protocol["primary_layer"] == readouts["primary_layer"] == LAYER
        and readouts["primary_target"] == TARGET,
        "Only predeclared primary layer16 nested heads may be exported",
    )
    source_module = manifest.get("source_sha256", {}).get("src/latent_probes/belief_readout.py")
    _require(
        source_module == file_sha256(inspect.getfile(belief_readout)),
        "Frozen readout source differs from capture-time code",
    )
    _require(readouts["row_sha256"] == canonical_hash(rows), "Readout/row identity mismatch")
    with np.load(source / "features.npz", allow_pickle=False) as archive:
        features = {key: archive[key].copy() for key in archive.files}
    _require(set(features) == {"layer_8", LAYER, "layer_24"}, "Source layers differ from protocol")
    for layer, values in features.items():
        _require(
            values.ndim == 2
            and len(values) == len(rows)
            and values.dtype.kind == "f"
            and np.isfinite(values).all(),
            "Invalid source feature matrix",
        )
        _require(
            _array_hash(values) == readouts["feature_sha256"][layer],
            "Readout/feature identity mismatch",
        )
    cases = {case["id"]: case for case in protocol["panel"]["cases"]}
    _require(len(cases) == len(protocol["panel"]["cases"]), "Duplicate source case")
    by_row = {row["example_id"]: row for row in rows}
    _require(len(by_row) == len(rows), "Duplicate source row")
    _require(len(rows) == len(cases) * 2, "Expected two views per case")
    for case in cases.values():
        _require(case["claim_place_bit"] == 1, "Candidate polarity changed")
        for view in ("omniscient", "sender"):
            row = by_row[case["id"] + "__" + view]
            _require(
                row["view"] == view
                and all(
                    row[key] == case[key] for key in ("template_id", "split", "phase", "labels")
                ),
                "Row/case metadata mismatch",
            )
            _require(
                row["messages_text"] == "\n".join(m["content"] for m in case["messages"][view]),
                "Row/story message mismatch",
            )
    captures = read_json(source / "captures.json")
    _require(
        len(captures) == len(rows) and len({r["example_id"] for r in captures}) == len(rows),
        "Capture row alignment mismatch",
    )
    index_by_id = {row["example_id"]: i for i, row in enumerate(rows)}
    for capture in captures:
        index = index_by_id[capture["example_id"]]
        row = rows[index]
        _require(
            capture["generated_answer_tokens_supplied"] == 0
            and capture["perspective_question_supplied"] is False,
            "Capture contains a question/answer",
        )
        _require(
            capture["prompt_token_ids"]
            and capture["selected_position"] == len(capture["prompt_token_ids"]) - 1,
            "Capture is not the last prompt token",
        )
        case_id = row["example_id"].rsplit("__", 1)[0]
        _require(
            capture["messages_sha256"] == canonical_hash(cases[case_id]["messages"][row["view"]]),
            "Capture/story identity mismatch",
        )
        name = capture["feature_file"]
        _require(
            name in outputs and capture["feature_sha256"] == actual[name],
            "Capture feature hash mismatch",
        )
        with np.load(source / name, allow_pickle=False) as archive:
            _require(
                all(np.array_equal(archive[layer], features[layer][index]) for layer in features),
                "Per-capture/bulk feature misalignment",
            )
    return manifest, protocol, rows, readouts, features[LAYER].astype(np.float64), actual


def export_readouts(source_run, output_dir):
    """Reconstruct and verify fixed primary heads; export only after both pass."""
    source, output = Path(source_run).resolve(), Path(output_dir).resolve()
    _require(not output.exists(), "Use a fresh export directory")
    _require(
        not output.is_relative_to(source) and not source.is_relative_to(output),
        "Export must be separate from source run",
    )
    manifest, protocol, rows, report, X, hashes = _source(source)
    _require(
        report["settings"]["activation_transform"]
        == "train-only StandardScaler, then per-row L2 normalization",
        "Unsupported source transform",
    )
    settings = report["settings"]
    _require(
        settings["solver"] == "lbfgs"
        and settings["max_iter"] == 2000
        and settings["fit_intercept"] is True
        and settings["class_weight"] is None
        and settings["seed"] == 20261120,
        "Source optimizer settings changed",
    )
    y = np.asarray([row["labels"][TARGET] for row in rows], dtype=np.int64)
    staged, verification = {}, {}
    for view in ("omniscient", "sender"):
        masks = {
            split: np.asarray(
                [i for i, row in enumerate(rows) if row["view"] == view and row["split"] == split],
                dtype=np.int64,
            )
            for split in ("train", "validation", "test", "control")
        }
        train = masks["train"]
        _require(
            len(train) > 0 and all(rows[i]["phase"] == "prior" for i in train),
            "Only prior training cases may fit",
        )
        target = report["readouts"][view][TARGET]["layers"][LAYER]
        c = target["selected_C"]
        eligible = [item for item in target["candidates"] if item["converged"]]
        recorded_choice = min(eligible, key=lambda item: (item["validation_log_loss"], item["C"]))
        _require(
            c in settings["C_grid"] and c == recorded_choice["C"],
            "Recorded C selection is inconsistent",
        )
        scaler = StandardScaler().fit(X[train])
        values = normalize(scaler.transform(X), norm="l2")
        _require(
            _array_hash(scaler.mean_) == target["train_statistics"]["mean_sha256"]
            and _array_hash(scaler.scale_) == target["train_statistics"]["scale_sha256"],
            "Training scaler fingerprint mismatch",
        )
        model = LogisticRegression(C=c, solver="lbfgs", max_iter=2000, random_state=20261120)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(values[train], y[train])
        _require(
            not any(issubclass(w.category, ConvergenceWarning) for w in caught),
            "Reconstruction did not converge",
        )
        _require(np.array_equal(model.classes_, [0, 1]), "Expected binary model")
        _require(
            _array_hash(model.coef_) == target["fit_fingerprint"]["coefficients_sha256"]
            and _array_hash(model.intercept_) == target["fit_fingerprint"]["intercept_sha256"],
            "Classifier fingerprint mismatch; reproduce the source numeric environment",
        )
        signature = {
            "schema": "belief_nested_probe_signature_v1",
            **{
                key: manifest["backend"].get(key)
                for key in (
                    "model_id",
                    "resolved_revision",
                    "dtype",
                    "quantization",
                    "chat_template_sha256",
                    "attention_implementation",
                    "is_test_fixture",
                )
            },
            "layer": 16,
            "hf_hidden_state_index": 17,
            "view": view,
            "pooling": "last_prompt_token",
            "timing": "story_only_before_perspective_question_or_generated_answer",
            "candidate": "token_is_in_case_place_b",
            "candidate_place_bit": 1,
            "target": TARGET,
            "story_target": "narrated_sender_thinks_reader_believes_candidate",
            "panel_id": protocol["panel"]["panel_id"],
            "is_natural_message_forecast": False,
        }
        probe = {
            "coef": model.coef_.copy(),
            "intercept": model.intercept_.copy(),
            "mean": scaler.mean_.copy(),
            "scale": scaler.scale_.copy(),
            "classes": model.classes_.copy(),
            "signature_json": np.asarray(json.dumps(signature, sort_keys=True)),
            "settings_json": np.asarray(
                json.dumps(
                    {
                        "C": c,
                        "transform": settings["activation_transform"],
                        "optimizer": "lbfgs",
                        "max_iter": 2000,
                        "seed": 20261120,
                    },
                    sort_keys=True,
                )
            ),
        }
        probabilities = apply_probe(X, probe)
        reference = model.predict_proba(values)
        max_error = float(np.max(np.abs(probabilities - reference)))
        _require(max_error <= 1e-12, "Independent probability reconstruction failed")
        metrics = {}
        for split, indices in masks.items():
            if len(indices):
                metrics[split] = _metrics(y[indices], probabilities[indices])
                _verify_metrics(metrics[split], target["metrics"][split])
        filename = view + "_layer_16_nested.npz"
        staged[filename] = probe
        verification[view] = {
            "selected_C_from_source_validation": c,
            "n_training_rows": len(train),
            "n_training_templates": len({rows[i]["template_id"] for i in train}),
            "scaler_and_classifier_fingerprints_exact": True,
            "independent_probability_max_abs_error": max_error,
            "probability_verification_scope": "NumPy scorer versus fixed-C reconstruction; aggregate metrics versus source report, which has no saved per-row probabilities",
            "independently_reproduced_metrics": metrics,
            "test_predictions": [
                {
                    "example_id": rows[i]["example_id"],
                    "label": int(y[i]),
                    "probability_one": float(probabilities[i, 1]),
                    "prediction": int(probabilities[i].argmax()),
                }
                for i in masks["test"]
            ],
            "artifact": filename,
        }
    _require(_snapshot(source) == hashes, "Original source changed during reconstruction")
    output.mkdir(parents=True)
    for name, arrays in staged.items():
        atomic_npz(output / name, **arrays)
    atomic_json(output / "verification.json", verification)
    unchanged = _snapshot(source) == hashes
    exported = {
        "schema": "belief_nested_probe_export_v1",
        "status": "complete" if unchanged else "failed_source_changed",
        "created_at": utc_now(),
        "is_test_fixture": manifest["is_test_fixture"],
        "source_run": str(source),
        "source_input_sha256": hashes,
        "source_artifacts_unchanged": unchanged,
        "script_sha256": file_sha256(__file__),
        "environment": environment_info(),
        "primary_layer": LAYER,
        "target": TARGET,
        "no_test_guided_setting_selection": True,
        "output_sha256": _snapshot(output),
        "limitations": [
            "Reconstruction and parameter hashes reproduce a declared source; this exporter does not independently prove that genuine model captures occurred.",
            "These heads decode stipulated fictional nested-perspective labels, not natural intent, deception or an unknown-message forecast.",
            "Exact reconstruction can depend on the source numerical environment; a fingerprint mismatch fails export instead of relaxing it.",
            "Future transfer must verify model, revision, layer, view, timing, candidate polarity and normalization compatibility.",
        ],
    }
    atomic_json(output / "manifest.json", exported)
    _require(unchanged, "Original source changed during export")
    return exported


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = export_readouts(args.run, args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "is_test_fixture": report["is_test_fixture"],
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
