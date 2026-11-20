"""Balanced, grouped toy inputs test readout isolation and information limits."""

from __future__ import annotations

from copy import deepcopy
from itertools import product

import numpy as np
import pytest

import latent_probes.belief_readout as module


def fixture():
    rows, vectors = [], []
    for split in ("train", "validation", "test"):
        template = "template-" + split
        for world, reader, nested in product((0, 1), repeat=3):
            for view in module.VIEWS:
                message = f"worldbit{world} nestedbit{nested} form {template}"
                if view == "omniscient":
                    message += f" actualreaderbit{reader}"
                rows.append(
                    {
                        "example_id": f"{template}-{world}-{reader}-{nested}",
                        "template_id": template,
                        "split": split,
                        "phase": "prior",
                        "view": view,
                        "messages_text": message,
                        "labels": {"world": world, "reader": reader, "nested": nested},
                    }
                )
                vectors.append([world, reader if view == "omniscient" else 0, nested, 5.0])
        for world in (0, 1):
            for view in module.VIEWS:
                rows.append(
                    {
                        "example_id": f"control-{template}-{world}",
                        "template_id": template,
                        "split": "control",
                        "phase": "delivered_current",
                        "view": view,
                        "messages_text": f"current delivered worldbit{world} nestedbit{world} form {template}",
                        "labels": {"world": world, "reader": world, "nested": world},
                    }
                )
                vectors.append([world, world if view == "omniscient" else 0, world, 5.0])
    base = np.asarray(vectors, dtype=float)
    features = {layer: base.copy() * (i + 1) for i, layer in enumerate(module.LAYERS)}
    return rows, features


def fingerprints(report):
    return {
        (view, target, layer): (
            report["readouts"][view][target]["layers"][layer]["selected_C"],
            report["readouts"][view][target]["layers"][layer]["validation_selection_loss"],
            report["readouts"][view][target]["layers"][layer]["fit_fingerprint"],
            report["readouts"][view][target]["layers"][layer]["train_statistics"]["mean_sha256"],
            report["readouts"][view][target]["layers"][layer]["train_statistics"]["scale_sha256"],
        )
        for view in module.VIEWS
        for target in module.TARGETS
        for layer in module.LAYERS
    }


def test_balanced_factorial_readout_world_copy_and_sender_impossibility():
    rows, features = fixture()
    original = {key: value.copy() for key, value in features.items()}
    report = module.evaluate_belief_readouts(rows, features)
    assert report["primary_layer"] == "layer_16"
    assert report["settings"]["C_grid"] == [0.001, 0.01, 0.1, 1.0, 10.0]
    assert report["design"]["n_prior_factorial_cases"] == 24
    assert report["design"]["n_prior_view_rows"] == 48
    assert report["design"]["n_control_view_rows"] == 12
    for view in module.VIEWS:
        for target in module.TARGETS:
            baselines = report["baselines"][view][target]
            assert baselines["source_world_copy"]["test"]["accuracy"] == (
                1 if target == "world" else 0.5
            )
            assert baselines["inverted_source_world_copy"]["test"]["accuracy"] == (
                0 if target == "world" else 0.5
            )
            assert baselines["constant_zero"]["test"]["accuracy"] == 0.5
            assert baselines["source_world_copy"]["control"]["accuracy"] == 1
        nested = report["primary_results"][view]
        assert nested["role"] == "primary"
        assert nested["metrics"]["test"]["accuracy"] == 1
        assert nested["train_statistics"]["group_stats"]["train"]["n_unique_groups"] == 1
    info = report["sender_reader_impossibility_control"]
    assert info["n_opposite_reader_pairs"] == {"train": 4, "validation": 4, "test": 4}
    for method in (*module.LAYERS, "text"):
        assert info["observed_accuracy"][method] == {"train": 0.5, "validation": 0.5, "test": 0.5}
    for layer in module.LAYERS:
        np.testing.assert_array_equal(features[layer], original[layer])


def test_controls_cannot_fit_scaling_vocabulary_or_classifier(monkeypatch):
    monkeypatch.setattr(module, "C_GRID", (0.01, 1.0))
    rows, features = fixture()
    initial = module.evaluate_belief_readouts(rows, features)
    changed = deepcopy(rows)
    altered = {key: value.copy() for key, value in features.items()}
    for i, row in enumerate(changed):
        if row["split"] == "control":
            row["messages_text"] = "CONTROL_ONLY_SENTINEL " + row["messages_text"]
            for value in altered.values():
                value[i] += 1000000
    original_fit = module.TfidfVectorizer.fit

    def checked_fit(self, raw_documents, *args, **kwargs):
        assert all("CONTROL_ONLY_SENTINEL" not in text for text in raw_documents)
        return original_fit(self, raw_documents, *args, **kwargs)

    monkeypatch.setattr(module.TfidfVectorizer, "fit", checked_fit)
    report = module.evaluate_belief_readouts(changed, altered)
    assert fingerprints(report) == fingerprints(initial)
    for view in module.VIEWS:
        for target in module.TARGETS:
            first, second = (
                initial["readouts"][view][target]["text"],
                report["readouts"][view][target]["text"],
            )
            assert first["fit_fingerprint"] == second["fit_fingerprint"]
            assert first["vocabulary_sha256"] == second["vocabulary_sha256"]


def test_test_labels_do_not_choose_C_layer_or_coefficients(monkeypatch):
    monkeypatch.setattr(module, "C_GRID", (0.01, 1.0))
    rows, features = fixture()
    first = module.evaluate_belief_readouts(rows, features)
    changed = deepcopy(rows)
    for row in changed:
        if row["split"] == "test":
            row["labels"]["nested"] = 1 - row["labels"]["nested"]
    second = module.evaluate_belief_readouts(changed, features)
    assert first["primary_layer"] == second["primary_layer"] == "layer_16"
    assert fingerprints(first) == fingerprints(second)
    assert first["primary_results"]["sender"]["metrics"]["test"]["accuracy"] == 1
    assert second["primary_results"]["sender"]["metrics"]["test"]["accuracy"] == 0
    for view in module.VIEWS:
        for target in module.TARGETS:
            assert (
                first["readouts"][view][target]["text"]["fit_fingerprint"]
                == second["readouts"][view][target]["text"]["fit_fingerprint"]
            )


def test_template_leakage_fails_before_any_fit(monkeypatch):
    rows, features = fixture()
    rows[0]["split"] = "test"

    def no_fit(*args, **kwargs):
        raise AssertionError("Bad partitions must fail before fitting")

    monkeypatch.setattr(module.LogisticRegression, "fit", no_fit)
    with pytest.raises(ValueError, match="Template leakage"):
        module.evaluate_belief_readouts(rows, features)


@pytest.mark.parametrize("corruption", ["messages", "features"])
def test_sender_reader_bit_must_be_hidden_and_identical(corruption):
    rows, features = fixture()
    index = next(
        i
        for i, row in enumerate(rows)
        if row["phase"] == "prior" and row["view"] == "sender" and row["labels"]["reader"] == 1
    )
    if corruption == "messages":
        rows[index]["messages_text"] += " actualreaderbit1"
        message = "messages expose"
    else:
        features["layer_16"][index, 0] += 0.125
        message = "features differ"
    with pytest.raises(ValueError, match=message):
        module.evaluate_belief_readouts(rows, features)


@pytest.mark.parametrize(
    "corruption", ["missing_target", "bad_bit", "bad_control", "bad_phase", "missing_cell"]
)
def test_target_phase_and_factorial_masks_are_explicit(corruption):
    rows, features = fixture()
    if corruption == "missing_target":
        del rows[0]["labels"]["nested"]
        message = "Exactly world/reader/nested"
    elif corruption == "bad_bit":
        rows[0]["labels"]["reader"] = True
        message = "binary integers"
    elif corruption == "bad_control":
        row = next(row for row in rows if row["split"] == "control")
        row["labels"]["nested"] = 1 - row["labels"]["world"]
        message = "all equal world"
    elif corruption == "bad_phase":
        rows[0]["phase"] = "delivered_current"
        message = "Only prior rows"
    else:
        rows.pop(0)
        features = {key: value[1:] for key, value in features.items()}
        message = "full balanced"
    with pytest.raises(ValueError, match=message):
        module.evaluate_belief_readouts(rows, features)


def test_binary_metrics_keep_fixed_classes_and_validate_probabilities():
    metrics = module.binary_metrics([0, 1], [[0.8, 0.2], [0.1, 0.9]])
    assert metrics["accuracy"] == metrics["macro_f1"] == 1
    assert metrics["binary_log_loss"] == pytest.approx(-np.log([0.8, 0.9]).mean())
    assert module.binary_metrics([0], [[1, 0]])["macro_f1"] == 0.5
    with pytest.raises(ValueError, match="Invalid binary probabilities"):
        module.binary_metrics([0], [[0.8, 0.8]])


def test_feature_alignment_and_prespecified_layer_checks():
    rows, features = fixture()
    with pytest.raises(ValueError, match="Primary layer"):
        module.evaluate_belief_readouts(rows, features, primary_layer="layer_test_winner")
    features["layer_8"] = features["layer_8"][:-1]
    with pytest.raises(ValueError, match="alignment/shape"):
        module.evaluate_belief_readouts(rows, features)
