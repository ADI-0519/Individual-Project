"""Grouped toy banks verify matched refits and immutable fresh scoring."""

from __future__ import annotations

import csv
from copy import deepcopy

import numpy as np
import pytest

import latent_probes.belief_query_readout as module
from latent_probes.storage import read_json


def bank(*, fresh=False):
    rows, vectors = [], []
    templates = (
        [("new_template", "fresh")]
        if fresh
        else [("old_train", "train"), ("old_val", "validation"), ("old_test", "test")]
    )
    for template, split in templates:
        entities = [
            "person:reader-" + template,
            "person:sender-" + template,
            "place:left-" + template,
            "place:right-" + template,
        ]
        cases = [(w, r, n, "prior", split) for w in (0, 1) for r in (0, 1) for n in (0, 1)]
        if not fresh and split == "train":
            cases += [(w, w, w, "delivered_current", "control") for w in (0, 1)]
        for world, reader, nested, phase, row_split in cases:
            case = f"{template}-{phase}-{world}-{reader}-{nested}"
            for view in module.VIEWS:
                for condition in module.CONDITIONS:
                    text = f"worldbit{world} nestedbit{nested} story {template} {phase} condition {condition}"
                    if view == "omniscient":
                        text += f" actualreaderbit{reader}"
                    rows.append(
                        {
                            "example_id": case + "__" + view,
                            "case_id": case,
                            "template_id": template,
                            "split": row_split,
                            "phase": phase,
                            "view": view,
                            "condition": condition,
                            "messages_text": text,
                            "labels": {"world": world, "reader": reader, "nested": nested},
                            "entity_ids": list(entities),
                        }
                    )
                    actual = reader if view == "omniscient" else 0
                    vector = (
                        [nested, actual, world, 2, 5]
                        if condition == "nested_question"
                        else [world, actual, nested, 4 if condition == "world_question" else 0, 5]
                    )
                    vectors.append(vector)
    base = np.asarray(vectors, dtype=float)
    return rows, {layer: base.copy() * (i + 1) for i, layer in enumerate(module.LAYERS)}


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    rows, features = bank()
    output = tmp_path_factory.mktemp("query_fit") / "fit"
    report = module.fit_query_comparison(rows, features, output)
    return output, report


def test_primary_accessibility_and_fixed_cross_condition_stability_are_distinct(fitted):
    output, report = fitted
    assert report["settings"]["primary_layer"] == "layer_16"
    assert len(read_json(output / "fit_metadata.json")["heads"]) == 72
    for condition in module.CONDITIONS:
        for view in module.VIEWS:
            primary = report["equal_condition"][condition][view]["nested"]["layer_16"]
            assert primary["metrics"]["test"]["accuracy"] == 1
        for layer in (*module.LAYERS, "text"):
            assert (
                report["equal_condition"][condition]["sender"]["reader"][layer]["metrics"]["test"][
                    "accuracy"
                ]
                == 0.5
            )
    selected = next(
        item
        for item in report["cross_condition"]
        if item["source_condition"] == "story"
        and item["evaluation_condition"] == "nested_question"
        and item["view"] == "sender"
        and item["target"] == "nested"
        and item["layer"] == "layer_16"
    )
    assert selected["metrics"]["test"]["accuracy"] == 0.5
    with np.load(output / "parameters.npz", allow_pickle=False) as parameters:
        assert all(parameters[key].dtype.kind in "fiu" for key in parameters.files)


def test_scalers_and_vocabulary_use_train_only_and_test_labels_never_select(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "C_GRID", (0.01, 1.0))
    rows, features = bank()
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    module.fit_query_comparison(rows, features, first_dir)
    changed = deepcopy(rows)
    altered = {key: value.copy() for key, value in features.items()}
    for i, row in enumerate(changed):
        if row["split"] == "test":
            row["labels"]["nested"] ^= 1
        if row["split"] == "control":
            row["messages_text"] += " ONLY_CONTROL_SENTINEL"
            for value in altered.values():
                value[i] += 1000000
    module.fit_query_comparison(changed, altered, second_dir)
    first, second = (
        read_json(first_dir / "fit_metadata.json"),
        read_json(second_dir / "fit_metadata.json"),
    )
    assert first["heads"] == second["heads"]
    assert read_json(first_dir / "text_transforms.json") == read_json(
        second_dir / "text_transforms.json"
    )
    with (
        np.load(first_dir / "parameters.npz", allow_pickle=False) as a,
        np.load(second_dir / "parameters.npz", allow_pickle=False) as b,
    ):
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
        selected = [
            i
            for i, row in enumerate(rows)
            if row["split"] == "train" and row["view"] == "sender" and row["condition"] == "story"
        ]
        np.testing.assert_allclose(
            a["transform__story__sender__layer_16__mean"],
            features["layer_16"][selected].mean(axis=0),
            atol=0,
            rtol=0,
        )


def test_fresh_uses_only_saved_parameters_without_fitting_or_changing_source(
    fitted, tmp_path, monkeypatch
):
    fit_dir, _ = fitted
    before = module._snapshot(fit_dir)

    def no_fit(*args, **kwargs):
        raise AssertionError("Fresh scoring must never fit")

    monkeypatch.setattr(module.LogisticRegression, "fit", no_fit)
    monkeypatch.setattr(module.StandardScaler, "fit", no_fit)
    monkeypatch.setattr(module.TfidfVectorizer, "fit", no_fit)
    rows, features = bank(fresh=True)
    result = module.evaluate_fresh_query_comparison(rows, features, fit_dir, tmp_path / "fresh")
    assert result["no_refit"] and result["no_hyperparameter_selection"]
    assert result["fresh_entity_ids_disjoint"] and result["fitted_artifacts_unchanged"]
    assert module._snapshot(fit_dir) == before
    for condition in module.CONDITIONS:
        assert (
            result["equal_condition"][condition]["sender"]["nested"]["layer_16"]["metrics"][
                "fresh"
            ]["accuracy"]
            == 1
        )
        assert (
            result["equal_condition"][condition]["sender"]["reader"]["layer_16"]["metrics"][
                "fresh"
            ]["accuracy"]
            == 0.5
        )
    changed = deepcopy(rows)
    for row in changed:
        row["labels"]["nested"] ^= 1
    second = module.evaluate_fresh_query_comparison(
        changed, features, fit_dir, tmp_path / "fresh_changed_labels"
    )
    assert (
        second["equal_condition"]["story"]["sender"]["nested"]["layer_16"]["metrics"]["fresh"][
            "accuracy"
        ]
        == 0
    )
    with (
        (tmp_path / "fresh/predictions.csv").open(newline="") as a,
        (tmp_path / "fresh_changed_labels/predictions.csv").open(newline="") as b,
    ):
        for first, changed_row in zip(csv.DictReader(a), csv.DictReader(b), strict=True):
            assert first["p0"] == changed_row["p0"] and first["p1"] == changed_row["p1"]
    assert module._snapshot(fit_dir) == before


@pytest.mark.parametrize("corruption", ["entity", "template", "case", "width"])
def test_fresh_bank_novelty_and_coordinate_width_guards(fitted, tmp_path, corruption):
    fit_dir, _ = fitted
    rows, features = bank(fresh=True)
    if corruption == "entity":
        for row in rows:
            row["entity_ids"][0] = "person:reader-old_train"
        expected = "entity IDs overlap"
    elif corruption == "template":
        for row in rows:
            row["template_id"] = "old_train"
        expected = "template_id overlaps"
    elif corruption == "case":
        old = read_json(fit_dir / "rows.json")[0]["case_id"]
        target = rows[0]["case_id"]
        for row in rows:
            if row["case_id"] == target:
                row["case_id"] = old
                row["example_id"] = old + "__" + row["view"]
        expected = "case_id overlaps"
    else:
        features["layer_16"] = np.column_stack([features["layer_16"], np.zeros(len(rows))])
        expected = "feature width changed"
    with pytest.raises(ValueError, match=expected):
        module.evaluate_fresh_query_comparison(rows, features, fit_dir, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("corruption", ["template_leak", "reader_leak", "missing_condition"])
def test_invalid_legacy_design_fails_before_fitting(tmp_path, monkeypatch, corruption):
    rows, features = bank()
    if corruption == "template_leak":
        for row in rows:
            if row["template_id"] == "old_val":
                row["template_id"] = "old_train"
                row["entity_ids"] = list(rows[0]["entity_ids"])
        expected = "Template leakage"
    elif corruption == "reader_leak":
        index = next(
            i
            for i, row in enumerate(rows)
            if row["view"] == "sender" and row["phase"] == "prior" and row["labels"]["reader"] == 1
        )
        features["layer_16"][index, 0] += 0.125
        expected = "different feature rows"
    else:
        rows.pop(0)
        features = {key: value[1:] for key, value in features.items()}
        expected = "all three conditions"

    def no_fit(*args, **kwargs):
        raise AssertionError("Bad design must fail before fitting")

    monkeypatch.setattr(module.LogisticRegression, "fit", no_fit)
    with pytest.raises(ValueError, match=expected):
        module.fit_query_comparison(rows, features, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_saved_parameter_tampering_prevents_fresh_scoring(fitted, tmp_path):
    import shutil

    copied = tmp_path / "copy"
    shutil.copytree(fitted[0], copied)
    with (copied / "parameters.npz").open("ab") as stream:
        stream.write(b"tampered")
    rows, features = bank(fresh=True)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        module.evaluate_fresh_query_comparison(rows, features, copied, tmp_path / "out")
