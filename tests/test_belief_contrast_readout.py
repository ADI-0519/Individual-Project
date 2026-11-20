"""Synthetic paired claims test linearity, isolation and immutable transfer."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from latent_probes import belief_contrast_readout as contrast
from latent_probes import belief_query_readout as query
from latent_probes.storage import read_json


def old_bank():
    rows, vectors = [], []
    for split in ("train", "validation", "test"):
        template = "old_" + split
        entities = [
            "person:r-" + template,
            "person:s-" + template,
            "place:a-" + template,
            "place:b-" + template,
        ]
        for w in (0, 1):
            for r in (0, 1):
                for n in (0, 1):
                    case = f"{template}-w{w}r{r}n{n}"
                    for view in query.VIEWS:
                        for condition in query.CONDITIONS:
                            text = (
                                f"worldbit{w} nestedbit{n} {template} {condition} candidate placeB"
                            )
                            if view == "omniscient":
                                text += f" actualreaderbit{r}"
                            rows.append(
                                {
                                    "example_id": case + "__" + view,
                                    "case_id": case,
                                    "template_id": template,
                                    "split": split,
                                    "phase": "prior",
                                    "view": view,
                                    "condition": condition,
                                    "messages_text": text,
                                    "entity_ids": list(entities),
                                    "labels": {"world": w, "reader": r, "nested": n},
                                }
                            )
                            vectors.append([w, r if view == "omniscient" else 0, n, 1, 5])
    X = np.asarray(vectors, dtype=float)
    return rows, {layer: X.copy() * (i + 1) for i, layer in enumerate(query.LAYERS)}


def paired_bank(*, fresh=False):
    old_rows, old_features = old_bank()
    rows, indices, claims = [], [], []
    for i, original in enumerate(old_rows):
        if original["condition"] not in contrast.CONDITIONS or (
            fresh and original["split"] != "test"
        ):
            continue
        for claim in (0, 1):
            row = deepcopy(original)
            if fresh:
                row["case_id"] = row["case_id"].replace("old_test", "fresh_family")
                row["template_id"] = "fresh_family"
                row["split"] = "fresh"
                row["entity_ids"] = [
                    x.replace("old_test", "fresh_family") for x in row["entity_ids"]
                ]
                row["messages_text"] = row["messages_text"].replace("old_test", "fresh_family")
            row["channel"] = "new_family" if fresh else "legacy"
            row["claim_bit"] = claim
            row["example_id"] = "__".join(
                (row["case_id"], row["view"], row["condition"], str(claim))
            )
            row["messages_text"] = row["messages_text"].replace(
                "placeB", "placeB" if claim else "placeA"
            )
            row["candidate_text"] = "token at " + row["template_id"] + (" B" if claim else " A")
            if not claim:
                row["labels"] = {target: 1 - value for target, value in row["labels"].items()}
            rows.append(row)
            indices.append(i)
            claims.append(claim)
    features = {layer: old_features[layer][indices].copy() for layer in query.LAYERS}
    for j, row in enumerate(rows):
        if not claims[j]:
            for layer_index, layer in enumerate(query.LAYERS):
                factor = layer_index + 1
                features[layer][j, 0] = factor * row["labels"]["world"]
                features[layer][j, 1] = (
                    factor * row["labels"]["reader"] if row["view"] == "omniscient" else 0
                )
                features[layer][j, 2] = factor * row["labels"]["nested"]
                features[layer][j, 3] = 0
    return rows, features


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    root = tmp_path_factory.mktemp("contrast_fit")
    rows, features = old_bank()
    original = root / "original"
    query.fit_query_comparison(rows, features, original)
    paired_rows, paired_features = paired_bank()
    output = root / "paired"
    report = contrast.fit_contrast_readouts(paired_rows, paired_features, output, original)
    return original, output, report


def test_pair_endpoint_counts_and_saved_models_are_numeric(fitted):
    original, output, report = fitted
    assert report["settings"]["primary_endpoints"] == ["original_relative", "paired_linear"]
    assert "NO per-row" in report["settings"]["paired_transform"]
    metrics = report["equal_condition"]["story"]["sender"]["nested"]["paired_linear"]["metrics"][
        "legacy"
    ]["test"]
    assert metrics["n_rows"] == 8 and metrics["accuracy"] == 1
    for target in contrast.TARGETS:
        for method in contrast.METHODS:
            metric = report["equal_condition"]["story"]["sender"][target][method]["metrics"][
                "legacy"
            ]["test"]
            assert metric["n_rows"] == 8  # A never doubles the number of observations.
            if target == "reader":
                assert metric["accuracy"] == 0.5
    checks = report["candidate_only_control"]
    assert checks["literal_text_supplied_for_all_rows"]
    assert all(
        all(
            value == 0.5
            for value in item["candidate_only_best_deterministic_accuracy_bound"].values()
        )
        for item in checks["checks"]
    )
    with np.load(output / "parameters.npz", allow_pickle=False) as archive:
        assert all(archive[key].dtype.kind in "fiu" for key in archive.files)
        assert all(
            np.array_equal(archive[key], [0])
            for key in archive.files
            if "paired_linear" in key and key.endswith("__intercept")
        )
    assert query._snapshot(original)


def test_primary_pair_is_strictly_linear_without_difference_normalization():
    head = {"method": "paired_linear", "parameter_prefix": "p", "transform": "z"}
    params = {
        "z__mean": np.asarray([20.0, 30.0]),
        "z__scale": np.asarray([1.0, 1.0]),
        "p__coef": np.asarray([[1.0, 0.0]]),
        "p__intercept": np.asarray([0.0]),
    }
    A = np.asarray([[1.0, 2.0], [1.0, 2.0]])
    B = np.asarray([[2.0, 2.0], [3.0, 2.0]])
    probability, detail = contrast._predict(head, params, {}, A, B, ["", ""], ["", ""])
    np.testing.assert_array_equal(detail[0], [1, 2])
    reverse, reverse_detail = contrast._predict(head, params, {}, B, A, ["", ""], ["", ""])
    np.testing.assert_allclose(reverse[:, 1], 1 - probability[:, 1], atol=1e-15)
    np.testing.assert_array_equal(reverse_detail[0], -detail[0])


def test_train_only_scaling_and_heldout_features_cannot_change_fitted_parameters(fitted, tmp_path):
    original, fit_dir, _ = fitted
    rows, features = paired_bank()
    for layer in features:
        features[layer][[i for i, row in enumerate(rows) if row["split"] == "test"]] += 1000
    changed = tmp_path / "changed_test_features"
    contrast.fit_contrast_readouts(rows, features, changed, original)
    with (
        np.load(fit_dir / "parameters.npz", allow_pickle=False) as before,
        np.load(changed / "parameters.npz", allow_pickle=False) as after,
    ):
        assert before.files == after.files
        for key in before.files:
            np.testing.assert_array_equal(before[key], after[key])
        for condition in contrast.CONDITIONS:
            for view in contrast.VIEWS:
                training = np.asarray(
                    [
                        i
                        for i, row in enumerate(rows)
                        if row["split"] == "train"
                        and row["condition"] == condition
                        and row["view"] == view
                    ]
                )
                X = features[contrast.LAYER][training]
                prefix = "new__" + condition + "__" + view
                np.testing.assert_allclose(before[prefix + "__mean"], X.mean(axis=0), atol=0)
                expected_scale = X.std(axis=0)
                expected_scale[expected_scale == 0] = 1
                np.testing.assert_allclose(before[prefix + "__scale"], expected_scale, atol=0)


def test_fresh_no_refits_originals_unchanged_and_labels_do_not_change_predictions(
    fitted, tmp_path, monkeypatch
):
    original, fit_dir, _ = fitted
    before, old_before = query._snapshot(fit_dir), query._snapshot(original)
    rows, features = paired_bank(fresh=True)

    def no_fit(*args, **kwargs):
        raise AssertionError("Fresh scoring cannot fit")

    monkeypatch.setattr(contrast.LogisticRegression, "fit", no_fit)
    monkeypatch.setattr(contrast.StandardScaler, "fit", no_fit)
    monkeypatch.setattr(query.TfidfVectorizer, "fit", no_fit)
    report = contrast.evaluate_contrast_readouts(rows, features, fit_dir, tmp_path / "fresh")
    assert report["no_refit"] and report["fitted_artifacts_unchanged"]
    primary = report["equal_condition"]["story"]["sender"]["nested"]["paired_linear"]["metrics"][
        "new_family"
    ]["fresh"]
    assert primary["accuracy"] == 1 and primary["binary_log_loss"] < np.log(2)
    assert primary["n_rows"] == 8
    assert query._snapshot(fit_dir) == before and query._snapshot(original) == old_before
    changed = deepcopy(rows)
    for row in changed:
        row["labels"]["nested"] ^= 1
    second = contrast.evaluate_contrast_readouts(changed, features, fit_dir, tmp_path / "changed")
    assert (
        second["equal_condition"]["story"]["sender"]["nested"]["paired_linear"]["metrics"][
            "new_family"
        ]["fresh"]["accuracy"]
        == 0
    )
    import csv

    with (
        (tmp_path / "fresh/predictions.csv").open(newline="") as a,
        (tmp_path / "changed/predictions.csv").open(newline="") as b,
    ):
        for x, y in zip(csv.DictReader(a), csv.DictReader(b), strict=True):
            assert x["margin"] == y["margin"] and x["p_B"] == y["p_B"]


def test_invalid_complements_or_exposed_private_reader_fail_before_fit(
    fitted, tmp_path, monkeypatch
):
    original, _, _ = fitted
    rows, features = paired_bank()
    rows[0]["labels"]["nested"] = rows[1]["labels"]["nested"]
    with pytest.raises(ValueError, match="not complements"):
        contrast.fit_contrast_readouts(rows, features, tmp_path / "bad", original)
    rows, features = paired_bank()
    i = next(
        i
        for i, row in enumerate(rows)
        if row["view"] == "sender" and row["claim_bit"] == 1 and row["labels"]["reader"] == 1
    )
    features["layer_16"][i, 0] += 0.125
    with pytest.raises(ValueError, match="exposes private"):
        contrast.fit_contrast_readouts(rows, features, tmp_path / "bad2", original)


def test_post_message_keeps_hidden_reader_impossibility_and_exposure_guard(fitted, tmp_path):
    _, fit_dir, _ = fitted
    rows, features = paired_bank(fresh=True)
    for row in rows:
        row["phase"] = "post_message"
    result = contrast.evaluate_contrast_readouts(rows, features, fit_dir, tmp_path / "post")
    metrics = result["equal_condition"]["story"]["sender"]["reader"]["paired_linear"]["metrics"][
        "new_family"
    ]["fresh"]
    assert metrics["n_rows"] == 8 and metrics["accuracy"] == 0.5
    index = next(
        i
        for i, row in enumerate(rows)
        if row["view"] == "sender" and row["claim_bit"] == 1 and row["labels"]["reader"] == 1
    )
    features["layer_16"][index, 0] += 0.125
    with pytest.raises(ValueError, match="exposes private"):
        contrast.evaluate_contrast_readouts(rows, features, fit_dir, tmp_path / "post_exposed")
    assert not (tmp_path / "post_exposed").exists()


@pytest.mark.parametrize("kind", ["entity", "case", "width"])
def test_fresh_disjointness_and_coordinate_guards(fitted, tmp_path, kind):
    _, fit_dir, _ = fitted
    rows, features = paired_bank(fresh=True)
    if kind == "entity":
        for row in rows:
            row["entity_ids"][0] = "person:r-old_train"
        message = "entity IDs overlap"
    elif kind == "case":
        old = read_json(fit_dir / "rows.json")[0]["case_id"]
        selected = rows[0]["case_id"]
        for row in rows:
            if row["case_id"] == selected:
                row["case_id"] = old
                row["example_id"] = "__".join(
                    (old, row["view"], row["condition"], str(row["claim_bit"]))
                )
        message = "case_id overlaps"
    else:
        features["layer_16"] = np.column_stack([features["layer_16"], np.zeros(len(rows))])
        message = "feature width changed"
    with pytest.raises(ValueError, match=message):
        contrast.evaluate_contrast_readouts(rows, features, fit_dir, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
