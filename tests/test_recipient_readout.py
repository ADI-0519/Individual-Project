"""Synthetic grouped fixtures test selection, target binding and immutable transfer."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from latent_probes import recipient_readout as rr
from latent_probes.outcome_readout import CLASSES


def write_inputs(root, *, fresh=False, rows=None, vectors=None, policy="a" * 64, signature=None):
    root.mkdir(parents=True, exist_ok=True)
    if rows is None:
        rows, vectors = [], []
        for case in range(24):
            split = "test" if fresh or case >= 20 else "validation" if case >= 16 else "train"
            case_id = f"{'ct' if fresh else 'source'}_{case}"
            states = (
                [(goal, label) for goal in ("neutral", "favorable") for label in range(3)]
                if fresh
                else [("neutral", state % 3) for state in range(9)]
            )
            for state, (goal, label) in enumerate(states):
                rows.append(
                    {
                        "example_id": f"{case_id}_{state}",
                        "case_id": case_id,
                        "family": f"rule_{case % 4}",
                        "objective": goal,
                        "split": split,
                        "decision": CLASSES[label],
                        "prompt_text": f"{goal} source evidence {CLASSES[label].lower()} case {case_id}",
                        "goal_stripped_prompt_text": f"source evidence {CLASSES[label].lower()} case {case_id}",
                        "report_text": f"literal excerpt {CLASSES[label].lower()}",
                    }
                )
                vector = np.zeros(5)
                vector[label] = 2.0 if split == "train" else 200.0
                vector[3], vector[4] = float(case), 7.0
                vectors.append(vector)
    features, labels = root / "features.npz", root / "labels.csv"
    np.savez_compressed(
        features,
        X=np.asarray(vectors, dtype=float),
        example_ids=np.asarray([row["example_id"] for row in rows]),
        feature_signature=np.asarray(
            json.dumps(
                signature
                or {"is_test_fixture": True, "pooling": "last_non_special_report_token", "layer": 8}
            )
        ),
        provenance_json=np.asarray(
            json.dumps(
                {
                    "is_test_fixture": True,
                    "reader_policy_sha256": policy,
                    "label_row_sha256": {
                        row["example_id"]: rr.label_row_sha256(row) for row in rows
                    },
                }
            )
        ),
    )
    with labels.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rr.LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return features, labels, rows, np.asarray(vectors)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def source_fit(tmp_path):
    features, labels, rows, X = write_inputs(tmp_path / "source")
    destination = tmp_path / "fit"
    report = rr.fit_recipient_readouts(features, labels, destination)
    return destination, report, rows, X


def test_train_only_transforms_complete_candidate_records_and_numeric_artifacts(source_fit):
    destination, report, rows, X = source_fit
    assert report["settings"]["seed"] == 42
    assert report["settings"]["C_grid"] == [0.01, 0.1, 1.0, 10.0, 100.0]
    assert report["counts"]["train"]["n_rows"] == 144
    assert report["counts"]["validation"]["n_rows"] == report["counts"]["test"]["n_rows"] == 36
    train = [i for i, row in enumerate(rows) if row["split"] == "train"]
    with np.load(destination / "activation.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["mean"], X[train].mean(axis=0))
        assert not np.allclose(archive["mean"], X.mean(axis=0))
        assert archive["scale"][-1] == 1
    for method in rr.METHODS:
        result = report["methods"][method]
        assert [row["C"] for row in result["validation_candidates"]] == list(rr.C_GRID)
        expected = min(
            result["validation_candidates"],
            key=lambda row: (row["validation"]["multiclass_log_loss"], row["C"]),
        )
        assert result["selected_C"] == expected["C"]
        with np.load(destination / result["artifact"], allow_pickle=False) as archive:
            assert all(archive[key].dtype.kind != "O" for key in archive.files)
            if "vocabulary_terms" in archive.files:
                terms = archive["vocabulary_terms"].tolist()
                assert not any("source_16" in term or "source_20" in term for term in terms)
    with (destination / "predictions.csv").open(newline="", encoding="utf-8") as stream:
        predictions = list(csv.DictReader(stream))
    assert len(predictions) == 216 * 3
    for method in rr.METHODS:
        actual_method, probabilities = rr._score_artifact(destination / f"{method}.npz", X, rows)
        assert actual_method == method
        saved = [row for row in predictions if row["method"] == method]
        expected = np.asarray([[float(row[f"p_{name}"]) for name in CLASSES] for row in saved])
        np.testing.assert_allclose(probabilities, expected, rtol=1e-12, atol=1e-12)


def test_goal_stripped_head_uses_only_the_declared_stripped_context(source_fit):
    destination, _, rows, X = source_fit
    _, original = rr._score_artifact(destination / "goal_stripped_text.npz", X, rows)
    for row in rows:
        row["prompt_text"] = "A completely different reporting objective and private history"
    _, changed = rr._score_artifact(destination / "goal_stripped_text.npz", X, rows)
    np.testing.assert_array_equal(original, changed)


def test_source_test_features_and_labels_cannot_change_selected_fit(tmp_path):
    features, labels, rows, X = write_inputs(tmp_path / "original")
    first = tmp_path / "first"
    rr.fit_recipient_readouts(features, labels, first)
    for i, row in enumerate(rows):
        if row["split"] == "test":
            row["decision"] = CLASSES[(CLASSES.index(row["decision"]) + 1) % 3]
            row["prompt_text"] = "unseen-only-source-test-marker"
            row["report_text"] = "another unseen test marker"
            row["goal_stripped_prompt_text"] = "unseen test context"
            X[i] *= -999
    features, labels, _, _ = write_inputs(tmp_path / "changed", rows=rows, vectors=X)
    second = tmp_path / "second"
    rr.fit_recipient_readouts(features, labels, second)
    for method in rr.METHODS:
        with np.load(first / f"{method}.npz") as a, np.load(second / f"{method}.npz") as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])


@pytest.mark.parametrize(
    "corruption", ["csv_label", "missing_class", "pressure_training", "case_count"]
)
def test_input_or_split_corruption_is_rejected_before_fitting(tmp_path, corruption):
    features, labels, rows, X = write_inputs(tmp_path / "inputs")
    if corruption == "csv_label":
        text = labels.read_text(encoding="utf-8")
        labels.write_text(text.replace(",APPROVE,", ",REJECT,", 1), encoding="utf-8")
    else:
        if corruption == "missing_class":
            for row in rows:
                if row["split"] == "train" and row["decision"] == "DEFER":
                    row["decision"] = "APPROVE"
        elif corruption == "pressure_training":
            rows[0]["objective"] = "favorable"
        else:
            rows[0]["case_id"] = "extra_case"
        features, labels, _, _ = write_inputs(tmp_path / "rebound", rows=rows, vectors=X)
    output = tmp_path / "out"
    with pytest.raises(ValueError):
        rr.fit_recipient_readouts(features, labels, output)
    assert not output.exists()


def test_fresh_evaluation_reconstructs_selected_fits_without_refitting(
    tmp_path, source_fit, monkeypatch
):
    fitted, source, _, _ = source_fit
    before = {path.name: sha(path) for path in fitted.iterdir()}
    features, labels, rows, _ = write_inputs(tmp_path / "fresh", fresh=True)
    monkeypatch.setattr(
        rr, "LogisticRegression", lambda **_: pytest.fail("Frozen evaluation attempted fitting")
    )
    report = rr.evaluate_recipient_readouts(
        features,
        labels,
        fitted,
        tmp_path / "evaluation",
        expected_case_ids={row["case_id"] for row in rows},
    )
    assert report["no_refit"]
    for method in rr.METHODS:
        assert report["methods"][method]["selected_C"] == source["methods"][method]["selected_C"]
        assert report["methods"][method]["evaluations"]["neutral"]["n_valid_rows"] == 72
    assert before == {path.name: sha(path) for path in fitted.iterdir()}


@pytest.mark.parametrize(
    "corruption", ["signature", "reader_policy", "source_overlap", "forbidden_overlap", "hash"]
)
def test_fresh_transfer_guards_preserve_original_fit(tmp_path, source_fit, corruption):
    fitted, _, _, _ = source_fit
    features, labels, rows, X = write_inputs(tmp_path / "fresh", fresh=True)
    forbidden = ()
    if corruption == "signature":
        features, labels, rows, _ = write_inputs(
            tmp_path / "modified", fresh=True, signature={"other_capture": True}
        )
    elif corruption == "reader_policy":
        features, labels, rows, _ = write_inputs(tmp_path / "modified", fresh=True, policy="b" * 64)
    elif corruption == "source_overlap":
        for row in rows:
            if row["case_id"] == "ct_0":
                row["case_id"] = "source_0"
        features, labels, _, _ = write_inputs(tmp_path / "modified", rows=rows, vectors=X)
    elif corruption == "forbidden_overlap":
        forbidden = {"ct_0"}
    else:
        with (fitted / "activation.npz").open("ab") as stream:
            stream.write(b"tamper")
    with pytest.raises(ValueError):
        rr.evaluate_recipient_readouts(
            features,
            labels,
            fitted,
            tmp_path / "evaluation",
            expected_case_ids={row["case_id"] for row in rows},
            forbidden_case_ids=forbidden,
        )
    assert not (tmp_path / "evaluation").exists()


def test_exact_validation_loss_tie_selects_smaller_c(tmp_path, monkeypatch):
    class UniformModel:
        def __init__(self, **kwargs):
            self.c_value = kwargs["C"]

        def fit(self, matrix, labels):
            assert len(labels) == 144
            self.classes_ = np.arange(3)
            self.coef_ = np.zeros((3, matrix.shape[1]))
            self.intercept_ = np.zeros(3)
            self.n_iter_ = np.asarray([1])
            return self

        def predict_proba(self, matrix):
            return np.full((matrix.shape[0], 3), 1 / 3)

    monkeypatch.setattr(rr, "LogisticRegression", UniformModel)
    features, labels, _, _ = write_inputs(tmp_path / "source")
    report = rr.fit_recipient_readouts(features, labels, tmp_path / "fit")
    assert all(method["selected_C"] == 0.01 for method in report["methods"].values())
