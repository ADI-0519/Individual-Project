"""Synthetic safeguards for the exposed-bank diagnostic; never native-data fits."""

import json
import warnings
from copy import deepcopy
from itertools import product

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from latent_probes import cached_readout_adequacy as cf


@pytest.fixture
def data():
    rows, features = [], []
    for group in range(30):
        gid = f"correction_{group:02d}"
        for i, u, b1, b2, w, m in product((0, 1), repeat=6):
            case_id = f"{gid}__i{i}_u{u}_b{b2}_b1{b1}_w{w}_map{m}"
            state = {
                "group_id": gid,
                "mechanism": "correction",
                "initial": i,
                "update": u,
                "first_receipt": b1,
                "receipt": b2,
                "world": w,
                "name_map": m,
            }
            rows.append(
                {
                    "case_id": case_id,
                    "row_id": case_id,
                    "group_id": gid,
                    "split": "test",
                    "mechanism": "correction",
                    "state": state,
                    "gold": 1 - u if b2 else u if b1 else i,
                }
            )
            features.append([2 * x - 1 for x in (i, u, b1, b2, w, m)])
    result = cf.CorrectionData(tuple(rows), np.array(features, dtype=np.float32), {"fixture": True})
    cf.validate_data(result)
    return result


def test_factor_rule_truth_table_and_ties():
    bits = np.array(list(product((0, 1), repeat=4)))
    assert cf.fold_target(bits).tolist() == [
        1 - u if b2 else u if b1 else i for i, u, b1, b2 in bits
    ]
    # With I=U=0, B1=1 and varying B2, the target is XOR-like, not a receipt label.
    assert cf.fold_target(np.array([[0, 0, 1, 0], [0, 0, 1, 1]])).tolist() == [0, 1]
    assert cf.fold_target((np.zeros((1, 4)) > 0).astype(int)).tolist() == [0]
    with pytest.raises(ValueError):
        cf.fold_target(np.array([[0, 0, 2, 0]]))


def test_complete_group_folds_and_rejects_split_factorial_id_corruption(data):
    folds = cf.make_folds([r["group_id"] for r in data.rows])
    cf.validate_folds(folds, [r["group_id"] for r in data.rows])
    assert len({g for f in folds for g in f["outer_test_groups"]}) == 30
    for fold in folds:
        assert not set(fold["outer_test_groups"]) & set(fold["outer_train_groups"])
        assert set(fold["inner_train_groups"]) | set(fold["inner_dev_groups"]) == set(
            fold["outer_train_groups"]
        )
    damaged = deepcopy(folds)
    damaged[0]["inner_train_groups"][0] = damaged[0]["outer_test_groups"][0]
    with pytest.raises(ValueError, match="membership"):
        cf.validate_folds(damaged, [r["group_id"] for r in data.rows])
    for field, value in (("split", "train"), ("case_id", "wrong")):
        rows = deepcopy(data.rows)
        rows[0][field] = value
        with pytest.raises(ValueError):
            cf.validate_data(cf.CorrectionData(rows, data.features, {}))
    with pytest.raises(ValueError, match="thirty complete"):
        cf.validate_data(cf.CorrectionData(data.rows[:-1], data.features[:-1], {}))


def test_heldout_features_labels_cannot_change_transform_selection_or_heads(
    data, monkeypatch, tmp_path
):
    monkeypatch.setattr(cf, "LR_GRID", (1.0,))
    monkeypatch.setattr(cf, "RBF_C_GRID", (1.0,))
    monkeypatch.setattr(cf, "RBF_GAMMA_GRID", (1.0,))
    fold = cf.make_folds([r["group_id"] for r in data.rows])[0]
    first, arrays = cf.run_fold(data, fold)
    heldout = cf.indices_for(data.rows, fold["outer_test_groups"])
    changed_x, changed_rows = data.features.copy(), deepcopy(data.rows)
    changed_x[heldout] = -100 * changed_x[heldout] + 25
    for index in heldout:
        changed_rows[index]["gold"] = 1 - changed_rows[index]["gold"]
    second, changed_arrays = cf.run_fold(cf.CorrectionData(changed_rows, changed_x, {}), fold)
    for name in arrays:
        np.testing.assert_array_equal(arrays[name], changed_arrays[name])
    for method in cf.METHODS:
        assert (
            first["methods"][method]["selected_parameters"]
            == second["methods"][method]["selected_parameters"]
        )
        assert first["methods"][method]["candidates"] == second["methods"][method]["candidates"]
    # Round-trip safe numeric NPZ and JSON, then independently reproduce margins.
    np.savez(tmp_path / "heads.npz", **arrays)
    (tmp_path / "results.json").write_text(json.dumps(first), encoding="utf-8")
    restored = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    with np.load(tmp_path / "heads.npz", allow_pickle=False) as saved:
        transformed = cf.transform(
            data.features[heldout], {"mean": saved["outer_mean"], "scale": saved["outer_scale"]}
        )
        for method in cf.METHODS:
            local = {
                k.split("__", 1)[1]: saved[k] for k in saved.files if k.startswith(method + "__")
            }
            margins = cf.serialized_margins(
                method, transformed, restored["methods"][method]["head"], local
            )
            scores = restored["methods"][method]["heldout_scores"]
            expected = np.array(
                [
                    s["factor_margins"] if method == "fourfactor_lr" else [s["margin"]]
                    for s in scores
                ]
            )
            np.testing.assert_allclose(margins, expected, atol=1e-9, rtol=1e-9)
    assert all(r["split"] == "test" for r in data.rows)


def test_nonconverged_candidate_rejected_and_all_failure_preserved(data, monkeypatch):
    class BadLR:
        def __init__(self, **kwargs):
            pass

        def fit(self, x, y):
            self.classes_ = np.array([0, 1])
            warnings.warn("fixture nonconvergence", ConvergenceWarning)
            return self

    monkeypatch.setattr(cf, "LogisticRegression", BadLR)
    with pytest.raises(cf.CandidateFailure, match="did not converge"):
        cf.fit_candidate(
            "direct_lr",
            data.features[:64],
            np.array([r["gold"] for r in data.rows[:64]]),
            {"C": 1.0},
        )
    events = []
    with pytest.raises(cf.CandidateFailure, match="No converged") as failure:
        cf.select_method(
            "direct_lr",
            data.features[:64],
            np.array([r["gold"] for r in data.rows[:64]]),
            data.features[64:128],
            data.rows[64:128],
            cf.factor_labels(data.rows[:64]),
            events.append,
        )
    assert len(events) == len(cf.LR_GRID) == len(failure.value.details)
    assert all(event["status"] == "failed" for event in events)


def test_nonfinite_dev_margins_fail_closed(data, monkeypatch):
    class NonfiniteHead:
        def decision_function(self, x):
            return np.full(len(x), np.nan)

    monkeypatch.setattr(cf, "fit_candidate", lambda *args: NonfiniteHead())
    events = []
    with pytest.raises(cf.CandidateFailure, match="No converged"):
        cf.select_method(
            "direct_lr",
            data.features[:64],
            np.array([r["gold"] for r in data.rows[:64]]),
            data.features[64:128],
            data.rows[64:128],
            cf.factor_labels(data.rows[:64]),
            events.append,
        )
    assert all(e["status"] == "failed" and "Nonfinite" in e["error"] for e in events)


def test_selected_outer_refit_failure_is_terminal(data, monkeypatch):
    monkeypatch.setattr(cf, "LR_GRID", (1.0,))
    original = cf.fit_candidate
    calls = []

    def fail_outer(kind, features, labels, parameters):
        calls.append(len(features))
        if len(features) == 1536:
            raise cf.CandidateFailure("fixture outer failure")
        return original(kind, features, labels, parameters)

    monkeypatch.setattr(cf, "fit_candidate", fail_outer)
    with pytest.raises(cf.CandidateFailure, match="outer refit") as failure:
        cf.run_fold(data, cf.make_folds([r["group_id"] for r in data.rows])[0])
    assert calls == [1152, 1536]
    assert failure.value.details["candidates"][0]["status"] == "complete"


def test_source_hash_corruption_rejected_before_loading(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    for name in ("rows.json", "features.npz", "capture_manifest.json"):
        (capture / name).write_bytes(b"fixture")
    hashes = {p.name: cf.file_hash(p) for p in capture.iterdir()}
    (capture / "rows.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        cf.load_data({"capture_directory": str(capture), "capture_sha256": hashes})


def test_protocol_document_source_binding_is_required_and_changes_rejected(tmp_path, monkeypatch):
    assert any(
        p.name == "CACHED_READOUT_ADEQUACY_PROTOCOL_V1.md" for p in cf.required_source_paths()
    )
    source, folds, protocol = (
        tmp_path / "source.py",
        tmp_path / "folds.json",
        tmp_path / "protocol.json",
    )
    source.write_text("frozen", encoding="utf-8")
    folds.write_text(
        json.dumps({"schema": "cached-readout-adequacy-folds-v1", "folds": []}), encoding="utf-8"
    )
    value = {
        "schema": "cached-readout-adequacy-protocol-v1",
        "research_status": cf.STATUS,
        "seed": cf.SEED,
        "source_sha256": {str(source): cf.file_hash(source)},
        "fold_manifest": {"path": str(folds), "sha256": cf.file_hash(folds)},
        "models": [
            {"key": key, "capture_directory": str(tmp_path), "capture_sha256": {}}
            for key in ("llama", "qwen")
        ],
    }
    protocol.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(cf, "required_source_paths", lambda: {source})
    cf.verify_protocol(protocol)
    source.write_text("retuned", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen source"):
        cf.verify_protocol(protocol)


def test_metric_denominators_histogram_and_receipt_pairs(data):
    rows = data.rows[:64]
    scores = [
        {
            "case_id": r["case_id"],
            "prediction": r["gold"],
            "factor_predictions": [r["state"][n] for n in cf.FACTORS],
        }
        for r in rows
    ]
    metrics = cf.summarize(rows, scores, "fourfactor_lr")
    assert metrics["quartet_correct_cells_histogram"] == {"0": 0, "1": 0, "2": 0, "3": 0, "4": 8}
    assert metrics["factor_joint_accuracy"] == metrics["primary_quartet_accuracy"] == 1
    assert metrics["receipt_pair_metrics"]["receipt"]["changing"]["both_correct_fraction"] == 1
    assert metrics["nuisance_verdict_flips"]["world"]["flips"] == 0
    assert metrics["nuisance_verdict_flips"]["name_map"]["pairs"] == 32
    with pytest.raises(ValueError, match="coverage"):
        cf.summarize(rows, scores[:-1], "fourfactor_lr")
