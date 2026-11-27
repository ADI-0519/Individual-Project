"""Synthetic tests of the sealed old-data relevance/oracle diagnostic."""

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyse_logical_relevance_old.py"
SPEC = importlib.util.spec_from_file_location("logical_relevance_old", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def all_bits():
    return np.asarray(list(itertools.product((0, 1), repeat=4)), dtype=np.int64)


def test_fold_matches_event_updates_for_every_truth_tuple():
    bits = all_bits()
    expected = []
    for initial, report, first_read, replacement_read in bits:
        belief = initial
        if first_read:
            belief = report
        if replacement_read:
            belief = 1 - report
        expected.append(belief)
    assert np.array_equal(analysis.fold(bits), expected)


def test_true_factor_relevance_matches_all_closed_form_conditions():
    bits = all_bits()
    initial, update, first, final = bits.T
    expected = np.column_stack(
        (
            (first == 0) & (final == 0),
            (first == 1) | (final == 1),
            (final == 0) & (initial != update),
            (first == 1) | (initial == update),
        )
    )
    relevant = analysis.relevance(bits)
    assert np.array_equal(relevant, expected)
    assert relevant.sum(axis=0).tolist() == [4, 12, 4, 12]
    # This definition holds true others fixed, so changed predicted others can change relevance.
    truth, predicted_others = np.asarray([[0, 1, 0, 0]]), np.asarray([[0, 1, 1, 0]])
    assert analysis.relevance(truth)[0, 0]
    assert not analysis.relevance(predicted_others)[0, 0]


def test_metrics_keep_zero_ties_class0_and_raw_score_orientation():
    tied = analysis.metrics([0, 1], [0.0, 0.0])
    assert tied["zero_accuracy"] == tied["raw_oriented_auc"] == 0.5
    assert tied["class0_recall"] == 1.0 and tied["class1_recall"] == 0.0
    reversed_score = analysis.metrics([0, 0, 1, 1], [2.0, 1.0, -1.0, -2.0])
    assert reversed_score["raw_oriented_auc"] == 0.0
    assert reversed_score["zero_accuracy"] == 0.0


def test_auc_ties_match_independent_sklearn_implementation():
    labels = np.asarray([0, 1, 1, 0, 1, 0, 0, 1])
    scores = np.asarray([-2, -2, 0, 0, 1, 1, 4, 5])
    assert analysis.metrics(labels, scores)["raw_oriented_auc"] == roc_auc_score(labels, scores)


def test_empty_and_one_class_metrics_have_explicit_nulls():
    empty = analysis.metrics([], [])
    assert empty["n"] == 0 and empty["raw_oriented_auc"] is None
    assert empty["auc_null_reason"] == "empty" and empty["zero_accuracy"] is None
    one = analysis.metrics([1, 1], [-1, 1])
    assert one["raw_oriented_auc"] is None and one["auc_null_reason"] == "one_class"
    assert one["class0_recall"] is None and one["class0_recall_null_reason"] == "class_absent"
    assert one["class1_recall"] == one["zero_accuracy"] == 0.5


@pytest.mark.parametrize("labels,scores", [([0], [np.nan]), ([2], [0]), ([0, 1], [1])])
def test_metric_invalid_inputs_rejected(labels, scores):
    with pytest.raises(ValueError):
        analysis.metrics(labels, scores)


def test_exact_join_reorders_and_rejects_duplicate_or_missing_identities():
    rows = [{"row_id": "b", "gold": 1}, {"row_id": "a", "gold": 0}]
    assert [r["gold"] for r in analysis.align_rows(rows, ["a", "b"])] == [0, 1]
    for records, ids in ((rows, ["a", "a"]), (rows, ["a", "c"]), (rows + [rows[0]], ["a", "b"])):
        with pytest.raises(ValueError):
            analysis.align_rows(records, ids)


def correction_rows(group="fresh_test_00", renderer="A"):
    rows = []
    for initial, update, first, final, world, mapping in itertools.product((0, 1), repeat=6):
        bits = np.asarray([[initial, update, first, final]])
        rows.append(
            {
                "row_id": f"{group}_{renderer}_{len(rows)}",
                "group_id": group,
                "renderer": renderer,
                "partition": "test",
                "condition": "correction_easy" if initial == update else "correction_hard",
                "state": {
                    "initial": initial,
                    "update": update,
                    "first_receipt": first,
                    "receipt": final,
                    "world": world,
                    "name_map": mapping,
                },
                "gold": int(analysis.fold(bits)[0]),
            }
        )
    return rows


def test_receipt_quartets_preserve_all_four_cells_and_reject_missing_or_duplicate():
    rows = correction_rows()
    quartets = analysis.quartet_indices(rows, np.arange(len(rows)))
    assert quartets.shape == (16, 4)
    assert sorted(quartets.ravel().tolist()) == list(range(64))
    with pytest.raises(ValueError, match="Incomplete"):
        analysis.quartet_indices(rows, np.arange(63))
    with pytest.raises(ValueError, match="Duplicate"):
        analysis.quartet_indices(rows, np.asarray([*range(64), 0]))


def test_all_sixteen_masks_substitute_only_selected_true_bits():
    truth = all_bits()
    prediction = 1 - truth
    for mask in range(16):
        replaced = analysis.substitute(prediction, truth, mask)
        for column in range(4):
            assert np.array_equal(
                replaced[:, column],
                truth[:, column] if mask & (1 << column) else prediction[:, column],
            )
    assert np.array_equal(
        analysis.fold(analysis.substitute(prediction, truth, 0)), analysis.fold(prediction)
    )
    assert np.array_equal(
        analysis.fold(analysis.substitute(prediction, truth, 15)), analysis.fold(truth)
    )
    assert np.array_equal(prediction, 1 - truth)
    with pytest.raises(ValueError):
        analysis.substitute(prediction, truth, 16)


def full_rows():
    rows = []
    for context in range(24):
        group = f"fresh_test_{context:02d}"
        for renderer in ("A", "B"):
            rows.extend(correction_rows(group, renderer))
            for initial, update, receipt, world, mapping in itertools.product((0, 1), repeat=5):
                rows.append(
                    {
                        "row_id": f"one_{group}_{renderer}_{len(rows)}",
                        "group_id": group,
                        "renderer": renderer,
                        "partition": "test",
                        "condition": "one_step",
                        "state": {
                            "initial": initial,
                            "update": update,
                            "receipt": receipt,
                            "world": world,
                            "name_map": mapping,
                        },
                        "gold": update if receipt else initial,
                    }
                )
            for asserted, world, mapping in itertools.product((0, 1), repeat=3):
                rows.append(
                    {
                        "row_id": f"asserted_{group}_{renderer}_{len(rows)}",
                        "group_id": group,
                        "renderer": renderer,
                        "partition": "test",
                        "condition": "asserted",
                        "state": {"asserted": asserted, "world": world, "name_map": mapping},
                        "gold": asserted,
                    }
                )
    return rows


def test_all_old_contexts_and_factorial_cells_required():
    rows = full_rows()
    analysis.validate_rows(rows)
    changed = [*rows]
    changed[0] = {**changed[0], "gold": 1 - changed[0]["gold"]}
    with pytest.raises(ValueError, match="Formal target"):
        analysis.validate_rows(changed)
    changed = [*rows]
    changed[0] = changed[1]
    with pytest.raises(ValueError, match="Duplicate factorial"):
        analysis.validate_rows(changed)


def test_complete_report_retains_every_population_mask_and_context():
    rows = full_rows()
    prediction = np.zeros((4992, 4), dtype=np.int64)
    scores = prediction.astype(float)
    factors, comparisons, oracles = analysis.analyze_model("synthetic", rows, scores, prediction)
    assert len(factors) == 1200 and len(comparisons) == 600 and len(oracles) == 2400
    assert {r["factor"] for r in factors} == set(analysis.FACTORS)
    assert {r["relevance"] for r in factors} == {"relevant", "irrelevant"}
    assert {r["stratum"] for r in factors} == set(analysis.STRATA)
    assert {r["mask"] for r in oracles} == set(range(16))
    assert len({r["context"] for r in oracles}) == 25
    for record in oracles:
        assert sum(record[f"quartets_with_{k}_correct"] for k in range(5)) == record["quartets"]
        if record["mask"] == 15:
            assert record["row_accuracy"] == record["quartet_all_correct"] == 1.0
    no_relevance = [
        r
        for r in factors
        if r["factor"] == "B1"
        and r["stratum"] == "correction_easy"
        and r["relevance"] == "relevant"
    ]
    assert no_relevance and all(r["n"] == 0 and r["raw_oriented_auc"] is None for r in no_relevance)
