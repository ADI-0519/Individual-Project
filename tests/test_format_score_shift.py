"""Meaningful independent checks for POSTHOC score orientation and exact joins."""

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "analyse_format_score_shift.py"
SPEC = spec_from_file_location("format_score_shift", SOURCE)
diagnostic = module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def test_raw_auc_keeps_wrong_orientation_and_exact_ties():
    labels = [0, 0, 1, 1]
    assert diagnostic.raw_auc(labels, [4, 3, 2, 1]) == 0
    assert diagnostic.raw_auc(labels, [1, 2, 3, 4]) == 1
    assert diagnostic.raw_auc(labels, [2, 2, 2, 2]) == 0.5
    assert diagnostic.raw_auc([1, 1], [1, 2]) is None
    with pytest.raises(ValueError, match="Invalid AUROC"):
        diagnostic.raw_auc(labels, [1, 2, np.inf, 4])
    with pytest.raises(ValueError, match="Invalid AUROC"):
        diagnostic.raw_auc([0, 0, 2, 1], [1, 2, 3, 4])


def test_zero_is_negative_and_raw_auc_is_invariant_to_common_offset():
    labels, scores = np.array([0, 1]), np.array([0.0, 1.0])
    assert diagnostic.score_metrics(labels, scores)["accuracy_at_zero"] == 1
    shifted = diagnostic.score_metrics(labels, scores + 9)
    assert shifted["accuracy_at_zero"] == 0.5
    assert shifted["raw_auroc"] == 1


def test_order_transitions_distinguish_jointly_wrong_and_changed_order():
    # Four positive-negative comparisons, two retaining correct order and two reversing.
    counts = diagnostic.order_counts([0, 0, 1, 1], [0, 2, 1, 3], [0, 2, 3, 1])
    assert counts["n_pairs"] == 4
    assert counts["both_correct_order"] == 2
    assert counts["A_only_correct_order"] == 1
    assert counts["B_only_correct_order"] == 1
    assert counts["both_wrong_order"] == 0
    result = diagnostic.summarize_order(counts)
    assert result["non_tied_order_retention"] == 0.5
    assert result["A_within_context_pair_auroc"] == 0.75
    assert result["B_within_context_pair_auroc"] == 0.75
    wrong = diagnostic.summarize_order(diagnostic.order_counts([0, 1], [2, 1], [4, 3]))
    assert wrong["non_tied_order_retention"] == 1
    assert wrong["both_wrong_order_fraction"] == 1
    tied = diagnostic.summarize_order(diagnostic.order_counts([0, 1], [2, 2], [1, 3]))
    assert tied["A_within_context_pair_auroc"] == 0.5
    assert tied["B_within_context_pair_auroc"] == 1
    assert tied["non_tied_order_retention"] is None


def row(renderer, world=0):
    return {
        "row_id": renderer + str(world),
        "renderer": renderer,
        "group_id": "g",
        "gold": 1,
        "condition": "correction_hard",
        "state": {"group_id": "g", "world": world, "name_map": 0, "receipt": 1},
        "messages_sha256": renderer + str(world),
    }


def test_exact_id_alignment_and_formal_pairing_survive_permutation():
    rows = [row("A"), row("B"), row("A", 1), row("B", 1)]
    shuffled = [rows[3], rows[0], rows[1], rows[2]]
    aligned = diagnostic.align_rows(shuffled, [r["row_id"] for r in rows])
    assert aligned == rows
    pairs = diagnostic.ab_pairs(shuffled)
    assert len(pairs) == 2
    for _, a, b in pairs:
        assert shuffled[a]["state"] == shuffled[b]["state"]
        assert shuffled[a]["renderer"] == "A"
        assert shuffled[b]["renderer"] == "B"


def test_joins_reject_duplicates_missing_or_mismatched_states():
    a, b = row("A"), row("B")
    with pytest.raises(ValueError, match="Duplicate metadata"):
        diagnostic.align_rows([a, a], ["A0"])
    with pytest.raises(ValueError, match="Duplicate array"):
        diagnostic.align_rows([a, b], ["A0", "A0"])
    with pytest.raises(ValueError, match="Missing/extra"):
        diagnostic.align_rows([a, b], ["A0"])
    with pytest.raises(ValueError, match="Duplicate formal"):
        diagnostic.ab_pairs([a, a, b])
    broken = deepcopy(b)
    broken["state"]["receipt"] = 0
    with pytest.raises(ValueError, match="Missing formal"):
        diagnostic.ab_pairs([a, broken])
    broken = deepcopy(b)
    broken["gold"] = 0
    with pytest.raises(ValueError, match="target mismatch"):
        diagnostic.ab_pairs([a, broken])


def test_nuisance_pairs_hold_other_state_and_labels_fixed():
    rows = [row("A", 1), row("A", 0)]
    pairs = diagnostic.nuisance_pairs(rows, [0, 1], "world")
    assert pairs == [(1, 0)]
    measured = diagnostic.nuisance_metrics(pairs, np.array([1, 1]), np.array([2.0, -1.0]))
    assert measured["verdict_flip_fraction"] == 1
    assert measured["signed_state1_minus_state0_mean"] == 3
    with pytest.raises(ValueError, match="Nuisance changes target"):
        diagnostic.nuisance_metrics(pairs, np.array([1, 0]), np.array([2.0, -1.0]))


def test_factor_fold_truth_table_and_constant_correlation():
    bits = np.array([[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 0, 1], [1, 0, 1, 1]])
    assert diagnostic.factor_fold(bits).tolist() == [0, 1, 0, 1]
    assert diagnostic.correlation([1, 1], [2, 3]) is None
    assert diagnostic.correlation([1, 2, 2, 4], [8, 6, 6, 2], rank=True) == pytest.approx(-1)


def test_existing_output_is_never_overwritten(tmp_path):
    (tmp_path / "unrelated.txt").write_text("preserve me")
    with pytest.raises(ValueError, match="refuse overwrite"):
        diagnostic.run(tmp_path, tmp_path, tmp_path / "absent_specification.md")
    assert (tmp_path / "unrelated.txt").read_text() == "preserve me"
