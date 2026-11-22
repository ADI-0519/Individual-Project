"""Analytic ceilings and counterexamples, without experimental model outputs."""

from itertools import product

import numpy as np
import pytest

from latent_probes.fresh_readout_analysis import (
    metrics,
    primary_difference,
    strict_first_token,
    verify_aggregate_measurement,
)


def test_strict_first_token_retains_outside_candidates_in_denominator():
    rows = [
        {
            "gold": value,
            "output_logits": {
                "argmax_token_id": token,
                "argmax_tie_count": ties,
                "yes_logprob": -2.0,
                "no_logprob": -3.0,
            },
        }
        for value, token, ties in ((1, 10, 1), (0, 20, 2), (1, 99, 1))
    ]
    result = strict_first_token(rows, {"YES": 10, "NO": 20})
    assert result["rows"] == 3 and result["candidate_first_tokens"] == 2
    assert result["outside_candidate_first_tokens"] == 1
    assert result["accuracy_full_denominator"] == 2 / 3
    assert result["full_vocab_argmax_ties"] == 1


def correction_rows(condition="correction_hard"):
    rows = []
    for u, b1, b2, w, m in product((0, 1), repeat=5):
        i = 1 - u if condition == "correction_hard" else u
        gold = 1 - u if b2 else u if b1 else i
        rows.append(
            {
                "group_id": "analytic_fixture",
                "condition": condition,
                "renderer": "A",
                "gold": gold,
                "state": {
                    "initial": i,
                    "update": u,
                    "first_receipt": b1,
                    "receipt": b2,
                    "world": w,
                    "name_map": m,
                },
            }
        )
    return rows


def test_oracle_joint_panels_and_all_preserving_pairs():
    rows = correction_rows()
    result = metrics(rows, [r["gold"] for r in rows], [2 * r["gold"] - 1 for r in rows])
    assert result["rows"] == 32 and result["correct"] == 32
    assert result["quartets"] == result["complete_quartets"] == 8
    assert result["quartet_accuracy"] == 1
    assert result["quartet_correct_cells_histogram"] == {"0": 0, "1": 0, "2": 0, "3": 0, "4": 8}
    for value in result["nuisance_pairs"].values():
        assert value == {"pairs": 16, "verdict_flips": 0, "both_correct": 16}
    for factor in ("first_receipt", "receipt"):
        assert result["receipt_pairs"][factor]["changing"] == {
            "pairs": 8,
            "verdict_flips": 8,
            "both_correct": 8,
        }
        assert result["receipt_pairs"][factor]["preserving"] == {
            "pairs": 8,
            "verdict_flips": 0,
            "both_correct": 8,
        }


def test_initial_belief_shortcut_gets_three_cells_but_zero_quartets():
    rows = correction_rows()
    result = metrics(rows, [r["state"]["initial"] for r in rows])
    assert result["row_accuracy"] == 0.75
    assert result["quartet_accuracy"] == 0
    assert result["quartet_correct_cells_histogram"]["3"] == 8


def test_easy_panels_keep_their_different_receipt_label_structure():
    rows = correction_rows("correction_easy")
    result = metrics(rows, [r["gold"] for r in rows])
    assert result["receipt_pairs"]["first_receipt"]["preserving"]["pairs"] == 16
    assert result["receipt_pairs"]["first_receipt"]["changing"]["pairs"] == 0
    assert result["receipt_pairs"]["receipt"]["changing"]["pairs"] == 16
    assert result["receipt_pairs"]["receipt"]["preserving"]["pairs"] == 0


def test_world_shortcut_has_complete_nuisance_failure():
    rows = correction_rows()
    result = metrics(rows, [r["state"]["world"] for r in rows])
    assert result["nuisance_pairs"]["world"]["verdict_flips"] == 16
    assert result["quartet_accuracy"] == 0


def test_asserted_states_do_not_invent_receipt_or_quartet_factors():
    rows = [
        {
            "group_id": "fixture",
            "condition": "asserted",
            "renderer": "B",
            "gold": e,
            "state": {"asserted": e, "world": w, "name_map": m},
        }
        for e, w, m in product((0, 1), repeat=3)
    ]
    result = metrics(rows, [r["gold"] for r in rows])
    assert result["quartets"] == 0 and result["quartet_accuracy"] is None
    assert result["receipt_pairs"]["receipt"]["changing"]["pairs"] == 0
    assert result["nuisance_pairs"]["world"]["pairs"] == 4


def test_primary_bootstrap_resamples_contexts_and_has_fixed_policy():
    result = primary_difference({f"g{i}": {"S0": 0, "S1": 1} for i in range(24)})
    assert result["groups"] == 24 and result["quartets"] == 192
    assert result["difference"] == 1 and result["conditional_percentile_interval_95"] == [1, 1]
    assert result["bootstrap_seed"] == 20261122 and result["bootstrap_samples"] == 10000
    with pytest.raises(ValueError):
        primary_difference({f"g{i}": {"S0": 0, "S1": 1} for i in range(23)})
    with pytest.raises(ValueError):
        primary_difference({f"g{i}": {"S0": 0, "S1": 1} for i in range(24)}, samples=500)


@pytest.mark.parametrize("change", ["missing", "duplicate", "nonfinite"])
def test_incomplete_counterparts_and_nonfinite_measurements_fail(change):
    rows = correction_rows()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0]
    with pytest.raises(ValueError):
        metrics(
            rows,
            [r["gold"] for r in rows],
            np.full(len(rows), np.nan) if change == "nonfinite" else None,
        )


@pytest.mark.parametrize("change", ["feature", "output_margin"])
def test_aggregate_cannot_diverge_from_real_journal_measurement(change):
    feature = np.array([[1.0, 2.0]], dtype=np.float32)
    journal = feature.copy()
    row = {"output_yes_no_logitdiff": 2.5}
    complete = {"output_logits": {"yes_minus_no": 2.5}}
    verify_aggregate_measurement(feature, journal, row, complete)
    if change == "feature":
        feature[0, 0] = 3
    else:
        row["output_yes_no_logitdiff"] = -2.5
    with pytest.raises(ValueError):
        verify_aggregate_measurement(feature, journal, row, complete)
