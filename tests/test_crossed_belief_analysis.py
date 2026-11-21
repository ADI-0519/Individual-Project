"""Meaningful shortcut, pairing, raw-logit and failure-preservation checks."""

import copy

import numpy as np
import pytest

from latent_probes.crossed_belief_analysis import (
    HEAD,
    auc,
    condition_analysis,
    counterfactual_pairs,
    hidden_reader_control,
    reconstruct_logits,
    reconstruct_probe,
    verify_answer,
)
from latent_probes.crossed_belief_bank import build_bank


def measured_rows(condition="nested_question"):
    rows = copy.deepcopy(build_bank()["sender_inputs"])
    for row in rows:
        row["sender_input_id"] = row["id"]
        row["id"] += "__" + condition
        row["condition"] = condition
        margin = float(2 * row["gold"] - 1)
        row["probe"] = {
            "margin": margin,
            "probability": 0.75 if row["gold"] else 0.25,
            "prediction": row["gold"],
        }
        row["output_logits"] = {"yes_minus_no": margin}
        row["answer"] = {"prediction": row["gold"]} if condition != "story" else None
        row["generation"] = (
            {"text": "YES" if row["gold"] else "NO"} if condition != "story" else None
        )
    return rows


def test_full_factorial_shortcuts_and_pair_denominators():
    result = condition_analysis(measured_rows())
    assert result["primary"]["n"] == 64
    assert result["primary"]["mapping_paired_cells"] == 32
    assert set(result["primary"]["shortcut_correct"].values()) == {32}
    assert result["overall"]["shortcut_correct"]["initial_slot"] == 96
    assert result["overall"]["shortcut_correct"]["message_slot"] == 96
    assert result["pairs"]["receipt_primary"]["n_pairs"] == 32
    assert result["pairs"]["receipt_invariant"]["n_pairs"] == 32
    assert result["pairs"]["world_all"]["n_pairs"] == 64
    assert result["pairs"]["world_primary"]["n_pairs"] == 32
    assert result["pairs"]["mapping_all"]["n_pairs"] == 64
    assert result["probe_qualification"]["passed"]
    assert result["output_ranking_qualification"]["passed"]
    assert result["behavioral_qualification"]["passed"]


def test_old_complement_initial_shortcut_cannot_qualify_new_instrument():
    rows = measured_rows()
    for row in rows:
        source = row["source"]
        old_rule = (
            source["message_slot"] if source["receipt_reports_read"] else 1 - source["message_slot"]
        )
        row["probe"] = {
            "margin": float(2 * old_rule - 1),
            "probability": 0.75 if old_rule else 0.25,
            "prediction": old_rule,
        }
    result = condition_analysis(rows)
    assert result["primary"]["probe_correct"] == 64
    assert result["invariant"]["probe_correct"] == 32
    assert not result["probe_qualification"]["passed"]
    assert any("invariant" in item for item in result["probe_qualification"]["failed_checks"])


def test_likelihood_success_does_not_rescue_constant_greedy_no():
    rows = measured_rows()
    for row in rows:
        row["answer"]["prediction"] = 0
        row["generation"]["text"] = "NO"
    result = condition_analysis(rows)
    assert result["output_ranking_qualification"]["passed"]
    assert not result["behavioral_qualification"]["passed"]
    assert result["primary"]["answer_correct"] == 32


def test_single_wording_failure_is_not_hidden_by_overall_accuracy():
    rows = measured_rows()
    for row in rows:
        if row["template_id"] == "cb_t01":
            row["probe"]["prediction"] = 0
    result = condition_analysis(rows)
    assert result["primary"]["probe_correct"] == 56
    assert not result["probe_qualification"]["passed"]
    assert any("template/cb_t01" in item for item in result["probe_qualification"]["failed_checks"])


def test_unqueried_logits_do_not_get_answer_or_ranking_qualification():
    result = condition_analysis(measured_rows("story"))
    assert "output_score_auroc" not in result["primary"]
    assert "behavioral_qualification" not in result
    assert "output_ranking_qualification" not in result


def test_missing_counterfactual_pair_is_rejected():
    with pytest.raises(ValueError, match="Incomplete counterfactual"):
        counterfactual_pairs(measured_rows()[1:], "world_slot")


def test_hidden_delivery_pairing_is_exact_chance_even_with_perfect_nested_head():
    rows = [r for r in measured_rows() if r["stratum"] == "receipt_sensitive"]
    result = hidden_reader_control(rows)
    assert result == {
        "fictional_rows": 128,
        "unique_sender_inputs": 64,
        "accuracy": 0.5,
        "auroc": 0.5,
    }
    with pytest.raises(ValueError, match="only to I != M"):
        hidden_reader_control(measured_rows())


def test_raw_full_vocabulary_uses_argmax_and_stable_normalizer():
    values = np.array([1000.0, 999.0, -1000.0, 1001.0], dtype=np.float32)
    result = reconstruct_logits(values, {"YES": 0, "NO": 1})
    assert result["argmax_token_id"] == 3
    assert result["yes_minus_no"] == 1.0
    assert result["full_vocab_logsumexp"] == pytest.approx(1001 + np.log1p(np.exp(-1) + np.exp(-2)))
    assert result["p_yes_given_yes_no"] == pytest.approx(1 / (1 + np.exp(-1)))
    with pytest.raises(ValueError, match="float32"):
        reconstruct_logits(values.astype(np.float64), {"YES": 0, "NO": 1})


def test_scaler_then_l2_normalization_and_tie_zero():
    metadata = {
        "feature_widths": {"layer_16": 2},
        "heads": {
            HEAD: {"layer": "layer_16", "activation_transform": "a", "parameter_prefix": "p"}
        },
    }
    parameters = {
        "a__mean": np.array([1.0, 3.0]),
        "a__scale": np.array([2.0, 4.0]),
        "p__coef": np.array([[3.0, 4.0]]),
        "p__intercept": np.array([-5.0]),
    }
    result = reconstruct_probe(np.array([7.0, 19.0]), metadata, parameters)
    assert result["margin"] == pytest.approx(0.0)
    assert result["probability"] == 0.5
    assert result["prediction"] == 0
    with pytest.raises(ValueError, match="Invalid saved feature"):
        reconstruct_probe(np.array([7.0, np.nan]), metadata, parameters)


def test_auc_ties_count_half_and_missing_class_rejected():
    assert auc([0, 1, 0, 1], [0.0, 0.0, 1.0, 1.0]) == 0.5
    assert auc([0, 1], [0.0, 1.0]) == 1.0
    with pytest.raises(ValueError, match="both classes"):
        auc([0, 0], [1.0, 2.0])


def test_answer_token_binding_cannot_be_replaced_by_recorded_text():
    protocol = {
        "generation": {"max_new_tokens": 12, "eos_token_ids": [9]},
        "tokenizer": {"all_special_ids": [8, 9]},
    }
    row = {
        "condition": "nested_question",
        "prompt_token_ids": [1, 2],
        "output_logits": {"argmax_token_id": 4},
        "generation": {
            "prompt_token_ids": [1, 2],
            "generated_token_ids": [4, 9],
            "finish_reason": "eos",
            "text": "NO",
        },
        "answer": {"prediction": 0, "capped": False, "unexpected_special": False},
    }
    verify_answer(row, protocol)
    row["generation"]["generated_token_ids"][0] = 5
    with pytest.raises(ValueError, match="first token"):
        verify_answer(row, protocol)
