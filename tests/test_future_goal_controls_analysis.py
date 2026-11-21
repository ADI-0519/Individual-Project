"""Fabricated records test estimands and stopping; they are not model evidence."""

from copy import deepcopy

import numpy as np
import pytest

from latent_probes import future_goal_controls_analysis as analysis


def records():
    rows = deepcopy(analysis.reviewed_bank()["planned_conditions"])
    shifts = {
        "neutral": 0,
        "own_token_a": -1,
        "own_token_b": 1,
        "archive_token_a": -0.25,
        "archive_token_b": 0.25,
        "own_disk_a": -0.5,
        "own_disk_b": 0.5,
    }
    for row in rows:
        margin = (3 if row["gold"] else -3) + shifts[row["condition"]]
        row["probe"] = {
            "head_id": analysis.HEAD,
            "margin": margin,
            "probability": float(1 / (1 + np.exp(-margin))),
            "prediction": int(margin > 0),
        }
        row["generation"] = {"text": "YES." if row["gold"] else "NO!", "finish_reason": "eos"}
    return rows


def test_mapping_offsets_cancel_and_group_bootstrap_preserves_scene_units():
    rows = records()
    # Large offsets differ by scene, mapping, and control family. Within-family
    # b-minus-a subtraction must remove them before either mapping is averaged.
    for row in rows:
        if row["condition"] == "neutral":
            continue
        family = (
            50
            if row["condition"].startswith("archive")
            else -20
            if row["condition"].startswith("own_disk")
            else 90
        )
        offset = family + row["mapping_id"] * 100 + int(row["scene_id"][-2:])
        margin = row["probe"]["margin"] + offset
        row["probe"].update(
            margin=margin,
            probability=float(np.exp(-np.logaddexp(0, -margin))),
            prediction=int(margin > 0),
        )
    result = analysis.analyse_rows(rows)
    assert result["qualification"]["passed"]
    assert result["primary"]["n_scenes"] == 16
    assert result["primary"]["mean"] == pytest.approx(1.5)
    assert result["secondary"]["mean"] == pytest.approx(1)
    assert result["primary"]["channel_means"] == {"observation": 1.5, "message": 1.5}
    assert result["primary"]["percentile_95_interval"] == pytest.approx([1.5, 1.5])
    assert all(
        v["mean"] == pytest.approx(1.5)
        for v in result["primary"]["leave_one_wording_group_out"].values()
    )
    assert all(s["primary"] == pytest.approx(1.5) for s in result["per_scene"])
    assert result["specificity"]["primary"]["n_scenes"] == 16


def test_failed_neutral_gate_never_emits_nonneutral_estimates():
    rows = records()
    for row in [r for r in rows if r["condition"] == "neutral"][:9]:
        row["generation"] = {"text": "NO" if row["gold"] else "YES", "finish_reason": "eos"}
    result = analysis.analyse_rows(rows)
    assert result["status"] == "qualification_failed"
    assert result["qualification"]["overall"]["answer_correct"] == 23
    assert result["nonneutral_conditions_present_after_failed_gate"] == 192
    assert (
        not {"primary", "secondary", "per_scene", "raw_scores", "metrics", "specificity"}
        & result.keys()
    )


def test_capped_parseable_answer_is_invalid_and_specificity_is_separate():
    rows = records()
    target = next(r for r in rows if r["condition"] == "own_disk_a")
    target["generation"]["finish_reason"] = "length"
    neutral = next(r for r in rows if r["condition"] == "neutral")
    neutral["generation"]["finish_reason"] = "length"
    result = analysis.analyse_rows(rows)
    assert result["qualification"]["overall"]["answer_correct"] == 31
    assert result["qualification"]["overall"]["invalid"] == 1
    assert result["specificity"]["primary"]["n_scenes"] == 16
    assert result["specificity"]["secondary"]["n_scenes"] == 15
    assert result["specificity"]["joint_primary"]["n_scenes"] == 15
    assert result["specificity"]["primary"]["required_correct_answers_per_scene"] == 8
    assert result["specificity"]["joint_primary"]["required_correct_answers_per_scene"] == 12


def test_duplicate_and_missing_ids_and_bank_metadata_are_rejected():
    rows = records()
    with pytest.raises(ValueError, match="Duplicate condition ID"):
        analysis.analyse_rows([*rows, deepcopy(rows[0])])
    with pytest.raises(ValueError, match="all 32 exact neutral"):
        analysis.analyse_rows(rows[1:])
    rows[0]["gold"] = 1 - rows[0]["gold"]
    with pytest.raises(ValueError, match="differs from bank"):
        analysis.analyse_rows(rows)


@pytest.mark.parametrize("text", ["YES NO", "YES?", "YES!!", "YES because", "1", ""])
def test_strict_answer_parser_rejects_explanations_and_extra_punctuation(text):
    rows = records()
    rows[0]["generation"]["text"] = text
    result = analysis.qualify_neutral(rows)
    assert result["overall"]["invalid"] == 1
    assert result["overall"]["answer_correct"] == 31


def test_subgroup_floors_are_checked_even_when_overall_passes():
    rows = records()
    neutral = [r for r in rows if r["condition"] == "neutral" and r["mapping_id"] == 0]
    for row in neutral[:5]:
        row["generation"]["finish_reason"] = "length"
    result = analysis.qualify_neutral(rows)
    assert result["overall"]["answer_correct"] == 27
    assert not result["passed"]
    assert any("mapping/0/answer_correct" in check for check in result["failed_checks"])


def test_exact_half_probability_preserves_source_argmax_tie_zero():
    rows = records()
    rows[0]["probe"].update(margin=0.0, probability=0.5, prediction=0)
    assert analysis.qualify_neutral(rows)["overall"]["probe_correct"] == 31
    rows[0]["probe"]["prediction"] = 1
    with pytest.raises(ValueError, match="threshold differs"):
        analysis.qualify_neutral(rows)
    rows[0]["probe"].update(margin=float(np.nextafter(0.0, 1.0)), probability=0.5, prediction=0)
    assert analysis.qualify_neutral(rows)["overall"]["probe_correct"] == 31


def test_wording_pair_is_resampled_intact_and_ranking_survives_threshold_shift():
    rows = records()
    for row in rows:
        if row["condition"] == "own_token_b":
            margin = row["probe"]["margin"] + (5 if row["gold"] else -5) + 20
            row["probe"].update(
                margin=margin,
                probability=float(np.exp(-np.logaddexp(0, -margin))),
                prediction=int(margin > 0),
            )
    result = analysis.analyse_rows(rows)
    # Each wording group contains both opposing labels; resampling a whole group
    # cancels its +/-5 component, whereas an incorrect scene bootstrap would vary.
    assert {s["primary"] for s in result["per_scene"]} == {16.5, 26.5}
    assert result["primary"]["percentile_95_interval"] == [21.5, 21.5]
    metrics = result["metrics"]["per_condition"]["own_token_b"]
    assert metrics["probe_accuracy"] == 0.5
    assert metrics["score_auroc"] == 1


def test_saved_generation_requires_prefix_hash_text_and_special_disposition():
    protocol = {"generation": {"eos_token_ids": [99]}, "tokenizer": {"all_special_ids": [98, 99]}}
    saved = {
        "prompt_token_ids": [1, 2],
        "generated_token_ids": [7, 99],
        "finish_reason": "eos",
        "text": "YES",
        "raw_decoded_text": "YES<eos>",
        "raw_text_sha256": analysis.canonical_hash("YES<eos>"),
        "unexpected_special_tokens": [],
    }
    analysis._generation_binding(saved, [1, 2], protocol)
    with pytest.raises(ValueError, match="prefix differs"):
        analysis._generation_binding(saved, [1, 3], protocol)
    altered = {**saved, "raw_decoded_text": "NO<eos>"}
    with pytest.raises(ValueError, match="text hash differs"):
        analysis._generation_binding(altered, [1, 2], protocol)
    altered = {**saved, "generated_token_ids": [98, 7, 99]}
    with pytest.raises(ValueError, match="special-token disposition"):
        analysis._generation_binding(altered, [1, 2], protocol)
    altered["unexpected_special_tokens"] = [98]
    analysis._generation_binding(altered, [1, 2], protocol)
    assert analysis._answer({"generation": altered, "gold": 1})["invalid"]


def test_reviewed_bank_byte_hash_is_required(tmp_path):
    path = tmp_path / "bank.json"
    path.write_text(analysis.BANK.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 differs"):
        analysis.reviewed_bank(path)


def test_incomplete_attempt_is_preserved_unscored_even_if_rows_were_saved():
    for status in ("failed_incomplete_no_retry", "running", "prepared_no_model_calls"):
        with pytest.raises(ValueError, match="incomplete attempts are preserved unscored"):
            analysis._terminal_study({"status": status, "n_completed_rows": 224})
    analysis._terminal_study({"status": "complete"})
    analysis._terminal_study({"status": "neutral_failed_stopped"})
