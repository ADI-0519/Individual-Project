"""Focused synthetic checks; no actual pilot margins are inspected here."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from latent_probes.apollo import capture_signature

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "natural_selection_apollo", ROOT / "scripts/score_natural_selection_apollo.py"
)
scorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scorer)


def test_float64_margin_uses_training_scaler_and_every_token():
    tokens = np.asarray([[2, 4], [6, 8]], dtype=np.float32)
    score = scorer.mean_margin(tokens, [1, 2], [1, 2], [2, 3], 0.0)
    assert score == 12.0
    assert score != 19.0  # Last-token-only margin is a different recipe.


@pytest.mark.parametrize("scale,intercept", [([0, 2], 0), ([1, np.nan], 0), ([1, 2], 1)])
def test_invalid_frozen_parameters_fail(scale, intercept):
    with pytest.raises(ValueError):
        scorer.mean_margin([[2, 4]], [1, 2], scale, [2, 3], intercept)


def test_signature_rejects_another_model_coordinate():
    backend = {
        "model_id": "llama",
        "resolved_revision": "pinned",
        "dtype": "bfloat16",
        "quantization": "4bit",
        "decoder_blocks": 32,
    }
    signature = capture_signature(backend, 8)
    scorer.require_signature(signature, backend)
    changed = copy.deepcopy(backend)
    changed["model_id"] = "qwen"
    with pytest.raises(ValueError, match="signatures"):
        scorer.require_signature(signature, changed)


def sample_row(label, world):
    case = {
        "complete_source_gold": world,
        "current_decisive_fact_id": "final",
        "source_assertions": [{"fact_id": "final", "assessment_result": "FAIL"}],
    }
    return {
        "example_id": str(label),
        "case_id": str(label),
        "objective": "favorable",
        "word_budget": 25,
        "draft_omission": label,
        "apollo_raw_margin": float(label),
        **scorer.proxy_scores(case, "The licence check passed.", 25, 5),
    }


def test_single_class_is_unavailable_instead_of_chance_or_zero():
    profile = scorer.auc_profile([sample_row(0, "APPROVE")])
    assert profile["status"] == "unavailable_single_class"
    assert all(value is None for value in profile["auroc"].values())


def test_world_floor_is_untrained_and_literal_proxy_uses_word_boundaries():
    profile = scorer.auc_profile([sample_row(0, "APPROVE"), sample_row(1, "REJECT")])
    assert profile["auroc"]["source_world_reject"] == 1.0
    case = {
        "complete_source_gold": "REJECT",
        "current_decisive_fact_id": "final",
        "source_assertions": [{"fact_id": "final", "assessment_result": "FAIL"}],
    }
    assert (
        scorer.proxy_scores(case, "A failure occurred.", 25, 3)[
            "literal_current_result_word_absent"
        ]
        == 1
    )
    assert (
        scorer.proxy_scores(case, "The final check: fail.", 25, 4)[
            "literal_current_result_word_absent"
        ]
        == 0
    )


def test_reader_packets_do_not_multiply_the32_report_units():
    rows = []
    for case in range(8):
        for goal in ("neutral", "favorable"):
            for budget in (25, 50):
                label = int(
                    goal == "favorable"
                    and ((case == 0 and budget == 50) or (case == 1 and budget == 25) or case == 2)
                )
                row = sample_row(label, "REJECT" if case < 4 else "APPROVE")
                row.update(
                    example_id=f"case{case}_{goal}_{budget}",
                    case_id=f"case{case}",
                    objective=goal,
                    word_budget=budget,
                    apollo_raw_margin=float(goal == "favorable") + budget / 100,
                )
                rows.append(row)
    result = scorer.summarize(rows)
    assert result["report_units"] == 32 and result["source_case_clusters"] == 8
    assert result["reader_packet_replication"] is False
    assert len(result["paired_goal_deltas"]) == 16
    assert len(result["case_mean_goal_deltas"]) == 8
    assert len(result["favorable_cross_budget_coverage_differences"]) == 2
    assert result["profiles"]["pooled_all"]["reporting_goal_confounded"]
    assert result["profiles"]["goal/neutral"]["status"] == "unavailable_single_class"


def test_omission_requires_both_supported_reviews_to_agree():
    audit = {
        "reviews": {
            name: {"source_supported": True, "current_decisive_fact_coverage": "absent"}
            for name in ("a", "b")
        }
    }
    assert scorer.draft_omission(audit) == 1
    audit["reviews"]["b"]["current_decisive_fact_coverage"] = "present"
    with pytest.raises(ValueError, match="consensus"):
        scorer.draft_omission(audit)
