"""Synthetic-only semantic review gates; these fixtures are not real annotations."""

from copy import deepcopy

import pytest

from latent_probes.fresh_readout_review import compare


@pytest.fixture
def review():
    packet = {
        "schema": "fresh-readout-blind-review-v1",
        "base_contexts": [{"group_id": f"fixture_{i}"} for i in range(48)],
        "prompts": [{"review_id": "R001"}, {"review_id": "R002"}],
    }
    key = {
        "schema": "fresh-readout-review-key-v1",
        "records": [{"review_id": "R001", "gold": 0}, {"review_id": "R002", "gold": 1}],
    }
    answers = {
        "schema": "fresh-readout-blind-answers-v1",
        "reviewer_type": "agent",
        "human_review_complete": False,
        "context_reviews": [
            {"group_id": f"fixture_{i}", "acceptable": True, "notes": "synthetic context"}
            for i in range(48)
        ],
        "answers": [
            {"review_id": "R001", "answer": "NO", "reason": "synthetic negative"},
            {"review_id": "R002", "answer": "YES", "reason": "synthetic positive"},
        ],
    }
    return packet, key, answers


def test_complete_synthetic_review_compares_without_human_claim(review):
    result = compare(*review)
    assert result["all_answers_match"] and not result["context_issues"]
    assert result["coverage"] == {"base_contexts": 48, "prompts": 2, "review_ids": ["R001", "R002"]}


def test_disagreement_is_explicit_not_relabelled_as_pass(review):
    packet, key, answers = deepcopy(review)
    answers["answers"][0]["answer"] = "YES"
    result = compare(packet, key, answers)
    assert not result["all_answers_match"] and result["mismatches"][0]["review_id"] == "R001"


@pytest.mark.parametrize(
    "change",
    ["missing_answer", "duplicate_answer", "missing_context", "human_claim", "empty_reason"],
)
def test_incomplete_or_fabricated_acceptance_rejected(review, change):
    packet, key, answers = deepcopy(review)
    if change == "missing_answer":
        answers["answers"].pop()
    elif change == "duplicate_answer":
        answers["answers"].append(answers["answers"][0])
    elif change == "missing_context":
        answers["context_reviews"].pop()
    elif change == "human_claim":
        answers["human_review_complete"] = True
    else:
        answers["answers"][0]["reason"] = ""
    with pytest.raises(ValueError):
        compare(packet, key, answers)


def test_context_ambiguity_is_preserved(review):
    packet, key, answers = deepcopy(review)
    answers["context_reviews"][0].update(acceptable=False, notes="synthetic ambiguity")
    result = compare(packet, key, answers)
    assert result["context_issues"] and result["all_answers_match"]
