"""Check the committed agent audit against the exact bank and blinded packet."""

from pathlib import Path

from latent_probes.main_study_bank import load_bank
from latent_probes.main_study_execution import validate_agent_audit
from latent_probes.storage import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def test_actual_agent_answers_remain_bound_and_distinct_from_human_review():
    path = ROOT / "data/annotations/main_study_v1_agent_semantic_20261122.json"
    assert file_sha256(path) == "71b021952823a65ba345763100c8d8a493e16363ce7317dc068fba429ee028af"
    review = validate_agent_audit(
        load_bank(ROOT / "data/belief_main/main_v1/main_bank_v1.json"),
        path,
        ROOT / "docs/review/main_study_v1/review_prompts.jsonl",
    )
    assert review["representative_prompts"] == 736
    assert review["groups"] == 154
    assert review["answers_agree_with_formal_rule"] is True
    assert review["reviewer_type"] == "agent"
    assert review["human_review_complete"] is False
    assert review["review_complete"] is False
