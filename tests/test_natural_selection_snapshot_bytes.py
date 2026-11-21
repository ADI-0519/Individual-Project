"""Checkout bytes must retain the provenance of the executed source/reviews."""

from pathlib import Path

import pytest

from latent_probes.storage import file_sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative,expected",
    [
        (
            "data/communication/natural_selection_pilot_draft_v1.json",
            "d7fa9e42ecc10affbcbd417e3329a9fea5fa7ed6e64a8150ad482a23950e9c07",
        ),
        (
            "data/annotations/natural_selection_pilot_v1_reviewer_a_agent_draft.json",
            "0bc88ecb92ab3169bc3e2032c92fe51e08f0b23814fe28745488222d9d27bfb5",
        ),
        (
            "data/annotations/natural_selection_pilot_v1_reviewer_b_agent_draft.json",
            "d18a8a08a179f1fdb36c1430dd7d6ffa9672d0c0c2f93072ecd478fa19f12bf2",
        ),
        (
            "data/annotations/natural_selection_review_protocol_v1.json",
            "dcca1d96963b6b6244367ab3006c0a3ccb0cebbeb4082da3f5114ea2284fcd56",
        ),
    ],
)
def test_executed_json_snapshot_bytes_survive_checkout(relative, expected):
    assert file_sha256(ROOT / relative) == expected
