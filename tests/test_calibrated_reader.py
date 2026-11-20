"""Pure input/policy checks; these tests do not claim model capability."""

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from latent_probes.calibrated_reader import (
    MODEL_REVISION,
    build_calibrated_reader_messages,
    build_reader_messages,
    policy_metadata,
    worked_examples,
)
from latent_probes.communication_panel import authored_panel
from latent_probes.communication_panel import build_reader_messages as original_reader
from latent_probes.forecast_panel import PRECEDENCE_POLICY, READER_INSTRUCTION, load_forecast_panel
from latent_probes.forecast_panel import build_reader_messages as forecast_reader

ROOT = Path(__file__).resolve().parents[1]


def test_exactly_nine_balanced_source_grounded_examples_without_rendered_rationale():
    examples = worked_examples()
    assert len(examples) == len({example["id"] for example in examples}) == 9
    assert Counter(example["answer"] for example in examples) == {
        "APPROVE": 3,
        "REJECT": 3,
        "DEFER": 3,
    }
    assert Counter(example["family"] for example in examples) == {
        "conjunction": 3,
        "threshold": 3,
        "supersession": 2,
        "arithmetic": 1,
    }
    for example in examples:
        assert example["messages"][1] == {"role": "assistant", "content": example["answer"]}
        assert example["entity"] in example["messages"][0]["content"]
        assert example["rationale"] not in example["messages"][0]["content"]


def test_example_decisions_follow_available_evidence_without_inventing_a_new_record():
    examples = {example["id"]: example for example in worked_examples()}
    assert 166 + 1 == 167
    assert examples["cal_conjunction_all"]["answer"] == "APPROVE"
    assert "C167 passed" in examples["cal_conjunction_all"]["report"]
    assert examples["cal_conjunction_failure"]["answer"] == "REJECT"
    assert "C167 failed" in examples["cal_conjunction_failure"]["report"]
    assert examples["cal_conjunction_missing"]["answer"] == "DEFER"
    assert "C167" not in " ".join(examples["cal_conjunction_missing"]["evidence_packet"])
    assert 5971 < 6007 <= 6041
    assert examples["cal_threshold_updated_pass"]["answer"] == "APPROVE"
    assert 6173 < 6203 <= 6247
    assert examples["cal_threshold_updated_failure"]["answer"] == "REJECT"
    assert examples["cal_threshold_file_only"]["answer"] == "DEFER"
    assert "score" not in " ".join(examples["cal_threshold_file_only"]["evidence_packet"])
    for key, answer in (
        ("cal_single_available_pass", "APPROVE"),
        ("cal_single_available_failure", "REJECT"),
    ):
        assert examples[key]["answer"] == answer
        assert len(examples[key]["evidence_packet"]) == 1
        assert examples[key]["report"] == ""
    assert examples["cal_arithmetic_missing_cost"]["answer"] == "DEFER"
    assert "cost" not in " ".join(examples["cal_arithmetic_missing_cost"]["evidence_packet"])


def test_existing_and_current_cases_are_not_few_shot_examples():
    paths = [
        ROOT / "data/reader_forecast/development_v1.json",
        ROOT / "data/communication/fresh_transfer_v1.json",
        ROOT / "data/pilot_scenarios.jsonl",
    ]
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for example in worked_examples():
        assert example["entity"] not in source_text
        assert example["rule"] not in source_text
        for assertion in example["evidence_packet"]:
            assert assertion not in source_text
    for number in (166, 167, 6007, 5971, 6041, 6203, 6247, 6173, 680153):
        assert re.search(rf"\b{number}\b", source_text) is None
    # A separately authored fresh gate may arrive after this module; it must
    # independently check the same disjointness before model execution.


def test_actual_payload_and_original_system_are_byte_exact_for_all_contexts():
    for case in authored_panel()["cases"]:
        for context in ("informed", "absent", "stale"):
            for report in ("", "Verbatim report:  spaces  and\nnewlines remain."):
                original = original_reader(case, report, context)
                calibrated = build_calibrated_reader_messages(case, report, context)
                assert len(calibrated) == 20
                assert (
                    calibrated[0]
                    == original[0]
                    == {"role": "system", "content": READER_INSTRUCTION}
                )
                assert PRECEDENCE_POLICY in calibrated[0]["content"]
                assert calibrated[-1] == original[-1]
                assert calibrated == build_reader_messages(case, report, context)
                assert [message["role"] for message in calibrated] == [
                    "system",
                    *(role for _ in range(9) for role in ("user", "assistant")),
                    "user",
                ]


def test_metadata_binds_fixed_policy_and_detached_outputs_cannot_mutate_it():
    before = policy_metadata()
    examples = worked_examples()
    examples[0]["messages"][0]["content"] = "changed"
    examples[0]["answer"] = "DEFER"
    case = authored_panel()["cases"][0]
    messages = build_reader_messages(case, "", "absent")
    messages[1]["content"] = "changed"
    assert policy_metadata() == before
    assert len(before["policy_sha256"]) == len(before["example_material_sha256"]) == 64
    assert before["model_revision"] == MODEL_REVISION
    assert before["max_new_tokens"] == 8 and before["do_sample"] is False
    assert before["max_input_tokens"] == 2048


@pytest.mark.parametrize("context,report", [("unknown", ""), ("informed", None)])
def test_original_validation_is_not_bypassed(context, report):
    with pytest.raises(ValueError):
        build_reader_messages(authored_panel()["cases"][0], report, context)


def test_changes_to_private_case_metadata_never_enter_the_actual_payload():
    case = authored_panel()["cases"][0]
    before = build_reader_messages(case, "", "absent")
    case["id"] = "private-target-id"
    case["private_annotation"] = "private gold"
    assert build_reader_messages(case, "", "absent") == before
    assert "private gold" not in json.dumps(before)


def test_corrupted_normative_metadata_still_fails_original_integrity_checks():
    case = authored_panel()["cases"][0]
    case["packet_expected_decisions"] = {"informed": "private gold"}
    with pytest.raises(ValueError, match="Reader facts differ"):
        build_reader_messages(case, "", "absent")


def test_explicit_original_case_adapter_preserves_all_source_three_fact_cases():
    panel = load_forecast_panel(ROOT / "data/reader_forecast/development_v1.json")
    for case in panel["cases"]:
        assert len(case["source"]) == 3
        for context in ("informed", "absent", "stale"):
            for report in ("", "Verbatim  report\nwith original spacing."):
                original = forecast_reader(case, report, context)
                adapted = build_reader_messages(
                    case, report, context, reader_builder=forecast_reader
                )
                assert adapted[0] == original[0]
                assert adapted[-1] == original[-1]
                assert len(adapted) == 20
    changed = panel["cases"][0]
    changed["source"][-1] = "Corrupted source material"
    with pytest.raises(ValueError):
        build_reader_messages(changed, "", "absent", reader_builder=forecast_reader)


def test_no_arbitrary_adapter_or_automatic_fallback_bypasses_case_validation():
    case = authored_panel()["cases"][0]
    with pytest.raises(ValueError, match="validated case adapter"):
        build_reader_messages(case, "", "absent", reader_builder=lambda *_: [])
    # Passing a four-source communication case to the original three-source
    # adapter must fail rather than falling back to another rendering schema.
    with pytest.raises(ValueError):
        build_reader_messages(case, "", "absent", reader_builder=forecast_reader)
