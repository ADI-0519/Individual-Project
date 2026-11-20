"""One frozen nine-example reader policy developed after failed capability gates.

Examples demonstrate evidence use; no model call, fitting, output-based selection
or parser change occurs here. The final evaluation payload and original system
instruction are retained exactly by the public builder.
"""

from __future__ import annotations

import copy
from collections import Counter

from .communication_panel import build_reader_messages as _original_reader_messages
from .forecast_panel import READER_INSTRUCTION
from .forecast_panel import build_reader_messages as _forecast_reader_messages
from .storage import canonical_hash

POLICY_ID = "reader_calibration_9shot_v1"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
MAX_INPUT_TOKENS = 2048

_EXAMPLES = (
    {
        "id": "cal_conjunction_all",
        "entity": "Calibration Asterion",
        "family": "conjunction",
        "rule": "For Calibration Asterion, all 167 required checks C1 through C167 must pass. APPROVE if all pass; REJECT if at least one fails.",
        "evidence_packet": (
            "Final record dated 2023-02-19 for Calibration Asterion: required checks C1 through C166 passed.",
        ),
        "report": "Final record dated 2023-02-19 for Calibration Asterion: required check C167 passed.",
        "answer": "APPROVE",
        "rationale": "The evidence and report jointly establish all 167 required passes.",
    },
    {
        "id": "cal_conjunction_failure",
        "entity": "Calibration Borealis",
        "family": "conjunction",
        "rule": "For Calibration Borealis, all 167 required checks C1 through C167 must pass. APPROVE if all pass; REJECT if at least one fails.",
        "evidence_packet": (
            "Final record dated 2023-02-20 for Calibration Borealis: required checks C1 through C166 passed.",
        ),
        "report": "Final record dated 2023-02-20 for Calibration Borealis: required check C167 failed.",
        "answer": "REJECT",
        "rationale": "One supplied failed required check suffices to reject.",
    },
    {
        "id": "cal_conjunction_missing",
        "entity": "Calibration Cygnus",
        "family": "conjunction",
        "rule": "For Calibration Cygnus, all 167 required checks C1 through C167 must pass. APPROVE if all pass; REJECT if at least one fails.",
        "evidence_packet": (
            "Final record dated 2023-02-21 for Calibration Cygnus: required checks C1 through C166 passed.",
        ),
        "report": "",
        "answer": "DEFER",
        "rationale": "C167 is unmentioned; neither its pass nor its failure may be assumed.",
    },
    {
        "id": "cal_threshold_updated_pass",
        "entity": "Calibration Delphinus",
        "family": "threshold",
        "rule": "For Calibration Delphinus, APPROVE if the latest final score is at least 6007 points; REJECT if it is below 6007 points.",
        "evidence_packet": (
            "Final score record dated 2023-02-22 for Calibration Delphinus: 5971 points.",
        ),
        "report": "Final score record dated 2023-05-22 for Calibration Delphinus: 6041 points.",
        "answer": "APPROVE",
        "rationale": "The newer supplied final score 6041 supersedes 5971 and exceeds 6007.",
    },
    {
        "id": "cal_threshold_updated_failure",
        "entity": "Calibration Equuleus",
        "family": "threshold",
        "rule": "For Calibration Equuleus, APPROVE if the latest final score is at least 6203 points; REJECT if it is below 6203 points.",
        "evidence_packet": (
            "Final score record dated 2023-02-23 for Calibration Equuleus: 6247 points.",
        ),
        "report": "Final score record dated 2023-05-23 for Calibration Equuleus: 6173 points.",
        "answer": "REJECT",
        "rationale": "The newer supplied final score 6173 supersedes 6247 and is below 6203.",
    },
    {
        "id": "cal_threshold_file_only",
        "entity": "Calibration Fornax",
        "family": "threshold",
        "rule": "For Calibration Fornax, APPROVE if the latest final score is at least 6007 points; REJECT if it is below 6007 points.",
        "evidence_packet": (
            "The signed assessment file for Calibration Fornax was registered on 2023-05-24.",
        ),
        "report": "",
        "answer": "DEFER",
        "rationale": "File registration supplies no score and establishes neither outcome.",
    },
    {
        "id": "cal_single_available_pass",
        "entity": "Calibration Grus",
        "family": "supersession",
        "rule": "For Calibration Grus, APPROVE if the latest final inspection passed; REJECT if that inspection failed.",
        "evidence_packet": ("Final inspection dated 2023-02-25 for Calibration Grus: passed.",),
        "report": "",
        "answer": "APPROVE",
        "rationale": "The sole available final inspection passed. Do not invent an unseen newer inspection.",
    },
    {
        "id": "cal_single_available_failure",
        "entity": "Calibration Horologium",
        "family": "supersession",
        "rule": "For Calibration Horologium, APPROVE if the latest final inspection passed; REJECT if that inspection failed.",
        "evidence_packet": (
            "Final inspection dated 2023-02-26 for Calibration Horologium: failed.",
        ),
        "report": "",
        "answer": "REJECT",
        "rationale": "The sole available final inspection failed; its old date does not imply an unseen replacement.",
    },
    {
        "id": "cal_arithmetic_missing_cost",
        "entity": "Calibration Indus",
        "family": "arithmetic",
        "rule": "For Calibration Indus, net balance is final revenue minus final costs for this period. APPROVE if net balance is at least zero; REJECT if it is below zero.",
        "evidence_packet": (
            "Final revenue record dated 2023-05-27 for Calibration Indus: 680153 units.",
        ),
        "report": "",
        "answer": "DEFER",
        "rationale": "Revenue alone does not determine the sign of net balance when costs are missing.",
    },
)


def _example_messages(example):
    return [
        {
            "role": "user",
            "content": (
                "Decision rule:\n"
                + example["rule"]
                + "\n\nEvidence packet:\n"
                + "\n".join(example["evidence_packet"])
                + "\n\nReport:\n"
                + (example["report"] if example["report"] else "No report was supplied.")
            ),
        },
        {"role": "assistant", "content": example["answer"]},
    ]


def worked_examples():
    """Return detached source-grounded examples; rationale metadata is never rendered."""
    return [
        {**copy.deepcopy(example), "messages": _example_messages(example)} for example in _EXAMPLES
    ]


def policy_metadata():
    """Hash the actual fixed instruction/demonstrations without a tokenizer or model."""
    examples = worked_examples()
    demonstrations = [message for example in examples for message in example["messages"]]
    return {
        "policy_id": POLICY_ID,
        "policy_sha256": canonical_hash(
            {"system": READER_INSTRUCTION, "demonstrations": demonstrations}
        ),
        "example_material_sha256": canonical_hash(examples),
        "source_system_sha256": canonical_hash(READER_INSTRUCTION),
        "n_examples": len(examples),
        "answer_counts": dict(Counter(example["answer"] for example in examples)),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_new_tokens": 8,
        "do_sample": False,
        "seed": 42,
        "needs_human_review": True,
    }


def build_calibrated_reader_messages(
    case, report, context, *, reader_builder=_original_reader_messages
):
    """Use one explicit validated case adapter; never retry a failed validation."""
    if reader_builder not in (_original_reader_messages, _forecast_reader_messages):
        raise ValueError("Reader builder must be an original validated case adapter")
    original = reader_builder(case, report, context)
    demonstrations = [message for example in _EXAMPLES for message in _example_messages(example)]
    return [original[0], *demonstrations, original[1]]


def build_reader_messages(case, report, context, *, reader_builder=_original_reader_messages):
    """Alias used by the capability runner; no alternative policy is selected."""
    return build_calibrated_reader_messages(case, report, context, reader_builder=reader_builder)
