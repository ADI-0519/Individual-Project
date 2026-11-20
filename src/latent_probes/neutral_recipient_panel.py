"""Source-extracted neutral communication states for recipient-outcome readouts.

Reports are fixed, source-supported continuations, not natural model generations.
No recipient outcomes, probe fits, or model loading occur in this module.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from .forecast_panel import (
    CONTEXTS,
    build_sender_messages,
    validate_forecast_panel,
)
from .storage import canonical_hash

REPORT_KINDS = ("full", "omission", "historical")
SOURCE_PANEL_SHA256 = "a2ac1ba2260c2d09555f740debb2b7c96c1e0bfaa6cd326b4a216b339a9a8bb4"
SOURCE_CANONICAL_SHA256 = "74467094069591bd3f8d63c5f69a11808d89f7b4e9e0b1cc3f34c314d8b58dd0"
SOURCE_RELATIVE_PATH = "data/reader_forecast/development_v1.json"


def source_reports(case):
    """Concatenate complete original assertions verbatim, without paraphrasing."""
    old, common, current = case["source"]
    return {"full": common + " " + current, "omission": common, "historical": common + " " + old}


def source_projection(panel):
    return {
        **copy.deepcopy(panel["source_metadata"]),
        "cases": [
            {key: copy.deepcopy(value) for key, value in case.items() if key != "reports"}
            for case in panel["cases"]
        ],
    }


def validate_panel(panel):
    fields = {
        "schema_version",
        "panel_id",
        "source_bank",
        "source_metadata",
        "status",
        "authoring_notes",
        "report_kinds",
        "cases",
    }
    if not isinstance(panel, Mapping) or set(panel) != fields or panel["schema_version"] != 1:
        raise ValueError("Expected complete neutral-recipient excerpt panel schema 1")
    if panel["panel_id"] != "neutral_recipient_excerpts_v1":
        raise ValueError("Unexpected fixed neutral-recipient panel identity")
    if panel["source_bank"] != {
        "relative_path": SOURCE_RELATIVE_PATH,
        "sha256": SOURCE_PANEL_SHA256,
        "canonical_sha256": SOURCE_CANONICAL_SHA256,
    }:
        raise ValueError("Neutral training must bind the unchanged original case bank")
    if panel["report_kinds"] != list(REPORT_KINDS):
        raise ValueError("The three fixed report kinds must retain their authored order")
    if not isinstance(panel["authoring_notes"], list) or not panel["authoring_notes"]:
        raise ValueError("The off-policy authoring limitations must be retained")
    if not isinstance(panel["status"], str) or not panel["status"].strip():
        raise ValueError("The authored status must be declared")
    original = validate_forecast_panel(source_projection(panel))
    if canonical_hash(original) != SOURCE_CANONICAL_SHA256:
        raise ValueError("Original cases, facts, wording, IDs or splits were changed")
    for case in panel["cases"]:
        if set(case) != set(original["cases"][0]) | {"reports"}:
            raise ValueError("Unexpected neutral-recipient case fields")
        if case["reports"] != source_reports(case):
            raise ValueError("A fixed report is not the declared exact source concatenation")
    return copy.deepcopy(dict(panel))


def authored_panel(source):
    source = validate_forecast_panel(source)
    if canonical_hash(source) != SOURCE_CANONICAL_SHA256:
        raise ValueError("Only the unchanged original authored bank may supply training")
    return validate_panel(
        {
            "schema_version": 1,
            "panel_id": "neutral_recipient_excerpts_v1",
            "source_bank": {
                "relative_path": SOURCE_RELATIVE_PATH,
                "sha256": SOURCE_PANEL_SHA256,
                "canonical_sha256": SOURCE_CANONICAL_SHA256,
            },
            "source_metadata": {key: value for key, value in source.items() if key != "cases"},
            "status": "Fixed neutral source excerpts; off-policy communication states, no observed recipient labels.",
            "authoring_notes": [
                "Original 24 cases and 16/4/4 case splits are unchanged. No case or generated outcome is replaced.",
                "Each of full, omission and historical reports is an exact concatenation of complete source assertions; no report is generated or paraphrased by a language model.",
                "The full report is common plus current decisive evidence; omission is common alone; historical is common plus older decisive evidence. All are source-supported as dated assertions, not necessarily sufficient for every recipient.",
                "Cross all three report kinds with all three recipient packets. Recipient decisions must be actually measured using the frozen independently validated reader policy; world truth is not substituted.",
                "Actor states are off-policy forced continuations under the original neutral reporting instruction. They are not naturally generated reports and do not establish anticipation or deceptive intent.",
                "Do not collect labels or fit this new readout unless the separately frozen reader policy passes its fresh capability gate. Gate failure leaves this protocol unexecuted for adviser review.",
                "Use only the original 16 training cases for fitting, original four validation cases for fixed-grid model selection, and original four source test cases for a diagnostic. Fresh natural pressure transfer uses separately authored cases, not these source cases.",
            ],
            "report_kinds": list(REPORT_KINDS),
            "cases": [{**case, "reports": source_reports(case)} for case in source["cases"]],
        }
    )


def load_panel(path):
    path = Path(path)
    panel = validate_panel(json.loads(path.read_text(encoding="utf-8")))
    reference = path.parent.parent / "reader_forecast" / "development_v1.json"
    if (
        reference.exists()
        and hashlib.sha256(reference.read_bytes()).hexdigest() != SOURCE_PANEL_SHA256
    ):
        raise ValueError("Original source-bank bytes no longer match their frozen binding")
    return panel


def neutral_jobs(panel):
    panel = validate_panel(panel)
    original = source_projection(panel)
    originals = {case["id"]: case for case in original["cases"]}
    jobs = []
    for case in panel["cases"]:
        for kind in REPORT_KINDS:
            for context in CONTEXTS:
                text = case["reports"][kind]
                jobs.append(
                    {
                        "example_id": "nr_"
                        + canonical_hash([panel["panel_id"], case["id"], kind, context])[:24],
                        "case_id": case["id"],
                        "family": case["family"],
                        "split": case["split"],
                        "objective": "neutral",
                        "context": context,
                        "report_kind": kind,
                        "messages": build_sender_messages(
                            originals[case["id"]], context, "neutral"
                        ),
                        "report": text,
                        "report_text": text,
                        "off_policy": True,
                        "source_assertion_indices": {
                            "full": [1, 2],
                            "omission": [1],
                            "historical": [1, 0],
                        }[kind],
                    }
                )
    if Counter(job["split"] for job in jobs) != {"train": 144, "validation": 36, "test": 36}:
        raise ValueError("Unexpected fixed neutral case partitions")
    return jobs
