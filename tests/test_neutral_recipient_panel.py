"""Neutral source-excerpt support, immutable case splits and actor visibility."""

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from latent_probes.forecast_panel import build_sender_messages, render_world
from latent_probes.neutral_recipient_panel import (
    SOURCE_PANEL_SHA256,
    authored_panel,
    load_panel,
    neutral_jobs,
    source_projection,
    validate_panel,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/communication/neutral_excerpts_v1.json"
SOURCE = ROOT / "data/reader_forecast/development_v1.json"


@pytest.fixture
def panel():
    return load_panel(DATA)


def test_source_bank_and_original_partitions_are_unchanged(panel):
    original = json.loads(SOURCE.read_text())
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_PANEL_SHA256
    assert source_projection(panel) == original
    assert panel == authored_panel(original)
    assert Counter(c["split"] for c in panel["cases"]) == {
        "train": 16,
        "validation": 4,
        "test": 4,
    }


def test_reports_are_exact_whole_assertions_not_model_fabrications(panel):
    for case in panel["cases"]:
        old, common, current = case["source"]
        assert case["reports"] == {
            "full": common + " " + current,
            "omission": common,
            "historical": common + " " + old,
        }
        for kind, indices in (("full", [1, 2]), ("omission", [1]), ("historical", [1, 0])):
            assert case["reports"][kind] == " ".join(case["source"][i] for i in indices)


def test_all_216_states_cross_reports_and_packets_without_case_leakage(panel):
    jobs = neutral_jobs(panel)
    assert len(jobs) == len({j["example_id"] for j in jobs}) == 216
    assert Counter(j["split"] for j in jobs) == {"train": 144, "validation": 36, "test": 36}
    originals = {c["id"]: c for c in source_projection(panel)["cases"]}
    for case_id, original in originals.items():
        members = [j for j in jobs if j["case_id"] == case_id]
        assert len(members) == 9
        assert len({(j["report_kind"], j["context"]) for j in members}) == 9
        for job in members:
            assert job["split"] == original["split"]
            assert job["messages"] == build_sender_messages(original, job["context"], "neutral")
            assert job["report"] == job["report_text"]
            assert job["off_policy"] is True
            assert job["objective"] == "neutral"
            assert [m["role"] for m in job["messages"]] == ["system", "user"]
            assert "observed_reader" not in json.dumps(job["messages"])
            assert "source_assertion_indices" not in json.dumps(job["messages"])


def test_report_content_is_fixed_across_recipient_packets(panel):
    jobs = neutral_jobs(panel)
    for case in panel["cases"]:
        for kind in ("full", "omission", "historical"):
            members = [j for j in jobs if j["case_id"] == case["id"] and j["report_kind"] == kind]
            assert len({j["report_text"] for j in members}) == 1
            assert len({json.dumps(j["messages"]) for j in members}) == 3


@pytest.mark.parametrize(
    "corruption",
    ["report", "source", "split", "case_id", "metadata", "bank_sha", "new_valid_world"],
)
def test_any_change_to_source_identity_or_verbatim_extraction_is_rejected(panel, corruption):
    changed = copy.deepcopy(panel)
    case = changed["cases"][0]
    if corruption == "report":
        case["reports"]["omission"] += " Every test passed."
    elif corruption == "source":
        case["source"][0] += " Added assertion."
    elif corruption == "split":
        case["split"] = "test"
    elif corruption == "case_id":
        case["id"] = "rf_conjunction_99"
    elif corruption == "metadata":
        changed["source_metadata"]["status"] = "Changed source provenance"
    elif corruption == "bank_sha":
        changed["source_bank"]["sha256"] = "0" * 64
    else:
        case["world"]["entity"] = "A Newly Authored Entity"
        case.update(render_world(case["world"], case["family"]))
    with pytest.raises(ValueError):
        validate_panel(changed)


def test_no_model_decision_labels_are_authored_into_jobs(panel):
    for job in neutral_jobs(panel):
        assert not {
            "decision",
            "observed_reader_decision",
            "label",
            "predictions",
            "probabilities",
        } & set(job)
