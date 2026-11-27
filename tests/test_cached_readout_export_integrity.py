"""Portable checks that the committed report and frozen sources retain their bytes."""

import json
from datetime import datetime
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "docs/results/cached-readout-adequacy-2026-11-22"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def workspace_path(original):
    value = original.replace("\\", "/")
    marker = "/Individual-Project/"
    assert value.count(marker) == 1
    relative = Path(value.split(marker)[1])
    assert not relative.is_absolute() and ".." not in relative.parts
    return ROOT / relative


def test_committed_export_bytes_match_promoted_manifest():
    manifest = read(EXPORT / "manifest.json")
    assert manifest["research_status"] == "posthoc_exposed_bank_group_crossfit"
    assert set(manifest["output_sha256"]) == {
        "summary.json",
        "results.md",
        "readout_adequacy.pdf",
        "readout_adequacy.svg",
        "readout_adequacy.png",
    }
    for name, expected in manifest["output_sha256"].items():
        assert digest(EXPORT / name) == expected
    for original, expected in manifest["source_sha256"].items():
        if "/Individual-Project/" in original.replace("\\", "/"):
            assert digest(workspace_path(original)) == expected


def test_frozen_fitting_sources_match_execution_protocol():
    protocol = read(ROOT / "docs/validation/2026-11-22-cached-readout-adequacy-protocol.json")
    for original, expected in protocol["source_sha256"].items():
        assert digest(workspace_path(original)) == expected
    assert (
        digest(workspace_path(protocol["fold_manifest"]["path"]))
        == protocol["fold_manifest"]["sha256"]
    )


def test_committed_protocol_and_preparation_are_exact_execution_receipts():
    validation = ROOT / "docs/validation"
    prepared = read(validation / "2026-11-22-cached-readout-adequacy-preparation.json")
    complete = read(validation / "2026-11-22-cached-readout-adequacy-completion.json")
    assert prepared["classifier_fits"] == prepared["model_calls"] == 0
    assert prepared["protocol_sha256"] == digest(
        validation / "2026-11-22-cached-readout-adequacy-protocol.json"
    )
    assert complete["status"] == "complete" and complete["confirmatory_result"] is False
    assert complete["prepared_manifest_sha256"] == digest(
        validation / "2026-11-22-cached-readout-adequacy-preparation.json"
    )


def test_summary_keeps_exposure_and_supervision_limits_explicit():
    summary = read(EXPORT / "summary.json")
    assert (
        summary["fresh_test"]
        is summary["confirmatory_result"]
        is summary["human_review_complete"]
        is False
    )
    assert summary["outer_train_rows"] == 1536
    assert summary["independent_narrative_groups"] == 30
    assert summary["new_language_model_calls"] == 0
    for model in summary["models"].values():
        assert set(model["methods"]) == {"direct_lr", "rbf_svc", "fourfactor_lr"}
        for method in model["methods"].values():
            assert len(method["groups"]) == 30 and len(method["fold_values"]) == 5


def test_fresh_agent_check_keeps_source_identity_and_precomparison_freeze():
    review = read(ROOT / "docs/validation/2026-11-22-cached-readout-adequacy-second-agent-v2.json")
    assert review["status"] == "passed"
    assert review["model_calls"] == review["fits"] == review["project_code_imports"] == 0
    assert review["comparison_totals"]["verdict_mismatches"] == 0
    assert review["comparison_totals"]["factor_prediction_mismatches"] == 0
    checker = review["checker_provenance"]
    assert digest(workspace_path(checker["current_checker"])) == checker["current_checker_sha256"]
    events = {r["event"]: r["time"] for r in review["chronology"]}
    assert datetime.fromisoformat(
        events["independent_predictions_and_metrics_frozen"]
    ) < datetime.fromisoformat(events["comparison_started"])
    assert review["independence"]["saved_readout_score_or_metric_values_before_freeze"] is False
