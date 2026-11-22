"""Protect the committed report export without native captures or model inference."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "docs/results/main-study-2026-11-22-interim"
OUTPUTS = {
    "learning_curves.pdf",
    "learning_curves.png",
    "learning_curves.svg",
    "primary.pdf",
    "primary.png",
    "primary.svg",
    "results.md",
    "results_summary.json",
}
AUDITS = {
    "Llama": ROOT / "docs/validation/2026-11-22-main-study-llama-heldout-independent-v2.json",
    "Qwen": ROOT / "docs/validation/2026-11-22-main-study-qwen-heldout-independent.json",
}
MODEL_IDS = {
    "Llama": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen": "Qwen/Qwen2.5-7B-Instruct",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_outputs(directory, manifest):
    assert set(manifest["outputs"]) == OUTPUTS, "Declared output inventory differs"
    actual = {
        str(path.relative_to(directory)).replace("\\", "/")
        for path in directory.rglob("*")
        if path.is_file()
    }
    assert actual == OUTPUTS | {"report_manifest.json"}, "Snapshot file inventory differs"
    for name, expected in manifest["outputs"].items():
        actual_hash = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        assert actual_hash == expected, f"Snapshot output hash differs: {name}"


def check_markers(manifest, summary, audits):
    assert manifest["schema"] == "main-study-report-manifest-v1"
    assert summary["schema"] == "main-study-report-v1"
    for value in (manifest, summary):
        assert value["test_fixture"] is False, "Real export became a fixture"
    for value in (manifest, summary, *audits.values()):
        assert value["research_status"] == "agent_audited_exploratory", "Exploratory status differs"
        assert value["human_review_complete"] is False, "Human-review marker differs"
        assert value["confirmatory_result"] is False, "Confirmatory marker differs"
    for name, audit in audits.items():
        assert audit["schema"] == "main-study-heldout-independent-audit-v1"
        assert audit["model"] == name.lower()
        assert audit["reviewer_type"] == "agent"
        assert audit["status"] == "passed"
        assert audit["new_model_calls"] == audit["new_fits_or_selections"] == 0


def check_primary_bindings(manifest, summary, audits):
    models = {model["model"]: model for model in summary["models"]}
    assert len(summary["models"]) == len(models) == 2
    assert set(models) == set(audits) == set(MODEL_IDS)
    inputs = {
        PurePosixPath(item["path"].replace("\\", "/")).name: item["sha256"]
        for item in manifest["inputs"]
    }
    expected_names = {f"{name.lower()}_analysis_without_monitor.json" for name in models}
    assert len(manifest["inputs"]) == len(inputs) == 2
    assert set(inputs) == expected_names
    for name, model in models.items():
        audit = audits[name]
        primary, sampling = model["primary"], audit["primary_sampling"]
        assert model["model_id"] == MODEL_IDS[name]
        assert primary["status"] == "complete"
        assert (
            inputs[f"{name.lower()}_analysis_without_monitor.json"]
            == audit["source_hashes"]["analysis"]
        ), "Analysis hash binding differs"
        actual_primary = {key: primary[key] for key in audit["primary_reconstructed"]}
        assert actual_primary == audit["primary_reconstructed"], "Independent primary differs"
        assert primary["independent_scenario_groups"] == sampling["correction_groups"]
        assert primary["allocation_seeds"] == sampling["paired_allocation_seeds"]
        assert primary["bootstrap_draws"] == sampling["bootstrap_draws"], "Bootstrap draws differ"
        assert primary["bootstrap_seed"] == sampling["bootstrap_seed"], "Bootstrap seed differs"
        assert sampling["interval_exactly_matches_saved_analysis"] is True
        by_seed = {item["seed"]: item for item in primary["by_seed"]}
        allocations = {item["allocation_seed"]: item for item in model["primary_by_allocation"]}
        assert set(by_seed) == set(allocations) == set(primary["allocation_seeds"])
        for seed, item in allocations.items():
            assert item["status"] == "complete"
            for field in ("restricted", "full", "difference"):
                assert item[field] == pytest.approx(by_seed[seed][field], rel=0, abs=1e-15)
    # Reporter and auditor source hashes identify historical executions. Their
    # current source files can evolve without invalidating this frozen export.


@pytest.fixture
def snapshot_records():
    return (
        read(SNAPSHOT / "report_manifest.json"),
        read(SNAPSHOT / "results_summary.json"),
        {name: read(path) for name, path in AUDITS.items()},
    )


def test_committed_snapshot_has_exact_manifest_inventory_and_hashes(snapshot_records):
    manifest, _, _ = snapshot_records
    check_outputs(SNAPSHOT, manifest)


def test_committed_snapshot_and_audits_are_real_exploratory(snapshot_records):
    check_markers(*snapshot_records)


def test_committed_primary_and_analysis_hashes_match_final_independent_audits(snapshot_records):
    check_primary_bindings(*snapshot_records)


def test_committed_slide_outputs_and_source_bindings_survive_checkout():
    directory = ROOT / "presentations/progress2-main-study-2026-11-22/reviewed"
    receipt = read(directory / "source-receipt.json")
    assert receipt["slides"] == 4
    assert receipt["test_fixture"] is False
    assert receipt["research_status"] == "agent_audited_exploratory"
    assert receipt["human_review_complete"] is False
    assert receipt["confirmatory_result"] is False
    assert set(receipt["outputs"]) == {
        "Progress-2-main-study.pptx",
        "Progress-2-main-study.pdf",
        "speaker-notes.md",
    }
    for name, digest in receipt["outputs"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    for key, name in (
        ("summary", "results_summary.json"),
        ("report_manifest", "report_manifest.json"),
    ):
        assert (
            hashlib.sha256((SNAPSHOT / name).read_bytes()).hexdigest()
            == receipt["inputs"][key]["sha256"]
        )
    for name in ("speaker-notes.md", "source-receipt.json"):
        raw = (directory / name).read_bytes()
        assert b"\r\n" not in raw
        assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")


def test_committed_monitor_development_analysis_matches_independent_audit():
    analysis_path = ROOT / "docs/validation/2026-11-22-main-study-monitor-development-analysis.json"
    audit = read(
        ROOT / "docs/validation/2026-11-22-main-study-monitor-development-independent.json"
    )
    analysis = read(analysis_path)
    assert hashlib.sha256(analysis_path.read_bytes()).hexdigest() == audit["analysis_sha256"]
    assert analysis["partition"] == audit["partition"] == "development"
    assert audit["status"] == "passed"
    assert audit["test_result_files_accessed"] is False
    for data in (analysis, audit):
        assert data["research_status"] == "agent_audited_exploratory"
        assert data["human_review_complete"] is False
        assert data["confirmatory_result"] is False
    assert analysis["overall"] == audit["independent_overall"]
    assert analysis["by_mechanism"] == audit["independent_by_mechanism"]
    assert analysis["source_sha256"] == audit["source_sha256"]
    assert (
        analysis["overall"]["rows"] == audit["coverage"]["raw_generation_texts_and_hashes_checked"]
    )


@pytest.mark.parametrize("corruption", ["asset_bytes", "undeclared_file", "omitted_manifest_entry"])
def test_snapshot_corruption_is_detected(snapshot_records, tmp_path, corruption):
    manifest, _, _ = snapshot_records
    target = tmp_path / "snapshot"
    shutil.copytree(SNAPSHOT, target)
    if corruption == "asset_bytes":
        with (target / "primary.svg").open("ab") as stream:
            stream.write(b"\nchanged")
        message = "output hash differs"
    elif corruption == "undeclared_file":
        (target / "unrecorded.txt").write_text("extra", encoding="utf-8")
        message = "file inventory differs"
    else:
        manifest = copy.deepcopy(manifest)
        del manifest["outputs"]["primary.svg"]
        message = "Declared output inventory differs"
    with pytest.raises(AssertionError, match=message):
        check_outputs(target, manifest)


@pytest.mark.parametrize(
    "corruption", ["primary", "analysis_hash", "bootstrap_seed", "human_review"]
)
def test_metadata_changes_cannot_hide_broken_audit_bindings(snapshot_records, corruption):
    manifest, summary, audits = copy.deepcopy(snapshot_records)
    if corruption == "primary":
        summary["models"][0]["primary"]["full"] += 0.01
        message = "Independent primary differs"
    elif corruption == "analysis_hash":
        manifest["inputs"][0]["sha256"] = "0" * 64
        message = "Analysis hash binding differs"
    elif corruption == "bootstrap_seed":
        summary["models"][0]["primary"]["bootstrap_seed"] += 1
        message = "Bootstrap seed differs"
    else:
        summary["human_review_complete"] = True
        message = "Human-review marker differs"
    with pytest.raises(AssertionError, match=message):
        check_markers(manifest, summary, audits)
        check_primary_bindings(manifest, summary, audits)
