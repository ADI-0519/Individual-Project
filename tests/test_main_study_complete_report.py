"""Check completed exports against independent evidence without rerunning models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/results/main-study-2026-11-22-complete"
VALIDATION = ROOT / "docs/validation"


def read(path):
    return json.loads(path.read_bytes())


def test_complete_report_outputs_retain_their_manifest_bytes():
    manifest = read(REPORT / "report_manifest.json")
    assert {p.name for p in REPORT.iterdir()} == set(manifest["outputs"]) | {"report_manifest.json"}
    assert manifest["test_fixture"] is False
    assert manifest["human_review_complete"] is False
    assert manifest["confirmatory_result"] is False
    for name, expected in manifest["outputs"].items():
        assert hashlib.sha256((REPORT / name).read_bytes()).hexdigest() == expected


def test_complete_monitor_metrics_match_the_raw_response_audit():
    analysis_path = VALIDATION / "2026-11-22-main-study-monitor-test-analysis.json"
    analysis = read(analysis_path)
    audit = read(VALIDATION / "2026-11-22-main-study-monitor-test-independent.json")
    assert hashlib.sha256(analysis_path.read_bytes()).hexdigest() == audit["analysis_sha256"]
    assert analysis["partition"] == audit["partition"] == "test"
    assert audit["status"] == "passed"
    assert analysis["overall"] == audit["independent_overall"]
    assert analysis["by_mechanism"] == audit["independent_by_mechanism"]
    assert analysis["source_sha256"] == audit["source_sha256"]
    assert (
        analysis["overall"]["rows"] == audit["coverage"]["raw_generation_texts_and_hashes_checked"]
    )
    for record in (analysis, audit):
        assert record["research_status"] == "agent_audited_exploratory"
        assert record["human_review_complete"] is False
        assert record["confirmatory_result"] is False


def test_shared_baseline_and_unchanged_results_bind_both_final_analyses():
    manifest = read(REPORT / "report_manifest.json")
    summary = read(REPORT / "results_summary.json")
    previous = read(ROOT / "docs/results/main-study-2026-11-22-interim/results_summary.json")
    integration = read(VALIDATION / "2026-11-22-main-study-monitor-integration-independent.json")
    monitor = read(VALIDATION / "2026-11-22-main-study-monitor-test-analysis.json")
    models = {m["model"]: m for m in summary["models"]}
    old_models = {m["model"]: m for m in previous["models"]}
    assert set(models) == set(old_models) == {"Llama", "Qwen"}
    assert integration["status"] == "passed"
    for label, model in models.items():
        source = next(
            i for i in manifest["inputs"] if Path(i["path"]).name.startswith(label.lower())
        )
        proof = integration["comparisons"][label.lower()]
        old_audit_name = (
            "2026-11-22-main-study-llama-heldout-independent-v2.json"
            if label == "Llama"
            else "2026-11-22-main-study-qwen-heldout-independent.json"
        )
        assert source["sha256"] == proof["after"]["sha256"]
        assert (
            proof["before"]["sha256"]
            == read(VALIDATION / old_audit_name)["source_hashes"]["analysis"]
        )
        assert proof["only_changed_field"] == "prose_monitor"
        for field in (
            "primary",
            "learning_and_mechanisms",
            "failed_comparisons",
            "primary_by_allocation",
        ):
            assert model[field] == old_models[label][field]
        assert model["prose_monitor"] == {
            "status": "complete",
            "invalid": monitor["overall"]["invalid"],
            "rows": monitor["overall"]["rows"],
        }
        for mechanism, metrics in monitor["by_mechanism"].items():
            assert (
                model["baselines"]["prose_monitor"][mechanism]
                == metrics["macro_complete_quartet_accuracy"]
            )
    assert (
        models["Llama"]["baselines"]["prose_monitor"]
        == models["Qwen"]["baselines"]["prose_monitor"]
    )


def test_completed_slides_bind_the_completed_report():
    directory = ROOT / "presentations/progress2-main-study-2026-11-22/completed"
    receipt = read(directory / "source-receipt.json")
    assert receipt["slides"] == 4
    assert receipt["test_fixture"] is False
    assert receipt["human_review_complete"] is False
    for name, expected in receipt["outputs"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    for key, filename in (
        ("summary", "results_summary.json"),
        ("report_manifest", "report_manifest.json"),
    ):
        assert (
            hashlib.sha256((REPORT / filename).read_bytes()).hexdigest()
            == receipt["inputs"][key]["sha256"]
        )
