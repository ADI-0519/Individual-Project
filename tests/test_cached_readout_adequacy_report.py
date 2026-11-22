"""Reporter-only provenance and arithmetic corruption checks; no model fitting."""

import importlib.util
import json
from itertools import product
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cached_adequacy_report", ROOT / "scripts/report_cached_readout_adequacy.py"
)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def metrics(groups):
    quartets = [
        {
            "group_id": group,
            "update": u,
            "world": w,
            "name_map": m,
            "correct_cells": 4,
            "all_correct": True,
        }
        for group in groups
        for u, w, m in product((0, 1), repeat=3)
    ]
    return {
        "rows": 64 * len(groups),
        "primary_quartet_accuracy": 1.0,
        "row_accuracy": 1.0,
        "primary_row_accuracy": 1.0,
        "secondary_row_accuracy": 1.0,
        "quartets": quartets,
        "quartet_correct_cells_histogram": {
            str(k): len(quartets) if k == 4 else 0 for k in range(5)
        },
        "all_class_recalls": {"0": 1.0, "1": 1.0},
        "primary_class_recalls": {"0": 1.0, "1": 1.0},
        "secondary_class_recalls": {"0": 1.0, "1": 1.0},
        "nuisance_verdict_flips": {
            factor: {"pairs": 32 * len(groups), "flips": 0, "fraction": 0.0}
            for factor in ("world", "name_map")
        },
        "factor_joint_accuracy": 1.0,
        "factor_accuracies": {
            name: 1.0 for name in ("initial", "update", "first_receipt", "receipt")
        },
    }


def method_record():
    groups = [f"fixture_group_{i:02d}" for i in range(30)]
    return {
        "summary": metrics(groups),
        "groups": [dict(group_id=g, fold=i // 6, **metrics([g])) for i, g in enumerate(groups)],
        "fold_values": [dict(fold=f, **metrics(groups[f * 6 : (f + 1) * 6])) for f in range(5)],
        "oof_scores": [],
    }


def bind_completion(run, audit):
    complete_path = run / "completed_manifest.json"
    complete = REPORT.read(complete_path) if complete_path.exists() else {}
    complete.update(
        status="complete",
        research_status=REPORT.STATUS,
        confirmatory_result=False,
        model_calls=0,
        prepared_manifest_sha256=REPORT.digest(run / "prepared_manifest.json"),
        output_sha256={
            p.relative_to(run).as_posix(): REPORT.digest(p)
            for p in sorted(run.rglob("*"))
            if p.is_file() and p != complete_path
        },
    )
    write(complete_path, complete)
    review = {
        "schema": "cached-readout-adequacy-independent-audit-v1",
        "status": "all_independent_checks_passed",
        "research_status": REPORT.STATUS,
        "pipeline_imported": False,
        "classifier_refits": 0,
        "model_calls": 0,
        "models": {
            model: {
                "rows": 1920,
                "groups": 30,
                "outer_folds": 5,
                "reconstructed_oof_predictions": 5760,
                "all_group_fold_and_aggregate_metrics_match": True,
            }
            for model in ("llama", "qwen")
        },
        "source_sha256": {
            str(p): REPORT.digest(p)
            for p in [
                complete_path,
                run / "prepared_manifest.json",
                run / "llama/oof_results.json",
                run / "qwen/oof_results.json",
            ]
        },
    }
    write(audit, review)


@pytest.fixture
def source_fixture(tmp_path):
    run = tmp_path / "fixture_run"
    audit = tmp_path / "independent_audit.json"
    write(
        run / "prepared_manifest.json",
        {
            "schema": "cached-readout-adequacy-preparation-v1",
            "research_status": REPORT.STATUS,
            "model_calls": 0,
            "classifier_fits": 0,
            "environment": {"fixture": True},
        },
    )
    for model in ("llama", "qwen"):
        write(
            run / model / "oof_results.json",
            {
                "research_status": REPORT.STATUS,
                "confirmatory_result": False,
                "human_review_complete": False,
                "fresh_test": False,
                "original_split": "test",
                "model_calls": 0,
                "methods": {name: method_record() for name in REPORT.METHODS},
            },
        )
        for fold in range(5):
            write(
                run / model / f"fold_{fold}/results.json",
                {
                    "fold": fold,
                    "methods": {
                        name: {
                            "selected_parameters": {"C": 1.0},
                            "candidates": [{"parameters": {"C": 1.0}, "status": "complete"}],
                        }
                        for name in REPORT.METHODS
                    },
                },
            )
    bind_completion(run, audit)
    return run, audit


def test_compact_method_accepts_consistent_complete_metrics():
    compact = REPORT.compact_method(method_record())
    assert compact["summary"]["primary_quartet_accuracy"] == 1.0
    assert len(compact["groups"]) == 30 and len(compact["fold_values"]) == 5
    assert "quartets" not in compact["summary"]


@pytest.mark.parametrize(
    "corruption",
    [
        "duplicate_group",
        "duplicate_fold",
        "group_rows",
        "fold_rows",
        "summary_quartet_rate",
        "weighted_row_rate",
        "quartet_histogram",
        "group_quartet_count",
        "fold_quartet_count",
    ],
)
def test_compact_method_rejects_numerically_inconsistent_or_duplicate_metrics(corruption):
    record = method_record()
    if corruption == "duplicate_group":
        record["groups"][1]["group_id"] = record["groups"][0]["group_id"]
    elif corruption == "duplicate_fold":
        record["fold_values"][1]["fold"] = record["fold_values"][0]["fold"]
    elif corruption == "group_rows":
        record["groups"][0]["rows"] = 63
    elif corruption == "fold_rows":
        record["fold_values"][0]["rows"] = 383
    elif corruption == "summary_quartet_rate":
        record["summary"]["primary_quartet_accuracy"] = 0.5
    elif corruption == "weighted_row_rate":
        record["groups"][0]["row_accuracy"] = 0.0
    elif corruption == "quartet_histogram":
        record["summary"]["quartet_correct_cells_histogram"] = {
            "0": 0,
            "1": 0,
            "2": 0,
            "3": 1,
            "4": 239,
        }
    elif corruption == "group_quartet_count":
        record["groups"][0]["quartets"].pop()
    else:
        record["fold_values"][0]["quartets"].pop()
    with pytest.raises(ValueError):
        REPORT.compact_method(record)


def test_load_summary_accepts_complete_explicitly_exposed_fixture(source_fixture):
    summary = REPORT.load_summary(*source_fixture)
    assert summary["fresh_test"] is False and summary["confirmatory_result"] is False
    assert (
        summary["models"]["llama"]["methods"]["direct_lr"]["summary"]["primary_quartet_accuracy"]
        == 1.0
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("fresh_test", True),
        ("original_split", "fresh_test"),
        ("confirmatory_result", True),
        ("research_status", "fresh_confirmatory"),
    ],
)
def test_load_summary_rejects_false_fresh_or_confirmatory_provenance(source_fixture, field, value):
    run, audit = source_fixture
    path = run / "llama/oof_results.json"
    record = REPORT.read(path)
    record[field] = value
    write(path, record)
    bind_completion(run, audit)
    with pytest.raises(ValueError):
        REPORT.load_summary(run, audit)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "unrelated-audit"),
        ("research_status", "fresh_confirmatory"),
        ("pipeline_imported", True),
        ("classifier_refits", 1),
    ],
)
def test_load_summary_rejects_false_independent_audit_provenance(source_fixture, field, value):
    run, audit = source_fixture
    record = REPORT.read(audit)
    record[field] = value
    write(audit, record)
    with pytest.raises(ValueError):
        REPORT.load_summary(run, audit)


@pytest.mark.parametrize(
    "source", ["prepared_manifest.json", "llama/oof_results.json", "qwen/oof_results.json"]
)
def test_load_summary_rejects_tampered_explicit_audit_source_hash(source_fixture, source):
    run, audit = source_fixture
    record = REPORT.read(audit)
    record["source_sha256"][str(run / source)] = "0" * 64
    write(audit, record)
    with pytest.raises(ValueError):
        REPORT.load_summary(run, audit)


def test_load_summary_rejects_completion_digest_under_unrelated_audit_key(source_fixture):
    run, audit = source_fixture
    record = REPORT.read(audit)
    digest = record["source_sha256"].pop(str(run / "completed_manifest.json"))
    record["source_sha256"][str(run / "unrelated.json")] = digest
    write(audit, record)
    with pytest.raises(ValueError):
        REPORT.load_summary(run, audit)


@pytest.mark.parametrize(
    "source", ["llama/oof_results.json", "llama/fold_0/results.json", "qwen/fold_4/results.json"]
)
def test_load_summary_rejects_unbound_consumed_output(source_fixture, source):
    run, audit = source_fixture
    complete_path = run / "completed_manifest.json"
    complete = REPORT.read(complete_path)
    complete["output_sha256"].pop(source)
    write(complete_path, complete)
    review = REPORT.read(audit)
    review["source_sha256"][str(complete_path)] = REPORT.digest(complete_path)
    write(audit, review)
    with pytest.raises(ValueError):
        REPORT.load_summary(run, audit)


def test_load_summary_rejects_changed_bound_output_bytes(source_fixture):
    run, audit = source_fixture
    path = run / "llama/fold_0/results.json"
    value = REPORT.read(path)
    value["methods"]["direct_lr"]["selected_parameters"]["C"] = 100.0
    write(path, value)
    with pytest.raises(ValueError):
        REPORT.load_summary(run, audit)


@pytest.mark.parametrize(
    "corruption", ["primary_cells", "half_primary_half_secondary", "fold_rows"]
)
def test_compact_method_rejects_cross_metric_arithmetic_contradictions(corruption):
    record = method_record()
    if corruption == "primary_cells":
        # Coherent averages still contradict the saved 4/4 quartet cells.
        for item in [record["summary"], *record["groups"], *record["fold_values"]]:
            item["primary_row_accuracy"] = 0.5
    elif corruption == "half_primary_half_secondary":
        # The primary and secondary subsets each contain half the rows.
        for item in [record["summary"], *record["groups"], *record["fold_values"]]:
            item["row_accuracy"] = 0.75
    else:
        record["fold_values"][0]["row_accuracy"] = 0.5
    with pytest.raises(ValueError):
        REPORT.compact_method(record)


@pytest.mark.parametrize(
    "native,logical",
    [
        (
            "//wsl.localhost/Ubuntu/home/user/run/completed_manifest.json",
            "/home/user/run/completed_manifest.json",
        ),
        ("C:/Users/user/project/audit.json", "/mnt/c/Users/user/project/audit.json"),
        ("/prefix/wsl.localhost/Ubuntu/home/run", "/prefix/wsl.localhost/Ubuntu/home/run"),
    ],
)
def test_logical_source_path_mapping_normalizes_only_known_prefixes(native, logical):
    assert REPORT.logical_path(native) == logical
