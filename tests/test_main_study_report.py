"""CPU-only checks of honest reporting; all invented scores are marked fixtures."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "report_main_study.py"
SPEC = importlib.util.spec_from_file_location("main_study_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def _groups(score):
    return [
        {"group_id": f"synthetic_{m}", "mechanism": m, "complete_quartet_accuracy": score}
        for m in report.MECHANISMS
    ]


def _analysis(label="Llama"):
    # These are arbitrary reporting fixtures, never activation/model measurements.
    return {
        "schema": "main-study-analysis-v1",
        "test_fixture": True,
        "scientific_result": False,
        "real_empirical_result": False,
        "research_status": "agent_audited_exploratory",
        "human_review_complete": False,
        "confirmatory_result": False,
        "capture_provenance": {
            "model_settings": {"model_id": report.MODEL_IDS[label]},
            "bank_sha256": "a" * 64,
        },
        "primary": {
            "status": "complete",
            "restricted": 0.5,
            "full": 0.75,
            "full_minus_restricted": 0.25,
            "percentile_95_interval": [0.1, 0.4],
            "independent_scenario_groups": 1,
            "allocation_seeds": list(report.SEEDS),
        },
        "comparisons": [
            {
                "feature_kind": kind,
                "training_n": n,
                "allocation_seed": seed,
                "status": "complete",
                "arms": {"restricted": {"groups": _groups(0.5)}, "full": {"groups": _groups(0.75)}},
            }
            for kind in ("activation", "tfidf")
            for n in report.SIZES
            for seed in report.SEEDS
        ],
        "baselines": {
            "symbolic_oracle": _groups(1),
            "initial": _groups(0),
            "output_raw": _groups(0.25),
        },
        "prose_monitor": {"status": "not_provided"},
    }


def _failed(data, *, kind="activation", n=512, seed=37):
    record = next(
        r
        for r in data["comparisons"]
        if (r["feature_kind"], r["training_n"], r["allocation_seed"]) == (kind, n, seed)
    )
    record.pop("arms")
    record["status"] = "all_candidates_failed_to_converge"
    if kind == "activation" and n == 512:
        data["primary"] = {
            "status": "unavailable_missing_or_failed_primary_comparisons",
            "available_allocation_seeds": [s for s in report.SEEDS if s != seed],
            "missing_allocation_seeds": [seed],
        }
    return data


def test_fixture_refused_by_default():
    with pytest.raises(ValueError, match="Fixture"):
        report.validate_analysis(_analysis(), "Llama")


@pytest.mark.parametrize(
    "key,value",
    [
        ("test_fixture", "false"),
        ("scientific_result", True),
        ("real_empirical_result", True),
        ("confirmatory_result", True),
        ("human_review_complete", True),
        ("research_status", None),
    ],
)
def test_contradictory_provenance_refused(key, value):
    data = _analysis()
    data[key] = value
    with pytest.raises(ValueError):
        report.validate_analysis(data, "Llama", allow_fixture=True)


def test_complete_inventory_including_failures_required():
    data = _analysis()
    data["comparisons"].pop()
    with pytest.raises(ValueError, match="thirty"):
        report.validate_analysis(data, "Llama", allow_fixture=True)


def test_duplicate_allocation_refused():
    data = _analysis()
    data["comparisons"][-1] = copy.deepcopy(data["comparisons"][0])
    with pytest.raises(ValueError, match="duplicate"):
        report.validate_analysis(data, "Llama", allow_fixture=True)


def test_group_omission_refused_even_for_one_method():
    data = _analysis()
    data["comparisons"][0]["arms"]["restricted"]["groups"].pop()
    with pytest.raises(ValueError, match="same scenario"):
        report.validate_analysis(data, "Llama", allow_fixture=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("restricted", 0.7),
        ("full_minus_restricted", 0.8),
        ("percentile_95_interval", [0.5, -0.1]),
        ("independent_scenario_groups", 999),
    ],
)
def test_misleading_primary_summary_refused(field, value):
    data = _analysis()
    data["primary"][field] = value
    with pytest.raises(ValueError):
        report.validate_analysis(data, "Llama", allow_fixture=True)


def test_nonfinite_score_refused():
    data = _analysis()
    data["baselines"]["initial"][0]["complete_quartet_accuracy"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        report.validate_analysis(data, "Llama", allow_fixture=True)


def test_failed_primary_not_averaged_over_survivors():
    data = _failed(_analysis())
    report.validate_analysis(data, "Llama", allow_fixture=True)
    summary = report.summarize(data, "Llama")
    row = summary["learning_and_mechanisms"]["activation:512:correction"]
    assert row["complete_allocations"] == 4
    assert row["restricted"] is None and row["full"] is None
    text = report.render_markdown([summary], data)
    assert "UNAVAILABLE" in text and "4/5" in text
    assert "allocation 37: all_candidates_failed_to_converge" in text


def test_lying_complete_primary_with_failed_allocation_refused():
    data = _analysis()
    complete = copy.deepcopy(data["primary"])
    _failed(data)
    data["primary"] = complete
    with pytest.raises(ValueError, match="failed allocation"):
        report.validate_analysis(data, "Llama", allow_fixture=True)


def test_missing_prose_and_output_baselines_are_conspicuous():
    data = _analysis()
    text = report.render_markdown([report.summarize(data, "Llama")], data)
    assert "PROSE MONITOR MISSING" in text
    assert "| Llama | output_calibrated | UNAVAILABLE" in text
    assert "SYNTHETIC FIXTURE" in text
    assert "An interval including zero does not demonstrate equivalence" in text


def test_monitor_invalid_denominator_retained():
    data = _analysis()
    groups = _groups(0.25)
    for group, invalid in zip(groups, (2, 2, 3), strict=True):
        group.update(rows=64 if group["mechanism"] == "correction" else 32, invalid=invalid)
    data["prose_monitor"] = {
        "status": "complete",
        "test_fixture": True,
        "research_status": data["research_status"],
        "human_review_complete": False,
        "bank_sha256": "a" * 64,
        "partition": "test",
        "groups": groups,
        "overall": {"invalid": 7, "rows": 128},
    }
    report.validate_analysis(data, "Llama", allow_fixture=True)
    text = report.render_markdown([report.summarize(data, "Llama")], data)
    assert "7/128" in text and "counted incorrect" in text
    assert "PROSE MONITOR MISSING" not in text
    data["prose_monitor"]["overall"]["invalid"] = 0
    with pytest.raises(ValueError, match="denominators differ"):
        report.validate_analysis(data, "Llama", allow_fixture=True)


def test_saved_calibration_reported_separately_without_changing_primary():
    data = _analysis()
    for comparison in data["comparisons"]:
        for arm in report.ARMS:
            comparison["arms"][arm]["calibrated_groups"] = _groups(0.875)
    report.validate_analysis(data, "Llama", allow_fixture=True)
    summary = report.summarize(data, "Llama")
    assert summary["primary"]["full"] == 0.75
    assert summary["baselines"]["activation_full_calibrated_N512"]["correction"] == 0.875
    text = report.render_markdown([summary], data)
    assert "| Llama | activation_full_calibrated_N512 | 87.5 | 87.5 | 87.5 |" in text
    assert "| Llama | 37 | complete | 50.0 | 75.0 | +25.0 |" in text


def test_incomplete_calibration_not_averaged_over_survivors():
    data = _analysis()
    for comparison in data["comparisons"]:
        if comparison["allocation_seed"] != 37:
            comparison["arms"]["full"]["calibrated_groups"] = _groups(0.875)
    summary = report.summarize(data, "Llama")
    assert "activation_full_calibrated_N512" not in summary["baselines"]


def test_complete_comparisons_cannot_hide_primary_as_unavailable():
    data = _analysis()
    data["primary"] = {
        "status": "unavailable",
        "available_allocation_seeds": list(report.SEEDS),
        "missing_allocation_seeds": [],
    }
    with pytest.raises(ValueError, match="Unavailable primary"):
        report.validate_analysis(data, "Llama", allow_fixture=True)


@pytest.mark.parametrize("path", ["relative/report", "/mnt/c/research/report"])
def test_output_must_be_native_and_absolute(path):
    if path.startswith("/mnt/") and report.os.name != "posix":
        pytest.skip("Linux mount policy")
    with pytest.raises(ValueError):
        report._output_path(path)


def test_prior_report_is_preserved(tmp_path):
    existing = tmp_path / "prior"
    existing.mkdir()
    (existing / "keep.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        report.write_report("missing-a", "missing-b", existing, allow_fixture=True)
    assert (existing / "keep.txt").read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("mismatch", ["bank", "model", "fixture"])
def test_two_input_provenance_must_match(tmp_path, mismatch):
    first, second = _analysis("Llama"), _analysis("Qwen")
    if mismatch == "bank":
        second["capture_provenance"]["bank_sha256"] = "b" * 64
    elif mismatch == "model":
        second["capture_provenance"]["model_settings"]["model_id"] = report.MODEL_IDS["Llama"]
    else:
        second["test_fixture"] = False
    paths = []
    for i, data in enumerate((first, second)):
        path = tmp_path / f"synthetic_{i}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        paths.append(path)
    output = tmp_path / "unused"
    with pytest.raises(ValueError):
        report.write_report(*paths, output, allow_fixture=True)
    assert not output.exists()


def test_real_provenance_cannot_reuse_tiny_fixture_bank():
    data = _analysis()
    data.update(
        test_fixture=False,
        scientific_result=True,
        real_empirical_result=True,
        fit_settings_sha256="b" * 64,
        fit_heads_sha256="c" * 64,
    )
    with pytest.raises(ValueError, match="ninety groups"):
        report.validate_analysis(data, "Llama")


def test_figures_and_manifest_are_marked_fixtures(tmp_path):
    pytest.importorskip("matplotlib")
    paths = []
    for label in report.MODEL_IDS:
        path = tmp_path / f"synthetic_{label}.json"
        path.write_text(json.dumps(_analysis(label)), encoding="utf-8")
        paths.append(path)
    output = tmp_path / "fixture_report"
    manifest = report.write_report(*paths, output, allow_fixture=True)
    expected = {
        "results.md",
        "results_summary.json",
        "report_manifest.json",
        "primary.svg",
        "primary.pdf",
        "primary.png",
        "learning_curves.svg",
        "learning_curves.pdf",
        "learning_curves.png",
    }
    assert {p.name for p in output.iterdir()} == expected
    assert manifest["test_fixture"] is True
    for item in manifest["inputs"]:
        assert report._hash(Path(item["path"])) == item["sha256"]
    for name, digest in manifest["outputs"].items():
        assert report._hash(output / name) == digest
    svg = (output / "primary.svg").read_text(encoding="utf-8")
    assert "SYNTHETIC FIXTURE" in svg and "1 correction groups" in svg
    assert "5/5 allocations" in svg
    with pytest.raises(FileExistsError):
        report.write_report(*paths, output, allow_fixture=True)


def test_input_mutation_prevents_completion_manifest(tmp_path, monkeypatch):
    paths = []
    for label in report.MODEL_IDS:
        path = tmp_path / f"synthetic_{label}.json"
        path.write_text(json.dumps(_analysis(label)), encoding="utf-8")
        paths.append(path)

    def mutate_input(*args):
        paths[0].write_text("changed", encoding="utf-8")

    # The optional plot extra is absent in lightweight CI; fake only that import,
    # never mathematical data or provenance validation.
    import sys
    import types

    monkeypatch.setitem(sys.modules, "matplotlib", types.ModuleType("matplotlib"))
    monkeypatch.setattr(report, "_plot", mutate_input)
    output = tmp_path / "partial_fixture_report"
    with pytest.raises(RuntimeError, match="changed during"):
        report.write_report(*paths, output, allow_fixture=True)
    assert not (output / "report_manifest.json").exists()
