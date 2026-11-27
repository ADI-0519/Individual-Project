"""Synthetic export evidence, immutable inputs and audit/source binding guards."""

from __future__ import annotations

import copy
import gzip
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

from latent_probes.storage import file_sha256

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/report_format_adaptation_text.py"
SPEC = importlib.util.spec_from_file_location("lexical_control_report_test", SCRIPT)
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2) + "\n").encode("utf-8")
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)


def panel(correct=12):
    return {
        "complete_quartets": correct,
        "quartets": 192,
        "quartet_accuracy": correct / 192,
        "rows": 768,
        "correct": 400,
        "row_accuracy": 400 / 768,
        "class_recalls": {"0": 0.4, "1": 0.6},
        "nuisance_pairs": {
            "world": {"pairs": 384, "verdict_flips": 10, "both_correct": 200},
            "name_map": {"pairs": 384, "verdict_flips": 210, "both_correct": 70},
        },
    }


def fixture(tmp_path, monkeypatch):
    source, audit, latent, output = (
        tmp_path / name for name in ("source", "audit", "latent", "report")
    )
    source.mkdir()
    audit.mkdir()
    monkeypatch.setattr(reporter, "verify_pipeline", lambda source: {"status": "pass"})
    put(
        source / "preparation.json",
        {
            "schema": "adaptation-lexical-preparation-v1",
            "test_fixture": True,
            "human_review_complete": False,
            "source_sha256": {reporter.AUDITOR: file_sha256(reporter.ROOT / reporter.AUDITOR)},
        },
    )
    views = {}
    for view in reporter.VIEWS:
        views[view] = {"arms": {}, "factor_diagnostics": {}, "comparisons": {}}
        for arm in reporter.ARMS:
            views[view]["arms"][arm] = {
                fmt + "/" + condition: {
                    "aggregate": panel(),
                    "groups": ["all fake contexts retained"],
                }
                for fmt in ("A", "B")
                for condition in ("one_step", "correction_easy", "correction_hard", "asserted")
            }
            views[view]["factor_diagnostics"][arm] = {
                fmt + "/" + condition: {
                    "aggregate": {"factors": {"B1": {"raw_auroc": 0.75}, "B2": {"raw_auroc": 0.8}}},
                    "groups": ["fake factor groups"],
                }
                for fmt in ("A", "B")
                for condition in ("correction_all", "correction_easy", "correction_hard")
            }
    summary = {
        "schema": "adaptation-lexical-summary-v1",
        "research_status": "posthoc_exposed_bank_diagnostic",
        "human_review_complete": False,
        "shared_text_comparator_not_checkpoint_replications": True,
        "new_language_model_forwards": 0,
        "views": views,
    }
    metrics = {
        **{k: v for k, v in summary.items() if k != "views"},
        "views": {
            view: {
                "arms": {
                    arm: {scope: data["aggregate"] for scope, data in scopes.items()}
                    for arm, scopes in result["arms"].items()
                },
                "factor_diagnostics": {
                    arm: {scope: data["aggregate"] for scope, data in scopes.items()}
                    for arm, scopes in result["factor_diagnostics"].items()
                },
                "comparisons": result["comparisons"],
            }
            for view, result in views.items()
        },
    }
    put(source / "summary.json.gz", summary)
    put(source / "metrics.json", metrics)
    put(source / "rows.json.gz", {"synthetic": "metadata"})
    (source / "predictions.npz").write_bytes(b"fake prediction bytes; no scientific arrays")
    for name in ("analysis_complete.json", "fit_started.json", "analysis_started.json"):
        put(source / name, {"synthetic": True})
    for name in reporter.FIT_NAMES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("synthetic safe fixture " + name).encode())
    put(
        source / "fit_seal.json",
        {"files_sha256": {name: file_sha256(source / name) for name in reporter.FIT_NAMES}},
    )
    reconstruction_names = [view + "_reconstruction.json" for view in reporter.VIEWS]
    for view, name in zip(reporter.VIEWS, reconstruction_names, strict=True):
        put(audit / name, views[view])
    put(
        audit / "reconstruction_seal.json",
        {
            "schema": "format-adaptation-text-metric-reconstruction-seal-v1",
            "saved_summary_opened": False,
            "views_reconstructed": list(reporter.VIEWS),
            "output_sha256": {name: file_sha256(audit / name) for name in reconstruction_names},
        },
    )
    put(
        audit / "audit_receipt.json",
        {
            "schema": "format-adaptation-text-independent-metric-audit-v1",
            "status": "pass",
            "source": str(source.resolve()),
            "model_calls": 0,
            "classifier_fits": 0,
            "input_sha256": {
                name: file_sha256(source / name)
                for name in ("rows.json.gz", "summary.json.gz", "predictions.npz")
            },
            "auditor_sha256": file_sha256(reporter.ROOT / reporter.AUDITOR),
            "views": {
                view: {"numeric": 1, "exact": 1, "maximum_numeric_difference": 0.0}
                for view in reporter.VIEWS
            },
        },
    )
    put(
        latent / "summary.json",
        {
            "schema": "format-adaptation-report-v1",
            "test_fixture": True,
            "models": {
                model: {"arms": copy.deepcopy(views["full_messages"]["arms"])}
                for model in ("llama", "qwen")
            },
        },
    )
    put(
        latent / "manifest.json",
        {
            "schema": "format-adaptation-report-manifest-v1",
            "status": "complete",
            "output_sha256": {"summary.json": file_sha256(latent / "summary.json")},
        },
    )
    return source, audit, latent, output


def export(paths, **kwargs):
    source, audit, latent, output = paths
    return reporter.report(
        source, audit, output, latent_root=latent, allow_test_fixture=True, **kwargs
    )


def test_export_preserves_all_original_bytes_and_controlled_archive(tmp_path, monkeypatch):
    paths = fixture(tmp_path, monkeypatch)
    source, audit, _, output = paths
    manifest = export(paths)
    for name in reporter.SOURCE_NAMES:
        assert (output / name).read_bytes() == (source / name).read_bytes()
    for name in ("audit_receipt.json", "reconstruction_seal.json"):
        assert (output / "audit" / name).read_bytes() == (audit / name).read_bytes()
    for view in reporter.VIEWS:
        name = view + "_reconstruction.json"
        assert (
            gzip.decompress((output / "audit" / (name + ".gz")).read_bytes())
            == (audit / name).read_bytes()
        )
    with zipfile.ZipFile(output / "fitted-text-readouts.zip") as archive:
        assert archive.namelist() == list(reporter.FIT_NAMES)
        for name in archive.namelist():
            assert archive.read(name) == (source / name).read_bytes()
    assert manifest["new_language_model_forwards"] == manifest["classifier_fits"] == 0
    for name, digest in manifest["output_sha256"].items():
        assert file_sha256(output / name) == digest
    markdown = (output / "README.md").read_text(encoding="utf-8")
    assert "TEST FIXTURE" in markdown
    assert "Text full_messages / A" in markdown and "Text system_only / B" in markdown
    assert "Previous latent llama / A" in markdown and "Previous latent qwen / B" in markdown
    assert "World flips / pairs" in markdown and "Name-map flips / pairs" in markdown
    assert "not a capable semantic judge" in markdown


def test_production_refuses_fixture_and_output_overwrite(tmp_path, monkeypatch):
    paths = fixture(tmp_path, monkeypatch)
    source, audit, latent, output = paths
    with pytest.raises(ValueError, match="Unauthorized fixture"):
        reporter.report(source, audit, output, latent_root=latent)
    export(paths)
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        export(paths)
    assert (output / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("damage", ("receipt", "audited_bytes", "auditor", "reconstruction", "fit"))
def test_changed_audit_or_fit_is_rejected_before_report_creation(tmp_path, monkeypatch, damage):
    paths = fixture(tmp_path, monkeypatch)
    source, audit, _, output = paths
    if damage == "receipt":
        path = audit / "audit_receipt.json"
        value = reporter.read_json(path)
        value["status"] = "failed_preserved"
        put(path, value)
    elif damage == "audited_bytes":
        (source / "predictions.npz").write_bytes(b"changed")
    elif damage == "auditor":
        path = audit / "audit_receipt.json"
        value = reporter.read_json(path)
        value["auditor_sha256"] = "unbound replacement"
        put(path, value)
    elif damage == "reconstruction":
        (audit / "full_messages_reconstruction.json").write_text("changed")
    else:
        (source / reporter.FIT_NAMES[0]).write_bytes(b"changed fit")
    with pytest.raises(ValueError):
        export(paths)
    assert not output.exists()


def test_compact_display_cannot_disagree_with_audited_full_summary(tmp_path, monkeypatch):
    paths = fixture(tmp_path, monkeypatch)
    source, _, _, output = paths
    metrics = reporter.read_json(source / "metrics.json")
    metrics["views"]["full_messages"]["arms"]["Mixed512"]["B/correction_hard"]["correct"] = 700
    put(source / "metrics.json", metrics)
    with pytest.raises(ValueError, match="Compact display"):
        export(paths)
    assert not output.exists()


def test_archive_inventory_and_latent_source_are_bound(tmp_path, monkeypatch):
    paths = fixture(tmp_path, monkeypatch)
    source, _, latent, output = paths
    seal = reporter.read_json(source / "fit_seal.json")
    seal["files_sha256"]["../unsafe.py"] = "invalid"
    put(source / "fit_seal.json", seal)
    with pytest.raises(ValueError, match="archive inventory"):
        export(paths)
    assert not output.exists()
    del seal["files_sha256"]["../unsafe.py"]
    put(source / "fit_seal.json", seal)
    (latent / "summary.json").write_text("changed")
    with pytest.raises(ValueError, match="Latent display summary"):
        export(paths)


def test_sealed_pipeline_failure_prevents_export(tmp_path, monkeypatch):
    paths = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(reporter, "verify_pipeline", lambda source: {"status": "fail"})
    with pytest.raises(ValueError, match="pipeline verification"):
        export(paths)
    assert not paths[3].exists()


def test_optional_figure_is_copied_and_hash_bound(tmp_path, monkeypatch):
    paths = fixture(tmp_path, monkeypatch)
    figure = tmp_path / "figure.svg"
    figure.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
    manifest = export(paths, figures=[figure])
    name = "figure/lexical-controls.svg"
    assert manifest["output_sha256"][name] == file_sha256(figure)
    assert (paths[3] / name).read_bytes() == figure.read_bytes()
