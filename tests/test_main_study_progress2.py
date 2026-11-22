"""Provenance refusals and transparently synthetic four-slide layout checks."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_main_study_progress2.py"
SPEC = importlib.util.spec_from_file_location("main_study_progress2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def _metadata():
    # Synthetic metadata for schema tests, not real empirical measurements.
    return {
        "schema": "main-study-report-v1",
        "test_fixture": False,
        "research_status": "agent_audited_exploratory",
        "human_review_complete": False,
        "confirmatory_result": False,
        "models": [
            {
                "model": name,
                "model_id": identity,
                "primary": {
                    "status": "complete",
                    "allocation_seeds": builder.SEEDS,
                    "independent_scenario_groups": 30,
                    "restricted": 0.02,
                    "full": 0.01,
                    "full_minus_restricted": -0.01,
                    "percentile_95_interval": [-0.03, 0.01],
                },
                "learning_and_mechanisms": {
                    f"activation:512:{m}": {
                        "scenario_groups": 30,
                        "complete_allocations": 5,
                        "planned_allocations": 5,
                        "restricted": 0.02,
                        "full": 0.01,
                    }
                    for m in ("observation", "message", "correction")
                },
                "prose_monitor": {"status": "not_provided"},
            }
            for name, identity in builder.MODEL_IDS.items()
        ],
    }


def _save(tmp_path, data=None):
    data = _metadata() if data is None else data
    summary = tmp_path / "results_summary.json"
    summary.write_text(json.dumps(data), encoding="utf-8")
    manifest = {
        key: data[key]
        for key in (
            "test_fixture",
            "research_status",
            "human_review_complete",
            "confirmatory_result",
        )
    }
    manifest.update(
        schema="main-study-report-manifest-v1",
        bank_sha256="a" * 64,
        outputs={summary.name: hashlib.sha256(summary.read_bytes()).hexdigest()},
        inputs=[
            {"path": "synthetic-llama", "sha256": "b" * 64},
            {"path": "synthetic-qwen", "sha256": "c" * 64},
        ],
        report_script_sha256=builder._hash(SCRIPT.with_name("report_main_study.py")),
    )
    path = tmp_path / "report_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return summary, path


def test_summary_hash_must_match(tmp_path):
    summary, manifest = _save(tmp_path)
    summary.write_text(summary.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="Summary hash"):
        builder.load_sources(summary, manifest)


def test_valid_bound_metadata_loads(tmp_path):
    summary, manifest = _save(tmp_path)
    data, bindings = builder.load_sources(summary, manifest)
    assert {m["model"] for m in data["models"]} == {"Llama", "Qwen"}
    assert bindings["report_manifest"]["sha256"] == builder._hash(manifest)


@pytest.mark.parametrize("mutation", ["reporter", "input_hash"])
def test_reporter_and_source_hashes_required(tmp_path, mutation):
    summary, manifest = _save(tmp_path)
    data = json.loads(manifest.read_bytes())
    if mutation == "reporter":
        data["report_script_sha256"] = "d" * 64
    else:
        data["inputs"][0]["sha256"] = "missing"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        builder.load_sources(summary, manifest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("test_fixture", True),
        ("test_fixture", "false"),
        ("human_review_complete", True),
        ("research_status", "unreviewed"),
        ("confirmatory_result", True),
    ],
)
def test_public_loader_refuses_provenance_overclaims(tmp_path, field, value):
    data = _metadata()
    data[field] = value
    with pytest.raises(ValueError, match="Only real"):
        builder.load_sources(*_save(tmp_path, data))


@pytest.mark.parametrize("mutation", ["model", "duplicate", "count", "seeds", "delta", "interval"])
def test_wrong_or_incomplete_results_refused(tmp_path, mutation):
    data = _metadata()
    first = data["models"][0]
    if mutation == "model":
        first["model_id"] = "wrong-checkpoint"
    elif mutation == "duplicate":
        data["models"][1] = copy.deepcopy(first)
    elif mutation == "count":
        first["primary"]["independent_scenario_groups"] = 31
    elif mutation == "seeds":
        first["primary"]["allocation_seeds"] = [11, 23]
    elif mutation == "delta":
        first["primary"]["full_minus_restricted"] = 0.75
    else:
        first["primary"]["percentile_95_interval"] = [0.1, -0.1]
    with pytest.raises(ValueError):
        builder.load_sources(*_save(tmp_path, data))


def test_existing_output_protected_before_inputs_read(tmp_path):
    with pytest.raises(FileExistsError):
        builder.build(tmp_path / "missing", tmp_path / "also-missing", tmp_path)


def test_layout_exactly_four_and_fixture_visible(tmp_path):
    pytest.importorskip("pptx")
    pytest.importorskip("reportlab")
    if not Path("C:/Windows/Fonts/arial.ttf").exists():
        pytest.skip("Original editable deck's Arial fonts are unavailable")
    data = _metadata()
    data["test_fixture"] = True
    slides = builder._slides(data, {"fixture": "synthetic layout test only"})
    assert len(slides) == 4
    text = "\n".join(
        " ".join(obj[1]) for slide in slides for obj in slide.objects if obj[0] == "text"
    )
    assert "SYNTHETIC FIXTURE" in text
    assert "Human semantic review incomplete" in text
    assert "11,776" not in text and "154" not in text
    assert "prior belief differs from incoming information" in text
    assert "90 test contexts/model" in text
    assert all(slide.notes for slide in slides)
    template_hash = builder._hash(builder.TEMPLATE)
    pptx, pdf = tmp_path / "synthetic-fixture-only.pptx", tmp_path / "synthetic-fixture-only.pdf"
    builder._layout()._write_deck(builder.TEMPLATE, pptx, pdf, slides, builder.PROJECT)
    assert builder._hash(builder.TEMPLATE) == template_hash
    from pptx import Presentation

    deck = Presentation(pptx)
    assert len(deck.slides) == 4
    assert all(s.notes_slide.notes_text_frame.text for s in deck.slides)
    with zipfile.ZipFile(pptx) as archive:
        xml = "\n".join(
            archive.read(n).decode()
            for n in archive.namelist()
            if n.startswith("ppt/slides/slide") and n.endswith(".xml")
        )
    assert "SYNTHETIC FIXTURE" in xml
    assert pdf.read_bytes().startswith(b"%PDF")


def test_generated_notes_and_receipt_use_portable_bound_lf_bytes(tmp_path):
    """Re-render saved real report data; no inference or invented model values."""
    pytest.importorskip("pptx")
    pytest.importorskip("reportlab")
    if not Path("C:/Windows/Fonts/arial.ttf").exists():
        pytest.skip("Original editable deck's Arial fonts are unavailable")
    report = SCRIPT.parents[1] / "docs" / "results" / "main-study-2026-11-22-interim"
    output = tmp_path / "portable-deck"
    receipt = builder.build(
        report / "results_summary.json", report / "report_manifest.json", output
    )
    for name in ("speaker-notes.md", "source-receipt.json"):
        raw = (output / name).read_bytes()
        assert b"\r\n" not in raw, name
        assert raw.endswith(b"\n") and not raw.endswith(b"\n\n"), name
    notes_sha = hashlib.sha256((output / "speaker-notes.md").read_bytes()).hexdigest()
    assert receipt["outputs"]["speaker-notes.md"] == notes_sha
    saved_receipt = json.loads((output / "source-receipt.json").read_bytes())
    assert saved_receipt["outputs"]["speaker-notes.md"] == notes_sha
