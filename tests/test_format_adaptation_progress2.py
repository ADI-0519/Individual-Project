"""Reporting-schema refusals and explicitly synthetic four-slide rendering."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_format_adaptation_progress2.py"
SPEC = importlib.util.spec_from_file_location("format_adaptation_progress2", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


def metadata(fixture=False):
    # Schema fixtures only; no real result file is read to author these values.
    models = {}
    for key in ("llama", "qwen"):
        model = {
            "schema": "format-adaptation-analysis-v1",
            "model_key": key,
            "site": builder.SITES[key],
            "test_fixture": fixture,
            "confirmatory": False,
            "research_status": "agent_audited_exploratory",
            "arms": {},
        }
        b_counts = {"A512": 4, "Mixed512": 3 if key == "llama" else 5, "Ahalf256": 2, "Offset": 5}
        for arm in builder.ARMS:
            model["arms"][arm] = {}
            for fmt in ("A", "B"):
                complete = 6 if fmt == "A" else b_counts[arm]
                groups = [
                    {
                        "group_id": f"fixture_context_{i:02d}",
                        "rows": 32,
                        "quartets": 8,
                        "complete_quartets": complete,
                        "quartet_accuracy": complete / 8,
                        "correct": 4 * complete,
                        "row_accuracy": complete / 8,
                    }
                    for i in range(24)
                ]
                model["arms"][arm][fmt + "/correction_hard"] = {
                    "groups": groups,
                    "aggregate": {
                        "rows": 768,
                        "quartets": 192,
                        "complete_quartets": complete * 24,
                        "quartet_accuracy": complete / 8,
                        "correct": complete * 96,
                        "row_accuracy": complete / 8,
                    },
                }
        for name, comparator in (("primary", "A512"), ("named_diagnostic", "Offset")):
            a, b = b_counts["Mixed512"] / 8, b_counts[comparator] / 8
            model[name] = {
                "contrast": "Mixed512_minus_" + comparator,
                "groups": 24,
                "quartets": 192,
                "bootstrap_samples": 10000,
                "bootstrap_seed": 20261127,
                "renderer": "B",
                "condition": "correction_hard",
                "Mixed512": a,
                comparator: b,
                "difference": a - b,
                "conditional_percentile_interval_95": [a - b - 0.05, a - b + 0.05],
                "group_values": [
                    {
                        "group_id": f"fixture_context_{i:02d}",
                        "Mixed512": a,
                        comparator: b,
                        "difference": a - b,
                    }
                    for i in range(24)
                ],
            }
        model["primary"]["role"] = "primary" if key == "llama" else "named_secondary_replication"
        models[key] = model
    return {
        "schema": "format-adaptation-report-v1",
        "test_fixture": fixture,
        "human_review_complete": False,
        "research_status": "agent_audited_exploratory",
        "new_report_model_calls": 0,
        "classifier_fits": 0,
        "models": models,
    }


def save(tmp_path, data=None, audit_mutation=None):
    data = copy.deepcopy(data or metadata())
    campaign = "/native/synthetic-campaign"
    audited = {}
    for key, model in data["models"].items():
        audited[campaign + "/" + key + "_analysis/results.json"] = hashlib.sha256(
            builder.analysis_bytes(model)
        ).hexdigest()
        audited[campaign + "/" + key + "_analysis/predictions.npz"] = "b" * 64
    audit = {
        "schema": "format-adaptation-independent-audit-v1",
        "status": "pass",
        "campaign": campaign,
        "model_calls": 0,
        "classifier_fits": 0,
        "models": {key: {} for key in ("llama", "qwen")},
        "saved_analysis_opened_before_both_reconstructions_frozen": False,
        "analysis_output_sha256": audited,
    }
    if audit_mutation:
        audit_mutation(audit)
    audit_path = tmp_path / "audit_receipt.json"
    write(audit_path, audit)
    inputs = {
        campaign + "/" + key + "_analysis/results.json": value
        for key, value in (
            (key, audited[campaign + "/" + key + "_analysis/results.json"])
            for key in data["models"]
        )
    }
    inputs["/native/synthetic-audit/audit_receipt.json"] = builder.file_hash(audit_path)
    data["input_sha256"] = inputs
    summary = tmp_path / "summary.json"
    write(summary, data)
    manifest = tmp_path / "manifest.json"
    write(
        manifest,
        {
            "schema": "format-adaptation-report-manifest-v1",
            "status": "complete",
            "source_sha256": builder.file_hash(builder.REPORTER),
            "input_sha256": inputs,
            "output_sha256": {"summary.json": builder.file_hash(summary)},
        },
    )
    return summary, manifest, audit_path


def test_completed_bound_report_loads_and_survives_copy(tmp_path):
    sources = save(tmp_path / "original")
    data, bindings = builder.load_sources(*sources)
    assert data["models"]["llama"]["primary"]["difference"] < 0
    assert bindings["campaign_identity"]["path"] == "/native/synthetic-campaign"
    copied = tmp_path / "copied"
    copied.mkdir()
    new_sources = []
    for source in sources:
        destination = copied / source.name
        shutil.copyfile(source, destination)
        new_sources.append(destination)
    assert builder.load_sources(*new_sources)[0] == data


@pytest.mark.parametrize(
    "mutation", ["count", "groups", "interval", "difference", "role", "model", "nan"]
)
def test_rehashed_but_invalid_scientific_metadata_refused(tmp_path, mutation):
    data = metadata()
    first = data["models"]["llama"]
    if mutation == "count":
        first["arms"]["Mixed512"]["B/correction_hard"]["aggregate"]["quartets"] = 191
    elif mutation == "groups":
        first["arms"]["Offset"]["A/correction_hard"]["groups"].pop()
    elif mutation == "interval":
        first["primary"]["conditional_percentile_interval_95"] = [0.1, -0.1]
    elif mutation == "difference":
        first["primary"]["difference"] = 0.5
    elif mutation == "role":
        data["models"]["qwen"]["primary"]["role"] = "primary"
    elif mutation == "model":
        first["model_key"] = "qwen"
    else:
        first["primary"]["difference"] = float("nan")
    with pytest.raises(ValueError):
        builder.load_sources(*save(tmp_path, data))


@pytest.mark.parametrize(
    "mutation", ["failed_audit", "wrong_campaign", "missing_model", "early_open", "unaudited"]
)
def test_nonpassing_or_misbound_audit_refused(tmp_path, mutation):
    def damage(audit):
        if mutation == "failed_audit":
            audit["status"] = "failed_preserved"
        elif mutation == "wrong_campaign":
            audit["campaign"] = "/native/another-campaign"
        elif mutation == "missing_model":
            audit["models"].pop("qwen")
        elif mutation == "early_open":
            audit["saved_analysis_opened_before_both_reconstructions_frozen"] = True
        else:
            audit["analysis_output_sha256"][
                "/native/synthetic-campaign/llama_analysis/results.json"
            ] = "0" * 64

    with pytest.raises(ValueError):
        builder.load_sources(*save(tmp_path, audit_mutation=damage))


@pytest.mark.parametrize("target", ["summary", "manifest", "audit"])
def test_byte_hash_mismatches_refused(tmp_path, target):
    paths = save(tmp_path)
    index = {"summary": 0, "manifest": 1, "audit": 2}[target]
    if target == "manifest":
        data = json.loads(paths[index].read_text())
        data["source_sha256"] = "0" * 64
        write(paths[index], data)
    else:
        paths[index].write_bytes(paths[index].read_bytes() + b" ")
    with pytest.raises(ValueError):
        builder.load_sources(*paths)


def test_public_build_and_cli_refuse_fixtures_before_renderer(tmp_path):
    paths = save(tmp_path / "inputs", metadata(fixture=True))
    with pytest.raises(ValueError, match="Only real"):
        builder.build(*paths, tmp_path / "refused")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--summary",
            str(paths[0]),
            "--manifest",
            str(paths[1]),
            "--audit",
            str(paths[2]),
            "--output",
            str(tmp_path / "also-refused"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "Only real" in result.stderr
    assert not (tmp_path / "also-refused").exists()


def test_existing_deck_refused_before_reading_inputs(tmp_path):
    with pytest.raises(FileExistsError):
        builder.build(
            tmp_path / "missing-summary",
            tmp_path / "missing-manifest",
            tmp_path / "missing-audit",
            tmp_path,
        )


def presentation_python():
    candidates = [Path(sys.executable), SCRIPT.parents[2] / ".venv/Scripts/python.exe"]
    for candidate in candidates:
        if candidate.is_file():
            probe = subprocess.run(
                [str(candidate), "-c", "import pptx,reportlab"], capture_output=True
            )
            if probe.returncode == 0:
                return candidate
    pytest.skip("Existing presentation environment is unavailable; no packages installed")


def test_synthetic_export_has_exactly_four_editable_slides_vector_pdf_and_bound_notes(tmp_path):
    interpreter = presentation_python()
    paths = save(tmp_path / "inputs", metadata(fixture=True))
    output = tmp_path / "synthetic-only"
    template_hash = builder.file_hash(builder.TEMPLATE)
    renderer_hash = builder.file_hash(builder.RENDERER)
    code = (
        "import importlib.util,sys; from pathlib import Path; "
        "s=importlib.util.spec_from_file_location('fixture_builder',sys.argv[1]); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "m._build(*[Path(p) for p in sys.argv[2:]],allow_fixture=True)"
    )
    result = subprocess.run(
        [str(interpreter), "-c", code, str(SCRIPT), *(str(p) for p in paths), str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((output / "source-receipt.json").read_text())
    assert receipt["test_fixture"] is True and receipt["slides"] == 4
    assert (
        receipt["model_calls"] == receipt["classifier_fits"] == 0
        and receipt["analysis_performed"] is False
    )
    for name, expected in receipt["outputs"].items():
        assert builder.file_hash(output / name) == expected
    assert builder.file_hash(builder.TEMPLATE) == template_hash
    assert builder.file_hash(builder.RENDERER) == renderer_hash
    with zipfile.ZipFile(output / "SYNTHETIC-FIXTURE-ONLY.pptx") as archive:
        names = sorted(
            n for n in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)
        )
        assert len(names) == 4
        xml = [archive.read(n).decode() for n in names]
        assert all("SYNTHETIC FIXTURE" in text and "<p:pic>" not in text for text in xml)
        assert all("<a:t>" in text for text in xml)
        assert "Aditya Ranjan" in xml[0]
        assert "72/192" in xml[1] and "120/192" in xml[1] and "-12.50" in xml[1]
        assert "Qwen (secondary)" in xml[1]
        assert "NF4" in xml[2] and "spontaneous anticipation" in xml[2]
        assert "CREATE BF16" in xml[3] and "Proposed, not performed" in xml[3]
        assert (
            len(
                [
                    n
                    for n in archive.namelist()
                    if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml", n)
                ]
            )
            == 4
        )
    pdf = (output / "SYNTHETIC-FIXTURE-ONLY.pdf").read_bytes()
    assert pdf.startswith(b"%PDF") and b"/Subtype /Image" not in pdf
    assert len(re.findall(rb"/Type\s*/Page\b", pdf)) == 4
    notes = (output / "speaker-notes.md").read_bytes()
    assert b"SYNTHETIC FIXTURE" in notes and b"\r\n" not in notes
    assert notes.endswith(b"\n") and not notes.endswith(b"\n\n")
    assert b"exclude uncertainty from resampling the fitting cohort" in notes
