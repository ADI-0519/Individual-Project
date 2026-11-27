"""CPU-only checks of the real coordinator's ordering, CLI and seal gates."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from latent_probes.format_adaptation_fit import MODEL_SITES
from latent_probes.storage import atomic_json, file_sha256, read_json

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


campaign = load_script("adaptation_campaign_test", "run_format_adaptation_campaign.py")


@pytest.fixture
def prepared(tmp_path):
    output = tmp_path / "campaign"
    output.mkdir()
    source = tmp_path / "frozen_policy.py"
    source.write_text("# Synthetic frozen policy\n", encoding="utf-8")
    sources = {str(source): file_sha256(source)}
    preparation = {"source_sha256": sources, "bank_sha256": "b" * 64}
    for key in ("llama", "qwen"):
        directory = output / key
        directory.mkdir()
        atomic_json(
            directory / "protocol.json",
            {
                "old_bundle": str(tmp_path / "old" / (key + "_fit")),
                "source_sha256": sources,
                "bank_sha256": preparation["bank_sha256"],
            },
        )
        atomic_json(
            directory / "fit/manifest.json",
            {
                "schema": "format-adaptation-capture-manifest-v1",
                "status": "complete",
                "partition": "fit",
                "model_key": key,
                "completed_rows": 768,
            },
        )
        bundle = output / (key + "_fit")
        bundle.mkdir()
        for name in ("coefficients.npz", "settings.json", "fit_manifest.json"):
            (bundle / name).write_bytes((key + "/" + name).encode())
    return output, preparation, source


def test_real_plan_seals_both_fits_then_tests_analyses_audit_and_report(prepared):
    output, _, _ = prepared
    stages = campaign.commands(output)
    assert [name for name, _ in stages] == [
        "llama_fit_capture",
        "qwen_fit_capture",
        "llama_fit",
        "qwen_fit",
        "seal_both_fits",
        "llama_test_capture",
        "qwen_test_capture",
        "llama_analysis",
        "qwen_analysis",
        "independent_audit",
        "report",
    ]
    assert dict(stages)["seal_both_fits"] is None
    for key in ("llama", "qwen"):
        fit = dict(stages)[key + "_fit"]
        assert fit[fit.index("--capture") + 1] == str(output / key / "fit")
        assert fit[fit.index("--model-key") + 1] == key
        assert fit[fit.index("--old-bundle") + 1] == str(output.parent / "old" / (key + "_fit"))
        assert fit[fit.index("--output") + 1] == str(output / (key + "_fit"))
        test = dict(stages)[key + "_test_capture"]
        assert test[test.index("--partition") + 1] == "test"
        assert test[test.index("--seal") + 1] == str(output / "test_seal.json")
    audit = dict(stages)["independent_audit"]
    assert audit[1:] == [
        str(ROOT / "scripts/audit_format_adaptation.py"),
        "--campaign",
        str(output),
        "--output",
        str(output) + "_audit",
    ]
    report = dict(stages)["report"]
    assert report[1:] == [
        str(ROOT / "scripts/report_format_adaptation.py"),
        "--campaign",
        str(output),
        "--audit",
        str(output) + "_audit",
        "--output",
        str(output) + "_report",
    ]


def test_coordinator_arguments_parse_in_actual_fit_analysis_and_capture_clis(prepared, monkeypatch):
    output, _, _ = prepared
    fit = load_script("adaptation_fit_cli_test", "fit_format_adaptation.py")
    analysis = load_script("adaptation_analysis_cli_test", "analyse_format_adaptation.py")
    capture = load_script("adaptation_capture_cli_test", "run_format_adaptation_capture.py")
    calls = []

    def fake_fit(capture_path, old_bundle, model_key, output_path):
        calls.append(("fit", model_key, capture_path, old_bundle, output_path))
        return {
            "status": "fixture",
            "model_key": model_key,
            "site": MODEL_SITES[model_key],
            "old_C2_C": 100,
        }

    def fake_analysis(capture_path, bundle, seal, output_path):
        calls.append(("analysis", capture_path, bundle, seal, output_path))
        return {"model_key": "fixture", "primary": {}, "named_diagnostic": {}}

    monkeypatch.setattr(fit, "fit_bundle", fake_fit)
    monkeypatch.setattr(analysis, "analyse", fake_analysis)
    monkeypatch.setattr(capture, "capture", lambda *args: calls.append(("capture", *args)))
    for name, command in campaign.commands(output):
        if name in ("seal_both_fits", "independent_audit", "report"):
            continue
        monkeypatch.setattr(sys, "argv", [command[1], *command[2:]])
        if name.endswith("_capture"):
            capture.main()
        elif name.endswith("_analysis"):
            analysis.main()
        else:
            fit.main()
    assert [call[0] for call in calls] == [
        "capture",
        "capture",
        "fit",
        "fit",
        "capture",
        "capture",
        "analysis",
        "analysis",
    ]


def test_report_command_parses_in_actual_reporter_cli_without_rendering(prepared, monkeypatch):
    output, _, _ = prepared
    command = dict(campaign.commands(output))["report"]
    reporter = load_script("adaptation_report_cli_test", "report_format_adaptation.py")
    calls = []
    monkeypatch.setattr(reporter, "report", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", command[1:])
    reporter.main()
    assert calls == [(output, Path(str(output) + "_audit"), Path(str(output) + "_report"))]


def fake_validated_bundle(path):
    key = path.name.removesuffix("_fit")
    return {"model_key": key, "site": MODEL_SITES[key]}, {}


def test_seal_hashes_both_validated_model_bundles_and_refuses_overwrite(prepared, monkeypatch):
    output, preparation, _ = prepared
    validated = []

    def validate(path):
        validated.append(path.name)
        return fake_validated_bundle(path)

    monkeypatch.setattr(campaign, "validate_bundle", validate)
    campaign.seal_both(output, preparation)
    assert validated == ["llama_fit", "qwen_fit"]
    seal_path = output / "test_seal.json"
    seal = read_json(seal_path)
    assert set(seal["models"]) == {"llama", "qwen"}
    assert seal["source_sha256"] == preparation["source_sha256"]
    for key in ("llama", "qwen"):
        for name, field in (
            ("coefficients.npz", "coefficients_sha256"),
            ("settings.json", "settings_sha256"),
            ("fit_manifest.json", "fit_manifest_sha256"),
        ):
            assert seal["models"][key][field] == file_sha256(output / (key + "_fit") / name)
    before = seal_path.read_bytes()
    with pytest.raises(FileExistsError):
        campaign.seal_both(output, preparation)
    assert seal_path.read_bytes() == before


@pytest.mark.parametrize("key", ["llama", "qwen"])
def test_seal_rejects_either_prior_test_attempt(prepared, monkeypatch, key):
    output, preparation, _ = prepared
    (output / key / "test").mkdir()
    monkeypatch.setattr(campaign, "validate_bundle", fake_validated_bundle)
    with pytest.raises(ValueError, match="Test attempt predates"):
        campaign.seal_both(output, preparation)
    assert not (output / "test_seal.json").exists()


@pytest.mark.parametrize("key", ["llama", "qwen"])
@pytest.mark.parametrize("mutation", ["status", "row_count", "invalid_bundle"])
def test_seal_rejects_either_incomplete_capture_or_fit(prepared, monkeypatch, key, mutation):
    output, preparation, _ = prepared
    metadata = output / key / "fit/manifest.json"
    manifest = read_json(metadata)
    if mutation == "status":
        manifest["status"] = "failed"
    elif mutation == "row_count":
        manifest["completed_rows"] = 767
    atomic_json(metadata, manifest)

    def validate(path):
        if mutation == "invalid_bundle" and path.name == key + "_fit":
            raise ValueError("Synthetic fit is not frozen")
        return fake_validated_bundle(path)

    monkeypatch.setattr(campaign, "validate_bundle", validate)
    with pytest.raises(ValueError, match="finish before sealing|not frozen"):
        campaign.seal_both(output, preparation)
    assert not (output / "test_seal.json").exists()


def test_seal_rejects_changed_predeclared_source(prepared, monkeypatch):
    output, preparation, source = prepared
    monkeypatch.setattr(campaign, "validate_bundle", fake_validated_bundle)
    source.write_text("# Changed unprepared policy\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Changed input"):
        campaign.seal_both(output, preparation)
    assert not (output / "test_seal.json").exists()


def test_seal_rejects_bundle_stored_under_other_model_slot(prepared, monkeypatch):
    output, preparation, _ = prepared

    def wrong_model(path):
        return {"model_key": "llama", "site": MODEL_SITES["llama"]}, {}

    monkeypatch.setattr(campaign, "validate_bundle", wrong_model)
    with pytest.raises(ValueError, match="model|Model|slot"):
        campaign.seal_both(output, preparation)
    assert not (output / "test_seal.json").exists()
