"""Reporter gates and visibly marked software-fixture export; no inference."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from latent_probes.storage import file_sha256


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/report_format_adaptation.py"
    spec = importlib.util.spec_from_file_location("adaptation_report_test", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


@pytest.fixture
def fixture(tmp_path):
    campaign = tmp_path / "fixture_campaign"
    audit = tmp_path / "fixture_audit"
    hashes = {}
    aggregate = {
        "complete_quartets": 2,
        "quartets": 8,
        "quartet_accuracy": 0.25,
        "correct": 20,
        "rows": 32,
        "row_accuracy": 0.625,
    }
    for key in ("llama", "qwen"):
        folder = campaign / (key + "_analysis")
        folder.mkdir(parents=True)
        data = {
            "test_fixture": True,
            "primary": {"difference": 0.0},
            "named_diagnostic": {"difference": 0.0},
            "arms": {
                arm: {
                    renderer + "/" + cond: {"aggregate": copy.deepcopy(aggregate)}
                    for renderer in ("A", "B")
                    for cond in ("correction_hard", "correction_easy", "one_step", "asserted")
                }
                for arm in ("A512", "Mixed512", "Ahalf256", "Offset")
            },
        }
        write(folder / "results.json", data)
        (folder / "predictions.npz").write_bytes(
            b"EXPLICIT SOFTWARE FIXTURE; not model measurements"
        )
        outputs = {name: file_sha256(folder / name) for name in ("results.json", "predictions.npz")}
        write(folder / "manifest.json", {"status": "complete", "output_sha256": outputs})
        hashes.update({str((folder / name).resolve()): h for name, h in outputs.items()})
    write(
        audit / "audit_receipt.json",
        {"status": "pass", "campaign": str(campaign.resolve()), "analysis_output_sha256": hashes},
    )
    return campaign, audit, tmp_path / "report"


def test_default_rejects_fixture_as_scientific_result(fixture):
    campaign, audit, output = fixture
    with pytest.raises(ValueError, match="Fixture"):
        module().report(campaign, audit, output)
    assert not output.exists()


@pytest.mark.parametrize("change", ["audit_failure", "wrong_campaign", "changed_analysis"])
def test_report_requires_exact_audited_outputs(fixture, change):
    campaign, audit, output = fixture
    p = audit / "audit_receipt.json"
    receipt = json.loads(p.read_text())
    if change == "audit_failure":
        receipt["status"] = "failed"
    elif change == "wrong_campaign":
        receipt["campaign"] = str(campaign / "different")
    else:
        receipt["analysis_output_sha256"][
            str((campaign / "llama_analysis/results.json").resolve())
        ] = "bad"
    write(p, receipt)
    with pytest.raises(ValueError):
        module().report(campaign, audit, output, allow_test_fixture=True)
    assert not output.exists()


def test_all_export_formats_are_marked_as_fixture_and_byte_bound(fixture):
    pytest.importorskip("matplotlib")
    campaign, audit, output = fixture
    manifest = module().report(campaign, audit, output, allow_test_fixture=True)
    assert json.loads((output / "summary.json").read_text())["test_fixture"] is True
    assert "TEST FIXTURE" in (output / "results.md").read_text(encoding="utf-8")
    assert "TEST FIXTURE" in (output / "adaptation.svg").read_text(encoding="utf-8")
    assert {"adaptation.pdf", "adaptation.svg", "adaptation.png"} <= set(manifest["output_sha256"])
    assert all(file_sha256(output / name) == h for name, h in manifest["output_sha256"].items())
    with pytest.raises(FileExistsError):
        module().report(campaign, audit, output, allow_test_fixture=True)
