"""Figures cannot silently display altered or fixture summaries."""

import importlib.util
import json
from pathlib import Path

import pytest

from latent_probes.storage import atomic_json, file_sha256

MODULE = Path(__file__).resolve().parents[1] / "scripts/plot_format_adaptation_text.py"
spec = importlib.util.spec_from_file_location("lexical_plot_test", MODULE)
plotting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plotting)


def source_fixture(tmp_path):
    lexical, latent = tmp_path / "lexical", tmp_path / "latent"
    atomic_json(
        lexical / "metrics.json",
        {
            "human_review_complete": False,
            "research_status": "posthoc_exposed_bank_diagnostic",
            "views": {"full_messages": {}},
        },
    )
    atomic_json(lexical / "audit/audit_receipt.json", {"status": "pass"})
    atomic_json(
        lexical / "manifest.json",
        {
            "output_sha256": {
                "metrics.json": file_sha256(lexical / "metrics.json"),
                "audit/audit_receipt.json": file_sha256(lexical / "audit/audit_receipt.json"),
            }
        },
    )
    atomic_json(
        latent / "summary.json", {"test_fixture": False, "models": {"llama": {}, "qwen": {}}}
    )
    atomic_json(
        latent / "manifest.json",
        {
            "output_sha256": {
                "summary.json": file_sha256(latent / "summary.json"),
            }
        },
    )
    return lexical, latent


@pytest.mark.parametrize("damage", ["metrics", "audit", "latent", "fixture"])
def test_source_tampering_or_fixture_prevents_plot(tmp_path, damage):
    lexical, latent = source_fixture(tmp_path)
    assert len(plotting.load_inputs(lexical, latent)) == 3
    if damage == "metrics":
        path = lexical / "metrics.json"
    elif damage == "audit":
        path = lexical / "audit/audit_receipt.json"
    else:
        path = latent / "summary.json"
    value = json.loads(path.read_text())
    value["test_fixture"] = True
    atomic_json(path, value)
    with pytest.raises(ValueError):
        plotting.load_inputs(lexical, latent)
