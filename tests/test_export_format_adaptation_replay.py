"""Synthetic portable-score replay; no real campaign artifacts are opened."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_format_adaptation_analysis import scores_for, test_rows

from latent_probes import format_adaptation_analysis as metrics
from latent_probes.storage import atomic_json, atomic_npz

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_format_adaptation_replay.py"
SPEC = importlib.util.spec_from_file_location("portable_replay_test", SCRIPT)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    base = tmp_path_factory.mktemp("SYNTHETIC-PORTABLE-INPUTS")
    campaign, audit = base / "campaign", base / "audit"
    campaign.mkdir()
    audit.mkdir()
    identity = replay.canonical_path(campaign)
    sources = {
        str(replay.ROOT / "src/latent_probes" / name): replay.sha(
            (replay.ROOT / "src/latent_probes" / name).read_bytes()
        )
        for name in replay.SOURCE_NAMES
    }
    seal = {
        "schema": "format-adaptation-test-seal-v1",
        "models": {m: {} for m in replay.MODELS},
        "source_sha256": sources,
        "bank_sha256": "b" * 64,
    }
    atomic_json(campaign / "test_seal.json", seal)
    seal_hash = replay.sha((campaign / "test_seal.json").read_bytes())
    reconstruction = {
        "schema": "format-adaptation-independent-reconstruction-seal-v1",
        "SYNTHETIC_FIXTURE_ONLY": True,
    }
    atomic_json(audit / "reconstruction_seal.json", reconstruction)
    audit_data = {
        "schema": "format-adaptation-independent-audit-v1",
        "status": "pass",
        "campaign": identity,
        "model_calls": 0,
        "classifier_fits": 0,
        "production_functions_imported": False,
        "saved_analysis_opened_before_both_reconstructions_frozen": False,
        "SYNTHETIC_FIXTURE_ONLY": True,
        "models": {},
        "reconstruction_seal_sha256": replay.sha((audit / "reconstruction_seal.json").read_bytes()),
        "analysis_output_sha256": {},
        "input_sha256": {identity + "/test_seal.json": seal_hash},
    }
    rows = test_rows()
    tokens = {"YES": 1, "NO": 0}
    for key in replay.MODELS:
        rng = np.random.default_rng(11 if key == "llama" else 23)
        arm_scores = scores_for(rows)
        for score in arm_scores.values():
            score["factor_margins"] *= rng.uniform(0.1, 2.0, (4992, 4))
        margins = np.array(
            [(2 * r["gold"] - 1) * (0.5 + 0.05 * (i % 7)) for i, r in enumerate(rows)]
        )
        copied = copy.deepcopy(rows)
        strict = []
        for i, row in enumerate(copied):
            outside = i % 11 == 0
            argmax = 7 if outside else tokens["YES" if row["gold"] else "NO"]
            norm = float(np.logaddexp(0, margins[i]) - np.log(0.2))
            row.update(
                output_logits={
                    "argmax_token_id": argmax,
                    "argmax_tie_count": 2 if i % 17 == 0 else 1,
                    "yes_logit": float(margins[i]),
                    "no_logit": 0.0,
                    "yes_minus_no": float(margins[i]),
                    "yes_logprob": float(margins[i]) - norm,
                    "no_logprob": -norm,
                },
                messages=[{"role": "user", "content": "SYNTHETIC PROMPT MUST NOT BE EXPORTED"}],
                prompt_token_ids=[10, 20, 30],
                messages_sha256="a" * 64,
                rendered_chat_sha256="b" * 64,
                prompt_token_ids_sha256="c" * 64,
            )
            strict.append(not outside)
        strict = np.array(strict, dtype=bool)
        capture = campaign / key / "test"
        capture.mkdir(parents=True)
        atomic_json(
            capture / "rows.json",
            {
                "schema": "format-adaptation-capture-rows-v1",
                "model_key": key,
                "site": replay.SITES[key],
                "test_fixture": True,
                "rows": copied,
                "answer_token_ids": tokens,
            },
        )
        raw_hash = replay.sha((capture / "rows.json").read_bytes())
        atomic_json(
            capture / "manifest.json",
            {
                "schema": "format-adaptation-capture-manifest-v1",
                "status": "complete",
                "partition": "test",
                "completed_rows": 4992,
                "rows_sha256": raw_hash,
                "features_sha256": "f" * 64,
                "seal_sha256": seal_hash,
                "bank_sha256": seal["bank_sha256"],
            },
        )
        output = campaign / (key + "_analysis")
        output.mkdir()
        result = metrics.score_arrays(copied, arm_scores, margins, strict, model_key=key)
        result["output_first_token_detail"] = metrics.output_token_detail(copied, tokens)
        result.update(
            schema="format-adaptation-analysis-v1",
            model_key=key,
            site=replay.SITES[key],
            test_fixture=True,
            research_status="agent_audited_exploratory",
            confirmatory=False,
        )
        atomic_json(output / "results.json", result)
        atomic_npz(
            output / "predictions.npz",
            row_ids=np.array([r["row_id"] for r in rows]),
            arms=np.array(replay.ARMS),
            factor_order=np.array(replay.FACTORS),
            verdicts=np.column_stack([arm_scores[a]["verdicts"] for a in replay.ARMS]),
            factor_predictions=np.stack(
                [arm_scores[a]["factor_predictions"] for a in replay.ARMS], axis=1
            ),
            factor_margins=np.stack([arm_scores[a]["factor_margins"] for a in replay.ARMS], axis=1),
            output_margins=margins,
            strict_first_token_correct=strict,
        )
        output_hashes = {
            name: replay.sha((output / name).read_bytes())
            for name in ("results.json", "predictions.npz")
        }
        atomic_json(
            output / "manifest.json",
            {
                "schema": "format-adaptation-analysis-manifest-v1",
                "status": "complete",
                "output_sha256": output_hashes,
                "input_artifact_sha256": {identity + "/test_seal.json": seal_hash},
            },
        )
        audit_data["models"][key] = {
            "analysis_artifact_sha256": {
                identity + "/" + key + "_analysis/" + name: replay.sha((output / name).read_bytes())
                for name in ("manifest.json", "results.json", "predictions.npz")
            }
        }
        audit_data["analysis_output_sha256"].update(
            {
                identity + "/" + key + "_analysis/" + name: value
                for name, value in output_hashes.items()
            }
        )
        audit_data["input_sha256"].update(
            {
                identity + "/" + key + "/test/" + name: replay.sha((capture / name).read_bytes())
                for name in ("rows.json", "manifest.json")
            }
        )
    atomic_json(audit / "audit_receipt.json", audit_data)
    return campaign, audit


@pytest.fixture(scope="module")
def bundle(inputs, tmp_path_factory):
    path = tmp_path_factory.mktemp("SYNTHETIC-PORTABLE-OUTPUT") / "bundle"
    result = replay._export_bundle(*inputs, path, allow_fixture=True)
    assert result["status"] == "pass" and result["test_fixture"] is True
    return path


def rehash(bundle):
    manifest = replay.read_json(bundle / "manifest.json")
    manifest["files"] = {
        str(p.relative_to(bundle)).replace("\\", "/"): replay.sha(p.read_bytes())
        for p in bundle.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    (bundle / "manifest.json").write_bytes(replay.json_bytes(manifest))


def test_round_trip_retains_exact_original_bytes_and_all_metric_scopes(inputs, bundle):
    campaign, audit = inputs
    result = replay._replay_bundle(bundle, allow_fixture=True)
    assert result["independent_audit"] is result["feature_to_prediction_verified"] is False
    assert result["model_calls"] == result["classifier_fits"] == 0
    for key in replay.MODELS:
        assert (bundle / key / "predictions.npz").read_bytes() == (
            campaign / (key + "_analysis/predictions.npz")
        ).read_bytes()
        assert (
            replay.unpack(bundle / key / "results.json.gz")
            == (campaign / (key + "_analysis/results.json")).read_bytes()
        )
        assert (
            replay.unpack(bundle / key / "analysis_manifest.json.gz")
            == (campaign / (key + "_analysis/manifest.json")).read_bytes()
        )
        assert (
            replay.unpack(bundle / key / "capture_manifest.json.gz")
            == (campaign / key / "test/manifest.json").read_bytes()
        )
        assert set(result["models"][key]["replayed_scopes"]) == {
            "arms",
            "output_baseline",
            "factor_diagnostics",
            "paired_format",
            "primary",
            "named_diagnostic",
            "factor_truth_scope",
            "output_first_token_detail",
        }
        assert result["models"][key]["comparisons"]["numeric"] > 1000
        projected = json.loads(replay.unpack(bundle / key / "rows.json.gz"))
        assert all(
            "messages" not in r and "prompt_token_ids" not in r and "features" not in r
            for r in projected["rows"]
        )
    assert (
        replay.unpack(bundle / "audit_receipt.json.gz")
        == (audit / "audit_receipt.json").read_bytes()
    )
    assert "SYNTHETIC FIXTURE" in (bundle / "README.txt").read_text()
    assert all("features" not in p.name and "coefficients" not in p.name for p in bundle.rglob("*"))


def test_standalone_copy_replays_without_native_files_repo_import_or_pythonpath(
    inputs, bundle, tmp_path
):
    copied = tmp_path / "standalone"
    shutil.copytree(bundle, copied)
    campaign, _ = inputs
    offline = campaign.with_name("campaign-offline-preserved")
    assert campaign.is_relative_to(campaign.parent) and offline.is_relative_to(campaign.parent)
    campaign.rename(offline)
    code = """import importlib.abc,importlib.util,json,pathlib,sys
sys.dont_write_bytecode=True
root=pathlib.Path(sys.argv[2]); sys.path=[p for p in sys.path if p not in (str(root),str(root/'src'))]
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname=='latent_probes' or fullname.startswith('latent_probes.') or fullname in ('torch','transformers'):
   raise AssertionError('Forbidden repo/model import: '+fullname)
sys.meta_path.insert(0,Guard())
spec=importlib.util.spec_from_file_location('standalone',sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
r=m._replay_bundle(pathlib.Path(sys.argv[1]).parent,allow_fixture=True)
print(json.dumps({'status':r['status'],'module':r['production_metric_module'],'fixture':r['test_fixture']}))
"""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    try:
        child = subprocess.run(
            [sys.executable, "-I", "-c", code, str(copied / "replay.py"), str(replay.ROOT)],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
        )
    finally:
        offline.rename(campaign)
    assert child.returncode == 0, child.stderr
    result = json.loads(child.stdout)
    assert result == {
        "status": "pass",
        "module": "production/latent_probes/format_adaptation_analysis.py",
        "fixture": True,
    }


@pytest.mark.parametrize(
    "mutation", ["extra", "arm_order", "factor_order", "row_order", "nan", "strict_dtype"]
)
def test_safe_prediction_inventory_and_joins_reject_corruption(bundle, tmp_path, mutation):
    rows = json.loads(replay.unpack(bundle / "llama/rows.json.gz"))["rows"]
    with np.load(bundle / "llama/predictions.npz", allow_pickle=False) as source:
        values = {name: source[name].copy() for name in source.files}
    if mutation == "extra":
        values["features"] = np.zeros((4992, 2))
    elif mutation == "arm_order":
        values["arms"] = values["arms"][::-1]
    elif mutation == "factor_order":
        values["factor_order"] = values["factor_order"][::-1]
    elif mutation == "row_order":
        values["row_ids"] = values["row_ids"][::-1]
    elif mutation == "nan":
        values["factor_margins"][0, 0, 0] = np.nan
    else:
        values["strict_first_token_correct"] = values["strict_first_token_correct"].astype(np.int64)
    path = tmp_path / "corrupt.npz"
    atomic_npz(path, **values)
    with pytest.raises(ValueError):
        replay._arrays(path, rows)


@pytest.mark.parametrize("mutation", ["state_vector", "hash_vector", "identity_vector"])
def test_projection_refuses_nested_or_misplaced_vectors(mutation):
    row = copy.deepcopy(test_rows()[0])
    row.update(
        messages_sha256="a" * 64,
        output_logits={
            "argmax_token_id": 0,
            "argmax_tie_count": 1,
            "yes_logprob": -2.0,
            "no_logprob": -1.0,
            "yes_minus_no": -1.0,
        },
    )
    payload = {
        "schema": "format-adaptation-capture-rows-v1",
        "model_key": "llama",
        "site": replay.SITES["llama"],
        "test_fixture": True,
        "answer_token_ids": {"YES": 1, "NO": 0},
        "rows": [row],
    }
    if mutation == "state_vector":
        row["state"]["features"] = [1.0, 2.0]
    elif mutation == "hash_vector":
        row["messages_sha256"] = [1.0, 2.0]
    else:
        row["group_id"] = [1.0, 2.0]
    with pytest.raises(ValueError):
        replay._project(payload, "llama", True)


def test_array_refusal_precedes_any_export_output(inputs, tmp_path, monkeypatch):
    output = tmp_path / "must-not-exist"

    def refuse(*args):
        raise ValueError("Unexpected prediction arrays; derived scores only")

    monkeypatch.setattr(replay, "_arrays", refuse)
    with pytest.raises(ValueError, match="derived scores only"):
        replay._export_bundle(*inputs, output, allow_fixture=True)
    assert not output.exists()


@pytest.mark.parametrize(
    "mutation", ["failed", "campaign", "missing_model", "rows", "seal", "prediction"]
)
def test_export_rejects_partial_stale_or_wrong_audit(inputs, tmp_path, mutation):
    campaign, original = inputs
    audit = tmp_path / "audit"
    shutil.copytree(original, audit)
    data = replay.read_json(audit / "audit_receipt.json")
    if mutation == "failed":
        data["status"] = "failed_preserved"
    elif mutation == "campaign":
        data["campaign"] += "-wrong"
    elif mutation == "missing_model":
        data["models"].pop("qwen")
    elif mutation == "rows":
        data["input_sha256"][replay.canonical_path(campaign) + "/llama/test/rows.json"] = "0" * 64
    elif mutation == "seal":
        data["input_sha256"][replay.canonical_path(campaign) + "/test_seal.json"] = "0" * 64
    else:
        data["analysis_output_sha256"][
            replay.canonical_path(campaign) + "/qwen_analysis/predictions.npz"
        ] = "0" * 64
    (audit / "audit_receipt.json").write_bytes(replay.json_bytes(data))
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        replay._export_bundle(campaign, audit, output, allow_fixture=True)
    if output.exists():
        assert replay.read_json(output / "manifest.json")["status"] == "failed_preserved"


def test_file_tamper_and_rehashed_prediction_tamper_fail(bundle, tmp_path):
    copied = tmp_path / "tampered"
    shutil.copytree(bundle, copied)
    predictions = copied / "llama/predictions.npz"
    predictions.write_bytes(predictions.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="Changed/unsafe"):
        replay._replay_bundle(copied, allow_fixture=True)
    rehash(copied)
    with pytest.raises(ValueError, match="manifest/result binding"):
        replay._replay_bundle(copied, allow_fixture=True)


def test_rehashed_bad_pairing_and_metric_scope_change_fail(bundle, tmp_path):
    copied = tmp_path / "bad-pair"
    shutil.copytree(bundle, copied)
    rows = json.loads(replay.unpack(copied / "llama/rows.json.gz"))
    rows["rows"][0]["state"]["world"] = 1 - rows["rows"][0]["state"]["world"]
    replay.compressed(copied / "llama/rows.json.gz", replay.json_bytes(rows))
    rehash(copied)
    with pytest.raises(ValueError, match="Duplicate"):
        replay._replay_bundle(copied, allow_fixture=True)
    expected = {"primary": {"difference": 0.1}, "factor_diagnostics": {"B1": 0.5}}
    changed = copy.deepcopy(expected)
    changed["factor_diagnostics"]["B1"] = 0.6
    with pytest.raises(ValueError, match="factor_diagnostics"):
        replay.compare_tree(expected, changed)


def test_public_apis_cli_fixture_and_overwrite_guards(inputs, bundle, tmp_path):
    with pytest.raises(ValueError, match="synthetic"):
        replay.export_bundle(*inputs, tmp_path / "public-export")
    with pytest.raises(ValueError, match="synthetic"):
        replay.replay_bundle(bundle)
    with pytest.raises(ValueError, match="already exists"):
        replay._export_bundle(*inputs, bundle, allow_fixture=True)
    command = subprocess.run(
        [sys.executable, str(SCRIPT), "replay", "--bundle", str(bundle)],
        capture_output=True,
        text=True,
    )
    assert command.returncode != 0 and "synthetic" in command.stderr
    assert (
        "--fixture"
        not in subprocess.run(
            [sys.executable, str(SCRIPT), "export", "--help"], capture_output=True, text=True
        ).stdout
    )
