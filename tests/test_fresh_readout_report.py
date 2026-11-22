"""Engineering-only report fixtures; no saved experimental result is read."""

from __future__ import annotations

import copy
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest

from latent_probes.fresh_readout_analysis import score_arrays
from latent_probes.storage import atomic_json, atomic_npz, file_sha256, read_json

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fresh_report", ROOT / "scripts/report_fresh_readout.py"
)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def rows():
    result = []
    for group in range(24):
        for renderer in ("A", "B"):
            for i, u, b, w, m in itertools.product((0, 1), repeat=5):
                result.append(
                    {
                        "group_id": f"fixture_test_{group:02d}",
                        "renderer": renderer,
                        "condition": "one_step",
                        "gold": u if b else i,
                        "state": dict(
                            initial=i, update=u, receipt=b, first_receipt=None, world=w, name_map=m
                        ),
                    }
                )
            for i, u, b1, b2, w, m in itertools.product((0, 1), repeat=6):
                result.append(
                    {
                        "group_id": f"fixture_test_{group:02d}",
                        "renderer": renderer,
                        "condition": "correction_easy" if i == u else "correction_hard",
                        "gold": 1 - u if b2 else u if b1 else i,
                        "state": dict(
                            initial=i, update=u, first_receipt=b1, receipt=b2, world=w, name_map=m
                        ),
                    }
                )
            for e, w, m in itertools.product((0, 1), repeat=3):
                result.append(
                    {
                        "group_id": f"fixture_test_{group:02d}",
                        "renderer": renderer,
                        "condition": "asserted",
                        "gold": e,
                        "state": dict(asserted=e, world=w, name_map=m),
                    }
                )
    for index, row in enumerate(result):
        row["row_id"] = f"fixture_case_{index:04d}"
    return result


@pytest.fixture(scope="module")
def synthetic_results():
    cases = rows()
    gold = np.asarray([r["gold"] for r in cases])
    initial = np.asarray([r["state"].get("initial", 0) for r in cases])
    factors = np.asarray(
        [
            [s["initial"], s["update"], s.get("first_receipt") or 0, s["receipt"]]
            if "initial" in s
            else [0, 0, 0, 0]
            for s in (r["state"] for r in cases)
        ]
    )
    for index, row in enumerate(cases):
        if row["condition"] == "one_step":
            factors[index, 2], factors[index, 3] = row["state"]["receipt"], 0
    i, u, b1, b2 = factors.T
    fold = np.where(b2, 1 - u, np.where(b1, u, i))
    result = {}
    for model in ("llama", "qwen"):
        sites = [f"layer_{b}_{p}" for b in REPORT.BLOCKS[model] for p in REPORT.POSITIONS]
        s0, s1 = (initial, gold) if model == "llama" else (gold, initial)
        panel = np.column_stack([gold if j % 2 == 0 else initial for j in range(12)])
        panel[:, 0], panel[:, 5] = s1, s0
        scores = {
            "S0": s0,
            "S1": s1,
            "C1": gold,
            "C2": fold,
            "output": 1 - gold,
            "source_sites": sites,
            "source_all12": panel,
            "source_all12_margins": panel * 2 - 1,
            "factor_predictions": factors,
            "factor_margins": factors * 2 - 1,
        }
        for key in ("S0", "S1", "C1", "output"):
            scores[key + "_margin"] = scores[key] * 2 - 1
        scored, methods, margins = score_arrays(cases, scores)
        scored["strict_first_token_diagnostic"] = {
            stratum: {
                "rows": n,
                "candidate_first_tokens": n // 2,
                "outside_candidate_first_tokens": n // 2,
                "correct_full_denominator": 0,
                "accuracy_full_denominator": 0.0,
                "candidate_first_token_rate": 0.5,
                "full_vocab_argmax_ties": 3,
                "mean_candidate_token_mass": 0.2,
                "scope": "Engineering fixture of lowest-index full-vocabulary argmax; no response generation",
            }
            for stratum in REPORT.STRATA
            for n in [192 if stratum.endswith("/asserted") else 768]
        }
        result[model] = (
            scored,
            {
                "row_ids": np.asarray([r["row_id"] for r in cases]),
                "methods": np.asarray(list(methods)),
                "verdicts": np.column_stack(list(methods.values())),
                "margins": np.column_stack(
                    [margins[m] if m in margins else methods[m] * 2 - 1 for m in methods]
                ),
                "factor_predictions": factors,
                "factor_margins": factors * 2 - 1,
            },
        )
    return result


def make_analysis_pair(tmp_path, synthetic_results):
    bank = tmp_path / "fixture-bank.json"
    groups = [
        {"group_id": f"fixture_{p}_{i:02d}", "split": p}
        for p, n in (("correction_fit", 16), ("correction_dev", 8), ("test", 24))
        for i in range(n)
    ]
    atomic_json(bank, {"schema": "fresh-readout-bank-v2", "groups": groups})
    directories = []
    for model in ("llama", "qwen"):
        directory = tmp_path / model
        directory.mkdir()
        raw, arrays = synthetic_results[model]
        result = copy.deepcopy(raw)
        sites = [f"layer_{b}_{p}" for b in REPORT.BLOCKS[model] for p in REPORT.POSITIONS]
        settings_path = tmp_path / f"{model}-fit" / "settings.json"
        heads = {f"source_{j:02d}": {"site": site, "C": 0.1} for j, site in enumerate(sites)}
        atomic_json(
            settings_path,
            {
                "schema": "fresh-readout-fit-v2",
                "status": "frozen",
                "test_fixture": True,
                "research_status": REPORT.STATUS,
                "human_review_complete": False,
                "confirmatory": False,
                "capture_provenance": {
                    "model_key": model,
                    "model_settings": {
                        "model_id": REPORT.MODELS[model][0],
                        "revision": REPORT.MODELS[model][1],
                    },
                    "sites": sites,
                    "bank_file_sha256": file_sha256(bank),
                },
                "source_panel": [
                    {
                        "site": site,
                        "C": 0.1,
                        "head_key": f"source_{j:02d}",
                        "dev_group_log_loss": 0.5,
                    }
                    for j, site in enumerate(sites)
                ],
                "heads": heads,
                "pipelines": {"S0": {"head_key": "source_05"}, "S1": {"head_key": "source_00"}},
                "fit_context": {"bank_path": str(bank)},
                **{
                    p + "_ids": [f"{p}_{i}" for i in range(512)]
                    for p in ("source_fit", "source_dev", "correction_fit", "correction_dev")
                },
            },
        )
        result.update(
            schema="fresh-readout-analysis-v2",
            model_key=model,
            test_fixture=True,
            research_status=REPORT.STATUS,
            human_review_complete=False,
            confirmatory_result=False,
            fresh_test=True,
            primary_role="primary" if model == "llama" else "predeclared_replication",
            original_source_site=sites[5],
            selected_source_site=sites[0],
            bank_sha256=REPORT.canonical(read_json(bank)),
            source_sha256={
                str(bank): file_sha256(bank),
                str(settings_path): file_sha256(settings_path),
            },
        )
        atomic_json(directory / "results.json", result)
        atomic_npz(directory / "predictions.npz", **arrays)
        refresh_manifest(directory)
        directories.append(directory)
    return directories


def refresh_manifest(directory):
    result = read_json(directory / "results.json")
    atomic_json(
        directory / "manifest.json",
        {
            "schema": "fresh-readout-analysis-manifest-v2",
            "source_sha256": result["source_sha256"],
            "output_sha256": {
                name: file_sha256(directory / name) for name in ("results.json", "predictions.npz")
            },
        },
    )


def test_preserves_negative_replication_raw_orientation_and_all_methods(
    tmp_path, synthetic_results, monkeypatch
):
    llama, qwen = make_analysis_pair(tmp_path, synthetic_results)
    monkeypatch.setattr(
        np.random, "default_rng", lambda *a: pytest.fail("Reporter recomputed bootstrap")
    )
    summary = REPORT.load_report(llama, qwen, allow_test_fixture=True)
    assert summary["models"]["llama"]["primary"]["difference"] == 1
    assert summary["models"]["qwen"]["primary"]["difference"] == -1
    assert summary["paired_test_contexts"] == 24 and not summary["independent_models_pooled"]
    assert summary["test_fixture"] and not summary["fresh_test"]
    assert summary["report_kind"] == "engineering_fixture_not_scientific_result"
    for model in ("llama", "qwen"):
        assert len(summary["models"][model]["methods"]) == 17
        for value in summary["models"][model]["methods"].values():
            assert len(value["strata"]) == 8
            assert len(value["renderer_A_hard_group_values"]) == 24
        output = summary["models"][model]["methods"]["output"]["strata"]["A/correction_hard"]
        assert output["unrecalibrated_score_diagnostics"]["auroc"] == 0
        diagnostic = summary["models"][model]["strict_first_token_diagnostic"]
        assert len(diagnostic) == 8
        assert diagnostic["A/asserted"]["outside_candidate_first_tokens"] == 96
        assert diagnostic["A/asserted"]["accuracy_full_denominator"] == 0
    text = REPORT.markdown(summary)
    assert "ENGINEERING FIXTURE" in text and "-100.00 percentage points" in text
    assert "48 independent" in text and "not provide 48 independent" in text
    assert "1,536" in text and "512" in text and "privileged" in text
    assert "never sign-flipped" in text
    assert "Strict first-token elicitation diagnostic" in text
    assert "Outside YES/NO" in text
    assert "does not observe a complete generated response" in text


def test_strict_first_token_uses_full_denominator_not_only_candidate_outputs(synthetic_results):
    diagnostics = copy.deepcopy(synthetic_results["llama"][0]["strict_first_token_diagnostic"])
    for value in diagnostics.values():
        value["correct_full_denominator"] = value["rows"] // 4
        value["accuracy_full_denominator"] = 0.25
    REPORT.check_strict_first_token(diagnostics)
    diagnostics["A/asserted"]["accuracy_full_denominator"] = 0.5
    with pytest.raises(ValueError, match="full-denominator arithmetic"):
        REPORT.check_strict_first_token(diagnostics)


@pytest.mark.parametrize(
    "defect", ["missing_stratum", "outside_count", "correct_count", "mass", "tie_count"]
)
def test_strict_first_token_rejects_missing_or_inconsistent_diagnostics(synthetic_results, defect):
    diagnostics = copy.deepcopy(synthetic_results["llama"][0]["strict_first_token_diagnostic"])
    value = diagnostics["B/asserted"]
    if defect == "missing_stratum":
        diagnostics.pop("A/one_step")
    elif defect == "outside_count":
        value["outside_candidate_first_tokens"] += 1
    elif defect == "correct_count":
        value["correct_full_denominator"] = value["candidate_first_tokens"] + 1
    elif defect == "mass":
        value["mean_candidate_token_mass"] = float("nan")
    else:
        value["full_vocab_argmax_ties"] = value["rows"] + 1
    with pytest.raises(ValueError):
        REPORT.check_strict_first_token(diagnostics)


@pytest.mark.parametrize(
    "defect",
    [
        "upstream",
        "manifest",
        "confirmatory",
        "fresh",
        "model",
        "groups",
        "primary",
        "interval",
        "fixture",
    ],
)
def test_report_rejects_corrupt_sources_status_and_primary_arithmetic(
    tmp_path, synthetic_results, defect
):
    llama, qwen = make_analysis_pair(tmp_path, synthetic_results)
    result = read_json(llama / "results.json")
    if defect == "upstream":
        path = next(Path(p) for p in result["source_sha256"] if Path(p).name == "settings.json")
        path.write_text("{}", encoding="utf-8")
    elif defect == "manifest":
        with (llama / "results.json").open("ab") as stream:
            stream.write(b" ")
    elif defect == "confirmatory":
        result["confirmatory_result"] = True
    elif defect == "fresh":
        result["fresh_test"] = False
    elif defect == "model":
        result["model_key"] = "qwen"
    elif defect == "groups":
        result["methods"]["S0"]["strata"]["A/correction_hard"]["groups"][0]["group_id"] = "alien"
    elif defect == "primary":
        result["primary"]["difference"] = 0.5
    elif defect == "interval":
        result["primary"]["conditional_percentile_interval_95"] = [1, -1]
    elif defect == "fixture":
        result["test_fixture"] = False
    if defect not in ("upstream", "manifest"):
        atomic_json(llama / "results.json", result)
        refresh_manifest(llama)
    with pytest.raises(ValueError):
        REPORT.load_report(llama, qwen, allow_test_fixture=True)


def test_both_complete_manifests_required_before_reading_either_result(
    tmp_path, synthetic_results, monkeypatch
):
    llama, qwen = make_analysis_pair(tmp_path, synthetic_results)
    manifest = read_json(qwen / "manifest.json")
    manifest["status"] = "running"
    atomic_json(qwen / "manifest.json", manifest)
    original = REPORT.read

    def guarded(path):
        if Path(path).name == "results.json":
            pytest.fail("Read a model result before both analyses completed")
        return original(path)

    monkeypatch.setattr(REPORT, "read", guarded)
    with pytest.raises(ValueError, match="Both analyses must be complete"):
        REPORT.load_report(llama, qwen, allow_test_fixture=True)


def test_fixture_not_accepted_by_native_report_and_swapped_models_rejected(
    tmp_path, synthetic_results
):
    llama, qwen = make_analysis_pair(tmp_path, synthetic_results)
    with pytest.raises(ValueError, match="Fixture analyses"):
        REPORT.load_report(llama, qwen)
    with pytest.raises(ValueError, match="Wrong model"):
        REPORT.load_report(qwen, llama, allow_test_fixture=True)


def test_false_pinned_model_id_rejected_after_source_hash_rebinding(tmp_path, synthetic_results):
    llama, qwen = make_analysis_pair(tmp_path, synthetic_results)
    result = read_json(qwen / "results.json")
    settings_path = next(
        Path(p) for p in result["source_sha256"] if Path(p).name == "settings.json"
    )
    settings = read_json(settings_path)
    settings["capture_provenance"]["model_settings"]["model_id"] = "different/model"
    atomic_json(settings_path, settings)
    result["source_sha256"][str(settings_path)] = file_sha256(settings_path)
    atomic_json(qwen / "results.json", result)
    refresh_manifest(qwen)
    with pytest.raises(ValueError, match="pinned model identity"):
        REPORT.load_report(llama, qwen, allow_test_fixture=True)


def test_prediction_sign_flip_is_rejected_even_when_aggregate_hash_is_rebound(
    tmp_path, synthetic_results
):
    llama, qwen = make_analysis_pair(tmp_path, synthetic_results)
    with np.load(qwen / "predictions.npz", allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    arrays["margins"][:, 0] *= -1
    atomic_npz(qwen / "predictions.npz", **arrays)
    refresh_manifest(qwen)
    with pytest.raises(ValueError, match="positive orientation"):
        REPORT.load_report(llama, qwen, allow_test_fixture=True)


def test_export_binds_artifacts_and_does_not_overwrite(tmp_path, synthetic_results, monkeypatch):
    llama, qwen = make_analysis_pair(tmp_path, synthetic_results)

    def engineering_plot(summary, output):
        assert summary["test_fixture"]
        for suffix in ("pdf", "svg", "png"):
            (output / f"fresh_readout.{suffix}").write_bytes(b"engineering plot placeholder")

    monkeypatch.setattr(REPORT, "figures", engineering_plot)
    output = tmp_path / "export"
    REPORT.export_report(llama, qwen, output, allow_test_fixture=True)
    manifest = read_json(output / "manifest.json")
    assert manifest["status"] == "complete" and manifest["test_fixture"]
    assert len(manifest["input_sha256"]) == 6
    assert all(file_sha256(output / p) == d for p, d in manifest["output_sha256"].items())
    assert "report_fresh_readout.py" in next(iter(manifest["reporter_source_sha256"]))
    before = (output / "results.md").read_bytes()
    with pytest.raises(ValueError, match="fresh report directory"):
        REPORT.export_report(llama, qwen, output, allow_test_fixture=True)
    assert (output / "results.md").read_bytes() == before


def test_matplotlib_creates_all_requested_standalone_formats(tmp_path, synthetic_results):
    pytest.importorskip("matplotlib")
    llama, qwen = make_analysis_pair(tmp_path, synthetic_results)
    figures = tmp_path / "figures"
    REPORT.export_report(llama, qwen, figures, allow_test_fixture=True)
    assert all(
        (figures / f"fresh_readout.{s}").stat().st_size > 1000 for s in ("pdf", "svg", "png")
    )
    assert "ENGINEERING FIXTURE" in (figures / "fresh_readout.svg").read_text(encoding="utf-8")
