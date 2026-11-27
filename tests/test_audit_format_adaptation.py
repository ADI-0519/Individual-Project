"""Independent auditor fixtures: synthetic states/parameters only, never fits."""

from __future__ import annotations

import importlib.util
import json
from itertools import product
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_format_adaptation.py"
SPEC = importlib.util.spec_from_file_location("independent_adaptation_auditor", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def rows_for(split):
    rows = []
    for index in range(16 if split == "fit" else 24):
        group = f"synthetic_{split}_{index:02d}"
        for i, u, b1, b2, w, m in product((0, 1), repeat=6):
            if split == "fit" and i ^ u ^ w ^ m != index % 2:
                continue
            assigned = "A" if w == (index // 2) % 2 else "B"
            formats = (("A",) if assigned == "A" else ("A", "B")) if split == "fit" else ("A", "B")
            for fmt in formats:
                state = dict(
                    group_id=group,
                    mechanism="correction",
                    initial=i,
                    update=u,
                    first_receipt=b1,
                    receipt=b2,
                    world=w,
                    name_map=m,
                )
                rows.append(
                    dict(
                        group_id=group,
                        partition=split,
                        split=split,
                        renderer=fmt,
                        condition="correction_easy" if i == u else "correction_hard",
                        state=state,
                        gold=1 - u if b2 else u if b1 else i,
                        mixed_format=assigned if split == "fit" else None,
                    )
                )
        if split == "test":
            for fmt in ("A", "B"):
                for i, u, b, w, m in product((0, 1), repeat=5):
                    rows.append(
                        dict(
                            group_id=group,
                            partition=split,
                            split=split,
                            renderer=fmt,
                            condition="one_step",
                            gold=u if b else i,
                            state=dict(
                                group_id=group,
                                mechanism="message",
                                initial=i,
                                update=u,
                                first_receipt=None,
                                receipt=b,
                                world=w,
                                name_map=m,
                            ),
                        )
                    )
                for e, w, m in product((0, 1), repeat=3):
                    rows.append(
                        dict(
                            group_id=group,
                            partition=split,
                            split=split,
                            renderer=fmt,
                            condition="asserted",
                            gold=e,
                            state=dict(
                                group_id=group,
                                mechanism="asserted",
                                asserted=e,
                                world=w,
                                name_map=m,
                            ),
                        )
                    )
    for index, row in enumerate(rows):
        row["row_id"] = f"{split}_row_{index:05d}"
    return rows


def token_summary(row):
    y, n = (1.0, -1.0) if row["gold"] else (-1.0, 1.0)
    normalizer = np.log(np.exp(y) + np.exp(n) + np.exp(-4))
    return dict(
        yes_logit=y,
        no_logit=n,
        yes_minus_no=y - n,
        yes_logprob=float(y - normalizer),
        no_logprob=float(n - normalizer),
        argmax_token_id=2 if y > n else 3,
        argmax_tie_count=1,
        vocab_size=7,
    )


def make_campaign(path):
    campaign = path / "campaign"
    campaign.mkdir()
    fit_rows, test_rows = rows_for("fit"), rows_for("test")
    tokens = {"YES": 2, "NO": 3}
    for row in test_rows:
        row["output_candidate_order"] = ["YES", "NO"]
        row["output_logits"] = token_summary(row)
    mean, scale = np.array([2.0, -1.0, 3.0, -4.0]), np.array([1.0, 2.0, 3.0, 4.0])
    latent = np.array(
        [
            [r["state"][f] for f in audit.FIELDS]
            if r["condition"].startswith("correction_")
            else [r["gold"], 0, 0, 0]
            for r in test_rows
        ]
    )
    features = (mean + scale * (2 * latent - 1)).astype(np.float32)
    parameters = {
        "mean": mean,
        "scale": scale,
        "Offset__B_offset": np.array([0.25, -0.25, 0.1, -0.1]),
    }
    for arm in audit.ARMS:
        parameters[arm + "__coef"] = np.zeros((4, 4)) if arm == "A512" else np.eye(4) * 2
        parameters[arm + "__intercept"] = np.zeros(4)
    source = {str(p): audit.digest(p) for p in audit.CODE}
    models = {}
    for model in ("llama", "qwen"):
        fitting, capture, old = (
            campaign / (model + "_fit"),
            campaign / model / "test",
            campaign / (model + "_old"),
        )
        for directory in (fitting, capture, old):
            directory.mkdir(parents=True)
        fit_path = campaign / model / "fit/rows.json"
        put(fit_path, {"rows": fit_rows})
        old_settings = {
            "pipelines": {"C2": {"shared_C": 0.125, "factor_heads": {f: f for f in audit.FACTORS}}},
            "heads": {
                f: {"array_prefix": f, "site": audit.SITES[model], "C": 0.125}
                for f in audit.FACTORS
            },
        }
        put(old / "settings.json", old_settings)
        np.savez_compressed(
            old / "heads.npz",
            **{f + "__" + n: parameters[n] for f in audit.FACTORS for n in ("mean", "scale")},
        )
        training = {name: [] for name in (*audit.ARMS[:3], "B_offset")}
        for row in fit_rows:
            if row["renderer"] == "A":
                training["A512"].append(row["row_id"])
            if row["renderer"] == row["mixed_format"]:
                training["Mixed512"].append(row["row_id"])
                training["Ahalf256" if row["renderer"] == "A" else "B_offset"].append(row["row_id"])
        inputs = {
            str(p): audit.digest(p) for p in (fit_path, old / "settings.json", old / "heads.npz")
        }
        settings = {
            "schema": "format-adaptation-fit-v1",
            "status": "frozen",
            "model_key": model,
            "site": audit.SITES[model],
            "test_fixture": True,
            "hidden_size": 4,
            "training_group_ids": sorted({r["group_id"] for r in fit_rows}),
            "training_ids": training,
            "old_C2_C": 0.125,
            "old_settings_sha256": audit.digest(old / "settings.json"),
            "input_artifact_sha256": inputs,
            "arms": {
                arm: {"n": n, "C": 0.125 * 512 / n}
                for arm, n in (("A512", 512), ("Mixed512", 512), ("Ahalf256", 256))
            },
        }
        np.savez_compressed(fitting / "coefficients.npz", **parameters)
        settings["coefficients_sha256"] = audit.digest(fitting / "coefficients.npz")
        put(fitting / "settings.json", settings)
        put(
            fitting / "fit_manifest.json",
            {
                "schema": "format-adaptation-fit-manifest-v1",
                "status": "frozen",
                "input_artifact_sha256": inputs,
                "output_sha256": {
                    n: audit.digest(fitting / n) for n in ("settings.json", "coefficients.npz")
                },
            },
        )
        models[model] = {
            field: audit.digest(fitting / name)
            for name, field in (
                ("settings.json", "settings_sha256"),
                ("coefficients.npz", "coefficients_sha256"),
                ("fit_manifest.json", "fit_manifest_sha256"),
            )
        }
    seal = {
        "schema": "format-adaptation-test-seal-v1",
        "models": models,
        "bank_sha256": "synthetic-bank",
        "source_sha256": source,
    }
    put(campaign / "test_seal.json", seal)
    for model in ("llama", "qwen"):
        protocol = campaign / model / "protocol.json"
        put(
            protocol,
            {
                "site": audit.SITES[model],
                "bank_sha256": seal["bank_sha256"],
                "source_sha256": source,
                "tokenizer": {"answer_token_ids": tokens},
            },
        )
        capture = campaign / model / "test"
        put(
            capture / "rows.json",
            {
                "schema": "format-adaptation-capture-rows-v1",
                "model_key": model,
                "site": audit.SITES[model],
                "test_fixture": True,
                "answer_token_ids": tokens,
                "rows": test_rows,
            },
        )
        output = np.array([r["output_logits"]["yes_minus_no"] for r in test_rows], dtype=np.float64)
        np.savez_compressed(
            capture / "features.npz",
            features=features,
            row_ids=np.array([r["row_id"] for r in test_rows]),
            output_margins=output,
            strict_first_token_correct=np.ones(4992, dtype=bool),
        )
        put(
            capture / "manifest.json",
            {
                "schema": "format-adaptation-capture-manifest-v1",
                "status": "complete",
                "partition": "test",
                "completed_rows": 4992,
                "seal_sha256": audit.digest(campaign / "test_seal.json"),
                "bank_sha256": seal["bank_sha256"],
                "input_artifact_sha256": {str(protocol): audit.digest(protocol)},
                "rows_sha256": audit.digest(capture / "rows.json"),
                "features_sha256": audit.digest(capture / "features.npz"),
            },
        )
    return campaign, latent


def revise_capture(campaign, model, name, value):
    directory = campaign / model / "test"
    put(directory / name, value)
    manifest = audit.read(directory / "manifest.json")
    manifest[{"rows.json": "rows_sha256", "features.npz": "features_sha256"}[name]] = audit.digest(
        directory / name
    )
    put(directory / "manifest.json", manifest)


def test_rank_ties_and_fixed_zero_are_hand_checked():
    assert audit.rank_auc([0, 1, 0, 1], [0.0, 0.0, 1.0, 1.0]) == 0.5
    assert audit.rank_auc([0, 0, 1, 1], [-1.0, -1.0, 1.0, 1.0]) == 1
    assert audit.rank_auc([0, 0, 1, 1], [1.0, 1.0, -1.0, -1.0]) == 0
    assert audit.binary_metrics([0, 1], np.zeros(2, dtype=int))["row_accuracy"] == 0.5


def test_argmax_tie_and_conditional_tie_have_distinct_answers():
    row = {
        "gold": 1,
        "output_candidate_order": ["YES", "NO"],
        "output_logits": {
            "yes_logit": 0.0,
            "no_logit": 0.0,
            "yes_minus_no": 0.0,
            "yes_logprob": -1.0,
            "no_logprob": -1.0,
            "vocab_size": 7,
            "argmax_token_id": 2,
            "argmax_tie_count": 2,
        },
    }
    margins, strict = audit.output_scores([row], {"YES": 2, "NO": 3}, {"YES": 2, "NO": 3})
    assert margins.tolist() == [0.0] and strict.tolist() == [True]
    assert (margins > 0).tolist() == [False]
    row["output_logits"]["argmax_tie_count"] = 1
    with pytest.raises(ValueError, match="tie"):
        audit.output_scores([row], {"YES": 2, "NO": 3}, {"YES": 2, "NO": 3})
    row["output_logits"].update(argmax_tie_count=2, argmax_token_id=3)
    with pytest.raises(ValueError, match="tie-break"):
        audit.output_scores([row], {"YES": 2, "NO": 3}, {"YES": 2, "NO": 3})


@pytest.mark.parametrize("damage", ["gold", "candidate_ids", "reordering", "seal", "source"])
def test_corrupted_inputs_stop_before_saved_analysis(tmp_path, monkeypatch, damage):
    campaign, _ = make_campaign(tmp_path)
    if damage in ("gold", "candidate_ids"):
        payload = audit.read(campaign / "llama/test/rows.json")
        if damage == "gold":
            payload["rows"][0]["gold"] = 1 - payload["rows"][0]["gold"]
        else:
            payload["answer_token_ids"] = {"YES": 3, "NO": 2}
        revise_capture(campaign, "llama", "rows.json", payload)
    elif damage == "reordering":
        path = campaign / "llama/test/features.npz"
        with np.load(path, allow_pickle=False) as archive:
            values = {k: archive[k].copy() for k in archive.files}
        values["row_ids"][[0, 1]] = values["row_ids"][[1, 0]]
        np.savez_compressed(path, **values)
        manifest = audit.read(path.parent / "manifest.json")
        manifest["features_sha256"] = audit.digest(path)
        put(path.parent / "manifest.json", manifest)
    else:
        seal = audit.read(campaign / "test_seal.json")
        if damage == "seal":
            seal["models"]["llama"]["coefficients_sha256"] = "0" * 64
        else:
            seal["source_sha256"][str(SCRIPT)] = "0" * 64
        put(campaign / "test_seal.json", seal)
    original_read = audit.read

    def protected_read(path):
        assert "_analysis" not in str(path), "Saved analysis opened before reconstruction passed"
        return original_read(path)

    monkeypatch.setattr(audit, "read", protected_read)
    output = tmp_path / "failed_audit"
    with pytest.raises(ValueError):
        audit.audit(campaign, output, allow_fixture=True)
    assert original_read(output / "audit_receipt.json")["status"] == "failed_preserved"
    assert not (output / "reconstruction_seal.json").exists()


def test_full_reconstruction_is_frozen_before_either_saved_analysis(tmp_path, monkeypatch):
    campaign, latent = make_campaign(tmp_path)
    seal_path = campaign / "test_seal.json"
    seal = audit.read(seal_path)
    for model in ("llama", "qwen"):
        rows, features, parameters, pairs, output_scores, strict, tokens, _ = audit.load_inputs(
            campaign, model, seal, seal_path, allow_fixture=True
        )
        arrays, metrics = audit.reconstruct(
            rows, features, parameters, pairs, output_scores, strict, tokens, model
        )
        # These predictions and primary values follow directly from the authored
        # diagonal parameters; no production implementation supplies an oracle.
        np.testing.assert_array_equal(arrays["factor_margins"][:, 0], np.zeros((4992, 4)))
        np.testing.assert_array_equal(arrays["factor_margins"][:, 1], 2 * latent - 1)
        np.testing.assert_array_equal(arrays["factor_predictions"][:, 1], latent)
        assert metrics["primary"]["difference"] == 1
        assert metrics["primary"]["conditional_percentile_interval_95"] == [1.0, 1.0]
        assert metrics["named_diagnostic"]["difference"] == 0
        assert metrics["arms"]["A512"]["B/correction_hard"]["aggregate"]["row_accuracy"] == 0.5
        assert (
            metrics["arms"]["Mixed512"]["B/correction_hard"]["aggregate"]["complete_quartets"]
            == 192
        )
        assert (
            metrics["factor_diagnostics"]["Mixed512"]["B/correction_all"]["aggregate"]["rows"]
            == 1536
        )
        assert metrics["output_baseline"]["B/one_step"]["conditional"]["row_accuracy"] == 1
        analysis = campaign / (model + "_analysis")
        analysis.mkdir()
        np.savez_compressed(analysis / "predictions.npz", **arrays)
        put(
            analysis / "results.json",
            {
                **metrics,
                "schema": "format-adaptation-analysis-v1",
                "model_key": model,
                "site": audit.SITES[model],
                "confirmatory": False,
            },
        )
        bindings = {str(seal_path): audit.digest(seal_path)}
        bindings.update(
            {
                str(campaign / (model + "_fit") / name): audit.digest(
                    campaign / (model + "_fit") / name
                )
                for name in ("settings.json", "coefficients.npz", "fit_manifest.json")
            }
        )
        put(
            analysis / "manifest.json",
            {
                "schema": "format-adaptation-analysis-manifest-v1",
                "status": "complete",
                "input_artifact_sha256": bindings,
                "output_sha256": {
                    n: audit.digest(analysis / n) for n in ("results.json", "predictions.npz")
                },
            },
        )
    target = tmp_path / "independent_audit"
    original_read = audit.read
    observed = []

    def guarded_read(path):
        if "_analysis" in str(path):
            frozen = original_read(target / "reconstruction_seal.json")
            assert frozen["saved_analysis_artifacts_opened"] == 0
            assert frozen["both_models_reconstructed"] == ["llama", "qwen"]
            for name, expected in frozen["output_sha256"].items():
                assert audit.digest(target / name) == expected
            observed.append(str(path))
        return original_read(path)

    monkeypatch.setattr(audit, "read", guarded_read)
    receipt = audit.audit(campaign, target, allow_fixture=True)
    assert receipt["status"] == "pass" and len(observed) == 4
    assert receipt["campaign"] == str(campaign.resolve())
    assert len(receipt["analysis_output_sha256"]) == 4
    assert all(
        audit.digest(path) == value for path, value in receipt["analysis_output_sha256"].items()
    )
    assert receipt["model_calls"] == receipt["classifier_fits"] == 0
    assert receipt["models"]["llama"]["maximum_prediction_margin_difference"] == 0
    assert receipt["models"]["qwen"]["metric_comparisons"]["numeric"] > 1000
    with pytest.raises(ValueError, match="fresh"):
        audit.audit(campaign, target, allow_fixture=True)


def test_saved_metric_or_prediction_changes_are_detected(tmp_path):
    path = tmp_path / "analysis"
    path.mkdir()
    with pytest.raises(ValueError, match="Numeric metric"):
        audit.compare_tree({"primary": {"difference": 0.5}}, {"primary": {"difference": 0.4}})
    with pytest.raises(ValueError, match="Exact metric"):
        audit.compare_tree({"quartets": 192}, {"quartets": 191})


def test_bootstrap_resamples_whole_contexts_with_nonconstant_differences():
    contexts = {
        f"g{i:02d}": {"Mixed512": float(i % 2), "A512": 0.0, "Offset": 0.0} for i in range(24)
    }
    result = audit.contrast(contexts, "A512")
    assert result["difference"] == 0.5
    assert result["conditional_percentile_interval_95"] == pytest.approx([7 / 24, 17 / 24])
    assert len(result["group_values"]) == 24 and result["quartets"] == 192
