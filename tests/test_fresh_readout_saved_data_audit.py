"""Synthetic checks for the independent audit; never load real study outcomes."""

from __future__ import annotations

import ast
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "audit_fresh_readout_saved_data.py"
SPEC = importlib.util.spec_from_file_location("independent_fresh_audit", SOURCE)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def head(arrays, key, coefficient, intercept=0):
    width = len(coefficient)
    arrays.update(
        {
            key + "__mean": np.zeros(width, dtype=np.float64),
            key + "__scale": np.ones(width, dtype=np.float64),
            key + "__var": np.ones(width, dtype=np.float64),
            key + "__coef": np.array(coefficient, dtype=np.float64),
            key + "__intercept": np.array([intercept], dtype=np.float64),
        }
    )


def fixture_heads():
    sites = [
        f"layer_{layer}_{position}" for layer in (8, 16, 24, 30) for position in audit.POSITIONS
    ]
    arrays, heads, panel = {}, {}, []
    for index, site in enumerate(sites):
        key = f"source_{index:02d}"
        head(arrays, key, [index + 1, 1])
        heads[key] = {"site": site, "array_prefix": key}
        panel.append({"site": site, "head_key": key})
    for key, coefficient, intercept in (
        ("C1", [0, 1], 0),
        ("C2_I", [1, 0], 0),
        ("C2_M1", [0, 1], 0),
        ("C2_B1", [0, 0], 1),
        ("C2_B2", [0, 0], 1),
    ):
        head(arrays, key, coefficient, intercept)
        heads[key] = {"site": sites[0], "array_prefix": key}
    settings = {
        "hidden_size": 2,
        "capture_provenance": {"sites": sites},
        "heads": heads,
        "source_panel": panel,
        "pipelines": {
            "S0": {"head_key": "source_00"},
            "S1": {"head_key": "source_01"},
            "C1": {"head_key": "C1"},
            "C2": {"factor_heads": {factor: "C2_" + factor for factor in audit.FACTORS}},
        },
    }
    return settings, arrays


def selection_fixture():
    settings, arrays = fixture_heads()
    loss = float(np.log(2))
    original = settings["source_panel"][5]["site"]
    settings["original_site"] = original
    history = {"source_candidates": [], "C1_candidates": [], "C2_candidates": []}
    for entry in settings["source_panel"]:
        for c in audit.DIRECT_GRID:
            history["source_candidates"].append(
                {
                    "site": entry["site"],
                    "C": c,
                    "status": "ok",
                    "training_partition": "source_fit",
                    "development_partition": "source_dev",
                    "dev_group_log_loss": loss,
                }
            )
        entry.update(history["source_candidates"][-5])
        settings["heads"][entry["head_key"]].update(C=0.001, target="Y", dev_group_log_loss=loss)
    settings["pipelines"]["S0"]["head_key"] = "source_05"
    settings["pipelines"]["S1"]["head_key"] = "source_05"
    for method, grid in (("C1", audit.DIRECT_GRID), ("C2", audit.FACTOR_GRID)):
        for c in grid:
            record = {
                "site": original,
                "C": c,
                "status": "ok",
                "dev_group_log_loss": loss,
                "training_partition": "correction_fit",
                "development_partition": "correction_dev",
            }
            if method == "C2":
                record["factors"] = {
                    f: {"status": "ok", "dev_group_log_loss": loss} for f in audit.FACTORS
                }
            history[method + "_candidates"].append(record)
    settings["heads"]["C1"].update(site=original, C=0.001, target="Y", dev_group_log_loss=loss)
    for factor in audit.FACTORS:
        settings["heads"]["C2_" + factor].update(
            site=original, C=0.01, target=factor, dev_group_log_loss=loss
        )
    settings["pipelines"]["C2"].update(
        shared_C=0.01, privileged_supervision=True, external_fold="1-M1 if B2 else M1 if B1 else I"
    )
    settings.update(history)
    return settings, arrays, history


def panel_rows(condition="correction_hard", group="g", renderer="A"):
    rows = []
    for initial, update, first, receipt, world, names in itertools.product((0, 1), repeat=6):
        if condition == "correction_hard" and initial == update:
            continue
        if condition == "correction_easy" and initial != update:
            continue
        if condition == "one_step" and first:
            continue
        target = (
            (update if receipt else initial)
            if condition == "one_step"
            else (1 - update if receipt else update if first else initial)
        )
        rows.append(
            {
                "row_id": f"{group}-{renderer}-{condition}-{len(rows)}",
                "group_id": group,
                "condition": condition,
                "renderer": renderer,
                "gold": target,
                "state": {
                    "initial": initial,
                    "update": update,
                    "first_receipt": None if condition == "one_step" else first,
                    "receipt": receipt,
                    "world": world,
                    "name_map": names,
                },
                "output_yes_no_logitdiff": 0,
                "output_logits": {
                    "argmax_token_id": 1,
                    "argmax_tie_count": 1,
                    "yes_logprob": -2.0,
                    "no_logprob": -3.0,
                },
            }
        )
    return rows


def test_float64_scaler_then_normalize_with_zero_ties_and_not_reverse_order():
    arrays = {}
    head(arrays, "h", [5, -7], 0)
    arrays["h__mean"] = np.array([10.0, -4.0])
    arrays["h__scale"] = np.array([2.0, 4.0])
    features = np.array([[16, 12], [10, -4], [6, 8]], dtype=np.float32)
    observed = audit.linear_margin(features, arrays, "h")
    expected = np.array([(3 * 5 - 4 * 7) / 5, 0, (-2 * 5 - 3 * 7) / np.sqrt(13)])
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-14)
    assert observed.dtype == np.float64
    assert (observed > 0).tolist() == [False, False, False]


def test_all_17_columns_site_order_and_c2_uses_learned_b2_on_asserted():
    settings, arrays = fixture_heads()
    features = np.zeros((3, 12, 2), dtype=np.float32)
    features[0, :, :] = [3, 4]
    features[2, :, :] = [-3, -4]
    rows = [
        {
            "row_id": f"r{i}",
            "condition": "asserted",
            "state": {"receipt": None},
            "output_yes_no_logitdiff": margin,
        }
        for i, margin in enumerate((0.0, -2.0, 2.0))
    ]
    scores = audit.reconstruct(settings, arrays, features, rows)
    assert scores["verdicts"].shape == scores["margins"].shape == (3, 17)
    assert scores["methods"].tolist()[:5] == ["S0", "S1", "C1", "C2", "output"]
    np.testing.assert_allclose(scores["margins"][0, 5:], (np.arange(1, 13) * 3 + 4) / 5)
    assert scores["factor_predictions"].tolist() == [[1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]
    assert scores["verdicts"][:, 3].tolist() == [0, 1, 1]
    assert scores["verdicts"][:, 4].tolist() == [0, 0, 1]
    assert scores["verdicts"][1, :3].tolist() == [0, 0, 0]


def test_metrics_quartets_nuisance_and_receipt_controls_with_negative_result():
    rows = panel_rows()
    labels = np.array([r["gold"] for r in rows])
    perfect = audit.measure(rows, labels, np.where(labels, 3.0, -3.0))
    assert perfect["rows"] == 32
    assert perfect["quartets"] == perfect["complete_quartets"] == 8
    assert perfect["nuisance_pairs"]["world"] == {
        "pairs": 16,
        "verdict_flips": 0,
        "both_correct": 16,
    }
    assert perfect["receipt_pairs"]["receipt"]["changing"] == {
        "pairs": 8,
        "verdict_flips": 8,
        "both_correct": 8,
    }
    for receipt in ("first_receipt", "receipt"):
        assert perfect["receipt_pairs"][receipt]["preserving"] == {
            "pairs": 8,
            "verdict_flips": 0,
            "both_correct": 8,
        }
        assert perfect["receipt_pairs"][receipt]["changing"] == {
            "pairs": 8,
            "verdict_flips": 8,
            "both_correct": 8,
        }
    assert perfect["unrecalibrated_score_diagnostics"]["auroc"] == 1
    inverted = audit.measure(rows, 1 - labels, np.where(labels, -3.0, 3.0))
    assert inverted["row_accuracy"] == inverted["quartet_accuracy"] == 0
    assert inverted["quartet_correct_cells_histogram"] == {"0": 8, "1": 0, "2": 0, "3": 0, "4": 0}
    assert inverted["unrecalibrated_score_diagnostics"]["auroc"] == 0
    with pytest.raises(audit.AuditFailure, match="Incomplete quartet"):
        audit.measure(rows[:-1], labels[:-1])


def test_rank_auc_exact_ties_single_class_and_extreme_sigmoid():
    assert audit.auc_from_ranks([0, 1, 0, 1], [0, 0, 1, 1]) == 0.5
    assert audit.auc_from_ranks([1, 1], [2, 3]) is None
    np.testing.assert_array_equal(audit.sigmoid(np.array([-1000.0, 0, 1000])), [0, 0.5, 1])


def test_actual_easy_and_one_step_receipt_geometry():
    easy = panel_rows("correction_easy")
    assert all(r["state"]["initial"] == r["state"]["update"] for r in easy)
    easy_metrics = audit.measure(easy, [r["gold"] for r in easy])
    assert easy_metrics["receipt_pairs"]["first_receipt"]["preserving"]["pairs"] == 16
    assert easy_metrics["receipt_pairs"]["first_receipt"]["changing"]["pairs"] == 0
    assert easy_metrics["receipt_pairs"]["receipt"]["changing"]["pairs"] == 16
    assert easy_metrics["receipt_pairs"]["receipt"]["preserving"]["pairs"] == 0
    one_step = panel_rows("one_step")
    assert all(r["state"]["first_receipt"] is None for r in one_step)
    assert {r["state"]["receipt"] for r in one_step} == {0, 1}
    one_step_metrics = audit.measure(one_step, [r["gold"] for r in one_step])
    assert one_step_metrics["receipt_pairs"]["first_receipt"]["preserving"]["pairs"] == 0
    assert one_step_metrics["receipt_pairs"]["first_receipt"]["changing"]["pairs"] == 0
    assert one_step_metrics["receipt_pairs"]["receipt"]["changing"]["pairs"] == 8
    assert one_step_metrics["receipt_pairs"]["receipt"]["preserving"]["pairs"] == 8


def test_whole_context_bootstrap_fixed_seed_and_negative_contrast():
    scores = {f"c{i:02d}": {"S0": 0.75, "S1": 0.25 + (i % 3) / 8} for i in range(24)}
    result, draws = audit.primary(scores)
    repeated, repeated_draws = audit.primary(dict(reversed(list(scores.items()))))
    assert result == repeated
    np.testing.assert_array_equal(draws, repeated_draws)
    assert result["difference"] == -0.375
    assert result["conditional_percentile_interval_95"] == [
        -0.4166666666666667,
        -0.3333333333333333,
    ]
    assert len(draws) == 10000 and result["bootstrap_seed"] == 20261122
    with pytest.raises(audit.AuditFailure, match="24 complete"):
        audit.primary({"only": {"S0": 0, "S1": 1}})


def test_group_balanced_development_loss_is_not_row_weighted():
    rows = [{"group_id": "a", "gold": 0}] + [{"group_id": "b", "gold": 1}] * 3
    margins = np.array([0.0, 2.0, 2.0, 2.0])
    observed = audit.development_loss(rows, margins, "Y")
    expected = (np.log(2) + np.logaddexp(0, -2)) / 2
    assert observed == pytest.approx(expected, abs=1e-15)
    assert (
        abs(observed - np.mean(np.logaddexp(0, margins) - np.array([0, 1, 1, 1]) * margins)) > 0.1
    )


def test_train_only_scaler_audit_detects_development_absorption():
    settings, arrays = fixture_heads()
    # Keep one source head here: the full inventory is checked separately.
    settings["heads"] = {
        "source_00": {
            "site": settings["source_panel"][0]["site"],
            "array_prefix": "source_00",
            "training_partition": "source_fit",
            "scaler_n_samples_seen": 512,
            "target": "Y",
            "dev_group_log_loss": float(np.log(2)),
        }
    }
    settings["fit_context"] = {"test_rows_loaded": 0, "development_absorbed": False}
    rows = []
    for partition in audit.PARTITIONS:
        subset = [
            {
                "row_id": f"{partition}-{i}",
                "group_id": partition + "g",
                "partition": partition,
                "gold": i % 2,
            }
            for i in range(512)
        ]
        rows.extend(subset)
        settings[partition + "_ids"] = [r["row_id"] for r in subset]
    settings["heads"]["source_00"]["training_ids"] = settings["source_fit_ids"]
    features = np.zeros((2048, 12, 2), dtype=np.float32)
    features[:512, 0, 0] = np.tile([-1, 1], 256)
    features[512:1024, 0, 0] = 10
    head(arrays, "source_00", [0, 0])
    arrays["source_00__var"] = np.array([1.0, 0.0])
    result = audit.check_training(
        settings, arrays, rows, features, [{"row_id": "test", "group_id": "testg"}]
    )
    assert result["source_00"]["scaler_max_abs_differences"] == {"mean": 0, "var": 0, "scale": 0}
    arrays["source_00__mean"] = np.array([5.0, 0.0])
    with pytest.raises(audit.AuditFailure, match="train-only mean"):
        audit.check_training(
            settings, arrays, rows, features, [{"row_id": "test", "group_id": "testg"}]
        )


def test_exact_source_tie_precedence():
    candidates = [
        {"site": site, "C": c, "dev_group_log_loss": 1.0}
        for site, c in (
            ("layer_8_account_last", 0.1),
            ("layer_16_prompt_last", 0.1),
            ("layer_8_prompt_last", 0.01),
        )
    ]
    assert min(candidates, key=lambda r: audit.source_key(r, "layer_16_prompt_last"))["C"] == 0.01
    assert (
        min(candidates[:2], key=lambda r: audit.source_key(r, "layer_16_prompt_last"))["site"]
        == "layer_16_prompt_last"
    )


def test_complete_85_fit_history_selection_and_retained_candidate_binding():
    settings, _, history = selection_fixture()
    result = audit.validate_selection(settings, history)
    assert result["individual_candidate_fits"] == 85 and result["S0_equals_S1"] is True
    assert result["selected_site"] == "layer_16_prompt_last"
    settings["heads"]["C1"]["dev_group_log_loss"] += 0.1
    with pytest.raises(audit.AuditFailure, match="Retained C1 loss"):
        audit.validate_selection(settings, history)


def test_native_capture_schema_raw_logits_geometry_and_swapped_interface_detection(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    audit_output = tmp_path / "audit"
    audit_output.mkdir()
    planned = {
        "row_id": "one",
        "partition": "test",
        "condition": "asserted",
        "gold": 1,
        "group_id": "g",
        "renderer": "A",
        "state": {"attributed_belief": 1},
    }
    protocol = {
        "planned_rows": [planned],
        "sites": [f"site{i}" for i in range(12)],
        "hidden_size": 2,
        "tokenizer": {"answer_token_ids": {"YES": 11, "NO": 22}},
    }
    journal = capture / "row_journal" / "one"
    journal.mkdir(parents=True)
    features = np.arange(24, dtype=np.float32).reshape(12, 2)
    candidates = np.array([3.0, 1.0], dtype=np.float32)
    np.savez(journal / "raw_forward.npz", features=features, output_candidate_logits=candidates)
    np.savez(
        journal / "arrays.npz",
        features=features,
        output_candidate_logits=candidates,
        output_candidate_token_ids=np.array([11, 22]),
    )
    logits = {"yes_logit": 3.0, "no_logit": 1.0, "yes_minus_no": 2.0}
    complete = {
        **planned,
        "status": "complete",
        "output_logits": logits,
        "arrays_sha256": audit.digest(journal / "arrays.npz"),
        "raw_forward_sha256": audit.digest(journal / "raw_forward.npz"),
    }
    (journal / "complete.json").write_text(json.dumps(complete))
    (journal / "started.json").write_text(json.dumps({"started_at": "2026-11-22T01:00:01+00:00"}))
    row = {**complete, "output_yes_no_logitdiff": 2.0}
    (capture / "rows.json").write_text(
        json.dumps({"schema": "fresh-readout-capture-v1", "rows": [row]})
    )
    np.savez(
        capture / "features.npz",
        features=features[None],
        row_ids=np.array(["one"]),
        sites=np.array(protocol["sites"]),
    )
    manifest = {
        "schema": "fresh-readout-capture-manifest-v1",
        "status": "complete",
        "partition": "test",
        "completed_rows": 1,
        "rows_sha256": audit.digest(capture / "rows.json"),
        "features_sha256": audit.digest(capture / "features.npz"),
    }
    (capture / "manifest.json").write_text(json.dumps(manifest))
    rows, cache = audit.capture(
        audit.Reader(audit_output),
        capture,
        protocol,
        "test",
        {"created_at": "2026-11-22T01:00:00+00:00"},
    )
    assert rows[0]["output_yes_no_logitdiff"] == 2.0
    np.testing.assert_array_equal(cache[0], features)
    # Detect the same row/site array shape carrying swapped physical sites.
    np.savez(
        capture / "features.npz",
        features=features[None],
        row_ids=np.array(["one"]),
        sites=np.array(list(reversed(protocol["sites"]))),
    )
    manifest["features_sha256"] = audit.digest(capture / "features.npz")
    (capture / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(audit.AuditFailure, match="Wrong cache identities"):
        audit.capture(audit.Reader(audit_output), capture, protocol, "test")


def test_native_authorization_and_both_completion_gate_precede_data_access(tmp_path):
    output = tmp_path / "audit"
    models = [
        (model, "unopened-fit", "unopened-test", tmp_path / model) for model in ("llama", "qwen")
    ]
    with pytest.raises(audit.AuditFailure, match="explicit BOTH"):
        audit.run_audit(models, "unopened-seal", output)
    assert not output.exists()
    for model in ("llama", "qwen"):
        (tmp_path / model).mkdir()
        (tmp_path / model / "manifest.json").write_text(json.dumps({"schema": "incomplete"}))
    with pytest.raises(audit.AuditFailure, match="final manifests"):
        audit.run_audit(models, "unopened-seal", output, authorized_both_complete=True)
    events = [json.loads(line) for line in (output / "access_log.jsonl").read_text().splitlines()]
    assert all("unopened" not in event.get("path", "") for event in events)


def test_both_frozen_before_either_saved_outcome_read(tmp_path, monkeypatch):
    models = []
    manifest = {"schema": "fresh-readout-analysis-manifest-v2", "output_sha256": {}}
    for model in ("llama", "qwen"):
        directory = tmp_path / model
        directory.mkdir()
        np.savez(directory / "predictions.npz", values=np.array([7]))
        (directory / "results.json").write_text(
            json.dumps(
                {
                    "metric": -1.0,
                    "schema": "fresh-readout-analysis-v2",
                    "model_key": model,
                    "test_fixture": False,
                }
            )
        )
        manifest["output_sha256"] = {
            name: audit.digest(directory / name) for name in ("predictions.npz", "results.json")
        }
        (directory / "manifest.json").write_text(json.dumps(manifest))
        models.append((model, "fit", "test", directory))
    seal = tmp_path / "seal.json"
    seal.write_text(
        json.dumps(
            {
                "schema": "fresh-readout-fit-seal-v2",
                "status": "frozen",
                "fitted_artifact_sha256": {str(seal): "ignored"},
            }
        )
    )
    monkeypatch.setattr(audit.Reader, "bindings", lambda *args: None)
    monkeypatch.setattr(
        audit,
        "reconstruct_model",
        lambda *args: ({"values": np.array([7])}, {"metric": -1.0}, np.zeros(10000), {}),
    )
    result = audit.run_audit(models, seal, tmp_path / "audit", authorized_both_complete=True)
    assert result["status"] == "pass"
    events = [
        json.loads(line)
        for line in (tmp_path / "audit" / "access_log.jsonl").read_text().splitlines()
    ]
    freeze_index = next(
        i for i, event in enumerate(events) if event["kind"] == "both_reconstructions_frozen"
    )
    saved_reads = [
        i
        for i, event in enumerate(events)
        if event["kind"] == "read_begin" and "saved analysis" in event.get("role", "")
    ]
    assert len(saved_reads) == 4 and min(saved_reads) > freeze_index
    assert not json.loads((tmp_path / "audit" / "reconstruction_freeze.json").read_text())[
        "saved_results_or_predictions_parsed"
    ]


def test_recursive_comparison_reports_negative_mismatch_and_integer_exactness():
    expected = {"negative": -0.1, "count": 12, "context": [{"score": 0.5}]}
    observed = {
        "negative": 0.1,
        "count": 12 + 1e-10,
        "context": [{"score": 0.5 + 1e-10}],
        "prose": "extra",
    }
    differences = audit.compare_tree(expected, observed)
    assert [d["path"] for d in differences] == ["/negative", "/count"]


@pytest.mark.parametrize("extra", [999, -0.1, True, [1, 2], {"nested_numeric_metric": 0.99}])
@pytest.mark.parametrize(
    "path",
    [
        "",
        "/primary",
        "/methods/S1/strata/A/correction_hard/aggregate",
        "/methods/S1/strata/A/correction_hard/groups/0",
        "/methods/S1/strata/A/correction_hard/groups/0/nuisance_pairs/world",
    ],
)
def test_comparison_rejects_unexpected_numeric_and_structured_metrics(extra, path):
    differences = audit.compare_tree(
        {"checked": 0.5}, {"checked": 0.5, "unreconstructed": extra}, path
    )
    assert differences == [
        {"path": path + "/unreconstructed", "reason": "unexpected nonprose metric field"}
    ]


def test_comparison_accepts_current_result_prose_and_explicit_metadata_only():
    expected = {
        "primary": {"difference": -0.2},
        "factor_diagnostics": {},
        "methods": {"S0": {"score": 0.5}},
    }
    observed = {
        "primary": {"difference": -0.2, "uncertainty_scope": "Conditional only"},
        "factor_diagnostics": {"scope": "Two-message only"},
        "methods": {"S0": {"score": 0.5, "interpretation": "Unrecalibrated"}},
        "schema": "fresh-readout-analysis-v2",
        "model_key": "llama",
        "created_at": "2026-11-22",
        "test_fixture": False,
        "human_review_complete": False,
        "confirmatory_result": False,
        "fresh_test": True,
        "source_sha256": {"/synthetic/source.py": "a" * 64},
    }
    assert audit.compare_tree(expected, observed) == []
    observed["source_sha256"]["unreviewed_numeric"] = 99
    assert audit.compare_tree(expected, observed) == [
        {"path": "/source_sha256", "reason": "unexpected nonprose metric field"}
    ]


def complete_test_rows():
    rows = []
    for group in (f"g{i:02d}" for i in range(24)):
        for renderer in ("A", "B"):
            for condition in ("one_step", "correction_easy", "correction_hard"):
                subset = panel_rows(condition, group, renderer)
                rows.extend(subset)
            for target, world, names in itertools.product((0, 1), repeat=3):
                rows.append(
                    {
                        "row_id": f"{group}-{renderer}-asserted-{target}{world}{names}",
                        "group_id": group,
                        "renderer": renderer,
                        "condition": "asserted",
                        "gold": target,
                        "state": {"attributed_belief": target, "world": world, "name_map": names},
                        "output_yes_no_logitdiff": 2 * target - 1,
                        "output_logits": {
                            "argmax_token_id": 1,
                            "argmax_tie_count": 1,
                            "yes_logprob": -2.0,
                            "no_logprob": -3.0,
                        },
                    }
                )
    return rows


def test_full_4992_rows_all_17_methods_and_fixed_negative_primary():
    rows = complete_test_rows()
    assert len(rows) == 4992
    labels = np.array([r["gold"] for r in rows])
    predictions = np.tile(labels[:, None], (1, 17))
    selected = [
        i
        for i, r in enumerate(rows)
        if r["renderer"] == "A" and r["condition"] == "correction_hard"
    ]
    predictions[selected, 1] = 1 - labels[selected]
    bits = np.array([[r["state"].get(field) or 0 for field in audit.FACTOR_FIELDS] for r in rows])
    settings, _ = fixture_heads()
    reconstructed = {
        "methods": np.array(
            ["S0", "S1", "C1", "C2", "output"]
            + ["source_" + s for s in settings["capture_provenance"]["sites"]]
        ),
        "verdicts": predictions,
        "margins": predictions * 6 - 3,
        "factor_predictions": bits,
    }
    results, draws = audit.summarize(rows, reconstructed, {"YES": 1, "NO": 2})
    assert len(results["methods"]) == 17
    assert all(len(method["strata"]) == 8 for method in results["methods"].values())
    assert all(
        len(stratum["groups"]) == 24
        for method in results["methods"].values()
        for stratum in method["strata"].values()
    )
    assert results["primary"]["difference"] == -1
    assert results["primary"]["conditional_percentile_interval_95"] == [-1, -1]
    np.testing.assert_array_equal(draws, np.full(10000, -1.0))
    assert results["factor_diagnostics"]["B/correction_hard"]["accuracy"] == [1, 1, 1, 1]
    assert results["methods"]["C2"]["strata"]["B/asserted"]["aggregate"]["quartet_accuracy"] is None
    assert (
        results["strict_first_token_diagnostic"]["A/asserted"]["accuracy_full_denominator"] == 0.5
    )


def test_full_current_schema_run_through_reconstruct_model_and_comparison(tmp_path, monkeypatch):
    """Actual fit/protocol/seal/results files; full native row/tensor dimensions.

    Capture is the sole replaced boundary: its real raw-journal schema is covered
    above. Replacing it avoids 70,400 tiny OneDrive test files. All fitting
    metadata, candidate selection, scaler recomputation, inference, metrics,
    freezing, and final comparisons run through their real audit code.
    """
    test_rows = complete_test_rows()
    training_rows = []
    counts = {"source_fit": 32, "source_dev": 16, "correction_fit": 16, "correction_dev": 8}
    for partition in audit.PARTITIONS:
        for i in range(512):
            training_rows.append(
                {
                    "row_id": f"{partition}-{i}",
                    "partition": partition,
                    "group_id": f"{partition}-g{i // (512 // counts[partition])}",
                    "gold": i % 2,
                    "state": {field: i % 2 for field in audit.FACTOR_FIELDS},
                }
            )
    training_features = np.tile(np.array([[-1, -1], [1, 1]], dtype=np.float32), (1024, 12))
    training_features = training_features.reshape(2048, 12, 2)
    test_features = np.zeros((4992, 12, 2), dtype=np.float32)
    safe_source = tmp_path / "synthetic_fitting_source.txt"
    safe_source.write_text("Synthetic fixture only. No scientific/model fitting occurred.\n")
    source_binding = {str(safe_source): audit.digest(safe_source)}
    expected_scores = {
        "row_ids": np.array([r["row_id"] for r in test_rows]),
        "verdicts": np.zeros((4992, 17), dtype=np.int64),
        "margins": np.zeros((4992, 17)),
        "factor_margins": np.zeros((4992, 4)),
        "factor_predictions": np.zeros((4992, 4), dtype=np.int64),
    }
    expected_scores["margins"][:, 3] = -1
    expected_scores["margins"][:, 4] = [r["output_yes_no_logitdiff"] for r in test_rows]
    expected_scores["verdicts"][:, 4] = expected_scores["margins"][:, 4] > 0
    models, fitted_hashes, protocol_hashes = [], {}, {}
    for model in ("llama", "qwen"):
        fit, analysis = tmp_path / (model + "_fit"), tmp_path / (model + "_analysis")
        fit.mkdir()
        analysis.mkdir()
        settings, arrays, history = selection_fixture()
        if model == "qwen":

            def convert(value):
                encoded = json.dumps(value)
                for old, new in ((8, 7), (16, 14), (24, 21), (30, 26)):
                    encoded = encoded.replace(f"layer_{old}_", f"layer_{new}_")
                return json.loads(encoded)

            settings, history = convert(settings), convert(history)
        for key in settings["heads"]:
            arrays[key + "__coef"][:] = 0
            arrays[key + "__intercept"][:] = 0
        settings.update(schema="fresh-readout-fit-v2", status="frozen", test_fixture=False)
        settings["capture_provenance"]["model_key"] = model
        for partition in audit.PARTITIONS:
            settings[partition + "_ids"] = [
                r["row_id"] for r in training_rows if r["partition"] == partition
            ]
        for key, metadata in settings["heads"].items():
            partition = "source_fit" if key.startswith("source_") else "correction_fit"
            metadata.update(
                training_partition=partition,
                training_ids=settings[partition + "_ids"],
                scaler_n_samples_seen=512,
            )
        protocol_path = tmp_path / (model + "_protocol.json")
        protocol = {
            "model_key": model,
            "test_fixture": False,
            "sites": settings["capture_provenance"]["sites"],
            "hidden_size": 2,
            "planned_rows": training_rows + test_rows,
            "source_sha256": source_binding,
            "tokenizer": {"answer_token_ids": {"YES": 1, "NO": 2}},
        }
        protocol_path.write_text(json.dumps(protocol))
        protocol_hashes[model] = audit.digest(protocol_path)
        settings["fit_context"] = {
            "protocol_path": str(protocol_path),
            "capture_dir": "synthetic-training",
            "test_rows_loaded": 0,
            "development_absorbed": False,
        }
        settings["fitting_source_sha256"] = source_binding
        settings["input_artifact_sha256"] = {str(protocol_path): protocol_hashes[model]}
        settings["policy"] = {
            "source_allocation": {"n": 512, "seed": 11, "support": "full"},
            "training_partitions": list(audit.PARTITIONS),
            "development_absorbed": False,
            "transform": ["float64", "training_only_StandardScaler", "row_L2"],
            "logistic_regression": {
                "solver": "lbfgs",
                "penalty": "l2",
                "fit_intercept": True,
                "max_iter": 2000,
                "tol": 1e-4,
                "class_weight": None,
                "sample_weight": None,
                "random_state": 11,
            },
            "direct_C_grid": list(audit.DIRECT_GRID),
            "factor_C_grid": list(audit.FACTOR_GRID),
            "selected_refit": False,
            "calibration": None,
            "margin_orientation": "positive_is_symbolic_target_1",
            "threshold": "margin > 0; exact zero predicts 0",
            "numerical_thread_limit": 4,
        }
        np.savez(fit / "heads.npz", **arrays)
        (fit / "settings.json").write_text(json.dumps(settings))
        (fit / "candidates.json").write_text(json.dumps(history))
        (fit / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "frozen",
                    "output_sha256": {
                        name: audit.digest(fit / name)
                        for name in ("settings.json", "heads.npz", "candidates.json")
                    },
                }
            )
        )
        fitted_hashes.update(
            {
                str((fit / name).resolve()): audit.digest(fit / name)
                for name in ("settings.json", "heads.npz", "candidates.json", "manifest.json")
            }
        )
        expected_scores["methods"] = np.array(
            ["S0", "S1", "C1", "C2", "output"]
            + ["source_" + site for site in settings["capture_provenance"]["sites"]]
        )
        # This round trip is an interface test; independent numeric oracles are
        # the separate normalization, control metrics and negative-primary tests.
        metrics, _ = audit.summarize(test_rows, expected_scores, {"YES": 1, "NO": 2})
        metrics.update(schema="fresh-readout-analysis-v2", model_key=model, test_fixture=False)
        (analysis / "results.json").write_text(json.dumps(metrics))
        np.savez(analysis / "predictions.npz", **expected_scores)
        (analysis / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "fresh-readout-analysis-manifest-v2",
                    "output_sha256": {
                        name: audit.digest(analysis / name)
                        for name in ("results.json", "predictions.npz")
                    },
                }
            )
        )
        models.append((model, fit, "synthetic-test", analysis))
    seal = tmp_path / "seal.json"
    seal.write_text(
        json.dumps(
            {
                "schema": "fresh-readout-fit-seal-v2",
                "status": "frozen",
                "model_protocol_sha256": protocol_hashes,
                "fitted_artifact_sha256": fitted_hashes,
            }
        )
    )

    def synthetic_capture(reader, directory, protocol, partition, seal=None):
        assert directory == ("synthetic-test" if partition == "test" else "synthetic-training")
        assert protocol["test_fixture"] is False
        reader.event("synthetic_capture_boundary", partition=partition)
        return (
            (test_rows, test_features)
            if partition == "test"
            else (training_rows, training_features)
        )

    monkeypatch.setattr(audit, "capture", synthetic_capture)
    result = audit.run_audit(models, seal, tmp_path / "audit", authorized_both_complete=True)
    assert result["status"] == "pass"
    assert set(result["models"]) == {"llama", "qwen"}
    assert all(m["all_metrics_match"] for m in result["models"].values())
    for model in ("llama", "qwen"):
        training = json.loads(
            (tmp_path / "audit" / model / "training_selection_checks.json").read_text()
        )
        assert len(training["retained_heads"]) == 17
        assert training["selection"]["S0_equals_S1"] is True


def test_fixture_fit_metadata_cannot_enter_native_reconstruction(tmp_path):
    fit, output = tmp_path / "fit", tmp_path / "audit"
    fit.mkdir()
    output.mkdir()
    (fit / "settings.json").write_text(
        json.dumps({"schema": "fresh-readout-fit-v2", "status": "frozen", "test_fixture": True})
    )
    with pytest.raises(audit.AuditFailure, match="nonfixture fit"):
        audit.reconstruct_model(audit.Reader(output), "llama", fit, "unopened-test", {})


def test_auditor_has_no_production_sklearn_or_model_imports():
    modules = []
    for node in ast.walk(ast.parse(SOURCE.read_text())):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    assert all(
        module.split(".")[0]
        in {
            "__future__",
            "argparse",
            "hashlib",
            "json",
            "re",
            "collections",
            "datetime",
            "pathlib",
            "numpy",
        }
        for module in modules
    )
