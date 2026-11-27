"""Whole-context pairing and formal endpoint tests on synthetic predictions."""

from __future__ import annotations

import copy
from itertools import product

import numpy as np
import pytest
from test_format_adaptation_fit import make_inputs

from latent_probes import format_adaptation_analysis as analysis
from latent_probes import format_adaptation_fit as fit
from latent_probes.format_adaptation_fit import ARMS, FACTOR_FIELDS, fold_factors
from latent_probes.storage import atomic_json, atomic_npz, file_sha256, read_json, utc_now


def test_rows():
    rows = []
    for group_index, renderer in product(range(24), ("A", "B")):
        group = f"new_test_{group_index:02d}"
        for bits in product((0, 1), repeat=6):
            i, u, b1, b2, w, m = bits
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
                {
                    "group_id": group,
                    "partition": "test",
                    "split": "test",
                    "renderer": renderer,
                    "state": state,
                    "condition": "correction_hard" if i == 1 - u else "correction_easy",
                    "gold": 1 - u if b2 else u if b1 else i,
                }
            )
        for i, u, b, w, m in product((0, 1), repeat=5):
            rows.append(
                {
                    "group_id": group,
                    "partition": "test",
                    "split": "test",
                    "renderer": renderer,
                    "condition": "one_step",
                    "gold": u if b else i,
                    "state": dict(
                        group_id=group,
                        mechanism="message",
                        initial=i,
                        update=u,
                        first_receipt=None,
                        receipt=b,
                        world=w,
                        name_map=m,
                    ),
                }
            )
        for e, w, m in product((0, 1), repeat=3):
            rows.append(
                {
                    "group_id": group,
                    "partition": "test",
                    "split": "test",
                    "renderer": renderer,
                    "condition": "asserted",
                    "gold": e,
                    "state": dict(
                        group_id=group, mechanism="asserted", asserted=e, world=w, name_map=m
                    ),
                }
            )
    for index, row in enumerate(rows):
        row["row_id"] = f"synthetic_{index:05d}"
    return rows


# This is a data helper rather than a pytest test.
test_rows.__test__ = False


def scores_for(rows):
    factors = []
    for row in rows:
        if row["condition"].startswith("correction_"):
            factors.append([row["state"][f] for f in FACTOR_FIELDS])
        else:
            factors.append([row["gold"], 0, 0, 0])
    bits = np.array(factors)
    result = {
        arm: {
            "factor_predictions": bits.copy(),
            "factor_margins": (bits * 2 - 1).astype(float),
            "verdicts": fold_factors(bits),
        }
        for arm in ARMS
    }
    # Every hard B quartet has 3:1 gold. A512 choosing I gets row accuracy .75,
    # yet zero complete quartets; Mixed512 retains every quartet.
    hard_b = np.array([r["renderer"] == "B" and r["condition"] == "correction_hard" for r in rows])
    result["A512"]["factor_predictions"][hard_b] = 0
    result["A512"]["factor_predictions"][hard_b, 0] = bits[hard_b, 0]
    result["A512"]["factor_margins"] = (result["A512"]["factor_predictions"] * 2 - 1).astype(float)
    result["A512"]["verdicts"] = fold_factors(result["A512"]["factor_predictions"])
    return result


def test_complete_quartets_and_bootstrap_keep_paired_context_denominator():
    rows = test_rows()
    scores = scores_for(rows)
    margins = np.array([r["gold"] * 2 - 1 for r in rows], dtype=float)
    result = analysis.score_arrays(
        rows, scores, margins, np.ones(len(rows), dtype=bool), model_key="llama"
    )
    primary = result["primary"]
    assert primary["groups"] == 24 and primary["quartets"] == 192
    assert primary["difference"] == 1
    assert primary["conditional_percentile_interval_95"] == [1, 1]
    assert primary["bootstrap_seed"] == 20261127
    assert result["named_diagnostic"]["difference"] == 0
    assert result["arms"]["A512"]["B/correction_hard"]["aggregate"]["quartets"] == 192
    assert result["arms"]["A512"]["B/correction_hard"]["aggregate"]["complete_quartets"] == 0
    assert result["arms"]["A512"]["B/correction_hard"]["aggregate"]["row_accuracy"] == 0.75
    assert result["factor_diagnostics"]["Mixed512"]["B/correction_all"]["aggregate"]["rows"] == 1536
    assert result["arms"]["Mixed512"]["A/asserted"]["aggregate"]["quartets"] == 0
    assert result["output_baseline"]["A/one_step"]["conditional"]["row_accuracy"] == 1


def test_pairing_is_order_independent_and_full_state_sensitive():
    rows = test_rows()
    pairs = analysis.validate_test_rows(rows)
    assert len(pairs) == 2496
    shuffled = [rows[i] for i in np.random.default_rng(11).permutation(len(rows))]
    assert len(analysis.validate_test_rows(shuffled)) == 2496
    damaged = copy.deepcopy(rows)
    damaged[0]["state"]["world"] = 1 - damaged[0]["state"]["world"]
    with pytest.raises(ValueError, match="Duplicate"):
        analysis.validate_test_rows(damaged)
    with pytest.raises(ValueError, match="disjoint"):
        analysis.validate_test_rows(rows, [rows[0]["group_id"]])


def test_context_bootstrap_does_not_treat_quartets_as_independent():
    values = {f"g{i:02d}": {"Mixed512": i % 2, "A512": 0, "Offset": 0} for i in range(24)}
    result = analysis.paired_difference(values, "A512")
    expected_indices = np.random.default_rng(20261127).integers(0, 24, size=(10000, 24))
    expected = np.array([i % 2 for i in range(24)])[expected_indices].mean(axis=1)
    assert result["difference"] == 0.5
    np.testing.assert_array_equal(
        result["conditional_percentile_interval_95"], np.quantile(expected, [0.025, 0.975])
    )
    with pytest.raises(ValueError, match="fixed 24-context"):
        analysis.paired_difference(dict(list(values.items())[:-1]), "A512")


def test_factor_diagnostics_keep_class_recalls_and_reject_noncorrection_truth():
    rows = [
        r for r in test_rows() if r["renderer"] == "A" and r["condition"].startswith("correction_")
    ]
    truth = np.array([[r["state"][f] for f in FACTOR_FIELDS] for r in rows])
    metrics = analysis.factor_metrics(rows, truth * 2 - 1)
    assert metrics["all_four_correct"] == 1
    assert metrics["factors"]["B1"]["class_recalls"] == {"0": 1.0, "1": 1.0}
    assert metrics["factors"]["B1"]["class_counts"] == {"0": 768, "1": 768}
    one_step = next(r for r in test_rows() if r["condition"] == "one_step")
    with pytest.raises(ValueError, match="only on correction"):
        analysis.factor_metrics([one_step], np.zeros((1, 4)))


def test_safe_bundle_to_sealed_analysis_round_trip_and_plan_gate(tmp_path):
    capture, old, *_ = make_inputs(tmp_path)
    fitted = tmp_path / "fitted"
    fit.fit_bundle(capture, old, "llama", fitted, allow_test_fixture=True)
    rows = test_rows()
    tokens = {"YES": 1, "NO": 0}
    sources = {
        **fit.source_bindings(),
        **{
            str((analysis.ROOT / name).resolve()): file_sha256(analysis.ROOT / name)
            for name in analysis.POLICY_FILES
        },
    }
    protocol = tmp_path / "protocol.json"
    atomic_json(
        protocol,
        {
            "schema": "format-adaptation-capture-protocol-v1",
            "model_key": "llama",
            "site": fit.MODEL_SITES["llama"],
            "bank_sha256": "synthetic-bank",
            "source_sha256": sources,
            "planned_rows": rows,
            "tokenizer": {"answer_token_ids": tokens},
        },
    )
    entry = {
        field: file_sha256(fitted / name)
        for field, name in (
            ("coefficients_sha256", "coefficients.npz"),
            ("settings_sha256", "settings.json"),
            ("fit_manifest_sha256", "fit_manifest.json"),
        )
    }
    seal = tmp_path / "seal.json"
    atomic_json(
        seal,
        {
            "schema": "format-adaptation-test-seal-v1",
            "models": {"llama": entry, "qwen": entry},
            "source_sha256": sources,
            "bank_sha256": "synthetic-bank",
            "created_at": utc_now(),
        },
    )
    directory = tmp_path / "test_capture"
    directory.mkdir()
    atomic_json(
        directory / "rows.json",
        {
            "schema": "format-adaptation-capture-rows-v1",
            "test_fixture": True,
            "model_key": "llama",
            "site": fit.MODEL_SITES["llama"],
            "answer_token_ids": tokens,
            "rows": rows,
        },
    )
    rng = np.random.default_rng(11)
    margins = np.array([r["gold"] * 2 - 1 for r in rows], dtype=np.float64)
    atomic_npz(
        directory / "features.npz",
        features=rng.normal(size=(4992, 6)).astype(np.float32),
        row_ids=np.array([r["row_id"] for r in rows]),
        output_margins=margins,
        strict_first_token_correct=np.ones(4992, dtype=bool),
    )
    atomic_json(
        directory / "manifest.json",
        {
            "schema": "format-adaptation-capture-manifest-v1",
            "status": "complete",
            "partition": "test",
            "completed_rows": 4992,
            "rows_sha256": file_sha256(directory / "rows.json"),
            "features_sha256": file_sha256(directory / "features.npz"),
            "seal_sha256": file_sha256(seal),
            "bank_sha256": "synthetic-bank",
            "protocol_sha256": file_sha256(protocol),
            "input_artifact_sha256": {str(protocol.resolve()): file_sha256(protocol)},
        },
    )
    output = tmp_path / "analysis"
    result = analysis.analyse(directory, fitted, seal, output, allow_test_fixture=True)
    assert result["model_key"] == "llama"
    assert result["journal_verification"]["verified_rows"] == 0
    with np.load(output / "predictions.npz", allow_pickle=False) as archive:
        assert archive["verdicts"].shape == (4992, 4)
        assert archive["factor_margins"].shape == (4992, 4, 4)
        assert archive["row_ids"].tolist() == [r["row_id"] for r in rows]
    assert read_json(output / "manifest.json")["status"] == "complete"
    with pytest.raises(ValueError, match="preserve"):
        analysis.analyse(directory, fitted, seal, output, allow_test_fixture=True)
    # A self-consistent changed capture is still rejected by its original plan.
    changed = read_json(directory / "rows.json")
    changed["rows"][0]["gold"] = 1 - changed["rows"][0]["gold"]
    # Exercise the metadata gate directly before formal-gold validation.
    data = fit.load_capture(directory, "test", "llama", allow_test_fixture=True)
    data["rows"] = changed["rows"]
    with pytest.raises(ValueError, match="pre-inference prepared metadata"):
        analysis.verify_test_plan(data, directory, read_json(seal))
