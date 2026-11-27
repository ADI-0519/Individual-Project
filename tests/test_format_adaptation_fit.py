"""Synthetic protocol checks; no model forwards or scientific data are used."""

from __future__ import annotations

import copy
from itertools import product

import numpy as np
import pytest
from sklearn.preprocessing import normalize

from latent_probes import format_adaptation_fit as fit
from latent_probes.storage import atomic_json, atomic_npz, file_sha256, read_json


def training_rows():
    rows = []
    for group_index in range(16):
        group = f"new_fit_{group_index:02d}"
        for initial, update, b1, b2, world, mapping in product((0, 1), repeat=6):
            if initial ^ update ^ world ^ mapping != group_index % 2:
                continue
            state = dict(
                group_id=group,
                mechanism="correction",
                initial=initial,
                update=update,
                first_receipt=b1,
                receipt=b2,
                world=world,
                name_map=mapping,
            )
            assignment = "A" if world == (group_index // 2) % 2 else "B"
            for renderer in ("A",) if assignment == "A" else ("A", "B"):
                identifier = f"{group}_{initial}{update}{b1}{b2}{world}{mapping}_{renderer}"
                rows.append(
                    {
                        "row_id": identifier,
                        "case_id": identifier,
                        "group_id": group,
                        "partition": "fit",
                        "split": "fit",
                        "condition": "correction_hard"
                        if initial == 1 - update
                        else "correction_easy",
                        "state": state.copy(),
                        "renderer": renderer,
                        "mixed_format": assignment,
                        "gold": 1 - update if b2 else update if b1 else initial,
                    }
                )
    return rows


def make_inputs(tmp_path):
    rows = training_rows()
    mean = np.array([2, -3, 4, -5, 6, -7], dtype=np.float64)
    scale = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    rng = np.random.default_rng(11)
    raw = rng.normal(0, 0.01, (768, 6))
    for index, row in enumerate(rows):
        raw[index, :4] += 2 * np.array([row["state"][f] for f in fit.FACTOR_FIELDS]) - 1
        raw[index, 4:] += [row["state"]["world"], row["state"]["name_map"]]
        if row["renderer"] == "B":
            raw[index, :4] += [0.2, -0.3, 0.1, 0.4]
    features = (mean + scale * raw).astype(np.float32)
    capture = tmp_path / "capture"
    capture.mkdir()
    atomic_json(
        capture / "rows.json",
        {
            "schema": "format-adaptation-capture-rows-v1",
            "test_fixture": True,
            "model_key": "llama",
            "site": fit.MODEL_SITES["llama"],
            "rows": rows,
        },
    )
    atomic_npz(
        capture / "features.npz",
        row_ids=np.array([r["row_id"] for r in rows]),
        features=features,
        output_margins=np.zeros(768, dtype=np.float64),
        strict_first_token_correct=np.zeros(768, dtype=bool),
    )
    plan = tmp_path / "fit_plan.json"
    preparation = tmp_path / "preparation.json"
    atomic_json(
        plan,
        {
            "schema": "format-adaptation-fit-plan-v1",
            "model_key": "llama",
            "site": fit.MODEL_SITES["llama"],
            "bank_sha256": "synthetic-bank",
            "rows": rows,
        },
    )
    atomic_json(
        preparation,
        {
            "schema": "format-adaptation-preparation-v1",
            "source_sha256": fit.source_bindings(),
            "fit_plan_sha256": {"llama": file_sha256(plan)},
        },
    )
    atomic_json(
        capture / "manifest.json",
        {
            "schema": "format-adaptation-capture-manifest-v1",
            "status": "complete",
            "partition": "fit",
            "completed_rows": 768,
            "rows_sha256": file_sha256(capture / "rows.json"),
            "features_sha256": file_sha256(capture / "features.npz"),
            "bank_sha256": "synthetic-bank",
            "fit_plan_path": str(plan.resolve()),
            "preparation_path": str(preparation.resolve()),
            "input_artifact_sha256": {
                **fit.source_bindings(),
                **{str(p.resolve()): file_sha256(p) for p in (plan, preparation)},
            },
        },
    )
    old = tmp_path / "old"
    old.mkdir()
    heads = {
        f"C2_{f}": {
            "site": fit.MODEL_SITES["llama"],
            "C": 0.125,
            "target": f,
            "array_prefix": f"C2_{f}",
        }
        for f in fit.FACTORS
    }
    heads["old_S1"] = {"site": fit.MODEL_SITES["llama"]}
    atomic_json(
        old / "settings.json",
        {
            "schema": "fresh-readout-fit-v2",
            "status": "frozen",
            "test_fixture": True,
            "capture_provenance": {"model_key": "llama"},
            "hidden_size": 6,
            "heads": heads,
            "pipelines": {
                "S1": {"head_key": "old_S1"},
                "C2": {"shared_C": 0.125, "factor_heads": {f: f"C2_{f}" for f in fit.FACTORS}},
            },
        },
    )
    arrays = {
        f"C2_{f}__{name}": value.copy()
        for f in fit.FACTORS
        for name, value in (("mean", mean), ("scale", scale))
    }
    atomic_npz(old / "heads.npz", **arrays)
    atomic_json(old / "candidates.json", {"synthetic": True})
    atomic_json(
        old / "manifest.json",
        {
            "schema": "fresh-readout-fit-manifest-v2",
            "status": "frozen",
            "output_sha256": {
                name: file_sha256(old / name)
                for name in ("settings.json", "heads.npz", "candidates.json")
            },
        },
    )
    planned = read_json(plan)
    planned.update(
        old_bundle_path=str(old.resolve()),
        old_bundle_sha256={
            name: file_sha256(old / name)
            for name in ("settings.json", "heads.npz", "manifest.json", "candidates.json")
        },
    )
    atomic_json(plan, planned)
    prepared = read_json(preparation)
    prepared["fit_plan_sha256"]["llama"] = file_sha256(plan)
    prepared["source_sha256"].update(
        {
            str((old / name).resolve()): file_sha256(old / name)
            for name in ("settings.json", "heads.npz", "manifest.json", "candidates.json")
        }
    )
    atomic_json(preparation, prepared)
    metadata = read_json(capture / "manifest.json")
    metadata["input_artifact_sha256"][str(preparation.resolve())] = file_sha256(preparation)
    metadata["input_artifact_sha256"][str(plan.resolve())] = file_sha256(plan)
    atomic_json(capture / "manifest.json", metadata)
    return capture, old, rows, features, mean, scale


def test_matched_fit_uses_old_map_penalty_and_unmodified_offset_directions(tmp_path, monkeypatch):
    capture, old, rows, features, mean, scale = make_inputs(tmp_path)
    seen = []
    original = fit.fit_lr

    def observed(x, y, C):
        seen.append((len(x), C))
        return original(x, y, C)

    monkeypatch.setattr(fit, "fit_lr", observed)
    output = tmp_path / "fitted"
    settings = fit.fit_bundle(capture, old, "llama", output, allow_test_fixture=True)
    _, arrays = fit.validate_bundle(output, allow_test_fixture=True)
    assert seen == [(512, 0.125)] * 8 + [(256, 0.25)] * 4
    np.testing.assert_array_equal(arrays["mean"], mean)
    np.testing.assert_array_equal(arrays["scale"], scale)
    np.testing.assert_allclose(
        fit.transform(features, mean, scale),
        normalize((features.astype(np.float64) - mean) / scale),
        atol=1e-15,
    )
    renderers = [r["renderer"] for r in rows]
    scored = fit.score_bundle(settings, arrays, features, renderers)
    a = np.array(renderers) == "A"
    np.testing.assert_array_equal(
        scored["Offset"]["factor_margins"][a], scored["Ahalf256"]["factor_margins"][a]
    )
    np.testing.assert_allclose(
        scored["Offset"]["factor_margins"][~a] - scored["Ahalf256"]["factor_margins"][~a],
        np.broadcast_to(arrays["Offset__B_offset"], (256, 4)),
    )
    assert settings["fit_context"]["test_rows_loaded"] == 0
    assert settings["fit_context"]["scaler_fitted"] is False
    with pytest.raises(ValueError, match="preserve"):
        fit.fit_bundle(capture, old, "llama", output, allow_test_fixture=True)


def test_offset_solver_recovers_constant_shift_and_rejects_separation():
    margins = np.array([-1.0, -1.0, 1.0, 1.0]) + 2.75
    labels = np.array([0, 1, 0, 1])
    offset, record = fit.fit_intercept_offset(margins, labels)
    assert offset == pytest.approx(-2.75, abs=1e-12)
    assert abs(record["mean_loss_derivative"]) < 1e-12
    with pytest.raises(ValueError, match="balanced"):
        fit.fit_intercept_offset(margins, np.zeros(4))


def test_heldout_row_is_rejected_before_any_feature_load(tmp_path, monkeypatch):
    capture, _, _, _, _, _ = make_inputs(tmp_path)
    payload = read_json(capture / "rows.json")
    payload["rows"][0]["partition"] = "test"
    atomic_json(capture / "rows.json", payload)

    def forbidden(*args, **kwargs):
        pytest.fail("Feature archive was opened before held-out exclusion")

    monkeypatch.setattr(fit.np, "load", forbidden)
    with pytest.raises(ValueError, match="forbidden"):
        fit.load_capture(capture, "fit", "llama", allow_test_fixture=True)


def test_membership_requires_same_states_and_preassigned_balanced_paths():
    rows = training_rows()
    indices = fit.training_indices(rows)
    assert len(indices["Mixed512"]) == 512
    changed = copy.deepcopy(rows)
    changed[0]["mixed_format"] = "B" if changed[0]["mixed_format"] == "A" else "A"
    with pytest.raises(ValueError, match="replacement|assignment|paths"):
        fit.training_indices(changed)
    changed = copy.deepcopy(rows)
    changed[0]["gold"] = 1 - changed[0]["gold"]
    with pytest.raises(ValueError, match="Gold"):
        fit.training_indices(changed)


def test_fit_plan_binds_original_metadata_before_feature_io(tmp_path, monkeypatch):
    capture, _, *_ = make_inputs(tmp_path)
    payload = read_json(capture / "rows.json")
    payload["rows"][0]["case_id"] = "unplanned_prompt_identity"
    atomic_json(capture / "rows.json", payload)

    def forbidden(*args, **kwargs):
        pytest.fail("Features were opened before original planned metadata was checked")

    monkeypatch.setattr(fit.np, "load", forbidden)
    with pytest.raises(ValueError, match="predeclared fit-only plan"):
        fit.load_capture(capture, "fit", "llama", allow_test_fixture=True)


def test_failed_fit_stops_and_retains_completed_arm(tmp_path, monkeypatch):
    capture, old, *_ = make_inputs(tmp_path)
    original = fit.fit_lr
    seen = []

    def fail_after_arm(*args):
        seen.append(1)
        if len(seen) == 5:
            raise ValueError("synthetic convergence failure")
        return original(*args)

    monkeypatch.setattr(fit, "fit_lr", fail_after_arm)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="synthetic convergence"):
        fit.fit_bundle(capture, old, "llama", output, allow_test_fixture=True)
    assert len(seen) == 5
    manifest = read_json(output / "fit_manifest.json")
    assert manifest["status"] == "failed_preserved"
    assert manifest["failed_stage"] == "Mixed512/I"
    with np.load(output / "coefficients.npz", allow_pickle=False) as archive:
        assert "A512__coef" in archive.files
    with pytest.raises(ValueError, match="not frozen"):
        fit.validate_bundle(output, allow_test_fixture=True)
