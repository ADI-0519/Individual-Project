"""Synthetic fitting-policy and boundary checks; no real model observations."""

from __future__ import annotations

import copy
import itertools
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler, normalize

from latent_probes import fresh_readout_fit as fit
from latent_probes.config import ModelSettings
from latent_probes.fresh_readout_capture import BLOCKS, panel_sites
from latent_probes.main_study_run import MODELS
from latent_probes.storage import atomic_json, atomic_npz, file_sha256, read_json


def fixture_rows():
    rows = []
    for partition in fit.FIT_PARTITIONS:
        for index, (i, u, b1, b2, nuisance) in enumerate(itertools.product((0, 1), repeat=5)):
            state = dict(initial=i, update=u, first_receipt=b1, receipt=b2)
            target = int(1 - u if b2 else u if b1 else i)
            row_id = f"fixture_{partition}_{index:03d}"
            rows.append(
                {
                    "record_id": row_id,
                    "case_id": row_id,
                    "partition": partition,
                    "group_id": f"{partition}_{index // 8}",
                    "renderer": "A",
                    "gold": i if partition.startswith("source") else target,
                    "state": state,
                    "nuisance": nuisance,
                    "messages": [{"role": "user", "content": row_id}],
                }
            )
    return rows


def make_capture(tmp_path, *, model="llama", bank_path=None, rows=None):
    rows = copy.deepcopy(rows if rows is not None else fixture_rows())
    bank_path = bank_path or tmp_path / "fixture-bank.json"
    if not bank_path.exists():
        atomic_json(bank_path, {"rows": rows})
    sources = fit._fitting_sources()
    protocol_path = tmp_path / f"{model}-protocol.json"
    sources[str(Path(__file__).resolve())] = file_sha256(__file__)
    settings = ModelSettings(
        model_id=MODELS[model]["model_id"],
        revision=MODELS[model]["revision"],
        capture_layers=list(BLOCKS[model]),
        dtype="bfloat16",
        quantization="4bit",
        device="cuda",
        max_input_tokens=2048,
        local_files_only=True,
    )
    planned = [{**r, "row_id": r["record_id"]} for r in rows]
    protocol = {
        "schema": "fresh-readout-capture-protocol-v1",
        "test_fixture": True,
        "bank_path": str(bank_path.resolve()),
        "bank_file_sha256": file_sha256(bank_path),
        "source_sha256": sources,
        "model_key": model,
        "model_settings": asdict(settings),
        "sites": panel_sites(model),
        "original_site": f"layer_{16 if model == 'llama' else 14}_prompt_last",
        "hidden_size": 6,
        "planned_rows": planned,
        "forward_contract": {
            "source_path": str(Path(__file__).resolve()),
            "source_sha256": file_sha256(__file__),
        },
    }
    atomic_json(protocol_path, protocol)
    digest = file_sha256(protocol_path)
    completed = [
        {**r, "status": "complete", "test_fixture": True, "protocol_sha256": digest}
        for r in planned
    ]
    rng = np.random.default_rng(20261122)
    features = rng.normal(0, 0.02, size=(len(rows), 12, 6)).astype(np.float32)
    for index, row in enumerate(rows):
        s = row["state"]
        bits = [s[f] for f in fit.FACTOR_FIELDS]
        features[index, :, :4] += 2 * np.asarray(bits, dtype=np.float32) - 1
        features[index, :, 4] += row["nuisance"]
        if row["partition"].endswith("dev"):
            features[index, :, 5] += 5  # Makes development absorption detectable.
    directory = tmp_path / f"{model}-training"
    directory.mkdir()
    provenance = {
        "protocol_sha256": digest,
        "bank_file_sha256": file_sha256(bank_path),
        "model_key": model,
        "model_settings": asdict(settings),
        "sites": panel_sites(model),
        "original_site": protocol["original_site"],
        "partition": "training",
    }
    atomic_json(
        directory / "rows.json",
        {
            "schema": "fresh-readout-capture-v1",
            "test_fixture": True,
            "rows": completed,
            "provenance": provenance,
        },
    )
    atomic_npz(
        directory / "features.npz",
        features=features,
        row_ids=np.asarray([r["row_id"] for r in planned]),
        sites=np.asarray(panel_sites(model)),
    )
    atomic_json(
        directory / "manifest.json",
        {
            "schema": "fresh-readout-capture-manifest-v1",
            "status": "complete",
            "partition": "training",
            "test_fixture": True,
            "protocol_sha256": digest,
            "completed_rows": len(rows),
            "authorization": {"sha256": sources},
            "rows_sha256": file_sha256(directory / "rows.json"),
            "features_sha256": file_sha256(directory / "features.npz"),
        },
    )
    return directory, bank_path, protocol_path, features


@pytest.fixture
def trained(tmp_path):
    directory, bank, protocol, features = make_capture(tmp_path)
    output = tmp_path / "fitted"
    settings = fit.fit_bundle(directory, bank, protocol, output, allow_test_fixture=True)
    return output, settings, features, (directory, bank, protocol)


def test_round_trip_has_all_sites_train_only_scalers_and_no_refit(tmp_path, monkeypatch):
    directory, bank, protocol, features = make_capture(tmp_path)
    called = []
    original = fit._fit_candidate

    def record(x, labels, C):
        called.append((x.shape, len(labels), C))
        return original(x, labels, C)

    monkeypatch.setattr(fit, "_fit_candidate", record)
    output = tmp_path / "fitted"
    fit.fit_bundle(directory, bank, protocol, output, allow_test_fixture=True)
    settings, arrays = fit.validate_fit_bundle(output, allow_test_fixture=True)
    assert len(called) == 60 + 5 + 20  # Actual selected candidates are saved, never refitted.
    assert {n for _, n, _ in called} == {32}
    assert len(settings["source_panel"]) == 12 and len(settings["heads"]) == 17
    assert len(settings["source_candidates"]) == 60
    assert len(settings["C1_candidates"]) == len(settings["C2_candidates"]) == 5
    assert settings["policy"]["development_absorbed"] is False
    assert settings["fit_context"]["test_rows_loaded"] == 0
    assert settings["numerical_execution"]["thread_limit"] == 4
    for index, panel in enumerate(settings["source_panel"]):
        key = panel["head_key"]
        expected = StandardScaler().fit(features[:32, index].astype(np.float64))
        np.testing.assert_array_equal(arrays[key + "__mean"], expected.mean_)
        np.testing.assert_array_equal(arrays[key + "__scale"], expected.scale_)
        assert len(settings["heads"][key]["training_ids"]) == 32
    chosen_site = settings["heads"][settings["pipelines"]["S1"]["head_key"]]["site"]
    index = settings["capture_provenance"]["sites"].index(chosen_site)
    np.testing.assert_array_equal(
        arrays["C1__mean"], features[64:96, index].astype(np.float64).mean(axis=0)
    )
    for factor in fit.FACTORS:
        np.testing.assert_array_equal(arrays[f"C2_{factor}__mean"], arrays["C1__mean"])
        assert settings["heads"][f"C2_{factor}"]["training_ids"] == settings["correction_fit_ids"]
    result = fit.score_bundle(settings, arrays, features)
    assert result["source_all12"].shape == (128, 12)
    assert result["factor_margins"].shape == (128, 4)
    np.testing.assert_array_equal(result["C2"], fit.fold_factors(result["factor_predictions"]))
    np.testing.assert_allclose(result["S1_probability"], 1 / (1 + np.exp(-result["S1_margin"])))


def test_strict_zero_threshold_and_factor_fold(trained):
    output, settings, features, _ = trained
    _, arrays = fit.validate_fit_bundle(output, allow_test_fixture=True)
    for key in settings["heads"]:
        arrays[key + "__coef"][:] = 0
        arrays[key + "__intercept"][:] = 0
    result = fit.score_bundle(settings, arrays, features[:3])
    assert all(np.all(result[name] == 0) for name in ("S0", "S1", "C1", "C2", "source_all12"))
    all_bits = np.asarray(list(itertools.product((0, 1), repeat=4)))
    expected = [int(1 - u if b2 else u if b1 else i) for i, u, b1, b2 in all_bits]
    np.testing.assert_array_equal(fit.fold_factors(all_bits), expected)


def test_group_log_loss_has_equal_context_weight_and_stable_extremes():
    labels = np.asarray([0, 0, 1])
    margins = np.asarray([0.0, 0.0, -4.0])
    expected = (np.log(2) + np.logaddexp(0, 4)) / 2
    assert fit.equal_group_log_loss(labels, margins, ["a", "a", "b"]) == pytest.approx(expected)
    assert fit.equal_group_log_loss([0, 1], [-1e300, 1e300], ["a", "b"]) == 0


def test_exact_ties_prefer_C_before_original_then_block_position():
    def candidate(site, C=1.0, loss=0.5):
        return {"site": site, "C": C, "dev_group_log_loss": loss}

    original = "layer_16_prompt_last"
    candidates = [
        candidate("layer_8_query_location_last"),
        candidate(original),
        candidate("layer_8_account_last"),
        candidate("layer_24_account_last", C=0.1),
    ]
    assert min(candidates, key=lambda c: fit.site_selection_key(c, original))["C"] == 0.1
    assert (
        min(candidates[:3], key=lambda c: fit.site_selection_key(c, original))["site"] == original
    )
    assert (
        min(
            [c for c in candidates if c["site"] != original and c["C"] == 1],
            key=lambda c: fit.site_selection_key(c, original),
        )["site"]
        == "layer_8_account_last"
    )
    # Tiny nonzero differences are not treated as ties.
    assert fit.site_selection_key(candidate(original, loss=0.5), original) > fit.site_selection_key(
        candidate("layer_8_account_last", loss=np.nextafter(0.5, 0)), original
    )


@pytest.mark.parametrize("partition", ["test", "unknown", None])
def test_forbidden_partition_rejected_before_any_numpy_load(tmp_path, monkeypatch, partition):
    directory, bank, protocol, _ = make_capture(tmp_path)
    payload = read_json(directory / "rows.json")
    payload["rows"].append({"partition": partition, "row_id": "never_seen_test_id"})
    atomic_json(directory / "rows.json", payload)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Held-out features were opened"))
    with pytest.raises(ValueError, match="forbidden"):
        fit.load_training_capture(directory, bank, protocol, allow_test_fixture=True)


@pytest.mark.parametrize(
    "defect", ["fixture", "wrong_ids", "row_metadata", "model", "sites", "hash", "policy"]
)
def test_bad_capture_binding_rejected_before_fitting(tmp_path, monkeypatch, defect):
    directory, bank, protocol, _ = make_capture(tmp_path)
    payload = read_json(directory / "rows.json")
    plan = read_json(protocol)
    if defect == "wrong_ids":
        payload["rows"][0]["row_id"] = "wrong"
    elif defect == "row_metadata":
        payload["rows"][0]["gold"] ^= 1
    elif defect == "fixture":
        payload["test_fixture"] = False
    elif defect == "model":
        plan["model_settings"]["revision"] = "mutable"
    elif defect == "sites":
        plan["sites"] = plan["sites"][::-1]
    elif defect == "policy":
        plan["source_sha256"].pop(str((fit.ROOT / fit.POLICY_FILES[0]).resolve()))
    else:
        with (directory / "features.npz").open("ab") as stream:
            stream.write(b"changed")
    atomic_json(directory / "rows.json", payload)
    if defect in ("model", "sites", "policy"):
        atomic_json(protocol, plan)
    manifest = read_json(directory / "manifest.json")
    manifest["rows_sha256"] = file_sha256(directory / "rows.json")
    atomic_json(directory / "manifest.json", manifest)
    monkeypatch.setattr(
        fit, "_fit_candidate", lambda *a: pytest.fail("Invalid capture reached fitting")
    )
    with pytest.raises(ValueError):
        fit.fit_bundle(directory, bank, protocol, tmp_path / "fitted", allow_test_fixture=True)


def test_float64_training_transform_matches_declared_sklearn_order():
    training = np.asarray([[0, 3, 0], [2, 3, 2], [4, 3, -2]], dtype=np.float32)
    dev = np.asarray([[300, 3, -7]], dtype=np.float32)
    scaler = StandardScaler().fit(training.astype(np.float64))
    params = fit.fit_transform(training)
    np.testing.assert_allclose(
        fit.transform(dev, params), normalize(scaler.transform(dev.astype(np.float64)))
    )
    assert params["scale"][1] == 1  # StandardScaler's constant-feature convention.


def test_convergence_warning_invalidates_candidate(monkeypatch):
    class Nonconverged:
        def __init__(self, **kwargs):
            assert kwargs["max_iter"] == 2000 and kwargs["tol"] == 1e-4

        def fit(self, x, labels):
            warnings.warn("synthetic failed solve", ConvergenceWarning, stacklevel=2)

    monkeypatch.setattr(fit, "LogisticRegression", Nonconverged)
    with pytest.raises(fit.CandidateFailure, match="did not converge"):
        fit._fit_candidate(np.eye(2), np.asarray([0, 1]), 1.0)


def test_nonfinite_candidate_margin_is_preserved_as_candidate_failure(monkeypatch):
    class Nonfinite:
        def decision_function(self, x):
            return np.full(len(x), np.nan)

    monkeypatch.setattr(fit, "_fit_candidate", lambda *a: Nonfinite())
    with pytest.raises(fit.CandidateFailure, match="Nonfinite development margins"):
        fit._candidate(np.eye(2), [0, 1], np.eye(2), [0, 1], ["g", "g"], 1)


def test_failed_candidates_preserved_without_rescue_or_grid_changes(tmp_path, monkeypatch):
    directory, bank, protocol, _ = make_capture(tmp_path)
    original = fit._fit_candidate

    def failed_small_C(x, labels, C):
        if C == 0.001:
            raise fit.CandidateFailure("synthetic nonconvergence")
        return original(x, labels, C)

    monkeypatch.setattr(fit, "_fit_candidate", failed_small_C)
    output = tmp_path / "fitted"
    settings = fit.fit_bundle(directory, bank, protocol, output, allow_test_fixture=True)
    failed = [r for r in settings["source_candidates"] if r["status"] == "failed"]
    assert len(failed) == 12 and {r["C"] for r in failed} == {0.001}
    assert all("synthetic nonconvergence" in r["failure"]["message"] for r in failed)
    assert all(r["C"] != 0.001 for r in settings["source_panel"])


def test_all_failed_site_stops_and_preserves_full_source_candidate_record(tmp_path, monkeypatch):
    directory, bank, protocol, _ = make_capture(tmp_path)
    monkeypatch.setattr(
        fit, "_fit_candidate", lambda *a: (_ for _ in ()).throw(fit.CandidateFailure("failure"))
    )
    output = tmp_path / "fitted"
    with pytest.raises(ValueError, match="All source candidates failed"):
        fit.fit_bundle(directory, bank, protocol, output, allow_test_fixture=True)
    assert read_json(output / "manifest.json")["status"] == "failed_preserved"
    assert len(read_json(output / "candidates.json")["source_candidates"]) == 60
    assert not (output / "heads.npz").exists()
    with pytest.raises(ValueError, match="not frozen"):
        fit.validate_fit_bundle(output, allow_test_fixture=True)


def test_correction_development_cannot_change_source_selection(tmp_path):
    first = tmp_path / "first"
    first.mkdir()
    d1, b1, p1, _ = make_capture(first)
    s1 = fit.fit_bundle(d1, b1, p1, first / "fit", allow_test_fixture=True)
    second = tmp_path / "second"
    second.mkdir()
    rows = fixture_rows()
    for row in rows:
        if row["partition"] == "correction_dev":
            row["gold"] ^= 1
    d2, b2, p2, _ = make_capture(second, rows=rows)
    s2 = fit.fit_bundle(d2, b2, p2, second / "fit", allow_test_fixture=True)
    assert s1["source_panel"] == s2["source_panel"]
    assert s1["pipelines"]["S1"] == s2["pipelines"]["S1"]
    _, a1 = fit.validate_fit_bundle(first / "fit", allow_test_fixture=True)
    _, a2 = fit.validate_fit_bundle(second / "fit", allow_test_fixture=True)
    for key in a1:
        if key.startswith("source_"):
            np.testing.assert_array_equal(a1[key], a2[key])


def test_fixture_cannot_validate_as_native_and_array_mutation_is_detected(trained):
    output, _, _, _ = trained
    with pytest.raises(ValueError, match="Fixture bundle"):
        fit.validate_fit_bundle(output)
    with (output / "heads.npz").open("ab") as stream:
        stream.write(b"tampering")
    with pytest.raises(ValueError, match="changed"):
        fit.validate_fit_bundle(output, allow_test_fixture=True)


def test_hidden_size_must_match_bound_protocol_before_array_loading(trained, monkeypatch):
    output, settings, _, _ = trained
    settings["hidden_size"] += 1
    atomic_json(output / "settings.json", settings)
    manifest = read_json(output / "manifest.json")
    manifest["output_sha256"]["settings.json"] = file_sha256(output / "settings.json")
    atomic_json(output / "manifest.json", manifest)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Wrong hidden size reached arrays"))
    with pytest.raises(ValueError, match="hidden size differs from bound protocol"):
        fit.validate_fit_bundle(output, allow_test_fixture=True)


def test_both_model_seal_binds_all_eight_bundle_files_and_exact_protocols(tmp_path):
    d1, bank, p1, _ = make_capture(tmp_path, model="llama")
    d2, _, p2, _ = make_capture(tmp_path, model="qwen", bank_path=bank)
    lfit, qfit = tmp_path / "llama-fit", tmp_path / "qwen-fit"
    fit.fit_bundle(d1, bank, p1, lfit, allow_test_fixture=True)
    fit.fit_bundle(d2, bank, p2, qfit, allow_test_fixture=True)
    seal = fit.freeze_fits(lfit, qfit, [p2, p1], tmp_path / "seal.json", allow_test_fixture=True)
    assert len(seal["fitted_artifact_sha256"]) == 8
    assert seal["model_protocol_sha256"] == {"llama": file_sha256(p1), "qwen": file_sha256(p2)}
    assert all(
        file_sha256(path) == digest for path, digest in seal["fitted_artifact_sha256"].items()
    )
    assert seal["test_fixture"] and seal["human_review_complete"] is False
    saved_seal = (tmp_path / "seal.json").read_bytes()
    with pytest.raises(FileExistsError):
        fit.freeze_fits(lfit, qfit, [p2, p1], tmp_path / "seal.json", allow_test_fixture=True)
    assert (tmp_path / "seal.json").read_bytes() == saved_seal
    with pytest.raises(ValueError, match="wrong model/protocol"):
        fit.freeze_fits(qfit, lfit, [p1, p2], tmp_path / "wrong.json", allow_test_fixture=True)


def test_policy_explicitly_discloses_privileged_fold_and_no_calibration():
    policy = fit.fit_policy()
    assert policy["C2"] == "fourfactor_lr"
    assert policy["factor_C_grid"] == [0.01, 0.1, 1, 10, 100]
    assert policy["calibration"] is None and not policy["equal_capacity_claim"]
