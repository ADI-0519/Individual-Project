"""CPU-only synthetic engineering checks; these are not model results."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from latent_probes.main_study_analysis import (
    _head_margins,
    analyse_study,
    belief_group_metrics,
    binary_metrics,
    cluster_primary_summary,
    decisions,
    load_fit_bundle,
)
from latent_probes.main_study_design import (
    ALLOCATION_SEEDS,
    correction_states,
    default_training_groups,
    matched_training_allocation,
    one_step_states,
)
from latent_probes.main_study_fit import (
    FeatureCache,
    _array_hash,
    _fit_candidate,
    _fit_transform,
    _transform,
    fit_comparison,
    fit_monotone_calibration,
    fit_study,
    group_weights,
    grouped_log_loss,
    load_feature_cache,
    save_fit_bundle,
)
from latent_probes.storage import atomic_json, atomic_npz, file_sha256


def _rows(cases, split):
    return [
        {
            "row_id": case.case_id,
            "case_id": case.case_id,
            "group_id": case.group_id,
            "mechanism": case.mechanism,
            "split": split,
            "state": asdict(case),
            "gold": case.gold,
            "messages": [
                {"role": "user", "content": f"Synthetic value {case.gold}; {case.case_id}"}
            ],
            "output_yes_no_logitdiff": float(2 * case.gold - 1),
        }
        for case in cases
    ]


def _cache(rows, **overrides):
    # Deliberate target-bearing fixture features test plumbing, not scientific inference.
    x = np.array(
        [[2 * r["gold"] - 1, r["state"]["world"], r["state"]["name_map"]] for r in rows],
        dtype=np.float32,
    )
    args = {
        "rows": tuple(rows),
        "features": x,
        "provenance": {
            "bank_sha256": "synthetic-bank",
            "model_settings": {"model": "fixture"},
            "capture_layer": 1,
        },
        "test_fixture": True,
        "source_hashes": {"synthetic": "fixture-only"},
    }
    args.update(overrides)
    return FeatureCache(**args)


@pytest.fixture(scope="module")
def training_cache():
    rows = []
    for group in default_training_groups():
        rows.extend(_rows(one_step_states(group.mechanism, group.group_id), "train"))
    for mechanism in ("observation", "message"):
        rows.extend(_rows(one_step_states(mechanism, f"development_{mechanism}"), "development"))
    return _cache(rows)


@pytest.fixture(scope="module")
def fitted(training_cache):
    return fit_study(
        training_cache, default_training_groups(), kinds=("activation",), sizes=(128,), seeds=(11,)
    )


@pytest.fixture
def test_cache():
    rows = _rows(correction_states("held_correction"), "test")
    rows += _rows(one_step_states("observation", "held_observation"), "test")
    rows += _rows(one_step_states("message", "held_message"), "test")
    return _cache(rows)


@pytest.mark.parametrize("mutation", ["gold", "row_id", "group", "duplicate", "split"])
def test_capture_refuses_bad_semantic_bindings(mutation):
    rows = _rows(one_step_states("message", "group"), "train")
    if mutation == "gold":
        rows[0]["gold"] = 1 - rows[0]["gold"]
    elif mutation == "row_id":
        rows[0]["row_id"] = "wrong"
    elif mutation == "group":
        rows[0]["group_id"] = "wrong"
    elif mutation == "duplicate":
        rows[1] = dict(rows[0])
    else:
        rows[0]["split"] = "development"
    with pytest.raises(ValueError):
        _cache(rows)


def test_feature_hash_detects_tampering():
    rows = _rows(one_step_states("message", "group"), "train")
    rows[0]["feature_sha256"] = "incorrect"
    with pytest.raises(ValueError, match="Feature hash"):
        _cache(rows)


def test_fitting_api_refuses_any_test_rows_before_computation(
    training_cache, test_cache, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("No fitting may occur after a test cache is supplied")

    monkeypatch.setattr("latent_probes.main_study_fit._fit_candidate", forbidden)
    cache = _cache(list(training_cache.rows) + list(test_cache.rows))
    with pytest.raises(ValueError, match="Test rows"):
        fit_comparison(cache, default_training_groups(), training_n=128, seed=11)


def test_selected_unique_ids_scalers_and_C_match_policy(training_cache, fitted):
    settings, arrays = fitted
    comparison = settings["comparisons"][0]
    allocation = matched_training_allocation(default_training_groups(), training_n=128, seed=11)
    lookup = {r["case_id"]: i for i, r in enumerate(training_cache.rows)}
    for arm in ("restricted", "full"):
        head = comparison["arms"][arm]
        selected = [case.case_id for case in getattr(allocation, arm)]
        assert head["training_ids"] == selected
        assert len(set(selected)) == 128
        assert not set(selected) & set(comparison["development_ids"])
        observed_mean = arrays[f"{head['array_prefix']}__mean"]
        assert np.array_equal(
            observed_mean, training_cache.features[[lookup[x] for x in selected]].mean(axis=0)
        )
    successful = [c for c in comparison["candidates"] if not c["excluded_arms"]]
    expected = min(successful, key=lambda candidate: (candidate["mean_loss"], candidate["C"]))
    assert comparison["selected_C"] == expected["C"]
    assert settings["test_fixture"] and not settings["scientific_result"]


def test_development_loss_assigns_equal_scenario_weight():
    rows = ({"group_id": "a"}, {"group_id": "b"}, {"group_id": "b"}, {"group_id": "b"})
    assert group_weights(rows).tolist() == pytest.approx([0.5, 1 / 6, 1 / 6, 1 / 6])
    y = np.array([0, 1, 1, 1])
    margins = np.array([2.0, 2.0, 2.0, 2.0])
    assert grouped_log_loss(y, margins, rows) == pytest.approx(
        (np.logaddexp(0, 2) + np.logaddexp(0, 2) - 2) / 2
    )


def test_shared_C_tie_break_is_lower_C(training_cache, monkeypatch):
    monkeypatch.setattr("latent_probes.main_study_fit.grouped_log_loss", lambda *args: 0.75)
    comparison, _ = fit_comparison(
        training_cache, default_training_groups(), training_n=128, seed=11
    )
    assert comparison["selected_C"] == 0.001


def test_one_arm_nonconvergence_excludes_candidate_for_both(training_cache, monkeypatch):
    from latent_probes import main_study_fit as module

    original = module._fit_candidate
    calls = []

    def candidate(x, labels, c_value):
        calls.append(c_value)
        return None if len(calls) == 2 else original(x, labels, c_value)

    monkeypatch.setattr(module, "_fit_candidate", candidate)
    comparison, _ = fit_comparison(
        training_cache, default_training_groups(), training_n=128, seed=11
    )
    assert comparison["candidates"][0]["excluded_arms"] == ["full"]
    assert comparison["candidates"][0]["mean_loss"] is None
    assert comparison["selected_C"] != 0.001


def test_all_candidates_failing_is_terminal(training_cache, monkeypatch):
    monkeypatch.setattr("latent_probes.main_study_fit._fit_candidate", lambda *args: None)
    comparison, arrays = fit_comparison(
        training_cache, default_training_groups(), training_n=128, seed=11
    )
    assert comparison["status"] == "no_converged_shared_candidate"
    assert not arrays and "selected_C" not in comparison


def test_monotone_calibration_never_reverses_scores():
    rows = tuple({"case_id": str(i), "group_id": str(i // 2)} for i in range(8))
    calibration = fit_monotone_calibration(np.arange(8), np.array([1, 1, 1, 1, 0, 0, 0, 0]), rows)
    assert calibration["status"] == "complete"
    assert calibration["slope"] >= 0
    assert calibration["source_ids"] == [str(i) for i in range(8)]


def test_decision_ties_and_invalid_probabilities():
    assert decisions(np.array([0.1, 0.5, 0.9])).tolist() == [0, 0, 1]
    with pytest.raises(ValueError):
        decisions(np.array([np.nan]))
    with pytest.raises(ValueError):
        binary_metrics(np.array([0, 1]), np.array([0.5]))


@pytest.mark.parametrize("mechanism", ["observation", "message", "correction"])
def test_perfect_gold_and_75percent_shortcut_quartet_separation(mechanism):
    cases = correction_states("g") if mechanism == "correction" else one_step_states(mechanism, "g")
    rows = tuple(_rows(cases, "test"))
    exact = belief_group_metrics(rows, np.array([2 * c.gold - 1 for c in cases], dtype=float))
    shortcut = belief_group_metrics(
        rows, np.array([2 * c.final_update - 1 for c in cases], dtype=float)
    )
    assert exact["complete_quartet_accuracy"] == 1
    assert shortcut["complete_quartet_accuracy"] == 0
    if mechanism != "correction":
        assert shortcut["all_rows"]["accuracy"] == 0.75
    else:
        assert exact["secondary_correction_cells"]["rows"] == 32


def test_incomplete_counterfactuals_fail():
    rows = tuple(_rows(correction_states("g"), "test"))
    with pytest.raises(ValueError):
        belief_group_metrics(rows[:-1], np.zeros(len(rows) - 1))


def test_primary_aggregation_preserves_groups_and_seeds():
    data = {
        seed: {"a": {"restricted": 0.0, "full": 1.0}, "b": {"restricted": 1.0, "full": 0.0}}
        for seed in ALLOCATION_SEEDS
    }
    result = cluster_primary_summary(data)
    assert result["independent_scenario_groups"] == 2
    assert result["full_minus_restricted"] == 0
    assert result["percentile_95_interval"] == [-1, 1]
    assert len(result["by_seed"]) == 5
    assert result == cluster_primary_summary(data)
    del data[11]["a"]
    with pytest.raises(ValueError, match="same scenario"):
        cluster_primary_summary(data)


def test_fixture_analysis_cannot_become_scientific(fitted, test_cache):
    settings, arrays = fitted
    with pytest.raises(ValueError, match="Fixture outputs"):
        analyse_study(test_cache, settings, arrays)
    report = analyse_study(test_cache, settings, arrays, allow_fixture=True)
    assert report["test_fixture"] and not report["scientific_result"]
    assert report["primary"] is None
    assert len(report["baselines"]["symbolic_oracle"]) == 3


def test_provenance_mismatch_and_source_overlap_are_rejected(fitted, test_cache):
    settings, arrays = fitted
    wrong = replace(test_cache, provenance=dict(test_cache.provenance, capture_layer=9))
    with pytest.raises(ValueError, match="provenance mismatch"):
        analyse_study(wrong, settings, arrays, allow_fixture=True)
    altered = json.loads(json.dumps(settings))
    altered["comparisons"][0]["arms"]["restricted"]["training_ids"].append(
        test_cache.rows[0]["case_id"]
    )
    with pytest.raises(ValueError, match="overlaps"):
        analyse_study(test_cache, altered, arrays, allow_fixture=True)


def test_pickle_free_bundle_roundtrip_and_tamper_detection(tmp_path, fitted, test_cache):
    settings, arrays = fitted
    out = tmp_path / "fit"
    save_fit_bundle(out, settings, arrays)
    restored, restored_arrays = load_fit_bundle(out)
    assert (
        analyse_study(test_cache, restored, restored_arrays, allow_fixture=True)["comparisons"]
        == analyse_study(test_cache, settings, arrays, allow_fixture=True)["comparisons"]
    )
    with pytest.raises(FileExistsError):
        save_fit_bundle(out, settings, arrays)
    with (out / "heads.npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_fit_bundle(out)


def test_tfidf_vocab_is_train_only_and_roundtrips(training_cache, test_cache, tmp_path):
    rows = [dict(row) for row in training_cache.rows]
    for row in rows:
        if row["split"] == "development":
            row["messages"] = [{"role": "user", "content": "DevelopmentOnlyMarker"}]
    settings, arrays = fit_study(
        _cache(rows), default_training_groups(), kinds=("tfidf",), sizes=(128,), seeds=(11,)
    )
    for head in settings["comparisons"][0]["arms"].values():
        assert "developmentonlymarker" not in head["vocabulary"]
    save_fit_bundle(tmp_path / "text_fit", settings, arrays)
    restored, restored_arrays = load_fit_bundle(tmp_path / "text_fit")
    report = analyse_study(test_cache, restored, restored_arrays, allow_fixture=True)
    assert len(report["comparisons"]) == 1


def test_feature_cache_exact_order_and_binding(tmp_path):
    cases = one_step_states("message", "g")
    cache = _cache(_rows(cases, "train"))
    rows = [
        dict(r, feature_sha256=_array_hash(f))
        for r, f in zip(cache.rows, cache.features, strict=True)
    ]
    atomic_json(
        tmp_path / "rows.json",
        {
            "schema": "main-study-capture-v1",
            "rows": rows,
            "provenance": cache.provenance,
            "test_fixture": True,
        },
    )
    atomic_npz(
        tmp_path / "features.npz",
        features=cache.features,
        row_ids=np.array([r["case_id"] for r in rows]),
    )
    hashes = {name: file_sha256(tmp_path / name) for name in ("rows.json", "features.npz")}
    atomic_json(tmp_path / "capture_manifest.json", {"status": "complete", "output_sha256": hashes})
    assert np.array_equal(load_feature_cache(tmp_path).features, cache.features)
    atomic_npz(
        tmp_path / "features.npz",
        features=cache.features,
        row_ids=np.array([r["case_id"] for r in reversed(rows)]),
    )
    hashes["features.npz"] = file_sha256(tmp_path / "features.npz")
    atomic_json(tmp_path / "capture_manifest.json", {"status": "complete", "output_sha256": hashes})
    with pytest.raises(ValueError, match="order"):
        load_feature_cache(tmp_path)


def test_float32_capture_uses_identical_float64_fit_and_saved_inference(tmp_path):
    rows = _rows(one_step_states("message", "fixture"), "train")
    rng = np.random.default_rng(17)
    x = rng.normal(size=(32, 7)).astype(np.float32)
    cache = _cache(rows, features=x)
    indices = list(range(24))
    transformed, transform = _fit_transform(cache, indices, "activation")
    assert transformed.dtype == np.float64
    assert np.array_equal(transformed, _transform(cache, indices, transform))
    model = _fit_candidate(transformed, np.array([rows[i]["gold"] for i in indices]), 1.0)
    key = "activation_n128_seed11_restricted"
    arrays = {
        f"{key}__mean": transform["mean"],
        f"{key}__scale": transform["scale"],
        f"{key}__coef": model.coef_[0],
        f"{key}__intercept": model.intercept_,
    }
    comparison = {"feature_kind": "activation", "arms": {"restricted": {"array_prefix": key}}}
    settings = {"schema": "main-study-fit-v1", "test_fixture": True, "comparisons": [comparison]}
    save_fit_bundle(tmp_path / "float64_head", settings, arrays)
    _, restored = load_fit_bundle(tmp_path / "float64_head")
    expected = model.decision_function(_transform(cache, list(range(32)), transform))
    actual = _head_margins(cache, comparison, "restricted", restored)
    assert np.allclose(expected, actual, atol=1e-13, rtol=1e-13)


def test_failed_primary_allocations_are_explicit_not_silently_dropped(fitted, test_cache):
    settings, arrays = fitted
    settings = json.loads(json.dumps(settings))
    settings["comparisons"] = [
        {
            "training_n": 512,
            "allocation_seed": seed,
            "feature_kind": "activation",
            "status": "no_converged_shared_candidate",
        }
        for seed in ALLOCATION_SEEDS
    ]
    result = analyse_study(test_cache, settings, arrays, allow_fixture=True)
    assert result["primary"]["status"] == "unavailable_missing_or_failed_primary_comparisons"
    assert result["primary"]["missing_allocation_seeds"] == list(ALLOCATION_SEEDS)
    assert len(result["comparisons"]) == 5


def test_whole_scenario_overlap_is_rejected_even_with_different_case_ids(fitted, test_cache):
    settings, arrays = fitted
    settings = json.loads(json.dumps(settings))
    settings["fitting_group_splits"]["held_correction"] = "development"
    with pytest.raises(ValueError, match="scenario group overlaps"):
        analyse_study(test_cache, settings, arrays, allow_fixture=True)


def test_nonfixture_analysis_requires_preheldout_fit_binding(fitted, test_cache):
    settings, arrays = fitted
    settings = dict(settings, test_fixture=False)
    real = replace(test_cache, test_fixture=False)
    with pytest.raises(ValueError, match="bound before held-out"):
        analyse_study(real, settings, arrays)
