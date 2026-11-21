"""Bound CPU analysis tests: fixtures exercise auditing, never empirical claims."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from latent_probes.recipient_risk_panel import actor_jobs, load_panel, reader_plan
from latent_probes.storage import atomic_json, file_sha256, read_json

ROOT = Path(__file__).resolve().parents[1]


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared = module_at("risk_analysis_runner_fixtures", ROOT / "tests/test_recipient_risk_runner.py")


@pytest.fixture
def runner(monkeypatch, tmp_path):
    return shared.runner.__wrapped__(monkeypatch, tmp_path)


@pytest.fixture
def analysis(monkeypatch, runner):
    module = module_at(
        "risk_analysis_under_test", ROOT / "scripts/analyse_recipient_risk_validation.py"
    )
    monkeypatch.setattr(module, "_runner_module", lambda: runner)
    return module


def attempt(runner, tmp_path, *, full=False, pass_controls=False):
    output = shared.prepare(runner, tmp_path)
    factory = shared.RiskFactory(output, controls_pass=pass_controls, empty_first=full)
    runner.execute(output, stage="controls", backend_factory=factory)
    if full:
        runner.execute(output, stage="reports", backend_factory=factory)
        runner.execute(output, stage="readers", backend_factory=factory)
    return output, factory


def reseal(output, name, value):
    atomic_json(output / f"{name}.json", value)
    manifest = read_json(output / "manifest.json")
    manifest[name + "_sha256"] = file_sha256(output / f"{name}.json")
    atomic_json(output / "manifest.json", manifest)


def test_failed_qualification_is_a_bound_zero_report_hard_stop(analysis, runner, tmp_path):
    output, factory = attempt(runner, tmp_path)
    result = analysis.analyse(output, allow_fixture=True)
    assert result["attempt_status"] == "qualification_failed"
    assert result["counts"]["reports"] == 0
    assert result["counts"]["recipient_conditions"] == 72
    assert result["counts"]["unique_reader_calls"] == factory.calls["reader"]
    assert result["counts"]["generation_attempts"]["actor"] == 0
    assert result["feasibility"]["overall_status"] == "qualification_failed_hard_stop"
    assert result["feasibility"]["dependent_report_artifacts_absent"]
    assert not result["feasibility"]["dependent_report_collection_permitted"]
    assert result["human_review"] is result["new_probe_fits_permitted"] is False
    assert len(result["bindings"]["reader_prompt_token_sha256"]) == 72
    assert result["controls"]["n_case_clusters"] == 12
    assert result["controls"]["errors"]
    assert result["verification_scope"]["independent_tokenization_or_decoding_repeated"] is False


def test_passing_controls_allow_only_report_collection(analysis, runner, tmp_path):
    output, _ = attempt(runner, tmp_path, pass_controls=True)
    result = analysis.analyse(output, allow_fixture=True)
    assert result["attempt_status"] == "controls_passed"
    assert result["feasibility"]["dependent_report_collection_permitted"]
    assert result["new_probe_fits_permitted"] is False
    assert result["controls"]["errors"] == []
    assert result["counts"]["reports"] == 0


def test_cpu_fixture_requires_explicit_acknowledgement(analysis, runner, tmp_path):
    output, _ = attempt(runner, tmp_path)
    with pytest.raises(ValueError, match="explicit allow_fixture"):
        analysis.analyse(output)


def test_full_analysis_preserves_every_group_and_keeps_semantics_pending(
    analysis, runner, tmp_path
):
    output, factory = attempt(runner, tmp_path, pass_controls=True, full=True)
    result = analysis.analyse(output, allow_fixture=True)
    assert result["attempt_status"] == "complete"
    assert result["counts"]["reports"] == factory.calls["actor"] == 48
    assert result["counts"]["recipient_conditions"] == 504
    assert result["counts"]["unique_reader_calls"] == factory.calls["reader"]
    assert len(result["bindings"]["feature_sha256"]) == 47
    behavior = result["behavior"]
    assert len(behavior["groups"]) == 12
    assert len(behavior["groups_by_packet_and_rule"]) == 108
    assert len(behavior["no_report_to_original"]) == 144
    assert len(behavior["original_to_edits"]) == 288
    assert len(behavior["paired_conditions"]) == 144
    assert sum(g["n_conditions"] for g in behavior["groups"].values()) == 432
    assert all(
        behavior["groups"][f"{b}/{g}/original"]["n_conditions"] == 36
        for b in (50, 25)
        for g in ("neutral", "favorable")
    )
    # The empty report exactly reuses no-report control answers, including
    # definite informed/stale outcomes. It must not be assumed to yield DEFER.
    empty_id = result["report_audits"][0]["example_id"]
    assert all(
        r["from"] == r["to"]
        for r in behavior["no_report_to_original"]
        if r["source_example_id"] == empty_id
    )
    assert (
        result["feasibility"]["gate3_within_goal_world_risk_classes"]["status"] == "observed_fail"
    )
    assert result["feasibility"]["gate4_packet_relative_world_risk"]["status"] == "observed_fail"
    for gate in (1, 2, 5, 6):
        assert (
            next(v for k, v in result["feasibility"].items() if k.startswith(f"gate{gate}_"))[
                "status"
            ]
            == "pending_semantic_review"
        )
    assert result["new_probe_fits_permitted"] is False
    assert result["report_audits"][0]["technical_exclusions"] == ["empty_report"]
    assert all(
        a["source_support"] == "pending_blind_semantic_review" for a in result["report_audits"]
    )


@pytest.mark.parametrize("name", ("control_rows", "reader_cache", "qualification"))
def test_journal_mutation_is_rejected(analysis, runner, tmp_path, name):
    output, _ = attempt(runner, tmp_path)
    (output / f"{name}.json").write_text("{}")
    with pytest.raises(ValueError, match="changed|removed"):
        analysis.analyse(output, allow_fixture=True)


def test_even_resealed_wrong_decision_cannot_pass_parser_binding(analysis, runner, tmp_path):
    output, _ = attempt(runner, tmp_path)
    rows = read_json(output / "control_rows.json")
    rows[0]["decision"] = "APPROVE"
    reseal(output, "control_rows", rows)
    with pytest.raises(ValueError, match="Saved decision"):
        analysis.analyse(output, allow_fixture=True)


def test_even_resealed_changed_labels_cannot_replace_the_control_plan(analysis, runner, tmp_path):
    output, _ = attempt(runner, tmp_path)
    rows = read_json(output / "control_rows.json")
    rows[0]["normative_control_decision"] = "DEFER"
    reseal(output, "control_rows", rows)
    with pytest.raises(ValueError, match="exact planned condition"):
        analysis.analyse(output, allow_fixture=True)


def test_attempt_binding_detects_source_drift(analysis, runner, tmp_path):
    output, _ = attempt(runner, tmp_path)
    runner.fixture_source.write_text("Changed after execution.")
    with pytest.raises(ValueError, match="source changed"):
        analysis.analyse(output, allow_fixture=True)


@pytest.mark.parametrize("damage", ("checkpoint", "runtime"))
def test_saved_backend_checkpoint_and_runtime_are_bound(analysis, runner, tmp_path, damage):
    output, _ = attempt(runner, tmp_path)
    manifest = read_json(output / "manifest.json")
    if damage == "checkpoint":
        manifest["reader_backend"]["resolved_revision"] = "different-checkpoint"
    else:
        manifest["reader_backend"]["pilot_runtime_identity"]["python"] = "different-runtime"
    atomic_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="backend identity|runtime signature"):
        analysis.analyse(output, allow_fixture=True)


def test_failed_qualification_cannot_hide_dependent_artifacts(analysis, runner, tmp_path):
    output, _ = attempt(runner, tmp_path)
    atomic_json(output / "blind_reports.json", [])
    with pytest.raises(ValueError, match="Dependent report artifacts"):
        analysis.analyse(output, allow_fixture=True)


def test_changed_feature_bytes_are_rejected(analysis, runner, tmp_path):
    output, _ = attempt(runner, tmp_path, pass_controls=True, full=True)
    source = next(iter(read_json(output / "sender_rows.json")[1:]))
    np.savez(output / source["feature_file"], layer_8_tokens=np.zeros((2, 4)))
    with pytest.raises(ValueError, match="features changed"):
        analysis.analyse(output, allow_fixture=True)


def test_changed_blind_export_cannot_be_presented_as_exact_reports(analysis, runner, tmp_path):
    output, _ = attempt(runner, tmp_path, pass_controls=True, full=True)
    blind = read_json(output / "blind_reports.json")
    blind[0]["report"] = "An edited report."
    atomic_json(output / "blind_reports.json", blind)
    with pytest.raises(ValueError, match="Blind review export"):
        analysis.analyse(output, allow_fixture=True)


@pytest.fixture
def behavioral_rows():
    panel = load_panel(ROOT / "data/communication/recipient_risk_validation_v2_draft.json")
    rows = reader_plan(
        panel, {j["example_id"]: "Synthetic unit-test report." for j in actor_jobs(panel)}
    )
    for row in rows:
        row["decision"] = row["normative_control_decision"] or row["complete_source_gold"]
    return rows


def test_invalid_is_not_scored_as_safe_or_as_defer(analysis):
    row = {
        "decision": "INVALID",
        "complete_source_gold": "REJECT",
        "case_id": "test",
        "rule_family": "test",
    }
    assert analysis.outcome(row) == "invalid"
    assert analysis.wrong(row) is None
    assert analysis.counts([row])["outcome_counts"] == {
        "world_correct": 0,
        "world_wrong_definite": 0,
        "defer": 0,
        "invalid": 1,
    }
    row["decision"] = "DEFER"
    assert analysis.wrong(row) == 0
    assert analysis.outcome(row) == "defer"


def test_induced_and_maintained_errors_and_placebo_specificity_are_distinct(
    analysis, behavioral_rows
):
    rows = behavioral_rows
    report_id = next(
        r["source_example_id"]
        for r in rows
        if r["version"] == "original"
        and r["word_budget"] == 50
        and r["complete_source_gold"] == "REJECT"
    )
    for row in rows:
        if (
            row["source_example_id"] == report_id
            and row["context"] in ("absent", "stale")
            and row["version"] in ("original", "irrelevant")
        ):
            row["decision"] = "APPROVE"
    result = analysis.behavioral_analysis(rows, [])
    pairs = {
        p["context"]: p for p in result["paired_conditions"] if p["source_example_id"] == report_id
    }
    assert pairs["absent"]["induced_specific_correction_candidate"]
    assert not pairs["absent"]["maintained_specific_correction_candidate"]
    assert pairs["stale"]["maintained_specific_correction_candidate"]
    baseline = {
        p["context"]: p
        for p in result["no_report_to_original"]
        if p["source_example_id"] == report_id
    }
    assert baseline["absent"]["report_induced_world_error"]
    assert baseline["stale"]["maintained_preexisting_world_error"]
    assert pairs["absent"]["wrong_irrelevant_minus_restored"] == 1
    assert any(
        p["world_outcomes"]["original"] == "world_correct" for p in result["paired_conditions"]
    )


def test_informed_control_error_does_not_qualify_as_stale_maintained_correction(
    analysis, behavioral_rows
):
    rows = behavioral_rows
    chosen = next(r for r in rows if r["version"] == "original" and r["context"] == "informed")
    wrong_answer = "APPROVE" if chosen["complete_source_gold"] == "REJECT" else "REJECT"
    for row in rows:
        if row["case_id"] == chosen["case_id"] and row["context"] == "informed":
            if row["version"] == "no_report" or (
                row["source_example_id"] == chosen["source_example_id"]
                and row["version"] in ("original", "irrelevant")
            ):
                row["decision"] = wrong_answer
    result = analysis.behavioral_analysis(rows, [])
    pair = next(
        p
        for p in result["paired_conditions"]
        if p["source_example_id"] == chosen["source_example_id"] and p["context"] == "informed"
    )
    baseline = next(
        p
        for p in result["no_report_to_original"]
        if p["source_example_id"] == chosen["source_example_id"] and p["context"] == "informed"
    )
    assert baseline["maintained_preexisting_world_error"]
    assert not pair["maintained_specific_correction_candidate"]


def test_control_invalidity_and_full_disclosure_failure_exclude_paired_contrast(
    analysis, behavioral_rows
):
    rows = behavioral_rows
    chosen = next(r for r in rows if r["version"] == "original")
    for row in rows:
        if row["case_id"] == chosen["case_id"] and row["context"] == chosen["context"]:
            if row["version"] == "no_report":
                row["decision"] = "INVALID"
            elif row["version"] == "full_disclosure":
                row["decision"] = "DEFER"
    result = analysis.behavioral_analysis(rows, [])
    selected = [
        p
        for p in result["paired_conditions"]
        if p["case_id"] == chosen["case_id"] and p["context"] == chosen["context"]
    ]
    assert selected and all(not p["gate6_behavioral_stratum"] for p in selected)
    assert all(p["wrong_irrelevant_minus_restored"] is None for p in selected)
    assert all("no_report" in p["invalid_members"] for p in selected)


def test_equal_case_contrast_weights_cases_and_includes_negative_effects(analysis):
    pairs = [
        {"case_id": "a", "gate6_behavioral_stratum": True, "wrong_irrelevant_minus_restored": 1},
        {"case_id": "b", "gate6_behavioral_stratum": True, "wrong_irrelevant_minus_restored": -1},
        {"case_id": "b", "gate6_behavioral_stratum": True, "wrong_irrelevant_minus_restored": -1},
        {"case_id": "b", "gate6_behavioral_stratum": True, "wrong_irrelevant_minus_restored": 0},
        {"case_id": "c", "gate6_behavioral_stratum": False, "wrong_irrelevant_minus_restored": 1},
    ]
    result = analysis.equal_case_contrast(pairs)
    assert result["n_tuples"] == 4 and result["n_case_clusters"] == 2
    assert result["equal_case_mean"] == pytest.approx((1 - 2 / 3) / 2)
    assert result["equal_case_mean"] != pytest.approx(
        sum(p["wrong_irrelevant_minus_restored"] for p in pairs[:4]) / 4
    )


def test_secondary_budget_cannot_rescue_primary_packet_gate(analysis, behavioral_rows):
    rows = behavioral_rows
    for row in rows:
        if row["version"] == "original" and row["word_budget"] == 25 and row["context"] == "absent":
            row["decision"] = "APPROVE" if row["complete_source_gold"] == "REJECT" else "REJECT"
    behavior = analysis.behavioral_analysis(rows, [])
    gates = analysis.feasibility_observations(behavior, rows)
    assert gates["gate4_packet_relative_world_risk"]["status"] == "observed_fail"
    assert gates["new_probe_fits_permitted"] is False
    assert any(r["word_budget"] == 25 for r in behavior["packet_sensitive_reports"])


def test_qualification_record_cannot_be_replaced_by_an_approved_fit(analysis, runner, tmp_path):
    output, _ = attempt(runner, tmp_path)
    qualification = read_json(output / "qualification.json")
    qualification["new_probe_fits_permitted"] = True
    reseal(output, "qualification", qualification)
    with pytest.raises(ValueError, match="Qualification differs"):
        analysis.analyse(output, allow_fixture=True)
