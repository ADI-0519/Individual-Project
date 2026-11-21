"""Metadata-only engineering fixtures; none is an actual human attestation or fit."""

from __future__ import annotations

import copy
import json

import pytest
from test_main_study_bank import synthetic_bank as synthetic_bank

from latent_probes import main_study_execution as execution
from latent_probes.main_study_bank import bank_digest, representative_cases
from latent_probes.storage import file_sha256, read_json


@pytest.fixture
def engineering_freeze(tmp_path, monkeypatch):
    bank = {
        "pytest_fixture_only": True,
        "groups": [
            {"group_id": "fixture_train", "split": "train"},
            {"group_id": "fixture_development", "split": "development"},
            {"group_id": "fixture_test", "split": "test"},
        ],
    }
    bank_path, ledger_path = tmp_path / "fixture_bank.json", tmp_path / "fixture_ledger.json"
    bank_path.write_text(json.dumps(bank), encoding="utf-8")
    ledger_path.write_text(
        '{"pytest_fixture_only": true, "no_human_records": true}', encoding="utf-8"
    )
    calls = []
    monkeypatch.setattr(execution, "load_bank", lambda p: read_json(p))

    def fixture_review_only(received, path):
        assert received["pytest_fixture_only"] is True
        assert read_json(path)["no_human_records"] is True
        calls.append(str(path))
        # Scoped pytest substitution; no real ledger gains accepted records.
        return {"review_complete": True, "pytest_review_stub_only": True}

    monkeypatch.setattr(execution, "verify_review_ledger", fixture_review_only)
    return bank, bank_path, ledger_path, tmp_path / "fixture_freeze.json", calls


def build(fixture, **kwargs):
    _, bank_path, ledger_path, output, _ = fixture
    return execution.build_execution_freeze(
        bank_path, ledger_path, execution.DEFAULT_BASELINE_PROTOCOL, output, **kwargs
    )


def test_declared_baseline_policy_matches_current_numerical_constants():
    policy = read_json(execution.DEFAULT_BASELINE_PROTOCOL)
    assert policy == execution.expected_baseline_policy()
    assert policy["tfidf"]["dtype"] == "float64"
    assert policy["raw_threshold"] == 0.5 and policy["tie_class"] == 0
    assert policy["calibration"]["slope_min"] == 0
    assert policy["text_monitor"]["protocol_sha256"] == file_sha256(execution.DEFAULT_PROTOCOL)


def test_initial_freeze_binds_source_union_and_never_overwrites(engineering_freeze):
    result = build(engineering_freeze)
    assert result["status"] == "frozen" and result["stage"] == "train_dev"
    assert result["model_calls"] == 0 and result["fitted_artifact_sha256"] == {}
    assert result["review"]["pytest_review_stub_only"] is True
    assert len(engineering_freeze[4]) == 1
    expected = {
        *execution.CAPTURE_POLICY_FILES,
        *execution.MONITOR_POLICY_FILES,
        *execution.EXTRA_POLICY_FILES,
    }
    for name in expected:
        path = execution.ROOT / name
        assert result["source_sha256"][str(path.resolve())] == file_sha256(path)
    assert result["text_monitor_protocol_path"] == str(execution.DEFAULT_PROTOCOL.resolve())
    original = engineering_freeze[3].read_bytes()
    with pytest.raises(FileExistsError):
        build(engineering_freeze)
    assert engineering_freeze[3].read_bytes() == original


@pytest.mark.parametrize("questionnaire_mismatch", [False, True])
def test_real_review_gate_rejects_blank_or_disputed_fixture_ledger(
    synthetic_bank, tmp_path, questionnaire_mismatch
):
    bank_path, ledger_path, output = [
        tmp_path / f"fixture_{n}.json" for n in ("bank", "ledger", "freeze")
    ]
    bank_path.write_text(json.dumps(synthetic_bank), encoding="utf-8")
    records = []
    if questionnaire_mismatch:
        group = synthetic_bank["groups"][0]
        cases = representative_cases(group)
        answers = {c.case_id: "YES" if c.gold else "NO" for c in cases}
        answers[cases[0].case_id] = "NO" if cases[0].gold else "YES"
        records.append(
            {
                "group_id": group["group_id"],
                "role": "student",
                "reviewer": "EXPLICIT PYTEST FIXTURE, NOT A HUMAN REVIEW",
                "decision": "accepted",
                "independent_answers": answers,
            }
        )
    ledger_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bank_sha256": bank_digest(synthetic_bank),
                "records": records,
                "disputed_groups": [],
            }
        ),
        encoding="utf-8",
    )
    message = (
        "independent-answer mismatches" if questionnaire_mismatch else "Human review incomplete"
    )
    with pytest.raises(ValueError, match=message):
        execution.build_execution_freeze(
            bank_path, ledger_path, execution.DEFAULT_BASELINE_PROTOCOL, output
        )
    assert not output.exists()


def test_policy_drift_rejects_before_creating_freeze(engineering_freeze, tmp_path):
    policy = execution.expected_baseline_policy()
    policy["tfidf"]["lowercase"] = False
    changed = tmp_path / "changed_baseline.json"
    changed.write_text(json.dumps(policy), encoding="utf-8")
    _, bank_path, ledger_path, output, _ = engineering_freeze
    with pytest.raises(ValueError, match="Baseline policy differs"):
        execution.build_execution_freeze(bank_path, ledger_path, changed, output)
    assert not output.exists()


def test_test_stage_requires_fits_and_initial_stage_excludes_them(engineering_freeze, tmp_path):
    with pytest.raises(ValueError, match="requires fitted"):
        build(engineering_freeze, stage="test")
    with pytest.raises(ValueError, match="requires fitted"):
        build(engineering_freeze, fit_directories=(tmp_path,))
    assert not engineering_freeze[3].exists()


@pytest.fixture
def metadata_fit(engineering_freeze, tmp_path, monkeypatch):
    directory = tmp_path / "explicit_pytest_metadata_fit"
    directory.mkdir()
    for name in ("settings.json", "heads.npz"):
        (directory / name).write_text(
            "Explicit metadata fixture; not fitted weights.", encoding="utf-8"
        )
    bank = engineering_freeze[0]
    settings = {
        "test_fixture": False,  # Exercise the gate's false branch using a scoped loader stub.
        "pytest_metadata_fixture_only": True,
        "capture_provenance": {
            "bank_sha256": bank_digest(bank),
            "partition": "train_dev",
            "model_settings": {"model_id": "explicit-pytest-model-fixture"},
        },
        "fitting_group_splits": {
            g["group_id"]: g["split"] for g in bank["groups"] if g["split"] != "test"
        },
        "comparisons": [
            {
                "feature_kind": kind,
                "training_n": n,
                "allocation_seed": seed,
                "status": "no_converged_shared_candidate",
            }
            for kind in ("activation", "tfidf")
            for n in execution.TRAINING_SIZES
            for seed in execution.ALLOCATION_SEEDS
        ],
    }

    def fixture_bundle(path):
        assert path == directory.resolve() and settings["pytest_metadata_fixture_only"]
        return copy.deepcopy(settings), {}

    monkeypatch.setattr(execution, "load_fit_bundle", fixture_bundle)
    return directory, settings


def test_test_freeze_binds_complete_preselected_fit_inventory(engineering_freeze, metadata_fit):
    directory, _ = metadata_fit
    result = build(engineering_freeze, stage="test", fit_directories=(directory,))
    assert result["stage"] == "test"
    assert result["fitted_artifact_sha256"] == {
        str((directory / name).resolve()): file_sha256(directory / name)
        for name in ("settings.json", "heads.npz")
    }


@pytest.mark.parametrize("mutation", ["fixture", "bank", "inventory", "test_group", "partition"])
def test_invalid_fitted_metadata_fails_closed(engineering_freeze, metadata_fit, mutation):
    directory, settings = metadata_fit
    if mutation == "fixture":
        settings["test_fixture"] = True
    elif mutation == "bank":
        settings["capture_provenance"]["bank_sha256"] = "different-bank"
    elif mutation == "inventory":
        settings["comparisons"].pop()
    elif mutation == "test_group":
        settings["fitting_group_splits"]["fixture_test"] = "test"
    else:
        settings["capture_provenance"]["partition"] = "test"
    with pytest.raises(ValueError):
        build(engineering_freeze, stage="test", fit_directories=(directory,))
    assert not engineering_freeze[3].exists()
