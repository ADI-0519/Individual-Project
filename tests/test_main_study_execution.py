"""Metadata-only engineering fixtures; none is an actual human attestation or fit."""

from __future__ import annotations

import copy
import json

import pytest
from test_main_study_bank import synthetic_bank as synthetic_bank

from latent_probes import main_study_execution as execution
from latent_probes.main_study_bank import (
    bank_digest,
    prompt_digest,
    render_messages,
    representative_cases,
)
from latent_probes.storage import file_sha256, read_json, utc_now


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


def test_human_freeze_refuses_to_relabel_agent_fits(engineering_freeze, metadata_fit):
    directory, settings = metadata_fit
    settings["capture_provenance"]["research_status"] = execution.AGENT_RESEARCH_STATUS
    with pytest.raises(ValueError, match="cannot relabel"):
        build(engineering_freeze, stage="test", fit_directories=(directory,))


@pytest.fixture
def agent_materials(synthetic_bank, tmp_path, monkeypatch):
    """Entirely synthetic gate fixtures; these never claim actual semantic review."""
    bank_path, audit_path, packet = [
        tmp_path / name for name in ("synthetic_bank.json", "synthetic_agent.json", "packet.jsonl")
    ]
    bank_path.write_text(json.dumps(synthetic_bank), encoding="utf-8")
    rows, records = [], []
    for group in synthetic_bank["groups"]:
        for case in representative_cases(group):
            messages = render_messages(group, case)
            digest = prompt_digest(messages)
            rows.append(
                {
                    "case_id": case.case_id,
                    "group_id": group["group_id"],
                    "prompt_sha256": digest,
                    "messages": messages,
                }
            )
            records.append(
                {
                    "case_id": case.case_id,
                    "prompt_sha256": digest,
                    "answer": "YES" if case.gold else "NO",
                    "assumptions": ["SYNTHETIC PYTEST FIXTURE ONLY, not a real independent answer"],
                    "issues": [],
                }
            )
    packet.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    audit = {
        "schema": "main-study-agent-semantic-audit-v1",
        "reviewer_type": "agent",
        "reviewer_id": "EXPLICIT PYTEST FIXTURE, NOT AN ACTUAL AGENT REVIEW",
        "author_independent": True,
        "bank_sha256": bank_digest(synthetic_bank),
        "review_packet_sha256": file_sha256(packet),
        "review_scope": "all_representative_prompts_and_transformations",
        "transformation_review": {
            "location_complement": True,
            "world_independence": True,
            "global_name_exchange": True,
        },
        "records": records,
        "unresolved_blockers": [],
        "limitations": ["Synthetic pytest fixture; not real review or human acceptance."],
        "created_at": utc_now(),
    }
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    monkeypatch.setattr(execution, "_require_versioned_amendment", lambda: None)
    return synthetic_bank, bank_path, audit_path, packet, audit


def agent_build(materials, output, **kwargs):
    _, bank, audit, packet, _ = materials
    return execution.build_agent_execution_freeze(
        bank,
        audit,
        execution.DEFAULT_BASELINE_PROTOCOL,
        output,
        review_packet_path=packet,
        **kwargs,
    )


def test_agent_freeze_keeps_original_human_gate_and_provenance_separate(agent_materials, tmp_path):
    _, bank, audit, _, _ = agent_materials
    output = tmp_path / "agent_freeze.json"
    result = agent_build(agent_materials, output)
    assert result["schema"] != "main-study-execution-freeze-v1"
    assert result["research_status"] == "agent_audited_exploratory"
    assert result["human_review_complete"] is False and result["confirmatory_result"] is False
    assert result["review"]["representative_prompts"] == 736
    assert result["review"]["review_complete"] is False
    assert result["source_sha256"][str(execution.AGENT_AMENDMENT.resolve())] == file_sha256(
        execution.AGENT_AMENDMENT
    )
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        agent_build(agent_materials, output)
    assert output.read_bytes() == original
    with pytest.raises(ValueError, match="Review ledger must bind"):
        execution.build_execution_freeze(
            bank, audit, execution.DEFAULT_BASELINE_PROTOCOL, tmp_path / "human.json"
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_answer",
        "unsure",
        "missing",
        "duplicate",
        "prompt_hash",
        "bank_hash",
        "packet_hash",
        "blocker",
        "case_issue",
        "human",
        "not_independent",
        "no_transform",
        "future",
        "naive_time",
        "no_assumptions",
        "no_limits",
    ],
)
def test_agent_audit_rejects_incomplete_or_disagreeing_review(agent_materials, mutation):
    bank, _, path, packet, audit = agent_materials
    if mutation == "wrong_answer":
        audit["records"][0]["answer"] = "NO" if audit["records"][0]["answer"] == "YES" else "YES"
    elif mutation == "unsure":
        audit["records"][0]["answer"] = "UNSURE"
    elif mutation == "missing":
        audit["records"].pop()
    elif mutation == "duplicate":
        audit["records"][-1] = audit["records"][0]
    elif mutation == "prompt_hash":
        audit["records"][0]["prompt_sha256"] = "wrong"
    elif mutation == "bank_hash":
        audit["bank_sha256"] = "wrong"
    elif mutation == "packet_hash":
        audit["review_packet_sha256"] = "wrong"
    elif mutation == "blocker":
        audit["unresolved_blockers"] = ["ambiguous prose"]
    elif mutation == "case_issue":
        audit["records"][0]["issues"] = ["ambiguous prose"]
    elif mutation == "human":
        audit["reviewer_type"] = "human"
    elif mutation == "not_independent":
        audit["author_independent"] = False
    elif mutation == "no_transform":
        audit["transformation_review"]["world_independence"] = False
    elif mutation == "future":
        audit["created_at"] = "2999-01-01T00:00:00+00:00"
    elif mutation == "naive_time":
        audit["created_at"] = "2020-01-01T00:00:00"
    elif mutation == "no_assumptions":
        audit["records"][0].pop("assumptions")
    else:
        audit["limitations"] = []
    path.write_text(json.dumps(audit), encoding="utf-8")
    saved = path.read_bytes()
    with pytest.raises(ValueError):
        execution.validate_agent_audit(bank, path, packet)
    assert path.read_bytes() == saved  # Disagreement is never auto-corrected.


def test_agent_packet_cannot_rebind_unrelated_prose(agent_materials):
    bank, _, audit_path, packet, audit = agent_materials
    rows = [json.loads(line) for line in packet.read_text(encoding="utf-8").splitlines()]
    rows[0]["messages"][0]["content"] = "Unrelated text"
    packet.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    audit["review_packet_sha256"] = file_sha256(packet)
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly match"):
        execution.validate_agent_audit(bank, audit_path, packet)


def test_agent_authorization_preserves_source_and_stage_checks(agent_materials, tmp_path):
    _, bank_path, audit, _, _ = agent_materials
    freeze_path = tmp_path / "agent_freeze.json"
    freeze = agent_build(agent_materials, freeze_path)
    protocol = {
        "bank_path": str(bank_path),
        "bank_sha256": freeze["bank_sha256"],
        "policy_sha256": {
            str((execution.ROOT / p).resolve()): file_sha256(execution.ROOT / p)
            for p in execution.CAPTURE_POLICY_FILES
        },
    }
    authorized = execution.authorize_agent_execution(
        protocol, audit, freeze_path, partition="train_dev", fit_dir=None
    )
    assert authorized["human_review_complete"] is False
    assert authorized["review"]["review_complete"] is False
    with pytest.raises(ValueError, match="stage"):
        execution.authorize_agent_execution(
            protocol, audit, freeze_path, partition="test", fit_dir=None
        )
    freeze["source_sha256"].pop(
        str((execution.ROOT / "src/latent_probes/main_study_fit.py").resolve())
    )
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    with pytest.raises(ValueError, match="all source"):
        execution.authorize_agent_execution(
            protocol, audit, freeze_path, partition="train_dev", fit_dir=None
        )


def test_agent_test_freeze_requires_both_fits_and_initial_binding(
    agent_materials, tmp_path, monkeypatch
):
    initial_path = tmp_path / "initial_agent.json"
    initial = agent_build(agent_materials, initial_path)
    bank = agent_materials[0]
    directories, all_settings = [], {}
    for model in execution.MODELS.values():
        directory = tmp_path / model["model_type"]
        directory.mkdir()
        for name in ("heads.npz", "settings.json"):
            (directory / name).write_text(
                "METADATA PYTEST FIXTURE, NOT FITTED HEADS", encoding="utf-8"
            )
        directories.append(directory)
        all_settings[directory] = {
            "test_fixture": False,
            "capture_provenance": {
                "partition": "train_dev",
                "bank_sha256": bank_digest(bank),
                "model_settings": {"model_id": model["model_id"]},
                "research_status": execution.AGENT_RESEARCH_STATUS,
                "human_review_complete": False,
                "source_bindings": {str(initial_path): file_sha256(initial_path)},
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
    monkeypatch.setattr(execution, "load_fit_bundle", lambda path: (all_settings[path], {}))
    with pytest.raises(ValueError, match="Both pinned"):
        agent_build(
            agent_materials,
            tmp_path / "one_model.json",
            stage="test",
            fit_directories=tuple(directories[:1]),
            initial_freeze_path=initial_path,
        )
    all_settings[directories[0]]["capture_provenance"]["source_bindings"] = {}
    with pytest.raises(ValueError, match="exact initial"):
        agent_build(
            agent_materials,
            tmp_path / "missing_initial.json",
            stage="test",
            fit_directories=tuple(directories),
            initial_freeze_path=initial_path,
        )
    all_settings[directories[0]]["capture_provenance"]["source_bindings"] = {
        str(initial_path): file_sha256(initial_path)
    }
    result = agent_build(
        agent_materials,
        tmp_path / "both_models.json",
        stage="test",
        fit_directories=tuple(directories),
        initial_freeze_path=initial_path,
    )
    assert result["stage"] == "test" and result["bank_sha256"] == initial["bank_sha256"]
    assert len(result["fitted_artifact_sha256"]) == 4
