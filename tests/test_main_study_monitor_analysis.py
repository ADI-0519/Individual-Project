"""Invalid-output and journal replay regression tests on explicit fixtures."""

from __future__ import annotations

import json
import re

import pytest
from test_main_study_bank import synthetic_bank as synthetic_bank

from latent_probes.main_study_bank import bank_digest, iter_rows, prompt_digest
from latent_probes.main_study_design import BeliefCase
from latent_probes.main_study_monitor import parse_monitor_output
from latent_probes.main_study_monitor_analysis import analyse_monitor_run, monitor_group_metrics
from latent_probes.storage import atomic_json, file_sha256


def fixture_record(row, *, invalid=False, finish="eos"):
    """Construct oracle extractions solely for verifying software, never inference."""
    case = BeliefCase(**row["state"])
    # These fixture entities are literal and the renderer repeats them verbatim.
    introduction = row["messages"][1]["content"]
    initial_name = re.search(r"inspect the token inside the (.+?)\. ", introduction).group(1)
    candidate_name = re.search(
        r'Candidate proposition: "The token is in the (.+?)\."', introduction
    ).group(1)
    # Infer the other visible fixture name from the initial/candidate or event.
    places = [m for m in re.findall(r"SYNTHETIC_[A-Za-z0-9_]+_(?:ALPHA|BETA)", introduction)]
    assert 1 <= len(set(places)) <= 2
    # A fully redundant state can mention only one name. This fixture's known
    # naming convention supplies the unused name; no production parser does so.
    entity_prefix = places[0].rsplit("_", 1)[0]
    zero, one = entity_prefix + "_ALPHA", entity_prefix + "_BETA"
    names = [zero, one] if not case.name_map else [one, zero]
    assert initial_name == names[case.initial]
    assert candidate_name == names[1]
    updates = [
        {
            "location": names[case.update],
            "accepted_access": bool(
                case.first_receipt if case.mechanism == "correction" else case.receipt
            ),
        }
    ]
    if case.mechanism == "correction":
        updates.append(
            {"location": names[case.final_update], "accepted_access": bool(case.receipt)}
        )
    raw = (
        "malformed fixture"
        if invalid
        else json.dumps(
            {
                "initial_location": initial_name,
                "updates": updates,
                "queried_location": candidate_name,
            }
        )
    )
    return {
        **row,
        **parse_monitor_output(raw, row["messages"], finish_reason=finish),
        "prompt_sha256": prompt_digest(row["messages"]),
        "raw_response": raw,
        "_fixture_finish": finish,
    }


def test_one_invalid_prediction_is_incorrect_without_gold_dependent_imputation(synthetic_bank):
    rows = list(iter_rows(synthetic_bank, "development"))[:32]
    records = [fixture_record(r, invalid=i == 0) for i, r in enumerate(rows)]
    assert rows[0]["gold"] == 0  # Filling invalid with zero would falsely mark this correct.
    result = monitor_group_metrics(rows, records)
    assert (result["rows"], result["valid"], result["invalid"], result["correct"]) == (
        32,
        31,
        1,
        31,
    )
    assert result["accuracy"] == 31 / 32
    assert result["complete_quartet_accuracy"] == 7 / 8
    assert result["valid_only_binary_metrics"]["rows"] == 31
    assert result["valid_only_binary_metrics"]["accuracy"] == 1
    assert result["preserving"]["both_valid_pairs"] == 7
    assert records[0]["prediction"] is records[0]["score"] is None


def test_all_invalid_and_secondary_correction_outcomes_are_retained(synthetic_bank):
    rows = [r for r in iter_rows(synthetic_bank, "test") if r["mechanism"] == "correction"][:64]
    result = monitor_group_metrics(rows, [fixture_record(r, invalid=True) for r in rows])
    assert result["rows"] == result["invalid"] == 64
    assert result["accuracy"] == result["complete_quartet_accuracy"] == 0
    assert result["valid_only_binary_metrics"] is None
    assert result["changing"]["verdict_flip_fraction_of_valid_pairs"] is None
    assert result["secondary_correction_cells"] == {
        "rows": 32,
        "invalid": 32,
        "correct": 0,
        "accuracy": 0,
    }


def test_exact_scenario_coverage_and_null_invalid_contract(synthetic_bank):
    rows = list(iter_rows(synthetic_bank, "development"))[:32]
    records = [fixture_record(r) for r in rows]
    with pytest.raises(ValueError, match="exactly cover"):
        monitor_group_metrics(rows, records[:-1])
    with pytest.raises(ValueError, match="exactly cover"):
        monitor_group_metrics(rows, records + records[:1])
    records[0].update(valid=False, prediction=0, score=0.0)
    with pytest.raises(ValueError, match="stay null"):
        monitor_group_metrics(rows, records)


@pytest.fixture
def complete_fixture_run(synthetic_bank, tmp_path):
    bank_path = tmp_path / "fixture_bank.json"
    atomic_json(bank_path, synthetic_bank)
    protocol = {
        "bank_sha256": bank_digest(synthetic_bank),
        "test_fixture": True,
        "monitor_protocol_sha256": "fixture_policy",
        "model_settings": {"fixture": True},
    }
    rows = list(iter_rows(synthetic_bank, "development"))
    protocol["planned_rows"] = [{"case_id": r["case_id"], "prompt_token_ids": [42]} for r in rows]
    atomic_json(tmp_path / "protocol.json", protocol)
    protocol_hash = file_sha256(tmp_path / "protocol.json")
    records = []
    for index, row in enumerate(rows):
        record = fixture_record(row, finish="length" if index == 0 else "eos")
        finish = record.pop("_fixture_finish")
        raw_path = tmp_path / "row_journal" / row["case_id"] / "raw_generation.json"
        atomic_json(
            raw_path,
            {
                "raw_response": record["raw_response"],
                "generation": {
                    "finish_reason": finish,
                    "prompt_token_ids": [42],
                    "generated_token_ids": [1],
                },
            },
        )
        record.update(
            raw_generation_sha256=file_sha256(raw_path),
            status="complete",
            test_fixture=True,
            protocol_sha256=protocol_hash,
        )
        records.append(record)  # Actual runner exports no top-level finish_reason.
    atomic_json(
        tmp_path / "records.json",
        {
            "schema": "main-study-text-monitor-v1",
            "bank_sha256": bank_digest(synthetic_bank),
            "protocol_sha256": protocol_hash,
            "monitor_protocol_sha256": "fixture_policy",
            "model_settings": {"fixture": True},
            "partition": "development",
            "test_fixture": True,
            "records": records,
        },
    )
    atomic_json(
        tmp_path / "run_manifest.json",
        {
            "status": "complete",
            "partition": "development",
            "protocol_sha256": protocol_hash,
            "records_sha256": file_sha256(tmp_path / "records.json"),
        },
    )
    return bank_path, tmp_path


def test_real_runner_record_shape_replays_bound_finish_reason_and_capped_output(
    complete_fixture_run,
):
    bank_path, run_dir = complete_fixture_run
    result = analyse_monitor_run(bank_path, run_dir, allow_fixture=True)
    assert result["scientific_result"] is False
    assert result["overall"]["rows"] == 512
    assert result["overall"]["invalid"] == 1
    assert result["overall"]["correct"] == 511
    assert result["by_mechanism"]["observation"]["scenario_groups"] == 8
    with pytest.raises(ValueError, match="Fixture outputs"):
        analyse_monitor_run(bank_path, run_dir)


def test_raw_journal_tampering_is_rejected(complete_fixture_run):
    bank_path, run_dir = complete_fixture_run
    raw = next((run_dir / "row_journal").glob("*/raw_generation.json"))
    raw.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Raw monitor generation hash"):
        analyse_monitor_run(bank_path, run_dir, allow_fixture=True)
