"""Immutable review merging on explicit synthetic fixtures, never human attestations."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_main_study_bank import synthetic_bank as synthetic_bank

from latent_probes.main_study_bank import (
    bank_digest,
    prompt_digest,
    render_messages,
    representative_cases,
)
from latent_probes.main_study_review import (
    merge_review_fragments,
    validate_questionnaire_answers,
    write_fresh_json,
)
from latent_probes.storage import file_sha256


def record(group, role="student", *, questionnaire=True):
    cases = representative_cases(group)
    value = {
        "role": role,
        "group_id": group["group_id"],
        "reviewer": f"SYNTHETIC {role.upper()} (test only)",
        "reviewed_at": "2020-01-01T00:00:00Z",
        "decision": "accepted",
        "notes": "Explicit software fixture; not actual human validation.",
        "prompt_sha256s": [prompt_digest(render_messages(group, c)) for c in cases],
    }
    if questionnaire:
        value["independent_answers"] = {c.case_id: "YES" if c.gold else "NO" for c in cases}
    return value


@pytest.fixture
def source(synthetic_bank, tmp_path):
    path = tmp_path / "synthetic_bank.json"
    path.write_text(json.dumps(synthetic_bank), encoding="utf-8")

    def fragment(name, records=(), disputes=()):
        output = tmp_path / name
        output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "bank_sha256": bank_digest(synthetic_bank),
                    "records": list(records),
                    "disputed_groups": list(disputes),
                }
            ),
            encoding="utf-8",
        )
        return output

    return synthetic_bank, path, fragment


def test_blank_ledger_reports_154_missing_without_creating_human_records(source):
    _, bank_path, fragment = source
    path = fragment("blank.json")
    original = path.read_bytes()
    merged, readiness = merge_review_fragments(bank_path, [path])
    assert merged["records"] == [] and readiness["status"] == "incomplete"
    assert len(readiness["missing_student_groups"]) == 154
    assert readiness["human_records_created"] == 0 and readiness["humanity_authenticated"] is False
    assert path.read_bytes() == original
    assert readiness["source_fragments"][0]["sha256"] == file_sha256(path)


def test_disputes_are_unioned_without_relabelling_records(source):
    bank, bank_path, fragment = source
    first, second = bank["groups"][:2]
    a = record(first)
    b = record(second, "independent")
    b["decision"] = "disputed"
    paths = [fragment("first.json", [a], [first["group_id"]]), fragment("second.json", [b])]
    merged, readiness = merge_review_fragments(bank_path, paths)
    assert set(merged["disputed_groups"]) == {first["group_id"], second["group_id"]}
    assert sorted(merged["records"], key=lambda r: r["group_id"]) == sorted(
        [a, b], key=lambda r: r["group_id"]
    )
    assert readiness["review_complete"] is False


@pytest.mark.parametrize("answer", ("YES", "UNSURE"))
def test_wrong_or_unsure_accepted_answers_are_unresolved_not_repaired(source, answer):
    bank, bank_path, fragment = source
    human = record(bank["groups"][0])
    case_id = next(iter(human["independent_answers"]))
    # The first representative fixture has formal NO; these simulate disagreement.
    assert human["independent_answers"][case_id] == "NO"
    human["independent_answers"][case_id] = answer
    path = fragment("disagreement.json", [human])
    original = path.read_bytes()
    merged, readiness = merge_review_fragments(bank_path, [path])
    assert readiness["review_complete"] is False
    assert readiness["unresolved_answer_mismatches"][0]["case_id"] == case_id
    assert readiness["unresolved_answer_mismatches"][0]["recorded_answer"] == answer
    assert merged["records"][0] == human and path.read_bytes() == original


@pytest.mark.parametrize(
    "corruption", ("bank", "case", "name", "role", "hash", "timestamp", "answer", "partial")
)
def test_stale_or_invalid_fields_are_rejected(source, corruption):
    bank, bank_path, fragment = source
    human = record(bank["groups"][0])
    path = fragment("invalid.json", [human])
    value = json.loads(path.read_text(encoding="utf-8"))
    r = value["records"][0]
    if corruption == "bank":
        value["bank_sha256"] = "stale"
    elif corruption == "case":
        r["independent_answers"]["unknown_case"] = "NO"
    elif corruption == "name":
        r["reviewer"] = "   "
    elif corruption == "role":
        r["role"] = "agent"
    elif corruption == "hash":
        r["prompt_sha256s"][0] = "stale"
    elif corruption == "timestamp":
        r["reviewed_at"] = "2099-01-01T00:00:00Z"
    elif corruption == "answer":
        r["independent_answers"][next(iter(r["independent_answers"]))] = True
    else:
        r["independent_answers"].pop(next(iter(r["independent_answers"])))
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        merge_review_fragments(bank_path, [path])


def test_identical_and_conflicting_duplicate_role_group_records_are_rejected(source):
    bank, bank_path, fragment = source
    a = record(bank["groups"][0])
    b = copy.deepcopy(a)
    paths = [fragment("a.json", [a]), fragment("b.json", [b])]
    with pytest.raises(ValueError, match="Duplicate.*role, group"):
        merge_review_fragments(bank_path, paths)
    b["notes"] = "Different fixture notes still require explicit reconciliation."
    paths[1].write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bank_sha256": bank_digest(bank),
                "records": [b],
                "disputed_groups": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate.*role, group"):
        merge_review_fragments(bank_path, paths)


def test_complete_synthetic_records_satisfy_existing_gate_without_claiming_humanity(source):
    bank, bank_path, fragment = source
    students = [record(group) for group in bank["groups"]]
    strata = {}
    for group in bank["groups"]:
        strata.setdefault((group["split"], group["mechanism"]), group)
    independent = [record(group, "independent") for group in strata.values()]
    paths = [
        fragment("synthetic_students.json", students),
        fragment("synthetic_independent.json", independent),
    ]
    merged, readiness = merge_review_fragments(bank_path, paths)
    assert readiness["status"] == "accepted" and readiness["review_complete"] is True
    assert readiness["humanity_authenticated"] is False and readiness["human_records_created"] == 0
    assert readiness["questionnaire_records"] == len(students) + len(independent)
    assert not readiness["missing_student_groups"] and not readiness["gate_errors"]
    wrong = copy.deepcopy(merged)
    first = wrong["records"][0]
    case = next(iter(first["independent_answers"]))
    first["independent_answers"][case] = "UNSURE"
    assert validate_questionnaire_answers(bank, wrong)


def test_legacy_records_are_reported_without_inventing_answers(source):
    bank, bank_path, fragment = source
    legacy = record(bank["groups"][0], questionnaire=False)
    merged, readiness = merge_review_fragments(bank_path, [fragment("legacy.json", [legacy])])
    assert merged["records"] == [legacy]
    assert readiness["legacy_records_without_questionnaire"] == 1
    assert readiness["questionnaire_records"] == 0


def test_cli_check_only_exits_two_and_preserves_existing_output(source, tmp_path):
    _, bank_path, fragment = source
    blank = fragment("blank.json")
    report_path = tmp_path / "readiness.json"
    script = Path(__file__).resolve().parents[1] / "scripts/validate_main_study_review.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--bank",
            str(bank_path),
            "--fragments",
            str(blank),
            "--report",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 2, result.stderr
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "incomplete"
    original = report_path.read_bytes()
    with pytest.raises(FileExistsError):
        write_fresh_json(report_path, {"must_not_overwrite": True})
    assert report_path.read_bytes() == original
