"""Merge immutable human-review fragments and report actual gate readiness.

Field validation cannot authenticate humanity. No function creates human
answers, resolves a dispute, changes a decision or repairs a mismatch against
the formal state rule. A new human record is needed for reconciliation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from .storage import file_sha256


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON field: {key}")
        value[key] = item
    return value


def validate_questionnaire_answers(bank: dict, ledger: dict) -> list[dict]:
    """Return unresolved accepted-answer mismatches; never alter human records.

    This helper intentionally imports bank functions locally so the existing
    execution review gate can call it without creating an import cycle.
    Legacy schema-1 records may omit questionnaires; coverage is reported by
    the merger rather than fabricated. Accepted supplied questionnaires must
    cover every representative prompt and agree with the formal rule.
    """
    from .main_study_bank import representative_cases

    cases = {g["group_id"]: {c.case_id: c for c in representative_cases(g)} for g in bank["groups"]}
    mismatches = []
    for record in ledger["records"]:
        if "independent_answers" not in record:
            continue
        answers = record["independent_answers"]
        group = cases.get(record["group_id"])
        if group is None or not isinstance(answers, dict) or not set(answers) <= set(group):
            raise ValueError("Questionnaire has an unknown group/case or malformed answers")
        if any(
            type(answer) is not str or answer not in {"YES", "NO", "UNSURE"}
            for answer in answers.values()
        ):
            raise ValueError("Questionnaire answers must be YES, NO or UNSURE")
        if record["decision"] != "accepted":
            continue
        if set(answers) != set(group):
            raise ValueError("Accepted questionnaire must cover every representative case")
        for case_id, answer in answers.items():
            if answer != ("YES" if group[case_id].gold else "NO"):
                mismatches.append(
                    {
                        "group_id": record["group_id"],
                        "case_id": case_id,
                        "reviewer": record["reviewer"],
                        "role": record["role"],
                        "recorded_answer": answer,
                        "reason": "UNSURE"
                        if answer == "UNSURE"
                        else "differs_from_formal_state_rule",
                    }
                )
    return mismatches


def _validate_fragment(bank, ledger):
    from .main_study_bank import bank_digest, prompt_digest, render_messages, representative_cases

    fields = {"schema_version", "bank_sha256", "records", "disputed_groups"}
    if (
        not isinstance(ledger, dict)
        or set(ledger) != fields
        or type(ledger["schema_version"]) is not int
        or ledger["schema_version"] != 1
    ):
        raise ValueError("Expected exactly the schema-1 human-review ledger fields")
    if ledger["bank_sha256"] != bank_digest(bank):
        raise ValueError("Review fragment bank hash differs from the exact bank")
    groups = {g["group_id"]: g for g in bank["groups"]}
    if not isinstance(ledger["records"], list) or not isinstance(ledger["disputed_groups"], list):
        raise ValueError("Review records and disputed_groups must be arrays")
    if any(not isinstance(g, str) or g not in groups for g in ledger["disputed_groups"]):
        raise ValueError("Unknown disputed group")
    required = {
        "role",
        "group_id",
        "reviewer",
        "reviewed_at",
        "decision",
        "notes",
        "prompt_sha256s",
    }
    for record in ledger["records"]:
        if (
            not isinstance(record, dict)
            or not required <= set(record)
            or set(record) - (required | {"independent_answers"})
        ):
            raise ValueError("Malformed human-review record fields")
        if (
            not isinstance(record["role"], str)
            or record["role"] not in {"student", "independent"}
            or not isinstance(record["group_id"], str)
            or record["group_id"] not in groups
        ):
            raise ValueError("Unknown reviewer role or group")
        name = record["reviewer"]
        if not isinstance(name, str) or not name.strip() or any(ord(c) < 32 for c in name):
            raise ValueError("An actual nonempty reviewer name is required")
        if (
            not isinstance(record["notes"], str)
            or not isinstance(record["decision"], str)
            or record["decision"]
            not in {
                "accepted",
                "disputed",
                "rejected",
            }
        ):
            raise ValueError("Invalid review decision or notes")
        try:
            timestamp = datetime.fromisoformat(record["reviewed_at"].replace("Z", "+00:00"))
        except (TypeError, AttributeError, ValueError) as error:
            raise ValueError("An actual timezone-aware review timestamp is required") from error
        if timestamp.tzinfo is None or timestamp > datetime.now(UTC):
            raise ValueError("Review timestamp must be aware and not in the future")
        group = groups[record["group_id"]]
        expected = {prompt_digest(render_messages(group, c)) for c in representative_cases(group)}
        actual = record["prompt_sha256s"]
        if (
            not isinstance(actual, list)
            or any(not isinstance(h, str) for h in actual)
            or (len(actual) != len(set(actual)))
            or not set(actual) <= expected
        ):
            raise ValueError("Unknown/duplicate/stale representative prompt hash")
        if record["decision"] == "accepted" and set(actual) != expected:
            raise ValueError("Accepted record must bind every representative prompt")
    return validate_questionnaire_answers(bank, ledger)


def merge_review_fragments(bank_path, fragment_paths) -> tuple[dict, dict]:
    """Merge without side effects; incomplete records remain incomplete."""
    from .main_study_bank import bank_digest, load_bank, verify_review_ledger

    bank = load_bank(bank_path)
    if not fragment_paths:
        raise ValueError("Need at least one existing human-review fragment")
    records, disputed, sources, seen, paths, mismatches = [], set(), [], set(), set(), []
    for path in fragment_paths:
        path = Path(path).resolve()
        if path in paths:
            raise ValueError("Duplicate review fragment path")
        paths.add(path)
        original = path.read_bytes()
        ledger = json.loads(original.decode("utf-8"), object_pairs_hook=_unique_object)
        mismatches.extend(_validate_fragment(bank, ledger))
        for record in ledger["records"]:
            identity = (record["role"], record["group_id"])
            if identity in seen:
                raise ValueError(
                    "Duplicate (role, group) review record; reconcile in a new fragment"
                )
            seen.add(identity)
            records.append(record)
            if record["decision"] != "accepted":
                disputed.add(record["group_id"])
        disputed.update(ledger["disputed_groups"])
        sources.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(original).hexdigest(),
                "records": len(ledger["records"]),
            }
        )
    merged = {
        "schema_version": 1,
        "bank_sha256": bank_digest(bank),
        "records": sorted(records, key=lambda r: (r["role"], r["group_id"])),
        "disputed_groups": sorted(disputed),
    }
    errors, gate = [], None
    if mismatches:
        errors.append(f"Unresolved independent-answer mismatches: {len(mismatches)}")
    with TemporaryDirectory(prefix="main-study-review-check-") as directory:
        check_path = Path(directory) / "merged.json"
        check_path.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
        try:
            gate = verify_review_ledger(bank, check_path)
        except ValueError as error:
            errors.append(str(error))
    accepted_students = {
        r["group_id"] for r in records if r["role"] == "student" and r["decision"] == "accepted"
    }
    report = {
        "schema": "main-study-human-review-readiness-v1",
        "status": "accepted" if not errors else "incomplete",
        "review_complete": not errors,
        "bank_sha256": bank_digest(bank),
        "bank_file_sha256": file_sha256(bank_path),
        "source_fragments": sources,
        "human_records": len(records),
        "missing_student_groups": sorted(
            {g["group_id"] for g in bank["groups"]} - accepted_students
        ),
        "questionnaire_records": sum("independent_answers" in r for r in records),
        "legacy_records_without_questionnaire": sum(
            "independent_answers" not in r for r in records
        ),
        "unresolved_answer_mismatches": mismatches,
        "disputed_groups": sorted(disputed),
        "gate_errors": errors,
        "existing_ledger_gate": gate,
        "humanity_authenticated": False,
        "human_records_created": 0,
        "scope": "Field/provenance and formal-answer checks; software cannot authenticate humanity. Human records are preserved unchanged. No automatic reconciliation.",
    }
    return merged, report


def write_fresh_json(path, value):
    """Exclusively create an output; never overwrite a human review artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
