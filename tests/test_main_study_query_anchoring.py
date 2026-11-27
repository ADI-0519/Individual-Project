"""Fail-closed and cancellation tests for the NEW same-response control."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

from latent_probes.main_study_bank import iter_rows

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "query_anchoring", ROOT / "scripts/audit_main_study_query_anchoring.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def fixture_pair():
    messages = [
        {"role": "system", "content": "Read this complete fictional account."},
        {
            "role": "user",
            "content": 'The initial location is blue cabinet. The other place is red trunk.\n\nCandidate proposition: "The token is in the red trunk."\n\nAt the end of the scene, does Sender think that Reader believes that the token is in the blue cabinet? Answer only YES or NO.',
        },
    ]
    row = {
        "case_id": "fixture_case",
        "group_id": "fixture_group",
        "mechanism": "correction",
        "split": "test",
        "gold": 1,
        "state": {"initial": 1, "update": 0},
        "messages": messages,
    }
    record = {
        "case_id": row["case_id"],
        "messages": deepcopy(messages),
        "valid": True,
        "prediction": 1,
        "score": 1.0,
        "final_location": "red trunk",
        "extraction": {
            "initial_location": "red trunk",
            "queried_location": "red trunk",
            "updates": [{"location": "blue cabinet", "accepted_access": False}],
        },
    }
    return row, record


def test_query_uses_final_question_not_candidate_proposition():
    row, _ = fixture_pair()
    assert AUDIT.actual_question_location(row["messages"]) == "blue cabinet"


def test_question_parser_accepts_all_full_bank_contexts_without_state_argument():
    rows = list(iter_rows(AUDIT.read(ROOT / "data/belief_main/main_v1/main_bank_v1.json")))
    assert len(rows) == 5888
    assert all(AUDIT.actual_question_location(r["messages"]) for r in rows)


@pytest.mark.parametrize("mutation", ["missing", "ambiguous", "anaphoric"])
def test_query_parser_fails_without_one_explicit_question(mutation):
    row, _ = fixture_pair()
    messages = row["messages"]
    if mutation == "missing":
        messages[-1]["content"] = "No final question is supplied."
    elif mutation == "ambiguous":
        messages.append(deepcopy(messages[-1]))
    else:
        messages[-1]["content"] = "Does Reader believe it is there?"
    with pytest.raises(ValueError, match="exactly one explicit"):
        AUDIT.actual_question_location(messages)


@pytest.mark.parametrize("mutation", ["different_id", "duplicate_id", "different_prose"])
def test_alignment_corruption_fails(mutation):
    row, record = fixture_pair()
    records = [record]
    if mutation == "different_id":
        record["case_id"] = "another_case"
    elif mutation == "duplicate_id":
        records.append(deepcopy(record))
    else:
        record["messages"][-1]["content"] += " Changed."
    with pytest.raises(ValueError):
        AUDIT.anchor_records([row], records)


def test_two_component_errors_can_cancel_before_query_anchoring():
    row, record = fixture_pair()
    frozen = deepcopy(record)
    anchored = AUDIT.anchor_records([row], [record])
    assert record == frozen
    assert anchored[0]["original_prediction"] == 1 == row["gold"]
    assert anchored[0]["anchored_prediction"] == 0
    counts = AUDIT.comparison_counts(anchored)
    assert counts["wrong_query_original_correct_anchored_wrong"] == 1


def test_invalid_validity_and_null_prediction_remain_frozen():
    row, record = fixture_pair()
    record.update(
        valid=False,
        prediction=None,
        score=None,
        extraction=None,
        final_location=None,
        invalid_reason="frozen invalid",
    )
    result = AUDIT.anchor_records([row], [record])[0]
    assert result["valid"] is False
    assert result["original_prediction"] is result["anchored_prediction"] is None
    assert result["invalid_reason"] == "frozen invalid"


def test_inconsistent_saved_final_location_fails():
    row, record = fixture_pair()
    record["final_location"] = "blue cabinet"
    with pytest.raises(ValueError, match="final location"):
        AUDIT.anchor_records([row], [record])


def test_portable_binding_uses_logical_root_instead_of_recorded_native_path(tmp_path):
    moved_root = tmp_path / "relocated_workspace"
    source = moved_root / "scripts" / "control.py"
    source.parent.mkdir(parents=True)
    source.write_text("portable source\n", encoding="utf-8")
    binding = AUDIT.source_binding(source, "workspace", moved_root)
    binding["path"] = "C:\\nonexistent\\historical-machine\\control.py"
    assert AUDIT.resolve_bound_source(binding, {"workspace": moved_root}) == source.resolve()
    assert AUDIT.verify_bound_sources({"example": binding}, {"workspace": moved_root}) == 1
    source.write_text("changed source\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash differs"):
        AUDIT.verify_bound_sources({"example": binding}, {"workspace": moved_root})


@pytest.mark.parametrize(
    "relative", ["../escape.py", "/absolute.py", "C:/drive.py", "dir\\file.py"]
)
def test_portable_binding_rejects_escaping_or_native_relative_paths(tmp_path, relative):
    binding = {"root": "workspace", "relative_path": relative}
    with pytest.raises(ValueError, match="relative POSIX"):
        AUDIT.resolve_bound_source(binding, {"workspace": tmp_path})


@pytest.mark.parametrize(
    "name",
    [
        "2026-11-22-main-study-measurement-forensics-v2.json",
        "2026-11-22-main-study-query-anchoring-posthoc-v2.json",
    ],
)
def test_current_v2_workspace_bindings_survive_recorded_path_relocation(name):
    report = AUDIT.read(ROOT / "docs/validation" / name)
    assert report["audit_version"] == 2
    bindings = {
        key: dict(binding, path="/obsolete/native/path")
        for key, binding in report["source_files"].items()
        if binding["root"] == "workspace"
    }
    assert AUDIT.verify_bound_sources(bindings, {"workspace": ROOT}) == len(bindings)
    if "measurement_report_selection" in report:
        assert report["measurement_report_selection"]["relative_path"] == (
            "docs/validation/2026-11-22-main-study-measurement-forensics.json"
        )
        assert bindings["current_measurement_code"]["sha256"] == AUDIT.sha(
            ROOT / "scripts/audit_main_study_measurement_forensics.py"
        )


def test_historical_source_snapshots_prove_original_report_bindings():
    index = AUDIT.read(ROOT / "docs/validation/audit-source-snapshots-2026-11-22/index.json")
    original_query = AUDIT.read(
        ROOT / "docs/validation/2026-11-22-main-study-query-anchoring-posthoc.json"
    )
    assert len(index["snapshots"]) == 4
    for entry in index["snapshots"]:
        snapshot = ROOT / entry["snapshot_path"]
        assert snapshot.name.endswith(".py.txt")
        assert AUDIT.sha(snapshot) == entry["sha256"]
        assert snapshot.stat().st_size == entry["bytes"]
        original_name = Path(entry["original_repo_path"]).name
        if "measurement" in original_name:
            expected = original_query["protected_measurement_artifact_sha256_before_and_after"][
                entry["original_path"]
            ]
        else:
            key = "audit_tests" if original_name.startswith("test_") else "audit_code"
            expected = original_query["source_files"][key]["sha256"]
        assert expected == entry["sha256"]
    for relative, digest in index["historical_reports"].items():
        assert AUDIT.sha(ROOT / relative) == digest
