"""Draft renderer checks on synthetic fixtures, not authored research material.

The repeated mechanical fixture prose intentionally supplies no claim of actual
narrative independence or human review. No fixture bank is saved as study data.
"""

from __future__ import annotations

import copy
import csv
import json
from collections import Counter

import pytest

from latent_probes.main_study_bank import (
    ROSTER,
    bank_digest,
    case_from_row,
    export_review,
    iter_rows,
    load_bank,
    training_groups,
    validate_bank,
    verify_review_ledger,
)
from latent_probes.main_study_design import default_training_groups


@pytest.fixture
def synthetic_bank():
    """Generate only test records; these are not 154 authored story skeletons."""
    groups = []
    for (split, mechanism), count in ROSTER.items():
        roster = [g for g in default_training_groups() if g.mechanism == mechanism]
        for index in range(count):
            group_id = (
                roster[index].group_id if split == "train" else f"{split}_{mechanism}_{index:02d}"
            )
            entity_prefix = f"SYNTHETIC_{split}_{mechanism}_{index:02d}"
            group = {
                "group_id": group_id,
                "split": split,
                "mechanism": mechanism,
                "wording_group": f"SYNTHETIC_WORDING_{group_id}",
                "sender": f"{entity_prefix}_sender",
                "reader": f"{entity_prefix}_reader",
                "locations": [f"{entity_prefix}_ALPHA", f"{entity_prefix}_BETA"],
                "introduction": f"Synthetic fixture {group_id}: {{sender}} works with {{reader}}.",
                "event_description": f"Synthetic fixture event {group_id}: a timed evidence session occurs.",
            }
            if split == "train":
                group.update(pair_id=roster[index].pair_id, offset=roster[index].offset)
            groups.append(group)
    return {
        "schema_version": 1,
        "status": "draft_unreviewed",
        "groups": groups,
        "excluded_group_ids": [f"design_group_{index}" for index in range(6)],
    }


def test_full_synthetic_roster_split_sizes_unique_cases_and_state_bindings(synthetic_bank):
    result = validate_bank(synthetic_bank)
    rows = list(iter_rows(synthetic_bank))
    assert result["groups"] == 154 and result["prefixes"] == 5888
    assert result["human_review_complete"] is False
    assert Counter(row["split"] for row in rows) == {
        "train": 1536,
        "development": 512,
        "test": 3840,
    }
    assert len({row["case_id"] for row in rows}) == 5888
    assert set(training_groups(synthetic_bank)) == set(default_training_groups())
    group_counts = Counter(row["group_id"] for row in rows)
    for group in synthetic_bank["groups"]:
        assert group_counts[group["group_id"]] == (64 if group["mechanism"] == "correction" else 32)
    for row in rows:
        case = case_from_row(row)
        prior = (
            case.update if case.mechanism == "correction" and case.first_receipt else case.initial
        )
        final = 1 - case.update if case.mechanism == "correction" else case.update
        assert row["gold"] == (final if case.receipt else prior)
    assert len(list(iter_rows(synthetic_bank, "train"))) == 1536
    with pytest.raises(ValueError, match="Unknown split"):
        list(iter_rows(synthetic_bank, "unregistered"))


def test_messages_keep_gold_metadata_out_and_query_tracks_global_name_exchange(synthetic_bank):
    groups = {g["group_id"]: g for g in synthetic_bank["groups"]}
    for row in iter_rows(synthetic_bank):
        assert [message["role"] for message in row["messages"]] == ["system", "user"]
        assert all(set(message) == {"role", "content"} for message in row["messages"])
        content = row["messages"][1]["content"]
        case, group = case_from_row(row), groups[row["group_id"]]
        target = group["locations"][1 - case.name_map]
        assert content.endswith(
            f"At the end of the scene, does {group['sender']} think that {group['reader']} "
            f"believes that the token is in the {target}? Answer only YES or NO."
        )
        assert '"gold"' not in content and '"state"' not in content
        assert '"name_map"' not in content and "SYNTHETIC_WORDING" not in content
        assert "09:05" in content and "09:15" in content and "09:20" in content
        if case.mechanism == "observation":
            update = group["locations"][case.update ^ case.name_map]
            assert f"At 09:05 the token is actually in the {update}." in content
        if case.mechanism == "correction":
            assert "09:10" in content and "09:12" in content
            assert "does not merely withdraw" in content


def test_bank_digest_roundtrip_and_roster_and_row_corruption_are_rejected(synthetic_bank, tmp_path):
    path = tmp_path / "synthetic_bank.json"
    path.write_text(json.dumps(synthetic_bank, indent=2), encoding="utf-8")
    assert bank_digest(load_bank(path)) == bank_digest(synthetic_bank)
    changed = copy.deepcopy(synthetic_bank)
    changed["groups"][0]["introduction"] += " Changed synthetic prose."
    assert bank_digest(changed) != bank_digest(synthetic_bank)
    changed = copy.deepcopy(synthetic_bank)
    changed["groups"][0]["mechanism"] = "correction"
    with pytest.raises(ValueError, match="roster"):
        validate_bank(changed)
    row = next(iter_rows(synthetic_bank))
    row["gold"] = 1 - row["gold"]
    with pytest.raises(ValueError, match="binding mismatch"):
        case_from_row(row)


def test_stale_and_missing_actual_human_review_fail_closed(synthetic_bank, tmp_path):
    ledger = tmp_path / "review_ledger.json"
    ledger.write_text(
        json.dumps(
            {"schema_version": 1, "bank_sha256": "0" * 64, "records": [], "disputed_groups": []}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exact canonical bank hash"):
        verify_review_ledger(synthetic_bank, ledger)
    ledger.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bank_sha256": bank_digest(synthetic_bank),
                "records": [],
                "disputed_groups": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="student acceptance missing for 154 groups"):
        verify_review_ledger(synthetic_bank, ledger)


def test_review_exports_are_blinded_blank_and_cannot_overwrite_annotations(
    synthetic_bank, tmp_path
):
    export_review(synthetic_bank, tmp_path)
    prompts = [
        json.loads(line)
        for line in (tmp_path / "review_prompts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(prompts) == 736  # 124 one-step groups x4 +30 correction groups x8.
    assert all(set(row) == {"case_id", "group_id", "prompt_sha256", "messages"} for row in prompts)
    assert all("gold" not in row and "state" not in row for row in prompts)
    path = tmp_path / "annotations.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns, records = reader.fieldnames, list(reader)
    assert len(records) == 154
    fields = ("reviewer", "reviewed_at", "decision", "notes", "prompt_sha256s")
    assert all(not row[field] for row in records for field in fields)
    ledger = json.loads((tmp_path / "review_ledger.json").read_text(encoding="utf-8"))
    assert ledger["records"] == [] and ledger["bank_sha256"] == bank_digest(synthetic_bank)
    records[0]["notes"] = "Synthetic test annotation: must not be overwritten."
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    with pytest.raises(ValueError, match="overwrite human annotations"):
        export_review(synthetic_bank, tmp_path)
