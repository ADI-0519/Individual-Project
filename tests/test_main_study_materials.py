"""Input provenance and allocation checks; all narratives here are test fixtures."""

from __future__ import annotations

import copy
import json
from collections import Counter

import pytest
from test_main_study_bank import synthetic_bank as synthetic_bank

from latent_probes.main_study_bank import bank_digest, iter_rows, prompt_digest
from latent_probes.main_study_materials import (
    assemble_bank,
    build_materials,
    material_manifests,
    read_authored_json,
)
from latent_probes.storage import file_sha256


@pytest.fixture
def fragments(synthetic_bank, tmp_path):
    # The renderer fixture deliberately embeds IDs to expose it as synthetic.
    # Assembly additionally checks that actual visible framings omit those IDs.
    for index, group in enumerate(synthetic_bank["groups"]):
        group["introduction"] = f"Synthetic context {index}: {{sender}} works with {{reader}}."
        group["event_description"] = (
            f"Synthetic evidence setting {index}: one timed session occurs."
        )
    examples = tmp_path / "examples.json"
    examples.write_text(
        json.dumps(
            {
                "groups": [{"group_id": f"excluded_{i}"} for i in range(6)],
                "cases": [
                    {
                        "sender": "Excluded Person",
                        "reader": "Absent Reviewer",
                        "canonical_places": ["excluded chest", "absent trunk"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    paths = []
    for index, splits in enumerate(({"train", "development"}, {"test"})):
        path = tmp_path / f"synthetic_fragment_{index}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "draft_unreviewed",
                    "groups": [g for g in synthetic_bank["groups"] if g["split"] in splits],
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    return paths, examples


def test_order_independent_assembly_has_exact_source_hashes_and_no_review_claim(fragments):
    paths, examples = fragments
    bank = assemble_bank(paths, examples)
    assert bank_digest(bank) == bank_digest(assemble_bank(list(reversed(paths)), examples))
    assert len(bank["groups"]) == 154 and len(list(iter_rows(bank))) == 5888
    assert bank["human_review_complete"] is False and bank["model_observations"] == 0
    assert {r["name"]: r["sha256"] for r in bank["authored_source_files"]} == {
        p.name: file_sha256(p) for p in paths
    }


def test_manifest_pins_every_prompt_and_allocates_only_training_groups(fragments):
    bank = assemble_bank(*fragments)
    splits, allocations = material_manifests(bank)
    rows = {r["case_id"]: r for r in iter_rows(bank)}
    assert splits["split_rows"] == {"train": 1536, "development": 512, "test": 3840}
    assert len(splits["rows"]) == len(rows) == 5888
    assert len(allocations["allocations"]) == 15
    assert json.loads(json.dumps(allocations)) == allocations
    for row in splits["rows"]:
        assert row["prompt_sha256"] == prompt_digest(rows[row["case_id"]]["messages"])
    for allocation in allocations["allocations"]:
        size = allocation["training_n"]
        arms = [allocation[name]["case_ids"] for name in ("restricted", "full")]
        assert all(len(ids) == len(set(ids)) == size for ids in arms)
        for ids in arms:
            assert {rows[i]["split"] for i in ids} == {"train"}
            assert Counter(rows[i]["gold"] for i in ids) == {0: size // 2, 1: size // 2}

        def nuisance(ids):
            return Counter(
                (
                    rows[i]["group_id"],
                    *(rows[i]["state"][k] for k in ("update", "receipt", "world", "name_map")),
                )
                for i in ids
            )

        assert nuisance(arms[0]) == nuisance(arms[1])


@pytest.mark.parametrize("corruption", ["placeholder", "excluded", "metadata", "locations"])
def test_invalid_or_excluded_visible_prose_is_rejected(fragments, corruption):
    paths, examples = fragments
    source = read_authored_json(paths[0])
    group = source["groups"][0]
    if corruption == "placeholder":
        group["introduction"] = "Unreviewed {sender.__class__}"
    elif corruption == "excluded":
        group["introduction"] += " Excluded Person appears."
    elif corruption == "metadata":
        group["introduction"] += f" Identifier {group['group_id']}"
    else:
        group["locations"] = ["red locker", "large red locker"]
    paths[0].write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError):
        assemble_bank(paths, examples)


def test_material_bundle_is_immutable_and_does_not_fill_human_annotations(fragments, tmp_path):
    paths, examples = fragments
    output, review = tmp_path / "material", tmp_path / "review"
    result = build_materials(paths, examples, output, review)
    assert result["groups"] == 154 and result["allocations"] == 15
    assert result["human_review_complete"] is False and result["model_calls"] == 0
    assert all(file_sha256(output / name) == digest for name, digest in result["bindings"].items())
    ledger = read_authored_json(review / "review_ledger.json")
    assert ledger["records"] == []
    original = copy.deepcopy(ledger)
    with pytest.raises(FileExistsError):
        build_materials(paths, examples, output, review)
    assert read_authored_json(review / "review_ledger.json") == original


def test_duplicate_json_fields_are_not_silently_overwritten(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"status":"draft_unreviewed","status":"accepted"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        read_authored_json(path)
