"""Assemble authored drafts and bind all splits/allocations before observations.

This module creates research materials, never activations, fitted heads or human
attestations. Introductory framings share the explicitly declared state renderer.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from string import Formatter

from .main_study_bank import (
    bank_digest,
    export_review,
    iter_rows,
    prompt_digest,
    training_groups,
    validate_bank,
)
from .main_study_design import ALLOCATION_SEEDS, TRAINING_SIZES, matched_training_allocation
from .storage import atomic_json, file_sha256

ROOT = Path(__file__).resolve().parents[2]


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_authored_json(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    if not isinstance(value, dict):
        raise ValueError("Authored source must be a JSON object")
    return value


def _check_draft_group(group: dict) -> None:
    """Reject broken rendering syntax and accidental metadata in visible prose."""
    if not isinstance(group, dict):
        raise ValueError("Each group must be a narrative record")
    for field in ("group_id", "wording_group", "sender", "reader"):
        if not isinstance(group.get(field), str) or not group[field].strip():
            raise ValueError(f"Missing narrative field {field}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", group["group_id"]):
        raise ValueError("Group ID must be a safe journal basename")
    if not isinstance(group.get("locations"), list) or len(group["locations"]) != 2:
        raise ValueError("Need exactly two physical locations")
    entities = (group["sender"], group["reader"], *group["locations"])
    if any(not isinstance(s, str) or not s.strip() or "\n" in s for s in entities):
        raise ValueError("Actor/location names must be nonempty single-line strings")
    if len({s.casefold() for s in entities}) != 4:
        raise ValueError("Actor and physical location names must be distinct")
    first, second = group["locations"]
    if first.casefold() in second.casefold() or second.casefold() in first.casefold():
        raise ValueError("Location names may not contain one another")
    for field in ("introduction", "event_description"):
        text = group.get(field)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Missing narrative prose: {field}")
        try:
            fields = list(Formatter().parse(text))
        except ValueError as exc:
            raise ValueError("Malformed prose formatting") from exc
        if any(
            name not in {None, "sender", "reader"} or spec or conversion
            for _, name, spec, conversion in fields
        ):
            raise ValueError("Prose placeholders may only name sender or reader")
        if group["group_id"] in text or group["wording_group"] in text:
            raise ValueError("Research metadata leaked into narrative prose")


def assemble_bank(fragment_paths: list[str | Path], examples_path: str | Path) -> dict:
    """Combine authored fragments; preserve excluded actors, locations and IDs."""
    examples = read_authored_json(examples_path)
    excluded_groups = {g["group_id"] for g in examples["groups"]}
    if len(excluded_groups) != 6:
        raise ValueError("Expected all six permanently excluded design groups")
    excluded_entities = {
        entity.casefold()
        for row in examples["cases"]
        for entity in (row["sender"], row["reader"], *row["canonical_places"])
    }
    groups, bindings = [], []
    for path in fragment_paths:
        path = Path(path)
        fragment = read_authored_json(path)
        if fragment.get("schema_version") != 1 or fragment.get("status") != "draft_unreviewed":
            raise ValueError("Only explicit unreviewed draft fragments may be assembled")
        if not isinstance(fragment.get("groups"), list) or not fragment["groups"]:
            raise ValueError("Authored fragment has no narratives")
        for group in fragment["groups"]:
            _check_draft_group(group)
            visible = " ".join(
                (
                    group["sender"],
                    group["reader"],
                    *group["locations"],
                    group["introduction"],
                    group["event_description"],
                )
            ).casefold()
            reused = sorted(
                name
                for name in excluded_entities
                if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", visible)
            )
            if reused:
                raise ValueError(f"Excluded design entity reused: {group['group_id']} {reused}")
            groups.append(group)
        bindings.append({"name": path.name, "sha256": file_sha256(path)})
    # Canonical order does not depend on fragment order or filesystem enumeration.
    groups.sort(
        key=lambda g: (
            {"train": 0, "development": 1, "test": 2}[g["split"]],
            g["mechanism"],
            g["group_id"],
        )
    )
    actor_words = {split: set() for split in ("train", "development", "test")}
    for group in groups:
        for name in (group["sender"], group["reader"]):
            actor_words[group["split"]].update(re.split(r"[\s'-]+", name.casefold()))
    split_names = list(actor_words)
    for index, split in enumerate(split_names):
        for other in split_names[index + 1 :]:
            overlap = actor_words[split] & actor_words[other]
            if overlap:
                raise ValueError(f"Actor name components reused across splits: {sorted(overlap)}")
    bank = {
        "schema_version": 1,
        "status": "draft_unreviewed",
        "authorship": "agent-authored contextual framings; awaiting recorded human semantic review",
        "independence": "154 contextual groups sharing three mechanisms and one declared renderer; not 154 reasoning families",
        "human_review_complete": False,
        "model_observations": 0,
        "excluded_group_ids": sorted(excluded_groups),
        "excluded_examples_sha256": file_sha256(examples_path),
        "authored_source_files": sorted(bindings, key=lambda b: b["name"]),
        "groups": groups,
    }
    validate_bank(bank)
    return bank


def material_manifests(bank: dict) -> tuple[dict, dict]:
    """Bind all 5,888 exact prefixes and all 15 matched allocations."""
    validate_bank(bank)
    rows = list(iter_rows(bank))
    identifiers = {row["case_id"] for row in rows}
    split_manifest = {
        "schema": "main-study-split-manifest-v1",
        "bank_sha256": bank_digest(bank),
        "status": "input_materials_only_no_model_observations",
        "human_review_complete": False,
        "model_calls": 0,
        "split_rows": dict(Counter(row["split"] for row in rows)),
        "groups": [
            {key: group[key] for key in ("group_id", "split", "mechanism", "wording_group")}
            for group in bank["groups"]
        ],
        "rows": [
            {
                "case_id": row["case_id"],
                "group_id": row["group_id"],
                "split": row["split"],
                "mechanism": row["mechanism"],
                "prompt_sha256": prompt_digest(row["messages"]),
            }
            for row in rows
        ],
    }
    allocations = []
    for size in TRAINING_SIZES:
        for seed in ALLOCATION_SEEDS:
            allocation = matched_training_allocation(
                training_groups(bank), training_n=size, seed=seed
            )
            record = {
                "training_n": size,
                "allocation_seed": seed,
                "selected_groups": [group.group_id for group in allocation.selected_groups],
            }
            for arm in ("restricted", "full"):
                cases = getattr(allocation, arm)
                ids = [case.case_id for case in cases]
                if len(ids) != size or not set(ids) <= identifiers:
                    raise ValueError(
                        "Allocation selected missing or incorrectly counted authored rows"
                    )
                record[arm] = {
                    "case_ids": ids,
                    "label_counts": {
                        str(label): count for label, count in Counter(c.gold for c in cases).items()
                    },
                }
            allocations.append(record)
    allocation_manifest = {
        "schema": "main-study-matched-allocation-manifest-v1",
        "bank_sha256": bank_digest(bank),
        "source": "input-only mathematical allocation; never uses model scores",
        "training_sizes": list(TRAINING_SIZES),
        "allocation_seeds": list(ALLOCATION_SEEDS),
        "allocations": allocations,
    }
    return split_manifest, allocation_manifest


def build_materials(
    fragment_paths: list[str | Path],
    examples_path: str | Path,
    output_directory: str | Path,
    review_directory: str | Path,
) -> dict:
    """Create a new bank and review pack without overwriting any previous material."""
    output, review = Path(output_directory), Path(review_directory)
    if output.exists() or review.exists():
        raise FileExistsError(
            "Preserve prior bank/review material; choose fresh versioned directories"
        )
    bank = assemble_bank(fragment_paths, examples_path)
    splits, allocations = material_manifests(bank)
    output.mkdir(parents=True)
    atomic_json(output / "main_bank_v1.json", bank)
    atomic_json(output / "split_manifest_v1.json", splits)
    atomic_json(output / "matched_allocations_v1.json", allocations)
    export_review(bank, review)
    summary = {
        "schema": "main-study-materials-build-v1",
        "bank_sha256": bank_digest(bank),
        "groups": len(bank["groups"]),
        "unique_prefixes": len(splits["rows"]),
        "allocations": len(allocations["allocations"]),
        "split_rows": splits["split_rows"],
        "human_review_complete": False,
        "model_calls": 0,
        "bindings": {path.name: file_sha256(path) for path in sorted(output.glob("*.json"))},
        "review_bindings": {path.name: file_sha256(path) for path in sorted(review.iterdir())},
    }
    atomic_json(output / "build_manifest_v1.json", summary)
    return summary


def audit_materials(bank_path: str | Path, *, preflights=()) -> dict:
    """Reconstruct source/manifests and check prepared inputs without weights."""
    from .main_study_execution import DEFAULT_BASELINE_PROTOCOL, expected_baseline_policy
    from .storage import canonical_hash

    bank_path = Path(bank_path)
    bank = read_authored_json(bank_path)
    fragment_paths = [ROOT / "data/belief_main" / b["name"] for b in bank["authored_source_files"]]
    rebuilt = assemble_bank(fragment_paths, ROOT / "data/belief_main/design_examples_v1.json")
    if rebuilt != bank:
        raise ValueError("Authored sources do not reconstruct the recorded bank")
    if read_authored_json(DEFAULT_BASELINE_PROTOCOL) != expected_baseline_policy():
        raise ValueError("Declared baseline protocol differs from the implemented policy")
    splits, allocations = material_manifests(bank)
    for name, expected in (
        ("split_manifest_v1.json", splits),
        ("matched_allocations_v1.json", allocations),
    ):
        if read_authored_json(bank_path.parent / name) != expected:
            raise ValueError(f"Derived material differs from the source bank: {name}")
    build = read_authored_json(bank_path.parent / "build_manifest_v1.json")
    for name, digest in build["bindings"].items():
        if file_sha256(bank_path.parent / name) != digest:
            raise ValueError(f"Built input byte hash differs: {name}")
    expected_rows = {r["case_id"]: r for r in iter_rows(bank)}
    prepared = []
    for protocol_path in preflights:
        protocol_path = Path(protocol_path)
        protocol = read_authored_json(protocol_path)
        if (
            protocol.get("bank_sha256") != bank_digest(bank)
            or protocol.get("test_fixture") is not False
        ):
            raise ValueError("Prepared input is not bound to the real authored bank")
        planned = protocol["planned_rows"]
        expected_ids = set(expected_rows)
        if "monitor_protocol_sha256" in protocol:
            expected_ids = {i for i, r in expected_rows.items() if r["split"] != "train"}
        if len(planned) != len(expected_ids) or {r["case_id"] for r in planned} != expected_ids:
            raise ValueError("Prepared prefix roster differs")
        for row in planned:
            expected = expected_rows[row["case_id"]]
            if row["messages"] != expected["messages"]:
                raise ValueError("Prepared source prose differs")
            ids = row["prompt_token_ids"]
            if not ids or any(type(i) is not int or i < 0 for i in ids):
                raise ValueError("Prepared prefix token IDs must be nonempty integers")
            if canonical_hash(ids) != row["prompt_token_ids_sha256"]:
                raise ValueError("Prepared token ID hash differs")
            if len(ids) > protocol["model_settings"]["max_input_tokens"]:
                raise ValueError("Prepared prefix exceeds fixed context")
        for path, digest in protocol["policy_sha256"].items():
            if file_sha256(path) != digest:
                raise ValueError(f"Prepared capture source changed: {path}")
        preflight = protocol["preflight"]
        if (
            preflight["model_weights_loaded"] is not False
            or preflight.get("model_forwards", 0)
            or preflight.get("model_generations", 0)
        ):
            raise ValueError("A preflight record cannot claim model observations")
        prepared.append(
            {
                "path": str(protocol_path),
                "sha256": file_sha256(protocol_path),
                "model_id": protocol["model_settings"]["model_id"],
                "prefixes": len(planned),
                "minimum_tokens": min(len(r["prompt_token_ids"]) for r in planned),
                "maximum_tokens": max(len(r["prompt_token_ids"]) for r in planned),
            }
        )
    return {
        "schema": "main-study-materials-audit-v1",
        "bank_sha256": bank_digest(bank),
        "source_fragments_reconstructed": len(fragment_paths),
        "groups": len(bank["groups"]),
        "prefixes": len(expected_rows),
        "allocations": len(allocations["allocations"]),
        "human_review_complete": False,
        "main_study_model_calls": 0,
        "prepared_inputs": prepared,
    }
