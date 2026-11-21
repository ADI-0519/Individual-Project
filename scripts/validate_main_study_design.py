"""Export mathematical design checks without asserting narrative or human validity."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from latent_probes.main_study_design import (
    ALLOCATION_SEEDS,
    TRAINING_SIZES,
    BeliefCase,
    correction_states,
    default_training_groups,
    matched_training_allocation,
    one_step_states,
    primary_quartets,
)
from latent_probes.storage import atomic_json, read_json

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = Path("data/belief_main/design_examples_v1.json")
POLICY = Path("docs/MAIN_STUDY_SPECIFICATION_V1.md")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def id_hash(cases):
    """UTF-8 SHA256 of sorted IDs separated by LF, without a trailing LF."""
    return sha256("\n".join(sorted(case.case_id for case in cases)).encode("utf-8")).hexdigest()


def cells(cases, fields):
    return Counter(tuple(getattr(case, field) for field in fields) for case in cases)


def support_record(cases):
    quartets = primary_quartets(cases)
    primary = sum(case.primary_eligible for case in cases)
    return {
        "rows": len(cases),
        "unique_case_ids": len({case.case_id for case in cases}),
        "gold_0": sum(case.gold == 0 for case in cases),
        "gold_1": sum(case.gold == 1 for case in cases),
        "primary_quartets": len(quartets),
        "primary_rows": primary,
        "secondary_rows": len(cases) - primary,
    }


def example_audit():
    bank = read_json(ROOT / EXAMPLES)
    require(bank["exclusion"]["human_review_complete"] is False, "Review status changed")
    require(
        bank["exclusion"]["model_calls"] == bank["exclusion"]["fit_calls"] == 0, "Examples executed"
    )
    records = []
    for row in bank["cases"]:
        case = BeliefCase(row["group_id"], row["mechanism"], **row["state"])
        expected = row["gold"]
        require(case.gold == expected["attributed_belief"], f"Gold mismatch {row['id']}")
        require(
            case.prior_final_update == expected["prior_final_update"], f"Prior mismatch {row['id']}"
        )
        require(
            case.final_update == expected["final_update_location"], f"Final mismatch {row['id']}"
        )
        require(
            case.changes_if_received == expected["final_receipt_informative"],
            f"Change mismatch {row['id']}",
        )
        require(
            expected["answer"] == ("YES" if case.gold else "NO"), f"Answer mismatch {row['id']}"
        )
        require(row["review"]["human_review_complete"] is False, "Human status changed")
        require(row["review"]["exclude_all_future_partitions"] is True, "Design scene not excluded")
        records.append(case)
    require(len(records) == 24, "Expected 24 representative design records")
    require(len({case.group_id for case in records}) == 6, "Expected six design groups")
    require(
        Counter(case.mechanism for case in records)
        == {"observation": 8, "message": 8, "correction": 8},
        "Mechanism coverage changed",
    )
    require(len(primary_quartets(tuple(records))) == 6, "Expected six complete example quartets")
    return {
        "cases_checked": 24,
        "groups": 6,
        "complete_quartets": 6,
        "gold_prior_final_informativeness_answer_checks": "pass",
        "human_semantic_validation": False,
        "excluded_from_all_future_partitions": True,
    }


def allocation_record(allocation):
    restricted, full = allocation.restricted, allocation.full
    n = allocation.training_n
    require(len(restricted) == len(full) == n, "Wrong training N")
    require(
        len({case.case_id for case in restricted}) == len({case.case_id for case in full}) == n,
        "Duplicate training input IDs",
    )
    joint = ("group_id", "update", "receipt", "world", "name_map")
    require(cells(restricted, joint) == cells(full, joint), "Unmatched story/U/B/W/map joint")
    require(
        all(case.initial != case.update for case in restricted),
        "Restricted arm contains redundant evidence",
    )
    masks = {}
    for group in allocation.selected_groups:
        rows = [case for case in full if case.group_id == group.group_id]
        changing = sorted(
            {
                (case.update, case.world, case.name_map)
                for case in rows
                if case.initial != case.update
            }
        )
        require(len(rows) == 16 and len(changing) == 4, "Wrong full-arm group mask")
        require(Counter(cell[0] for cell in changing) == {0: 2, 1: 2}, "Unbalanced U mask")
        masks[group.group_id] = changing
    arm_support = {}
    for name, rows in (("restricted", restricted), ("full", full)):
        require(
            Counter(case.gold for case in rows) == {0: n // 2, 1: n // 2},
            "Unbalanced training labels",
        )
        require(
            Counter(case.initial for case in rows) == {0: n // 2, 1: n // 2},
            "Unbalanced initial labels",
        )
        require(
            all(
                count == 2
                for count in cells(
                    rows, ("group_id", "initial", "update", "world", "name_map")
                ).values()
            ),
            "Broken receipt pair",
        )
        arm_support[name] = {
            "rows": len(rows),
            "unique_case_ids": len({case.case_id for case in rows}),
            "gold_0": n // 2,
            "gold_1": n // 2,
            "sorted_case_ids_utf8_lf_no_trailing_lf_sha256": id_hash(rows),
        }
    pairs = defaultdict(list)
    by_group = {group.group_id: group for group in allocation.selected_groups}
    for case in full:
        pairs[by_group[case.group_id].pair_id].append(case)
    for rows in pairs.values():
        require(
            len(cells(rows, ("initial", "update", "receipt", "world", "name_map"))) == 32,
            "Incomplete antithetic pair",
        )
    return {
        "seed": allocation.seed,
        "training_n": n,
        "selected_group_count": len(allocation.selected_groups),
        "selected_groups": [asdict(group) for group in allocation.selected_groups],
        "full_changing_cells_U_W_map_by_group": masks,
        "arms": arm_support,
        "exact_group_U_B_W_map_joint_match": True,
        "all_receipt_pairs_intact": True,
        "antithetic_partner_factorial_coverage": True,
    }


def validate():
    groups = default_training_groups()
    allocations = [
        matched_training_allocation(groups, training_n=n, seed=seed)
        for seed in ALLOCATION_SEEDS
        for n in TRAINING_SIZES
    ]
    for seed in ALLOCATION_SEEDS:
        draws = [allocation for allocation in allocations if allocation.seed == seed]
        for smaller, larger in zip(draws, draws[1:]):
            require(
                set(smaller.restricted) < set(larger.restricted),
                "Restricted learning curve not nested",
            )
            require(set(smaller.full) < set(larger.full), "Full learning curve not nested")
    for n in TRAINING_SIZES:
        selected = {
            tuple(sorted(group.group_id for group in allocation.selected_groups))
            for allocation in allocations
            if allocation.training_n == n
        }
        require(
            len(selected) == len(ALLOCATION_SEEDS),
            "Allocation draws repeat identical group subsets",
        )
    factorial = {
        "observation": support_record(one_step_states("observation", "support_observation")),
        "message": support_record(one_step_states("message", "support_message")),
        "correction": support_record(correction_states("support_correction")),
    }
    for mechanism in ("observation", "message"):
        require(
            factorial[mechanism]
            == {
                "rows": 32,
                "unique_case_ids": 32,
                "gold_0": 16,
                "gold_1": 16,
                "primary_quartets": 8,
                "primary_rows": 32,
                "secondary_rows": 0,
            },
            f"One-step support changed: {mechanism}",
        )
    require(
        factorial["correction"]["rows"] == 64
        and factorial["correction"]["primary_rows"]
        == factorial["correction"]["secondary_rows"]
        == 32,
        "Correction support changed",
    )
    bindings = {}
    for path in (
        Path("src/latent_probes/main_study_design.py"),
        Path("scripts/validate_main_study_design.py"),
        EXAMPLES,
        POLICY,
    ):
        # Git's text policy uses LF. Bind semantic source bytes consistently
        # before/after checkout, explicitly rather than pretending these are
        # immutable raw-run files. This remains an unexecuted design proposal.
        encoded = (ROOT / path).read_bytes().decode("utf-8").replace("\r\n", "\n").encode("utf-8")
        bindings[path.as_posix()] = sha256(encoded).hexdigest()
    return {
        "schema": "main_study_design_cpu_validation_v2",
        "status": "design_invariants_pass_narrative_review_not_assessed",
        "model_calls": 0,
        "fit_calls": 0,
        "human_review_complete": False,
        "policies_declared_not_frozen": True,
        "group_ids_define_allocation_roster": True,
        "authored_materials_audited_by_this_script": False,
        "binding_algorithm": "SHA256 of UTF-8 text normalized to LF; no other transformations",
        "file_hashes_utf8_lf": bindings,
        "training_sizes": list(TRAINING_SIZES),
        "allocation_seeds": list(ALLOCATION_SEEDS),
        "pair_selection": "SHA256 UTF-8 pipe-joined(pair-selection-v1,seed,mechanism,pair_id), ascending digest then pair_id; nested prefixes",
        "full_mask": "For each(seed,pair_id,U), SHA256 UTF-8 pipe-joined(coverage-mask-v1,seed,pair_id,U,W,map), ascending digest then(W,map); first2of4 changing atoffset0, complementary atoffset1",
        "case_id_hash_contract": "SHA256 of lexicographically sorted case IDs joined by LF with no trailing LF",
        "reserved_training_groups": [asdict(group) for group in groups],
        "factorial_support_per_group": factorial,
        "design_example_audit": example_audit(),
        "allocations": [allocation_record(allocation) for allocation in allocations],
        "all_fifteen_allocations_checked": True,
        "nested_training_sizes": True,
        "distinct_group_subsets_across_five_seeds_per_N": True,
        "interpretation_limit": "Mathematical design checks do not validate narrative semantics, model capabilities, novelty or research outcomes. Finite within-story correlations remain despite pooled balance.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        not args.output.resolve().is_relative_to((ROOT / "runs").resolve()),
        "Output must stay outside raw runs",
    )
    record = validate()
    atomic_json(args.output, record)
    print({"status": record["status"], "allocations": 15, "examples": 24, "model_calls": 0})


if __name__ == "__main__":
    main()
