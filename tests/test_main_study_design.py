"""Scientific-design invariants, independent of any model or fitted probe."""

from collections import Counter, defaultdict
from dataclasses import replace
from itertools import product

import pytest

from latent_probes.main_study_design import (
    ALLOCATION_SEEDS,
    TRAINING_SIZES,
    BeliefCase,
    TrainingGroup,
    correction_states,
    default_training_groups,
    matched_training_allocation,
    one_step_states,
    primary_quartets,
)


@pytest.mark.parametrize("mechanism", ["observation", "message"])
def test_one_step_access_changes_only_attributed_information(mechanism):
    # Initial left; a right update is observed/reported; private world is left.
    no_access = BeliefCase("example", mechanism, 0, 1, 0, 0, 0)
    access = replace(no_access, receipt=1)
    assert (no_access.gold, access.gold) == (0, 1)
    # Current truth and physical names cannot alter the attributed symbolic content.
    for case in (no_access, access):
        assert replace(case, world=1, name_map=1).gold == case.gold
    # Repeating initial left evidence leaves the belief left regardless of access.
    redundant = replace(no_access, update=0)
    assert (redundant.gold, replace(redundant, receipt=1).gold) == (0, 0)


def test_correction_composes_both_receipts_and_latest_access_overrides():
    # Initial left, first message right, correction left; late private world right.
    cases = [
        BeliefCase("correction", "correction", 0, 1, b2, 1, 0, b1)
        for b1, b2 in product((0, 1), repeat=2)
    ]
    assert [case.gold for case in cases] == [0, 0, 1, 0]
    assert [case.prior_final_update for case in cases] == [0, 0, 1, 1]
    assert [case.changes_if_received for case in cases] == [False, False, True, True]
    # The secondary path is not deleted: initial already equals message one.
    secondary = [replace(case, initial=1) for case in cases]
    assert [case.gold for case in secondary] == [1, 0, 1, 0]
    assert not any(case.primary_eligible for case in secondary)


@pytest.mark.parametrize(
    "mechanism,count", [("observation", 32), ("message", 32), ("correction", 64)]
)
def test_full_factorial_banks_and_valid_primary_quartets(mechanism, count):
    cases = (
        correction_states("group")
        if mechanism == "correction"
        else one_step_states(mechanism, "group")
    )
    assert len(cases) == len({case.case_id for case in cases}) == count
    assert Counter(case.gold for case in cases) == {0: count // 2, 1: count // 2}
    quartets = primary_quartets(cases)
    assert len(quartets) == len({quartet.quartet_id for quartet in quartets}) == 8
    primary_ids = {case.case_id for quartet in quartets for case in quartet.cases}
    assert len(primary_ids) == 32
    assert primary_ids == {case.case_id for case in cases if case.primary_eligible}
    assert len(set(case.case_id for case in cases) - primary_ids) == count - 32
    for quartet in quartets:
        assert {case.receipt for case in quartet.changing} == {0, 1}
        assert {case.receipt for case in quartet.preserving} == {0, 1}
        assert quartet.changing[0].gold != quartet.changing[1].gold
        assert quartet.preserving[0].gold == quartet.preserving[1].gold
        assert len({(case.world, case.name_map, case.update) for case in quartet.cases}) == 1


@pytest.mark.parametrize("mechanism", ["observation", "message", "correction"])
def test_complete_quartets_defeat_the_three_quarters_shortcut(mechanism):
    cases = (
        correction_states("group")
        if mechanism == "correction"
        else one_step_states(mechanism, "group")
    )
    for quartet in primary_quartets(cases):
        shortcuts = (
            lambda c: 0,
            lambda c: 1,
            lambda c: c.initial,
            lambda c: c.final_update,
            lambda c: c.world,
            lambda c: c.receipt,
        )
        for shortcut in shortcuts:
            assert not all(shortcut(case) == case.gold for case in quartet.cases)
        # Pretend every final update differs from the prior: 75% row accuracy,
        # but zero complete quartets. For corrections this ignores B1 entirely.
        assert (
            sum(
                (case.final_update if case.receipt else 1 - case.final_update) == case.gold
                for case in quartet.cases
            )
            == 3
        )


def test_incomplete_or_duplicated_primary_counterfactuals_fail():
    cases = one_step_states("message", "group")
    with pytest.raises(ValueError, match="Incomplete"):
        primary_quartets(cases[1:])
    with pytest.raises(ValueError, match="Duplicate"):
        primary_quartets(cases + cases[:1])


def _cells(cases, fields):
    return Counter(tuple(getattr(case, field) for field in fields) for case in cases)


@pytest.mark.parametrize("seed", ALLOCATION_SEEDS)
@pytest.mark.parametrize("training_n", TRAINING_SIZES)
def test_all_allocations_exactly_match_budget_groups_and_nontreatment_factors(seed, training_n):
    allocation = matched_training_allocation(
        default_training_groups(), training_n=training_n, seed=seed
    )
    arms = (allocation.restricted, allocation.full)
    assert all(len(arm) == len({case.case_id for case in arm}) == training_n for arm in arms)
    assert len(allocation.selected_groups) == training_n // 16
    assert Counter(group.mechanism for group in allocation.selected_groups) == {
        "observation": training_n // 32,
        "message": training_n // 32,
    }
    # Same groups and entire U/B/W/map joint distribution in each story.
    joint = ("group_id", "update", "receipt", "world", "name_map")
    assert _cells(arms[0], joint) == _cells(arms[1], joint)
    for arm in arms:
        assert Counter(case.gold for case in arm) == {0: training_n // 2, 1: training_n // 2}
        assert Counter(case.initial for case in arm) == {0: training_n // 2, 1: training_n // 2}
        assert _cells(arm, ("mechanism", "world", "gold")) == {
            (mechanism, world, gold): training_n // 8
            for mechanism, world, gold in product(("observation", "message"), (0, 1), (0, 1))
        }
        for group in allocation.selected_groups:
            rows = [case for case in arm if case.group_id == group.group_id]
            assert len(rows) == 16
            assert Counter(case.gold for case in rows) == {0: 8, 1: 8}
            assert Counter(case.initial for case in rows) == {0: 8, 1: 8}
            # Both access endpoints are retained for every selected initial/update.
            assert set(_cells(rows, ("initial", "update", "world", "name_map")).values()) == {2}
    assert all(case.initial != case.update for case in allocation.restricted)
    for group in allocation.selected_groups:
        assert Counter(
            case.initial == case.update
            for case in allocation.full
            if case.group_id == group.group_id
        ) == {False: 8, True: 8}


@pytest.mark.parametrize("seed", ALLOCATION_SEEDS)
def test_antithetic_story_partners_cover_every_full_factorial_cell(seed):
    allocation = matched_training_allocation(default_training_groups(), training_n=512, seed=seed)
    rows_by_pair = defaultdict(list)
    groups = {group.group_id: group for group in allocation.selected_groups}
    for case in allocation.full:
        rows_by_pair[groups[case.group_id].pair_id].append(case)
    expected = Counter(product((0, 1), repeat=5))
    for rows in rows_by_pair.values():
        assert _cells(rows, ("initial", "update", "receipt", "world", "name_map")) == expected


def test_seeded_draws_are_distinct_nested_and_independent_of_input_order():
    groups = default_training_groups()
    for training_n in TRAINING_SIZES:
        selected_sets = []
        for seed in ALLOCATION_SEEDS:
            allocation = matched_training_allocation(groups, training_n=training_n, seed=seed)
            assert allocation == matched_training_allocation(
                tuple(reversed(groups)), training_n=training_n, seed=seed
            )
            selected_sets.append(frozenset(group.group_id for group in allocation.selected_groups))
        assert len(set(selected_sets)) == len(ALLOCATION_SEEDS)
    for seed in ALLOCATION_SEEDS:
        allocations = [
            matched_training_allocation(groups, training_n=n, seed=seed) for n in TRAINING_SIZES
        ]
        for lower, higher in zip(allocations, allocations[1:]):
            assert set(lower.restricted) < set(higher.restricted)
            assert set(lower.full) < set(higher.full)


def test_invalid_bits_and_malformed_training_partitions_fail():
    with pytest.raises(ValueError, match="integer bit"):
        BeliefCase("group", "message", True, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="first_receipt"):
        BeliefCase("group", "correction", 0, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="no first_receipt"):
        BeliefCase("group", "message", 0, 0, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="one-step"):
        TrainingGroup("group", "correction", "pair", 0)
    groups = default_training_groups()
    for invalid in (groups[:-1], groups[:-1] + groups[:1]):
        with pytest.raises(ValueError, match="48 distinct"):
            matched_training_allocation(invalid, training_n=128, seed=11)
    with pytest.raises(ValueError, match="offsets"):
        matched_training_allocation(
            (replace(groups[0], offset=1),) + groups[1:], training_n=128, seed=11
        )
    with pytest.raises(ValueError, match="training_n"):
        matched_training_allocation(groups, training_n=64, seed=11)
    with pytest.raises(ValueError, match="seed"):
        matched_training_allocation(groups, training_n=128, seed=99)
