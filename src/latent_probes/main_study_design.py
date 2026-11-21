"""Pure state rules and matched allocations for the proposed main study.

This module creates mathematical design records, not reviewed narratives or
model observations. Gold is the *content* attributed to a fictional reader by
the fictional sender; it is not whether that belief agrees with world truth.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from typing import Literal

OneStepMechanism = Literal["observation", "message"]
Mechanism = Literal["observation", "message", "correction"]
ONE_STEP_MECHANISMS: tuple[OneStepMechanism, ...] = ("observation", "message")
TRAINING_SIZES = (128, 256, 512)
ALLOCATION_SEEDS = (11, 23, 37, 53, 71)


def _bit(value: object, name: str) -> None:
    if type(value) is not int or value not in (0, 1):
        raise ValueError(f"{name} must be an integer bit")


@dataclass(frozen=True)
class BeliefCase:
    """One unrendered input; the mapping changes names, never symbolic gold.

    ``update`` is U for one-step cases and M1 for correction cases. ``receipt``
    is the sender's attributed access B or B2. Correction M2 is fixed to 1-M1;
    ``first_receipt`` is B1. The current private world is independent of these
    attributed information paths. Actual reader access is not an extra factor.
    """

    group_id: str
    mechanism: Mechanism
    initial: int
    update: int
    receipt: int
    world: int
    name_map: int
    first_receipt: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, str) or not self.group_id.strip():
            raise ValueError("A nonempty scenario group ID is required")
        if self.mechanism not in (*ONE_STEP_MECHANISMS, "correction"):
            raise ValueError("Unknown mechanism")
        for name in ("initial", "update", "receipt", "world", "name_map"):
            _bit(getattr(self, name), name)
        if self.mechanism == "correction":
            _bit(self.first_receipt, "first_receipt")
        elif self.first_receipt is not None:
            raise ValueError("One-step cases have no first_receipt factor")

    @property
    def final_update(self) -> int:
        return 1 - self.update if self.mechanism == "correction" else self.update

    @property
    def prior_final_update(self) -> int:
        if self.mechanism == "correction" and self.first_receipt:
            return self.update
        return self.initial

    @property
    def gold(self) -> int:
        return self.final_update if self.receipt else self.prior_final_update

    @property
    def changes_if_received(self) -> bool:
        return self.prior_final_update != self.final_update

    @property
    def primary_eligible(self) -> bool:
        return self.mechanism != "correction" or self.initial == self.final_update

    @property
    def case_id(self) -> str:
        factors = f"i{self.initial}_u{self.update}_b{self.receipt}"
        if self.mechanism == "correction":
            factors += f"_b1{self.first_receipt}"
        return f"{self.group_id}__{factors}_w{self.world}_map{self.name_map}"


def one_step_states(mechanism: OneStepMechanism, group_id: str) -> tuple[BeliefCase, ...]:
    """All 32 unique I/U/B/W/name-map cells for one authored story group.

    Observation U describes an observed event before the independent later
    private W update. Message U is message content, independently of W.
    Rendering must make those information-access rules explicit.
    """
    if mechanism not in ONE_STEP_MECHANISMS:
        raise ValueError("One-step mechanism must be observation or message")
    return tuple(BeliefCase(group_id, mechanism, *bits) for bits in product((0, 1), repeat=5))


def correction_states(group_id: str) -> tuple[BeliefCase, ...]:
    """All 64 I/M1/B1/B2/W/map cells, including 32 secondary-only cells."""
    return tuple(
        BeliefCase(group_id, "correction", initial, first, b2, world, mapping, b1)
        for initial, first, b1, b2, world, mapping in product((0, 1), repeat=6)
    )


@dataclass(frozen=True)
class BeliefQuartet:
    """A changing pair plus a preserving pair, each ordered no-access/access."""

    changing: tuple[BeliefCase, BeliefCase]
    preserving: tuple[BeliefCase, BeliefCase]

    @property
    def cases(self) -> tuple[BeliefCase, ...]:
        return self.changing + self.preserving

    @property
    def quartet_id(self) -> str:
        case = self.changing[0]
        return f"{case.group_id}__u{case.update}_w{case.world}_map{case.name_map}"


def primary_quartets(cases: tuple[BeliefCase, ...]) -> tuple[BeliefQuartet, ...]:
    """Construct only complete valid blocks; incomplete or duplicate cells fail.

    In corrections I=M2 is necessary: varying B1 then creates preserving and
    changing priors before the final update. I!=M2 cells remain in the input
    bank for secondary analysis and are never invented or relabelled here.
    """
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Duplicate case ID")
    blocks: dict[tuple, list[BeliefCase]] = defaultdict(list)
    for case in cases:
        if case.primary_eligible:
            key = (case.group_id, case.mechanism, case.update, case.world, case.name_map)
            blocks[key].append(case)
    result = []
    for key in sorted(blocks):
        block = blocks[key]
        changing = tuple(
            sorted((c for c in block if c.changes_if_received), key=lambda c: c.receipt)
        )
        preserving = tuple(
            sorted((c for c in block if not c.changes_if_received), key=lambda c: c.receipt)
        )
        if len(changing) != 2 or len(preserving) != 2:
            raise ValueError(f"Incomplete primary quartet {key}")
        if any(tuple(c.receipt for c in pair) != (0, 1) for pair in (changing, preserving)):
            raise ValueError(f"Missing receipt counterpart in quartet {key}")
        if changing[0].gold == changing[1].gold or preserving[0].gold != preserving[1].gold:
            raise ValueError(f"Invalid gold transition in quartet {key}")
        result.append(BeliefQuartet((changing[0], changing[1]), (preserving[0], preserving[1])))
    return tuple(result)


@dataclass(frozen=True)
class TrainingGroup:
    """An authored one-step story, paired with a distinct story at opposite offset."""

    group_id: str
    mechanism: OneStepMechanism
    pair_id: str
    offset: int

    def __post_init__(self) -> None:
        if self.mechanism not in ONE_STEP_MECHANISMS:
            raise ValueError("Training contains one-step mechanisms only")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.group_id, self.pair_id)
        ):
            raise ValueError("Training group and pair IDs must be nonempty")
        _bit(self.offset, "offset")


def default_training_groups() -> tuple[TrainingGroup, ...]:
    """Reserve 48 IDs; these are not 48 independently authored narratives yet."""
    return tuple(
        TrainingGroup(
            f"train_{mechanism}_{pair:02d}{suffix}",
            mechanism,
            f"train_{mechanism}_pair_{pair:02d}",
            offset,
        )
        for mechanism in ONE_STEP_MECHANISMS
        for pair in range(12)
        for offset, suffix in enumerate(("a", "b"))
    )


@dataclass(frozen=True)
class MatchedAllocation:
    seed: int
    training_n: int
    selected_groups: tuple[TrainingGroup, ...]
    restricted: tuple[BeliefCase, ...]
    full: tuple[BeliefCase, ...]


def _rank(*parts: object) -> bytes:
    return sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()


def matched_training_allocation(
    groups: tuple[TrainingGroup, ...], *, training_n: int, seed: int
) -> MatchedAllocation:
    """Select nested whole pairs and exactly 16 unique cells/group/arm.

    For each pair/seed/U, SHA ranking marks two of the four W/map cells changing
    and two preserving. The offset-1 story takes the complementary mask. Each
    story therefore has eight changing and eight preserving rows, with B pairs
    intact. Pooling its two distinct stories covers the complete I/U/B/W/map
    factorial once. Finite within-story correlations still exist; global
    balance does not prove their absence. The five allocations vary both the
    selected stories and these balanced masks before model outcomes exist.

    This function neither captures activations nor chooses fitted settings.
    """
    if type(training_n) is not int or training_n not in TRAINING_SIZES:
        raise ValueError(f"training_n must be one of {TRAINING_SIZES}")
    if type(seed) is not int or seed not in ALLOCATION_SEEDS:
        raise ValueError(f"seed must be one of {ALLOCATION_SEEDS}")
    if len(groups) != 48 or len({group.group_id for group in groups}) != 48:
        raise ValueError("Need 48 distinct authored training-group IDs")
    pairs: dict[str, list[TrainingGroup]] = defaultdict(list)
    for group in groups:
        pairs[group.pair_id].append(group)
    if len(pairs) != 24:
        raise ValueError("Need 24 distinct whole story pairs")
    for pair in pairs.values():
        if len(pair) != 2 or {group.offset for group in pair} != {0, 1}:
            raise ValueError("Each story pair needs two distinct groups at offsets 0/1")
        if pair[0].mechanism != pair[1].mechanism:
            raise ValueError("A story pair cannot mix mechanisms")
    selected = []
    for mechanism in ONE_STEP_MECHANISMS:
        candidates = [key for key, pair in pairs.items() if pair[0].mechanism == mechanism]
        if len(candidates) != 12:
            raise ValueError("Need 12 story pairs per mechanism")
        candidates.sort(key=lambda key: (_rank("pair-selection-v1", seed, mechanism, key), key))
        for key in candidates[: training_n // 64]:
            selected.extend(sorted(pairs[key], key=lambda group: group.offset))
    restricted = []
    full = []
    for group in selected:
        change_cells = {}
        for update in (0, 1):
            cells = sorted(
                product((0, 1), repeat=2),
                key=lambda cell: (
                    _rank("coverage-mask-v1", seed, group.pair_id, update, *cell),
                    cell,
                ),
            )
            change_cells[update] = set(cells[:2])
        for case in one_step_states(group.mechanism, group.group_id):
            if case.initial != case.update:
                restricted.append(case)
            change = int((case.world, case.name_map) in change_cells[case.update]) ^ group.offset
            if case.initial == (case.update ^ change):
                full.append(case)
    if len(restricted) != training_n or len(full) != training_n:
        raise ValueError("Matched allocation has incorrect training size")
    return MatchedAllocation(seed, training_n, tuple(selected), tuple(restricted), tuple(full))
