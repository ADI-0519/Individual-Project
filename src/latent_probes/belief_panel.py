"""Deterministic fictional belief-state diagnostic, not empirical belief labels.

The 64 prior cases exhaust a three-bit cube within each wording template.
The 16 delivered-current controls are not training/fitting material. Character
visibility is part of the construct: sender messages never reveal the reader's
actual attendance. An omniscient explicit query tests perspective reasoning;
it is not evidence of a spontaneously represented or causally used belief.
"""

from __future__ import annotations

import copy
import itertools
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

PANEL_ID = "belief_diagnostic_development_v1"
TARGETS = ("world", "reader", "nested")
PRIOR_SPLITS = ("train", "validation", "test")
PHASES = ("prior", "delivered_current")

ASSUMPTIONS = (
    "Labels are stipulated fictional belief contents, not measured LLM or human beliefs.",
    "The token occupies exactly one of two places at the end of each scene.",
    "People remember their last observation and update it after an observed move or "
    "a trusted current-location message; they do not infer unobserved moves.",
    "Both people see the initial placement together; the sender observes the later move.",
    "An absent reader does not learn that a move occurred and receives no other update.",
    "The sender cannot observe actual reader attendance and treats the particular "
    "attendance display as accurate; no other attendance information is available.",
    "The display can disagree with actual attendance in the authored narrator's "
    "record, but its possible unreliability is not information available to the sender.",
    "Every authored world changes location, but this is an authoring invariant, "
    "not a public rule known to the reader.",
    "Prior scenes contain no sent or planned message and no unspecified future-message target.",
    "Delivered-current controls contain a truthful current-location message, trusted "
    "receipt by the reader, and confirmation observed by the sender.",
    "The fixed claim is that the token is in place_b; 1 means that claim is true "
    "under the queried perspective, not that a belief is accurate.",
    "Messages stop before perspective questions and before any generated answer; "
    "labels and IDs are metadata, never part of those messages.",
    "Four wording templates train, two validate, and two test; this is wording/template "
    "holdout, not held-out story-family generalization.",
    "Delivered-current rows have split control and must not be used for fitting.",
    "Materials require independent human review before any human-validity claim.",
)

SYSTEM = (
    "Read the fictional scene. Questions will follow. "
    "Each person has only the observations and messages explicitly available to them. "
    "People remember the last place they observed the token and update after seeing "
    "a move or receiving a trusted message about its current place. "
    "They do not infer an unobserved move. Narrator-only information is not shared "
    "with a character merely because the narrator describes it."
)


@dataclass(frozen=True)
class _Frame:
    template_id: str
    split: str
    reader: str
    sender: str
    place_a: str
    place_b: str
    order: tuple[str, ...]
    wording: int


_FRAMES = (
    _Frame(
        "bd_t01",
        "train",
        "Lee",
        "Rowan",
        "amber locker",
        "teal locker",
        ("world", "actual", "record"),
        0,
    ),
    _Frame(
        "bd_t02",
        "train",
        "Mira",
        "Owen",
        "north cabinet",
        "south cabinet",
        ("record", "world", "actual"),
        1,
    ),
    _Frame(
        "bd_t03",
        "train",
        "Niko",
        "Tess",
        "cedar crate",
        "maple crate",
        ("actual", "record", "world"),
        2,
    ),
    _Frame(
        "bd_t04",
        "train",
        "Asha",
        "Bram",
        "copper drawer",
        "silver drawer",
        ("world", "record", "actual"),
        3,
    ),
    _Frame(
        "bd_t05",
        "validation",
        "Dara",
        "Finn",
        "violet chest",
        "ochre chest",
        ("record", "actual", "world"),
        4,
    ),
    _Frame(
        "bd_t06",
        "validation",
        "Emi",
        "Gus",
        "left safe",
        "right safe",
        ("actual", "world", "record"),
        5,
    ),
    _Frame(
        "bd_t07",
        "test",
        "Iris",
        "Jude",
        "round cubby",
        "square cubby",
        ("world", "actual", "record"),
        6,
    ),
    _Frame(
        "bd_t08", "test", "Kira", "Luca", "red bin", "blue bin", ("record", "world", "actual"), 7
    ),
)
_FRAME_BY_ID = {frame.template_id: frame for frame in _FRAMES}
_CONTROL_TEMPLATES = (_FRAMES[0].template_id, _FRAMES[-1].template_id)


def _bit(value, name):
    if type(value) is not int or value not in (0, 1):
        raise ValueError(f"{name} must be an integer bit")
    return value


def _labels(world, observed, recorded, phase):
    if phase == "delivered_current":
        return dict.fromkeys(TARGETS, world)
    return {
        "world": world,
        "reader": world if observed else 1 - world,
        "nested": world if recorded else 1 - world,
    }


def _paragraphs(frame, world, observed, recorded):
    reader, sender = frame.reader, frame.sender
    initial = frame.place_a if world else frame.place_b
    current = frame.place_b if world else frame.place_a
    initials = (
        f"At the start, {reader} and {sender} together watch a token being placed in "
        f"the {initial}.",
        f"Initially, {sender} and {reader} watch the token's placement in the {initial} together.",
        f"The token begins in the {initial}; {reader} watches its placement alongside {sender}.",
        f"Both {sender} and {reader} are watching when the token is first placed in the {initial}.",
        f"At opening time, {reader} watches with {sender} as a token is put in the {initial}.",
        f"{sender} and {reader} jointly observe a token being put in the {initial} "
        "at the beginning.",
        f"The initial placement puts a token in the {initial}, with {reader} and "
        f"{sender} watching together.",
        f"A token is first put in the {initial} while {sender} watches together with {reader}.",
    )
    worlds = (
        f"At noon, a caretaker moves the token from the {initial} into the {current}.",
        f"A caretaker transfers the token from the {initial} to the {current} at noon.",
        f"At noon the token is moved by a caretaker out of the {initial} and into the {current}.",
        f"The noon action is a caretaker moving the token from the {initial} into the {current}.",
        f"At noon, the caretaker takes the token out of the {initial} and puts it "
        f"in the {current}.",
        f"The caretaker's noon move takes the token from the {initial} to the {current}.",
        f"During the noon transfer, a caretaker relocates the token from the {initial} "
        f"to the {current}.",
        f"A caretaker removes the token from the {initial} at noon and places it "
        f"inside the {current}.",
    )
    present = (
        f"{reader} stays and watches the entire noon move.",
        f"{reader} does not leave and sees every part of the noon transfer.",
        f"{reader} is there for the complete move and watches it.",
        f"{reader} remains in view of the token throughout the noon action.",
        f"{reader} witnesses the whole transfer at noon.",
        f"{reader} is not away during the move and watches it from start to finish.",
        f"{reader} watches the caretaker perform the complete noon relocation.",
        f"{reader} stays for the noon action and sees the token's whole journey.",
    )
    absent = (
        f"{reader} is outside throughout the noon move.",
        f"{reader} leaves before the noon transfer and returns only after it is over.",
        f"{reader} is not there for any part of the move.",
        f"{reader} is away throughout the noon action and does not see the token.",
        f"{reader} misses the whole transfer at noon.",
        f"{reader} is away from before the move starts until after it finishes.",
        f"{reader} does not watch any of the caretaker's noon relocation.",
        f"{reader} leaves for the noon action and sees none of the token's journey.",
    )
    recorded_present = (
        "present and watching the whole noon move",
        "not absent and watching the entire noon transfer",
        "there and observing every part of the move",
        "remaining and watching throughout the noon action",
        "a witness to the complete noon transfer",
        "not away and watching the move from start to finish",
        "watching all of the noon relocation",
        "staying and seeing the entire noon action",
    )
    recorded_absent = (
        "outside and not watching any of the noon move",
        "away for the entire noon transfer",
        "not there and not observing any part of the move",
        "away and watching none of the noon action",
        "absent for the complete noon transfer",
        "away from before the move starts until after it finishes",
        "not watching any of the noon relocation",
        "leaving and seeing none of the noon action",
    )
    index = frame.wording
    world_text = (
        worlds[index] + f" {sender} watches the whole move on a camera feed that shows the token "
        f"but does not show {reader} or reveal whether {reader} is watching. "
        "There are no later moves."
    )
    actual_text = (
        present[index]
        if observed
        else (
            absent[index] + f" {reader} receives no update and does not learn that a move occurred."
        )
    )
    actual_text = f"Narrator-only attendance, not available to {sender}: " + actual_text
    status = recorded_present[index] if recorded else recorded_absent[index]
    record_text = (
        f"Attendance information available to {sender}: the display records "
        f"{reader} as {status}. {sender} treats this particular display as accurate "
        f"and has no other information about {reader}'s attendance."
    )
    return initials[index], {"world": world_text, "actual": actual_text, "record": record_text}


def _messages(frame, world, observed, recorded, phase):
    initial, paragraphs = _paragraphs(frame, world, observed, recorded)
    current = frame.place_b if world else frame.place_a
    update = (
        f"After the move, {frame.sender} sends {frame.reader} this message: "
        f'"The token is now in the {current}." '
        f"{frame.reader} receives and trusts this current-location message. "
        f"{frame.sender} sees its delivery confirmation and knows that "
        f"{frame.reader} trusts the message. There are no later events."
    )
    result = {}
    for view in ("omniscient", "sender"):
        parts = [initial]
        parts.extend(
            paragraphs[key] for key in frame.order if view == "omniscient" or key != "actual"
        )
        if phase == "delivered_current":
            parts.append(update)
        else:
            parts.append("No message has been sent or planned. The scene ends here.")
        parts.append(
            "Candidate statement for later assessment, not a new observation or "
            "message to either person: "
            f'"The token is in the {frame.place_b}."'
        )
        parts.append("The scene is complete. Questions follow.")
        result[view] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n\n".join(parts)},
        ]
    return result


def _questions(frame):
    suffix = f"the token is in the {frame.place_b}?"
    return {
        "world": f"At the end of the scene, is the token actually in the {frame.place_b}? "
        "Answer only YES or NO.",
        "reader": f"At the end of the scene, does {frame.reader} believe that {suffix} "
        "Answer only YES or NO.",
        "nested": f"At the end of the scene, does {frame.sender} think that "
        f"{frame.reader} believes that {suffix} Answer only YES or NO.",
    }


def _case(frame, world, observed, recorded, phase):
    identifier = f"{frame.template_id}_w{world}_a{observed}_b{recorded}"
    if phase != "prior":
        identifier += "_" + phase
    return {
        "id": identifier,
        "template_id": frame.template_id,
        "split": frame.split if phase == "prior" else "control",
        "phase": phase,
        "world_bit": world,
        "reader_observed": observed,
        "sender_recorded": recorded,
        "reader_name": frame.reader,
        "sender_name": frame.sender,
        "place_a": frame.place_a,
        "place_b": frame.place_b,
        "claim_place_bit": 1,
        "labels": _labels(world, observed, recorded, phase),
        "messages": _messages(frame, world, observed, recorded, phase),
        "questions": _questions(frame),
    }


def build_panel():
    """Return a fresh, validated 64-prior/16-control deterministic panel."""
    cases = [
        _case(frame, world, observed, recorded, "prior")
        for frame in _FRAMES
        for world, observed, recorded in itertools.product((0, 1), repeat=3)
    ]
    cases.extend(
        _case(_FRAME_BY_ID[template_id], world, observed, recorded, "delivered_current")
        for template_id in _CONTROL_TEMPLATES
        for world, observed, recorded in itertools.product((0, 1), repeat=3)
    )
    return validate_panel(
        {
            "schema": 1,
            "panel_id": PANEL_ID,
            "assumptions": list(ASSUMPTIONS),
            "cases": cases,
        }
    )


def validate_panel(panel):
    """Check exact factual rendering, gold, visibility, grouping, and controls.

    This is a deterministic structural/source check, not human semantic review.
    Return an independent copy; never repair changed data or infer missing labels.
    """
    if not isinstance(panel, Mapping) or set(panel) != {
        "schema",
        "panel_id",
        "assumptions",
        "cases",
    }:
        raise ValueError("Expected the complete belief-panel schema")
    if type(panel["schema"]) is not int or panel["schema"] != 1:
        raise ValueError("Unknown belief-panel schema")
    if panel["panel_id"] != PANEL_ID or panel["assumptions"] != list(ASSUMPTIONS):
        raise ValueError("Changed panel identity or assumptions")
    cases = panel["cases"]
    if not isinstance(cases, list) or len(cases) != 80:
        raise ValueError("Expected exactly 64 prior cases and 16 update controls")
    groups, identifiers = {}, set()
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("Each case must be a mapping")
        template_id = case.get("template_id")
        if template_id not in _FRAME_BY_ID:
            raise ValueError("Unknown wording template")
        phase = case.get("phase")
        if phase not in PHASES:
            raise ValueError("Unknown belief phase")
        world, observed, recorded = (
            _bit(case.get(name), name)
            for name in ("world_bit", "reader_observed", "sender_recorded")
        )
        frame = _FRAME_BY_ID[template_id]
        if phase == "delivered_current" and template_id not in _CONTROL_TEMPLATES:
            raise ValueError("Update controls are restricted to the first and last templates")
        labels = case.get("labels")
        if not isinstance(labels, Mapping) or set(labels) != set(TARGETS):
            raise ValueError("Missing or extra gold target")
        for target in TARGETS:
            _bit(labels[target], f"labels.{target}")
        if type(case.get("claim_place_bit")) is not int or case["claim_place_bit"] != 1:
            raise ValueError("The fixed claim must concern place_b")
        expected = _case(frame, world, observed, recorded, phase)
        if dict(case) != expected:
            raise ValueError(
                "Case differs from its source facts, gold, visibility, or role questions"
            )
        if case["id"] in identifiers:
            raise ValueError("Duplicate case identifier")
        identifiers.add(case["id"])
        groups.setdefault((phase, template_id), []).append(case)

    expected_groups = {("prior", frame.template_id) for frame in _FRAMES} | {
        ("delivered_current", key) for key in _CONTROL_TEMPLATES
    }
    if set(groups) != expected_groups:
        raise ValueError("Missing prior template or update-control group")
    cube = set(itertools.product((0, 1), repeat=3))
    for (phase, _template_id), members in groups.items():
        observed_cube = {
            (case["world_bit"], case["reader_observed"], case["sender_recorded"])
            for case in members
        }
        if len(members) != 8 or observed_cube != cube:
            raise ValueError("Every group must contain all eight source-bit combinations")
        if phase == "prior":
            triples = {tuple(case["labels"][key] for key in TARGETS) for case in members}
            if triples != cube:
                raise ValueError("Prior target triples must exhaust the independent cube")
            for first, second in itertools.combinations(TARGETS, 2):
                counts = Counter(
                    (case["labels"][first], case["labels"][second]) for case in members
                )
                if set(counts.values()) != {2} or len(counts) != 4:
                    raise ValueError("Prior target labels are not pairwise independent")
            for world, recorded in itertools.product((0, 1), repeat=2):
                pair = [
                    case
                    for case in members
                    if case["world_bit"] == world and case["sender_recorded"] == recorded
                ]
                if pair[0]["messages"]["sender"] != pair[1]["messages"]["sender"]:
                    raise ValueError("Actual private attendance leaked into sender messages")

    frames_by_split = {
        split: {frame.template_id for frame in _FRAMES if frame.split == split}
        for split in PRIOR_SPLITS
    }
    if [len(frames_by_split[key]) for key in PRIOR_SPLITS] != [4, 2, 2]:
        raise ValueError("The wording-template split must be 4/2/2")
    names = [name for frame in _FRAMES for name in (frame.reader, frame.sender)]
    places = [place for frame in _FRAMES for place in (frame.place_a, frame.place_b)]
    if len(set(names)) != len(names) or len(set(places)) != len(places):
        raise ValueError("People and place pairs must be distinct between wording templates")
    return copy.deepcopy(dict(panel))


def load_panel(path):
    """Load and validate a frozen JSON panel without generating model outputs."""
    return validate_panel(json.loads(Path(path).read_text(encoding="utf-8")))
