"""Fresh authored perspective-query validation material, with no model outputs.

This is the same finite logical family as the development belief task, with
four new wording groups. Gold describes stipulated fictional belief content.
Nothing here establishes independent human validity or story-family OOD.
"""

from __future__ import annotations

import copy
import itertools
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

PANEL_ID = "belief_query_fresh_validation_v1"
TARGETS = ("world", "reader", "nested")
SYSTEM = (
    "Read the fictional scene. Questions will follow. "
    "Each person has only the observations and messages explicitly available to them. "
    "People remember the last place they observed the token and update after seeing "
    "a move or receiving a trusted message about its current place. "
    "They do not infer an unobserved move. Narrator-only information is not shared "
    "with a character merely because the narrator describes it."
)
ASSUMPTIONS = (
    "Author-assisted draft; annotation_origin=agent_assisted_draft; "
    "needs_human_review=true. This material has not been independently human validated.",
    "Thirty-two fresh prior cases exhaust eight world/actual-attendance/recorded-attendance "
    "combinations in each of four new wording groups.",
    "This is the same logical family with new wording, names, and places; it is not "
    "independent story-family OOD evidence.",
    "Both people observe the initial placement together, and the sender observes "
    "the entire later transfer through a view that cannot reveal reader attendance.",
    "The reader retains the last observed location unless they see the transfer "
    "or receive a trusted current-location report.",
    "An absent reader sees none of the transfer, receives no update, and does not "
    "learn that a transfer occurred.",
    "The sender accepts the particular attendance record as accurate and has no "
    "other reader-attendance information; the actual private attendance is hidden.",
    "The attendance record explicitly says the reader watched the entire transfer "
    "or did not watch any of it, and records no later update.",
    "Actual attendance and the attendance record vary independently in authoring; "
    "the record's possible error is not disclosed to the sender.",
    "Initial and final locations differ in each authored case. This is an authoring "
    "invariant, not a rule known to the reader.",
    "Labels mean the fixed place_b proposition under world, reader-belief, or "
    "sender-attributed-reader-belief perspective, not belief accuracy.",
    "The base messages contain the scene and a candidate proposition but no perspective "
    "question, generated answer, gold bit, planned message, or unknown-future target.",
    "All cases have split fresh and phase prior; the before/after comparison refers "
    "to adding a question, not to any change in the characters' evidence.",
    "Authoring rules and review metadata are not model-visible scene instructions.",
)


@dataclass(frozen=True)
class _Frame:
    template_id: str
    reader: str
    sender: str
    place_a: str
    place_b: str
    order: tuple[str, ...]
    wording: int


_FRAMES = (
    _Frame(
        "bq_t01",
        "Pavel",
        "Zara",
        "ivory cupboard",
        "indigo cupboard",
        ("actual", "record", "world"),
        0,
    ),
    _Frame(
        "bq_t02", "Uma", "Vera", "linen trunk", "wicker trunk", ("record", "world", "actual"), 1
    ),
    _Frame(
        "bq_t03",
        "Silas",
        "Wren",
        "granite alcove",
        "marble alcove",
        ("world", "actual", "record"),
        2,
    ),
    _Frame("bq_t04", "Yuna", "Ezra", "east vault", "west vault", ("world", "record", "actual"), 3),
)
_BY_TEMPLATE = {frame.template_id: frame for frame in _FRAMES}


def _bit(value, name):
    if type(value) is not int or value not in (0, 1):
        raise ValueError(f"{name} must be an integer bit")
    return value


def _gold(world, observed, recorded):
    return {
        "world": world,
        "reader": world if observed else 1 - world,
        "nested": world if recorded else 1 - world,
    }


def _scene_parts(frame, world, observed, recorded):
    reader, sender = frame.reader, frame.sender
    start = frame.place_a if world else frame.place_b
    finish = frame.place_b if world else frame.place_a
    openings = (
        f"During the 09:00 inventory, {reader} stands beside {sender} while the "
        f"token is set inside the {start}. Both look directly at its placement.",
        f"At 09:00, the token is deposited in the {start} under the joint inspection "
        f"of {sender} and {reader}; each can see that the other is inspecting it.",
        f"The 09:00 check opens with {reader} and {sender} examining the token as "
        f"it is put into the {start}. They observe this together.",
        f"{sender} and {reader} are side by side for the 09:00 placement of a "
        f"token in the {start}, with both watching it go inside.",
    )
    movements = (
        f"At 09:15 an attendant transfers the token out of the {start} and into "
        f"the {finish}. {sender} follows the complete transfer on an overhead "
        f"object camera. Its image is cropped to the containers and the attendant's "
        f"hands, so {sender} cannot see {reader} or determine whether {reader} watches.",
        f"The 09:15 handling takes the token from the {start} to the {finish}. "
        f"{sender} sees the entire handling through a close-up link showing only "
        f"the containers and moving hands. The link gives {sender} no view of "
        f"{reader} and no information about whether {reader} observes the handling.",
        f"At 09:15, an attendant lifts the token from the {start} before setting it "
        f"in the {finish}. {sender}'s live object view shows every step but excludes "
        f"{reader}. The camera provides no way for {sender} to check {reader}'s attendance.",
        f"The token is relocated at 09:15: it leaves the {start} and is put in "
        f"the {finish}. {sender} monitors that relocation from a close-up image "
        f"which covers the token and containers but cannot show {reader} or whether "
        f"{reader} is watching.",
    )
    actual_present = (
        f"{reader} watches the 09:15 transfer directly, including its destination.",
        f"{reader} has an uninterrupted view of the token for the whole 09:15 handling.",
        f"{reader} sees both the removal and the new placement at 09:15.",
        f"{reader} attends the complete 09:15 relocation and watches it finish.",
    )
    actual_absent = (
        f"{reader} is in the waiting area from 09:10 until 09:20 and watches none "
        "of the 09:15 transfer.",
        f"{reader} is away from the containers for the entire 09:15 handling and "
        "does not watch any of it.",
        f"{reader} spends 09:10 to 09:20 elsewhere and sees neither the removal "
        "nor the new placement.",
        f"{reader} is absent throughout the 09:15 relocation and cannot watch it.",
    )
    record_present = (
        f"The witness log states that {reader} watched the complete 09:15 transfer.",
        f"The inspection note says {reader} watched all of the 09:15 handling.",
        f"The attendance screen reports that {reader} watched the entire 09:15 transfer.",
        f"The observation entry records {reader} as having watched the whole 09:15 relocation.",
    )
    record_absent = (
        f"The witness log states that {reader} did not watch any of the 09:15 transfer.",
        f"The inspection note says {reader} did not watch any of the 09:15 handling.",
        f"The attendance screen reports that {reader} did not watch the 09:15 transfer.",
        f"The observation entry records that {reader} did not watch any of the 09:15 relocation.",
    )
    index = frame.wording
    actual = (
        actual_present[index]
        if observed
        else (
            actual_absent[index]
            + f" No report reaches {reader}, and {reader} does not learn that a transfer occurred."
        )
    )
    actual = (
        f"Private attendance account, known to {reader} and the narrator but not "
        f"available to {sender}: " + actual
    )
    record = record_present[index] if recorded else record_absent[index]
    record = (
        f"Record supplied to {sender}: "
        + record
        + f" It also states that {reader} received no token-location messages after "
        f"the initial placement. {sender} accepts this record as accurate and "
        f"has no other information about {reader}'s attendance."
    )
    movement = movements[index] + " The token is not handled again before 09:20."
    return openings[index], {"actual": actual, "record": record, "world": movement}


def _messages(frame, world, observed, recorded):
    opening, paragraphs = _scene_parts(frame, world, observed, recorded)
    result = {}
    for view in ("omniscient", "sender"):
        parts = [opening]
        parts.extend(
            paragraphs[key] for key in frame.order if view == "omniscient" or key != "actual"
        )
        parts.append(
            "No token-location messages are exchanged in this account. "
            "No additional observation or event occurs before the account ends at 09:20."
        )
        parts.append(
            f'Candidate proposition: "The token is in the {frame.place_b}." '
            "This is a statement to assess, not new evidence or a message received "
            "by either person."
        )
        parts.append("End of account. A question may follow.")
        result[view] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n\n".join(parts)},
        ]
    return result


def _questions(frame):
    place = frame.place_b
    return {
        "world": f"At the end of the scene, is the token actually in the {place}? "
        "Answer only YES or NO.",
        "reader": f"At the end of the scene, does {frame.reader} believe that the token is in "
        f"the {place}? Answer only YES or NO.",
        "nested": f"At the end of the scene, does {frame.sender} think that "
        f"{frame.reader} believes that the token is in the {place}? Answer only YES or NO.",
    }


def _case(frame, world, observed, recorded):
    return {
        "id": f"{frame.template_id}_w{world}_a{observed}_b{recorded}",
        "template_id": frame.template_id,
        "split": "fresh",
        "phase": "prior",
        "world_bit": world,
        "reader_observed": observed,
        "sender_recorded": recorded,
        "reader_name": frame.reader,
        "sender_name": frame.sender,
        "place_a": frame.place_a,
        "place_b": frame.place_b,
        "claim_place_bit": 1,
        "labels": _gold(world, observed, recorded),
        "messages": _messages(frame, world, observed, recorded),
        "questions": _questions(frame),
    }


def build_fresh_panel():
    """Build the deterministic fresh 32-case draft before collecting model outputs."""
    return validate_fresh_panel(
        {
            "schema": 1,
            "panel_id": PANEL_ID,
            "assumptions": list(ASSUMPTIONS),
            "cases": [
                _case(frame, world, observed, recorded)
                for frame in _FRAMES
                for world, observed, recorded in itertools.product((0, 1), repeat=3)
            ],
        }
    )


def validate_fresh_panel(panel):
    """Validate source-derived labels and visibility; this is not human review."""
    if not isinstance(panel, Mapping) or set(panel) != {
        "schema",
        "panel_id",
        "assumptions",
        "cases",
    }:
        raise ValueError("Expected the complete fresh-query panel schema")
    if type(panel["schema"]) is not int or panel["schema"] != 1:
        raise ValueError("Unknown fresh-query schema")
    if panel["panel_id"] != PANEL_ID or panel["assumptions"] != list(ASSUMPTIONS):
        raise ValueError("Fresh panel identity or authoring assumptions changed")
    cases = panel["cases"]
    if not isinstance(cases, list) or len(cases) != 32:
        raise ValueError("Expected exactly 32 fresh prior cases")
    groups, identifiers = {}, set()
    for case in cases:
        if not isinstance(case, Mapping) or case.get("template_id") not in _BY_TEMPLATE:
            raise ValueError("Unknown fresh wording group")
        world, observed, recorded = (
            _bit(case.get(key), key) for key in ("world_bit", "reader_observed", "sender_recorded")
        )
        labels = case.get("labels")
        if not isinstance(labels, Mapping) or set(labels) != set(TARGETS):
            raise ValueError("Expected three explicitly named gold targets")
        for target in TARGETS:
            _bit(labels[target], f"labels.{target}")
        if type(case.get("claim_place_bit")) is not int or case["claim_place_bit"] != 1:
            raise ValueError("Fresh claim must concern place_b")
        expected = _case(_BY_TEMPLATE[case["template_id"]], world, observed, recorded)
        if dict(case) != expected:
            raise ValueError("Fresh source, gold, role question, split, or visibility changed")
        if case["id"] in identifiers:
            raise ValueError("Duplicate fresh case identifier")
        identifiers.add(case["id"])
        groups.setdefault(case["template_id"], []).append(case)
    if set(groups) != set(_BY_TEMPLATE):
        raise ValueError("Missing fresh wording group")
    cube = set(itertools.product((0, 1), repeat=3))
    for members in groups.values():
        if (
            len(members) != 8
            or {
                (case["world_bit"], case["reader_observed"], case["sender_recorded"])
                for case in members
            }
            != cube
        ):
            raise ValueError("Each fresh wording group must exhaust the eight source cells")
        if {tuple(case["labels"][key] for key in TARGETS) for case in members} != cube:
            raise ValueError("Each fresh wording group must exhaust the eight target triples")
        for left, right in itertools.combinations(TARGETS, 2):
            counts = Counter((case["labels"][left], case["labels"][right]) for case in members)
            if len(counts) != 4 or set(counts.values()) != {2}:
                raise ValueError("Fresh targets must be balanced and pairwise independent")
        for world, recorded in itertools.product((0, 1), repeat=2):
            pair = [
                case
                for case in members
                if case["world_bit"] == world and case["sender_recorded"] == recorded
            ]
            serialized = [
                json.dumps(case["messages"]["sender"], sort_keys=True, ensure_ascii=False)
                for case in pair
            ]
            if serialized[0].encode("utf-8") != serialized[1].encode("utf-8"):
                raise ValueError("Actual private attendance leaked into fresh sender view")
    names = [name for frame in _FRAMES for name in (frame.reader, frame.sender)]
    places = [place for frame in _FRAMES for place in (frame.place_a, frame.place_b)]
    if len(set(names)) != 8 or len(set(places)) != 8:
        raise ValueError("Fresh names and places must be distinct across wording groups")
    return copy.deepcopy(dict(panel))


def load_fresh_panel(path):
    return validate_fresh_panel(json.loads(Path(path).read_text(encoding="utf-8")))
