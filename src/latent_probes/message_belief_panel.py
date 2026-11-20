"""Authored known-message perspective diagnostic, frozen before model outputs.

This changes the evidence-update mechanism relative to the prior-state panel.
Labels specify fictional states under explicit trust rules, not measured beliefs.
The four groups vary wording/entities within one known-message logical family.
"""

from __future__ import annotations

import copy
import itertools
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .belief_query_panel import SYSTEM, TARGETS, _questions

PANEL_ID = "known_message_validation_v1"
VIEWS = ("omniscient", "sender")
SOURCE_BITS = ("world_bit", "message_bit", "delivered", "receipt")
ASSUMPTIONS = (
    "Author-assisted draft; annotation_origin=agent_assisted_draft; "
    "needs_human_review=true. No independent human validation has occurred.",
    "This is a frozen diagnostic of existing prior-state heads on known-message "
    "updates. It is not new probe training, pressure transfer, or a validation success claim.",
    "Four new wording/entity groups each exhaust all sixteen world/message/actual-"
    "delivery/sender-trusted-receipt combinations. They share one logical family.",
    "The token occupies exactly one of two places. Both people observe the initial "
    "placement; its bit is one minus the later message bit.",
    "The actual current world location varies independently of the message content. "
    "The message may be false without the recipient knowing that it is false.",
    "The recipient has no observation of the token after the initial placement and "
    "learns no new location except by fully reading the specified delivered message.",
    "The recipient treats this service's message as authoritative, accepts its "
    "location claim even if false, and replaces the remembered initial location.",
    "Actual delivery means the recipient fully reads the exact message; non-delivery "
    "means no content or notification reaches the recipient.",
    "The sender knows the initial location, current world location and exact message. "
    "It knows the recipient's trust and update rules but cannot observe actual delivery.",
    "The sender accepts its particular delivery receipt as definitive evidence of "
    "whether the recipient fully read the message. No other delivery evidence exists.",
    "Actual delivery is independently varied in the narrator's private record. "
    "Possible receipt error is not disclosed to the sender.",
    "The sender view omits private actual delivery. Opposing actual-reader labels "
    "therefore share identical sender inputs and require exactly 50% paired accuracy.",
    "Gold world=W, reader=M if delivered else 1-M, nested=M if receipt else 1-M. "
    "Each label evaluates the fixed place_b proposition, not belief accuracy.",
    "Capture is after the exact known message and receipt, before any appended "
    "perspective question or generated answer. No future unknown message is forecast.",
    "System text and world/reader/nested question stems exactly reuse the existing "
    "query panel. Base messages contain no gold, source bits, case IDs or questions.",
    "All rows are fresh evaluation only. Existing weights, transforms, regularization "
    "and primary block16 stay frozen. These results must not select a new head.",
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
        "km_t01",
        "Elara",
        "Quentin",
        "brass recess",
        "pewter recess",
        ("world", "message", "actual", "receipt"),
        0,
    ),
    _Frame(
        "km_t02",
        "Juniper",
        "Otto",
        "coral hamper",
        "jade hamper",
        ("message", "world", "receipt", "actual"),
        1,
    ),
    _Frame(
        "km_t03",
        "Esme",
        "Felix",
        "upper niche",
        "lower niche",
        ("receipt", "message", "actual", "world"),
        2,
    ),
    _Frame(
        "km_t04",
        "Galen",
        "Tamsin",
        "oval coffer",
        "triangular coffer",
        ("actual", "world", "message", "receipt"),
        3,
    ),
)
_BY_TEMPLATE = {frame.template_id: frame for frame in _FRAMES}


def _bit(value, name):
    if type(value) is not int or value not in (0, 1):
        raise ValueError(f"{name} must be integer 0 or 1")
    return value


def labels(case):
    """Derive all three normative targets from the four source bits, not stored gold."""
    if not isinstance(case, Mapping):
        raise ValueError("Expected a known-message case mapping")
    world, message, delivered, receipt = (_bit(case.get(name), name) for name in SOURCE_BITS)
    return {
        "world": world,
        "reader": message if delivered else 1 - message,
        "nested": message if receipt else 1 - message,
    }


def _messages(frame, world, message, delivered, receipt):
    reader, sender = frame.reader, frame.sender
    initial = frame.place_a if message else frame.place_b
    current = frame.place_b if world else frame.place_a
    asserted = frame.place_b if message else frame.place_a
    index = frame.wording
    openings = (
        f"At 09:00, {reader} and {sender} stand together while the token is put "
        f"in the {initial}. Both inspect the placement and see each other inspecting it.",
        f"During the 09:00 inventory, {sender} and {reader} jointly observe the "
        f"token being placed inside the {initial}; each knows the other saw it.",
        f"The 09:00 check finds {reader} and {sender} side by side as the token "
        f"is set in the {initial}. They watch this initial placement together.",
        f"At 09:00, both {sender} and {reader} look directly at the token as it "
        f"is put into the {initial}, observing the placement and each other's attendance.",
    )
    worlds = (
        f"At 09:10, {sender} checks a private live object camera: the token is "
        f"currently in the {current}. {reader} cannot see that feed.",
        f"A private live view at 09:10 shows {sender} that the token is currently "
        f"in the {current}. This camera is inaccessible to {reader}.",
        f"{sender}'s private 09:10 object feed shows the token inside the {current}. "
        f"The feed is live and accurate; {reader} has no access to it.",
        f"At 09:10, a live camera available only to {sender} shows the token in "
        f"the {current}. {reader} sees none of this current-location evidence.",
    )
    message_text = f"The token is now in the {asserted}."
    messages = (
        f"At 09:15, a location service attempts to send {reader} the exact message "
        f'"{message_text}" {sender} sees this outgoing text but cannot see delivery.',
        f"The location service's 09:15 outgoing text to {reader} reads exactly "
        f'"{message_text}" Its content is visible to {sender}; actual delivery is not.',
        f"The exact text queued by the location service for {reader} at 09:15 is "
        f'"{message_text}" {sender} knows that text without observing its delivery.',
        f"At 09:15 the location service tries to transmit this exact text to {reader}: "
        f'"{message_text}" {sender} can inspect the text but cannot observe its arrival.',
    )
    delivered_text = (
        f"The message reaches {reader}, who fully reads it and replaces the "
        "remembered location with the location it states.",
        f"{reader} receives and reads the complete message, then adopts its "
        "stated location instead of the earlier remembered location.",
        f"The complete message is delivered to {reader} and read in full; its "
        "location claim replaces the initial remembered location.",
        f"{reader} reads the entire delivered message and updates the remembered "
        "location to the one asserted in that message.",
    )
    undelivered_text = (
        f"No message content or notification reaches {reader}; the initial "
        "remembered location is retained.",
        f"{reader} receives neither this message nor an arrival notification, "
        "and retains the initial remembered location.",
        f"The attempted message and any notification fail to reach {reader}, "
        "whose remembered location remains the initial one.",
        f"{reader} sees no message content and no notification of this attempt, "
        "so the initial remembered location is unchanged.",
    )
    actual = (
        f"Private delivery account for the 09:15 attempt, known to {reader} and the narrator but not "
        f"available to {sender}: "
        + (delivered_text[index] if delivered else undelivered_text[index])
    )
    read_receipt = (
        f"The delivery receipt says {reader} received and fully read the 09:15 message.",
        f"The receipt records successful delivery and complete reading by {reader}.",
        f"The transmission log states that {reader} received the entire message and read it.",
        f"The status entry reports that the complete 09:15 text was delivered to and read by {reader}.",
    )
    no_receipt = (
        f"The delivery receipt says no content or notification from the 09:15 attempt reached {reader}.",
        f"The receipt records that neither the message nor a notification reached {reader}.",
        f"The transmission log states that {reader} received no content or notification from this attempt.",
        f"The status entry reports that none of the 09:15 message or any notification was delivered to {reader}.",
    )
    receipt_text = (
        f"Receipt supplied to {sender} at 09:16: "
        + (read_receipt[index] if receipt else no_receipt[index])
        + f" {sender} accepts this receipt as definitive, knows {reader}'s update "
        "rule, and has no other evidence about actual delivery."
    )
    paragraphs = {
        "world": worlds[index],
        "message": messages[index],
        "actual": actual,
        "receipt": receipt_text,
    }
    result = {}
    for view in VIEWS:
        parts = [
            openings[index],
            f"After 09:00, {reader} has no further view of the token and no source "
            "of location information except the particular service message described "
            f"below. {reader} treats this service as authoritative: if a message is "
            "fully read, its location claim replaces the earlier remembered location "
            "without checking, even when the claim differs from reality. Without "
            "a delivered message the initial remembered location is retained. "
            f"{sender} knows these rules. Neither person has other location updates "
            "beyond those described in this account.",
        ]
        parts.extend(
            paragraphs[key] for key in frame.order if view == "omniscient" or key != "actual"
        )
        parts.append(
            "The token stays at its 09:10 location through the end at 09:20. "
            "There are no further observations, messages, receipts or conversations. "
            f"{sender} knows the initial location, the live current location, the "
            "outgoing message and the supplied receipt. The private delivery account "
            f"is not available to {sender}."
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


def _case(frame, world, message, delivered, receipt):
    source = dict(zip(SOURCE_BITS, (world, message, delivered, receipt), strict=True))
    return {
        "id": f"{frame.template_id}_w{world}_m{message}_a{delivered}_b{receipt}",
        "template_id": frame.template_id,
        "split": "fresh",
        "phase": "known_message",
        **source,
        "reader_name": frame.reader,
        "sender_name": frame.sender,
        "place_a": frame.place_a,
        "place_b": frame.place_b,
        "claim_place_bit": 1,
        "needs_human_review": True,
        "labels": labels(source),
        "messages": _messages(frame, world, message, delivered, receipt),
        "questions": _questions(frame),
    }


def build_panel():
    """Build the deterministic prospective diagnostic without model calls."""
    return validate_panel(
        {
            "schema": 1,
            "panel_id": PANEL_ID,
            "assumptions": list(ASSUMPTIONS),
            "cases": [
                _case(frame, *bits)
                for frame in _FRAMES
                for bits in itertools.product((0, 1), repeat=4)
            ],
        }
    )


def validate_panel(panel):
    """Enforce frozen construction and visibility; this is not human validation."""
    if not isinstance(panel, Mapping) or set(panel) != {
        "schema",
        "panel_id",
        "assumptions",
        "cases",
    }:
        raise ValueError("Expected complete known-message panel schema")
    if type(panel["schema"]) is not int or panel["schema"] != 1:
        raise ValueError("Unknown known-message panel schema")
    if panel["panel_id"] != PANEL_ID or panel["assumptions"] != list(ASSUMPTIONS):
        raise ValueError("Known-message identity or assumptions changed")
    cases = panel["cases"]
    if not isinstance(cases, list) or len(cases) != 64:
        raise ValueError("Expected exactly 64 known-message cases")
    groups, ids = {}, set()
    for case in cases:
        if not isinstance(case, Mapping) or case.get("template_id") not in _BY_TEMPLATE:
            raise ValueError("Unknown known-message wording group")
        bits = tuple(_bit(case.get(key), key) for key in SOURCE_BITS)
        expected = _case(_BY_TEMPLATE[case["template_id"]], *bits)
        stored = case.get("labels")
        if not isinstance(stored, Mapping) or set(stored) != set(TARGETS):
            raise ValueError("Expected three named gold targets")
        for key in TARGETS:
            _bit(stored[key], f"labels.{key}")
        if type(case.get("claim_place_bit")) is not int or case["claim_place_bit"] != 1:
            raise ValueError("Candidate must be place_b")
        if case.get("needs_human_review") is not True or dict(case) != expected:
            raise ValueError("Frozen source, gold, question, review flag or visibility changed")
        if case["id"] in ids:
            raise ValueError("Duplicate case identifier")
        ids.add(case["id"])
        groups.setdefault(case["template_id"], []).append(case)
    if set(groups) != set(_BY_TEMPLATE):
        raise ValueError("Missing known-message wording group")
    cube = set(itertools.product((0, 1), repeat=4))
    for members in groups.values():
        if (
            len(members) != 16
            or {tuple(case[key] for key in SOURCE_BITS) for case in members} != cube
        ):
            raise ValueError("Each group must exhaust sixteen source cells")
        for target in TARGETS:
            if Counter(case["labels"][target] for case in members) != {0: 8, 1: 8}:
                raise ValueError("Every target must be balanced per wording group")
        for left, right in itertools.combinations(TARGETS, 2):
            if Counter((case["labels"][left], case["labels"][right]) for case in members) != {
                pair: 4 for pair in itertools.product((0, 1), repeat=2)
            }:
                raise ValueError("Gold targets must be pairwise independent")
        for world, message, receipt in itertools.product((0, 1), repeat=3):
            pair = [
                case
                for case in members
                if (case["world_bit"], case["message_bit"], case["receipt"])
                == (world, message, receipt)
            ]
            if pair[0]["messages"]["sender"] != pair[1]["messages"]["sender"]:
                raise ValueError("Private actual delivery leaked into sender view")
    return copy.deepcopy(dict(panel))


def load_panel(path):
    """Load and validate the prospectively saved JSON panel."""
    return validate_panel(json.loads(Path(path).read_text(encoding="utf-8")))


def render_messages(case, view):
    """Return detached base chat messages, without a perspective question."""
    if view not in VIEWS:
        raise ValueError("View must be omniscient or sender")
    if not isinstance(case, Mapping) or case.get("template_id") not in _BY_TEMPLATE:
        raise ValueError("Unknown known-message case")
    bits = tuple(_bit(case.get(key), key) for key in SOURCE_BITS)
    expected = _case(_BY_TEMPLATE[case["template_id"]], *bits)
    if dict(case) != expected:
        raise ValueError("Known-message case changed")
    return copy.deepcopy(expected["messages"][view])
