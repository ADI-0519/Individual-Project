"""Fresh complementary-claim validation across two evidence channels.

Authored fictional gold is not an empirical LM-belief label. Observation access
and trusted-message access share a Boolean truth function; this bank tests fresh
wording/evidence-channel transfer, not broad reasoning-family OOD. No model
outputs are read or generated here.
"""

from __future__ import annotations

import copy
import itertools
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .belief_panel import SYSTEM, build_panel
from .belief_query_panel import build_fresh_panel

PANEL_ID = "belief_contrast_validation_v1"
TARGETS = ("world", "reader", "nested")
CHANNELS = ("observation_prior", "trusted_message_update")
ASSUMPTIONS = (
    "Agent-assisted authored draft; annotation_origin=agent_assisted_draft; "
    "needs_human_review=true. No independent human validation is claimed.",
    "All 128 cases are fresh scoring material: eight novel wording/entity groups "
    "per evidence channel and all eight world/access/record combinations per group.",
    "The channels share the same binary gold function. This is evidence-channel "
    "transfer with new wording, not broad reasoning-family OOD.",
    "The world ends opposite its initial state in each authored scene. This is an "
    "authoring invariant, not a public rule available to an absent reader.",
    "Both people observe the initial placement together, and the sender observes "
    "the entire world transfer without seeing the reader's private access.",
    "In observation_prior, reader_updated means the reader actually watches the "
    "transfer; sender_recorded means the trusted attendance entry says so.",
    "In trusted_message_update, the reader always misses the world transfer and "
    "the sender knows this. reader_updated means private receipt/read/trust of the "
    "truthful current-location note; sender_recorded is the independently varied "
    "receipt accepted by the sender.",
    "The actual delivery can fail despite a positive trusted receipt, or succeed "
    "despite a negative trusted receipt. The sender has no independent view of the "
    "reader or their inbox. Possible receipt errors are narrator-only information.",
    "The reader remembers the last observed location and updates after observing "
    "a transfer or reading a trusted current-location note. There are no other updates.",
    "Base labels encode location content for place_b, not accuracy of a person's belief. "
    "The reader and nested labels equal world if their respective access bit is one, "
    "and otherwise equal the initial location.",
    "For each target, place_a and place_b candidate labels are exact complements. "
    "Candidate claims name a location only and do not name a perspective or give gold.",
    "Each sender candidate prefix is byte-identical across actual-access pairs; "
    "actual hidden reader labels are therefore structurally unknowable in that view.",
    "SYSTEM, neutral candidate disclaimer, and perspective-question stems match the "
    "legacy task. Questions are separate metadata and are not in candidate prefixes.",
    "Actor-role assignment is fixed within each complete cube and independent of "
    "source bits; initial reader-first/sender-first mention order is balanced across groups.",
    "All groups have new people and places, including between the two channels. "
    "Channel comparisons therefore also include fresh lexical variation and are not "
    "an isolated causal effect of the evidence operation.",
    "Pairwise score subtraction does not prove causal belief tracking, remove every "
    "candidate interaction, or validate an unknown-future-message forecast.",
)


@dataclass(frozen=True)
class _Frame:
    template_id: str
    channel: str
    reader: str
    sender: str
    place_a: str
    place_b: str
    style: int
    order: tuple[str, ...]


_ORDERS = (
    ("world", "actual", "record"),
    ("record", "world", "actual"),
    ("actual", "record", "world"),
    ("world", "record", "actual"),
    ("record", "actual", "world"),
    ("actual", "world", "record"),
    ("world", "actual", "record"),
    ("record", "world", "actual"),
)
_IDENTITIES = (
    ("Aiden", "Bea", "oak chest", "pine chest"),
    ("Cora", "Devon", "bronze cabinet", "pewter cabinet"),
    ("Elena", "Felix", "upper locker", "lower locker"),
    ("Greta", "Hugo", "scarlet case", "turquoise case"),
    ("Ida", "Jasper", "glass-front cupboard", "mesh-front cupboard"),
    ("Lena", "Milo", "narrow drawer", "wide drawer"),
    ("Nora", "Otis", "black trunk", "white trunk"),
    ("Pia", "Quinn", "star-marked crate", "moon-marked crate"),
    ("Ravi", "Sana", "alpha vault", "beta vault"),
    ("Theo", "Una", "jade cubby", "coral cubby"),
    ("Vik", "Willa", "large coffer", "small coffer"),
    ("Xander", "Yara", "arched recess", "flat recess"),
    ("Zane", "Alma", "yellow safe", "purple safe"),
    ("Boris", "Celia", "dotted bin", "striped bin"),
    ("Dorian", "Esther", "east cupboard", "west cupboard"),
    ("Florian", "Hazel", "first compartment", "second compartment"),
)
_FRAMES = tuple(
    _Frame(
        f"bc{'p' if channel_index == 0 else 'm'}_{style + 1:02d}",
        channel,
        *_IDENTITIES[channel_index * 8 + style],
        style,
        _ORDERS[style],
    )
    for channel_index, channel in enumerate(CHANNELS)
    for style in range(8)
)
_BY_TEMPLATE = {frame.template_id: frame for frame in _FRAMES}


def _bit(value, name):
    if type(value) is not int or value not in (0, 1):
        raise ValueError(f"{name} must be an integer bit")
    return value


def _content_labels(world, actual, recorded):
    return {
        "world": world,
        "reader": world if actual else 1 - world,
        "nested": world if recorded else 1 - world,
    }


def _opening(frame, initial):
    r, s = frame.reader, frame.sender
    return (
        f"At 16:00, {r} and {s} jointly inspect a token as it is stored in the {initial}.",
        f"The 16:00 inventory is witnessed together by {s} and {r}; a token goes "
        f"into the {initial} while both watch.",
        f"{r} is beside {s} at 16:00 when the token is deposited in the {initial}. "
        "Each sees the placement and the other observer.",
        f"During the opening check at 16:00, {s} and {r} stand together and see a "
        f"token being stored inside the {initial}.",
        f"The scene opens at 16:00 with {r} watching beside {s} as the token is "
        f"secured in the {initial}.",
        f"{s} observes the 16:00 placement alongside {r}: the token is put in the "
        f"{initial}, in full view of both.",
        f"At the 16:00 check, {r} and {s} are together as a token is set down "
        f"inside the {initial}.",
        f"The first inspection, at 16:00, has {s} and {r} jointly watch the token "
        f"enter the {initial}.",
    )[frame.style]


def _world(frame, initial, current):
    s, r = frame.sender, frame.reader
    movement = (
        f"At 16:10 the custodian takes the token from the {initial} to the {current}.",
        f"The custodian's 16:10 transfer removes the token from the {initial} and "
        f"leaves it inside the {current}.",
        f"A custodian relocates the token at 16:10, lifting it from the {initial} "
        f"and depositing it in the {current}.",
        f"The 16:10 handling moves the token out of the {initial} and into the {current}.",
        f"At 16:10, a custodian exchanges the token's storage place: from the "
        f"{initial} to the {current}.",
        f"During the 16:10 relocation, the token leaves the {initial} and is placed "
        f"inside the {current}.",
        f"The token is transferred at 16:10 by a custodian from the {initial} into the {current}.",
        f"The custodian carries out a transfer at 16:10, taking the token out of "
        f"the {initial} and storing it in the {current}.",
    )[frame.style]
    return (
        movement + f" {s} sees the full transfer on an object-only video feed. "
        f"The feed cannot show {r} or whether {r} watches it. "
        "The token is not moved again."
    )


def _observation_parts(frame, actual, recorded):
    r, s = frame.reader, frame.sender
    positive = (
        f"{r} directly watches the token throughout the 16:10 transfer.",
        f"{r} sees every stage of the custodian's transfer at 16:10.",
        f"{r} is present for, and watches, the full 16:10 relocation.",
        f"{r} observes the custodian remove the token and put it away at 16:10.",
        f"{r} watches from the beginning to the end of the 16:10 storage change.",
        f"{r} witnesses both endpoints of the token's 16:10 relocation.",
        f"{r} stays with the token and sees the entire 16:10 transfer.",
        f"{r} has a direct view of the complete 16:10 transfer and follows it.",
    )
    negative = (
        f"{r} is elsewhere for the entire 16:10 transfer and sees none of it.",
        f"{r} is away before the transfer begins and returns only once it is finished.",
        f"{r} is absent throughout the 16:10 relocation and does not watch it.",
        f"{r} does not see the token being removed or put away at 16:10.",
        f"{r} misses every part of the 16:10 storage change.",
        f"{r} sees neither endpoint of the custodian's 16:10 relocation.",
        f"{r} leaves before 16:10 and sees none of the transfer.",
        f"{r} has no view of any part of the 16:10 transfer.",
    )
    yes_record = (
        f"The attendance entry says {r} watched the whole 16:10 transfer.",
        f"The observer register records that {r} watched every stage at 16:10.",
        f"The access log reports that {r} watched the complete 16:10 relocation.",
        f"The witness entry states that {r} watched the removal and new placement.",
        f"The attendance note lists {r} as watching the full 16:10 storage change.",
        f"The observer record states that {r} watched both endpoints of the relocation.",
        f"The attendance report records {r} watching the entire transfer at 16:10.",
        f"The witness register says {r} watched the full custodian transfer.",
    )
    no_record = (
        f"The attendance entry says {r} did not watch any of the 16:10 transfer.",
        f"The observer register records that {r} did not watch any stage at 16:10.",
        f"The access log reports that {r} did not watch the 16:10 relocation.",
        f"The witness entry states that {r} did not watch the removal or new placement.",
        f"The attendance note lists {r} as not watching any of the 16:10 storage change.",
        f"The observer record states that {r} did not watch either endpoint.",
        f"The attendance report records that {r} did not watch the transfer at 16:10.",
        f"The witness register says {r} did not watch any of the custodian transfer.",
    )
    actual_text = (
        positive[frame.style]
        if actual
        else (
            negative[frame.style]
            + f" {r} receives no update and does not learn that a transfer occurred."
        )
    )
    actual_text = (
        f"Private attendance, unavailable to {s}: "
        + actual_text
        + f" {s} has no independent way to see this attendance."
    )
    record_text = (
        f"Attendance record available to {s}: "
        + (yes_record[frame.style] if recorded else no_record[frame.style])
        + f" It records no later token-location update to {r}. "
        f"{s} accepts that particular entry as accurate and has no other attendance "
        f"information. No token-location messages are exchanged."
    )
    return actual_text, record_text


def _message_parts(frame, actual, recorded):
    r, s = frame.reader, frame.sender
    positive = (
        f"At 16:14, {r} receives, reads, and trusts the current-location note.",
        f"The note reaches {r} at 16:14; {r} reads it and accepts it as accurate.",
        f"{r} opens the delivered note at 16:14 and trusts its current-location statement.",
        f"At 16:14 the delivered note is read and accepted as accurate by {r}.",
        f"{r} gets the note at 16:14, reads it, and relies on its current-location information.",
        f"At 16:14, {r} reads the note that has arrived and trusts it.",
        f"The note is delivered to {r} by 16:14 and is then read and trusted.",
        f"{r} takes delivery at 16:14, reads the note, and accepts its contents.",
    )
    negative = (
        f"The note never reaches {r}; {r} reads no note by 16:20.",
        f"Delivery fails, and {r} receives no current-location note before 16:20.",
        f"{r}'s inbox receives no note, so {r} does not read it.",
        f"No copy of the note is delivered to {r} by the end of the scene.",
        f"{r} does not get the note and has no access to its contents.",
        f"The attempted delivery does not reach {r}, who reads no note.",
        f"The note remains undelivered, and {r} never sees it.",
        f"{r} takes no delivery and cannot read the note before 16:20.",
    )
    yes_receipt = (
        f"The receipt says the note reached {r} and was read and trusted.",
        f"The delivery entry records that {r} received, read, and accepted the note.",
        f"The read-receipt reports the note delivered to {r}, read, and trusted.",
        f"The dispatch log says {r} received the note and accepted it after reading.",
        f"The delivery certificate records {r} reading and trusting the delivered note.",
        f"The receipt reports successful delivery, reading, and acceptance by {r}.",
        f"The note's status entry says {r} received, read, and trusted it.",
        f"The dispatch receipt confirms that {r} read and accepted the delivered note.",
    )
    no_receipt = (
        f"The receipt says the note did not reach {r} and was not read.",
        f"The delivery entry records that {r} did not receive or read the note.",
        f"The read-receipt reports no delivery to {r} and no reading of the note.",
        f"The dispatch log says {r} received no note and did not read it.",
        f"The delivery certificate records no receipt or reading by {r}.",
        f"The receipt reports delivery failure and no reading by {r}.",
        f"The note's status entry says {r} did not receive or read it.",
        f"The dispatch receipt reports that {r} received no note and did not read it.",
    )
    actual_text = (
        positive[frame.style]
        if actual
        else (
            negative[frame.style]
            + f" {r} does not learn that the transfer occurred or that a note was sent."
        )
    )
    actual_text = (
        f"Actual delivery, private to {r} and the narrator and unavailable to {s}: " + actual_text
    )
    record_text = (
        f"Receipt supplied to {s} at 16:15: "
        + (yes_receipt[frame.style] if recorded else no_receipt[frame.style])
        + f" {s} accepts this particular receipt as accurate. {s} cannot see {r}, "
        f"{r}'s inbox, or any independent acknowledgment, and has no other delivery information."
    )
    return actual_text, record_text


def _scene(frame, world, actual, recorded, view):
    initial = frame.place_a if world else frame.place_b
    current = frame.place_b if world else frame.place_a
    parts = [_opening(frame, initial)]
    world_text = _world(frame, initial, current)
    if frame.channel == "observation_prior":
        actual_text, record_text = _observation_parts(frame, actual, recorded)
    else:
        parts.append(
            f"At 16:05 {frame.reader} leaves for a separate room. {frame.sender} "
            f"sees the departure and knows {frame.reader} cannot see the 16:10 "
            "transfer or the object video feed. No other person supplies an update."
        )
        world_text += (
            f" At 16:12 {frame.sender} sends {frame.reader} this note: "
            f'"The token is now in the {current}." '
            f"{frame.reader} has agreed to trust a current-location note from "
            f"{frame.sender}, and {frame.sender} knows that agreement."
        )
        actual_text, record_text = _message_parts(frame, actual, recorded)
    paragraphs = {"world": world_text, "actual": actual_text, "record": record_text}
    parts.extend(paragraphs[key] for key in frame.order if view == "omniscient" or key != "actual")
    parts.append(
        "The scene ends at 16:20. After the stated events, there are no additional "
        "token observations, contacts, or updates."
    )
    return "\n\n".join(parts)


def _questions(frame, place_bit):
    place = (frame.place_a, frame.place_b)[place_bit]
    return {
        "world": f"At the end of the scene, is the token actually in the {place}? "
        "Answer only YES or NO.",
        "reader": f"At the end of the scene, does {frame.reader} believe that "
        f"the token is in the {place}? Answer only YES or NO.",
        "nested": f"At the end of the scene, does {frame.sender} think that "
        f"{frame.reader} believes that the token is in the {place}? Answer only YES or NO.",
    }


def _case(frame, world, actual, recorded):
    labels = _content_labels(world, actual, recorded)
    candidates = []
    for place_bit, place in enumerate((frame.place_a, frame.place_b)):
        claim = f"The token is in the {place}."
        messages = {}
        for view in ("omniscient", "sender"):
            text = _scene(frame, world, actual, recorded, view)
            text += (
                "\n\nCandidate statement for later assessment, not a new observation "
                f'or message to either person: "{claim}"'
                "\n\nThe scene is complete. Questions follow."
            )
            messages[view] = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": text},
            ]
        candidates.append(
            {
                "place_bit": place_bit,
                "claim": claim,
                "labels": {target: int(value == place_bit) for target, value in labels.items()},
                "messages": messages,
                "questions": _questions(frame, place_bit),
            }
        )
    return {
        "id": f"{frame.template_id}_w{world}_a{actual}_b{recorded}",
        "template_id": frame.template_id,
        "split": "fresh",
        "channel": frame.channel,
        "phase": "prior" if frame.channel == "observation_prior" else "post_message",
        "world_bit": world,
        "reader_updated": actual,
        "sender_recorded": recorded,
        "reader_name": frame.reader,
        "sender_name": frame.sender,
        "place_a": frame.place_a,
        "place_b": frame.place_b,
        "labels": labels,
        "candidates": candidates,
    }


def build_contrast_panel():
    return validate_contrast_panel(
        {
            "id": PANEL_ID,
            "schema": 1,
            "assumptions": list(ASSUMPTIONS),
            "cases": [
                _case(frame, world, actual, recorded)
                for frame in _FRAMES
                for world, actual, recorded in itertools.product((0, 1), repeat=3)
            ],
        }
    )


def validate_contrast_panel(panel):
    """Check exact rendering and gold; never repair or inspect model outputs."""
    if not isinstance(panel, Mapping) or set(panel) != {"id", "schema", "assumptions", "cases"}:
        raise ValueError("Expected the complete complementary-claim panel schema")
    if (
        panel["id"] != PANEL_ID
        or type(panel["schema"]) is not int
        or panel["schema"] != 1
        or panel["assumptions"] != list(ASSUMPTIONS)
    ):
        raise ValueError("Changed panel identity or authoring declarations")
    cases = panel["cases"]
    if not isinstance(cases, list) or len(cases) != 128:
        raise ValueError("Expected 64 fresh cases in each of two evidence channels")
    groups, identifiers = {}, set()
    for case in cases:
        if not isinstance(case, Mapping) or case.get("template_id") not in _BY_TEMPLATE:
            raise ValueError("Unknown contrast wording group")
        world, actual, recorded = (
            _bit(case.get(key), key) for key in ("world_bit", "reader_updated", "sender_recorded")
        )
        if not isinstance(case.get("labels"), Mapping) or set(case["labels"]) != set(TARGETS):
            raise ValueError("Expected world, reader, and nested content labels")
        for target in TARGETS:
            _bit(case["labels"][target], f"labels.{target}")
        if not isinstance(case.get("candidates"), list) or len(case["candidates"]) != 2:
            raise ValueError("Expected two complementary candidates")
        for candidate in case["candidates"]:
            if not isinstance(candidate, Mapping):
                raise ValueError("Each candidate must be a mapping")
            _bit(candidate.get("place_bit"), "candidate.place_bit")
            labels = candidate.get("labels")
            if not isinstance(labels, Mapping) or set(labels) != set(TARGETS):
                raise ValueError("Missing candidate-specific gold")
            for target in TARGETS:
                _bit(labels[target], f"candidate.labels.{target}")
        expected = _case(_BY_TEMPLATE[case["template_id"]], world, actual, recorded)
        if dict(case) != expected:
            raise ValueError(
                "Source, candidate, role, gold, channel, or private visibility changed"
            )
        if case["id"] in identifiers:
            raise ValueError("Duplicate contrast case")
        identifiers.add(case["id"])
        groups.setdefault(case["template_id"], []).append(case)
    if set(groups) != set(_BY_TEMPLATE):
        raise ValueError("Missing evidence-channel wording group")
    cube = set(itertools.product((0, 1), repeat=3))
    for members in groups.values():
        if (
            len(members) != 8
            or {
                tuple(case[key] for key in ("world_bit", "reader_updated", "sender_recorded"))
                for case in members
            }
            != cube
        ):
            raise ValueError("Each group must exhaust the source cube")
        if {tuple(case["labels"][target] for target in TARGETS) for case in members} != cube:
            raise ValueError("Each group must exhaust the target cube")
        for left, right in itertools.combinations(TARGETS, 2):
            counts = Counter((case["labels"][left], case["labels"][right]) for case in members)
            if len(counts) != 4 or set(counts.values()) != {2}:
                raise ValueError("Target bits must be balanced and pairwise independent")
        for world, recorded in itertools.product((0, 1), repeat=2):
            pair = [
                case
                for case in members
                if case["world_bit"] == world and case["sender_recorded"] == recorded
            ]
            for place_bit in (0, 1):
                prefixes = [
                    json.dumps(
                        case["candidates"][place_bit]["messages"]["sender"],
                        sort_keys=True,
                        ensure_ascii=False,
                    ).encode("utf-8")
                    for case in pair
                ]
                if prefixes[0] != prefixes[1]:
                    raise ValueError("Actual private access leaked into sender prefix")
    old_cases = build_panel()["cases"] + build_fresh_panel()["cases"]
    old_names = {case[key] for case in old_cases for key in ("reader_name", "sender_name")}
    old_places = {case[key] for case in old_cases for key in ("place_a", "place_b")}
    names = [name for frame in _FRAMES for name in (frame.reader, frame.sender)]
    places = [place for frame in _FRAMES for place in (frame.place_a, frame.place_b)]
    if len(set(names)) != 32 or len(set(places)) != 32:
        raise ValueError("Contrast groups must have distinct people and places")
    if set(names) & old_names or set(places) & old_places:
        raise ValueError("Contrast entities must be novel relative to earlier belief panels")
    return copy.deepcopy(dict(panel))


def load_contrast_panel(path):
    return validate_contrast_panel(json.loads(Path(path).read_text(encoding="utf-8")))
