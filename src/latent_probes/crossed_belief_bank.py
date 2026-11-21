"""Fully crossed, agent-authored neutral attribution material; no model calls.

The fictional attribution is a conditional update, not a measured reader LM
belief. Four renderings share one logical family. All material awaits human
semantic review. Executed older banks and learned heads are not modified.
"""

from __future__ import annotations

import copy
import itertools
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .belief_query_panel import SYSTEM, _questions

SCHEMA = "crossed_belief_validation_draft_v1"
HEAD = "nested_question__sender__nested__layer_16"
SOURCE_BITS = ("initial_slot", "message_slot", "receipt_reports_read", "world_slot")
CONDITIONS = ("nested_question", "story")


@dataclass(frozen=True)
class _Frame:
    template_id: str
    reader: str
    sender: str
    place_a: str
    place_b: str
    wording: int


_FRAMES = (
    _Frame("cb_t01", "Daria", "Lennox", "amber strongbox", "teal strongbox", 0),
    _Frame("cb_t02", "Mireya", "Bastian", "plum compartment", "saffron compartment", 1),
    _Frame("cb_t03", "Nadia", "Corin", "larch crate", "elm crate", 2),
    _Frame("cb_t04", "Soraya", "Alistair", "hexagonal recess", "pentagonal recess", 3),
)
_BY_TEMPLATE = {frame.template_id: frame for frame in _FRAMES}

ASSUMPTIONS = (
    "Agent-assisted draft. Human semantic review is incomplete; no model outputs "
    "were used to author these inputs. Gold describes fictional states, not "
    "measured beliefs in an independent reader model.",
    "Four wording/entity templates fully cross initial reader location belief I, "
    "exact message location M, sender-trusted receipt B, and current world W, "
    "with both global physical-name mappings: 128 distinct sender inputs.",
    "Each sender input has two private actual-delivery A records, giving 256 "
    "fictional rows. A is never included in a sender-visible message. A pairs "
    "have identical inputs; they are not independent model observations.",
    "Gold for the fixed place_b proposition: sender-attributed reader belief "
    "T = M if B else I; actual reader belief R = M if A else I; world = W.",
    "I is independent of M. The prospectively primary I!=M subset has 64 unique "
    "sender inputs (32 mapping-paired factor/template cells). I=M supplies 64 "
    "receipt-invariance controls; its attribution does not depend on B.",
    "I-only and M-only label rules each reach 75% on all 128 inputs and 50% on "
    "the primary I!=M subset. W-only and B-only rules reach 50% on both subsets.",
    "A hidden-actual-reader prediction can achieve at most 75% across all A-paired "
    "rows, and exactly 50% on I!=M. R is identifiable without A when I=M.",
    "Each template crosses every factor independently, removing the earlier "
    "wording/record-polarity association. Templates still differ in names and "
    "places as well as wording and share one conditional-update reasoning family.",
    "Mapping swaps exchange complete physical names globally; they are paired "
    "lexical controls, not independent scenarios or complementary questions.",
    "The old neutral system and exact nested slot-b question are reused. Story "
    "capture includes the candidate proposition but no perspective question. "
    "Both captures end with the model's normal assistant-generation header.",
    "The sender knows the actual world but attributes beliefs according to its "
    "trusted receipt, including when the reader accepts a false location claim. "
    "Possible receipt error and A are not disclosed to the sender.",
    "The message is fixed and sent by a location service. No sender-generated "
    "report, incentive manipulation, omission or unknown future belief is tested.",
    "No new probe fitting, threshold/layer selection or calibration uses this "
    "bank. Before/after transfer localizes a fixed readout; failure before the "
    "question does not prove the model lacks a belief representation.",
)


def _require(value, message):
    if not value:
        raise ValueError(message)


def _bit(value, name):
    _require(type(value) is int and value in (0, 1), name + " must be an integer bit")
    return value


def labels(row, actual_delivery=None):
    """Recompute semantic targets; optionally include the private reader target."""
    _require(isinstance(row, Mapping), "Expected a sender row mapping")
    source = row.get("source")
    _require(isinstance(source, Mapping), "Expected source factors")
    initial, message, receipt, world = (_bit(source.get(k), k) for k in SOURCE_BITS)
    result = {"world": world, "nested": message if receipt else initial}
    if actual_delivery is not None:
        delivered = _bit(actual_delivery, "actual_delivery")
        result["reader"] = message if delivered else initial
    return result


def _frame(row):
    _require(isinstance(row, Mapping), "Expected a sender row mapping")
    _require(row.get("template_id") in _BY_TEMPLATE, "Unknown template")
    _require(type(row.get("mapping_id")) is int, "Mapping must be an integer")
    _require(row["mapping_id"] in (0, 1), "Unknown physical mapping")
    original = _BY_TEMPLATE[row["template_id"]]
    places = (original.place_a, original.place_b)[:: -1 if row["mapping_id"] else 1]
    return _Frame(original.template_id, original.reader, original.sender, *places, original.wording)


def build_messages(row, queried):
    """Render only sender-available evidence; stored gold/A never affect text."""
    _require(type(queried) is bool, "queried must be bool")
    frame = _frame(row)
    labels(row)
    source = row["source"]
    initial, message, receipt, world = (source[k] for k in SOURCE_BITS)
    places = (frame.place_a, frame.place_b)
    initial_place, message_place, world_place = (places[x] for x in (initial, message, world))
    reader, sender, n = frame.reader, frame.sender, frame.wording
    openings = (
        f"At 09:00, {reader} and {sender} inspect the token together inside the "
        f"{initial_place}. Each sees the other complete this inspection. "
        f"{reader} believes the token is in the {initial_place}, and {sender} knows this.",
        f"The 09:00 inventory is witnessed jointly by {sender} and {reader}: "
        f"the token is in the {initial_place}. Both watch each other's inspection. "
        f"{sender} knows that {reader} believes it is in the {initial_place}.",
        f"During their shared 09:00 check, {reader} and {sender} look directly "
        f"inside the {initial_place} and find the token. They notice each other "
        f"looking. {reader}'s belief is that the token is in the {initial_place}; "
        f"{sender} knows that initial belief.",
        f"At the joint inspection at 09:00, the token is visible in the "
        f"{initial_place} to both {sender} and {reader}. Each observes the other's "
        f"inspection. {reader} believes the token is in the {initial_place}, "
        f"which {sender} knows.",
    )
    worlds = (
        f"At 09:03, {sender}'s private live camera shows the token in the "
        f"{world_place}. That view is accurate and unavailable to {reader}.",
        f"An accurate object feed at 09:03 shows {sender} that the token is "
        f"in the {world_place}. {reader} cannot access this feed.",
        f"At 09:03, {sender} alone sees an accurate current image of the token "
        f"inside the {world_place}. {reader} has no view of this image.",
        f"The private camera check at 09:03 establishes for {sender} that the "
        f"token is in the {world_place}. The image is live and correct, and "
        f"{reader} cannot see it.",
    )
    exact_message = f"The token is now in the {message_place}."
    outgoing = (
        f"At 09:05, a location service attempts to transmit this exact text to "
        f'{reader}: "{exact_message}" {sender} can inspect the outgoing text '
        "but cannot observe whether it arrives or is read.",
        f"At 09:05, the location service queues an exact message addressed to "
        f'{reader}: "{exact_message}" The queued words are visible to {sender}; '
        "the recipient and the delivery process are not visible.",
        f"The location service's 09:05 outgoing message for {reader} is exactly "
        f'"{exact_message}" {sender} knows these words but has no direct view '
        "of arrival or reading.",
        f"At 09:05, a service makes a transmission attempt to {reader}, using "
        f'the exact words "{exact_message}" {sender} sees that outgoing message '
        "without seeing delivery or reading.",
    )
    receipt_yes = (
        f"The complete 09:05 message reached {reader} and was read in full at 09:06.",
        f"{reader} received the whole 09:05 text and fully read it at 09:06.",
        f"At 09:06, {reader} read every word of the delivered 09:05 message.",
        f"The entire 09:05 message was delivered to and completely read by {reader} at 09:06.",
    )
    receipt_no = (
        f"Neither the 09:05 message nor a notification of it reached {reader}.",
        f"{reader} received none of the 09:05 text and no notification of its existence.",
        f"No content or notification from the 09:05 attempt was delivered to {reader}.",
        f"Neither any part of the 09:05 message nor any arrival notice reached {reader}.",
    )
    receipts = (
        f"At 09:10, {sender} receives a delivery receipt stating: ",
        f"The receipt available to {sender} at 09:10 records: ",
        f"At 09:10, a transmission record is supplied to {sender}; it says: ",
        f"A status entry provided to {sender} at 09:10 reports: ",
    )
    rules = (
        f"After 09:00, {reader} has no direct observation of the token and no "
        "location evidence except this one possible service message. If the "
        "message is fully read, the reader accepts its stated location and replaces "
        "the initial location belief, even if the message is false. If no message "
        "or notification arrives, the reader retains the initial location belief. "
        f"{sender} knows these update rules."
    )
    receipt_text = (
        receipts[n]
        + (receipt_yes[n] if receipt else receipt_no[n])
        + f" {sender} accepts this particular receipt as definitive evidence of "
        f"whether {reader} read the message. {sender} has no other delivery "
        "evidence and cannot independently check delivery."
    )
    closing = (
        "The scene ends at 09:20. The token has stayed at its 09:03 location "
        "since that camera check. There have been no further observations, "
        "messages, receipts or conversations. No direct observation or independent "
        f"delivery account is available to {sender}."
    )
    candidate = (
        f'Candidate proposition: "The token is in the {frame.place_b}." '
        "This is a statement to assess, not new evidence or a message received "
        "by either person."
    )
    parts = [openings[n], rules, worlds[n], outgoing[n], receipt_text, closing, candidate]
    parts.append("End of account. A question may follow.")
    if queried:
        parts.append("Question: " + _questions(frame)["nested"])
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def _sender_row(frame, factors, mapping):
    source = dict(zip(SOURCE_BITS, factors, strict=True))
    initial, message, receipt, world = factors
    row = {
        "id": f"{frame.template_id}_i{initial}_m{message}_b{receipt}_w{world}_map{mapping}",
        "template_id": frame.template_id,
        "mapping_id": mapping,
        "source": source,
        "reader_name": frame.reader,
        "sender_name": frame.sender,
        "place_a": (frame.place_a, frame.place_b)[mapping],
        "place_b": (frame.place_b, frame.place_a)[mapping],
        "claim_slot": 1,
        "stratum": "receipt_sensitive" if initial != message else "receipt_invariant",
        "gold": message if receipt else initial,
        "needs_human_review": True,
    }
    row["messages"] = build_messages(row, queried=True)
    row["story_messages"] = build_messages(row, queried=False)
    return row


def build_bank():
    """Build the complete draft factorial, independently of model outputs."""
    visible = [
        _sender_row(frame, factors, mapping)
        for frame, factors, mapping in itertools.product(
            _FRAMES, itertools.product((0, 1), repeat=4), (0, 1)
        )
    ]
    hidden = [
        {
            "id": f"{row['id']}_a{actual}",
            "sender_input_id": row["id"],
            "actual_delivery": actual,
            "labels": labels(row, actual),
        }
        for row, actual in itertools.product(visible, (0, 1))
    ]
    return {
        "schema": SCHEMA,
        "primary_head": HEAD,
        "source_bits": list(SOURCE_BITS),
        "conditions": list(CONDITIONS),
        "annotation_origin": "agent_assisted_draft",
        "needs_human_review": True,
        "model_outputs_seen": False,
        "assumptions": list(ASSUMPTIONS),
        "sender_inputs": visible,
        "hidden_delivery_rows": hidden,
    }


def validate_bank(bank):
    """Reject missing cells, altered labels, hidden leakage and unreviewed edits.

    Equality to the authored deterministic build intentionally binds both text
    and metadata. The runner additionally freezes the actual input byte hash.
    """
    _require(isinstance(bank, Mapping), "Expected a bank mapping")
    _require(bank == build_bank(), "Crossed bank differs from the authored full factorial")
    visible = bank["sender_inputs"]
    _require(len(visible) == 128, "Need 128 unique sender rows")
    _require(len({r["id"] for r in visible}) == 128, "Duplicate sender IDs")
    _require(Counter(r["gold"] for r in visible) == {0: 64, 1: 64}, "Gold imbalance")
    for queried in (False, True):
        rendered = [json.dumps(build_messages(r, queried), sort_keys=True) for r in visible]
        _require(len(set(rendered)) == 128, "Duplicate visible prompts")
    return bank


def load_bank(path):
    """Read and validate the exact authored draft, never infer missing cells."""
    return validate_bank(json.loads(Path(path).read_text(encoding="utf-8")))


def sender_rows(bank):
    """Return unique sender inputs only; private A records remain separate."""
    validate_bank(bank)
    return copy.deepcopy(bank["sender_inputs"])
