"""One fixed reader-only evidence-role diagnostic; no hidden world or model calls."""

from __future__ import annotations

import copy
import json
from collections import Counter
from datetime import date
from pathlib import Path

from .calibrated_reader import policy_metadata, worked_examples
from .forecast_panel import READER_INSTRUCTION
from .storage import canonical_hash

CONDITIONS = ("decisive", "metadata", "explicit_missing")
SIDES = ("low", "high")


def render(case, condition, side):
    if condition not in CONDITIONS or side not in SIDES:
        raise ValueError("Unknown evidence-role condition or side")
    entity, value = case["entity"], case[side]
    if case["family"] == "date":
        rule = (
            f"For {entity}, APPROVE if the completion date is on or before {case['boundary']}; "
            "REJECT if completion was after that deadline."
        )
        decisive = f"The completion date for {entity} is {value}."
        administrative = f"The administrative file registration date for {entity} is {value}."
        missing = f"The completion date for {entity} is not provided."
        answer = (
            "APPROVE"
            if date.fromisoformat(value) <= date.fromisoformat(case["boundary"])
            else "REJECT"
        )
    elif case["family"] == "score":
        rule = (
            f"For {entity}, APPROVE if the evaluation score is at least {case['boundary']} points; "
            "REJECT if the evaluation score is below that threshold."
        )
        decisive = f"The evaluation score for {entity} is {value} points."
        administrative = f"The administrative file-ID number for {entity} is {value}."
        missing = f"The evaluation score for {entity} is not provided."
        answer = "APPROVE" if value >= case["boundary"] else "REJECT"
    else:
        raise ValueError("Unknown evidence-role family")
    packet = [decisive] if condition == "decisive" else [administrative]
    if condition == "explicit_missing":
        packet.append(missing)
    return {
        "rule": rule,
        "evidence_packet": packet,
        "expected_decision": answer if condition == "decisive" else "DEFER",
    }


def prefix():
    return [
        {"role": "system", "content": READER_INSTRUCTION},
        *[message for example in worked_examples() for message in example["messages"]],
    ]


def validate_panel(panel):
    if (
        set(panel)
        != {
            "schema_version",
            "panel_id",
            "status",
            "authoring_notes",
            "reader_policy_sha256",
            "cases",
        }
        or panel["schema_version"] != 1
        or panel["panel_id"] != "evidence_role_substitution_v1"
    ):
        raise ValueError("Expected frozen evidence-role panel schema 1")
    if panel["reader_policy_sha256"] != policy_metadata()["policy_sha256"]:
        raise ValueError("The frozen nine-shot reader policy changed")
    if not panel["status"] or not panel["authoring_notes"] or len(panel["cases"]) != 16:
        raise ValueError("Expected sixteen documented cases")
    if Counter(c["family"] for c in panel["cases"]) != {"date": 8, "score": 8}:
        raise ValueError("Expected eight date and eight score cases")
    ids, entities = set(), set()
    for case in panel["cases"]:
        if set(case) != {"id", "family", "entity", "boundary", "low", "high", "conditions"}:
            raise ValueError("Unexpected evidence-role case fields")
        if case["id"] in ids or case["entity"] in entities:
            raise ValueError("Case IDs and entities must be unique")
        ids.add(case["id"])
        entities.add(case["entity"])
        if case["family"] == "date":
            values = [date.fromisoformat(case[k]) for k in ("low", "boundary", "high")]
        else:
            values = [case[k] for k in ("low", "boundary", "high")]
            if any(type(v) is not int or v < 0 for v in values):
                raise ValueError("Scores and file IDs must be nonnegative integers")
        if not values[0] < values[1] < values[2]:
            raise ValueError("Low and high must straddle the same fixed boundary")
        expected = {
            f"{condition}/{side}": render(case, condition, side)
            for condition in CONDITIONS
            for side in SIDES
        }
        if case["conditions"] != expected:
            raise ValueError("Evidence packet or normative role label differs from exact templates")
    for family in ("date", "score"):
        if [c["id"] for c in panel["cases"] if c["family"] == family] != [
            f"er_{family}_{i:02d}" for i in range(1, 9)
        ]:
            raise ValueError("Case identities must preserve frozen authored order")
    return copy.deepcopy(panel)


def authored_panel():
    names = ("Aster", "Birch", "Clover", "Dune", "Elder", "Flint", "Grove", "Heath")
    cases = []
    for family in ("date", "score"):
        for i, name in enumerate(names, 1):
            boundary = f"2022-{i:02d}-15" if family == "date" else 900019 + 211 * i
            case = {
                "id": f"er_{family}_{i:02d}",
                "family": family,
                "entity": f"Evidence {family.title()} {name}",
                "boundary": boundary,
                "low": f"2022-{i:02d}-13" if family == "date" else boundary - 31,
                "high": f"2022-{i:02d}-17" if family == "date" else boundary + 37,
            }
            case["conditions"] = {
                f"{condition}/{side}": render(case, condition, side)
                for condition in CONDITIONS
                for side in SIDES
            }
            cases.append(case)
    return validate_panel(
        {
            "schema_version": 1,
            "panel_id": "evidence_role_substitution_v1",
            "reader_policy_sha256": policy_metadata()["policy_sha256"],
            "status": "Prospective reader-only failure diagnostic; no outputs observed.",
            "authoring_notes": [
                "Six conditions per case cross two value sides with decisive, administrative metadata, or metadata plus explicit missing-decisive-attribute wording.",
                "Date comparisons use completion dates versus administrative registration dates; score comparisons use evaluation scores versus administrative file-ID numbers. A missing target attribute supports DEFER.",
                "The exact original nine-shot prefix/system/checkpoint/parser/eight-token budget remain frozen. This panel neither qualifies a reader nor licenses probe fitting or ct generation.",
                "All96conditions are reported from one batch. Repeated conditions share16case clusters and two authored grammars; no broad population or intent claim.",
            ],
            "cases": cases,
        }
    )


def load_panel(path):
    return validate_panel(json.loads(Path(path).read_text(encoding="utf-8")))


def jobs(panel):
    panel = validate_panel(panel)
    result = []
    for case in panel["cases"]:
        for condition in CONDITIONS:
            for side in SIDES:
                content = case["conditions"][f"{condition}/{side}"]
                user = (
                    "Decision rule:\n"
                    + content["rule"]
                    + "\n\nEvidence packet:\n"
                    + "\n".join(content["evidence_packet"])
                    + "\n\nReport:\nNo report was supplied."
                )
                result.append(
                    {
                        "example_id": "er_"
                        + canonical_hash([panel["panel_id"], case["id"], condition, side])[:24],
                        "case_id": case["id"],
                        "family": case["family"],
                        "condition": condition,
                        "side": side,
                        "expected_decision": content["expected_decision"],
                        "messages": [*prefix(), {"role": "user", "content": user}],
                    }
                )
    return result
