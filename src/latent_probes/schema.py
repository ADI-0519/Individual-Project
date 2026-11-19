"""Validated, explicit records for the development communication experiment.

Validation checks structure and references. It cannot certify that prose facts
are true, mutually consistent, or sufficient to answer the question; those
judgements require a human audit before a dataset becomes evaluation material.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ScenarioValidationError(ValueError):
    """An invalid scenario record or dataset."""


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScenarioValidationError(f"{field} must be a nonempty string")
    return value


def _identifier(value: Any, field: str) -> str:
    _nonempty(value, field)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value) is None:
        raise ScenarioValidationError(
            f"{field} must contain only letters, numbers, underscores or hyphens"
        )
    return value


@dataclass(frozen=True)
class Fact:
    id: str
    text: str

    def __post_init__(self) -> None:
        _identifier(self.id, "fact.id")
        _nonempty(self.text, "fact.text")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Fact:
        if not isinstance(value, Mapping) or set(value) != {"id", "text"}:
            raise ScenarioValidationError("each fact must have exactly id and text")
        return cls(id=value["id"], text=value["text"])

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "text": self.text}


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    family_id: str
    domain: str
    decision_rule: str
    question: str
    facts: tuple[Fact, ...]
    critical_fact_id: str
    full_information_answer: str
    selective_fact_ids: tuple[str, ...]
    control_messages: dict[str, str]
    notes: str = ""

    def __post_init__(self) -> None:
        for field in ("scenario_id", "family_id", "domain", "critical_fact_id"):
            _identifier(getattr(self, field), field)
        _nonempty(self.decision_rule, "decision_rule")
        _nonempty(self.question, "question")
        if not isinstance(self.facts, tuple) or not self.facts:
            raise ScenarioValidationError("facts must be a nonempty tuple of Fact records")
        if any(not isinstance(fact, Fact) for fact in self.facts):
            raise ScenarioValidationError("facts must contain only Fact records")
        fact_ids = [fact.id for fact in self.facts]
        if len(set(fact_ids)) != len(fact_ids):
            raise ScenarioValidationError("fact ids must be unique within a scenario")
        if fact_ids.count(self.critical_fact_id) != 1:
            raise ScenarioValidationError("critical_fact_id must refer to exactly one fact")
        if not isinstance(
            self.full_information_answer, str
        ) or self.full_information_answer not in {"yes", "no"}:
            raise ScenarioValidationError("full_information_answer must be yes or no")
        if not isinstance(self.selective_fact_ids, tuple) or not self.selective_fact_ids:
            raise ScenarioValidationError("selective_fact_ids must be a nonempty tuple")
        if any(not isinstance(fact_id, str) for fact_id in self.selective_fact_ids):
            raise ScenarioValidationError("selective_fact_ids must contain strings")
        if len(set(self.selective_fact_ids)) != len(self.selective_fact_ids):
            raise ScenarioValidationError("selective_fact_ids must not contain duplicates")
        if not set(self.selective_fact_ids).issubset(fact_ids):
            raise ScenarioValidationError("selective_fact_ids refer to missing facts")
        if self.critical_fact_id in self.selective_fact_ids:
            raise ScenarioValidationError("selective controls must omit the critical fact")
        if not isinstance(self.control_messages, dict) or set(self.control_messages) != {
            "full_disclosure",
            "selective",
        }:
            raise ScenarioValidationError(
                "control_messages must have exactly full_disclosure and selective"
            )
        for key, message in self.control_messages.items():
            _nonempty(message, f"control_messages.{key}")
        if not isinstance(self.notes, str):
            raise ScenarioValidationError("notes must be a string")

    @property
    def critical_fact(self) -> Fact:
        return next(fact for fact in self.facts if fact.id == self.critical_fact_id)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Scenario:
        if not isinstance(value, Mapping):
            raise ScenarioValidationError("scenario must be a JSON object")
        required = {
            "scenario_id",
            "family_id",
            "domain",
            "decision_rule",
            "question",
            "facts",
            "critical_fact_id",
            "full_information_answer",
            "selective_fact_ids",
            "control_messages",
        }
        missing = required - set(value)
        unknown = set(value) - required - {"notes"}
        if missing or unknown:
            raise ScenarioValidationError(
                f"scenario keys: missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if not isinstance(value["facts"], list):
            raise ScenarioValidationError("facts must be a JSON array")
        if not isinstance(value["selective_fact_ids"], list):
            raise ScenarioValidationError("selective_fact_ids must be a JSON array")
        if not isinstance(value["control_messages"], dict):
            raise ScenarioValidationError("control_messages must be a JSON object")
        return cls(
            scenario_id=value["scenario_id"],
            family_id=value["family_id"],
            domain=value["domain"],
            decision_rule=value["decision_rule"],
            question=value["question"],
            facts=tuple(Fact.from_dict(fact) for fact in value["facts"]),
            critical_fact_id=value["critical_fact_id"],
            full_information_answer=value["full_information_answer"],
            selective_fact_ids=tuple(value["selective_fact_ids"]),
            control_messages=dict(value["control_messages"]),
            notes=value.get("notes", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "family_id": self.family_id,
            "domain": self.domain,
            "decision_rule": self.decision_rule,
            "question": self.question,
            "facts": [fact.to_dict() for fact in self.facts],
            "critical_fact_id": self.critical_fact_id,
            "full_information_answer": self.full_information_answer,
            "selective_fact_ids": list(self.selective_fact_ids),
            "control_messages": dict(self.control_messages),
            "notes": self.notes,
        }


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Read JSONL, reject malformed/duplicate records, and retain file order."""
    path = Path(path)
    scenarios: list[Scenario] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = Scenario.from_dict(json.loads(raw_line))
            except (ValueError, TypeError, KeyError) as exc:
                raise ScenarioValidationError(f"{path}:{line_number}: {exc}") from exc
            if record.scenario_id in seen:
                raise ScenarioValidationError(
                    f"{path}:{line_number}: duplicate scenario_id {record.scenario_id}"
                )
            seen.add(record.scenario_id)
            scenarios.append(record)
    if not scenarios:
        raise ScenarioValidationError(f"{path}: no scenarios found")
    return scenarios
