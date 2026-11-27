"""Draft narrative bank, deterministic rendering, and fail-closed human review.

Factorial variants are dependent observations. The bank has three mechanisms,
not 154 independent reasoning families. Rendering never exposes symbolic gold.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from latent_probes.main_study_design import (
    BeliefCase,
    TrainingGroup,
    correction_states,
    default_training_groups,
    one_step_states,
    primary_quartets,
)

SYSTEM = (
    "Read the fictional scene. Questions will follow. Each person has only the "
    "observations and messages explicitly available to them. People remember the last "
    "place they observed the token and update after seeing a move or receiving a "
    "trusted message about its current place. They do not infer an unobserved move. "
    "Narrator-only information is not shared with a character merely because the "
    "narrator describes it."
)
SCHEMA_VERSION = 1
SPLITS = ("train", "development", "test")
ROSTER = {
    ("train", "observation"): 24,
    ("train", "message"): 24,
    ("development", "observation"): 8,
    ("development", "message"): 8,
    ("test", "observation"): 30,
    ("test", "message"): 30,
    ("test", "correction"): 30,
}


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256((payload + "\n").encode("utf-8")).hexdigest()


def bank_digest(bank: dict) -> str:
    """Hash canonical JSON, independent of indentation/newline storage choices."""
    return _digest(bank)


def prompt_digest(messages: list[dict]) -> str:
    return _digest(messages)


def load_bank(path: str | Path) -> dict:
    bank = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_bank(bank)
    return bank


def _locations(group: dict, case: BeliefCase) -> tuple[str, str]:
    places = tuple(group["locations"])
    return places[::-1] if case.name_map else places


def render_messages(group: dict, case: BeliefCase) -> list[dict]:
    """Render a fictional account using visible attribution evidence only.

    The sender sees a trusted record, not actual private reader access. U is an
    accurate earlier observation only in observation cases. W is a later private
    check. Message content U may be inaccurate without contradicting the rules.
    """
    if case.group_id != group["group_id"] or case.mechanism != group["mechanism"]:
        raise ValueError("State/group binding mismatch")
    places = _locations(group, case)
    sender, reader = group["sender"], group["reader"]
    initial, update, world = (places[x] for x in (case.initial, case.update, case.world))
    lines = [group["introduction"].format(sender=sender, reader=reader)]
    lines.append(
        f"At 09:00, {sender} and {reader} jointly inspect the token inside the {initial}. "
        f"They see each other complete this inspection. {reader} believes the token is "
        f"in the {initial}, and {sender} knows that initial belief."
    )
    if case.mechanism == "observation":
        lines.append(
            group["event_description"].format(sender=sender, reader=reader)
            + f" At 09:05 the token is actually in the {update}. {sender} receives an "
            "accurate image of the token from this session, with the reader area "
            f"excluded. This gives {sender} the displayed location but no evidence "
            f"about whether {reader} observed the session."
        )
        receipt = (
            f"{reader} directly observed the complete 09:05 session and saw the token "
            f"in the {update}."
            if case.receipt
            else f"{reader} saw no part of the 09:05 session and received no notification "
            "that it occurred."
        )
        lines.append(
            f'At 09:07, an access record supplied to {sender} states: "{receipt}" '
            f"{sender} accepts this particular record as definitive and has no other "
            f"evidence about {reader}'s access. The account does not disclose whether "
            "the access record agrees with actual private access."
        )
        lines.append(
            f"If {reader} directly observes that session, its displayed location "
            f"replaces the initial location belief. Otherwise {reader} retains the "
            f"initial belief. {sender} knows and applies this rule."
        )
    else:
        lines.append(
            group["event_description"].format(sender=sender, reader=reader)
            + f' At 09:05, {sender} sends exactly: "The token is in the {update}." '
            "The report's content is not guaranteed to match the actual token location."
        )
        first_receipt = case.first_receipt if case.mechanism == "correction" else case.receipt
        record = (
            f"{reader} fully read the 09:05 report."
            if first_receipt
            else f"{reader} received no content or notification from the 09:05 report."
        )
        lines.append(
            f'At 09:07, a reading record supplied to {sender} states: "{record}" '
            f"{sender} accepts this particular record as definitive."
        )
        if case.mechanism == "correction":
            final = places[case.final_update]
            lines.append(
                f'At 09:10, {sender} sends a separate replacement report: "Correction: '
                f"the token is in the {final}. Use this location instead of any earlier "
                'location report you read." The replacement explicitly supplies a '
                "location and does not merely withdraw the previous report."
            )
            record = (
                f"{reader} fully read the 09:10 replacement report."
                if case.receipt
                else f"{reader} received no content or notification from the 09:10 replacement report."
            )
            lines.append(
                f'At 09:12, a second reading record supplied to {sender} states: "{record}" '
                f"{sender} accepts this particular record as definitive."
            )
        lines.append(
            f"{reader} trusts each fully read location report. A fully read report "
            "replaces the preceding location belief even when its content is false "
            "in the world. The most recently read report takes priority. A report "
            "that is not read supplies neither content nor a notification; in that "
            f"case the preceding belief is retained. {sender} knows and applies these "
            "rules and has no access evidence besides the accepted records. The "
            "account does not disclose whether those records agree with actual private reading."
        )
    lines.append(
        f"At 09:15, {sender} alone makes an accurate private live check: the token is "
        f"now in the {world}. {reader} cannot access this check and receives no notice "
        "of it. Any movement between the timed checks occurs outside the reader's "
        "view without a movement notice. The token remains in this privately checked "
        "location through 09:20."
    )
    lines.append(
        f"The question concerns 09:20. Apart from any access described in the accepted "
        f"records, {reader} gets no additional observation, report or location evidence. "
        f"{sender} retains those records; neither person infers an unobserved move."
    )
    lines.append(
        f'Candidate proposition: "The token is in the {places[1]}." This is a statement '
        "to assess, not new evidence or a message received by either person."
    )
    lines.append("End of account. A question may follow.")
    lines.append(
        f"At the end of the scene, does {sender} think that {reader} believes that the "
        f"token is in the {places[1]}? Answer only YES or NO."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(lines)}]


def states_for_group(group: dict) -> tuple[BeliefCase, ...]:
    if group["mechanism"] == "correction":
        return correction_states(group["group_id"])
    return one_step_states(group["mechanism"], group["group_id"])


def iter_rows(bank: dict, split: str | None = None) -> Iterator[dict]:
    if split is not None and split not in SPLITS:
        raise ValueError("Unknown split")
    for group in bank["groups"]:
        if split is None or group["split"] == split:
            for case in states_for_group(group):
                yield {
                    "case_id": case.case_id,
                    "group_id": case.group_id,
                    "mechanism": case.mechanism,
                    "split": group["split"],
                    "state": asdict(case),
                    "gold": case.gold,
                    "messages": render_messages(group, case),
                    "wording_group": group["wording_group"],
                }


def case_from_row(row: dict) -> BeliefCase:
    case = BeliefCase(**row["state"])
    if any(row[key] != getattr(case, key) for key in ("case_id", "group_id", "mechanism", "gold")):
        raise ValueError("Row/state binding mismatch")
    return case


def training_groups(bank: dict) -> tuple[TrainingGroup, ...]:
    return tuple(
        TrainingGroup(g["group_id"], g["mechanism"], g["pair_id"], g["offset"])
        for g in bank["groups"]
        if g["split"] == "train"
    )


def validate_bank(bank: dict) -> dict:
    """Check roster, global exchange, all factorials, prose and split bindings."""
    if bank.get("schema_version") != SCHEMA_VERSION or bank.get("status") != "draft_unreviewed":
        raise ValueError("Expected version-1 unreviewed draft; approval lives in a separate ledger")
    groups = bank.get("groups", [])
    if len(groups) != 154 or len({g["group_id"] for g in groups}) != 154:
        raise ValueError("Need exactly 154 distinct group IDs")
    if Counter((g["split"], g["mechanism"]) for g in groups) != Counter(ROSTER):
        raise ValueError("Incorrect split/mechanism roster")
    if set(training_groups(bank)) != set(default_training_groups()):
        raise ValueError("Training IDs/pairs differ from frozen allocation roster")
    entities: dict[str, set[str]] = {split: set() for split in SPLITS}
    prose = set()
    wording = set()
    for group in groups:
        if group["wording_group"] in wording:
            raise ValueError("Wording group reused")
        wording.add(group["wording_group"])
        if len(group["locations"]) != 2 or len(set(group["locations"])) != 2:
            raise ValueError("Need two distinct physical locations")
        if group["sender"] == group["reader"]:
            raise ValueError("Sender and reader must differ")
        for field in ("introduction", "event_description"):
            value = group[field]
            if not isinstance(value, str) or not value.strip() or value in prose:
                raise ValueError("Missing or reused draft prose skeleton")
            prose.add(value)
        entities[group["split"]].update((group["sender"], group["reader"], *group["locations"]))
        cases = states_for_group(group)
        if len(cases) != (64 if group["mechanism"] == "correction" else 32):
            raise ValueError("Incomplete factorial")
        if len(primary_quartets(cases)) != 8:
            raise ValueError("Incorrect primary quartet coverage")
        for case in cases:
            swapped = BeliefCase(**{**asdict(case), "name_map": 1 - case.name_map})
            if case.gold != swapped.gold:
                raise ValueError("Global name exchange changed symbolic gold")
            rendered = render_messages(group, case)[1]["content"]
            a, b = group["locations"]
            exchanged = rendered.replace(a, "<LOCATION-A>").replace(b, a).replace("<LOCATION-A>", b)
            if exchanged != render_messages(group, swapped)[1]["content"]:
                raise ValueError("Physical names were not exchanged globally")
            target = _locations(group, case)[1]
            if not rendered.endswith(f"token is in the {target}? Answer only YES or NO."):
                raise ValueError("Question must target symbolic slot1")
    for index, split in enumerate(SPLITS):
        for other in SPLITS[index + 1 :]:
            if entities[split] & entities[other]:
                raise ValueError("Actor/location entity leaked across splits")
    hashes = [prompt_digest(row["messages"]) for row in iter_rows(bank)]
    if len(hashes) != 5888 or len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate visible message or incorrect total factorial count")
    forbidden = set(bank.get("excluded_group_ids", []))
    if not forbidden or forbidden & {g["group_id"] for g in groups}:
        raise ValueError("Missing or violated excluded design-group roster")
    return {
        "groups": len(groups),
        "prefixes": len(hashes),
        "mechanisms": 3,
        "bank_sha256": bank_digest(bank),
        "human_review_complete": False,
    }


def representative_cases(group: dict) -> tuple[BeliefCase, ...]:
    """Review all normalized logical paths; complements use a documented exchange."""
    return tuple(
        c for c in states_for_group(group) if c.update == 0 and c.world == 0 and c.name_map == 0
    )


def _logical_path(case: BeliefCase) -> str:
    if case.mechanism == "correction":
        return f"correction/initial_matches_first{int(case.initial == case.update)}/b1{case.first_receipt}/b2{case.receipt}"
    return f"{case.mechanism}/changes{int(case.initial != case.update)}/b{case.receipt}"


def verify_review_ledger(bank: dict, ledger_path: str | Path) -> dict:
    """Require actual named human records bound to this bank and rendered prompts.

    Software cannot authenticate humanity or prevent a deliberately falsified
    attestation. This gate prevents accidental execution of a blank/partial bank;
    it does not manufacture independent semantic evidence.
    """
    validate_bank(bank)
    ledger = json.loads(Path(ledger_path).read_text(encoding="utf-8"))
    if ledger.get("schema_version") != 1 or ledger.get("bank_sha256") != bank_digest(bank):
        raise ValueError("Review ledger must bind the exact canonical bank hash")
    from .main_study_review import validate_questionnaire_answers

    mismatches = validate_questionnaire_answers(bank, ledger)
    if mismatches:
        raise ValueError(f"Unresolved independent-answer mismatches: {mismatches}")
    groups = {g["group_id"]: g for g in bank["groups"]}
    accepted: dict[tuple[str, str], dict] = {}
    human_names: dict[str, set[str]] = {"student": set(), "independent": set()}
    disputed = set(ledger.get("disputed_groups", []))
    if not disputed <= groups.keys():
        raise ValueError("Unknown disputed group")
    for record in ledger.get("records", []):
        role, group_id = record.get("role"), record.get("group_id")
        if role not in human_names or group_id not in groups:
            raise ValueError("Unknown reviewer role/group")
        name = record.get("reviewer", "").strip()
        if not name or not isinstance(record.get("notes"), str):
            raise ValueError("Named reviewer and notes are required")
        try:
            reviewed_at = datetime.fromisoformat(record["reviewed_at"].replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Actual timezone-aware review timestamp required") from exc
        if reviewed_at.tzinfo is None or reviewed_at > datetime.now(UTC):
            raise ValueError("Review timestamp must be aware and not in the future")
        decision = record.get("decision")
        if decision not in ("accepted", "disputed", "rejected"):
            raise ValueError("Unknown review decision")
        if decision != "accepted":
            raise ValueError("Unresolved disputed/rejected review record")
        key = (role, group_id)
        if key in accepted:
            raise ValueError("Duplicate review acceptance")
        expected = {
            prompt_digest(render_messages(groups[group_id], c))
            for c in representative_cases(groups[group_id])
        }
        actual = record.get("prompt_sha256s", [])
        if (
            not isinstance(actual, list)
            or len(actual) != len(set(actual))
            or set(actual) != expected
        ):
            raise ValueError("Review must cover every representative logical path")
        accepted[key] = record
        human_names[role].add(name.casefold())
    if human_names["student"] & human_names["independent"]:
        raise ValueError("Independent reviewer must differ from student reviewer")
    missing = [g for g in groups if ("student", g) not in accepted]
    if missing:
        raise ValueError(
            f"Human review incomplete: student acceptance missing for {len(missing)} groups"
        )
    independent = [groups[g] for role, g in accepted if role == "independent"]
    strata = {(g["split"], g["mechanism"]) for g in independent}
    if strata != set(ROSTER):
        raise ValueError("Independent sample must cover every split/mechanism stratum")
    required_paths = {
        _logical_path(c) for group in groups.values() for c in representative_cases(group)
    }
    reviewed_paths = {
        _logical_path(c) for group in independent for c in representative_cases(group)
    }
    if reviewed_paths != required_paths:
        raise ValueError("Independent sample lacks a normalized logical path")
    if any(("independent", g) not in accepted for g in disputed):
        raise ValueError("Every disputed group needs independent accepted reconciliation")
    return {
        "bank_sha256": bank_digest(bank),
        "ledger_sha256": _digest(ledger),
        "student_groups": len(groups),
        "independent_groups": len(independent),
        "logical_paths": len(required_paths),
        "review_complete": True,
        "sampled_review_is_not_full_independent_review": True,
    }


def export_review(bank: dict, directory: str | Path) -> None:
    """Write blinded draft cards, exact review prompts, and empty human annotations."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ledger = {
        "schema_version": 1,
        "bank_sha256": bank_digest(bank),
        "records": [],
        "disputed_groups": [],
    }
    ledger_path = directory / "review_ledger.json"
    if ledger_path.exists() and json.loads(ledger_path.read_text(encoding="utf-8")) != ledger:
        raise ValueError("Refusing to overwrite completed or edited human review ledger")
    annotation_path = directory / "annotations.csv"
    if annotation_path.exists():
        with annotation_path.open(encoding="utf-8", newline="") as handle:
            previous = list(csv.DictReader(handle))
        blank = ("reviewer", "reviewed_at", "decision", "notes", "prompt_sha256s")
        if any(any(row.get(key, "") for key in blank) for row in previous):
            raise ValueError("Refusing to overwrite human annotations")
    lines = [
        "# Main-study draft bank: human semantic review",
        "",
        "**Unreviewed agent-authored draft. No model observations or fitted results.**",
        "154 authored contextual framings share three state mechanisms; they are not "
        "154 independent reasoning families. The common timeline and update-rule prose "
        "are shared by design. Distinct introductory/event prose does not establish "
        "independent semantic difficulty. All labels and model scores are withheld here.",
        "",
        f"Canonical bank SHA256: `{bank_digest(bank)}`.",
        "",
        "The student must inspect every base narrative, representative exact prompt, "
        "and the transformations. An independent named human reviews a stratified "
        "sample covering every split/mechanism and normalized logical path, plus all "
        "disputes. Agreement by agents and passing code checks do not count as human review.",
        "",
        "## Shared system text",
        "",
        SYSTEM,
        "",
        "## Transformation review",
        "",
        "Each base group expands all binary initial location, update location, "
        "accepted-access record, later private world, and global physical-name mappings. "
        "Correction groups additionally cross both access records; the second report "
        "always names the opposite location. Actual private access is never supplied. "
        "Only accepted records determine the sender's attribution. Observation U is "
        "the true earlier 09:05 observed state, with W independently checked at 09:15. "
        "Message contents can differ from W. The candidate always asks about symbolic "
        "slot1, even when its physical name is exchanged. The query is not reader evidence.",
        "",
        "Four representative prompts per one-step group and eight per correction "
        "group cover all normalized logical paths at update slot0/world slot0/map0. "
        "Review their complete messages in [review_prompts.jsonl](./review_prompts.jsonl). "
        "The remaining cells require review of location-complement, world and global "
        "name-exchange rules; they have not each been individually human-read. "
        "Any reviewer may demand additional expanded cells before accepting a group.",
        "",
        "Record independent answers and ambiguities in notes before reconciliation. "
        "Blank [annotations.csv](./annotations.csv) is a worksheet, not approval. "
        "The machine-readable [review_ledger.json](./review_ledger.json) requires named "
        "human acceptances, actual timestamps and every representative prompt hash. "
        "Do not replace review with a boolean or fill accepted fields with agent judgments.",
        "",
    ]
    prompts = []
    with annotation_path.open("w", encoding="utf-8", newline="") as handle:
        columns = [
            "group_id",
            "split",
            "mechanism",
            "representative_count",
            "reviewer",
            "reviewed_at",
            "decision",
            "notes",
            "prompt_sha256s",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for group in bank["groups"]:
            cases = representative_cases(group)
            writer.writerow(
                {
                    "group_id": group["group_id"],
                    "split": group["split"],
                    "mechanism": group["mechanism"],
                    "representative_count": len(cases),
                }
            )
            lines.extend(
                [
                    f"## {group['group_id']}",
                    "",
                    group["introduction"].format(sender=group["sender"], reader=group["reader"]),
                    "",
                    group["event_description"].format(
                        sender=group["sender"], reader=group["reader"]
                    ),
                    "",
                    f"Physical locations: {group['locations'][0]}; {group['locations'][1]}.",
                    "",
                    "Representative exact prompt bindings:",
                    "",
                ]
            )
            for case in cases:
                messages = render_messages(group, case)
                digest = prompt_digest(messages)
                prompts.append(
                    {
                        "case_id": case.case_id,
                        "group_id": case.group_id,
                        "prompt_sha256": digest,
                        "messages": messages,
                    }
                )
                lines.append(f"- `{case.case_id}`: `{digest}`")
            lines.append("")
    (directory / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (directory / "review_prompts.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prompts), encoding="utf-8"
    )
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
