"""Fresh, paired-renderer contexts for a development-selected readout study.

This module creates authored inputs and formal labels, never model observations.
The original source messages/IDs are reused exactly; old outcomes are not inputs.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from itertools import product
from pathlib import Path

from .main_study_bank import SYSTEM, training_groups
from .main_study_bank import iter_rows as original_rows
from .main_study_bank import load_bank as original_bank
from .main_study_bank import render_messages as original_render
from .main_study_design import BeliefCase, correction_states, matched_training_allocation

ROOT = Path(__file__).resolve().parents[2]
PARTITIONS = ("source_fit", "source_dev", "correction_fit", "correction_dev", "test")
COUNTS = dict(zip(PARTITIONS, (512, 512, 512, 512, 4992), strict=True))
SCHEMA = "fresh-readout-bank-v2"
SYSTEM_B = (
    "Interpret this fictional account using only the evidence explicitly available to each person. "
    "A character does not learn a narrator-only fact or an unmentioned message. People retain a "
    "location belief until an observation or a trusted, fully read location report replaces it. "
    "Use the stated event times to order updates, even if paragraphs describe them in another order."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def relative_binding(path):
    path = Path(path).resolve()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)}


def verify_binding(binding):
    relative = Path(binding["path"])
    require(
        not relative.is_absolute() and ".." not in relative.parts, "Input path escapes workspace"
    )
    path = ROOT / relative
    require(digest(path) == binding["sha256"], "Bound input changed: " + str(path))
    return path


def _entities(groups):
    actors, locations = set(), set()
    for group in groups:
        for field in ("sender", "reader"):
            components = re.findall(r"[A-Za-z]+", group[field].casefold())
            require(len(components) == 2, "Use two-word actor names consistently across cohorts")
            require(not actors.intersection(components), "Actor component reused in fresh contexts")
            actors.update(components)
        for location in group["locations"]:
            require(
                location.isascii() and 2 <= len(location.split()) <= 4,
                "Use natural ASCII location names of two to four words",
            )
            require(location.casefold() not in locations, "Fresh physical location reused")
            locations.add(location.casefold())
    return actors, locations


def build_bank(train_dev_path, test_path, source_path=None):
    source_path = Path(source_path or ROOT / "data/belief_main/main_v1/main_bank_v1.json")
    source = original_bank(source_path)
    authored = [read(train_dev_path), read(test_path)]
    for packet in authored:
        require(
            packet["schema_version"] == "fresh-readout-authored-contexts-v1"
            and packet["status"] == "draft_unreviewed"
            and packet["human_review_complete"] is False
            and packet["model_observations"] == 0,
            "Authored inputs must remain explicitly unreviewed and unmeasured",
        )
    allocation = matched_training_allocation(training_groups(source), training_n=512, seed=11)
    groups, identity_assignment = assign_actor_pairs(authored)
    result = {
        "schema": SCHEMA,
        "status": "draft_unreviewed",
        "human_review_complete": False,
        "model_observations": 0,
        "seed": 20261122,
        "research_status": "agent_audited_exploratory_pending_semantic_review",
        "independence": "48 contextual groups, three task conditions and two fixed renderers; shared mechanisms are not independent reasoning families",
        "branch": "twelve_site_source_development_selection",
        "C2": "fourfactor_lr",
        "source_bank": relative_binding(source_path),
        "authored_inputs": [relative_binding(train_dev_path), relative_binding(test_path)],
        "source_fit_case_ids": [c.case_id for c in allocation.full],
        "source_dev_case_ids": [r["case_id"] for r in original_rows(source, "development")],
        "groups": groups,
        "identity_assignment": identity_assignment,
        "engineering_group_ids_excluded": ["fresh_engineering_excluded_v2"],
    }
    validate_bank(result)
    return result


def assign_actor_pairs(authored):
    """Balance each author's name pool across cohorts without reusing any name.

    Contexts keep their authored split. Only placeholder actor pairs are assigned
    from two independently shuffled pools; this is fixed before model outcomes.
    """
    groups = [dict(g) for packet in authored for g in packet["groups"]]
    pools = []
    for author, packet in enumerate(authored):
        pool = sorted(
            packet["groups"],
            key=lambda g: sha256(
                f"fresh-actor-pairs-v2|20261122|{author}|{g['group_id']}".encode()
            ).digest(),
        )
        require(len(pool) == 24, "Need two author pools of 24 actor pairs")
        pools.append(pool)
    cursors = [0, 0]
    assignment = []
    for cohort in ("correction_fit", "correction_dev", "test"):
        cohort_groups = sorted(
            (g for g in groups if g["split"] == cohort), key=lambda g: g["group_id"]
        )
        for index, group in enumerate(cohort_groups):
            author = index % 2
            origin = pools[author][cursors[author]]
            cursors[author] += 1
            group.update(sender=origin["sender"], reader=origin["reader"])
            assignment.append(
                {
                    "group_id": group["group_id"],
                    "actor_pair_origin": origin["group_id"],
                    "author_pool": author,
                    "sender": origin["sender"],
                    "reader": origin["reader"],
                }
            )
    require(cursors == [24, 24], "Every actor pair must be assigned once")
    return groups, {
        "policy": "SHA256 actor-pair pools, seed20261122, alternate author pools within each cohort",
        "records": assignment,
    }


def load_bank(path):
    bank = read(path)
    validate_bank(bank)
    return bank


def validate_bank(bank):
    require(
        bank["schema"] == SCHEMA and bank["status"] == "draft_unreviewed",
        "Wrong fresh bank schema/status",
    )
    require(
        bank["human_review_complete"] is False and bank["model_observations"] == 0,
        "Bank metadata cannot fabricate human review or observations",
    )
    require(
        bank["branch"] == "twelve_site_source_development_selection"
        and bank["C2"] == "fourfactor_lr",
        "Wrong outcome-independent branch",
    )
    source = original_bank(verify_binding(bank["source_bank"]))
    for binding in bank["authored_inputs"]:
        verify_binding(binding)
    authored = [read(verify_binding(binding)) for binding in bank["authored_inputs"]]
    assigned, assignment = assign_actor_pairs(authored)
    require(
        bank["groups"] == assigned and bank["identity_assignment"] == assignment,
        "Compiled contexts/actor assignments differ from authored source policy",
    )
    groups = bank["groups"]
    require(
        len(groups) == len({g["group_id"] for g in groups}) == 48, "Need 48 distinct fresh contexts"
    )
    require(
        Counter(g["split"] for g in groups)
        == Counter(correction_fit=16, correction_dev=8, test=24),
        "Incorrect contextual roster",
    )
    require(len({g["wording_group"] for g in groups}) == 48, "Duplicate contextual wording group")
    actors, locations = _entities(groups)
    old_actors = {
        c.casefold()
        for g in source["groups"]
        for k in ("sender", "reader")
        for c in re.findall(r"[A-Za-z]+", g[k])
    }
    old_places = {p.casefold() for g in source["groups"] for p in g["locations"]}
    require(
        not actors.intersection(old_actors) and not locations.intersection(old_places),
        "Fresh/old entity overlap",
    )
    framings = set()
    for group in groups:
        require(
            len(group["locations"]) == 2 and group["sender"] != group["reader"],
            "Invalid actor/location binding",
        )
        require(
            re.fullmatch(r"fresh_(correction_fit|correction_dev|test)_\d{2}", group["group_id"])
            is not None,
            "Invalid fresh group identifier",
        )
        framing = " ".join(
            (group["introduction"] + " " + group["event_description"]).casefold().split()
        )
        require(framing not in framings, "Duplicate fresh contextual framing")
        framings.add(framing)
        for key in ("introduction", "event_description"):
            require(
                group[key].isascii()
                and group[key].format(sender=group["sender"], reader=group["reader"]),
                "Invalid authored framing",
            )
    expected = matched_training_allocation(training_groups(source), training_n=512, seed=11)
    require(
        bank["source_fit_case_ids"] == [c.case_id for c in expected.full],
        "Source fitting allocation changed",
    )
    require(
        bank["source_dev_case_ids"] == [r["case_id"] for r in original_rows(source, "development")],
        "Source development IDs changed",
    )
    return {
        "groups": 48,
        "source_fit_rows": 512,
        "source_dev_rows": 512,
        "human_review_complete": False,
    }


def locations(group, mapping):
    return group["locations"][::-1] if mapping else group["locations"]


def attach_spans(messages, query_prefix, queried_location, query_suffix):
    require(len(messages) == 2 and messages[1]["role"] == "user", "Need one explicit user account")
    content = messages[1]["content"]
    account, separator, question = content.rpartition("\n\n")
    require(
        separator and account and question == query_prefix + queried_location + query_suffix,
        "Explicit final question differs",
    )
    question_start = len(account) + len(separator)
    start = question_start + len(query_prefix)
    return {
        "user_content_char_spans": {
            # Include the paragraph separator: strictly before the question, including any separator token.
            "account": [0, question_start],
            "query_location": [start, start + len(queried_location)],
        },
        "span_boundary_policy": {"query_location": "leading_whitespace"},
    }


def _question(group, place, renderer):
    if renderer == "A":
        return (
            f"At the end of the scene, does {group['sender']} think that {group['reader']} believes that the token is in the ",
            place,
            "? Answer only YES or NO.",
        )
    return (
        f"At 09:20, is it {group['sender']}'s belief that {group['reader']} believes that the token is in the ",
        place,
        "? Respond with YES or NO.",
    )


def render(group, state, renderer, condition):
    require(renderer in ("A", "B"), "Unknown renderer")
    places = locations(group, state["name_map"])
    sender, reader = group["sender"], group["reader"]
    prefix, target, suffix = _question(group, places[1], renderer)
    if renderer == "A" and condition != "asserted":
        case = BeliefCase(**state)
        messages = original_render({**group, "mechanism": case.mechanism}, case)
        return messages, attach_spans(messages, prefix, target, suffix)
    lines = [group["introduction"].format(sender=sender, reader=reader)]
    if condition == "asserted":
        if renderer == "A":
            lines += [
                f"At 09:15, {sender} alone checks the physical token: it is in the {places[state['world']]}. {reader} cannot access this check and gets no notice. This private location remains unchanged through 09:20.",
                f"The account explicitly states the final attribution at 09:20: {sender} believes that {reader} currently believes that the token is in the {places[state['asserted']]}. This states {sender}'s attribution directly, without a sequence of earlier reports to infer. It does not state {reader}'s actual private belief or the physical truth.",
                "The attribution is a narrator statement about a character's belief, not a new message supplied to either character.",
            ]
        else:
            lines += [
                f"Directly supplied final belief: at 09:20, {sender} attributes to {reader} the belief that the token is in the {places[state['asserted']]}. Take this as an explicit description of {sender}'s current attribution, not as a report delivered to {reader}. Neither the actual private belief nor the physical location is determined by that attribution.",
                f"Physical-state note, 09:15: a private check available only to {sender} found the token in the {places[state['world']]}, where it stays until 09:20. No notification or result of this check reaches {reader}.",
                "There is no earlier location-message sequence to reconstruct for this directly stated attribution.",
            ]
    else:
        initial, update = (places[state[k]] for k in ("initial", "update"))
        lines += [
            f"Interpretation rule known to {sender}: {reader} accepts every fully read location report as a replacement belief, including a false report. Among reports read, the latest event time determines priority. An unread report gives no content and no notification. {sender} reconstructs {reader}'s belief using the accepted reading records alone; their agreement with actual private reading is not disclosed.",
            group["event_description"].format(sender=sender, reader=reader),
            f"Report event, 09:05: {sender} writes privately, 'The token is in the {update}.' The written location can disagree with the actual physical location.",
            f"Earlier shared baseline, 09:00: both {sender} and {reader} inspect the token in the {initial} and see the other complete that inspection. At this baseline {reader}'s belief is the {initial}, and {sender} knows this belief.",
        ]
        first = state["first_receipt"] if state["mechanism"] == "correction" else state["receipt"]
        access = (
            f"{reader} read the full 09:05 location report"
            if first
            else f"{reader} obtained neither the content of the 09:05 report nor a notification about it"
        )
        lines.append(
            f"Record event, 09:07: {sender} receives and accepts as definitive this reading entry: {access}."
        )
        if state["mechanism"] == "correction":
            replacement = places[1 - state["update"]]
            lines.append(
                f"Replacement event, 09:10: {sender} writes a separate report, 'Use the {replacement} as the token location in place of any earlier location report you read.' This report supplies a replacement location; it does not merely cancel an earlier claim."
            )
            access = (
                f"{reader} read the full 09:10 replacement report"
                if state["receipt"]
                else f"{reader} obtained neither the replacement report's content nor a notification about it"
            )
            lines.append(
                f"Record event, 09:12: {sender} accepts as definitive a second reading entry: {access}."
            )
        lines += [
            f"Private physical check, 09:15: {sender} alone accurately finds the token in the {places[state['world']]}. {reader} has no access or notification. Any movement between checks took place outside {reader}'s view without a movement notice; the privately checked position persists until 09:20.",
            f"For 09:20, there is no other observation, message or location evidence for {reader} beyond what the accepted records describe. {sender} retains those records. Neither character infers an unseen move.",
        ]
    if renderer == "A":
        lines += [
            f'Candidate proposition: "The token is in the {places[1]}." This is a statement to assess, not new evidence or a message received by either person.',
            "End of account. A question may follow.",
        ]
    else:
        lines += [
            f'Proposition for assessment: "The token is in the {places[1]}." Quoting this proposition gives neither character new evidence or a location message.',
            "Account closed. The question follows separately.",
        ]
    lines.append(prefix + target + suffix)
    messages = [
        {"role": "system", "content": SYSTEM if renderer == "A" else SYSTEM_B},
        {"role": "user", "content": "\n\n".join(lines)},
    ]
    return messages, attach_spans(messages, prefix, target, suffix)


def _fresh_row(group, state, renderer, condition):
    if condition == "asserted":
        bits = f"e{state['asserted']}_w{state['world']}_map{state['name_map']}"
        gold = state["asserted"]
    else:
        case = BeliefCase(**state)
        bits = case.case_id.split("__", 1)[1]
        gold = case.gold
    case_id = f"{group['group_id']}__{condition}_{renderer}__{bits}"
    messages, spans = render(group, state, renderer, condition)
    return {
        "record_id": case_id,
        "case_id": case_id,
        "group_id": group["group_id"],
        "partition": group["split"],
        "split": group["split"],
        "original_split": None,
        "condition": condition,
        "mechanism": state["mechanism"],
        "renderer": renderer,
        "state": state,
        "gold": gold,
        "messages": messages,
        **spans,
    }


def iter_rows(bank, partition=None):
    require(partition is None or partition in PARTITIONS, "Unknown fresh partition")
    source = original_bank(verify_binding(bank["source_bank"]))
    source_by_id = {
        r["case_id"]: r for r in original_rows(source) if r["split"] in ("train", "development")
    }
    for part, key in (("source_fit", "source_fit_case_ids"), ("source_dev", "source_dev_case_ids")):
        if partition is not None and partition != part:
            continue
        for identity in bank[key]:
            row = source_by_id[identity]
            group = next(g for g in source["groups"] if g["group_id"] == row["group_id"])
            place = locations(group, row["state"]["name_map"])[1]
            prefix, _, suffix = _question(group, place, "A")
            yield {
                **row,
                "record_id": identity,
                "partition": part,
                "original_split": row["split"],
                "split": part,
                "condition": "source_" + row["mechanism"],
                "renderer": "A",
                **attach_spans(row["messages"], prefix, place, suffix),
            }
    for group in bank["groups"]:
        if partition is not None and partition != group["split"]:
            continue
        renderers = ("A", "B") if group["split"] == "test" else ("A",)
        for renderer in renderers:
            if group["split"] == "test":
                for i, u, b, w, m in product((0, 1), repeat=5):
                    state = asdict(BeliefCase(group["group_id"], "message", i, u, b, w, m))
                    yield _fresh_row(group, state, renderer, "one_step")
            for case in correction_states(group["group_id"]):
                if group["split"] == "correction_fit":
                    index = int(group["group_id"].rsplit("_", 1)[1])
                    if case.initial ^ case.update ^ case.world ^ case.name_map != index % 2:
                        continue
                condition = (
                    "correction_hard" if case.initial == 1 - case.update else "correction_easy"
                )
                yield _fresh_row(group, asdict(case), renderer, condition)
            if group["split"] == "test":
                for e, w, m in product((0, 1), repeat=3):
                    state = {
                        "group_id": group["group_id"],
                        "mechanism": "asserted",
                        "asserted": e,
                        "world": w,
                        "name_map": m,
                    }
                    yield _fresh_row(group, state, renderer, "asserted")


def audit_rows(bank):
    rows = list(iter_rows(bank))
    require(Counter(r["partition"] for r in rows) == Counter(COUNTS), "Wrong full prefix inventory")
    require(
        len(rows) == len({r["record_id"] for r in rows}) == 7040,
        "Duplicate/missing record identity",
    )
    require(len({canonical(r["messages"]) for r in rows}) == 7040, "Duplicate exact input prose")
    for row in rows:
        text = row["messages"][1]["content"]
        spans = row["user_content_char_spans"]
        require(
            spans["account"][0] == 0
            and 0
            < spans["account"][1]
            <= spans["query_location"][0]
            < spans["query_location"][1]
            < len(text),
            "Invalid span order",
        )
        require(
            row["record_id"] not in text and row["group_id"] not in text,
            "Identifier exposed in source prose",
        )
        require(type(row["gold"]) is int and row["gold"] in (0, 1), "Invalid gold bit")
    counts = Counter((r["condition"], r["renderer"]) for r in rows if r["partition"] == "test")
    require(
        counts
        == Counter(
            {
                (c, fmt): n
                for fmt in ("A", "B")
                for c, n in (
                    ("one_step", 768),
                    ("correction_hard", 768),
                    ("correction_easy", 768),
                    ("asserted", 192),
                )
            }
        ),
        "Wrong fresh test subgroups",
    )
    return {
        "rows": len(rows),
        "partitions": dict(Counter(r["partition"] for r in rows)),
        "fresh_contexts": 48,
        "paired_test_contexts": 24,
        "test_primary_quartets_per_renderer": 192,
        "human_review_complete": False,
    }


def review_packet(bank):
    """Opaque IDs hide all factors; no answers/state/encoded IDs enter blind prompts."""
    rows = list(iter_rows(bank))
    groups = [g for g in bank["groups"] if g["split"] == "test"]
    chosen = []
    # Every normalized I/U/B path in both renderers; nuisance assignments are balanced,
    # then explicit matched nuisance controls are added below without using outcomes.
    for renderer in ("A", "B"):
        for condition in ("one_step", "correction", "asserted"):
            templates = {}
            for row in rows:
                if row["partition"] != "test" or row["renderer"] != renderer:
                    continue
                if condition == "correction" and not row["condition"].startswith("correction_"):
                    continue
                if condition != "correction" and row["condition"] != condition:
                    continue
                s = row["state"]
                key = (
                    (s["asserted"],)
                    if condition == "asserted"
                    else (s["initial"], s["update"], s.get("first_receipt"), s["receipt"])
                )
                templates.setdefault(key, []).append(row)
            for index, key in enumerate(sorted(templates, key=str)):
                group = groups[len(chosen) % len(groups)]["group_id"]
                candidates = [
                    r
                    for r in templates[key]
                    if r["group_id"] == group
                    and r["state"]["world"] == index % 2
                    and r["state"]["name_map"] == (index // 2) % 2
                ]
                require(len(candidates) == 1, "Review path selection ambiguous")
                chosen.append(candidates[0])
    for group in bank["groups"]:
        if not any(r["group_id"] == group["group_id"] for r in chosen):
            chosen.append(next(r for r in rows if r["group_id"] == group["group_id"]))
    # Add world/name counterpart panels for four distinct receipt paths, in each format.
    for renderer in ("A", "B"):
        candidates = [
            r
            for r in rows
            if r["group_id"] == groups[0]["group_id"]
            and r["renderer"] == renderer
            and r["condition"] == "correction_hard"
            and r["state"]["initial"] == 0
        ]
        chosen.extend(candidates)
    unique = {r["record_id"]: r for r in chosen}
    ordered = sorted(
        unique.values(),
        key=lambda r: sha256(("fresh-blind-review-v2|" + r["record_id"]).encode()).digest(),
    )
    packet, key = [], []
    for index, row in enumerate(ordered):
        opaque = f"R{index + 1:03d}"
        packet.append({"review_id": opaque, "messages": row["messages"]})
        key.append(
            {
                "review_id": opaque,
                "record_id": row["record_id"],
                "gold": row["gold"],
                "condition": row["condition"],
                "renderer": row["renderer"],
            }
        )
    return {
        "schema": "fresh-readout-blind-review-v1",
        "human_review_complete": False,
        "base_contexts": [
            {
                k: g[k]
                for k in (
                    "group_id",
                    "sender",
                    "reader",
                    "locations",
                    "introduction",
                    "event_description",
                )
            }
            for g in bank["groups"]
        ],
        "prompts": packet,
    }, {"schema": "fresh-readout-review-key-v1", "records": key}
