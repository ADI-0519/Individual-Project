"""Compare a blind agent's answers with the separately retained formal key."""

from pathlib import Path

from .fresh_readout_bank import (
    ROOT,
    canonical,
    digest,
    load_bank,
    read,
    require,
    review_packet,
    write_new,
)


def resolve(binding):
    relative = Path(binding["path"])
    require(
        not relative.is_absolute() and ".." not in relative.parts and "\\" not in binding["path"],
        "Review path must be relative POSIX within workspace",
    )
    path = ROOT / relative
    require(digest(path) == binding["sha256"], "Bound review source changed")
    return path


def compare(packet, key, answers):
    require(packet["schema"] == "fresh-readout-blind-review-v1", "Unknown review packet")
    require(key["schema"] == "fresh-readout-review-key-v1", "Unknown private answer key")
    require(
        answers["schema"] == "fresh-readout-blind-answers-v1"
        and answers["reviewer_type"] == "agent"
        and answers["human_review_complete"] is False,
        "Expected explicit agent answers, not human acceptance",
    )
    expected = {r["review_id"] for r in packet["prompts"]}
    keyed = {r["review_id"]: r for r in key["records"]}
    judged = {r["review_id"]: r for r in answers["answers"]}
    require(
        len(expected)
        == len(packet["prompts"])
        == len(keyed)
        == len(key["records"])
        == len(judged)
        == len(answers["answers"])
        and expected == set(keyed) == set(judged),
        "Missing/duplicate/extra opaque review answers",
    )
    contexts = {g["group_id"] for g in packet["base_contexts"]}
    reviewed = {g["group_id"]: g for g in answers["context_reviews"]}
    require(
        len(contexts) == len(reviewed) == len(answers["context_reviews"]) == 48
        and contexts == set(reviewed),
        "Review must explicitly cover all 48 base contexts",
    )
    issues = [g for g in reviewed.values() if g.get("acceptable") is not True]
    mismatches = []
    for identity in sorted(expected):
        judgment = judged[identity]
        require(
            judgment["answer"] in ("YES", "NO")
            and isinstance(judgment.get("reason"), str)
            and judgment["reason"].strip(),
            "Each answer needs YES/NO and a reason",
        )
        if int(judgment["answer"] == "YES") != keyed[identity]["gold"]:
            mismatches.append(
                {
                    "review_id": identity,
                    "answer": judgment["answer"],
                    "gold": keyed[identity]["gold"],
                }
            )
    return {
        "all_answers_match": not mismatches,
        "mismatches": mismatches,
        "context_issues": issues,
        "coverage": {"base_contexts": 48, "prompts": len(expected), "review_ids": sorted(expected)},
    }


def make_receipt(bank_path, packet_path, key_path, answers_path, output):
    bank = load_bank(bank_path)
    packet, key, answers = (read(p) for p in (packet_path, key_path, answers_path))
    require(
        packet["bank_sha256"] == canonical(bank)
        and packet["bank_file_sha256"] == digest(bank_path),
        "Packet/bank binding differs",
    )
    result = compare(packet, key, answers)
    status = (
        "passed"
        if result["all_answers_match"] and not result["context_issues"]
        else "disagreement_preserved"
    )
    receipt = {
        "schema": "fresh-readout-agent-semantic-review-v1",
        "status": status,
        "reviewer_type": "agent",
        "human_review_complete": False,
        "research_status": "agent_audited_exploratory",
        "bank_file_sha256": digest(bank_path),
        "bank_sha256": canonical(bank),
        **result,
        "source_files": {
            key: {"path": Path(path).resolve().relative_to(ROOT).as_posix(), "sha256": digest(path)}
            for key, path in (("packet", packet_path), ("key", key_path), ("answers", answers_path))
        },
    }
    write_new(output, receipt)
    return receipt


def validate_review_receipt(receipt_path, bank_path):
    receipt = read(receipt_path)
    bank = load_bank(bank_path)
    require(
        receipt["schema"] == "fresh-readout-agent-semantic-review-v1"
        and receipt["status"] == "passed"
        and receipt["reviewer_type"] == "agent"
        and receipt["human_review_complete"] is False
        and receipt["research_status"] == "agent_audited_exploratory",
        "Need passed, explicitly agent-only review",
    )
    require(
        receipt["bank_file_sha256"] == digest(bank_path)
        and receipt["bank_sha256"] == canonical(bank),
        "Semantic review belongs to another bank",
    )
    require(
        set(receipt["source_files"]) == {"packet", "key", "answers"},
        "Need exact three review source bindings",
    )
    sources = {k: read(resolve(b)) for k, b in receipt["source_files"].items()}
    expected_packet, expected_key = review_packet(bank)
    expected_packet.update(bank_sha256=canonical(bank), bank_file_sha256=digest(bank_path))
    require(
        sources["packet"] == expected_packet and sources["key"] == expected_key,
        "Review text/path coverage or formal key differs from compiled bank",
    )
    require(
        sources["packet"]["bank_file_sha256"] == digest(bank_path)
        and sources["packet"]["bank_sha256"] == canonical(bank),
        "Review packet bank differs",
    )
    result = compare(sources["packet"], sources["key"], sources["answers"])
    require(
        result["all_answers_match"] and not result["context_issues"],
        "Semantic disagreements remain",
    )
    require(
        all(
            receipt[k] == result[k]
            for k in ("all_answers_match", "context_issues", "mismatches", "coverage")
        ),
        "Review comparison record changed",
    )
    require(
        {g["group_id"] for g in sources["packet"]["base_contexts"]}
        == {g["group_id"] for g in bank["groups"]},
        "Reviewed contexts differ from bank",
    )
    return receipt
