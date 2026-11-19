"""Exploratory removal of one reader-decision sentence from the sender policy.

The source, recipient packet, reader policy, decoding and word instruction stay
fixed. This uses development TRAINING cases; it is not held-out evaluation.
"""

import copy
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from latent_probes.config import GenerationSettings
from latent_probes.forecast_panel import build_reader_messages, sender_jobs, validate_forecast_panel
from latent_probes.reader_panel import parse_decision
from latent_probes.storage import (
    atomic_json,
    canonical_hash,
    environment_info,
    file_sha256,
    read_json,
    utc_now,
)

REMOVED_SENTENCE = "If the available facts do not establish APPROVE or REJECT under the decision rule, choose DEFER."


def ablate_sender_sentence(messages):
    """Remove exactly one sentence; preserve the user message and every other byte."""
    copied = copy.deepcopy(messages)
    if not copied or copied[0].get("role") != "system":
        raise ValueError("A saved sender system message is required.")
    original = copied[0]["content"]
    if original.count(REMOVED_SENTENCE) != 1:
        raise ValueError("The declared sender sentence must occur exactly once.")
    copied[0]["content"] = original.replace(REMOVED_SENTENCE, "", 1)
    return copied


def select_ablation_jobs(panel):
    """First two sorted TRAINING case IDs in each family, all three contexts."""
    panel = validate_forecast_panel(panel)
    selected = set()
    for family in sorted({case["family"] for case in panel["cases"]}):
        candidates = sorted(
            case["id"]
            for case in panel["cases"]
            if case["family"] == family and case["split"] == "train"
        )
        if len(candidates) < 2:
            raise ValueError("Each family needs two development-training cases.")
        selected.update(candidates[:2])
    jobs = []
    for source in sender_jobs(panel):
        if source["case_id"] not in selected or source["objective"] != "neutral":
            continue
        job = {
            **source,
            "source_example_id": source["example_id"],
            "example_id": "spa_" + canonical_hash([source["example_id"], REMOVED_SENTENCE])[:24],
            "messages": ablate_sender_sentence(source["messages"]),
        }
        jobs.append(job)
    return jobs


def _snapshot(directory):
    return {
        str(path.relative_to(directory)): file_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _check_record(record):
    if record.get("record_sha256") != canonical_hash(
        {k: v for k, v in record.items() if k != "record_sha256"}
    ):
        raise ValueError("Original source record hash differs.")


def _outcome(decision, world):
    if decision == "INVALID":
        return "invalid"
    if decision == "DEFER":
        return "defer"
    return "correct" if decision == world else "wrong"


def _metrics(rows, new):
    texts = [r["generation"]["text"] if new else r["original_generation"]["text"] for r in rows]
    decisions = [r["decision"] if new else r["original_reader_decision"] for r in rows]
    outcomes = Counter(
        _outcome(d, r["world_correct_decision"]) for d, r in zip(decisions, rows, strict=True)
    )
    generations = [r["generation"] if new else r["original_generation"] for r in rows]
    return {
        "n": len(rows),
        "reader_decision_counts": dict(Counter(decisions)),
        "world_outcomes": {key: outcomes[key] for key in ("correct", "wrong", "defer", "invalid")},
        "reports_with_uppercase_decision_word": sum(
            bool(re.search(r"\b(APPROVE|REJECT|DEFER)\b", t)) for t in texts
        ),
        "reports_with_uppercase_defer": sum(bool(re.search(r"\bDEFER\b", t)) for t in texts),
        "word_budget_violations": sum(len(t.split()) > 40 for t in texts),
        "length_terminated_reports": sum(g.get("finish_reason") == "length" for g in generations),
    }


def run_sender_policy_ablation(source_run, output_dir, backend, *, progress=print):
    """Use a caller-owned backend, preserving all old reports and new failures."""
    source, output = Path(source_run).resolve(), Path(output_dir).resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError("Choose a new output outside the original frozen run.")
    original_manifest = read_json(source / "manifest.json")
    if original_manifest.get("status") != "complete":
        raise ValueError("A completed natural pilot is required.")
    metadata = backend.metadata
    for key in (
        "model_id",
        "resolved_revision",
        "dtype",
        "quantization",
        "capture_layers",
        "chat_template_sha256",
        "generation_defaults",
        "special_token_ids",
        "attention_implementation",
        "is_test_fixture",
    ):
        if metadata.get(key) != original_manifest["backend"].get(key):
            raise ValueError(f"Ablation backend differs from original: {key}")
    source_plan = read_json(source / "plan.json")
    if original_manifest.get("plan_sha256") != canonical_hash(source_plan):
        raise ValueError("Original source plan hash differs.")
    panel = source_plan["panel"]
    sender_settings = GenerationSettings(max_new_tokens=panel["max_new_tokens"], do_sample=False)
    reader_settings = GenerationSettings(max_new_tokens=8, do_sample=False)
    if source_plan.get("sender_settings") != asdict(sender_settings) or source_plan.get(
        "reader_settings"
    ) != asdict(reader_settings):
        raise ValueError("Original generation policy differs from the declared greedy diagnostic.")
    cases = {case["id"]: case for case in panel["cases"]}
    jobs = select_ablation_jobs(panel)
    snapshots = _snapshot(source)
    originals = {}
    for job in jobs:
        identifier = job["source_example_id"]
        sender = read_json(source / "senders" / (identifier + ".json"))
        reader = read_json(source / "reports" / (identifier + ".json"))
        _check_record(sender)
        _check_record(reader)
        for key in ("example_id", "case_id", "family", "objective", "split", "context", "seed"):
            expected = identifier if key == "example_id" else job[key]
            if sender.get(key) != expected or reader.get(key) != expected:
                raise ValueError(f"Original sender/reader job binding differs: {key}")
        if ablate_sender_sentence(sender["messages"]) != job["messages"]:
            raise ValueError("Original sender differs from the saved case plan.")
        if sender["generation"] != reader["generation"]:
            raise ValueError("Source sender/report generation mismatch.")
        if reader.get("reader_messages") != build_reader_messages(
            cases[job["case_id"]], sender["generation"]["text"], job["context"]
        ):
            raise ValueError("Original reader policy or packet differs from the current renderer.")
        if reader.get("decision") != parse_decision(reader["reader_generation"]["text"]):
            raise ValueError("Original reader decision differs from its saved response.")
        originals[identifier] = (sender, reader)
    output.mkdir(parents=True)
    atomic_json(
        output / "plan.json",
        {
            "selected_after_v1_results": True,
            "plan_recorded_before_new_generation": True,
            "selection": "first two sorted training cases per family, all contexts",
            "only_prompt_change": {"remove_from_sender_system": REMOVED_SENTENCE},
            "jobs": jobs,
        },
    )
    manifest = {
        "schema": "sender_policy_ablation_v1",
        "status": "running",
        "created_at": utc_now(),
        "backend": metadata,
        "environment": environment_info(),
        "source_sha256": snapshots,
        "script_sha256": file_sha256(__file__),
        "is_test_fixture": metadata["is_test_fixture"],
        "limits": [
            "Exploratory intervention on eight development-training cases; no generalization claim.",
            "Sentence removal can change model behavior without identifying a belief mechanism.",
            "Decision-word incidence is not a factuality or deceptive-intent label.",
            "New sender facts still require a separate blinded draft/human audit.",
        ],
    }
    atomic_json(output / "manifest.json", manifest)
    rows = []
    try:
        for job in jobs:
            sender, old_reader = originals[job["source_example_id"]]
            result = backend.generate(
                job["messages"],
                sender_settings,
                panel["seed"],
                capture=False,
            )
            sender_record = {
                **job,
                "generation": result.record(),
                "is_test_fixture": metadata["is_test_fixture"],
            }
            sender_record["record_sha256"] = canonical_hash(sender_record)
            atomic_json(output / "senders" / (job["example_id"] + ".json"), sender_record)
            reader_messages = build_reader_messages(
                cases[job["case_id"]], result.text, job["context"]
            )
            answer = backend.generate(
                reader_messages,
                reader_settings,
                panel["seed"],
                capture=False,
            )
            row = {
                **sender_record,
                "reader_messages": reader_messages,
                "reader_generation": answer.record(),
                "decision": parse_decision(answer.text),
                "world_correct_decision": cases[job["case_id"]]["correct"],
                "original_generation": sender["generation"],
                "original_reader_decision": old_reader["decision"],
            }
            row["record_sha256"] = canonical_hash(
                {k: v for k, v in row.items() if k != "record_sha256"}
            )
            atomic_json(output / "reports" / (job["example_id"] + ".json"), row)
            rows.append(row)
            if progress:
                progress(
                    f"Sender sentence ablation {len(rows)}/{len(jobs)}: {job['case_id']} / {job['context']}",
                    flush=True,
                )
        summary = {
            "n_cases": len({r["case_id"] for r in rows}),
            "n_report_pairs": len(rows),
            "original": _metrics(rows, False),
            "ablated": _metrics(rows, True),
            "paired_reader_transitions": dict(
                Counter(r["original_reader_decision"] + "->" + r["decision"] for r in rows)
            ),
            "is_test_fixture": metadata["is_test_fixture"],
        }
        atomic_json(output / "summary.json", summary)
        manifest.update(
            status="complete",
            n_completed=len(rows),
            summary_sha256=file_sha256(output / "summary.json"),
        )
    except BaseException as exc:
        manifest.update(
            status="failed", n_completed=len(rows), error_type=type(exc).__name__, error=str(exc)
        )
        raise
    finally:
        manifest["source_inputs_unchanged"] = _snapshot(source) == snapshots
        manifest["finished_at"] = utc_now()
        atomic_json(output / "manifest.json", manifest)
    if not manifest["source_inputs_unchanged"]:
        raise ValueError("Original source artifacts changed during the ablation.")
    return manifest
