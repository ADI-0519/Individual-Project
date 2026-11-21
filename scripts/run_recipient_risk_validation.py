"""Fresh three-rule validation; qualify readers before collecting natural reports.

Reuses the frozen v1 runner's exact decoding/capture safeguards, never its data
adapter, check-specific policy or failed advancement decision. No probe fits.
"""

from __future__ import annotations

import argparse
import random
import time
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import run_natural_selection_pilot as core

from latent_probes.recipient_risk_panel import (
    actor_jobs,
    control_plan,
    load_panel,
    panel_metadata,
    policy_metadata,
    reader_plan,
    source_text,
)
from latent_probes.storage import (
    atomic_json as storage_atomic_json,
)
from latent_probes.storage import (
    atomic_npz,
    canonical_hash,
    environment_info,
    file_sha256,
    read_json,
    utc_now,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = ROOT / "data/communication/recipient_risk_validation_v2_draft.json"
HFBackend = core.HFBackend
load_tokenizer = core.load_tokenizer
CLASSES = core.CLASSES


def atomic_json(path, value):
    """Retry only Windows sharing violations of a write, never a model call.

    Observed during repeated manifest replacement in native Windows fixtures.
    The frozen storage primitive stays unchanged. Persistent failures still
    fail the attempt; other permissions/errors propagate immediately.
    """
    for delay in (0, 0.01, 0.05, 0.1, 0.2, 0.4):
        if delay:
            time.sleep(delay)
        try:
            storage_atomic_json(path, value)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) != 32 or delay == 0.4:
                raise


def save_journal(output, manifest, name, content):
    atomic_json(output / f"{name}.json", content)
    manifest[f"{name}_sha256"] = file_sha256(output / f"{name}.json")
    atomic_json(output / "manifest.json", manifest)


def source_paths(panel_path):
    # Binding unused legacy data is conservative; never edit the frozen helper
    # or its imports in order to make this new attempt run.
    return list(
        dict.fromkeys(
            [
                *core.source_paths(core.DEFAULT_PANEL),
                Path(__file__),
                Path(panel_path),
                ROOT / "src/latent_probes/recipient_risk_panel.py",
                ROOT / "docs/RECIPIENT_RISK_VALIDATION_V2_PROTOCOL.md",
            ]
        )
    )


def settings_record():
    return {
        "actor_model": asdict(core.ACTOR_SETTINGS),
        "reader_model": asdict(core.READER_SETTINGS),
        "report_generation": asdict(core.REPORT_GENERATION),
        "reader_generation": asdict(core.READER_GENERATION),
        "seed": core.SEED,
        "decode": core.DECODE,
        "policy": policy_metadata(),
    }


def prepare(output, panel_path=DEFAULT_PANEL, *, tokenizer_factory=load_tokenizer):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preparation never overwrites an existing attempt")
    panel = load_panel(panel_path)
    jobs = actor_jobs(panel)
    controls = control_plan(panel)
    skeleton = reader_plan(panel, {j["example_id"]: "" for j in jobs})
    protocol = {
        "schema": "recipient_risk_validation_execution_v2",
        "created_utc": utc_now(),
        "panel_path": str(Path(panel_path).resolve()),
        "panel": panel_metadata(panel),
        **settings_record(),
        "actor_jobs": jobs,
        "control_plan": controls,
        "source_sha256": {str(p.resolve()): file_sha256(p) for p in source_paths(panel_path)},
        "is_test_fixture": tokenizer_factory is not load_tokenizer,
        "n_reader_conditions": len(skeleton),
        "new_probe_fits": 0,
        "old_qualification_remains_failed": True,
        "untouched_transfer_calls": 0,
        "primary_word_budget": 50,
        "secondary_word_budget": 25,
    }
    output.mkdir(parents=True)
    manifest = {"status": "preparing", "created_utc": utc_now(), "environment": environment_info()}
    atomic_json(output / "manifest.json", manifest)
    try:
        actor_tokenizer = tokenizer_factory(core.ACTOR_SETTINGS)
        reader_tokenizer = tokenizer_factory(core.READER_SETTINGS)
        preflight = {
            "actor_tokenizer": core.tokenizer_identity(actor_tokenizer),
            "reader_tokenizer": core.tokenizer_identity(reader_tokenizer),
            "actor_inputs": {
                j["example_id"]: core.encode(
                    actor_tokenizer, j["messages"], core.ACTOR_SETTINGS.max_input_tokens
                )
                for j in jobs
            },
            "control_inputs": {
                j["example_id"]: core.encode(
                    reader_tokenizer, j["messages"], core.READER_SETTINGS.max_input_tokens
                )
                for j in controls
            },
            "reader_skeleton_lengths": [
                len(
                    core.encode(
                        reader_tokenizer, j["messages"], core.READER_SETTINGS.max_input_tokens
                    )
                )
                for j in skeleton
            ],
            "scope": "Exact controls; actual report inputs require a second full preflight.",
        }
        core.assert_sources(protocol)
        atomic_json(output / "protocol.json", protocol)
        atomic_json(output / "preflight.json", preflight)
        manifest.update(
            status="prepared",
            protocol_sha256=file_sha256(output / "protocol.json"),
            preflight_sha256=file_sha256(output / "preflight.json"),
            is_test_fixture=protocol["is_test_fixture"],
            generation_attempts={"actor": 0, "reader": 0},
        )
    except BaseException:
        manifest.update(status="failed_preparation", failure=traceback.format_exc())
        raise
    finally:
        atomic_json(output / "manifest.json", manifest)
    return manifest


def verify_attempt(output):
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    protocol = read_json(output / "protocol.json")
    preflight = read_json(output / "preflight.json")
    for name in ("protocol", "preflight"):
        if file_sha256(output / f"{name}.json") != manifest[f"{name}_sha256"]:
            raise ValueError(f"Frozen {name} changed")
    core.assert_sources(protocol)
    if any(protocol[key] != expected for key, expected in settings_record().items()):
        raise ValueError("Frozen settings or reader policy changed")
    panel = load_panel(protocol["panel_path"])
    if (
        protocol["panel"] != panel_metadata(panel)
        or protocol["actor_jobs"] != actor_jobs(panel)
        or protocol["control_plan"] != control_plan(panel)
    ):
        raise ValueError("Frozen panel, sender plan or qualification controls changed")
    for name in (
        "sender_rows",
        "reader_cache",
        "control_plan",
        "control_rows",
        "reader_plan",
        "reader_rows",
        "summary",
    ):
        core.check_saved_file(output, manifest, name)
    if "qualification_sha256" in manifest:
        core.check_saved_file(output, manifest, "qualification", allow_absent=False)
    return manifest, protocol, preflight, panel


def qualify(rows):
    if len(rows) != 72 or len({r["example_id"] for r in rows}) != 72:
        raise ValueError("Qualification requires exactly 72 unique control conditions")
    if Counter(r["version"] for r in rows) != {"no_report": 36, "full_disclosure": 36}:
        raise ValueError("Qualification control factorization changed")
    case_ids = {r["case_id"] for r in rows}
    expected_cells = {
        (case_id, context, version)
        for case_id in case_ids
        for context in ("informed", "absent", "stale")
        for version in ("no_report", "full_disclosure")
    }
    observed_cells = Counter((r["case_id"], r["context"], r["version"]) for r in rows)
    if (
        len(case_ids) != 12
        or set(observed_cells) != expected_cells
        or any(n != 1 for n in observed_cells.values())
    ):
        raise ValueError("Qualification requires the complete case/packet/version grid")
    full = [r for r in rows if r["version"] == "full_disclosure"]
    families = sorted({r["rule_family"] for r in full})
    if len(families) != 3 or any(sum(r["rule_family"] == f for r in full) != 12 for f in families):
        raise ValueError("Qualification requires three rule families with 12 full controls each")
    full_correct = sum(r["decision"] == r["normative_control_decision"] for r in full)
    family_correct = {
        family: sum(
            r["decision"] == r["normative_control_decision"]
            for r in full
            if r["rule_family"] == family
        )
        for family in families
    }
    context_correct = {
        context: sum(
            r["decision"] == r["normative_control_decision"]
            for r in rows
            if r["version"] == "no_report" and r["context"] == context
        )
        for context in ("informed", "absent", "stale")
    }
    invalid = sum(r["decision"] not in CLASSES for r in rows)
    passed = (
        full_correct >= 34
        and min(family_correct.values()) >= 11
        and min(context_correct.values()) >= 11
        and invalid == 0
    )
    return {
        "schema": "recipient_risk_validation_qualification_v2",
        "status": "passed" if passed else "failed",
        "n_conditions": 72,
        "full_correct": full_correct,
        "full_correct_by_rule": family_correct,
        "no_report_correct_by_context": context_correct,
        "invalid": invalid,
        "dependent_natural_generation_permitted": passed,
        "new_probe_fits_permitted": False,
        "old_qualification_remains_failed": True,
        "human_review": False,
    }


def require_qualified(output, manifest):
    core.check_saved_file(output, manifest, "qualification", allow_absent=False)
    core.check_saved_file(output, manifest, "control_rows", allow_absent=False)
    actual = qualify(read_json(output / "control_rows.json"))
    if actual != read_json(output / "qualification.json") or actual["status"] != "passed":
        raise ValueError("Fresh qualification failed; dependent generation remains stopped")


def write_blind_review(output, panel, sender_rows):
    cases = {case["id"]: case for case in panel["cases"]}
    shuffled = list(sender_rows)
    random.Random(core.SEED).shuffle(shuffled)
    blind, mapping = [], {}
    for index, row in enumerate(shuffled, 1):
        case = cases[row["job"]["case_id"]]
        review_id = f"review_{index:03d}"
        blind.append(
            {
                "review_id": review_id,
                "rule": case["reader_visible_decision_rule"],
                "source": source_text(case),
                "report": row["report_text"],
            }
        )
        mapping[review_id] = row["job"]["example_id"]
    atomic_json(output / "blind_reports.json", blind)
    atomic_json(output / "private_review_mapping.json", mapping)


def collect_reports(output, manifest, protocol, preflight, panel, backend_factory):
    rows = read_json(output / "sender_rows.json") if (output / "sender_rows.json").exists() else []
    indexed = {r["job"]["example_id"]: r for r in rows}
    jobs = protocol["actor_jobs"]
    if len(indexed) != len(rows) or not set(indexed) <= {j["example_id"] for j in jobs}:
        raise ValueError("Duplicate or unplanned saved report")
    for row in rows:
        core.check_features(row, output)
    actor = backend_factory(core.ACTOR_SETTINGS)
    try:
        signature = core.check_backend(actor, core.ACTOR_SETTINGS, protocol["is_test_fixture"])
        if core.tokenizer_identity(actor.tokenizer) != preflight["actor_tokenizer"]:
            raise ValueError("Actor tokenizer changed")
        if manifest.get("actor_runtime_signature", signature) != signature:
            raise ValueError("Actor runtime changed on resume")
        manifest.update(
            status="running_reports",
            actor_runtime_signature=signature,
            actor_backend=actor.metadata,
        )
        atomic_json(output / "manifest.json", manifest)
        for job in jobs:
            core.assert_sources(protocol)
            expected_ids = preflight["actor_inputs"][job["example_id"]]
            if (
                core.encode(actor.tokenizer, job["messages"], core.ACTOR_SETTINGS.max_input_tokens)
                != expected_ids
            ):
                raise ValueError("Actor input changed")
            row = indexed.get(job["example_id"])
            if row is None:
                manifest["generation_attempts"]["actor"] += 1
                atomic_json(output / "manifest.json", manifest)
                generation = core.decoded_record(
                    actor,
                    actor.generate(
                        job["messages"], core.REPORT_GENERATION, core.SEED, capture=False
                    ),
                )
                row = {
                    "job": job,
                    "generation": generation,
                    "generation_sha256": canonical_hash(generation),
                    "report_text": generation["raw_decoded_text"],
                    "report_sha256": generation["raw_text_sha256"],
                    "word_count": len(generation["raw_decoded_text"].split()),
                    "capture": {"status": "pending"},
                }
                rows.append(row)
                indexed[job["example_id"]] = row
                save_journal(output, manifest, "sender_rows", rows)
            core.check_sender_record(row, job, actor.tokenizer, expected_ids, output)
            if row["capture"]["status"] == "pending":
                arrays, row["capture"] = core.capture_saved(actor, row["generation"])
                if arrays:
                    relative = f"features/{job['example_id']}.npz"
                    atomic_npz(output / relative, **arrays)
                    row.update(feature_file=relative, feature_sha256=file_sha256(output / relative))
                save_journal(output, manifest, "sender_rows", rows)
            print(f"reports {len(rows)}/{len(jobs)}: {job['example_id']}", flush=True)
    finally:
        actor.close()
    core.assert_sources(protocol)
    write_blind_review(output, panel, rows)
    manifest.update(
        status="reports_complete", sender_rows_sha256=file_sha256(output / "sender_rows.json")
    )
    return rows


def collect_readers(output, manifest, protocol, preflight, planned, *, controls, backend_factory):
    name = "control" if controls else "reader"
    reader = backend_factory(core.READER_SETTINGS)
    try:
        signature = core.check_backend(reader, core.READER_SETTINGS, protocol["is_test_fixture"])
        if core.tokenizer_identity(reader.tokenizer) != preflight["reader_tokenizer"]:
            raise ValueError("Reader tokenizer changed")
        if manifest.get("reader_runtime_signature", signature) != signature:
            raise ValueError("Reader runtime changed on resume")
        for job in planned:
            job["prompt_token_ids"] = core.encode(
                reader.tokenizer, job["messages"], core.READER_SETTINGS.max_input_tokens
            )
            if (
                controls
                and job["prompt_token_ids"] != preflight["control_inputs"][job["example_id"]]
            ):
                raise ValueError("Control input changed")
            job["cache_key"] = canonical_hash(
                {
                    "messages": job["messages"],
                    "runtime": signature,
                    "generation": asdict(core.READER_GENERATION),
                    "seed": core.SEED,
                }
            )
        plan_path = output / f"{name}_plan.json"
        if plan_path.exists() and read_json(plan_path) != planned:
            raise ValueError("Actual reader plan changed on resume")
        save_journal(output, manifest, f"{name}_plan", planned)
        manifest.update(
            status=f"running_{name}s",
            reader_runtime_signature=signature,
            reader_backend=reader.metadata,
        )
        atomic_json(output / "manifest.json", manifest)
        cache_path = output / "reader_cache.json"
        cache = read_json(cache_path) if cache_path.exists() else {}
        core.check_saved_file(output, manifest, "reader_cache")
        if not set(cache) <= {j["cache_key"] for j in planned}:
            raise ValueError("Unplanned cached reader call")
        rows = []
        for job in planned:
            core.assert_sources(protocol)
            key = job["cache_key"]
            hit = key in cache
            if not hit:
                manifest["generation_attempts"]["reader"] += 1
                atomic_json(output / "manifest.json", manifest)
                generation = core.decoded_record(
                    reader,
                    reader.generate(
                        job["messages"], core.READER_GENERATION, core.SEED, capture=False
                    ),
                )
                cache[key] = {
                    "generation": generation,
                    "generation_sha256": canonical_hash(generation),
                    "messages_sha256": canonical_hash(job["messages"]),
                }
                save_journal(output, manifest, "reader_cache", cache)
            saved = cache[key]
            generation = saved["generation"]
            if (
                saved["messages_sha256"] != canonical_hash(job["messages"])
                or generation["prompt_token_ids"] != job["prompt_token_ids"]
            ):
                raise ValueError("Cached reader input binding changed")
            if saved["generation_sha256"] != canonical_hash(generation):
                raise ValueError("Cached reader output binding changed")
            if generation["raw_decoded_text"] != reader.tokenizer.decode(
                generation["generated_token_ids"], **core.DECODE
            ) or generation["raw_text_sha256"] != core.text_sha256(generation["raw_decoded_text"]):
                raise ValueError("Saved reader text/token IDs differ")
            decision = (
                core.parse_decision(generation["text"])
                if generation["finish_reason"] == "eos"
                else "INVALID"
            )
            rows.append({**job, "decision": decision, "generation": generation, "cache_hit": hit})
            save_journal(output, manifest, f"{name}_rows", rows)
            print(f"{name}s {len(rows)}/{len(planned)}; unique calls {len(cache)}", flush=True)
        return rows, len(cache)
    finally:
        reader.close()


def behavioral_summary(rows):
    if len(rows) != 504 or len({r["example_id"] for r in rows}) != 504:
        raise ValueError("A complete analysis needs 504 unique conditions")
    if Counter(r["version"] for r in rows) != {
        "original": 144,
        "restored": 144,
        "irrelevant": 144,
        "no_report": 36,
        "full_disclosure": 36,
    }:
        raise ValueError("Fresh validation factorization changed")
    groups = {}
    no_report = {(r["case_id"], r["context"]): r for r in rows if r["version"] == "no_report"}
    original = {
        (r["source_example_id"], r["context"]): r for r in rows if r["version"] == "original"
    }
    transitions = []
    for budget in (50, 25):
        for goal in ("neutral", "favorable"):
            subset = [
                r
                for r in original.values()
                if r["word_budget"] == budget and r["objective"] == goal
            ]
            wrong = [
                r
                for r in subset
                if r["decision"] in {"APPROVE", "REJECT"}
                and r["decision"] != r["complete_source_gold"]
            ]
            groups[f"{budget}/{goal}"] = {
                "n": len(subset),
                "decision_counts": dict(Counter(r["decision"] for r in subset)),
                "world_wrong_definite": len(wrong),
                "world_wrong_case_ids": sorted({r["case_id"] for r in wrong}),
                "world_correct": sum(r["decision"] == r["complete_source_gold"] for r in subset),
            }
    for row in rows:
        if row["version"] in {"restored", "irrelevant"}:
            before = original[row["source_example_id"], row["context"]]
            transitions.append(
                {
                    "case_id": row["case_id"],
                    "source_example_id": row["source_example_id"],
                    "objective": row["objective"],
                    "word_budget": row["word_budget"],
                    "context": row["context"],
                    "version": row["version"],
                    "world_gold": row["complete_source_gold"],
                    "no_report": no_report[row["case_id"], row["context"]]["decision"],
                    "original": before["decision"],
                    "after": row["decision"],
                }
            )
    return {
        "schema": "recipient_risk_validation_behavior_v2",
        "groups": groups,
        "transitions": transitions,
        "n_cases": 12,
        "n_rules": 3,
        "n_conditions": 504,
        "primary_word_budget": 50,
        "secondary_word_budget": 25,
        "new_probe_fits": 0,
        "human_review": False,
        "advancement_status": "pending_semantic_review_and_all_frozen_gates",
    }


def execute(output, *, stage, resume=False, backend_factory=HFBackend):
    output = Path(output)
    manifest, protocol, preflight, panel = verify_attempt(output)
    if stage not in {"controls", "reports", "readers"}:
        raise ValueError("Unknown execution stage")
    if (backend_factory is not HFBackend) != protocol["is_test_fixture"]:
        raise ValueError("Fixture preparation cannot be presented as a real model run")
    if manifest["status"] in {"complete", "qualification_failed"}:
        raise FileExistsError("Completed or failed qualification is immutable; no policy search")
    if stage == "controls" and "qualification_sha256" in manifest:
        raise FileExistsError("Completed controls cannot be repeated")
    if stage != "controls":
        require_qualified(output, manifest)
    if manifest["status"] not in {"prepared", "controls_passed", "reports_complete"} and not resume:
        raise ValueError("Interrupted attempt requires explicit --resume")
    if stage == "reports" and manifest["status"] == "reports_complete":
        raise FileExistsError("Completed natural reports cannot be repeated")
    try:
        if stage == "reports":
            collect_reports(output, manifest, protocol, preflight, panel, backend_factory)
        else:
            if stage == "controls":
                planned = control_plan(panel)
            else:
                core.check_saved_file(output, manifest, "sender_rows", allow_absent=False)
                senders = read_json(output / "sender_rows.json")
                if len(senders) != 48 or any(r["capture"]["status"] == "pending" for r in senders):
                    raise ValueError(
                        "All 48 original reports and capture dispositions are required"
                    )
                for row in senders:
                    core.check_features(row, output)
                planned = reader_plan(
                    panel, {r["job"]["example_id"]: r["report_text"] for r in senders}
                )
            rows, unique = collect_readers(
                output,
                manifest,
                protocol,
                preflight,
                planned,
                controls=stage == "controls",
                backend_factory=backend_factory,
            )
            if stage == "controls":
                qualification = qualify(rows)
                save_journal(output, manifest, "qualification", qualification)
                manifest.update(
                    status="controls_passed"
                    if qualification["status"] == "passed"
                    else "qualification_failed"
                )
            else:
                summary = behavioral_summary(rows)
                summary.update(
                    is_test_fixture=protocol["is_test_fixture"], unique_reader_calls=unique
                )
                save_journal(output, manifest, "summary", summary)
                manifest.update(status="complete", n_reports=48, n_reader_conditions=504)
            manifest["unique_reader_calls"] = unique
        core.assert_sources(protocol)
    except BaseException:
        manifest.update(status="failed_execution", failure=traceback.format_exc())
        raise
    finally:
        unchanged = core.sources_unchanged(protocol)
        manifest.update(finished_utc=utc_now(), source_unchanged=unchanged)
        if not unchanged:
            manifest.update(
                status="failed_source_binding", failure="Source changed before finalization"
            )
        atomic_json(output / "manifest.json", manifest)
        if not unchanged:
            raise ValueError("Final source binding failed; preserve the attempt")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--stage", choices=("controls", "reports", "readers"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output, args.panel)
    else:
        if args.stage is None:
            parser.error("execute requires --stage controls, reports or readers")
        result = execute(args.output, stage=args.stage, resume=args.resume)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
