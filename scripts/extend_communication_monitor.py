"""One frozen budget recovery for capped communication forecasts, without refitting."""

from __future__ import annotations

import argparse
import copy
import traceback
from pathlib import Path

from run_communication_transfer import (
    artifact_hashes,
    discrete_metrics,
    monitor_decision,
    monitor_messages,
)

from latent_probes.apollo import capture_signature
from latent_probes.backends import HFBackend
from latent_probes.communication_panel import validate_panel
from latent_probes.config import GenerationSettings, ModelSettings
from latent_probes.storage import (
    atomic_json,
    canonical_hash,
    environment_info,
    file_sha256,
    read_json,
    utc_now,
)

ROOT = Path(__file__).resolve().parents[1]
OLD_BUDGET = 160
NEW_BUDGET = 512
SEED = 42
EXPECTED_TOTAL = 144
EXPECTED_SELECTED = 66


class ReplayMismatch(ValueError):
    """The attempted recovery is not an exact continuation of the original run."""


def select_recovery_rows(rows):
    """Use termination alone, never correctness, goal, content labels, or scores."""
    identifiers = [row["example_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate original forecast identity")
    selected, cached = [], []
    for row in rows:
        generation = row["monitor_generation"]
        finish = generation["finish_reason"]
        if finish == "length":
            if len(generation["generated_token_ids"]) != OLD_BUDGET:
                raise ValueError("A selected original forecast did not reach the frozen cap")
            selected.append(row)
        elif finish == "eos":
            cached.append(row)
        else:
            raise ValueError("Unexpected original monitor termination reason")
    return selected, cached


def prefix_audit(original, attempted):
    """Require the whole original generated sequence, including its final token."""
    old_ids = original["generated_token_ids"]
    new_ids = attempted["generated_token_ids"]
    if not old_ids or any(type(token) is not int for token in old_ids + new_ids):
        raise ValueError("Expected nonempty original and integer generation token IDs")
    mismatches = [
        index
        for index, token in enumerate(old_ids)
        if index >= len(new_ids) or new_ids[index] != token
    ]
    return {
        "prompt_exact": original["prompt_token_ids"] == attempted["prompt_token_ids"],
        "original_generated_tokens": len(old_ids),
        "attempted_generated_tokens": len(new_ids),
        "entire_original_prefix_exact": not mismatches,
        "first_prefix_mismatch": mismatches[0] if mismatches else None,
    }


def forecast_row(original, generation, *, provenance, audit=None):
    """Evaluation metadata is joined after generation, never supplied to the model."""
    predictions = copy.deepcopy(original["predictions"])
    predictions["full_context_forecaster_160"] = monitor_decision(original["monitor_generation"])
    predictions["full_context_forecaster_512_recovery"] = monitor_decision(generation)
    return {
        **{
            key: original[key]
            for key in (
                "example_id",
                "case_id",
                "family",
                "objective",
                "context",
                "transfer_scope",
                "world_correct_decision",
                "decision",
            )
        },
        "source_record_sha256": original["record_sha256"],
        "provenance": provenance,
        "predictions": predictions,
        "generation": generation,
        "replay_audit": audit,
    }


def summarize_recovery(rows, original_rows, attempts):
    groups = {"all": rows}
    for key in ("objective", "context", "transfer_scope", "world_correct_decision"):
        for value in sorted({row[key] for row in rows}):
            groups[f"{key}/{value}"] = [row for row in rows if row[key] == value]
    methods = list(rows[0]["predictions"])
    original_generations = [row["monitor_generation"] for row in original_rows]
    new_generations = [attempt["generation"] for attempt in attempts]

    def cost(generations):
        return {
            "calls": len(generations),
            "prompt_tokens": sum(len(g["prompt_token_ids"]) for g in generations),
            "generated_tokens": sum(len(g["generated_token_ids"]) for g in generations),
            "elapsed_generation_seconds": sum(g["elapsed_seconds"] for g in generations),
        }

    return {
        "schema": "communication_monitor_budget_recovery_summary_v1",
        "n_reports": len(rows),
        "n_case_clusters": len({row["case_id"] for row in rows}),
        "n_reused_eos_forecasts": sum(row["provenance"] == "cached_original_eos" for row in rows),
        "n_extended_forecasts": sum(row["provenance"] == "extended_exact_prefix" for row in rows),
        "all_attempt_prompts_exact": all(a["replay_audit"]["prompt_exact"] for a in attempts),
        "all_original_160_token_prefixes_exact": all(
            a["replay_audit"]["entire_original_prefix_exact"] for a in attempts
        ),
        "new_generation_budget": NEW_BUDGET,
        "evaluations": {
            group: {method: discrete_metrics(subset, method) for method in methods}
            for group, subset in groups.items()
        },
        "forecast_termination": {
            reason: sum(row["generation"]["finish_reason"] == reason for row in rows)
            for reason in ("eos", "length")
        },
        "cost": {
            "original_all_144_monitor_calls": cost(original_generations),
            "additional_from_scratch_recovery_calls": cost(new_generations),
            "original_prefix_tokens_regenerated": sum(
                a["replay_audit"]["original_generated_tokens"] for a in attempts
            ),
            "new_tokens_beyond_old_cap": sum(
                max(0, len(a["generation"]["generated_token_ids"]) - OLD_BUDGET) for a in attempts
            ),
            "additional_cached_forecast_calls": 0,
            "scope": "New calls include regeneration of the original 160-token prefixes.",
        },
        "limits": [
            "Post-observation budget repair, not untouched confirmatory baseline evidence.",
            "Selection is original termination only; the 512-token cap is never increased again.",
            "Strict original parser and all invalid-target exclusions remain visible.",
            "No new reports, readers, probe fitting, prompts or independent reader model.",
            "The larger compute budget must accompany every recovered-baseline comparison.",
        ],
    }


def prepare(source):
    """Bind the immutable source and freeze outcome-free model jobs."""
    manifest = read_json(source / "manifest.json")
    protocol = read_json(source / "protocol.json")
    if (
        manifest["status"] != "complete"
        or file_sha256(source / "protocol.json") != manifest["protocol_sha256"]
    ):
        raise ValueError("Original communication run is incomplete or protocol hash changed")
    if protocol["schema"] != "fresh_communication_transfer_v1" or protocol["seed"] != SEED:
        raise ValueError("Unexpected source experiment or seed")
    old_settings = GenerationSettings(**protocol["monitor_settings"])
    if old_settings.max_new_tokens != OLD_BUDGET or old_settings.do_sample:
        raise ValueError("Source must be the frozen greedy 160-token monitor")
    helper = ROOT / "scripts/run_communication_transfer.py"
    if file_sha256(helper) != protocol["source_hashes"]["scripts/run_communication_transfer.py"]:
        raise ValueError("Original monitor prompt/parser helper source has changed")
    panel = validate_panel(protocol["panel"])
    cases = {case["id"]: case for case in panel["cases"]}
    source_jobs = {job["example_id"]: job for job in protocol["jobs"]}
    rows, bindings = [], []
    for path in sorted((source / "reports").glob("*.json")):
        row = read_json(path)
        recorded = row["record_sha256"]
        if canonical_hash({k: v for k, v in row.items() if k != "record_sha256"}) != recorded:
            raise ValueError("Original report record hash mismatch")
        job = source_jobs[row["example_id"]]
        if path.stem != row["example_id"] or any(row[key] != value for key, value in job.items()):
            raise ValueError("Source report identity or original sender job changed")
        rows.append(row)
        bindings.append(
            {
                "example_id": row["example_id"],
                "source_report_file": str(path.relative_to(source)),
                "source_report_sha256": file_sha256(path),
                "monitor_messages": monitor_messages(
                    cases[row["case_id"]], row, row["report_text"]
                ),
                "original_monitor_generation": row["monitor_generation"],
            }
        )
    selected, cached = select_recovery_rows(rows)
    if len(rows) != EXPECTED_TOTAL or len(selected) != EXPECTED_SELECTED or len(cached) != 78:
        raise ValueError("Frozen recovery requires exactly 144 source rows, 66 capped and 78 EOS")
    if {row["example_id"] for row in rows} != set(source_jobs):
        raise ValueError("Original report registry is incomplete")
    if any(monitor_decision(row["monitor_generation"]) == "INVALID" for row in cached):
        raise ValueError("An original EOS forecast has a separate parsing failure")
    selected_ids = {row["example_id"] for row in selected}
    settings = ModelSettings(**protocol["settings"])
    new_settings = GenerationSettings(
        **{**protocol["monitor_settings"], "max_new_tokens": NEW_BUDGET}
    )
    plan = {
        "schema": "communication_monitor_budget_recovery_v1",
        "created_utc": utc_now(),
        "source_directory": str(source.resolve()),
        "source_tree_sha256": artifact_hashes(source),
        "original_protocol_sha256": file_sha256(source / "protocol.json"),
        "model_settings": settings.__dict__,
        "old_monitor_settings": old_settings.__dict__,
        "new_monitor_settings": new_settings.__dict__,
        "seed": SEED,
        "selection_rule": "Every original monitor finish_reason == length; no correctness/goal selection.",
        "all_forecast_bindings": bindings,
        "selected_example_ids": sorted(selected_ids),
        "cached_example_ids": sorted(row["example_id"] for row in cached),
        "source_sha256": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in (
                Path(__file__),
                helper,
                ROOT / "src/latent_probes/backends.py",
                ROOT / "src/latent_probes/config.py",
                ROOT / "src/latent_probes/communication_panel.py",
                ROOT / "docs/MONITOR_BUDGET_RECOVERY_PROTOCOL.md",
            )
        },
        "prefix_requirement": "All original 160 generated IDs and encoded prompt IDs match exactly.",
        "failure_policy": "Abort inconclusive on any mismatch; preserve every attempted generation.",
        "new_reports": 0,
        "new_reader_calls": 0,
        "new_fit_calls": 0,
    }
    return manifest, rows, plan, settings, new_settings


def run(args, backend_factory=HFBackend):
    source, output = Path(args.source), Path(args.output)
    if output.exists():
        raise FileExistsError("Use the one frozen fresh output directory; preserve all attempts")
    if source.resolve().is_relative_to(output.resolve()) or output.resolve().is_relative_to(
        source.resolve()
    ):
        raise ValueError("Source and recovery output trees must be disjoint")
    original_manifest, originals, plan, settings, generation_settings = prepare(source)
    output.mkdir(parents=True)
    atomic_json(output / "protocol.json", plan)
    manifest = {
        "schema": "communication_monitor_budget_recovery_manifest_v1",
        "status": "prepared",
        "created_utc": utc_now(),
        "protocol_sha256": file_sha256(output / "protocol.json"),
        "environment": environment_info(),
        "n_completed_attempts": 0,
    }
    atomic_json(output / "manifest.json", manifest)
    backend, attempts, prompt_checks = None, [], []
    selected_ids = set(plan["selected_example_ids"])
    original_by_id = {row["example_id"]: row for row in originals}
    recovered = [
        forecast_row(row, row["monitor_generation"], provenance="cached_original_eos")
        for row in originals
        if row["example_id"] not in selected_ids
    ]
    try:
        print("Loading the original pinned model for one monitor budget recovery", flush=True)
        backend = backend_factory(settings)
        if backend.metadata.get("is_test_fixture") is not False or capture_signature(
            backend.metadata, 8
        ) != capture_signature(original_manifest["backend"], 8):
            raise ReplayMismatch("Original model/precision/template identity differs")
        manifest.update(status="running", started_utc=utc_now(), backend=backend.metadata)
        atomic_json(output / "manifest.json", manifest)
        # Check every selected and cached prompt before the first new token is generated.
        for binding in plan["all_forecast_bindings"]:
            encoded = backend.tokenizer.apply_chat_template(
                binding["monitor_messages"],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            ids = encoded["input_ids"][0].tolist()
            exact = ids == binding["original_monitor_generation"]["prompt_token_ids"]
            prompt_checks.append({"example_id": binding["example_id"], "prompt_exact": exact})
            atomic_json(output / "prompt_checks.json", prompt_checks)
            if not exact:
                raise ReplayMismatch("Reconstructed original monitor prompt IDs differ")
        for binding in plan["all_forecast_bindings"]:
            identifier = binding["example_id"]
            if identifier not in selected_ids:
                continue
            generation = backend.generate(
                binding["monitor_messages"], generation_settings, SEED
            ).record()
            audit = prefix_audit(binding["original_monitor_generation"], generation)
            attempt = {
                "example_id": identifier,
                "generation": generation,
                "replay_audit": audit,
                "decision_used": audit["prompt_exact"] and audit["entire_original_prefix_exact"],
            }
            attempts.append(attempt)
            atomic_json(output / "attempts" / f"{identifier}.json", attempt)
            manifest["n_completed_attempts"] = len(attempts)
            atomic_json(output / "manifest.json", manifest)
            if not attempt["decision_used"]:
                raise ReplayMismatch("Original 160-token prefix changed; recovery is inconclusive")
            recovered.append(
                forecast_row(
                    original_by_id[identifier],
                    generation,
                    provenance="extended_exact_prefix",
                    audit=audit,
                )
            )
            print(f"Recovery {len(attempts)}/{len(selected_ids)}: exact old prefix", flush=True)
        if len(recovered) != EXPECTED_TOTAL:
            raise ValueError("Incomplete recovered forecast registry")
        recovered.sort(key=lambda row: row["example_id"])
        if artifact_hashes(source) != plan["source_tree_sha256"]:
            raise ValueError("Original communication run changed during recovery")
        atomic_json(output / "forecasts.json", recovered)
        atomic_json(output / "summary.json", summarize_recovery(recovered, originals, attempts))
        manifest.update(
            status="complete",
            completed_utc=utc_now(),
            source_unchanged=True,
            n_forecasts=len(recovered),
            n_extended=len(attempts),
            n_cached=78,
            all_prompt_checks_exact=True,
            all_original_prefixes_exact=True,
            summary_sha256=file_sha256(output / "summary.json"),
            forecasts_sha256=file_sha256(output / "forecasts.json"),
        )
    except ReplayMismatch:
        manifest.update(status="inconclusive_replay_mismatch", failure=traceback.format_exc())
        raise
    except BaseException:
        manifest.update(status="failed", failure=traceback.format_exc())
        raise
    finally:
        if backend is not None:
            backend.close()
        manifest.update(
            finished_utc=utc_now(),
            source_unchanged=artifact_hashes(source) == plan["source_tree_sha256"],
            additional_attempt_cost={
                "calls": len(attempts),
                "generated_tokens": sum(
                    len(a["generation"]["generated_token_ids"]) for a in attempts
                ),
                "elapsed_generation_seconds": sum(
                    a["generation"]["elapsed_seconds"] for a in attempts
                ),
            },
        )
        atomic_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/communication_transfer_v1")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/adirj/kcl-latent-probes-runs/communication_monitor_512_v1"),
    )
    run(parser.parse_args())
