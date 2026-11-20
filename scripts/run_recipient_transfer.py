"""Gated source-excerpt training and untouched natural communication transfer.

The source reports are off-policy fixed continuations. Labels are actual answers
from one separately qualified Qwen policy, never normative replacements.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import traceback
from pathlib import Path

import numpy as np
from replay_communication_reader import capability_gate
from validate_calibrated_reader import validate_bank

from latent_probes.apollo import capture_signature
from latent_probes.backends import HFBackend
from latent_probes.calibrated_reader import build_calibrated_reader_messages, policy_metadata
from latent_probes.communication_panel import build_sender_messages
from latent_probes.config import GenerationSettings, load_config
from latent_probes.forecast_panel import build_reader_messages as source_reader_builder
from latent_probes.neutral_recipient_panel import load_panel, neutral_jobs, source_projection
from latent_probes.reader_panel import parse_decision
from latent_probes.storage import (
    atomic_json,
    atomic_npz,
    canonical_hash,
    environment_info,
    file_sha256,
    read_json,
    utc_now,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
REPORT_SETTINGS = GenerationSettings(max_new_tokens=96, do_sample=False)
READER_SETTINGS = GenerationSettings(max_new_tokens=8, do_sample=False)


def require_fresh_gate(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    protocol = read_json(directory / "protocol.json")
    summary = read_json(directory / "summary.json")
    if (
        manifest["status"] != "complete_fresh_gate_passed"
        or not manifest["source_unchanged"]
        or not summary["fresh_gate"]["passed"]
        or manifest["protocol_sha256"] != file_sha256(directory / "protocol.json")
        or manifest["summary_sha256"] != file_sha256(directory / "summary.json")
        or protocol["policy"] != policy_metadata()
    ):
        raise ValueError("A completed, hash-bound fresh gate for this exact policy is required")
    for path, digest in protocol["source_sha256"].items():
        if file_sha256(Path(path)) != digest:
            raise ValueError("A frozen reader qualification input changed")
    controls = read_json(directory / "fresh_controls.json")
    jobs = protocol["fresh_jobs"]
    if len(controls) != 72 or len(jobs) != 72:
        raise ValueError("A complete fresh 72-condition capability gate is required")
    for control, job in zip(controls, jobs, strict=True):
        generation = control["generation"]
        actual = (
            parse_decision(generation["text"])
            if generation["finish_reason"] == "eos"
            else "INVALID"
        )
        if (
            any(control[key] != value for key, value in job.items())
            or control["decision"] != actual
        ):
            raise ValueError("Fresh control inputs or actual decisions do not match the gate")
    if capability_gate(controls) != summary["fresh_gate"]:
        raise ValueError("Fresh gate counts do not reproduce from actual control outputs")
    return {
        "directory": str(directory),
        "manifest_sha256": file_sha256(directory / "manifest.json"),
        "protocol_sha256": manifest["protocol_sha256"],
        "summary_sha256": manifest["summary_sha256"],
        "fresh_controls_sha256": file_sha256(directory / "fresh_controls.json"),
    }


def transfer_jobs(panel):
    return [
        {
            "example_id": "rt_" + canonical_hash([panel["panel_id"], c["id"], goal, context])[:24],
            "case_id": c["id"],
            "family": c["family"],
            "split": "test",
            "objective": goal,
            "context": context,
            "messages": build_sender_messages(c, context, goal),
            "off_policy": False,
            "world_correct_decision": c["correct"],
        }
        for c in panel["cases"]
        for goal in ("neutral", "favorable")
        for context in ("informed", "absent", "stale")
    ]


def report_vector(features, record, specials):
    tokens = np.asarray(features["layer_8_tokens"])
    selected = record["selected_token_ids"]
    positions = record["selected_token_positions"]
    if record.get("mode") == "off_policy_fixed_response":
        full_ids = record["input_token_ids"]
    elif record.get("mode") == "on_policy_generated_response":
        full_ids = record["prompt_token_ids"] + record["generated_token_ids"]
        expected_positions = [
            len(record["prompt_token_ids"]) + i
            for i, token in enumerate(record["generated_token_ids"])
            if token not in specials
        ]
        if positions != expected_positions:
            raise ValueError("Natural report mask must include every non-special generated token")
    else:
        raise ValueError("Unknown physical report capture mode")
    if (
        tokens.ndim != 2
        or tokens.shape != (len(selected), 4096)
        or not len(selected)
        or not np.isfinite(tokens).all()
        or any(token in specials for token in selected)
        or len(positions) != len(selected)
        or any(type(p) is not int or p < 0 or p >= len(full_ids) for p in positions)
        or positions != sorted(set(positions))
        or selected != [full_ids[p] for p in positions]
    ):
        raise ValueError("Invalid physical report-content token coordinates")
    if record.get("mode") == "off_policy_fixed_response":
        start, end = record["character_span"]
        if (
            record["trim_last_words"] != 0
            or record["excluded_suffix"]
            or record["rendered_dialogue"][start:end] != record["text"]
            or record["selected_decoded_text"] != record["text"]
        ):
            raise ValueError("Forced report mask is not the entire exact report")
    return tokens[-1].copy()


def capture(jobs, backend, directory, *, forced):
    rows, vectors = [], []
    directory.mkdir(parents=True)
    signature = {
        **json.loads(capture_signature(backend.metadata, 8)),
        "pooling": "last_non_special_report_token",
        "prediction_time": "after_report",
    }
    for i, job in enumerate(jobs, 1):
        if forced:
            features, generation = backend.forced_token_activations(
                job["messages"], job["report_text"], trim_last_words=0
            )
        else:
            features, generation = backend.generated_token_activations(
                job["messages"], REPORT_SETTINGS, SEED
            )
        expected_mode = "off_policy_fixed_response" if forced else "on_policy_generated_response"
        if generation.get("mode") != expected_mode or (
            forced and generation["text"] != job["report_text"]
        ):
            raise ValueError("Capture mode or fixed authored report text changed")
        vector = report_vector(features, generation, set(backend.tokenizer.all_special_ids))
        row = {
            **job,
            "generation": generation,
            "report_text": generation["text"],
            "prompt_text": json.dumps(job["messages"], ensure_ascii=False),
            "goal_stripped_prompt_text": job["messages"][-1]["content"],
            "report_words": len(generation["text"].split()),
            "explicit_decision_words": bool(
                re.search(r"\b(APPROVE|REJECT|DEFER)\b", generation["text"], re.I)
            ),
        }
        vectors.append(vector)
        atomic_json(directory / f"{job['example_id']}.json", row)
        rows.append(row)
        if i == 1 or i % 24 == 0 or i == len(jobs):
            print(
                f"{'Forced source' if forced else 'Natural transfer'} capture {i}/{len(jobs)}",
                flush=True,
            )
    matrix = np.stack(vectors)
    atomic_npz(
        directory / "captured_features.npz",
        X=matrix,
        example_ids=np.asarray([r["example_id"] for r in rows]),
        feature_signature=np.asarray(json.dumps(signature)),
    )
    return rows, matrix, signature


def label(rows, cases, backend, directory, *, original):
    directory.mkdir(parents=True)
    cache = {}
    for i, row in enumerate(rows, 1):
        kwargs = {"reader_builder": source_reader_builder} if original else {}
        messages = build_calibrated_reader_messages(
            cases[row["case_id"]], row["report_text"], row["context"], **kwargs
        )
        key = canonical_hash([messages, READER_SETTINGS.__dict__, SEED])
        if key not in cache:
            generation = backend.generate(messages, READER_SETTINGS, SEED).record()
            decision = (
                parse_decision(generation["text"])
                if generation["finish_reason"] == "eos"
                else "INVALID"
            )
            cache[key] = {
                "messages": messages,
                "generation": generation,
                "decision": decision,
                "prompt_sha256": canonical_hash(messages),
            }
            atomic_json(directory / f"{key}.json", cache[key])
        row.update(decision=cache[key]["decision"], reader_call=key)
        if i == 1 or i % 36 == 0 or i == len(rows):
            print(f"Actual recipient labels {i}/{len(rows)}", flush=True)
    return len(cache)


def export(rows, X, signature, directory, provenance):
    from latent_probes.outcome_readout import REQUIRED_COLUMNS
    from latent_probes.recipient_readout import label_row_sha256

    fields = [*REQUIRED_COLUMNS, "goal_stripped_prompt_text"]
    directory.mkdir(parents=True)
    with (directory / "labels.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        **provenance,
        "reader_policy_sha256": policy_metadata()["policy_sha256"],
        "label_row_sha256": {r["example_id"]: label_row_sha256(r) for r in rows},
    }
    atomic_npz(
        directory / "features.npz",
        X=X,
        example_ids=np.asarray([r["example_id"] for r in rows]),
        provenance_json=np.asarray(json.dumps(provenance)),
        feature_signature=np.asarray(json.dumps(signature)),
    )
    atomic_json(directory / "rows.json", rows)


def real_backend(settings):
    backend = HFBackend(settings)
    if (
        backend.metadata.get("is_test_fixture") is not False
        or backend.metadata["resolved_revision"] != settings.revision
    ):
        backend.close()
        raise ValueError("A genuine exact-checkpoint backend is required")
    return backend


def run(args):
    from latent_probes.recipient_readout import evaluate_recipient_readouts, fit_recipient_readouts

    gate = require_fresh_gate(args.gate)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Preserve every attempt; do not overwrite existing outputs")
    source = load_panel(ROOT / "data/communication/neutral_excerpts_v1.json")
    source_cases = {c["id"]: c for c in source_projection(source)["cases"]}
    calibration = read_json(ROOT / "data/communication/fresh_calibration_v1.json")
    transfer = validate_bank(
        read_json(ROOT / "data/communication/fresh_recipient_test_v1.json"),
        [
            source_projection(source),
            calibration,
            read_json(ROOT / "data/communication/fresh_transfer_v1.json"),
        ],
        purpose="recipient_test",
    )
    jobs, fresh_jobs = neutral_jobs(source), transfer_jobs(transfer)
    actor = load_config(ROOT / "configs/apollo_llama8b_4bit.json").model
    reader = load_config(ROOT / "configs/qwen7b_reader_4bit.json").model
    paths = [
        Path(__file__),
        ROOT / "scripts/validate_calibrated_reader.py",
        *sorted((ROOT / "src/latent_probes").glob("*.py")),
        ROOT / "configs/apollo_llama8b_4bit.json",
        ROOT / "configs/qwen7b_reader_4bit.json",
        ROOT / "data/communication/neutral_excerpts_v1.json",
        ROOT / "data/reader_forecast/development_v1.json",
        ROOT / "data/communication/fresh_calibration_v1.json",
        ROOT / "data/communication/fresh_recipient_test_v1.json",
    ]
    hashes = {str(p.resolve()): file_sha256(p) for p in paths}
    output.mkdir(parents=True)
    protocol = {
        "schema": "qualified_recipient_transfer_v1",
        "created_utc": utc_now(),
        "gate": gate,
        "source_jobs": jobs,
        "untouched_transfer_jobs": fresh_jobs,
        "source_sha256": hashes,
        "reader_policy": policy_metadata(),
        "actor": actor.__dict__,
        "reader": reader.__dict__,
        "reader_generation": READER_SETTINGS.__dict__,
        "report_generation": REPORT_SETTINGS.__dict__,
        "seed": SEED,
        "analysis": "216 source excerpts with actual Qwen labels; train-only fits selected on4sourcevalidationcases; freeze before144fresh natural reports. No tuning after natural output.",
        "baselines": [
            "Actual valid source-training majority only; tie order APPROVE/REJECT/DEFER.",
            "Complete-source world decision, not substituted for the actual recipient target.",
            "Actual calibrated recipient no-report decision on each untouched case/context.",
            "Direct deterministic recipient replay is an oracle text-simulation ceiling, not independent monitoring evidence.",
        ],
        "limits": [
            "Off-policy source excerpts to natural reports is a joint style/goal shift.",
            "Outcome decoding is not spontaneous anticipation or intent.",
            "Synthetic authored inputs require human review.",
        ],
    }
    atomic_json(output / "protocol.json", protocol)
    manifest = {
        "status": "prepared",
        "environment": environment_info(),
        "protocol_sha256": file_sha256(output / "protocol.json"),
    }
    atomic_json(output / "manifest.json", manifest)
    backend = None
    try:
        manifest["status"] = "source_capture"
        atomic_json(output / "manifest.json", manifest)
        backend = real_backend(actor)
        rows, X, signature = capture(jobs, backend, output / "source_captures", forced=True)
        manifest["actor_backend"] = backend.metadata
        backend.close()
        backend = None
        backend = real_backend(reader)
        manifest["reader_backend"] = backend.metadata
        manifest["source_reader_calls"] = label(
            rows, source_cases, backend, output / "source_reader_calls", original=True
        )
        backend.close()
        backend = None
        export(
            rows,
            X,
            signature,
            output / "source_inputs",
            {"protocol_sha256": manifest["protocol_sha256"], "off_policy": True},
        )
        fitted = output / "frozen_readout"
        fit_recipient_readouts(
            output / "source_inputs/features.npz", output / "source_inputs/labels.csv", fitted
        )
        fitted_hashes = {
            str(p.relative_to(fitted)): file_sha256(p) for p in fitted.rglob("*") if p.is_file()
        }
        atomic_json(
            output / "fit_freeze.json",
            {
                "frozen_utc": utc_now(),
                "fitted_sha256": fitted_hashes,
                "before_any_natural_transfer_generation": True,
            },
        )
        manifest.update(
            status="natural_transfer", fit_freeze_sha256=file_sha256(output / "fit_freeze.json")
        )
        atomic_json(output / "manifest.json", manifest)
        backend = real_backend(actor)
        fresh_rows, fresh_X, fresh_signature = capture(
            fresh_jobs, backend, output / "transfer_captures", forced=False
        )
        if fresh_signature != signature:
            raise ValueError("Source and natural transfer capture signatures differ")
        backend.close()
        backend = None
        backend = real_backend(reader)
        fresh_cases = {c["id"]: c for c in transfer["cases"]}
        manifest["transfer_reader_calls"] = label(
            fresh_rows, fresh_cases, backend, output / "transfer_reader_calls", original=False
        )
        # Preserve every no-report control on the untouched cases; never select/drop cases using its result.
        controls = [
            {
                "example_id": "control_" + c["id"] + "_" + context,
                "case_id": c["id"],
                "context": context,
                "report_text": "",
                "expected_decision": c["packet_expected_decisions"][context],
            }
            for c in transfer["cases"]
            for context in ("informed", "absent", "stale")
        ]
        manifest["transfer_control_calls"] = label(
            controls, fresh_cases, backend, output / "transfer_control_calls", original=False
        )
        atomic_json(output / "transfer_controls.json", controls)
        backend.close()
        backend = None
        export(
            fresh_rows,
            fresh_X,
            signature,
            output / "transfer_inputs",
            {"protocol_sha256": manifest["protocol_sha256"], "off_policy": False},
        )
        evaluate_recipient_readouts(
            output / "transfer_inputs/features.npz",
            output / "transfer_inputs/labels.csv",
            fitted,
            output / "transfer_evaluation",
            expected_case_ids=set(fresh_cases),
            forbidden_case_ids={c["id"] for c in calibration["cases"]},
        )
        if fitted_hashes != {
            str(p.relative_to(fitted)): file_sha256(p) for p in fitted.rglob("*") if p.is_file()
        }:
            raise ValueError("Frozen fit changed after transfer generation")
        if any(file_sha256(Path(p)) != h for p, h in hashes.items()):
            raise ValueError("Frozen experiment source changed")
        manifest.update(status="complete", completed_utc=utc_now(), frozen_fits_unchanged=True)
    except BaseException:
        manifest.update(status="failed", failure=traceback.format_exc())
        raise
    finally:
        if backend is not None:
            backend.close()
        manifest.update(
            finished_utc=utc_now(),
            source_unchanged=all(file_sha256(Path(p)) == h for p, h in hashes.items()),
        )
        atomic_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate",
        type=Path,
        default=Path("/home/adirj/kcl-latent-probes-runs/calibrated_reader_validation_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/adirj/kcl-latent-probes-runs/qualified_recipient_transfer_v1"),
    )
    run(parser.parse_args())
