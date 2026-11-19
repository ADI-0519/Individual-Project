"""Natural communication pilot: neutral fitting precedes pressure generation.

The target is the observed reader decision under the packet attributed to that
reader. Post-report features do not establish a pre-report forecast or belief.
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .apollo import capture_signature
from .apollo_probe import score_apollo_probe
from .backends import HFBackend
from .config import GenerationSettings
from .forecast_panel import build_reader_messages, control_jobs, load_forecast_panel, sender_jobs
from .reader_panel import parse_decision
from .storage import (
    atomic_json,
    atomic_npz,
    canonical_hash,
    environment_info,
    file_sha256,
    utc_now,
)


def _persist_record(path, record):
    record = dict(record)
    record["record_sha256"] = canonical_hash(record)
    atomic_json(path, record)
    return record


def _artifact_hashes(directory):
    return {
        str(path.relative_to(directory)): file_sha256(path)
        for path in sorted(Path(directory).rglob("*"))
        if path.is_file()
    }


def _export(rows, destination, signature, provenance):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    matrices, identifiers = [], []
    for row in rows:
        with np.load(row["absolute_feature_path"], allow_pickle=False) as archive:
            tokens = np.asarray(archive["X"])
        if tokens.ndim != 2 or not len(tokens) or not np.isfinite(tokens).all():
            raise ValueError("Natural report requires a nonempty finite token matrix.")
        matrices.append(tokens[-1].copy())
        identifiers.append(row["example_id"])
    atomic_npz(
        destination / "features.npz",
        X=np.stack(matrices),
        example_ids=np.asarray(identifiers),
        feature_signature=np.asarray(signature),
        provenance_json=np.asarray(
            json.dumps(
                {
                    **provenance,
                    "record_sha256": {row["example_id"]: row["record_sha256"] for row in rows},
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
    )
    fields = [
        "example_id",
        "case_id",
        "family",
        "objective",
        "split",
        "decision",
        "prompt_text",
        "report_text",
    ]
    with (destination / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{
                        key: row[key] for key in fields if key not in {"prompt_text", "report_text"}
                    },
                    "prompt_text": json.dumps(row["messages"], ensure_ascii=False),
                    "report_text": row["generation"]["text"],
                }
            )
    return destination / "features.npz", destination / "labels.csv"


def run_forecast_experiment(
    config,
    panel_path,
    output_dir,
    *,
    apollo_probe=None,
    backend_factory=HFBackend,
    fit_function=None,
    score_function=None,
    progress=print,
):
    """Run one immutable development panel with a caller-injectable real backend."""
    from .outcome_readout import evaluate_frozen_readout, fit_evaluate_readout

    fit_function = fit_function or fit_evaluate_readout
    score_function = score_function or evaluate_frozen_readout
    panel = load_forecast_panel(panel_path)
    jobs, controls = sender_jobs(panel), control_jobs(panel)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("Use a fresh forecast-pilot directory; original results stay frozen.")
    if len(config.model.capture_layers) != 1 or config.sender.do_sample:
        raise ValueError(
            "This prespecified pilot requires one capture layer and greedy generation."
        )
    output.mkdir(parents=True)
    layer = config.model.capture_layers[0]
    cases = {case["id"]: case for case in panel["cases"]}
    report_settings = GenerationSettings(max_new_tokens=panel["max_new_tokens"], do_sample=False)
    reader_settings = GenerationSettings(max_new_tokens=8, do_sample=False)
    plan = {
        "panel": panel,
        "sender_jobs": jobs,
        "control_jobs": controls,
        "sender_settings": report_settings.__dict__,
        "reader_settings": reader_settings.__dict__,
        "target": "observed decision under sender-attributed reader packet",
        "feature_time": "last non-special generated report token, after the report",
        "split_scope": "case-held-out development; families appear across splits",
        "prospective_order": "controls, neutral generation, neutral-only fit, favorable generation, frozen scoring",
    }
    atomic_json(output / "plan.json", plan)
    manifest = {
        "schema": "natural_reader_forecast_v1",
        "status": "preparing",
        "created_at": utc_now(),
        "environment": environment_info(),
        "plan_sha256": canonical_hash(plan),
        "panel_sha256": file_sha256(panel_path),
        "config_sha256": canonical_hash(config.to_dict()),
        "limitations": [
            "Authored development material, not a human-audited or family-held-out benchmark.",
            "Observed reader decisions are labels; source correctness is a separate diagnostic.",
            "Natural reports require factuality/disclosure audit before omission or intent claims.",
            "Post-report decodability is not evidence of an anticipated second-order belief.",
            "The primary reader packet equals the sender-attributed packet; later independent replay is separate.",
        ],
    }
    atomic_json(output / "manifest.json", manifest)
    backend, rows, control_records = None, [], []
    started = time.perf_counter()
    try:
        if progress:
            progress("Loading pinned model for natural reader-forecast pilot", flush=True)
        backend = backend_factory(config.model)
        metadata = backend.metadata
        if type(metadata.get("is_test_fixture")) is not bool:
            raise ValueError("Backend must explicitly declare is_test_fixture.")
        token_signature = capture_signature(metadata, layer)
        signature = json.dumps(
            {
                **json.loads(token_signature),
                "pooling": "last_non_special_natural_report_token",
                "prediction_time": "after_report",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        manifest.update(
            status="running",
            backend=metadata,
            is_test_fixture=metadata["is_test_fixture"],
            feature_signature=signature,
            load_seconds=time.perf_counter() - started,
        )
        atomic_json(output / "manifest.json", manifest)
        for index, job in enumerate(controls, 1):
            result = backend.generate(
                job["messages"], reader_settings, panel["seed"], capture=False
            )
            record = {
                **job,
                "generation": result.record(),
                "decision": parse_decision(result.text),
                "is_test_fixture": metadata["is_test_fixture"],
            }
            control_records.append(
                _persist_record(output / "controls" / (job["example_id"] + ".json"), record)
            )
            if progress and index % 12 == 0:
                progress(f"Reader controls {index}/{len(controls)}", flush=True)

        def capture_job(job):
            features, generation = backend.generated_token_activations(
                job["messages"], report_settings, panel["seed"]
            )
            tokens = np.asarray(features[f"layer_{layer}_tokens"])
            if tokens.ndim != 2 or not len(tokens) or not np.isfinite(tokens).all():
                raise ValueError("Natural sender produced empty or nonfinite activations.")
            feature_file = Path("report_features") / (job["example_id"] + ".npz")
            atomic_npz(
                output / feature_file, X=tokens, feature_signature=np.asarray(token_signature)
            )
            _persist_record(
                output / "senders" / (job["example_id"] + ".json"),
                {
                    **job,
                    "generation": generation,
                    "feature_file": str(feature_file),
                    "feature_sha256": file_sha256(output / feature_file),
                    "is_test_fixture": metadata["is_test_fixture"],
                },
            )
            messages = build_reader_messages(
                cases[job["case_id"]], generation["text"], job["context"]
            )
            reader = backend.generate(messages, reader_settings, panel["seed"], capture=False)
            row = {
                **job,
                "generation": generation,
                "reader_messages": messages,
                "reader_generation": reader.record(),
                "decision": parse_decision(reader.text),
                "world_correct_decision": cases[job["case_id"]]["correct"],
                "feature_file": str(feature_file),
                "feature_sha256": file_sha256(output / feature_file),
                "is_test_fixture": metadata["is_test_fixture"],
            }
            row = _persist_record(output / "reports" / (job["example_id"] + ".json"), row)
            rows.append({**row, "absolute_feature_path": output / feature_file})
            if progress:
                progress(
                    f"Natural report + reader {len(rows)}/{len(jobs)}: {job['objective']} / {job['case_id']} / {row['decision']}",
                    flush=True,
                )

        for job in jobs:
            if job["objective"] == "neutral":
                capture_job(job)
        provenance = {
            "plan_sha256": manifest["plan_sha256"],
            "panel_sha256": manifest["panel_sha256"],
            "backend": metadata,
            "target": plan["target"],
            "feature_time": plan["feature_time"],
        }
        features_path, labels_path = _export(
            rows, output / "neutral_dataset", signature, provenance
        )
        if progress:
            progress(
                "Fitting neutral-only readouts before any favorable reports are generated",
                flush=True,
            )
        fit_report = fit_function(
            features_path, labels_path, output / "neutral_readout", allow_two_class_diagnostic=True
        )
        frozen_hashes = _artifact_hashes(output / "neutral_readout")
        atomic_json(
            output / "frozen_readout.json",
            {
                "frozen_at": utc_now(),
                "artifacts": frozen_hashes,
                "neutral_reports_completed": len(rows),
                "favorable_reports_completed": 0,
            },
        )
        manifest["neutral_fit_status"] = fit_report.get("status")
        atomic_json(output / "manifest.json", manifest)
        for job in jobs:
            if job["objective"] != "neutral":
                capture_job(job)
        _export(rows, output / "all_dataset", signature, provenance)
        test_features, test_labels = _export(
            [row for row in rows if row["split"] == "test"],
            output / "test_dataset",
            signature,
            provenance,
        )
        if _artifact_hashes(output / "neutral_readout") != frozen_hashes:
            raise ValueError("Neutral fitted readout artifacts changed during pressure generation.")
        if fit_report.get("status") != "not_trainable":
            score_function(
                test_features, test_labels, output / "neutral_readout", output / "frozen_transfer"
            )
        matrices, token_ids = [], []
        for row in rows:
            with np.load(row["absolute_feature_path"], allow_pickle=False) as archive:
                tokens = archive["X"]
                matrices.append(tokens.copy())
                token_ids.extend([row["example_id"]] * len(tokens))
        atomic_npz(
            output / "apollo_tokens.npz",
            X=np.concatenate(matrices),
            example_ids=np.asarray(token_ids),
            feature_signature=np.asarray(token_signature),
        )
        if apollo_probe is not None:
            score_apollo_probe(
                output / "apollo_tokens.npz", apollo_probe, output / "apollo_scores.csv"
            )
        with (output / "report_audit.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "example_id",
                    "case_id",
                    "objective",
                    "context",
                    "report_text",
                    "finish_reason",
                    "within_word_budget",
                    "factuality_audit",
                    "disclosure_audit",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        **{
                            key: row[key]
                            for key in ("example_id", "case_id", "objective", "context")
                        },
                        "report_text": row["generation"]["text"],
                        "finish_reason": row["generation"].get("finish_reason"),
                        "within_word_budget": len(row["generation"]["text"].split())
                        <= panel["word_budget"],
                        "factuality_audit": "",
                        "disclosure_audit": "",
                    }
                )
        full_controls = [r for r in control_records if r["kind"] == "full_evidence"]
        summary = {
            "n_natural_reports": len(rows),
            "n_primary_reader_calls": len(rows),
            "n_reader_controls": len(control_records),
            "full_evidence_controls": {
                "n": len(full_controls),
                "correct": sum(r["decision"] == r["expected_decision"] for r in full_controls),
            },
            "neutral_fit_status": fit_report.get("status"),
            "by_condition": {},
            "is_test_fixture": metadata["is_test_fixture"],
        }
        prior_lookup = {
            (r["case_id"], r["context"]): r["decision"]
            for r in control_records
            if r["kind"] == "no_report"
        }
        for objective, split in (
            ("neutral", "train"),
            ("neutral", "validation"),
            ("neutral", "test"),
            ("favorable", "test"),
        ):
            selected = [r for r in rows if r["objective"] == objective and r["split"] == split]
            valid = [r for r in selected if r["decision"] != "INVALID"]
            summary["by_condition"][f"{objective}_{split}"] = {
                "n": len(selected),
                "n_valid_targets": len(valid),
                "n_invalid_targets": len(selected) - len(valid),
                "decision_counts": dict(Counter(r["decision"] for r in selected)),
                "source_world_rule_accuracy_all_conditions": sum(
                    r["decision"] == r["world_correct_decision"] for r in selected
                )
                / len(selected),
                "no_report_reader_accuracy_all_conditions": sum(
                    r["decision"] != "INVALID"
                    and r["decision"] == prior_lookup[r["case_id"], r["context"]]
                    for r in selected
                )
                / len(selected),
                "source_world_rule_accuracy_valid_targets": (
                    sum(r["decision"] == r["world_correct_decision"] for r in valid) / len(valid)
                    if valid
                    else None
                ),
                "no_report_reader_accuracy_valid_targets": (
                    sum(r["decision"] == prior_lookup[r["case_id"], r["context"]] for r in valid)
                    / len(valid)
                    if valid
                    else None
                ),
                "length_terminated_reports": sum(
                    r["generation"].get("finish_reason") == "length" for r in selected
                ),
            }
        atomic_json(output / "summary.json", summary)
        manifest.update(
            status="complete",
            summary_sha256=file_sha256(output / "summary.json"),
            source_artifacts_unchanged=_artifact_hashes(output / "neutral_readout")
            == frozen_hashes,
            n_natural_reports=len(rows),
            n_reader_controls=len(control_records),
            n_primary_reader_calls=len(rows),
        )
    except BaseException as exc:
        manifest.update(
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc),
            completed_reports=len(rows),
            completed_controls=len(control_records),
        )
        raise
    finally:
        if backend is not None:
            backend.close()
        manifest.update(finished_at=utc_now(), elapsed_seconds=time.perf_counter() - started)
        atomic_json(output / "manifest.json", manifest)
    return manifest
