"""Auditable sender generation and independent recipient replay.

The default backend runs a real model. Tests inject a clearly marked scripted
backend; there is no fake-output CLI mode.
"""

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from .backends import HFBackend
from .config import ExperimentConfig
from .prompts import (
    build_forecast_messages,
    build_recipient_messages,
    build_sender_messages,
    parse_answer,
)
from .schema import load_scenarios
from .storage import (
    atomic_json,
    atomic_npz,
    canonical_hash,
    code_fingerprint,
    environment_info,
    file_sha256,
    read_json,
    utc_now,
    write_jsonl,
)


def _records(directory):
    return [read_json(path) for path in sorted(Path(directory).glob("*.json"))]


def _select(scenarios_path, limit):
    scenarios = load_scenarios(scenarios_path)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        scenarios = scenarios[:limit]
    return scenarios


def sender_jobs(scenarios, config):
    for scenario in scenarios:
        for objective in config.objectives:
            for assumed in config.assumed_knowledge:
                for seed in config.seeds:
                    identity = [scenario.scenario_id, objective, assumed, seed]
                    yield {
                        "example_id": "s_" + canonical_hash(identity)[:24],
                        "scenario": scenario,
                        "objective": objective,
                        "assumed_knowledge": assumed,
                        "seed": seed,
                    }


def plan(config, scenarios_path, output_path, limit=None):
    scenarios = _select(scenarios_path, limit)
    jobs = list(sender_jobs(scenarios, config))
    recipient_repetitions = len(config.seeds) if config.recipient.do_sample else 1
    payload = {
        "status": "plan_only_no_model_outputs",
        "created_at": utc_now(),
        "scenario_count": len(scenarios),
        "family_count": len({s.family_id for s in scenarios}),
        "sender_generations": len(jobs),
        "elicited_forecasts": len(jobs) if config.elicited_forecast else 0,
        "recipient_generations": (len(jobs) + 3 * len(scenarios)) * 2 * recipient_repetitions,
        "config": config.to_dict(),
        "sender_jobs": [
            {
                **{k: v for k, v in j.items() if k != "scenario"},
                "scenario_id": j["scenario"].scenario_id,
                "messages": build_sender_messages(
                    j["scenario"], j["objective"], j["assumed_knowledge"]
                ),
            }
            for j in jobs
        ],
        "control_recipient_examples": [
            {
                "scenario_id": s.scenario_id,
                "actual_knowledge": actual,
                "condition": kind,
                "messages": build_recipient_messages(s, message, actual),
            }
            for s in scenarios
            for actual in (False, True)
            for kind, message in [
                ("no_message", None),
                ("full_disclosure", s.control_messages["full_disclosure"]),
                ("selective_control", s.control_messages["selective"]),
            ]
        ],
    }
    if Path(output_path).exists():
        raise FileExistsError(f"Plan already exists: {output_path}; choose a new output path")
    atomic_json(output_path, payload)
    return {
        k: v
        for k, v in payload.items()
        if k not in {"config", "sender_jobs", "control_recipient_examples"}
    }


def _prepare(output, identity, resume, stage):
    output = Path(output)
    if output.exists():
        if not resume:
            raise FileExistsError(
                f"Output exists: {output}. Use --resume for the identical experiment or a new directory."
            )
        if not (output / "manifest.json").is_file():
            raise ValueError("Existing output lacks a manifest; choose a new directory")
        manifest = read_json(output / "manifest.json")
        if manifest.get("identity") != identity or manifest.get("stage") != stage:
            raise ValueError("Resume inputs differ from the recorded experiment")
        if manifest.get("environment", {}).get("code_sha256") != code_fingerprint():
            raise ValueError(
                "Source code changed since this run. Use a new output directory to preserve provenance."
            )
        if manifest.get("environment", {}).get("packages") != environment_info()["packages"]:
            raise ValueError("Package versions changed since this run. Use a new output directory.")
        return output, manifest
    if resume:
        raise FileNotFoundError("Cannot resume a nonexistent run")
    output.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "stage": stage,
        "status": "preparing",
        "identity": identity,
        "created_at": utc_now(),
        "environment": environment_info(),
    }
    atomic_json(output / "manifest.json", manifest)
    return output, manifest


def _load_backend(settings, manifest, factory):
    previous = manifest.get("backend", {})
    if previous.get("resolved_revision"):
        settings = replace(settings, revision=previous["resolved_revision"])
    backend = factory(settings)
    # Device placement may legitimately differ on a restarted cluster job.
    fields = (
        "model_id",
        "resolved_revision",
        "dtype",
        "quantization",
        "chat_template_sha256",
        "capture_layers",
        "extraction",
        "positions",
        "is_test_fixture",
    )
    if previous and any(previous.get(k) != backend.metadata.get(k) for k in fields):
        backend.close()
        raise ValueError("Loaded checkpoint/capture specification differs from the original run")
    return backend


def _validate_sender_snapshot(run_dir, manifest):
    """Reject accidental edits to the inputs used by an existing sender run."""
    run_dir = Path(run_dir)
    config = ExperimentConfig.from_dict(read_json(run_dir / "config.json"))
    scenarios = load_scenarios(run_dir / "scenarios.jsonl")
    recorded = manifest.get("identity", {})
    if canonical_hash(config.to_dict()) != recorded.get("config_hash"):
        raise ValueError("Saved sender configuration differs from its manifest")
    if canonical_hash([s.to_dict() for s in scenarios]) != recorded.get("scenarios_hash"):
        raise ValueError("Saved scenario snapshot differs from its sender manifest")
    return config, scenarios


def _validate_feature_file(path, keys, backend_metadata):
    if not isinstance(keys, list) or not keys or len(set(keys)) != len(keys):
        raise ValueError("Invalid activation feature-key metadata")
    expected_keys = {
        f"layer_{layer}_{position}"
        for layer in backend_metadata["capture_layers"]
        for position in backend_metadata["positions"]
    }
    if set(keys) != expected_keys:
        raise ValueError("Activation feature keys differ from the recorded capture specification")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(keys):
                raise ValueError("Activation archive keys differ from its sender record")
            dimensions = set()
            for key in keys:
                vector = archive[key]
                if vector.ndim != 1 or vector.size == 0 or vector.dtype.kind not in "fiu":
                    raise ValueError("Activations must be nonempty numeric vectors")
                if not np.isfinite(vector).all():
                    raise ValueError("Activation vector contains non-finite values")
                dimensions.add(vector.size)
            if len(dimensions) != 1:
                raise ValueError("Activation vectors have inconsistent dimensions")
            if backend_metadata.get("hidden_size") is not None and dimensions != {
                backend_metadata["hidden_size"]
            }:
                raise ValueError("Activation dimension differs from the recorded model")
    except (OSError, TypeError, EOFError) as exc:
        raise ValueError("Activation archive cannot be read safely") from exc


def _validate_sender_record(record, job, messages, run_dir, backend_metadata):
    """Check recorded inputs and feature bytes before reusing a completed item."""
    scenario = job["scenario"]
    expected = {
        "example_id": job["example_id"],
        "scenario_id": scenario.scenario_id,
        "family_id": scenario.family_id,
        "domain": scenario.domain,
        "objective": job["objective"],
        "assumed_knowledge": job["assumed_knowledge"],
        "seed": job["seed"],
        "generation_mode": "natural",
        "messages": messages,
        "prompt_hash": canonical_hash(messages),
        "feature_file": f"features/{job['example_id']}.npz",
        "is_test_fixture": backend_metadata.get("is_test_fixture", False),
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved sender record identity or prompt differs from the experiment")
    checksum = canonical_hash(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )
    if record.get("record_sha256") != checksum:
        raise ValueError("Saved sender record checksum differs; the record was modified")
    feature_path = Path(run_dir) / record["feature_file"]
    if not feature_path.is_file():
        raise ValueError("Checkpoint record is incomplete: its activation file is missing")
    if record.get("feature_sha256") != file_sha256(feature_path):
        raise ValueError("Saved activation checksum differs; the feature file was modified")
    _validate_feature_file(feature_path, record.get("feature_keys"), backend_metadata)


def _check_sender_record_ids(run_dir, jobs, require_complete=False):
    expected = {job["example_id"] for job in jobs}
    existing = {path.stem for path in (Path(run_dir) / "sender").glob("*.json")}
    if existing - expected:
        raise ValueError("Unexpected sender record files in run directory")
    if require_complete and existing != expected:
        raise ValueError("Completed sender run is missing expected records")


def _validate_recipient_record(record, job, scenario, messages, is_test_fixture):
    expected = {
        **job,
        "family_id": scenario.family_id,
        "domain": scenario.domain,
        "full_information_answer": scenario.full_information_answer,
        "messages": messages,
        "prompt_hash": canonical_hash(messages),
        "is_test_fixture": is_test_fixture,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved recipient record identity or prompt differs from the experiment")
    checksum = canonical_hash(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )
    if record.get("record_sha256") != checksum:
        raise ValueError("Saved recipient record checksum differs; the record was modified")
    generation = record.get("generation")
    if not isinstance(generation, dict) or not isinstance(generation.get("text"), str):
        raise ValueError("Recipient generation text is missing or invalid")
    if record.get("answer") != parse_answer(generation["text"]):
        raise ValueError("Recipient answer differs from its raw generation")


def export_annotations(run_dir, output_path):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("Refusing to overwrite an annotation file")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _records(Path(run_dir) / "sender")
    fieldnames = [
        "example_id",
        "scenario_id",
        "family_id",
        "objective",
        "assumed_knowledge",
        "text",
        "factuality",
        "critical_fact_disclosed",
        "ambiguous",
        "label",
        "label_target",
        "annotation_notes",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {**{k: row[k] for k in fieldnames if k in row}, "text": row["generation"]["text"]}
            )
    return {"rows": len(rows), "output": str(output_path), "labels": "blank_pending_manual_audit"}


def run_senders(
    config,
    scenarios_path,
    output_dir,
    limit=None,
    resume=False,
    backend_factory=HFBackend,
    progress=None,
):
    scenarios = _select(scenarios_path, limit)
    snapshot = [s.to_dict() for s in scenarios]
    identity = {
        "config_hash": canonical_hash(config.to_dict()),
        "scenarios_hash": canonical_hash(snapshot),
    }
    output, manifest = _prepare(output_dir, identity, resume, "sender")
    if not resume:
        atomic_json(output / "config.json", config.to_dict())
        write_jsonl(output / "scenarios.jsonl", snapshot)
    backend = None
    try:
        if resume:
            _validate_sender_snapshot(output, manifest)
        jobs = list(sender_jobs(scenarios, config))
        _check_sender_record_ids(
            output, jobs, require_complete=manifest.get("status") == "complete"
        )
        if manifest.get("status") == "complete":
            if manifest.get("sender_records_hash") != canonical_hash(_records(output / "sender")):
                raise ValueError("Completed sender records differ from their manifest")
        backend = _load_backend(config.model, manifest, backend_factory)
        manifest.update(status="running", backend=backend.metadata)
        atomic_json(output / "manifest.json", manifest)
        for index, job in enumerate(jobs, start=1):
            scenario, example_id = job["scenario"], job["example_id"]
            record_path = output / "sender" / f"{example_id}.json"
            feature_path = output / "features" / f"{example_id}.npz"
            messages = build_sender_messages(scenario, job["objective"], job["assumed_knowledge"])
            if record_path.exists():
                existing = read_json(record_path)
                _validate_sender_record(existing, job, messages, output, backend.metadata)
                continue
            result = backend.generate(messages, config.sender, job["seed"], capture=True)
            if not result.features:
                raise ValueError("Backend returned no requested sender activations")
            # Save passive activations before making any explicit forecast query.
            atomic_npz(feature_path, **result.features)
            _validate_feature_file(feature_path, sorted(result.features), backend.metadata)
            record = {
                "example_id": example_id,
                "scenario_id": scenario.scenario_id,
                "family_id": scenario.family_id,
                "domain": scenario.domain,
                "objective": job["objective"],
                "assumed_knowledge": job["assumed_knowledge"],
                "seed": job["seed"],
                "messages": messages,
                "prompt_hash": canonical_hash(messages),
                "generation": result.record(),
                "feature_file": feature_path.relative_to(output).as_posix(),
                "feature_sha256": file_sha256(feature_path),
                "feature_keys": sorted(result.features),
                "generation_mode": "natural",
                "is_test_fixture": backend.metadata.get("is_test_fixture", False),
            }
            if config.elicited_forecast:
                forecast_messages = build_forecast_messages(scenario, messages, result.text)
                forecast = backend.generate(
                    forecast_messages, config.recipient, job["seed"], capture=False
                )
                record["elicited_forecast"] = {
                    "messages": forecast_messages,
                    "generation": forecast.record(),
                    "answer": parse_answer(forecast.text),
                    "interpretation": "post-message elicited forecast, not ground truth of passive anticipation",
                }
            record["record_sha256"] = canonical_hash(record)
            atomic_json(record_path, record)
            if progress:
                progress(f"sender {index}/{len(jobs)} {scenario.scenario_id}")
        rows = _records(output / "sender")
        write_jsonl(output / "senders.jsonl", rows)
        if not (output / "annotations.csv").exists():
            export_annotations(output, output / "annotations.csv")
        manifest.update(
            status="complete",
            completed_at=utc_now(),
            sender_count=len(rows),
            sender_records_hash=canonical_hash(rows),
            scenario_count=len(scenarios),
            family_count=len({s.family_id for s in scenarios}),
        )
        atomic_json(output / "manifest.json", manifest)
        return {
            "output": str(output),
            "sender_count": len(rows),
            "status": "complete",
            "is_test_fixture": backend.metadata.get("is_test_fixture", False),
        }
    except Exception as exc:
        manifest.update(status="failed", failure_type=type(exc).__name__, last_updated=utc_now())
        atomic_json(output / "manifest.json", manifest)
        raise
    finally:
        if backend is not None:
            backend.close()


def replay_recipients(
    run_dir, config=None, output_dir=None, resume=False, backend_factory=HFBackend, progress=None
):
    run_dir = Path(run_dir)
    sender_manifest = read_json(run_dir / "manifest.json")
    if sender_manifest.get("stage") != "sender" or sender_manifest.get("status") != "complete":
        raise ValueError("Recipient replay requires a completed sender run")
    sender_config, scenarios = _validate_sender_snapshot(run_dir, sender_manifest)
    config = config or sender_config
    by_id = {s.scenario_id: s for s in scenarios}
    sender_job_list = list(sender_jobs(scenarios, sender_config))
    _check_sender_record_ids(run_dir, sender_job_list, require_complete=True)
    sender_job_by_id = {job["example_id"]: job for job in sender_job_list}
    senders = _records(run_dir / "sender")
    if len(senders) != sender_manifest["sender_count"]:
        raise ValueError("Sender record count does not match its manifest")
    if canonical_hash(senders) != sender_manifest.get("sender_records_hash"):
        raise ValueError("Sender records differ from their completed-run manifest")
    for row in senders:
        job = sender_job_by_id[row["example_id"]]
        messages = build_sender_messages(
            job["scenario"], job["objective"], job["assumed_knowledge"]
        )
        _validate_sender_record(row, job, messages, run_dir, sender_manifest["backend"])
    identity = {
        "sender_identity": sender_manifest["identity"],
        "sender_records_hash": canonical_hash(senders),
        "recipient_config_hash": canonical_hash(config.to_dict()),
    }
    output, manifest = _prepare(
        output_dir or run_dir / "recipients" / "default", identity, resume, "recipient"
    )
    if not resume:
        atomic_json(output / "config.json", config.to_dict())
    jobs = []
    reader_seeds = config.seeds if config.recipient.do_sample else config.seeds[:1]
    messages_to_replay = [
        {
            "scenario_id": row["scenario_id"],
            "message": row["generation"]["text"],
            "sender_example_id": row["example_id"],
            "message_kind": "generated",
            "objective": row["objective"],
            "assumed_knowledge": row["assumed_knowledge"],
        }
        for row in senders
    ]
    for scenario in scenarios:
        for kind, text in [
            ("no_message", None),
            ("full_disclosure", scenario.control_messages["full_disclosure"]),
            ("selective_control", scenario.control_messages["selective"]),
        ]:
            messages_to_replay.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "message": text,
                    "sender_example_id": None,
                    "message_kind": kind,
                    "objective": None,
                    "assumed_knowledge": None,
                }
            )
    for replay in messages_to_replay:
        for actual in (False, True):
            for seed in reader_seeds:
                job = {**replay, "actual_knowledge": actual, "recipient_seed": seed}
                job["example_id"] = "r_" + canonical_hash(job)[:24]
                jobs.append(job)
    backend = None
    try:
        saved_config = ExperimentConfig.from_dict(read_json(output / "config.json"))
        if canonical_hash(saved_config.to_dict()) != manifest["identity"]["recipient_config_hash"]:
            raise ValueError("Saved recipient configuration differs from its manifest")
        expected_ids = {job["example_id"] for job in jobs}
        existing_ids = {path.stem for path in (output / "records").glob("*.json")}
        if existing_ids - expected_ids:
            raise ValueError("Unexpected recipient record files in run directory")
        if manifest.get("status") == "complete":
            if existing_ids != expected_ids:
                raise ValueError("Completed recipient run is missing expected records")
            if manifest.get("recipient_records_hash") != canonical_hash(
                _records(output / "records")
            ):
                raise ValueError("Completed recipient records differ from their manifest")
        backend = _load_backend(config.model, manifest, backend_factory)
        manifest.update(
            status="running",
            backend=backend.metadata,
            sender_backend=sender_manifest["backend"],
            recipient_prompt_condition="careful_reader_explicit_unknown",
            is_test_fixture=bool(
                backend.metadata.get("is_test_fixture")
                or sender_manifest["backend"].get("is_test_fixture")
            ),
        )
        atomic_json(output / "manifest.json", manifest)
        for index, job in enumerate(jobs, start=1):
            scenario = by_id[job["scenario_id"]]
            record_path = output / "records" / f"{job['example_id']}.json"
            messages = build_recipient_messages(scenario, job["message"], job["actual_knowledge"])
            if record_path.exists():
                _validate_recipient_record(
                    read_json(record_path), job, scenario, messages, manifest["is_test_fixture"]
                )
                continue
            result = backend.generate(
                messages, config.recipient, job["recipient_seed"], capture=False
            )
            record = {
                **job,
                "family_id": scenario.family_id,
                "domain": scenario.domain,
                "full_information_answer": scenario.full_information_answer,
                "messages": messages,
                "prompt_hash": canonical_hash(messages),
                "generation": result.record(),
                "answer": parse_answer(result.text),
                "is_test_fixture": manifest["is_test_fixture"],
            }
            record["record_sha256"] = canonical_hash(record)
            atomic_json(record_path, record)
            if progress:
                progress(f"recipient {index}/{len(jobs)} {scenario.scenario_id}")
        rows = _records(output / "records")
        write_jsonl(output / "recipients.jsonl", rows)
        manifest.update(
            status="complete",
            completed_at=utc_now(),
            recipient_count=len(rows),
            recipient_records_hash=canonical_hash(rows),
            scenario_count=len(scenarios),
        )
        atomic_json(output / "manifest.json", manifest)
        return {
            "output": str(output),
            "recipient_count": len(rows),
            "status": "complete",
            "is_test_fixture": manifest["is_test_fixture"],
        }
    except Exception as exc:
        manifest.update(status="failed", failure_type=type(exc).__name__, last_updated=utc_now())
        atomic_json(output / "manifest.json", manifest)
        raise
    finally:
        if backend is not None:
            backend.close()
