"""Frozen prose-only evidence extraction, followed by an explicit belief fold.

This baseline is not a structured-state oracle. Only original prose messages
reach the model. All outputs, including malformed/capped outputs, are retained;
invalid outcomes have null scores and count incorrect in full-denominator
analysis. Preparing prefixes never loads model weights.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from .backends import HFBackend
from .config import GenerationSettings, ModelSettings
from .main_study_bank import bank_digest, iter_rows, load_bank, prompt_digest, verify_review_ledger
from .main_study_run import MODELS, ROOT, _native_output, require, snapshot_files, token_ids
from .storage import atomic_json, canonical_hash, environment_info, file_sha256, read_json, utc_now

DEFAULT_PROTOCOL = ROOT / "data/belief_main/text_monitor_protocol_v1.json"
POLICY_FILES = (
    "src/latent_probes/main_study_monitor.py",
    "scripts/run_main_study_monitor.py",
    "src/latent_probes/main_study_bank.py",
    "src/latent_probes/main_study_review.py",
    "src/latent_probes/main_study_design.py",
    "src/latent_probes/main_study_run.py",
    "src/latent_probes/backends.py",
    "src/latent_probes/config.py",
    "src/latent_probes/storage.py",
)


def load_monitor_protocol(path: str | Path = DEFAULT_PROTOCOL) -> dict:
    """Validate the versioned fixed policy before producing model inputs."""
    protocol = read_json(path)
    fixed = {
        "schema": "main-study-text-monitor-protocol-v1",
        "status": "specified_unexecuted",
        "method": "prose_evidence_extractor_then_symbolic_fold",
        "is_symbolic_oracle": False,
        "model_id": MODELS["qwen"]["model_id"],
        "revision": MODELS["qwen"]["revision"],
        "dtype": "bfloat16",
        "quantization": "4bit",
        "max_input_tokens": 2048,
        "max_new_tokens": 384,
        "do_sample": False,
        "seed": 20261121,
        "retries": 0,
    }
    for key, value in fixed.items():
        require(
            type(protocol.get(key)) is type(value) and protocol[key] == value,
            f"Text monitor policy differs: {key}",
        )
    for key in ("system_prompt", "user_prefix", "scope", "fold"):
        require(
            isinstance(protocol.get(key), str) and protocol[key].strip(),
            f"Missing monitor policy: {key}",
        )
    require(protocol.get("parser", {}).get("version") == 1, "Unsupported monitor parser")
    return protocol


def _prose_messages(messages: list[dict]) -> list[dict]:
    require(isinstance(messages, list) and messages, "Need original prose messages only")
    for message in messages:
        require(
            isinstance(message, dict)
            and set(message) == {"role", "content"}
            and message["role"] in {"system", "user", "assistant"}
            and isinstance(message["content"], str)
            and message["content"].strip(),
            "Source messages must contain only role and prose content",
        )
    return messages


def monitor_messages(messages: list[dict], protocol: dict) -> list[dict]:
    """Build model payload from prose alone; metadata/gold have no parameter."""
    prose = _prose_messages(messages)
    return [
        {"role": "system", "content": protocol["system_prompt"]},
        {
            "role": "user",
            "content": protocol["user_prefix"]
            + json.dumps(prose, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Nonfinite JSON constant: " + value)


def parse_monitor_output(raw_text: str, messages: list[dict], *, finish_reason="eos") -> dict:
    """Strictly parse extraction and fold it without accessing symbolic states.

    Literal occurrence checks prevent invented physical names, but do not prove
    semantic correctness. A valid extraction can still be wrong. No parser repair,
    markdown removal, fuzzy name matching, coercion, retry or gold lookup occurs.
    """
    prose = "\n".join(message["content"] for message in _prose_messages(messages))
    invalid = {
        "valid": False,
        "prediction": None,
        "score": None,
        "extraction": None,
        "final_location": None,
        "invalid_reason": None,
    }
    try:
        require(isinstance(raw_text, str), "Generated output must be text")
        require(finish_reason == "eos", "Generation did not finish at EOS")
        data = json.loads(
            raw_text.strip(), object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
        require(
            isinstance(data, dict)
            and set(data) == {"initial_location", "updates", "queried_location"},
            "Extraction must have exactly the three required keys",
        )
        updates = data["updates"]
        require(
            isinstance(updates, list) and len(updates) in {1, 2},
            "Need one or two ordered candidate updates",
        )
        locations = [data["initial_location"], data["queried_location"]]
        for update in updates:
            require(
                isinstance(update, dict)
                and set(update) == {"location", "accepted_access"}
                and type(update["accepted_access"]) is bool,
                "Update requires exactly location and boolean accepted_access",
            )
            locations.append(update["location"])
        for place in locations:
            require(
                isinstance(place, str)
                and 0 < len(place) <= 160
                and place == place.strip()
                and not any(ord(ch) < 32 for ch in place)
                and re.search(r"(?<!\w)" + re.escape(place) + r"(?!\w)", prose) is not None,
                "Location must occur verbatim in source prose",
            )
        require(len(set(locations)) <= 2, "Binary study permits at most two physical names")
        current = data["initial_location"]
        for update in updates:
            if update["accepted_access"]:
                current = update["location"]
        prediction = int(current == data["queried_location"])
        return {
            "valid": True,
            "prediction": prediction,
            "score": float(prediction),
            "extraction": data,
            "final_location": current,
            "invalid_reason": None,
        }
    except (ValueError, TypeError, RecursionError) as error:
        return {**invalid, "invalid_reason": str(error)}


def prepare_monitor(
    bank_path,
    output,
    *,
    protocol_path=DEFAULT_PROTOCOL,
    tokenizer=None,
    model_config=None,
    test_fixture=False,
    bank_loader=None,
    row_builder=None,
):
    """Prepare every exact monitor prefix using tokenizer/config only."""
    require(type(test_fixture) is bool, "Fixture identity must be boolean")
    bank_path, protocol_path = Path(bank_path).resolve(), Path(protocol_path).resolve()
    output = _native_output(output, test_fixture=test_fixture)
    require(
        not output.exists() and not output.is_relative_to(ROOT),
        "Preparation needs a fresh directory outside the source tree",
    )
    policy = load_monitor_protocol(protocol_path)
    bank = (bank_loader or load_bank)(bank_path)
    if tokenizer is None:
        from transformers import AutoConfig, AutoTokenizer

        common = {
            "revision": policy["revision"],
            "local_files_only": True,
            "trust_remote_code": False,
        }
        tokenizer = AutoTokenizer.from_pretrained(policy["model_id"], **common)
        model_config = AutoConfig.from_pretrained(policy["model_id"], **common)
    require(
        model_config is not None
        and model_config._commit_hash == policy["revision"]
        and model_config.model_type == "qwen2",
        "Pinned monitor config differs",
    )
    settings = ModelSettings(
        model_id=policy["model_id"],
        revision=policy["revision"],
        dtype=policy["dtype"],
        quantization=policy["quantization"],
        device="cuda",
        capture_layers=[0],
        max_input_tokens=policy["max_input_tokens"],
        local_files_only=True,
    )
    planned = []
    for row in (row_builder or iter_rows)(bank):
        require(row["split"] in {"train", "development", "test"}, "Unknown split")
        if row["split"] == "train":
            # This fixed method is not fitted; training generations add no data
            # needed for its development comparison or held-out evaluation.
            continue
        case_id = row["case_id"]
        require(
            isinstance(case_id, str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case_id) is not None,
            "Case ID must be a safe journal basename",
        )
        messages = monitor_messages(row["messages"], policy)
        ids = token_ids(tokenizer, messages)
        require(
            len(ids) <= settings.max_input_tokens
            and len(ids) + policy["max_new_tokens"] <= model_config.max_position_embeddings,
            "Monitor prefix exceeds frozen context; truncation forbidden",
        )
        planned.append(
            {
                "case_id": case_id,
                "split": row["split"],
                "messages": row["messages"],
                "prompt_sha256": prompt_digest(row["messages"]),
                "monitor_messages": messages,
                "monitor_messages_sha256": canonical_hash(messages),
                "prompt_token_ids": ids,
                "prompt_token_ids_sha256": canonical_hash(ids),
            }
        )
    require(
        planned
        and len({row["case_id"] for row in planned}) == len(planned)
        and len({row["prompt_sha256"] for row in planned}) == len(planned),
        "Need nonempty unique case IDs and prose prefixes",
    )
    protocol = {
        "schema": "main-study-text-monitor-preparation-v1",
        "created_at": utc_now(),
        "test_fixture": test_fixture,
        "bank_path": str(bank_path),
        "bank_file_sha256": file_sha256(bank_path),
        "bank_sha256": bank_digest(bank),
        "monitor_protocol_path": str(protocol_path),
        "monitor_protocol_sha256": file_sha256(protocol_path),
        "policy_sha256": {str(ROOT / p): file_sha256(ROOT / p) for p in POLICY_FILES},
        "model_settings": asdict(settings),
        "generation_settings": asdict(GenerationSettings(max_new_tokens=384, do_sample=False)),
        "seed": policy["seed"],
        "tokenizer": {
            "chat_template_sha256": canonical_hash(tokenizer.chat_template),
            "special_token_ids": {"eos": tokenizer.eos_token_id, "pad": tokenizer.pad_token_id},
        },
        "decoder_blocks": model_config.num_hidden_layers,
        "environment": environment_info(),
        "planned_rows": planned,
        "preflight": {
            "rows": len(planned),
            "model_weights_loaded": False,
            "model_generations": 0,
            "human_review_validated": False,
            "minimum_tokens": min(len(r["prompt_token_ids"]) for r in planned),
            "maximum_tokens": max(len(r["prompt_token_ids"]) for r in planned),
        },
    }
    output.mkdir(parents=True)
    atomic_json(output / "protocol.json", protocol)
    atomic_json(
        output / "run_manifest.json",
        {
            "schema": "main-study-text-monitor-run-v1",
            "status": "prepared_no_weights",
            "protocol_sha256": file_sha256(output / "protocol.json"),
            "completed_rows": 0,
            "test_fixture": test_fixture,
        },
    )
    return protocol


def _verify_protocol(protocol):
    snapshot_files(
        {
            protocol["bank_path"]: protocol["bank_file_sha256"],
            protocol["monitor_protocol_path"]: protocol["monitor_protocol_sha256"],
            **protocol["policy_sha256"],
        }
    )


def _authorize(protocol, ledger_path, freeze_path, partition, fit_dir):
    require(
        ledger_path is not None and freeze_path is not None,
        "Execution requires a human-review ledger and frozen execution specification",
    )
    review = verify_review_ledger(load_bank(protocol["bank_path"]), ledger_path)
    freeze = read_json(freeze_path)
    require(
        freeze.get("schema") == "main-study-execution-freeze-v1"
        and freeze.get("status") == "frozen"
        and freeze.get("bank_sha256") == protocol["bank_sha256"],
        "Execution specification is not frozen for this bank",
    )
    snapshot_files(freeze.get("source_sha256"))
    require(
        all(
            freeze["source_sha256"].get(path) == digest
            for path, digest in protocol["policy_sha256"].items()
        ),
        "Execution freeze must bind every monitor source file",
    )
    require(
        str(Path(freeze["text_monitor_protocol_path"]).resolve())
        == protocol["monitor_protocol_path"]
        and freeze.get("text_monitor_protocol_sha256") == protocol["monitor_protocol_sha256"],
        "Execution freeze must bind the exact text-monitor policy",
    )
    baseline = Path(freeze["baseline_protocol_path"]).resolve()
    require(
        file_sha256(baseline) == freeze["baseline_protocol_sha256"], "Baseline protocol changed"
    )
    bindings = {
        str(Path(ledger_path).resolve()): file_sha256(ledger_path),
        str(Path(freeze_path).resolve()): file_sha256(freeze_path),
        str(baseline): file_sha256(baseline),
    }
    if partition == "test":
        from .main_study_analysis import load_fit_bundle

        require(fit_dir is not None, "Held-out monitor requires frozen development-selected fits")
        directory = Path(fit_dir).resolve()
        settings, _ = load_fit_bundle(directory)
        require(
            settings.get("test_fixture") == protocol["test_fixture"]
            and settings.get("capture_provenance", {}).get("bank_sha256")
            == protocol["bank_sha256"],
            "Fit bank/fixture identity differs",
        )
        for name in ("settings.json", "heads.npz"):
            path = str(directory / name)
            digest = file_sha256(path)
            require(
                freeze.get("fitted_artifact_sha256", {}).get(path) == digest,
                "Held-out fitted artifacts are absent from execution freeze",
            )
            bindings[path] = digest
    return {"review": review, "source_bindings": bindings}


def _completed(output, protocol, selected):
    journal = output / "row_journal"
    expected = {r["case_id"] for r in selected}
    if journal.exists():
        require(
            all(p.is_dir() and p.name in expected for p in journal.iterdir()),
            "Unexpected monitor row journal entry",
        )
    results = []
    for planned in selected:
        directory = journal / planned["case_id"]
        if not directory.exists():
            continue
        require(
            (directory / "started.json").is_file()
            and (directory / "complete.json").is_file()
            and not (directory / "failure.json").exists(),
            "Incomplete/failed attempted row blocks resumption; no implicit retry",
        )
        row = read_json(directory / "complete.json")
        started = read_json(directory / "started.json")
        require(
            row.get("case_id") == planned["case_id"]
            and row.get("status") == "complete"
            and row.get("test_fixture") is protocol["test_fixture"]
            and all(row.get(key) == value for key, value in planned.items())
            and row.get("protocol_sha256") == file_sha256(output / "protocol.json")
            and started.get("protocol_sha256") == row["protocol_sha256"]
            and started.get("case_id") == row["case_id"]
            and file_sha256(directory / "raw_generation.json") == row["raw_generation_sha256"],
            "Completed monitor row binding changed",
        )
        raw = read_json(directory / "raw_generation.json")
        require(
            row["raw_response"] == raw["raw_response"]
            and raw["generation"]["prompt_token_ids"] == planned["prompt_token_ids"]
            and 0 < len(raw["generation"]["generated_token_ids"]) <= 384
            and all(
                row.get(key) == value
                for key, value in parse_monitor_output(
                    raw["raw_response"],
                    planned["messages"],
                    finish_reason=raw["generation"]["finish_reason"],
                ).items()
            ),
            "Saved monitor parsing changed",
        )
        results.append(row)
    return results


def execute_monitor(
    output,
    *,
    partition,
    ledger_path,
    freeze_path,
    fit_dir=None,
    resume=False,
    backend_factory=HFBackend,
    progress=print,
):
    """Generate one greedy extraction per authorized row; never retry an outcome."""
    output = Path(output).resolve()
    protocol = read_json(output / "protocol.json")
    manifest = read_json(output / "run_manifest.json")
    require(partition in {"development", "test"}, "Unknown execution partition")
    require(manifest.get("status") != "complete", "Completed monitor cannot be rerun")
    require(
        manifest.get("protocol_sha256") == file_sha256(output / "protocol.json"),
        "Prepared monitor protocol changed",
    )
    _verify_protocol(protocol)
    environment = environment_info()
    require(
        all(
            environment[k] == protocol["environment"][k] for k in ("python", "platform", "packages")
        ),
        "Prepared Python/platform/package environment changed",
    )
    authorization = _authorize(protocol, ledger_path, freeze_path, partition, fit_dir)
    if protocol["test_fixture"]:
        require(backend_factory is not HFBackend, "Fixture cannot load real model weights")
    selected = [r for r in protocol["planned_rows"] if r["split"] == partition]
    require(selected, "No rows in requested monitor partition")
    completed = _completed(output, protocol, selected)
    if manifest["status"] == "prepared_no_weights":
        require(
            not resume and not completed and not (output / "attempts").exists(),
            "Fresh monitor preparation must have no attempts",
        )
    else:
        require(
            resume
            and manifest.get("partition") == partition
            and manifest.get("authorization") == authorization,
            "Use explicit resume with identical partition and authorization",
        )
    attempts = output / "attempts"
    attempts.mkdir(exist_ok=True)
    attempt = attempts / f"{len(list(attempts.iterdir())):04d}.json"
    with attempt.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "started_at": utc_now(),
                "explicit_resume": resume,
                "protocol_sha256": manifest["protocol_sha256"],
            },
            stream,
        )
    manifest.update(
        status="running",
        partition=partition,
        authorization=authorization,
        completed_rows=len(completed),
        environment=environment,
    )
    atomic_json(output / "run_manifest.json", manifest)
    pending = [r for r in selected if r["case_id"] not in {c["case_id"] for c in completed}]
    backend = None
    try:
        if pending:
            backend = backend_factory(ModelSettings(**protocol["model_settings"]))
            require(
                bool(backend.metadata["is_test_fixture"]) == protocol["test_fixture"],
                "Fixture/real backend identity mismatch",
            )
            require(
                all(
                    backend.metadata[key] == protocol["model_settings"][key]
                    for key in ("model_id", "dtype", "quantization", "capture_layers")
                )
                and backend.metadata["resolved_revision"] == protocol["model_settings"]["revision"]
                and backend.metadata["model_type"] == "qwen2"
                and backend.metadata["decoder_blocks"] == protocol["decoder_blocks"]
                and canonical_hash(backend.tokenizer.chat_template)
                == protocol["tokenizer"]["chat_template_sha256"]
                and {"eos": backend.tokenizer.eos_token_id, "pad": backend.tokenizer.pad_token_id}
                == protocol["tokenizer"]["special_token_ids"],
                "Pinned monitor backend/tokenizer identity differs",
            )
            backend.model.eval()
            require(not backend.model.training, "Monitor model must be in evaluation mode")
            manifest["backend"] = backend.metadata
            atomic_json(output / "run_manifest.json", manifest)
        for planned in pending:
            _verify_protocol(protocol)
            snapshot_files(authorization["source_bindings"])
            require(
                file_sha256(output / "protocol.json") == manifest["protocol_sha256"],
                "Monitor protocol changed during execution",
            )
            ids = token_ids(backend.tokenizer, planned["monitor_messages"])
            require(ids == planned["prompt_token_ids"], "Monitor prefix changed")
            directory = output / "row_journal" / planned["case_id"]
            directory.mkdir(parents=True, exist_ok=False)
            atomic_json(
                directory / "started.json",
                {
                    "case_id": planned["case_id"],
                    "started_at": utc_now(),
                    "protocol_sha256": manifest["protocol_sha256"],
                },
            )
            try:
                result = backend.generate(
                    planned["monitor_messages"],
                    GenerationSettings(**protocol["generation_settings"]),
                    seed=protocol["seed"],
                    capture=False,
                )
                raw_text = backend.tokenizer.decode(
                    result.generated_token_ids, skip_special_tokens=True
                )
                raw = {
                    "raw_response": raw_text,
                    "raw_response_with_special_tokens": backend.tokenizer.decode(
                        result.generated_token_ids, skip_special_tokens=False
                    ),
                    "generation": result.record(),
                    "normalization": "Parse outer whitespace only; no location normalization",
                }
                atomic_json(directory / "raw_generation.json", raw)
                require(result.prompt_token_ids == ids, "Generated prompt token IDs differ")
                require(
                    0 < len(result.generated_token_ids) <= 384,
                    "Generated token count violates frozen cap",
                )
                row = {
                    **planned,
                    **parse_monitor_output(
                        raw_text, planned["messages"], finish_reason=result.finish_reason
                    ),
                    "raw_response": raw_text,
                    "protocol_sha256": manifest["protocol_sha256"],
                    "raw_generation_sha256": file_sha256(directory / "raw_generation.json"),
                    "test_fixture": protocol["test_fixture"],
                    "status": "complete",
                }
                atomic_json(directory / "complete.json", row)
            except BaseException as error:
                atomic_json(
                    directory / "failure.json",
                    {
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "failed_at": utc_now(),
                        "preserved_files_sha256": {
                            path.name: file_sha256(path)
                            for path in directory.iterdir()
                            if path.is_file()
                        },
                    },
                )
                raise
            manifest["completed_rows"] += 1
            atomic_json(output / "run_manifest.json", manifest)
            if progress:
                progress(f"Monitor completed {manifest['completed_rows']}/{len(selected)} rows")
        _verify_protocol(protocol)
        snapshot_files(authorization["source_bindings"])
        records = _completed(output, protocol, selected)
        require(len(records) == len(selected), "Cannot finalize incomplete monitor records")
        atomic_json(
            output / "records.json",
            {
                "schema": "main-study-text-monitor-v1",
                "bank_sha256": protocol["bank_sha256"],
                "protocol_sha256": manifest["protocol_sha256"],
                "monitor_protocol_sha256": protocol["monitor_protocol_sha256"],
                "model_settings": protocol["model_settings"],
                "partition": partition,
                "test_fixture": protocol["test_fixture"],
                "records": records,
            },
        )
        manifest.update(
            status="complete",
            finished_at=utc_now(),
            records_sha256=file_sha256(output / "records.json"),
            source_bindings=authorization["source_bindings"],
        )
        atomic_json(output / "run_manifest.json", manifest)
        return manifest
    except BaseException as error:
        manifest.update(
            status="failed_or_interrupted",
            error_type=type(error).__name__,
            error=str(error),
            stopped_at=utc_now(),
        )
        atomic_json(output / "run_manifest.json", manifest)
        raise
    finally:
        if backend is not None:
            backend.close()
