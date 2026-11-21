"""Bound, review-gated main-study feature capture with complete row journals.

Preparation loads a pinned tokenizer/config only. Real execution requires an
accepted human ledger and a separate frozen execution specification. No label,
answer, threshold, or fitted model influences which rows are captured.
"""

from __future__ import annotations

import inspect
import json
import re
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .backends import HFBackend
from .config import ModelSettings
from .crossed_belief_validation import summarize_logits
from .storage import (
    atomic_json,
    atomic_npz,
    canonical_hash,
    environment_info,
    file_sha256,
    read_json,
    utc_now,
)

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "llama": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "model_type": "llama",
    },
    "qwen": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "model_type": "qwen2",
    },
}
POLICY_FILES = (
    "scripts/run_main_study.py",
    "src/latent_probes/main_study_run.py",
    "src/latent_probes/main_study_bank.py",
    "src/latent_probes/main_study_review.py",
    "src/latent_probes/main_study_design.py",
    "src/latent_probes/backends.py",
    "src/latent_probes/config.py",
    "src/latent_probes/storage.py",
    "src/latent_probes/crossed_belief_validation.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def snapshot_files(bindings):
    """Recheck bytes of every explicitly frozen input; never trust mtime."""
    require(isinstance(bindings, dict) and bindings, "Need nonempty source bindings")
    for name, digest in bindings.items():
        require(Path(name).is_file() and file_sha256(name) == digest, f"Changed input: {name}")


def token_ids(tokenizer, messages):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    require(
        isinstance(ids, list) and ids and all(type(i) is int and i >= 0 for i in ids),
        "Tokenizer must return a nonempty flat integer prefix",
    )
    return ids


def _token_map(tokenizer):
    tokens = {}
    for word in ("YES", "NO"):
        ids = tokenizer.encode(word, add_special_tokens=False)
        require(
            len(ids) == 1
            and tokenizer.decode(ids, skip_special_tokens=False) == word
            and ids[0] not in tokenizer.all_special_ids,
            "Require distinct unambiguous bare single-token YES/NO candidates",
        )
        tokens[word] = ids[0]
    require(tokens["YES"] != tokens["NO"], "YES/NO token collision")
    return tokens


def _forward_contract(forward):
    signature = inspect.signature(forward)
    require("logits_to_keep" in signature.parameters, "Forward lacks logits_to_keep support")
    source = inspect.getsourcefile(inspect.unwrap(forward))
    require(source is not None, "Need inspectable installed forward source")
    return {
        "signature": str(
            signature.replace(
                parameters=[p for p in signature.parameters.values() if p.name != "self"]
            )
        ),
        "source_path": str(Path(source).resolve()),
        "source_sha256": file_sha256(source),
    }


def _native_output(output, *, test_fixture):
    path = Path(output).resolve()
    if not test_fixture:
        require(
            "onedrive" not in str(path).lower() and not str(path).startswith("/mnt/"),
            "Real capture outputs must use native storage outside OneDrive and Windows mounts",
        )
    return path


def prepare(
    bank_path,
    output,
    *,
    model_key="llama",
    tokenizer=None,
    model_config=None,
    forward=None,
    test_fixture=False,
    bank_loader=None,
    row_builder=None,
):
    """Bind every exact prefix without loading weights or requesting human approval."""
    from .main_study_bank import bank_digest, iter_rows, load_bank

    require(model_key in MODELS and type(test_fixture) is bool, "Invalid model/fixture choice")
    bank_path = Path(bank_path).resolve()
    output = _native_output(output, test_fixture=test_fixture)
    require(not output.exists(), "Use a new preparation directory; no overwrite")
    require(not output.is_relative_to(ROOT), "Capture outputs must be outside the source tree")
    bank = (bank_loader or load_bank)(bank_path)
    model = MODELS[model_key]
    if tokenizer is None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        common = {
            "revision": model["revision"],
            "local_files_only": True,
            "trust_remote_code": False,
        }
        model_config = AutoConfig.from_pretrained(model["model_id"], **common)
        tokenizer = AutoTokenizer.from_pretrained(model["model_id"], **common)
        forward = AutoModelForCausalLM._model_mapping[type(model_config)].forward
    require(
        model_config is not None
        and model_config._commit_hash == model["revision"]
        and model_config.model_type == model["model_type"],
        "Pinned model config differs",
    )
    blocks = model_config.num_hidden_layers
    require(type(blocks) is int and blocks > 1, "Invalid decoder block count")
    layer = 16 if model_key == "llama" else blocks // 2
    require(0 <= layer < blocks - 1, "Capture must precede final decoder normalization")
    settings = ModelSettings(
        model_id=model["model_id"],
        revision=model["revision"],
        dtype="bfloat16",
        quantization="4bit",
        device="cuda",
        capture_layers=[layer],
        max_input_tokens=2048,
        local_files_only=True,
    )
    tokens = _token_map(tokenizer)
    planned, seen = [], set()
    for raw in (row_builder or iter_rows)(bank):
        row = dict(raw)
        row["row_id"] = row["case_id"]
        require(
            isinstance(row["row_id"], str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", row["row_id"]) is not None,
            "Case IDs must be safe journal basenames",
        )
        require(row["split"] in {"train", "development", "test"}, "Unknown bank split")
        require(row["row_id"] not in seen, "Duplicate bank row ID")
        seen.add(row["row_id"])
        ids = token_ids(tokenizer, row["messages"])
        require(
            len(ids) <= min(settings.max_input_tokens, model_config.max_position_embeddings),
            "Prompt exceeds frozen context; no truncation permitted",
        )
        planned.append(
            {
                **row,
                "messages_sha256": canonical_hash(row["messages"]),
                "prompt_token_ids": ids,
                "prompt_token_ids_sha256": canonical_hash(ids),
            }
        )
    require(
        planned and len({r["messages_sha256"] for r in planned}) == len(planned),
        "Need unique nonempty exact prefixes",
    )
    policy = {str(ROOT / p): file_sha256(ROOT / p) for p in POLICY_FILES}
    protocol = {
        "schema": "main-study-capture-protocol-v1",
        "created_at": utc_now(),
        "test_fixture": test_fixture,
        "bank_path": str(bank_path),
        "bank_file_sha256": file_sha256(bank_path),
        "bank_sha256": bank_digest(bank),
        "policy_sha256": policy,
        "model_settings": asdict(settings),
        "model_key": model_key,
        "capture_layer": layer,
        "decoder_blocks": blocks,
        "model_type": model["model_type"],
        "hidden_size": model_config.hidden_size,
        "vocab_size": model_config.vocab_size,
        "forward_contract": _forward_contract(forward),
        "tokenizer": {
            "chat_template_sha256": canonical_hash(tokenizer.chat_template),
            "special_token_ids": {"eos": tokenizer.eos_token_id, "pad": tokenizer.pad_token_id},
            "answer_token_ids": tokens,
        },
        "environment": environment_info(),
        "planned_rows": planned,
        "preflight": {
            "model_weights_loaded": False,
            "model_forwards": 0,
            "rows": len(planned),
            "minimum_tokens": min(len(r["prompt_token_ids"]) for r in planned),
            "maximum_tokens": max(len(r["prompt_token_ids"]) for r in planned),
            "human_review_validated": False,
        },
        "capture": "Last complete chat-template token including assistant header; block output before final norm; same forward as full-vocabulary logits; zero answer tokens.",
    }
    output.mkdir(parents=True)
    atomic_json(output / "protocol.json", protocol)
    atomic_json(
        output / "manifest.json",
        {
            "schema": "main-study-capture-manifest-v1",
            "status": "prepared_no_weights",
            "protocol_sha256": file_sha256(output / "protocol.json"),
            "completed_rows": 0,
            "test_fixture": test_fixture,
        },
    )
    return protocol


def verify_protocol(protocol):
    require(file_sha256(protocol["bank_path"]) == protocol["bank_file_sha256"], "Bank changed")
    snapshot_files(protocol["policy_sha256"])
    forward = protocol["forward_contract"]
    require(
        file_sha256(forward["source_path"]) == forward["source_sha256"],
        "Installed model-forward source changed",
    )


def _authorization(protocol, ledger_path, freeze_path, *, partition, fit_dir):
    """Real and fixture execution share the same fail-closed provenance gates."""
    from .main_study_bank import load_bank, verify_review_ledger

    require(
        ledger_path is not None and freeze_path is not None,
        "Execution requires a human-review ledger and frozen execution specification",
    )
    bank = load_bank(protocol["bank_path"])
    review = verify_review_ledger(bank, ledger_path)
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
        "Execution freeze must bind every capture source file",
    )
    baseline = Path(freeze["baseline_protocol_path"])
    require(
        file_sha256(baseline) == freeze["baseline_protocol_sha256"], "Baseline protocol changed"
    )
    required = {
        str(Path(ledger_path).resolve()): file_sha256(ledger_path),
        str(Path(freeze_path).resolve()): file_sha256(freeze_path),
        str(baseline.resolve()): file_sha256(baseline),
    }
    if partition == "test":
        from .main_study_analysis import load_fit_bundle

        require(fit_dir is not None, "Test capture requires a frozen fitted head bundle")
        fit_dir = Path(fit_dir).resolve()
        settings, _ = load_fit_bundle(fit_dir)
        require(
            settings.get("test_fixture") == protocol["test_fixture"],
            "Fitted-head fixture identity differs",
        )
        capture = settings.get("capture_provenance", {})
        require(
            capture.get("bank_sha256") == protocol["bank_sha256"]
            and capture.get("model_settings") == protocol["model_settings"]
            and capture.get("capture_layer") == protocol["capture_layer"],
            "Fitted-head bank/model/capture binding differs",
        )
        for name in ("settings.json", "heads.npz"):
            path = fit_dir / name
            required[str(path)] = file_sha256(path)
        frozen_fits = freeze.get("fitted_artifact_sha256", {})
        require(
            all(
                frozen_fits.get(path) == digest
                for path, digest in required.items()
                if Path(path).name in {"settings.json", "heads.npz"}
            ),
            "Held-out capture requires fitted artifacts bound in execution freeze",
        )
    return {"review": review, "sha256": required}


def forward_measurement(backend, ids, tokens, *, check_layer=False, raw_sink=None):
    """Capture feature and all output logits in one full, untruncated forward."""
    layer = backend.settings.capture_layers[0]
    require(len(backend.settings.capture_layers) == 1, "Need exactly one frozen capture layer")
    require(ids and len(ids) <= backend.settings.max_input_tokens, "Oversized prompt")
    observed = []

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        observed.append(hidden[0, -1].detach().float().cpu().numpy().copy())

    handle = backend.base.layers[layer].register_forward_hook(hook)
    started = time.perf_counter()
    try:
        sequence = backend.torch.tensor([ids], device=backend.input_device)
        with backend.torch.inference_mode():
            result = backend.model(
                input_ids=sequence,
                attention_mask=backend.torch.ones_like(sequence),
                logits_to_keep=1,
                use_cache=False,
                return_dict=True,
                **({"output_hidden_states": True} if check_layer else {}),
            )
    finally:
        handle.remove()
    require(len(observed) == 1, "Need exactly one decoder-hook observation")
    vector = observed[0]
    values = result.logits.detach().float().cpu().numpy()
    if raw_sink is not None:
        # Preserve every available tensor before measurement assertions. A failed
        # alignment/shape/finiteness check must not erase an observed outcome.
        raw_sink(vector, values)
    require(
        vector.shape == (backend.metadata["hidden_size"],)
        and vector.dtype == np.dtype("float32")
        and np.isfinite(vector).all(),
        "Invalid feature vector",
    )
    if check_layer:
        reference = result.hidden_states[layer + 1][0, -1].detach().float().cpu().numpy()
        require(np.array_equal(vector, reference), "Hook/HF hidden-state alignment differs")
    require(values.shape == (1, 1, backend.model.config.vocab_size), "Unexpected vocabulary shape")
    logits = values[0, 0].copy()
    return vector, logits, summarize_logits(logits, tokens), time.perf_counter() - started


def _complete_rows(output, protocol, selected):
    """Validate all durable completed rows; a started unknown row forbids resumption."""
    result = []
    journal = output / "row_journal"
    expected = {row["row_id"] for row in selected}
    if journal.exists():
        require(
            all(p.is_dir() and p.name in expected for p in journal.iterdir()),
            "Unexpected row journal entry",
        )
    for planned in selected:
        directory = journal / planned["row_id"]
        if not directory.exists():
            continue
        require(
            (directory / "started.json").is_file()
            and (directory / "complete.json").is_file()
            and not (directory / "failure.json").exists(),
            "Incomplete/failed attempted row blocks resumption; preserve it and stop",
        )
        row = read_json(directory / "complete.json")
        started = read_json(directory / "started.json")
        require(
            row["row_id"] == planned["row_id"]
            and row["status"] == "complete"
            and started["row_id"] == planned["row_id"]
            and started["protocol_sha256"] == row["protocol_sha256"]
            and all(row.get(key) == value for key, value in planned.items())
            and row["protocol_sha256"] == file_sha256(output / "protocol.json")
            and row["prompt_token_ids_sha256"] == planned["prompt_token_ids_sha256"]
            and file_sha256(directory / "arrays.npz") == row["arrays_sha256"],
            "Completed row binding changed",
        )
        require(
            file_sha256(directory / "raw_forward.npz") == row["raw_forward_sha256"],
            "Raw forward journal changed",
        )
        with np.load(directory / "arrays.npz", allow_pickle=False) as data:
            vector, logits = data["feature"], data["next_token_logits"]
            require(
                vector.dtype == np.float32
                and vector.shape == (protocol["hidden_size"],)
                and np.isfinite(vector).all(),
                "Invalid saved feature",
            )
            require(
                canonical_hash(vector.tolist()) == row["feature_values_sha256"]
                and summarize_logits(logits, protocol["tokenizer"]["answer_token_ids"])
                == row["output_logits"],
                "Saved row measurements changed",
            )
        result.append(row)
    return result


def _finalize(output, protocol, selected, authorization, manifest):
    rows = _complete_rows(output, protocol, selected)
    require(len(rows) == len(selected), "Cannot export an incomplete capture cache")
    features = []
    for row in rows:
        with np.load(
            output / "row_journal" / row["row_id"] / "arrays.npz", allow_pickle=False
        ) as data:
            features.append(data["feature"].copy())
    provenance = {
        "bank_sha256": protocol["bank_sha256"],
        "protocol_sha256": file_sha256(output / "protocol.json"),
        "model_settings": protocol["model_settings"],
        "capture_layer": protocol["capture_layer"],
        "partition": manifest["partition"],
        "source_bindings": authorization["sha256"],
    }
    atomic_npz(
        output / "features.npz",
        features=np.stack(features).astype(np.float32),
        row_ids=np.asarray([r["row_id"] for r in rows], dtype=str),
    )
    atomic_json(
        output / "rows.json",
        {
            "schema": "main-study-capture-v1",
            "test_fixture": protocol["test_fixture"],
            "provenance": provenance,
            "rows": [
                {**row, "output_yes_no_logitdiff": row["output_logits"]["yes_minus_no"]}
                for row in rows
            ],
        },
    )
    manifest.update(
        status="complete",
        completed_rows=len(rows),
        finished_at=utc_now(),
        provenance=provenance,
        output_sha256={name: file_sha256(output / name) for name in ("features.npz", "rows.json")},
    )
    atomic_json(output / "capture_manifest.json", manifest)
    atomic_json(output / "manifest.json", manifest)
    return manifest


def execute(
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
    """Execute only authorized rows; explicit resume never retries an unknown row."""
    output = Path(output).resolve()
    protocol, manifest = read_json(output / "protocol.json"), read_json(output / "manifest.json")
    require(partition in {"train_dev", "test"}, "Partition must be train_dev or test")
    require(
        manifest["protocol_sha256"] == file_sha256(output / "protocol.json"),
        "Prepared protocol changed",
    )
    require(manifest["status"] != "complete", "Completed capture cannot be rerun")
    verify_protocol(protocol)
    environment = environment_info()
    require(
        all(
            environment[key] == protocol["environment"][key]
            for key in ("python", "platform", "packages")
        ),
        "Prepared Python/platform/package environment changed",
    )
    authorization = _authorization(
        protocol, ledger_path, freeze_path, partition=partition, fit_dir=fit_dir
    )
    if protocol["test_fixture"]:
        require(backend_factory is not HFBackend, "Fixture protocol cannot load real model weights")
    selected = [
        r for r in protocol["planned_rows"] if (r["split"] == "test") == (partition == "test")
    ]
    require(selected, "No rows in requested partition")
    if manifest["status"] != "prepared_no_weights":
        require(
            resume
            and manifest.get("partition") == partition
            and manifest.get("authorization") == authorization,
            "Use explicit resume with the exact prior authorization and partition",
        )
    else:
        require(not resume, "Fresh preparation does not require resume")
    completed = _complete_rows(output, protocol, selected)
    if not resume:
        require(not completed and not (output / "attempts").exists(), "Attempt already exists")
    manifest.update(
        status="running",
        partition=partition,
        authorization=authorization,
        environment=environment_info(),
        completed_rows=len(completed),
    )
    atomic_json(output / "manifest.json", manifest)
    attempts = output / "attempts"
    attempts.mkdir(exist_ok=True)
    attempt_file = attempts / f"{len(list(attempts.iterdir())):04d}.json"
    with attempt_file.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "started_at": utc_now(),
                "explicit_resume": resume,
                "protocol_sha256": manifest["protocol_sha256"],
            },
            stream,
        )
    pending = [r for r in selected if r["row_id"] not in {c["row_id"] for c in completed}]
    backend = None
    try:
        if pending:
            backend = backend_factory(ModelSettings(**protocol["model_settings"]))
            require(
                bool(backend.metadata["is_test_fixture"]) == protocol["test_fixture"],
                "Fixture/real backend identity mismatch",
            )
            for key in ("model_id", "dtype", "quantization", "capture_layers"):
                require(
                    backend.metadata[key] == protocol["model_settings"][key],
                    "Backend setting differs: " + key,
                )
            require(
                backend.metadata["resolved_revision"] == protocol["model_settings"]["revision"]
                and backend.metadata["decoder_blocks"] == protocol["decoder_blocks"]
                and backend.metadata["hidden_size"] == protocol["hidden_size"]
                and _forward_contract(backend.model.forward) == protocol["forward_contract"]
                and canonical_hash(backend.tokenizer.chat_template)
                == protocol["tokenizer"]["chat_template_sha256"]
                and _token_map(backend.tokenizer) == protocol["tokenizer"]["answer_token_ids"]
                and {"eos": backend.tokenizer.eos_token_id, "pad": backend.tokenizer.pad_token_id}
                == protocol["tokenizer"]["special_token_ids"],
                "Pinned backend/tokenizer/forward identity differs",
            )
            backend.model.eval()
            require(not backend.model.training, "Model must be in evaluation mode")
            manifest["backend"] = backend.metadata
            atomic_json(output / "manifest.json", manifest)
        for index, planned in enumerate(pending):
            verify_protocol(protocol)
            snapshot_files(authorization["sha256"])
            require(
                file_sha256(output / "protocol.json") == manifest["protocol_sha256"],
                "Protocol changed during capture",
            )
            ids = token_ids(backend.tokenizer, planned["messages"])
            require(
                ids == planned["prompt_token_ids"]
                and canonical_hash(planned["messages"]) == planned["messages_sha256"],
                "Exact prefix changed",
            )
            directory = output / "row_journal" / planned["row_id"]
            directory.mkdir(parents=True, exist_ok=False)
            atomic_json(
                directory / "started.json",
                {
                    "row_id": planned["row_id"],
                    "protocol_sha256": manifest["protocol_sha256"],
                    "started_at": utc_now(),
                },
            )
            try:
                vector, logits, summary, elapsed = forward_measurement(
                    backend,
                    ids,
                    protocol["tokenizer"]["answer_token_ids"],
                    check_layer=index == 0,
                    raw_sink=lambda vector, values: atomic_npz(
                        directory / "raw_forward.npz",
                        feature=vector,
                        full_forward_logits=values,
                    ),
                )
                atomic_npz(directory / "arrays.npz", feature=vector, next_token_logits=logits)
                row = {
                    **planned,
                    "status": "complete",
                    "test_fixture": protocol["test_fixture"],
                    "protocol_sha256": manifest["protocol_sha256"],
                    "arrays_sha256": file_sha256(directory / "arrays.npz"),
                    "raw_forward_sha256": file_sha256(directory / "raw_forward.npz"),
                    "feature_values_sha256": canonical_hash(vector.tolist()),
                    "output_logits": summary,
                    "elapsed_seconds": elapsed,
                    "capture": {
                        "layer": protocol["capture_layer"],
                        "hf_hidden_state_index": protocol["capture_layer"] + 1,
                        "selected_position": len(ids) - 1,
                        "selected_token_id": ids[-1],
                        "same_forward_as_logits": True,
                        "hook_alignment_checked": index == 0,
                        "answer_tokens_supplied": 0,
                    },
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
            atomic_json(output / "manifest.json", manifest)
            if progress:
                progress(
                    f"Main-study {partition}: {manifest['completed_rows']}/{len(selected)}",
                    flush=True,
                )
        verify_protocol(protocol)
        snapshot_files(authorization["sha256"])
        return _finalize(output, protocol, selected, authorization, manifest)
    except BaseException as error:
        manifest.update(
            status="interrupted_preserved",
            error_type=type(error).__name__,
            error=str(error),
            finished_at=utc_now(),
        )
        atomic_json(output / "manifest.json", manifest)
        raise
    finally:
        if backend is not None:
            backend.close()
