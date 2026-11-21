"""One non-research prefix per pinned model; no fitting or main-bank inference.

Run Llama and Qwen in separate sequential processes. The only real prefix comes
from the dedicated engineering definition; main-bank text is used for exclusion
checks and provenance, never passed to a model. There is no retry/resume mode.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from latent_probes.backends import HFBackend
from latent_probes.config import ModelSettings
from latent_probes.main_study_bank import bank_digest, iter_rows, load_bank
from latent_probes.main_study_run import (
    MODELS,
    POLICY_FILES,
    ROOT,
    _forward_contract,
    _token_map,
    forward_measurement,
    require,
    snapshot_files,
    token_ids,
)
from latent_probes.storage import (
    atomic_json,
    atomic_npz,
    canonical_hash,
    environment_info,
    file_sha256,
    read_json,
    utc_now,
)

PROFILE_PATH = ROOT / "data/belief_main/engineering_profile_v1.json"
EXCLUDED_PATH = ROOT / "data/belief_main/design_examples_v1.json"
SCRIPT_PATH = Path(__file__).resolve()


def validate_profile(profile, bank, excluded):
    """Check that the sole engineering prompt and its actors are not study material."""
    require(
        profile.get("schema") == "main-study-engineering-profile-input-v1"
        and profile.get("status") == "engineering_only"
        and profile.get("research_observation") is False
        and type(profile.get("permitted_forwards_per_model")) is int
        and profile["permitted_forwards_per_model"] == 1,
        "Expected the explicit single-forward non-research engineering definition",
    )
    require(not {"gold", "label", "target", "prediction"} & profile.keys(), "No fit labels")
    messages = profile.get("messages")
    require(
        isinstance(messages, list)
        and [m.get("role") for m in messages] == ["system", "user"]
        and all(isinstance(m.get("content"), str) and m["content"].strip() for m in messages),
        "Need exactly one system message and one user prefix",
    )
    actors, places = profile.get("actors"), profile.get("locations")
    require(
        isinstance(actors, list)
        and isinstance(places, list)
        and len(actors) == len(places) == 2
        and len(set(actors + places)) == 4
        and all(isinstance(x, str) and len(x.split()) >= 2 for x in actors + places),
        "Need two dedicated actors and two distinct dedicated storage places",
    )
    combined = "\n".join(m["content"] for m in messages)
    require(all(x in combined for x in actors + places), "Declared profile entity missing")
    cases = excluded.get("cases", [])
    require(isinstance(cases, list) and cases, "Missing excluded design cases")
    source_records = [*bank["groups"], *cases]
    names = [r[k] for r in source_records for k in ("sender", "reader")]
    used_components = {part.casefold() for name in names for part in name.split()}
    profile_components = {part.casefold() for name in actors for part in name.split()}
    require(not used_components & profile_components, "Profile actor component reuses study entity")
    corpus_text = json.dumps([bank, excluded], ensure_ascii=False).casefold()
    require(
        all(entity.casefold() not in corpus_text for entity in actors + places),
        "Profile actor/location phrase already occurs in study material",
    )
    profile_hash = canonical_hash(messages)
    require(
        all(canonical_hash(row["messages"]) != profile_hash for row in iter_rows(bank)),
        "Profile duplicates a main-study prefix",
    )
    require(
        all(canonical_hash(row["messages"]) != profile_hash for row in cases),
        "Profile duplicates an excluded design prefix",
    )
    return {
        "main_group_count": len(bank["groups"]),
        "excluded_design_prefix_count": len(cases),
        "actor_name_components_disjoint": True,
        "actor_and_location_phrases_disjoint": True,
        "exact_prefix_disjoint": True,
        "human_main_study_review_validated": False,
    }


def _cached_model_inputs(model_key):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model = MODELS[model_key]
    options = {
        "revision": model["revision"],
        "local_files_only": True,
        "trust_remote_code": False,
    }
    config = AutoConfig.from_pretrained(model["model_id"], **options)
    tokenizer = AutoTokenizer.from_pretrained(model["model_id"], **options)
    forward = AutoModelForCausalLM._model_mapping[type(config)].forward
    return tokenizer, config, forward


def _gpu_state(torch):
    return {
        str(i): {
            "name": torch.cuda.get_device_properties(i).name,
            "total_bytes": torch.cuda.get_device_properties(i).total_memory,
            "compute_capability": list(torch.cuda.get_device_capability(i)),
            "allocated_bytes": torch.cuda.memory_allocated(i),
            "reserved_bytes": torch.cuda.memory_reserved(i),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(i),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(i),
        }
        for i in range(torch.cuda.device_count())
    }


def _synchronize(torch, *, reset=False):
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)
        if reset:
            torch.cuda.reset_peak_memory_stats(i)


def _snapshot_sources(output, sources):
    result = {}
    for index, name in enumerate(sorted(sources)):
        source = Path(name)
        data = source.read_bytes()
        destination = output / "source_snapshot" / f"{index:02d}_{source.name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(data)
        require(file_sha256(destination) == sources[name], "Source changed while snapshotting")
        result[name] = {
            "sha256": sources[name],
            "snapshot": str(destination.relative_to(output)),
        }
    return result


def profile_backend(
    bank_path,
    output,
    *,
    model_key,
    profile_path=PROFILE_PATH,
    excluded_path=EXCLUDED_PATH,
    test_fixture=False,
    backend_factory=HFBackend,
    tokenizer=None,
    model_config=None,
    forward=None,
):
    """Bind source before loading, then preserve exactly one attempted forward.

    Fixture injection is a Python test interface only. The CLI has no fixture
    flag, no prompt argument, no generation mode and no research-review bypass.
    """
    require(model_key in MODELS and type(test_fixture) is bool, "Invalid model/fixture choice")
    require(
        not test_fixture or backend_factory is not HFBackend,
        "Synthetic fixtures cannot load real model weights",
    )
    require(
        test_fixture or (tokenizer is model_config is forward is None),
        "Real profiles use only the pinned cached tokenizer/config/forward",
    )
    bank_path, output = Path(bank_path).resolve(), Path(output).resolve()
    profile_path, excluded_path = Path(profile_path).resolve(), Path(excluded_path).resolve()
    require(not output.exists(), "No overwrite or retry: preserve the previous profile attempt")
    require(not output.is_relative_to(ROOT), "Profile outputs must be outside the source tree")
    if not test_fixture:
        require(
            "onedrive" not in str(output).casefold() and not str(output).startswith("/mnt/"),
            "Real profile outputs must use native storage outside Windows/OneDrive mounts",
        )
        require(
            profile_path == PROFILE_PATH.resolve(), "Real profile uses the dedicated definition"
        )
    bank = load_bank(bank_path)
    profile, excluded = read_json(profile_path), read_json(excluded_path)
    exclusion = validate_profile(profile, bank, excluded)
    if tokenizer is None:
        tokenizer, model_config, forward = _cached_model_inputs(model_key)
    model = MODELS[model_key]
    require(
        model_config is not None
        and model_config._commit_hash == model["revision"]
        and model_config.model_type == model["model_type"],
        "Pinned model configuration differs",
    )
    blocks = model_config.num_hidden_layers
    require(type(blocks) is int and blocks > 1, "Invalid decoder block count")
    layer = 16 if model_key == "llama" else blocks // 2
    require(0 <= layer < blocks - 1, "Invalid decoder capture layer")
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
    ids = token_ids(tokenizer, profile["messages"])
    require(
        len(ids) <= min(settings.max_input_tokens, model_config.max_position_embeddings),
        "Dedicated engineering prefix exceeds context; no truncation permitted",
    )
    tokens, contract = _token_map(tokenizer), _forward_contract(forward)
    sources = {
        str(path): file_sha256(path)
        for path in {
            SCRIPT_PATH,
            bank_path,
            profile_path,
            excluded_path,
            Path(contract["source_path"]),
            *(ROOT / p for p in POLICY_FILES),
        }
    }
    require(bank_digest(bank) == bank_digest(read_json(bank_path)), "Bank changed during preflight")
    snapshot_files(sources)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": "main-study-engineering-profile-protocol-v1",
        "created_at": utc_now(),
        "test_fixture": test_fixture,
        "research_observation": False,
        "main_bank_rows_passed_to_model": 0,
        "permitted_forwards": 1,
        "bank_path": str(bank_path),
        "bank_sha256": bank_digest(bank),
        "bank_file_sha256": file_sha256(bank_path),
        "profile_path": str(profile_path),
        "profile_file_sha256": file_sha256(profile_path),
        "messages": profile["messages"],
        "messages_sha256": canonical_hash(profile["messages"]),
        "prompt_token_ids": ids,
        "prompt_token_ids_sha256": canonical_hash(ids),
        "model_settings": asdict(settings),
        "model_key": model_key,
        "decoder_blocks": blocks,
        "capture_layer": layer,
        "hidden_size": model_config.hidden_size,
        "vocab_size": model_config.vocab_size,
        "forward_contract": contract,
        "source_sha256": sources,
        "source_snapshots": _snapshot_sources(output, sources),
        "environment": environment_info(),
        "tokenizer": {
            "chat_template_sha256": canonical_hash(tokenizer.chat_template),
            "special_token_ids": {"eos": tokenizer.eos_token_id, "pad": tokenizer.pad_token_id},
            "answer_token_ids": tokens,
        },
        "exclusion_checks": exclusion,
        "claim_boundary": "Engineering timing/memory/tensor alignment only. No accuracy, fit labels, belief mechanism, prompt selection, held-out or deception result.",
    }
    atomic_json(output / "protocol.json", protocol)
    manifest = {
        "schema": "main-study-engineering-profile-manifest-v1",
        "status": "prepared_no_weights",
        "test_fixture": test_fixture,
        "research_observation": False,
        "completed_forwards": 0,
        "protocol_sha256": file_sha256(output / "protocol.json"),
    }
    atomic_json(output / "manifest.json", manifest)
    snapshot_files(sources)
    backend = None
    try:
        atomic_json(
            output / "started.json",
            {"started_at": utc_now(), "protocol_sha256": manifest["protocol_sha256"]},
        )
        if not test_fixture:
            import torch

            require(torch.cuda.is_available(), "Engineering profile requires the declared CUDA GPU")
            _synchronize(torch, reset=True)
            manifest["gpu_before_load"] = _gpu_state(torch)
        manifest["status"] = "loading_weights"
        atomic_json(output / "manifest.json", manifest)
        load_started = time.perf_counter()
        backend = backend_factory(settings)
        if not test_fixture:
            _synchronize(backend.torch)
            manifest["gpu_after_load"] = _gpu_state(backend.torch)
        manifest["load_seconds"] = time.perf_counter() - load_started
        require(
            backend.metadata["is_test_fixture"] is test_fixture,
            "Fixture/real backend identity differs",
        )
        require(
            all(
                backend.metadata[key] == asdict(settings)[key]
                for key in ("model_id", "dtype", "quantization", "capture_layers")
            )
            and backend.metadata["resolved_revision"] == model["revision"]
            and backend.metadata["decoder_blocks"] == blocks
            and backend.metadata["hidden_size"] == model_config.hidden_size
            and _forward_contract(backend.model.forward) == contract,
            "Loaded backend differs from the pinned preparation",
        )
        require(
            token_ids(backend.tokenizer, profile["messages"]) == ids
            and _token_map(backend.tokenizer) == tokens
            and canonical_hash(backend.tokenizer.chat_template)
            == protocol["tokenizer"]["chat_template_sha256"],
            "Loaded tokenizer differs from the source-bound prefix",
        )
        backend.model.eval()
        require(not backend.model.training, "Model must be in evaluation mode")
        snapshot_files(sources)
        require(
            file_sha256(output / "protocol.json") == manifest["protocol_sha256"],
            "Engineering protocol changed after preparation",
        )
        manifest.update(status="forward_started", backend=backend.metadata)
        atomic_json(output / "manifest.json", manifest)
        atomic_json(output / "forward_started.json", {"started_at": utc_now()})
        if not test_fixture:
            _synchronize(backend.torch, reset=True)
        forward_started = time.perf_counter()
        vector, logits, summary, capture_seconds = forward_measurement(
            backend,
            ids,
            tokens,
            check_layer=True,
            raw_sink=lambda vector, values: atomic_npz(
                output / "raw_forward.npz", feature=vector, full_forward_logits=values
            ),
        )
        if not test_fixture:
            _synchronize(backend.torch)
            manifest["gpu_after_forward"] = _gpu_state(backend.torch)
        elapsed = time.perf_counter() - forward_started
        atomic_npz(output / "arrays.npz", feature=vector, next_token_logits=logits)
        snapshot_files(sources)
        require(
            file_sha256(output / "protocol.json") == manifest["protocol_sha256"],
            "Engineering protocol changed during the attempted forward",
        )
        manifest.update(
            status="complete_engineering_only",
            finished_at=utc_now(),
            completed_forwards=1,
            model_forward_seconds=elapsed,
            capture_function_seconds=capture_seconds,
            input_tokens=len(ids),
            feature_shape=list(vector.shape),
            logits_shape=list(logits.shape),
            feature_dtype=str(vector.dtype),
            logits_dtype=str(logits.dtype),
            feature_values_sha256=canonical_hash(vector.tolist()),
            logits_values_sha256=canonical_hash(logits.tolist()),
            output_logits=summary,
            capture={
                "layer": layer,
                "hf_hidden_state_index": layer + 1,
                "position": len(ids) - 1,
                "token_id": ids[-1],
                "same_forward_as_full_vocabulary_logits": True,
                "hook_alignment_checked": True,
                "generated_answer_tokens": 0,
            },
            arrays_sha256=file_sha256(output / "arrays.npz"),
            raw_forward_sha256=file_sha256(output / "raw_forward.npz"),
        )
        atomic_json(output / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        manifest.update(
            status="failed_attempt_preserved",
            failed_at=utc_now(),
            error_type=type(error).__name__,
            error=str(error),
            preserved_files_sha256={
                path.name: file_sha256(path)
                for path in output.iterdir()
                if path.is_file() and path.name != "manifest.json"
            },
        )
        atomic_json(output / "failure.json", manifest)
        atomic_json(output / "manifest.json", manifest)
        raise
    finally:
        if backend is not None:
            backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    args = parser.parse_args()
    result = profile_backend(args.bank, args.output, model_key=args.model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
