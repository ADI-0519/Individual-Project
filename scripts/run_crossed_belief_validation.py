"""Prepare without weights, then execute one frozen crossed-belief validation attempt."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import subprocess
from pathlib import Path

import numpy as np
from run_belief_diagnostic import validate_layer_indices
from run_belief_query_comparison import SIGNATURE_FIELDS
from run_future_goal_answer_logits import forward_contract
from run_future_goal_controls import _reference, native_output, score_vector, token_ids

from latent_probes.backends import HFBackend
from latent_probes.belief_query_readout import _load_fit, _snapshot
from latent_probes.config import GenerationSettings, ModelSettings
from latent_probes.crossed_belief_bank import build_messages, load_bank, sender_rows
from latent_probes.crossed_belief_validation import (
    CONDITIONS,
    HEAD,
    answer_record,
    forward_measurement,
    require,
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

ROOT = Path(__file__).resolve().parents[1]
GENERATION = GenerationSettings(max_new_tokens=12, do_sample=False)
SEED = 20261121
POLICY_FILES = (
    "scripts/run_crossed_belief_validation.py",
    "src/latent_probes/crossed_belief_validation.py",
    "src/latent_probes/crossed_belief_bank.py",
    "scripts/analyse_crossed_belief_validation.py",
    "src/latent_probes/crossed_belief_analysis.py",
    "scripts/run_future_goal_controls.py",
    "scripts/run_future_goal_answer_logits.py",
    "scripts/run_belief_diagnostic.py",
    "scripts/run_belief_query_comparison.py",
    "src/latent_probes/belief_query_readout.py",
    "src/latent_probes/belief_query_panel.py",
    "src/latent_probes/belief_readout.py",
    "src/latent_probes/backends.py",
    "src/latent_probes/config.py",
    "src/latent_probes/storage.py",
    "docs/CROSSED_BELIEF_VALIDATION_PROTOCOL_V1.md",
    "docs/CROSSED_BELIEF_DESIGN_REVIEW_2026-11-21.md",
)


def git_binding():
    """Bind the local research source revision, including its uncommitted-file state."""

    def run(*args):
        return subprocess.run(
            ["git", "-C", str(ROOT), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "status_porcelain": run("status", "--porcelain")}


def verify(protocol, *, full_source=False):
    require(file_sha256(protocol["bank_path"]) == protocol["bank_sha256"], "Frozen bank changed")
    for path, digest in protocol["policy_sha256"].items():
        require(file_sha256(path) == digest, "Frozen source changed: " + path)
    require(
        _snapshot(Path(protocol["fit_dir"])) == protocol["fitted_artifact_sha256"],
        "Frozen fitted bundle changed",
    )
    require(
        file_sha256(protocol["config_path"]) == protocol["config_sha256"],
        "Frozen model config changed",
    )
    if full_source:
        require(git_binding() == protocol["git"], "Frozen Git source revision/state changed")
        for path_key, hash_key in (
            ("source_run", "source_artifact_sha256"),
            ("known_run", "known_artifact_sha256"),
        ):
            require(
                _snapshot(Path(protocol[path_key])) == protocol[hash_key],
                "Frozen source run changed: " + path_key,
            )


def prepare(bank_path, source_run, output, *, tokenizer=None, model_config=None, forward=None):
    """Pinned-tokenizer preflight, frozen-head source reproduction, zero model forwards."""
    bank_path, source_run, output = (
        Path(bank_path).resolve(),
        Path(source_run).resolve(),
        native_output(output),
    )
    require(not output.exists(), "Use a new attempt directory; no overwrite")
    old = read_json(source_run / "protocol.json")
    manifest = read_json(source_run / "manifest.json")
    source_hashes = _snapshot(source_run)
    require(
        manifest["status"] == "neutral_failed_stopped" and manifest["n_completed_rows"] == 32,
        "Bind the stopped 32-row goal-controls source; its qualification stays failed",
    )
    require(
        file_sha256(source_run / "protocol.json") == manifest["protocol_sha256"]
        and all(source_hashes.get(p) == h for p, h in manifest["output_sha256"].items()),
        "Goal-controls source bytes differ",
    )
    known_run, fit_dir, config_path = (
        Path(old[key]).resolve() for key in ("known_run", "fit_dir", "config_path")
    )
    for protected in (ROOT, source_run, known_run, fit_dir, bank_path, config_path):
        require(
            not output.is_relative_to(protected) and not protected.is_relative_to(output),
            "Output overlaps protected research inputs",
        )
    require(_snapshot(known_run) == old["known_artifact_sha256"], "Known-message source differs")
    metadata, parameters, texts, fit_hashes = _load_fit(fit_dir)
    require(fit_hashes == old["fitted_artifact_sha256"], "Original fitted head bundle differs")
    require(old["primary_head"] == HEAD, "Original head identity differs")
    require(file_sha256(config_path) == old["config_sha256"], "Original model config differs")
    settings = ModelSettings(**old["model_settings"])
    backend = manifest["backend"]
    require(
        settings.capture_layers == [16]
        and settings.local_files_only
        and settings.revision == backend["resolved_revision"]
        and settings.model_id == "meta-llama/Llama-3.1-8B-Instruct"
        and settings.revision == "0e9e39f249a16976918f6564b8830bc894c89659"
        and settings.dtype == "bfloat16"
        and settings.quantization == "4bit",
        "Frozen Llama NF4/BF16 model binding differs",
    )
    if tokenizer is None:
        from transformers import AutoConfig, AutoTokenizer
        from transformers.models.llama.modeling_llama import LlamaForCausalLM

        common = {
            "revision": settings.revision,
            "local_files_only": True,
            "trust_remote_code": False,
        }
        model_config = AutoConfig.from_pretrained(settings.model_id, **common)
        tokenizer = AutoTokenizer.from_pretrained(settings.model_id, **common)
        forward = LlamaForCausalLM.forward
    require(
        model_config is not None
        and model_config._commit_hash == settings.revision
        and model_config.model_type == "llama",
        "Pinned cached model config differs",
    )
    contract = forward_contract(forward)
    require(
        canonical_hash(tokenizer.chat_template) == backend["chat_template_sha256"]
        and {"eos": tokenizer.eos_token_id, "pad": tokenizer.pad_token_id}
        == backend["special_token_ids"],
        "Pinned tokenizer identity differs",
    )
    token_map = {word: tokenizer.encode(word, add_special_tokens=False) for word in ("YES", "NO")}
    require(
        all(
            len(ids) == 1
            and tokenizer.decode(ids, skip_special_tokens=False) == word
            and ids[0] not in tokenizer.all_special_ids
            for word, ids in token_map.items()
        )
        and token_map["YES"] != token_map["NO"],
        "Need unambiguous bare single-token YES/NO mappings",
    )
    reference = _reference(known_run, metadata, parameters, texts)
    require(
        token_ids(tokenizer, reference["messages"]) == reference["prompt_token_ids"],
        "Tokenizer source-reference prefix differs",
    )
    bank = load_bank(bank_path)
    inputs = sender_rows(bank)
    require(len(inputs) == 128, "Need exactly 128 unique visible sender inputs")
    planned, messages_seen = [], set()
    for row in inputs:
        require("actual_delivery" not in row, "Hidden actual-delivery metadata leaked into input")
        for condition in CONDITIONS:
            messages = build_messages(row, queried=condition == "nested_question")
            ids = token_ids(tokenizer, messages)
            require(
                len(ids) <= settings.max_input_tokens
                and len(ids) + (GENERATION.max_new_tokens if condition == "nested_question" else 0)
                <= model_config.max_position_embeddings,
                "Planned prompt or answer budget exceeds context",
            )
            digest = canonical_hash(messages)
            require(digest not in messages_seen, "Duplicate visible prefix in 256-capture roster")
            messages_seen.add(digest)
            planned.append(
                {
                    **copy.deepcopy(row),
                    "sender_input_id": row["id"],
                    "id": row["id"] + "__" + condition,
                    "condition": condition,
                    "messages": messages,
                    "messages_sha256": digest,
                    "prompt_token_ids": ids,
                    "prompt_token_ids_sha256": canonical_hash(ids),
                    "will_generate_answer": condition == "nested_question",
                }
            )
    forward_source = inspect.getsourcefile(inspect.unwrap(forward))
    require(forward_source is not None, "Need inspectable installed model-forward source")
    policy = {str(ROOT / p): file_sha256(ROOT / p) for p in POLICY_FILES}
    policy[str(Path(forward_source).resolve())] = file_sha256(forward_source)
    eos = backend["generation_defaults"].get("eos_token_id")
    eos = tokenizer.eos_token_id if eos is None else eos
    eos = eos if isinstance(eos, list) else [eos]
    require(eos and all(type(t) is int for t in eos), "Need frozen generation EOS IDs")
    protocol = {
        "schema": "crossed_belief_validation_protocol_v1",
        "created_at": utc_now(),
        "bank_path": str(bank_path),
        "bank_sha256": file_sha256(bank_path),
        "bank_schema": bank["schema"],
        "source_run": str(source_run),
        "source_artifact_sha256": source_hashes,
        "known_run": str(known_run),
        "known_artifact_sha256": old["known_artifact_sha256"],
        "fit_dir": str(fit_dir),
        "fitted_artifact_sha256": fit_hashes,
        "config_path": str(config_path),
        "config_sha256": old["config_sha256"],
        "model_settings": old["model_settings"],
        "source_backend": backend,
        "git": git_binding(),
        "policy_sha256": policy,
        "forward_signature": contract,
        "tokenizer": {
            "chat_template_sha256": canonical_hash(tokenizer.chat_template),
            "all_special_ids": list(tokenizer.all_special_ids),
        },
        "answer_token_ids": {word: ids[0] for word, ids in token_map.items()},
        "max_position_embeddings": model_config.max_position_embeddings,
        "generation": {
            "max_new_tokens": GENERATION.max_new_tokens,
            "do_sample": False,
            "seed": SEED,
            "eos_token_ids": eos,
        },
        "primary_head": HEAD,
        "threshold": 0.5,
        "threshold_tie": 0,
        "reference": reference,
        "planned_rows": planned,
        "n_visible_inputs": 128,
        "n_prefix_captures": 256,
        "n_queried_generations": 128,
        "hidden_actual_delivery_supplied": False,
        "capture": "Last complete chat-template token including assistant header; zero-based block16=HF17; same forward as full vocabulary logits; zero answer tokens supplied.",
        "unqueried_output_logits_are_taskless_diagnostic": True,
        "no_refit": True,
        "no_threshold_selection": True,
        "human_review_complete": bank.get("human_review_complete", False),
        "preflight": {
            "model_weights_loaded": False,
            "model_forwards": 0,
            "source_saved_full_vector_loaded_and_score_reproduced": True,
            "pinned_source_token_ids_exact": True,
            "minimum_tokens": min(len(r["prompt_token_ids"]) for r in planned),
            "maximum_tokens": max(len(r["prompt_token_ids"]) for r in planned),
        },
    }
    verify(protocol, full_source=True)
    output.mkdir(parents=True)
    atomic_json(output / "protocol.json", protocol)
    atomic_json(
        output / "manifest.json",
        {
            "schema": "crossed_belief_validation_run_v1",
            "status": "prepared_no_weights",
            "created_at": utc_now(),
            "protocol_sha256": file_sha256(output / "protocol.json"),
            "n_completed_rows": 0,
            "generated_answers": 0,
            "model_measurement_forwards": 0,
        },
    )
    return protocol


def execute(output, *, backend_factory=HFBackend, progress=print):
    """Every claimed attempt terminates permanently; partial outputs are retained."""
    output = native_output(output)
    protocol, manifest = read_json(output / "protocol.json"), read_json(output / "manifest.json")
    digest = file_sha256(output / "protocol.json")
    require(
        manifest["status"] == "prepared_no_weights" and manifest["protocol_sha256"] == digest,
        "No overwrite, rerun, or resumption of an attempted assay",
    )
    require(
        not (output / "row_outputs").exists() and not (output / "reference_check.json").exists(),
        "Prepared attempt already contains execution outputs",
    )
    verify(protocol, full_source=True)
    with (output / "attempt_started.json").open("x", encoding="utf-8") as stream:
        json.dump({"protocol_sha256": digest, "started_at": utc_now()}, stream)
    manifest.update(status="running", started_at=utc_now(), environment=environment_info())
    atomic_json(output / "manifest.json", manifest)
    backend = None
    try:
        backend = backend_factory(ModelSettings(**protocol["model_settings"]))
        for key in SIGNATURE_FIELDS:
            require(
                backend.metadata.get(key) == protocol["source_backend"].get(key),
                "Backend signature differs: " + key,
            )
        require(
            forward_contract(backend.model.forward) == protocol["forward_signature"],
            "Installed model forward signature differs",
        )
        backend.model.eval()
        require(not backend.model.training, "Model must be in eval mode")
        manifest["backend"] = backend.metadata
        atomic_json(output / "manifest.json", manifest)
        metadata, parameters, texts, _ = _load_fit(Path(protocol["fit_dir"]))
        reference = protocol["reference"]
        ids = token_ids(backend.tokenizer, reference["messages"])
        vector, logits, summary = forward_measurement(backend, ids, protocol["answer_token_ids"])
        check = {
            "ids_exact": ids == reference["prompt_token_ids"],
            "feature_exact": bool(
                np.array_equal(vector, np.asarray(reference["feature"], np.float32))
            ),
            "feature_values_sha256": canonical_hash(vector.tolist()),
            "probe": score_vector(vector, metadata, parameters, texts),
            "output_logits": summary,
        }
        check["score_exact"] = check["probe"] == reference["probe"]
        atomic_npz(output / "reference_forward.npz", layer_16=vector, next_token_logits=logits)
        atomic_json(output / "reference_check.json", check)
        require(
            check["ids_exact"] and check["feature_exact"] and check["score_exact"],
            "Original full-vector/token/score reproduction failed before new assay inputs",
        )
        manifest["layer_index_check"] = validate_layer_indices(backend, ids)
        atomic_json(output / "manifest.json", manifest)
        for index, planned in enumerate(protocol["planned_rows"]):
            verify(protocol)
            require(file_sha256(output / "protocol.json") == digest, "Frozen protocol changed")
            ids = token_ids(backend.tokenizer, planned["messages"])
            require(
                ids == planned["prompt_token_ids"]
                and canonical_hash(planned["messages"]) == planned["messages_sha256"]
                and canonical_hash(ids) == planned["prompt_token_ids_sha256"],
                "Frozen prompt binding differs",
            )
            path = output / "row_outputs" / (planned["id"] + ".json")
            feature_path = Path("forward_arrays") / (planned["id"] + ".npz")
            require(
                not path.exists() and not (output / feature_path).exists(),
                "Row already attempted; no regeneration",
            )
            atomic_json(path, {"id": planned["id"], "status": "started", "protocol_sha256": digest})
            vector, logits, summary = forward_measurement(
                backend, ids, protocol["answer_token_ids"]
            )
            manifest["model_measurement_forwards"] += 1
            atomic_npz(output / feature_path, layer_16=vector, next_token_logits=logits)
            row = {
                **planned,
                "status": "measured_before_answer",
                "protocol_sha256": digest,
                "capture": {
                    "layer": 16,
                    "hf_hidden_state_index": 17,
                    "selected_position": len(ids) - 1,
                    "selected_token_id": ids[-1],
                    "generated_answer_tokens_supplied": 0,
                    "perspective_question_supplied": planned["will_generate_answer"],
                    "prompt_token_ids_sha256": canonical_hash(ids),
                    "feature_path": feature_path.as_posix(),
                    "feature_sha256": file_sha256(output / feature_path),
                    "feature_values_sha256": canonical_hash(vector.tolist()),
                    "same_forward_as_logits": True,
                },
                "probe": score_vector(vector, metadata, parameters, texts),
                "output_logits": summary,
                "generation": None,
                "answer": None,
            }
            atomic_json(path, row)
            if planned["will_generate_answer"]:
                generation = backend.generate(
                    planned["messages"], GENERATION, SEED, capture=False
                ).record()
                row["generation"] = generation
                # Preserve the observed answer before asserting generation/capture binding.
                atomic_json(path, row)
                require(
                    backend.tokenizer.decode(
                        generation["generated_token_ids"], skip_special_tokens=True
                    ).strip()
                    == generation["text"],
                    "Generated text differs from exact token decoding",
                )
                row["answer"] = answer_record(
                    generation,
                    ids,
                    summary,
                    max_new_tokens=GENERATION.max_new_tokens,
                    eos_ids=protocol["generation"]["eos_token_ids"],
                    special_ids=protocol["tokenizer"]["all_special_ids"],
                )
                manifest["generated_answers"] += 1
            row["status"] = "complete"
            atomic_json(path, row)
            manifest["n_completed_rows"] += 1
            atomic_json(output / "manifest.json", manifest)
            if progress:
                progress(f"Crossed-belief {index + 1}/256: {planned['condition']}", flush=True)
        verify(protocol, full_source=True)
        require(
            manifest["n_completed_rows"] == 256
            and manifest["generated_answers"] == 128
            and manifest["model_measurement_forwards"] == 256,
            "Completed execution counts differ from frozen roster",
        )
        manifest["status"] = "complete_fresh_development_validation"
    except BaseException as error:
        manifest.update(
            status="failed_incomplete_no_retry", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        manifest["finished_at"] = utc_now()
        manifest["output_sha256"] = {
            p: h for p, h in _snapshot(output).items() if p != "manifest.json"
        }
        atomic_json(output / "manifest.json", manifest)
        if backend is not None:
            backend.close()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "execute"))
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        require(
            args.bank is not None and args.source_run is not None, "prepare requires bank/source"
        )
        protocol = prepare(args.bank, args.source_run, args.output)
        print(json.dumps(protocol["preflight"], indent=2))
    else:
        print(execute(args.output)["status"])


if __name__ == "__main__":
    main()
