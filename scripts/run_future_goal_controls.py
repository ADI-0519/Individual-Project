"""Prepare offline, then run 32 neutral rows and conditionally 192 frozen rows once."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from run_belief_diagnostic import parse_yes_no, query_messages, validate_layer_indices
from run_belief_query_comparison import SIGNATURE_FIELDS

from latent_probes.backends import HFBackend
from latent_probes.belief_query_readout import _load_fit, _predict, _row_normalize, _snapshot
from latent_probes.config import GenerationSettings, ModelSettings, load_config
from latent_probes.future_goal_controls import (
    BANK_SHA256,
    GATE,
    HEAD,
    load_material,
    qualify_neutral,
)
from latent_probes.future_goal_controls import require as require
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
    "scripts/run_future_goal_controls.py",
    "scripts/analyse_future_goal_controls.py",
    "src/latent_probes/future_goal_controls.py",
    "src/latent_probes/future_goal_controls_analysis.py",
    "scripts/run_belief_goal_stress.py",
    "scripts/run_message_belief_validation.py",
    "scripts/run_belief_diagnostic.py",
    "scripts/run_belief_query_comparison.py",
    "src/latent_probes/belief_query_readout.py",
    "src/latent_probes/belief_readout.py",
    "src/latent_probes/backends.py",
    "src/latent_probes/config.py",
    "src/latent_probes/storage.py",
    "docs/FUTURE_GOAL_CONTENT_CONTROLS_DESIGN.md",
    "docs/FUTURE_GOAL_CONTENT_CONTROLS_PROTOCOL_V1.md",
)


def native_output(output):
    output = Path(output).resolve()
    require(
        sys.platform == "linux"
        and not output.is_relative_to(Path("/mnt"))
        and "onedrive" not in str(output).lower(),
        "Use native WSL output outside OneDrive",
    )
    return output


def token_ids(tokenizer, messages):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    require(
        isinstance(ids, list) and ids and all(type(i) is int and i >= 0 for i in ids),
        "Tokenizer did not return a nonempty integer prefix",
    )
    return ids


def score_vector(vector, metadata, parameters, texts):
    """Reuse the fitted scaler/L2 transform and score, preserving the raw margin."""
    head = metadata["heads"][HEAD]
    x = np.asarray(vector, dtype=np.float64).reshape(1, -1)
    require(
        x.shape == (1, metadata["feature_widths"]["layer_16"]) and np.isfinite(x).all(),
        "Invalid layer-16 feature",
    )
    prefix = head["activation_transform"]
    z = _row_normalize((x - parameters[prefix + "__mean"]) / parameters[prefix + "__scale"])
    margin = float(
        (z @ parameters[head["parameter_prefix"] + "__coef"].T)[0, 0]
        + parameters[head["parameter_prefix"] + "__intercept"][0]
    )
    probabilities = _predict(head, parameters, texts, x, [""])[0]
    require(np.isfinite(margin) and np.isfinite(probabilities).all(), "Nonfinite frozen score")
    return {
        "head_id": HEAD,
        "margin": margin,
        "probability": float(probabilities[1]),
        "prediction": int(probabilities.argmax()),
    }


def _completed_source(path):
    manifest = read_json(path / "manifest.json")
    require(manifest.get("status") == "complete", "Source run must be complete")
    hashes = _snapshot(path)
    require(
        all(hashes.get(name) == digest for name, digest in manifest["output_sha256"].items()),
        "Source output hash mismatch",
    )
    return manifest, hashes


def _reference(known_run, metadata, parameters, texts):
    rows = read_json(known_run / "rows.json")
    index, row = next(
        (i, r)
        for i, r in enumerate(rows)
        if r["view"] == "sender" and r["condition"] == "nested_question"
    )
    panel = read_json(known_run / "protocol.json")["panel"]
    case = next(c for c in panel["cases"] if c["id"] == row["case_id"])
    messages = query_messages(case, "sender", "nested")
    capture = next(
        c
        for c in read_json(known_run / "captures.json")
        if c["example_id"] == row["example_id"] and c["condition"] == row["condition"]
    )
    require(
        capture["messages_sha256"] == canonical_hash(messages)
        and row["messages_text"] == "\n".join(m["content"] for m in messages)
        and capture["generated_answer_tokens_supplied"] == 0,
        "Source reference binding differs",
    )
    with np.load(known_run / "features.npz", allow_pickle=False) as archive:
        vector = archive["layer_16"][index].copy()
    probe = score_vector(vector, metadata, parameters, texts)
    with (known_run / "frozen_evaluation/predictions.csv").open(newline="") as stream:
        saved = next(
            r
            for r in csv.DictReader(stream)
            if r["head_id"] == HEAD
            and r["example_id"] == row["example_id"]
            and r["evaluation_condition"] == "nested_question"
        )
    require(
        abs(float(saved["p1"]) - probe["probability"]) <= 1e-14
        and int(saved["prediction"]) == probe["prediction"],
        "Frozen head does not reproduce the saved source prediction",
    )
    return {
        "example_id": row["example_id"],
        "messages": messages,
        "messages_sha256": canonical_hash(messages),
        "prompt_token_ids": capture["prompt_token_ids"],
        "feature": vector.tolist(),
        "feature_values_sha256": canonical_hash(vector.tolist()),
        "probe": probe,
    }


def prepare_run(bank_path, config_path, source_run, known_run, fit_dir, output, *, tokenizer=None):
    """Freeze every condition and source before any model weights are loaded."""
    bank_path, config_path, source_run, known_run, fit_dir = (
        Path(p).resolve() for p in (bank_path, config_path, source_run, known_run, fit_dir)
    )
    output = native_output(output)
    if output.exists():
        raise FileExistsError("Use a new attempt directory; no overwrite or implicit rerun")
    protected = [bank_path, config_path, source_run, known_run, fit_dir, ROOT]
    require(
        all(
            not (p == output or p.is_relative_to(output) or output.is_relative_to(p))
            for p in protected
        ),
        "Output overlaps protected sources",
    )
    bank = load_material(bank_path)
    source, source_hashes = _completed_source(source_run)
    known, known_hashes = _completed_source(known_run)
    metadata, parameters, texts, fit_hashes = _load_fit(fit_dir)
    source_protocol = read_json(source_run / "protocol.json")
    known_protocol = read_json(known_run / "protocol.json")
    require(
        fit_hashes
        == source_protocol["fitted_artifact_sha256"]
        == known_protocol["fitted_artifact_sha256"],
        "Fitted bundle differs from the exact original assay bindings",
    )
    require(
        known_hashes == source_protocol["source_artifact_sha256"],
        "Known-message artifacts differ from the goal-stress source binding",
    )
    require(metadata["heads"][HEAD]["layer"] == "layer_16", "Frozen head layer differs")
    for key in SIGNATURE_FIELDS:
        require(
            source["backend"].get(key) == known["backend"].get(key),
            "Goal-stress and known-message backend differ: " + key,
        )
    backend = source["backend"]
    require(
        isinstance(backend["resolved_revision"], str) and len(backend["resolved_revision"]) == 40,
        "Need an exact cached model revision",
    )
    settings = load_config(config_path).model
    for key in ("model_id", "dtype", "quantization"):
        require(getattr(settings, key) == backend[key], "Config differs from source: " + key)
    require(settings.revision == backend["resolved_revision"], "Config model revision differs")
    settings = replace(settings, capture_layers=[16], local_files_only=True)
    if tokenizer is None:
        from transformers import AutoConfig, AutoTokenizer

        common = {
            "revision": settings.revision,
            "local_files_only": True,
            "trust_remote_code": False,
        }
        cached_config = AutoConfig.from_pretrained(settings.model_id, **common)
        require(cached_config._commit_hash == settings.revision, "Cached model revision differs")
        tokenizer = AutoTokenizer.from_pretrained(settings.model_id, **common)
    tokenizer_metadata = {
        "model_id": settings.model_id,
        "revision": settings.revision,
        "local_files_only": True,
        "chat_template_sha256": canonical_hash(tokenizer.chat_template),
        "all_special_ids": list(tokenizer.all_special_ids),
        "special_token_ids": {"eos": tokenizer.eos_token_id, "pad": tokenizer.pad_token_id},
    }
    require(
        tokenizer_metadata["chat_template_sha256"] == backend["chat_template_sha256"]
        and tokenizer_metadata["special_token_ids"] == backend["special_token_ids"],
        "Pinned tokenizer identity differs",
    )
    reference = _reference(known_run, metadata, parameters, texts)
    require(
        token_ids(tokenizer, reference["messages"]) == reference["prompt_token_ids"],
        "Pinned tokenizer does not reproduce source-reference tokens",
    )
    planned = []
    for row in bank["planned_conditions"]:
        ids = token_ids(tokenizer, row["messages"])
        require(len(ids) <= settings.max_input_tokens, "Prompt exceeds frozen input limit")
        planned.append(
            {
                **copy.deepcopy(row),
                "messages_sha256": canonical_hash(row["messages"]),
                "prompt_token_ids": ids,
                "prompt_token_ids_sha256": canonical_hash(ids),
            }
        )
    lookup = {(r["scene_id"], r["mapping_id"], r["condition"]): r for r in planned}
    lengths = {
        "per_row": {r["id"]: len(r["prompt_token_ids"]) for r in planned},
        "own_minus_archive": {
            f"{s['id']}/map{m}/{slot}": len(
                lookup[(s["id"], m, "own_token_" + slot)]["prompt_token_ids"]
            )
            - len(lookup[(s["id"], m, "archive_token_" + slot)]["prompt_token_ids"])
            for s in bank["scenes"]
            for m in (0, 1)
            for slot in ("a", "b")
        },
        "map1_minus_map0": {
            f"{s['id']}/{c}": len(lookup[(s["id"], 1, c)]["prompt_token_ids"])
            - len(lookup[(s["id"], 0, c)]["prompt_token_ids"])
            for s in bank["scenes"]
            for c in bank["conditions"]
        },
    }
    eos = backend["generation_defaults"].get("eos_token_id")
    if eos is None:
        eos = tokenizer.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    require(eos and all(type(i) is int for i in eos), "Need explicit frozen EOS IDs")
    pad = backend["generation_defaults"].get("pad_token_id")
    if pad is None:
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos[0]
    protocol = {
        "schema": "future_goal_controls_protocol_v1",
        "created_at": utc_now(),
        "bank_path": str(bank_path),
        "bank_sha256": BANK_SHA256,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "config": read_json(config_path),
        "model_settings": asdict(settings),
        "source_run": str(source_run),
        "source_artifact_sha256": source_hashes,
        "known_run": str(known_run),
        "known_artifact_sha256": known_hashes,
        "fit_dir": str(fit_dir),
        "fitted_artifact_sha256": fit_hashes,
        "source_backend": backend,
        "tokenizer": tokenizer_metadata,
        "primary_head": HEAD,
        "threshold": 0.5,
        "threshold_tie": 0,
        "layers": [16],
        "hf_hidden_state_index": 17,
        "capture": "Last complete chat-template prompt token, add_generation_prompt=True, before any answer.",
        "planned_rows": planned,
        "reference": reference,
        "token_lengths": lengths,
        "generation": {
            "settings": asdict(GENERATION),
            "seed": SEED,
            "eos_token_ids": eos,
            "pad_token_id": pad,
            "parser": "run_belief_diagnostic.parse_yes_no; capped outputs invalid",
        },
        "gate": GATE,
        "execution": "32 neutral first; remaining 192 only on pass, within this attempt",
        "policy_sha256": {p: file_sha256(ROOT / p) for p in POLICY_FILES},
        "no_refit": True,
        "no_hyperparameter_selection": True,
        "human_review_complete": bank["human_review_complete"],
    }
    output.mkdir(parents=True)
    atomic_json(output / "protocol.json", protocol)
    atomic_json(
        output / "manifest.json",
        {
            "schema": "future_goal_controls_run_v1",
            "status": "prepared_no_model_calls",
            "created_at": utc_now(),
            "protocol_sha256": file_sha256(output / "protocol.json"),
            "bank_sha256": BANK_SHA256,
            "n_completed_rows": 0,
            "generated_answers": 0,
        },
    )
    return protocol


def verify_frozen(protocol, *, full_source=False):
    require(
        file_sha256(protocol["bank_path"]) == protocol["bank_sha256"] == BANK_SHA256,
        "Frozen bank changed",
    )
    require(
        file_sha256(protocol["config_path"]) == protocol["config_sha256"], "Frozen config changed"
    )
    for name, digest in protocol["policy_sha256"].items():
        require(file_sha256(ROOT / name) == digest, "Frozen policy changed: " + name)
    require(
        _snapshot(Path(protocol["fit_dir"])) == protocol["fitted_artifact_sha256"],
        "Frozen fitted artifacts changed",
    )
    if full_source:
        for key, hashes in (
            ("source_run", "source_artifact_sha256"),
            ("known_run", "known_artifact_sha256"),
        ):
            require(
                _snapshot(Path(protocol[key])) == protocol[hashes], "Frozen source changed: " + key
            )


def capture_vector(backend, messages):
    """The existing backend hook selects zero-based block16, before final norm."""
    ids = token_ids(backend.tokenizer, messages)
    require(len(ids) <= backend.settings.max_input_tokens, "Prompt exceeds input limit")
    sequence = backend.torch.tensor([ids], device=backend.input_device)
    captured = backend._capture_token_rows(sequence, [len(ids) - 1])
    require(set(captured) == {"layer_16_tokens"}, "Unexpected capture layers")
    vector = np.asarray(captured["layer_16_tokens"], dtype=np.float32)
    require(
        vector.shape == (1, backend.metadata["hidden_size"]) and np.isfinite(vector).all(),
        "Invalid prefix-only activation",
    )
    return ids, vector[0].copy()


def execute_run(output, *, backend_factory=HFBackend, progress=print):
    """Claim a prepared attempt before weight loading; every failure is terminal."""
    output = native_output(output)
    protocol, manifest = read_json(output / "protocol.json"), read_json(output / "manifest.json")
    digest = file_sha256(output / "protocol.json")
    require(
        manifest["status"] == "prepared_no_model_calls" and digest == manifest["protocol_sha256"],
        "No overwrite or implicit rerun of an attempt",
    )
    require(
        not (output / "row_outputs").exists() and not (output / "reference_check.json").exists(),
        "Prepared attempt already contains execution outputs",
    )
    verify_frozen(protocol, full_source=True)
    # An exclusive, permanent claim also prevents two processes from starting the same plan.
    with (output / "attempt_started.json").open("x", encoding="utf-8") as stream:
        json.dump({"protocol_sha256": digest, "started_at": utc_now()}, stream)
    manifest.update(status="running", started_at=utc_now(), environment=environment_info())
    atomic_json(output / "manifest.json", manifest)
    backend, completed = None, []
    try:
        backend = backend_factory(ModelSettings(**protocol["model_settings"]))
        for key in SIGNATURE_FIELDS:
            expected = [16] if key == "capture_layers" else protocol["source_backend"].get(key)
            require(backend.metadata.get(key) == expected, "Backend signature differs: " + key)
        require(list(backend.settings.capture_layers) == [16], "Capture only frozen block16")
        manifest["backend"] = backend.metadata
        atomic_json(output / "manifest.json", manifest)
        metadata, parameters, texts, _ = _load_fit(Path(protocol["fit_dir"]))
        reference = protocol["reference"]
        ids, vector = capture_vector(backend, reference["messages"])
        check = {
            "ids_exact": ids == reference["prompt_token_ids"],
            "feature_exact": np.array_equal(vector, np.asarray(reference["feature"], np.float32)),
            "feature_values_sha256": canonical_hash(vector.tolist()),
            "probe": score_vector(vector, metadata, parameters, texts),
        }
        check["feature_exact"] = bool(check["feature_exact"])
        check["score_exact"] = check["probe"] == reference["probe"]
        atomic_npz(output / "reference_feature.npz", layer_16=vector)
        atomic_json(output / "reference_check.json", check)
        require(
            check["ids_exact"] and check["feature_exact"] and check["score_exact"],
            "Source-reference backend/head recomputation failed; stop before neutral assay",
        )
        manifest["layer_index_check"] = validate_layer_indices(backend, ids)
        neutral = [r for r in protocol["planned_rows"] if r["condition"] == "neutral"]
        remaining = [r for r in protocol["planned_rows"] if r["condition"] != "neutral"]
        require(len(neutral) == 32 and len(remaining) == 192, "Frozen execution roster differs")

        def collect(planned):
            verify_frozen(protocol)
            require(file_sha256(output / "protocol.json") == digest, "Frozen protocol changed")
            require(
                canonical_hash(planned["messages"]) == planned["messages_sha256"],
                "Frozen messages changed",
            )
            row_path = output / "row_outputs" / (planned["id"] + ".json")
            feature_path = Path("prefix_features") / (planned["id"] + ".npz")
            require(
                not row_path.exists() and not (output / feature_path).exists(),
                "Row already attempted; no regeneration",
            )
            atomic_json(
                row_path,
                {"id": planned["id"], "status": "capture_started", "protocol_sha256": digest},
            )
            ids, vector = capture_vector(backend, planned["messages"])
            require(
                ids == planned["prompt_token_ids"]
                and canonical_hash(ids) == planned["prompt_token_ids_sha256"],
                "Capture prefix differs from tokenizer preflight",
            )
            atomic_npz(output / feature_path, layer_16=vector)
            record = {
                **copy.deepcopy(planned),
                "status": "generation_started",
                "protocol_sha256": digest,
                "capture": {
                    "prompt_token_ids_sha256": canonical_hash(ids),
                    "selected_position": len(ids) - 1,
                    "selected_token_id": ids[-1],
                    "layer": 16,
                    "hf_hidden_state_index": 17,
                    "feature_path": feature_path.as_posix(),
                    "feature_sha256": file_sha256(output / feature_path),
                    "generated_answer_tokens_supplied": 0,
                },
                "probe": score_vector(vector, metadata, parameters, texts),
            }
            atomic_json(row_path, record)
            result = backend.generate(planned["messages"], GENERATION, SEED, capture=False)
            generation = result.record()
            raw = backend.tokenizer.decode(
                result.generated_token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            generation.update(raw_decoded_text=raw, raw_text_sha256=canonical_hash(raw))
            generation["unexpected_special_tokens"] = [
                t
                for i, t in enumerate(result.generated_token_ids)
                if t in protocol["tokenizer"]["all_special_ids"]
                and not (
                    i == len(result.generated_token_ids) - 1
                    and t in protocol["generation"]["eos_token_ids"]
                )
            ]
            # Preserve every returned output even if a consistency check below fails.
            record["generation"] = generation
            atomic_json(row_path, record)
            require(result.prompt_token_ids == ids, "Answer prefix differs from captured prefix")
            generated = result.generated_token_ids
            require(
                0 < len(generated) <= 12 and all(type(i) is int for i in generated),
                "Invalid generated token count or IDs",
            )
            eos = protocol["generation"]["eos_token_ids"]
            expected_finish = "eos" if generated[-1] in eos else "length"
            require(
                result.finish_reason == expected_finish
                and (expected_finish == "eos" or len(generated) == 12),
                "Incoherent EOS/length stop",
            )
            decoded = backend.tokenizer.decode(generated, skip_special_tokens=True).strip()
            require(
                decoded == result.text, "Returned answer text differs from generated token decoding"
            )
            capped = result.finish_reason == "length"
            prediction = (
                None
                if capped or generation["unexpected_special_tokens"]
                else parse_yes_no(result.text)
            )
            record.update(
                status="complete",
                answer={
                    "prediction": prediction,
                    "invalid": prediction is None,
                    "length_capped": capped,
                    "correct": prediction == planned["gold"],
                },
            )
            atomic_json(row_path, record)
            completed.append(record)
            manifest.update(n_completed_rows=len(completed), generated_answers=len(completed))
            atomic_json(output / "manifest.json", manifest)
            if progress:
                progress(
                    f"Future-goal conditions {len(completed)}/224: {planned['id']}", flush=True
                )

        for row in neutral:
            collect(row)
        qualification = qualify_neutral(completed)
        atomic_json(output / "qualification.json", qualification)
        manifest["neutral_qualification_passed"] = qualification["passed"]
        if qualification["passed"]:
            manifest["neutral_passed_at"] = utc_now()
            atomic_json(output / "manifest.json", manifest)
            for row in remaining:
                collect(row)
            manifest["status"] = "complete"
        else:
            manifest["status"] = "neutral_failed_stopped"
        verify_frozen(protocol, full_source=True)
        require(file_sha256(output / "protocol.json") == digest, "Frozen protocol changed")
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
    sub = parser.add_subparsers(dest="mode", required=True)
    prepare = sub.add_parser("prepare", help="offline pinned-tokenizer preflight, no weights")
    prepare.add_argument(
        "--bank",
        type=Path,
        default=ROOT / "data/belief_diagnostic/future_goal_content_controls_draft_v1.json",
    )
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--source-run", type=Path, required=True)
    prepare.add_argument("--known-run", type=Path, required=True)
    prepare.add_argument("--fit-dir", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    execute = sub.add_parser(
        "execute", help="single attempt; conditional continuation is automatic"
    )
    execute.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        protocol = prepare_run(
            args.bank, args.config, args.source_run, args.known_run, args.fit_dir, args.output
        )
        print(
            f"Prepared 224 exact prefixes; token range "
            f"{min(protocol['token_lengths']['per_row'].values())}.."
            f"{max(protocol['token_lengths']['per_row'].values())}; no weights loaded."
        )
    else:
        manifest = execute_run(args.output)
        print(manifest["status"])


if __name__ == "__main__":
    main()
