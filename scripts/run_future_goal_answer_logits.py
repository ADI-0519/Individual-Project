"""Posthoc development diagnostic: exact neutral prefixes, no generation or gate rescue."""

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np

from latent_probes.backends import HFBackend
from latent_probes.config import ModelSettings
from latent_probes.storage import (
    atomic_json,
    canonical_hash,
    environment_info,
    file_sha256,
    read_json,
    utc_now,
)

ROOT = Path(__file__).resolve().parents[1]
BANK_SHA256 = "8c1b64638d12a93585e1bef8260388437ca2404feff6809a214f18a598e78a79"


def require(value, message):
    if not value:
        raise ValueError(message)


def snapshot(path):
    return {
        p.relative_to(path).as_posix(): file_sha256(p)
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


def native(path):
    path = Path(path).resolve()
    require(
        sys.platform == "linux"
        and not path.is_relative_to(Path("/mnt"))
        and "onedrive" not in str(path).lower(),
        "Use native WSL output outside OneDrive",
    )
    return path


def forward_contract(function):
    signature = inspect.signature(function)
    require(
        "logits_to_keep" in signature.parameters,
        "Local forward lacks explicit logits_to_keep support",
    )
    return str(
        signature.replace(parameters=[p for p in signature.parameters.values() if p.name != "self"])
    )


def prefix_ids(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)


def verify(protocol):
    require(
        snapshot(Path(protocol["source_run"])) == protocol["source_sha256"],
        "Frozen old run changed",
    )
    for path, digest in protocol["policy_sha256"].items():
        require(file_sha256(path) == digest, "Frozen source code changed: " + path)
    old = protocol["source_bindings"]
    require(
        file_sha256(old["bank_path"]) == old["bank_sha256"] == BANK_SHA256, "Frozen bank changed"
    )
    require(file_sha256(old["config_path"]) == old["config_sha256"], "Frozen config changed")
    require(
        snapshot(Path(old["fit_dir"])) == old["fitted_artifact_sha256"], "Frozen fit bundle changed"
    )


def prepare(source_run, output, *, tokenizer=None, forward=None):
    source_run, output = Path(source_run).resolve(), native(output)
    require(not output.exists(), "Use a new attempt directory; no overwrite")
    require(
        not output.is_relative_to(source_run)
        and not source_run.is_relative_to(output)
        and not output.is_relative_to(ROOT),
        "Output overlaps protected sources",
    )
    old, manifest = read_json(source_run / "protocol.json"), read_json(source_run / "manifest.json")
    hashes = snapshot(source_run)
    require(
        manifest["status"] == "neutral_failed_stopped" and manifest["n_completed_rows"] == 32,
        "Need the stopped 32-neutral source attempt",
    )
    require(
        all(hashes.get(p) == h for p, h in manifest["output_sha256"].items()),
        "Old output hashes differ",
    )
    require(
        file_sha256(source_run / "protocol.json") == manifest["protocol_sha256"],
        "Old protocol differs",
    )
    settings = old["model_settings"]
    require(
        settings["local_files_only"]
        and settings["capture_layers"] == [16]
        and settings["revision"] == manifest["backend"]["resolved_revision"],
        "Source model binding differs",
    )
    if tokenizer is None:
        from transformers import AutoTokenizer
        from transformers.models.llama.modeling_llama import LlamaForCausalLM

        require(manifest["backend"]["model_type"] == "llama", "Unsupported source architecture")
        tokenizer = AutoTokenizer.from_pretrained(
            settings["model_id"],
            revision=settings["revision"],
            local_files_only=True,
            trust_remote_code=False,
        )
        forward = LlamaForCausalLM.forward
    contract = forward_contract(forward)
    require(
        canonical_hash(tokenizer.chat_template) == manifest["backend"]["chat_template_sha256"]
        and tokenizer.eos_token_id == manifest["backend"]["special_token_ids"]["eos"],
        "Tokenizer differs",
    )
    tokens = {word: tokenizer.encode(word, add_special_tokens=False) for word in ("YES", "NO")}
    require(
        all(
            len(ids) == 1
            and tokenizer.decode(ids, skip_special_tokens=False) == word
            and ids[0] not in tokenizer.all_special_ids
            for word, ids in tokens.items()
        )
        and tokens["YES"] != tokens["NO"],
        "Ambiguous bare YES/NO single-token mapping; stop",
    )
    rows = []
    for planned in (r for r in old["planned_rows"] if r["condition"] == "neutral"):
        row_path = source_run / "row_outputs" / (planned["id"] + ".json")
        row = read_json(row_path)
        require(
            row["status"] == "complete"
            and all(row[k] == v for k, v in planned.items())
            and row["generation"]["text"] == "NO"
            and row["generation"]["generated_token_ids"][0] == tokens["NO"][0],
            "Saved NO generation or exact source row binding differs",
        )
        require(
            prefix_ids(tokenizer, row["messages"])
            == row["prompt_token_ids"]
            == row["generation"]["prompt_token_ids"]
            and canonical_hash(row["prompt_token_ids"])
            == row["prompt_token_ids_sha256"]
            == row["capture"]["prompt_token_ids_sha256"],
            "Exact source prefix differs",
        )
        capture, ids = row["capture"], row["prompt_token_ids"]
        require(
            capture["layer"] == 16
            and capture["hf_hidden_state_index"] == 17
            and capture["generated_answer_tokens_supplied"] == 0
            and capture["selected_position"] == len(ids) - 1
            and capture["selected_token_id"] == ids[-1],
            "Source capture convention differs",
        )
        feature = source_run / row["capture"]["feature_path"]
        require(
            file_sha256(feature) == row["capture"]["feature_sha256"], "Saved feature hash differs"
        )
        rows.append(
            {
                k: row[k]
                for k in (
                    "id",
                    "scene_id",
                    "channel",
                    "wording_group",
                    "mapping_id",
                    "gold",
                    "messages",
                    "prompt_token_ids",
                    "prompt_token_ids_sha256",
                    "capture",
                    "probe",
                )
            }
        )
    require(
        len(rows) == 32 and len({r["id"] for r in rows}) == 32,
        "Need exactly 32 distinct neutral prefixes",
    )
    policy = {str(ROOT / p): h for p, h in old["policy_sha256"].items()}
    policy[str(Path(__file__).resolve())] = file_sha256(__file__)
    diagnostic_protocol = ROOT / "docs/FUTURE_GOAL_ANSWER_LOGITS_PROTOCOL_V1.md"
    policy[str(diagnostic_protocol)] = file_sha256(diagnostic_protocol)
    forward_source = inspect.getsourcefile(inspect.unwrap(forward))
    require(forward_source is not None, "Local model-forward source is unavailable")
    policy[str(Path(forward_source).resolve())] = file_sha256(forward_source)
    protocol = {
        "schema": "future_goal_answer_logits_posthoc_v1",
        "created_at": utc_now(),
        "scope": "Posthoc development output-head ranking diagnostic; does not qualify or rescue the failed goal study.",
        "source_run": str(source_run),
        "source_sha256": hashes,
        "source_bindings": old,
        "model_settings": settings,
        "source_backend": manifest["backend"],
        "tokens": {k: v[0] for k, v in tokens.items()},
        "forward_signature": contract,
        "planned_rows": rows,
        "policy_sha256": policy,
        "generated_answers": 0,
    }
    verify(protocol)
    output.mkdir(parents=True)
    atomic_json(output / "protocol.json", protocol)
    atomic_json(
        output / "manifest.json",
        {
            "status": "prepared_no_weights",
            "n_rows": 0,
            "protocol_sha256": file_sha256(output / "protocol.json"),
            "generated_answers": 0,
        },
    )
    return protocol


def one_forward(backend, row, tokens, expected):
    ids = row["prompt_token_ids"]
    require(prefix_ids(backend.tokenizer, row["messages"]) == ids, "Runtime prefix differs")
    captured = []

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden[0, -1].detach().float().cpu().numpy().copy())

    handle = backend.base.layers[16].register_forward_hook(hook)
    try:
        sequence = backend.torch.tensor([ids], device=backend.input_device)
        with backend.torch.inference_mode():
            result = backend.model(
                input_ids=sequence,
                attention_mask=backend.torch.ones_like(sequence),
                logits_to_keep=1,
                use_cache=False,
                return_dict=True,
            )
    finally:
        handle.remove()
    require(len(captured) == 1, "Expected exactly one block16 hook observation")
    check = {
        "feature_exact": bool(np.array_equal(captured[0], expected)),
        "feature_values_sha256": canonical_hash(captured[0].tolist()),
    }
    if not check["feature_exact"]:
        return check  # Preserve the failed feature check without accepting any logits.
    values = result.logits.detach().float().cpu().numpy()
    require(
        values.ndim == 3 and values.shape[:2] == (1, 1) and np.isfinite(values).all(),
        "Invalid next-token logits",
    )
    values = values[0, 0].astype(np.float64)
    yes, no = (float(values[tokens[word]]) for word in ("YES", "NO"))
    log_z = float(values.max() + np.log(np.exp(values - values.max()).sum()))
    return {
        **check,
        "yes_logit": yes,
        "no_logit": no,
        "yes_logprob": yes - log_z,
        "no_logprob": no - log_z,
        "yes_minus_no": yes - no,
        "p_yes_given_yes_no": float(np.exp(-np.logaddexp(0.0, no - yes))),
        "full_vocab_logsumexp": log_z,
        "vocab_size": len(values),
        "argmax_token_id": int(values.argmax()),
    }


def execute(output, *, backend_factory=HFBackend):
    output = native(output)
    protocol, manifest = read_json(output / "protocol.json"), read_json(output / "manifest.json")
    digest = file_sha256(output / "protocol.json")
    require(
        manifest["status"] == "prepared_no_weights" and manifest["protocol_sha256"] == digest,
        "No overwrite or rerun",
    )
    verify(protocol)
    with (output / "attempt_started").open("x") as stream:
        stream.write(digest)
    manifest.update(status="running", started_at=utc_now(), environment=environment_info())
    atomic_json(output / "manifest.json", manifest)
    backend = None
    try:
        backend = backend_factory(ModelSettings(**protocol["model_settings"]))
        require(backend.metadata == protocol["source_backend"], "Exact source backend differs")
        require(
            forward_contract(backend.model.forward) == protocol["forward_signature"],
            "Runtime forward signature differs",
        )
        backend.model.eval()
        require(not backend.model.training, "Model must be in eval mode")
        manifest["backend"] = backend.metadata
        for row in protocol["planned_rows"]:
            require(
                file_sha256(output / "protocol.json") == digest, "Frozen posthoc protocol changed"
            )
            path = output / "row_outputs" / (row["id"] + ".json")
            require(not path.exists(), "Row already attempted; no rerun")
            atomic_json(path, {"id": row["id"], "status": "started", "protocol_sha256": digest})
            with np.load(
                Path(protocol["source_run"]) / row["capture"]["feature_path"], allow_pickle=False
            ) as archive:
                expected = archive["layer_16"].copy()
            result = one_forward(backend, row, protocol["tokens"], expected)
            atomic_json(
                path,
                {
                    **row,
                    "status": "complete" if result["feature_exact"] else "feature_mismatch",
                    "protocol_sha256": digest,
                    **result,
                },
            )
            require(
                result["feature_exact"],
                "Full-model block16 feature differs from saved neutral vector; stop",
            )
            manifest["n_rows"] += 1
            atomic_json(output / "manifest.json", manifest)
        verify(protocol)
        manifest["status"] = "complete_posthoc_development"
    except BaseException as error:
        manifest.update(
            status="failed_incomplete_no_retry", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        manifest["finished_at"] = utc_now()
        manifest["output_sha256"] = {
            p: h for p, h in snapshot(output).items() if p != "manifest.json"
        }
        atomic_json(output / "manifest.json", manifest)
        if backend is not None:
            backend.close()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "execute"))
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        require(args.source_run is not None, "prepare requires --source-run")
        prepare(args.source_run, args.output)
        print("Prepared 32 exact neutral prefixes; no weights loaded. Posthoc only.")
    else:
        print(execute(args.output)["status"])


if __name__ == "__main__":
    main()
