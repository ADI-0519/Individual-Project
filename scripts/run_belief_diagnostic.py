"""Fresh development gate for distinct world, reader and nested-perspective labels.

Query-free story features and separately elicited answers are different measures.
This is a narrated task, not a natural sender or an unknown-message forecast.
"""

import argparse
import copy
import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from latent_probes.backends import HFBackend
from latent_probes.belief_panel import validate_panel
from latent_probes.belief_readout import evaluate_belief_readouts
from latent_probes.config import GenerationSettings, load_config
from latent_probes.storage import (
    atomic_json,
    atomic_npz,
    canonical_hash,
    environment_info,
    file_sha256,
    read_json,
    utc_now,
)

LAYERS = (8, 16, 24)
PRIMARY_LAYER = "layer_16"


def parse_yes_no(text):
    match = re.fullmatch(r"\s*(YES|NO)[.!]?\s*", text, flags=re.IGNORECASE)
    return int(match.group(1).upper() == "YES") if match else None


def query_messages(case, view, target):
    if view == "sender" and target == "reader":
        raise ValueError("Actual private reader information is not available in this view.")
    messages = copy.deepcopy(case["messages"][view])
    messages[-1]["content"] += "\n\nQuestion: " + case["questions"][target]
    return messages


def capture_prefix(backend, messages):
    encoded = backend.tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    ids = encoded["input_ids"][0].tolist()
    if not ids or len(ids) > backend.settings.max_input_tokens:
        raise ValueError("Diagnostic prompt is empty or exceeds the declared input limit.")
    sequence = backend.torch.tensor([ids], device=backend.input_device)
    captured = backend._capture_token_rows(sequence, [len(ids) - 1])
    vectors = {}
    for layer in LAYERS:
        vector = np.asarray(captured[f"layer_{layer}_tokens"], dtype=np.float32)
        if vector.ndim != 2 or vector.shape[0] != 1 or not np.isfinite(vector).all():
            raise ValueError("Invalid prefix-only activation vector.")
        vectors[f"layer_{layer}"] = vector[0].copy()
    return ids, vectors


def validate_layer_indices(backend, ids):
    """Compare hook outputs with HF hidden-state indices in ONE genuine forward."""
    if backend.metadata["is_test_fixture"]:
        return {"status": "not_run_test_fixture", "is_test_fixture": True}
    observed, handles = {}, []
    sequence = backend.torch.tensor([ids], device=backend.input_device)

    def hook_for(layer):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            observed[layer] = hidden[:, -1, :].detach().float().cpu().numpy().copy()

        return hook

    try:
        for layer in LAYERS:
            handles.append(backend.base.layers[layer].register_forward_hook(hook_for(layer)))
        with backend.torch.inference_mode():
            output = backend.model(sequence, output_hidden_states=True, use_cache=False)
        checks = {}
        for layer in LAYERS:
            reference = output.hidden_states[layer + 1][:, -1, :].float().cpu().numpy()
            checks[str(layer)] = {
                "hf_hidden_state_index": layer + 1,
                "exact": bool(np.array_equal(observed[layer], reference)),
            }
        if not all(item["exact"] for item in checks.values()):
            raise ValueError("A decoder hook differs from its HF hidden-state index.")
        return {"status": "passed", "n_prompts": 1, "checks": checks}
    finally:
        for handle in handles:
            handle.remove()


def summarize_behavior(records):
    groups, joint = defaultdict(list), defaultdict(list)
    for row in records:
        groups[f"{row['view']}/{row['phase']}/{row['target']}"].append(row)
        joint[(row["case_id"], row["view"], row["phase"])].append(row)
    result = {}
    for name, selected in groups.items():
        correct = sum(r["prediction"] == r["gold"] for r in selected)
        result[name] = {
            "n_rows": len(selected),
            "n_templates": len({r["template_id"] for r in selected}),
            "n_unique_prompts": len({r["prompt_sha256"] for r in selected}),
            "correct": correct,
            "invalid": sum(r["prediction"] is None for r in selected),
            "accuracy_including_invalid": correct / len(selected),
        }
    joint_groups = defaultdict(list)
    for (_case, view, phase), selected in joint.items():
        joint_groups[f"{view}/{phase}"].append(all(r["prediction"] == r["gold"] for r in selected))
    return {
        "perspectives": result,
        "joint_correct": {
            key: {
                "n_cases": len(values),
                "correct": sum(values),
                "accuracy": sum(values) / len(values),
            }
            for key, values in joint_groups.items()
        },
        "interpretation": "Elicited answers against stipulated fictional-state gold; not measured hidden beliefs.",
    }


def run_belief_diagnostic(panel, output_dir, backend, *, progress=print):
    validate_panel(panel)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("Use a new belief-diagnostic output path.")
    if list(backend.settings.capture_layers) != list(LAYERS):
        raise ValueError("Use exactly the three prespecified decoder blocks.")
    output.mkdir(parents=True)
    protocol = {
        "schema": "belief_construct_development_protocol_v1",
        "created_at": utc_now(),
        "panel": panel,
        "layers": list(LAYERS),
        "primary_layer": PRIMARY_LAYER,
        "capture": "last chat-prompt token; story only; no perspective question or answer",
        "behavior": "Separate explicit perspective questions; actual reader question omitted in sender view",
        "readout": "train-only StandardScaler then per-row L2; C chosen only by neutral validation NLL",
        "C_grid": [0.001, 0.01, 0.1, 1.0, 10.0],
        "scope": "Fresh authored development task, not a new story-family main-study evaluation",
        "messages_are_known": True,
        "unknown_natural_message_forecast": False,
        "sender_is_narrated_character_not_live_generator": True,
    }
    atomic_json(output / "protocol.json", protocol)
    manifest = {
        "schema": "belief_construct_development_run_v1",
        "status": "running",
        "created_at": utc_now(),
        "backend": backend.metadata,
        "environment": environment_info(),
        "script_sha256": file_sha256(__file__),
        "panel_sha256": canonical_hash(panel),
        "protocol_sha256": file_sha256(output / "protocol.json"),
        "source_sha256": {
            name: file_sha256(name)
            for name in ("src/latent_probes/belief_panel.py", "src/latent_probes/belief_readout.py")
        },
        "is_test_fixture": backend.metadata["is_test_fixture"],
        "limits": [
            "Stipulated fictional perspective labels need independent human semantic review.",
            "Query-free narrated-story decodability does not establish spontaneous sender belief or causal use.",
            "Held-out wording templates share the same logical task; not new story families.",
            "Corrective-message controls collapse the three labels to world truth and are excluded from training.",
            "Sender-view actual reader labels are deliberately unknowable; duplicate inputs are not independent samples.",
        ],
    }
    atomic_json(output / "manifest.json", manifest)
    rows, matrices, captures, behavior, repeats = [], {f"layer_{x}": [] for x in LAYERS}, [], [], []
    capture_cache, generation_cache, repeated = {}, {}, set()
    try:
        for index, case in enumerate(panel["cases"]):
            for view in ("omniscient", "sender"):
                messages = case["messages"][view]
                cache_key = canonical_hash(messages)
                cached = cache_key in capture_cache
                if not cached:
                    capture_cache[cache_key] = capture_prefix(backend, messages)
                ids, vectors = capture_cache[cache_key]
                feature_path = Path("prefix_features") / (cache_key + ".npz")
                if not cached:
                    atomic_npz(output / feature_path, **vectors)
                if not captures:
                    manifest["layer_index_check"] = validate_layer_indices(backend, ids)
                repeat_key = (view, case["phase"])
                if repeat_key not in repeated:
                    repeat_ids, repeat_vectors = capture_prefix(backend, messages)
                    repeat_check = {
                        "case_id": case["id"],
                        "view": view,
                        "phase": case["phase"],
                        "prompt_ids_exact": ids == repeat_ids,
                        "layers_exact": {
                            k: bool(np.array_equal(v, repeat_vectors[k]))
                            for k, v in vectors.items()
                        },
                    }
                    repeats.append(repeat_check)
                    repeated.add(repeat_key)
                    atomic_json(output / "repeat_checks.json", repeats)
                    if not repeat_check["prompt_ids_exact"] or not all(
                        repeat_check["layers_exact"].values()
                    ):
                        raise ValueError(
                            "Identical query-free capture was not numerically repeatable."
                        )
                row = {
                    "example_id": case["id"] + "__" + view,
                    "template_id": case["template_id"],
                    "split": case["split"],
                    "phase": case["phase"],
                    "view": view,
                    "messages_text": "\n".join(m["content"] for m in messages),
                    "labels": case["labels"],
                }
                rows.append(row)
                for key, vector in vectors.items():
                    matrices[key].append(vector)
                captures.append(
                    {
                        "example_id": row["example_id"],
                        "messages_sha256": cache_key,
                        "feature_file": feature_path.as_posix(),
                        "feature_sha256": file_sha256(output / feature_path),
                        "prompt_token_ids": ids,
                        "selected_position": len(ids) - 1,
                        "reused_identical_capture": cached,
                        "generated_answer_tokens_supplied": 0,
                        "perspective_question_supplied": False,
                    }
                )
                atomic_json(output / "captures.json", captures)
                for target in (
                    ("world", "reader", "nested") if view == "omniscient" else ("world", "nested")
                ):
                    query = query_messages(case, view, target)
                    query_hash = canonical_hash(query)
                    reused = query_hash in generation_cache
                    if not reused:
                        generation_cache[query_hash] = backend.generate(
                            query,
                            GenerationSettings(max_new_tokens=6, do_sample=False),
                            42,
                            capture=False,
                        ).record()
                    generation = generation_cache[query_hash]
                    record = {
                        "example_id": row["example_id"] + "__" + target,
                        "case_id": case["id"],
                        "template_id": case["template_id"],
                        "phase": case["phase"],
                        "split": case["split"],
                        "view": view,
                        "target": target,
                        "messages": query,
                        "prompt_sha256": query_hash,
                        "generation": generation,
                        "reused_identical_generation": reused,
                        "prediction": parse_yes_no(generation["text"]),
                        "gold": case["labels"][target],
                    }
                    record["record_sha256"] = canonical_hash(record)
                    atomic_json(output / "behavior" / (record["example_id"] + ".json"), record)
                    behavior.append(record)
            manifest.update(n_cases_completed=index + 1, n_unique_generations=len(generation_cache))
            atomic_json(output / "manifest.json", manifest)
            if progress:
                progress(
                    f"Belief diagnostic {index + 1}/{len(panel['cases'])}: {case['phase']}",
                    flush=True,
                )
        features = {key: np.asarray(values, dtype=np.float32) for key, values in matrices.items()}
        atomic_npz(output / "features.npz", **features)
        atomic_json(output / "rows.json", rows)
        readouts = evaluate_belief_readouts(rows, features, primary_layer=PRIMARY_LAYER)
        atomic_json(output / "readouts.json", readouts)
        atomic_json(output / "behavior_summary.json", summarize_behavior(behavior))
        if canonical_hash(panel) != manifest["panel_sha256"] or any(
            file_sha256(name) != digest for name, digest in manifest["source_sha256"].items()
        ):
            raise ValueError("Panel or diagnostic source changed during execution.")
        manifest.update(
            status="complete",
            n_capture_rows=len(rows),
            n_behavior_rows=len(behavior),
            n_unique_captures=len(capture_cache),
            n_unique_generations=len(generation_cache),
        )
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        manifest["finished_at"] = utc_now()
        manifest["output_sha256"] = {
            str(path.relative_to(output)): file_sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        atomic_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output path.")
    panel = read_json(args.panel)
    validate_panel(panel)
    panel_digest = file_sha256(args.panel)
    settings = replace(
        load_config(args.config).model, capture_layers=list(LAYERS), local_files_only=True
    )
    print("Loading pinned cached model for fresh perspective diagnostic", flush=True)
    backend = HFBackend(settings)
    try:
        run_belief_diagnostic(panel, args.output, backend)
        if file_sha256(args.panel) != panel_digest:
            raise ValueError("Source panel bytes changed during execution.")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
