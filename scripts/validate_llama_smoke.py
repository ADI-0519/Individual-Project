"""Validate one real NF4/BF16 Llama smoke; this is not a deception evaluation."""

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from latent_probes.apollo import capture_signature, load_token_examples
from latent_probes.apollo_probe import score_apollo_probe
from latent_probes.backends import HFBackend
from latent_probes.config import GenerationSettings, load_config
from latent_probes.storage import (
    atomic_json,
    atomic_npz,
    canonical_hash,
    code_fingerprint,
    environment_info,
    file_sha256,
    read_json,
    utc_now,
)

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
LAYER, WIDTH = 8, 4096


def error_metrics(reference, observed):
    assert reference.shape == observed.shape, "Compared activation shapes differ"
    assert np.isfinite(reference).all() and np.isfinite(observed).all()
    errors = observed.astype(np.float64) - reference.astype(np.float64)
    return {
        "values": int(errors.size),
        "exact_fraction": float(np.mean(reference == observed)),
        "max_absolute_error": float(np.max(np.abs(errors))),
        "rms_error": float(np.sqrt(np.mean(errors**2))),
    }


def require_exact(report, name, reference, observed, message, **details):
    check = error_metrics(reference, observed)
    check.update(details)
    check["passed"] = bool(np.array_equal(reference, observed))
    report["checks"][name] = check
    assert check["passed"], message


def validate(args, report):
    training = args.training.resolve()
    manifest = read_json(training / "manifest.json")
    assert manifest["stage"] == "apollo_tokens" and manifest["status"] == "complete"
    assert manifest["examples"] == 14, "Expected fourteen paired instruction examples"
    captured_backend = manifest["backend"]
    for key, expected in {
        "backend": "huggingface",
        "is_test_fixture": False,
        "model_id": MODEL,
        "resolved_revision": REVISION,
        "dtype": "bfloat16",
        "quantization": "4bit",
        "capture_layers": [LAYER],
        "hidden_size": WIDTH,
    }.items():
        assert captured_backend[key] == expected, f"Unexpected capture metadata: {key}"
    config = load_config(training / "config.json")
    assert config.model.model_id == MODEL and config.model.dtype == "bfloat16"
    assert config.model.quantization == "4bit" and config.model.capture_layers == [LAYER]
    assert config.model.device == "cuda", "A CUDA smoke capture is required"
    inputs = [
        training / name
        for name in (
            "manifest.json",
            "config.json",
            "examples.jsonl",
            "features.npz",
            "metadata.csv",
        )
    ]
    inputs += [args.probe / "manifest.json", args.probe / "probe.npz", Path(__file__)]
    report["input_sha256"] = {str(path): file_sha256(path) for path in inputs}
    for filename, field in (
        ("features.npz", "features_sha256"),
        ("metadata.csv", "metadata_sha256"),
    ):
        assert file_sha256(training / filename) == manifest[field], "Capture artifact changed"
    examples = load_token_examples(training / "examples.jsonl")
    assert len(examples) == 14
    assert len(list((training / "records").glob("*.json"))) == 14
    records, selected = [], None
    for example in examples:
        path = training / "records" / (canonical_hash(example["example_id"])[:32] + ".json")
        record = read_json(path)
        assert record["example_id"] == example["example_id"]
        assert record["input_hash"] == canonical_hash(example)
        assert record["record_sha256"] == canonical_hash(
            {key: value for key, value in record.items() if key != "record_sha256"}
        )
        assert record["is_test_fixture"] is False
        feature_path = (training / record["feature_file"]).resolve()
        assert feature_path.is_relative_to(training), "Activation path escapes training run"
        assert file_sha256(feature_path) == record["feature_sha256"]
        capture = record["capture"]
        positions, ids = capture["selected_token_positions"], capture["input_token_ids"]
        assert positions and positions == sorted(set(positions))
        assert all(type(p) is int and 0 <= p < len(ids) for p in positions)
        assert capture["selected_token_ids"] == [ids[p] for p in positions]
        with np.load(feature_path, allow_pickle=False) as archive:
            matrix = archive["X"]
            assert matrix.shape == (len(positions), WIDTH) and np.isfinite(matrix).all()
            if selected is None:
                assert example["mode"] == "fixed" and example["trim_last_words"] == 5
                assert capture["mode"] == "off_policy_fixed_response"
                selected = (example, record, matrix.copy())
        report["input_sha256"].update(
            {str(path): file_sha256(path), str(feature_path): file_sha256(feature_path)}
        )
        records.append(record)
    assert canonical_hash(records) == manifest["records_hash"]
    with np.load(training / "features.npz", allow_pickle=False) as archive:
        assert archive["X"].ndim == 2 and archive["X"].shape[1] == WIDTH
        assert len(archive["X"]) == len(archive["example_ids"]) == manifest["token_rows"]
        assert np.isfinite(archive["X"]).all()
        assert str(archive["feature_signature"].item()) == manifest["feature_signature"]
    example, record, saved = selected
    report.update(
        checkpoint=MODEL,
        resolved_revision=REVISION,
        training_examples=14,
        selected_example_id=example["example_id"],
        captured_backend=captured_backend,
    )
    settings = replace(config.model, revision=REVISION)
    load_start = time.perf_counter()
    try:
        backend = HFBackend(settings)
    finally:
        report["load_seconds"] = time.perf_counter() - load_start
    try:
        import bitsandbytes as bnb

        torch = backend.torch
        assert torch.cuda.is_available() and backend.input_device.type == "cuda"
        assert bool(getattr(backend.model, "is_loaded_in_4bit", False))
        quantized = [m for m in backend.model.modules() if isinstance(m, bnb.nn.Linear4bit)]
        assert quantized, "No actual bitsandbytes Linear4bit modules were loaded"
        assert all(m.compute_dtype == torch.bfloat16 for m in quantized)
        torch.cuda.synchronize(backend.input_device)
        signature = capture_signature(backend.metadata, LAYER)
        assert signature == manifest["feature_signature"], "Reloaded capture profile differs"
        report.update(
            load_seconds=time.perf_counter() - load_start,
            runtime_backend=backend.metadata,
            actual_input_device=str(backend.input_device),
            gpu_name=torch.cuda.get_device_name(backend.input_device),
            torch_cuda=torch.version.cuda,
            linear4bit_modules=len(quantized),
        )
        peaks = {}

        def record_peaks():
            for index in range(torch.cuda.device_count()):
                peaks[str(index)] = max(
                    peaks.get(str(index), 0.0), torch.cuda.max_memory_allocated(index) / 2**30
                )
            report["peak_allocated_gib"] = dict(peaks)

        record_peaks()
        _, fresh = backend.forced_token_activations(
            example["messages"], example["response"], example["trim_last_words"]
        )
        record_peaks()
        for field in (
            "input_token_ids",
            "selected_token_positions",
            "selected_token_ids",
            "character_span",
            "excluded_suffix",
            "trim_last_words",
        ):
            assert fresh[field] == record["capture"][field], f"Replay mask/IDs differ: {field}"
        report["checks"]["replayed_mask_and_token_ids"] = {"passed": True}
        positions = fresh["selected_token_positions"]
        full = torch.tensor([fresh["input_token_ids"]], device=backend.input_device)
        hooked, native_dtypes = [], []
        report["native_hook_dtypes"] = native_dtypes

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            native_dtypes.append(str(hidden.dtype))
            assert hidden.dtype == torch.bfloat16, "Native residual output is not BF16"
            hooked.append(hidden[0, positions, :].detach().float().cpu().numpy().copy())

        handle = backend.base.layers[LAYER].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                out = backend.base(
                    input_ids=full,
                    attention_mask=torch.ones_like(full),
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                expected = out.hidden_states[9][0, positions, :].float().cpu().numpy().copy()
        finally:
            handle.remove()
        assert len(hooked) == 1
        require_exact(
            report,
            "same_forward_hook_equals_hf_hidden_state_9",
            hooked[0],
            expected,
            "Block-8 hook differs from HF hidden_states[9] in the same forward",
        )
        saved_bf16 = torch.from_numpy(saved).to(torch.bfloat16)
        assert np.array_equal(saved, saved_bf16.float().numpy()), "Saved values are not exact BF16"
        lower = torch.nextafter(saved_bf16, torch.full_like(saved_bf16, -float("inf")))
        upper = torch.nextafter(saved_bf16, torch.full_like(saved_bf16, float("inf")))
        outside = (expected < lower.float().numpy()) | (expected > upper.float().numpy())
        replay_check = error_metrics(saved, expected)
        replay_check.update(
            outside_neighbor_count=int(outside.sum()),
            allowed_bf16_neighbor_steps=1,
            passed=not bool(outside.any()),
        )
        report["checks"]["saved_vectors_vs_reload"] = replay_check
        assert replay_check["passed"], "Reload exceeds the predeclared one-BF16-neighbor bound"
        first = backend._capture_token_rows(full, positions)[f"layer_{LAYER}_tokens"]
        second = backend._capture_token_rows(full, positions)[f"layer_{LAYER}_tokens"]
        require_exact(
            report,
            "unchanged_same_shape_repeats",
            first,
            second,
            "Unchanged repeats differ: causal validation is numerically inconclusive",
        )
        vocabulary = backend.model.get_input_embeddings().num_embeddings
        for name, row_index in (("first", 0), ("last", len(positions) - 1)):
            position = positions[row_index]
            assert position + 1 < full.shape[1], "Causal check needs a nonempty future suffix"
            mutant = full.clone()
            mutant[:, position + 1 :] = (mutant[:, position + 1 :] + 1) % vocabulary
            assert mutant.shape == full.shape
            assert torch.equal(mutant[:, : position + 1], full[:, : position + 1])
            assert bool(torch.all(mutant[:, position + 1 :] != full[:, position + 1 :]).item())
            observed = backend._capture_token_rows(mutant, [position])[f"layer_{LAYER}_tokens"]
            key = name + "_token_future_suffix_invariance"
            require_exact(
                report,
                key,
                first[row_index : row_index + 1],
                observed,
                f"Changing only future IDs changed the {name} selected token",
                token_position=position,
                input_length=int(full.shape[1]),
                changed_future_tokens=int(full.shape[1] - position - 1),
                mutated_input_token_ids=mutant[0].tolist(),
            )
        record_peaks()
        messages = [{"role": "user", "content": "What is 17 plus 25? Answer briefly."}]
        features, generation = backend.generated_token_activations(
            messages, GenerationSettings(max_new_tokens=16, do_sample=False), 42
        )
        record_peaks()
        matrix = features[f"layer_{LAYER}_tokens"]
        assert matrix.shape == (len(generation["selected_token_ids"]), WIDTH)
        assert len(matrix) and np.isfinite(matrix).all()
        feature_path = args.output / "neutral_features.npz"
        atomic_npz(
            feature_path,
            X=matrix,
            feature_signature=np.asarray(signature),
            example_ids=np.asarray(["llama8b_neutral_arithmetic_smoke"] * len(matrix)),
        )
        atomic_json(
            args.output / "neutral_generation.json",
            {
                "is_test_fixture": False,
                "is_deception_evaluation": False,
                "messages": messages,
                "capture": generation,
                "backend": backend.metadata,
                "features_sha256": file_sha256(feature_path),
            },
        )
        scores_path = args.output / "neutral_scores.csv"
        score_apollo_probe(feature_path, args.probe, scores_path)
        with scores_path.open(newline="", encoding="utf-8") as stream:
            scores = list(csv.DictReader(stream))
        assert len(scores) == 1 and np.isfinite(float(scores[0]["score"]))
        report.update(
            neutral_raw_margin=float(scores[0]["score"]),
            neutral_generated_text=generation["text"],
            neutral_features_sha256=file_sha256(feature_path),
            neutral_scores_sha256=file_sha256(scores_path),
        )
    finally:
        backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "created_at": utc_now(),
        "is_deception_evaluation": False,
        "checks": {},
        "comparison_policy": {
            "same_forward_and_same_shape_causality": "exact numeric equality; zero tolerance",
            "saved_replay": "at most one adjacent representable BF16 value per coordinate",
            "future_mutation": "fresh original clone per endpoint; (future_id + 1) % vocabulary",
        },
        "limitations": [
            "Llama 8B NF4/BF16 engineering validation, not a published performance reproduction.",
            "One saved record and one neutral generation do not establish deception detection.",
            "The neutral raw margin is not a probability or an FPR calibration.",
        ],
    }
    try:
        report.update(environment=environment_info(), code_sha256=code_fingerprint())
        validate(args, report)
        report["status"] = "pass"
    except Exception as exc:
        report.update(status="fail", error_type=type(exc).__name__, error_message=str(exc))
        raise
    finally:
        report["completed_at"] = utc_now()
        atomic_json(args.output / "validation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
