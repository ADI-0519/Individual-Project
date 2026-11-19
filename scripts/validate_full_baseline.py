"""Audit a full saved Apollo capture and replay a prespecified Llama subset.

This performs engineering checks, not a deception evaluation. The public API
accepts an already loaded backend and never closes it or generates responses.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from latent_probes.apollo import capture_signature, load_token_examples
from latent_probes.backends import HFBackend
from latent_probes.config import load_config
from latent_probes.storage import (
    atomic_json,
    canonical_hash,
    code_fingerprint,
    environment_info,
    file_sha256,
    read_json,
    utc_now,
)

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
BLOCK, HF_STATE, WIDTH, N_EXAMPLES = 8, 9, 4096, 536
STRIDE = 27


class ValidationFailure(RuntimeError):
    """A failed audit whose complete report has already been persisted."""

    def __init__(self, report):
        self.report = report
        super().__init__(f"Full baseline validation failed ({len(report['failures'])} checks)")


def error_metrics(reference, observed):
    reference, observed = np.asarray(reference), np.asarray(observed)
    if reference.shape != observed.shape or not reference.size:
        raise ValueError("Compared activation arrays require the same nonempty shape")
    if not np.isfinite(reference).all() or not np.isfinite(observed).all():
        raise ValueError("Compared activation arrays must be finite")
    difference = observed.astype(np.float64) - reference.astype(np.float64)
    return {
        "shape": list(reference.shape),
        "values": int(reference.size),
        "exact_fraction": float(np.mean(reference == observed)),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "rms_error": float(np.sqrt(np.mean(difference**2))),
    }


def bf16_neighbor_check(reference, observed):
    """Use adjacent BF16 values, including asymmetric spacing and signed zero.

    Inputs are finite FP32 storage of BF16 values. This integer construction
    avoids using a relative epsilon or requiring torch in numerical unit tests.
    """
    check = error_metrics(reference, observed)
    reference = np.asarray(reference, dtype=np.float32)
    observed = np.asarray(observed)
    raw = np.ascontiguousarray(reference).view(np.uint32)
    if np.any(raw & np.uint32(0xFFFF)):
        raise ValueError("Reference values are not exactly representable in BF16")
    bits = raw >> np.uint32(16)
    lower, upper = bits.copy(), bits.copy()
    positive, negative, zero = reference > 0, reference < 0, reference == 0
    lower[positive], upper[positive] = bits[positive] - 1, bits[positive] + 1
    lower[negative], upper[negative] = bits[negative] + 1, bits[negative] - 1
    lower[zero], upper[zero] = 0x8001, 0x0001
    lower_values = (lower.astype(np.uint32) << np.uint32(16)).view(np.float32)
    upper_values = (upper.astype(np.uint32) << np.uint32(16)).view(np.float32)
    outside = (observed < lower_values) | (observed > upper_values)
    check.update(
        allowed_bf16_neighbor_steps=1,
        outside_neighbor_count=int(outside.sum()),
        passed=not bool(outside.any()),
    )
    return check


def select_replay_examples(examples, stride=STRIDE):
    """Select both conditions of every 27th sorted source, plus endpoints."""
    if type(stride) is not int or stride < 1:
        raise ValueError("Selection stride must be a positive integer")
    pairs = sorted({row["pair_id"] for row in examples})
    if not pairs:
        raise ValueError("Cannot select from an empty source list")
    chosen = set(pairs[::stride]) | {pairs[0], pairs[-1]}
    return sorted(
        [row for row in examples if row["pair_id"] in chosen],
        key=lambda row: (row["pair_id"], row["example_id"]),
    )


def _record(report, name, passed, **details):
    entry = {"name": name, "passed": bool(passed), **details}
    report["checks"].append(entry)
    if not entry["passed"]:
        report["failures"].append(entry)
    return bool(passed)


def _attempt(report, name, function):
    try:
        details = function() or {}
        passed = details.pop("passed", True)
        return _record(report, name, passed, **details)
    except Exception as exc:
        return _record(report, name, False, error_type=type(exc).__name__, error=str(exc))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash_input(report, path):
    report["input_sha256"][str(path)] = file_sha256(path)


def _audit_artifacts(capture, probe, report):
    manifest = read_json(capture / "manifest.json")
    examples = load_token_examples(capture / "examples.jsonl")
    report["capture_manifest"] = {
        "stage": manifest.get("stage"),
        "status": manifest.get("status"),
        "examples": manifest.get("examples"),
        "token_rows": manifest.get("token_rows"),
        "backend": manifest.get("backend"),
        "identity": manifest.get("identity"),
    }
    report["source_fingerprints"] = {
        "capture_package_code_sha256": manifest.get("environment", {}).get("code_sha256"),
        "current_package_code_sha256": code_fingerprint(),
        "capture_git_commit": manifest.get("environment", {}).get("git_commit"),
        "current_core_files_sha256": {
            name: file_sha256(Path(inspect.getfile(HFBackend)).parent / name)
            for name in ("backends.py", "apollo.py", "storage.py", "config.py", "apollo_probe.py")
        },
    }
    fingerprint = report["source_fingerprints"]
    fingerprint["package_fingerprint_matches_capture"] = (
        fingerprint["capture_package_code_sha256"] == fingerprint["current_package_code_sha256"]
    )
    if not fingerprint["package_fingerprint_matches_capture"]:
        report["limitations"].append(
            "Package source fingerprint differs from capture; current core hashes are recorded. "
            "Successful numerical replay does not establish identical historical source code."
        )
    _require(
        manifest.get("stage") == "apollo_tokens" and manifest.get("status") == "complete",
        "A completed Apollo token capture is required",
    )
    _require(
        manifest.get("examples") == len(examples) == N_EXAMPLES,
        "Expected all 536 original captured examples",
    )
    ids = [row["example_id"] for row in examples]
    _require(len(set(ids)) == len(ids), "Duplicate example IDs in original capture")
    for filename in (
        "manifest.json",
        "config.json",
        "examples.jsonl",
        "features.npz",
        "metadata.csv",
    ):
        _hash_input(report, capture / filename)
    if (capture / "preparation.json").exists():
        _hash_input(report, capture / "preparation.json")
    for filename in ("manifest.json", "probe.npz"):
        _hash_input(report, probe / filename)
    _require(
        file_sha256(capture / "features.npz") == manifest["features_sha256"],
        "Packed feature hash differs from capture manifest",
    )
    _require(
        file_sha256(capture / "metadata.csv") == manifest["metadata_sha256"],
        "Metadata hash differs from capture manifest",
    )
    _require(
        canonical_hash(examples) == manifest["identity"]["examples_hash"],
        "Original example content differs from capture identity",
    )
    _require(
        canonical_hash(read_json(capture / "config.json")) == manifest["identity"]["config_hash"],
        "Original configuration differs from capture identity",
    )
    if (capture / "preparation.json").exists():
        _require(
            canonical_hash(read_json(capture / "preparation.json"))
            == manifest["identity"]["preparation_hash"],
            "Preparation identity differs",
        )
    with np.load(capture / "features.npz", allow_pickle=False) as archive:
        X, packed_ids = archive["X"].copy(), archive["example_ids"].copy()
        signature = str(archive["feature_signature"].item())
    _require(
        X.ndim == 2 and X.shape[1] == WIDTH and np.isfinite(X).all(),
        "Packed activations require finite width-4096 rows",
    )
    _require(X.dtype == np.float32, "Packed storage must be FP32")
    _require(
        not np.any(X.view(np.uint32) & np.uint32(0xFFFF)),
        "Packed stored values are not exactly representable in BF16",
    )
    _require(
        packed_ids.ndim == 1
        and packed_ids.dtype.kind == "U"
        and len(packed_ids) == len(X) == manifest["token_rows"],
        "Packed row count/IDs differ",
    )
    _require(signature == manifest["feature_signature"], "Packed feature signature differs")
    starts = np.flatnonzero(np.r_[True, packed_ids[1:] != packed_ids[:-1]])
    ends = np.r_[starts[1:], len(packed_ids)]
    _require(
        packed_ids[starts].tolist() == ids, "Packed example order or contiguous grouping differs"
    )
    bounds = {
        identifier: (int(start), int(end))
        for identifier, start, end in zip(ids, starts, ends, strict=True)
    }
    with (capture / "metadata.csv").open(newline="", encoding="utf-8-sig") as stream:
        metadata = list(csv.DictReader(stream))
    _require([row["example_id"] for row in metadata] == ids, "Metadata example order differs")
    for row, meta in zip(examples, metadata, strict=True):
        for key in ("pair_id", "split", "label", "label_target"):
            _require(str(row[key]) == meta[key], f"Metadata {key} differs: {row['example_id']}")
    expected_names = {canonical_hash(identifier)[:32] + ".json" for identifier in ids}
    _require(
        {path.name for path in (capture / "records").glob("*.json")} == expected_names,
        "Original record directory has missing or unexpected JSON files",
    )
    records, matrices = {}, {}
    for row in examples:
        identifier = row["example_id"]
        record_path = capture / "records" / (canonical_hash(identifier)[:32] + ".json")

        def inspect_record(row=row, identifier=identifier, record_path=record_path):
            _hash_input(report, record_path)
            record = read_json(record_path)
            records[identifier] = record
            _require(record["example_id"] == identifier, "Record example identity differs")
            _require(record["input_hash"] == canonical_hash(row), "Record input hash differs")
            _require(
                record["record_sha256"]
                == canonical_hash(
                    {key: value for key, value in record.items() if key != "record_sha256"}
                ),
                "Record content checksum differs",
            )
            _require(
                record.get("is_test_fixture") is False, "Fixture record is not a model capture"
            )
            feature = (capture / record["feature_file"]).resolve()
            _require(feature.is_relative_to(capture), "Record activation path escapes capture")
            _hash_input(report, feature)
            _require(
                file_sha256(feature) == record["feature_sha256"], "Record feature hash differs"
            )
            detail = record["capture"]
            positions, token_ids = detail["selected_token_positions"], detail["input_token_ids"]
            _require(
                positions and positions == sorted(set(positions)), "Invalid selected positions"
            )
            _require(
                all(type(pos) is int and 0 <= pos < len(token_ids) for pos in positions),
                "Selected token index out of range",
            )
            _require(
                detail["selected_token_ids"] == [token_ids[pos] for pos in positions],
                "Selected IDs do not agree with recorded input IDs",
            )
            _require(
                row["mode"] == "fixed"
                and row["trim_last_words"] == 5
                and detail["mode"] == "off_policy_fixed_response",
                "Wrong capture mode",
            )
            with np.load(feature, allow_pickle=False) as archive:
                matrix = archive["X"]
                _require(
                    matrix.shape == (len(positions), WIDTH) and np.isfinite(matrix).all(),
                    "Record array shape or finite-value check failed",
                )
                start, end = bounds[identifier]
                _require(
                    np.array_equal(matrix, X[start:end]), "Packed rows differ from record array"
                )
                matrices[identifier] = matrix.copy()
            return {
                "example_id": identifier,
                "n_tokens": len(positions),
                "packed_rows": [start, end],
            }

        _attempt(report, "artifact_record:" + identifier, inspect_record)
    if len(records) == len(examples):
        _record(
            report,
            "ordered_records_hash",
            canonical_hash([records[i] for i in ids]) == manifest["records_hash"],
        )
    expected_tokens = {record["feature_file"] for record in records.values()}
    actual_tokens = {
        path.relative_to(capture).as_posix() for path in (capture / "tokens").glob("*.npz")
    }
    _record(report, "token_files_exact_set", expected_tokens == actual_tokens)
    probe_manifest = read_json(probe / "manifest.json")
    _require(
        probe_manifest["features_sha256"] == manifest["features_sha256"]
        and probe_manifest["labels_sha256"] == manifest["metadata_sha256"]
        and probe_manifest["feature_signature"] == signature,
        "Frozen probe was not fitted on this full capture/profile",
    )
    _require(
        file_sha256(probe / "probe.npz") == probe_manifest["probe_sha256"],
        "Frozen numerical probe hash differs",
    )
    for split, key in (("train", "train_example_ids"), ("validation", "validation_example_ids")):
        _require(
            set(probe_manifest[key]) == {r["example_id"] for r in examples if r["split"] == split},
            f"Frozen probe {split} identity differs",
        )
    with np.load(probe / "probe.npz", allow_pickle=False) as archive:
        for key in ("mean", "scale", "coefficient"):
            values = archive[key]
            _require(
                values.shape == (WIDTH,) and np.isfinite(values).all(),
                f"Invalid frozen probe {key}",
            )
        _require(
            np.all(archive["scale"] > 0) and float(archive["intercept"]) == 0,
            "Frozen probe scale/intercept differs from declared recipe",
        )
    report["artifact_summary"] = {
        "examples": len(examples),
        "unique_sources": len({r["pair_id"] for r in examples}),
        "token_rows": len(X),
        "width": WIDTH,
        "finite_values": int(X.size),
        "storage_dtype": str(X.dtype),
        "all_stored_values_on_bf16_grid": True,
    }
    return manifest, examples, records, matrices


def _gpu_memory(torch):
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)
    return {
        str(index): {
            "allocated_gib": torch.cuda.memory_allocated(index) / 2**30,
            "reserved_gib": torch.cuda.memory_reserved(index) / 2**30,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(index) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(index) / 2**30,
        }
        for index in range(torch.cuda.device_count())
    }


def _phase(report, backend, name, function):
    torch = backend.torch
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)
        torch.cuda.reset_peak_memory_stats(index)
    start = time.perf_counter()
    try:
        return function()
    finally:
        report["memory_phases"].append(
            {
                "name": name,
                "seconds": time.perf_counter() - start,
                "devices": _gpu_memory(torch),
            }
        )


def _direct_capture(backend, full, positions):
    torch = backend.torch
    captured, dtypes = [], []

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        dtypes.append(str(hidden.dtype))
        captured.append(hidden[0, positions, :].detach().float().cpu().numpy().copy())

    handle = backend.base.layers[BLOCK].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output = backend.base(
                input_ids=full,
                attention_mask=torch.ones_like(full),
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            expected = output.hidden_states[HF_STATE][0, positions, :].float().cpu().numpy().copy()
    finally:
        handle.remove()
    _require(len(captured) == 1, "Expected one decoder-block hook invocation")
    return captured[0], expected, dtypes


def _runtime_checks(backend, manifest, report):
    import bitsandbytes as bnb

    torch = backend.torch
    modules = [
        module for module in backend.model.modules() if isinstance(module, bnb.nn.Linear4bit)
    ]
    quant_types = {
        str(getattr(getattr(m.weight, "quant_state", None), "quant_type", None)) for m in modules
    }
    report["runtime"] = {
        "backend": backend.metadata,
        "actual_input_device": str(backend.input_device),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "linear4bit_modules": len(modules),
        "actual_quantization_types": sorted(quant_types),
        "actual_compute_dtypes": sorted({str(m.compute_dtype) for m in modules}),
        "decoder_blocks": len(backend.base.layers),
        "hidden_size": backend.model.config.hidden_size,
    }
    expected = {
        "backend": "huggingface",
        "is_test_fixture": False,
        "model_id": MODEL,
        "resolved_revision": REVISION,
        "dtype": "bfloat16",
        "quantization": "4bit",
        "capture_layers": [BLOCK],
        "hidden_size": WIDTH,
    }
    for key, value in expected.items():
        _require(
            manifest["backend"][key] == value and backend.metadata[key] == value,
            f"Capture/runtime backend mismatch: {key}",
        )
    _require(torch.cuda.is_available() and backend.input_device.type == "cuda", "CUDA is required")
    _require(len(modules) == 224, "Expected 224 actual bitsandbytes Linear4bit modules")
    _require(
        all(module.compute_dtype == torch.bfloat16 for module in modules),
        "Linear4bit compute dtype is not uniformly BF16",
    )
    _require(quant_types == {"nf4"}, "Actual quantization states are not uniformly NF4")
    _require(
        len(backend.base.layers) == 32 and backend.model.config.hidden_size == WIDTH,
        "Actual model block count/hidden width differs",
    )
    _require(
        capture_signature(backend.metadata, BLOCK) == manifest["feature_signature"],
        "Loaded backend capture signature differs",
    )
    report["runtime"]["gpu_name"] = torch.cuda.get_device_name(backend.input_device)
    report["memory_at_validation_entry"] = _gpu_memory(torch)


def _replay_one(example, record, saved, backend, report):
    torch = backend.torch
    identifier = example["example_id"]
    prefix = identifier + ":"
    features, fresh = _phase(
        report,
        backend,
        prefix + "backend_replay",
        lambda: backend.forced_token_activations(example["messages"], example["response"], 5),
    )
    compared_fields = (
        "mode",
        "text",
        "rendered_dialogue",
        "input_token_ids",
        "selected_token_positions",
        "selected_token_ids",
        "selected_decoded_text",
        "character_span",
        "excluded_suffix",
        "trim_last_words",
        "mask",
    )
    differences = [key for key in compared_fields if fresh[key] != record["capture"][key]]
    _record(
        report,
        prefix + "fresh_tokenization_and_mask",
        not differences,
        differing_fields=differences,
        compared_fields=list(compared_fields),
    )
    replay = features[f"layer_{BLOCK}_tokens"]
    _attempt(report, prefix + "saved_vs_backend_replay", lambda: bf16_neighbor_check(saved, replay))
    _require(not differences, "Cannot continue causal comparison with changed tokenization/mask")
    positions = fresh["selected_token_positions"]
    full = torch.tensor([fresh["input_token_ids"]], device=backend.input_device)
    first, expected, first_dtypes = _phase(
        report,
        backend,
        prefix + "hf_hidden_states",
        lambda: _direct_capture(backend, full, positions),
    )
    second, repeated_expected, second_dtypes = _phase(
        report,
        backend,
        prefix + "same_options_repeat",
        lambda: _direct_capture(backend, full, positions),
    )
    _record(
        report,
        prefix + "native_bf16_residual",
        first_dtypes == second_dtypes == ["torch.bfloat16"],
        first_dtypes=first_dtypes,
        repeated_dtypes=second_dtypes,
    )
    for name, left, right in (
        ("same_forward_hook_equals_hf9", first, expected),
        ("repeat_forward_hook_equals_hf9", second, repeated_expected),
        ("unchanged_same_options_repeat", first, second),
    ):
        _attempt(
            report,
            prefix + name,
            lambda left=left, right=right: {
                **error_metrics(left, right),
                "passed": bool(np.array_equal(left, right)),
            },
        )
    _attempt(
        report, prefix + "backend_replay_vs_hf9", lambda: bf16_neighbor_check(replay, expected)
    )
    vocabulary = backend.model.get_input_embeddings().num_embeddings
    for endpoint, row_index in (("first", 0), ("last", len(positions) - 1)):
        position = positions[row_index]
        _require(position + 1 < full.shape[1], "Causal mutation requires a nonempty future suffix")
        mutant = full.clone()
        mutant[:, position + 1 :] = (mutant[:, position + 1 :] + 1) % vocabulary
        _require(
            mutant.shape == full.shape
            and torch.equal(mutant[:, : position + 1], full[:, : position + 1])
            and bool(torch.all(mutant[:, position + 1 :] != full[:, position + 1 :]).item()),
            "Future-only mutation construction failed",
        )
        observed = _phase(
            report,
            backend,
            prefix + endpoint + "_future_mutation",
            lambda: backend._capture_token_rows(mutant, [position])[f"layer_{BLOCK}_tokens"],
        )
        reference = replay[row_index : row_index + 1]
        _attempt(
            report,
            prefix + endpoint + "_future_suffix_invariance",
            lambda: {
                **error_metrics(reference, observed),
                "passed": bool(np.array_equal(reference, observed)),
                "token_position": position,
                "input_length": int(full.shape[1]),
                "changed_future_tokens": int(full.shape[1] - position - 1),
                "mutated_input_token_ids": mutant[0].tolist(),
                "original_input_token_ids_sha256": canonical_hash(full[0].tolist()),
            },
        )


def run_validation(capture_dir, probe_dir, output_dir, backend):
    """Audit all saved examples and replay the fixed subset on a caller-owned backend.

    On failure, persist validation.json before raising ValidationFailure. This
    function never closes the supplied backend, including on failure.
    """
    capture, probe, output = (
        Path(capture_dir).resolve(),
        Path(probe_dir).resolve(),
        Path(output_dir).resolve(),
    )
    if output.exists():
        raise FileExistsError("Validation output must be a new directory")
    if output.is_relative_to(capture) or output.is_relative_to(probe):
        raise ValueError("Validation output must not be inside immutable capture/probe directories")
    output.mkdir(parents=True)
    started = time.perf_counter()
    report = {
        "schema": "full_baseline_validation_v1",
        "status": "running",
        "created_at": utc_now(),
        "is_deception_evaluation": False,
        "checks": [],
        "failures": [],
        "input_sha256": {},
        "memory_phases": [],
        "environment": environment_info(),
        "validator_sha256": file_sha256(__file__),
        "comparison_policy": {
            "saved_replay": "at most one adjacent representable BF16 value per coordinate",
            "backend_vs_hf9": "at most one adjacent representable BF16 value per coordinate",
            "same_forward_and_repeat": "exact equality, identical full sequence shape and forward flags",
            "causality": "exact equality; independent original clone for each endpoint; all future IDs +1 modulo vocabulary",
            "forwards_per_selected_example": 5,
        },
        "limitations": [
            "NF4/BF16 engineering verification, not deception detection or a paper-performance reproduction.",
            "All 536 artifacts are audited; only the prespecified source subset is replayed on the GPU.",
            "Timing excludes caller-owned backend loading; no new model responses are generated.",
            "Each memory phase resets CUDA peak counters and includes already-resident model allocations.",
            "Entry memory records preexisting lifetime counters; it does not isolate backend-load peak memory.",
        ],
    }
    try:
        manifest, examples, records, matrices = _audit_artifacts(capture, probe, report)
        _record(report, "full_artifact_audit", not report["failures"])
        selected = select_replay_examples(examples)
        selection = {
            "recorded_at": utc_now(),
            "rule": "every 27th sorted unique pair_id, plus first and last; both conditions",
            "stride": STRIDE,
            "capture_manifest_sha256": file_sha256(capture / "manifest.json"),
            "selected_example_ids": [row["example_id"] for row in selected],
            "selected_pair_ids": sorted({row["pair_id"] for row in selected}),
            "n_examples": len(selected),
            "n_sources": len({row["pair_id"] for row in selected}),
            "selection_does_not_use_activation_values_or_probe_scores": True,
        }
        atomic_json(output / "selection.json", selection)
        report["selection"] = selection
        if report["failures"]:
            raise ValueError("Artifact audit failed; GPU comparisons were not attempted")
        _runtime_checks(backend, manifest, report)
        _record(report, "actual_runtime_model_profile", True)
        for row in selected:
            identifier = row["example_id"]
            _attempt(
                report,
                identifier + ":replay_sequence",
                lambda row=row, identifier=identifier: _replay_one(
                    row, records[identifier], matrices[identifier], backend, report
                ),
            )
    except Exception as exc:
        _record(
            report, "validation_execution", False, error_type=type(exc).__name__, error=str(exc)
        )
    finally:
        for path, original in list(report["input_sha256"].items()):
            _attempt(
                report,
                "input_unchanged:" + path,
                lambda path=path, original=original: {"passed": file_sha256(path) == original},
            )
        if report["memory_phases"]:
            report["validation_forward_peak_allocated_gib"] = {
                index: max(
                    phase["devices"][index]["peak_allocated_gib"]
                    for phase in report["memory_phases"]
                )
                for index in report["memory_phases"][0]["devices"]
            }
        report.update(
            status="fail" if report["failures"] else "pass",
            completed_at=utc_now(),
            validation_seconds_excluding_backend_load=time.perf_counter() - started,
        )
        atomic_json(output / "validation.json", report)
    if report["failures"]:
        raise ValidationFailure(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Validation output must be a new directory")
    config = load_config(args.capture / "config.json")
    started = time.perf_counter()
    backend = HFBackend(replace(config.model, revision=REVISION, local_files_only=True))
    load_seconds = time.perf_counter() - started
    try:
        report = run_validation(args.capture, args.probe, args.output, backend)
        print(json.dumps({"backend_load_seconds": load_seconds, "validation": report}, indent=2))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
