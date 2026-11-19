"""Verify genuine checkpoint captures; never interpret this as deception evidence.

Defaults match the manual CPU smoke workflow; explicit paths support other
float32 devices. The second model load is cached and checks actual Hugging Face
layer indexing and causal-prefix invariance.
"""

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from latent_probes.backends import HFBackend
from latent_probes.config import load_config
from latent_probes.storage import atomic_json, file_sha256, read_json


def main(argv=None):
    root = Path("runs")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=root / "apollo_smoke_cpu")
    parser.add_argument("--controls", type=Path, default=root / "apollo_smoke_controls")
    parser.add_argument("--scores", type=Path, default=root / "apollo_smoke_control_scores.csv")
    parser.add_argument("--threshold", type=Path, default=root / "apollo_smoke_threshold.json")
    parser.add_argument("--output", type=Path, default=root / "apollo_smoke_validation.json")
    args = parser.parse_args(argv)
    training, controls = args.training, args.controls
    # Prespecified float32 tolerances: never adapt these to observed errors.
    rtol, atol = 1e-4, 1e-4
    manifests = [read_json(p / "manifest.json") for p in (training, controls)]
    for manifest in manifests:
        assert manifest["status"] == "complete", "A model capture run is incomplete"
        assert manifest["backend"]["backend"] == "huggingface"
        assert manifest["backend"]["is_test_fixture"] is False
        assert manifest["backend"]["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
        assert manifest["backend"]["resolved_revision"] == (
            "7ae557604adf67be50417f59c2c2f167def9a775"
        ), "Unexpected Qwen checkpoint revision"
        assert manifest["backend"]["dtype"] == "float32", "This validator requires float32 captures"
    assert manifests[0]["examples"] == 14, "Expected seven eligible paired smoke facts"
    assert manifests[1]["examples"] == 6
    assert manifests[0]["feature_signature"] == manifests[1]["feature_signature"]
    rows = [read_json(p) for p in sorted((training / "records").glob("*.json"))]
    record = rows[0]
    capture = record["capture"]
    assert capture["mode"] == "off_policy_fixed_response"
    assert len(capture["excluded_suffix"].split()) == 5
    assert capture["selected_token_ids"]
    assert len(capture["selected_token_ids"]) == len(capture["selected_token_positions"])
    config = load_config(training / "config.json")
    control_config = load_config(controls / "config.json")
    assert config.model.dtype == control_config.model.dtype == "float32", (
        "The declared 1e-4 tolerance is only validated for float32"
    )
    model_settings = replace(config.model, revision=manifests[0]["backend"]["resolved_revision"])
    backend = HFBackend(model_settings)
    try:
        layer = model_settings.capture_layers[0]
        torch = backend.torch
        actual_backend = dict(backend.metadata)
        actual_input_device = str(backend.input_device)
        runtime = {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "device": actual_input_device,
        }
        if backend.input_device.type == "cuda":
            runtime.update(
                gpu_name=torch.cuda.get_device_name(backend.input_device),
                compute_capability=list(torch.cuda.get_device_capability(backend.input_device)),
            )
        parameter_dtypes = sorted(
            {str(parameter.dtype) for parameter in backend.model.parameters()}
        )
        assert parameter_dtypes == ["torch.float32"], "Reloaded model parameters must be float32"
        token_ids = capture["input_token_ids"]
        positions = capture["selected_token_positions"]
        full = torch.tensor([token_ids], device=backend.input_device)
        with np.load(training / record["feature_file"], allow_pickle=False) as archive:
            saved = archive["X"]
        with torch.inference_mode():
            out = backend.base(
                input_ids=full,
                attention_mask=torch.ones_like(full),
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            expected = out.hidden_states[layer + 1][0, positions, :].float().cpu().numpy()
        np.testing.assert_allclose(saved, expected, rtol=rtol, atol=atol)
        hf_errors = np.abs(saved - expected)
        prefix_checks = {}
        for name, row_index in (("first", 0), ("last", len(positions) - 1)):
            position = positions[row_index]
            prefix = full[:, : position + 1]
            prefix_row = backend._capture_token_rows(prefix, [position])[f"layer_{layer}_tokens"][0]
            np.testing.assert_allclose(saved[row_index], prefix_row, rtol=rtol, atol=atol)
            errors = np.abs(saved[row_index] - prefix_row)
            prefix_checks[name] = {
                "selected_row_index": row_index,
                "token_position": position,
                "max_absolute_error": float(np.max(errors)),
                "mean_absolute_error": float(np.mean(errors)),
            }
        max_error = max(check["max_absolute_error"] for check in prefix_checks.values())
    finally:
        backend.close()
    with args.scores.open(newline="") as stream:
        scores = list(csv.DictReader(stream))
    assert len(scores) == 6
    assert all(np.isfinite(float(row["score"])) for row in scores)
    threshold = read_json(args.threshold)
    assert threshold["n_control"] == 6
    assert any("unreliable" in text for text in threshold["warnings"])
    report = {
        "status": "real_model_engineering_smoke_passed",
        "is_deception_evaluation": False,
        "checkpoint": manifests[0]["backend"]["model_id"],
        "resolved_revision": manifests[0]["backend"]["resolved_revision"],
        "model_dtype": model_settings.dtype,
        "configured_device": model_settings.device,
        "control_configured_device": control_config.model.device,
        "actual_input_device": actual_input_device,
        "actual_parameter_dtypes": parameter_dtypes,
        "runtime": runtime,
        "capture_environment": manifests[0]["environment"],
        "backend_metadata": {
            "training_capture": manifests[0]["backend"],
            "control_capture": manifests[1]["backend"],
            "validation_reload": actual_backend,
        },
        "input_paths": {
            "training": str(training),
            "controls": str(controls),
            "scores": str(args.scores),
            "threshold": str(args.threshold),
        },
        "training_examples": 14,
        "control_examples": 6,
        "actual_hf_hidden_state_index_verified": layer + 1,
        "hf_hidden_state_parity": {
            "passed": True,
            "max_absolute_error": float(np.max(hf_errors)),
            "mean_absolute_error": float(np.mean(hf_errors)),
            "rtol": rtol,
            "atol": atol,
        },
        "causal_prefix_invariance": {
            "passed": True,
            "max_absolute_error": max_error,
            "checks": prefix_checks,
            "rtol": rtol,
            "atol": atol,
        },
        "uv_lock_sha256": file_sha256("uv.lock"),
        "limitations": [
            "Qwen 0.5B float32 "
            + ("CPU" if actual_input_device == "cpu" else actual_input_device)
            + " engineering check, not the 8B study.",
            "Two validation instruction examples cannot establish detection performance.",
            "Six neutral controls cannot establish a reliable 1% FPR operating point.",
        ],
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
