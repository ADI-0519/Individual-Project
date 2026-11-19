"""Verify genuine checkpoint captures; never interpret this as deception evidence.

Run after the manual CPU smoke workflow. The second model load is cached and
checks actual Hugging Face layer indexing and causal-prefix invariance.
"""

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from latent_probes.backends import HFBackend
from latent_probes.config import load_config
from latent_probes.storage import atomic_json, file_sha256, read_json


def main():
    root = Path("runs")
    training = root / "apollo_smoke_cpu"
    controls = root / "apollo_smoke_controls"
    manifests = [read_json(p / "manifest.json") for p in (training, controls)]
    for manifest in manifests:
        assert manifest["status"] == "complete", "A model capture run is incomplete"
        assert manifest["backend"]["backend"] == "huggingface"
        assert manifest["backend"]["is_test_fixture"] is False
        assert manifest["backend"]["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
        assert manifest["backend"]["resolved_revision"], "Unresolved model revision"
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
    model_settings = replace(config.model, revision=manifests[0]["backend"]["resolved_revision"])
    backend = HFBackend(model_settings)
    try:
        layer = model_settings.capture_layers[0]
        torch = backend.torch
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
        np.testing.assert_allclose(saved, expected, rtol=1e-4, atol=1e-4)
        prefix = full[:, : positions[0] + 1]
        prefix_row = backend._capture_token_rows(prefix, [positions[0]])[f"layer_{layer}_tokens"][0]
        np.testing.assert_allclose(saved[0], prefix_row, rtol=1e-4, atol=1e-4)
        max_error = float(np.max(np.abs(saved[0] - prefix_row)))
    finally:
        backend.close()
    with (root / "apollo_smoke_control_scores.csv").open(newline="") as stream:
        scores = list(csv.DictReader(stream))
    assert len(scores) == 6
    assert all(np.isfinite(float(row["score"])) for row in scores)
    threshold = read_json(root / "apollo_smoke_threshold.json")
    assert threshold["n_control"] == 6
    assert any("unreliable" in text for text in threshold["warnings"])
    report = {
        "status": "real_model_engineering_smoke_passed",
        "is_deception_evaluation": False,
        "checkpoint": manifests[0]["backend"]["model_id"],
        "resolved_revision": manifests[0]["backend"]["resolved_revision"],
        "training_examples": 14,
        "control_examples": 6,
        "actual_hf_hidden_state_index_verified": layer + 1,
        "causal_prefix_invariance": {
            "passed": True,
            "max_absolute_error": max_error,
            "rtol": 1e-4,
            "atol": 1e-4,
        },
        "uv_lock_sha256": file_sha256("uv.lock"),
        "limitations": [
            "Qwen 0.5B float32 CPU engineering check, not the 8B study.",
            "Two validation instruction examples cannot establish detection performance.",
            "Six neutral controls cannot establish a reliable 1% FPR operating point.",
        ],
    }
    atomic_json(root / "apollo_smoke_validation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
