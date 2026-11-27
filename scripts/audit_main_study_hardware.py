"""Independently verify engineering captures without importing project modules.

Reads saved arrays and source snapshots only. It never loads model weights or
compares later source edits against an earlier execution's frozen originals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_profile(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    require(manifest["status"] == "complete_engineering_only", "Incomplete profile")
    require(manifest["completed_forwards"] == 1, "Expected one declared engineering forward")
    require(
        not manifest["test_fixture"] and not manifest["research_observation"], "Wrong provenance"
    )
    require(protocol["main_bank_rows_passed_to_model"] == 0, "Main cases entered a profile")
    for name, key in (
        ("protocol.json", "protocol_sha256"),
        ("arrays.npz", "arrays_sha256"),
        ("raw_forward.npz", "raw_forward_sha256"),
    ):
        require(digest(root / name) == manifest[key], f"Changed profile artifact: {name}")
    dims = {"llama": (4096, 128256), "qwen": (3584, 152064)}[protocol["model_key"]]
    with (
        np.load(root / "arrays.npz", allow_pickle=False) as arrays,
        np.load(root / "raw_forward.npz", allow_pickle=False) as raw,
    ):
        require(set(arrays.files) == {"feature", "next_token_logits"}, "Wrong array inventory")
        require(set(raw.files) == {"feature", "full_forward_logits"}, "Wrong raw inventory")
        require(arrays["feature"].shape == (dims[0],), "Wrong activation dimension")
        require(arrays["next_token_logits"].shape == (dims[1],), "Incomplete vocabulary")
        require(
            raw["full_forward_logits"].shape == (1, 1, dims[1]),
            "Wrong raw batch/token/vocabulary axes",
        )
        require(
            np.isfinite(arrays["feature"]).all() and np.isfinite(arrays["next_token_logits"]).all(),
            "Nonfinite observations",
        )
        require(np.array_equal(arrays["feature"], raw["feature"]), "Raw/saved feature differs")
        require(
            np.array_equal(arrays["next_token_logits"], raw["full_forward_logits"][0, 0]),
            "Raw/saved logits differ",
        )
        require(
            canonical(arrays["feature"].tolist()) == manifest["feature_values_sha256"],
            "Feature value hash differs",
        )
        require(
            canonical(arrays["next_token_logits"].tolist()) == manifest["logits_values_sha256"],
            "Logit value hash differs",
        )
    require(canonical(protocol["messages"]) == protocol["messages_sha256"], "Prompt differs")
    require(
        canonical(protocol["prompt_token_ids"]) == protocol["prompt_token_ids_sha256"],
        "Token prefix differs",
    )
    for original, snapshot in protocol["source_snapshots"].items():
        require(
            digest(root / snapshot["snapshot"])
            == snapshot["sha256"]
            == protocol["source_sha256"][original],
            "Frozen source snapshot differs",
        )
    require(manifest["capture"]["hook_alignment_checked"], "Missing actual alignment check")
    return {
        "model": protocol["model_key"],
        "native_directory": str(root),
        "protocol_sha256": digest(root / "protocol.json"),
        "manifest_sha256": digest(root / "manifest.json"),
        "input_tokens": manifest["input_tokens"],
        "feature_dimensions": dims[0],
        "vocabulary_logits": dims[1],
        "source_snapshots_verified": len(protocol["source_snapshots"]),
        "load_seconds": manifest["load_seconds"],
        "cold_forward_and_capture_seconds": manifest["model_forward_seconds"],
        "gpu_after_forward": manifest["gpu_after_forward"],
        "capture": manifest["capture"],
        "raw_array_equal": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve the previous audit; use a fresh output")
    records = [audit_profile(root) for root in args.runs]
    require(len({r["model"] for r in records}) == len(records), "Duplicate model profiles")
    report = {
        "schema": "main-study-independent-engineering-profile-audit-v1",
        "models": records,
        "project_modules_imported": False,
        "new_model_calls": 0,
        "profile_forwards_verified": len(records),
        "main_study_forwards": 0,
        "fits": 0,
        "human_review_complete": False,
        "auditor_source_sha256": digest(__file__),
        "limitation": "Cold one-case engineering checks; no throughput, accuracy or mechanism claim. Frozen source snapshots verified; later review-gate code changes are separate.",
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"profiles_verified": len(records), "new_model_calls": 0}))


if __name__ == "__main__":
    main()
