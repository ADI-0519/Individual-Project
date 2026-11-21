"""Audit saved posthoc YES/NO scores without models, fitting or threshold selection."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in root.rglob("*") if p.is_file()}


def canonical(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode()).hexdigest()


def metrics(rows):
    def auc(field):
        pairs = [
            (r["gold"], r["probe"]["margin"] if field == "probe" else r["yes_minus_no"])
            for r in rows
        ]
        positive = [s for y, s in pairs if y]
        negative = [s for y, s in pairs if not y]
        return sum((a > b) + 0.5 * (a == b) for a in positive for b in negative) / (
            len(positive) * len(negative)
        )

    return {
        "conditions": len(rows),
        "scene_units": len({r["scene_id"] for r in rows}),
        "probe_correct": sum(r["probe"]["prediction"] == r["gold"] for r in rows),
        "output_fixed_correct": sum(int(r["yes_minus_no"] > 0) == r["gold"] for r in rows),
        "probe_auroc": auc("probe"),
        "output_score_auroc": auc("output"),
        "p_yes_range": [
            min(r["p_yes_given_yes_no"] for r in rows),
            max(r["p_yes_given_yes_no"] for r in rows),
        ],
    }


def analyse(run, source, native=None):
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    protocol, manifest = read(run / "protocol.json"), read(run / "manifest.json")
    old_manifest = read(source / "manifest.json")
    if (
        protocol.get("schema") != "future_goal_answer_logits_posthoc_v1"
        or old_manifest.get("status") != "neutral_failed_stopped"
        or read(source / "qualification.json")["passed"]
    ):
        raise ValueError("Need the stopped original gate and its posthoc diagnostic")
    if (
        manifest["status"] != "complete_posthoc_development"
        or manifest["n_rows"] != 32
        or manifest["generated_answers"] != 0
    ):
        raise ValueError("Need the complete 32-forward, zero-generation diagnostic")
    local = files(run)
    if set(local) != set(manifest["output_sha256"]) | {"manifest.json"} or any(
        local[k] != v for k, v in manifest["output_sha256"].items()
    ):
        raise ValueError("Diagnostic output ledger differs")
    if (
        sha(run / "protocol.json") != manifest["protocol_sha256"]
        or files(source) != protocol["source_sha256"]
    ):
        raise ValueError("Protocol or original neutral evidence differs")
    expected = {r["id"]: r for r in protocol["planned_rows"]}
    rows = [read(p) for p in sorted((run / "row_outputs").glob("*.json"))]
    if (
        len(rows) != 32
        or len({r["id"] for r in rows}) != 32
        or {r["id"] for r in rows} != set(expected)
    ):
        raise ValueError("Need every exact unique neutral prefix")
    errors = {"logit_difference": 0.0, "logprob_identity": 0.0, "conditional_probability": 0.0}
    for row in rows:
        scalar_fields = (
            "yes_logit",
            "no_logit",
            "yes_logprob",
            "no_logprob",
            "yes_minus_no",
            "p_yes_given_yes_no",
            "full_vocab_logsumexp",
        )
        if not all(math.isfinite(row[k]) for k in scalar_fields):
            raise ValueError("Nonfinite saved output score")
        original = read(source / "row_outputs" / (row["id"] + ".json"))
        if (
            row["status"] != "complete"
            or row["feature_exact"] is not True
            or any(row[k] != v for k, v in expected[row["id"]].items())
        ):
            raise ValueError("Saved row differs from exact planned prefix")
        if (
            row["probe"] != original["probe"]
            or row["messages"] != original["messages"]
            or row["gold"] != original["gold"]
            or row["prompt_token_ids"] != original["prompt_token_ids"]
        ):
            raise ValueError("Posthoc row differs from original neutral observation")
        feature_path = source / original["capture"]["feature_path"]
        if sha(feature_path) != original["capture"]["feature_sha256"]:
            raise ValueError("Original feature bytes differ")
        with np.load(feature_path, allow_pickle=False) as archive:
            feature = archive["layer_16"]
        if canonical(feature.tolist()) != row["feature_values_sha256"]:
            raise ValueError("Exact same-forward feature binding differs")
        errors["logit_difference"] = max(
            errors["logit_difference"],
            abs(row["yes_minus_no"] - (row["yes_logit"] - row["no_logit"])),
        )
        errors["logprob_identity"] = max(
            errors["logprob_identity"],
            abs(row["yes_logprob"] - (row["yes_logit"] - row["full_vocab_logsumexp"])),
            abs(row["no_logprob"] - (row["no_logit"] - row["full_vocab_logsumexp"])),
        )
        errors["conditional_probability"] = max(
            errors["conditional_probability"],
            abs(row["p_yes_given_yes_no"] - 1 / (1 + math.exp(-row["yes_minus_no"]))),
        )
        if (
            row["argmax_token_id"] != original["generation"]["generated_token_ids"][0]
            or row["argmax_token_id"] != protocol["tokens"]["NO"]
        ):
            raise ValueError("Recorded argmax differs from original greedy NO")
    if any(error > 1e-12 for error in errors.values()):
        raise ValueError("Saved scalar score identity differs")
    if native is not None and files(native) != local:
        raise ValueError("Native/mirror files differ")
    keys = (
        "id",
        "scene_id",
        "channel",
        "wording_group",
        "mapping_id",
        "gold",
        "probe",
        "yes_logit",
        "no_logit",
        "yes_logprob",
        "no_logprob",
        "yes_minus_no",
        "p_yes_given_yes_no",
        "full_vocab_logsumexp",
        "argmax_token_id",
        "feature_values_sha256",
    )
    return {
        "schema": "future_goal_output_head_posthoc_analysis_v1",
        "status": "complete_posthoc_development",
        "scope": "Same 32 neutral prefixes, 16 paired scene units; no gate rescue or goal comparison.",
        "execution_commit": "20d4101",
        "run_directory": str(run),
        "source_protocol_sha256": sha(run / "protocol.json"),
        "source_manifest_sha256": sha(run / "manifest.json"),
        "old_neutral_gate_passed": False,
        "goal_archive_conditions_collected": 0,
        "new_generated_answers": 0,
        "full_model_forward_conditions": 32,
        "overall": metrics(rows),
        "per_channel": {
            c: metrics([r for r in rows if r["channel"] == c]) for c in ("observation", "message")
        },
        "tokens": protocol["tokens"],
        "all_block16_vectors_exact": True,
        "all_argmax_matches_old_no": True,
        "identity_max_errors": errors,
        "mirror": {
            "files": len(local),
            "bytes": sum(p.stat().st_size for p in run.rglob("*") if p.is_file()),
            "native_mirror_checked": native is not None,
        },
        "human_review_complete": False,
        "limits": [
            "Posthoc development comparison on authored scenarios; no reasoning-family holdout.",
            "16 scenes each appear under two paired name mappings.",
            "No threshold calibration, polarity flip or fitting.",
            "YES/NO conditional token probability is not calibrated belief confidence.",
            "Full-vocabulary logits were not saved; logZ/argmax cannot independently be reconstructed, although scalar identities and old greedy NO agree.",
        ],
        "records": [{k: row[k] for k in keys} for row in rows],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--native-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(args.run.resolve())
        or args.output.resolve().is_relative_to(args.source_run.resolve())
    ):
        parser.error("Use a new report outside the immutable run directories")
    result = analyse(args.run, args.source_run, args.native_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "overall", "per_channel", "identity_max_errors", "mirror")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
