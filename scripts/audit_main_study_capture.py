"""Verify a completed capture from raw arrays without importing project modules.

This audits recorded measurements, not the validity of fictional belief labels.
No model weights are loaded and no predictions are selected or retried.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reconstruct_output(logits, tokens):
    """Independently reconstruct every summary from the saved float32 vocabulary."""
    require(
        set(tokens) == {"YES", "NO"}
        and all(type(token) is int and 0 <= token < len(logits) for token in tokens.values())
        and tokens["YES"] != tokens["NO"],
        "Invalid answer token IDs",
    )
    values = logits.astype(np.float64)
    yes, no = float(values[tokens["YES"]]), float(values[tokens["NO"]])
    maximum = values.max()
    log_z = float(maximum + np.log(np.exp(values - maximum).sum()))
    return {
        "yes_logit": yes,
        "no_logit": no,
        "yes_logprob": yes - log_z,
        "no_logprob": no - log_z,
        # Subtract after float64 promotion, never round a float32 subtraction.
        "yes_minus_no": yes - no,
        "p_yes_given_yes_no": float(np.exp(-np.logaddexp(0.0, no - yes))),
        "full_vocab_logsumexp": log_z,
        "vocab_size": len(values),
        "argmax_token_id": int(values.argmax()),
        "logits_values_sha256": canonical(logits.tolist()),
    }


def audit_capture(directory):
    root = Path(directory)
    protocol, manifest = read(root / "protocol.json"), read(root / "capture_manifest.json")
    payload = read(root / "rows.json")
    require(protocol["schema"] == "main-study-capture-protocol-v1", "Unsupported protocol schema")
    require(manifest["schema"] == "main-study-capture-manifest-v1", "Unsupported manifest schema")
    require(manifest["status"] == "complete", "Capture is incomplete")
    require(payload["schema"] == "main-study-capture-v1", "Unsupported capture schema")
    require(manifest["protocol_sha256"] == digest(root / "protocol.json"), "Protocol changed")
    for name in ("rows.json", "features.npz"):
        require(manifest["output_sha256"][name] == digest(root / name), f"Changed {name}")
    require(payload["provenance"] == manifest["provenance"], "Export provenance differs")
    require(payload["test_fixture"] == protocol["test_fixture"], "Fixture identity differs")
    require(manifest["test_fixture"] == protocol["test_fixture"], "Manifest fixture differs")
    partition = manifest["partition"]
    require(partition in {"train_dev", "test"}, "Unknown capture partition")
    provenance = payload["provenance"]
    for key in ("bank_sha256", "model_settings", "capture_layer"):
        require(provenance[key] == protocol[key], f"Protocol provenance differs: {key}")
    require(provenance["partition"] == partition, "Provenance partition differs")
    require(
        provenance["protocol_sha256"] == manifest["protocol_sha256"],
        "Provenance protocol differs",
    )
    require(
        provenance["source_bindings"] == manifest["authorization"]["sha256"],
        "Provenance source bindings differ",
    )
    backend = manifest["backend"]
    for key in ("model_id", "dtype", "quantization", "capture_layers"):
        require(backend[key] == protocol["model_settings"][key], f"Backend metadata differs: {key}")
    require(
        backend["resolved_revision"] == protocol["model_settings"]["revision"]
        and backend["hidden_size"] == protocol["hidden_size"]
        and backend["decoder_blocks"] == protocol["decoder_blocks"]
        and backend["is_test_fixture"] == protocol["test_fixture"],
        "Backend identity differs",
    )
    require(
        all(row["split"] in {"train", "development", "test"} for row in protocol["planned_rows"]),
        "Unknown planned split",
    )
    require(
        len({row["row_id"] for row in protocol["planned_rows"]}) == len(protocol["planned_rows"]),
        "Duplicate planned identity",
    )
    planned = [
        row for row in protocol["planned_rows"] if (row["split"] == "test") == (partition == "test")
    ]
    rows = payload["rows"]
    require(rows, "Empty capture partition")
    require(len(rows) == len(planned) == manifest["completed_rows"], "Row count differs")
    require(
        [row["row_id"] for row in rows] == [row["row_id"] for row in planned],
        "Rows are missing, reordered or duplicated",
    )
    expected_ids = {row["row_id"] for row in rows}
    require(len(expected_ids) == len(rows), "Duplicate row identity")
    require(
        {path.name for path in (root / "row_journal").iterdir()} == expected_ids
        and all(path.is_dir() for path in (root / "row_journal").iterdir()),
        "Unexpected or missing journal entries",
    )
    hidden, vocab = protocol["hidden_size"], protocol["vocab_size"]
    require(type(hidden) is int and hidden > 0, "Invalid hidden size")
    require(type(vocab) is int and vocab > 1, "Invalid vocabulary size")
    with np.load(root / "features.npz", allow_pickle=False) as exported:
        require(set(exported.files) == {"features", "row_ids"}, "Unexpected export arrays")
        features = exported["features"]
        require(features.dtype == np.float32, "Export dtype differs")
        require(features.shape == (len(rows), hidden), "Export dimensions differ")
        require(exported["row_ids"].tolist() == [row["row_id"] for row in rows], "IDs differ")
        times, groups = [], set()
        for index, (row, plan) in enumerate(zip(rows, planned, strict=True)):
            item = root / "row_journal" / row["row_id"]
            require(not (item / "failure.json").exists(), "Failed row in complete capture")
            recorded, started = read(item / "complete.json"), read(item / "started.json")
            require(recorded["status"] == "complete", "Incomplete journal row")
            require(started["row_id"] == row["row_id"], "Started identity differs")
            require(
                started["protocol_sha256"]
                == manifest["protocol_sha256"]
                == recorded["protocol_sha256"],
                "Journal protocol differs",
            )
            require(
                plan.keys() <= recorded.keys()
                and all(recorded[key] == value for key, value in plan.items()),
                "Plan differs",
            )
            require(
                row.keys() == recorded.keys() | {"output_yes_no_logitdiff"}
                and all(row[key] == value for key, value in recorded.items()),
                "Export differs",
            )
            require(row["test_fixture"] == protocol["test_fixture"], "Row fixture differs")
            require(canonical(row["messages"]) == row["messages_sha256"], "Prompt hash differs")
            require(
                canonical(row["prompt_token_ids"]) == row["prompt_token_ids_sha256"],
                "Token hash differs",
            )
            token_ids = row["prompt_token_ids"]
            require(
                isinstance(token_ids, list)
                and token_ids
                and all(type(token) is int and token >= 0 for token in token_ids),
                "Invalid prompt token IDs",
            )
            require(
                len(token_ids) <= min(2048, protocol["model_settings"]["max_input_tokens"]),
                "Oversized prompt",
            )
            require(digest(item / "arrays.npz") == row["arrays_sha256"], "Saved arrays changed")
            require(
                digest(item / "raw_forward.npz") == row["raw_forward_sha256"],
                "Raw arrays changed",
            )
            with (
                np.load(item / "arrays.npz", allow_pickle=False) as saved,
                np.load(item / "raw_forward.npz", allow_pickle=False) as raw,
            ):
                require(
                    set(saved.files) == {"feature", "next_token_logits"},
                    "Unexpected saved arrays",
                )
                require(
                    set(raw.files) == {"feature", "full_forward_logits"}, "Unexpected raw arrays"
                )
                vector, logits = saved["feature"], saved["next_token_logits"]
                require(vector.shape == (hidden,) and logits.shape == (vocab,), "Wrong shape")
                require(vector.dtype == logits.dtype == np.float32, "Wrong saved dtype")
                require(np.isfinite(vector).all() and np.isfinite(logits).all(), "Nonfinite row")
                require(
                    raw["feature"].dtype == raw["full_forward_logits"].dtype == np.float32,
                    "Wrong raw dtype",
                )
                require(np.array_equal(vector, raw["feature"]), "Raw feature differs")
                require(raw["full_forward_logits"].shape == (1, 1, vocab), "Raw axes differ")
                require(
                    np.array_equal(logits, raw["full_forward_logits"][0, 0]),
                    "Raw logits differ",
                )
                require(np.array_equal(vector, features[index]), "Exported feature differs")
                require(canonical(vector.tolist()) == row["feature_values_sha256"], "Values differ")
                summary = reconstruct_output(logits, protocol["tokenizer"]["answer_token_ids"])
                require(summary == row["output_logits"], "Output summary differs")
                difference = summary["yes_minus_no"]
                require(difference == row["output_yes_no_logitdiff"], "Output margin differs")
            capture = row["capture"]
            require(capture["answer_tokens_supplied"] == 0, "Answer tokens supplied")
            require(capture["same_forward_as_logits"] is True, "Separate forward")
            require(capture["selected_position"] == len(row["prompt_token_ids"]) - 1, "Position")
            require(capture["selected_token_id"] == row["prompt_token_ids"][-1], "Token differs")
            require(capture["layer"] == protocol["capture_layer"], "Layer differs")
            require(capture["hf_hidden_state_index"] == protocol["capture_layer"] + 1, "HF index")
            if index == 0:
                require(capture["hook_alignment_checked"] is True, "Missing alignment check")
            elapsed = row["elapsed_seconds"]
            require(
                type(elapsed) in {int, float} and np.isfinite(elapsed) and elapsed >= 0,
                "Invalid elapsed time",
            )
            times.append(elapsed)
            groups.add(row["group_id"])
    return {
        "directory": str(root),
        "model": protocol["model_key"],
        "partition": partition,
        "rows_verified": len(rows),
        "groups_verified": len(groups),
        "test_fixture": protocol["test_fixture"],
        "provenance": payload["provenance"],
        "raw_export_array_identity": True,
        "full_vocabulary_summary_reconstructed": True,
        "protocol_journal_export_identity": True,
        "feature_dimensions": hidden,
        "vocabulary_logits": vocab,
        "recorded_forward_seconds_sum": float(sum(times)),
        "capture_manifest_sha256": digest(root / "capture_manifest.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve the prior audit; use a fresh output path")
    report = {
        "schema": "main-study-independent-capture-audit-v1",
        "capture": audit_capture(args.capture),
        "auditor_source_sha256": digest(__file__),
        "project_modules_imported": any(
            name == "latent_probes" or name.startswith("latent_probes.") for name in sys.modules
        ),
        "new_model_calls": 0,
        "scope": "Saved measurement integrity; not human semantic validation or a result analysis",
        "limitations": [
            "Checks saved arrays and recorded metadata; does not rerun tokenization or a model.",
            "Checks internal hash consistency, not authenticity against an external trusted digest.",
            "Same-forward and hook-alignment claims are recorded assertions; raw alignment references are not saved.",
            "Does not validate semantic labels, authorization evidence, or current external source files.",
        ],
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"rows_verified": report["capture"]["rows_verified"]}))


if __name__ == "__main__":
    main()
