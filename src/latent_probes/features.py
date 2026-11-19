"""Explicit feature packing; no inference of deception labels from objectives."""

import csv
import json
from pathlib import Path

import numpy as np

from .storage import atomic_json, atomic_npz, file_sha256, read_json


def feature_spec(backend, layer, position):
    fields = (
        "backend",
        "is_test_fixture",
        "model_id",
        "resolved_revision",
        "dtype",
        "quantization",
        "chat_template_sha256",
        "extraction",
        "attention_implementation",
    )
    return {
        **{key: backend.get(key) for key in fields},
        "decoder_block_index": layer,
        "position": position,
        "position_definition": backend.get("positions", {}).get(position),
    }


def pack_features(run_dir, output_dir, layer, position="response_last", annotations_path=None):
    if position not in {"prompt_last", "response_last"}:
        raise ValueError("position must be prompt_last or response_last")
    run_dir, output = Path(run_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError("Feature export already exists; choose a new directory")
    manifest = read_json(run_dir / "manifest.json")
    if manifest.get("status") != "complete" or manifest.get("stage") not in {"sender", "examples"}:
        raise ValueError("Pack features from a completed sender or independent-example run")
    rows = [read_json(p) for p in sorted((run_dir / "sender").glob("*.json"))]
    if not rows or len({r["example_id"] for r in rows}) != len(rows):
        raise ValueError("No records or duplicate record identifiers")
    if len(rows) != manifest.get("sender_count"):
        raise ValueError("Record count does not match manifest")
    from .storage import canonical_hash

    if manifest.get("sender_records_hash") != canonical_hash(rows):
        raise ValueError("Saved records changed after run completion")
    annotations = {}
    if annotations_path is not None:
        with Path(annotations_path).open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                key = row.get("example_id")
                if not key or key in annotations:
                    raise ValueError("Annotation example_id is missing or duplicated")
                annotations[key] = row
        if set(annotations) != {r["example_id"] for r in rows}:
            raise ValueError("Annotation IDs must match the feature examples exactly")
    key = f"layer_{layer}_{position}"
    vectors, metadata = [], []
    for row in rows:
        path = (run_dir / row["feature_file"]).resolve()
        if not path.is_relative_to(run_dir.resolve()):
            raise ValueError("Feature path escapes its run directory")
        if file_sha256(path) != row.get("feature_sha256"):
            raise ValueError(f"Feature checksum mismatch for {row['example_id']}")
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive:
                raise ValueError(
                    f"Feature {key} was not captured; available: {list(archive.files)}"
                )
            vector = archive[key]
            if vector.ndim != 1 or vector.dtype.kind not in "fi" or not np.isfinite(vector).all():
                raise ValueError("Each activation must be one finite numeric vector")
            vectors.append(vector.copy())
        if len({v.shape for v in vectors}) > 1:
            raise ValueError("Activation dimensions differ across examples")
        messages = row["messages"]
        output_text = row["generation"]["text"]
        full_context = json.dumps(
            messages + [{"role": "assistant", "content": output_text}], ensure_ascii=False
        )
        objective = row.get("objective")
        supervision = row.get("supervision", {})
        annotation = annotations.get(row["example_id"], {})
        metadata.append(
            {
                "example_id": row["example_id"],
                "scenario_id": row["scenario_id"],
                "family_id": row["family_id"],
                "objective": objective or "",
                "assumed_knowledge": row.get("assumed_knowledge", ""),
                "objective_pressure": int(objective == "favourable")
                if objective is not None
                else "",
                "label": annotation.get("label", supervision.get("label", "")),
                "label_target": annotation.get(
                    "label_target", supervision.get("label_target", "manual_definition_required")
                ),
                "factuality": annotation.get("factuality", ""),
                "critical_fact_disclosed": annotation.get("critical_fact_disclosed", ""),
                "ambiguous": annotation.get("ambiguous", ""),
                "text": json.dumps(messages, ensure_ascii=False)
                if position == "prompt_last"
                else full_context,
                "prompt_text": json.dumps(messages, ensure_ascii=False),
                "output_text": output_text,
            }
        )
    spec = feature_spec(manifest["backend"], layer, position)
    signature = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    output.mkdir(parents=True)
    atomic_npz(
        output / "features.npz",
        X=np.stack(vectors).astype(np.float32),
        example_ids=np.asarray([r["example_id"] for r in rows]),
        feature_signature=np.asarray(signature),
    )
    with (output / "metadata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    result = {
        "examples": len(rows),
        "dimension": len(vectors[0]),
        "feature_spec": spec,
        "source_run_identity": manifest["identity"],
        "source_records_hash": manifest["sender_records_hash"],
        "annotations_sha256": file_sha256(annotations_path) if annotations_path else None,
        "text_baseline_information": (
            "full prompt; no generated response"
            if position == "prompt_last"
            else "full prompt including source and objective, plus generated response"
        ),
        "label_warning": "objective_pressure is an instruction label; label is not inferred from it",
        "output": str(output),
    }
    atomic_json(output / "manifest.json", result)
    return result
