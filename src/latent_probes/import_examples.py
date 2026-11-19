"""Extract independent probe examples without copying another model's activations.

Provided labels must have a declared meaning. For generated behaviour, leave
labels empty until the actual outputs have been audited.
"""

import json
from pathlib import Path

from .backends import HFBackend
from .experiment import _load_backend, _prepare, _records
from .storage import (
    atomic_json,
    atomic_npz,
    canonical_hash,
    file_sha256,
    read_json,
    utc_now,
    write_jsonl,
)


def load_examples(path, limit=None):
    rows, seen = [], set()
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Line {line_number}: each example must be a JSON object")
        for key in ("example_id", "family_id"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"Line {line_number}: missing {key}")
        if row["example_id"] in seen:
            raise ValueError("Duplicate independent example_id")
        seen.add(row["example_id"])
        if "response" in row:
            raise ValueError(
                "This importer generates own-model responses; fixed external responses require a separate off-policy protocol"
            )
        messages = row.get("messages")
        if (
            not isinstance(messages, list)
            or not messages
            or not isinstance(messages[-1], dict)
            or messages[-1].get("role") != "user"
        ):
            raise ValueError("messages must be a nonempty conversation ending in a user turn")
        if any(
            not isinstance(m, dict)
            or m.get("role") not in {"system", "user", "assistant"}
            or not isinstance(m.get("content"), str)
            or not m["content"].strip()
            for m in messages
        ):
            raise ValueError("Each message must have a valid role and nonempty content")
        label = row.get("label")
        if label is not None:
            if (
                type(label) is not int
                or label not in {0, 1}
                or not isinstance(row.get("label_target"), str)
                or not row["label_target"].strip()
            ):
                raise ValueError(
                    "A provided label must be integer 0/1 with a declared label_target"
                )
        rows.append(row)
    if not rows:
        raise ValueError("Independent example file is empty")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    return rows


def extract_examples(
    config,
    examples_path,
    output_dir,
    limit=None,
    resume=False,
    backend_factory=HFBackend,
    progress=None,
):
    examples = load_examples(examples_path, limit)
    identity = {
        "config_hash": canonical_hash(config.to_dict()),
        "examples_hash": canonical_hash(examples),
    }
    output, manifest = _prepare(output_dir, identity, resume, "examples")
    if not resume:
        atomic_json(output / "config.json", config.to_dict())
        write_jsonl(output / "examples.jsonl", examples)
    else:
        if canonical_hash(load_examples(output / "examples.jsonl")) != identity["examples_hash"]:
            raise ValueError("Independent example snapshot changed")
        if canonical_hash(read_json(output / "config.json")) != identity["config_hash"]:
            raise ValueError("Configuration snapshot changed")
    expected_ids = {
        "e_" + canonical_hash([r["example_id"], seed])[:24]
        for r in examples
        for seed in config.seeds
    }
    existing_ids = {p.stem for p in (output / "sender").glob("*.json")}
    if not existing_ids <= expected_ids:
        raise ValueError("Unexpected cached independent example records")
    if resume and manifest.get("status") == "complete":
        if canonical_hash(_records(output / "sender")) != manifest.get("sender_records_hash"):
            raise ValueError(
                "Completed independent example records changed; refusing to bless edits on resume"
            )
    backend = None
    try:
        backend = _load_backend(config.model, manifest, backend_factory)
        manifest.update(status="running", backend=backend.metadata)
        atomic_json(output / "manifest.json", manifest)
        count = 0
        for example in examples:
            for seed in config.seeds:
                count += 1
                example_id = "e_" + canonical_hash([example["example_id"], seed])[:24]
                record_path = output / "sender" / f"{example_id}.json"
                feature_path = output / "features" / f"{example_id}.npz"
                supervision = {
                    key: example[key]
                    for key in ("label", "label_target", "expected_answer", "notes")
                    if key in example
                }
                if record_path.exists():
                    old = read_json(record_path)
                    identity_fields = {
                        "example_id": example_id,
                        "source_example_id": example["example_id"],
                        "scenario_id": example.get("scenario_id", example["example_id"]),
                        "family_id": example["family_id"],
                        "seed": seed,
                        "messages": example["messages"],
                        "supervision": supervision,
                        "feature_file": feature_path.relative_to(output).as_posix(),
                    }
                    if (
                        any(old.get(k) != v for k, v in identity_fields.items())
                        or old.get("record_sha256")
                        != canonical_hash({k: v for k, v in old.items() if k != "record_sha256"})
                        or old.get("prompt_hash") != canonical_hash(example["messages"])
                        or old.get("feature_sha256") != file_sha256(feature_path)
                    ):
                        raise ValueError("Independent example checkpoint changed")
                    continue
                result = backend.generate(example["messages"], config.sender, seed, capture=True)
                if not result.features:
                    raise ValueError("No activations were captured")
                atomic_npz(feature_path, **result.features)
                record = {
                    "example_id": example_id,
                    "source_example_id": example["example_id"],
                    "scenario_id": example.get("scenario_id", example["example_id"]),
                    "family_id": example["family_id"],
                    "seed": seed,
                    "messages": example["messages"],
                    "prompt_hash": canonical_hash(example["messages"]),
                    "generation": result.record(),
                    "feature_file": feature_path.relative_to(output).as_posix(),
                    "feature_sha256": file_sha256(feature_path),
                    "feature_keys": sorted(result.features),
                    "generation_mode": "natural_independent_task",
                    "supervision": supervision,
                    "is_test_fixture": backend.metadata.get("is_test_fixture", False),
                }
                record["record_sha256"] = canonical_hash(record)
                atomic_json(record_path, record)
                if progress:
                    progress(f"independent example {count}/{len(expected_ids)}")
        rows = _records(output / "sender")
        write_jsonl(output / "senders.jsonl", rows)
        manifest.update(
            status="complete",
            completed_at=utc_now(),
            sender_count=len(rows),
            sender_records_hash=canonical_hash(rows),
        )
        atomic_json(output / "manifest.json", manifest)
        return {
            "output": str(output),
            "example_count": len(rows),
            "is_test_fixture": backend.metadata.get("is_test_fixture", False),
        }
    except Exception as exc:
        manifest.update(status="failed", failure_type=type(exc).__name__, last_updated=utc_now())
        atomic_json(output / "manifest.json", manifest)
        raise
    finally:
        if backend is not None:
            backend.close()
