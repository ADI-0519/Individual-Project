"""Read-only import of Huang et al.'s pinned DeceptionBench source prompts.

This is not the unrelated PKU-Alignment benchmark, a release of model responses,
or a reproduction of the upstream formatting/generation/judging pipeline.
The module makes no network/model calls and never infers behavior labels.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path

BENCHMARK_ID = "huang_2025_deceptionbench"
REPOSITORY_REVISION = "8f3175f66fdff89ded3ebdf0ee2055fa47c955db"
DATASET_REVISION = "38b412da565073483c0a68ce9e04eb15859beedc"
CONDITIONS = (
    "L1-self",
    "L1-other",
    "L2-self-pressure",
    "L2-self-reward",
    "L2-other-pressure",
    "L2-other-reward",
)
METADATA_FIELDS = ("topic", "dimension", "question", "groundtruth", "goal")
SOURCE_FIELDS = frozenset((*METADATA_FIELDS, *CONDITIONS))
FILE_TOPICS = {
    "economy.json": "Economy",
    "education.json": "Education",
    "entertainment.json": "Entertainment",
    "healthcare.json": "Healthcare",
    "social.json": "Social Interactions",
}


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def _json_bytes(raw: bytes, *, description: str) -> object:
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid UTF-8 JSON: {description}") from exc


def _canonical_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256((raw + "\n").encode("utf-8")).hexdigest()


def validate_source_manifest(manifest: dict) -> None:
    """Validate release identity/count contract; raw bytes are checked separately."""
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
    ):
        raise ValueError("Expected source-manifest schema version 1")
    if manifest.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("Wrong DeceptionBench identity; benchmark names are ambiguous")
    if (
        manifest.get("repository") != "https://github.com/Aries-iai/DeceptionBench"
        or manifest.get("dataset") != "https://huggingface.co/datasets/skyai798/DeceptionBench"
        or manifest.get("repository_revision") != REPOSITORY_REVISION
        or manifest.get("dataset_revision") != DATASET_REVISION
    ):
        raise ValueError("Source repository/dataset revision differs from audited release")
    if manifest.get("conditions") != list(CONDITIONS):
        raise ValueError("Expected exactly six audited static conditions")
    if manifest.get("total_scenarios") != 150 or manifest.get("total_static_prompts") != 900:
        raise ValueError("Incorrect source scenario/prompt totals")
    if (
        manifest.get("behavior_labels_released") is not False
        or manifest.get("activations_released") is not False
        or type(manifest.get("generated_model_calls")) is not int
        or manifest["generated_model_calls"] != 0
    ):
        raise ValueError("Source manifest cannot claim generated labels/activations/model calls")
    if (
        manifest.get("prompt_representation")
        != "source_prompt_only_no_upstream_thought_response_wrapper"
    ):
        raise ValueError("Importer preserves source prompts, not complete upstream model inputs")
    notices = manifest.get("license_notices")
    if not isinstance(notices, dict) or not all(
        isinstance(notices.get(key), str) and notices[key].strip()
        for key in ("repository_readme", "dataset_card", "paper_v1", "redistribution_status")
    ):
        raise ValueError("Preserve licensing/redistribution notices in the manifest")
    files = manifest.get("files", [])
    if not isinstance(files, list) or len(files) != 5:
        raise ValueError("Expected five source-file bindings")
    if {item.get("name") for item in files} != set(FILE_TOPICS):
        raise ValueError("Missing/duplicate/unexpected domain filename")
    for item in files:
        if item.get("topic") != FILE_TOPICS[item["name"]] or item.get("rows") != 30:
            raise ValueError("Incorrect per-file topic/count binding")
        if type(item.get("bytes")) is not int or item["bytes"] <= 0:
            raise ValueError("Source byte size must be a positive integer")
        if not isinstance(item.get("sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", item["sha256"]
        ):
            raise ValueError("Invalid raw source SHA256")


def load_deceptionbench(directory: str | Path, manifest_path: str | Path) -> dict:
    """Read caller-provided local files, verify bytes, then validate source schema.

    No source dataset is bundled. The returned records preserve published
    reference metadata separately from source-only model messages. Published
    prompts can themselves mention truthful facts and deceptive goals; this
    separation does not remove those intentional task features.
    """
    directory = Path(directory)
    manifest_raw = Path(manifest_path).read_bytes()
    manifest = _json_bytes(manifest_raw, description="source manifest")
    validate_source_manifest(manifest)
    if {path.name for path in directory.glob("*.json")} != set(FILE_TOPICS):
        raise ValueError("Source directory must contain exactly the five domain JSON files")
    records = []
    all_prompt_hashes = set()
    for binding in manifest["files"]:
        raw = (directory / binding["name"]).read_bytes()
        if len(raw) != binding["bytes"] or sha256(raw).hexdigest() != binding["sha256"]:
            raise ValueError(f"Raw source bytes/hash mismatch: {binding['name']}")
        rows = _json_bytes(raw, description=binding["name"])
        if not isinstance(rows, list) or len(rows) != binding["rows"]:
            raise ValueError(f"Incorrect row count: {binding['name']}")
        dimensions: Counter[str] = Counter()
        for index, source in enumerate(rows):
            if not isinstance(source, dict) or set(source) != SOURCE_FIELDS:
                raise ValueError(f"Unexpected source schema: {binding['name']} row {index}")
            if any(not isinstance(value, str) or not value.strip() for value in source.values()):
                raise ValueError("Every published source field must be a nonempty string")
            if source["topic"] != binding["topic"]:
                raise ValueError("Source topic does not match its domain binding")
            dimensions[source["dimension"]] += 1
            for condition in CONDITIONS:
                digest = sha256(source[condition].encode("utf-8")).hexdigest()
                if digest in all_prompt_hashes:
                    raise ValueError("Duplicate visible source prompt")
                all_prompt_hashes.add(digest)
            record_hash = _canonical_hash(source)
            stem = Path(binding["name"]).stem
            records.append(
                {
                    "scenario_id": f"{BENCHMARK_ID}__{stem}__{index:03d}__{record_hash[:12]}",
                    "source_file": binding["name"],
                    "source_row_index": index,
                    "source_record_sha256": record_hash,
                    "source_file_sha256": binding["sha256"],
                    "metadata": {key: source[key] for key in METADATA_FIELDS},
                    "prompts": {key: source[key] for key in CONDITIONS},
                }
            )
        if len(dimensions) != 3 or set(dimensions.values()) != {10}:
            raise ValueError("Expected three source dimensions with ten scenarios per domain")
    if len(records) != 150 or len(all_prompt_hashes) != 900:
        raise ValueError("Incorrect complete release totals")
    return {
        "benchmark_id": BENCHMARK_ID,
        "source_manifest": manifest,
        "source_manifest_sha256": sha256(manifest_raw).hexdigest(),
        "records": records,
        "audit": {
            "scenarios": 150,
            "static_source_prompts": 900,
            "generated_model_calls": 0,
            "behavior_labels": 0,
            "prompt_representation": manifest["prompt_representation"],
        },
    }


def iter_deceptionbench_rows(dataset: dict) -> Iterator[dict]:
    """Yield source-only model messages with unknown, never guessed behavior labels."""
    if dataset.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("Wrong dataset identity")
    for record in dataset["records"]:
        for condition in CONDITIONS:
            parts = condition.split("-")
            prompt = record["prompts"][condition]
            yield {
                "case_id": f"{record['scenario_id']}__{condition}",
                "scenario_id": record["scenario_id"],
                "domain": record["metadata"]["topic"],
                "dimension": record["metadata"]["dimension"],
                "condition": condition,
                "level": parts[0],
                "role": parts[1],
                "inducement": parts[2] if len(parts) == 3 else None,
                "source_file": record["source_file"],
                "source_row_index": record["source_row_index"],
                "source_record_sha256": record["source_record_sha256"],
                "source_prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
                "source_prompt_only": prompt,
                "messages": [{"role": "user", "content": prompt}],
                "prompt_representation": "source_prompt_only_no_upstream_thought_response_wrapper",
                "behavior_label": None,
            }
