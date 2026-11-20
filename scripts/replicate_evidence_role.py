"""Post-discovery Llama replication of all frozen Qwen evidence-role messages."""

from __future__ import annotations

import argparse
import copy
import traceback
from pathlib import Path

from run_evidence_role_diagnostic import strict_decision, summarise

from latent_probes.backends import HFBackend
from latent_probes.config import GenerationSettings, ModelSettings
from latent_probes.storage import (
    atomic_json,
    canonical_hash,
    environment_info,
    file_sha256,
    read_json,
    utc_now,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
SETTINGS = ModelSettings(
    model_id=MODEL_ID,
    revision=REVISION,
    dtype="bfloat16",
    quantization="4bit",
    device="cuda",
    capture_layers=[8],
    max_input_tokens=2048,
    local_files_only=True,
)
GENERATION = GenerationSettings(max_new_tokens=8, do_sample=False)
SEED = 42


def tree_hashes(directory):
    directory = Path(directory)
    return {
        str(p.relative_to(directory)): file_sha256(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def prepare(source):
    """Read all source conditions without selecting for correctness or value effects."""
    source = Path(source)
    manifest = read_json(source / "manifest.json")
    protocol = read_json(source / "protocol.json")
    rows = read_json(source / "rows.json")
    summary = read_json(source / "summary.json")
    if manifest["status"] != "complete" or not manifest["source_unchanged"]:
        raise ValueError("A completed immutable Qwen discovery batch is required")
    for filename, key in (
        ("protocol.json", "protocol_sha256"),
        ("rows.json", "rows_sha256"),
        ("summary.json", "summary_sha256"),
    ):
        if file_sha256(source / filename) != manifest[key]:
            raise ValueError("Qwen source artifact hash mismatch: " + filename)
    if (
        manifest["backend"].get("is_test_fixture") is not False
        or manifest["backend"]["model_id"] != "Qwen/Qwen2.5-7B-Instruct"
    ):
        raise ValueError("Source must be the genuine original Qwen batch")
    for path, digest in protocol["source_sha256"].items():
        if file_sha256(Path(path)) != digest:
            raise ValueError("A frozen discovery input changed")
    planned = protocol["jobs"]
    if len(planned) != 96 or len(rows) != 96 or len({j["example_id"] for j in planned}) != 96:
        raise ValueError("Replication requires every one of the original 96 conditions")
    first_prefix = planned[0]["messages"][:-1]
    if len(first_prefix) != 19 or first_prefix[0]["role"] != "system":
        raise ValueError("Expected the original system and nine worked-example pairs")
    prefix_hash = canonical_hash(
        {"system": first_prefix[0]["content"], "demonstrations": first_prefix[1:]}
    )
    if prefix_hash != protocol["policy"]["policy_sha256"]:
        raise ValueError("Frozen message prefix does not match the source policy hash")
    for job, row in zip(planned, rows, strict=True):
        if job["messages"][:-1] != first_prefix or any(row[k] != v for k, v in job.items()):
            raise ValueError("Source job order, metadata or frozen messages changed")
        if row["decision"] != strict_decision(row["generation"]):
            raise ValueError("Source decision differs from its raw generation")
        record = source / "records" / f"{job['example_id']}.json"
        if read_json(record) != row:
            raise ValueError("Source combined and per-call records differ")
    if summarise(rows) != summary:
        raise ValueError("Source summary does not reproduce from all 96 outcomes")
    return copy.deepcopy(planned), {
        "source_directory": str(source.resolve()),
        "source_tree_sha256": tree_hashes(source),
        "source_model": manifest["backend"],
        "source_policy_metadata": protocol["policy"],
        "source_protocol_sha256": manifest["protocol_sha256"],
        "source_rows_sha256": manifest["rows_sha256"],
        "source_summary_sha256": manifest["summary_sha256"],
        "reused_text_policy_sha256": prefix_hash,
    }


def run(args, backend_factory=HFBackend):
    source, output = Path(args.source), Path(args.output)
    if output.exists():
        raise FileExistsError("Preserve the one frozen independent-reader replication")
    if source.resolve().is_relative_to(output.resolve()) or output.resolve().is_relative_to(
        source.resolve()
    ):
        raise ValueError("Replication and discovery directories must be separate")
    planned, provenance = prepare(source)
    code_paths = [
        Path(__file__),
        ROOT / "scripts/run_evidence_role_diagnostic.py",
        ROOT / "src/latent_probes/backends.py",
        ROOT / "src/latent_probes/config.py",
        ROOT / "src/latent_probes/reader_panel.py",
        ROOT / "docs/EVIDENCE_ROLE_REPLICATION_PROTOCOL.md",
    ]
    source_hashes = {str(p.resolve()): file_sha256(p) for p in code_paths}
    output.mkdir(parents=True)
    protocol = {
        "schema": "post_discovery_evidence_role_llama_replication_v1",
        "created_utc": utc_now(),
        "selection": "One independent Llama checkpoint after Qwen discovery; all96 original messages, not model or prompt search.",
        "jobs": planned,
        "model": SETTINGS.__dict__,
        "generation": GENERATION.__dict__,
        "seed": SEED,
        "source": provenance,
        "source_sha256": source_hashes,
        "limits": [
            "Post-discovery replication, not an originally preregistered two-model study.",
            "Same16 cases and two authored grammars, not new held-out content.",
            "Qwen metadata identifies the source policy only; Llama is not claimed to inherit reader qualification.",
            "No fitting, new policy, checkpoint search, ct calls or capability-gate change.",
        ],
    }
    atomic_json(output / "protocol.json", protocol)
    manifest = {
        "status": "prepared",
        "protocol_sha256": file_sha256(output / "protocol.json"),
        "environment": environment_info(),
        "new_probe_fits": 0,
        "untouched_transfer_calls": 0,
        "qualifies_reader": False,
    }
    atomic_json(output / "manifest.json", manifest)
    backend = None
    try:
        backend = backend_factory(SETTINGS)
        if (
            backend.metadata.get("is_test_fixture") is not False
            or backend.metadata["model_id"] != MODEL_ID
            or backend.metadata["resolved_revision"] != REVISION
            or backend.metadata["dtype"] != "bfloat16"
            or backend.metadata["quantization"] != "4bit"
        ):
            raise ValueError("The exact genuine Llama NF4/BF16 checkpoint is required")
        manifest.update(status="running", started_utc=utc_now(), backend=backend.metadata)
        atomic_json(output / "manifest.json", manifest)
        lengths = []
        for job in planned:
            encoded = backend.tokenizer.apply_chat_template(
                job["messages"],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            token_ids = encoded["input_ids"][0].tolist()
            lengths.append(
                {
                    "example_id": job["example_id"],
                    "prompt_tokens": len(token_ids),
                    "prompt_token_ids": token_ids,
                    "prompt_sha256": canonical_hash(job["messages"]),
                }
            )
            if len(token_ids) > SETTINGS.max_input_tokens:
                atomic_json(output / "prompt_lengths.json", lengths)
                raise ValueError(
                    "Llama context exceeds2048; no truncation or policy repair allowed"
                )
        atomic_json(output / "prompt_lengths.json", lengths)
        if tree_hashes(source) != provenance["source_tree_sha256"] or any(
            file_sha256(Path(p)) != h for p, h in source_hashes.items()
        ):
            raise ValueError("A frozen input changed before replication generation")
        rows = []
        for i, (job, length) in enumerate(zip(planned, lengths, strict=True), 1):
            generation = backend.generate(job["messages"], GENERATION, SEED).record()
            if generation["prompt_token_ids"] != length["prompt_token_ids"]:
                raise ValueError("Llama generated prompt differs from its exact preflight IDs")
            row = {**job, "generation": generation, "decision": strict_decision(generation)}
            atomic_json(output / "records" / f"{job['example_id']}.json", row)
            rows.append(row)
            if i % 16 == 0:
                print(f"Independent Llama evidence-role replication {i}/96", flush=True)
        atomic_json(output / "rows.json", rows)
        atomic_json(output / "summary.json", summarise(rows))
        if tree_hashes(source) != provenance["source_tree_sha256"] or any(
            file_sha256(Path(p)) != h for p, h in source_hashes.items()
        ):
            raise ValueError("A frozen discovery source or replication input changed")
        manifest.update(
            status="complete",
            completed_utc=utc_now(),
            n_calls=96,
            rows_sha256=file_sha256(output / "rows.json"),
            summary_sha256=file_sha256(output / "summary.json"),
        )
    except BaseException:
        manifest.update(status="failed", failure=traceback.format_exc())
        raise
    finally:
        if backend is not None:
            backend.close()
        manifest.update(
            finished_utc=utc_now(),
            discovery_unchanged=tree_hashes(source) == provenance["source_tree_sha256"],
            source_unchanged=all(file_sha256(Path(p)) == h for p, h in source_hashes.items()),
        )
        atomic_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/adirj/kcl-latent-probes-runs/evidence_role_diagnostic_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/adirj/kcl-latent-probes-runs/evidence_role_llama_replication_v1"),
    )
    run(parser.parse_args())
