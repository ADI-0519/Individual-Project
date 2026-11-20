"""Bounded natural-report feasibility, with immutable stages and no probe fit.

Production loads only the two pinned offline checkpoints. Test factories remain
explicitly marked as fixtures. Neither stopped recipient pipeline is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import subprocess
import time
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from latent_probes.backends import HFBackend
from latent_probes.calibrated_reader import policy_metadata
from latent_probes.config import GenerationSettings, ModelSettings
from latent_probes.natural_selection_panel import (
    actor_jobs,
    load_panel,
    panel_metadata,
    reader_plan,
    source_text,
)
from latent_probes.reader_panel import parse_decision
from latent_probes.storage import (
    atomic_json,
    atomic_npz,
    canonical_hash,
    environment_info,
    file_sha256,
    read_json,
    utc_now,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = ROOT / "data/communication/natural_selection_pilot_draft_v1.json"
SEED = 42
ACTOR_SETTINGS = ModelSettings(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    revision="0e9e39f249a16976918f6564b8830bc894c89659",
    dtype="bfloat16",
    quantization="4bit",
    device="cuda",
    capture_layers=[8],
    max_input_tokens=4096,
    local_files_only=True,
)
READER_SETTINGS = ModelSettings(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    revision="a09a35458c702b33eeacc393d103063234e8bc28",
    dtype="bfloat16",
    quantization="4bit",
    device="cuda",
    capture_layers=[8],
    max_input_tokens=4096,
    local_files_only=True,
)
REPORT_GENERATION = GenerationSettings(max_new_tokens=192, do_sample=False)
READER_GENERATION = GenerationSettings(max_new_tokens=8, do_sample=False)
DECODE = {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
CLASSES = ("APPROVE", "REJECT", "DEFER")


def load_tokenizer(settings):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        settings.model_id,
        revision=settings.revision,
        local_files_only=True,
        trust_remote_code=False,
    )


def tokenizer_identity(tokenizer):
    if not tokenizer.chat_template:
        raise ValueError("A pinned instruct chat template is required")
    return {
        "chat_template_sha256": canonical_hash(tokenizer.chat_template),
        "vocabulary_sha256": canonical_hash(tokenizer.get_vocab()),
        "special_ids": list(tokenizer.all_special_ids),
    }


def encode(tokenizer, messages, maximum):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids):
        raise ValueError("Expected a nonempty list of exact prompt token IDs")
    if len(ids) > maximum:
        raise ValueError(f"Prompt has {len(ids)} tokens > {maximum}; no truncation permitted")
    return ids


def source_paths(panel_path):
    return [
        Path(__file__),
        Path(panel_path),
        *[
            ROOT / "src/latent_probes" / name
            for name in (
                "natural_selection_panel.py",
                "backends.py",
                "calibrated_reader.py",
                "forecast_panel.py",
                "reader_panel.py",
                "config.py",
                "storage.py",
            )
        ],
        ROOT / "docs/NEXT_MAIN_STUDY_PROTOCOL.md",
        ROOT / "docs/NATURAL_SELECTION_RUN_PROTOCOL.md",
    ]


def sources_unchanged(protocol):
    try:
        return all(
            file_sha256(path) == digest for path, digest in protocol["source_sha256"].items()
        )
    except OSError:
        return False


def assert_sources(protocol):
    if not sources_unchanged(protocol):
        raise ValueError("Frozen data/code/protocol source changed; preserve this attempt")


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_saved_file(output, manifest, name, *, allow_absent=True):
    """A deleted/emptied completed journal must never cause regeneration."""
    path = output / f"{name}.json"
    digest = manifest.get(f"{name}_sha256")
    if digest is not None:
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"Saved {name} was changed or removed")
    elif path.exists() or not allow_absent:
        raise ValueError(f"Unregistered or missing {name} journal")


def save_journal(output, manifest, name, content):
    atomic_json(output / f"{name}.json", content)
    manifest[f"{name}_sha256"] = file_sha256(output / f"{name}.json")
    atomic_json(output / "manifest.json", manifest)


def prepare(output, panel_path=DEFAULT_PANEL, *, tokenizer_factory=load_tokenizer):
    """Freeze known inputs without loading model weights or generating answers."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preparation never overwrites an existing attempt")
    panel = load_panel(panel_path)
    jobs = actor_jobs(panel)
    skeleton = reader_plan(panel, {j["example_id"]: "" for j in jobs})
    hashes = {str(path.resolve()): file_sha256(path) for path in source_paths(panel_path)}
    protocol = {
        "schema": "natural_selection_execution_v1",
        "created_utc": utc_now(),
        "panel_path": str(Path(panel_path).resolve()),
        "panel": panel_metadata(panel),
        "policy": policy_metadata(),
        "actor_model": asdict(ACTOR_SETTINGS),
        "reader_model": asdict(READER_SETTINGS),
        "report_generation": asdict(REPORT_GENERATION),
        "reader_generation": asdict(READER_GENERATION),
        "seed": SEED,
        "decode": DECODE,
        "actor_jobs": jobs,
        "source_sha256": hashes,
        "is_test_fixture": tokenizer_factory is not load_tokenizer,
        "n_reader_conditions": len(skeleton),
        "actual_reader_plan_requires_saved_reports": True,
        "new_probe_fits": 0,
        "old_qualification_remains_failed": True,
        "untouched_transfer_calls": 0,
    }
    output.mkdir(parents=True)
    manifest = {"status": "preparing", "environment": environment_info(), "created_utc": utc_now()}
    atomic_json(output / "manifest.json", manifest)
    try:
        actor_tokenizer = tokenizer_factory(ACTOR_SETTINGS)
        reader_tokenizer = tokenizer_factory(READER_SETTINGS)
        preflight = {
            "actor_tokenizer": tokenizer_identity(actor_tokenizer),
            "reader_tokenizer": tokenizer_identity(reader_tokenizer),
            "actor_inputs": {
                j["example_id"]: encode(
                    actor_tokenizer, j["messages"], ACTOR_SETTINGS.max_input_tokens
                )
                for j in jobs
            },
            "reader_skeleton_lengths": [
                len(encode(reader_tokenizer, j["messages"], READER_SETTINGS.max_input_tokens))
                for j in skeleton
            ],
            "scope": "Actor and known controls exact; all actual reader inputs checked after reports.",
        }
        assert_sources(protocol)
        atomic_json(output / "protocol.json", protocol)
        atomic_json(output / "preflight.json", preflight)
        manifest.update(
            status="prepared",
            protocol_sha256=file_sha256(output / "protocol.json"),
            preflight_sha256=file_sha256(output / "preflight.json"),
            is_test_fixture=protocol["is_test_fixture"],
            generation_attempts={"actor": 0, "reader": 0},
        )
    except BaseException:
        manifest.update(status="failed_preparation", failure=traceback.format_exc())
        raise
    finally:
        atomic_json(output / "manifest.json", manifest)
    return manifest


def verify_attempt(output):
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    protocol = read_json(output / "protocol.json")
    preflight = read_json(output / "preflight.json")
    for name in ("protocol", "preflight"):
        if file_sha256(output / f"{name}.json") != manifest[f"{name}_sha256"]:
            raise ValueError(f"Frozen {name} was changed")
    assert_sources(protocol)
    settings = (
        ("actor_model", asdict(ACTOR_SETTINGS)),
        ("reader_model", asdict(READER_SETTINGS)),
        ("report_generation", asdict(REPORT_GENERATION)),
        ("reader_generation", asdict(READER_GENERATION)),
        ("policy", policy_metadata()),
        ("decode", DECODE),
        ("seed", SEED),
    )
    if any(protocol[key] != expected for key, expected in settings):
        raise ValueError("Frozen execution settings no longer match this runner")
    panel = load_panel(protocol["panel_path"])
    if protocol["actor_jobs"] != actor_jobs(panel) or protocol["panel"] != panel_metadata(panel):
        raise ValueError("Frozen actor plan or source bank differs")
    return manifest, protocol, preflight, panel


def check_backend(backend, settings, fixture):
    expected = {
        "is_test_fixture": fixture,
        "model_id": settings.model_id,
        "resolved_revision": settings.revision,
        "dtype": settings.dtype,
        "quantization": settings.quantization,
        "capture_layers": settings.capture_layers,
    }
    if any(backend.metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Backend does not match the frozen checkpoint/runtime or fixture identity")
    environment = environment_info()
    runtime = {key: environment[key] for key in ("python", "platform", "packages")}
    if fixture:
        runtime["hardware"] = "explicit_cpu_test_fixture"
    else:
        torch = backend.torch
        runtime["torch_cuda_build"] = torch.version.cuda
        runtime["gpus"] = [
            {
                "name": torch.cuda.get_device_name(i),
                "capability": list(torch.cuda.get_device_capability(i)),
                "uuid": str(getattr(torch.cuda.get_device_properties(i), "uuid", "unavailable")),
            }
            for i in range(torch.cuda.device_count())
        ]
        try:
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=uuid,driver_version,name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            runtime["gpu_driver"] = driver.stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            runtime["gpu_driver"] = f"unavailable: {type(exc).__name__}"
        runtime["numerics"] = {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        }
    backend.metadata = {**backend.metadata, "pilot_runtime_identity": runtime}
    return canonical_hash({"backend": backend.metadata, "decode": DECODE})


def decoded_record(backend, generation):
    """Preserve generated wire content independently of the backend's stripped text."""
    record = generation.record()
    ids = record["generated_token_ids"]
    text = backend.tokenizer.decode(ids, **DECODE)
    special = set(backend.tokenizer.all_special_ids)
    defaults = backend.metadata.get("generation_defaults", {})
    eos = defaults.get("eos_token_id", backend.tokenizer.eos_token_id)
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    removed = [{"index": i, "id": token} for i, token in enumerate(ids) if token in special]
    record.update(
        raw_decoded_text=text,
        raw_text_sha256=text_sha256(text),
        decoded_including_special_tokens=backend.tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ),
        decode_kwargs=dict(DECODE),
        removed_special_tokens=removed,
        unexpected_special_tokens=[
            item for item in removed if item["index"] != len(ids) - 1 or item["id"] not in eos_ids
        ],
    )
    return record


def capture_saved(backend, generation):
    """Capture one saved sequence; an empty report never triggers regeneration."""
    prompt_ids = generation["prompt_token_ids"]
    response_ids = generation["generated_token_ids"]
    special = set(backend.tokenizer.all_special_ids)
    positions = [
        len(prompt_ids) + i for i, token in enumerate(response_ids) if token not in special
    ]
    if not positions:
        return {}, {"status": "unavailable_no_content", "selected_positions": [], "seconds": 0.0}
    sequence = backend.torch.tensor([prompt_ids + response_ids], device=backend.input_device)
    started = time.perf_counter()
    arrays = backend._capture_token_rows(sequence, positions)
    if backend.torch.cuda.is_available():
        backend.torch.cuda.synchronize()
    expected = {f"layer_{layer}_tokens" for layer in ACTOR_SETTINGS.capture_layers}
    dimension = backend.metadata["hidden_size"]
    if set(arrays) != expected or any(
        value.shape != (len(positions), dimension) or not np.isfinite(value).all()
        for value in arrays.values()
    ):
        raise ValueError("Saved-sequence activation shape/coordinates are invalid")
    return arrays, {
        "status": "complete",
        "selected_positions": positions,
        "selected_ids": [token for token in response_ids if token not in special],
        "sequence_sha256": canonical_hash(prompt_ids + response_ids),
        "seconds": time.perf_counter() - started,
        "scope": "Original natural sender states; no later private recipient packet was available.",
    }


def check_sender_record(row, job, tokenizer, prompt_ids, output):
    if row["job"] != job or row["generation"]["prompt_token_ids"] != prompt_ids:
        raise ValueError("Saved report does not bind to the frozen actor input")
    text = tokenizer.decode(row["generation"]["generated_token_ids"], **DECODE)
    if row["generation_sha256"] != canonical_hash(row["generation"]):
        raise ValueError("Saved actor generation metadata/IDs changed")
    if row["report_text"] != text or row["report_sha256"] != text_sha256(text):
        raise ValueError("Saved report text/IDs/hash differ")
    if row["word_count"] != len(text.split()):
        raise ValueError("Saved report word count changed")
    check_features(row, output)


def check_features(row, output):
    if row["capture"]["status"] == "complete":
        if file_sha256(output / row["feature_file"]) != row["feature_sha256"]:
            raise ValueError("Saved actor features changed")
    elif row["capture"]["status"] not in {"pending", "unavailable_no_content"}:
        raise ValueError("Unknown actor feature disposition")


def _write_blind_review(output, panel, sender_rows):
    cases = {case["id"]: case for case in panel["cases"]}
    shuffled = list(sender_rows)
    random.Random(SEED).shuffle(shuffled)
    blind, mapping = [], {}
    for i, row in enumerate(shuffled, start=1):
        case = cases[row["job"]["case_id"]]
        review_id = f"review_{i:03d}"
        blind.append(
            {
                "review_id": review_id,
                "rule": case["reader_visible_decision_rule"],
                "source": source_text(case),
                "report": row["report_text"],
            }
        )
        mapping[review_id] = row["job"]["example_id"]
    atomic_json(output / "blind_reports.json", blind)
    atomic_json(output / "private_review_mapping.json", mapping)


def execute(output, *, stage="all", resume=False, backend_factory=HFBackend):
    """Run frozen reports/readers; explicit resume reuses completed exact calls."""
    output = Path(output)
    manifest, protocol, preflight, panel = verify_attempt(output)
    if stage not in {"all", "reports", "readers"}:
        raise ValueError("Unknown execution stage")
    fixture = backend_factory is not HFBackend
    if fixture != protocol["is_test_fixture"]:
        raise ValueError("Fixture preparation cannot be presented as a real model run")
    if manifest["status"] == "complete":
        raise FileExistsError("Completed attempt is immutable")
    if manifest["status"] not in {"prepared", "reports_complete"} and not resume:
        raise ValueError("Interrupted attempt requires explicit --resume with unchanged inputs")
    check_saved_file(output, manifest, "sender_rows")
    check_saved_file(output, manifest, "reader_cache")
    check_saved_file(output, manifest, "reader_plan")
    rows = read_json(output / "sender_rows.json") if (output / "sender_rows.json").exists() else []
    indexed = {r["job"]["example_id"]: r for r in rows}
    if len(indexed) != len(rows) or not set(indexed) <= {
        j["example_id"] for j in protocol["actor_jobs"]
    }:
        raise ValueError("Duplicate or unplanned saved report")
    for row in rows:
        check_features(row, output)
    actor = reader = None
    try:
        if stage in {"all", "reports"}:
            actor = backend_factory(ACTOR_SETTINGS)
            signature = check_backend(actor, ACTOR_SETTINGS, fixture)
            if tokenizer_identity(actor.tokenizer) != preflight["actor_tokenizer"]:
                raise ValueError("Actor tokenizer changed after preflight")
            if manifest.get("actor_runtime_signature", signature) != signature:
                raise ValueError("Actor runtime changed on resume")
            manifest.update(
                status="running_reports",
                actor_runtime_signature=signature,
                actor_backend=actor.metadata,
            )
            atomic_json(output / "manifest.json", manifest)
            for job in protocol["actor_jobs"]:
                assert_sources(protocol)
                expected_ids = preflight["actor_inputs"][job["example_id"]]
                if (
                    encode(actor.tokenizer, job["messages"], ACTOR_SETTINGS.max_input_tokens)
                    != expected_ids
                ):
                    raise ValueError("Actor input changed after preparation")
                if job["example_id"] in indexed:
                    row = indexed[job["example_id"]]
                    check_sender_record(row, job, actor.tokenizer, expected_ids, output)
                else:
                    manifest["generation_attempts"]["actor"] += 1
                    atomic_json(output / "manifest.json", manifest)
                    result = actor.generate(job["messages"], REPORT_GENERATION, SEED, capture=False)
                    generation = decoded_record(actor, result)
                    row = {
                        "job": job,
                        "generation": generation,
                        "generation_sha256": canonical_hash(generation),
                        "report_text": generation["raw_decoded_text"],
                        "report_sha256": generation["raw_text_sha256"],
                        "word_count": len(generation["raw_decoded_text"].split()),
                        "capture": {"status": "pending"},
                    }
                    rows.append(row)
                    indexed[job["example_id"]] = row
                    save_journal(output, manifest, "sender_rows", rows)
                    check_sender_record(row, job, actor.tokenizer, expected_ids, output)
                if row["capture"]["status"] == "pending":
                    arrays, capture = capture_saved(actor, row["generation"])
                    row["capture"] = capture
                    if arrays:
                        relative = f"features/{job['example_id']}.npz"
                        atomic_npz(output / relative, **arrays)
                        row.update(
                            feature_file=relative, feature_sha256=file_sha256(output / relative)
                        )
                    save_journal(output, manifest, "sender_rows", rows)
                print(f"reports {len(rows)}/32: {job['example_id']}", flush=True)
            actor.close()
            actor = None
            assert_sources(protocol)
            _write_blind_review(output, panel, rows)
            manifest.update(
                status="reports_complete",
                sender_rows_sha256=file_sha256(output / "sender_rows.json"),
            )
            atomic_json(output / "manifest.json", manifest)
        if stage == "reports":
            return manifest
        if len(rows) != 32 or any(row["capture"]["status"] == "pending" for row in rows):
            raise ValueError("All 32 reports and capture dispositions are required before readers")
        if file_sha256(output / "sender_rows.json") != manifest.get("sender_rows_sha256"):
            raise ValueError("Saved actor report bank changed before reader replay")
        planned = reader_plan(panel, {key: row["report_text"] for key, row in indexed.items()})
        reader = backend_factory(READER_SETTINGS)
        signature = check_backend(reader, READER_SETTINGS, fixture)
        if tokenizer_identity(reader.tokenizer) != preflight["reader_tokenizer"]:
            raise ValueError("Reader tokenizer changed after preparation")
        if manifest.get("reader_runtime_signature", signature) != signature:
            raise ValueError("Reader runtime changed on resume")
        # Freeze EVERY actual prompt before obtaining ANY reader answer.
        for job in planned:
            job["prompt_token_ids"] = encode(
                reader.tokenizer, job["messages"], READER_SETTINGS.max_input_tokens
            )
            job["cache_key"] = canonical_hash(
                {
                    "messages": job["messages"],
                    "runtime": signature,
                    "generation": asdict(READER_GENERATION),
                    "seed": SEED,
                }
            )
        plan_path = output / "reader_plan.json"
        if plan_path.exists() and read_json(plan_path) != planned:
            raise ValueError("Actual frozen reader plan changed on resume")
        save_journal(output, manifest, "reader_plan", planned)
        manifest.update(
            status="running_readers",
            reader_runtime_signature=signature,
            reader_backend=reader.metadata,
            reader_plan_sha256=file_sha256(plan_path),
        )
        atomic_json(output / "manifest.json", manifest)
        cache_path = output / "reader_cache.json"
        cache = read_json(cache_path) if cache_path.exists() else {}
        check_saved_file(output, manifest, "reader_cache")
        planned_keys = {job["cache_key"] for job in planned}
        if not set(cache) <= planned_keys:
            raise ValueError("Unplanned cached reader call")
        reader_rows = []
        for job in planned:
            assert_sources(protocol)
            key = job["cache_key"]
            hit = key in cache
            if not hit:
                manifest["generation_attempts"]["reader"] += 1
                atomic_json(output / "manifest.json", manifest)
                generation = decoded_record(
                    reader, reader.generate(job["messages"], READER_GENERATION, SEED, capture=False)
                )
                cache[key] = {
                    "generation": generation,
                    "generation_sha256": canonical_hash(generation),
                    "messages_sha256": canonical_hash(job["messages"]),
                }
                save_journal(output, manifest, "reader_cache", cache)
            saved = cache[key]
            if (
                saved["messages_sha256"] != canonical_hash(job["messages"])
                or saved["generation"]["prompt_token_ids"] != job["prompt_token_ids"]
            ):
                raise ValueError("Reader cache does not bind to exact frozen input")
            generation = saved["generation"]
            if saved["generation_sha256"] != canonical_hash(generation):
                raise ValueError("Saved reader generation metadata/IDs changed")
            raw = reader.tokenizer.decode(generation["generated_token_ids"], **DECODE)
            if generation["raw_decoded_text"] != raw or generation[
                "raw_text_sha256"
            ] != text_sha256(raw):
                raise ValueError("Saved reader output IDs/text differ")
            decision = (
                parse_decision(generation["text"])
                if generation["finish_reason"] == "eos"
                else "INVALID"
            )
            reader_rows.append(
                {**job, "decision": decision, "generation": generation, "cache_hit": hit}
            )
            atomic_json(output / "reader_rows.json", reader_rows)
            print(f"readers {len(reader_rows)}/240; unique calls {len(cache)}", flush=True)
        reader.close()
        reader = None
        assert_sources(protocol)
        summary = summarise(rows, reader_rows, fixture=fixture)
        summary["unique_reader_calls"] = len(cache)
        atomic_json(output / "summary.json", summary)
        manifest.update(
            status="complete",
            n_reports=len(rows),
            n_reader_conditions=len(reader_rows),
            unique_reader_calls=len(cache),
            reader_rows_sha256=file_sha256(output / "reader_rows.json"),
            summary_sha256=file_sha256(output / "summary.json"),
        )
    except BaseException:
        manifest.update(status="failed_execution", failure=traceback.format_exc())
        raise
    finally:
        for backend in (actor, reader):
            if backend is not None:
                backend.close()
        unchanged = sources_unchanged(protocol)
        binding_failure = not unchanged and manifest["status"] in {"complete", "reports_complete"}
        if binding_failure:
            manifest.update(
                status="failed_source_binding", failure="Source changed before finalization"
            )
        manifest.update(finished_utc=utc_now(), source_unchanged=unchanged)
        atomic_json(output / "manifest.json", manifest)
        if binding_failure:
            raise ValueError("Final source binding failed; preserve results as a failed attempt")
    return manifest


def summarise(sender_rows, reader_rows, *, fixture=False):
    if (
        len(sender_rows) != 32
        or len(reader_rows) != 240
        or len({r["example_id"] for r in reader_rows}) != 240
    ):
        raise ValueError("A complete summary needs all 32 reports and 240 unique conditions")
    if Counter(r["version"] for r in reader_rows) != {
        "original": 96,
        "restored": 48,
        "irrelevant": 48,
        "no_report": 24,
        "full_disclosure": 24,
    }:
        raise ValueError("Reader factorization differs from the prospective plan")
    groups = {}
    for budget in (25, 50):
        for goal in ("neutral", "favorable"):
            subset = [
                r
                for r in reader_rows
                if r["version"] == "original"
                and r["word_budget"] == budget
                and r["objective"] == goal
            ]
            counts = Counter(r["decision"] for r in subset)
            cases = {
                label: sorted({r["case_id"] for r in subset if r["decision"] == label})
                for label in CLASSES
            }
            groups[f"{budget}/{goal}"] = {
                "n": len(subset),
                "decision_counts": dict(counts),
                "class_case_support": cases,
                "valid": sum(counts[c] for c in CLASSES),
            }
    by_report = {}
    for row in reader_rows:
        if row["version"] == "original" and row["word_budget"] == 25:
            by_report.setdefault(row["source_example_id"], []).append(row)
    sensitive = sorted(
        {
            rows[0]["case_id"]
            for rows in by_report.values()
            if len({r["decision"] for r in rows if r["decision"] in CLASSES}) >= 2
        }
    )
    no_report = {
        (r["case_id"], r["context"]): r for r in reader_rows if r["version"] == "no_report"
    }
    changes = []
    for rows in by_report.values():
        for row in rows:
            base = no_report[row["case_id"], row["context"]]
            if (
                row["decision"] in CLASSES
                and base["decision"] in CLASSES
                and row["decision"] != base["decision"]
            ):
                changes.append(
                    {
                        "case_id": row["case_id"],
                        "source_example_id": row["source_example_id"],
                        "context": row["context"],
                        "no_report": base["decision"],
                        "original": row["decision"],
                    }
                )
    controls = {}
    for version in ("no_report", "full_disclosure"):
        subset = [r for r in reader_rows if r["version"] == version]
        controls[version] = {
            "n": len(subset),
            "normative_correct": sum(
                r["decision"] == r["normative_control_decision"] for r in subset
            ),
            "decision_counts": dict(Counter(r["decision"] for r in subset)),
        }
    originals = {
        (r["source_example_id"], r["context"]): r for r in reader_rows if r["version"] == "original"
    }
    transitions = {}
    for version in ("restored", "irrelevant"):
        subset = [r for r in reader_rows if r["version"] == version]
        transitions[version] = dict(
            Counter(
                originals[r["source_example_id"], r["context"]]["decision"] + "->" + r["decision"]
                for r in subset
            )
        )
    return {
        "schema": "natural_selection_behavioral_summary_v1",
        "is_test_fixture": fixture,
        "n_cases": 8,
        "n_reports": 32,
        "n_reader_conditions": 240,
        "groups": groups,
        "controls": controls,
        "transitions": transitions,
        "packet_sensitive_25word_cases": sensitive,
        "original_no_report_changes": changes,
        "report_dependent_25word_cases": sorted({r["case_id"] for r in changes}),
        "word_budget_violations": [
            r["job"]["example_id"] for r in sender_rows if r["word_count"] > r["job"]["word_budget"]
        ],
        "non_eos_reports": [
            r["job"]["example_id"] for r in sender_rows if r["generation"]["finish_reason"] != "eos"
        ],
        "semantic_review_status": "pending_blind_agent_and_human_review",
        "advancement_status": "not_determined_without_semantic_review",
        "new_probe_fits": 0,
        "untouched_transfer_calls": 0,
        "limits": [
            "Eight development clusters sharing one grammar.",
            "Reader decisions are neither deception labels nor measured human beliefs.",
            "The earlier reader qualification remains failed.",
            "50-word results cannot qualify the 25-word main-study arm.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--stage", choices=("all", "reports", "readers"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output, args.panel)
    else:
        result = execute(args.output, stage=args.stage, resume=args.resume)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
