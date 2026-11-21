"""Export, inspect and replay the completed natural-selection pilot on another CPU.

Archives contain exact captured bytes, never weights, credentials or new inference.
Default replay checks saved records; optional local tokenizers independently check IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import stat
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "natural_selection_portable_bundle_v1"
MAX_FILES = 150
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 80 * 1024 * 1024
ANALYSIS = "docs/validation/2026-11-20-natural-selection-pilot.json"
APOLLO = "docs/validation/2026-11-20-natural-selection-apollo.json"
REVIEW_A = "data/annotations/natural_selection_pilot_v1_reviewer_a_agent_draft.json"
REVIEW_B = "data/annotations/natural_selection_pilot_v1_reviewer_b_agent_draft.json"
REVIEW_PROTOCOL = "data/annotations/natural_selection_review_protocol_v1.json"
RUBRIC = "docs/NATURAL_SELECTION_REVIEW_RUBRIC.md"
SCRIPTS = (
    "scripts/analyse_natural_selection_pilot.py",
    "scripts/score_natural_selection_apollo.py",
    "scripts/run_natural_selection_pilot.py",
)
MODULES = (
    "__init__",
    "apollo",
    "backends",
    "calibrated_reader",
    "communication_panel",
    "config",
    "experiment",
    "forecast_panel",
    "natural_selection_panel",
    "prompts",
    "reader_panel",
    "schema",
    "storage",
)
README = """# Portable natural-selection development pilot

This is an exact-byte research archive, not new inference or a held-out result.
It includes 32 saved reports, 240 reader conditions, 222 unique reader records,
32 token-feature matrices, the frozen Apollo adaptation, draft agent annotations,
source/protocol snapshots and the analysis code. Human semantic review is pending.
No model weights, Hugging Face tokens, model cache or recipient-transfer bank is included.

Verify the ZIP against the separately recorded archive SHA256 before extraction.
Inspect source code before executing an archive from an untrusted publisher.
Inventory hashes detect accidental changes; they are not a publisher signature.

With Python 3.11+ and the project's NumPy/scikit-learn/SciPy dependencies installed:

    python reproduce.py reproduce --bundle . --output ../replayed.json

This works offline without a GPU, PyTorch, transformers or access to the original
Windows/WSL directories. It recomputes annotations, groups, gates, transitions,
behavioral summaries and all frozen Apollo margins/AUROCs from saved records.
It verifies captured token-ID and wire-text bindings but DOES NOT independently
retokenize text or decode token IDs. This limitation is explicit in the output.

For independent tokenizer verification, also install the locked `models` extra,
and supply separately acquired, pinned local tokenizer assets (no weights needed):

    python reproduce.py reproduce --bundle . --output ../replayed-tokenized.json \\
      --actor-tokenizer /local/llama-tokenizer --reader-tokenizer /local/qwen-tokenizer

Both tokenizer identities must exactly match the captured vocabulary/template/
special-token hashes. Network access is disabled for loading them. Their source
models' access and license terms apply; tokenizer assets are not redistributed.
The original captured files and their absolute provenance strings are unchanged.
A separate inventory maps those strings to local bundle members.

Neither replay repeats stochastic/numerical model inference or validates the
draft semantics. No probe fit, target change, threshold selection or main-study
advancement is permitted. Use the included original lockfile for the exact
dependency reference; numerical comparisons allow 1e-12 absolute/relative error.
"""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON object key")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def member_name(name):
    require(isinstance(name, str) and bool(name), "Empty archive member")
    path = PurePosixPath(name)
    require(
        "\\" not in name
        and ":" not in name
        and not path.is_absolute()
        and name == path.as_posix()
        and all(part not in {"", ".", ".."} for part in name.split("/")),
        "Unsafe archive member: " + name,
    )
    for part in path.parts:
        require(
            part == part.rstrip(" .")
            and not re.search(r'[<>"|?*\x00-\x1f]', part)
            and not re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", part.split(".")[0], re.I),
            "Nonportable archive member: " + name,
        )
    return name


def inventory_for(payload, original_paths):
    return {
        "schema": SCHEMA,
        "files": {
            member_name(name): {"bytes": len(data), "sha256": digest(data)}
            for name, data in sorted(payload.items())
        },
        "original_paths": dict(sorted(original_paths.items())),
        "tokenizer_assets_included": False,
        "model_weights_included": False,
        "new_model_calls": 0,
        "new_probe_fits": 0,
    }


def validate_inventory(value):
    require(value.get("schema") == SCHEMA, "Wrong portable inventory schema")
    require(set(value) == set(inventory_for({}, {})), "Unexpected inventory fields")
    require(
        value["tokenizer_assets_included"] is False
        and value["model_weights_included"] is False
        and type(value["new_model_calls"]) is int
        and value["new_model_calls"] == 0
        and type(value["new_probe_fits"]) is int
        and value["new_probe_fits"] == 0,
        "Wrong archive scope",
    )
    files = value["files"]
    require(isinstance(files, dict) and 0 < len(files) < MAX_FILES, "Invalid inventory length")
    total = 0
    for name, item in files.items():
        member_name(name)
        require(name != "inventory.json", "Inventory cannot hash itself")
        require(set(item) == {"bytes", "sha256"}, "Unexpected file descriptor")
        require(
            type(item["bytes"]) is int
            and 0 <= item["bytes"] <= MAX_FILE_BYTES
            and isinstance(item["sha256"], str)
            and re.fullmatch("[0-9a-f]{64}", item["sha256"]),
            "Invalid file descriptor",
        )
        total += item["bytes"]
    require(total <= MAX_TOTAL_BYTES, "Bundle exceeds total uncompressed limit")
    require(isinstance(value["original_paths"], dict), "Missing original path mapping")
    for original, name in value["original_paths"].items():
        require(isinstance(original, str) and original and name in files, "Invalid path mapping")
    return value


def _repo_relative(original):
    normalized = str(original).replace("\\", "/")
    marker = "/Individual-Project/"
    if marker in normalized:
        return normalized.split(marker, 1)[1]
    require(not normalized.startswith("/") and ":" not in normalized, "Unknown captured path")
    return member_name(normalized)


def export_bundle(repo, run, probe, output):
    """Export a bounded allowlist, preserving every original artifact byte."""
    repo, run, probe, output = map(Path, (repo, run, probe, output))
    require(not output.exists(), "Archive already exists; no overwrite")
    analysis, apollo = read_json(repo / ANALYSIS), read_json(repo / APOLLO)
    protocol = read_json(run / "protocol.json")
    payload, paths = {}, {}

    def add(name, path, expected=None):
        name = member_name(name)
        require(Path(path).is_file(), "Missing required artifact: " + str(path))
        data = Path(path).read_bytes()
        require(expected is None or digest(data) == expected, "Captured SHA256 differs: " + name)
        require(name not in payload or payload[name] == data, "Conflicting bundle member")
        payload[name] = data

    frozen = analysis["provenance"]
    raw = {name + ".json": sha for name, sha in frozen["run_sha256"].items()}
    raw.update(frozen["feature_sha256"])
    raw.update(
        {
            "blind_reports.json": frozen["review_sha256"]["blind_reports"],
            "private_review_mapping.json": frozen["review_sha256"]["private_mapping"],
        }
    )
    require(len(raw) == 42, "Expected precisely 42 saved raw artifacts")
    actual = {p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file()}
    require(actual == set(raw), "Raw directory differs from the bounded captured inventory")
    original_run = apollo["provenance"]["run_directory"].rstrip("/")
    for name, sha in raw.items():
        add("run/" + name, run / name, sha)
        paths[original_run + "/" + name] = "run/" + name
    for original, sha in protocol["source_sha256"].items():
        relative = _repo_relative(original)
        name = "source/" + relative
        add(name, repo / relative, sha)
        paths[original] = name
    for original, sha in apollo["provenance"]["input_sha256"].items():
        if original in paths:
            require(digest(payload[paths[original]]) == sha, "Conflicting original binding")
            continue
        relative = _repo_relative(original)
        if relative == ".cache/natural_selection_pilot_analysis_draft.json":
            relative = ANALYSIS
        if relative.endswith("apollo_llama8b_full_probe_v1/probe.npz"):
            name, local = "probe/probe.npz", probe / "probe.npz"
        elif relative.endswith("apollo_llama8b_full_probe_v1/manifest.json"):
            name, local = "probe/manifest.json", probe / "manifest.json"
        else:
            name, local = "source/" + relative, repo / relative
        add(name, local, sha)
        paths[original] = name
    for relative in (
        ANALYSIS,
        APOLLO,
        REVIEW_A,
        REVIEW_B,
        REVIEW_PROTOCOL,
        RUBRIC,
        "docs/NATURAL_SELECTION_PILOT_RESULTS.md",
        *SCRIPTS,
    ):
        add("source/" + relative, repo / relative)
    # Only the transitive imports of the frozen analyzers/runner are exported.
    # New studies or unrelated package modules cannot silently enter this bundle.
    for name in MODULES:
        relative = "src/latent_probes/" + name + ".py"
        add("source/" + relative, repo / relative)
    for relative in ("pyproject.toml", "uv.lock"):
        add("source/" + relative, repo / relative)
    add("reproduce.py", Path(__file__))
    payload["README.md"] = README.encode()
    inventory = validate_inventory(inventory_for(payload, paths))
    payload["inventory.json"] = json_bytes(inventory)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, data, compresslevel=9)
    verified = verify_archive(output)
    return {
        **verified,
        "archive_sha256": digest(output.read_bytes()),
        "archive_bytes": output.stat().st_size,
    }


def verify_archive(path):
    """Validate all names, sizes and bytes before any extraction or execution."""
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [member_name(info.filename) for info in infos]
        require(
            len(names) == len(set(name.casefold() for name in names)) <= MAX_FILES,
            "Duplicate/excess archive members",
        )
        require("inventory.json" in names, "Missing inventory")
        for info in infos:
            kind = stat.S_IFMT(info.external_attr >> 16)
            require(kind in {0, stat.S_IFREG} and not info.is_dir(), "Nonregular archive member")
            require(not info.flag_bits & 1, "Encrypted archive members are forbidden")
            require(info.file_size <= MAX_FILE_BYTES, "Archive member too large")
        require(sum(i.file_size for i in infos) <= MAX_TOTAL_BYTES, "Archive too large")
        inventory = validate_inventory(
            json.loads(archive.read("inventory.json"), object_pairs_hook=unique_object)
        )
        require(
            set(names) == set(inventory["files"]) | {"inventory.json"}, "Unlisted/missing member"
        )
        for name, item in inventory["files"].items():
            data = archive.read(name)
            require(
                len(data) == item["bytes"] and digest(data) == item["sha256"],
                "Member hash differs: " + name,
            )
    return {
        "files": len(inventory["files"]),
        "inventory_sha256": digest(json_bytes(inventory)),
        "verified": True,
    }


def extract_bundle(archive_path, destination):
    verify_archive(archive_path)
    destination = Path(destination)
    require(not destination.exists(), "Extraction destination must not exist")
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            path = destination / member_name(info.filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(archive.read(info))
    return verify_directory(destination)


def verify_directory(bundle):
    bundle = Path(bundle)
    inventory = validate_inventory(read_json(bundle / "inventory.json"))
    files = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
    # Python imports may create __pycache__; never treat those as captured evidence.
    files = {name for name in files if "__pycache__" not in PurePosixPath(name).parts}
    require(files == set(inventory["files"]) | {"inventory.json"}, "Expanded inventory differs")
    for name, item in inventory["files"].items():
        path = bundle / name
        require(
            not path.is_symlink() and path.resolve().is_relative_to(bundle.resolve()),
            "Symlink/escape",
        )
        data = path.read_bytes()
        require(
            len(data) == item["bytes"] and digest(data) == item["sha256"],
            "Expanded hash differs: " + name,
        )
    return inventory


def load_script(source, relative, name):
    spec = importlib.util.spec_from_file_location(name, source / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(actual, expected, path="root"):
    """Allow numerical rounding only; never tolerate count, field or string drift."""
    require(type(actual) is type(expected), "Reproduction type differs: " + path)
    if isinstance(actual, dict):
        require(set(actual) == set(expected), "Reproduction keys differ: " + path)
        for key in actual:
            compare(actual[key], expected[key], path + "/" + str(key))
    elif isinstance(actual, list):
        require(len(actual) == len(expected), "Reproduction length differs: " + path)
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            compare(left, right, path + "/" + str(index))
    elif isinstance(actual, float):
        require(
            math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            "Numerical result differs: " + path,
        )
    else:
        require(actual == expected, "Reproduction value differs: " + path)


def record_verified_run(run, analysis, resolve):
    """Verify saved records without claiming independent tokenization/decoding."""
    import numpy as np

    from latent_probes.natural_selection_panel import actor_jobs, load_panel, reader_plan
    from latent_probes.reader_panel import parse_decision
    from latent_probes.storage import canonical_hash, file_sha256

    r = analysis._runner_module()
    manifest = read_json(run / "manifest.json")
    require(
        manifest["status"] == "complete" and manifest["source_unchanged"] is True, "Incomplete run"
    )
    names = (
        "protocol",
        "preflight",
        "sender_rows",
        "reader_plan",
        "reader_cache",
        "reader_rows",
        "summary",
    )
    hashes = {name: file_sha256(run / (name + ".json")) for name in names}
    require(
        all(sha == manifest[name + "_sha256"] for name, sha in hashes.items()),
        "Run journal binding differs",
    )
    objects = {name: read_json(run / (name + ".json")) for name in names}
    protocol, preflight = objects["protocol"], objects["preflight"]
    require(
        protocol["is_test_fixture"] is manifest["is_test_fixture"] is False, "Real run required"
    )
    for original, sha in protocol["source_sha256"].items():
        require(file_sha256(resolve(original)) == sha, "Captured source differs")
    for key, value in (
        ("actor_model", asdict(r.ACTOR_SETTINGS)),
        ("reader_model", asdict(r.READER_SETTINGS)),
        ("report_generation", asdict(r.REPORT_GENERATION)),
        ("reader_generation", asdict(r.READER_GENERATION)),
        ("seed", r.SEED),
        ("decode", r.DECODE),
        ("policy", r.policy_metadata()),
    ):
        require(protocol[key] == value, "Frozen setting differs: " + key)
    panel = load_panel(resolve(protocol["panel_path"]))
    jobs, sender = actor_jobs(panel), objects["sender_rows"]
    require(protocol["actor_jobs"] == jobs and len(sender) == 32, "Actor bank differs")
    indexed = {row["job"]["example_id"]: row for row in sender}
    require(
        len(indexed) == 32 and set(indexed) == {j["example_id"] for j in jobs}, "Actor IDs differ"
    )
    features = {}
    specials = set(preflight["actor_tokenizer"]["special_ids"])
    for job in jobs:
        row = indexed[job["example_id"]]
        generation, capture = row["generation"], row["capture"]
        prompt, generated = generation["prompt_token_ids"], generation["generated_token_ids"]
        require(
            row["job"] == job and prompt == preflight["actor_inputs"][job["example_id"]],
            "Saved actor prompt differs",
        )
        require(canonical_hash(generation) == row["generation_sha256"], "Actor generation differs")
        require(
            generation["raw_decoded_text"] == row["report_text"] == generation["text"],
            "Actor wire/report text differs",
        )
        require(
            digest(row["report_text"].encode())
            == row["report_sha256"]
            == generation["raw_text_sha256"],
            "Actor wire hash differs",
        )
        require(row["word_count"] == len(row["report_text"].split()), "Word count differs")
        positions = [len(prompt) + i for i, token in enumerate(generated) if token not in specials]
        selected = [token for token in generated if token not in specials]
        require(
            capture["status"] == "complete"
            and capture["selected_positions"] == positions
            and capture["selected_ids"] == selected
            and capture["sequence_sha256"] == canonical_hash(prompt + generated),
            "Actor capture differs",
        )
        name = member_name(row["feature_file"])
        path = run / name
        features[name] = file_sha256(path)
        require(features[name] == row["feature_sha256"], "Actor feature binding differs")
        with np.load(path, allow_pickle=False) as arrays:
            require(set(arrays.files) == {"layer_8_tokens"}, "Unexpected feature coordinates")
            values = arrays["layer_8_tokens"]
            require(
                values.shape == (len(selected), 4096) and np.isfinite(values).all(),
                "Invalid feature matrix",
            )
    plan = reader_plan(panel, {key: row["report_text"] for key, row in indexed.items()})
    saved_plan, rows, cache = (
        objects["reader_plan"],
        objects["reader_rows"],
        objects["reader_cache"],
    )
    require(len(plan) == len(saved_plan) == len(rows) == 240, "Reader conditions differ")
    for job, saved, row in zip(plan, saved_plan, rows, strict=True):
        require({key: saved[key] for key in job} == job, "Exact reader message differs")
        require({key: row[key] for key in saved} == saved, "Reader condition binding differs")
        key = canonical_hash(
            {
                "messages": job["messages"],
                "runtime": manifest["reader_runtime_signature"],
                "generation": asdict(r.READER_GENERATION),
                "seed": r.SEED,
            }
        )
        require(key == saved["cache_key"], "Cache key differs")
        cached, generation = cache[key], row["generation"]
        require(
            cached["generation"] == generation
            and cached["generation_sha256"] == canonical_hash(generation)
            and cached["messages_sha256"] == canonical_hash(job["messages"]),
            "Reader generation differs",
        )
        require(
            generation["prompt_token_ids"] == saved["prompt_token_ids"]
            and digest(generation["raw_decoded_text"].encode()) == generation["raw_text_sha256"],
            "Reader IDs/wire hash differ",
        )
        require(
            generation["text"] == generation["raw_decoded_text"].strip(),
            "Reader normalized text differs",
        )
        decision = (
            parse_decision(generation["text"])
            if generation["finish_reason"] == "eos"
            else "INVALID"
        )
        require(row["decision"] == decision, "Parsed decision differs")
    require(
        set(cache) == {row["cache_key"] for row in saved_plan}, "Unexpected reader cache records"
    )
    summary = r.summarise(sender, rows, fixture=False)
    summary["unique_reader_calls"] = len(cache)
    require(
        summary == objects["summary"] and len(cache) == manifest["unique_reader_calls"],
        "Behavioral summary differs",
    )
    return {
        "run": run,
        "manifest": manifest,
        "protocol": protocol,
        "panel": panel,
        "sender": sender,
        "reader": rows,
        "summary": summary,
        "feature_hashes": features,
        "hashes": {**hashes, "manifest": file_sha256(run / "manifest.json")},
    }


def reproduce(bundle, output, actor_tokenizer=None, reader_tokenizer=None):
    """Recompute all scientific results using relocated source and evidence."""
    import numpy as np

    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    require(
        not output.exists() and not output.is_relative_to(bundle),
        "Use a new output outside the immutable bundle",
    )
    require(
        bool(actor_tokenizer) == bool(reader_tokenizer),
        "Supply both local tokenizer paths or neither",
    )
    inventory = verify_directory(bundle)
    source, run = bundle / "source", bundle / "run"
    for name, module in tuple(sys.modules.items()):
        if name == "latent_probes" or name.startswith("latent_probes."):
            filename = getattr(module, "__file__", None)
            require(
                filename is not None
                and Path(filename).resolve().is_relative_to((source / "src").resolve()),
                "Run replay as a fresh CLI process; project modules were already imported elsewhere",
            )
    sys.path.insert(0, str(source / "src"))
    analysis = load_script(source, SCRIPTS[0], "portable_pilot_analysis")
    apollo = load_script(source, SCRIPTS[1], "portable_pilot_apollo")

    def resolve(original):
        require(
            str(original) in inventory["original_paths"],
            "Captured path has no relocation mapping: " + str(original),
        )
        return bundle / inventory["original_paths"][str(original)]

    analysis._path = resolve
    if actor_tokenizer:
        from transformers import AutoTokenizer

        local = {
            "meta-llama/Llama-3.1-8B-Instruct": actor_tokenizer,
            "Qwen/Qwen2.5-7B-Instruct": reader_tokenizer,
        }

        def factory(settings):
            return AutoTokenizer.from_pretrained(
                local[settings.model_id], local_files_only=True, trust_remote_code=False
            )

        tokenizer_status = "independently_verified_with_pinned_local_tokenizer_identities"
    else:
        # Reuse the unchanged semantic analysis after explicit saved-record validation.
        # This hook never substitutes recorded IDs for an actual tokenizer.
        analysis._verified_run = lambda location, tokenizer_factory=None: record_verified_run(
            location, analysis, resolve
        )
        factory = None
        tokenizer_status = "not_retokenized_or_independently_decoded_saved_record_bindings_only"
    rebuilt = analysis.analyse(
        run,
        source / REVIEW_A,
        source / REVIEW_B,
        source / REVIEW_PROTOCOL,
        source / RUBRIC,
        tokenizer_factory=factory,
    )
    reference = read_json(source / ANALYSIS)
    scientific_keys = set(reference) - {"created_utc", "provenance"}
    compare(
        {key: rebuilt[key] for key in scientific_keys},
        {key: reference[key] for key in scientific_keys},
        "pilot",
    )
    from latent_probes.storage import canonical_hash

    saved_apollo = read_json(source / APOLLO)
    for original, sha in saved_apollo["provenance"]["input_sha256"].items():
        require(digest(resolve(original).read_bytes()) == sha, "Apollo captured input differs")
    probe = bundle / "probe/probe.npz"
    require(digest(probe.read_bytes()) == apollo.PROBE_SHA256, "Original frozen probe required")
    manifest = read_json(run / "manifest.json")
    probe_manifest = read_json(bundle / "probe/manifest.json")
    apollo.require_signature(probe_manifest["feature_signature"], manifest["actor_backend"])
    require(
        probe_manifest["probe_sha256"] == apollo.PROBE_SHA256
        and probe_manifest["settings"]["score_aggregation"] == "mean_raw_margin",
        "Probe metadata differs",
    )
    with np.load(probe, allow_pickle=False) as arrays:
        mean, scale, coefficient, intercept = (
            arrays[name].copy() for name in ("mean", "scale", "coefficient", "intercept")
        )
        if "feature_signature" in arrays.files:
            apollo.require_signature(
                str(arrays["feature_signature"].item()), manifest["actor_backend"]
            )
    require(mean.shape == scale.shape == coefficient.shape == (4096,), "Probe dimension differs")
    cases = {
        case["id"]: case
        for case in read_json(resolve(read_json(run / "protocol.json")["panel_path"]))["cases"]
    }
    audits = {audit["example_id"]: audit for audit in rebuilt["report_audits"]}
    rows = []
    specials = set(read_json(run / "preflight.json")["actor_tokenizer"]["special_ids"])
    for sender in read_json(run / "sender_rows.json"):
        job, generation = sender["job"], sender["generation"]
        with np.load(run / sender["feature_file"], allow_pickle=False) as arrays:
            tokens = arrays["layer_8_tokens"].copy()
        require(
            sender["capture"]["sequence_sha256"]
            == canonical_hash(generation["prompt_token_ids"] + generation["generated_token_ids"]),
            "Probe capture sequence differs",
        )
        rows.append(
            {
                "example_id": job["example_id"],
                "case_id": job["case_id"],
                "objective": job["objective"],
                "word_budget": job["word_budget"],
                "draft_omission": apollo.draft_omission(audits[job["example_id"]]),
                "report_text": sender["report_text"],
                "selected_content_tokens": sum(
                    token not in specials for token in generation["generated_token_ids"]
                ),
                "feature_file": sender["feature_file"],
                "feature_sha256": sender["feature_sha256"],
                "apollo_raw_margin": apollo.mean_margin(
                    tokens, mean, scale, coefficient, intercept
                ),
                **apollo.proxy_scores(
                    cases[job["case_id"]],
                    sender["report_text"],
                    job["word_budget"],
                    sender["word_count"],
                ),
            }
        )
    scores = {"rows": rows, **apollo.summarize(rows)}
    compare(scores, {key: saved_apollo[key] for key in scores}, "apollo")
    # Detect input mutation during replay before writing a result.
    verify_directory(bundle)
    result = {
        "schema": "natural_selection_portable_replay_v1",
        "inventory_sha256": digest(json_bytes(inventory)),
        "pilot_scientific_fields_reproduced": sorted(scientific_keys),
        "apollo_fields_reproduced": sorted(scores),
        "tokenizer_verification": tokenizer_status,
        "new_model_calls": 0,
        "new_probe_fits": 0,
        "human_review": False,
        "model_inference_repeated": False,
        "numeric_tolerance": {"absolute": 1e-12, "relative": 1e-12},
        "pilot": {key: rebuilt[key] for key in scientific_keys},
        "apollo": scores,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(json_bytes(result))
    return {
        "scientific_results_match": True,
        "tokenizer_verification": tokenizer_status,
        "new_model_calls": 0,
        "new_probe_fits": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--repo", type=Path, default=ROOT)
    export.add_argument("--run", type=Path, required=True)
    export.add_argument("--probe", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--archive", type=Path, required=True)
    extract = sub.add_parser("extract")
    extract.add_argument("--archive", type=Path, required=True)
    extract.add_argument("--destination", type=Path, required=True)
    replay = sub.add_parser("reproduce")
    replay.add_argument("--bundle", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--actor-tokenizer", type=Path)
    replay.add_argument("--reader-tokenizer", type=Path)
    args = parser.parse_args()
    if args.command == "export":
        result = export_bundle(args.repo, args.run, args.probe, args.output)
    elif args.command == "verify":
        result = verify_archive(args.archive)
    elif args.command == "extract":
        value = extract_bundle(args.archive, args.destination)
        result = {"extracted": True, "files": len(value["files"])}
    else:
        result = reproduce(args.bundle, args.output, args.actor_tokenizer, args.reader_tokenizer)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
