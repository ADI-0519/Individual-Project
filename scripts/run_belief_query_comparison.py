"""Paired story/query captures, followed by frozen scoring of fresh wording groups.

No answer is generated. Gold describes the same fictional prior state before
and after a question. Question presence also changes length and token position.
"""

import argparse
import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
from export_belief_readout import _snapshot, _source
from run_belief_diagnostic import LAYERS, capture_prefix, query_messages, validate_layer_indices

from latent_probes.backends import HFBackend
from latent_probes.belief_panel import validate_panel
from latent_probes.belief_query_panel import validate_fresh_panel
from latent_probes.belief_query_readout import (
    evaluate_fresh_query_comparison,
    fit_query_comparison,
)
from latent_probes.config import load_config
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
CONDITIONS = ("story", "nested_question", "world_question")
SIGNATURE_FIELDS = (
    "backend",
    "is_test_fixture",
    "model_id",
    "resolved_revision",
    "model_type",
    "dtype",
    "quantization",
    "decoder_blocks",
    "hidden_size",
    "capture_layers",
    "chat_template_sha256",
    "generation_defaults",
    "special_token_ids",
    "attention_implementation",
)
SOURCE_FILES = (
    "scripts/run_belief_query_comparison.py",
    "scripts/run_belief_diagnostic.py",
    "scripts/export_belief_readout.py",
    "src/latent_probes/belief_query_panel.py",
    "src/latent_probes/belief_query_readout.py",
)


def condition_messages(case, view, condition):
    if condition not in CONDITIONS or view not in ("omniscient", "sender"):
        raise ValueError("Unknown condition or information view")
    if condition == "story":
        return copy.deepcopy(case["messages"][view])
    return query_messages(case, view, "nested" if condition == "nested_question" else "world")


def entity_ids(case):
    return [
        "person:" + case["reader_name"].casefold(),
        "person:" + case["sender_name"].casefold(),
        "place:" + case["place_a"].casefold(),
        "place:" + case["place_b"].casefold(),
    ]


def _separate_paths(source_run, output_dir):
    source, output = Path(source_run).resolve(), Path(output_dir).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Source and output must be disjoint")
    if output.exists():
        raise FileExistsError("Use a new query-comparison output path")
    return source, output


def _source_prefixes(source, panel):
    captures = {r["example_id"]: r for r in read_json(source / "captures.json")}
    with np.load(source / "features.npz", allow_pickle=False) as archive:
        old_features = {key: archive[key].copy() for key in archive.files}
    old_rows = read_json(source / "rows.json")
    indices = {r["example_id"]: i for i, r in enumerate(old_rows)}
    expected = {}
    for case in panel["cases"]:
        for view in ("omniscient", "sender"):
            identifier = case["id"] + "__" + view
            capture = captures[identifier]
            expected[(identifier, "story")] = {
                "ids": capture["prompt_token_ids"],
                "vectors": {key: value[indices[identifier]] for key, value in old_features.items()},
            }
            for target in ("nested", "world"):
                row = read_json(source / "behavior" / (identifier + "__" + target + ".json"))
                if row.get("record_sha256") != canonical_hash(
                    {key: value for key, value in row.items() if key != "record_sha256"}
                ):
                    raise ValueError("Original behavioral record hash mismatch")
                if row["messages"] != query_messages(case, view, target):
                    raise ValueError("Original question wording differs from paired query")
                expected[(identifier, target + "_question")] = {
                    "ids": row["generation"]["prompt_token_ids"],
                }
    return expected


def _collect(panel, stage, output, backend, expected, cache, progress):
    rows, captures, repeats, replay = [], [], [], []
    matrices = {f"layer_{layer}": [] for layer in LAYERS}
    repeated = set()
    initial_cache = set(cache)
    layer_check = None
    final_token_ids = set()
    for index, case in enumerate(panel["cases"]):
        for view in ("omniscient", "sender"):
            identifier = case["id"] + "__" + view
            for condition in CONDITIONS:
                messages = condition_messages(case, view, condition)
                digest = canonical_hash(messages)
                if stage == "fresh" and digest in initial_cache:
                    raise ValueError("Fresh model input already occurred in legacy captures")
                reused = digest in cache
                feature_path = Path("prefix_features") / (digest + ".npz")
                if not reused:
                    cache[digest] = capture_prefix(backend, messages)
                    atomic_npz(output / feature_path, **cache[digest][1])
                ids, vectors = cache[digest]
                final_token_ids.add(ids[-1])
                prior = expected.get((identifier, condition))
                if stage == "legacy" and (prior is None or ids != prior["ids"]):
                    raise ValueError("Re-rendered legacy prefix differs from its saved token IDs")
                if prior and "vectors" in prior:
                    replay.append(
                        {
                            "example_id": identifier,
                            "layers": {
                                key: {
                                    "exact": bool(np.array_equal(vector, prior["vectors"][key])),
                                    "max_absolute_difference": float(
                                        np.max(
                                            np.abs(
                                                vector.astype(np.float64) - prior["vectors"][key]
                                            )
                                        )
                                    ),
                                }
                                for key, vector in vectors.items()
                            },
                        }
                    )
                repeat_key = (case["phase"], view, condition)
                if repeat_key not in repeated:
                    repeat_ids, repeat_vectors = capture_prefix(backend, messages)
                    check = {
                        "case_id": case["id"],
                        "view": view,
                        "condition": condition,
                        "phase": case["phase"],
                        "ids_exact": ids == repeat_ids,
                        "layers_exact": {
                            key: bool(np.array_equal(vector, repeat_vectors[key]))
                            for key, vector in vectors.items()
                        },
                    }
                    repeats.append(check)
                    repeated.add(repeat_key)
                    if not check["ids_exact"] or not all(check["layers_exact"].values()):
                        raise ValueError("Same-input capture is not exactly repeatable")
                if layer_check is None:
                    layer_check = validate_layer_indices(backend, ids)
                rows.append(
                    {
                        "example_id": identifier,
                        "case_id": case["id"],
                        "template_id": case["template_id"],
                        "split": case["split"],
                        "phase": case["phase"],
                        "view": view,
                        "condition": condition,
                        "messages_text": "\n".join(message["content"] for message in messages),
                        "entity_ids": entity_ids(case),
                        "labels": copy.deepcopy(case["labels"]),
                    }
                )
                for key, vector in vectors.items():
                    matrices[key].append(vector.copy())
                captures.append(
                    {
                        "example_id": identifier,
                        "condition": condition,
                        "messages_sha256": digest,
                        "feature_file": feature_path.as_posix(),
                        "feature_sha256": file_sha256(output / feature_path),
                        "prompt_token_ids": ids,
                        "selected_position": len(ids) - 1,
                        "selected_token_id": ids[-1],
                        "generated_answer_tokens_supplied": 0,
                        "question_supplied": condition != "story",
                        "reused_identical_capture": reused,
                    }
                )
                atomic_json(output / stage / "captures.json", captures)
        if progress:
            progress(f"Query comparison {stage} {index + 1}/{len(panel['cases'])}", flush=True)
    if len(final_token_ids) != 1:
        raise ValueError("Capture token identity differs across paired conditions")
    features = {key: np.asarray(values, dtype=np.float32) for key, values in matrices.items()}
    atomic_json(output / stage / "rows.json", rows)
    atomic_npz(output / stage / "features.npz", **features)
    atomic_json(output / stage / "repeats.json", repeats)
    atomic_json(output / stage / "legacy_story_replay.json", replay)
    return (
        rows,
        features,
        {
            "n_rows": len(rows),
            "n_unique_new_prefixes": len(set(cache) - initial_cache),
            "n_cases": len(panel["cases"]),
            "last_token_id": next(iter(final_token_ids)),
            "layer_index_check": layer_check,
            "n_repeat_checks": len(repeats),
            "legacy_story_exact": all(
                item["exact"] for row in replay for item in row["layers"].values()
            ),
            "legacy_story_replay_interpretation": "Exact equality recorded; mismatch is numerical inconclusive, not a reason to rewrite original captures.",
        },
    )


def run_query_comparison(source_run, fresh_panel, output_dir, backend, *, progress=print):
    source, output = _separate_paths(source_run, output_dir)
    original, protocol, _rows, _readouts, _X, original_hashes = _source(source)
    legacy = validate_panel(protocol["panel"])
    validate_fresh_panel(fresh_panel)
    for key in SIGNATURE_FIELDS:
        if backend.metadata.get(key) != original["backend"].get(key):
            raise ValueError(f"Backend differs from frozen source: {key}")
    if list(backend.settings.capture_layers) != list(LAYERS):
        raise ValueError("Use the three declared decoder blocks")
    old_templates = {case["template_id"] for case in legacy["cases"]}
    old_entities = {entity for case in legacy["cases"] for entity in entity_ids(case)}
    if old_templates & {case["template_id"] for case in fresh_panel["cases"]} or old_entities & {
        entity for case in fresh_panel["cases"] for entity in entity_ids(case)
    }:
        raise ValueError("Fresh templates or named entities overlap legacy data")
    expected = _source_prefixes(source, legacy)
    source_files = {name: file_sha256(ROOT / name) for name in SOURCE_FILES}
    panels_hash = canonical_hash([legacy, fresh_panel])
    output.mkdir(parents=True)
    plan = {
        "schema": "paired_belief_query_protocol_v1",
        "created_at": utc_now(),
        "legacy_panel": legacy,
        "fresh_panel": fresh_panel,
        "conditions": list(CONDITIONS),
        "primary_target": "nested",
        "primary_layer": "layer_16",
        "layers": list(LAYERS),
        "capture": "last chat-prompt token, before any generated answer",
        "legacy_analysis": "same known cases, exploratory matched input-conditioning comparison",
        "fresh_analysis": "new wording groups frozen before these new model outputs; same logical family; no human-validation claim",
        "fit": "separate condition/view heads; train-only transformations; C from legacy validation only",
        "fresh_order": "legacy captures, fit and hash-freeze all models, then first fresh capture and no-refit scoring",
        "secondary": "frozen cross-condition scoring retains source condition preprocessing and coefficients",
        "limits": [
            "Questioning changes length, position and task; world-question control is not length matched.",
            "Gold remains the same fictional prior state; no belief update or behavior improvement is measured.",
            "No generated answers supplied; no spontaneous, causal or unknown-message-anticipation claim.",
        ],
    }
    atomic_json(output / "protocol.json", plan)
    manifest = {
        "schema": "paired_belief_query_run_v1",
        "status": "running",
        "created_at": utc_now(),
        "environment": environment_info(),
        "backend": backend.metadata,
        "is_test_fixture": backend.metadata["is_test_fixture"],
        "source_input_sha256": original_hashes,
        "source_files_sha256": source_files,
        "panels_sha256": panels_hash,
        "protocol_sha256": file_sha256(output / "protocol.json"),
        "source_run": str(source),
        "generated_answers": 0,
    }
    atomic_json(output / "manifest.json", manifest)
    cache = {}
    try:
        rows, features, ledger = _collect(
            legacy, "legacy", output, backend, expected, cache, progress
        )
        manifest["legacy"] = ledger
        if progress:
            progress("Fit legacy-only readouts and freeze before fresh captures", flush=True)
        fit_query_comparison(rows, features, output / "fits")
        frozen = _snapshot(output / "fits")
        atomic_json(output / "frozen_fits.json", {"frozen_at": utc_now(), "artifacts": frozen})
        manifest["frozen_fit_artifacts"] = frozen
        manifest["fresh_capture_started_at"] = utc_now()
        atomic_json(output / "manifest.json", manifest)
        fresh_rows, fresh_features, ledger = _collect(
            fresh_panel, "fresh", output, backend, {}, cache, progress
        )
        manifest["fresh"] = ledger
        if _snapshot(output / "fits") != frozen:
            raise ValueError("Fitted artifacts changed during fresh captures")
        evaluate_fresh_query_comparison(
            fresh_rows, fresh_features, output / "fits", output / "fresh_evaluation"
        )
        if _snapshot(output / "fits") != frozen:
            raise ValueError("Fresh evaluation changed fitted artifacts")
        if _snapshot(source) != original_hashes:
            raise ValueError("Original run changed during paired comparison")
        if canonical_hash([legacy, fresh_panel]) != panels_hash or any(
            file_sha256(ROOT / name) != digest for name, digest in source_files.items()
        ):
            raise ValueError("Panel or executed diagnostic source changed during run")
        manifest.update(
            status="complete",
            source_unchanged=True,
            frozen_fits_unchanged=True,
            n_unique_forward_prefixes=len(cache),
        )
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        manifest["finished_at"] = utc_now()
        manifest["output_sha256"] = {
            name: digest for name, digest in _snapshot(output).items() if name != "manifest.json"
        }
        atomic_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--fresh-panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _separate_paths(args.source_run, args.output)
    original = read_json(args.source_run / "manifest.json")["backend"]
    fresh = read_json(args.fresh_panel)
    validate_fresh_panel(fresh)
    fresh_digest = file_sha256(args.fresh_panel)
    settings = replace(
        load_config(args.config).model,
        model_id=original["model_id"],
        revision=original["resolved_revision"],
        dtype=original["dtype"],
        quantization=original["quantization"],
        capture_layers=list(LAYERS),
        local_files_only=True,
    )
    print("Loading cached model for paired captures and frozen fresh evaluation", flush=True)
    backend = HFBackend(settings)
    try:
        run_query_comparison(args.source_run, fresh, args.output, backend)
        if file_sha256(args.fresh_panel) != fresh_digest:
            raise ValueError("Fresh panel source file changed during run")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
