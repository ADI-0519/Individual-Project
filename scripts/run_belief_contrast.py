"""Prospective complementary-claim measurement and evidence-channel validation.

The two candidate propositions are scored independently; no answer is generated.
Legacy captures support fitting, then every fitted artifact freezes before the
new bank is forwarded. Original experiments are read-only inputs.
"""

import argparse
import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
from run_belief_diagnostic import capture_prefix, validate_layer_indices

from latent_probes.backends import HFBackend
from latent_probes.belief_contrast_panel import validate_contrast_panel
from latent_probes.belief_contrast_readout import (
    evaluate_contrast_readouts,
    fit_contrast_readouts,
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
LAYERS = (8, 16, 24)
CONDITIONS = ("story", "nested_question")
VIEWS = ("omniscient", "sender")
SOURCES = (
    "scripts/run_belief_contrast.py",
    "scripts/run_belief_diagnostic.py",
    "src/latent_probes/belief_contrast_panel.py",
    "src/latent_probes/belief_contrast_readout.py",
    "src/latent_probes/backends.py",
)
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


def snapshot(root):
    return {
        p.relative_to(root).as_posix(): file_sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def legacy_candidates(case):
    """Replace only the nominated claim/question; never edit the scene."""
    result = copy.deepcopy(case)
    result["channel"] = "legacy_observation"
    result["candidates"] = []
    old = f'"The token is in the {case["place_b"]}."'
    for bit, place in ((0, case["place_a"]), (1, case["place_b"])):
        claim = f"The token is in the {place}."
        messages = copy.deepcopy(case["messages"])
        new = f'"{claim}"'
        for value in messages.values():
            text = value[-1]["content"]
            if text.count(old) != 1:
                raise ValueError("Legacy candidate boundary is not unique")
            value[-1]["content"] = text.replace(old, new, 1)
        questions = {}
        for key, text in case["questions"].items():
            if text.count(case["place_b"]) != 1:
                raise ValueError("Legacy question claim boundary is not unique")
            questions[key] = text.replace(case["place_b"], place, 1)
        result["candidates"].append(
            {
                "place_bit": bit,
                "claim": claim,
                "labels": {k: int(v) if bit else 1 - int(v) for k, v in case["labels"].items()},
                "messages": messages,
                "questions": questions,
            }
        )
    return result


def condition_messages(candidate, view, condition):
    if view not in VIEWS or condition not in CONDITIONS:
        raise ValueError("Unknown view or condition")
    result = copy.deepcopy(candidate["messages"][view])
    if condition == "nested_question":
        result[-1]["content"] += "\n\nQuestion: " + candidate["questions"]["nested"]
    return result


def load_source(source):
    manifest = read_json(source / "manifest.json")
    if manifest.get("status") != "complete" or manifest.get("is_test_fixture"):
        raise ValueError("A completed genuine query-comparison source is required")
    for rel, digest in manifest["output_sha256"].items():
        if file_sha256(source / rel) != digest:
            raise ValueError("Changed source output: " + rel)
    for rel, digest in manifest["source_files_sha256"].items():
        if file_sha256(ROOT / rel) != digest:
            raise ValueError("Changed executed query source: " + rel)
    protocol = read_json(source / "protocol.json")
    cases = [legacy_candidates(c) for c in protocol["legacy_panel"]["cases"]]
    for case in protocol["fresh_panel"]["cases"]:
        case = legacy_candidates(case)
        case["split"] = "development"
        cases.append(case)
    expected = {}
    for stage in ("legacy", "fresh"):
        rows = read_json(source / stage / "rows.json")
        captures = read_json(source / stage / "captures.json")
        by_key = {(c["example_id"], c["condition"]): c for c in captures}
        with np.load(source / stage / "features.npz", allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        for index, row in enumerate(rows):
            if row["condition"] not in CONDITIONS:
                continue
            capture = by_key[(row["example_id"], row["condition"])]
            expected[(row["case_id"], row["view"], row["condition"])] = {
                "ids": capture["prompt_token_ids"],
                "features": {k: v[index] for k, v in arrays.items()},
            }
    return manifest, cases, expected


def _collect(cases, stage, output, backend, expected, cache, progress):
    rows, captures, repeats, replay = [], [], [], []
    arrays = {f"layer_{layer}": [] for layer in LAYERS}
    initial = set(cache)
    checked = set()
    hf_check = None
    for index, case in enumerate(cases):
        for view in VIEWS:
            for condition in CONDITIONS:
                for candidate in sorted(case["candidates"], key=lambda c: c["place_bit"]):
                    bit = candidate["place_bit"]
                    messages = condition_messages(candidate, view, condition)
                    digest = canonical_hash(messages)
                    if stage == "fresh" and digest in initial:
                        raise ValueError("Fresh input already forwarded in legacy stage")
                    reused = digest in cache
                    rel = Path("prefix_features") / (digest + ".npz")
                    if not reused:
                        cache[digest] = capture_prefix(backend, messages)
                        atomic_npz(output / rel, **cache[digest][1])
                    ids, features = cache[digest]
                    if hf_check is None:
                        hf_check = validate_layer_indices(backend, ids)
                    key = (case["channel"], view, condition, bit)
                    if key not in checked:
                        r_ids, r_features = capture_prefix(backend, messages)
                        exact = ids == r_ids and all(
                            np.array_equal(features[k], r_features[k]) for k in arrays
                        )
                        repeats.append({"key": list(key), "prompt_sha256": digest, "exact": exact})
                        checked.add(key)
                        if not exact:
                            raise ValueError("Repeated prefix differs; numerical inconclusive")
                    if stage == "legacy" and bit == 1:
                        old = expected[(case["id"], view, condition)]
                        exact = ids == old["ids"] and all(
                            np.array_equal(features[k], old["features"][k]) for k in arrays
                        )
                        replay.append(
                            {
                                "case_id": case["id"],
                                "view": view,
                                "condition": condition,
                                "exact": exact,
                            }
                        )
                        if not exact:
                            raise ValueError(
                                "Original claim-B replay differs; preserve source and stop"
                            )
                    example_id = f"{case['id']}__{view}__{condition}__{bit}"
                    entity_ids = [
                        "person:" + case["reader_name"].casefold(),
                        "person:" + case["sender_name"].casefold(),
                        "place:" + case["place_a"].casefold(),
                        "place:" + case["place_b"].casefold(),
                    ]
                    rows.append(
                        {
                            "example_id": example_id,
                            "case_id": case["id"],
                            "template_id": case["template_id"],
                            "split": case["split"],
                            "phase": case["phase"],
                            "channel": case["channel"],
                            "view": view,
                            "condition": condition,
                            "claim_bit": bit,
                            "candidate_text": candidate["claim"],
                            "entity_ids": entity_ids,
                            "messages_text": "\n".join(m["content"] for m in messages),
                            "labels": candidate["labels"],
                        }
                    )
                    captures.append(
                        {
                            "example_id": example_id,
                            "prompt_sha256": digest,
                            "messages": messages,
                            "prompt_token_ids": ids,
                            "position": len(ids) - 1,
                            "feature_path": rel.as_posix(),
                            "cache_reused": reused,
                        }
                    )
                    for layer in arrays:
                        arrays[layer].append(features[layer])
        progress(f"{stage}: {index + 1}/{len(cases)} cases; {len(cache)} unique prefixes")
    path = output / stage
    atomic_json(path / "rows.json", rows)
    atomic_json(path / "captures.json", captures)
    atomic_json(path / "repeats.json", repeats)
    atomic_json(path / "legacy_claim_b_replay.json", replay)
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()}
    atomic_npz(path / "features.npz", **arrays)
    return (
        rows,
        arrays,
        {
            "n_rows": len(rows),
            "n_cases": len(cases),
            "n_unique_new_prefixes": len(set(cache) - initial),
            "repeat_checks": len(repeats),
            "all_repeats_exact": all(r["exact"] for r in repeats),
            "legacy_replay_rows": len(replay),
            "legacy_replay_exact": all(r["exact"] for r in replay) if replay else None,
            "hf_layer_index_check": hf_check,
        },
    )


def run_comparison(source_run, fresh_panel_path, output_dir, backend, *, progress=print):
    source, fresh_path, output = map(
        lambda p: Path(p).resolve(), (source_run, fresh_panel_path, output_dir)
    )
    if output.exists():
        raise FileExistsError("Choose a new output directory")
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Source and output paths must be disjoint")
    if list(backend.settings.capture_layers) != list(LAYERS):
        raise ValueError("Capture only prespecified blocks8,16,24")
    old, cases, expected = load_source(source)
    if any(old["backend"].get(k) != backend.metadata.get(k) for k in SIGNATURE_FIELDS):
        raise ValueError("Backend differs from the frozen source profile")
    fresh_panel = read_json(fresh_path)
    validate_contrast_panel(fresh_panel)
    legacy_entities = {
        v.casefold()
        for c in cases
        for v in (c["reader_name"], c["sender_name"], c["place_a"], c["place_b"])
    }
    fresh_entities = {
        v.casefold()
        for c in fresh_panel["cases"]
        for v in (c["reader_name"], c["sender_name"], c["place_a"], c["place_b"])
    }
    if legacy_entities & fresh_entities:
        raise ValueError("Fresh named entities overlap source cases")
    hashes = snapshot(source)
    code_hashes = {rel: file_sha256(ROOT / rel) for rel in SOURCES}
    fresh_hash = file_sha256(fresh_path)
    output.mkdir(parents=True)
    protocol = {
        "schema": "belief_complement_protocol_v1",
        "created_at": utc_now(),
        "legacy_cases": cases,
        "fresh_panel": fresh_panel,
        "conditions": list(CONDITIONS),
        "primary_layer": "layer_16",
        "primary_target": "nested",
        "primary_diagnostic": "Original frozen story-head raw-margin B minus A; no refit",
        "new_readout_endpoint": "Supervised strictly linear scaled B-A features, signed augmentation, no intercept",
        "fresh_order": "Legacy capture, fit and save all models, freeze all artifacts, then fresh capture and unchanged evaluation",
        "scope": "Prior and known-message normative perspectives; no generated answers, unknown-message forecast or deception labels",
        "independence_unit": "wording group; complementary candidates and factorial cells are repeated measurements",
        "evidence_channel_transfer": "Trusted observation record to trusted delivered-message receipt; same Boolean task, not broad reasoning-family OOD",
        "success_gate": {
            "new_readout": "story/sender/nested/layer_16/paired_linear",
            "each_fresh_channel_accuracy_at_least": 0.75,
            "each_fresh_channel_class_recall_at_least": 0.60,
            "each_fresh_channel_negative_log_loss_below": 0.6931471805599453,
            "each_fresh_channel_wording_group_accuracy_at_least": 0.50,
            "note": "Development go/no-go criterion, not statistical significance or a publication claim",
        },
        "no_test_guided_layer_or_method_selection": True,
        "generated_answers": 0,
        "limits": [
            "Complement consistency is enforced algebraically and is not independent evidence.",
            "Original frozen heads use row normalization; paired margins need not remove common raw activation offsets.",
            "New raw-difference head cannot create an interaction from purely additive world, receipt and candidate features.",
            "Fresh materials are agent-assisted; human semantic review is outstanding.",
            "TF-IDF and same-checkpoint behavior are not a capable independent text monitor.",
        ],
    }
    atomic_json(output / "protocol.json", protocol)
    manifest = {
        "schema": "belief_complement_run_v1",
        "status": "running",
        "created_at": utc_now(),
        "environment": environment_info(),
        "backend": copy.deepcopy(backend.metadata),
        "is_test_fixture": backend.metadata.get("is_test_fixture", False),
        "source_run": str(source),
        "source_input_sha256": hashes,
        "source_files_sha256": code_hashes,
        "fresh_panel_sha256": fresh_hash,
        "protocol_sha256": file_sha256(output / "protocol.json"),
        "generated_answers": 0,
    }
    atomic_json(output / "manifest.json", manifest)
    cache = {}
    try:
        rows, features, manifest["legacy"] = _collect(
            cases, "legacy", output, backend, expected, cache, progress
        )
        fit_contrast_readouts(rows, features, output / "fits", source / "fits")
        frozen = snapshot(output / "fits")
        atomic_json(output / "frozen_fits.json", {"frozen_at": utc_now(), "artifacts": frozen})
        manifest["fresh_capture_started_at"] = utc_now()
        atomic_json(output / "manifest.json", manifest)
        f_rows, f_features, manifest["fresh"] = _collect(
            fresh_panel["cases"], "fresh", output, backend, {}, cache, progress
        )
        evaluate_contrast_readouts(f_rows, f_features, output / "fits", output / "fresh_evaluation")
        manifest["source_unchanged"] = snapshot(source) == hashes
        manifest["frozen_fits_unchanged"] = snapshot(output / "fits") == frozen
        manifest["fresh_panel_unchanged"] = file_sha256(fresh_path) == fresh_hash
        manifest["source_code_unchanged"] = all(
            file_sha256(ROOT / p) == h for p, h in code_hashes.items()
        )
        if not all(
            manifest[k]
            for k in (
                "source_unchanged",
                "frozen_fits_unchanged",
                "fresh_panel_unchanged",
                "source_code_unchanged",
            )
        ):
            raise ValueError("Source, fresh data or frozen artifacts changed during execution")
        manifest["n_unique_forward_prefixes"] = len(cache)
        manifest["status"] = "complete"
        manifest["finished_at"] = utc_now()
        manifest["output_sha256"] = {
            k: v for k, v in snapshot(output).items() if k != "manifest.json"
        }
        atomic_json(output / "manifest.json", manifest)
        progress("Completed complementary-claim capture and frozen fresh scoring")
        return manifest
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        manifest["finished_at"] = utc_now()
        atomic_json(output / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--fresh-panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    backend = HFBackend(replace(config.model, capture_layers=LAYERS))
    run_comparison(args.source_run, args.fresh_panel, args.output, backend)


if __name__ == "__main__":
    main()
