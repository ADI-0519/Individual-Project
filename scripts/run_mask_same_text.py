"""Frozen off-policy MASK comparison: identical generated tokens, two contexts.

Select the first clean ablated honest draft per case, never using a margin.
This script neither generates text nor fits a probe nor establishes new labels.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

from latent_probes.apollo import capture_signature
from latent_probes.backends import HFBackend
from latent_probes.config import ModelSettings
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
SPEC = importlib.util.spec_from_file_location(
    "mask_specificity_same_text_source", ROOT / "scripts/run_mask_specificity.py"
)
specificity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(specificity)
LIMITS = (
    "Off-policy prompt-context sensitivity with exact text fixed; not natural deception, "
    "intent, reader belief or an intervention on naturally generated honest responses. "
    "Agent-assisted draft labels need human review. The q90 cutoff is a 32-prompt neutral "
    "reference, not a validated honest-response false-positive rate."
)


def _snapshot(directory):
    return {
        str(path.resolve()): file_sha256(path)
        for path in sorted(Path(directory).rglob("*"))
        if path.is_file()
    }


def _verify(plan):
    for path, digest in plan["source_sha256"].items():
        if file_sha256(path) != digest:
            raise ValueError("Frozen same-text source changed: " + path)
    for name in ("capture_dir", "score_dir"):
        actual = _snapshot(plan[name])
        expected = {
            path: digest
            for path, digest in plan["source_sha256"].items()
            if Path(path).is_relative_to(Path(plan[name]))
        }
        if actual != expected:
            raise ValueError("Frozen source tree changed: " + name)


def select_first_clean(records, annotations, panel):
    """Selection consumes statuses and fixed repeat order; no scores are accepted."""
    _beliefs, classified = specificity.classify_records(records, annotations, panel)
    record_by_id = {record["example_id"]: record for record in records}
    selected = []
    for case in panel["cases"]:
        eligible = [
            row
            for row in classified
            if row["case_id"] == case["id"]
            and row["condition"] == "ablated"
            and row["status"] == "honest_assertion"
            and row["belief_equals_upstream_truth"] is True
            and not record_by_id[row["example_id"]]["length_capped"]
        ]
        eligible.sort(
            key=lambda row: (
                int(record_by_id[row["example_id"]]["repeat"]),
                row["example_id"],
            )
        )
        if eligible:
            selected.append(eligible[0])
    if len(selected) > 8:
        raise ValueError("At most eight source cases are permitted")
    return selected


def prepare_same_text(capture_dir, annotation_path, score_dir, output):
    """Freeze all selected IDs and source hashes without loading or calling a model."""
    capture_dir, annotation_path, score_dir = map(
        lambda path: Path(path).resolve(), (capture_dir, annotation_path, score_dir)
    )
    capture = read_json(capture_dir / "manifest.json")
    source_plan = read_json(capture_dir / "protocol.json")
    if capture["status"] != "captured_awaiting_blind_draft" or capture["n_completed"] != 120:
        raise ValueError("A complete immutable 120-response capture is required")
    if file_sha256(capture_dir / "protocol.json") != capture["protocol_sha256"]:
        raise ValueError("Capture protocol identity mismatch")
    for name, digest in capture["output_sha256"].items():
        path = (capture_dir / name).resolve()
        if not path.is_relative_to(capture_dir) or file_sha256(path) != digest:
            raise ValueError("Capture artifact identity mismatch: " + name)
    specificity.verify_sources(source_plan)
    records = []
    for job in source_plan["jobs"]:
        path = capture_dir / "responses" / (job["example_id"] + ".json")
        record = read_json(path)
        if record["status"] != "captured":
            raise ValueError("Every original response must have successful capture")
        record["record_sha256"] = file_sha256(path)
        records.append(record)
    annotations = specificity.load_draft_annotations(annotation_path, records)
    selected = select_first_clean(records, annotations, source_plan["panel"])
    # Only after selection may scores be opened, solely to bind/comparison-check originals.
    scoring = read_json(score_dir / "score_protocol.json")
    if (
        scoring["capture_manifest_sha256"] != file_sha256(capture_dir / "manifest.json")
        or scoring["annotations_sha256"] != file_sha256(annotation_path)
        or scoring["frozen_draft_annotations"] != annotations
    ):
        raise ValueError("Draft scoring provenance does not match capture and annotations")
    original_scores = {
        row["example_id"]: row for row in read_json(score_dir / "draft_scores.json")["rows"]
    }
    probe_manifest, _arrays = specificity._probe(source_plan["probe_dir"])
    by_id = {record["example_id"]: record for record in records}
    for row in selected:
        original = original_scores[row["example_id"]]
        if any(original[key] != row[key] for key in row):
            raise ValueError("Selected draft classification differs from scored source")
        row["original_mean_raw_margin"] = original["mean_raw_margin"]
        row["source_record_sha256"] = by_id[row["example_id"]]["record_sha256"]
    protected = [
        capture_dir,
        score_dir,
        annotation_path,
        Path(source_plan["probe_dir"]),
        *map(Path, source_plan["source_sha256"]),
        Path(__file__),
    ]
    output = specificity._new_output(output, protected)
    sources = {
        **_snapshot(capture_dir),
        **_snapshot(score_dir),
        **source_plan["source_sha256"],
        str(annotation_path): file_sha256(annotation_path),
        str(Path(__file__).resolve()): file_sha256(__file__),
    }
    plan = {
        "schema": "mask_same_text_plan_v1",
        "created_at": utc_now(),
        "capture_dir": str(capture_dir),
        "score_dir": str(score_dir),
        "annotations_path": str(annotation_path),
        "source_sha256": sources,
        "probe_dir": source_plan["probe_dir"],
        "probe_manifest": probe_manifest,
        "model_settings": source_plan["model_settings"],
        "panel": source_plan["panel"],
        "selection": "First repeat-ordered primary clean ablated S=B=T, no length cap, per case; no margin used.",
        "selected": selected,
        "expected_forward_passes": 2 * len(selected),
        "expected_generations": 0,
        "token_policy": "Replay entire original generated ID sequence including EOS; score identical non-special IDs under each prefix.",
        "threshold": read_json(capture_dir / "calibration_threshold.json")["q90"],
        "needs_human_review": True,
        "label_audited": False,
        "limits": LIMITS,
    }
    _verify(plan)
    output.mkdir(parents=True)
    atomic_json(output / "protocol.json", plan)
    atomic_json(
        output / "manifest.json",
        {
            "schema": "mask_same_text_run_v1",
            "status": "prepared_no_model_calls",
            "protocol_sha256": file_sha256(output / "protocol.json"),
            "n_completed_cases": 0,
        },
    )
    return plan


def execute_same_text(output, backend, *, progress=print):
    output = Path(output).resolve()
    plan, manifest = read_json(output / "protocol.json"), read_json(output / "manifest.json")
    protocol_hash = file_sha256(output / "protocol.json")
    if (
        manifest["status"] != "prepared_no_model_calls"
        or manifest["protocol_sha256"] != protocol_hash
    ):
        raise ValueError("Use an unchanged fresh plan; no overwrite or implicit resume")
    _verify(plan)
    probe_manifest, arrays = specificity._probe(plan["probe_dir"])
    signature = capture_signature(backend.metadata, 8)
    if signature != probe_manifest["feature_signature"]:
        raise ValueError("Backend block-8 coordinate identity mismatch")
    if 8 not in backend.settings.capture_layers:
        raise ValueError("Backend must capture block 8")
    manifest.update(status="running", started_at=utc_now(), environment=environment_info())
    atomic_json(output / "manifest.json", manifest)
    cases = {case["id"]: case for case in plan["panel"]["cases"]}
    results = []
    try:
        for selected in plan["selected"]:
            _verify(plan)
            if file_sha256(output / "protocol.json") != protocol_hash:
                raise ValueError("Same-text protocol changed")
            identifier = selected["example_id"]
            capture_dir = Path(plan["capture_dir"])
            source = read_json(capture_dir / "responses" / (identifier + ".json"))
            generated = source["generation"]["generated_token_ids"]
            old_prompt = source["generation"]["prompt_token_ids"]
            if not generated or not old_prompt:
                raise ValueError("Exact original prompt and generated IDs are required")
            content_offsets = [
                index
                for index, token in enumerate(generated)
                if token not in set(backend.tokenizer.all_special_ids)
            ]
            selected_ids = [generated[index] for index in content_offsets]
            if not content_offsets or selected_ids != source["selected_token_ids"]:
                raise ValueError("Original non-special response selection differs")
            with np.load(capture_dir / source["feature_file"], allow_pickle=False) as archive:
                original_X = archive["X"].copy()
                if str(archive["feature_signature"].item()) != signature:
                    raise ValueError("Original features have incompatible coordinates")
            original_margin = specificity.mean_margin(original_X, arrays)
            if original_margin != selected["original_mean_raw_margin"]:
                raise ValueError("Original saved score does not match frozen features and fit")
            vectors, contexts = {}, {}
            for condition in ("ablated", "pressure"):
                messages = cases[selected["case_id"]][condition + "_messages"]
                prompt = backend.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
                if not isinstance(prompt, list) or not all(type(token) is int for token in prompt):
                    raise ValueError("Tokenizer must return exact flat integer prompt IDs")
                if condition == "ablated" and (
                    prompt != old_prompt or messages != source["messages"]
                ):
                    raise ValueError("Original ablated prefix does not exactly replay")
                positions = [len(prompt) + index for index in content_offsets]
                sequence = backend.torch.tensor([prompt + generated], device=backend.input_device)
                X = np.asarray(
                    backend._capture_token_rows(sequence, positions)["layer_8_tokens"],
                    dtype=np.float32,
                )
                if X.shape != original_X.shape or not np.isfinite(X).all():
                    raise ValueError("Invalid same-text activation matrix")
                vectors[condition] = X
                margin = specificity.mean_margin(X, arrays)
                contexts[condition] = {
                    "messages": messages,
                    "prompt_token_ids": prompt,
                    "full_sequence_token_ids": prompt + generated,
                    "selected_token_positions": positions,
                    "selected_token_ids": selected_ids,
                    "mean_raw_margin": margin,
                    "above_q90_neutral_reference": margin > plan["threshold"]["threshold"],
                }
            difference = vectors["ablated"].astype(np.float64) - original_X.astype(np.float64)
            row = {
                **selected,
                "generated_token_ids": generated,
                "generated_token_ids_sha256": canonical_hash(generated),
                "response_text": source["generation"]["text"],
                "response_text_utf8_sha256": hashlib.sha256(
                    source["generation"]["text"].encode("utf-8")
                ).hexdigest(),
                "selected_token_ids": selected_ids,
                "contexts": contexts,
                "ablated_replay_exact": bool(np.array_equal(vectors["ablated"], original_X)),
                "ablated_replay_max_absolute_error": float(np.max(np.abs(difference))),
                "ablated_replay_changed_values": int(np.count_nonzero(difference)),
                "ablated_replay_margin_minus_original": contexts["ablated"]["mean_raw_margin"]
                - original_margin,
                "pressure_minus_ablated": contexts["pressure"]["mean_raw_margin"]
                - contexts["ablated"]["mean_raw_margin"],
                "feature_signature": signature,
                "needs_human_review": True,
                "label_audited": False,
            }
            atomic_npz(
                output / "features" / (identifier + ".npz"),
                X_original=original_X,
                X_ablated=vectors["ablated"],
                X_pressure=vectors["pressure"],
                feature_signature=np.asarray(signature),
            )
            atomic_json(output / "cases" / (identifier + ".json"), row)
            results.append(row)
            manifest["n_completed_cases"] = len(results)
            atomic_json(output / "manifest.json", manifest)
            if progress:
                progress(f"{len(results)}/{len(plan['selected'])} same-text {identifier}")
        _verify(plan)
        differences = [row["pressure_minus_ablated"] for row in results]
        summary = {
            "n_cases": len(results),
            "n_forward_passes": 2 * len(results),
            "n_generations": 0,
            "pressure_minus_ablated": differences,
            "positive": sum(value > 0 for value in differences),
            "negative": sum(value < 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "mean_difference": float(np.mean(differences)) if differences else None,
            "median_difference": float(np.median(differences)) if differences else None,
            "all_ablated_replays_exact": all(row["ablated_replay_exact"] for row in results),
            "q90_neutral_reference": plan["threshold"],
            "q90_exceedances": {
                condition: sum(
                    row["contexts"][condition]["above_q90_neutral_reference"] for row in results
                )
                for condition in ("ablated", "pressure")
            },
            "needs_human_review": True,
            "label_audited": False,
            "limits": LIMITS,
        }
        atomic_json(output / "summary.json", summary)
        manifest.update(status="complete", source_unchanged=True)
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        manifest["finished_at"] = utc_now()
        manifest["output_sha256"] = {
            path.relative_to(output).as_posix(): file_sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        atomic_json(output / "manifest.json", manifest)
    return summary


def run_same_text(capture_dir, annotation_path, score_dir, output, backend, progress=print):
    prepare_same_text(capture_dir, annotation_path, score_dir, output)
    return execute_same_text(output, backend, progress=progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    plan_parser = commands.add_parser("plan", help="Freeze selection and sources; no model calls")
    for flag in ("capture-dir", "annotations", "score-dir", "output"):
        plan_parser.add_argument("--" + flag, type=Path, required=True)
    execute = commands.add_parser("capture", help="Execute an immutable prepared same-text plan")
    execute.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "plan":
        prepare_same_text(args.capture_dir, args.annotations, args.score_dir, args.output)
    else:
        plan = read_json(args.run_dir / "protocol.json")
        _verify(plan)
        backend = HFBackend(ModelSettings(**plan["model_settings"]))
        try:
            execute_same_text(args.run_dir, backend)
        finally:
            backend.close()


if __name__ == "__main__":
    main()
