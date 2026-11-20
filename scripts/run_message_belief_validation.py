"""Frozen prior-readout transfer to known-message fictional belief updates.

The phase remains ``known_message``. This is an evidence-channel transfer, not
prior-state wording validation or anticipation of an unknown future message.
All prefix features precede separate direct/structured behavioral generations.
"""

from __future__ import annotations

import argparse
import copy
import re
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from run_belief_diagnostic import (
    LAYERS,
    capture_prefix,
    parse_yes_no,
    query_messages,
    validate_layer_indices,
)
from run_belief_query_comparison import CONDITIONS, SIGNATURE_FIELDS, condition_messages, entity_ids

from latent_probes.backends import HFBackend
from latent_probes.belief_query_readout import _load_fit, _score, _snapshot, _write_predictions
from latent_probes.config import GenerationSettings, ModelSettings, load_config
from latent_probes.message_belief_panel import load_panel
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
BEHAVIOR_SETTINGS = {
    "direct": GenerationSettings(max_new_tokens=8, do_sample=False),
    "structured": GenerationSettings(max_new_tokens=160, do_sample=False),
}
SEED = 20261120
POLICY_FILES = (
    "scripts/run_message_belief_validation.py",
    "scripts/run_belief_diagnostic.py",
    "scripts/run_belief_query_comparison.py",
    "src/latent_probes/message_belief_panel.py",
    "src/latent_probes/belief_panel.py",
    "src/latent_probes/belief_query_panel.py",
    "src/latent_probes/belief_query_readout.py",
    "src/latent_probes/belief_readout.py",
    "src/latent_probes/backends.py",
    "src/latent_probes/config.py",
    "src/latent_probes/storage.py",
)


def _require(value, message):
    if not value:
        raise ValueError(message)


def _output_path(output, protected):
    output = Path(output).resolve()
    for source in protected:
        source = Path(source).resolve()
        _require(
            not (
                output == source
                or source.is_relative_to(output)
                or (source.is_dir() and output.is_relative_to(source))
            ),
            "Output must be disjoint from protected sources",
        )
    if output.exists():
        raise FileExistsError("Use a new output directory; no overwrite or implicit retry")
    return output


def structured_messages(case):
    """Same visible story and semantic question, with a declared format change."""
    messages = copy.deepcopy(case["messages"]["sender"])
    question = case["questions"]["nested"]
    suffix = "Answer only YES or NO."
    _require(question.endswith(suffix), "Unexpected nested-question format clause")
    question = question[: -len(suffix)].rstrip()
    messages[-1]["content"] += (
        "\n\nQuestion: "
        + question
        + "\nResolve the question using only the information available to the sender. "
        "Give four short lines identifying: INITIAL (the reader's initial observed place), "
        "MESSAGE (the place asserted by the note), RECEIPT (the receipt the sender trusts), "
        "and ATTRIBUTED (the place the sender therefore attributes to the reader). "
        "Then give a final line exactly FINAL: YES or FINAL: NO. "
        "Do not add text after that final line."
    )
    return messages


def parse_structured(text):
    """Require one final marker at the end and all four requested evidence fields."""
    if not isinstance(text, str):
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or len(re.findall(r"\bFINAL\s*:", text, re.IGNORECASE)) != 1:
        return None
    final = re.fullmatch(r"FINAL:\s*(YES|NO)[.!]?", lines[-1], re.IGNORECASE)
    fields = ("INITIAL", "MESSAGE", "RECEIPT", "ATTRIBUTED")
    if not final or not all(
        sum(bool(re.fullmatch(field + r":\s*\S.*", line, re.IGNORECASE)) for line in lines[:-1])
        == 1
        for field in fields
    ):
        return None
    if len(lines) != 5:
        return None
    return int(final.group(1).upper() == "YES")


def parse_leading_yes_no(text):
    """Secondary leading-token extraction; not semantic adjudication of the rest."""
    match = re.match(r"^\s*(YES|NO)\b", text, re.IGNORECASE)
    return int(match.group(1).upper() == "YES") if match else None


def parse_final_marker(text):
    """Secondary unique final marker, independent of evidence-field formatting."""
    if len(re.findall(r"\bFINAL\s*:", text, re.IGNORECASE)) != 1:
        return None
    match = re.search(r"(?:^|\n)\s*FINAL:\s*(YES|NO)[.!]?\s*\Z", text, re.IGNORECASE)
    return int(match.group(1).upper() == "YES") if match else None


def _freshness(panel, metadata, source_protocol):
    cases = panel["cases"]
    old_cases = [
        case
        for key in ("legacy_panel", "fresh_panel")
        for case in source_protocol.get(key, {}).get("cases", [])
    ]
    for key, known in (
        ("id", set(metadata["known_case_ids"]) | {c["id"] for c in old_cases}),
        (
            "template_id",
            set(metadata["known_template_ids"]) | {c["template_id"] for c in old_cases},
        ),
    ):
        _require(not ({c[key] for c in cases} & known), "Fresh " + key + " overlaps old bank")
    known_entities = set(metadata["known_entity_ids"]) | {
        entity for case in old_cases for entity in entity_ids(case)
    }
    _require(
        not ({entity for case in cases for entity in entity_ids(case)} & known_entities),
        "Fresh named entities overlap old bank",
    )
    known_text = set(metadata["known_message_hashes"]) | {
        canonical_hash("\n".join(m["content"] for m in condition_messages(c, v, q)))
        for c in old_cases
        for v in ("omniscient", "sender")
        for q in CONDITIONS
    }
    _require(
        not (
            {
                canonical_hash("\n".join(m["content"] for m in condition_messages(c, v, q)))
                for c in cases
                for v in ("omniscient", "sender")
                for q in CONDITIONS
            }
            & known_text
        ),
        "Fresh model input duplicates old bank",
    )


def prepare_run(panel_path, fit_dir, source_run, output):
    """Freeze all source/fit/panel bytes before any capture or generation."""
    panel_path, fit_dir, source_run = (
        Path(path).resolve() for path in (panel_path, fit_dir, source_run)
    )
    output = _output_path(
        output, [panel_path, fit_dir, source_run, *(ROOT / p for p in POLICY_FILES)]
    )
    panel = load_panel(panel_path)
    metadata, _parameters, _text, fitted_hashes = _load_fit(fit_dir)
    source = read_json(source_run / "manifest.json")
    _require(source.get("status") == "complete", "Original comparison must be complete")
    _require(
        fit_dir == source_run / "fits",
        "Use the fitted heads belonging to the specified original comparison",
    )
    source_hashes = _snapshot(source_run)
    _require(
        all(source_hashes.get(p) == digest for p, digest in source["output_sha256"].items()),
        "Original comparison output hash mismatch",
    )
    _freshness(panel, metadata, read_json(source_run / "protocol.json"))
    _require(len(panel["cases"]) == 64, "Expected frozen 64-case known-message bank")
    _require(
        all(c["phase"] == "known_message" and c["split"] == "fresh" for c in panel["cases"]),
        "Known-message phase must remain explicit",
    )
    protocol = {
        "schema": "known_message_frozen_readout_protocol_v1",
        "created_at": utc_now(),
        "panel": panel,
        "panel_path": str(panel_path),
        "panel_file_sha256": file_sha256(panel_path),
        "fit_dir": str(fit_dir),
        "fitted_artifact_sha256": fitted_hashes,
        "source_run": str(source_run),
        "source_artifact_sha256": source_hashes,
        "source_backend": source["backend"],
        "policy_sha256": {name: file_sha256(ROOT / name) for name in POLICY_FILES},
        "conditions": list(CONDITIONS),
        "layers": list(LAYERS),
        "primary_target": "nested",
        "primary_layer": "layer_16",
        "capture": "Last chat-prompt token before any answer; all captures and frozen readout scoring precede any behavior generation.",
        "no_refit": True,
        "no_hyperparameter_selection": True,
        "behavior": {
            "settings": {name: asdict(value) for name, value in BEHAVIOR_SETTINGS.items()},
            "seed": SEED,
            "view": "sender",
            "target": "nested",
            "structured_prompt_example": structured_messages(panel["cases"][0]),
            "parsing": "Direct full-string YES/NO; structured exactly four named field lines and a single final YES/NO marker. Invalid and capped outputs retained, never retried.",
            "secondary_parsing": "Predeclared format/content diagnostic: direct leading YES/NO; structured unique final FINAL: YES/NO line even if evidence-field formatting is invalid. Does not adjudicate preceding explanation. Length-capped outputs remain ineligible for both primary and secondary accuracy.",
        },
        "limits": [
            "Known-message fictional-state labels, not hidden LM beliefs or unknown-message anticipation.",
            "Frozen prior-trained heads are transferred; no known-message fit or layer selection.",
            "Structured elicitation uses the same visible evidence but adds a reasoning/output scaffold; it is elicited model behavior, not an additional observer with privileged evidence.",
            "Four authored wording groups and duplicate sender inputs are not independent broad-family validation.",
            "Agent-assisted authored materials require independent human semantic review.",
        ],
    }
    output.mkdir(parents=True)
    atomic_json(output / "protocol.json", protocol)
    atomic_json(
        output / "manifest.json",
        {
            "schema": "known_message_frozen_readout_run_v1",
            "status": "prepared_no_model_calls",
            "protocol_sha256": file_sha256(output / "protocol.json"),
            "n_capture_rows": 0,
            "n_behavior_rows": 0,
        },
    )
    return protocol


def verify_frozen(protocol, *, full_source=False):
    _require(
        file_sha256(protocol["panel_path"]) == protocol["panel_file_sha256"],
        "Frozen panel source changed",
    )
    for name, digest in protocol["policy_sha256"].items():
        _require(file_sha256(ROOT / name) == digest, "Frozen executed policy changed: " + name)
    _require(
        _snapshot(Path(protocol["fit_dir"])) == protocol["fitted_artifact_sha256"],
        "Frozen fitted artifacts changed",
    )
    if full_source:
        _require(
            _snapshot(Path(protocol["source_run"])) == protocol["source_artifact_sha256"],
            "Original comparison changed",
        )


def _validate_features(rows, features, metadata):
    _require(set(features) == {f"layer_{layer}" for layer in LAYERS}, "Unexpected layers")
    arrays = {}
    for layer, values in features.items():
        values = np.asarray(values)
        _require(
            values.shape == (len(rows), metadata["feature_widths"][layer])
            and values.dtype.kind == "f"
            and np.isfinite(values).all(),
            "Feature shape/alignment/finiteness mismatch",
        )
        arrays[layer] = values.astype(np.float64)
    pairs = defaultdict(list)
    for i, row in enumerate(rows):
        _require(row["phase"] == "known_message", "Do not mislabel known-message phase")
        if row["view"] == "sender":
            pairs[(row["condition"], row["messages_text"])].append(i)
    for indices in pairs.values():
        _require(
            len(indices) == 2 and {rows[i]["labels"]["reader"] for i in indices} == {0, 1},
            "Sender inputs must hide actual delivery in opposing-reader pairs",
        )
        _require(
            len({rows[i]["labels"]["nested"] for i in indices}) == 1
            and all(
                np.array_equal(values[indices[0]], values[indices[1]]) for values in arrays.values()
            ),
            "Identical sender inputs must have identical nested labels and features",
        )
    return arrays, len(pairs)


def score_frozen(rows, features, fit_dir, output):
    """Use saved transforms/weights without invoking the prior-only validator."""
    output = Path(output)
    _require(not output.exists(), "Use a new frozen-score output")
    metadata, parameters, texts, hashes = _load_fit(Path(fit_dir))
    arrays, hidden_pairs = _validate_features(rows, features, metadata)
    equal, cross, predictions = _score(rows, arrays, metadata["heads"], parameters, texts)
    controls = [
        item["metrics"]["fresh"]["accuracy"]
        for condition in equal.values()
        for item in condition["sender"]["reader"].values()
    ] + [
        item["metrics"]["fresh"]["accuracy"]
        for item in cross
        if item["view"] == "sender" and item["target"] == "reader"
    ]
    _require(
        controls and all(value == 0.5 for value in controls),
        "Hidden actual-reader control must be exactly 50%",
    )
    _require(_snapshot(Path(fit_dir)) == hashes, "Scoring changed frozen fitted artifacts")
    output.mkdir(parents=True)
    _write_predictions(output / "predictions.csv", predictions)
    report = {
        "schema": "known_message_frozen_readout_report_v1",
        "phase": "known_message",
        "no_refit": True,
        "no_hyperparameter_selection": True,
        "equal_condition": equal,
        "cross_condition": cross,
        "n_prediction_rows": len(predictions),
        "hidden_reader_pairs": hidden_pairs,
        "hidden_reader_control_metric_cells": len(controls),
        "fitted_input_sha256": hashes,
        "interpretation": "Prior-state readouts transferred unchanged to a known-message update task; source-condition scaling and weights retained for every transfer.",
    }
    atomic_json(output / "report.json", report)
    return report


def summarize_behavior(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["style"]].append(record)
    result = {}
    for style, selected in groups.items():
        result[style] = {
            "n_rows": len(selected),
            "n_unique_prompts": len({row["prompt_sha256"] for row in selected}),
            "correct": sum(row["prediction"] == row["gold"] for row in selected),
            "invalid": sum(row["prediction"] is None for row in selected),
            "length_capped": sum(row["length_capped"] for row in selected),
            "accuracy_including_invalid": sum(row["prediction"] == row["gold"] for row in selected)
            / len(selected),
            "secondary_content_correct": sum(
                row["secondary_prediction"] == row["gold"] for row in selected
            ),
            "secondary_content_invalid": sum(
                row["secondary_prediction"] is None for row in selected
            ),
            "secondary_content_accuracy_including_invalid": sum(
                row["secondary_prediction"] == row["gold"] for row in selected
            )
            / len(selected),
            "secondary_interpretation": "Predeclared content extraction independent of strict format; no explanation-validity or internal-reasoning claim.",
            "per_template": {
                template: {
                    "n": sum(row["template_id"] == template for row in selected),
                    "correct": sum(
                        row["template_id"] == template and row["prediction"] == row["gold"]
                        for row in selected
                    ),
                    "invalid": sum(
                        row["template_id"] == template and row["prediction"] is None
                        for row in selected
                    ),
                    "secondary_content_correct": sum(
                        row["template_id"] == template
                        and row["secondary_prediction"] == row["gold"]
                        for row in selected
                    ),
                    "secondary_content_invalid": sum(
                        row["template_id"] == template and row["secondary_prediction"] is None
                        for row in selected
                    ),
                }
                for template in sorted({row["template_id"] for row in selected})
            },
        }
    return result


def capture_run(output, backend, *, progress=print):
    output = Path(output).resolve()
    protocol, manifest = read_json(output / "protocol.json"), read_json(output / "manifest.json")
    digest = file_sha256(output / "protocol.json")
    _require(
        manifest["status"] == "prepared_no_model_calls" and digest == manifest["protocol_sha256"],
        "Need an unchanged prepared plan; no automatic retry",
    )
    verify_frozen(protocol, full_source=True)
    for key in SIGNATURE_FIELDS:
        _require(
            backend.metadata.get(key) == protocol["source_backend"].get(key),
            "Backend signature differs: " + key,
        )
    _require(list(backend.settings.capture_layers) == list(LAYERS), "Use three frozen layers")
    manifest.update(
        status="running",
        started_at=utc_now(),
        backend=backend.metadata,
        environment=environment_info(),
    )
    atomic_json(output / "manifest.json", manifest)
    rows, captures, repeats, behavior = [], [], [], []
    matrices = {f"layer_{layer}": [] for layer in LAYERS}
    cache, repeated, generation_cache = {}, set(), {}
    final_ids = set()
    try:
        for index, case in enumerate(protocol["panel"]["cases"]):
            verify_frozen(protocol)
            _require(file_sha256(output / "protocol.json") == digest, "Protocol changed")
            for view in ("omniscient", "sender"):
                identifier = case["id"] + "__" + view
                for condition in CONDITIONS:
                    messages = condition_messages(case, view, condition)
                    key = canonical_hash(messages)
                    reused = key in cache
                    feature_path = Path("prefix_features") / (key + ".npz")
                    if not reused:
                        cache[key] = capture_prefix(backend, messages)
                        atomic_npz(output / feature_path, **cache[key][1])
                    ids, vectors = cache[key]
                    final_ids.add(ids[-1])
                    repeat_key = (view, condition)
                    if repeat_key not in repeated:
                        second_ids, second_vectors = capture_prefix(backend, messages)
                        check = {
                            "view": view,
                            "condition": condition,
                            "ids_exact": ids == second_ids,
                            "layers_exact": {
                                name: bool(np.array_equal(value, second_vectors[name]))
                                for name, value in vectors.items()
                            },
                        }
                        _require(
                            check["ids_exact"] and all(check["layers_exact"].values()),
                            "Same-input capture is not repeatable",
                        )
                        repeats.append(check)
                        repeated.add(repeat_key)
                    if not captures:
                        manifest["layer_index_check"] = validate_layer_indices(backend, ids)
                    rows.append(
                        {
                            "example_id": identifier,
                            "case_id": case["id"],
                            "template_id": case["template_id"],
                            "split": "fresh",
                            "phase": "known_message",
                            "view": view,
                            "condition": condition,
                            "messages_text": "\n".join(m["content"] for m in messages),
                            "entity_ids": entity_ids(case),
                            "labels": copy.deepcopy(case["labels"]),
                        }
                    )
                    for name, vector in vectors.items():
                        matrices[name].append(vector.copy())
                    captures.append(
                        {
                            "example_id": identifier,
                            "condition": condition,
                            "messages_sha256": key,
                            "feature_file": feature_path.as_posix(),
                            "feature_sha256": file_sha256(output / feature_path),
                            "prompt_token_ids": ids,
                            "selected_position": len(ids) - 1,
                            "selected_token_id": ids[-1],
                            "generated_answer_tokens_supplied": 0,
                            "reused_identical_capture": reused,
                        }
                    )
                    atomic_json(output / "captures.json", captures)
                    manifest["n_capture_rows"] = len(rows)
            atomic_json(output / "manifest.json", manifest)
            if progress:
                progress(f"Known-message captures {index + 1}/64", flush=True)
        _require(len(final_ids) == 1, "Final capture-token identity differs by condition")
        features = {name: np.asarray(value, dtype=np.float32) for name, value in matrices.items()}
        atomic_json(output / "rows.json", rows)
        atomic_npz(output / "features.npz", **features)
        atomic_json(output / "repeats.json", repeats)
        score_frozen(rows, features, protocol["fit_dir"], output / "frozen_evaluation")
        manifest.update(
            prefix_capture_complete_at=utc_now(),
            n_unique_capture_prefixes=len(cache),
            last_token_id=next(iter(final_ids)),
            n_repeat_checks=len(repeats),
        )
        atomic_json(output / "manifest.json", manifest)
        for style in BEHAVIOR_SETTINGS:
            for index, case in enumerate(protocol["panel"]["cases"]):
                verify_frozen(protocol)
                _require(file_sha256(output / "protocol.json") == digest, "Protocol changed")
                messages = (
                    query_messages(case, "sender", "nested")
                    if style == "direct"
                    else structured_messages(case)
                )
                key = canonical_hash([style, messages])
                reused = key in generation_cache
                if not reused:
                    request_path = output / "generated_responses" / (key + ".json")
                    atomic_json(
                        request_path,
                        {
                            "style": style,
                            "messages": messages,
                            "status": "started",
                            "created_at": utc_now(),
                        },
                    )
                    generated = backend.generate(
                        messages, BEHAVIOR_SETTINGS[style], SEED, capture=False
                    )
                    if style == "direct":
                        _require(
                            generated.prompt_token_ids == cache[canonical_hash(messages)][0],
                            "Direct-answer prompt differs from captured nested-question prefix",
                        )
                    generation_cache[key] = generated.record()
                    atomic_json(
                        request_path,
                        {
                            "style": style,
                            "messages": messages,
                            "status": "generated",
                            "generation": generated.record(),
                        },
                    )
                generated_record = generation_cache[key]
                text = generated_record["text"]
                capped = generated_record["finish_reason"] == "length"
                parsed = parse_yes_no(text) if style == "direct" else parse_structured(text)
                secondary = (
                    parse_leading_yes_no(text) if style == "direct" else parse_final_marker(text)
                )
                record = {
                    "case_id": case["id"],
                    "template_id": case["template_id"],
                    "phase": "known_message",
                    "view": "sender",
                    "target": "nested",
                    "style": style,
                    "prompt_sha256": key,
                    "messages": messages,
                    "generation": generated_record,
                    "length_capped": capped,
                    "parsed_prediction": parsed,
                    "prediction": None if capped else parsed,
                    "secondary_parsed_prediction": secondary,
                    "secondary_prediction": None if capped else secondary,
                    "gold": case["labels"]["nested"],
                    "reused_identical_generation": reused,
                    "needs_human_review": True,
                }
                behavior.append(record)
                atomic_json(output / "behavior" / (case["id"] + "__" + style + ".json"), record)
                manifest["n_behavior_rows"] = len(behavior)
                atomic_json(output / "manifest.json", manifest)
                if progress:
                    progress(f"Known-message {style} behavior {index + 1}/64", flush=True)
        atomic_json(output / "behavior_report.json", summarize_behavior(behavior))
        verify_frozen(protocol, full_source=True)
        manifest.update(
            status="complete",
            source_unchanged=True,
            frozen_fits_unchanged=True,
            n_unique_behavior_generations=len(generation_cache),
        )
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        manifest["finished_at"] = utc_now()
        manifest["output_sha256"] = {
            name: value for name, value in _snapshot(output).items() if name != "manifest.json"
        }
        atomic_json(output / "manifest.json", manifest)
    return manifest


def run_message_validation(panel_path, fit_dir, source_run, output, backend, *, progress=print):
    prepare_run(panel_path, fit_dir, source_run, output)
    return capture_run(output, backend, progress=progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = prepare_run(args.panel, args.fit_dir, args.source_run, args.output)
    original = protocol["source_backend"]
    settings = replace(
        load_config(args.config).model,
        model_id=original["model_id"],
        revision=original["resolved_revision"],
        dtype=original["dtype"],
        quantization=original["quantization"],
        capture_layers=list(LAYERS),
        local_files_only=True,
        max_input_tokens=4096,
    )
    backend = HFBackend(ModelSettings(**asdict(settings)))
    try:
        capture_run(args.output, backend)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
