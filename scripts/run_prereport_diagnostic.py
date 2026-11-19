"""Exploratory before-report readout of already observed reader decisions.

Every fitted activation comes from a prefix-only forward pass. Saved responses
are used only in separately identified numerical controls and never as readout
features or text inputs. This replays known outcomes, not a prospective study.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from latent_probes.apollo import capture_signature
from latent_probes.backends import HFBackend
from latent_probes.config import load_config
from latent_probes.outcome_readout import (
    REQUIRED_COLUMNS,
    evaluate_frozen_readout,
    fit_evaluate_readout,
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

BLOCK = 8
REPORT_PLACEHOLDER = "[REPORT NOT YET GENERATED]"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _snapshot(directory):
    return {
        path.relative_to(directory).as_posix(): file_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _checked_record(path):
    record = read_json(path)
    digest = record.pop("record_sha256", None)
    _require(digest == canonical_hash(record), f"Source record hash mismatch: {path.name}")
    record["record_sha256"] = digest
    return record


def _token_ids(value, name):
    _require(
        isinstance(value, list) and bool(value) and all(type(x) is int and x >= 0 for x in value),
        f"{name} must contain nonnegative integer token IDs",
    )
    return value


def _source_records(source):
    manifest, plan = read_json(source / "manifest.json"), read_json(source / "plan.json")
    _require(manifest.get("status") == "complete", "Source run must be complete")
    _require(manifest.get("plan_sha256") == canonical_hash(plan), "Source plan hash mismatch")
    frozen = read_json(source / "frozen_readout.json")["artifacts"]
    for name, digest in frozen.items():
        path = (source / "neutral_readout" / name).resolve()
        _require(path.is_relative_to(source / "neutral_readout"), "Unsafe original artifact path")
        _require(file_sha256(path) == digest, f"Original frozen artifact hash mismatch: {name}")
    jobs = plan["sender_jobs"]
    identifiers = [job["example_id"] for job in jobs]
    _require(bool(jobs) and len(set(identifiers)) == len(jobs), "Duplicate/empty source jobs")
    _require(
        {p.stem for p in (source / "reports").glob("*.json")} == set(identifiers),
        "Source report set differs from plan",
    )
    _require(
        {p.stem for p in (source / "senders").glob("*.json")} == set(identifiers),
        "Source sender set differs from plan",
    )
    rows = []
    for job in jobs:
        identifier = job["example_id"]
        _require(Path(identifier).name == identifier and identifier not in {".", ".."}, "Unsafe ID")
        report = _checked_record(source / "reports" / (identifier + ".json"))
        sender = _checked_record(source / "senders" / (identifier + ".json"))
        for key in (
            "example_id",
            "case_id",
            "family",
            "objective",
            "split",
            "context",
            "messages",
            "seed",
        ):
            _require(report[key] == sender[key] == job[key], f"Source job/record mismatch: {key}")
        _require(
            all(message["role"] in {"system", "user"} for message in report["messages"]),
            "Sender prefix must contain only pre-report system/user messages",
        )
        _require(
            report["generation"] == sender["generation"], "Source sender/report generation mismatch"
        )
        feature = (source / report["feature_file"]).resolve()
        _require(feature.is_relative_to(source), "Unsafe source feature path")
        _require(
            report["feature_file"] == sender["feature_file"]
            and report["feature_sha256"] == sender["feature_sha256"] == file_sha256(feature),
            "Source feature artifact hash mismatch",
        )
        _token_ids(report["generation"]["prompt_token_ids"], "Saved prompt")
        _token_ids(report["generation"]["generated_token_ids"], "Saved response")
        _require(
            report["decision"] == parse_decision(report["reader_generation"]["text"]),
            "Source observed decision differs from reader output",
        )
        rows.append(report)
    # Match the original generation order: all neutral jobs, then favorable jobs.
    rows.sort(key=lambda row: (row["objective"] != "neutral", identifiers.index(row["example_id"])))
    return manifest, plan, rows


def _capture_ids(backend, ids, position, layer):
    maximum = getattr(backend.model.config, "max_position_embeddings", None)
    _require(not maximum or len(ids) <= maximum, "Saved sequence exceeds model context")
    sequence = backend.torch.tensor([ids], device=backend.input_device)
    matrix = np.asarray(backend._capture_token_rows(sequence, [position])[f"layer_{layer}_tokens"])
    _require(
        matrix.ndim == 2 and matrix.shape[0] == 1 and matrix.shape[1] > 0,
        "Bad prefix capture shape",
    )
    _require(np.isfinite(matrix).all(), "Nonfinite prefix capture")
    return matrix.astype(np.float32, copy=True)


def prefix_vector(backend, messages, saved_prompt_ids, layer=BLOCK):
    """Capture last prompt token; this API accepts no response or outcome data."""
    _token_ids(saved_prompt_ids, "Saved prompt")
    encoded = backend.tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    rendered_ids = encoded["input_ids"][0].tolist()
    _require(rendered_ids == saved_prompt_ids, "Re-rendered prefix differs from saved prompt IDs")
    _require(
        len(saved_prompt_ids) <= backend.settings.max_input_tokens,
        "Saved prompt exceeds input limit",
    )
    vector = _capture_ids(backend, saved_prompt_ids, len(saved_prompt_ids) - 1, layer)
    return vector, {
        "supplied_token_count": len(saved_prompt_ids),
        "supplied_token_ids_sha256": canonical_hash(saved_prompt_ids),
        "selected_token_position": len(saved_prompt_ids) - 1,
        "generated_response_tokens_supplied": 0,
        "sequence_scope": "saved_prompt_only",
    }


def select_numerical_examples(rows):
    """First/last sorted case per family, first neutral context; no outcomes used."""
    selected = []
    for family in sorted({r["family"] for r in rows}):
        neutral = [r for r in rows if r["family"] == family and r["objective"] == "neutral"]
        cases = sorted({r["case_id"] for r in neutral})
        _require(bool(cases), "Each family needs a neutral example")
        for case in sorted({cases[0], cases[-1]}):
            selected.append(
                min(
                    (r for r in neutral if r["case_id"] == case),
                    key=lambda r: (r["context"], r["example_id"]),
                )
            )
    return selected


def bf16_neighbor_check(reference, observed):
    """One adjacent BF16 value, including asymmetric spacing around powers of two.

    Same integer-neighbor policy as validate_full_baseline.py, independently
    embedded here so standalone execution needs no sibling-script import.
    """
    reference, observed = np.asarray(reference, dtype=np.float32), np.asarray(observed)
    _require(reference.shape == observed.shape and reference.size > 0, "Comparison shape mismatch")
    _require(np.isfinite(reference).all() and np.isfinite(observed).all(), "Nonfinite comparison")
    raw = np.ascontiguousarray(reference).view(np.uint32)
    _require(not np.any(raw & np.uint32(0xFFFF)), "Reference is not on the BF16 grid")
    bits = raw >> np.uint32(16)
    lower, upper = bits.copy(), bits.copy()
    positive, negative, zero = reference > 0, reference < 0, reference == 0
    lower[positive], upper[positive] = bits[positive] - 1, bits[positive] + 1
    lower[negative], upper[negative] = bits[negative] + 1, bits[negative] - 1
    lower[zero], upper[zero] = 0x8001, 0x0001
    low = (lower.astype(np.uint32) << np.uint32(16)).view(np.float32)
    high = (upper.astype(np.uint32) << np.uint32(16)).view(np.float32)
    outside = (observed < low) | (observed > high)
    return {
        "allowed_bf16_neighbor_steps": 1,
        "outside_neighbor_count": int(outside.sum()),
        "exact_fraction": float(np.mean(reference == observed)),
        "max_absolute_error": float(np.max(np.abs(observed.astype(np.float64) - reference))),
        "passed": not bool(outside.any()),
    }


def numerical_controls(backend, row, prefix, layer=BLOCK):
    """Saved responses enter these controls only; their vectors are never fitted."""
    prompt, future = row["generation"]["prompt_token_ids"], row["generation"]["generated_token_ids"]
    position = len(prompt) - 1
    repeat_prefix = _capture_ids(backend, prompt, position, layer)
    full_ids = prompt + future
    full = _capture_ids(backend, full_ids, position, layer)
    repeat_full = _capture_ids(backend, full_ids, position, layer)
    prefix_stable, full_stable = (
        np.array_equal(prefix, repeat_prefix),
        np.array_equal(full, repeat_full),
    )
    parity = bf16_neighbor_check(prefix, full)
    mutations = []
    vocab_size = backend.model.config.vocab_size
    _require(type(vocab_size) is int and vocab_size > 1, "Need model vocabulary size")
    for name, index in (
        ("first_future_token", len(prompt)),
        ("last_future_token", len(full_ids) - 1),
    ):
        changed = list(full_ids)
        changed[index] = (changed[index] + 1) % vocab_size
        observed = _capture_ids(backend, changed, position, layer)
        mutations.append(
            {
                "name": name,
                "mutated_position": index,
                "same_sequence_shape": len(changed) == len(full_ids),
                "changed_token_count": sum(a != b for a, b in zip(changed, full_ids, strict=True)),
                "full_input_token_ids_sha256": canonical_hash(changed),
                "unchanged_full_input_repeat_passed": bool(full_stable),
                "exact_fraction": float(np.mean(full == observed)),
                "passed": bool(full_stable and np.array_equal(full, observed)),
                "status": "evaluated" if full_stable else "inconclusive_unstable_same_input_repeat",
            }
        )
    passed = bool(
        prefix_stable and full_stable and parity["passed"] and all(m["passed"] for m in mutations)
    )
    return {
        "example_id": row["example_id"],
        "status": "passed" if passed else "numerical_inconclusive",
        "prefix_only_repeat_exact": bool(prefix_stable),
        "full_sequence_repeat_exact": bool(full_stable),
        "prefix_vs_full_one_neighbor": parity,
        "full_sequence_token_count": len(full_ids),
        "full_sequence_token_ids_sha256": canonical_hash(full_ids),
        "future_mutations": mutations,
        "full_sequence_vectors_exported_as_features": False,
        "interpretation": "Engineering comparison only; failure does not establish causal response leakage.",
    }


def _export(rows, vectors, destination, signature, provenance):
    destination.mkdir(parents=True)
    atomic_npz(
        destination / "features.npz",
        X=np.stack([vectors[r["example_id"]] for r in rows]),
        example_ids=np.asarray([r["example_id"] for r in rows]),
        feature_signature=np.asarray(signature),
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
    )
    with (destination / "labels.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for row in rows:
            exported = {
                **{key: row[key] for key in REQUIRED_COLUMNS[:6]},
                "prompt_text": json.dumps(row["messages"], ensure_ascii=False),
                "report_text": REPORT_PLACEHOLDER,
            }
            _require(
                exported["report_text"] == REPORT_PLACEHOLDER, "Report leaked into pre-report text"
            )
            writer.writerow(exported)
    return destination / "features.npz", destination / "labels.csv"


def run_prereport_diagnostic(
    source_run,
    output_dir,
    backend,
    *,
    fit_function=fit_evaluate_readout,
    score_function=evaluate_frozen_readout,
    progress=print,
):
    """Reuse a loaded backend, preserve source artifacts, and never close it."""
    source, output = Path(source_run).resolve(), Path(output_dir).resolve()
    _require(
        not output.is_relative_to(source) and not source.is_relative_to(output),
        "Output must be separate from source run",
    )
    _require(not output.exists(), "Use a fresh diagnostic directory")
    hashes = _snapshot(source)
    source_manifest, _plan, rows = _source_records(source)
    _require(
        list(backend.settings.capture_layers) == [BLOCK], "Exactly decoder block 8 is required"
    )
    signature_base = capture_signature(backend.metadata, BLOCK)
    _require(
        signature_base == capture_signature(source_manifest["backend"], BLOCK),
        "Backend/source coordinate mismatch",
    )
    signature = json.dumps(
        {
            **json.loads(signature_base),
            "pooling": "last_prompt_token",
            "prediction_time": "before_report",
            "generated_response_tokens_supplied": 0,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    selected = select_numerical_examples(rows)
    output.mkdir(parents=True)
    manifest = {
        "schema": "prereport_diagnostic_v1",
        "status": "running",
        "created_at": utc_now(),
        "analysis_status": "exploratory_replay_of_already_observed_outcomes",
        "source_run": str(source),
        "source_input_sha256": hashes,
        "script_sha256": file_sha256(Path(__file__)),
        "environment": environment_info(),
        "feature_signature": signature,
        "numerical_selection_rule": "first and last sorted case per family, first sorted neutral context; no labels/features used",
        "selected_example_ids": [r["example_id"] for r in selected],
        "n_captured": 0,
        "limitations": [
            "Exploratory replay: these natural reports and reader outcomes were already observed before this diagnostic was designed.",
            "Every fitted activation is computed from saved prompt tokens only; full-response forwards are separate numerical controls.",
            "New before-report weights are fitted; after-report weights are never applied to these coordinates.",
            "Text controls contain pre-report prompts and a constant placeholder only; report-only is a prior baseline.",
            "Case-held-out development with shared families and repeated contexts; no population confidence or belief/deception inference.",
            "Original reader-control failures, report factual errors, truncations and INVALID labels are retained without regeneration.",
        ],
    }
    atomic_json(output / "manifest.json", manifest)
    vectors, capture_records, controls = {}, [], []
    try:
        for row in rows:
            matrix, details = prefix_vector(
                backend, row["messages"], row["generation"]["prompt_token_ids"]
            )
            vectors[row["example_id"]] = matrix[0].copy()
            feature_file = Path("prefix_features") / (row["example_id"] + ".npz")
            atomic_npz(output / feature_file, X=matrix, feature_signature=np.asarray(signature))
            capture_records.append(
                {
                    "example_id": row["example_id"],
                    "source_record_sha256": row["record_sha256"],
                    "feature_file": feature_file.as_posix(),
                    "feature_sha256": file_sha256(output / feature_file),
                    **details,
                }
            )
            manifest["n_captured"] = len(vectors)
            atomic_json(output / "captures.json", capture_records)
            atomic_json(output / "manifest.json", manifest)
            if progress:
                progress(
                    f"Prefix-only feature {len(vectors)}/{len(rows)}: {row['example_id']}",
                    flush=True,
                )
        for row in selected:
            try:
                control = numerical_controls(backend, row, vectors[row["example_id"]][None, :])
            except Exception as error:
                control = {
                    "example_id": row["example_id"],
                    "status": "numerical_inconclusive",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "full_sequence_vectors_exported_as_features": False,
                    "interpretation": "Numerical control could not be evaluated; valid prefix-only features are retained.",
                }
            controls.append(control)
            atomic_json(output / "numerical_controls.json", controls)
        provenance = {
            "source_manifest_sha256": hashes["manifest.json"],
            "source_record_sha256": {r["example_id"]: r["record_sha256"] for r in rows},
            "analysis_status": manifest["analysis_status"],
            "prediction_time": "before_report",
            "supplied_future_tokens": 0,
            "text_report_placeholder": REPORT_PLACEHOLDER,
        }
        neutral = [r for r in rows if r["objective"] == "neutral"]
        features, labels = _export(
            neutral, vectors, output / "neutral_dataset", signature, provenance
        )
        fitted = fit_function(
            features, labels, output / "neutral_readout", allow_two_class_diagnostic=True
        )
        frozen = _snapshot(output / "neutral_readout")
        atomic_json(
            output / "frozen_readout.json",
            {
                "frozen_at": utc_now(),
                "artifacts": frozen,
                "analysis_status": "exploratory_weights_frozen_before_scoring; outcomes_were_already_observed",
            },
        )
        features, labels = _export(
            [r for r in rows if r["split"] == "test"],
            vectors,
            output / "test_dataset",
            signature,
            provenance,
        )
        if fitted["status"] != "not_trainable":
            score_function(features, labels, output / "neutral_readout", output / "frozen_transfer")
        _require(
            _snapshot(output / "neutral_readout") == frozen,
            "New before-report fitted artifacts changed during scoring",
        )
        _require(_snapshot(source) == hashes, "Original source run changed during diagnostic")
        manifest.update(
            status="complete",
            source_artifacts_unchanged=True,
            neutral_fit_status=fitted["status"],
            numerical_status=(
                "passed"
                if all(c["status"] == "passed" for c in controls)
                else "numerical_inconclusive"
            ),
            n_numerical_examples=len(controls),
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    settings = replace(config.model, local_files_only=True)
    backend = HFBackend(settings)
    try:
        result = run_prereport_diagnostic(args.run, args.output, backend)
    finally:
        backend.close()
    print(json.dumps({"status": result["status"], "numerical_status": result["numerical_status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
