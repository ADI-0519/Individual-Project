"""Independent, CPU-only audit of a saved instruction-condition baseline.

No production extraction, fitting, scoring or metric helpers are imported.
This audits saved evidence and instruction-label diagnostics, not deception.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

TARGET = "instructed_honesty_condition"
RTOL = 1e-10
ATOL = 1e-12
STATIONARITY_BOUND = 1e-3
BASE_SEED = 20261119


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def _inside(root: Path, relative: str) -> Path:
    _require(isinstance(relative, str) and bool(relative), "Missing artifact path")
    candidate = (root / relative).resolve()
    _require(candidate.is_relative_to(root), "Artifact path escapes the capture directory")
    return candidate


def _signature(raw: Any) -> str:
    value = json.loads(raw)
    _require(isinstance(value, dict) and bool(value), "Invalid feature signature")
    return _canonical(value)


def manual_auroc(labels: Any, scores: Any) -> float | None:
    """Pairwise Mann-Whitney AUROC, with half credit for equal scores."""
    y, values = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    _require(y.ndim == values.ndim == 1 and len(y) == len(values), "AUROC shape mismatch")
    _require(len(y) > 0 and np.isin(y, [0, 1]).all(), "AUROC requires binary labels")
    _require(np.isfinite(values).all(), "AUROC requires finite scores")
    positive, negative = values[y == 1], values[y == 0]
    if not len(positive) or not len(negative):
        return None
    wins = 0.0
    for score in positive:
        wins += np.count_nonzero(score > negative)
        wins += 0.5 * np.count_nonzero(score == negative)
    return float(wins / (len(positive) * len(negative)))


def training_statistics(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Population statistics; account explicitly for numerically constant columns."""
    values = np.asarray(X, dtype=np.float64)
    _require(values.ndim == 2 and all(values.shape), "Training matrix must be nonempty")
    mean = values.mean(axis=0)
    variance = np.mean((values - mean) ** 2, axis=0)
    _require(np.isfinite(mean).all() and np.isfinite(variance).all(), "Nonfinite statistics")
    std = np.sqrt(variance)
    eps = np.finfo(np.float64).eps
    uncertainty = len(values) * eps * variance + (len(values) * mean * eps) ** 2
    scale = std.copy()
    scale[variance <= uncertainty] = 1.0
    return mean, std, scale


def _metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        _require(
            len(fields) == len(set(fields))
            and {"example_id", "pair_id", "split", "label", "label_target"} <= set(fields),
            "Invalid metadata columns",
        )
        rows = list(reader)
    _require(bool(rows) and all(None not in row for row in rows), "Invalid metadata rows")
    return rows


def _load_evidence(capture: Path, probe: Path) -> dict[str, Any]:
    capture, probe = capture.resolve(), probe.resolve()
    hashes: dict[str, str] = {}

    def checked(path: Path, expected: str | None = None) -> str:
        digest = _file_hash(path)
        hashes[str(path)] = digest
        if expected is not None:
            _require(digest == expected, f"Artifact hash mismatch: {path}")
        return digest

    manifest_path, probe_manifest_path = capture / "manifest.json", probe / "manifest.json"
    manifest, fitted = _object(manifest_path), _object(probe_manifest_path)
    checked(manifest_path)
    checked(probe_manifest_path)
    _require(manifest.get("status") == "complete", "Capture status must be complete")
    _require(manifest.get("stage") == "apollo_tokens", "Expected an Apollo token capture")
    _require(fitted.get("schema") == "apollo_probe_v1", "Unsupported probe schema")
    _require(fitted.get("status", "complete") == "complete", "Probe status is incomplete")
    _require(fitted.get("label_target") == TARGET, "Probe is not an instruction-label readout")
    features_path, labels_path = capture / "features.npz", capture / "metadata.csv"
    for path, key in ((features_path, "features_sha256"), (labels_path, "metadata_sha256")):
        _require(isinstance(manifest.get(key), str), f"Capture missing {key}")
        checked(path, manifest[key])
    _require(
        fitted.get("features_sha256") == hashes[str(features_path)], "Probe feature hash differs"
    )
    _require(fitted.get("labels_sha256") == hashes[str(labels_path)], "Probe label hash differs")

    config_path, examples_path = capture / "config.json", capture / "examples.jsonl"
    config = _object(config_path)
    checked(config_path)
    checked(examples_path)
    examples = [
        json.loads(line)
        for line in examples_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _require(bool(examples) and all(isinstance(row, dict) for row in examples), "Invalid examples")
    identity = manifest.get("identity", {})
    _require(identity.get("config_hash") == _digest(config), "Config identity mismatch")
    _require(identity.get("examples_hash") == _digest(examples), "Example identity mismatch")
    preparation_path = capture / "preparation.json"
    preparation = _object(preparation_path) if preparation_path.exists() else None
    if preparation is not None:
        checked(preparation_path)
    _require(
        identity.get("preparation_hash") == _digest(preparation), "Preparation identity mismatch"
    )

    with np.load(features_path, allow_pickle=False) as archive:
        _require(
            {"X", "example_ids", "feature_signature"} <= set(archive.files), "Missing packed arrays"
        )
        X = np.asarray(archive["X"], dtype=np.float64)
        ids = np.asarray(archive["example_ids"])
        signature_array = np.asarray(archive["feature_signature"])
    _require(X.ndim == 2 and all(X.shape) and np.isfinite(X).all(), "Invalid packed matrix")
    _require(ids.ndim == 1 and ids.dtype.kind == "U" and len(ids) == len(X), "Invalid packed IDs")
    _require(
        signature_array.shape == () and signature_array.dtype.kind == "U", "Invalid signature array"
    )
    signature = _signature(str(signature_array.item()))
    _require(
        signature == _signature(manifest.get("feature_signature", "")), "Capture signature differs"
    )
    _require(
        signature == _signature(fitted.get("feature_signature", "")), "Probe signature differs"
    )

    metadata = _metadata(labels_path)
    example_ids = [row.get("example_id") for row in examples]
    _require(
        all(
            isinstance(identifier, str) and identifier and identifier == identifier.strip()
            for identifier in example_ids
        )
        and len(set(example_ids)) == len(example_ids),
        "Example IDs must be unique, nonempty and trimmed",
    )
    _require([row["example_id"] for row in metadata] == example_ids, "Metadata order/IDs differ")
    by_id = {row["example_id"]: row for row in metadata}
    for example in examples:
        row = by_id[example["example_id"]]
        _require(
            example.get("split") in {"train", "validation"},
            "Only training/validation captures are audited",
        )
        _require(
            type(example.get("label")) is int and example["label"] in {0, 1},
            "Invalid example label",
        )
        _require(example.get("label_target") == TARGET, "Unexpected example target")
        _require(
            example.get("mode") == "fixed" and example.get("trim_last_words") == 5,
            "Unexpected mask profile",
        )
        _require(isinstance(example.get("response"), str), "Missing fixed response")
        _require(
            isinstance(example.get("messages"), list) and bool(example["messages"]),
            "Missing prompt",
        )
        for key in ("pair_id", "split", "label", "label_target"):
            _require(row[key] == str(example[key]), f"Metadata {key} differs from examples")
    pair_groups: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        _require(bool(example.get("pair_id")), "Empty pair ID")
        pair_groups.setdefault(example["pair_id"], []).append(example)
    for group in pair_groups.values():
        _require(
            len(group) == 2 and sorted(row["label"] for row in group) == [0, 1],
            "Incomplete instruction pair",
        )
        _require(len({row["split"] for row in group}) == 1, "Pair leaks across partitions")
        _require(group[0]["response"] == group[1]["response"], "Paired responses differ")
    split_ids = {
        split: sorted(row["example_id"] for row in examples if row["split"] == split)
        for split in ("train", "validation")
    }
    _require(all(split_ids.values()), "Both partitions must be nonempty")
    for split in split_ids:
        _require(
            fitted.get(f"{split}_example_ids") == split_ids[split], "Probe example split differs"
        )
        expected_pairs = sorted({by_id[identifier]["pair_id"] for identifier in split_ids[split]})
        _require(fitted.get(f"{split}_pair_ids") == expected_pairs, "Probe pair split differs")

    expected_stems = {_digest(identifier)[:32] for identifier in example_ids}
    actual_stems = {path.stem for path in (capture / "records").glob("*.json")}
    _require(actual_stems == expected_stems, "Missing or extra capture records")
    records, groups, histories, cursor = [], {}, [], 0
    for example in examples:
        identifier = example["example_id"]
        stem = _digest(identifier)[:32]
        record_path = _inside(capture, f"records/{stem}.json")
        record = _object(record_path)
        checked(record_path)
        _require(record.get("example_id") == identifier, "Record example ID differs")
        _require(record.get("input_hash") == _digest(example), "Record input hash differs")
        _require(
            record.get("record_sha256")
            == _digest({key: value for key, value in record.items() if key != "record_sha256"}),
            "Record self-hash mismatch",
        )
        relative = f"tokens/{stem}.npz"
        _require(record.get("feature_file") == relative, "Record feature path differs")
        token_path = _inside(capture, relative)
        checked(token_path, record.get("feature_sha256"))
        _require(isinstance(record.get("feature_sha256"), str), "Missing record feature hash")
        with np.load(token_path, allow_pickle=False) as archive:
            token_matrix = np.asarray(archive["X"], dtype=np.float64)
        _require(
            token_matrix.ndim == 2 and all(token_matrix.shape) and np.isfinite(token_matrix).all(),
            "Invalid record matrix",
        )
        count_rows = len(token_matrix)
        end = cursor + count_rows
        _require(
            end <= len(X) and np.array_equal(ids[cursor:end], np.repeat(identifier, count_rows)),
            "Packed example order differs",
        )
        _require(np.array_equal(X[cursor:end], token_matrix), "Per-record/packed features differ")
        detail = record.get("capture", {})
        positions, selected, input_ids = (
            detail.get("selected_token_positions"),
            detail.get("selected_token_ids"),
            detail.get("input_token_ids"),
        )
        _require(
            isinstance(positions, list) and len(positions) == count_rows,
            "Selected position count differs",
        )
        _require(
            isinstance(input_ids, list)
            and all(type(token) is int and token >= 0 for token in input_ids),
            "Invalid input token IDs",
        )
        _require(
            all(type(position) is int and 0 <= position < len(input_ids) for position in positions),
            "Invalid selected positions",
        )
        _require(
            positions == list(range(positions[0], positions[-1] + 1)),
            "Training positions are not contiguous",
        )
        _require(
            selected == [input_ids[position] for position in positions],
            "Selected token IDs differ from input",
        )
        retained = " ".join(example["response"].split(" ")[:-5])
        _require(bool(retained.strip()), "Empty retained response")
        _require(
            detail.get("mode") == "off_policy_fixed_response"
            and detail.get("text") == example["response"],
            "Fixed-response record differs",
        )
        _require(
            detail.get("trim_last_words") == 5
            and detail.get("excluded_suffix") == example["response"][len(retained) :],
            "Recorded word mask differs",
        )
        _require(
            detail.get("selected_decoded_text") == retained,
            "Decoded selection differs from retained text",
        )
        span, rendered = detail.get("character_span"), detail.get("rendered_dialogue")
        _require(
            isinstance(span, list) and len(span) == 2 and all(type(x) is int for x in span),
            "Invalid character span",
        )
        _require(
            isinstance(rendered, str) and 0 <= span[0] < span[1] <= len(rendered),
            "Invalid rendered span",
        )
        _require(
            rendered[span[0] : span[1]] == retained, "Character span differs from retained text"
        )
        _require(
            record.get("is_test_fixture") == manifest.get("backend", {}).get("is_test_fixture"),
            "Fixture provenance differs",
        )
        histories.extend(tuple(input_ids[: position + 1]) for position in positions)
        groups[identifier] = np.arange(cursor, end)
        records.append(record)
        cursor = end
    _require(cursor == len(X), "Unaccounted packed token rows")
    _require(
        manifest.get("examples") == len(examples) and manifest.get("token_rows") == len(X),
        "Capture counts differ",
    )
    _require(manifest.get("records_hash") == _digest(records), "Record collection hash differs")
    probe_path = probe / "probe.npz"
    checked(probe_path, fitted.get("probe_sha256"))
    _require(isinstance(fitted.get("probe_sha256"), str), "Missing probe hash")
    with np.load(probe_path, allow_pickle=False) as archive:
        mean, scale, coefficient = [
            np.asarray(archive[key], dtype=np.float64) for key in ("mean", "scale", "coefficient")
        ]
        intercept = np.asarray(archive["intercept"], dtype=np.float64)
    _require(
        all(
            value.shape == (X.shape[1],) and np.isfinite(value).all()
            for value in (mean, scale, coefficient)
        ),
        "Invalid probe dimensions/values",
    )
    _require(
        (scale > 0).all() and intercept.shape == () and float(intercept) == 0.0,
        "Invalid probe scale/intercept",
    )
    _require(fitted.get("feature_dimension") == X.shape[1], "Probe dimension declaration differs")
    return {
        "X": X,
        "ids": ids,
        "metadata": by_id,
        "examples": examples,
        "groups": groups,
        "histories": histories,
        "mean": mean,
        "scale": scale,
        "coefficient": coefficient,
        "manifest": manifest,
        "fitted": fitted,
        "signature": signature,
        "input_hashes": hashes,
        "preparation": preparation,
    }


def _metric(rows: list[dict[str, Any]], score_key: str = "score") -> dict[str, Any]:
    available = [row for row in rows if row.get(score_key) is not None]
    return {
        "n_examples": len(available),
        "n_positive": sum(row["label"] for row in available),
        "n_negative": sum(1 - row["label"] for row in available),
        "n_pairs": len({row["pair_id"] for row in available}),
        "roc_auc": manual_auroc(
            [row["label"] for row in available], [row[score_key] for row in available]
        )
        if available
        else None,
    }


def _vector_key(row: np.ndarray) -> bytes:
    canonical = np.asarray(row, dtype="<f8").copy()
    canonical[canonical == 0] = 0.0
    return hashlib.sha256(canonical.tobytes()).digest()


def overlap_diagnostics(data: dict[str, Any], margins: np.ndarray) -> dict[str, Any]:
    X, histories, metadata, groups = data["X"], data["histories"], data["metadata"], data["groups"]
    train_histories: dict[tuple[int, ...], list[int]] = {}
    train_vectors: dict[bytes, list[int]] = {}
    for index, identifier in enumerate(data["ids"]):
        if metadata[identifier]["split"] == "train":
            train_histories.setdefault(histories[index], []).append(index)
            train_vectors.setdefault(_vector_key(X[index]), []).append(index)
    rows, shared_total, exact_total, shared_exact_total = [], 0, 0, 0
    for identifier, indices in groups.items():
        if metadata[identifier]["split"] != "validation":
            continue
        shared = [histories[index] in train_histories for index in indices]
        exact = [
            any(
                np.array_equal(X[index], X[other])
                for other in train_vectors.get(_vector_key(X[index]), [])
            )
            for index in indices
        ]
        shared_exact = [
            any(
                np.array_equal(X[index], X[other])
                for other in train_histories.get(histories[index], [])
            )
            for index in indices
        ]
        novel = indices[~np.asarray(shared)]
        shared_total += sum(shared)
        exact_total += sum(exact)
        shared_exact_total += sum(shared_exact)
        rows.append(
            {
                "example_id": identifier,
                "pair_id": metadata[identifier]["pair_id"],
                "label": int(metadata[identifier]["label"]),
                "n_tokens": len(indices),
                "shared_history_tokens": sum(shared),
                "exact_training_vector_tokens": sum(exact),
                "all_histories_seen": all(shared),
                "all_vectors_seen": all(exact),
                "score": float(margins[indices].mean()),
                "novel_token_score": float(margins[novel].mean()) if len(novel) else None,
                "n_novel_tokens": len(novel),
            }
        )
    covered_pairs = {row["pair_id"] for row in rows if row["all_histories_seen"]}
    any_shared_pairs = {row["pair_id"] for row in rows if row["shared_history_tokens"]}
    novel_missing_pairs = {row["pair_id"] for row in rows if row["novel_token_score"] is None}
    return {
        "selection_uses_labels": False,
        "history_definition": "entire input-token prefix through and including each scored token",
        "vector_equality": "exact numeric equality against ANY training row; signed zeros equal",
        "unique_training_histories": len(train_histories),
        "training_token_rows": sum(len(indices) for indices in train_histories.values()),
        "validation_token_rows": sum(row["n_tokens"] for row in rows),
        "shared_history_validation_rows": shared_total,
        "exact_vector_validation_rows": exact_total,
        "exact_vector_rows_with_same_history": shared_exact_total,
        "fully_history_covered_examples": sum(row["all_histories_seen"] for row in rows),
        "fully_history_covered_pairs": sorted(covered_pairs),
        "fully_vector_covered_examples": sum(row["all_vectors_seen"] for row in rows),
        "original_validation": _metric(rows),
        "purge_fully_covered_pairs": {
            "rule": "remove a pair if either member has every scored history in training",
            "removed_pair_ids": sorted(covered_pairs),
            **_metric([row for row in rows if row["pair_id"] not in covered_pairs]),
        },
        "strict_purge_any_shared_history_pairs": {
            "rule": "remove a pair if either member has any scored history in training",
            "removed_pair_ids": sorted(any_shared_pairs),
            **_metric([row for row in rows if row["pair_id"] not in any_shared_pairs]),
        },
        "novel_token_only": {
            "rule": "average only token rows whose entire causal history is absent from training",
            "n_examples_without_novel_tokens": sum(
                row["novel_token_score"] is None for row in rows
            ),
            "available_examples": _metric(rows, "novel_token_score"),
            "complete_pairs": _metric(
                [row for row in rows if row["pair_id"] not in novel_missing_pairs],
                "novel_token_score",
            ),
        },
        "validation_cases": rows,
    }


def text_baselines(examples: list[dict[str, Any]]) -> dict[str, Any]:
    train = [row for row in examples if row["split"] == "train"]
    validation = [row for row in examples if row["split"] == "validation"]

    def prompt(row: dict[str, Any]) -> str:
        return "\n".join(f"{message['role']}: {message['content']}" for message in row["messages"])

    results: dict[str, Any] = {
        "unique_training_prompts": len({_canonical(row["messages"]) for row in train}),
        "unique_validation_prompts": len({_canonical(row["messages"]) for row in validation}),
        "vectorizer": {
            "fit_partition": "train only",
            "ngram_range": [1, 2],
            "lowercase": True,
            "norm": "l2",
        },
        "classifier": {
            "C": 1.0,
            "fit_intercept": True,
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": 42,
        },
    }
    for name, getter in (("prompt_only", prompt), ("response_only", lambda row: row["response"])):
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        try:
            fitted_text = vectorizer.fit_transform([getter(row) for row in train])
        except ValueError as exc:
            results[name] = {"available": False, "reason": str(exc)}
            continue
        estimator = LogisticRegression(
            C=1.0, fit_intercept=True, solver="lbfgs", max_iter=1000, random_state=42
        )
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(fitted_text, [row["label"] for row in train])
        scores = estimator.decision_function(
            vectorizer.transform([getter(row) for row in validation])
        )
        results[name] = {
            "available": True,
            "n_features": len(vectorizer.vocabulary_),
            "roc_auc": manual_auroc([row["label"] for row in validation], scores),
            "converged": not any(issubclass(item.category, ConvergenceWarning) for item in seen),
            "scores": [
                {"example_id": row["example_id"], "score": float(score)}
                for row, score in zip(validation, scores, strict=True)
            ],
        }
    results["response_text_definition"] = (
        "complete inserted fact, identical within each instruction pair"
    )
    return results


def pair_flip_controls(
    data: dict[str, Any], mean: np.ndarray, scale: np.ndarray, permutations: int
) -> dict[str, Any]:
    _require(
        type(permutations) is int and permutations >= 0,
        "permutations must be a nonnegative integer",
    )
    metadata, groups = data["metadata"], data["groups"]
    train_mask = np.asarray(
        [metadata[identifier]["split"] == "train" for identifier in data["ids"]]
    )
    base_labels = np.asarray([int(metadata[identifier]["label"]) for identifier in data["ids"]])
    validation_ids = [
        identifier for identifier in groups if metadata[identifier]["split"] == "validation"
    ]
    pair_ids = {
        split: sorted({row["pair_id"] for row in metadata.values() if row["split"] == split})
        for split in ("train", "validation")
    }
    _require(not set(pair_ids["train"]) & set(pair_ids["validation"]), "Null control pair leakage")
    Z = (data["X"] - mean) / scale
    records = []
    for seed in range(BASE_SEED, BASE_SEED + permutations):
        train_seed, validation_seed = np.random.SeedSequence(seed).spawn(2)
        flips = {}
        for split, child_seed in (("train", train_seed), ("validation", validation_seed)):
            bits = np.random.default_rng(child_seed).integers(0, 2, size=len(pair_ids[split]))
            flips[split] = dict(zip(pair_ids[split], bits.tolist(), strict=True))
        shuffled_train = np.asarray(
            [
                base_labels[index] ^ flips["train"][metadata[identifier]["pair_id"]]
                for index, identifier in enumerate(data["ids"])
                if train_mask[index]
            ]
        )
        estimator = LogisticRegression(
            C=0.1, fit_intercept=False, solver="lbfgs", max_iter=1000, random_state=42
        )
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(Z[train_mask], shuffled_train)
        margins = Z @ estimator.coef_[0]
        scores = [float(margins[groups[identifier]].mean()) for identifier in validation_ids]
        shuffled_validation = [
            int(metadata[identifier]["label"])
            ^ flips["validation"][metadata[identifier]["pair_id"]]
            for identifier in validation_ids
        ]
        records.append(
            {
                "seed": seed,
                "train_flip_sha256": _digest(flips["train"]),
                "validation_flip_sha256": _digest(flips["validation"]),
                "n_train_pairs_flipped": sum(flips["train"].values()),
                "n_validation_pairs_flipped": sum(flips["validation"].values()),
                "roc_auc": manual_auroc(shuffled_validation, scores),
                "converged": not any(
                    issubclass(item.category, ConvergenceWarning) for item in seen
                ),
            }
        )
    aucs = [record["roc_auc"] for record in records]
    return {
        "n_fits": permutations,
        "base_seed": BASE_SEED,
        "assignment": "independent train/validation SeedSequence children; one XOR label flip per sorted pair ID",
        "normalization": "fixed training-only statistics; never fitted on validation",
        "classifier": {
            "C": 0.1,
            "fit_intercept": False,
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": 42,
        },
        "predetermined_expected_result_required": False,
        "mean_auroc": float(np.mean(aucs)) if aucs else None,
        "population_std_auroc": float(np.std(aucs)) if aucs else None,
        "minimum_auroc": min(aucs) if aucs else None,
        "maximum_auroc": max(aucs) if aucs else None,
        "fits": records,
    }


def audit_baseline(
    capture: str | Path, probe: str | Path, output: str | Path, permutations: int = 20
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("Audit output already exists; use a new path")
    _require(type(permutations) is int and permutations >= 0, "permutations must be nonnegative")
    data = _load_evidence(Path(capture), Path(probe))
    X, ids, metadata, fitted = data["X"], data["ids"], data["metadata"], data["fitted"]
    train_mask = np.asarray([metadata[identifier]["split"] == "train" for identifier in ids])
    mean, std, scale = training_statistics(X[train_mask])
    _require(np.allclose(mean, data["mean"], rtol=RTOL, atol=ATOL), "Training-only mean mismatch")
    _require(
        np.allclose(scale, data["scale"], rtol=RTOL, atol=ATOL), "Training-only scale mismatch"
    )
    settings = fitted.get("settings", {})
    _require(
        settings.get("C") == 0.1
        and settings.get("fit_intercept") is False
        and settings.get("penalty") == "l2",
        "Unexpected logistic objective",
    )
    _require(
        settings.get("training_unit") == "token"
        and settings.get("score_aggregation") == "mean_raw_margin",
        "Unexpected fit/scoring unit",
    )
    _require(
        fitted.get("n_training_token_rows") == int(train_mask.sum()), "Training row count differs"
    )
    _require(
        fitted.get("n_validation_token_rows") == int((~train_mask).sum()),
        "Validation row count differs",
    )
    Z = (X - data["mean"]) / data["scale"]
    margins = Z @ data["coefficient"]
    affine = X @ (data["coefficient"] / data["scale"]) - data["mean"] @ (
        data["coefficient"] / data["scale"]
    )
    _require(
        np.allclose(margins, affine, rtol=RTOL, atol=ATOL),
        "Independent raw-margin identities differ",
    )
    responses = [
        {
            "example_id": identifier,
            "pair_id": metadata[identifier]["pair_id"],
            "split": metadata[identifier]["split"],
            "label": int(metadata[identifier]["label"]),
            "n_tokens": len(indices),
            "score": float(np.sum(margins[indices]) / len(indices)),
        }
        for identifier, indices in data["groups"].items()
    ]
    validation = [row for row in responses if row["split"] == "validation"]
    metrics = _metric(validation)
    reference = fitted.get("validation_diagnostic", {})
    for key in ("n_examples", "n_positive", "n_negative"):
        _require(reference.get(key) == metrics[key], "Stored validation counts differ")
    stored_auc, independent_auc = reference.get("roc_auc"), metrics["roc_auc"]
    _require(
        stored_auc is None
        and independent_auc is None
        or stored_auc is not None
        and independent_auc is not None
        and np.isclose(stored_auc, independent_auc, rtol=0, atol=ATOL),
        "Stored AUROC differs from independent pairwise calculation",
    )
    train_scores = margins[train_mask]
    probabilities = np.empty_like(train_scores)
    nonnegative = train_scores >= 0
    probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-train_scores[nonnegative]))
    exponent = np.exp(train_scores[~nonnegative])
    probabilities[~nonnegative] = exponent / (1.0 + exponent)
    y = np.asarray([int(metadata[identifier]["label"]) for identifier in ids])[train_mask]
    coefficient = data["coefficient"]
    gradient = Z[train_mask].T @ (probabilities - y) / len(y) + coefficient / (0.1 * len(y))
    objective = float(
        np.mean(np.logaddexp(0, train_scores) - y * train_scores)
        + coefficient @ coefficient / (2 * 0.1 * len(y))
    )
    gradient_norm = float(np.max(np.abs(gradient)))
    report = {
        "schema": "independent_instruction_baseline_audit_v1",
        "status": "saved_artifact_integrity_and_numeric_checks_passed",
        "created_at": datetime.now(UTC).isoformat(),
        "is_deception_evaluation": False,
        "label_target": TARGET,
        "input_sha256": data["input_hashes"],
        "auditor_sha256": _file_hash(Path(__file__)),
        "feature_signature": json.loads(data["signature"]),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "integrity": {
            "capture_status": data["manifest"]["status"],
            "probe_schema": fitted["schema"],
            "n_examples": len(responses),
            "n_token_rows": len(X),
            "feature_dimension": X.shape[1],
            "record_and_packed_alignment": True,
            "saved_mask_text_checks": True,
        },
        "normalization": {
            "rtol": RTOL,
            "atol": ATOL,
            "variance_ddof": 0,
            "constant_scale_policy": "scale=1 where variance is below its float64 constant-feature uncertainty bound",
            "training_only_mean_max_error": float(np.max(np.abs(mean - data["mean"]))),
            "training_only_scale_max_error": float(np.max(np.abs(scale - data["scale"]))),
            "population_std_minimum": float(std.min()),
            "unit_scale_columns": int(np.sum(scale == 1.0)),
        },
        "frozen_scores": {
            "aggregation": "sum raw margins / number of selected tokens",
            "affine_identity_max_error": float(np.max(np.abs(margins - affine))),
            "raw_margin_sha256_float64_le": hashlib.sha256(
                margins.astype("<f8").tobytes()
            ).hexdigest(),
            "validation": metrics,
            "responses": responses,
        },
        "logistic_stationarity": {
            "objective": "mean(logaddexp(0,z)-y*z) + ||w||^2/(2*C*N)",
            "C": 0.1,
            "N": len(y),
            "regularized_objective": objective,
            "gradient_max_absolute": gradient_norm,
            "diagnostic_bound": STATIONARITY_BOUND,
            "within_diagnostic_bound": gradient_norm <= STATIONARITY_BOUND,
            "interpretation": "optimizer diagnostic only; not evidence of deception detection",
        },
        "text_controls": text_baselines(data["examples"]),
        "overlap": overlap_diagnostics(data, margins),
        "pair_flip_negative_controls": pair_flip_controls(data, mean, scale, permutations),
        "limitations": [
            "Every metric concerns instruction labels, not observed lies, intent or recipient beliefs.",
            "This audit reads saved arrays and declared token records; it does not load a model or independently retokenize text.",
            "Record hashes establish internal consistency, not external authenticity or independent human annotation.",
            "Original source bytes outside the capture are not reopened; copied preparation provenance is hash-checked.",
            "Purging and novel-token selection use causal histories and pair IDs without looking at labels or scores.",
            "Token rows share prompts and causal prefixes; they are not independent statistical observations.",
            "A finite permutation panel is a negative-control diagnostic, not a required pass rate or a paper-level significance claim.",
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--permutations", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        result = audit_baseline(args.capture, args.probe, args.output, args.permutations)
    except (ValueError, OSError, KeyError) as exc:
        print(f"Audit refused: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": args.output,
                "is_deception_evaluation": False,
                "validation_auroc": result["frozen_scores"]["validation"]["roc_auc"],
                "permutation_fits": result["pair_flip_negative_controls"]["n_fits"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
