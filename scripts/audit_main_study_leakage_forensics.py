"""Independent, read-only leakage audit; never import or execute study code.

Reads JSON and numeric NPZ with pickle disabled. Reconstructs allocations and
preprocessing from retained inputs, without fitting classifiers or calling LMs.
The report deliberately contains no held-out performance scores.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from hashlib import sha256
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical(value, newline=False):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256((text + ("\n" if newline else "")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def native_path(path):
    """Resolve recorded WSL paths on Windows, with no shell invocation."""
    if path.startswith("/mnt/c/"):
        return Path("C:/" + path[len("/mnt/c/") :])
    if path.startswith("/") and Path.cwd().drive:
        return Path("//wsl.localhost/Ubuntu" + path)
    return Path(path)


def check_bindings(bindings):
    for name, expected in bindings.items():
        path = native_path(name)
        require(path.is_file() and digest(path) == expected, f"Changed bound input: {name}")
    return len(bindings)


def gold(state):
    previous = state["initial"]
    if state["mechanism"] == "correction" and state["first_receipt"]:
        previous = state["update"]
    update = 1 - state["update"] if state["mechanism"] == "correction" else state["update"]
    return update if state["receipt"] else previous


def identifier(group, bits):
    i, u, b, w, m = bits
    return f"{group}__i{i}_u{u}_b{b}_w{w}_map{m}"


def rank(*parts):
    return sha256("|".join(map(str, parts)).encode()).digest()


def allocation(groups, size, seed):
    selected = []
    for mechanism in ("observation", "message"):
        pairs = sorted(
            {g["pair_id"] for g in groups if g["mechanism"] == mechanism},
            key=lambda p: (rank("pair-selection-v1", seed, mechanism, p), p),
        )
        for pair in pairs[: size // 64]:
            selected.extend(
                sorted((g for g in groups if g["pair_id"] == pair), key=lambda g: g["offset"])
            )
    result = {"restricted": [], "full": []}
    for group in selected:
        masks = {}
        for update in (0, 1):
            cells = sorted(
                product((0, 1), repeat=2),
                key=lambda c: (rank("coverage-mask-v1", seed, group["pair_id"], update, *c), c),
            )
            masks[update] = set(cells[:2])
        for i, u, b, w, m in product((0, 1), repeat=5):
            case_id = identifier(group["group_id"], (i, u, b, w, m))
            if i != u:
                result["restricted"].append(case_id)
            if i == (u ^ (int((w, m) in masks[u]) ^ group["offset"])):
                result["full"].append(case_id)
    return [g["group_id"] for g in selected], result


def check_memberships(comparison, expected, train_ids, dev_ids):
    require(comparison["development_ids"] == dev_ids, "Development membership or order differs")
    for arm in ("restricted", "full"):
        ids = comparison["arms"][arm]["training_ids"]
        require(
            ids == expected[arm] and set(ids) <= train_ids, f"{arm}: fitting membership differs"
        )
        require(len(set(ids)) == comparison["training_n"], f"{arm}: duplicate fitting input")
        calibration = comparison["arms"][arm]["calibration"]
        if calibration["status"] == "complete":
            require(
                calibration["source_ids"] == dev_ids, f"{arm}: calibration not development-only"
            )


def check_scaler(selected, mean, scale):
    selected = np.asarray(selected, dtype=np.float64)
    expected_mean, expected_scale = selected.mean(axis=0), selected.std(axis=0)
    expected_scale[expected_scale == 0] = 1
    require(
        np.allclose(mean, expected_mean, atol=1e-12, rtol=1e-12),
        "Scaler mean not selected-train-only",
    )
    require(
        np.allclose(scale, expected_scale, atol=1e-12, rtol=1e-12),
        "Scaler scale not selected-train-only",
    )
    return float(max(np.max(np.abs(mean - expected_mean)), np.max(np.abs(scale - expected_scale))))


def text(row):
    return "\n".join(f"{m.get('role', '')}: {m['content']}" for m in row["messages"])


def check_vocabulary(rows, vocabulary, idf):
    # Build the prescribed analyzer, without fitting a vectorizer. Derive the
    # observed vocabulary and smoothed document-frequency IDF independently.
    analyzer = TfidfVectorizer(
        lowercase=True, analyzer="word", token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2)
    ).build_analyzer()
    frequency = Counter(term for row in rows for term in set(analyzer(text(row))))
    require(len(frequency) < 50000, "Audit needs explicit max_features selection for this corpus")
    require(
        set(vocabulary) == set(frequency), "Vocabulary differs from selected training documents"
    )
    require(set(vocabulary.values()) == set(range(len(vocabulary))), "Noncontiguous vocabulary")
    expected = np.empty(len(vocabulary))
    for term, index in vocabulary.items():
        expected[index] = np.log((1 + len(rows)) / (1 + frequency[term])) + 1
    require(np.allclose(idf, expected, atol=1e-12, rtol=1e-12), "IDF not selected-train-only")


def selected_development_loss(rows, features, comparison, head, arrays):
    """Use retained coefficients only; no fitting and no held-out scores."""
    prefix = head["array_prefix"]
    if comparison["feature_kind"] == "activation":
        values = (np.asarray(features, dtype=np.float64) - arrays[prefix + "__mean"]) / arrays[
            prefix + "__scale"
        ]
        norms = np.linalg.norm(values, axis=1)
        norms[norms == 0] = 1
        values /= norms[:, None]
    else:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
            max_features=50000,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float64,
            vocabulary=head["vocabulary"],
        )
        vectorizer.idf_ = arrays[prefix + "__idf"]
        values = vectorizer.transform([text(row) for row in rows])
    margins = (
        np.asarray(values @ arrays[prefix + "__coef"]).ravel() + arrays[prefix + "__intercept"][0]
    )
    counts = Counter(r["group_id"] for r in rows)
    weights = np.array([1 / (len(counts) * counts[r["group_id"]]) for r in rows])
    labels = np.array([gold(r["state"]) for r in rows])
    return float(weights @ (np.logaddexp(0, margins) - labels * margins))


def check_cache(directory, bank_groups, split_rows, partition):
    manifest, payload = read(directory / "capture_manifest.json"), read(directory / "rows.json")
    require(manifest["status"] == "complete", "Incomplete cache")
    for name in ("features.npz", "rows.json"):
        require(
            digest(directory / name) == manifest["output_sha256"][name],
            "Cache file binding differs",
        )
    rows = payload["rows"]
    with np.load(directory / "features.npz", allow_pickle=False) as data:
        features, ids = data["features"].copy(), data["row_ids"].tolist()
    require(
        ids == [r["case_id"] for r in rows] and len(set(ids)) == len(ids),
        "Cache row order/uniqueness",
    )
    expected = [r for r in split_rows if (r["split"] == "test") == (partition == "test")]
    require(ids == [r["case_id"] for r in expected], "Frozen row inclusion/order differs")
    require(
        features.shape[0] == len(rows) and np.isfinite(features).all(), "Invalid feature matrix"
    )
    protocol = read(directory / "protocol.json")
    require(
        digest(directory / "protocol.json") == manifest["protocol_sha256"],
        "Protocol file binding differs",
    )
    planned = {r["case_id"]: r for r in protocol["planned_rows"]}
    by_id = {r["case_id"]: r for r in split_rows}
    for row, values in zip(rows, features, strict=True):
        expected_row, group, prepared = (
            by_id[row["case_id"]],
            bank_groups[row["group_id"]],
            planned[row["case_id"]],
        )
        require(row["split"] == group["split"] == expected_row["split"], "Group/split mismatch")
        require(row["gold"] == gold(row["state"]), "Independent target rule differs")
        require(
            canonical(row["messages"], newline=True) == expected_row["prompt_sha256"],
            "Manifest/prompt differs",
        )
        require(canonical(row["messages"]) == row["messages_sha256"], "Messages hash differs")
        require(
            row["messages"] == prepared["messages"]
            and row["prompt_token_ids"] == prepared["prompt_token_ids"],
            "Prepared prefix differs",
        )
        require(
            canonical(values.tolist()) == row["feature_values_sha256"],
            "Retained feature values differ",
        )
        require(row["capture"]["answer_tokens_supplied"] == 0, "Answer tokens supplied")
        require(
            row["capture"]["selected_position"] == len(row["prompt_token_ids"]) - 1,
            "Position differs",
        )
        require(
            all(m["role"] in {"system", "user"} for m in row["messages"]),
            "Assistant completion in input",
        )
        visible = text(row)
        require(
            row["case_id"] not in visible and row["group_id"] not in visible,
            "Identifier in visible input",
        )
    return rows, features, protocol, manifest


def contextual_overlap(groups):
    normalized, pairs = {}, []
    for group in groups:
        value = (group["introduction"] + " " + group["event_description"]).casefold()
        value = re.sub(r"\{(?:sender|reader)\}", "person", value)
        normalized[group["group_id"]] = re.findall(r"\b\w+\b", value)
    for i, left in enumerate(groups):
        a = normalized[left["group_id"]]
        a5 = set(zip(*(a[n:] for n in range(5))))
        for right in groups[i + 1 :]:
            if left["split"] == right["split"]:
                continue
            b = normalized[right["group_id"]]
            b5 = set(zip(*(b[n:] for n in range(5))))
            require(a != b, "Exact normalized contextual framing across splits")
            score = len(a5 & b5) / len(a5 | b5)
            pairs.append((score, left["group_id"], right["group_id"]))
    return [
        {"fivegram_jaccard": s, "left": a, "right": b}
        for s, a, b in sorted(pairs, reverse=True)[:10]
    ]


def audit(root, run_root):
    bank_path = root / "data/belief_main/main_v1/main_bank_v1.json"
    bank = read(bank_path)
    split = read(root / "data/belief_main/main_v1/split_manifest_v1.json")
    materials = read(root / "data/belief_main/main_v1/matched_allocations_v1.json")
    require(
        canonical(bank, newline=True) == split["bank_sha256"] == materials["bank_sha256"],
        "Bank binding differs",
    )
    groups = bank["groups"]
    by_group = {g["group_id"]: g for g in groups}
    train_groups = [g for g in groups if g["split"] == "train"]
    entities, actor_components = {}, {}
    for partition in ("train", "development", "test"):
        entities[partition] = {
            v.casefold()
            for g in groups
            if g["split"] == partition
            for v in (g["sender"], g["reader"], *g["locations"])
        }
        actor_components[partition] = {
            w
            for g in groups
            if g["split"] == partition
            for v in (g["sender"], g["reader"])
            for w in re.split(r"[\s'-]+", v.casefold())
        }
    for a, b in (("train", "development"), ("train", "test"), ("development", "test")):
        require(not entities[a] & entities[b], "Full entity reused across splits")
        require(
            not actor_components[a] & actor_components[b], "Actor component reused across splits"
        )
    contexts = contextual_overlap(groups)
    require(len({r["prompt_sha256"] for r in split["rows"]}) == 5888, "Duplicate exact prompt")
    for record in materials["allocations"]:
        selected, expected = allocation(
            train_groups, record["training_n"], record["allocation_seed"]
        )
        require(selected == record["selected_groups"], "Selected group allocation differs")
        for arm in ("restricted", "full"):
            require(expected[arm] == record[arm]["case_ids"], "Allocation manifest differs")
    campaign, recovery = (
        run_root / "main_v1_agent_20261122_campaign",
        run_root / "main_v1_agent_20261122_recovery_v1",
    )
    freeze = read(campaign / "test_freeze.json")
    source_binding_count = check_bindings(freeze["source_sha256"])
    fitted_binding_count = check_bindings(freeze["fitted_artifact_sha256"])
    archive_root = run_root / "main_v1_agent_20261122_training_source_archive"
    archive = read(archive_root / "manifest.json")
    archive_entries = archive["files"] + archive.get("supplementary_exact_prefix_plans", [])
    require(len(archive_entries) == archive["copied_file_count"], "Archive inventory differs")
    for entry in archive_entries:
        require(
            digest(archive_root / entry["archive_relative_path"]) == entry["sha256"],
            "Archive bytes differ",
        )
        require(
            digest(native_path(entry["original_absolute_path"])) == entry["sha256"],
            "Original/archive differs",
        )
    result = {
        "schema": "main-study-leakage-forensics-v1",
        "scientific_code_imported": False,
        "model_calls": 0,
        "classifier_refits": 0,
        "heldout_performance_tables_read": False,
        "bank_sha256": split["bank_sha256"],
        "exact_unique_prompts": 5888,
        "group_counts": dict(Counter(g["split"] for g in groups)),
        "row_counts": dict(Counter(r["split"] for r in split["rows"])),
        "independently_reconstructed_allocations": len(materials["allocations"]),
        "cross_split_full_entity_overlap": 0,
        "cross_split_actor_component_overlap": 0,
        "exact_normalized_context_overlap": 0,
        "largest_contextual_overlaps": contexts,
        "source_bindings_checked": source_binding_count,
        "fit_bindings_checked": fitted_binding_count,
        "archived_files_checked": len(archive_entries),
        "models": {},
    }
    for model in ("llama", "qwen"):
        rows, features, protocol, manifest = check_cache(
            run_root / f"main_v1_agent_20261122_{model}_train_dev",
            by_group,
            split["rows"],
            "train_dev",
        )
        test_rows, _, test_protocol, test_manifest = check_cache(
            recovery / f"{model}_test", by_group, split["rows"], "test"
        )
        test_ids = {r["case_id"] for r in test_rows}
        require(not test_ids & {r["case_id"] for r in rows}, "Train/dev versus test case overlap")
        settings = read(campaign / f"{model}_fit/settings.json")
        require(
            digest(campaign / f"{model}_fit/heads.npz") == settings["heads_sha256"],
            "Fit file binding differs",
        )
        require(
            settings["capture_source_hashes"] == manifest["output_sha256"],
            "Fit origin cache differs",
        )
        require(
            settings["fitting_group_splits"]
            == {g["group_id"]: g["split"] for g in groups if g["split"] != "test"},
            "Fit group inventory differs",
        )
        with np.load(campaign / f"{model}_fit/heads.npz", allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in data.files}
        require(set(arrays) == set(settings["array_sha256"]), "Fit array inventory differs")
        for name, values in arrays.items():
            require(
                sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
                == settings["array_sha256"][name],
                "Fit array hash differs",
            )
        index = {r["case_id"]: i for i, r in enumerate(rows)}
        train_ids = {r["case_id"] for r in rows if r["split"] == "train"}
        dev_ids = [r["case_id"] for r in rows if r["split"] == "development"]
        dev_indices = [index[i] for i in dev_ids]
        dev_rows = [rows[i] for i in dev_indices]
        inventory = {
            (c["feature_kind"], c["training_n"], c["allocation_seed"])
            for c in settings["comparisons"]
        }
        require(
            len(settings["comparisons"]) == 30
            and inventory
            == set(product(("activation", "tfidf"), (128, 256, 512), (11, 23, 37, 53, 71))),
            "Fit inventory differs",
        )
        scaler_error = 0.0
        for comparison in settings["comparisons"]:
            selected, expected = allocation(
                train_groups, comparison["training_n"], comparison["allocation_seed"]
            )
            require(selected == comparison["selected_groups"], "Fit group selection differs")
            if comparison["status"] != "complete":
                require(
                    comparison["status"] == "no_converged_shared_candidate",
                    "Unexpected fit exclusion",
                )
                continue
            check_memberships(comparison, expected, train_ids, dev_ids)
            candidates = comparison["candidates"]
            require([c["C"] for c in candidates] == [0.001, 0.01, 0.1, 1.0, 10.0], "C grid differs")
            successful = [c for c in candidates if not c["excluded_arms"]]
            for candidate in successful:
                require(
                    abs(candidate["mean_loss"] - np.mean(list(candidate["losses"].values())))
                    < 1e-12,
                    "Shared-C mean differs from stored arm losses",
                )
            require(
                comparison["selected_C"]
                == min(successful, key=lambda c: (c["mean_loss"], c["C"]))["C"],
                "Shared-C selection differs",
            )
            selected_candidate = next(c for c in successful if c["C"] == comparison["selected_C"])
            for arm in ("restricted", "full"):
                head = comparison["arms"][arm]
                selected_rows = [rows[index[i]] for i in head["training_ids"]]
                require(
                    canonical(selected_rows) == head["training_rows_sha256"],
                    "Fitting row binding differs",
                )
                prefix = head["array_prefix"]
                if comparison["feature_kind"] == "activation":
                    scaler_error = max(
                        scaler_error,
                        check_scaler(
                            features[[index[i] for i in head["training_ids"]]],
                            arrays[prefix + "__mean"],
                            arrays[prefix + "__scale"],
                        ),
                    )
                else:
                    check_vocabulary(selected_rows, head["vocabulary"], arrays[prefix + "__idf"])
                reconstructed_loss = selected_development_loss(
                    dev_rows, features[dev_indices], comparison, head, arrays
                )
                require(
                    abs(reconstructed_loss - selected_candidate["losses"][arm]) < 1e-10,
                    "Retained head does not reproduce selected development loss",
                )
        require(
            settings["output_calibration"]["source_ids"] == dev_ids,
            "Output calibration membership differs",
        )
        first_test_attempt = read(recovery / f"{model}_test/attempts/0000.json")["started_at"]
        require(
            datetime.fromisoformat(manifest["finished_at"])
            < datetime.fromisoformat(freeze["created_at"])
            < datetime.fromisoformat(first_test_attempt),
            "Train/dev, test-freeze, test-attempt chronology differs",
        )
        result["models"][model] = {
            "train_dev_rows": len(rows),
            "test_rows": len(test_rows),
            "comparisons_checked": len(settings["comparisons"]),
            "activation_scalers_checked": 30,
            "tfidf_vocabularies_idfs_checked": 30,
            "maximum_scaler_absolute_error": scaler_error,
            "development_rows_for_selection_and_calibration": len(dev_ids),
            "test_freeze_created_at": freeze["created_at"],
            "test_first_attempt_at": first_test_attempt,
            "training_finished_at": manifest["finished_at"],
            "test_finished_at": test_manifest["finished_at"],
            "source_membership_verified": True,
            "stored_preprocessing_verified": True,
            "selected_C_losses_reproduced_from_retained_heads": 60,
        }
    receipt = read(recovery / "recovery_receipt.json")
    require(receipt["new_fit_or_selection"] is False, "Recovery introduced fitting")
    unknown = native_path(receipt["unknown_attempt"]["original_directory"])
    require(
        {p.name for p in unknown.iterdir()} == {"started.json"},
        "Recovered unknown had recorded outcome",
    )
    require(
        digest(unknown / "started.json")
        == receipt["unknown_attempt"]["artifact_sha256"]["started.json"],
        "Unknown journal differs",
    )
    result["recovery"] = {
        "retained_rows": receipt["retained_completed_rows"],
        "remaining_rows": receipt["remaining_rows"],
        "unknown_repeated_case": receipt["unknown_attempt"]["case_id"],
        "recorded_outcome_available": False,
        "new_fit_or_selection": False,
    }
    result["status"] = "all_executed_checks_passed"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--run-root", type=Path, default=native_path("/home/adirj/kcl-latent-probes-runs")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Choose a fresh audit output")
    result = audit(args.root, args.run_root)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "models": list(result["models"]),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
