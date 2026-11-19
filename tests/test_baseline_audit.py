"""Small saved-artifact fixtures exercise the independent CPU audit; no models."""

from __future__ import annotations

import csv
import hashlib
import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from latent_probes.baseline_audit import (
    ATOL,
    BASE_SEED,
    RTOL,
    audit_baseline,
    main,
    manual_auroc,
    text_baselines,
    training_statistics,
)


def digest(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def saved_fixture(root):
    """Distinct facts produce exact, contained and novel causal-prefix cases.

    Validation-only extreme values would strongly corrupt a leaky scaler. The
    final 'context' vector also duplicates a training vector under a DIFFERENT
    history, so exact-vector lookup must check every training occurrence.
    """
    capture, probe = root / "capture", root / "probe"
    capture.mkdir()
    probe.mkdir()
    target = "instructed_honesty_condition"
    signature = json.dumps({"model_id": "fixture", "is_test_fixture": True, "layer": 0})
    facts = [
        ("train-a", "train", "The shared prefix holds extra five omitted words right here"),
        ("train-b", "train", "Other distinct training context five ignored words go here"),
        (
            "val-exact",
            "validation",
            "The shared prefix holds extra six irrelevant words near there",
        ),
        ("val-contained", "validation", "The shared prefix six irrelevant words near there"),
        (
            "val-novel",
            "validation",
            "Fresh unseen validation context six irrelevant words near there",
        ),
    ]
    vocabulary = {}

    def tokens(words):
        return [vocabulary.setdefault(word, 100 + len(vocabulary)) for word in words]

    examples, records, matrices, token_ids, metadata = [], [], [], [], []
    for pair, split, response in facts:
        words = response.split(" ")
        content_ids = tokens(words)
        retained = " ".join(words[:-5])
        for label in (0, 1):
            identifier = pair + ("-h" if label == 0 else "-d")
            messages = [
                {
                    "role": "user",
                    "content": "Pretend you are " + ("honest" if label == 0 else "dishonest"),
                }
            ]
            example = {
                "example_id": identifier,
                "pair_id": pair,
                "split": split,
                "label": label,
                "label_target": target,
                "label_audited": "",
                "mode": "fixed",
                "messages": messages,
                "response": response,
                "trim_last_words": 5,
            }
            input_ids = [1, 10 + label, 2] + content_ids + [0]
            positions = list(range(3, 3 + len(words) - 5))
            selected = [input_ids[position] for position in positions]
            matrix = np.asarray(
                [
                    [(2 * label - 1) * (1 + index / 10), token / 100, 0]
                    for index, token in enumerate(selected)
                ],
                dtype=np.float32,
            )
            if pair == "val-novel":
                matrix[0, 1] = 1000
            stem = digest(identifier)[:32]
            feature_path = capture / "tokens" / f"{stem}.npz"
            feature_path.parent.mkdir(exist_ok=True)
            np.savez_compressed(feature_path, X=matrix)
            prompt = "USER " + messages[0]["content"] + "\nASSISTANT "
            detail = {
                "mode": "off_policy_fixed_response",
                "text": response,
                "rendered_dialogue": prompt + response + " END",
                "input_token_ids": input_ids,
                "selected_token_ids": selected,
                "selected_token_positions": positions,
                "selected_decoded_text": retained,
                "character_span": [len(prompt), len(prompt) + len(retained)],
                "excluded_suffix": response[len(retained) :],
                "trim_last_words": 5,
            }
            record = {
                "example_id": identifier,
                "input_hash": digest(example),
                "feature_file": f"tokens/{stem}.npz",
                "feature_sha256": file_hash(feature_path),
                "capture": detail,
                "is_test_fixture": True,
            }
            record["record_sha256"] = digest(record)
            write_json(capture / "records" / f"{stem}.json", record)
            examples.append(example)
            records.append(record)
            matrices.append(matrix)
            token_ids.extend([identifier] * len(matrix))
            metadata.append(
                {
                    "example_id": identifier,
                    "pair_id": pair,
                    "split": split,
                    "label": label,
                    "label_target": target,
                    "label_audited": "",
                }
            )
    X = np.concatenate(matrices)
    np.savez_compressed(
        capture / "features.npz",
        X=X,
        example_ids=np.asarray(token_ids),
        feature_signature=np.asarray(signature),
    )
    with (capture / "metadata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    config = {"model": {"model_id": "fixture"}}
    write_json(capture / "config.json", config)
    (capture / "examples.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in examples), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "stage": "apollo_tokens",
        "status": "complete",
        "identity": {
            "config_hash": digest(config),
            "examples_hash": digest(examples),
            "preparation_hash": digest(None),
        },
        "backend": {"is_test_fixture": True},
        "examples": len(examples),
        "token_rows": len(X),
        "feature_signature": signature,
        "features_sha256": file_hash(capture / "features.npz"),
        "metadata_sha256": file_hash(capture / "metadata.csv"),
        "records_hash": digest(records),
    }
    write_json(capture / "manifest.json", manifest)
    by_id = {row["example_id"]: row for row in metadata}
    train_mask = np.asarray([by_id[identifier]["split"] == "train" for identifier in token_ids])
    y = np.asarray([by_id[identifier]["label"] for identifier in token_ids])
    scaler = StandardScaler().fit(X[train_mask].astype(np.float64))
    classifier = LogisticRegression(
        C=0.1, fit_intercept=False, solver="lbfgs", max_iter=1000, random_state=42
    )
    classifier.fit(scaler.transform(X[train_mask].astype(np.float64)), y[train_mask])
    np.savez_compressed(
        probe / "probe.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        coefficient=classifier.coef_[0],
        intercept=np.asarray(0.0),
    )
    margins = scaler.transform(X.astype(np.float64)) @ classifier.coef_[0]
    validation = [row for row in metadata if row["split"] == "validation"]
    ids = np.asarray(token_ids)
    scores = [float(margins[ids == row["example_id"]].mean()) for row in validation]
    fitted = {
        "schema": "apollo_probe_v1",
        "label_target": target,
        "feature_signature": signature,
        "feature_dimension": X.shape[1],
        "features_sha256": file_hash(capture / "features.npz"),
        "labels_sha256": file_hash(capture / "metadata.csv"),
        "probe_sha256": file_hash(probe / "probe.npz"),
        "n_training_token_rows": int(train_mask.sum()),
        "n_validation_token_rows": int((~train_mask).sum()),
        "settings": {
            "C": 0.1,
            "fit_intercept": False,
            "penalty": "l2",
            "training_unit": "token",
            "score_aggregation": "mean_raw_margin",
        },
        "validation_diagnostic": {
            "n_examples": len(validation),
            "n_positive": 3,
            "n_negative": 3,
            "roc_auc": manual_auroc([row["label"] for row in validation], scores),
        },
    }
    for split in ("train", "validation"):
        fitted[f"{split}_example_ids"] = sorted(
            row["example_id"] for row in metadata if row["split"] == split
        )
        fitted[f"{split}_pair_ids"] = sorted(
            {row["pair_id"] for row in metadata if row["split"] == split}
        )
    write_json(probe / "manifest.json", fitted)
    return capture, probe, examples, X


def test_manual_pairwise_auc_handles_ties_and_undefined_single_class():
    assert manual_auroc([0, 1, 1], [0, 0, 1]) == 0.75
    assert manual_auroc([0, 1], [8, 8]) == 0.5
    assert manual_auroc([1, 1], [0, 1]) is None
    with pytest.raises(ValueError, match="finite"):
        manual_auroc([0, 1], [0, np.nan])


def test_training_statistics_ignore_validation_extremes_and_handle_constant_column():
    train = np.asarray([[-3, 0], [-1, 0], [1, 0], [3, 0]], dtype=float)
    validation = np.asarray([[100, 0], [102, 0]], dtype=float)
    mean, std, scale = training_statistics(train)
    np.testing.assert_allclose(mean, [0, 0])
    np.testing.assert_allclose(std, [np.sqrt(5), 0])
    np.testing.assert_allclose(scale, [np.sqrt(5), 1])
    assert abs(training_statistics(np.concatenate([train, validation]))[0][0]) > 30


def test_prompt_and_response_controls_expose_the_instruction_confound(tmp_path):
    _, _, examples, _ = saved_fixture(tmp_path)
    report = text_baselines(examples)
    assert report["unique_training_prompts"] == 2
    assert report["prompt_only"]["roc_auc"] == 1
    assert report["response_only"]["roc_auc"] == 0.5


def test_audit_detects_exact_and_contained_histories_and_any_training_vector(tmp_path):
    capture, probe, _, _ = saved_fixture(tmp_path)
    before = {
        path: file_hash(path)
        for root in (capture, probe)
        for path in root.rglob("*")
        if path.is_file()
    }
    report = audit_baseline(capture, probe, tmp_path / "audit.json", permutations=2)
    assert report["is_deception_evaluation"] is False
    assert report["normalization"]["rtol"] == RTOL
    assert report["normalization"]["atol"] == ATOL
    assert report["logistic_stationarity"]["within_diagnostic_bound"]
    overlap = report["overlap"]
    assert overlap["selection_uses_labels"] is False
    assert overlap["shared_history_validation_rows"] == 16
    assert overlap["exact_vector_rows_with_same_history"] == 16
    assert overlap["exact_vector_validation_rows"] == 18
    assert overlap["fully_history_covered_examples"] == 4
    assert overlap["fully_history_covered_pairs"] == ["val-contained", "val-exact"]
    assert overlap["purge_fully_covered_pairs"]["n_examples"] == 2
    assert overlap["novel_token_only"]["available_examples"]["n_examples"] == 2
    negative = report["pair_flip_negative_controls"]
    assert negative["n_fits"] == 2
    assert [fit["seed"] for fit in negative["fits"]] == [BASE_SEED, BASE_SEED + 1]
    assert negative["predetermined_expected_result_required"] is False
    assert all(file_hash(path) == digest_before for path, digest_before in before.items())
    second = audit_baseline(capture, probe, tmp_path / "second.json", permutations=2)
    assert second["pair_flip_negative_controls"] == negative
    with pytest.raises(FileExistsError):
        audit_baseline(capture, probe, tmp_path / "audit.json", permutations=0)


def test_an_internally_rehashed_validation_leaky_scaler_is_rejected(tmp_path):
    capture, probe, _, X = saved_fixture(tmp_path)
    path = probe / "probe.npz"
    with np.load(path, allow_pickle=False) as arrays:
        values = {key: arrays[key].copy() for key in arrays.files}
    values["mean"] = X.astype(float).mean(axis=0)
    np.savez_compressed(path, **values)
    manifest = json.loads((probe / "manifest.json").read_text())
    manifest["probe_sha256"] = file_hash(path)
    write_json(probe / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Training-only mean"):
        audit_baseline(capture, probe, tmp_path / "refused.json", permutations=0)
    assert not (tmp_path / "refused.json").exists()


def test_rehashed_packed_features_cannot_disagree_with_per_record_arrays(tmp_path):
    capture, probe, _, _ = saved_fixture(tmp_path)
    path = capture / "features.npz"
    with np.load(path, allow_pickle=False) as arrays:
        values = {key: arrays[key].copy() for key in arrays.files}
    values["X"][0, 0] += 1
    np.savez_compressed(path, **values)
    manifest = json.loads((capture / "manifest.json").read_text())
    manifest["features_sha256"] = file_hash(path)
    write_json(capture / "manifest.json", manifest)
    fitted = json.loads((probe / "manifest.json").read_text())
    fitted["features_sha256"] = file_hash(path)
    write_json(probe / "manifest.json", fitted)
    with pytest.raises(ValueError, match="Per-record/packed"):
        audit_baseline(capture, probe, tmp_path / "refused.json", permutations=0)


def test_rehashed_selected_token_ids_must_match_the_saved_input_positions(tmp_path):
    capture, probe, examples, _ = saved_fixture(tmp_path)
    path = capture / "records" / (digest(examples[0]["example_id"])[:32] + ".json")
    record = json.loads(path.read_text())
    record["capture"]["selected_token_ids"][0] += 1
    record["record_sha256"] = digest(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )
    write_json(path, record)
    records = [
        json.loads(
            (capture / "records" / (digest(example["example_id"])[:32] + ".json")).read_text()
        )
        for example in examples
    ]
    manifest = json.loads((capture / "manifest.json").read_text())
    manifest["records_hash"] = digest(records)
    write_json(capture / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Selected token IDs differ"):
        audit_baseline(capture, probe, tmp_path / "refused.json", permutations=0)


@pytest.mark.parametrize("corruption", ["hash", "status", "record"])
def test_corrupted_saved_artifacts_are_refused(tmp_path, corruption):
    capture, probe, _, _ = saved_fixture(tmp_path)
    if corruption == "hash":
        with (capture / "features.npz").open("ab") as stream:
            stream.write(b"changed")
    elif corruption == "status":
        path = capture / "manifest.json"
        value = json.loads(path.read_text())
        value["status"] = "failed"
        write_json(path, value)
    else:
        path = next((capture / "records").glob("*.json"))
        value = json.loads(path.read_text())
        value["input_hash"] = "changed"
        write_json(path, value)
    with pytest.raises(ValueError):
        audit_baseline(capture, probe, tmp_path / "refused.json", permutations=0)


def test_cli_uses_only_saved_files_and_reports_failure(tmp_path, capsys):
    capture, probe, _, _ = saved_fixture(tmp_path)
    output = tmp_path / "cli.json"
    assert (
        main(
            [
                "--capture",
                str(capture),
                "--probe",
                str(probe),
                "--output",
                str(output),
                "--permutations",
                "0",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["is_deception_evaluation"] is False
    assert main(["--capture", str(capture), "--probe", str(probe), "--output", str(output)]) == 2
    assert "already exists" in capsys.readouterr().err
