"""CPU-only fixtures for prespecified identical-text context comparisons."""

import csv
import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from latent_probes.storage import atomic_json, file_sha256, read_json

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


same_text = load_script("same_text_test", ROOT / "scripts/run_mask_same_text.py")
reference = load_script("mask_fixture_source", ROOT / "tests/test_mask_specificity.py")
prepared = reference.prepared


@pytest.fixture
def inputs():
    return (
        read_json(ROOT / "data/mask_specificity/mask_specificity_v1.json"),
        read_json(ROOT / "data/mask_specificity/mask_neutral_calibration_v1.json"),
    )


@pytest.fixture
def sources(prepared, tmp_path):
    reference.runner.capture_run(prepared.output, prepared.backend, progress=None)
    annotations = tmp_path / "draft.csv"
    rows = reference.draft_csv(prepared.output, annotations)
    truths = {case["id"]: case["ground_truth_option"] for case in prepared.plan["panel"]["cases"]}
    for row in rows:
        row["focal_resolution"] = str(truths[row["case_id"]])
    with annotations.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=reference.runner.ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    score_dir = tmp_path / "scores"
    reference.runner.score_draft(prepared.output, annotations, score_dir)
    backend = SimpleNamespace(
        metadata=prepared.backend.metadata,
        settings=prepared.settings,
        tokenizer=SimpleNamespace(all_special_ids=[2, 99]),
        input_device="cpu",
        torch=SimpleNamespace(tensor=lambda value, device: np.asarray(value)),
    )
    ablated = {
        tuple((message["role"], message["content"]) for message in case["ablated_messages"])
        for case in prepared.plan["panel"]["cases"]
    }

    def render(messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": True}
        key = tuple((message["role"], message["content"]) for message in messages)
        return [1, 4] if key in ablated else [1, 5]

    backend.tokenizer.apply_chat_template = render
    backend.calls = []

    def capture(sequence, positions):
        assert sequence.tolist()[0][2:] == [11, 99, 12, 2]
        assert positions == [2, 4]
        backend.calls.append((sequence.tolist(), positions))
        X = np.asarray([[3.0, 7.0], [5.0, 11.0]], dtype=np.float32)
        if sequence[0, 1] == 5:
            X += np.asarray([2.0, 0.0], dtype=np.float32)
        return {"layer_8_tokens": X}

    backend._capture_token_rows = capture
    return SimpleNamespace(
        capture=prepared.output,
        annotations=annotations,
        score=score_dir,
        backend=backend,
        tmp=tmp_path,
        panel=prepared.plan["panel"],
        probe=prepared.probe_dir,
    )


def test_plan_selection_first_repeat_never_scores(sources):
    scores = read_json(sources.score / "draft_scores.json")
    for row in scores["rows"]:
        row["mean_raw_margin"] = -1000 if row["example_id"].endswith("r2") else 1000
    atomic_json(sources.score / "draft_scores.json", scores)
    plan = same_text.prepare_same_text(
        sources.capture, sources.annotations, sources.score, sources.tmp / "plan"
    )
    assert len(plan["selected"]) == 8
    assert all(row["example_id"].endswith("r0") for row in plan["selected"])
    assert plan["expected_forward_passes"] == 16 and plan["expected_generations"] == 0
    assert not sources.backend.calls


def test_same_generated_ids_eos_and_special_selection(sources):
    output = sources.tmp / "same_text"
    before = same_text._snapshot(sources.capture)
    summary = same_text.run_same_text(
        sources.capture, sources.annotations, sources.score, output, sources.backend, progress=None
    )
    assert summary["n_cases"] == 8 and summary["n_forward_passes"] == 16
    assert summary["n_generations"] == 0 and len(sources.backend.calls) == 16
    assert summary["positive"] == 8 and summary["mean_difference"] == 2.0
    assert summary["all_ablated_replays_exact"] is True
    assert same_text._snapshot(sources.capture) == before
    record = read_json(next((output / "cases").glob("*.json")))
    assert record["generated_token_ids"] == [11, 99, 12, 2]
    assert record["selected_token_ids"] == [11, 12]
    assert record["contexts"]["pressure"]["full_sequence_token_ids"] == [1, 5, 11, 99, 12, 2]
    assert record["ablated_replay_max_absolute_error"] == 0
    assert record["label_audited"] is False
    assert record["original_mean_raw_margin"] == 1.5
    assert read_json(output / "manifest.json")["source_unchanged"] is True
    with np.load(next((output / "features").glob("*.npz")), allow_pickle=False) as arrays:
        assert np.array_equal(arrays["X_original"], arrays["X_ablated"])


def test_no_overwrite_and_protected_sources(sources):
    with pytest.raises(ValueError, match="disjoint"):
        same_text.prepare_same_text(
            sources.capture, sources.annotations, sources.score, sources.capture / "forbidden"
        )
    with pytest.raises(ValueError, match="disjoint"):
        same_text.prepare_same_text(
            sources.capture, sources.annotations, sources.score, sources.tmp
        )
    output = sources.tmp / "same_text"
    same_text.prepare_same_text(sources.capture, sources.annotations, sources.score, output)
    with pytest.raises(FileExistsError):
        same_text.prepare_same_text(sources.capture, sources.annotations, sources.score, output)
    assert not sources.backend.calls


def test_changed_annotations_rejected_before_forward(sources):
    output = sources.tmp / "same_text"
    same_text.prepare_same_text(sources.capture, sources.annotations, sources.score, output)
    with sources.annotations.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="source changed"):
        same_text.execute_same_text(output, sources.backend, progress=None)
    assert not sources.backend.calls


def test_changed_prefix_rejected_before_forward(sources):
    output = sources.tmp / "same_text"
    same_text.prepare_same_text(sources.capture, sources.annotations, sources.score, output)
    sources.backend.tokenizer.apply_chat_template = lambda *args, **kwargs: [1, 777]
    with pytest.raises(ValueError, match="prefix"):
        same_text.execute_same_text(output, sources.backend, progress=None)
    assert not sources.backend.calls


def test_shared_backend_extra_capture_layers_allowed(sources):
    sources.backend.settings = replace(sources.backend.settings, capture_layers=[8, 16, 24])
    summary = same_text.run_same_text(
        sources.capture,
        sources.annotations,
        sources.score,
        sources.tmp / "same_text",
        sources.backend,
        progress=None,
    )
    assert summary["n_cases"] == 8


def test_selection_skips_refusal_truncation_and_false_operational_belief(sources):
    records = []
    for job in read_json(sources.capture / "protocol.json")["jobs"]:
        path = sources.capture / "responses" / (job["example_id"] + ".json")
        record = read_json(path)
        record["record_sha256"] = file_sha256(path)
        records.append(record)
    annotations = same_text.specificity.load_draft_annotations(sources.annotations, records)
    first, second, third = [case["id"] for case in sources.panel["cases"][:3]]
    for record in records:
        identifier = record["example_id"]
        if (
            record["case_id"] == first
            and record["condition"] == "ablated"
            and record["repeat"] == 0
        ):
            annotations[identifier]["response_style"] = "refusal"
        if (
            record["case_id"] == third
            and record["condition"] == "ablated"
            and record["repeat"] == 0
        ):
            record["length_capped"] = True
        if record["case_id"] == second and record["condition"] != "calibration":
            annotations[identifier]["focal_resolution"] = str(
                1 - int(annotations[identifier]["focal_resolution"])
            )
    selected = same_text.select_first_clean(records, annotations, sources.panel)
    by_case = {row["case_id"]: row["example_id"] for row in selected}
    assert by_case[first].endswith("r1") and by_case[third].endswith("r1")
    assert second not in by_case and len(selected) == 7


def test_nonexact_replay_retained_without_tolerance_or_replacing_source(sources):
    output = sources.tmp / "same_text"
    original_capture = sources.backend._capture_token_rows

    def perturbed(sequence, positions):
        result = original_capture(sequence, positions)
        result["layer_8_tokens"][0, 0] += np.float32(0.001)
        return result

    sources.backend._capture_token_rows = perturbed
    before = file_sha256(sources.capture / "manifest.json")
    summary = same_text.run_same_text(
        sources.capture, sources.annotations, sources.score, output, sources.backend, progress=None
    )
    assert summary["all_ablated_replays_exact"] is False
    record = read_json(next((output / "cases").glob("*.json")))
    assert record["ablated_replay_changed_values"] == 1
    assert record["ablated_replay_max_absolute_error"] > 0
    assert file_sha256(sources.capture / "manifest.json") == before
