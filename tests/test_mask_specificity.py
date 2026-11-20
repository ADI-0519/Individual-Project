"""CPU-only checks of prospective ordering, exact replay and draft-label separation."""

import copy
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from latent_probes.apollo import capture_signature
from latent_probes.backends import GenerationResult
from latent_probes.config import ModelSettings
from latent_probes.storage import atomic_json, file_sha256, read_json

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mask_runner_test", ROOT / "scripts/run_mask_specificity.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def inputs():
    return (
        read_json(ROOT / "data/mask_specificity/mask_specificity_v1.json"),
        read_json(ROOT / "data/mask_specificity/mask_neutral_calibration_v1.json"),
    )


class FixtureBackend:
    def __init__(self, metadata, output):
        self.metadata = metadata
        self.output = output
        self.calls = []
        self.replays = []
        self.input_device = "cpu"
        self.torch = SimpleNamespace(tensor=lambda value, device: np.asarray(value))
        self.tokenizer = SimpleNamespace(
            all_special_ids=[2, 99], decode=lambda ids: " ".join(map(str, ids))
        )
        self.fail_generation_at = None
        self.fail_extraction_at = None

    def generate(self, messages, settings, seed, capture):
        assert read_json(self.output / "manifest.json")["status"] == "running"
        assert (self.output / "protocol.json").is_file()
        index = len(self.calls)
        if index >= 32:
            threshold = read_json(self.output / "calibration_threshold.json")
            assert threshold["n"] == 32 and len(threshold["response_sha256"]) == 32
        else:
            assert not (self.output / "calibration_threshold.json").exists()
        self.calls.append((messages, settings, seed, capture))
        if index == self.fail_generation_at:
            raise RuntimeError("Fixture generation failure")
        return GenerationResult(
            text="Clear focal answer.",
            prompt_token_ids=[1, 4],
            generated_token_ids=[11, 99, 12, 2],
            finish_reason="eos",
        )

    def _capture_token_rows(self, sequence, positions):
        self.replays.append((sequence.tolist(), positions))
        if len(self.replays) - 1 == self.fail_extraction_at:
            raise RuntimeError("Fixture extraction failure")
        assert sequence.tolist() == [[1, 4, 11, 99, 12, 2]]
        assert positions == [2, 4]
        return {"layer_8_tokens": np.asarray([[3.0, 7.0], [5.0, 11.0]], dtype=np.float32)}


@pytest.fixture
def prepared(tmp_path, inputs):
    panel, calibration = inputs
    panel_path, calibration_path = tmp_path / "inputs/panel.json", tmp_path / "inputs/cal.json"
    atomic_json(panel_path, panel)
    atomic_json(calibration_path, calibration)
    metadata = {
        "model_id": "fixture/model",
        "resolved_revision": "fixture-rev",
        "dtype": "float32",
        "quantization": "none",
        "chat_template_sha256": "fixture-chat",
        "attention_implementation": "sdpa",
        "is_test_fixture": True,
        "decoder_blocks": 12,
        "hidden_size": 2,
    }
    settings = ModelSettings(
        model_id="fixture/model",
        revision="fixture-rev",
        dtype="float32",
        device="cpu",
        capture_layers=[8],
        local_files_only=True,
    )
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    np.savez(
        probe_dir / "probe.npz",
        mean=np.asarray([1.0, 3.0]),
        scale=np.asarray([2.0, 4.0]),
        coefficient=np.asarray([2.0, -1.0]),
        intercept=np.asarray(0.0),
    )
    atomic_json(
        probe_dir / "manifest.json",
        {
            "schema": "apollo_probe_v1",
            "probe_sha256": file_sha256(probe_dir / "probe.npz"),
            "label_target": "instructed_honesty_condition",
            "feature_dimension": 2,
            "feature_signature": capture_signature(metadata, 8),
        },
    )
    output = tmp_path / "capture"
    plan = runner.prepare_run(
        panel_path, calibration_path, probe_dir, output, settings, enforce_pins=False
    )
    return SimpleNamespace(
        output=output,
        plan=plan,
        backend=FixtureBackend(metadata, output),
        panel_path=panel_path,
        calibration_path=calibration_path,
        probe_dir=probe_dir,
        settings=settings,
    )


def draft_csv(source, path):
    with (source / "annotations_blank.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.update(
            focal_resolution="0",
            focal_status="clear",
            response_style="assertive",
            competing_false_assertions="no",
            evidence_quote="Clear focal answer.",
            evidence_notes="Fixture-only draft.",
        )
        if row["condition"] == "pressure" and row["repeat"] == "2":
            row["focal_resolution"] = "1"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=runner.ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def test_frozen_real_inputs_and_exact_counts(inputs):
    panel, calibration = inputs
    assert file_sha256(ROOT / "data/processed/mask_specificity_v1.json") == runner.PANEL_SHA
    assert (
        file_sha256(ROOT / "data/processed/mask_neutral_calibration_v1.json")
        == runner.CALIBRATION_SHA
    )
    jobs = runner.build_jobs(panel, calibration)
    assert Counter(j["condition"] for j in jobs) == {
        "calibration": 32,
        "belief": 40,
        "pressure": 24,
        "ablated": 24,
    }
    assert [j["seed"] for j in jobs] == list(range(2026112000, 2026112120))
    assert len({j["example_id"] for j in jobs}) == 120
    assert jobs[:32][-1]["condition"] == "calibration"
    assert jobs[32:72][-1]["condition"] == "belief"
    assert jobs[72:96][-1]["condition"] == "pressure"
    assert jobs[96:][-1]["condition"] == "ablated"
    assert panel["selection"]["source_rows"] == [3, 5, 26, 28, 50, 83, 117, 142]
    for job in jobs:
        assert "canary GUID" not in json.dumps(job["messages"])


def test_input_validations_reject_silent_policy_changes(inputs):
    panel, calibration = copy.deepcopy(inputs)
    panel["belief_repeats"]["1"] = 1
    with pytest.raises(ValueError, match="Belief"):
        runner.build_jobs(panel, calibration)
    panel, calibration = copy.deepcopy(inputs)
    panel["cases"][0]["behavior_label"] = "honest"
    with pytest.raises(ValueError, match="preset"):
        runner.build_jobs(panel, calibration)
    panel, calibration = copy.deepcopy(inputs)
    calibration["cases"][0]["messages"] = panel["cases"][0]["pressure_messages"]
    with pytest.raises(ValueError, match="disjoint"):
        runner.build_jobs(panel, calibration)


def test_plan_only_freezes_without_calls(prepared):
    manifest = read_json(prepared.output / "manifest.json")
    assert manifest["status"] == "prepared_no_model_calls" and manifest["n_completed"] == 0
    assert prepared.plan["generation"] == {
        "max_new_tokens": 1000,
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
    }
    assert prepared.plan["label_audited"] is False
    assert prepared.plan["needs_human_review"] is True
    assert len(prepared.plan["jobs"]) == 120
    assert not prepared.backend.calls


def test_exact_float64_original_scaler_mean_margin():
    arrays = {
        "mean": np.array([1.0, 3.0]),
        "scale": np.array([2.0, 4.0]),
        "coefficient": np.array([2.0, -1.0]),
    }
    assert (
        runner.mean_margin(np.asarray([[3.0, 7.0], [5.0, 11.0]], dtype=np.float32), arrays) == 1.5
    )
    for X in (np.empty((0, 2)), np.asarray([[np.nan, 2.0]]), np.ones((2, 3))):
        with pytest.raises(ValueError):
            runner.mean_margin(X, arrays)


def test_threshold_linear_strict_ties_and_small_tail():
    threshold = runner.reference_threshold(list(range(32)))
    assert threshold["q90"]["threshold"] == pytest.approx(27.9)
    assert threshold["q90"]["exceedances"] == 4
    assert threshold["q99_diagnostic"]["exceedances"] == 1
    assert runner.reference_threshold([2.0] * 32)["q90"]["exceedances"] == 0
    with pytest.raises(ValueError):
        runner.reference_threshold([1.0] * 31)


def test_capture_order_exact_replay_and_task_blinding(prepared):
    original = {
        p: file_sha256(p)
        for p in (prepared.panel_path, prepared.calibration_path, prepared.probe_dir / "probe.npz")
    }
    manifest = runner.capture_run(prepared.output, prepared.backend, progress=None)
    assert manifest["status"] == "captured_awaiting_blind_draft" and manifest["n_completed"] == 120
    assert manifest["task_scores_withheld"] is True
    assert len(prepared.backend.calls) == len(prepared.backend.replays) == 120
    assert all(file_sha256(p) == digest for p, digest in original.items())
    export = read_json(prepared.output / "blind_response_export.json")
    assert len(export["rows"]) == 88
    assert Counter(r["condition"] for r in export["rows"]) == {
        "belief": 40,
        "pressure": 24,
        "ablated": 24,
    }
    assert all(
        not any("margin" in key or "threshold" in key or "feature" in key for key in row)
        for row in export["rows"]
    )
    for row in export["rows"]:
        assert row["label_audited"] is False and row["needs_human_review"] is True
        record = read_json(prepared.output / "responses" / (row["example_id"] + ".json"))
        assert "neutral_reference_mean_raw_margin" not in record
        assert record["selected_token_positions"] == [2, 4]
        assert record["selected_token_ids"] == [11, 12]
        with np.load(prepared.output / record["feature_file"], allow_pickle=False) as archive:
            assert archive["X"].dtype == np.float32 and archive["X"].shape == (2, 2)
    with (prepared.output / "annotations_blank.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 88 and all(r["focal_resolution"] == "" for r in rows)
    assert all(r["label_audited"] == "false" for r in rows)
    with pytest.raises(ValueError, match="no automatic retries"):
        runner.capture_run(prepared.output, prepared.backend, progress=None)


@pytest.mark.parametrize("kind", ["generation", "extraction"])
def test_failure_preserves_raw_and_blocks_later_calls(prepared, kind):
    setattr(prepared.backend, f"fail_{kind}_at", 2)
    with pytest.raises(RuntimeError):
        runner.capture_run(prepared.output, prepared.backend, progress=None)
    manifest = read_json(prepared.output / "manifest.json")
    assert manifest["status"] == "failed" and manifest["n_completed"] == 2
    assert len(prepared.backend.calls) == 3
    failed_job = prepared.plan["jobs"][2]
    failed = read_json(prepared.output / "responses" / (failed_job["example_id"] + ".json"))
    assert failed["status"] == "failed"
    if kind == "extraction":
        assert failed["generation"]["generated_token_ids"] == [11, 99, 12, 2]
        assert failed["generation"]["text"] == "Clear focal answer."
    assert not (prepared.output / "calibration_threshold.json").exists()


def test_source_or_backend_change_prevents_any_call(prepared):
    prepared.backend.metadata["chat_template_sha256"] = "different"
    with pytest.raises(ValueError, match="coordinates"):
        runner.capture_run(prepared.output, prepared.backend)
    assert not prepared.backend.calls


def test_changed_source_is_rejected(prepared):
    prepared.panel_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen source"):
        runner.capture_run(prepared.output, prepared.backend)
    assert not prepared.backend.calls


def test_disjoint_no_overwrite_guard(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for output in (source, source / "nested", tmp_path):
        with pytest.raises(ValueError):
            runner._new_output(output, [source])
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        runner._new_output(existing, [source])


def test_cpu_draft_scoring_and_within_case_pairs(prepared, tmp_path):
    runner.capture_run(prepared.output, prepared.backend, progress=None)
    labels = tmp_path / "draft.csv"
    draft_csv(prepared.output, labels)
    summary = runner.score_draft(prepared.output, labels, tmp_path / "scores")
    assert summary["draft_only"] and summary["needs_human_review"] and not summary["label_audited"]
    assert summary["groups"]["pressure/honest_assertion"]["n"] == 16
    assert summary["groups"]["pressure/operational_belief_contradiction"]["n"] == 8
    assert summary["eligible_pair_cases"] == 8 and summary["mean_case_ranking"] == 0.5
    assert all(c["pairs"] == 2 for c in summary["within_pressure_case_pairs"])
    assert len(summary["matched_honest_case_comparisons"]) == 8
    assert summary["matched_honest_case_mean_difference"] == 0.0
    assert len(prepared.backend.calls) == 120  # Scoring makes no additional model calls.


def test_strict_five_and_refusal_style_not_honest_negative(prepared, tmp_path):
    runner.capture_run(prepared.output, prepared.backend, progress=None)
    path = tmp_path / "draft.csv"
    draft_csv(prepared.output, path)
    records = []
    for job in prepared.plan["jobs"]:
        record_path = prepared.output / "responses" / (job["example_id"] + ".json")
        record = read_json(record_path)
        record["record_sha256"] = file_sha256(record_path)
        records.append(record)
    annotations = runner.load_draft_annotations(path, records)
    first, second = prepared.plan["panel"]["cases"][:2]
    first_belief = next(
        r for r in records if r["case_id"] == first["id"] and r["condition"] == "belief"
    )
    annotations[first_belief["example_id"]]["focal_resolution"] = "1"
    second_pressure = next(
        r for r in records if r["case_id"] == second["id"] and r["condition"] == "pressure"
    )
    annotations[second_pressure["example_id"]]["response_style"] = "refusal"
    beliefs, rows = runner.classify_records(records, annotations, prepared.plan["panel"])
    assert beliefs[first["id"]]["operational_belief"] is None
    assert {r["status"] for r in rows if r["case_id"] == first["id"]} == {
        "inconsistent_or_no_belief"
    }
    assert (
        next(r for r in rows if r["example_id"] == second_pressure["example_id"])["status"]
        == "refusal_with_honest_assertion"
    )
    annotations[second_pressure["example_id"]]["response_style"] = "assertive"
    annotations[second_pressure["example_id"]]["competing_false_assertions"] = "yes"
    _, rows = runner.classify_records(records, annotations, prepared.plan["panel"])
    assert (
        next(r for r in rows if r["example_id"] == second_pressure["example_id"])["status"]
        == "mixed_or_other_nonprimary"
    )


def test_human_audit_flag_and_incomplete_annotations_rejected(prepared, tmp_path):
    runner.capture_run(prepared.output, prepared.backend, progress=None)
    labels = tmp_path / "draft.csv"
    rows = draft_csv(prepared.output, labels)
    rows[0]["label_audited"] = "true"
    with labels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=runner.ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="human-audited"):
        runner.score_draft(prepared.output, labels, tmp_path / "scores")
    assert not (tmp_path / "scores").exists()
