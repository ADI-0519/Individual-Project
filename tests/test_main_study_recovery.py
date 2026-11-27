"""Conservative recovery plumbing using explicit fake measurements, never models."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from latent_probes.crossed_belief_validation import summarize_logits
from latent_probes.main_study_run import _complete_rows
from latent_probes.storage import atomic_json, atomic_npz, canonical_hash, file_sha256

SOURCE = Path(__file__).resolve().parents[1] / "scripts/recover_main_study_campaign.py"
spec = importlib.util.spec_from_file_location("recovery_script_fixture", SOURCE)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


@pytest.fixture
def interrupted(tmp_path):
    source = tmp_path / "old_capture"
    planned = [
        {
            "row_id": f"explicit_fixture_{index}",
            "split": "test",
            "prompt_token_ids_sha256": "fixture",
        }
        for index in range(4)
    ]
    protocol = {
        "hidden_size": 2,
        "tokenizer": {"answer_token_ids": {"YES": 3, "NO": 4}},
        "planned_rows": planned,
    }
    atomic_json(source / "protocol.json", protocol)
    sha = file_sha256(source / "protocol.json")
    for index in range(2):
        directory = source / "row_journal" / planned[index]["row_id"]
        feature = np.asarray([index, 1], dtype=np.float32)
        logits = np.asarray([0, 0, 0, 2, 1], dtype=np.float32)
        atomic_npz(directory / "arrays.npz", feature=feature, next_token_logits=logits)
        atomic_npz(
            directory / "raw_forward.npz", feature=feature, full_forward_logits=logits[None, None]
        )
        atomic_json(
            directory / "started.json", {"row_id": planned[index]["row_id"], "protocol_sha256": sha}
        )
        atomic_json(
            directory / "complete.json",
            {
                **planned[index],
                "status": "complete",
                "protocol_sha256": sha,
                "arrays_sha256": file_sha256(directory / "arrays.npz"),
                "raw_forward_sha256": file_sha256(directory / "raw_forward.npz"),
                "feature_values_sha256": canonical_hash(feature.tolist()),
                "output_logits": summarize_logits(
                    logits, protocol["tokenizer"]["answer_token_ids"]
                ),
            },
        )
    atomic_json(
        source / "row_journal" / planned[2]["row_id"] / "started.json",
        {"row_id": planned[2]["row_id"], "protocol_sha256": sha},
    )
    atomic_json(
        source / "manifest.json",
        {
            "status": "running",
            "partition": "test",
            "protocol_sha256": sha,
            "completed_rows": 2,
            "authorization": {"fixture_only": True},
        },
    )
    atomic_json(source / "attempts" / "0000.json", {"explicit_fixture": True})
    return source, tmp_path / "new_capture", protocol


def snapshot(root):
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_completed_rows_reused_byte_identically_and_unknown_preserved(interrupted):
    source, destination, protocol = interrupted
    original = snapshot(source)
    receipt = recovery.copy_interrupted_capture(source, destination, validator=_complete_rows)
    assert snapshot(source) == original
    assert receipt["retained_completed_rows"] == 2 and receipt["remaining_rows"] == 2
    assert receipt["unknown_attempt"]["case_id"] == "explicit_fixture_2"
    assert receipt["unknown_attempt"]["planned_repeat_count"] == 1
    assert receipt["unknown_attempt"]["counts_as_new_independent_case"] is False
    assert not (destination / "row_journal" / "explicit_fixture_2").exists()
    assert receipt["remaining_case_ids"] == ["explicit_fixture_2", "explicit_fixture_3"]
    assert _complete_rows(destination, protocol, protocol["planned_rows"])
    for name, value in receipt["copied_artifact_sha256"].items():
        assert value == original[name] == file_sha256(destination / name)
    with pytest.raises(ValueError, match="fresh"):
        recovery.copy_interrupted_capture(source, destination, validator=_complete_rows)


@pytest.mark.parametrize(
    "mutation",
    [
        "raw_unknown",
        "failure_unknown",
        "second_unknown",
        "count_mismatch",
        "bad_complete",
        "extra_file",
    ],
)
def test_invalid_or_more_ambiguous_recovery_is_not_automatically_authorized(interrupted, mutation):
    source, destination, _ = interrupted
    unknown = source / "row_journal" / "explicit_fixture_2"
    if mutation == "raw_unknown":
        (unknown / "raw_forward.npz").write_bytes(b"An outcome may exist; never discard")
    elif mutation == "failure_unknown":
        atomic_json(unknown / "failure.json", {"error": "Explicit failure"})
    elif mutation == "second_unknown":
        atomic_json(
            source / "row_journal" / "explicit_fixture_3" / "started.json", {"fixture": True}
        )
    elif mutation == "count_mismatch":
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        atomic_json(source / "manifest.json", dict(manifest, completed_rows=1))
    elif mutation == "bad_complete":
        (source / "row_journal" / "explicit_fixture_0" / "arrays.npz").write_bytes(b"corrupted")
    else:
        (source / "row_journal" / "explicit_fixture_0" / "unrecorded.txt").write_text(
            "unknown", encoding="utf-8"
        )
    original = snapshot(source)
    with pytest.raises(ValueError):
        recovery.copy_interrupted_capture(source, destination, validator=_complete_rows)
    assert snapshot(source) == original


def test_original_validator_failure_leaves_copy_and_never_returns_receipt(interrupted):
    source, destination, _ = interrupted
    calls = []

    def reject(*arguments):
        calls.append(arguments)
        raise ValueError("Original checker rejected saved observations")

    with pytest.raises(ValueError, match="checker rejected"):
        recovery.copy_interrupted_capture(source, destination, validator=reject)
    assert len(calls) == 1 and destination.exists()
    assert not (destination / "recovery_receipt.json").exists()


def test_fixed_continuation_never_refits_or_refreezes_and_repeats_only_explicit_row(tmp_path):
    old = {
        "campaign": str(tmp_path / "old"),
        "bank": "bank",
        "agent_audit": "audit",
        "training_freeze": "initial",
        "llama_train": "llama_train",
        "qwen_train": "qwen_train",
        "monitor_development": "monitor",
    }
    folder = tmp_path / "new"
    stages = recovery.continuation_plan(old, folder)
    assert stages[0]["name"] == "explicit_resume_llama_test"
    assert stages[0]["command"][-1] == "--resume"
    assert sum("--resume" in stage["command"] for stage in stages) == 1
    assert not any(
        "fit_main_study.py" in Path(argument).name
        or "prepare_main_study_execution.py" in Path(argument).name
        for stage in stages
        for argument in stage["command"]
    )
    assert not any(stage["name"] == "prepare_llama_test" for stage in stages)
    assert stages[-1]["name"] == "report_frozen_results"
    for stage in stages:
        if "--fit-dir" in stage["command"] or "--fit" in stage["command"]:
            assert str(folder / "llama_fit") not in stage["command"]
            assert str(folder / "qwen_fit") not in stage["command"]
        assert str(folder / "test_freeze.json") not in stage["command"]


def test_machine_process_check_identifies_producers(tmp_path):
    for number, args in (
        (1001, ["python", "/repo/scripts/run_main_study.py", "execute"]),
        (1002, ["python", "unrelated.py"]),
    ):
        directory = tmp_path / str(number)
        directory.mkdir()
        (directory / "cmdline").write_bytes(b"\0".join(value.encode() for value in args))
    assert [row["pid"] for row in recovery.active_producers(tmp_path)] == [1001]


def test_prepare_requires_explicit_absence_and_refuses_live_producer(tmp_path):
    with pytest.raises(ValueError, match="attestation"):
        recovery.prepare_recovery(
            tmp_path / "old",
            tmp_path / "new",
            attest_no_running_producers=False,
            process_check=lambda: [],
        )
    with pytest.raises(ValueError, match="still running"):
        recovery.prepare_recovery(
            tmp_path / "old",
            tmp_path / "new",
            attest_no_running_producers=True,
            process_check=lambda: [{"pid": 123}],
        )
    assert not (tmp_path / "new").exists()


@pytest.fixture
def prepared_fixture(tmp_path):
    """Explicit fake sealed recovery, never a scientific configuration."""
    root = tmp_path / "fake_recovery"
    root.mkdir()
    (root / "logs").mkdir()
    saved = root / "llama_test" / "fixture_artifact.txt"
    saved.parent.mkdir()
    saved.write_text("explicit test artifact", encoding="utf-8")
    receipt = {
        "copied_artifact_sha256": {"fixture_artifact.txt": file_sha256(saved)},
        "unknown_attempt": {"explicit_fixture_only": True},
    }
    atomic_json(root / "recovery_receipt.json", receipt)
    config = {
        "schema": "main-study-explicit-recovery-campaign-v1",
        "campaign": str(root),
        "source_sha256": {
            str(root / "recovery_receipt.json"): file_sha256(root / "recovery_receipt.json")
        },
        "recovery_receipt_sha256": file_sha256(root / "recovery_receipt.json"),
        "stages": [
            {"name": "fake_test_stage", "command": ["explicit-pytest-command"], "outputs": []}
        ],
    }
    atomic_json(root / "config.json", config)
    atomic_json(
        root / "prepared_manifest.json",
        {
            "schema": "main-study-recovery-preparation-seal-v1",
            "sha256": {
                name: file_sha256(root / name) for name in ("config.json", "recovery_receipt.json")
            },
        },
    )
    atomic_json(root / "state.json", {"status": "prepared_no_model_calls"})
    return root


def test_prepared_configuration_and_copied_measurements_are_checked_before_child(prepared_fixture):
    root = prepared_fixture
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config["stages"][0]["command"] = ["unapproved edited command"]
    atomic_json(root / "config.json", config)
    with pytest.raises(ValueError, match="Changed campaign input"):
        recovery.run_prepared(
            root,
            child=lambda *a: pytest.fail("No edited command may run"),
            process_check=lambda: [],
        )
    assert not (root / "execution_started.json").exists()


def test_failed_recovery_child_is_never_automatically_retried(prepared_fixture):
    root, calls = prepared_fixture, []

    def fail(*arguments):
        calls.append(arguments)
        return {"returncode": 1, "output_tail": "explicit engineering failure"}

    with pytest.raises(ValueError, match="no retry"):
        recovery.run_prepared(root, child=fail, process_check=lambda: [])
    assert len(calls) == 1
    assert (root / "execution_started.json").exists()
    assert (
        json.loads((root / "state.json").read_text(encoding="utf-8"))["status"]
        == "failed_or_interrupted"
    )
    with pytest.raises(ValueError, match="never automatically retry"):
        recovery.run_prepared(
            root, child=lambda *a: pytest.fail("No retry"), process_check=lambda: []
        )


def test_corrupted_prepared_measurement_never_starts_child(prepared_fixture):
    root = prepared_fixture
    (root / "llama_test" / "fixture_artifact.txt").write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="Changed campaign input"):
        recovery.run_prepared(
            root,
            child=lambda *a: pytest.fail("No corrupted measurement may run"),
            process_check=lambda: [],
        )
    assert not (root / "execution_started.json").exists()
