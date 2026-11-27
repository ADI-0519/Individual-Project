"""Zero-unknown recovery mechanics use explicit synthetic journals, never models."""

import importlib.util
from pathlib import Path

import pytest
from threadpoolctl import threadpool_info

from latent_probes.storage import atomic_json, file_sha256, read_json

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fresh_recovery", ROOT / "scripts/recover_fresh_readout_campaign.py"
)
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def row_validator(directory, planned, protocol, digest):
    """Replace numerical validation only; frozen numerical validation has its own tests."""
    assert all(pool["num_threads"] <= 4 for pool in threadpool_info())
    complete = read_json(directory / "complete.json")
    assert complete["row_id"] == planned["row_id"]
    assert read_json(directory / "started.json")["protocol_sha256"] == digest
    for name in ("arrays", "raw_forward"):
        assert complete[name + "_sha256"] == file_sha256(directory / (name + ".npz"))
    return complete


def aggregate_validator(directory, bank, protocol, **kwargs):
    manifest = read_json(directory / "manifest.json")
    for name in ("features", "rows"):
        suffix = ".npz" if name == "features" else ".json"
        assert manifest[name + "_sha256"] == file_sha256(directory / (name + suffix))


def write_row(directory, planned, digest, *, engineering=False):
    directory.mkdir(parents=True)
    atomic_json(
        directory / "started.json", {"row_id": planned["row_id"], "protocol_sha256": digest}
    )
    for name in ("arrays", "raw_forward"):
        (directory / (name + ".npz")).write_bytes(b"explicit synthetic journal bytes")
    atomic_json(
        directory / "complete.json",
        {
            **planned,
            "arrays_sha256": file_sha256(directory / "arrays.npz"),
            "raw_forward_sha256": file_sha256(directory / "raw_forward.npz"),
            "capture": {"hook_alignment_checked": engineering},
        },
    )


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    original, target = tmp_path / "original", tmp_path / "recovery"
    original.mkdir()
    source, bank, review = tmp_path / "source.py", tmp_path / "bank.json", tmp_path / "review.json"
    source.write_text("# Explicit engineering fixture\n", encoding="utf-8")
    atomic_json(bank, {"fixture": True})
    atomic_json(review, {"fixture": True})
    sources = {str(source): file_sha256(source)}
    rows = [
        {"row_id": f"fixture_{i}", "partition": part}
        for i, part in enumerate((*recovery.PARTITIONS, "test"))
    ]
    engineering = {"row_id": "fixture_excluded_engineering"}

    def authorization(protocol, digest, path, partition):
        return {
            "bindings_path": str(path),
            "bindings_sha256": file_sha256(path),
            "sha256": {str(path): file_sha256(path)},
        }

    monkeypatch.setattr(recovery, "_authorization", authorization)
    monkeypatch.setattr(recovery, "verify_protocol", lambda protocol: None)
    for model, count in (("llama", 4), ("qwen", 2)):
        directory = original / model
        directory.mkdir()
        protocol = {
            "model_key": model,
            "test_fixture": True,
            "planned_rows": rows,
            "engineering_row": engineering,
            "bank_path": str(bank),
            "source_sha256": sources,
        }
        atomic_json(directory / "protocol.json", protocol)
        digest = file_sha256(directory / "protocol.json")
        atomic_json(directory / "preparation.json", {"protocol_sha256": digest})
        binding = original / (model + "_training_bindings.json")
        atomic_json(binding, {"review_artifact_sha256": {str(review): file_sha256(review)}})
        write_row(directory / "engineering", engineering, digest, engineering=True)
        training = directory / "training"
        (training / "attempts").mkdir(parents=True)
        atomic_json(training / "attempts/0000.json", {"fixture": True})
        for row in rows[:count]:
            write_row(training / "row_journal" / row["row_id"], row, digest)
        manifest = {
            "partition": "training",
            "protocol_sha256": digest,
            "completed_rows": count,
            "status": "complete" if model == "llama" else "running",
            "authorization": authorization(protocol, digest, binding, "training"),
        }
        if model == "llama":
            (training / "features.npz").write_bytes(b"explicit complete aggregate fixture")
            atomic_json(training / "rows.json", {"rows": rows[:count]})
            manifest.update(
                features_sha256=file_sha256(training / "features.npz"),
                rows_sha256=file_sha256(training / "rows.json"),
            )
        else:
            (directory / "execution.lock").write_text("stale preserved lock\n", encoding="utf-8")
        atomic_json(training / "manifest.json", manifest)
        sources[str(binding)] = file_sha256(binding)
        sources[str(directory / "protocol.json")] = digest
    atomic_json(
        original / "campaign_state.json",
        {
            "schema": "fresh-readout-campaign-v1",
            "status": "running",
            "inventory_sha256": "explicit-fixture-inventory",
            "source_sha256": sources,
            "stages": [
                {"name": "llama_training", "status": "complete"},
                {"name": "qwen_training", "status": "running"},
            ],
        },
    )
    return original, target


def prepare(fixture, **kwargs):
    original, target = fixture
    return recovery.prepare_recovery(
        original,
        target,
        attest_no_running_producers=True,
        process_check=lambda: [],
        validator=row_validator,
        aggregate_validator=aggregate_validator,
        allow_test_fixture=True,
        **kwargs,
    )


def test_prepare_copies_exact_completed_prefix_and_preserves_original(fixture):
    original, target = fixture
    before = recovery.inventory(original)
    receipt = prepare(fixture)
    assert receipt["retained_completed_rows"] == 6 and receipt["remaining_qwen_training_rows"] == 2
    assert receipt["unknown_rows"] == [] and receipt["repeat_count"] == 0
    assert (
        receipt["numerical_read_thread_limit"] == 4
        and receipt["engineering_prefills_remaining"] == 0
    )
    assert recovery.inventory(original) == before
    assert not (target / "qwen/execution.lock").exists()
    copied = read_json(target / "copied_inventory.json")
    for name, digest in copied.items():
        assert file_sha256(target / name) == digest
    plan = read_json(target / "campaign_plan.json")
    assert [stage["name"] for stage in plan] == [
        stage["name"]
        for stage in recovery.launcher().build_plan(read_json(target / "recovery_config.json"))
    ]
    assert plan[0]["kind"] == "adopt_completed" and "command" not in plan[0]
    assert plan[1]["command"][-1] == "--resume"
    assert plan[1]["command"][plan[1]["command"].index("--bindings") + 1] == str(
        original / "qwen_training_bindings.json"
    )


@pytest.mark.parametrize("defect", ["started_only", "partial", "failure", "prior_fit"])
def test_any_unknown_or_outcome_bearing_partial_artifact_forbids_recovery(fixture, defect):
    original, target = fixture
    if defect == "started_only":
        directory = original / "qwen/training/row_journal/fixture_2"
        directory.mkdir()
        atomic_json(directory / "started.json", {"row_id": "fixture_2"})
    elif defect == "prior_fit":
        (original / "llama_fit").mkdir()
    else:
        (
            original
            / "qwen/training/row_journal/fixture_1"
            / ("failure.json" if defect == "failure" else "raw.tmp")
        ).write_text("preserved", encoding="utf-8")
    before = recovery.inventory(original)
    with pytest.raises(ValueError):
        prepare(fixture)
    assert recovery.inventory(original) == before and not target.exists()


def fake_runner(target, *, fail_on=None, corrupt_retained=False):
    plan = read_json(target / "campaign_plan.json")
    calls = []

    def child(args, stdout, stderr):
        stage = next(stage for stage in plan if stage.get("command") == args)
        calls.append(stage["name"])
        stdout.write_text("engineering fixture\n", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        if stage["name"] == fail_on:
            return 5
        for filename in stage["outputs"]:
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            if stage["name"] == "freeze_both_fits":
                atomic_json(
                    path,
                    {
                        "fitted_artifact_sha256": {
                            str(target / (model + "_fit") / name): file_sha256(
                                target / (model + "_fit") / name
                            )
                            for model in ("llama", "qwen")
                            for name in recovery.launcher().FIT_FILES
                        }
                    },
                )
            else:
                path.write_bytes(b"explicit continuation fixture output")
        if corrupt_retained and stage["name"] == "qwen_training":
            (target / "qwen/training/row_journal/fixture_0/arrays.npz").write_bytes(
                b"changed retained bytes"
            )
        return 0

    def binder(config, *, test):
        assert test
        result = {}
        for model in ("llama", "qwen"):
            path = target / (model + "_test_bindings.json")
            atomic_json(path, {"fixture": True})
            result[str(path)] = file_sha256(path)
        return result

    return calls, child, binder


def test_execution_adopts_llama_resumes_qwen_once_and_keeps_watcher_compatible_order(fixture):
    original, target = fixture
    prepare(fixture)
    before = recovery.inventory(original)
    calls, child, binder = fake_runner(target)
    state = recovery.run_prepared(
        target,
        attest_no_running_producers=True,
        process_check=lambda: [],
        child=child,
        binder=binder,
        allow_test_fixture=True,
    )
    assert state["status"] == "complete" and state["unchanged_completed"] is True
    assert len(state["stages"]) == 10 and state["stages"][0]["adopted_complete"] is True
    assert calls[0] == "qwen_training" and "llama_training" not in calls
    assert calls.index("fit_qwen") < calls.index("freeze_both_fits") < calls.index("llama_test")
    assert recovery.inventory(original) == before
    with pytest.raises(ValueError, match="already attempted"):
        recovery.run_prepared(
            target,
            attest_no_running_producers=True,
            process_check=lambda: [],
            allow_test_fixture=True,
        )


@pytest.mark.parametrize("corrupt", [False, True])
def test_failure_or_changed_retained_values_stops_before_fitting_and_never_retries(
    fixture, corrupt
):
    _, target = fixture
    prepare(fixture)
    calls, child, binder = fake_runner(
        target, fail_on=None if corrupt else "qwen_training", corrupt_retained=corrupt
    )
    with pytest.raises(ValueError):
        recovery.run_prepared(
            target,
            attest_no_running_producers=True,
            process_check=lambda: [],
            child=child,
            binder=binder,
            allow_test_fixture=True,
        )
    assert calls == ["qwen_training"]
    assert read_json(target / "campaign_state.json")["status"] == "failed_preserved"


def test_active_producer_or_missing_attestation_blocks_copy(fixture):
    original, target = fixture
    for attestation, processes in ((False, []), (True, [123])):
        with pytest.raises(ValueError, match="attestation"):
            recovery.prepare_recovery(
                original,
                target,
                attest_no_running_producers=attestation,
                process_check=lambda: processes,
                allow_test_fixture=True,
            )
    assert not target.exists()
