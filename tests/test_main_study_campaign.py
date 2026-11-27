"""Launcher plumbing only: no models, scientific cases or GPU jobs are executed."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/run_main_study_campaign.py"
spec = importlib.util.spec_from_file_location("main_study_campaign_script", SOURCE)
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


@pytest.fixture
def config(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("explicit engineering fixture", encoding="utf-8")
    folder = tmp_path / "new_campaign"
    return {
        "campaign": str(folder),
        "stop_file": str(folder / "STOP"),
        "llama_train": str(tmp_path / "llama"),
        "qwen_train": str(tmp_path / "qwen"),
        "monitor_development": str(tmp_path / "monitor"),
        "bank": str(tmp_path / "bank.json"),
        "agent_audit": str(tmp_path / "agent.json"),
        "training_freeze": str(tmp_path / "initial.json"),
        "bank_sha256": "explicit engineering fixture",
        "wait_timeout_seconds": 10,
        "poll_seconds": 0.01,
        "source_sha256": {str(source): campaign.digest(source)},
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_fixed_plan_waits_external_llama_and_freezes_after_both_fits(config):
    stages = campaign.build_plan(config)
    names = [stage["name"] for stage in stages]
    assert stages[0]["kind"] == "wait_llama"
    assert "execute_llama_train_dev" not in names
    assert names.index("fit_llama") < names.index("freeze_test")
    assert names.index("fit_qwen") < names.index("freeze_test")
    assert names.index("freeze_test") < names.index("prepare_llama_test")
    assert names.index("analyse_qwen_without_monitor") < names.index("execute_monitor_development")
    assert names[-2:] == ["analyse_llama_with_monitor", "analyse_qwen_with_monitor"]
    assert len(names) == len(set(names))
    for stage in stages[1:]:
        assert stage["command"][0] == sys.executable
        assert "--resume" not in stage["command"] and "--allow-fixture" not in stage["command"]
    freeze = next(stage for stage in stages if stage["name"] == "freeze_test")
    assert freeze["command"].count("--fit-dir") == 2
    assert (
        freeze["command"][freeze["command"].index("--initial-freeze") + 1]
        == config["training_freeze"]
    )


def test_wait_adopts_external_run_without_loading_or_retrying(tmp_path):
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"status": "running", "completed_rows": 1})
    slept, validated = [], []

    def sleeper(seconds):
        slept.append(seconds)
        write_json(manifest, {"status": "complete"})

    result = campaign.wait_for_capture(
        tmp_path,
        timeout=1,
        poll=0.01,
        stop_file=tmp_path / "STOP",
        sleeper=sleeper,
        validator=lambda: validated.append(True) or {"complete": True},
        progress=lambda *a, **k: None,
    )
    assert slept == [0.01] and validated == [True] and result == {"complete": True}


@pytest.mark.parametrize(
    "status", ["prepared_no_weights", "failed_or_interrupted", "unknown", None]
)
def test_wait_never_restarts_unknown_or_failed_capture(tmp_path, status):
    write_json(tmp_path / "manifest.json", {"status": status})
    with pytest.raises(ValueError, match="no retry"):
        campaign.wait_for_capture(
            tmp_path,
            timeout=1,
            poll=0.01,
            stop_file=tmp_path / "STOP",
            validator=lambda: pytest.fail("Must not validate a failed run"),
        )


def test_external_wait_is_bounded_and_stop_does_not_touch_run(tmp_path):
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"status": "running"})
    times = iter((0, 2))
    with pytest.raises(TimeoutError):
        campaign.wait_for_capture(
            tmp_path,
            timeout=1,
            poll=0.01,
            stop_file=tmp_path / "STOP",
            clock=lambda: next(times),
            validator=lambda: None,
            progress=lambda *a, **k: None,
        )
    (tmp_path / "STOP").touch()
    with pytest.raises(InterruptedError):
        campaign.wait_for_capture(
            tmp_path, timeout=1, poll=0.01, stop_file=tmp_path / "STOP", validator=lambda: None
        )
    assert campaign.read(manifest)["status"] == "running"


def test_existing_output_and_failed_child_are_not_retried(config, tmp_path):
    existing = tmp_path / "already_there.json"
    existing.write_text("keep", encoding="utf-8")
    stage = {"name": "fixture", "kind": "command", "command": ["fixture"], "fresh": [str(existing)]}
    with pytest.raises(ValueError, match="no overwrite/retry"):
        campaign.execute_stage(stage, config, child=lambda *a: pytest.fail("Must not start child"))
    calls = []
    stage["fresh"] = []
    with pytest.raises(ValueError, match="Child failed"):
        campaign.execute_stage(
            stage,
            config,
            child=lambda *a: calls.append(a) or {"returncode": 2, "output_tail": "fixture failure"},
        )
    assert len(calls) == 1 and existing.read_text(encoding="utf-8") == "keep"


def test_child_output_is_bounded_and_arguments_are_literal(tmp_path):
    literal = "$(not-a-command); `also literal`"
    log = tmp_path / "child.log"
    result = campaign.run_child(
        [sys.executable, "-c", "import sys; print(sys.argv[1]); print('x'*1000)", literal],
        log,
        limit=100,
    )
    assert result["returncode"] == 0 and result["log_bytes"] == 100
    assert result["discarded_stdout_bytes"] > 0 and log.stat().st_size == 100
    assert literal in result["output_tail"]


def test_campaign_stop_keeps_state_and_refuses_repeat(config):
    calls = []

    def stop_after_first(stage, cfg):
        calls.append(stage["name"])
        Path(cfg["stop_file"]).touch()
        return {"fixture_only": True}

    with pytest.raises(InterruptedError):
        campaign.run_campaign(config, executor=stop_after_first)
    state = campaign.read(Path(config["campaign"]) / "state.json")
    assert state["status"] == "stopped" and calls == ["adopt_llama_train"]
    assert state["stages"][0]["status"] == "complete"
    with pytest.raises(ValueError, match="never resumed"):
        campaign.run_campaign(config, executor=lambda *a: pytest.fail("No rerun"))


def test_source_change_fails_before_campaign_folder(config):
    Path(next(iter(config["source_sha256"]))).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Changed campaign input"):
        campaign.run_campaign(config, executor=lambda *a: pytest.fail("Must not execute"))
    assert not Path(config["campaign"]).exists()


def test_all_successes_record_fixed_command_plan_and_hashes(config):
    seen = []
    state = campaign.run_campaign(
        config, executor=lambda stage, _: seen.append(stage["name"]) or {"fixture_only": True}
    )
    assert state["status"] == "complete"
    plan = campaign.read(Path(config["campaign"]) / "config.json")
    assert seen == [stage["name"] for stage in plan["stages"]]
    assert all(stage["status"] == "complete" for stage in state["stages"])
