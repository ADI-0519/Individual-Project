"""Coordinator order, failure preservation and detached launch use no model calls."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from latent_probes.storage import atomic_json, file_sha256, read_json

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_fresh_readout_campaign.py"
SPEC = importlib.util.spec_from_file_location("fresh_readout_campaign", SCRIPT)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


@pytest.fixture
def config(tmp_path):
    output = tmp_path / "prepared"
    output.mkdir()
    source, review = tmp_path / "source.txt", tmp_path / "review.json"
    source.write_text("explicit coordinator engineering fixture", encoding="utf-8")
    atomic_json(review, {"schema": "explicit-review-fixture"})
    return {
        "output": str(output),
        "bank_path": str(tmp_path / "bank.json"),
        "inventory_sha256": "explicit-engineering-fixture",
        "source_sha256": {str(source): file_sha256(source)},
        "review_artifact_sha256": {str(review): file_sha256(review)},
    }


def test_plan_seals_both_fits_before_either_test_and_scores_only_after_both(config):
    stages = campaign.build_plan(config)
    assert [stage["name"] for stage in stages] == [
        "llama_training",
        "qwen_training",
        "fit_llama",
        "fit_qwen",
        "freeze_both_fits",
        "bind_both_tests",
        "llama_test",
        "qwen_test",
        "analyze_llama",
        "analyze_qwen",
    ]
    assert sum(stage.get("cohort_rows", 0) for stage in stages) == 14080
    assert all("--resume" not in stage.get("command", []) for stage in stages)
    assert all(isinstance(arg, str) for stage in stages for arg in stage.get("command", []))


def fake_executor(config, *, fail_on=None):
    stages = campaign.build_plan(config)
    calls = []

    def child(args, stdout, stderr):
        stage = next(stage for stage in stages if stage.get("command") == args)
        calls.append(stage["name"])
        stdout.write_text("engineering fixture output\n", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        if stage["name"] == fail_on:
            return 7
        for name in stage["outputs"]:
            path = Path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            if stage["name"] == "freeze_both_fits":
                output = Path(config["output"])
                atomic_json(
                    path,
                    {
                        "fitted_artifact_sha256": {
                            str(output / (key + "_fit") / filename): file_sha256(
                                output / (key + "_fit") / filename
                            )
                            for key in campaign.MODELS_ORDER
                            for filename in campaign.FIT_FILES
                        }
                    },
                )
            else:
                path.write_bytes(b"explicit coordinator fixture artifact")
        return 0

    def binder(config, *, test=False):
        calls.append("bind_tests" if test else "bind_training")
        paths = {}
        for key in campaign.MODELS_ORDER:
            path = Path(config["output"]) / f"{key}_{'test' if test else 'training'}_bindings.json"
            atomic_json(path, {"schema": "explicit-binding-fixture"})
            paths[str(path)] = file_sha256(path)
        return paths

    return calls, child, binder


def test_complete_campaign_runs_fixed_stages_once_and_archives_sources(config):
    calls, child, binder = fake_executor(config)
    result = campaign.run_campaign(config, child=child, binder=binder)
    assert result["status"] == "complete" and len(result["stages"]) == 10
    assert calls.index("bind_tests") < calls.index("llama_test") < calls.index("qwen_test")
    assert calls.index("qwen_test") < calls.index("analyze_llama")
    assert result["cohort_prefills_planned"] == 14080
    assert result["excluded_engineering_prefills_planned"] == 2
    assert len(read_json(result["source_archive_manifest"])) == 4
    assert not (Path(config["output"]) / "campaign.lock").exists()
    with pytest.raises(ValueError, match="already attempted"):
        campaign.run_campaign(config, child=child, binder=binder)


def test_failed_stage_stops_without_retry_and_keeps_logs_and_state(config):
    calls, child, binder = fake_executor(config, fail_on="qwen_training")
    with pytest.raises(ValueError, match="exit 7"):
        campaign.run_campaign(config, child=child, binder=binder)
    assert calls == ["bind_training", "llama_training", "qwen_training"]
    output = Path(config["output"])
    state = read_json(output / "campaign_state.json")
    assert state["status"] == state["stages"][-1]["status"] == "failed_preserved"
    assert state["stages"][-1]["returncode"] == 7
    assert (output / "campaign_logs/qwen_training.stdout.log").is_file()
    assert not (output / "llama_fit").exists()
    with pytest.raises(ValueError, match="already attempted"):
        campaign.run_campaign(config, child=child, binder=binder)


def test_unknown_lock_and_prior_stage_output_fail_before_child(config):
    output = Path(config["output"])
    (output / "campaign.lock").write_text("unknown", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown campaign lock"):
        campaign.run_campaign(config)
    (output / "campaign.lock").unlink()
    (output / "llama/training").mkdir(parents=True)
    (output / "llama/training/manifest.json").write_text("{}", encoding="utf-8")
    calls, child, binder = fake_executor(config)
    with pytest.raises(ValueError, match="Prior stage output"):
        campaign.run_campaign(config, child=child, binder=binder)
    assert calls == []


def test_changed_source_fails_before_binding_or_process(config):
    source = next(iter(config["source_sha256"]))
    Path(source).write_text("changed", encoding="utf-8")
    calls, child, binder = fake_executor(config)
    with pytest.raises(ValueError, match="Changed input"):
        campaign.run_campaign(config, child=child, binder=binder)
    assert calls == []


def test_detached_launch_has_no_shell_and_uses_offline_fixed_threads(config, monkeypatch):
    monkeypatch.setattr(campaign, "preflight", lambda *args: config)
    observed = []

    def popen(args, **kwargs):
        observed.append((args, kwargs))
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(campaign.subprocess, "Popen", popen)
    result = campaign.detach(config["output"], "review.json")
    assert result["pid"] == 12345 and result["status"] == "launched"
    args, options = observed[0]
    assert "--detach" not in args
    assert options["shell"] is False and options["start_new_session"] is True
    assert options["env"]["HF_HUB_OFFLINE"] == options["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert all(
        options["env"][name] == "4"
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    )
    with pytest.raises(FileExistsError):
        campaign.detach(config["output"], "review.json")
    assert len(observed) == 1
