"""Synthetic completion gating uses the real frozen coordinator's stage plan."""

import ast
import importlib.util
import json
from pathlib import Path

import pytest


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


ROOT = Path(__file__).resolve().parents[1]
watcher = module("fresh_readout_completion", ROOT / "scripts/complete_fresh_readout_campaign.py")
coordinator = module(
    "fresh_readout_coordinator_plan", ROOT / "scripts/run_fresh_readout_campaign.py"
)


@pytest.fixture
def fixture(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    audit_script, reporter_script = tmp_path / "auditor.py", tmp_path / "reporter.py"
    audit_script.write_text("# Independent engineering auditor fixture\n", encoding="utf-8")
    reporter_script.write_text("# Engineering reporter fixture\n", encoding="utf-8")
    config = {
        "campaign_root": str(campaign),
        "watcher_output": str(tmp_path / "watcher"),
        "audit_output": str(tmp_path / "watcher/audit"),
        "report_output": str(tmp_path / "report"),
        "publish_dir": str(tmp_path / "publish"),
        "audit_script": str(audit_script),
        "reporter_script": str(reporter_script),
        "poll_seconds": 0.01,
    }
    # Stage names/order/output maps come from the actual frozen launch code.
    plan = coordinator.build_plan(
        {"output": str(campaign), "bank_path": str(tmp_path / "bank.json")}
    )
    stages = []
    for model in watcher.MODELS:
        directory = campaign / model
        (directory / "test").mkdir(parents=True)
        protocol_path = directory / "protocol.json"
        watcher.fresh_json(protocol_path, {"model_key": model, "test_fixture": False})
        watcher.fresh_json(
            directory / "test/manifest.json",
            {
                "status": "complete",
                "test_fixture": False,
                "completed_rows": 4992,
                "protocol_sha256": watcher.digest(protocol_path),
            },
        )
        analysis = campaign / (model + "_analysis")
        analysis.mkdir()
        # Deliberately invalid JSON proves the watcher never parses outcome files.
        (analysis / "results.json").write_bytes(b"DO NOT PARSE SAVED OUTCOMES IN THE WATCHER")
        (analysis / "predictions.npz").write_bytes(b"explicit engineering prediction fixture")
        watcher.fresh_json(
            analysis / "manifest.json",
            {
                "schema": "fresh-readout-analysis-manifest-v2",
                "source_sha256": {str(protocol_path): watcher.digest(protocol_path)},
                "output_sha256": {
                    name: watcher.digest(analysis / name)
                    for name in ("results.json", "predictions.npz")
                },
            },
        )
    for stage in plan:
        paths = stage["outputs"]
        hashes = {path: watcher.digest(path) for path in paths if Path(path).is_file()}
        stages.append({"name": stage["name"], "status": "complete", "output_sha256": hashes})
    state = {"schema": "fresh-readout-campaign-v1", "status": "complete", "stages": stages}
    watcher.fresh_json(campaign / "campaign_state.json", state)
    return config, state, plan


def fake_child(config, *, audit_failure=None, corrupt_report=False):
    calls = []

    def run(args, stdout, stderr):
        name = "audit" if args[1] == config["audit_script"] else "report"
        calls.append(name)
        stdout.write_text("explicit engineering output\n", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        if name == "audit":
            output = Path(config["audit_output"])
            output.mkdir()
            frozen = {}
            for model in watcher.MODELS:
                directory = output / model
                directory.mkdir()
                watcher.fresh_json(directory / "reconstructed_metrics.json", {"fixture": True})
                frozen[model] = {
                    "reconstructed_metrics.json": watcher.digest(
                        directory / "reconstructed_metrics.json"
                    )
                }
            watcher.fresh_json(
                output / "reconstruction_freeze.json",
                {
                    "saved_results_or_predictions_parsed": False,
                    "completed_output_bytes_hashed_before_reconstruction": True,
                    "sha256": frozen,
                },
            )
            watcher.fresh_json(
                output / "audit_result.json",
                {
                    "schema": "independent-fresh-readout-saved-data-audit-v1",
                    "status": "mismatch" if audit_failure == "mismatch" else "pass",
                    "models": {
                        model: {"all_metrics_match": audit_failure != "mismatch"}
                        for model in watcher.MODELS
                    },
                },
            )
            (output / "access_log.jsonl").write_text(
                json.dumps(
                    {
                        "kind": "audit_started",
                        "source_sha256": "wrong"
                        if audit_failure == "checker"
                        else watcher.digest(config["audit_script"]),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return 1 if audit_failure == "exit" else 0
        output = Path(config["report_output"])
        output.mkdir()
        for name in watcher.EXPORTS:
            (output / name).write_bytes(b"explicit engineering export")
        campaign = Path(config["campaign_root"])
        inputs = {
            str(campaign / (model + "_analysis") / name): watcher.digest(
                campaign / (model + "_analysis") / name
            )
            for model in watcher.MODELS
            for name in ("manifest.json", "results.json", "predictions.npz")
        }
        hashes = {name: watcher.digest(output / name) for name in watcher.EXPORTS}
        if corrupt_report:
            hashes["fresh_readout.pdf"] = "wrong"
        watcher.fresh_json(
            output / "manifest.json",
            {
                "schema": "fresh-readout-report-manifest-v2",
                "status": "complete",
                "test_fixture": False,
                "report_kind": "completed_fresh_pipeline_comparison",
                "research_status": "agent_audited_exploratory",
                "human_review_complete": False,
                "confirmatory_result": False,
                "input_sha256": inputs,
                "reporter_source_sha256": {
                    config["reporter_script"]: watcher.digest(config["reporter_script"])
                },
                "output_sha256": hashes,
            },
        )
        return 0

    return calls, run


def test_real_coordinator_shaped_gate_audits_then_reports_and_copies_exact_bytes(fixture):
    config, _, plan = fixture
    assert tuple(stage["name"] for stage in plan) == watcher.STAGES
    calls, child = fake_child(config)
    result = watcher.watch(config, child=child)
    assert calls == ["audit", "report"] and result["status"] == "complete"
    assert set(Path(config["publish_dir"]).iterdir()) == {
        Path(config["publish_dir"]) / name for name in watcher.EXPORTS | {"manifest.json"}
    }
    for path, expected in result["published_sha256"].items():
        assert watcher.digest(path) == expected
    assert result["new_language_model_calls"] == result["classifier_fits"] == 0
    with pytest.raises(ValueError, match="fresh exclusive outputs"):
        watcher.watch(config, child=child)


def test_running_campaign_polls_only_state_until_both_complete(fixture, monkeypatch):
    config, complete, _ = fixture
    campaign_path = Path(config["campaign_root"]) / "campaign_state.json"
    watcher.update_json(campaign_path, {"schema": "fresh-readout-campaign-v1", "status": "running"})
    reads, original = [], watcher.read

    def observed(path):
        reads.append(Path(path))
        return original(path)

    def sleeper(seconds):
        assert 0 < seconds <= 45
        assert reads == [campaign_path]
        watcher.update_json(campaign_path, complete)

    monkeypatch.setattr(watcher, "read", observed)
    calls, child = fake_child(config)
    assert watcher.watch(config, child=child, sleeper=sleeper)["poll_count"] == 2
    assert calls == ["audit", "report"]


@pytest.mark.parametrize("defect", ["missing_second", "fixture", "incomplete_stage"])
def test_both_native_completion_gates_precede_any_saved_result_hashing(
    fixture, monkeypatch, defect
):
    config, complete, _ = fixture
    campaign = Path(config["campaign_root"])
    if defect == "missing_second":
        (campaign / "qwen_analysis/manifest.json").unlink()
    elif defect == "fixture":
        watcher.update_json(
            campaign / "qwen/protocol.json", {"test_fixture": True, "model_key": "qwen"}
        )
    else:
        next(stage for stage in complete["stages"] if stage["name"] == "analyze_qwen")["status"] = (
            "running"
        )
        watcher.update_json(campaign / "campaign_state.json", complete)
    original, outcome_hashes = watcher.digest, []

    def guarded(path):
        if Path(path).name in {"results.json", "predictions.npz"}:
            outcome_hashes.append(str(path))
        return original(path)

    monkeypatch.setattr(watcher, "digest", guarded)
    calls, child = fake_child(config)
    with pytest.raises((ValueError, FileNotFoundError)):
        watcher.watch(config, child=child)
    assert calls == outcome_hashes == []


@pytest.mark.parametrize("failure", ["exit", "mismatch", "checker"])
def test_audit_must_pass_before_report_or_publication(fixture, failure):
    config, _, _ = fixture
    calls, child = fake_child(config, audit_failure=failure)
    with pytest.raises(ValueError):
        watcher.watch(config, child=child)
    assert calls == ["audit"] and not Path(config["publish_dir"]).exists()
    assert not Path(config["report_output"]).exists()
    receipt = watcher.read(Path(config["watcher_output"]) / "completion_receipt.json")
    assert receipt["status"] == "failed_preserved"


def test_failed_campaign_and_changed_helper_never_start_auditor(fixture):
    config, _, _ = fixture
    watcher.update_json(
        Path(config["campaign_root"]) / "campaign_state.json", {"status": "failed_preserved"}
    )
    calls, child = fake_child(config)
    with pytest.raises(ValueError, match="Campaign stopped or failed"):
        watcher.watch(config, child=child)
    assert calls == []


def test_changed_reporter_while_waiting_stops_before_result_access(fixture):
    config, _, _ = fixture
    watcher.update_json(
        Path(config["campaign_root"]) / "campaign_state.json", {"status": "running"}
    )
    calls, child = fake_child(config)

    def sleeper(_seconds):
        Path(config["reporter_script"]).write_text("changed helper", encoding="utf-8")

    with pytest.raises(ValueError, match="Changed frozen file"):
        watcher.watch(config, child=child, sleeper=sleeper)
    assert calls == []


def test_corrupt_report_never_copies_exports(fixture):
    config, _, _ = fixture
    calls, child = fake_child(config, corrupt_report=True)
    with pytest.raises(ValueError, match="Changed frozen file"):
        watcher.watch(config, child=child)
    assert calls == ["audit", "report"] and not Path(config["publish_dir"]).exists()


def test_watcher_source_imports_only_standard_library():
    tree = ast.parse(
        (ROOT / "scripts/complete_fresh_readout_campaign.py").read_text(encoding="utf-8")
    )
    modules = {
        node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    modules |= {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert modules <= {
        "__future__",
        "argparse",
        "hashlib",
        "json",
        "os",
        "platform",
        "shutil",
        "subprocess",
        "sys",
        "tempfile",
        "time",
        "datetime",
        "pathlib",
    }
