"""Model-free failure gates, exact token coordinates and fresh transfer lineage."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from latent_probes.calibrated_reader import policy_metadata
from latent_probes.storage import file_sha256, read_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "recipient_transfer_test_subject", ROOT / "scripts/run_recipient_transfer.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


@pytest.fixture
def qualified_fixture(tmp_path, runner):
    """Synthetic gate records exercise validation; they are not research evidence."""
    gate = tmp_path / "explicit_fixture_gate"
    gate.mkdir()
    dependency = tmp_path / "explicit_fixture_source.txt"
    dependency.write_text("Fixture source remains bound.\n")
    jobs, controls = [], []
    for i in range(24):
        for context in ("informed", "absent", "stale"):
            answer = "DEFER" if context == "absent" else ("APPROVE", "REJECT")[i % 2]
            job = {
                "example_id": f"fixture_{i}_{context}",
                "case_id": f"fixture_{i}",
                "context": context,
                "expected_decision": answer,
                "messages": [{"role": "user", "content": "Explicit unit-test fixture."}],
            }
            jobs.append(job)
            controls.append(
                {**job, "generation": {"text": answer, "finish_reason": "eos"}, "decision": answer}
            )
    protocol = {
        "policy": policy_metadata(),
        "source_sha256": {str(dependency): file_sha256(dependency)},
        "fresh_jobs": jobs,
        "is_test_fixture": True,
    }
    summary = {"fresh_gate": runner.capability_gate(controls), "is_test_fixture": True}
    save(gate / "protocol.json", protocol)
    save(gate / "summary.json", summary)
    save(gate / "fresh_controls.json", controls)
    save(
        gate / "manifest.json",
        {
            "status": "complete_fresh_gate_passed",
            "source_unchanged": True,
            "protocol_sha256": file_sha256(gate / "protocol.json"),
            "summary_sha256": file_sha256(gate / "summary.json"),
            "is_test_fixture": True,
        },
    )
    return gate, dependency


def test_gate_is_reconstructed_from_actual_bound_control_answers(runner, qualified_fixture):
    gate, _ = qualified_fixture
    result = runner.require_fresh_gate(gate)
    assert result["fresh_controls_sha256"] == file_sha256(gate / "fresh_controls.json")
    assert result["protocol_sha256"] == file_sha256(gate / "protocol.json")


@pytest.mark.parametrize(
    "corruption", ["failed", "policy", "source", "record", "counts", "missing"]
)
def test_bad_or_changed_qualification_never_authorizes_transfer(
    runner, qualified_fixture, corruption
):
    gate, dependency = qualified_fixture
    if corruption == "failed":
        manifest = read_json(gate / "manifest.json")
        manifest["status"] = "fresh_reader_controls_failed"
        save(gate / "manifest.json", manifest)
    elif corruption == "policy":
        protocol = read_json(gate / "protocol.json")
        protocol["policy"]["policy_id"] = "another_policy"
        save(gate / "protocol.json", protocol)
    elif corruption == "source":
        dependency.write_text("Changed source\n")
    else:
        controls = read_json(gate / "fresh_controls.json")
        if corruption == "record":
            controls[0]["decision"] = "DEFER"  # Disagrees with its raw generation.
        elif corruption == "counts":
            for control in [r for r in controls if r["context"] == "absent"][:5]:
                control["decision"] = "APPROVE"
                control["generation"]["text"] = "APPROVE"
        else:
            controls.pop()
        save(gate / "fresh_controls.json", controls)
    with pytest.raises(ValueError):
        runner.require_fresh_gate(gate)


def test_failed_gate_prevents_backend_fit_output_and_fresh_generation(
    runner, qualified_fixture, monkeypatch, tmp_path
):
    gate, _ = qualified_fixture
    manifest = read_json(gate / "manifest.json")
    manifest["status"] = "fresh_reader_controls_failed"
    save(gate / "manifest.json", manifest)

    def forbidden(*args, **kwargs):
        pytest.fail("A failed capability gate must prevent every model and fitting call")

    monkeypatch.setattr(runner, "real_backend", forbidden)
    import latent_probes.recipient_readout as fitting

    monkeypatch.setattr(fitting, "fit_recipient_readouts", forbidden)
    monkeypatch.setattr(fitting, "evaluate_recipient_readouts", forbidden)
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="fresh gate"):
        runner.run(SimpleNamespace(gate=gate, output=output))
    assert not output.exists()


def physical_record(forced):
    common = {
        "text": "A.",
        "selected_token_ids": [11, 12],
        "selected_token_positions": [1, 2],
        "selected_decoded_text": "A.",
    }
    if forced:
        return {
            **common,
            "mode": "off_policy_fixed_response",
            "input_token_ids": [90, 11, 12, 99],
            "rendered_dialogue": "HA.Z",
            "character_span": [1, 3],
            "trim_last_words": 0,
            "excluded_suffix": "",
        }
    return {
        **common,
        "mode": "on_policy_generated_response",
        "prompt_token_ids": [90],
        "generated_token_ids": [11, 12, 99],
        "finish_reason": "eos",
    }


@pytest.mark.parametrize("forced", [True, False])
def test_physical_feature_is_final_report_content_row_not_mean(runner, forced):
    matrix = np.stack([np.ones(4096), np.full(4096, 7.0)]).astype(np.float32)
    result = runner.report_vector({"layer_8_tokens": matrix}, physical_record(forced), {90, 99})
    assert np.array_equal(result, matrix[-1])
    result[:] = 0
    assert np.all(matrix[-1] == 7)  # The returned last row is detached.


@pytest.mark.parametrize(
    "corruption",
    [
        "nan",
        "width",
        "special",
        "positions_count",
        "position_range",
        "id_binding",
        "mode",
        "trim",
        "span",
        "decode",
    ],
)
def test_bad_physical_coordinates_or_forced_report_mask_are_rejected(runner, corruption):
    matrix = np.ones((2, 4096), dtype=np.float32)
    record = physical_record(True)
    if corruption == "nan":
        matrix[0, 0] = np.nan
    elif corruption == "width":
        matrix = matrix[:, :-1]
    elif corruption == "special":
        record["selected_token_ids"][0] = 90
    elif corruption == "positions_count":
        record["selected_token_positions"] = [1]
    elif corruption == "position_range":
        record["selected_token_positions"] = [1, 4]
    elif corruption == "id_binding":
        record["selected_token_ids"] = [12, 11]
    elif corruption == "mode":
        record["mode"] = "unverified"
    elif corruption == "trim":
        record["trim_last_words"] = 1
    elif corruption == "span":
        record["character_span"] = [0, 2]
    else:
        record["selected_decoded_text"] = "Other text"
    with pytest.raises(ValueError):
        runner.report_vector({"layer_8_tokens": matrix}, record, {90, 99})


@pytest.mark.parametrize("corruption", ["prompt_row", "missing_final_body_token"])
def test_natural_mask_must_include_all_and_only_generated_content(runner, corruption):
    record = physical_record(False)
    if corruption == "prompt_row":
        record["prompt_token_ids"] = [31]
        record["selected_token_positions"] = [0, 1]
        record["selected_token_ids"] = [31, 11]
        matrix = np.ones((2, 4096), dtype=np.float32)
    else:
        record["selected_token_positions"] = [1]
        record["selected_token_ids"] = [11]
        matrix = np.ones((1, 4096), dtype=np.float32)
    with pytest.raises(ValueError):
        runner.report_vector({"layer_8_tokens": matrix}, record, {90, 99})


def fixture_capture_backend(*, wrong_text=False):
    def fixed(messages, text, trim_last_words):
        assert trim_last_words == 0
        record = physical_record(True)
        if wrong_text:
            record.update(text="B.", rendered_dialogue="HB.Z", selected_decoded_text="B.")
        return {"layer_8_tokens": np.ones((2, 4096), dtype=np.float32)}, record

    return SimpleNamespace(
        metadata={"model_id": "explicit_fixture", "is_test_fixture": True, "decoder_blocks": 32},
        tokenizer=SimpleNamespace(all_special_ids=[90, 99]),
        forced_token_activations=fixed,
    )


def test_forced_capture_cannot_replace_authored_report_and_saves_vectors(runner, tmp_path):
    job = {
        "example_id": "explicit_fixture",
        "messages": [{"role": "user", "content": "input"}],
        "report_text": "A.",
    }
    with pytest.raises(ValueError, match="authored report"):
        runner.capture(
            [job], fixture_capture_backend(wrong_text=True), tmp_path / "bad", forced=True
        )
    rows, matrix, signature = runner.capture(
        [job], fixture_capture_backend(), tmp_path / "good", forced=True
    )
    assert rows[0]["report_text"] == "A."
    assert signature["pooling"] == "last_non_special_report_token"
    with np.load(tmp_path / "good/captured_features.npz", allow_pickle=False) as archive:
        assert np.array_equal(archive["X"], matrix)
        assert archive["example_ids"].tolist() == [job["example_id"]]


def test_fresh_jobs_cover_all_pairs_without_source_or_qualification_case_reuse(runner):
    panel = read_json(ROOT / "data/communication/fresh_recipient_test_v1.json")
    jobs = runner.transfer_jobs(panel)
    assert len(jobs) == len({j["example_id"] for j in jobs}) == 144
    assert len({(j["case_id"], j["objective"], j["context"]) for j in jobs}) == 144
    forbidden = set()
    for path in (
        "data/communication/fresh_calibration_v1.json",
        "data/reader_forecast/development_v1.json",
    ):
        forbidden.update(c["id"] for c in read_json(ROOT / path)["cases"])
    assert not {j["case_id"] for j in jobs} & forbidden
    for job in jobs:
        assert job["case_id"].startswith("ct_") and job["split"] == "test"
        assert job["off_policy"] is False
        assert "expected_decision" not in json.dumps(job["messages"])


def test_export_binds_exact_csv_label_rows_to_feature_ids(runner, tmp_path):
    row = {
        "example_id": "explicit_fixture",
        "case_id": "fixture_case",
        "family": "fixture",
        "objective": "neutral",
        "split": "train",
        "decision": "DEFER",
        "prompt_text": "source and packet",
        "report_text": "source excerpt",
        "goal_stripped_prompt_text": "source and packet",
    }
    signature = {"capture": "explicit_fixture"}
    runner.export([row], np.zeros((1, 4096)), signature, tmp_path / "export", {"off_policy": True})
    from latent_probes.recipient_readout import label_row_sha256

    with np.load(tmp_path / "export/features.npz", allow_pickle=False) as archive:
        provenance = json.loads(str(archive["provenance_json"].item()))
        assert provenance["label_row_sha256"] == {row["example_id"]: label_row_sha256(row)}
        assert provenance["reader_policy_sha256"] == policy_metadata()["policy_sha256"]
        assert archive["example_ids"].tolist() == [row["example_id"]]
