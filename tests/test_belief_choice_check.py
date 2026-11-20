"""Pinned-policy, prompt isolation, caching and provenance checks without a GPU."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from latent_probes.belief_panel import build_panel
from latent_probes.storage import atomic_json, canonical_hash, file_sha256, read_json

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "belief_choice_check", ROOT / "scripts/run_belief_choice_check.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BackendStub:
    def __init__(self, output, *, callback=None, token_count=2):
        self.output = output
        self.callback = callback
        self.token_count = token_count
        self.calls = []
        self.metadata = {field: "fixture" for field in MODULE.SIGNATURE_FIELDS}
        self.metadata.update(
            is_test_fixture=True,
            dtype="bfloat16",
            quantization="none",
            requested_revision="fixture-pin",
            resolved_revision="fixture-pin",
            capture_layers=[8, 16, 24],
            device_map={"": "cpu"},
            special_token_ids={"eos": 2, "pad": None},
        )

    def generate(self, messages, settings, seed, capture):
        plan = read_json(self.output / "plan.json")
        assert len(plan["jobs"]) == 400
        assert plan["generation"] == {"max_new_tokens": 6, "do_sample": False, "seed": 42}
        assert settings.max_new_tokens == 6 and settings.do_sample is False
        assert seed == 42 and capture is False
        self.calls.append(copy.deepcopy(messages))
        if self.callback:
            self.callback()
        result = {
            "text": "A",
            "generated_token_ids": [32] * self.token_count,
            "finish_reason": "eos",
        }
        return SimpleNamespace(record=lambda: copy.deepcopy(result))


@pytest.fixture
def inputs(tmp_path):
    source, output = tmp_path / "source", tmp_path / "choice"
    source.mkdir()
    backend = BackendStub(output)
    panel = build_panel()
    atomic_json(source / "protocol.json", {"panel": panel})
    jobs = MODULE.build_jobs(panel)
    for job in jobs:
        atomic_json(
            source / "behavior" / (job["source_example_id"] + ".json"),
            {"example_id": job["source_example_id"], "is_test_fixture": True},
        )
    outputs = MODULE._tree_hashes(source)
    atomic_json(
        source / "manifest.json",
        {
            "schema": "belief_construct_development_run_v1",
            "status": "complete",
            "n_cases_completed": 80,
            "n_behavior_rows": 400,
            "backend": backend.metadata,
            "panel_sha256": canonical_hash(panel),
            "protocol_sha256": file_sha256(source / "protocol.json"),
            "script_sha256": file_sha256(MODULE.SOURCE_SCRIPT),
            "source_sha256": {
                name: file_sha256(ROOT / name)
                for name in (
                    "src/latent_probes/belief_panel.py",
                    "src/latent_probes/belief_readout.py",
                )
            },
            "output_sha256": outputs,
        },
    )
    return source, output, backend


@pytest.mark.parametrize(
    "text, expected",
    (
        ("A", "A"),
        (" b. ", "B"),
        ("A!", "A"),
        ("A because", None),
        ("Option A", None),
        ("A or B", None),
        ("YES", None),
        ("", None),
        ("B\nThe reason", None),
        ("A..", None),
    ),
)
def test_strict_choice_parser(text, expected):
    assert MODULE.parse_choice(text) == expected


def test_scene_candidate_and_information_boundary_stay_exact():
    panel = build_panel()
    jobs = MODULE.build_jobs(panel)
    assert len(jobs) == 400
    assert len({job["prompt_sha256"] for job in jobs}) == 320
    templates = {}
    for case in panel["cases"]:
        order = MODULE.option_order(case["template_id"])
        templates[case["template_id"]] = order
        for view, messages in case["messages"].items():
            targets = ("world", "reader", "nested") if view == "omniscient" else ("world", "nested")
            for target in targets:
                changed = MODULE.choice_messages(case, view, target)
                assert changed[0] == messages[0]
                assert changed[-1]["content"].startswith(messages[-1]["content"] + "\n\nQuestion: ")
                assert changed[-1]["content"].endswith("Answer only A or B.")
                assert f'"The token is in the {case["place_b"]}."' in changed[-1]["content"]
                altered = copy.deepcopy(case)
                altered["labels"] = {key: 1 - value for key, value in case["labels"].items()}
                altered["world_bit"] ^= 1
                altered["reader_observed"] ^= 1
                altered["sender_recorded"] ^= 1
                assert MODULE.choice_messages(altered, view, target) == changed
    assert list(templates.values()).count((0, 1)) == 4
    assert list(templates.values()).count((1, 0)) == 4
    for job in jobs:
        assert job["gold_choice"] == "AB"[job["option_place_bits"].index(job["gold"])]
    with pytest.raises(ValueError):
        MODULE.choice_messages(panel["cases"][0], "sender", "reader")


def test_complete_run_plan_precedes_calls_cache_and_source_remain_intact(inputs):
    source, output, backend = inputs
    before = MODULE._tree_hashes(source)
    manifest = MODULE.run_choice_check(source, output, backend, progress=None)
    assert manifest["status"] == "complete"
    assert manifest["source_unchanged"] is True
    assert manifest["capture_features"] is False
    assert manifest["is_test_fixture"] is True
    assert manifest["n_behavior_rows"] == 400
    assert manifest["n_unique_generations"] == len(backend.calls) == 320
    assert MODULE._tree_hashes(source) == before
    assert len(list((output / "behavior").glob("*.json"))) == 400
    assert not list(output.rglob("*.npz"))
    summary = read_json(output / "summary.json")
    for group in summary["perspectives"].values():
        assert group["invalid"] == 0
        assert group["accuracy_including_invalid"] == 0.5
    rows = [read_json(path) for path in (output / "behavior").glob("*.json")]
    assert sum(row["reused_identical_generation"] for row in rows) == 80
    assert all(
        row["source_behavior_sha256"] == before[row["source_behavior_reference"]] for row in rows
    )
    assert len([row for row in rows if row["phase"] == "delivered_current"]) == 80


@pytest.mark.parametrize("relation", ("equal", "inside", "ancestor"))
def test_output_cannot_overlap_source(inputs, relation):
    source, _output, backend = inputs
    bad = {"equal": source, "inside": source / "new", "ancestor": source.parent}[relation]
    with pytest.raises(ValueError, match="disjoint"):
        MODULE.run_choice_check(source, bad, backend, progress=None)
    assert not backend.calls


@pytest.mark.parametrize("field", ("resolved_revision", "quantization", "chat_template_sha256"))
def test_changed_backend_signature_stops_before_calls(inputs, field):
    source, output, backend = inputs
    backend.metadata[field] = "different"
    with pytest.raises(ValueError, match="signature"):
        MODULE.run_choice_check(source, output, backend, progress=None)
    assert not backend.calls and not output.exists()


def test_source_policy_change_stops_before_another_call(inputs):
    source, output, backend = inputs

    def mutate_protocol():
        with (source / "protocol.json").open("a", encoding="utf-8") as stream:
            stream.write(" ")

    backend.callback = mutate_protocol
    with pytest.raises(ValueError, match="changed during"):
        MODULE.run_choice_check(source, output, backend, progress=None)
    assert len(backend.calls) == 1
    assert read_json(output / "manifest.json")["status"] == "failed"


def test_six_token_cap_is_enforced(inputs):
    source, output, backend = inputs
    backend.token_count = 7
    with pytest.raises(ValueError, match="six-token"):
        MODULE.run_choice_check(source, output, backend, progress=None)
    assert len(backend.calls) == 1
    assert read_json(output / "manifest.json")["status"] == "failed"


def test_source_bytes_must_match_completed_ledger(inputs):
    source, output, backend = inputs
    target = next((source / "behavior").glob("*.json"))
    target.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="source bytes"):
        MODULE.run_choice_check(source, output, backend, progress=None)
    assert not backend.calls and not output.exists()
