"""The paired collector supplies no answers and freezes fits before fresh inputs."""

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from latent_probes.belief_panel import build_panel
from latent_probes.belief_query_panel import build_fresh_panel
from latent_probes.storage import atomic_json, atomic_npz, canonical_hash, read_json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "query_comparison", ROOT / "scripts/run_belief_query_comparison.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FixtureTokenizer:
    def __init__(self):
        self.calls = []
        self.corrupt = False

    def apply_chat_template(self, messages, **_kwargs):
        digest = canonical_hash(messages)
        self.calls.append(digest)
        ids = [int(digest[i : i + 2], 16) for i in range(0, 16, 2)] + [271]
        if self.corrupt:
            ids[0] += 1
        return {"input_ids": np.asarray([ids])}


class FixtureBackend:
    def __init__(self):
        self.metadata = {"backend": "fixture", "model_id": "fixture", "is_test_fixture": True}
        self.settings = SimpleNamespace(capture_layers=[8, 16, 24], max_input_tokens=4096)
        self.tokenizer = FixtureTokenizer()
        self.torch = SimpleNamespace(tensor=lambda values, **_kwargs: np.asarray(values))
        self.input_device = "cpu"
        self.n_captures = 0
        self.fail_after = None

    def _capture_token_rows(self, sequence, positions):
        self.n_captures += 1
        if self.fail_after == self.n_captures:
            raise RuntimeError("fixture capture failure")
        assert positions == [sequence.shape[1] - 1]
        return {
            f"layer_{layer}_tokens": sequence[:, :3].astype(np.float32) + layer
            for layer in MODULE.LAYERS
        }


def fixture_source(path, backend, monkeypatch):
    panel = build_panel()
    rows, captures = [], []
    matrices = {f"layer_{layer}": [] for layer in MODULE.LAYERS}
    for case in panel["cases"]:
        for view in ("omniscient", "sender"):
            identifier = case["id"] + "__" + view
            ids, vectors = MODULE.capture_prefix(backend, case["messages"][view])
            rows.append({"example_id": identifier})
            captures.append({"example_id": identifier, "prompt_token_ids": ids})
            for key, value in vectors.items():
                matrices[key].append(value)
            for target in ("world", "nested"):
                messages = MODULE.query_messages(case, view, target)
                query_ids = backend.tokenizer.apply_chat_template(messages)["input_ids"][0].tolist()
                record = {"messages": messages, "generation": {"prompt_token_ids": query_ids}}
                record["record_sha256"] = canonical_hash(record)
                atomic_json(path / "behavior" / (identifier + "__" + target + ".json"), record)
    atomic_json(path / "captures.json", captures)
    atomic_json(path / "rows.json", rows)
    atomic_npz(
        path / "features.npz", **{k: np.asarray(v, dtype=np.float32) for k, v in matrices.items()}
    )
    original = {"backend": copy.deepcopy(backend.metadata)}
    hashes = MODULE._snapshot(path)
    monkeypatch.setattr(
        MODULE, "_source", lambda _path: (original, {"panel": panel}, rows, {}, None, hashes)
    )
    backend.tokenizer.calls.clear()
    backend.n_captures = 0
    return panel, hashes


def test_conditions_keep_story_and_gold_out_of_model_inputs():
    case = build_panel()["cases"][0]
    untouched = copy.deepcopy(case)
    for view in ("omniscient", "sender"):
        story = MODULE.condition_messages(case, view, "story")
        assert story == case["messages"][view]
        for condition, target in (("nested_question", "nested"), ("world_question", "world")):
            messages = MODULE.condition_messages(case, view, condition)
            assert messages[0] == story[0]
            assert (
                messages[-1]["content"]
                == story[-1]["content"] + "\n\nQuestion: " + case["questions"][target]
            )
            changed = copy.deepcopy(case)
            changed["labels"] = {k: 1 - v for k, v in case["labels"].items()}
            assert MODULE.condition_messages(changed, view, condition) == messages
    assert case == untouched


def test_collection_freezes_fit_before_any_fresh_capture(tmp_path, monkeypatch):
    backend = FixtureBackend()
    source = tmp_path / "source"
    _legacy, hashes = fixture_source(source, backend, monkeypatch)
    fresh = build_fresh_panel()
    fresh_hashes = {
        canonical_hash(MODULE.condition_messages(c, v, q))
        for c in fresh["cases"]
        for v in ("omniscient", "sender")
        for q in MODULE.CONDITIONS
    }
    events = []

    def fit(rows, features, output):
        assert len(rows) == 480 and all(len(v) == 480 for v in features.values())
        assert not fresh_hashes.intersection(backend.tokenizer.calls)
        atomic_json(output / "weights.json", {"fixture": True})
        events.append("fit")

    def evaluate(rows, features, fitted, output):
        assert events == ["fit"] and len(rows) == 192
        assert all(row["split"] == "fresh" for row in rows)
        assert all(len(v) == 192 for v in features.values())
        assert read_json(fitted / "weights.json") == {"fixture": True}
        events.append("fresh_evaluation")
        atomic_json(output / "report.json", {"fixture": True})

    monkeypatch.setattr(MODULE, "fit_query_comparison", fit)
    monkeypatch.setattr(MODULE, "evaluate_fresh_query_comparison", evaluate)
    output = tmp_path / "new_run"
    manifest = MODULE.run_query_comparison(source, fresh, output, backend, progress=None)
    assert manifest["status"] == "complete" and manifest["is_test_fixture"]
    assert manifest["generated_answers"] == 0 and events == ["fit", "fresh_evaluation"]
    assert manifest["legacy"]["n_unique_new_prefixes"] == 360
    assert manifest["fresh"]["n_unique_new_prefixes"] == 144
    assert manifest["n_unique_forward_prefixes"] == 504
    assert MODULE._snapshot(source) == hashes
    assert read_json(output / "frozen_fits.json")["artifacts"] == MODULE._snapshot(output / "fits")
    captures = read_json(output / "legacy/captures.json") + read_json(
        output / "fresh/captures.json"
    )
    assert len(captures) == 672
    assert all(c["generated_answer_tokens_supplied"] == 0 for c in captures)
    assert len({c["selected_token_id"] for c in captures}) == 1


def test_prefix_mismatch_is_not_silently_accepted(tmp_path, monkeypatch):
    backend = FixtureBackend()
    source = tmp_path / "source"
    _, hashes = fixture_source(source, backend, monkeypatch)
    backend.tokenizer.corrupt = True
    output = tmp_path / "bad_prefix"
    with pytest.raises(ValueError, match="saved token IDs"):
        MODULE.run_query_comparison(source, build_fresh_panel(), output, backend, progress=None)
    assert MODULE._snapshot(source) == hashes
    assert read_json(output / "manifest.json")["status"] == "failed"


def test_failure_preserves_partial_prefixes(tmp_path, monkeypatch):
    backend = FixtureBackend()
    source = tmp_path / "source"
    _, hashes = fixture_source(source, backend, monkeypatch)
    backend.fail_after = 3
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="fixture capture failure"):
        MODULE.run_query_comparison(source, build_fresh_panel(), output, backend, progress=None)
    assert len(list((output / "prefix_features").glob("*.npz"))) == 1
    assert len(read_json(output / "legacy/captures.json")) == 1
    assert read_json(output / "manifest.json")["status"] == "failed"
    assert MODULE._snapshot(source) == hashes


def test_rejects_contained_output_before_source_reads(tmp_path):
    source = tmp_path / "frozen"
    source.mkdir()
    with pytest.raises(ValueError, match="disjoint"):
        MODULE.run_query_comparison(source, build_fresh_panel(), source / "new", FixtureBackend())
    assert not list(source.iterdir())
