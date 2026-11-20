"""Fixtures verify capture lineage and freeze ordering, not model competence."""

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from latent_probes.belief_contrast_panel import build_contrast_panel
from latent_probes.belief_panel import build_panel
from latent_probes.storage import atomic_json, canonical_hash, read_json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "contrast_runner", ROOT / "scripts/run_belief_contrast.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FixtureTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        digest = canonical_hash(messages)
        self.calls.append(digest)
        return {
            "input_ids": np.asarray([[int(digest[i : i + 2], 16) for i in range(0, 16, 2)] + [271]])
        }


class FixtureBackend:
    def __init__(self):
        self.metadata = {"backend": "fixture", "is_test_fixture": True}
        self.settings = SimpleNamespace(capture_layers=[8, 16, 24], max_input_tokens=4096)
        self.tokenizer = FixtureTokenizer()
        self.torch = SimpleNamespace(tensor=lambda x, **kw: np.asarray(x))
        self.input_device = "cpu"
        self.n_calls = 0
        self.fail_after = None

    def _capture_token_rows(self, sequence, positions):
        self.n_calls += 1
        if self.n_calls == self.fail_after:
            raise RuntimeError("fixture forward failure")
        assert positions == [sequence.shape[1] - 1]
        return {
            f"layer_{layer}_tokens": sequence[:, :3].astype(np.float32) + layer
            for layer in MODULE.LAYERS
        }


def source_fixture(tmp_path, monkeypatch, backend):
    source = tmp_path / "source"
    atomic_json(source / "preserved.json", {"never": "edit"})
    cases = [MODULE.legacy_candidates(c) for c in build_panel()["cases"]]
    expected = {}
    for case in cases:
        candidate = case["candidates"][1]
        for view in MODULE.VIEWS:
            for condition in MODULE.CONDITIONS:
                messages = MODULE.condition_messages(candidate, view, condition)
                ids, features = MODULE.capture_prefix(backend, messages)
                expected[(case["id"], view, condition)] = {"ids": ids, "features": features}
    monkeypatch.setattr(
        MODULE,
        "load_source",
        lambda p: ({"backend": copy.deepcopy(backend.metadata)}, cases, expected),
    )
    fresh = tmp_path / "fresh.json"
    atomic_json(fresh, build_contrast_panel())
    backend.tokenizer.calls.clear()
    backend.n_calls = 0
    return source, fresh


def test_candidate_changes_only_proposition_and_preserves_gold():
    original = build_panel()["cases"][0]
    untouched = copy.deepcopy(original)
    converted = MODULE.legacy_candidates(original)
    a, b = converted["candidates"]
    for target in ("world", "reader", "nested"):
        assert a["labels"][target] == 1 - b["labels"][target]
    for view in MODULE.VIEWS:
        assert b["messages"][view] == original["messages"][view]
        old = b["messages"][view][-1]["content"]
        expected = old.replace('"' + b["claim"] + '"', '"' + a["claim"] + '"', 1)
        assert a["messages"][view][-1]["content"] == expected
    assert original == untouched


def test_fit_freezes_before_any_fresh_forward(tmp_path, monkeypatch):
    backend = FixtureBackend()
    source, fresh = source_fixture(tmp_path, monkeypatch, backend)
    preserved = MODULE.snapshot(source)
    new_panel = read_json(fresh)
    hashes = {
        canonical_hash(MODULE.condition_messages(p, v, q))
        for c in new_panel["cases"]
        for p in c["candidates"]
        for v in MODULE.VIEWS
        for q in MODULE.CONDITIONS
    }
    events = []

    def fit(rows, features, outdir, original):
        assert len(rows) == 640
        assert all(len(x) == 640 for x in features.values())
        assert not hashes.intersection(backend.tokenizer.calls)
        assert all(r["split"] != "fresh" for r in rows)
        atomic_json(outdir / "weights.json", {"fixture": True})
        events.append("fit")

    def evaluate(rows, features, fitdir, outdir):
        assert events == ["fit"]
        assert len(rows) == 1024 and all(len(x) == 1024 for x in features.values())
        assert all(r["split"] == "fresh" for r in rows)
        assert read_json(fitdir / "weights.json") == {"fixture": True}
        events.append("evaluate")
        atomic_json(outdir / "report.json", {"fixture": True})

    monkeypatch.setattr(MODULE, "fit_contrast_readouts", fit)
    monkeypatch.setattr(MODULE, "evaluate_contrast_readouts", evaluate)
    result = MODULE.run_comparison(
        source, fresh, tmp_path / "new", backend, progress=lambda x: None
    )
    assert result["status"] == "complete" and result["is_test_fixture"]
    assert result["source_unchanged"] and result["frozen_fits_unchanged"]
    assert result["fresh_panel_unchanged"] and result["source_code_unchanged"]
    assert result["generated_answers"] == 0 and events == ["fit", "evaluate"]
    assert result["legacy"]["legacy_replay_rows"] == 320
    assert result["legacy"]["legacy_replay_exact"]
    assert result["fresh"]["legacy_replay_rows"] == 0
    assert result["fresh"]["legacy_replay_exact"] is None
    assert MODULE.snapshot(source) == preserved


def test_output_inside_source_rejected_before_loading(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="disjoint"):
        MODULE.run_comparison(source, tmp_path / "unused", source / "new", FixtureBackend())
    assert not list(source.iterdir())


def test_failed_forward_retains_originals_and_partial_artifacts(tmp_path, monkeypatch):
    backend = FixtureBackend()
    source, fresh = source_fixture(tmp_path, monkeypatch, backend)
    preserved = MODULE.snapshot(source)
    backend.fail_after = 4
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="fixture forward failure"):
        MODULE.run_comparison(source, fresh, output, backend, progress=lambda x: None)
    assert read_json(output / "manifest.json")["status"] == "failed"
    assert list((output / "prefix_features").glob("*.npz"))
    assert MODULE.snapshot(source) == preserved


def test_original_claim_mismatch_stops_before_fit(tmp_path, monkeypatch):
    backend = FixtureBackend()
    source, fresh = source_fixture(tmp_path, monkeypatch, backend)
    original = MODULE.load_source
    manifest, cases, expected = original(source)
    first = next(iter(expected))
    expected[first]["features"]["layer_16"] += 0.25
    monkeypatch.setattr(MODULE, "load_source", lambda p: (manifest, cases, expected))
    monkeypatch.setattr(MODULE, "fit_contrast_readouts", lambda *a: pytest.fail("must not fit"))
    output = tmp_path / "mismatch"
    with pytest.raises(ValueError, match="claim-B replay differs"):
        MODULE.run_comparison(source, fresh, output, backend, progress=lambda x: None)
    assert read_json(output / "manifest.json")["status"] == "failed"
