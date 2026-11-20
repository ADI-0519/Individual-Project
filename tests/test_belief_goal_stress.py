"""Future reporting goals leave current labels and saved readout settings fixed."""

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from latent_probes.belief_query_readout import TEXT_SETTINGS
from latent_probes.message_belief_panel import build_panel
from latent_probes.storage import atomic_json, atomic_npz, canonical_hash, read_json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "belief_goal_stress_test", ROOT / "scripts/run_belief_goal_stress.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FixtureBackend:
    def __init__(self):
        self.metadata = {"backend": "fixture", "is_test_fixture": True, "model_id": "fixture"}
        self.settings = SimpleNamespace(capture_layers=[8, 16, 24], max_input_tokens=4096)
        self.tokenizer = SimpleNamespace(apply_chat_template=self.encode)
        self.torch = SimpleNamespace(tensor=lambda values, **_kwargs: np.asarray(values))
        self.input_device = "cpu"
        self.n_captures = 0
        self.fail_at = None
        self.drift = 0

    def encode(self, messages, **_kwargs):
        digest = canonical_hash(messages)
        return {"input_ids": np.asarray([[int(digest[:2], 16), int(digest[2:4], 16), 271]])}

    def _capture_token_rows(self, sequence, positions):
        self.n_captures += 1
        if self.n_captures == self.fail_at:
            raise RuntimeError("Fixture goal capture failure")
        assert positions == [sequence.shape[1] - 1]
        return {
            f"layer_{layer}_tokens": sequence[:, :2].astype(np.float32) + layer + self.drift
            for layer in runner.LAYERS
        }

    def generate(self, *_args, **_kwargs):
        raise AssertionError("Goal-context robustness must not generate an answer or report")


def fixture_models():
    params, heads = {"classes": np.asarray([0, 1])}, {}
    texts = {"source_text": {"terms": ["token"], "idf": [1.0], "settings": TEXT_SETTINGS}}
    for head_id in (runner.HEAD, runner.TEXT_HEAD, *runner.READER_HEADS):
        layer = "text" if head_id.endswith("__text") else "layer_16"
        target = "reader" if "__reader__" in head_id else "nested"
        heads[head_id] = {
            "target": target,
            "layer": layer,
            "parameter_prefix": head_id,
            "selected_C": 1.0,
        }
        params[head_id + "__coef"] = np.zeros((1, 1 if layer == "text" else 2))
        params[head_id + "__intercept"] = np.asarray([0.1])
        if layer == "text":
            heads[head_id]["text_transform"] = "source_text"
        else:
            heads[head_id]["activation_transform"] = "source_scale"
    params["source_scale__mean"] = np.zeros(2)
    params["source_scale__scale"] = np.ones(2)
    return (
        {"heads": heads, "feature_widths": {f"layer_{layer}": 2 for layer in runner.LAYERS}},
        params,
        texts,
    )


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    panel, backend = build_panel(), FixtureBackend()
    source, fitted, output = tmp_path / "source", tmp_path / "fit", tmp_path / "goal_run"
    atomic_json(fitted / "weights.json", {"immutable": True})
    models = fixture_models()
    monkeypatch.setattr(runner, "_load_fit", lambda path: (*models, runner._snapshot(path)))
    rows, captures = [], []
    features = {f"layer_{layer}": [] for layer in runner.LAYERS}
    for case in panel["cases"]:
        messages = runner.query_messages(case, "sender", "nested")
        ids, vectors = runner.capture_prefix(backend, messages)
        rows.append(
            {
                "case_id": case["id"],
                "example_id": case["id"] + "__sender",
                "template_id": case["template_id"],
                "phase": "known_message",
                "view": "sender",
                "condition": "nested_question",
                "labels": case["labels"],
                "messages_text": "\n".join(m["content"] for m in messages),
            }
        )
        captures.append(
            {
                "example_id": case["id"] + "__sender",
                "condition": "nested_question",
                "messages_sha256": canonical_hash(messages),
                "generated_answer_tokens_supplied": 0,
                "prompt_token_ids": ids,
            }
        )
        for name, vector in vectors.items():
            features[name].append(vector)
    atomic_json(source / "rows.json", rows)
    atomic_json(source / "captures.json", captures)
    atomic_npz(
        source / "features.npz",
        **{name: np.asarray(values, dtype=np.float32) for name, values in features.items()},
    )
    atomic_json(
        source / "protocol.json",
        {
            "panel": panel,
            "fit_dir": str(fitted.resolve()),
            "fitted_artifact_sha256": runner._snapshot(fitted),
        },
    )
    atomic_json(
        source / "manifest.json",
        {
            "status": "complete",
            "backend": backend.metadata,
            "output_sha256": runner._snapshot(source),
        },
    )
    backend.n_captures = 0
    protocol = runner.prepare_run(source, fitted, output)
    return SimpleNamespace(
        source=source,
        fitted=fitted,
        output=output,
        backend=backend,
        panel=panel,
        protocol=protocol,
        models=models,
    )


def test_same_scene_question_and_role_with_no_gold_leak():
    case = build_panel()["cases"][0]
    original = copy.deepcopy(case)
    rendered = [runner.goal_messages(case, condition) for condition in runner.CONDITIONS]
    unchanged = runner.query_messages(case, "sender", "nested")
    assert all(messages[1] == unchanged[1] for messages in rendered)
    assert all(
        messages[0]["content"].startswith(unchanged[0]["content"] + "\n\n") for messages in rendered
    )
    assert (
        len({messages[0]["content"].split("For that later report,")[0] for messages in rendered})
        == 1
    )
    assert "has not been written or delivered" in rendered[0][0]["content"]
    assert all(messages[0]["content"].count('"The token is in the ') == 1 for messages in rendered)
    assert "believe the candidate claim." in rendered[1][0]["content"]
    assert "believe the negation of the candidate claim." in rendered[2][0]["content"]
    changed = copy.deepcopy(case)
    changed["labels"] = {key: 1 - value for key, value in case["labels"].items()}
    assert [runner.goal_messages(changed, condition) for condition in runner.CONDITIONS] == rendered
    assert case == original


def test_planned_current_nested_labels_and_hidden_pairs_unchanged(prepared):
    planned = prepared.protocol["planned_rows"]
    assert len(planned) == 192 and len({row["messages_sha256"] for row in planned}) == 96
    cases = {case["id"]: case for case in prepared.panel["cases"]}
    assert all(row["labels"] == cases[row["case_id"]]["labels"] for row in planned)
    assert all(row["phase"] == "known_message" for row in planned)
    features = {f"layer_{layer}": np.zeros((192, 2)) for layer in runner.LAYERS}
    assert len(runner._hidden_pairs(planned, features)) == 96
    altered = copy.deepcopy(planned)
    altered[0]["messages_text"] += "Actual delivery disclosed"
    with pytest.raises(ValueError, match="Actual reader state"):
        runner._hidden_pairs(altered, features)


def test_frozen_no_fit_no_generation_and_exact_reference(prepared, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("No fresh fitting allowed")

    for cls in (LogisticRegression, StandardScaler, TfidfVectorizer):
        monkeypatch.setattr(cls, "fit", forbidden)
    source_hashes, fit_hashes = runner._snapshot(prepared.source), runner._snapshot(prepared.fitted)
    manifest = runner.capture_run(prepared.output, prepared.backend, progress=None)
    assert manifest["status"] == "complete" and manifest["generated_answers"] == 0
    assert manifest["n_capture_rows"] == 192 and manifest["n_unique_capture_prefixes"] == 96
    assert manifest["n_reference_checks"] == 4 and manifest["n_repeat_checks"] == 3
    assert manifest["reference_exact"] and prepared.backend.n_captures == 103
    assert runner._snapshot(prepared.source) == source_hashes
    assert runner._snapshot(prepared.fitted) == fit_hashes
    report = read_json(prepared.output / "report.json")
    assert report["n_hidden_reader_pairs_including_baseline"] == 128
    for condition in (runner.BASELINE, *runner.CONDITIONS):
        assert report["metrics"][runner.HEAD][condition]["n_unique_sender_inputs"] == 32
        for head in runner.READER_HEADS:
            assert report["metrics"][head][condition]["metrics"]["accuracy"] == 0.5
    for head in (runner.HEAD, runner.TEXT_HEAD):
        contrasts = report["paired_contrasts"][head]
        assert contrasts["goal_true_minus_false"]["summary"]["n_unique_sender_inputs"] == 32
        assert contrasts["goal_true_minus_false"]["summary"]["mean_p1_difference"] == 0.0
    assert all(
        row["generated_answer_tokens_supplied"] == 0
        for row in read_json(prepared.output / "captures.json")
    )
    with pytest.raises(ValueError, match="no automatic retry"):
        runner.capture_run(prepared.output, prepared.backend, progress=None)


def test_reference_drift_is_inconclusive_without_overwriting_source(prepared):
    original = runner._snapshot(prepared.source)
    prepared.backend.drift = 1
    result = runner.capture_run(prepared.output, prepared.backend, progress=None)
    assert result["status"] == "complete" and not result["reference_exact"]
    assert result["original_to_role_comparison_status"] == "numerically_inconclusive"
    assert runner._snapshot(prepared.source) == original


def test_changed_fit_or_source_blocks_any_new_forward(prepared):
    atomic_json(prepared.fitted / "weights.json", {"changed": True})
    with pytest.raises(ValueError, match="fitted heads changed"):
        runner.capture_run(prepared.output, prepared.backend, progress=None)
    assert prepared.backend.n_captures == 0


def test_no_overwrite_or_contained_output_before_forward(prepared, tmp_path):
    for output in (prepared.source, prepared.source / "child", prepared.fitted / "child", tmp_path):
        with pytest.raises(ValueError, match="disjoint"):
            runner.prepare_run(prepared.source, prepared.fitted, output)
    with pytest.raises(FileExistsError):
        runner.prepare_run(prepared.source, prepared.fitted, prepared.output)
    assert prepared.backend.n_captures == 0


def test_failure_preserves_reference_capture(prepared):
    prepared.backend.fail_at = 2
    original = runner._snapshot(prepared.source)
    with pytest.raises(RuntimeError, match="Fixture goal capture"):
        runner.capture_run(prepared.output, prepared.backend, progress=None)
    assert len(list((prepared.output / "reference_features").glob("*.npz"))) == 1
    assert read_json(prepared.output / "manifest.json")["status"] == "failed"
    assert runner._snapshot(prepared.source) == original
