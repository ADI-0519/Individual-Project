"""Known-message phase, unchanged saved transforms and answer-free capture order."""

import copy
import importlib.util
import sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from latent_probes import belief_query_readout
from latent_probes.backends import GenerationResult
from latent_probes.message_belief_panel import build_panel
from latent_probes.storage import atomic_json, canonical_hash, read_json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "known_message_runner_test", ROOT / "scripts/run_message_belief_validation.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FixtureBackend:
    def __init__(self, output):
        self.output = output
        self.metadata = {"backend": "fixture", "is_test_fixture": True, "model_id": "fixture"}
        self.settings = SimpleNamespace(capture_layers=[8, 16, 24], max_input_tokens=4096)
        self.tokenizer = SimpleNamespace(apply_chat_template=self.encode)
        self.torch = SimpleNamespace(tensor=lambda values, **_kwargs: np.asarray(values))
        self.input_device = "cpu"
        self.n_captures = 0
        self.n_generations = 0
        self.fail_capture_at = None
        self.capped = False

    def encode(self, messages, **_kwargs):
        value = canonical_hash(messages)
        return {"input_ids": np.asarray([[int(value[:2], 16), int(value[2:4], 16), 271]])}

    def _capture_token_rows(self, sequence, positions):
        assert self.n_generations == 0, "An answer contaminated a later feature capture"
        assert positions == [sequence.shape[1] - 1]
        assert read_json(self.output / "manifest.json")["status"] == "running"
        self.n_captures += 1
        if self.n_captures == self.fail_capture_at:
            raise RuntimeError("Fixture capture failure")
        return {
            f"layer_{layer}_tokens": sequence[:, :2].astype(np.float32) + layer
            for layer in runner.LAYERS
        }

    def generate(self, messages, settings, seed, capture):
        assert capture is False and seed == runner.SEED
        assert (self.output / "frozen_evaluation/report.json").is_file()
        assert len(read_json(self.output / "captures.json")) == 384
        assert read_json(self.output / "manifest.json")["prefix_capture_complete_at"]
        self.n_generations += 1
        structured = "INITIAL (" in messages[-1]["content"]
        assert settings.max_new_tokens == (160 if structured else 8)
        text = (
            "INITIAL: first place\nMESSAGE: second place\nRECEIPT: read\nATTRIBUTED: second place\nFINAL: YES"
            if structured
            else "NO"
        )
        return GenerationResult(
            text=text,
            prompt_token_ids=self.encode(messages)["input_ids"][0].tolist(),
            generated_token_ids=[9999],
            finish_reason="length" if self.capped else "eos",
        )


def fixture_models():
    parameters, heads, texts = {"classes": np.asarray([0, 1])}, {}, {}
    layers = ("layer_8", "layer_16", "layer_24", "text")
    for condition, view in product(runner.CONDITIONS, ("omniscient", "sender")):
        text_key = condition + "__" + view
        texts[text_key] = {
            "terms": ["token"],
            "idf": [1.0],
            "settings": belief_query_readout.TEXT_SETTINGS,
        }
        for layer in layers[:-1]:
            transform = "transform__" + condition + "__" + view + "__" + layer
            parameters[transform + "__mean"] = np.zeros(2)
            parameters[transform + "__scale"] = np.ones(2)
        for target, layer in product(("world", "reader", "nested"), layers):
            head_id = "__".join((condition, view, target, layer))
            parameters[head_id + "__coef"] = np.zeros((1, 1 if layer == "text" else 2))
            parameters[head_id + "__intercept"] = np.asarray([0.1])
            heads[head_id] = {
                "condition": condition,
                "view": view,
                "target": target,
                "layer": layer,
                "parameter_prefix": head_id,
                "selected_C": 0.1,
            }
            if layer == "text":
                heads[head_id]["text_transform"] = text_key
            else:
                heads[head_id]["activation_transform"] = (
                    "transform__" + condition + "__" + view + "__" + layer
                )
    metadata = {
        "heads": heads,
        "feature_widths": {layer: 2 for layer in layers[:-1]},
        "known_case_ids": ["old"],
        "known_template_ids": ["old"],
        "known_entity_ids": ["person:old"],
        "known_message_hashes": ["old"],
    }
    return metadata, parameters, texts


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    panel_path = tmp_path / "panel.json"
    atomic_json(panel_path, build_panel())
    source = tmp_path / "source"
    fitted = source / "fits"
    atomic_json(fitted / "fixture_weights.json", {"fixture": True})
    atomic_json(
        source / "protocol.json", {"legacy_panel": {"cases": []}, "fresh_panel": {"cases": []}}
    )
    output = tmp_path / "new_run"
    backend = FixtureBackend(output)
    atomic_json(
        source / "manifest.json",
        {
            "status": "complete",
            "backend": backend.metadata,
            "output_sha256": runner._snapshot(source),
        },
    )
    models = fixture_models()
    monkeypatch.setattr(runner, "_load_fit", lambda path: (*models, runner._snapshot(path)))
    protocol = runner.prepare_run(panel_path, fitted, source, output)
    return SimpleNamespace(
        output=output,
        backend=backend,
        protocol=protocol,
        panel_path=panel_path,
        source=source,
        fitted=fitted,
    )


def no_fit(*_args, **_kwargs):
    raise AssertionError("Fresh validation must never fit or choose hyperparameters")


def test_structured_prompt_keeps_evidence_and_gold_out():
    case = build_panel()["cases"][0]
    before = copy.deepcopy(case)
    messages = runner.structured_messages(case)
    assert messages[0] == case["messages"]["sender"][0]
    assert messages[-1]["content"].startswith(
        case["messages"]["sender"][-1]["content"] + "\n\nQuestion: "
    )
    changed = copy.deepcopy(case)
    changed["labels"] = {key: 1 - value for key, value in case["labels"].items()}
    assert runner.structured_messages(changed) == messages
    assert case == before


@pytest.mark.parametrize(
    "value,expected",
    [
        ("INITIAL: a\nMESSAGE: b\nRECEIPT: read\nATTRIBUTED: b\nFINAL: YES", 1),
        ("INITIAL: a\nMESSAGE: b\nRECEIPT: unread\nATTRIBUTED: a\nFINAL: NO.", 0),
        ("FINAL: YES", None),
        ("INITIAL: a\nMESSAGE: b\nRECEIPT: read\nATTRIBUTED: b\nFINAL: YES\nExtra", None),
        ("INITIAL: a\nMESSAGE: b\nRECEIPT: read\nATTRIBUTED: b\nFINAL: YES\nFINAL: NO", None),
        ("INITIAL: a\nMESSAGE: b\nRECEIPT: read\nATTRIBUTED: b\nFINAL: MAYBE", None),
    ],
)
def test_strict_structured_parser(value, expected):
    assert runner.parse_structured(value) == expected


def test_secondary_content_extraction_preserves_primary_format_failure():
    direct = "YES, because the receipt says the note was read."
    assert runner.parse_yes_no(direct) is None
    assert runner.parse_leading_yes_no(direct) == 1
    assert runner.parse_leading_yes_no("Yesterday, no receipt arrived") is None
    structured = "The note was read, so the sender attributes its location.\nFINAL: NO"
    assert runner.parse_structured(structured) is None
    assert runner.parse_final_marker(structured) == 0
    assert runner.parse_final_marker("FINAL: YES\nFINAL: NO") is None
    assert runner.parse_final_marker("FINAL: NO\nExtra commentary") is None
    report = runner.summarize_behavior(
        [
            {
                "style": "direct",
                "template_id": "fixture",
                "prompt_sha256": "fixture",
                "prediction": None,
                "secondary_prediction": 1,
                "gold": 1,
                "length_capped": False,
            }
        ]
    )
    assert report["direct"]["invalid"] == 1 and report["direct"]["correct"] == 0
    assert report["direct"]["secondary_content_invalid"] == 0
    assert report["direct"]["secondary_content_correct"] == 1


def test_all_captures_precede_behavior_and_fits_remain_frozen(prepared, monkeypatch):
    for cls in (LogisticRegression, StandardScaler, TfidfVectorizer):
        monkeypatch.setattr(cls, "fit", no_fit)
    original = runner._snapshot(prepared.source)
    manifest = runner.capture_run(prepared.output, prepared.backend, progress=None)
    assert manifest["status"] == "complete"
    assert manifest["n_capture_rows"] == 384 and manifest["n_behavior_rows"] == 128
    assert manifest["n_unique_capture_prefixes"] == 288
    assert manifest["n_repeat_checks"] == 6
    assert prepared.backend.n_captures == 294
    assert prepared.backend.n_generations == manifest["n_unique_behavior_generations"] == 64
    assert runner._snapshot(prepared.source) == original
    captures = read_json(prepared.output / "captures.json")
    assert all(row["generated_answer_tokens_supplied"] == 0 for row in captures)
    rows = read_json(prepared.output / "rows.json")
    assert all(row["phase"] == "known_message" and row["split"] == "fresh" for row in rows)
    scores = read_json(prepared.output / "frozen_evaluation/report.json")
    assert scores["no_refit"] and scores["hidden_reader_control_metric_cells"] == 28
    assert scores["hidden_reader_pairs"] == 96
    behavior = read_json(prepared.output / "behavior_report.json")
    assert (
        behavior["direct"]["n_unique_prompts"] == behavior["structured"]["n_unique_prompts"] == 32
    )
    assert behavior["direct"]["correct"] == behavior["structured"]["correct"] == 32
    assert behavior["direct"]["invalid"] == behavior["structured"]["invalid"] == 0
    with (prepared.output / "frozen_evaluation/predictions.csv").open() as handle:
        assert ",known_message," in handle.read()
    with pytest.raises(ValueError, match="automatic retry"):
        runner.capture_run(prepared.output, prepared.backend, progress=None)


def test_capped_outputs_are_retained_as_invalid_not_retried(prepared):
    prepared.backend.capped = True
    runner.capture_run(prepared.output, prepared.backend, progress=None)
    report = read_json(prepared.output / "behavior_report.json")
    assert report["direct"]["invalid"] == report["structured"]["invalid"] == 64
    assert report["direct"]["length_capped"] == report["structured"]["length_capped"] == 64
    record = read_json(next((prepared.output / "behavior").glob("*__direct.json")))
    assert record["parsed_prediction"] == 0 and record["prediction"] is None
    assert record["secondary_parsed_prediction"] == 0 and record["secondary_prediction"] is None
    assert prepared.backend.n_generations == 64


def test_changed_source_blocks_model_calls(prepared):
    atomic_json(prepared.panel_path, {})
    with pytest.raises(ValueError, match="panel source changed"):
        runner.capture_run(prepared.output, prepared.backend, progress=None)
    assert prepared.backend.n_captures == prepared.backend.n_generations == 0


def test_failure_preserves_partial_feature_artifact(prepared):
    prepared.backend.fail_capture_at = 2
    with pytest.raises(RuntimeError, match="Fixture capture"):
        runner.capture_run(prepared.output, prepared.backend, progress=None)
    assert len(list((prepared.output / "prefix_features").glob("*.npz"))) == 1
    assert read_json(prepared.output / "manifest.json")["status"] == "failed"
    assert prepared.backend.n_generations == 0


def test_disjoint_output_rejected_before_model_calls(prepared, tmp_path):
    for output in (prepared.source, prepared.fitted / "child", tmp_path):
        with pytest.raises(ValueError, match="disjoint"):
            runner.prepare_run(prepared.panel_path, prepared.fitted, prepared.source, output)
    with pytest.raises(FileExistsError):
        runner.prepare_run(prepared.panel_path, prepared.fitted, prepared.source, prepared.output)
    assert prepared.backend.n_captures == prepared.backend.n_generations == 0


def test_actual_delivery_cannot_change_sender_features(prepared):
    panel = read_json(prepared.panel_path)
    cases = panel["cases"]
    pair = next(
        (a, b)
        for a in cases
        for b in cases
        if a["id"] != b["id"] and a["messages"]["sender"] == b["messages"]["sender"]
    )
    rows = [
        {
            "phase": "known_message",
            "view": "sender",
            "condition": "story",
            "messages_text": "same",
            "labels": case["labels"],
        }
        for case in pair
    ]
    features = {f"layer_{layer}": np.ones((2, 2), dtype=np.float32) for layer in runner.LAYERS}
    metadata = fixture_models()[0]
    arrays, pairs = runner._validate_features(rows, features, metadata)
    assert pairs == 1 and np.array_equal(arrays["layer_8"][0], arrays["layer_8"][1])
    features["layer_8"][1, 0] = 2
    with pytest.raises(ValueError, match="Identical sender inputs"):
        runner._validate_features(rows, features, metadata)
