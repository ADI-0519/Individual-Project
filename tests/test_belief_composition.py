"""Nonlinear toy facts distinguish external composition from a direct head."""

from __future__ import annotations

import importlib.util
import inspect
import shutil
from pathlib import Path

import numpy as np
import pytest

from latent_probes import belief_readout
from latent_probes.belief_panel import build_panel
from latent_probes.storage import atomic_json, atomic_npz, canonical_hash, file_sha256, read_json

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_belief_composition.py"
SPEC = importlib.util.spec_from_file_location("belief_composition_diagnostic", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture(scope="module")
def saved_run(tmp_path_factory):
    source = tmp_path_factory.mktemp("composition_source")
    panel = build_panel()
    rows, captures, arrays = [], [], []
    for case in panel["cases"]:
        for view in ("omniscient", "sender"):
            identifier = case["id"] + "__" + view
            # Synthetic linear coordinates encode the two ingredients, not XNOR.
            vector = np.asarray(
                [
                    case["world_bit"],
                    case["sender_recorded"],
                    case["labels"]["reader"] if view == "omniscient" else 0,
                    5.0,
                ],
                dtype=np.float32,
            )
            vectors = {layer: vector * (i + 1) for i, layer in enumerate(belief_readout.LAYERS)}
            arrays.append(vectors)
            rows.append(
                {
                    "example_id": identifier,
                    "template_id": case["template_id"],
                    "split": case["split"],
                    "phase": case["phase"],
                    "view": view,
                    "messages_text": "\n".join(m["content"] for m in case["messages"][view]),
                    "labels": case["labels"],
                }
            )
            feature = "prefix_features/" + identifier + ".npz"
            atomic_npz(source / feature, **vectors)
            captures.append(
                {
                    "example_id": identifier,
                    "messages_sha256": canonical_hash(case["messages"][view]),
                    "feature_file": feature,
                    "feature_sha256": file_sha256(source / feature),
                    "prompt_token_ids": [1, 2],
                    "selected_position": 1,
                    "generated_answer_tokens_supplied": 0,
                    "perspective_question_supplied": False,
                }
            )
    features = {
        layer: np.stack([value[layer] for value in arrays]) for layer in belief_readout.LAYERS
    }
    report = belief_readout.evaluate_belief_readouts(rows, features)
    atomic_json(source / "protocol.json", {"primary_layer": "layer_16", "panel": panel})
    atomic_json(source / "rows.json", rows)
    atomic_json(source / "captures.json", captures)
    atomic_json(source / "readouts.json", report)
    atomic_npz(source / "features.npz", **features)
    hashes = MODULE.HELPER._snapshot(source)
    atomic_json(
        source / "manifest.json",
        {
            "schema": "belief_construct_development_run_v1",
            "status": "complete",
            "is_test_fixture": True,
            "backend": {
                "backend": "fixture",
                "model_id": "fixture",
                "resolved_revision": "fixture-pin",
                "is_test_fixture": True,
            },
            "protocol_sha256": hashes["protocol.json"],
            "panel_sha256": canonical_hash(panel),
            "source_sha256": {
                "src/latent_probes/belief_readout.py": file_sha256(inspect.getfile(belief_readout))
            },
            "output_sha256": hashes,
        },
    )
    return source


def test_ingredients_can_be_readable_when_direct_linear_composition_is_not(saved_run, tmp_path):
    before = MODULE.HELPER._snapshot(saved_run)
    report = MODULE.diagnose_composition(saved_run, tmp_path / "diagnostic.json")
    assert report["is_test_fixture"] is True
    assert report["gold_rule_agreement_on_prior_rows"] == 1
    assert report["source_artifacts_unchanged"]
    assert before == MODULE.HELPER._snapshot(saved_run)
    for view in ("omniscient", "sender"):
        test = report["views"][view]["groups"]["test"]
        assert test["world"]["n_rows"] == test["sender_recorded"]["n_rows"] == 16
        assert test["world"]["accuracy"] == test["sender_recorded"]["accuracy"] == 1
        assert test["composed_nested"]["accuracy"] == 1
        assert test["original_direct_nested"]["accuracy"] < 1
        assert "binary_log_loss" not in test["composed_nested"]
        control = report["views"][view]["groups"]["control"]["composed_nested"]
        assert control == {"status": "out_of_scope_delivered_message_overrides_prior_rule"}


def test_hard_rule_and_both_wrong_error_cancellation():
    world, recorded = np.asarray([0, 0, 1, 1]), np.asarray([0, 1, 0, 1])
    nested = MODULE.hard_composition(world, recorded)
    np.testing.assert_array_equal(nested, [1, 0, 0, 1])
    np.testing.assert_array_equal(MODULE.hard_composition(1 - world, 1 - recorded), nested)
    assert MODULE.classification_metrics(nested, nested)["accuracy"] == 1
    with pytest.raises(ValueError, match="binary"):
        MODULE.hard_composition([2], [0])


def test_receipt_selection_api_omits_test_and_controls():
    assert list(inspect.signature(MODULE.fit_receipt_head).parameters) == [
        "X_train",
        "y_train",
        "X_validation",
        "y_validation",
    ]
    train = np.asarray([[-1.0], [-0.5], [0.5], [1.0]])
    labels = np.asarray([0, 0, 1, 1])
    _, selected = MODULE.fit_receipt_head(train, labels, train, labels)
    assert (
        selected["selected_C"]
        == min(
            selected["candidates"], key=lambda item: (item["validation_binary_log_loss"], item["C"])
        )["C"]
    )


def test_all_fits_use_only_32_prior_training_rows(saved_run, tmp_path, monkeypatch):
    original = MODULE.LogisticRegression.fit
    sizes = []

    def checked_fit(self, X, y, *args, **kwargs):
        sizes.append(len(y))
        assert len(X) == len(y) == 32
        return original(self, X, y, *args, **kwargs)

    monkeypatch.setattr(MODULE.LogisticRegression, "fit", checked_fit)
    MODULE.diagnose_composition(saved_run, tmp_path / "diagnostic.json")
    assert len(sizes) == 14  # Two fixed heads + five receipt candidates, per view.


def test_inconsistent_authored_rule_fails_before_any_fit(saved_run, tmp_path, monkeypatch):
    source = Path(shutil.copytree(saved_run, tmp_path / "source"))
    protocol = read_json(source / "protocol.json")
    protocol["panel"]["cases"][0]["sender_recorded"] ^= 1
    atomic_json(source / "protocol.json", protocol)
    manifest = read_json(source / "manifest.json")
    manifest["panel_sha256"] = canonical_hash(protocol["panel"])
    manifest["protocol_sha256"] = manifest["output_sha256"]["protocol.json"] = file_sha256(
        source / "protocol.json"
    )
    atomic_json(source / "manifest.json", manifest)

    def no_fit(*args, **kwargs):
        raise AssertionError("Invalid rule must fail before fitting")

    monkeypatch.setattr(MODULE.LogisticRegression, "fit", no_fit)
    with pytest.raises(ValueError, match="specified Boolean rule"):
        MODULE.diagnose_composition(source, tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()
