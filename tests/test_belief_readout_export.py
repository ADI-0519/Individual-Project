"""Matched fixture exports reproduce neutral heads without model loading."""

from __future__ import annotations

import importlib.util
import inspect
import shutil
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from latent_probes import belief_readout
from latent_probes.storage import atomic_json, atomic_npz, canonical_hash, file_sha256, read_json

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_belief_readout.py"
SPEC = importlib.util.spec_from_file_location("belief_readout_export", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture(scope="module")
def saved_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("belief_export_source")
    rows, cases, matrices, captures = [], [], [], []
    for split in ("train", "validation", "test"):
        template = "template-" + split
        cells = [(w, r, n, "prior", split) for w, r, n in product((0, 1), repeat=3)]
        cells += [(w, w, w, "delivered_current", "control") for w in (0, 1)]
        for world, reader, nested, phase, row_split in cells:
            case = {
                "id": f"{template}-{phase}-{world}-{reader}-{nested}",
                "template_id": template,
                "split": row_split,
                "phase": phase,
                "labels": {"world": world, "reader": reader, "nested": nested},
                "claim_place_bit": 1,
                "messages": {},
            }
            for view in ("omniscient", "sender"):
                text = f"worldbit{world} nestedbit{nested} {template} {phase}"
                if view == "omniscient":
                    text += f" actualreaderbit{reader}"
                messages = [{"role": "user", "content": text}]
                case["messages"][view] = messages
                identifier = case["id"] + "__" + view
                vector = np.asarray(
                    [world, reader if view == "omniscient" else 0, nested, 5.0], dtype=np.float32
                )
                features = {
                    layer: vector * (i + 1) for i, layer in enumerate(belief_readout.LAYERS)
                }
                file = "prefix_features/" + identifier + ".npz"
                atomic_npz(root / file, **features)
                matrices.append(features)
                rows.append(
                    {
                        "example_id": identifier,
                        "template_id": template,
                        "split": row_split,
                        "phase": phase,
                        "view": view,
                        "messages_text": text,
                        "labels": case["labels"],
                    }
                )
                captures.append(
                    {
                        "example_id": identifier,
                        "messages_sha256": canonical_hash(messages),
                        "feature_file": file,
                        "feature_sha256": file_sha256(root / file),
                        "prompt_token_ids": [1, 2],
                        "selected_position": 1,
                        "generated_answer_tokens_supplied": 0,
                        "perspective_question_supplied": False,
                    }
                )
            cases.append(case)
    features = {
        layer: np.stack([item[layer] for item in matrices]) for layer in belief_readout.LAYERS
    }
    report = belief_readout.evaluate_belief_readouts(rows, features)
    panel = {"panel_id": "fixture_belief_panel", "cases": cases}
    protocol = {"primary_layer": "layer_16", "panel": panel}
    atomic_json(root / "protocol.json", protocol)
    atomic_json(root / "rows.json", rows)
    atomic_json(root / "captures.json", captures)
    atomic_json(root / "readouts.json", report)
    atomic_npz(root / "features.npz", **features)
    outputs = MODULE._snapshot(root)
    atomic_json(
        root / "manifest.json",
        {
            "schema": "belief_construct_development_run_v1",
            "status": "complete",
            "is_test_fixture": True,
            "backend": {
                "backend": "fixture",
                "is_test_fixture": True,
                "model_id": "fixture",
                "resolved_revision": "fixture-pin",
                "dtype": "float32",
                "quantization": "none",
            },
            "protocol_sha256": outputs["protocol.json"],
            "panel_sha256": canonical_hash(panel),
            "source_sha256": {
                "src/latent_probes/belief_readout.py": file_sha256(inspect.getfile(belief_readout))
            },
            "output_sha256": outputs,
        },
    )
    return root


def copy_run(source, tmp_path):
    return Path(shutil.copytree(source, tmp_path / "source"))


def update_report_hash(source):
    manifest = read_json(source / "manifest.json")
    manifest["output_sha256"]["readouts.json"] = file_sha256(source / "readouts.json")
    atomic_json(source / "manifest.json", manifest)


def test_exported_numeric_heads_match_all_metrics_and_preserve_source(saved_run, tmp_path):
    before = MODULE._snapshot(saved_run)
    output = tmp_path / "export"
    manifest = MODULE.export_readouts(saved_run, output)
    assert manifest["status"] == "complete" and manifest["is_test_fixture"] is True
    assert manifest["source_artifacts_unchanged"]
    assert before == MODULE._snapshot(saved_run)
    verification = read_json(output / "verification.json")
    for view in ("omniscient", "sender"):
        assert verification[view]["n_training_rows"] == 8
        assert verification[view]["n_training_templates"] == 1
        assert verification[view]["scaler_and_classifier_fingerprints_exact"]
        assert verification[view]["independent_probability_max_abs_error"] <= 1e-12
        assert len(verification[view]["test_predictions"]) == 8
        path = output / (view + "_layer_16_nested.npz")
        with np.load(path, allow_pickle=False) as archive:
            assert all(archive[key].dtype.kind != "O" for key in archive.files)
            assert np.array_equal(archive["classes"], [0, 1])
            import json

            signature = json.loads(archive["signature_json"].item())
            assert signature["view"] == view and signature["is_test_fixture"] is True
            assert signature["layer"] == 16 and signature["candidate_place_bit"] == 1
            assert signature["is_natural_message_forecast"] is False


def test_pure_numpy_probe_has_correct_zero_norm_and_sigmoid_behavior():
    probe = {
        "mean": np.asarray([1.0, 2.0]),
        "scale": np.asarray([2.0, 1.0]),
        "coef": np.asarray([[2.0, 0.0]]),
        "intercept": np.asarray([0.3]),
        "classes": np.asarray([0, 1]),
    }
    probabilities = MODULE.apply_probe([[1.0, 2.0], [3.0, 2.0], [-1.0, 2.0]], probe)
    np.testing.assert_allclose(probabilities[:, 1], expit([0.3, 2.3, -1.7]), rtol=0, atol=0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    with pytest.raises(ValueError, match="shape mismatch"):
        MODULE.apply_probe([[1.0]], probe)


def test_export_fits_only_prior_training_rows_at_recorded_C(saved_run, tmp_path, monkeypatch):
    expected = read_json(saved_run / "readouts.json")
    observed = []
    original_fit = MODULE.LogisticRegression.fit

    def checked_fit(self, X, y, *args, **kwargs):
        assert len(X) == len(y) == 8
        observed.append(self.C)
        return original_fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(MODULE.LogisticRegression, "fit", checked_fit)
    MODULE.export_readouts(saved_run, tmp_path / "export")
    assert observed == [
        expected["readouts"][view]["nested"]["layers"]["layer_16"]["selected_C"]
        for view in ("omniscient", "sender")
    ]


@pytest.mark.parametrize("corruption", ["artifact", "fingerprint", "metric"])
def test_source_tampering_or_unreproducible_report_prevents_export(saved_run, tmp_path, corruption):
    source = copy_run(saved_run, tmp_path)
    if corruption == "artifact":
        with (source / "features.npz").open("ab") as stream:
            stream.write(b"tampered")
        message = "Source artifact hash mismatch"
    else:
        report = read_json(source / "readouts.json")
        head = report["readouts"]["omniscient"]["nested"]["layers"]["layer_16"]
        if corruption == "fingerprint":
            head["fit_fingerprint"]["coefficients_sha256"] = "0" * 64
            message = "Classifier fingerprint mismatch"
        else:
            head["metrics"]["test"]["accuracy"] = 0.123
            message = "Source metric mismatch"
        atomic_json(source / "readouts.json", report)
        update_report_hash(source)
    with pytest.raises(ValueError, match=message):
        MODULE.export_readouts(source, tmp_path / "export")
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("corruption", ["running", "flag_mismatch", "masquerading_fixture"])
def test_incomplete_or_mismatched_fixture_sources_are_rejected(saved_run, tmp_path, corruption):
    source = copy_run(saved_run, tmp_path)
    manifest = read_json(source / "manifest.json")
    if corruption == "running":
        manifest["status"] = "running"
        message = "complete belief diagnostic"
    elif corruption == "flag_mismatch":
        manifest["is_test_fixture"] = False
        message = "flags mismatch"
    else:
        manifest["is_test_fixture"] = manifest["backend"]["is_test_fixture"] = False
        message = "pinned HF identity"
    atomic_json(source / "manifest.json", manifest)
    with pytest.raises(ValueError, match=message):
        MODULE.export_readouts(source, tmp_path / "export")


def test_export_cannot_write_inside_source_or_overwrite_existing_directory(saved_run, tmp_path):
    with pytest.raises(ValueError, match="separate from source"):
        MODULE.export_readouts(saved_run, saved_run / "export")
    output = tmp_path / "already-exists"
    output.mkdir()
    with pytest.raises(ValueError, match="fresh export"):
        MODULE.export_readouts(saved_run, output)
