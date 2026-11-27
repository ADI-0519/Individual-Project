"""Synthetic restoration, strict prediction comparisons and failure preservation."""

from __future__ import annotations

import copy
import gzip
import importlib.util
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from latent_probes.storage import canonical_hash, file_sha256

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/replay_format_adaptation_text.py"
SPEC = importlib.util.spec_from_file_location("lexical_prediction_replay_test", SCRIPT)
replayer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replayer)


def put(path, value):
    data = (json.dumps(value, indent=2) + "\n").encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)


def save_predictions(path, rows, scores):
    np.savez_compressed(
        path,
        row_ids=np.array([r["row_id"] for r in rows]),
        **{
            f"{view}__{arm}__factor_margins": scores[view][arm]["factor_margins"]
            for view in replayer.VIEWS
            for arm in replayer.ARMS
        },
    )


def fake_scores(size):
    margin = ((np.arange(size * 4).reshape(size, 4) % 5) - 2).astype(np.float64)
    return {
        view: {
            arm: {
                "factor_margins": margin.copy(),
                "factor_predictions": (margin > 0).astype(int),
                "verdicts": replayer.fold_factors((margin > 0).astype(int)),
            }
            for arm in replayer.ARMS
        }
        for view in replayer.VIEWS
    }


def fixture(tmp_path, monkeypatch):
    source, output = tmp_path / "report", tmp_path / "replay"
    source.mkdir()
    monkeypatch.setattr(replayer, "ROW_COUNT", 4)
    rows = [
        {
            "row_id": f"synthetic_{i}",
            "partition": "test",
            "renderer": "A" if i % 2 else "B",
            "messages": [
                {"role": "system", "content": "Follow the test instructions"},
                {"role": "user", "content": f"Synthetic fixture content {i}"},
            ],
            "messages_sha256": str(i),
        }
        for i in range(4)
    ]
    scores = fake_scores(4)
    for name in replayer.REPORT_NAMES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic test-only placeholder")
    preparation = {"test_fixture": False, "test_input_sha256": canonical_hash(rows)}
    put(source / "preparation.json", preparation)
    members = {name: ("synthetic member " + name).encode() for name in replayer.FIT_NAMES}
    with zipfile.ZipFile(source / "fitted-text-readouts.zip", "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    import hashlib

    fit_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in members.items()}
    put(
        source / "fit_seal.json",
        {
            "schema": "adaptation-lexical-fit-seal-v1",
            "all_views_frozen_before_scoring": True,
            "preparation_sha256": file_sha256(source / "preparation.json"),
            "files_sha256": fit_hashes,
        },
    )
    metadata = [
        {k: r[k] for k in ("row_id", "partition", "renderer", "messages_sha256")} for r in rows
    ]
    put(source / "rows.json.gz", {"rows": metadata})
    save_predictions(source / "predictions.npz", rows, scores)
    manifest = {
        "schema": "format-adaptation-text-report-manifest-v1",
        "status": "complete",
        "test_fixture": False,
        "human_review_complete": False,
        "new_language_model_forwards": 0,
        "classifier_fits": 0,
        "output_sha256": {name: file_sha256(source / name) for name in replayer.REPORT_NAMES},
        "fitted_archive_members_sha256": fit_hashes,
    }
    put(source / "manifest.json", manifest)
    runner = SimpleNamespace(
        ROW_FIELDS=("row_id", "partition", "renderer"),
        check_preparation=lambda path: preparation,
        input_rows=lambda partition: copy.deepcopy(rows),
        verify_original_test=lambda supplied: supplied == rows,
    )
    monkeypatch.setattr(replayer, "original_runner", lambda: runner)

    class FrozenReadout:
        def __init__(self, view):
            self.view = view

        def score(self, records):
            assert all(set(r) == {"messages", "renderer"} for r in records)
            assert len(records) == 4
            return copy.deepcopy(scores[self.view])

    monkeypatch.setattr(replayer, "load_text", lambda path: FrozenReadout(path.name))
    return source, output, rows, scores, runner


def refresh_manifest(source, name):
    manifest = replayer.read_json(source / "manifest.json")
    manifest["output_sha256"][name] = file_sha256(source / name)
    put(source / "manifest.json", manifest)


def test_complete_prediction_replay_has_no_fitting_or_summary_read(tmp_path, monkeypatch):
    source, output, _, _, _ = fixture(tmp_path, monkeypatch)
    original_read = replayer.read_json

    def guarded(path):
        assert "summary" not in Path(path).name
        return original_read(path)

    monkeypatch.setattr(replayer, "read_json", guarded)
    (source / "unrelated-figure-extra.svg").write_text("Ignored separately bound figure")
    receipt = replayer.replay(source, output)
    assert receipt["status"] == "pass"
    assert receipt["verified_report_members"] == 16
    assert receipt["classifier_fits"] == receipt["new_language_model_forwards"] == 0
    assert receipt["full_summary_deserialized"] is False
    assert receipt["comparisons"]["margin_comparisons"] == 2 * 4 * 4 * 4
    assert receipt["comparisons"]["verdict_comparisons"] == 2 * 4 * 4
    assert receipt["comparisons"]["maximum_absolute_margin_difference"] == 0
    assert set(receipt["restored_fit_members_sha256"]) == set(replayer.FIT_NAMES)
    assert (
        original_read(output / "prediction_reconstruction_seal.json")[
            "original_prediction_arrays_opened"
        ]
        is False
    )


def test_public_replay_rejects_fixture_and_preserves_failure(tmp_path, monkeypatch):
    source, output, _, _, _ = fixture(tmp_path, monkeypatch)
    manifest = replayer.read_json(source / "manifest.json")
    manifest["test_fixture"] = True
    put(source / "manifest.json", manifest)
    with pytest.raises(ValueError, match="fixtures are forbidden"):
        replayer.replay(source, output)
    assert replayer.read_json(output / "replay_receipt.json")["status"] == "failed_preserved"
    before = (output / "replay_receipt.json").read_bytes()
    with pytest.raises(ValueError, match="output exists"):
        replayer.replay(source, output)
    assert (output / "replay_receipt.json").read_bytes() == before


def test_report_hash_inventory_and_outside_output_are_required(tmp_path, monkeypatch):
    source, output, _, _, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="outside the report"):
        replayer.replay(source, source / "forbidden")
    (source / "metrics.json").write_text("changed")
    with pytest.raises(ValueError, match="differs from manifest"):
        replayer.replay(source, output)
    assert replayer.read_json(output / "replay_receipt.json")["status"] == "failed_preserved"


@pytest.mark.parametrize("damage", ("extra", "traversal", "symlink", "wrong_bytes", "duplicate"))
def test_archive_extra_path_and_member_changes_are_rejected(tmp_path, monkeypatch, damage):
    source, output, _, _, _ = fixture(tmp_path, monkeypatch)
    archive_path = source / "fitted-text-readouts.zip"
    with zipfile.ZipFile(archive_path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    if damage == "extra":
        members["unexpected.json"] = b"extra"
    elif damage == "traversal":
        members["../escape.json"] = members.pop(replayer.FIT_NAMES[0])
    elif damage == "wrong_bytes":
        members[replayer.FIT_NAMES[0]] = b"changed"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, data in members.items():
            if damage == "symlink" and name == replayer.FIT_NAMES[0]:
                info = zipfile.ZipInfo(name)
                info.external_attr = 0o120777 << 16
                archive.writestr(info, b"/arbitrary/target")
            else:
                archive.writestr(name, data)
        if damage == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr(replayer.FIT_NAMES[0], b"duplicate")
    refresh_manifest(source, "fitted-text-readouts.zip")
    with pytest.raises(ValueError):
        replayer.replay(source, output)
    assert not (tmp_path / "escape.json").exists()
    assert replayer.read_json(output / "replay_receipt.json")["status"] == "failed_preserved"


@pytest.mark.parametrize("damage", ("order", "margins", "bits", "fold", "nonfinite", "dtype"))
def test_prediction_comparison_rejects_mismatches(tmp_path, monkeypatch, damage):
    _, _, rows, scores, _ = fixture(tmp_path, monkeypatch)
    expected = tmp_path / "saved.npz"
    save_predictions(expected, rows[::-1] if damage == "order" else rows, scores)
    if damage == "margins":
        scores["full_messages"]["Mixed512"]["factor_margins"][0, 0] += 0.1
    elif damage == "bits":
        scores["system_only"]["Offset"]["factor_predictions"][0, 0] ^= 1
    elif damage == "fold":
        scores["full_messages"]["A512"]["verdicts"][0] ^= 1
    elif damage == "nonfinite":
        scores["system_only"]["Ahalf256"]["factor_margins"][0, 0] = np.nan
    elif damage == "dtype":
        scores["full_messages"]["A512"]["factor_margins"] = scores["full_messages"]["A512"][
            "factor_margins"
        ].astype(np.float32)
    with pytest.raises(ValueError):
        replayer.compare_predictions(rows, scores, expected)


def test_tolerance_never_hides_a_strict_zero_threshold_flip(tmp_path, monkeypatch):
    _, _, rows, scores, _ = fixture(tmp_path, monkeypatch)
    expected = tmp_path / "saved.npz"
    save_predictions(expected, rows, scores)
    one = scores["full_messages"]["Mixed512"]
    one["factor_margins"][0, 2] = 1e-14  # Original exact zero; within margin tolerance.
    one["factor_predictions"][0, 2] = 1
    one["verdicts"] = replayer.fold_factors(one["factor_predictions"])
    with pytest.raises(ValueError, match="Strict factor decisions"):
        replayer.compare_predictions(rows, scores, expected)


def test_reconstructed_input_mismatch_stops_before_scoring(tmp_path, monkeypatch):
    source, output, _, _, runner = fixture(tmp_path, monkeypatch)
    runner.check_preparation = lambda path: {"test_input_sha256": "changed"}
    monkeypatch.setattr(
        replayer, "load_text", lambda path: pytest.fail("Must not score changed inputs")
    )
    with pytest.raises(ValueError, match="Reconstructed test inputs"):
        replayer.replay(source, output)
    assert not (output / "prediction_reconstruction_seal.json").exists()


def test_scoring_failure_remains_and_cannot_be_retried(tmp_path, monkeypatch):
    source, output, _, _, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        replayer,
        "load_text",
        lambda path: (_ for _ in ()).throw(ValueError("synthetic score failure")),
    )
    with pytest.raises(ValueError, match="synthetic score failure"):
        replayer.replay(source, output)
    receipt = replayer.read_json(output / "replay_receipt.json")
    assert receipt["status"] == "failed_preserved" and receipt["classifier_fits"] == 0
    with pytest.raises(ValueError, match="output exists"):
        replayer.replay(source, output)
