"""Synthetic roster, leakage, sparse fitting and safe restoration contracts."""

from __future__ import annotations

import copy
import itertools
import subprocess
import sys
import warnings

import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.exceptions import ConvergenceWarning

from latent_probes import format_adaptation_fit as latent
from latent_probes import format_adaptation_text as text
from latent_probes.storage import atomic_json, file_sha256, read_json


def synthetic_rows():
    rows = []
    for group in range(16):
        identity = f"synthetic_fit_{group:02d}"
        for initial, update, b1, b2, world, mapping in itertools.product((0, 1), repeat=6):
            if initial ^ update ^ world ^ mapping != group % 2:
                continue
            assignment = "A" if world == (group // 2) % 2 else "B"
            for renderer in ("A",) if assignment == "A" else ("A", "B"):
                state = dict(
                    group_id=identity,
                    mechanism="correction",
                    initial=initial,
                    update=update,
                    first_receipt=b1,
                    receipt=b2,
                    world=world,
                    name_map=mapping,
                )
                identifier = f"{identity}_{initial}{update}{b1}{b2}{world}{mapping}_{renderer}"
                words = ["positive" if bit else "negative" for bit in (initial, update, b1, b2)]
                rows.append(
                    dict(
                        row_id=identifier,
                        case_id=identifier,
                        group_id=identity,
                        partition="fit",
                        split="fit",
                        condition="correction_hard" if initial != update else "correction_easy",
                        state=state,
                        renderer=renderer,
                        mixed_format=assignment,
                        gold=1 - update if b2 else update if b1 else initial,
                        extra_metadata="forbiddenmetadatasentinel",
                        messages=[
                            {
                                "role": "system",
                                "content": f"Follow the shared instructions. Format {renderer}.",
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Initial {words[0]}; update {words[1]}; "
                                    f"firstreceipt {words[2]}; lastreceipt {words[3]}. "
                                    + ("beacononly" if renderer == "B" else "atlasone")
                                ),
                            },
                        ],
                    )
                )
    return rows


@pytest.fixture(scope="module")
def rows():
    return synthetic_rows()


@pytest.fixture(scope="module")
def fitted(rows):
    return text.fit_text(rows, "full_messages")


def project(rows):
    return [{"messages": row["messages"], "renderer": row["renderer"]} for row in rows]


def test_fixed_policy_and_matching_sparse_fit_counts(rows, monkeypatch):
    observed = []
    original = text.fit_lr

    def recorded(values, labels, C):
        assert issparse(values)
        np.testing.assert_allclose(np.asarray(values.multiply(values).sum(axis=1)).ravel(), 1)
        observed.append((values.shape[0], C, int(labels.sum())))
        return original(values, labels, C)

    monkeypatch.setattr(text, "fit_lr", recorded)
    fitted = text.fit_text(rows, "full_messages")
    assert observed == [(512, 100, 256)] * 8 + [(256, 200, 128)] * 4
    assert fitted.metadata["policy"] == text.text_policy()
    assert fitted.metadata["arms"]["Offset"]["factor_positive_counts_B"] == [128] * 4
    for arm, count in (("A512", 512), ("Mixed512", 512), ("Ahalf256", 256)):
        receipt = fitted.metadata["arms"][arm]
        assert receipt["observations"] == len(set(receipt["fit_row_ids"])) == count
        assert (
            receipt["factor_positive_counts"]
            == receipt["factor_negative_counts"]
            == [count // 2] * 4
        )


def test_each_arm_vocabulary_uses_only_its_own_selected_observations(fitted):
    assert "beacononly" in fitted.maps["Mixed512"].word.vocabulary_
    assert "beacononly" not in fitted.maps["A512"].word.vocabulary_
    assert "beacononly" not in fitted.maps["Ahalf256"].word.vocabulary_
    for mapping in fitted.maps.values():
        assert "forbiddenmetadatasentinel" not in mapping.word.vocabulary_
    assert set(fitted.maps) == set(text.FIT_ARMS)
    assert fitted.maps["A512"] is not fitted.maps["Mixed512"]


def test_scoring_new_text_never_refits_or_learns_test_vocabulary(fitted, rows, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("A held-out score attempted to fit a vocabulary")

    monkeypatch.setattr(text.TfidfVectorizer, "fit", forbidden)
    monkeypatch.setattr(text.TfidfVectorizer, "fit_transform", forbidden)
    records = project(copy.deepcopy(rows[:4]))
    for record in records:
        record["messages"][1]["content"] += " neverobservedvalidationword"
    scores = fitted.score(records)
    assert set(scores) == set(latent.ARMS)
    for mapping in fitted.maps.values():
        assert "neverobservedvalidationword" not in mapping.word.vocabulary_
    for values in scores.values():
        assert values["factor_margins"].shape == values["factor_predictions"].shape == (4, 4)
        np.testing.assert_array_equal(
            values["verdicts"], latent.fold_factors(values["factor_predictions"])
        )


def test_offset_reuses_exact_ahalf_map_and_heads_and_keeps_a_identical(fitted, rows):
    records = project(rows[:12])
    scores = fitted.score(records)
    is_b = np.array([row["renderer"] == "B" for row in records])
    base, shifted = scores["Ahalf256"], scores["Offset"]
    for field in ("factor_margins", "factor_predictions", "verdicts"):
        np.testing.assert_array_equal(base[field][~is_b], shifted[field][~is_b])
    np.testing.assert_allclose(
        shifted["factor_margins"][is_b] - base["factor_margins"][is_b],
        np.broadcast_to(fitted.offsets, (is_b.sum(), 4)),
        atol=1e-15,
        rtol=0,
    )
    assert "Offset" not in fitted.maps
    assert "Offset" not in fitted.coefficients
    assert (
        fitted.metadata["arms"]["Offset"]["A_fit_row_ids"]
        == fitted.metadata["arms"]["Ahalf256"]["fit_row_ids"]
    )


@pytest.mark.parametrize("mutation", ("test", "missing", "duplicate", "assignment", "gold"))
def test_bad_or_test_roster_rejected_before_text_access(rows, monkeypatch, mutation):
    damaged = copy.deepcopy(rows)
    if mutation == "test":
        damaged[0]["partition"] = "test"
    elif mutation == "missing":
        damaged.pop()
    elif mutation == "duplicate":
        damaged[-1] = copy.deepcopy(damaged[0])
    elif mutation == "assignment":
        damaged[0]["mixed_format"] = "B" if damaged[0]["mixed_format"] == "A" else "A"
    elif mutation == "gold":
        damaged[0]["gold"] = 1 - damaged[0]["gold"]

    def forbidden(*_args):
        raise AssertionError("Text was opened before validating the fitting roster")

    monkeypatch.setattr(text, "serialize_messages", forbidden)
    with pytest.raises(ValueError):
        text.fit_text(damaged, "full_messages")


def test_serializer_excludes_all_formal_metadata_and_system_view_excludes_user(rows):
    row = copy.deepcopy(rows[0])
    before = text.serialize_messages(row["messages"], "full_messages")
    row.update(row_id="gold1", gold=999, state={"secret": "neverfeature"}, metadata="neverfeature")
    assert text.serialize_messages(row["messages"], "full_messages") == before
    assert "neverfeature" not in before
    system = text.serialize_messages(row["messages"], "system_only")
    row["messages"][1]["content"] = "Completely different user text and answer."
    assert text.serialize_messages(row["messages"], "system_only") == system
    assert text.serialize_messages(row["messages"], "full_messages") != before


def test_system_only_fit_never_learns_user_vocabulary_and_scores_independently(rows):
    fitted = text.fit_text(rows, "system_only")
    for mapping in fitted.maps.values():
        for term in ("beacononly", "atlasone", "initial", "firstreceipt"):
            assert term not in mapping.word.vocabulary_
    records = project(copy.deepcopy(rows[:6]))
    original = fitted.score(records)
    for record in records:
        record["messages"][1]["content"] = "arbitrary different user location and receipt"
    after = fitted.score(records)
    for arm in latent.ARMS:
        np.testing.assert_array_equal(original[arm]["factor_margins"], after[arm]["factor_margins"])


@pytest.mark.parametrize("mutation", ("role", "content", "extra", "missing", "reordered"))
def test_serializer_rejects_malformed_or_enriched_messages(rows, mutation):
    messages = copy.deepcopy(rows[0]["messages"])
    if mutation == "role":
        messages[1]["role"] = "assistant"
    elif mutation == "content":
        messages[1]["content"] = 1
    elif mutation == "extra":
        messages[0]["answer"] = 1
    elif mutation == "missing":
        del messages[0]["role"]
    elif mutation == "reordered":
        messages.reverse()
    with pytest.raises(ValueError):
        text.serialize_messages(messages, "full_messages")


@pytest.mark.parametrize("field", ("gold", "state", "row_id", "output_margins"))
def test_score_refuses_label_or_metadata_fields(fitted, rows, field):
    records = project(copy.deepcopy(rows[:1]))
    records[0][field] = "forbidden"
    with pytest.raises(ValueError, match="labels and metadata are forbidden"):
        fitted.score(records)


def test_fit_nonconvergence_stops_without_rescue(rows, monkeypatch):
    calls = []

    def nonconvergent(self, *_args, **_kwargs):
        calls.append(True)
        warnings.warn("synthetic stop", ConvergenceWarning, stacklevel=2)
        return self

    monkeypatch.setattr(latent.LogisticRegression, "fit", nonconvergent)
    with pytest.raises(ValueError, match="did not converge"):
        text.fit_text(rows, "full_messages")
    assert len(calls) == 1


def test_fit_nonfinite_head_stops(rows, monkeypatch):
    def nonfinite(self, values, *_args, **_kwargs):
        self.classes_ = np.array([0, 1])
        self.coef_ = np.full((1, values.shape[1]), np.nan)
        self.intercept_ = np.zeros(1)
        return self

    monkeypatch.setattr(latent.LogisticRegression, "fit", nonfinite)
    with pytest.raises(ValueError, match="Invalid fitted factor parameters"):
        text.fit_text(rows, "full_messages")


def test_safe_json_npz_roundtrip_never_refits_and_preserves_exact_predictions(
    fitted, rows, tmp_path, monkeypatch
):
    output = tmp_path / "fit"
    fitted.save(output)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Loading fitted text must not refit")

    monkeypatch.setattr(text.TfidfVectorizer, "fit", forbidden)
    monkeypatch.setattr(text.TfidfVectorizer, "fit_transform", forbidden)
    restored = text.load_text(output)
    assert restored.metadata == fitted.metadata
    records = project(rows[:12])
    before, after = fitted.score(records), restored.score(records)
    for arm in latent.ARMS:
        for field in before[arm]:
            np.testing.assert_array_equal(before[arm][field], after[arm][field])
    with np.load(output / "arrays.npz", allow_pickle=False) as arrays:
        assert all(arrays[name].dtype == np.float64 for name in arrays.files)
    with pytest.raises(ValueError, match="already exists"):
        fitted.save(output)


def test_safe_load_rejects_changed_artifact(fitted, tmp_path):
    output = tmp_path / "fit"
    fitted.save(output)
    with (output / "settings.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="artifact changed"):
        text.load_text(output)


def test_safe_load_rejects_invalid_vocabulary_even_with_updated_manifest(fitted, tmp_path):
    output = tmp_path / "fit"
    fitted.save(output)
    settings = read_json(output / "settings.json")
    settings["transforms"]["A512"]["word"]["vocabulary"] = {"duplicate": 0, "index": 0}
    atomic_json(output / "settings.json", settings)
    manifest = read_json(output / "manifest.json")
    manifest["output_sha256"]["settings.json"] = file_sha256(output / "settings.json")
    atomic_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="vocabulary indices"):
        text.load_text(output)


def test_score_rejects_empty_wrong_format_and_nonfinite_heads(fitted, rows):
    with pytest.raises(ValueError, match="nonempty"):
        fitted.score([])
    records = project(copy.deepcopy(rows[:1]))
    records[0]["renderer"] = "C"
    with pytest.raises(ValueError, match="A/B renderer"):
        fitted.score(records)
    original = fitted.coefficients["A512"][0, 0]
    try:
        fitted.coefficients["A512"][0, 0] = np.nan
        with pytest.raises(ValueError, match="Bad margins"):
            fitted.score(project(rows[:1]))
    finally:
        fitted.coefficients["A512"][0, 0] = original


def test_cpu_module_imports_without_torch_or_transformers():
    code = """
import builtins
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'bitsandbytes'}:
        raise AssertionError('Model import forbidden: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = checked
import latent_probes.format_adaptation_text
"""
    result = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
