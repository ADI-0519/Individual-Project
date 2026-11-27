"""Independent lexical metric checker, using only hand-authored fake states."""

from __future__ import annotations

import copy
import gzip
import importlib.util
import json
from itertools import product
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_format_adaptation_text.py"
SPEC = importlib.util.spec_from_file_location("independent_text_control_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def synthetic_rows():
    rows = []
    for group_index in range(24):
        group = f"test_{group_index:02d}"
        for fmt in ("A", "B"):
            for initial, message, first, last, world, names in product((0, 1), repeat=6):
                state = dict(
                    group_id=group,
                    mechanism="correction",
                    initial=initial,
                    update=message,
                    first_receipt=first,
                    receipt=last,
                    world=world,
                    name_map=names,
                )
                rows.append(
                    dict(
                        group_id=group,
                        renderer=fmt,
                        partition="test",
                        split="test",
                        state=state,
                        condition="correction_easy" if initial == message else "correction_hard",
                        gold=(1 - message) if last else message if first else initial,
                    )
                )
            for initial, message, read, world, names in product((0, 1), repeat=5):
                state = dict(
                    group_id=group,
                    mechanism="message",
                    initial=initial,
                    update=message,
                    first_receipt=None,
                    receipt=read,
                    world=world,
                    name_map=names,
                )
                rows.append(
                    dict(
                        group_id=group,
                        renderer=fmt,
                        partition="test",
                        split="test",
                        state=state,
                        condition="one_step",
                        gold=message if read else initial,
                    )
                )
            for bit, world, names in product((0, 1), repeat=3):
                rows.append(
                    dict(
                        group_id=group,
                        renderer=fmt,
                        partition="test",
                        split="test",
                        condition="asserted",
                        gold=bit,
                        state=dict(
                            group_id=group,
                            mechanism="asserted",
                            asserted=bit,
                            world=world,
                            name_map=names,
                        ),
                    )
                )
    for index, row in enumerate(rows):
        row["row_id"] = f"synthetic_{index:04d}"
    return rows


def fake_margins(rows):
    truth = np.array(
        [
            [r["state"][field] for field in audit.FIELDS]
            if r["condition"].startswith("correction_")
            else [r["gold"], 0, 0, 0]
            for r in rows
        ],
        dtype=np.float64,
    )
    perfect = 2 * truth - 1
    return {
        "A512": np.zeros((4992, 4), dtype=np.float64),
        "Mixed512": perfect,
        "Ahalf256": np.zeros((4992, 4), dtype=np.float64),
        "Offset": perfect.copy(),
    }


def put(path, data):
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump(data, stream, allow_nan=False)
    else:
        path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")


def test_checker_imports_no_production_modules():
    import ast

    module = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert all(not name.startswith("latent_probes") for name in imported)


def test_binary_ties_use_zero_class_and_known_class_recalls():
    result = audit.binary([0, 1, 0, 1], np.zeros(4, dtype=int))
    assert result["correct"] == 2
    assert result["class_counts"] == {"0": 2, "1": 2}
    assert result["class_recalls"] == {"0": 1.0, "1": 0.0}


@pytest.mark.parametrize("damage", ["gold", "duplicate", "stratum", "split", "metadata"])
def test_formal_input_damage_is_rejected(damage):
    rows = synthetic_rows()
    if damage == "gold":
        rows[0]["gold"] = 1 - rows[0]["gold"]
    elif damage == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif damage == "stratum":
        rows[0]["condition"] = "correction_hard"
    elif damage == "split":
        rows[0]["partition"] = "fit"
    else:
        rows[0]["messages"] = [{"role": "user", "content": "not metadata"}]
    with pytest.raises(ValueError):
        audit.validate_rows(rows)


def test_hand_computed_perfect_panels_and_zero_score_failures():
    rows = [
        r for r in synthetic_rows() if r["renderer"] == "B" and r["condition"] == "correction_hard"
    ]
    perfect = audit.verdict(rows, [r["gold"] for r in rows])
    assert perfect["rows"] == 768 and perfect["complete_quartets"] == 192
    assert perfect["quartet_correct_cells_histogram"] == {"0": 0, "1": 0, "2": 0, "3": 0, "4": 192}
    assert perfect["nuisance_pairs"]["world"] == {
        "pairs": 384,
        "verdict_flips": 0,
        "both_correct": 384,
    }
    constant = audit.verdict(rows, np.zeros(768, dtype=int))
    assert constant["row_accuracy"] == 0.5 and constant["complete_quartets"] == 0
    assert constant["quartet_correct_cells_histogram"]["1"] == 96
    assert constant["quartet_correct_cells_histogram"]["3"] == 96


def test_auroc_orientation_and_all_four_correctness():
    rows = [
        r
        for r in synthetic_rows()
        if r["renderer"] == "B" and r["condition"].startswith("correction_")
    ]
    truth = np.array([[r["state"][field] for field in audit.FIELDS] for r in rows])
    perfect = audit.factors(rows, 2 * truth - 1)
    reversed_scores = audit.factors(rows, 1 - 2 * truth)
    assert perfect["all_four_correct"] == 1
    assert reversed_scores["all_four_correct"] == 0
    assert all(v["raw_auroc"] == 1 for v in perfect["factors"].values())
    assert all(v["raw_auroc"] == 0 for v in reversed_scores["factors"].values())


def test_bootstrap_preserves_context_units():
    values = {f"g{i:02d}": {"Mixed512": float(i % 2), "A512": 0.0} for i in range(24)}
    result = audit.contrast(values, "A512")
    assert result["difference"] == 0.5
    assert result["conditional_percentile_interval_95"] == pytest.approx([7 / 24, 17 / 24])
    assert result["role"] == "posthoc_descriptive_comparison" and result["quartets"] == 192


@pytest.mark.parametrize(
    "expected,actual",
    [
        ({"n": 192}, {"n": 191}),
        ({"x": 0.4}, {"x": 0.5}),
        ({"a": None}, {"a": 0}),
        ({"a": 1}, {"a": 1, "b": 2}),
    ],
)
def test_changed_summary_values_and_inventory_are_rejected(expected, actual):
    with pytest.raises(ValueError):
        audit.compare_tree(expected, actual)


def test_both_views_frozen_before_summary_and_gzip_supported(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "audit"
    source.mkdir()
    rows = synthetic_rows()
    margins = fake_margins(rows)
    expected = audit.reconstruct(rows, margins)
    assert expected["comparisons"]["Mixed512_minus_A512"]["difference"] == 1
    assert expected["comparisons"]["Mixed512_minus_Offset"]["difference"] == 0
    put(source / "rows.json.gz", {"rows": rows})
    put(source / "summary.json.gz", {"views": {view: expected for view in audit.VIEWS}})
    np.savez_compressed(
        source / "predictions.npz",
        row_ids=np.array([r["row_id"] for r in rows]),
        **{
            f"{view}__{arm}__factor_margins": values
            for view in audit.VIEWS
            for arm, values in margins.items()
        },
    )
    original_read, observed = audit.read, []

    def guarded_read(path):
        if Path(path).name.startswith("summary"):
            seal = original_read(output / "reconstruction_seal.json")
            assert seal["saved_summary_opened"] is False
            assert seal["views_reconstructed"] == list(audit.VIEWS)
            assert all(
                audit.digest(output / name) == sha for name, sha in seal["output_sha256"].items()
            )
            observed.append(str(path))
        return original_read(path)

    monkeypatch.setattr(audit, "read", guarded_read)
    receipt = audit.audit(source, output)
    assert receipt["status"] == "pass" and len(observed) == 1
    assert receipt["model_calls"] == receipt["classifier_fits"] == 0
    assert all(v["numeric"] > 1000 for v in receipt["views"].values())
    assert all(v["exact"] > 1000 for v in receipt["views"].values())
    with pytest.raises(ValueError, match="fresh"):
        audit.audit(source, output)


def test_row_order_damage_preserves_failure_without_reading_summary(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "audit"
    source.mkdir()
    rows = synthetic_rows()
    put(source / "rows.json", {"rows": rows})
    put(source / "summary.json", {"views": {}})
    ids = [r["row_id"] for r in rows]
    ids[0], ids[1] = ids[1], ids[0]
    np.savez_compressed(
        source / "predictions.npz",
        row_ids=np.array(ids),
        **{
            f"{view}__{arm}__factor_margins": np.zeros((4992, 4))
            for view in audit.VIEWS
            for arm in audit.ARMS
        },
    )
    original_read = audit.read

    def guarded_read(path):
        assert not Path(path).name.startswith("summary"), "Summary opened before reconstruction"
        return original_read(path)

    monkeypatch.setattr(audit, "read", guarded_read)
    with pytest.raises(ValueError, match="ordering"):
        audit.audit(source, output)
    assert original_read(output / "audit_receipt.json")["status"] == "failed_preserved"
    assert not (output / "reconstruction_seal.json").exists()
