"""Restore and replay the audited lexical predictions without fitting or models.

This deliberately reuses the production feature map and scoring implementation.
It is a portable fitted-feature prediction replay, not an independent refit or
independent numerical audit. The full result summary is never deserialized.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from threadpoolctl import threadpool_limits

from latent_probes.format_adaptation_fit import ARMS, fold_factors
from latent_probes.format_adaptation_text import VIEWS, load_text
from latent_probes.storage import (
    atomic_json,
    atomic_npz,
    canonical_hash,
    environment_info,
    file_sha256,
    read_json,
    utc_now,
)

ROOT = Path(__file__).resolve().parents[1]
ROW_COUNT = 4992
FIT_NAMES = tuple(
    f"fit/{view}/{name}"
    for view in VIEWS
    for name in ("settings.json", "arrays.npz", "manifest.json")
)
REPORT_NAMES = {
    "rows.json.gz",
    "predictions.npz",
    "summary.json.gz",
    "metrics.json",
    "preparation.json",
    "fit_seal.json",
    "analysis_complete.json",
    "fit_started.json",
    "analysis_started.json",
    "audit/audit_receipt.json",
    "audit/reconstruction_seal.json",
    "audit/full_messages_reconstruction.json.gz",
    "audit/system_only_reconstruction.json.gz",
    "fitted-text-readouts.zip",
    "latent-comparison.json",
    "README.md",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def original_runner():
    spec = importlib.util.spec_from_file_location(
        "lexical_prediction_replay_runner", ROOT / "scripts/run_format_adaptation_text.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def verify_report(directory):
    """Check the exact scientific export inventory; unrelated extras are ignored."""
    manifest = read_json(directory / "manifest.json")
    require(
        manifest.get("schema") == "format-adaptation-text-report-manifest-v1"
        and manifest.get("status") == "complete"
        and manifest.get("test_fixture") is False
        and manifest.get("human_review_complete") is False
        and manifest.get("new_language_model_forwards") == 0
        and manifest.get("classifier_fits") == 0,
        "Completed scientific lexical report required; fixtures are forbidden",
    )
    require(set(manifest.get("output_sha256", {})) == REPORT_NAMES, "Report inventory differs")
    for name, expected in manifest["output_sha256"].items():
        require(
            file_sha256(directory / name) == expected,
            "Report member differs from manifest: " + name,
        )
    return manifest


def restore_archive(directory, output):
    """Read exactly six regular members; never use archive-controlled extraction."""
    seal = read_json(directory / "fit_seal.json")
    require(
        seal.get("schema") == "adaptation-lexical-fit-seal-v1"
        and seal.get("all_views_frozen_before_scoring") is True
        and seal.get("preparation_sha256") == file_sha256(directory / "preparation.json")
        and set(seal.get("files_sha256", {})) == set(FIT_NAMES),
        "Invalid original fitted-instrument seal",
    )
    with zipfile.ZipFile(directory / "fitted-text-readouts.zip") as archive:
        members = archive.infolist()
        require(
            len(members) == len(FIT_NAMES)
            and {member.filename for member in members} == set(FIT_NAMES),
            "Archive must contain exactly six authorized fitting files",
        )
        for member in members:
            path = PurePosixPath(member.filename)
            require(
                not path.is_absolute()
                and ".." not in path.parts
                and "\\" not in member.filename
                and not member.is_dir()
                and (member.external_attr >> 16) & 0o170000 in (0, 0o100000),
                "Unsafe archive member",
            )
            # read() validates the member CRC. Only fixed authorized paths are written.
            data = archive.read(member)
            destination = output / member.filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(data)
            require(
                file_sha256(destination) == seal["files_sha256"][member.filename],
                "Restored fit member differs from original seal: " + member.filename,
            )
    return seal


def compare_predictions(rows, scores, prediction_path):
    expected_keys = {"row_ids"} | {
        f"{view}__{arm}__factor_margins" for view in VIEWS for arm in ARMS
    }
    expected_ids = [row["row_id"] for row in rows]
    result = {
        "margin_comparisons": 0,
        "strict_factor_bit_comparisons": 0,
        "verdict_comparisons": 0,
        "maximum_absolute_margin_difference": 0.0,
        "rtol": 1e-12,
        "atol": 1e-12,
        "views": {},
    }
    with np.load(prediction_path, allow_pickle=False) as original:
        require(set(original.files) == expected_keys, "Unexpected saved prediction inventory")
        ids = original["row_ids"]
        require(
            ids.dtype.kind == "U" and ids.shape == (ROW_COUNT,) and ids.tolist() == expected_ids,
            "Saved prediction row ordering differs",
        )
        require(set(scores) == set(VIEWS), "Missing or extra replayed view")
        for view in VIEWS:
            require(set(scores[view]) == set(ARMS), "Missing or extra replayed arm")
            result["views"][view] = {}
            for arm in ARMS:
                saved = original[f"{view}__{arm}__factor_margins"]
                current = scores[view][arm]
                margins = current["factor_margins"]
                require(
                    saved.dtype == np.float64
                    and saved.shape == np.shape(margins) == (ROW_COUNT, 4)
                    and np.asarray(margins).dtype == np.float64
                    and np.isfinite(saved).all()
                    and np.isfinite(margins).all(),
                    "Invalid saved or replayed margin shape/value",
                )
                difference = float(np.max(np.abs(saved - margins)))
                require(
                    np.allclose(saved, margins, rtol=1e-12, atol=1e-12),
                    f"Replayed factor margins differ: {view}/{arm}",
                )
                saved_bits = (saved > 0).astype(np.int64)
                require(
                    np.array_equal(current["factor_predictions"], (margins > 0).astype(np.int64))
                    and np.array_equal(current["factor_predictions"], saved_bits),
                    f"Strict factor decisions differ: {view}/{arm}",
                )
                require(
                    np.array_equal(current["verdicts"], fold_factors(saved_bits)),
                    f"Replayed external-fold verdicts differ: {view}/{arm}",
                )
                result["margin_comparisons"] += saved.size
                result["strict_factor_bit_comparisons"] += saved.size
                result["verdict_comparisons"] += ROW_COUNT
                result["maximum_absolute_margin_difference"] = max(
                    result["maximum_absolute_margin_difference"], difference
                )
                result["views"][view][arm] = {
                    "margins": saved.size,
                    "strict_factor_bits": saved.size,
                    "verdicts": ROW_COUNT,
                    "maximum_absolute_margin_difference": difference,
                }
    return result


def replay(report_dir, output_fresh_dir):
    directory, output = Path(report_dir).resolve(), Path(output_fresh_dir).resolve()
    require(directory.is_dir(), "Existing report directory required")
    require(not output.exists(), "Replay output exists; preserve all attempts")
    require(not output.is_relative_to(directory), "Replay output must be outside the report")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": "format-adaptation-text-portable-prediction-replay-v1",
        "status": "running",
        "started_at": utc_now(),
        "report": str(directory),
        "replay_source_sha256": file_sha256(__file__),
        "new_language_model_forwards": 0,
        "classifier_fits": 0,
        "scope": "Restored production text feature maps and prediction replay; not an independent audit or refit",
        "full_summary_deserialized": False,
        "environment": environment_info(),
    }
    atomic_json(output / "started.json", receipt)
    try:
        manifest = verify_report(directory)
        receipt["report_manifest_sha256"] = file_sha256(directory / "manifest.json")
        receipt["verified_report_members"] = len(manifest["output_sha256"])
        runner = original_runner()
        preparation = runner.check_preparation(directory)
        require(preparation.get("test_fixture", False) is False, "Fixture preparation forbidden")
        seal = restore_archive(directory, output)
        require(
            manifest.get("fitted_archive_members_sha256") == seal["files_sha256"],
            "Archive inventory differs from export manifest",
        )
        receipt["restored_fit_members_sha256"] = seal["files_sha256"]
        rows = runner.input_rows("test")
        require(
            isinstance(rows, list)
            and len(rows) == ROW_COUNT
            and canonical_hash(rows) == preparation.get("test_input_sha256"),
            "Reconstructed test inputs differ from prepared bank",
        )
        runner.verify_original_test(rows)
        exported = json.loads(gzip.decompress((directory / "rows.json.gz").read_bytes()))["rows"]
        require(len(exported) == len(rows), "Exported formal-row inventory differs")
        for row, old in zip(rows, exported, strict=True):
            require(
                all(row[key] == old[key] for key in (*runner.ROW_FIELDS, "messages_sha256")),
                "Exported test metadata differs from reconstructed input",
            )
        records = [{"messages": row["messages"], "renderer": row["renderer"]} for row in rows]
        scores = {}
        with threadpool_limits(limits=4):
            for view in VIEWS:
                scores[view] = load_text(output / "fit" / view).score(records)
        arrays = {"row_ids": np.array([row["row_id"] for row in rows])}
        for view in VIEWS:
            for arm in ARMS:
                arrays[f"{view}__{arm}__factor_margins"] = scores[view][arm]["factor_margins"]
        atomic_npz(output / "replayed_predictions.npz", **arrays)
        atomic_json(
            output / "prediction_reconstruction_seal.json",
            {
                "schema": "lexical-production-prediction-reconstruction-seal-v1",
                "original_prediction_arrays_opened": False,
                "full_summary_deserialized": False,
                "prediction_sha256": file_sha256(output / "replayed_predictions.npz"),
                "restored_fit_members_sha256": seal["files_sha256"],
            },
        )
        receipt["comparisons"] = compare_predictions(rows, scores, directory / "predictions.npz")
        verify_report(directory)
        runner.check_preparation(directory)
        receipt.update(
            status="pass",
            finished_at=utc_now(),
            reconstructed_rows=ROW_COUNT,
            test_input_sha256=canonical_hash(rows),
            prediction_reconstruction_sha256=file_sha256(output / "replayed_predictions.npz"),
            prediction_reconstruction_seal_sha256=file_sha256(
                output / "prediction_reconstruction_seal.json"
            ),
        )
        atomic_json(output / "replay_receipt.json", receipt)
        return receipt
    except BaseException as error:
        receipt.update(
            status="failed_preserved",
            finished_at=utc_now(),
            error_type=type(error).__name__,
            error=str(error),
        )
        atomic_json(output / "replay_receipt.json", receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    print(json.dumps(replay(options.report, options.output), indent=2))


if __name__ == "__main__":
    main()
