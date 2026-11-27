"""Sealed, CPU-only lexical controls for the exposed format-adaptation study."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from latent_probes.format_adaptation_bank import audit_rows, iter_rows, load
from latent_probes.format_adaptation_fit import ARMS, training_indices
from latent_probes.format_adaptation_text import VIEWS, fit_text, load_text
from latent_probes.format_adaptation_text_analysis import score_text_arrays
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
BANK = "data/belief_main/format_adaptation_v1/compiled_v1/bank.json"
OLD = "docs/results/format-adaptation-2026-11-27"
PROTOCOL = "docs/LEXICAL_CONTROL_PROTOCOL_V1.md"
ROW_FIELDS = ("row_id", "group_id", "partition", "split", "renderer", "condition", "state", "gold")
SOURCE_FILES = (
    "src/latent_probes/format_adaptation_text.py",
    "src/latent_probes/format_adaptation_text_analysis.py",
    "src/latent_probes/format_adaptation_fit.py",
    "src/latent_probes/format_adaptation_analysis.py",
    "src/latent_probes/format_adaptation_bank.py",
    "src/latent_probes/fresh_readout_bank.py",
    "src/latent_probes/main_study_design.py",
    "src/latent_probes/main_study_bank.py",
    "src/latent_probes/storage.py",
    "scripts/run_format_adaptation_text.py",
    "scripts/audit_format_adaptation_text.py",
    "tests/test_format_adaptation_text.py",
    "tests/test_format_adaptation_text_analysis.py",
    "tests/test_run_format_adaptation_text.py",
    "tests/test_audit_format_adaptation_text.py",
    "docs/validation/2026-11-27-lexical-control-prerun-review-v1.json",
    PROTOCOL,
)
INPUT_FILES = (
    BANK,
    f"{OLD}/preparation.json",
    f"{OLD}/replay/manifest.json",
    f"{OLD}/replay/llama/rows.json.gz",
    f"{OLD}/replay/qwen/rows.json.gz",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def gzip_json(path, value):
    """Deterministic compressed JSON, written only inside an exclusive run."""
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as stream:
        stream.write(gzip.compress(data, mtime=0))


def read_gzip(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def input_rows(partition):
    bank = load(ROOT / BANK)
    return [
        {**row, "row_id": row["record_id"], "messages_sha256": canonical_hash(row["messages"])}
        for row in iter_rows(bank, partition)
    ]


def check_original_sources():
    """Original study remains immutable; use repository-relative suffixes."""
    preparation = read_json(ROOT / OLD / "preparation.json")
    require(file_sha256(ROOT / BANK) == preparation["bank_sha256"], "Changed original bank")
    verified, historical = 0, {}
    for path, digest in preparation["source_sha256"].items():
        if "Individual-Project/" not in path:
            require(
                path.startswith("/home/adirj/kcl-latent-probes-runs/"), "Unexpected historical path"
            )
            historical[path] = digest
            continue
        relative = path.split("Individual-Project/", 1)[1]
        require(file_sha256(ROOT / relative) == digest, f"Changed original source: {relative}")
        verified += 1
    require(verified == 39 and len(historical) == 11, "Original source inventory changed")
    return {
        "repository_bindings_verified": verified,
        "historical_native_bindings_preserved_not_required_for_portable_run": historical,
    }


def verify_original_test(rows):
    """Every text input must match both previously recorded model inputs."""
    manifest = read_json(ROOT / OLD / "replay" / "manifest.json")
    require(
        manifest.get("schema") == "format-adaptation-portable-replay-v1"
        and manifest.get("status") == "complete"
        and manifest.get("test_fixture") is False,
        "Original projections must come from the completed scientific replay",
    )
    for model in ("llama", "qwen"):
        path = ROOT / OLD / "replay" / model / "rows.json.gz"
        require(
            file_sha256(path) == manifest["files"].get(f"{model}/rows.json.gz"),
            "Original row projection differs from replay manifest: " + model,
        )
        original = read_gzip(path)
        saved = original["rows"] if isinstance(original, dict) else original
        require(len(saved) == len(rows) == 4992, "Wrong saved test inventory")
        for row, old in zip(rows, saved, strict=True):
            require(
                all(row[field] == old[field] for field in (*ROW_FIELDS, "messages_sha256")),
                f"Text input differs from {model} original: {row['row_id']}",
            )


def check_preparation(output):
    preparation = read_json(Path(output) / "preparation.json")
    require(preparation.get("schema") == "adaptation-lexical-preparation-v1", "Wrong preparation")
    for field in ("source_sha256", "input_sha256"):
        for relative, digest in preparation[field].items():
            require(file_sha256(ROOT / relative) == digest, f"Changed sealed artifact: {relative}")
    check_original_sources()
    return preparation


def prepare(output):
    output = Path(output)
    require(not output.exists(), "Output exists; preserve all attempts")
    old_count = check_original_sources()
    audit = audit_rows(load(ROOT / BANK))
    fitting, test = input_rows("fit"), input_rows("test")
    indices = training_indices(fitting)
    verify_original_test(test)
    value = {
        "schema": "adaptation-lexical-preparation-v1",
        "created_at": utc_now(),
        "research_status": "predeclared_posthoc_exposed_bank_diagnostic",
        "human_review_complete": False,
        "source_sha256": {p: file_sha256(ROOT / p) for p in SOURCE_FILES},
        "input_sha256": {p: file_sha256(ROOT / p) for p in INPUT_FILES},
        "original_preparation_binding_check": old_count,
        "bank_audit": audit,
        "arm_indices_sha256": {k: canonical_hash(list(map(int, v))) for k, v in indices.items()},
        "fit_input_sha256": canonical_hash(fitting),
        "test_input_sha256": canonical_hash(test),
        "environment": environment_info(),
        "new_language_model_forwards": 0,
    }
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "preparation.json", value)
    return value


def exclusive_start(output, stage):
    with (Path(output) / f"{stage}_started.json").open("x", encoding="utf-8") as stream:
        json.dump({"created_at": utc_now(), "stage": stage}, stream)


def fit(output):
    output = Path(output)
    preparation = check_preparation(output)
    exclusive_start(output, "fit")
    rows = input_rows("fit")
    require(canonical_hash(rows) == preparation["fit_input_sha256"], "Fitting inputs changed")
    with threadpool_limits(limits=4):
        for view in VIEWS:
            fitted = fit_text(rows, view)
            fitted.save(output / "fit" / view)
    check_preparation(output)
    files = sorted(p for p in (output / "fit").rglob("*") if p.is_file())
    seal = {
        "schema": "adaptation-lexical-fit-seal-v1",
        "created_at": utc_now(),
        "preparation_sha256": file_sha256(output / "preparation.json"),
        "files_sha256": {
            str(p.relative_to(output)).replace("\\", "/"): file_sha256(p) for p in files
        },
        "all_views_frozen_before_scoring": True,
        "research_status": "posthoc_diagnostic_not_fresh_confirmation",
    }
    atomic_json(output / "fit_seal.json", seal)
    return seal


def check_fit_seal(output):
    output = Path(output)
    seal = read_json(output / "fit_seal.json")
    require(seal.get("schema") == "adaptation-lexical-fit-seal-v1", "Wrong fit seal")
    require(
        seal.get("preparation_sha256") == file_sha256(output / "preparation.json"),
        "Changed preparation",
    )
    expected = seal["files_sha256"]
    actual = {
        str(p.relative_to(output)).replace("\\", "/"): file_sha256(p)
        for p in (output / "fit").rglob("*")
        if p.is_file()
    }
    require(actual == expected and len(expected) >= 4, "Changed/missing/unexpected fitted artifact")
    return seal


def analyse(output):
    output = Path(output)
    preparation = check_preparation(output)
    check_fit_seal(output)
    exclusive_start(output, "analysis")
    rows = input_rows("test")
    require(canonical_hash(rows) == preparation["test_input_sha256"], "Test inputs changed")
    verify_original_test(rows)
    records = [{"messages": r["messages"], "renderer": r["renderer"]} for r in rows]
    summary = {
        "schema": "adaptation-lexical-summary-v1",
        "created_at": utc_now(),
        "research_status": "posthoc_exposed_bank_diagnostic",
        "human_review_complete": False,
        "shared_text_comparator_not_checkpoint_replications": True,
        "new_language_model_forwards": 0,
        "views": {},
    }
    arrays = {"row_ids": np.array([r["row_id"] for r in rows])}
    with threadpool_limits(limits=4):
        for view in VIEWS:
            scores = load_text(output / "fit" / view).score(records)
            for arm in ARMS:
                arrays[f"{view}__{arm}__factor_margins"] = scores[arm]["factor_margins"]
            summary["views"][view] = score_text_arrays(rows, scores)
    check_preparation(output)
    check_fit_seal(output)
    metadata = [{k: r[k] for k in (*ROW_FIELDS, "messages_sha256")} for r in rows]
    gzip_json(output / "rows.json.gz", {"rows": metadata})
    atomic_npz(output / "predictions.npz", **arrays)
    gzip_json(output / "summary.json.gz", summary)
    compact = {
        **{k: v for k, v in summary.items() if k != "views"},
        "views": {
            view: {
                "arms": {
                    arm: {scope: entry["aggregate"] for scope, entry in scopes.items()}
                    for arm, scopes in result["arms"].items()
                },
                "factor_diagnostics": {
                    arm: {scope: entry["aggregate"] for scope, entry in scopes.items()}
                    for arm, scopes in result["factor_diagnostics"].items()
                },
                "comparisons": result["comparisons"],
            }
            for view, result in summary["views"].items()
        },
    }
    atomic_json(output / "metrics.json", compact)
    atomic_json(
        output / "analysis_complete.json",
        {
            "created_at": utc_now(),
            "files_sha256": {
                name: file_sha256(output / name)
                for name in ("rows.json.gz", "predictions.npz", "summary.json.gz", "metrics.json")
            },
            "preparation_sha256": file_sha256(output / "preparation.json"),
            "fit_seal_sha256": file_sha256(output / "fit_seal.json"),
            "awaits_independent_numerical_audit": True,
        },
    )
    return compact


def verify(output):
    output = Path(output)
    check_preparation(output)
    check_fit_seal(output)
    receipt = read_json(output / "analysis_complete.json")
    for name, digest in receipt["files_sha256"].items():
        require(file_sha256(output / name) == digest, f"Changed scored output: {name}")
    require(
        receipt["preparation_sha256"] == file_sha256(output / "preparation.json"),
        "Preparation mismatch",
    )
    require(
        receipt["fit_seal_sha256"] == file_sha256(output / "fit_seal.json"), "Fit seal mismatch"
    )
    return {"status": "pass", "scored_files": len(receipt["files_sha256"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fit", "analyse", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    functions = {"prepare": prepare, "fit": fit, "analyse": analyse, "verify": verify}
    result = functions[args.stage](args.output)
    print(
        json.dumps({"stage": args.stage, "output": str(args.output), "result_keys": list(result)})
    )


if __name__ == "__main__":
    main()
