"""Prepare or execute a sealed, CPU-only post-hoc readout adequacy comparison."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_info, threadpool_limits

from latent_probes.cached_readout_adequacy import (
    STATUS,
    aggregate,
    canonical_hash,
    file_hash,
    load_data,
    policy,
    read_json,
    require,
    required_source_paths,
    run_fold,
    validate_folds,
    verify_protocol,
    write_json,
)


def environment():
    pools = [
        {k: entry.get(k) for k in ("user_api", "internal_api", "prefix", "version", "num_threads")}
        for entry in threadpool_info()
    ]
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "thread_limit": 4,
        "threadpools": sorted(pools, key=lambda entry: json.dumps(entry, sort_keys=True)),
    }


def now():
    return datetime.now(UTC).isoformat()


def prepare(protocol_path, output):
    with threadpool_limits(limits=4):
        return _prepare(protocol_path, output)


def _prepare(protocol_path, output):
    output = output.resolve()
    require(not output.exists(), "Use a fresh output directory")
    require(
        "onedrive" not in str(output).lower()
        and not str(output).startswith("/mnt/")
        and not str(output).startswith("\\\\"),
        "Use fresh native storage",
    )
    protocol_path = protocol_path.resolve()
    protocol, folds = verify_protocol(protocol_path)
    project = Path(__file__).resolve().parents[1]
    require(not output.is_relative_to(project), "Output must be outside the source repository")
    tracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            "--",
            *[str(p.relative_to(project)) for p in sorted(required_source_paths())],
        ],
        cwd=project,
        capture_output=True,
        text=True,
    )
    require(
        tracked.returncode == 0,
        "Track protocol/code in Git before preparation: " + tracked.stderr.strip(),
    )
    models = {}
    for spec in protocol["models"]:
        data = load_data(spec)
        validate_folds(folds, [r["group_id"] for r in data.rows])
        models[spec["key"]] = {
            "rows": len(data.rows),
            "features": data.features.shape[1],
            "case_ids_sha256": canonical_hash([r["case_id"] for r in data.rows]),
        }
    verify_protocol(protocol_path)
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "prepared_manifest.json",
        {
            "schema": "cached-readout-adequacy-preparation-v1",
            "created_at": now(),
            "protocol_path": str(protocol_path),
            "protocol_sha256": file_hash(protocol_path),
            "research_status": STATUS,
            "model_calls": 0,
            "classifier_fits": 0,
            "policy": policy(),
            "environment": environment(),
            "models": models,
        },
    )


def execute(output):
    with threadpool_limits(limits=4):
        return _execute(output)


def _execute(output):
    seal = read_json(output / "prepared_manifest.json")
    seal_hash = file_hash(output / "prepared_manifest.json")
    require(seal["schema"] == "cached-readout-adequacy-preparation-v1", "Unknown preparation seal")
    protocol_path = Path(seal["protocol_path"])
    require(
        file_hash(protocol_path) == seal["protocol_sha256"]
        and environment() == seal["environment"],
        "Protocol/environment changed",
    )
    protocol, folds = verify_protocol(protocol_path)
    write_json(
        output / "execution_started.json",
        {
            "started_at": now(),
            "prepared_manifest_sha256": file_hash(output / "prepared_manifest.json"),
            "model_calls": 0,
        },
    )
    try:
        for spec in protocol["models"]:
            data = load_data(spec)
            validate_folds(folds, [r["group_id"] for r in data.rows])
            require(
                canonical_hash([r["case_id"] for r in data.rows])
                == seal["models"][spec["key"]]["case_ids_sha256"],
                "Prepared source ID inventory differs",
            )
            model_dir = output / spec["key"]
            model_dir.mkdir(exist_ok=False)
            all_folds = []
            for fold in folds:
                require(
                    file_hash(output / "prepared_manifest.json") == seal_hash
                    and file_hash(protocol_path) == seal["protocol_sha256"]
                    and environment() == seal["environment"],
                    "Seal/protocol/environment changed before fitting",
                )
                verify_protocol(protocol_path)
                fold_dir = model_dir / f"fold_{fold['fold']}"
                fold_dir.mkdir(exist_ok=False)

                def event(value):
                    with (fold_dir / "candidate_events.jsonl").open(
                        "a", encoding="utf-8"
                    ) as stream:
                        stream.write(json.dumps(value, allow_nan=False) + "\n")

                result, arrays = run_fold(data, fold, event)
                np.savez_compressed(fold_dir / "heads.npz", **arrays)
                result["heads_sha256"] = file_hash(fold_dir / "heads.npz")
                result["array_sha256"] = {k: sha(v) for k, v in arrays.items()}
                write_json(fold_dir / "results.json", result)
                all_folds.append(result)
                print(f"{spec['key']} fold {fold['fold']} complete", flush=True)
            write_json(model_dir / "oof_results.json", aggregate(data, all_folds))
        require(
            file_hash(output / "prepared_manifest.json") == seal_hash
            and file_hash(protocol_path) == seal["protocol_sha256"]
            and environment() == seal["environment"],
            "Seal/protocol/environment changed during fitting",
        )
        verify_protocol(protocol_path)
        write_json(
            output / "completed_manifest.json",
            {
                "status": "complete",
                "finished_at": now(),
                "research_status": STATUS,
                "confirmatory_result": False,
                "model_calls": 0,
                "prepared_manifest_sha256": file_hash(output / "prepared_manifest.json"),
                "output_sha256": {
                    str(p.relative_to(output)): file_hash(p)
                    for p in sorted(output.rglob("*"))
                    if p.is_file()
                },
            },
        )
    except BaseException as error:
        write_json(
            output / "failure.json",
            {
                "status": "failed_preserved",
                "failed_at": now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "candidate_details": getattr(error, "details", None),
                "traceback": traceback.format_exc(),
                "model_calls": 0,
            },
        )
        raise


def sha(values):
    from hashlib import sha256

    return sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--protocol", type=Path, required=True)
    preparation.add_argument("--output", type=Path, required=True)
    execution = commands.add_parser("execute")
    execution.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.protocol, args.output)
    else:
        execute(args.output.resolve())


if __name__ == "__main__":
    main()
