"""Bind reviewed study materials and policies; this command never loads models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_probes.main_study_execution import DEFAULT_BASELINE_PROTOCOL, build_execution_freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--review-ledger", type=Path, required=True)
    parser.add_argument("--baseline-protocol", type=Path, default=DEFAULT_BASELINE_PROTOCOL)
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh JSON output; no overwrite"
    )
    parser.add_argument("--stage", choices=("train_dev", "test"), default="train_dev")
    parser.add_argument("--fit-dir", type=Path, action="append", default=[])
    args = parser.parse_args()
    result = build_execution_freeze(
        args.bank,
        args.review_ledger,
        args.baseline_protocol,
        args.output,
        stage=args.stage,
        fit_directories=tuple(args.fit_dir),
    )
    print(
        json.dumps(
            {key: result[key] for key in ("status", "stage", "bank_sha256", "model_calls")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
