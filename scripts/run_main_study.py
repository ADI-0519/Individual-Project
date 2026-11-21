"""Prepare exact main-study prefixes or execute an explicitly reviewed partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_probes.main_study_run import execute, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    pre = sub.add_parser("prepare", help="Tokenizer/config preflight; never loads weights")
    pre.add_argument("--bank", type=Path, required=True)
    pre.add_argument("--output", type=Path, required=True)
    pre.add_argument("--model", choices=("llama", "qwen"), default="llama")
    run = sub.add_parser("execute", help="Requires accepted review and execution freeze")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--partition", choices=("train_dev", "test"), required=True)
    run.add_argument("--review-ledger", type=Path, required=True)
    run.add_argument("--execution-freeze", type=Path, required=True)
    run.add_argument("--fit-dir", type=Path)
    run.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare(args.bank, args.output, model_key=args.model)["preflight"]
    else:
        result = execute(
            args.output,
            partition=args.partition,
            ledger_path=args.review_ledger,
            freeze_path=args.execution_freeze,
            fit_dir=args.fit_dir,
            resume=args.resume,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
