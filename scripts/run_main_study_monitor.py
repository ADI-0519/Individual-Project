"""Prepare or execute the fixed Qwen prose-only evidence-extraction baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_probes.main_study_monitor import DEFAULT_PROTOCOL, execute_monitor, prepare_monitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prepare = sub.add_parser("prepare", help="Tokenizer/config only; no model weights")
    prepare.add_argument("--bank", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--monitor-protocol", type=Path, default=DEFAULT_PROTOCOL)
    execute = sub.add_parser("execute", help="Requires actual human ledger and frozen sources")
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--partition", choices=("development", "test"), required=True)
    execute.add_argument("--review-ledger", type=Path, required=True)
    execute.add_argument("--execution-freeze", type=Path, required=True)
    execute.add_argument("--fit-dir", type=Path)
    execute.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare_monitor(args.bank, args.output, protocol_path=args.monitor_protocol)[
            "preflight"
        ]
    else:
        result = execute_monitor(
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
