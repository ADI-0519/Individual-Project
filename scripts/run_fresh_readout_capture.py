"""Prepare fresh-study exact prefixes or execute one frozen capture partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_probes.fresh_readout_capture import PARTITIONS, execute, prepare
from latent_probes.storage import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    pre = sub.add_parser("prepare", help="Pinned tokenizer/config only; never loads weights")
    pre.add_argument("--bank", type=Path, required=True)
    pre.add_argument("--output", type=Path, required=True)
    pre.add_argument("--model", choices=("llama", "qwen"), required=True)
    pre.add_argument(
        "--engineering-row", type=Path, help="JSON row outside every cohort, for hook/HF alignment"
    )
    pre.add_argument("--source-bindings", type=Path, help="Additional path-to-SHA256 JSON mapping")
    run = sub.add_parser("execute", help="Requires frozen review/source/protocol bindings")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--partition", choices=tuple(PARTITIONS), required=True)
    run.add_argument("--bindings", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare(
            args.bank,
            args.output,
            model_key=args.model,
            engineering_row=read_json(args.engineering_row) if args.engineering_row else None,
            source_bindings=read_json(args.source_bindings) if args.source_bindings else None,
        )
        result = result["preflight"]
    else:
        result = execute(
            args.output, partition=args.partition, bindings_path=args.bindings, resume=args.resume
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
