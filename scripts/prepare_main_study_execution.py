"""Bind reviewed study materials and policies; this command never loads models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_probes.main_study_execution import (
    DEFAULT_BASELINE_PROTOCOL,
    DEFAULT_REVIEW_PACKET,
    build_agent_execution_freeze,
    build_execution_freeze,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    review = parser.add_mutually_exclusive_group(required=True)
    review.add_argument("--review-ledger", type=Path, help="Actual human review ledger")
    review.add_argument(
        "--agent-audit", type=Path, help="Independent agent audit; exploratory only"
    )
    parser.add_argument("--review-packet", type=Path, default=DEFAULT_REVIEW_PACKET)
    parser.add_argument(
        "--initial-freeze", type=Path, help="Required for agent exploratory test stage"
    )
    parser.add_argument("--baseline-protocol", type=Path, default=DEFAULT_BASELINE_PROTOCOL)
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh JSON output; no overwrite"
    )
    parser.add_argument("--stage", choices=("train_dev", "test"), default="train_dev")
    parser.add_argument("--fit-dir", type=Path, action="append", default=[])
    args = parser.parse_args()
    builder = build_agent_execution_freeze if args.agent_audit else build_execution_freeze
    extra = (
        {"initial_freeze_path": args.initial_freeze, "review_packet_path": args.review_packet}
        if args.agent_audit
        else {}
    )
    if not args.agent_audit and args.initial_freeze:
        parser.error("--initial-freeze belongs only to the agent exploratory route")
    result = builder(
        args.bank,
        args.agent_audit or args.review_ledger,
        args.baseline_protocol,
        args.output,
        stage=args.stage,
        fit_directories=tuple(args.fit_dir),
        **extra,
    )
    print(
        json.dumps(
            {key: result[key] for key in ("status", "stage", "bank_sha256", "model_calls")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
