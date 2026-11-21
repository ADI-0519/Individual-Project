"""Merge human-review fragments and report readiness without fabricating review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_probes.main_study_review import merge_review_fragments, write_fresh_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--fragments", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, help="New combined ledger; omit for check-only")
    parser.add_argument("--report", type=Path, help="New readiness report; never overwrites")
    args = parser.parse_args()
    outputs = [p.resolve() for p in (args.output, args.report) if p is not None]
    try:
        if len(set(outputs)) != len(outputs) or any(p.exists() for p in outputs):
            raise ValueError("Use distinct fresh output paths; preserve existing human work")
        merged, report = merge_review_fragments(args.bank, args.fragments)
        if args.output:
            write_fresh_json(args.output, merged)
        if args.report:
            write_fresh_json(args.report, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["review_complete"] else 2
    except (ValueError, OSError) as error:
        print(
            json.dumps(
                {"status": "invalid", "review_complete": False, "error": str(error)}, indent=2
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
