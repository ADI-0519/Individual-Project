"""Evaluate the frozen prose monitor with invalid predictions kept incorrect."""

from __future__ import annotations

import argparse
from pathlib import Path

from latent_probes.main_study_monitor_analysis import analyse_monitor_run
from latent_probes.storage import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-fixture", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve the previous analysis; output already exists")
    result = analyse_monitor_run(args.bank, args.run, allow_fixture=args.allow_fixture)
    atomic_json(args.output, result)
    print(result["overall"])


if __name__ == "__main__":
    main()
