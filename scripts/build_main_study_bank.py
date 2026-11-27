"""Assemble authored main-study material and blank review records; no inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_probes.main_study_materials import build_materials


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragments", nargs="+", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, required=True)
    args = parser.parse_args()
    result = build_materials(args.fragments, args.examples, args.output, args.review_output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
