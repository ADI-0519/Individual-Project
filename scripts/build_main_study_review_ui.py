"""Build a blank offline HTML review worksheet from the exact bound review pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_probes.main_study_review_ui import build_review_ui


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--review-prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_review_ui(args.bank, args.review_prompts, args.output), indent=2))


if __name__ == "__main__":
    main()
