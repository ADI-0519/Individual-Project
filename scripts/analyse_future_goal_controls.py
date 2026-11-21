"""Verify a saved future-goal controls run and write its fixed CPU analysis."""

import argparse
from pathlib import Path

from latent_probes.future_goal_controls_analysis import analyse_run
from latent_probes.storage import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="JSON report outside the immutable run directory"
    )
    parser.add_argument(
        "--skip-tokenizer-check",
        action="store_true",
        help="Record binding only; explicitly omit independent pinned decoding",
    )
    args = parser.parse_args()
    output, run = args.output.resolve(), args.run.resolve()
    if output.is_relative_to(run):
        parser.error("Write the analysis outside the immutable source run directory")
    if output.exists():
        parser.error("Use a new report path; existing reports are not overwritten")
    result = analyse_run(run, verify_tokenizer=not args.skip_tokenizer_check)
    atomic_json(output, result)
    print(
        f"{result['status']}; neutral qualification passed={result['qualification']['passed']}; "
        f"saved conditions={result['n_saved_conditions']}; report={output}"
    )


if __name__ == "__main__":
    main()
