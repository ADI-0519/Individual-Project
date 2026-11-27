"""Prepare or capture the fixed new adaptation study; no retries or generation."""

import argparse
from pathlib import Path

from latent_probes.format_adaptation_capture import capture, prepare_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "capture"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--old-root", type=Path)
    parser.add_argument("--review-receipt", type=Path)
    parser.add_argument("--launch-script", type=Path)
    parser.add_argument("--partition", choices=("fit", "test"))
    parser.add_argument("--seal", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if not all((args.bank, args.old_root, args.review_receipt, args.launch_script)):
            parser.error("Preparation requires bank, old-root, review-receipt and launch-script")
        prepare_campaign(
            args.bank, args.old_root, args.output, args.review_receipt, args.launch_script
        )
    else:
        if not args.partition:
            parser.error("Capture requires partition")
        capture(args.output, args.partition, args.seal)


if __name__ == "__main__":
    main()
