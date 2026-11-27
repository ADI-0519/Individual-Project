"""Fit fresh readout heads, validate safe bundles, or seal both models pre-test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_probes.fresh_readout_fit import fit_bundle, freeze_fits, validate_fit_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    fit = sub.add_parser("fit", help="Fit all12 source heads and separate correction controls")
    fit.add_argument(
        "--capture", type=Path, required=True, help="Completed training capture directory"
    )
    fit.add_argument("--bank", type=Path, required=True)
    fit.add_argument("--protocol", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    validate = sub.add_parser(
        "validate", help="Check provenance, policy, safe arrays and selections"
    )
    validate.add_argument("--bundle", type=Path, required=True)
    freeze = sub.add_parser("freeze", help="Seal both models before any fresh test forward")
    freeze.add_argument("--llama-fit", type=Path, required=True)
    freeze.add_argument("--qwen-fit", type=Path, required=True)
    freeze.add_argument("--protocols", type=Path, nargs=2, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "fit":
        result = fit_bundle(args.capture, args.bank, args.protocol, args.output)
        result = {
            "status": result["status"],
            "model": result["capture_provenance"]["model_key"],
            "S0": result["pipelines"]["S0"],
            "S1": result["pipelines"]["S1"],
        }
    elif args.mode == "validate":
        settings, _ = validate_fit_bundle(args.bundle)
        result = {
            "status": settings["status"],
            "model": settings["capture_provenance"]["model_key"],
        }
    else:
        result = freeze_fits(args.llama_fit, args.qwen_fit, args.protocols, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
