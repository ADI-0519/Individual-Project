"""Fit the four prospectively fixed format-adaptation arms without selection."""

import argparse
import json
from pathlib import Path

from latent_probes.format_adaptation_fit import fit_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--old-bundle", type=Path, required=True)
    parser.add_argument("--model-key", choices=("llama", "qwen"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = fit_bundle(args.capture, args.old_bundle, args.model_key, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "model_key": result["model_key"],
                "site": result["site"],
                "old_C2_C": result["old_C2_C"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
