"""Analyse a complete held-out capture against the both-model frozen fit seal."""

import argparse
import json
from pathlib import Path

from latent_probes.format_adaptation_analysis import analyse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyse(args.capture, args.fit, args.seal, args.output)
    print(
        json.dumps(
            {
                "model_key": result["model_key"],
                "primary": result["primary"],
                "named_diagnostic": result["named_diagnostic"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
