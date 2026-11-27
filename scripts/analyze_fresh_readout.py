"""Score a complete fresh-test capture without fitting or choosing sites."""

import argparse
import json
from pathlib import Path

from latent_probes.fresh_readout_analysis import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("test-capture", "fit", "protocol", "bank", "seal", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.test_capture, args.fit, args.protocol, args.bank, args.seal, args.output)
    print(
        json.dumps(
            {
                "model": result["model_key"],
                "primary": result["primary"],
                "research_status": result["research_status"],
            }
        )
    )


if __name__ == "__main__":
    main()
