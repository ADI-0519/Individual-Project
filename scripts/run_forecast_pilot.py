"""Run the original neutral-to-pressure reader-forecast development milestone."""

import argparse
from dataclasses import replace
from pathlib import Path

from latent_probes.config import load_config
from latent_probes.forecast_experiment import run_forecast_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apollo-probe", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    config = replace(config, model=replace(config.model, local_files_only=True))
    run_forecast_experiment(config, args.panel, args.output, apollo_probe=args.apollo_probe)


if __name__ == "__main__":
    main()
