"""Fit matched heads from a reviewed train/development capture only."""

from __future__ import annotations

import argparse
from pathlib import Path

from latent_probes.main_study_bank import bank_digest, load_bank, training_groups
from latent_probes.main_study_fit import fit_study, load_feature_cache, save_fit_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-fixture", action="store_true", help="Engineering fixtures only")
    args = parser.parse_args()
    cache = load_feature_cache(args.capture)
    if cache.test_fixture and not args.allow_fixture:
        parser.error("Fixture fitting requires --allow-fixture and remains labelled synthetic")
    bank = load_bank(args.bank)
    if cache.provenance.get("bank_sha256") != bank_digest(bank):
        parser.error("Capture and bank SHA256 do not match")
    settings, arrays = fit_study(cache, training_groups(bank))
    save_fit_bundle(args.output, settings, arrays)
    print(f"Saved {len(settings['comparisons'])} fixed comparisons; fixture={cache.test_fixture}")


if __name__ == "__main__":
    main()
