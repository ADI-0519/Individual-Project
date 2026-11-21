"""Reconstruct main inputs and verify prepared token/source bindings, without weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_probes.main_study_materials import audit_materials
from latent_probes.storage import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--preflights", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Preserve the earlier audit; choose a fresh output")
    result = audit_materials(args.bank, preflights=args.preflights)
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
