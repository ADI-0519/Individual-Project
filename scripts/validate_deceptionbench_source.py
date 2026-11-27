"""Validate local published DeceptionBench files without generation or judging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_probes.deceptionbench_data import iter_deceptionbench_rows, load_deceptionbench


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    dataset = load_deceptionbench(args.source_directory, args.manifest)
    rows = list(iter_deceptionbench_rows(dataset))
    print(
        json.dumps(
            {
                **dataset["audit"],
                "unique_case_ids": len({r["case_id"] for r in rows}),
                "source_manifest_sha256": dataset["source_manifest_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
