"""Score frozen matched heads on the complete held-out capture."""

from __future__ import annotations

import argparse
from pathlib import Path

from latent_probes.main_study_analysis import analyse_study, load_fit_bundle
from latent_probes.main_study_fit import load_feature_cache
from latent_probes.main_study_monitor_analysis import analyse_monitor_run
from latent_probes.storage import atomic_json, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--monitor-run", type=Path, help="Completed fixed prose-monitor test run")
    parser.add_argument(
        "--bank", type=Path, help="Exact authored bank; required with --monitor-run"
    )
    parser.add_argument(
        "--allow-fixture", action="store_true", help="Never produces scientific results"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve earlier analysis; output already exists")
    cache = load_feature_cache(args.capture)
    settings, arrays = load_fit_bundle(args.fit)
    result = analyse_study(cache, settings, arrays, allow_fixture=args.allow_fixture)
    result["prose_monitor"] = {"status": "not_provided"}
    if args.monitor_run:
        if args.bank is None:
            parser.error("--monitor-run requires the exact --bank")
        monitor = analyse_monitor_run(args.bank, args.monitor_run, allow_fixture=args.allow_fixture)
        if (
            monitor["partition"] != "test"
            or monitor["bank_sha256"] != cache.provenance.get("bank_sha256")
            or monitor["test_fixture"] != cache.test_fixture
        ):
            parser.error("Prose monitor must match the held-out bank and fixture provenance")
        result["prose_monitor"] = {"status": "complete", **monitor}
    result["fit_settings_sha256"] = file_sha256(args.fit / "settings.json")
    result["fit_heads_sha256"] = file_sha256(args.fit / "heads.npz")
    atomic_json(args.output, result)
    print(f"Saved held-out analysis; scientific_result={result['scientific_result']}")


if __name__ == "__main__":
    main()
