"""One fixed-policy, reader-only diagnosis of irrelevant evidence substitution.

This is not a new capability qualification or a dependent recipient-probe fit.
It preserves the failed qualification and never opens the untouched ct bank.
"""

from __future__ import annotations

import argparse
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

from latent_probes.backends import HFBackend
from latent_probes.calibrated_reader import policy_metadata
from latent_probes.config import GenerationSettings, load_config
from latent_probes.evidence_role_panel import jobs, load_panel
from latent_probes.reader_panel import parse_decision
from latent_probes.storage import (
    atomic_json,
    canonical_hash,
    environment_info,
    file_sha256,
    utc_now,
)

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = GenerationSettings(max_new_tokens=8, do_sample=False)
SEED = 42


def strict_decision(generation):
    return parse_decision(generation["text"]) if generation["finish_reason"] == "eos" else "INVALID"


def bootstrap(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    return {
        "n_case_clusters": len(values),
        "mean": float(values.mean()),
        "case_bootstrap_percentile_95": np.quantile(
            rng.choice(values, (10000, len(values))).mean(axis=1), [0.025, 0.975]
        ).tolist(),
        "scope": "Conditional on authored instances of one rule grammar; not population family inference.",
    }


def summarise(rows):
    if (
        len(rows) != 96
        or len({r["example_id"] for r in rows}) != 96
        or len({r["case_id"] for r in rows}) != 16
    ):
        raise ValueError("Expected every condition on all16 case clusters")
    case_families = {
        case_id: {r["family"] for r in rows if r["case_id"] == case_id}
        for case_id in {r["case_id"] for r in rows}
    }
    if any(len(families) != 1 for families in case_families.values()) or Counter(
        next(iter(families)) for families in case_families.values()
    ) != {"date": 8, "score": 8}:
        raise ValueError("Each case must bind to one family, with eight distinct cases per family")
    grouped = {}
    for family in ("date", "score"):
        for condition in ("decisive", "metadata", "explicit_missing"):
            subset = [r for r in rows if r["family"] == family and r["condition"] == condition]
            if len(subset) != 16:
                raise ValueError("Expected two value sides for each of8 cases")
            grouped[family + "/" + condition] = {
                "n": len(subset),
                "correct": sum(r["decision"] == r["expected_decision"] for r in subset),
                "invalid": sum(r["decision"] == "INVALID" for r in subset),
                "decision_counts": dict(Counter(r["decision"] for r in subset)),
                "deferrals": sum(r["decision"] == "DEFER" for r in subset),
            }
    paired = {}
    for family in ("date", "score"):
        case_ids = sorted({r["case_id"] for r in rows if r["family"] == family})
        cases = []
        for case_id in case_ids:
            indexed = {(r["condition"], r["side"]): r for r in rows if r["case_id"] == case_id}
            if set(indexed) != {
                (c, s)
                for c in ("decisive", "metadata", "explicit_missing")
                for s in ("low", "high")
            }:
                raise ValueError("Missing paired condition")
            pairs = {}
            for condition in ("metadata", "explicit_missing"):
                low, high = (indexed[condition, s]["decision"] for s in ("low", "high"))
                expected_low, expected_high = (
                    indexed["decisive", s]["expected_decision"] for s in ("low", "high")
                )
                pairs[condition] = {
                    "low": low,
                    "high": high,
                    "any_flip": low != high,
                    "definite_flip": {low, high} == {"APPROVE", "REJECT"},
                    "follows_irrelevant_value_as_decisive": low == expected_low
                    and high == expected_high,
                    "both_defer": low == high == "DEFER",
                }
            improvement = float(
                np.mean(
                    [
                        int(indexed["explicit_missing", s]["decision"] == "DEFER")
                        - int(indexed["metadata", s]["decision"] == "DEFER")
                        for s in ("low", "high")
                    ]
                )
            )
            cases.append(
                {
                    "case_id": case_id,
                    "pairs": pairs,
                    "explicit_missing_minus_metadata_deferral": improvement,
                }
            )
        paired[family] = {
            "cases": cases,
            "n_cases": len(cases),
            "metadata_definite_flip_cases": sum(
                c["pairs"]["metadata"]["definite_flip"] for c in cases
            ),
            "metadata_value_following_cases": sum(
                c["pairs"]["metadata"]["follows_irrelevant_value_as_decisive"] for c in cases
            ),
            "explicit_missing_value_following_cases": sum(
                c["pairs"]["explicit_missing"]["follows_irrelevant_value_as_decisive"]
                for c in cases
            ),
            "deferral_intervention": bootstrap(
                [c["explicit_missing_minus_metadata_deferral"] for c in cases]
            ),
        }
    return {
        "schema": "evidence_role_diagnostic_summary_v1",
        "n_calls": 96,
        "n_cases": 16,
        "groups": grouped,
        "paired": paired,
        "qualifies_reader": False,
        "new_probe_fits": 0,
        "untouched_transfer_calls": 0,
        "limits": [
            "Post-failure diagnosis under an unchanged policy, not untouched main-study validation.",
            "Two authored rule grammars with8 instances each, not16 independent domains.",
            "Input/answer contrasts do not establish an internal substitution mechanism or deceptive intent.",
            "Explicitly marking missing evidence is a diagnostic condition, not a selected new reader policy.",
        ],
    }


def run(args, backend_factory=HFBackend):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Preserve this fixed diagnostic attempt")
    panel = load_panel(args.panel)
    planned = jobs(panel)
    settings = load_config(ROOT / "configs/qwen7b_reader_4bit.json").model
    policy = policy_metadata()
    if (
        settings.model_id != policy["model_id"]
        or settings.revision != policy["model_revision"]
        or settings.quantization != "4bit"
        or settings.dtype != "bfloat16"
        or not settings.local_files_only
    ):
        raise ValueError("Reader checkpoint, precision and offline profile must remain unchanged")
    paths = [
        Path(__file__),
        Path(args.panel),
        ROOT / "src/latent_probes/evidence_role_panel.py",
        ROOT / "src/latent_probes/calibrated_reader.py",
        ROOT / "src/latent_probes/forecast_panel.py",
        ROOT / "src/latent_probes/reader_panel.py",
        ROOT / "src/latent_probes/backends.py",
        ROOT / "configs/qwen7b_reader_4bit.json",
        ROOT / "docs/EVIDENCE_ROLE_DIAGNOSTIC_PROTOCOL.md",
    ]
    hashes = {str(p.resolve()): file_sha256(p) for p in paths}
    output.mkdir(parents=True)
    protocol = {
        "schema": "evidence_role_diagnostic_protocol_v1",
        "created_utc": utc_now(),
        "policy": policy,
        "model": settings.__dict__,
        "generation": SETTINGS.__dict__,
        "seed": SEED,
        "jobs": planned,
        "source_sha256": hashes,
        "scope": "One bounded reader-failure diagnosis. Does not changequalification, fit probes or open ct cases.",
    }
    atomic_json(output / "protocol.json", protocol)
    manifest = {
        "status": "prepared",
        "environment": environment_info(),
        "protocol_sha256": file_sha256(output / "protocol.json"),
    }
    atomic_json(output / "manifest.json", manifest)
    backend = None
    try:
        backend = backend_factory(settings)
        if (
            backend.metadata.get("is_test_fixture") is not False
            or backend.metadata["resolved_revision"] != settings.revision
        ):
            raise ValueError("Real exact-checkpoint reader required")
        manifest.update(status="running", backend=backend.metadata, started_utc=utc_now())
        atomic_json(output / "manifest.json", manifest)
        lengths = []
        for job in planned:
            encoded = backend.tokenizer.apply_chat_template(
                job["messages"],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            n = int(encoded["input_ids"].shape[1])
            lengths.append(
                {
                    "example_id": job["example_id"],
                    "prompt_tokens": n,
                    "prompt_token_ids": encoded["input_ids"][0].tolist(),
                    "prompt_sha256": canonical_hash(job["messages"]),
                }
            )
            if n > settings.max_input_tokens:
                raise ValueError(
                    "A full diagnostic prompt exceeds the limit; nothing may be truncated"
                )
        atomic_json(output / "prompt_lengths.json", lengths)
        if any(file_sha256(Path(p)) != h for p, h in hashes.items()):
            raise ValueError("Frozen diagnostic inputs changed before generation")
        rows = []
        for i, job in enumerate(planned, 1):
            generation = backend.generate(job["messages"], SETTINGS, SEED).record()
            if generation["prompt_token_ids"] != lengths[i - 1]["prompt_token_ids"]:
                raise ValueError(
                    "Generated diagnostic prompt does not match its frozen preflight IDs"
                )
            row = {**job, "generation": generation, "decision": strict_decision(generation)}
            atomic_json(output / "records" / f"{job['example_id']}.json", row)
            rows.append(row)
            if i % 16 == 0:
                print(f"Fixed-policy evidence-role diagnostic {i}/{len(planned)}", flush=True)
        atomic_json(output / "rows.json", rows)
        summary = summarise(rows)
        atomic_json(output / "summary.json", summary)
        if any(file_sha256(Path(p)) != h for p, h in hashes.items()):
            raise ValueError("Frozen diagnostic inputs changed")
        manifest.update(
            status="complete",
            completed_utc=utc_now(),
            summary_sha256=file_sha256(output / "summary.json"),
            rows_sha256=file_sha256(output / "rows.json"),
            n_calls=len(rows),
        )
        print(summary["groups"], flush=True)
    except BaseException:
        manifest.update(status="failed", failure=traceback.format_exc())
        raise
    finally:
        if backend is not None:
            backend.close()
        manifest.update(
            finished_utc=utc_now(),
            source_unchanged=all(file_sha256(Path(p)) == h for p, h in hashes.items()),
        )
        atomic_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel", type=Path, default=ROOT / "data/communication/evidence_role_v1.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/adirj/kcl-latent-probes-runs/evidence_role_diagnostic_v1"),
    )
    run(parser.parse_args())
