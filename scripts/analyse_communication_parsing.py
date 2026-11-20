"""Post-hoc punctuation-only sensitivity; never overwrite frozen strict results."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from latent_probes.storage import atomic_json, file_sha256, read_json, utc_now


def punctuation_tolerant_final(generation):
    """Allow one terminal punctuation character on a unique exact FINAL line."""
    if generation["finish_reason"] != "eos":
        return "INVALID"
    matches = re.findall(r"(?m)^FINAL:\s*(APPROVE|REJECT|DEFER)[.!?]?\s*$", generation["text"])
    return matches[0] if len(matches) == 1 else "INVALID"


def analyse(primary, capability, output):
    if output.exists():
        raise FileExistsError("Keep all parsing sensitivity records")
    reports = [(p, read_json(p)) for p in sorted((primary / "reports").glob("*.json"))]
    diagnostic = [(p, read_json(p)) for p in sorted((capability / "records").glob("*.json"))]
    rows = []
    for path, r in reports:
        rows.append(
            {
                "kind": "forecaster",
                "id": r["example_id"],
                "case_id": r["case_id"],
                "objective": r["objective"],
                "context": r["context"],
                "target": r["decision"],
                "strict": r["predictions"]["full_context_forecaster"],
                "secondary": punctuation_tolerant_final(r["monitor_generation"]),
                "source_sha256": file_sha256(path),
            }
        )
    for path, r in diagnostic:
        if r["condition"] == "reasoned_192":
            rows.append(
                {
                    "kind": "reasoned_control",
                    "id": path.name,
                    "case_id": r["case_id"],
                    "context": r["context"],
                    "target": r["expected_decision"],
                    "strict": r["prediction"],
                    "secondary": punctuation_tolerant_final(r["generation"]),
                    "source_sha256": file_sha256(path),
                }
            )
    summary = {}
    for kind in ("forecaster", "reasoned_control"):
        field = "objective" if kind == "forecaster" else "context"
        for group in sorted({r[field] for r in rows if r["kind"] == kind}):
            selected = [
                r
                for r in rows
                if r["kind"] == kind and r[field] == group and r["target"] != "INVALID"
            ]
            summary[f"{kind}/{group}"] = {
                "n_valid_targets": len(selected),
                "strict_correct": sum(r["strict"] == r["target"] for r in selected),
                "secondary_correct": sum(r["secondary"] == r["target"] for r in selected),
                "strict_invalid": sum(r["strict"] == "INVALID" for r in selected),
                "secondary_invalid": sum(r["secondary"] == "INVALID" for r in selected),
            }
    atomic_json(
        output,
        {
            "schema": "posthoc_final_punctuation_sensitivity_v1",
            "created_utc": utc_now(),
            "posthoc": True,
            "new_model_calls": 0,
            "primary_unchanged": True,
            "rule": "Unique exact FINAL line with at most one optional terminal .!?; EOS required; no rescue of caps/ambiguous lines",
            "script_sha256": file_sha256(__file__),
            "summary": summary,
            "rows": rows,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--capability", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyse(args.primary, args.capability, args.output)
