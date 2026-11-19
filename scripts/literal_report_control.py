"""Reproduce the exploratory last-decision-word control without loading a model."""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from latent_probes.storage import atomic_json, file_sha256, read_json

CLASSES = ("APPROVE", "REJECT", "DEFER")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh exploratory-control output.")
    labels_path = args.run / "neutral_dataset/labels.csv"
    with labels_path.open(newline="", encoding="utf-8") as handle:
        counts = Counter(
            row["decision"]
            for row in csv.DictReader(handle)
            if row["split"] == "train"
            and row["objective"] == "neutral"
            and row["decision"] in CLASSES
        )
    if not counts:
        raise ValueError("A valid neutral-training majority is required.")
    fallback = max(CLASSES, key=lambda label: counts[label])
    files = [(path, read_json(path)) for path in sorted((args.run / "reports").glob("*.json"))]
    files = [(path, row) for path, row in files if row["split"] == "test"]
    predictions = []
    for path, row in files:
        mentions = re.findall(r"\b(APPROVE|REJECT|DEFER)\b", row["generation"]["text"])
        predictions.append(
            {
                "example_id": row["example_id"],
                "case_id": row["case_id"],
                "objective": row["objective"],
                "prediction": mentions[-1] if mentions else fallback,
                "explicit_decision_mention": bool(mentions),
                "observed_reader_decision": row["decision"],
                "source_sha256": file_sha256(path),
            }
        )
    evaluations = {}
    for objective in sorted({row["objective"] for row in predictions}):
        rows = [row for row in predictions if row["objective"] == objective]
        valid = [row for row in rows if row["observed_reader_decision"] in CLASSES]
        per_class = []
        for label in CLASSES:
            tp = sum(row["prediction"] == label == row["observed_reader_decision"] for row in valid)
            fp = sum(row["prediction"] == label != row["observed_reader_decision"] for row in valid)
            fn = sum(row["prediction"] != label == row["observed_reader_decision"] for row in valid)
            per_class.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        evaluations[objective + "_test"] = {
            "n": len(rows),
            "n_valid_targets": len(valid),
            "n_case_clusters": len({r["case_id"] for r in rows}),
            "accuracy": sum(row["prediction"] == row["observed_reader_decision"] for row in valid)
            / len(valid)
            if valid
            else None,
            "macro_f1": sum(per_class) / 3 if valid else None,
            "explicit_decision_coverage": sum(row["explicit_decision_mention"] for row in valid),
        }
    output = {
        "schema": "posthoc_literal_report_control_v1",
        "selected_after_original_pilot": True,
        "rule": "last uppercase decision word in report; otherwise neutral-training majority",
        "fallback": fallback,
        "training_counts": dict(counts),
        "training_labels_sha256": file_sha256(labels_path),
        "script_sha256": file_sha256(__file__),
        "evaluations": evaluations,
        "predictions": predictions,
        "limitation": "Exploratory development shortcut control; not a prospectively selected primary comparator.",
    }
    atomic_json(args.output, output)
    print(json.dumps(evaluations, indent=2))


if __name__ == "__main__":
    main()
