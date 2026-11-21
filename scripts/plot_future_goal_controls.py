"""Plot the verified neutral results and explicitly posthoc output-score comparison."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output filename stem")
    args = parser.parse_args()
    data = json.loads(args.record.read_text(encoding="utf-8"))
    if data["status"] != "complete_posthoc_development" or data["old_neutral_gate_passed"]:
        raise ValueError("This figure is for the verified stopped-gate diagnostic")
    groups = [data["overall"], data["per_channel"]["observation"], data["per_channel"]["message"]]
    labels = [
        "All\n32 conditions\n16 scenes",
        "Observation\n16 conditions\n8 scenes",
        "Message / receipt\n16 conditions\n8 scenes",
    ]
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "future-goal-controls-v1",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5))
    x, width = np.arange(3), 0.34
    settings = [
        (
            [g["probe_correct"] / g["conditions"] for g in groups],
            [g["output_fixed_correct"] / g["conditions"] for g in groups],
            "Fixed-decision accuracy",
            "Greedy answer (all NO)",
            "#939aa6",
        ),
        (
            [g["probe_auroc"] for g in groups],
            [g["output_score_auroc"] for g in groups],
            "Ranking AUROC (output comparison: posthoc)",
            "Output YES−NO score",
            "#26877d",
        ),
    ]
    for panel, (ax, (probe, baseline, title, name, color)) in enumerate(
        zip(axes, settings, strict=True)
    ):
        bars = [
            ax.bar(x - width / 2, probe, width, color="#51478a", label="Frozen belief readout"),
            ax.bar(x + width / 2, baseline, width, color=color, label=name),
        ]
        for series, container in enumerate(bars):
            values = (
                [
                    f"{g['probe_correct'] if series == 0 else g['output_fixed_correct']}/{g['conditions']}"
                    for g in groups
                ]
                if panel == 0
                else [f"{bar.get_height():.4f}" for bar in container]
            )
            ax.bar_label(
                container,
                labels=values,
                padding=3,
                fontsize=8,
            )
        ax.axhline(0.5, ls="--", color="#69717a", lw=0.8)
        ax.set(ylim=(0, 1.08), xticks=x, xticklabels=labels, title=title)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e5e7eb", lw=0.6)
        ax.legend(loc="upper left", fontsize=8, frameon=True)
    fig.suptitle("Neutral belief readouts versus a constant NO answer", fontsize=13, y=0.97)
    fig.text(
        0.5,
        0.01,
        "Llama 3.1 8B · NF4 / BF16 compute · no refitting or threshold changes\nNeutral answer gate failed: 192 goal/archive conditions were not collected. Authored feasibility bank; human review pending.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.91))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix, metadata in [
        ("png", {}),
        ("pdf", {"CreationDate": None, "ModDate": None}),
        ("svg", {"Date": None}),
    ]:
        fig.savefig(args.output.with_suffix("." + suffix), dpi=220, metadata=metadata)
    plt.close(fig)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "record": str(args.record),
                "record_sha256": hashlib.sha256(args.record.read_bytes()).hexdigest(),
                "groups": groups,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
