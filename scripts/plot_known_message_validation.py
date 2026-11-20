"""Rebuild the known-message figure from committed measured aggregate evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def plot_record(record_path, output_dir):
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        if (output_dir / f"known_message_validation.{suffix}").exists():
            raise FileExistsError(
                "Choose fresh figure outputs; preserved plots are not overwritten"
            )
    conditions = ("story", "nested_question", "world_question")
    values = [100 * record["primary_sender"][key]["accuracy"] for key in conditions]
    counts = [record["primary_sender"][key]["correct"] for key in conditions]
    values += [
        100 * record["behavior"][style]["accuracy_including_invalid"]
        for style in ("direct", "structured")
    ]
    counts += [record["behavior"][style]["correct"] for style in ("direct", "structured")]
    labels = [
        "Story-only\nprobe",
        "Belief query\nprobe",
        "World query\nprobe",
        "Direct\nanswer",
        "Structured\nanswer",
    ]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), layout="constrained")
        colors = ["#8394a5", "#2065a8", "#8394a5", "#da8b2a", "#da8b2a"]
        bars = axes[0].bar(range(5), values, color=colors, width=0.65)
        axes[0].set_xticks(range(5), labels)
        axes[0].set_ylim(0, 106)
        axes[0].set_ylabel("Correct (%) including invalid answers")
        axes[0].axhline(50, color="black", linestyle="--", linewidth=1)
        axes[0].bar_label(bars, labels=[f"{n}/64" for n in counts], padding=4)
        axes[0].set_title("A. Frozen primary block-16 heads and model answers")
        groups = sorted(record["per_template_primary_sender"])
        x = np.arange(len(groups))
        label_levels = {}
        for offset, (key, label, color) in enumerate(
            (
                ("nested_question_correct", "Frozen queried probe", "#2065a8"),
                ("direct", "Direct answer", "#da8b2a"),
                ("structured", "Structured answer", "#b8a38a"),
            )
        ):
            correct = [
                record["per_template_primary_sender"][g][key]
                if offset == 0
                else record["behavior"][key]["per_template"][g]["correct"]
                for g in groups
            ]
            bars = axes[1].bar(
                x + (offset - 1) * 0.23,
                100 * np.array(correct) / 16,
                width=0.22,
                label=label,
                color=color,
            )
            for bar, count, group in zip(bars, correct, groups, strict=True):
                level = label_levels.get((group, count), 0)
                label_levels[(group, count)] = level + 1
                axes[1].annotate(
                    f"{count}/16",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3 + 12 * level),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
        axes[1].set_xticks(x, [g.replace("km_t", "Group ") for g in groups])
        axes[1].set_ylim(0, 112)
        axes[1].axhline(50, color="black", linestyle="--", linewidth=1)
        axes[1].set_title("B. Results by wording group")
        axes[1].legend(fontsize=8, loc="lower left")
        fig.suptitle("Known-message validation: question-conditioned transfer without retraining")
        fig.supxlabel(
            "Four wordings of one logical family; 64 rows = 32 unique sender inputs. "
            "Agent-assisted fictional labels; human review pending.",
            fontsize=9,
        )
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(output_dir / f"known_message_validation.{suffix}", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_record(args.record, args.output)
