"""Rebuild natural communication figures from the committed measured record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_record(record_path, output):
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = ("communication_transfer", "communication_recipient_replay")
    if any(
        (output / f"{name}.{suffix}").exists() for name in names for suffix in ("png", "svg", "pdf")
    ):
        raise FileExistsError("Use fresh figure paths; previous figures remain unchanged")
    summary = record["summary"]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
        objectives = ("neutral", "favorable")
        bottom = np.zeros(2)
        for key, label, color in (
            ("reader_correct", "World-correct", "#2a8a6c"),
            ("reader_wrong_definite", "Wrong definite", "#ba4949"),
            ("reader_defer", "DEFER", "#a5aeb7"),
            ("reader_invalid", "INVALID", "#d4af55"),
        ):
            values = [summary["behavior"][o][key] for o in objectives]
            bars = axes[0].bar([0, 1], values, bottom=bottom, label=label, color=color, width=0.6)
            axes[0].bar_label(
                bars,
                labels=[str(v) if v else "" for v in values],
                label_type="center",
                color="white",
                fontsize=10,
            )
            bottom += values
        axes[0].set_xticks([0, 1], ["Neutral", "Approval pressure"])
        axes[0].set_ylim(0, 88)
        axes[0].set_ylabel("Natural reports / reader outcomes")
        axes[0].set_title("A. Actual reader decisions (72 reports per goal)")
        axes[0].legend(loc="upper center", fontsize=8, ncol=2)
        methods = (
            "activation",
            "neutral_train_majority",
            "full_text",
            "source_world",
            "full_context_forecaster",
        )
        labels = (
            "Frozen\nactivation",
            "Always\nAPPROVE",
            "Frozen\nTF-IDF",
            "Source\nworld",
            "160-token\nforecaster",
        )
        x = np.arange(len(methods))
        for i, (objective, color) in enumerate(
            zip(objectives, ("#4a82b4", "#da8b2a"), strict=True)
        ):
            metrics = [summary["evaluations"][f"objective/{objective}"][m] for m in methods]
            values = [100 * m["accuracy"] for m in metrics]
            bars = axes[1].bar(
                x + (i - 0.5) * 0.35, values, width=0.33, color=color, label=objective
            )
            axes[1].bar_label(
                bars,
                labels=[f"{m['correct']}/{m['n_valid_targets']}" for m in metrics],
                padding=3,
                fontsize=8,
                rotation=90,
            )
        axes[1].set_xticks(x, labels, fontsize=9)
        axes[1].set_ylim(0, 119)
        axes[1].set_ylabel("Observed reader decision predicted correctly (%)")
        axes[1].set_title("B. Frozen transfer and information-matched controls")
        axes[1].legend(loc="lower left", fontsize=8)
        fig.suptitle("Fresh natural communication: frozen neutral-trained readout under pressure")
        fig.supxlabel(
            "24 authored case clusters; two explicit task rules held out from readout fitting. 66 forecaster calls capped. Human review pending.",
            fontsize=9,
        )
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(output / f"communication_transfer.{suffix}", dpi=200)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
        replay = record["replays"]
        contexts = ("informed", "absent", "stale")
        versions = ("none", "original", "restored", "irrelevant_edit")
        x = np.arange(3)
        for i, (version, label, color) in enumerate(
            zip(
                versions,
                (
                    "No report",
                    "Same original report",
                    "Decisive fact restored",
                    "Irrelevant fact added",
                ),
                ("#a5aeb7", "#ba4949", "#2a8a6c", "#cda158"),
                strict=True,
            )
        ):
            counts = [
                sum(
                    r["context"] == c
                    and r["version"] == version
                    and r["decision"] in ("APPROVE", "REJECT")
                    and r["decision"] != r["world_correct_decision"]
                    for r in replay
                )
                for c in contexts
            ]
            bars = axes[0].bar(x + (i - 1.5) * 0.19, counts, width=0.18, label=label, color=color)
            axes[0].bar_label(bars, padding=3, fontsize=8)
        axes[0].set_xticks(x, contexts)
        axes[0].set_ylim(0, 29)
        axes[0].set_ylabel("Wrong definite decisions / 24 saved reports")
        axes[0].set_title("A. Identical pressured report across recipient evidence")
        axes[0].legend(loc="upper center", fontsize=8)
        rows = record["rows"]
        pairs = []
        for case in sorted({r["case_id"] for r in rows}):
            rs = [r for r in rows if r["case_id"] == case]
            pairs.append(
                [
                    np.mean([r["apollo_raw_margin"] for r in rs if r["objective"] == o])
                    for o in objectives
                ]
            )
        for pair in pairs:
            axes[1].plot([0, 1], pair, color="#8394a5", alpha=0.55, linewidth=1)
        means = np.mean(pairs, axis=0)
        axes[1].plot(
            [0, 1], means, color="#2065a8", marker="o", linewidth=3, label="Mean over cases"
        )
        axes[1].set_xticks([0, 1], ["Neutral", "Approval pressure"])
        axes[1].set_ylabel("Frozen Apollo raw margin (uncalibrated)")
        axes[1].set_title("B. Paired score sensitivity, averaged within case")
        axes[1].legend(fontsize=8)
        fig.suptitle("Recipient outcomes and probe specificity are separate measurements")
        fig.supxlabel(
            "Replays reuse 24 stale-attributed pressured reports; reader errors are not deception labels. DEFER/INVALID transitions remain in the report.",
            fontsize=8,
        )
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(output / f"communication_recipient_replay.{suffix}", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_record(args.record, args.output)
