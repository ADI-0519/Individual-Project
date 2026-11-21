"""Rebuild a static research figure from the byte-verified crossed-assay record."""

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
    parser.add_argument("--output", type=Path, required=True, help="Filename stem")
    args = parser.parse_args()
    record = json.loads(args.record.read_text(encoding="utf-8"))
    if record["status"] != "complete_fresh_development_validation":
        raise ValueError("Need the complete verified development assay")
    query, story = (record["conditions"][c] for c in ("nested_question", "story"))
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "crossed-belief-validation-v1",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    colors = ["#56468d", "#9b91b5", "#217f78", "#9299a5"]
    for ax, subset, title in zip(
        axes[:2],
        ("primary", "invariant"),
        ("Initial belief differs from message", "Initial belief agrees with message"),
        strict=True,
    ):
        q, s = query[subset], story[subset]
        values = [
            q["probe_correct"],
            s["probe_correct"],
            q["output_fixed_correct"],
            q["answer_correct"],
        ]
        bars = ax.bar(np.arange(4), np.asarray(values) / 64, color=colors)
        ax.bar_label(bars, labels=[f"{v}/64" for v in values], fontsize=8, padding=3)
        ax.set(
            xticks=np.arange(4),
            xticklabels=["Queried\nprobe", "Before\nquestion", "Output\nYES/NO", "Greedy\nanswer"],
            title=title,
            ylim=(0, 1.13),
        )
        ax.axhline(0.5, color="#6c737c", ls="--", lw=0.8)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel("Fixed-decision accuracy")
        ax.grid(axis="y", color="#e5e7eb", lw=0.6)
        ax.set_axisbelow(True)
    x, width = np.arange(2), 0.23
    for shift, values, label, color in [
        (
            -width,
            [query[k]["probe_auroc"] for k in ("primary", "invariant")],
            "Queried probe",
            colors[0],
        ),
        (
            0,
            [story[k]["probe_auroc"] for k in ("primary", "invariant")],
            "Before question",
            colors[1],
        ),
        (
            width,
            [query[k]["output_score_auroc"] for k in ("primary", "invariant")],
            "Output YES-NO score",
            colors[2],
        ),
    ]:
        bars = axes[2].bar(x + shift, values, width, color=color, label=label)
        axes[2].bar_label(bars, labels=[f"{v:.3f}" for v in values], fontsize=7, padding=3)
    axes[2].set(
        xticks=x,
        xticklabels=["Primary\nI differs from M", "Controls\nI equals M"],
        title="Ranking at frozen score orientation",
        ylim=(0, 1.13),
        ylabel="AUROC",
    )
    axes[2].axhline(0.5, color="#6c737c", ls="--", lw=0.8)
    axes[2].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[2].legend(fontsize=7, loc="upper left", frameon=True)
    axes[2].grid(axis="y", color="#e5e7eb", lw=0.6)
    axes[2].set_axisbelow(True)
    fig.suptitle(
        "Does a frozen belief readout survive independently varied initial beliefs?", fontsize=13
    )
    fig.text(
        0.5,
        0.015,
        "Llama 3.1 8B; NF4 / BF16 compute; no new fits, layers or thresholds. Each subset: 64 inputs, 32 mapping-paired cells, four templates.\n"
        "Initial-only/message-only baselines: 50% primary, 100% controls. One authored update-rule family; human semantic review pending.\n"
        "Before-question capture still contains a candidate proposition and assessment framing; this is not natural spontaneous anticipation.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.93))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix, metadata in (
        ("png", {}),
        ("pdf", {"CreationDate": None, "ModDate": None}),
        ("svg", {"Date": None}),
    ):
        fig.savefig(args.output.with_suffix("." + suffix), dpi=220, metadata=metadata)
    svg = args.output.with_suffix(".svg")
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    plt.close(fig)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "record_sha256": hashlib.sha256(args.record.read_bytes()).hexdigest(),
                "execution_commit": record["execution_commit"],
                "primary": query["primary"],
                "invariant": query["invariant"],
                "probe_qualification": query["probe_qualification"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
