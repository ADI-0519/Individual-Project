"""Plot the already-computed posthoc correction-factor diagnostic, without fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary.get("analysis_scope") != "correction_all_addendum":
        raise ValueError("Require the separately sealed full-correction supplement")
    args.output.mkdir(parents=True, exist_ok=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "format-score-v1"})
    heads = ["I", "M1", "B1", "B2"]
    labels = [
        "Initial\nlocation",
        "First report\nlocation",
        "First report\nread",
        "Replacement\nread",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.1), sharey=True)
    x = np.arange(4)
    table = []
    for model, ax in zip(("llama", "qwen"), axes, strict=True):
        data = summary["results"][model]
        if len(data["contexts"]) != 24:
            raise ValueError("Unexpected context denominator")
        a, b, accuracy_b = [], [], []
        for head in heads:
            renderers = data["conditions"]["correction_all"][head]["renderers"]
            if any(renderers[r]["pooled"]["n"] != 1536 for r in ("A", "B")):
                raise ValueError("Unexpected factor denominator")
            ma, mb = (renderers[r]["pooled"] for r in ("A", "B"))
            a.append(ma["raw_auroc"])
            b.append(mb["raw_auroc"])
            accuracy_b.append(mb["accuracy_at_zero"])
            table.append(
                {
                    "model": model,
                    "factor": head,
                    "A_auroc": a[-1],
                    "B_auroc": b[-1],
                    "B_zero_accuracy": accuracy_b[-1],
                }
            )
        ax.bar(x - 0.16, a, 0.31, color="#296b98", label="A: raw AUROC")
        ax.bar(x + 0.16, b, 0.31, color="#d4912e", label="B: raw AUROC")
        ax.scatter(
            x + 0.16,
            accuracy_b,
            s=48,
            marker="D",
            facecolors="white",
            edgecolors="#171717",
            zorder=3,
            label="B: fixed-zero accuracy",
        )
        for index, value in enumerate(b):
            ax.text(index + 0.16, value + 0.017, f"{value:.3f}", ha="center", fontsize=9)
        ax.axhline(0.5, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_xticks(x, labels, fontsize=10)
        ax.set_ylim(0, 1.10)
        ax.set_title("Llama 3.1 8B" if model == "llama" else "Qwen 2.5 7B", fontsize=13)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Raw AUROC / fixed-zero accuracy")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.085),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "Format transfer preserves location ranking better than receipt ranking", fontsize=14
    )
    fig.text(
        0.5,
        0.018,
        "Posthoc exposed-bank diagnostic; 1,536 correction rows per format/model; "
        "24 shared contexts.\nFrozen factor heads and zero thresholds; descriptive values, no confidence intervals.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.19, 1, 0.94))
    for extension in ("pdf", "svg", "png"):
        metadata = (
            {"CreationDate": None}
            if extension == "pdf"
            else {"Date": None}
            if extension == "svg"
            else None
        )
        fig.savefig(args.output / ("factor_transfer." + extension), dpi=180, metadata=metadata)
    plt.close(fig)
    manifest = {
        "schema": "posthoc-factor-score-figure-v1",
        "new_model_calls": 0,
        "classifier_fits": 0,
        "source_sha256": digest(__file__),
        "summary_sha256": digest(args.summary),
        "plotted_values": table,
        "output_sha256": {p.name: digest(p) for p in sorted(args.output.iterdir())},
    }
    with (args.output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "complete", "output": str(args.output)}))


if __name__ == "__main__":
    main()
