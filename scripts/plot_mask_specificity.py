"""Plot measured draft MASK scores without fitting, selection or model calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

NAMES = {
    "677ca1661956d330887d379d": "Fyre Festival",
    "677d5546c1f8425ef003a7b7": "Volkswagen",
    "677d553cb6682dba851f3062": "Wells Fargo",
    "677ca1654b96f907dbf5b73b": "Armstrong",
    "677d5539500c5a8f11b186d5": "Moon landings",
    "6783135070bdc0d2bfe05042": "Evergreen",
    "677ca163422316de28d82f8d": "2019 NBA Finals",
    "b4e16587-3d4f-4eef-938b-d694bd8b42d3_err": "2017 Oscars",
}


def plot_results(score_dir, same_text_dir, output_dir, record_path=None):
    output_dir = Path(output_dir)
    if record_path:
        record = json.loads(Path(record_path).read_text(encoding="utf-8"))
        scores = {"rows": record["task_scores"]}
        pair_plan = {"threshold": record["reference_threshold"]["q90"]}
        pairs = []  # The prespecified strict replay had no eligible sources.
        n_input_cases = record["capture"]["n_cases"]
        n_belief_cases = record["strict_primary"]["eligible_belief_cases"]
    else:
        score_dir, same_text_dir = map(Path, (score_dir, same_text_dir))
        scores = json.loads((score_dir / "draft_scores.json").read_text(encoding="utf-8"))
        pair_plan = json.loads((same_text_dir / "protocol.json").read_text(encoding="utf-8"))
        pairs = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((same_text_dir / "cases").glob("*.json"))
        ]
        n_input_cases = len({row["case_id"] for row in scores["rows"]})
        beliefs = json.loads((score_dir / "draft_operational_beliefs.json").read_text())
        n_belief_cases = sum(c["operational_belief"] is not None for c in beliefs["cases"].values())
    threshold = pair_plan["threshold"]["threshold"]
    rows = scores["rows"]
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        if (output_dir / f"mask_specificity.{suffix}").exists():
            raise FileExistsError("Choose fresh figure outputs; existing results stay preserved")
    colors = dict(zip(NAMES, plt.get_cmap("tab10").colors, strict=False))
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.7), layout="constrained")
        groups = (
            ("ablated", "Incentive-ablated"),
            ("pressure", "Pressured"),
        )
        labels = []
        for column, (condition, label) in enumerate(groups):
            selected = [row for row in rows if row["condition"] == condition]
            offsets = np.linspace(-0.14, 0.14, len(selected))
            for row, offset in zip(selected, offsets, strict=True):
                axes[0].scatter(
                    column + offset,
                    row["mean_raw_margin"],
                    color=colors[row["case_id"]],
                    s=35,
                    alpha=0.85,
                    marker="x" if row["status"] == "truncated" else "o",
                )
            labels.append(f"{label}\n(n={len(selected)})")
        axes[0].set_xticks(range(len(groups)), labels)
        axes[0].set_title("A. All task responses (not audited honesty classes)")
        axes[0].set_ylabel("Frozen Apollo mean raw margin")
        axes[0].axhline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1,
            label="Frozen neutral-reference q90 (N=32)",
        )
        axes[0].legend(loc="upper left", fontsize=8)
        for case in pairs:
            case_id = case["case_id"]
            axes[1].plot(
                [0, 1],
                [case["contexts"][key]["mean_raw_margin"] for key in ("ablated", "pressure")],
                "o-",
                color=colors[case_id],
                label=NAMES[case_id],
                alpha=0.85,
            )
        if pairs:
            axes[1].set_xticks([0, 1], ["Incentive-ablated context", "Pressure context"])
            axes[1].set_title("B. Identical response token IDs replayed")
            axes[1].set_ylabel("Frozen Apollo mean raw margin")
            axes[1].axhline(threshold, color="black", linestyle="--", linewidth=1)
            axes[1].legend(fontsize=8, loc="best")
        else:
            bars = axes[1].bar(
                [0, 1, 2],
                [n_input_cases, n_belief_cases, 0],
                color=["#8394a5", "#2065a8", "#da8b2a"],
                width=0.6,
            )
            axes[1].bar_label(bars, padding=4)
            axes[1].set_xticks(
                [0, 1, 2],
                ["Selected\ncases", "Five-check\nbelief cases", "Clean ablated\nsource cases"],
            )
            axes[1].set_ylim(0, max(9, n_input_cases + 1))
            axes[1].set_ylabel("Cases passing the strict primary gates")
            axes[1].set_title("B. Primary contrast is not estimable")
            axes[1].text(
                0.97,
                0.97,
                "No eligible source for the prespecified replay.\n"
                "0 counterfactual forwards; no replay estimate.",
                ha="right",
                va="top",
                transform=axes[1].transAxes,
                fontsize=9,
            )
        fig.suptitle("MASK specificity development pilot — agent-assisted draft labels")
        fig.supxlabel(
            "Eight purposive cases; repeated responses are clustered. "
            "Crosses mark capped responses. The cutoff is not a validated honest-response FPR.",
            fontsize=9,
        )
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(output_dir / f"mask_specificity.{suffix}", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--same-text", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.record and not (args.scores and args.same_text):
        parser.error("Provide --record, or both --scores and --same-text")
    plot_results(args.scores, args.same_text, args.output, args.record)
