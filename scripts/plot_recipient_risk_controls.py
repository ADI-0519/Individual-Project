"""Render the measured failed v2 qualification without model calls or new labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot(analysis_path, output):
    analysis_path, output = Path(analysis_path), Path(output)
    raw = analysis_path.read_bytes()
    analysis = json.loads(raw)
    if (
        analysis["attempt_status"] != "qualification_failed"
        or analysis["is_test_fixture"]
        or analysis["counts"]["recipient_conditions"] != 72
        or analysis["counts"]["reports"] != 0
        or analysis["counts"]["generation_attempts"] != {"actor": 0, "reader": 72}
    ):
        raise ValueError("This figure requires the genuine completed v2 qualification failure")
    groups = analysis["controls"]["groups"]
    keys = ["no_report/informed", "no_report/absent", "no_report/stale"]
    correct = [groups[k]["normative_correct"] for k in keys]
    total = [groups[k]["n_conditions"] for k in keys]
    correct.append(analysis["qualification"]["full_correct"])
    total.append(36)
    before = groups["no_report/informed"]["outcome_counts"]
    after = groups["full_disclosure/informed"]["outcome_counts"]
    if before != {"world_correct": 12, "world_wrong_definite": 0, "defer": 0, "invalid": 0}:
        raise ValueError("The informed paired contrast requires all twelve initial answers correct")
    if sum(after.values()) != 12 or after["invalid"]:
        raise ValueError("Unexpected informed full-disclosure dispositions")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "svg.hashsalt": "recipient-risk-v2-controls",
        }
    )
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11.8, 5.5), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    fig.subplots_adjust(left=0.20, right=0.98, top=0.74, bottom=0.27, wspace=0.43)
    fig.text(
        0.035,
        0.93,
        "Reader qualification failed before natural reporting",
        fontsize=17,
        weight="bold",
        color="#172838",
    )
    fig.text(
        0.035,
        0.875,
        "Qwen2.5-7B Instruct · NF4 weights / BF16 compute · fixed nine-example policy",
        fontsize=10.5,
        color="#526271",
    )
    fig.text(
        0.035,
        0.83,
        "12 fresh cases · 3 predicate types · 72 unique calls · no natural reports or new fits",
        fontsize=10.5,
        color="#526271",
    )
    labels = [
        "No report: informed",
        "No report: absent",
        "No report: stale",
        "Full source: all packets",
    ]
    for y, (n, d) in enumerate(zip(correct, total, strict=True)):
        left.barh(y, 100, height=0.5, color="#E7ECF1")
        left.barh(y, n / d * 100, height=0.5, color="#326B91")
        left.text(103, y, f"{n}/{d}", va="center", fontsize=11, weight="bold")
    left.set(
        yticks=range(4),
        yticklabels=labels,
        xlim=(0, 117),
        xticks=[0, 25, 50, 75, 100],
        xlabel="Matches specified control answer (%)",
    )
    left.invert_yaxis()
    left.set_title("A  Control correctness", loc="left", weight="bold", pad=17)
    left.tick_params(axis="y", length=0)
    left.spines["bottom"].set_color("#CDD5DE")
    colors = ["#287D68", "#8A929E", "#BD4F54"]
    categories = ["world_correct", "defer", "world_wrong_definite"]
    for y, values in enumerate((before, after)):
        offset = 0
        for category, color in zip(categories, colors, strict=True):
            n = values[category]
            if n:
                right.barh(y, n, left=offset, height=0.48, color=color)
                right.text(
                    offset + n / 2,
                    y,
                    str(n),
                    color="white",
                    weight="bold",
                    ha="center",
                    va="center",
                )
            offset += n
    right.set(
        yticks=[0, 1],
        yticklabels=["No report", "Full source"],
        xlim=(0, 12),
        xticks=[0, 3, 6, 9, 12],
        xlabel="Same twelve informed cases",
    )
    right.invert_yaxis()
    right.set_title("B  Adding the full source", loc="left", weight="bold", pad=17)
    right.tick_params(axis="y", length=0)
    right.spines["bottom"].set_color("#CDD5DE")
    right.legend(
        handles=[
            Patch(color=c, label=t)
            for c, t in zip(colors, ["World-correct", "DEFER", "World-wrong"], strict=True)
        ],
        loc="lower center",
        bbox_to_anchor=(0.42, -0.37),
        ncols=3,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.035,
        0.115,
        "No-report correctness follows supplied evidence; full-source correctness follows the complete-source rule result.",
        fontsize=9,
        color="#526271",
    )
    fig.text(
        0.035,
        0.075,
        "Full source adds older records, optional facts and descriptions together; no individual cause or precision effect was isolated.",
        fontsize=9,
        color="#526271",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fixed_date = datetime(2026, 11, 21, tzinfo=UTC)
    for suffix in ("png", "pdf", "svg"):
        path = output.with_suffix("." + suffix)
        metadata = (
            {"CreationDate": fixed_date}
            if suffix == "pdf"
            else {"Date": fixed_date.isoformat()}
            if suffix == "svg"
            else {}
        )
        fig.savefig(path, dpi=180, facecolor="white", metadata=metadata)
        if suffix == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
    plt.close(fig)
    record = {
        "schema": "recipient_risk_control_figure_v2",
        "analysis_sha256": hashlib.sha256(raw).hexdigest(),
        "control_correct": correct,
        "control_denominators": total,
        "informed_before": before,
        "informed_after": after,
        "human_review": False,
        "new_model_calls": 0,
        "new_probe_fits": 0,
    }
    output.with_suffix(".json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="Output stem without a file extension"
    )
    args = parser.parse_args()
    plot(args.analysis, args.output)
