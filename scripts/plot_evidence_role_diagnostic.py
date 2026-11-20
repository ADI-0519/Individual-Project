"""Scientific figures from completed evidence-role records, without model calls.

The primary record must contain summary and provenance.model. An optional completed
replication uses the same fields; no replication values or model identity are
invented. Three columns separate decisive controls, warranted missing-evidence
deferrals, and matched low/high definite-answer switches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD = ROOT / "docs/validation/2026-11-20-evidence-role-diagnostic.json"
DEFAULT_OUTPUT = ROOT / "docs/figures/2026-11-20"
COLORS = {"metadata": "#657786", "explicit_missing": "#007F7B"}
LABELS = {
    "metadata": "Administrative value only",
    "explicit_missing": "Plus explicit missing attribute",
}
FAMILIES = ("date", "score")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_record(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    summary = data.get("summary")
    model = data.get("provenance", {}).get("model", data.get("model"))
    if not isinstance(summary, dict) or not isinstance(model, dict) or not model.get("model_id"):
        raise ValueError(
            "Each completed record requires summary and actual provenance.model.model_id"
        )
    if summary.get("n_calls") != 96 or summary.get("n_cases") != 16:
        raise ValueError("Expected the frozen 96 conditions in 16 case clusters")
    if summary.get("qualifies_reader") is not False:
        raise ValueError("This figure must not relabel a diagnostic as reader qualification")
    for family in FAMILIES:
        paired = summary["paired"][family]
        if paired["n_cases"] != 8 or len(paired["cases"]) != 8:
            raise ValueError("Each grammar requires eight paired case clusters")
        for condition in ("decisive", "metadata", "explicit_missing"):
            group = summary["groups"][family + "/" + condition]
            if group["n"] != 16 or not 0 <= group["correct"] <= 16:
                raise ValueError("Condition denominators must remain 16")
            if not 0 <= group["deferrals"] <= 16 or not 0 <= group["invalid"] <= 16:
                raise ValueError("Invalid deferral or invalid-output count")
            if sum(group["decision_counts"].values()) != 16:
                raise ValueError("Decision counts do not match the condition denominator")
        for condition in ("metadata", "explicit_missing"):
            switches = sum(case["pairs"][condition]["definite_flip"] for case in paired["cases"])
            if not 0 <= switches <= 8:
                raise ValueError("Invalid paired-switch count")
    return data, summary, model


def _model_label(model):
    name = model["model_id"].split("/")[-1]
    quantization = model.get("quantization")
    dtype = model.get("dtype", model.get("compute_dtype", ""))
    # The frozen HFBackend implementation's 4bit branch is explicitly NF4.
    if quantization == "4bit" and dtype == "bfloat16":
        return name + "  |  NF4 weights, BF16 compute"
    declared = ", ".join(str(value) for value in (quantization, dtype) if value)
    return name + ("  |  " + declared if declared else "")


def _style(axis):
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D6DFE3", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#9CAEB6")
    axis.set_xticks([0, 1], ["Date grammar", "Score grammar"])
    axis.tick_params(labelsize=10, length=0, pad=6)
    axis.set_xlim(-0.65, 1.65)


def _count_labels(axis, bars, counts, denominator, *, as_fraction=False):
    for bar, count in zip(bars, counts, strict=True):
        height = count / denominator if as_fraction else count
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            height + (0.025 if as_fraction else denominator * 0.025),
            f"{count}/{denominator}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#16323E",
        )


def plot_records(record_path, output_dir, replication_path=None):
    paths = [Path(record_path)] + ([Path(replication_path)] if replication_path else [])
    records = [load_record(path) for path in paths]
    if len({model["model_id"] for _, _, model in records}) != len(records):
        raise ValueError("An optional replication must identify a different actual checkpoint")
    source_hashes = {str(path): _sha256(path) for path in paths}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.hashsalt": "evidence-role-diagnostic-v1",
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        len(records), 3, figsize=(12.9, 4.55 if len(records) == 1 else 7.6), squeeze=False
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        top=0.72 if len(records) == 1 else 0.82,
        bottom=0.21 if len(records) == 1 else 0.14,
        wspace=0.30,
        hspace=0.85,
    )
    figure.suptitle(
        "Fixed-policy evidence-role diagnostic",
        x=0.065,
        y=0.97,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color="#16323E",
    )
    for row_index, (_, summary, model) in enumerate(records):
        control_axis, deferral_axis, switch_axis = axes[row_index]
        for axis in axes[row_index]:
            _style(axis)
        figure.text(
            0.065,
            control_axis.get_position().y1 + (0.105 if len(records) == 1 else 0.075),
            _model_label(model),
            ha="left",
            va="bottom",
            fontsize=11.5,
            fontweight="bold",
            color="#16323E",
        )
        control_axis.set_title("A  Decisive-evidence accuracy", loc="left", pad=20, fontsize=12)
        control_counts = [summary["groups"][family + "/decisive"]["correct"] for family in FAMILIES]
        bars = control_axis.bar(
            [0, 1], np.asarray(control_counts) / 16, color="#247A94", width=0.48
        )
        _count_labels(control_axis, bars, control_counts, 16, as_fraction=True)
        control_axis.set_ylim(0, 1.18)
        control_axis.set_yticks([0, 0.5, 1])
        control_axis.yaxis.set_major_formatter(PercentFormatter(1))
        control_axis.set_ylabel("Normatively correct responses")

        deferral_axis.set_title("B  Missing-evidence deferrals", loc="left", pad=20, fontsize=12)
        switch_axis.set_title(
            "C  Low/high APPROVE ↔ REJECT switches", loc="left", pad=20, fontsize=11.5
        )
        for condition, offset in (("metadata", -0.18), ("explicit_missing", 0.18)):
            counts = [
                summary["groups"][family + "/" + condition]["deferrals"] for family in FAMILIES
            ]
            bars = deferral_axis.bar(
                np.arange(2) + offset,
                counts,
                width=0.32,
                color=COLORS[condition],
                label=LABELS[condition],
            )
            _count_labels(deferral_axis, bars, counts, 16)
            switches = [
                sum(
                    case["pairs"][condition]["definite_flip"]
                    for case in summary["paired"][family]["cases"]
                )
                for family in FAMILIES
            ]
            bars = switch_axis.bar(
                np.arange(2) + offset, switches, width=0.32, color=COLORS[condition]
            )
            _count_labels(switch_axis, bars, switches, 8)
        deferral_axis.set_ylim(0, 19)
        deferral_axis.set_yticks([0, 8, 16])
        deferral_axis.set_ylabel("DEFER answers (of 16)")
        switch_axis.set_ylim(0, 9.5)
        switch_axis.set_yticks([0, 4, 8])
        switch_axis.set_ylabel("Switching pairs (of 8)")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.075 if len(records) == 1 else 0.065),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    figure.text(
        0.065,
        0.025,
        "Each model: 96 deterministic conditions; 16 case clusters in two authored grammars. Both missing-evidence conditions warrant DEFER.\n"
        "Post-failure diagnosis; no new reader qualification or probe fit. Counts do not establish an internal mechanism or population precision.",
        ha="left",
        va="bottom",
        fontsize=8.8,
        color="#425D69",
        linespacing=1.35,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    description = "Completed records: " + json.dumps(source_hashes, sort_keys=True)
    outputs = {}
    for extension in ("png", "svg", "pdf"):
        destination = output / ("evidence_role_diagnostic." + extension)
        metadata = (
            {"Software": "KCL evidence-role figure", "Description": description}
            if extension == "png"
            else {"Creator": "KCL evidence-role figure", "Description": description, "Date": None}
            if extension == "svg"
            else {
                "Creator": "KCL evidence-role figure",
                "Subject": description,
                "CreationDate": None,
                "ModDate": None,
            }
        )
        figure.savefig(destination, dpi=220, facecolor="white", metadata=metadata)
        outputs[extension] = {"path": str(destination), "sha256": _sha256(destination)}
    plt.close(figure)
    return {
        "models": [model["model_id"] for _, _, model in records],
        "source_sha256": source_hashes,
        "outputs": outputs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--replication", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(plot_records(args.record, args.output_dir, args.replication), indent=2))


if __name__ == "__main__":
    main()
