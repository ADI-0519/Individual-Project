"""Plot the audited lexical control beside the earlier latent study.

This descriptive comparison does not equalize feature geometry or capacity and
does not test incremental activation value against a capable neural monitor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_probes.storage import atomic_json, file_sha256


def load_inputs(lexical, latent):
    lexical, latent = Path(lexical), Path(latent)
    compact = json.loads((lexical / "metrics.json").read_text(encoding="utf-8"))
    audit_path = lexical / "audit" / "audit_receipt.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest = json.loads((lexical / "manifest.json").read_text(encoding="utf-8"))
    previous = json.loads((latent / "summary.json").read_text(encoding="utf-8"))
    old_manifest = json.loads((latent / "manifest.json").read_text(encoding="utf-8"))
    if (
        audit.get("status") != "pass"
        or compact.get("human_review_complete") is not False
        or compact.get("research_status") != "posthoc_exposed_bank_diagnostic"
        or previous.get("test_fixture") is not False
    ):
        raise ValueError("Require audited exploratory lexical results")
    if manifest["output_sha256"]["metrics.json"] != file_sha256(lexical / "metrics.json"):
        raise ValueError("Changed lexical metrics")
    if manifest["output_sha256"]["audit/audit_receipt.json"] != file_sha256(audit_path):
        raise ValueError("Changed lexical audit receipt")
    if old_manifest["output_sha256"]["summary.json"] != file_sha256(latent / "summary.json"):
        raise ValueError("Changed latent metrics")
    series = []
    for key, name in (("llama", "Llama latent"), ("qwen", "Qwen latent")):
        series.append((name, previous["models"][key], False))
    series.append(("Shared lexical", compact["views"]["full_messages"], True))
    return series


def plot(lexical, latent, output):
    lexical, latent, output = Path(lexical), Path(latent), Path(output)
    series = load_inputs(lexical, latent)
    lexical_result = series[2][1]
    lexical_hard = lexical_result["arms"]["Mixed512"]["B/correction_hard"]
    name_pairs = lexical_hard["nuisance_pairs"]["name_map"]
    names = [output / ("lexical-comparison." + suffix) for suffix in ("pdf", "svg", "png")]
    receipt = output / "figure-manifest.json"
    if any(p.exists() for p in (*names, receipt)):
        raise ValueError("Figure exists; preserve prior exports")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    colors = ("#2764a1", "#237c61", "#d18529")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))
    arms = ("A512", "Mixed512", "Ahalf256", "Offset")
    x, width = np.arange(4), 0.24
    for axis, renderer in zip(axes[0], ("A", "B"), strict=True):
        for j, ((name, result, is_text), color) in enumerate(zip(series, colors, strict=True)):
            values = []
            for arm in arms:
                metric = result["arms"][arm][renderer + "/correction_hard"]
                metric = metric if is_text else metric["aggregate"]
                if metric["quartets"] != 192:
                    raise ValueError("Wrong hard panel denominator")
                values.append(metric["complete_quartets"])
            bars = axis.bar(x + (j - 1) * width, values, width, color=color, label=name)
            axis.bar_label(bars, fontsize=8, padding=2)
        axis.set_xticks(x, arms)
        axis.set_ylim(0, 208)
        axis.set_yticks([0, 48, 96, 144, 192])
        axis.set_ylabel("Complete hard panels / 192")
        axis.set_title(f"Format {renderer}: all four judgments correct")
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    for axis, factors, title in (
        (axes[1, 0], ("I", "M1"), "Mixed: location factors"),
        (axes[1, 1], ("B1", "B2"), "Mixed: receipt factors"),
    ):
        positions = np.arange(2)
        for j, ((name, result, is_text), color) in enumerate(zip(series, colors, strict=True)):
            metrics = result["factor_diagnostics"]["Mixed512"]["B/correction_all"]
            metrics = metrics if is_text else metrics["aggregate"]
            values = [metrics["factors"][factor]["raw_auroc"] for factor in factors]
            bars = axis.bar(positions + (j - 1) * width, values, width, color=color)
            axis.bar_label(bars, labels=[f"{v:.3f}" for v in values], fontsize=8, padding=2)
        axis.set_xticks(positions, factors)
        axis.set_ylim(0, 1.09)
        axis.axhline(0.5, color="#888888", linestyle=":", linewidth=1)
        axis.set_ylabel("Signed raw AUROC")
        axis.set_title(title + " (B; easy + hard)")
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle(
        "Queried belief readouts: format repair and a lexical control", fontsize=15, y=0.98
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.5,
        0.045,
        "Same 24 test contexts; four supervised factors + supplied rule. Text control is posthoc and shallow.\n"
        f"Lexical Mixed B-hard row accuracy: {lexical_hard['correct']}/{lexical_hard['rows']} "
        f"({100 * lexical_hard['row_accuracy']:.2f}%); name flips: "
        f"{name_pairs['verdict_flips']}/{name_pairs['pairs']}. System-only results are reported separately.\n"
        "Per-arm lexical vocabularies differ; latent map is shared. No capacity-matched or spontaneous-belief claim; human review pending.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.95), h_pad=2)
    output.mkdir(parents=True, exist_ok=True)
    for path in names:
        fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    atomic_json(
        receipt,
        {
            "schema": "lexical-control-figure-v1",
            "input_sha256": {
                str(lexical / "metrics.json"): file_sha256(lexical / "metrics.json"),
                str(lexical / "audit" / "audit_receipt.json"): file_sha256(
                    lexical / "audit" / "audit_receipt.json"
                ),
                str(latent / "summary.json"): file_sha256(latent / "summary.json"),
            },
            "source_sha256": file_sha256(__file__),
            "plot_environment": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
            "output_sha256": {p.name: file_sha256(p) for p in names},
            "scope": "Descriptive observed comparison; text is one shared posthoc lexical control",
            "new_model_calls": 0,
            "classifier_fits": 0,
        },
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lexical", required=True, type=Path)
    parser.add_argument("--latent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(plot(args.lexical, args.latent, args.output))


if __name__ == "__main__":
    main()
