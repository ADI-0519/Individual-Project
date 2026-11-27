"""Export every prespecified arm after both analyses and independent audit pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_probes.storage import file_sha256, read_json, utc_now


def require(value, message):
    if not value:
        raise ValueError(message)


def report(campaign, audit, output, *, allow_test_fixture=False):
    campaign, audit, output = Path(campaign), Path(audit), Path(output)
    receipt = read_json(audit / "audit_receipt.json")
    require(receipt["status"] == "pass", "Passing independent audit required")
    require(
        Path(receipt["campaign"]).resolve() == campaign.resolve(),
        "Audit belongs to a different campaign",
    )
    inputs = {str(audit / "audit_receipt.json"): file_sha256(audit / "audit_receipt.json")}
    models = {}
    for key in ("llama", "qwen"):
        path = campaign / (key + "_analysis")
        manifest = read_json(path / "manifest.json")
        require(manifest["status"] == "complete", "Both analyses must be complete")
        for name in ("results.json", "predictions.npz"):
            require(
                file_sha256(path / name) == manifest["output_sha256"][name],
                "Analysis output changed",
            )
            require(
                receipt["analysis_output_sha256"].get(str((path / name).resolve()))
                == file_sha256(path / name),
                "Current result is not the independently audited result",
            )
        models[key] = read_json(path / "results.json")
        require(
            models[key]["test_fixture"] is False or allow_test_fixture,
            "Fixture cannot become scientific report",
        )
        inputs[str(path / "results.json")] = file_sha256(path / "results.json")
        inputs[str(path / "manifest.json")] = file_sha256(path / "manifest.json")
    output.mkdir(parents=True, exist_ok=False)
    fixture = any(m["test_fixture"] for m in models.values())
    summary = {
        "schema": "format-adaptation-report-v1",
        "created_at": utc_now(),
        "research_status": "agent_audited_exploratory",
        "human_review_complete": False,
        "models": models,
        "input_sha256": inputs,
        "new_report_model_calls": 0,
        "classifier_fits": 0,
        "test_fixture": fixture,
    }
    with (output / "summary.json").open("x", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    lines = [
        "# Fresh matched-format adaptation: completed fixed comparison",
        "",
        "Agent-audited exploratory. Human semantic review remains incomplete. "
        "Both models share 24 test contexts; the intervals condition on fixed fits and authoring.",
        "",
        "All four arms receive four factor targets and the same supplied rule. "
        "Mixed versus Offset compares procedures, not a unique geometric mechanism.",
        "",
        "| Model / format | A512 hard panels | Mixed512 | Ahalf256 | Offset |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    if fixture:
        lines.insert(0, "TEST FIXTURE — NO RESEARCH RESULT\n")
    arms = ("A512", "Mixed512", "Ahalf256", "Offset")
    for key in ("llama", "qwen"):
        for renderer in ("A", "B"):
            values = []
            for arm in arms:
                d = models[key]["arms"][arm][renderer + "/correction_hard"]["aggregate"]
                values.append(
                    f"{d['complete_quartets']}/{d['quartets']} ({100 * d['quartet_accuracy']:.2f}%)"
                )
            lines.append("| " + key + " / " + renderer + " | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Fixed primary and named diagnostic",
            "",
            "```json",
            json.dumps(
                {
                    k: {"primary": m["primary"], "named_diagnostic": m["named_diagnostic"]}
                    for k, m in models.items()
                },
                indent=2,
            ),
            "```",
            "",
            "## Every arm, format and condition",
            "",
            "| Model | Arm | Stratum | Correct rows | Row accuracy | Complete panels |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for key, m in models.items():
        for arm in arms:
            for stratum, d in m["arms"][arm].items():
                x = d["aggregate"]
                lines.append(
                    f"| {key} | {arm} | {stratum} | {x['correct']}/{x['rows']} | "
                    f"{100 * x['row_accuracy']:.2f}% | {x['complete_quartets']}/{x['quartets']} |"
                )
    lines.extend(
        [
            "",
            "Full class recalls, per-context values, factor diagnostics, nuisance controls, "
            "conditional output and strict first-token diagnostics are in [summary.json](summary.json). "
            "Four-factor truth applies only to correction conditions. One-step/asserted scores "
            "are out-of-training-construct controls, not separate four-factor validation.",
            "",
            "![Fixed hard-panel comparison](adaptation.svg)",
            "",
            "The source and prediction archives, row journals, both-model pretest seal, "
            "failed-attempt evidence (if any), and independently frozen reconstructions remain "
            "in the bound native campaign/audit folders. No test-selected result replaces "
            "the Llama Mixed-minus-A512 primary.",
            "",
        ]
    )
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), sharey=True)
    x = np.arange(len(arms))
    for key, ax in zip(("llama", "qwen"), axes, strict=True):
        for renderer, offset, color in (("A", -0.17, "#296b98"), ("B", 0.17, "#d4912e")):
            values = [
                100
                * models[key]["arms"][arm][renderer + "/correction_hard"]["aggregate"][
                    "quartet_accuracy"
                ]
                for arm in arms
            ]
            ax.bar(x + offset, values, 0.33, label="Format " + renderer, color=color)
            for pos, value in zip(x + offset, values, strict=True):
                ax.text(pos, value + 1, f"{value:.1f}", ha="center", fontsize=9)
        ax.set_xticks(x, arms, rotation=12)
        ax.set_ylim(0, 108)
        ax.set_title("Llama: primary" if key == "llama" else "Qwen: named replication")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Hard complete-quartet accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.08), ncol=2, frameon=False
    )
    fig.suptitle("Fresh matched-observation format adaptation")
    if fixture:
        fig.suptitle("TEST FIXTURE — NO RESEARCH RESULT")
    fig.text(
        0.5,
        0.02,
        "24 shared paired contexts; 192 hard quartets per format/model. "
        "Four learned factors + supplied rule; human review incomplete.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.18, 1, 0.94))
    for ext in ("pdf", "svg", "png"):
        fig.savefig(output / ("adaptation." + ext), dpi=180)
    plt.close(fig)
    fresh = {
        "schema": "format-adaptation-report-manifest-v1",
        "status": "complete",
        "created_at": utc_now(),
        "input_sha256": inputs,
        "source_sha256": file_sha256(__file__),
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir()},
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as f:
        json.dump(fresh, f, indent=2)
        f.write("\n")
    return fresh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report(args.campaign, args.audit, args.output)


if __name__ == "__main__":
    main()
