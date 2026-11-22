"""Export audited descriptive crossfit results, without fitting or model inference."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

METHODS = ("direct_lr", "rbf_svc", "fourfactor_lr")
LABELS = ("Direct linear", "RBF classifier", "Four factors + rule")
STATUS = "posthoc_exposed_bank_group_crossfit"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(value, message):
    if not value:
        raise ValueError(message)


def logical_path(path):
    value = str(path).replace("\\", "/")
    prefix = "//wsl.localhost/Ubuntu"
    if value.startswith(prefix + "/"):
        value = value[len(prefix) :]
    if value.startswith("C:/"):
        value = "/mnt/c/" + value[3:]
    return value


def check_quartets(record, count):
    values = record["quartets"]
    require(len(values) == count, "Quartet inventory differs")
    identities = [(v["group_id"], v["update"], v["world"], v["name_map"]) for v in values]
    require(len(set(identities)) == count, "Duplicate quartet identities")
    for value in values:
        require(
            type(value["correct_cells"]) is int and 0 <= value["correct_cells"] <= 4,
            "Invalid correct-cell count",
        )
        require(
            type(value["all_correct"]) is bool
            and value["all_correct"] == (value["correct_cells"] == 4),
            "Quartet verdict inconsistent with correct-cell count",
        )
    accuracy = sum(v["all_correct"] for v in values) / count
    require(
        math.isclose(record["primary_quartet_accuracy"], accuracy, abs_tol=1e-12),
        "Quartet accuracy inconsistent with saved verdicts",
    )
    expected_histogram = {str(i): sum(v["correct_cells"] == i for v in values) for i in range(5)}
    require(
        record["quartet_correct_cells_histogram"] == expected_histogram,
        "Quartet histogram inconsistent with saved verdicts",
    )
    require(
        math.isclose(
            record["primary_row_accuracy"],
            sum(v["correct_cells"] for v in values) / (4 * count),
            abs_tol=1e-12,
        ),
        "Primary row accuracy inconsistent with quartet correct-cell counts",
    )
    require(
        math.isclose(
            record["row_accuracy"],
            (record["primary_row_accuracy"] + record["secondary_row_accuracy"]) / 2,
            abs_tol=1e-12,
        ),
        "All-row accuracy inconsistent with primary and secondary row accuracy",
    )


def compact_method(record):
    """Keep all group/fold values, omitting redundant per-row/quartet traces."""

    def compact(value):
        return {k: v for k, v in value.items() if k != "quartets"}

    require(len(record["groups"]) == 30, "Need all thirty narrative groups")
    require(len(record["fold_values"]) == 5, "Need all five outer folds")
    require(record["summary"]["rows"] == 1920, "Need all correction rows")
    group_ids = [g["group_id"] for g in record["groups"]]
    require(len(set(group_ids)) == 30, "Duplicate narrative groups")
    require(
        {f["fold"] for f in record["fold_values"]} == set(range(5)),
        "Duplicate or missing fold identities",
    )
    check_quartets(record["summary"], 240)
    group_quartets = []
    for group in record["groups"]:
        require(group["rows"] == 64 and group["fold"] in range(5), "Invalid group inventory")
        check_quartets(group, 8)
        require(
            {q["group_id"] for q in group["quartets"]} == {group["group_id"]},
            "Quartet belongs to a different group",
        )
        group_quartets.extend(group["quartets"])

    def key(q):
        return q["group_id"], q["update"], q["world"], q["name_map"]

    require(
        sorted(group_quartets, key=key) == sorted(record["summary"]["quartets"], key=key),
        "Group and aggregate quartets differ",
    )
    for fold in record["fold_values"]:
        require(fold["rows"] == 384, "Fold row inventory differs")
        check_quartets(fold, 48)
        members = [g for g in record["groups"] if g["fold"] == fold["fold"]]
        require(len(members) == 6, "Fold must contain six narrative groups")
        expected = [q for g in members for q in g["quartets"]]
        require(
            sorted(expected, key=key) == sorted(fold["quartets"], key=key),
            "Fold and group quartets differ",
        )
        for metric in ("row_accuracy", "primary_row_accuracy", "secondary_row_accuracy"):
            require(
                math.isclose(fold[metric], sum(g[metric] for g in members) / 6, abs_tol=1e-12),
                "Fold row accuracy differs from equal-sized groups",
            )
    for metric in ("row_accuracy", "primary_row_accuracy", "secondary_row_accuracy"):
        require(
            math.isclose(
                record["summary"][metric],
                sum(g[metric] for g in record["groups"]) / 30,
                abs_tol=1e-12,
            ),
            "Aggregate row accuracy differs from equal-sized groups",
        )
    return {
        "summary": compact(record["summary"]),
        "groups": [compact(g) for g in record["groups"]],
        "fold_values": [compact(f) for f in record["fold_values"]],
    }


def load_summary(run, audit):
    complete_path = run / "completed_manifest.json"
    complete = read(complete_path)
    require(complete["status"] == "complete", "Execution must be complete")
    require(complete["research_status"] == STATUS, "Wrong research status")
    require(complete["confirmatory_result"] is False, "Exposed results cannot be confirmatory")
    required_outputs = {"prepared_manifest.json"} | {
        f"{m}/{name}"
        for m in ("llama", "qwen")
        for name in ("oof_results.json", *(f"fold_{f}/results.json" for f in range(5)))
    }
    require(
        required_outputs <= complete["output_sha256"].keys(),
        "Completion manifest omits files read by this report",
    )
    for relative, expected in complete["output_sha256"].items():
        path = Path(relative)
        require(
            not path.is_absolute() and ".." not in path.parts, "Completion path escapes the run"
        )
        require(digest(run / relative) == expected, "Completed output changed: " + relative)
    review = read(audit)
    require(
        review["schema"] == "cached-readout-adequacy-independent-audit-v1"
        and review["research_status"] == STATUS,
        "Wrong independent audit provenance",
    )
    require(review["status"] == "all_independent_checks_passed", "Independent audit must pass")
    require(review["pipeline_imported"] is False, "Need independently reconstructed results")
    require(review["classifier_refits"] == review["model_calls"] == 0, "Audit must use saved data")
    audit_sources = {logical_path(k): v for k, v in review["source_sha256"].items()}
    for path in [
        complete_path,
        run / "prepared_manifest.json",
        *(run / m / "oof_results.json" for m in ("llama", "qwen")),
    ]:
        require(
            audit_sources.get(logical_path(path)) == digest(path),
            "Independent audit does not bind this exact source: " + str(path),
        )
    prepared = read(run / "prepared_manifest.json")
    summary = {
        "schema": "cached-readout-adequacy-report-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "research_status": STATUS,
        "fresh_test": False,
        "confirmatory_result": False,
        "human_review_complete": False,
        "new_language_model_calls": 0,
        "outer_train_rows": 1536,
        "independent_narrative_groups": 30,
        "interpretation": "Descriptive exposed-bank group crossfit; overlapping training folds.",
        "environment": prepared["environment"],
        "source_sha256": {
            str(complete_path): digest(complete_path),
            str(run / "prepared_manifest.json"): digest(run / "prepared_manifest.json"),
            str(audit): digest(audit),
            str(Path(__file__).resolve()): digest(__file__),
        },
        "models": {},
    }
    for model in ("llama", "qwen"):
        path = run / model / "oof_results.json"
        value = read(path)
        require(
            value["fresh_test"] is False and value["original_split"] == "test",
            "Original exposed-test provenance must remain explicit",
        )
        require(
            value["research_status"] == STATUS and value["confirmatory_result"] is False,
            "Invalid research provenance",
        )
        require(set(value["methods"]) == set(METHODS), "Need exactly the declared three methods")
        require(
            review["models"][model]["reconstructed_oof_predictions"] == 5760,
            "Independent prediction inventory differs",
        )
        summary["source_sha256"][str(path)] = digest(path)
        methods = {name: compact_method(value["methods"][name]) for name in METHODS}
        selection = []
        for fold in range(5):
            fold_record = read(run / model / f"fold_{fold}" / "results.json")
            selection.append(
                {
                    "fold": fold,
                    "methods": {
                        name: {
                            "selected_parameters": fold_record["methods"][name][
                                "selected_parameters"
                            ],
                            "candidates": len(fold_record["methods"][name]["candidates"]),
                            "failed_candidates": [
                                c
                                for c in fold_record["methods"][name]["candidates"]
                                if c["status"] != "complete"
                            ],
                        }
                        for name in METHODS
                    },
                }
            )
        summary["models"][model] = {"methods": methods, "selection": selection}
    return summary


def figures(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    colors = ("#23649e", "#dd8b25", "#378263")
    x = np.arange(3)
    for ax, (model, heading) in zip(
        axes, (("llama", "Llama 3.1 8B"), ("qwen", "Qwen2.5 7B")), strict=True
    ):
        values = summary["models"][model]["methods"]
        heights = [100 * values[m]["summary"]["primary_quartet_accuracy"] for m in METHODS]
        ax.bar(x, heights, color=colors, width=0.58, alpha=0.8)
        for column, name in enumerate(METHODS):
            points = [100 * f["primary_quartet_accuracy"] for f in values[name]["fold_values"]]
            ax.scatter(column + np.linspace(-0.16, 0.16, 5), points, color="black", s=16, zorder=3)
            ax.text(
                column, max(points + [heights[column]]) + 3, f"{heights[column]:.1f}%", ha="center"
            )
        ax.set(
            title=heading, ylim=(0, 110), xticks=x, xticklabels=("Linear", "RBF", "Factors\n+ rule")
        )
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Complete primary correction quartets (%)")
    fig.suptitle("Correction-trained readouts on existing midpoint activations", fontsize=13)
    fig.text(
        0.5,
        0.025,
        "Exposed-bank crossfit: 30 narratives, 240 quartets/model. Dots: five outer folds; no confidence intervals.\n"
        "Factors use four labels per training row plus an external update rule.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"readout_adequacy.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown(summary):
    lines = [
        "# Correction readout adequacy on the exposed activation bank",
        "",
        "**Posthoc descriptive diagnostic.** Original test cases were already exposed; "
        "these are group-crossfit results, not fresh confirmation. All original study results remain unchanged.",
        "",
        "Five outer folds keep all 64 variants of each narrative together. Each selected readout is refitted "
        "on 24 narratives (1,536 rows) and predicts six excluded narratives. Hyperparameter selection uses "
        "18 separate inner-training and six inner-development groups. Every method receives one saved "
        "midpoint vector per prompt. There are 30 narrative units shared across models, 1,920 rows and "
        "240 primary quartets per model. A quartet requires all four verdicts correct.",
        "",
        "![Readout comparison](readout_adequacy.png)",
        "",
        "| Model | Readout | Primary quartet % | All rows % | Primary rows % | Secondary rows % |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for model in ("llama", "qwen"):
        for method, label in zip(METHODS, LABELS, strict=True):
            s = summary["models"][model]["methods"][method]["summary"]
            scores = [
                s[k] * 100
                for k in (
                    "primary_quartet_accuracy",
                    "row_accuracy",
                    "primary_row_accuracy",
                    "secondary_row_accuracy",
                )
            ]
            lines.append(
                f"| {model.title()} | {label} | " + " | ".join(f"{v:.2f}" for v in scores) + " |"
            )
    lines += [
        "",
        "## Factor diagnostic and nuisance sensitivity",
        "",
        "Four-factor decoding has four labels per training row plus the supplied Boolean update algorithm. "
        "It is a privileged diagnostic, not a supervision- or compute-matched improvement. RBF may retain "
        "all training examples as support vectors. Its margin is not a calibrated probability.",
        "",
        "| Model | Initial % | Message value % | First receipt % | Final receipt % | All four factors % |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in ("llama", "qwen"):
        s = summary["models"][model]["methods"]["fourfactor_lr"]["summary"]
        scores = [
            s["factor_accuracies"][k] * 100
            for k in ("initial", "update", "first_receipt", "receipt")
        ]
        scores.append(100 * s["factor_joint_accuracy"])
        lines.append(f"| {model.title()} | " + " | ".join(f"{v:.2f}" for v in scores) + " |")
    lines += [
        "",
        "| Model | Readout | World flips % | Name-map flips % | Foldwise primary quartet % |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for model in ("llama", "qwen"):
        for method, label in zip(METHODS, LABELS, strict=True):
            m = summary["models"][model]["methods"][method]
            flips = m["summary"]["nuisance_verdict_flips"]
            folds = ", ".join(
                f"{100 * f['primary_quartet_accuracy']:.1f}" for f in m["fold_values"]
            )
            lines.append(
                f"| {model.title()} | {label} | {100 * flips['world']['fraction']:.2f} | "
                f"{100 * flips['name_map']['fraction']:.2f} | {folds} |"
            )
    failed = sum(
        len(m["failed_candidates"])
        for v in summary["models"].values()
        for f in v["selection"]
        for m in f["methods"].values()
    )
    lines += [
        "",
        f"Failed inner candidates retained: **{failed}**. Full selections, per-group/fold metrics, "
        "class recalls, receipt pairs and quartet histograms are in [the source-bound summary](summary.json).",
        "",
        "## Interpretation limits",
        "",
        "Success shows accessibility under correction supervision and this readout family. It does not "
        "make the original one-step-trained head transfer successfully. Comparing those earlier heads "
        "with these heads changes supervision, support, fitting size and selection, so it is not a causal "
        "estimate of adding correction examples. This diagnostic uses 1,536 outer-training rows, while "
        "the planned fresh test may use N512; success here cannot certify that smaller instrument.",
        "",
        "The folds have overlapping training sets. Aggregate values and fold dots are descriptive; no "
        "naive bootstrap or significance test is reported. Both models share the same 30 narratives and "
        "renderer. Human semantic review is incomplete. These measurements concern sender-attributed "
        "reader belief under an explicit belief question, not spontaneous anticipation, deceptive intent "
        "or causal use of a belief representation.",
        "",
        "All saved margins, train-only transforms, selected-grid records, folds and verdict metrics were "
        "reconstructed by a separate auditor without importing or refitting the pipeline. The output "
        "manifest binds the source analyses, audit and renderer. No new language-model calls were made.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Use a fresh export directory")
    summary = load_summary(args.run.resolve(), args.audit.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    figures(summary, args.output)
    (args.output / "results.md").write_text(markdown(summary), encoding="utf-8")
    manifest = {
        "schema": "cached-readout-adequacy-export-v1",
        "research_status": STATUS,
        "source_sha256": summary["source_sha256"],
        "output_sha256": {p.name: digest(p) for p in sorted(args.output.iterdir()) if p.is_file()},
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "status": STATUS}))


if __name__ == "__main__":
    main()
