"""Render two saved main-study analyses without fitting or changing their policies.

Default inputs must be real empirical analyses. Synthetic fixture rendering is
an explicitly marked engineering check. Existing output directories are never
reused. Matplotlib is imported only when figures are requested by write_report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

MODEL_IDS = {
    "Llama": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen": "Qwen/Qwen2.5-7B-Instruct",
}
SIZES = (128, 256, 512)
SEEDS = (11, 23, 37, 53, 71)
MECHANISMS = ("observation", "message", "correction")
ARMS = ("restricted", "full")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: object, *, low: float = 0, high: float = 1) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError("Expected a finite numeric score in its declared range")
    return float(value)


def _groups(values: object) -> dict[str, dict]:
    if not isinstance(values, list) or not values:
        raise ValueError("Nonempty scenario groups are required")
    result = {}
    for group in values:
        if not isinstance(group, dict):
            raise ValueError("Scenario group must be an object")
        identity = group.get("group_id")
        if not isinstance(identity, str) or not identity or identity in result:
            raise ValueError("Unique nonempty scenario group IDs are required")
        if group.get("mechanism") not in MECHANISMS:
            raise ValueError("Unrecognized scenario mechanism")
        _number(group.get("complete_quartet_accuracy"))
        result[identity] = group
    return result


def validate_analysis(data: dict, label: str, *, allow_fixture: bool = False) -> dict:
    """Reject ambiguous provenance and inconsistent inventories before rendering.

    This validates saved JSON consistency. It does not reauthenticate source
    tensors or repeat the independent capture audit.
    """
    if not isinstance(data, dict) or data.get("schema") != "main-study-analysis-v1":
        raise ValueError("Unsupported main-study analysis schema")
    fixture = data.get("test_fixture")
    if type(fixture) is not bool:
        raise ValueError("Explicit boolean fixture provenance required")
    if fixture and not allow_fixture:
        raise ValueError("Fixture analyses require --allow-fixture and are not model results")
    for key in ("scientific_result", "real_empirical_result"):
        if data.get(key) is not (not fixture):
            raise ValueError("Empirical/fixture provenance is contradictory")
    if type(data.get("human_review_complete")) is not bool:
        raise ValueError("Explicit human review provenance required")
    if data.get("confirmatory_result") is not False:
        raise ValueError("This reporting protocol accepts explicitly non-confirmatory analyses")
    if not isinstance(data.get("research_status"), str) or not data["research_status"]:
        raise ValueError("Explicit research status required")
    if data["research_status"] == "agent_audited_exploratory" and data["human_review_complete"]:
        raise ValueError("Agent exploratory analysis cannot claim completed human review")
    provenance = data.get("capture_provenance", {})
    if provenance.get("model_settings", {}).get("model_id") != MODEL_IDS[label]:
        raise ValueError(f"{label} analysis has the wrong model identity")
    bank = provenance.get("bank_sha256")
    if not isinstance(bank, str) or not re.fullmatch(r"[0-9a-f]{64}", bank):
        raise ValueError("Canonical bank SHA256 is required")
    for key in ("research_status", "human_review_complete"):
        if key in provenance and provenance[key] != data[key]:
            raise ValueError(f"Capture/analysis provenance differs: {key}")
    if not fixture:
        for key in ("fit_settings_sha256", "fit_heads_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(data.get(key, ""))):
                raise ValueError("Real analyses require saved fit artifact hashes")
    baselines = data.get("baselines", {})
    reference = _groups(baselines.get("symbolic_oracle"))
    if any(not any(g["mechanism"] == m for g in reference.values()) for m in MECHANISMS):
        raise ValueError("All three mechanisms must be retained even in fixtures")
    if not fixture and (
        len(reference) != 90
        or any(sum(g["mechanism"] == m for g in reference.values()) != 30 for m in MECHANISMS)
    ):
        raise ValueError("Real analysis must retain ninety groups, thirty per mechanism")

    def same_groups(groups: object) -> dict[str, dict]:
        mapped = _groups(groups)
        if set(mapped) != set(reference) or any(
            mapped[key]["mechanism"] != reference[key]["mechanism"] for key in reference
        ):
            raise ValueError("Methods must retain the same scenario groups and mechanisms")
        return mapped

    for groups in baselines.values():
        same_groups(groups)
    inventory = {}
    for record in data.get("comparisons", []):
        key = (record.get("feature_kind"), record.get("training_n"), record.get("allocation_seed"))
        if (
            key[0] not in ("activation", "tfidf")
            or type(key[1]) is not int
            or key[1] not in SIZES
            or type(key[2]) is not int
            or key[2] not in SEEDS
            or key in inventory
        ):
            raise ValueError("Unexpected or duplicate comparison inventory")
        if not isinstance(record.get("status"), str) or not record["status"]:
            raise ValueError("Every comparison must retain its status")
        if record["status"] == "complete":
            if set(record.get("arms", {})) != set(ARMS):
                raise ValueError("Complete comparison requires both fitted arms")
            for arm in ARMS:
                same_groups(record["arms"][arm].get("groups"))
                if "calibrated_groups" in record["arms"][arm]:
                    same_groups(record["arms"][arm]["calibrated_groups"])
        inventory[key] = record
    expected = {
        (kind, n, seed) for kind in ("activation", "tfidf") for n in SIZES for seed in SEEDS
    }
    if set(inventory) != expected:
        raise ValueError("All thirty planned comparison records, including failures, are required")
    primary = data.get("primary")
    completed = [s for s in SEEDS if inventory[("activation", 512, s)]["status"] == "complete"]
    if not isinstance(primary, dict) or not isinstance(primary.get("status"), str):
        raise ValueError("Explicit primary analysis status required")
    if primary["status"] == "complete":
        if completed != list(SEEDS) or primary.get("allocation_seeds") != list(SEEDS):
            raise ValueError("Complete primary cannot omit a failed allocation")
        correction_ids = {k for k, g in reference.items() if g["mechanism"] == "correction"}
        if primary.get("independent_scenario_groups") != len(correction_ids):
            raise ValueError("Primary scenario count differs from retained correction groups")
        scores = {}
        for arm in ARMS:
            cells = [
                group["complete_quartet_accuracy"]
                for seed in SEEDS
                for group in inventory[("activation", 512, seed)]["arms"][arm]["groups"]
                if group["mechanism"] == "correction"
            ]
            scores[arm] = sum(cells) / len(cells)
            if not math.isclose(_number(primary.get(arm)), scores[arm], abs_tol=1e-10):
                raise ValueError("Primary arm score differs from saved group scores")
        difference = _number(primary.get("full_minus_restricted"), low=-1)
        if not math.isclose(difference, scores["full"] - scores["restricted"], abs_tol=1e-10):
            raise ValueError("Primary paired difference is inconsistent")
        interval = primary.get("percentile_95_interval")
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("Primary interval must be the saved pair of endpoints")
        low, high = (_number(value, low=-1) for value in interval)
        if low > high:
            raise ValueError("Primary interval endpoints are reversed")
    elif (
        completed == list(SEEDS)
        or primary.get("available_allocation_seeds") != completed
        or primary.get("missing_allocation_seeds") != sorted(set(SEEDS) - set(completed))
    ):
        raise ValueError("Unavailable primary must disclose exactly the missing/failed allocations")
    monitor = data.get("prose_monitor", {"status": "not_provided"})
    if monitor.get("status") == "complete":
        for key in ("test_fixture", "research_status", "human_review_complete"):
            if monitor.get(key) != data[key]:
                raise ValueError("Prose monitor provenance differs from analysis")
        if monitor.get("bank_sha256") != bank or monitor.get("partition") != "test":
            raise ValueError("Prose monitor must cover the same held-out bank")
        monitor_groups = same_groups(monitor.get("groups"))
        for group in monitor_groups.values():
            if (
                type(group.get("rows")) is not int
                or type(group.get("invalid")) is not int
                or group["rows"] != (64 if group["mechanism"] == "correction" else 32)
                or not 0 <= group["invalid"] <= group["rows"]
            ):
                raise ValueError("Prose monitor groups require full row/invalid denominators")
        overall = monitor.get("overall", {})
        if type(overall.get("invalid")) is not int or type(overall.get("rows")) is not int:
            raise ValueError("Prose monitor must disclose invalid and total output counts")
        if not 0 <= overall["invalid"] <= overall["rows"]:
            raise ValueError("Invalid prose monitor denominator")
        if any(
            overall[key] != sum(g[key] for g in monitor_groups.values())
            for key in ("invalid", "rows")
        ):
            raise ValueError("Prose monitor overall denominators differ from retained groups")
    elif monitor.get("status") != "not_provided":
        raise ValueError("Unknown prose monitor status")
    return data


def summarize(data: dict, label: str) -> dict:
    """Macro-average existing group scores; partial allocations remain unavailable."""
    results = {}
    grouped = defaultdict(list)
    for record in data["comparisons"]:
        grouped[(record["feature_kind"], record["training_n"])].append(record)
    failures = []
    for (kind, size), records in sorted(grouped.items()):
        failed = [r for r in records if r["status"] != "complete"]
        failures.extend(
            {key: r[key] for key in ("feature_kind", "training_n", "allocation_seed", "status")}
            for r in failed
        )
        for mechanism in MECHANISMS:
            scores = {}
            group_count = sum(
                g["mechanism"] == mechanism for g in data["baselines"]["symbolic_oracle"]
            )
            for arm in ARMS:
                cells = [
                    g["complete_quartet_accuracy"]
                    for r in records
                    if r["status"] == "complete"
                    for g in r["arms"][arm]["groups"]
                    if g["mechanism"] == mechanism
                ]
                scores[arm] = None if failed else sum(cells) / len(cells)
            results[f"{kind}:{size}:{mechanism}"] = {
                "feature_kind": kind,
                "training_n": size,
                "mechanism": mechanism,
                "scenario_groups": group_count,
                "complete_allocations": len(records) - len(failed),
                "planned_allocations": len(SEEDS),
                **scores,
            }
    baselines = {}
    for name, groups in data["baselines"].items():
        baselines[name] = {
            mechanism: sum(
                g["complete_quartet_accuracy"] for g in groups if g["mechanism"] == mechanism
            )
            / sum(g["mechanism"] == mechanism for g in groups)
            for mechanism in MECHANISMS
        }
    for kind in ("activation", "tfidf"):
        records = grouped[(kind, 512)]
        for arm in ARMS:
            complete = all(
                r["status"] == "complete" and "calibrated_groups" in r["arms"][arm] for r in records
            )
            if complete:
                baselines[f"{kind}_{arm}_calibrated_N512"] = {
                    mechanism: sum(
                        g["complete_quartet_accuracy"]
                        for r in records
                        for g in r["arms"][arm]["calibrated_groups"]
                        if g["mechanism"] == mechanism
                    )
                    / sum(
                        g["mechanism"] == mechanism
                        for r in records
                        for g in r["arms"][arm]["calibrated_groups"]
                    )
                    for mechanism in MECHANISMS
                }
    primary_allocations = []
    for r in sorted(grouped[("activation", 512)], key=lambda item: item["allocation_seed"]):
        row = {"allocation_seed": r["allocation_seed"], "status": r["status"]}
        for arm in ARMS:
            groups = (
                []
                if r["status"] != "complete"
                else [
                    g["complete_quartet_accuracy"]
                    for g in r["arms"][arm]["groups"]
                    if g["mechanism"] == "correction"
                ]
            )
            row[arm] = sum(groups) / len(groups) if groups else None
        row["difference"] = row["full"] - row["restricted"] if r["status"] == "complete" else None
        primary_allocations.append(row)
    monitor = data.get("prose_monitor", {"status": "not_provided"})
    if monitor["status"] == "complete":
        baselines["prose_monitor"] = {
            mechanism: sum(
                g["complete_quartet_accuracy"]
                for g in monitor["groups"]
                if g["mechanism"] == mechanism
            )
            / sum(g["mechanism"] == mechanism for g in monitor["groups"])
            for mechanism in MECHANISMS
        }
    return {
        "model": label,
        "model_id": data["capture_provenance"]["model_settings"]["model_id"],
        "primary": data["primary"],
        "learning_and_mechanisms": results,
        "baselines": baselines,
        "failed_comparisons": failures,
        "primary_by_allocation": primary_allocations,
        "prose_monitor": {
            "status": monitor["status"],
            "invalid": monitor.get("overall", {}).get("invalid"),
            "rows": monitor.get("overall", {}).get("rows"),
        },
    }


def _percent(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "UNAVAILABLE"
    return f"{value * 100:+.1f}" if signed else f"{value * 100:.1f}"


def _status(data: dict) -> str:
    if data["test_fixture"]:
        return "SYNTHETIC FIXTURE — NOT MODEL RESULTS"
    review = "complete" if data["human_review_complete"] else "INCOMPLETE"
    return f"{data['research_status']} | human review {review} | confirmatory result: false"


def render_markdown(summaries: list[dict], source: dict) -> str:
    lines = [
        "# Matched belief-study results",
        "",
        f"**{_status(source)}**",
        "",
        "Target: fictional sender-attributed reader belief content. These results do not "
        "establish deception, spontaneous anticipation, or absence of belief information.",
        "",
        "## Primary correction-quartet comparison (N=512)",
        "",
        "Scores are percentages; differences and saved intervals are percentage points. "
        "Arm-score intervals are unavailable and are not fabricated.",
        "",
        "| Model | Restricted | Full | Full − restricted | Saved 95% group interval | "
        "Correction groups | Allocations |",
        "| --- | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for summary in summaries:
        primary = summary["primary"]
        if primary["status"] == "complete":
            interval = primary["percentile_95_interval"]
            cells = [
                summary["model"],
                _percent(primary["restricted"]),
                _percent(primary["full"]),
                _percent(primary["full_minus_restricted"], signed=True),
                f"[{_percent(interval[0], signed=True)}, {_percent(interval[1], signed=True)}]",
                str(primary["independent_scenario_groups"]),
                f"{len(primary['allocation_seeds'])}/{len(SEEDS)}",
            ]
        else:
            row = summary["learning_and_mechanisms"]["activation:512:correction"]
            cells = [
                summary["model"],
                *(["UNAVAILABLE"] * 4),
                str(row["scenario_groups"]),
                f"{row['complete_allocations']}/{len(SEEDS)}",
            ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Intervals are copied from the saved analysis: authored correction groups are "
        "resampled with five paired allocations retained. They are conditional on this "
        "bank and allocation design. Factorial rows and allocations are not extra narratives.",
        "",
        "### Paired allocation variability",
        "",
        "Each row averages the retained correction groups; no allocation is an extra narrative.",
        "",
        "| Model | Allocation | Status | Restricted | Full | Difference (pp) |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        for row in summary["primary_by_allocation"]:
            lines.append(
                f"| {summary['model']} | {row['allocation_seed']} | {row['status']} | "
                f"{_percent(row['restricted'])} | {_percent(row['full'])} | "
                f"{_percent(row['difference'], signed=True)} |"
            )
    lines += [
        "",
        "## Learning curves and mechanism retention",
        "",
        "Equal-group, equal-allocation raw complete-quartet scores; no intervals were saved "
        "for these summaries. If any allocation failed, the aggregate remains unavailable.",
        "",
        "| Model | Method | N | Mechanism | Restricted | Full | Groups | Allocations |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for summary in summaries:
        for row in summary["learning_and_mechanisms"].values():
            lines.append(
                f"| {summary['model']} | {row['feature_kind']} | {row['training_n']} | "
                f"{row['mechanism']} | {_percent(row['restricted'])} | {_percent(row['full'])} | "
                f"{row['scenario_groups']} | "
                f"{row['complete_allocations']}/{row['planned_allocations']} |"
            )
    lines += [
        "",
        "## Comparators",
        "",
        "Rule and output rows below use the same groups. TF-IDF appears above under its "
        "matched fitting allocations. Symbolic oracle uses source factors and is a logic "
        "ceiling, not a prose monitor. Calibrated readouts use saved development calibration "
        "at N=512, while the primary contrast above remains raw. Missing comparators "
        "prevent an advantage claim.",
        "",
        "| Model | Comparator | Observation | Message | Correction |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    names = (
        "constant_0",
        "initial",
        "final_update",
        "world",
        "symbolic_oracle",
        "output_raw",
        "output_calibrated",
        "activation_restricted_calibrated_N512",
        "activation_full_calibrated_N512",
        "tfidf_restricted_calibrated_N512",
        "tfidf_full_calibrated_N512",
        "prose_monitor",
    )
    for summary in summaries:
        for name in names:
            scores = summary["baselines"].get(name, {})
            lines.append(
                "| "
                + " | ".join(
                    [summary["model"], name, *[_percent(scores.get(m)) for m in MECHANISMS]]
                )
                + " |"
            )
        monitor = summary["prose_monitor"]
        if monitor["status"] != "complete":
            lines += [
                "",
                f"**{summary['model']}: PROSE MONITOR MISSING — baseline comparison incomplete.**",
            ]
        else:
            lines += [
                "",
                f"{summary['model']} prose-monitor invalid outputs: "
                f"{monitor['invalid']}/{monitor['rows']}; counted incorrect in quartet accuracy.",
            ]
    lines += ["", "## Failed comparisons", ""]
    for summary in summaries:
        failures = summary["failed_comparisons"]
        if not failures:
            lines.append(f"{summary['model']}: 0 failed comparisons in the saved inventory.")
        else:
            for failure in failures:
                lines.append(
                    f"- {summary['model']}: {failure['feature_kind']}, N={failure['training_n']}, "
                    f"allocation {failure['allocation_seed']}: {failure['status']}."
                )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "Read the paired difference together with both absolute scores, preservation and "
        "genuine-update performance, allocation variability and comparators. An interval "
        "including zero does not demonstrate equivalence. A readout failure does not show "
        "that the model has no beliefs. Correction adds event count, length and wording "
        "changes; these results alone do not isolate an abstract composition mechanism.",
        "",
        "This tool renders saved analysis JSON and checks its consistency; it does not "
        "reauthenticate model captures. Source file hashes are recorded in the report manifest.",
        "",
    ]
    return "\n".join(lines)


def _plot(summaries: list[dict], status: str, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"restricted": "#67758b", "full": "#057d8a"}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.9), layout="constrained")
        for index, summary in enumerate(summaries):
            primary = summary["primary"]
            if primary["status"] != "complete":
                axes[0].text(index, 45, "Unavailable\nfailed allocation(s)", ha="center")
                axes[1].text(0, index, "Unavailable", ha="center", va="center")
                continue
            for offset, arm in ((-0.18, "restricted"), (0.18, "full")):
                axes[0].bar(
                    index + offset,
                    primary[arm] * 100,
                    width=0.32,
                    color=colors[arm],
                    label=arm.title()
                    if arm.title() not in axes[0].get_legend_handles_labels()[1]
                    else None,
                )
            low, high = [v * 100 for v in primary["percentile_95_interval"]]
            difference = primary["full_minus_restricted"] * 100
            # Draw the saved interval itself: percentile intervals need not contain the estimate.
            axes[1].plot([low, high], [index, index], color=colors["full"], linewidth=3)
            axes[1].plot(difference, index, "o", color=colors["full"])
        axes[0].set(
            xticks=range(len(summaries)),
            xticklabels=[s["model"] for s in summaries],
            ylim=(0, 105),
            ylabel="Complete-quartet accuracy (%)",
            title="Raw arm scores · N=512",
        )
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, frameon=False)
        axes[1].axvline(0, color="#687079", linestyle="--", linewidth=1)
        axes[1].set(
            yticks=range(len(summaries)),
            yticklabels=[s["model"] for s in summaries],
            xlabel="Full − restricted (percentage points)",
            title="Paired difference · saved 95% group interval",
        )
        axes[1].invert_yaxis()
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
        counts = []
        for s in summaries:
            row = s["learning_and_mechanisms"]["activation:512:correction"]
            counts.append(
                f"{s['model']}: {row['scenario_groups']} correction groups; "
                f"{row['complete_allocations']}/5 allocations"
            )
        fig.suptitle("Fitting support and correction-sequence transfer\n" + status, fontsize=11)
        missing_monitors = [
            s["model"] for s in summaries if s["prose_monitor"]["status"] != "complete"
        ]
        monitor_note = (
            "PROSE MONITOR MISSING: " + ", ".join(missing_monitors)
            if missing_monitors
            else "Prose monitor complete; see comparator tables."
        )
        fig.supxlabel(
            " · ".join(counts)
            + "\nArm-score intervals unavailable; rows are dependent. "
            + monitor_note,
            fontsize=8,
        )
        for suffix in ("svg", "pdf", "png"):
            fig.savefig(output / f"primary.{suffix}", dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True, layout="constrained")
        mechanism_colors = {"observation": "#4575b4", "message": "#009e73", "correction": "#b65d20"}
        for ax, summary in zip(axes, summaries, strict=True):
            for mechanism in MECHANISMS:
                for arm, style in (("restricted", "--"), ("full", "-")):
                    scores = [
                        summary["learning_and_mechanisms"][f"activation:{n}:{mechanism}"][arm]
                        for n in SIZES
                    ]
                    ax.plot(
                        SIZES,
                        [math.nan if s is None else s * 100 for s in scores],
                        linestyle=style,
                        marker="o",
                        color=mechanism_colors[mechanism],
                        label=f"{mechanism} · {arm}",
                    )
            ax.set(
                title=summary["model"], xticks=SIZES, ylim=(0, 105), xlabel="Unique fitting inputs"
            )
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.15)
        axes[0].set_ylabel("Complete-quartet accuracy (%)")
        axes[1].legend(frameon=False, fontsize=8, loc="best")
        fig.suptitle("Activation-readout learning curves\n" + status, fontsize=11)
        fig.supxlabel(
            "Five paired allocations per point; missing/failed aggregate points omitted "
            "and disclosed in results.md.\nNo learning-curve intervals were saved.",
            fontsize=8,
        )
        for suffix in ("svg", "pdf", "png"):
            fig.savefig(output / f"learning_curves.{suffix}", dpi=180)
        plt.close(fig)


def _output_path(directory: str | Path) -> Path:
    path = Path(directory)
    if not path.is_absolute():
        raise ValueError("Output directory must be an absolute native path")
    resolved = path.resolve()
    if os.name == "posix" and re.match(r"^/mnt/[a-zA-Z](?:/|$)", resolved.as_posix()):
        raise ValueError("Use native Linux storage, not a mounted Windows output directory")
    if resolved.exists():
        raise FileExistsError("Preserve prior reports: output directory already exists")
    return resolved


def write_report(
    llama_analysis: str | Path,
    qwen_analysis: str | Path,
    output_directory: str | Path,
    *,
    allow_fixture: bool = False,
) -> dict:
    output = _output_path(output_directory)
    paths = [Path(llama_analysis).resolve(), Path(qwen_analysis).resolve()]
    if paths[0] == paths[1]:
        raise ValueError("Two separate model analyses are required")
    source_bytes = [path.read_bytes() for path in paths]
    input_records = [
        {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
        for path, raw in zip(paths, source_bytes, strict=True)
    ]
    analyses = [
        validate_analysis(json.loads(raw.decode("utf-8")), label, allow_fixture=allow_fixture)
        for raw, label in zip(source_bytes, MODEL_IDS, strict=True)
    ]
    first, second = analyses
    for key in ("test_fixture", "research_status", "human_review_complete", "confirmatory_result"):
        if first[key] != second[key]:
            raise ValueError("Both models must have the same research/fixture provenance")
    if first["capture_provenance"]["bank_sha256"] != second["capture_provenance"]["bank_sha256"]:
        raise ValueError("Both models must use the same canonical bank")
    first_groups = _groups(first["baselines"]["symbolic_oracle"])
    second_groups = _groups(second["baselines"]["symbolic_oracle"])
    if set(first_groups) != set(second_groups) or any(
        first_groups[k]["mechanism"] != second_groups[k]["mechanism"] for k in first_groups
    ):
        raise ValueError("Models must retain identical scenario groups")
    summaries = [summarize(data, label) for data, label in zip(analyses, MODEL_IDS, strict=True)]
    markdown = render_markdown(summaries, first)
    # Fail clearly before creating a partial report if the optional plotting extra is absent.
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Matplotlib is required; use the existing plots environment") from exc
    output.mkdir(parents=True, exist_ok=False)
    (output / "results.md").write_text(markdown, encoding="utf-8")
    summary = {
        "schema": "main-study-report-v1",
        "test_fixture": first["test_fixture"],
        "research_status": first["research_status"],
        "human_review_complete": first["human_review_complete"],
        "confirmatory_result": False,
        "models": summaries,
    }
    (output / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    _plot(summaries, _status(first), output)
    if any(_hash(Path(item["path"])) != item["sha256"] for item in input_records):
        raise RuntimeError(
            "Analysis input changed during reporting; partial output is not complete"
        )
    manifest = {
        "schema": "main-study-report-manifest-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "test_fixture": first["test_fixture"],
        "research_status": first["research_status"],
        "human_review_complete": first["human_review_complete"],
        "confirmatory_result": False,
        "bank_sha256": first["capture_provenance"]["bank_sha256"],
        "inputs": input_records,
        "report_script_sha256": _hash(Path(__file__)),
        "outputs": {p.name: _hash(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    (output / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-analysis", type=Path, required=True)
    parser.add_argument("--qwen-analysis", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--allow-fixture", action="store_true", help="Engineering fixtures only")
    args = parser.parse_args()
    try:
        manifest = write_report(
            args.llama_analysis,
            args.qwen_analysis,
            args.output_directory,
            allow_fixture=args.allow_fixture,
        )
    except (ValueError, FileExistsError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))
    print(
        f"Saved report: {args.output_directory}; test_fixture={manifest['test_fixture']}; "
        f"research_status={manifest['research_status']}; confirmatory_result=False"
    )


if __name__ == "__main__":
    main()
