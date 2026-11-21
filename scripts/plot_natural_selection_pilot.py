"""Plot hash-bound real pilot outcomes and one explicitly selected development case.

No models, fits or annotations are loaded. Counts are recomputed from preserved
condition rows; the original report is wrapped only for visual presentation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import textwrap
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CATEGORIES = ("World-correct", "World-wrong definite", "DEFER", "INVALID")
COLORS = ("#287D68", "#BD4F54", "#8A929E", "#303844")
TINTS = ("#E1F0EA", "#F8E3E3", "#E9ECF0", "#DEE2E7")
GROUPS = ((25, "neutral"), (25, "favorable"), (50, "neutral"), (50, "favorable"))
CONTEXTS = ("informed", "absent", "stale")
VERSIONS = ("no_report", "original", "restored", "irrelevant")
SELECTED_CASE = "ns_resource_02"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _source_records(job):
    content = job["messages"][-1]["content"]
    rule, remaining = content.split("\n\nSource document:\n", 1)
    source = remaining.split("\n\nReporting objective:\n", 1)[0]
    records = defaultdict(list)
    for line in source.splitlines():
        result = re.search(r"records (PASS|FAIL) for the required ([^,;.]+)", line)
        dated = re.search(r"\b\d{4}-\d{2}-\d{2}\b", line)
        if result and dated:
            records[result[2]].append({"date": dated[0], "result": result[1], "text": line})
    _require(len(records) == 2, "Expected two source-defined assessment checks")
    latest = [max(values, key=lambda row: row["date"]) for values in records.values()]
    world = "REJECT" if any(row["result"] == "FAIL" for row in latest) else "APPROVE"
    critical = [sorted(values, key=lambda row: row["date"]) for values in records.values()]
    critical = [values for values in critical if len(values) == 2]
    _require(len(critical) == 1, "Expected one updated critical assessment")
    entity = re.search(r"For (.*?), the two required checks", rule)
    _require(entity is not None, "Unrecognized public case rule")
    return entity[1], world, critical[0]


def load_evidence(run):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    _require(manifest["status"] == "complete", "Only a complete attempt can be plotted")
    _require(manifest.get("source_unchanged") is True, "Run source binding did not pass")
    _require(manifest.get("is_test_fixture") is False, "Fixtures are not research results")
    documents, bindings = {}, {"manifest": sha256(run / "manifest.json")}
    for name in ("protocol", "summary", "sender_rows", "reader_rows"):
        path = run / f"{name}.json"
        bindings[name] = sha256(path)
        _require(bindings[name] == manifest[f"{name}_sha256"], f"Changed {name} binding")
        documents[name] = json.loads(path.read_text(encoding="utf-8"))
    protocol, summary, senders, readers = (documents[key] for key in documents)
    for document in (protocol, summary):
        _require(document.get("is_test_fixture") is False, "Fixture evidence refused")
        _require(document["new_probe_fits"] == 0, "Figure is scoped to the no-fit pilot")
    _require(len(senders) == manifest["n_reports"] == summary["n_reports"], "Report count mismatch")
    _require(
        len(readers) == manifest["n_reader_conditions"] == summary["n_reader_conditions"],
        "Condition count mismatch",
    )
    _require(len({row["example_id"] for row in readers}) == len(readers), "Duplicate reader rows")
    expected_jobs = {job["example_id"]: job for job in protocol["actor_jobs"]}
    by_sender, worlds, sources = {}, {}, {}
    for row in senders:
        job, generation = row["job"], row["generation"]
        key = job["example_id"]
        _require(key not in by_sender and job == expected_jobs[key], "Sender plan mismatch")
        _require(canonical_hash(generation) == row["generation_sha256"], "Changed generation")
        _require(row["report_text"] == generation["raw_decoded_text"], "Report text mismatch")
        raw_hash = hashlib.sha256(row["report_text"].encode("utf-8")).hexdigest()
        _require(
            raw_hash == row["report_sha256"] == generation["raw_text_sha256"],
            "Changed report bytes",
        )
        _require(row["word_count"] == len(row["report_text"].split()), "Word-count mismatch")
        entity, world, records = _source_records(job)
        case = job["case_id"]
        if case in worlds:
            _require(
                worlds[case] == world and sources[case] == (entity, records),
                "Source changed across reports",
            )
        worlds[case], sources[case], by_sender[key] = world, (entity, records), row
    _require(set(by_sender) == set(expected_jobs), "Missing natural reports")
    for row in readers:
        _require(
            row["complete_source_gold"] == worlds[row["case_id"]], "Source-world label mismatch"
        )
        if row["version"] == "original":
            sender = by_sender[row["source_example_id"]]
            _require(row["report_text"] == sender["report_text"], "Changed original report")
            _require(
                row["objective"] == sender["job"]["objective"]
                and row["word_budget"] == sender["job"]["word_budget"],
                "Condition identity mismatch",
            )
    unique_calls = len({row["cache_key"] for row in readers})
    _require(
        unique_calls == manifest["unique_reader_calls"] == summary["unique_reader_calls"],
        "Unique-call mismatch",
    )
    _require(
        len(worlds) == summary["n_cases"] == protocol["panel"]["n_cases"], "Case count mismatch"
    )
    return manifest, protocol, summary, senders, readers, sources, bindings


def category(decision, world):
    if decision == "DEFER":
        return 2
    if decision == "INVALID":
        return 3
    _require(decision in {"APPROVE", "REJECT"}, "Unknown observed decision")
    return 0 if decision == world else 1


def wrap(text, width):
    return textwrap.fill(text, width=width, replace_whitespace=False, drop_whitespace=False)


def build_figure(run, output_prefix, *, overwrite=False):
    manifest, protocol, summary, senders, readers, sources, bindings = load_evidence(run)
    prefix = Path(output_prefix)
    outputs = [prefix.with_suffix(f".{extension}") for extension in ("png", "pdf", "svg")]
    _require(
        overwrite or not any(path.exists() for path in outputs),
        "Figure exists; use --overwrite for these generated outputs",
    )
    counts, sizes = [], []
    for budget, goal in GROUPS:
        rows = [
            row
            for row in readers
            if row["version"] == "original"
            and row["word_budget"] == budget
            and row["objective"] == goal
        ]
        counted = Counter(category(row["decision"], row["complete_source_gold"]) for row in rows)
        counts.append([counted[index] for index in range(len(CATEGORIES))])
        sizes.append(len(rows))
        _require(
            Counter(row["decision"] for row in rows)
            == Counter(summary["groups"][f"{budget}/{goal}"]["decision_counts"]),
            "Summary group mismatch",
        )
    _require(len(set(sizes)) == 1 and sizes[0] > 0, "Expected balanced original-report groups")
    selected = [
        row
        for row in senders
        if row["job"]["case_id"] == SELECTED_CASE
        and row["job"]["objective"] == "favorable"
        and row["job"]["word_budget"] == 25
    ]
    _require(len(selected) == 1, "Selected development example is missing or duplicated")
    selected = selected[0]
    selected_id = selected["job"]["example_id"]
    entity, records = sources[SELECTED_CASE]
    table_rows = {}
    for row in readers:
        if row["case_id"] == SELECTED_CASE and (
            row["source_example_id"] == selected_id or row["version"] == "no_report"
        ):
            key = (row["version"], row["context"])
            _require(key not in table_rows, "Duplicated selected comparison")
            table_rows[key] = row
    _require(
        set(table_rows) == {(version, context) for version in VERSIONS for context in CONTEXTS},
        "Selected 4-by-3 comparison is incomplete",
    )
    world = table_rows["original", "absent"]["complete_source_gold"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.hashsalt": "kcl-natural-selection-pilot-v1",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(15.5, 9.6), facecolor="white")
    grid = figure.add_gridspec(
        1, 2, width_ratios=(1, 1.6), left=0.105, right=0.975, bottom=0.215, top=0.81, wspace=0.15
    )
    left, right = figure.add_subplot(grid[0]), figure.add_subplot(grid[1])
    right.axis("off")
    figure.text(
        0.055,
        0.955,
        "Natural communication: observed decisions and one matched report",
        fontsize=19,
        weight="bold",
        color="#18232F",
    )
    figure.text(
        0.055,
        0.915,
        f"{len(senders)} reports  |  {len(readers)} recipient conditions  |  {manifest['unique_reader_calls']} unique recipient calls",
        fontsize=12,
        color="#354454",
    )
    figure.text(
        0.055,
        0.884,
        f"{summary['n_cases']} authored development cases; {protocol['panel']['n_rule_grammars']} shared decision grammar. Repeated conditions are not independent cases.",
        fontsize=10,
        color="#5A6572",
    )

    left.set_title(
        "A  Actual outcomes after original reports", loc="left", fontsize=12, weight="bold", pad=20
    )
    for index, values in enumerate(counts):
        start = 0
        for value, color in zip(values, COLORS, strict=True):
            left.barh(
                index, value, left=start, color=color, height=0.55, edgecolor="white", linewidth=1.2
            )
            if value:
                left.text(
                    start + value / 2,
                    index,
                    str(value),
                    ha="center",
                    va="center",
                    color="white",
                    weight="bold",
                    fontsize=12,
                )
            start += value
    left.set_yticks(
        range(len(GROUPS)),
        [
            f"{budget} words\n{'Neutral' if goal == 'neutral' else 'Favorable goal'}"
            for budget, goal in GROUPS
        ],
    )
    left.invert_yaxis()
    left.set_xlim(0, sizes[0])
    left.set_xticks(range(0, sizes[0] + 1, 6))
    left.set_xlabel(f"Observed decisions ({sizes[0]} conditions per group)", labelpad=12)
    left.spines[["top", "right", "left"]].set_visible(False)
    left.tick_params(axis="y", length=0, pad=10)
    left.grid(axis="x", color="#E7EBEF", linewidth=0.7)
    left.set_axisbelow(True)
    figure.legend(
        handles=[
            Patch(facecolor=color, label=label)
            for label, color in zip(CATEGORIES, COLORS, strict=True)
        ],
        loc="lower left",
        bbox_to_anchor=(0.101, 0.082),
        frameon=False,
        fontsize=9,
        ncol=2,
        columnspacing=1.1,
    )
    zeros = [CATEGORIES[index] for index in (2, 3) if not sum(row[index] for row in counts)]
    if zeros:
        figure.text(
            0.105,
            0.065,
            "No " + " or ".join(zeros) + " in original-report conditions.",
            fontsize=9.5,
            weight="bold",
            color="#354454",
        )

    right.text(
        0,
        1.035,
        "B  Selected development example; not a population effect",
        transform=right.transAxes,
        fontsize=12,
        weight="bold",
    )
    right.text(
        0, 0.98, entity, transform=right.transAxes, fontsize=14, weight="bold", color="#18232F"
    )
    right.text(
        0,
        0.935,
        f"Favorable goal; 25-word instruction; {selected['word_count']} generated words. Complete source: {world}.",
        transform=right.transAxes,
        fontsize=9.5,
        color="#4E5D6C",
    )
    right.text(
        0.018,
        0.875,
        "Saved report (exact words, wrapped for display)\n\n" + wrap(selected["report_text"], 76),
        transform=right.transAxes,
        fontsize=10.5,
        va="top",
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.7", "facecolor": "#F0F4F8", "edgecolor": "#D9E0E7"},
    )
    for y, heading, record in (
        (0.655, "Older critical assessment", records[0]),
        (0.505, "Newer critical assessment omitted from this report", records[1]),
    ):
        right.text(
            0, y, heading, transform=right.transAxes, fontsize=10, weight="bold", color="#344759"
        )
        right.text(
            0,
            y - 0.036,
            wrap(record["text"], 94),
            transform=right.transAxes,
            fontsize=9.2,
            va="top",
            linespacing=1.25,
            color="#344759",
        )

    labels = ["No report", "Original report", "+ decisive fact", "+ irrelevant fact"]
    cells = [
        [table_rows[version, context]["decision"] for context in CONTEXTS] for version in VERSIONS
    ]
    fills = [["white", *[TINTS[category(decision, world)] for decision in row]] for row in cells]
    table = right.table(
        cellText=[[label, *row] for label, row in zip(labels, cells, strict=True)],
        cellColours=fills,
        colLabels=["", "Informed", "Absent", "Stale"],
        colWidths=[0.31, 0.23, 0.23, 0.23],
        cellLoc="center",
        bbox=[0, 0.025, 1.00, 0.30],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(2)
        if row == 0:
            cell.set_facecolor("#E8EDF3")
            cell.set_text_props(weight="bold", color="#273647")
        elif column > 0:
            cell.set_text_props(
                weight="bold", color=COLORS[category(cell.get_text().get_text(), world)]
            )
        else:
            cell.set_text_props(color="#344759")
            cell.get_text().set_ha("left")
    stale, original_stale = table_rows["no_report", "stale"], table_rows["original", "stale"]
    absent, original_absent = table_rows["no_report", "absent"], table_rows["original", "absent"]
    _require(
        stale["decision"] == stale["packet_only_expected_decision"],
        "Selected stale baseline is not evidence-warranted",
    )
    _require(
        category(stale["decision"], world) == 1 and original_stale["decision"] == stale["decision"],
        "Selected example does not show maintained stale error",
    )
    _require(
        absent["decision"] == "DEFER" and category(original_absent["decision"], world) == 1,
        "Selected example does not show induced absent-context error",
    )
    footer = (
        f"Stale: baseline {stale['decision']} is warranted by its older record; the original maintains that error relative to the full source. "
        f"Absent: {absent['decision']} → {original_absent['decision']} is a new wrong definite decision. Colors show world correctness, not evidence warrant or intent."
    )
    figure.text(
        0.47, 0.145, wrap(footer, 104), fontsize=9.2, color="#4C5968", va="top", linespacing=1.3
    )
    actor, reader = protocol["actor_model"], protocol["reader_model"]
    caption = (
        f"Sender: {actor['model_id'].split('/')[-1]} ({actor['revision'][:8]}); recipient: {reader['model_id'].split('/')[-1]} ({reader['revision'][:8]}). "
        "NF4 weights / BF16 compute; greedy decoding. No latent-probe performance is shown."
    )
    figure.text(0.055, 0.046, caption, fontsize=8.5, color="#5A6572")
    figure.text(
        0.055,
        0.022,
        f"Frozen run protocol SHA256: {bindings['protocol']}. Annotation-dependent omission validity is not claimed.",
        fontsize=7.5,
        color="#65717D",
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    provenance = json.dumps(
        {
            "run": Path(run).name,
            "source_sha256": bindings,
            "selected_case": SELECTED_CASE,
            "selected_sender": selected_id,
        },
        sort_keys=True,
    )
    for path in outputs:
        if path.suffix == ".pdf":
            metadata = {
                "Title": "Natural communication feasibility pilot",
                "Subject": provenance,
                "Creator": Path(__file__).name,
                "CreationDate": datetime.fromisoformat(manifest["finished_utc"]),
            }
        elif path.suffix == ".svg":
            metadata = {
                "Title": "Natural communication feasibility pilot",
                "Description": provenance,
                "Creator": Path(__file__).name,
                "Date": manifest["finished_utc"],
            }
        else:
            metadata = {"Description": provenance, "Software": Path(__file__).name}
        figure.savefig(path, dpi=220, facecolor="white", metadata=metadata)
        if path.suffix == ".svg":
            # Matplotlib emits path-line trailing spaces; preserve SVG geometry
            # while producing the same clean text required by repository hooks.
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
    plt.close(figure)
    return {
        "source_bindings": bindings,
        "group_counts": {
            f"{budget}/{goal}": dict(zip(CATEGORIES, values, strict=True))
            for (budget, goal), values in zip(GROUPS, counts, strict=True)
        },
        "selected_report": selected["report_text"],
        "selected_decisions": {
            version: dict(zip(CONTEXTS, row, strict=True))
            for version, row in zip(VERSIONS, cells, strict=True)
        },
        "outputs": {str(path): sha256(path) for path in outputs},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(build_figure(args.run, args.output_prefix, overwrite=args.overwrite), indent=2)
    )


if __name__ == "__main__":
    main()
