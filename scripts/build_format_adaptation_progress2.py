"""Reporting-only four-slide Progress 2 export from a completed audited report.

No model calls, fitting or analysis. The public build and CLI reject fixtures.
Copied reports are accepted by byte hashes and the original canonical campaign
identity, without requiring native Linux source paths to exist on Windows.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import posixpath
import tempfile
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "scripts/update_progress2.py"
REPORTER = ROOT / "scripts/report_format_adaptation.py"
TEMPLATE = ROOT / "presentations/templates/progress2-original.pptx"
PROJECT = "Latent Probes and Listener Beliefs"
ARMS = ("A512", "Mixed512", "Ahalf256", "Offset")
SITES = {"llama": "layer_16_query_location_last", "qwen": "layer_21_query_location_last"}
DEFAULT_OUTPUT = ROOT / "presentations/progress2-format-adaptation-2026-11-27"


def require(value, message):
    if not value:
        raise ValueError(message)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_path(path):
    value = posixpath.normpath(str(path).replace("\\", "/"))
    if len(value) > 1 and value[1] == ":":
        value = value[0].upper() + value[1:]
    return value


def hash_map(values):
    require(isinstance(values, dict) and values, "Missing source hash map")
    result = {}
    for key, value in values.items():
        name = canonical_path(key)
        require(
            name not in result
            and isinstance(value, str)
            and len(value) == 64
            and all(c in "0123456789abcdef" for c in value),
            "Invalid or duplicate hash binding",
        )
        result[name] = value
    return result


def number(value, minimum=-1.0, maximum=1.0):
    require(
        type(value) in (int, float) and math.isfinite(value) and minimum <= value <= maximum,
        "Nonfinite or out-of-range metric",
    )
    return float(value)


def count(value, maximum):
    require(type(value) is int and 0 <= value <= maximum, "Invalid measured count")
    return value


def analysis_bytes(model):
    """Reproduce storage.atomic_json's frozen native LF serialization exactly."""
    return (json.dumps(model, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def validate_model(model, key, fixture):
    require(
        model.get("schema") == "format-adaptation-analysis-v1"
        and model.get("model_key") == key
        and model.get("site") == SITES[key]
        and model.get("test_fixture") is fixture
        and model.get("confirmatory") is False
        and model.get("research_status") == "agent_audited_exploratory",
        "Wrong model provenance",
    )
    require(set(model.get("arms", {})) == set(ARMS), "All four arms are required")
    all_groups = None
    for arm in ARMS:
        for fmt in ("A", "B"):
            scope = model["arms"][arm][fmt + "/correction_hard"]
            aggregate = scope["aggregate"]
            require(
                aggregate["rows"] == 768 and aggregate["quartets"] == 192,
                "Need 768 rows and 192 hard quartets",
            )
            completed = count(aggregate["complete_quartets"], 192)
            require(
                math.isclose(
                    number(aggregate["quartet_accuracy"], 0), completed / 192, abs_tol=1e-12
                ),
                "Quartet score/count mismatch",
            )
            correct = count(aggregate["correct"], 768)
            require(
                math.isclose(number(aggregate["row_accuracy"], 0), correct / 768, abs_tol=1e-12),
                "Row score/count mismatch",
            )
            groups = scope["groups"]
            require(
                len(groups) == 24 and len({g["group_id"] for g in groups}) == 24,
                "Need all 24 context groups",
            )
            ids = {g["group_id"] for g in groups}
            require(all_groups is None or ids == all_groups, "Arm context identities differ")
            all_groups = ids
            for g in groups:
                require(g["rows"] == 32 and g["quartets"] == 8, "Context denominator mismatch")
                n = count(g["complete_quartets"], 8)
                require(
                    math.isclose(number(g["quartet_accuracy"], 0), n / 8, abs_tol=1e-12),
                    "Context quartet score/count mismatch",
                )
                c = count(g["correct"], 32)
                require(
                    math.isclose(number(g["row_accuracy"], 0), c / 32, abs_tol=1e-12),
                    "Context row score/count mismatch",
                )
            require(
                sum(g["complete_quartets"] for g in groups) == completed
                and sum(g["correct"] for g in groups) == correct,
                "Aggregate/context count mismatch",
            )
    for name, comparator in (("primary", "A512"), ("named_diagnostic", "Offset")):
        contrast = model[name]
        require(
            contrast.get("contrast") == "Mixed512_minus_" + comparator
            and contrast.get("groups") == 24
            and contrast.get("quartets") == 192
            and contrast.get("bootstrap_samples") == 10000
            and contrast.get("bootstrap_seed") == 20261127
            and contrast.get("renderer") == "B"
            and contrast.get("condition") == "correction_hard",
            "Wrong paired contrast inventory",
        )
        values = {
            arm: model["arms"][arm]["B/correction_hard"]["aggregate"]["quartet_accuracy"]
            for arm in ("Mixed512", comparator)
        }
        for arm, value in values.items():
            require(
                math.isclose(number(contrast[arm], 0), value, abs_tol=1e-12),
                "Contrast arm score mismatch",
            )
        difference = number(contrast["difference"])
        require(
            math.isclose(difference, values["Mixed512"] - values[comparator], abs_tol=1e-12),
            "Contrast difference mismatch",
        )
        interval = contrast["conditional_percentile_interval_95"]
        require(
            isinstance(interval, list)
            and len(interval) == 2
            and number(interval[0]) <= number(interval[1]),
            "Invalid or reversed interval",
        )
        paired = contrast["group_values"]
        require(
            len(paired) == 24 and {g["group_id"] for g in paired} == all_groups,
            "Missing paired context values",
        )
        for item in paired:
            scores = {}
            for arm in ("Mixed512", comparator):
                source = next(
                    g
                    for g in model["arms"][arm]["B/correction_hard"]["groups"]
                    if g["group_id"] == item["group_id"]
                )
                scores[arm] = source["quartet_accuracy"]
                require(
                    math.isclose(number(item[arm], 0), scores[arm], abs_tol=1e-12),
                    "Paired context score mismatch",
                )
            require(
                math.isclose(
                    number(item["difference"]),
                    scores["Mixed512"] - scores[comparator],
                    abs_tol=1e-12,
                ),
                "Paired context difference mismatch",
            )
    require(
        model["primary"]["role"]
        == ("primary" if key == "llama" else "named_secondary_replication"),
        "Primary/secondary role changed",
    )
    return all_groups


def _load_sources(summary_path, manifest_path, audit_path, *, allow_fixture=False):
    paths = [Path(p).resolve() for p in (summary_path, manifest_path, audit_path)]
    summary, manifest, audit = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    require(
        summary.get("schema") == "format-adaptation-report-v1"
        and manifest.get("schema") == "format-adaptation-report-manifest-v1"
        and manifest.get("status") == "complete",
        "Completed adaptation report required",
    )
    fixture = summary.get("test_fixture")
    require(
        type(fixture) is bool
        and (allow_fixture or fixture is False)
        and summary.get("human_review_complete") is False
        and summary.get("research_status") == "agent_audited_exploratory"
        and summary.get("new_report_model_calls") == summary.get("classifier_fits") == 0,
        "Only real exploratory human-incomplete reports accepted",
    )
    require(manifest.get("source_sha256") == file_hash(REPORTER), "Reporter source hash mismatch")
    require(
        manifest.get("output_sha256", {}).get("summary.json") == file_hash(paths[0]),
        "Summary hash mismatch",
    )
    inputs = hash_map(summary.get("input_sha256"))
    require(inputs == hash_map(manifest.get("input_sha256")), "Report input maps differ")
    require(
        audit.get("schema") == "format-adaptation-independent-audit-v1"
        and audit.get("status") == "pass"
        and audit.get("model_calls") == audit.get("classifier_fits") == 0
        and audit.get("saved_analysis_opened_before_both_reconstructions_frozen") is False
        and set(audit.get("models", {})) == {"llama", "qwen"},
        "Passing independent both-model audit required",
    )
    campaign = canonical_path(audit.get("campaign", ""))
    require(
        campaign not in ("", ".")
        and (campaign.startswith("/") or (len(campaign) > 1 and campaign[1] == ":")),
        "Audit campaign identity missing",
    )
    audit_bindings = [key for key in inputs if posixpath.basename(key) == "audit_receipt.json"]
    require(
        len(audit_bindings) == 1 and inputs[audit_bindings[0]] == file_hash(paths[2]),
        "Audit receipt hash mismatch",
    )
    audited = hash_map(audit.get("analysis_output_sha256"))
    require(set(summary.get("models", {})) == {"llama", "qwen"}, "Both model reports required")
    shared = None
    for key, model in summary["models"].items():
        result_path = campaign + "/" + key + "_analysis/results.json"
        prediction_path = campaign + "/" + key + "_analysis/predictions.npz"
        require(
            result_path in audited
            and prediction_path in audited
            and inputs.get(result_path) == audited[result_path],
            "Wrong campaign or unaudited model result",
        )
        require(
            hashlib.sha256(analysis_bytes(model)).hexdigest() == audited[result_path],
            "Embedded model differs from exact audited analysis bytes",
        )
        groups = validate_model(model, key, fixture)
        require(shared is None or shared == groups, "Models do not share the same 24 contexts")
        shared = groups
    bindings = {
        label: {"path": str(path), "sha256": file_hash(path)}
        for label, path in zip(("summary", "manifest", "audit"), paths, strict=True)
    }
    bindings["campaign_identity"] = {"path": campaign, "sha256": None}
    return summary, bindings


def load_sources(summary_path, manifest_path, audit_path):
    return _load_sources(summary_path, manifest_path, audit_path)


@lru_cache(maxsize=1)
def _layout():
    spec = importlib.util.spec_from_file_location("format_adaptation_progress_style", RENDERER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _slides(summary, bindings):
    style = _layout()
    fixture = summary["test_fixture"]
    status = (
        "SYNTHETIC FIXTURE - NO RESEARCH RESULT"
        if fixture
        else "AGENT-AUDITED EXPLORATORY - HUMAN REVIEW INCOMPLETE"
    )
    provenance = "\n\nReporting inputs (no new inference or analysis):\n" + json.dumps(
        bindings, indent=2
    )

    def footer(slide, index):
        slide.line(0.65, 6.92, 12.68, 6.92, "#C3D0D2")
        slide.text("PROGRESS 02 | " + status, 0.65, 7.08, 11.1, 8.3, style.MUTED)
        slide.text(f"{index} / 4", 11.98, 7.08, 0.7, 9, style.MUTED)
        if fixture:
            slide.rect(0.65, 6.43, 12.03, 0.34, style.GOLD)
            slide.text(status, 0.85, 6.47, 11.6, 12, style.INK, True, max_height=0.25)

    def header(label, title, index):
        slide = style.Slide()
        slide.text("KING'S COLLEGE LONDON / 6CCSAPRJ", 0.65, 0.35, 12, 10, style.MUTED, True)
        slide.text(label, 0.65, 0.84, 12, 11, style.TEAL, True)
        slide.text(title, 0.65, 1.2, 12.03, 28, style.INK, True, max_height=0.66)
        footer(slide, index)
        return slide

    title = style.Slide(style.NAVY)
    title.text("KING'S COLLEGE LONDON / 6CCSAPRJ", 0.7, 0.42, 12, 10, "#B7CBCF", True)
    title.text("PROGRESS 2 | FRESH FORMAT ADAPTATION", 0.7, 1.02, 12, 12, style.GOLD, True)
    title.text(
        "Latent Probes and\nListener Beliefs", 0.7, 1.78, 11.6, 40, style.WHITE, True, leading=1.12
    )
    title.text(
        "Testing whether supervised readouts of sender-attributed reader beliefs\nremain reliable across prompt formats.",
        0.73,
        3.65,
        11.55,
        22,
        "#DFE9E9",
        max_height=1.05,
    )
    title.text("Aditya Ranjan", 0.73, 5.35, 11, 23, style.WHITE, True)
    title.text("Individual Project | completed fixed comparison", 0.73, 5.95, 11.5, 16, "#DFE9E9")
    footer(title, 1)
    title.notes = (
        "This update reports a completed, independently audited fixed-format adaptation comparison. "
        "The target is the fictional sender's attribution of the reader's current location belief. "
        "It is not a measurement of spontaneous anticipation or deception. Human semantic review remains incomplete."
        + provenance
    )

    progress = header("WHAT HAS BEEN ACHIEVED", "Completed fixed-format adaptation comparison", 2)
    progress.text(
        "FORMAT B / HARD: all four receipt cases correct", 0.8, 2.02, 11.8, 16, style.TEAL, True
    )
    x = (3.15, 5.55, 7.95, 10.35)
    for position, arm in zip(x, ARMS, strict=True):
        progress.text(arm, position, 2.47, 2.1, 15, style.MUTED, True)
    for row_index, key in enumerate(("llama", "qwen")):
        y = 2.94 + row_index * 0.68
        progress.rect(0.72, y - 0.08, 11.9, 0.59, style.PALE)
        progress.text(
            "Llama (primary)" if key == "llama" else "Qwen (secondary)",
            0.9,
            y,
            2.15,
            16,
            style.TEAL,
            True,
        )
        for position, arm in zip(x, ARMS, strict=True):
            value = summary["models"][key]["arms"][arm]["B/correction_hard"]["aggregate"][
                "complete_quartets"
            ]
            progress.text(f"{value}/192", position, y - 0.01, 2.1, 23, style.INK, True)
        p = summary["models"][key]["primary"]
        low, high = p["conditional_percentile_interval_95"]
        progress.text(
            f"{key.title()} Mixed - A512: {100 * p['difference']:+.2f} pp; 95% CI [{100 * low:+.2f}, {100 * high:+.2f}] pp",
            0.85,
            4.4 + row_index * 0.4,
            11.8,
            16,
            style.INK,
            max_height=0.32,
        )
    for row_index, key in enumerate(("llama", "qwen")):
        values = " | ".join(
            str(
                summary["models"][key]["arms"][arm]["A/correction_hard"]["aggregate"][
                    "complete_quartets"
                ]
            )
            + "/192"
            for arm in ARMS
        )
        progress.text(
            f"A control, {key.title()} (same arm order): {values}",
            0.85,
            5.44 + row_index * 0.35,
            11.8,
            14,
            style.MUTED,
            max_height=0.30,
        )
    progress.text(
        "24 shared contexts per model; intervals condition on the fixed fits and authored task family.",
        0.85,
        6.1,
        11.7,
        12,
        style.MUTED,
        max_height=0.25,
    )
    progress.notes = (
        "All counts use complete B1-by-B2 receipt quartets, not independent row accuracy. "
        "The primary is Llama Mixed512 minus A512 in B hard correction; Qwen is a named secondary replication. "
        "The 95% percentile intervals resample the same 24 paired contexts 10,000 times at seed 20261127. "
        "They exclude uncertainty from resampling the fitting cohort. A512 and Mixed512 use the same 512 formal states; "
        "Mixed uses 256 A plus 256 B observations. Ahalf uses 256 A; Offset freezes those coefficients and fits only B intercepts. "
        "Mixed versus Offset compares these procedures and cannot uniquely identify geometric rotation or internal mechanism. "
        "Both format controls and every fixed arm remain visible regardless of the effect's sign.\n\nSaved contrasts:\n"
        + json.dumps(
            {
                key: {name: model[name] for name in ("primary", "named_diagnostic")}
                for key, model in summary["models"].items()
            },
            indent=2,
        )
        + provenance
    )

    barriers = header("CURRENT BARRIERS", "What the result can establish remains bounded", 3)
    items = (
        (
            "Human review",
            "Human semantic validation is missing; agent agreement does not establish human acceptance.",
        ),
        (
            "Measurement",
            "Privileged factor supervision + an external Boolean rule measure supervised accessibility.",
        ),
        (
            "Transfer scope",
            "NF4 weights, two known formats and one author pool; shared templates limit generalisation.",
        ),
        (
            "Research claim",
            "No spontaneous anticipation, deception detection or model-internal causal algorithm is established.",
        ),
    )
    for index, (label, body) in enumerate(items):
        y = 2.12 + index * 1.03
        if index % 2 == 0:
            barriers.rect(0.7, y - 0.08, 11.95, 0.93, style.PALE)
        barriers.text(label, 0.9, y + 0.08, 2.0, 18, style.TEAL, True, max_height=0.65)
        barriers.text(body, 3.05, y + 0.04, 9.25, 18, style.INK, max_height=0.82)
    barriers.notes = (
        "The common frozen source-selected site, old scaler and regularization convention reduce adaptation confounds. "
        "They do not turn decodability into construct validity. The supplied four-factor rule is external to the model. "
        "A and B bundle system, narrative and question differences; this experiment does not isolate one prompt component. "
        "The new contextual groups share formal rules and an author pool. Local NF4 measurements do not establish BF16 robustness. "
        "Preserve the complete failed-control inventory and keep agent review distinct from human acceptance."
        + provenance
    )

    next_steps = header(
        "NEXT STEPS / OVERALL PLAN", "Turn the fixed result into a scoped research argument", 4
    )
    plans = (
        (
            "01",
            "Supervisor scope + human review",
            "Agree the final claim; seek human semantic review and preserve failed controls.",
        ),
        (
            "02",
            "BSPR + dissertation evidence",
            "Integrate the result, uncertainty, all arms and limitations into the overall project plan.",
        ),
        (
            "03",
            "Proposed CREATE BF16 replication",
            "Freeze a matched BF16 replication on CREATE. Proposed, not performed in this update.",
        ),
    )
    for index, (label, heading, body) in enumerate(plans):
        y = 2.05 + index * 1.18
        next_steps.text(label, 0.8, y, 0.6, 23, style.TEAL, True)
        next_steps.text(heading, 1.6, y, 10.6, 21, style.INK, True, max_height=0.4)
        next_steps.text(body, 1.6, y + 0.47, 10.6, 17, style.MUTED, max_height=0.65)
    next_steps.rect(0.75, 5.93, 11.8, 0.4, style.PALE)
    next_steps.text(
        "PLAN: measurement validity > controlled transfer > scoped communication claims",
        0.95,
        6.0,
        11.35,
        13,
        style.TEAL,
        True,
        max_height=0.27,
    )
    next_steps.notes = (
        "The immediate deliverable is the empirical argument for BSPR and the dissertation, with supervisor agreement on scope. "
        "Later human review cannot retroactively make exposed outcomes confirmatory. A BF16 replication on CREATE is a proposal: "
        "freeze checkpoints, contexts, captures, heads, penalty convention and endpoint before another matched precision run. "
        "It has not been performed in this update. Broader communication or spontaneous-anticipation claims require a separately designed study, "
        "not a stronger label for these measurements. No scheduled date or additional experiment is claimed complete here."
        + provenance
    )
    return [title, progress, barriers, next_steps]


def _build(summary_path, manifest_path, audit_path, output, template=None, *, allow_fixture=False):
    output, template = Path(output).resolve(), Path(template or TEMPLATE).resolve()
    if output.exists():
        raise FileExistsError("Preserve earlier decks; output directory already exists")
    summary, bindings = _load_sources(
        summary_path, manifest_path, audit_path, allow_fixture=allow_fixture
    )
    source_files = [Path(__file__).resolve(), RENDERER, REPORTER, template]
    sources = {str(p): file_hash(p) for p in source_files}
    slides = _slides(summary, bindings)
    require(len(slides) == 4, "Exactly four slides required")
    with tempfile.TemporaryDirectory(prefix="format-adaptation-progress2-") as temporary:
        directory = Path(temporary)
        stem = (
            "SYNTHETIC-FIXTURE-ONLY" if summary["test_fixture"] else "Progress-2-format-adaptation"
        )
        pptx, pdf = directory / (stem + ".pptx"), directory / (stem + ".pdf")
        _layout()._write_deck(template, pptx, pdf, slides, PROJECT)
        for name, expected in sources.items():
            require(
                file_hash(name) == expected, "Renderer/template/source changed during generation"
            )
        for label, binding in bindings.items():
            if label != "campaign_identity":
                require(
                    file_hash(binding["path"]) == binding["sha256"],
                    "Report input changed during generation",
                )
        output.mkdir(parents=True, exist_ok=False)
        for source in (pptx, pdf):
            (output / source.name).write_bytes(source.read_bytes())
        notes = "# Progress 2 speaker notes\n\n" + "\n\n".join(
            f"## Slide {i}\n\n{s.notes}" for i, s in enumerate(slides, 1)
        )
        if summary["test_fixture"]:
            notes = "SYNTHETIC FIXTURE - NO RESEARCH RESULT\n\n" + notes
        (output / "speaker-notes.md").write_text(
            notes.rstrip() + "\n", encoding="utf-8", newline="\n"
        )
    receipt = {
        "schema": "format-adaptation-progress2-receipt-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "slides": 4,
        "test_fixture": summary["test_fixture"],
        "model_calls": 0,
        "classifier_fits": 0,
        "analysis_performed": False,
        "human_review_complete": False,
        "inputs": bindings,
        "source_sha256": sources,
        "outputs": {p.name: file_hash(p) for p in sorted(output.iterdir())},
        "pdf_renderer": "ReportLab vector text/shapes with the unchanged editable PPTX geometry",
    }
    (output / "source-receipt.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    return receipt


def build(summary_path, manifest_path, audit_path, output=DEFAULT_OUTPUT, template=None):
    return _build(summary_path, manifest_path, audit_path, output, template)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("summary", "manifest", "audit"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", "--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--template", type=Path)
    args = parser.parse_args()
    try:
        result = build(args.summary, args.manifest, args.audit, args.output, args.template)
    except (ValueError, FileExistsError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps({"slides": result["slides"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
