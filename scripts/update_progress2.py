"""Build the measured Progress 2 deck without changing the original draft.

Requires the existing presentation environment's python-pptx and reportlab.
No model calls, package installation, or remote services are used.

Example optional results JSON (a completed neutral_readout must provide its own
headline/detail; measured natural-pilot counts are required together)::

    {
      "baseline_examples": 536,
      "baseline_instruction_auroc": 1.0,
      "software_tests": 176,
      "software_subtests": 48,
      "reader_calls": 48,
      "fixed_replay_calls": 48,
      "fixed_replay_original_pairs": 12,
      "fixed_replay_pressure_positive_pairs": 12,
      "fixed_replay_mean_pressure_delta": 0.8121,
      "natural_reports": 84,
      "neutral_reports": 72,
      "pressure_reports": 12,
      "natural_reader_calls": 84,
      "test_cases": 4,
      "test_rows_per_condition": 12,
      "forecast_neutral_correct": 10,
      "forecast_pressure_correct": 10,
      "lexical_neutral_correct": 8,
      "lexical_pressure_correct": 8,
      "copy_neutral_correct": 7,
      "copy_pressure_correct": 11,
      "text_forecast_neutral_correct": 8,
      "text_forecast_pressure_correct": 9,
      "neutral_readout": {
        "status": "complete",
        "headline": "Neutral readout trained; text shortcut exposed",
        "detail": "Four held-out cases; added value beyond report text is not established.",
        "metrics": {}
      }
    }

The defaults are a dated, already measured snapshot, not a live test count.
Unprovided future results stay pending.  Additional JSON metrics appear only in
speaker notes; the caller must supply the intended measured interpretation.

Usage from the repository, using an environment with the presentation deps::

    python scripts/update_progress2.py --results presentations/progress2/results.json

build_progress2(results_path=None, template_path=None, pptx_output=None,
                pdf_output=None) also supports an in-process call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.util import Inches, Pt
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

REPO = Path(__file__).resolve().parents[1]
PRESENTATIONS = REPO / "presentations"
DEFAULT_TEMPLATE = PRESENTATIONS / "templates" / "progress2-original.pptx"
DEFAULT_PPTX = PRESENTATIONS / "progress2" / "Progress-2-measured-results.pptx"
DEFAULT_PDF = PRESENTATIONS / "progress2" / "Progress-2-measured-results.pdf"
NAVY = "#152638"
TEAL = "#006D77"
GOLD = "#E8B777"
PAPER = "#F7F5EF"
INK = "#172839"
MUTED = "#536575"
PALE = "#E8EEEA"
WHITE = "#FFFFFF"
WIDTH = 13.333
HEIGHT = 7.5

DEFAULT_RESULTS: dict[str, Any] = {
    "snapshot_date": "19 November 2026",
    "progress_date": "20 November 2026",
    "baseline_examples": 536,
    "baseline_instruction_auroc": 1.0,
    "software_tests": 176,
    "software_subtests": 48,
    "reader_calls": 48,
    "fixed_replay_calls": 48,
    "fixed_replay_original_pairs": 12,
    "fixed_replay_pressure_positive_pairs": 12,
    "fixed_replay_mean_pressure_delta": 0.8121,
    "neutral_readout": {
        "status": "planned",
        "headline": "Neutral reader forecast: next empirical gate",
        "detail": "Pipeline in preparation; no transfer result is reported.",
        "metrics": {},
    },
}

PILOT_COUNTS = (
    "natural_reports",
    "neutral_reports",
    "pressure_reports",
    "natural_reader_calls",
    "test_cases",
    "test_rows_per_condition",
    "forecast_neutral_correct",
    "forecast_pressure_correct",
    "lexical_neutral_correct",
    "lexical_pressure_correct",
    "copy_neutral_correct",
    "copy_pressure_correct",
    "text_forecast_neutral_correct",
    "text_forecast_pressure_correct",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_results(path: Path | None) -> dict[str, Any]:
    result = json.loads(json.dumps(DEFAULT_RESULTS))
    if path is not None:
        supplied = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(supplied, dict):
            raise ValueError("Results must be a JSON object.")
        neutral = supplied.get("neutral_readout")
        if neutral is not None and not isinstance(neutral, dict):
            raise ValueError("neutral_readout must be an object.")
        result.update({key: value for key, value in supplied.items() if key != "neutral_readout"})
        if neutral is not None:
            if neutral.get("status") == "complete" and not all(
                isinstance(neutral.get(key), str) and neutral[key].strip()
                for key in ("headline", "detail")
            ):
                raise ValueError("A completed readout requires its measured headline and detail.")
            result["neutral_readout"].update(neutral)
    count_keys = (
        "baseline_examples",
        "software_tests",
        "software_subtests",
        "reader_calls",
        "fixed_replay_calls",
        "fixed_replay_original_pairs",
        "fixed_replay_pressure_positive_pairs",
    )
    for key in count_keys:
        if type(result[key]) is not int or result[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer.")
    pairs = result["fixed_replay_original_pairs"]
    if result["fixed_replay_pressure_positive_pairs"] > pairs:
        raise ValueError("Positive matched pairs cannot exceed all matched pairs.")
    auroc = result["baseline_instruction_auroc"]
    delta = result["fixed_replay_mean_pressure_delta"]
    if not isinstance(auroc, (int, float)) or not math.isfinite(auroc) or not 0 <= auroc <= 1:
        raise ValueError("baseline_instruction_auroc must lie in [0, 1].")
    if not isinstance(delta, (int, float)) or not math.isfinite(delta):
        raise ValueError("fixed_replay_mean_pressure_delta must be finite.")
    readout = result["neutral_readout"]
    if readout["status"] not in {"planned", "running", "complete"}:
        raise ValueError("neutral_readout.status must be planned, running, or complete.")
    for key, limit in (("headline", 115), ("detail", 180)):
        text = readout[key]
        if not isinstance(text, str) or not text.strip() or len(text) > limit:
            raise ValueError(f"neutral_readout.{key} must be nonempty and at most {limit} chars.")
    if not isinstance(readout.get("metrics", {}), dict):
        raise ValueError("neutral_readout.metrics must be an object.")
    if "natural_reports" in result:
        for key in PILOT_COUNTS:
            if key not in result or type(result[key]) is not int or result[key] < 0:
                raise ValueError(f"A measured natural pilot requires explicit nonnegative {key}.")
        if result["natural_reports"] != result["neutral_reports"] + result["pressure_reports"]:
            raise ValueError("Natural report count must equal neutral plus pressure reports.")
        if result["natural_reader_calls"] != result["natural_reports"]:
            raise ValueError("This deck assumes one measured reader label per natural report.")
        n_test = result["test_rows_per_condition"]
        if n_test <= 0 or result["test_cases"] <= 0:
            raise ValueError("The measured forecast requires nonempty test cases and rows.")
        for key in (key for key in PILOT_COUNTS if key.endswith("_correct")):
            if result[key] > n_test:
                raise ValueError(f"{key} cannot exceed test_rows_per_condition.")
        if readout["status"] != "complete":
            raise ValueError("Supply measured natural-pilot counts only with a completed readout.")
    if "known_message" in result:
        known = result["known_message"]
        mask = result.get("mask_specificity")
        if not isinstance(known, dict) or not isinstance(mask, dict):
            raise ValueError("The latest milestone requires known_message and mask_specificity.")
        known_counts = (
            "n_rows",
            "n_unique_inputs",
            "n_wording_groups",
            "queried_correct",
            "story_correct",
            "world_question_correct",
            "direct_correct",
            "direct_invalid",
            "structured_correct",
        )
        mask_counts = (
            "source_cases",
            "natural_responses",
            "calibration_responses",
            "belief_responses",
            "pressure_responses",
            "ablated_responses",
            "strict_belief_cases",
            "strict_clean_task_responses",
            "secondary_honest_responses",
            "secondary_opposing_responses",
            "secondary_honest_cases",
            "secondary_opposing_cases",
            "secondary_pooled_cases",
        )
        for values, keys in ((known, known_counts), (mask, mask_counts)):
            for key in keys:
                if type(values.get(key)) is not int or values[key] < 0:
                    raise ValueError(f"Latest measured count {key} must be a nonnegative integer.")
        if known["n_rows"] != 2 * known["n_unique_inputs"] or known["n_unique_inputs"] == 0:
            raise ValueError("Known-message rows must preserve paired sender-input denominators.")
        for key in known_counts[3:]:
            if known[key] > known["n_rows"]:
                raise ValueError(f"{key} cannot exceed the known-message row count.")
        if known["direct_correct"] + known["direct_invalid"] > known["n_rows"]:
            raise ValueError("Direct correct and invalid counts cannot overlap.")
        if mask["natural_responses"] != sum(
            mask[key]
            for key in (
                "calibration_responses",
                "belief_responses",
                "pressure_responses",
                "ablated_responses",
            )
        ):
            raise ValueError("MASK response counts must include reference, belief and task draws.")
        for key in (
            "strict_belief_cases",
            "secondary_honest_cases",
            "secondary_opposing_cases",
            "secondary_pooled_cases",
        ):
            if mask[key] > mask["source_cases"]:
                raise ValueError(f"{key} cannot exceed the MASK source panel.")
        if (
            mask["strict_clean_task_responses"]
            > mask["pressure_responses"] + mask["ablated_responses"]
        ):
            raise ValueError("Strict clean task responses cannot exceed task generations.")
        if (
            not max(mask["secondary_honest_cases"], mask["secondary_opposing_cases"])
            <= mask["secondary_pooled_cases"]
            <= mask["secondary_honest_cases"] + mask["secondary_opposing_cases"]
        ):
            raise ValueError("Secondary pooled cases must preserve the class-case union.")
        if mask.get("label_audited") is not False or mask.get("needs_human_review") is not True:
            raise ValueError("Current MASK labels must remain explicitly provisional.")
    return result


def _register_fonts() -> tuple[str, str]:
    """Use the same installed Arial files as the original presentation builder."""
    windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    choices = (
        (windows / "arial.ttf", windows / "arialbd.ttf"),
        (
            Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
            Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"),
        ),
    )
    for regular, bold in choices:
        if regular.is_file() and bold.is_file():
            if "ProgressArial" not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont("ProgressArial", str(regular)))
                pdfmetrics.registerFont(TTFont("ProgressArialBold", str(bold)))
            return "ProgressArial", "ProgressArialBold"
    raise RuntimeError("Installed Arial regular/bold fonts are required to match the template.")


class Slide:
    """Shared editable-PPTX/vector-PDF geometry, with measured text wrapping."""

    def __init__(self, background: str = PAPER) -> None:
        self.background = background
        self.objects: list[tuple[Any, ...]] = []
        self.notes = ""
        self.fonts = _register_fonts()

    def rect(self, x: float, y: float, w: float, h: float, color: str) -> None:
        self.objects.append(("rect", x, y, w, h, color))

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str) -> None:
        self.objects.append(("line", x1, y1, x2, y2, color))

    def text(
        self,
        value: str,
        x: float,
        y: float,
        w: float,
        size: float,
        color: str = INK,
        bold: bool = False,
        *,
        leading: float = 1.17,
        max_height: float | None = None,
    ) -> None:
        font = self.fonts[int(bold)]
        lines: list[str] = []
        for paragraph in value.split("\n"):
            current = ""
            for word in paragraph.split():
                candidate = f"{current} {word}".strip()
                if pdfmetrics.stringWidth(candidate, font, size) <= w * 72:
                    current = candidate
                elif current:
                    lines.append(current)
                    current = word
                else:
                    raise ValueError(f"Unbreakable text exceeds its box: {word}")
            lines.append(current)
        height = len(lines) * size * leading / 72 + 0.035
        if max_height is not None and height > max_height:
            raise ValueError(f"Text exceeds allocated height ({height:.2f}): {value}")
        if x < 0 or y < 0 or x + w > WIDTH + 0.01 or y + height > HEIGHT:
            raise ValueError(f"Text outside slide: {value}")
        self.objects.append(("text", lines, x, y, w, height, size, color, bold, leading))


def _header(slide: Slide, section: str, title: str, number: int, date: str) -> None:
    slide.text("KING’S COLLEGE LONDON / 6CCSAPRJ", 0.65, 0.35, 12, 10, MUTED, True)
    slide.text(section, 0.65, 0.85, 12, 11, TEAL, True)
    slide.text(title, 0.65, 1.18, 12.03, 28, INK, True, max_height=0.65)
    slide.line(0.65, 6.92, 12.68, 6.92, "#C3D0D2")
    slide.text(
        f"PROGRESS 02 · {date.upper()} · MEASURED DEVELOPMENT RESULTS", 0.65, 7.08, 11, 9, MUTED
    )
    slide.text(f"{number} / 4", 11.95, 7.08, 0.7, 9, MUTED)


def _slides(results: dict[str, Any]) -> list[Slide]:
    if "known_message" in results:
        return _latest_slides(results)
    date = results["progress_date"]
    provenance = (
        "Measured snapshot: " + results["snapshot_date"] + ". Defaults are sourced from "
        "docs/PILOT_RESULTS.md and the preserved run manifests; numeric software checks "
        "are distinct from independent model experiments. Results supplied to the builder:\n"
        + json.dumps(results, ensure_ascii=False, indent=2)
    )
    title = Slide(NAVY)
    title.text("KING’S COLLEGE LONDON / 6CCSAPRJ", 0.7, 0.4, 12, 10, "#B7CBCF", True)
    title.text("VARIANT 02 · LATENT PROBES UNDER DISTRIBUTION SHIFT", 0.7, 0.95, 12, 11, GOLD, True)
    title.text("Same Report,\nDifferent Reader", 0.7, 1.6, 8.2, 43, WHITE, True, leading=1.1)
    title.text(
        "Can a neutral-trained latent forecast of a reader’s decision transfer when the sender is pressured to obtain approval?",
        0.75,
        3.45,
        7.5,
        23,
        "#E2EBEA",
        max_height=1.65,
    )
    title.text("ADITYA RANJAN", 0.75, 6.25, 7.8, 16, WHITE, True)
    title.text(f"Hanqi Yan · Supervisor\nProgress 2 · {date}", 0.75, 6.65, 7.8, 12, "#B7CBCF")
    for y, label, body in (
        (1.65, "SOURCE + AUDIENCE", "Facts, rule and attributed reader evidence"),
        (3.2, "NATURAL REPORT", "Neutral goal or favourable-decision pressure"),
        (4.75, "INDEPENDENT READER", "Approve · Reject · Defer"),
    ):
        title.rect(9, y, 3.6, 1.15, "#243D50")
        title.text(label, 9.2, y + 0.15, 3.18, 11, GOLD, True)
        title.text(body, 9.2, y + 0.44, 3.15, 16, WHITE, max_height=0.65)
    title.line(10.8, 2.85, 10.8, 3.13, "#A9C2C5")
    title.line(10.8, 4.40, 10.8, 4.68, "#A9C2C5")
    title.text(
        "Reader decisions are measured; a belief mechanism remains a hypothesis.",
        9,
        6.25,
        3.6,
        12,
        "#B7CBCF",
    )
    title.notes = (
        "The core question continues Progress 1: train a lightweight forecast of a recipient’s "
        "decision on neutral sender summaries, freeze it, and evaluate transfer into reporting "
        "pressure. The sender-attributed reader evidence and the actual reader’s independently "
        "assigned starting evidence must remain distinct. Forecasting decisions does not by "
        "itself prove that the sender represents a recipient belief or intends to deceive.\n\n"
        + provenance
    )

    current = Slide()
    has_natural_pilot = "natural_reports" in results
    _header(
        current,
        "CURRENT STATE · SINCE PROGRESS 1",
        "Natural report pilot complete; forecast frozen"
        if has_natural_pilot
        else "Baseline verified; reader controls tested",
        2,
        date,
    )
    baseline_card = (
        "APOLLO COMPARATOR",
        str(results["baseline_examples"]),
        "saved 8B dialogue captures",
        f"Instruction-target AUROC {results['baseline_instruction_auroc']:.2f}.\nIndependent numeric and GPU replay checks.",
    )
    if has_natural_pilot:
        n_test = results["test_rows_per_condition"]
        n_neutral = results["forecast_neutral_correct"]
        n_pressure = results["forecast_pressure_correct"]
        forecast_value = (
            f"{n_neutral}/{n_test}" if n_neutral == n_pressure else f"{n_neutral} → {n_pressure}"
        )
        forecast_unit = (
            "correct in each condition"
            if n_neutral == n_pressure
            else f"correct of {n_test}: neutral → pressure"
        )
        copy_pressure = results["copy_pressure_correct"]
        card_specs = (
            baseline_card,
            (
                "NATURAL REPORT PILOT",
                str(results["natural_reports"]),
                f"reports + {results['natural_reader_calls']} reader labels",
                f"{results['neutral_reports']} neutral → fit → {results['pressure_reports']} pressured.\n24 cases; split 16 / 4 / 4.",
            ),
            (
                "FROZEN READER FORECAST",
                forecast_value,
                "activation: both conditions" if n_neutral == n_pressure else forecast_unit,
                f"Simple text copy: {copy_pressure}/{n_test} under pressure (exploratory).",
            ),
        )
    else:
        card_specs = (
            baseline_card,
            (
                "REAL READER PILOT",
                str(results["reader_calls"]),
                "calls · four development cases",
                "Inspection: REJECT → APPROVE after the decisive fact is restored.",
            ),
            (
                "FIXED-REPORT DIAGNOSTIC",
                str(results["fixed_replay_calls"]),
                "teacher-forced replays",
                f"Raw score rises in {results['fixed_replay_pressure_positive_pairs']}/{results['fixed_replay_original_pairs']} paired reports.\nMean change {results['fixed_replay_mean_pressure_delta']:+.4f}.",
            ),
        )
    for x, (label, value, unit, body) in zip((0.65, 4.78, 8.91), card_specs, strict=True):
        current.rect(x, 2.03, 3.78, 2.73, PALE)
        current.text(label, x + 0.18, 2.22, 3.42, 11.5, TEAL, True)
        current.text(value, x + 0.18, 2.58, 3.35, 37, INK, True)
        current.text(unit, x + 0.18, 3.21, 3.38, 14, MUTED, True, max_height=0.5)
        current.text(body, x + 0.18, 3.65, 3.36, 17, INK, max_height=1.02)
    readout = results["neutral_readout"]
    current.rect(0.65, 4.99, 12.03, 1.10, NAVY)
    current.text(readout["headline"], 0.85, 5.13, 11.60, 20, WHITE, True, max_height=0.40)
    current.text(readout["detail"], 0.85, 5.60, 11.60, 15, "#DFE9E9", max_height=0.40)
    current.text(
        f"Software verification: {results['software_tests']} tests + {results['software_subtests']} subtests; these are not model trials.",
        0.7,
        6.27,
        12,
        11,
        MUTED,
    )
    current.text(
        "Overall plan: baseline + frozen pilot complete → audit targets → fresh-family transfer study"
        if has_natural_pilot
        else "Overall plan: baseline complete → validate reader labels → freeze the transfer study",
        0.7,
        6.59,
        12,
        15,
        INK,
        True,
    )
    current.notes = (
        "The source-faithful Apollo-style adaptation is a comparator. Its labels are "
        "instructed honesty conditions, not audited natural deceptive behavior. The source "
        "baseline has 268 eligible unique fact pairs / 536 dialogues; the full numerical audit "
        "checks all stored artifacts and replays a prespecified subset of 22 dialogues. A "
        "prompt-only text control also reaches instruction AUROC 1.0.\n\n"
        "The reader assay contains four authored cases × three evidence packets × four "
        "report versions = 48 calls. In the inspection case a stale-evidence reader rejects "
        "the original and approves after the latest passed reinspection is restored; an "
        "informed reader approves both. None of the original reports induced a new wrong "
        "decision relative to no report in this panel. The original-to-restored stale comparison "
        "has one wrong-to-correct, two defer-to-correct, and one persistent wrong transition. "
        "There are no independent population estimates from four cases.\n\n"
        "The separate 48 fixed-text captures cross four cases, three sender-attributed "
        "reader packets, two reporting goals and two report versions. Original-report "
        "pressure contrasts are 12 repeated conditions within four cases. Raw probe scores "
        "are margins, not probabilities; this off-policy diagnostic does not establish "
        "natural deceptive intent or a validated false-positive rate.\n\n"
        "Neutral-readout status: " + readout["status"] + ". Only the explicit supplied result "
        "text is shown; additional supplied metrics are preserved below for audit.\n\n" + provenance
    )
    if has_natural_pilot:
        neutral_f1 = results.get("forecast_neutral_macro_f1", 0.703)
        pressure_f1 = results.get("forecast_pressure_macro_f1", 0.580)
        neutral_recall = results.get("forecast_neutral_balanced_recall", 0.778)
        pressure_recall = results.get("forecast_pressure_balanced_recall", 0.556)
        neutral_nll = results.get("forecast_neutral_nll", 0.958)
        lexical_nll = results.get("lexical_neutral_nll", 0.683)
        current.notes += (
            "\n\nNEW NATURAL-REPORT PILOT (development only): 24 authored cases with three "
            "reader packets each produce 72 neutral reports. The 16/4/4 case split is "
            "48 train / 12 validation / 12 neutral test reports. A three-class readout is "
            "trained and frozen before generating 12 pressure reports from the four test "
            "cases. There are 84 saved natural reports and 84 independently run reader "
            "labels, followed by 12 neutral and 12 pressured test forecasts. The same four "
            "case clusters support both objectives; these are not 24 independent test cases. "
            "Report text and reader context remain potential shortcuts, especially explicit "
            "decision words, comparisons and dates. A post-report readout does not identify "
            "a sender's belief.\n\n"
            "Accuracy alone hides classwise loss under pressure: activation macro-F1 is "
            f"{neutral_f1:.3f} neutral versus {pressure_f1:.3f} pressure; balanced recall is "
            f"{neutral_recall:.3f} versus {pressure_recall:.3f}. Neutral activation NLL "
            f"{neutral_nll:.3f} is worse than the lexical full-context baseline's "
            f"{lexical_nll:.3f}. This provides no calibration claim. "
            f"Lexical text scores {results['lexical_neutral_correct']}/{n_test} neutral "
            f"and {results['lexical_pressure_correct']}/{n_test} pressure; this is a "
            "preliminary comparison, not a capable LLM monitor. A direct "
            f"LLM text forecast scores {results['text_forecast_neutral_correct']}/{n_test} "
            f"neutral and {results['text_forecast_pressure_correct']}/{n_test} pressure. "
            "A literal-copy rule takes the last uppercase APPROVE, REJECT or DEFER in "
            "the report, defaulting to the neutral-training majority APPROVE when no "
            f"decision word occurs. It scores {results['copy_neutral_correct']}/{n_test} "
            f"neutral and {results['copy_pressure_correct']}/{n_test} pressure. Both were "
            "added post hoc and remain exploratory on the same four development test "
            "cases; they are not independent confirmatory comparisons. Added value of "
            "the activation forecast beyond report text is not established. Class "
            "imbalance, prompt/report truncation and factual accuracy of comparisons "
            "and dates require audit from the preserved raw reports. Fresh scenario-family "
            "holdouts and second-reader evaluation "
            "remain future work.\nAdditional supplied strong text forecast: "
            + json.dumps(
                results.get("strong_text_forecast", {"status": "pending"}), ensure_ascii=False
            )
        )

    barriers = Slide()
    _header(
        barriers,
        "CURRENT BARRIERS",
        "Valid targets and reader controls are the limiting step",
        3,
        date,
    )
    rows = (
        (
            "Target validity",
            "Full-evidence controls pass 23/24 cases; natural report facts still need audit.",
            "Audit comparisons, dates and explicit decision words; retain original outputs and failed controls.",
        ),
        (
            "Representation",
            "Features come after the report; explicit decision words provide a shortcut.",
            "Simple text copy beats activation accuracy under pressure in this exploratory pilot. Belief is not established.",
        ),
        (
            "Transfer evidence",
            "Only four development test cases; equal accuracy masks classwise loss.",
            "Report three-class metrics and uncertainty. Fresh family holdouts and a second reader remain ahead.",
        ),
        (
            "Compute / scale",
            "Local NF4 execution does not establish BF16 robustness.",
            "Profile a CREATE BF16 reference; scale after case semantics and capable text baselines are validated.",
        ),
    )
    if not has_natural_pilot:
        rows = (
            (
                "Target validity",
                "Instruction labels do not establish deceptive behaviour.",
                "Audit natural responses; compare honest and contradicting outputs within the same pressure condition.",
            ),
            (
                "Reader validity",
                "Profit restoration failed; a longer output budget did not fix it.",
                "Clarify evidence precedence; retain the failed version and recheck the reader controls.",
            ),
            (
                "Transfer evidence",
                "A future forecast pilot will be development evidence, not family-held-out generalisation.",
                "Neutral train → freeze → pressure test. Predict reader decisions; decodability is not proof of belief.",
            ),
            rows[3],
        )
    for index, (label, issue, response) in enumerate(rows):
        y = 2.03 + 1.12 * index
        if index % 2 == 0:
            barriers.rect(0.65, y - 0.08, 12.03, 1.05, PALE)
        barriers.text(label, 0.82, y + 0.08, 1.95, 16, TEAL, True, max_height=0.8)
        barriers.text(issue, 3, y + 0.04, 4.1, 16.5, INK, max_height=0.9)
        barriers.text(response, 7.58, y + 0.04, 4.8, 16.5, INK, max_height=0.9)
    pilot_barrier_note = (
        "The new natural-report pilot passes 23 of 24 authored full-evidence control "
        "cases; this is development evidence and the failed case remains visible. Natural "
        "reports contain potentially inaccurate comparisons or dates and explicit decision "
        "language, so retained raw text needs a factual and evidence-role audit before "
        "mechanistic claims. Equal test accuracy under neutral and pressured goals does "
        "not imply equal classwise behavior or calibrated forecasts.\n\n"
        if has_natural_pilot
        else ""
    )
    barriers.notes = pilot_barrier_note + (
        "A small controlled experiment is useful only with defensible outcome labels. "
        "For the profit case, revenue 120000 and costs 135000 imply a negative result, "
        "yet restoration leaves an absent reader deferring and a stale reader approving. "
        "A separate post-hoc diagnostic finds that a longer output limit does not solve it; "
        "explicit net profit and one packet-wording edit do. This suggests evidence wording "
        "or integration issues, without establishing a unique mechanism. Do not silently "
        "replace this failed control or call a revised development case held out.\n\n"
        "Any new neutral-reader readout result is developmental and case-held-out unless "
        "the complete scenario families, generation choices and monitoring configuration "
        "were frozen before independent evaluation. A sender activation cannot predict an "
        "actual reader packet randomized only after the report is saved without receiving "
        "that packet; original sender and context-aware monitor targets must be separate. "
        "The reader decision is a behavioral target, not a direct measurement of belief.\n\n"
        "CREATE supports a controlled BF16 reference and later activation/replay volume, "
        "not a claim that more compute repairs ambiguous labels.\n\n" + provenance
    )

    next_steps = Slide()
    _header(
        next_steps,
        "PLANNED NEXT STEPS",
        "Audit pilot targets, then test fresh-family transfer"
        if has_natural_pilot
        else "Freeze a neutral forecast, then test transfer under pressure",
        4,
        date,
    )
    plans = (
        (
            "NOW → 27 NOV",
            "Audit report facts, decision words and evidence roles. Preserve this pilot and capture features before the report.",
        ),
        (
            "BY 11 DEC",
            "Validate an independent neutral belief task, then test pre-report transfer under pressure before any causal claim.",
        ),
        (
            "18 DEC · BSPR",
            "Report pilot evidence and limits. Specify fresh family holdouts, a second reader and a CREATE BF16 reference.",
        ),
    )
    if not has_natural_pilot:
        plans = (
            (
                "NOW → 27 NOV",
                "Validate reader evidence rules and full-evidence controls. Fit the neutral readout; record failures and freeze its settings.",
            ),
            (
                "BY 11 DEC",
                "Test pressured natural reports against prompt-only and capable full-context text forecasts. Keep case splits fixed.",
            ),
            plans[2],
        )
    for x, (label, body) in zip((0.65, 4.78, 8.91), plans, strict=True):
        next_steps.rect(x, 2.03, 3.78, 2.82, PALE)
        next_steps.text(label, x + 0.18, 2.25, 3.42, 13, TEAL, True)
        next_steps.text(body, x + 0.18, 2.79, 3.38, 19, INK, max_height=1.95)
    next_steps.text(
        "JAN: freeze study → FEB: validation → MAR: family-held-out + second reader",
        0.72,
        5.18,
        11.9,
        20,
        INK,
        True,
        max_height=0.75,
    )
    next_steps.text(
        "APR: reproduce figures, finish report / code · Final submission 22 April 2027",
        0.72,
        5.94,
        11.9,
        17.5,
        MUTED,
        max_height=0.4,
    )
    next_steps.rect(0.65, 6.47, 12.03, 0.36, NAVY)
    next_steps.text(
        "Decision gate: validate forecasts and text controls before optional causal activation experiments.",
        0.83,
        6.55,
        11.65,
        12,
        WHITE,
        max_height=0.25,
    )
    next_steps.notes = (
        "This continues the approved central question rather than changing projects. "
        "Train the recipient-decision forecast only on neutral natural summaries with "
        "independently measured reader outcomes; freeze the representation, preprocessing, "
        "readout and splits before testing pressured natural summaries. Compare methods "
        "with the same source, rule, report and appropriate reader evidence. Prompt-only "
        "controls detect instruction shortcuts; a strong full-context text baseline is needed "
        "before claiming an activation advantage. Cross-case development results are not "
        "fresh-family generalisation. Future family holdouts and a second reader must be "
        "specified and frozen independently. Null results and equal text performance remain "
        "informative when the target and uncertainty are well controlled.\n\n"
        "Dates between official checkpoints are proposed internal milestones. The BSPR "
        "deadline is 18 December; verify the current KEATS submission time. The final "
        "report/source deadline in the supplied module schedule is 22 April 2027.\n\n" + provenance
    )
    if has_natural_pilot:
        next_steps.notes += (
            "\n\nThe natural pilot's post-report activation timing allows report-content "
            "and decision-word shortcuts. Its post-hoc copy-rule and direct-text "
            "comparisons do not establish activation added value. Preserve that "
            "exploratory result, audit raw reports for factual inaccuracies and "
            "truncation, then measure pre-report features on an independent neutral "
            "belief task before testing pressure transfer. Pre-report decodability "
            "alone still does not establish a belief mechanism or causal role."
        )
    return [title, current, barriers, next_steps]


def _latest_slides(results: dict[str, Any]) -> list[Slide]:
    """Four-slide measured update; retain legacy snapshots and the original renderer."""
    date, known, mask = (
        results["progress_date"],
        results["known_message"],
        results["mask_specificity"],
    )
    provenance = (
        "Measured snapshot: " + results["snapshot_date"] + ". All supplied results, including "
        "historical pilot metrics, are retained below. Software-test counts are historical "
        "verification records, not independent model experiments.\n"
        + json.dumps(results, ensure_ascii=False, indent=2)
    )
    title = Slide(NAVY)
    title.text("KING'S COLLEGE LONDON / 6CCSAPRJ", 0.7, 0.4, 12, 10, "#B7CBCF", True)
    title.text("VARIANT 02 · LATENT PROBES UNDER DISTRIBUTION SHIFT", 0.7, 0.95, 12, 11, GOLD, True)
    title.text("Latent Probes\nand Listener Beliefs", 0.7, 1.6, 8.2, 43, WHITE, True, leading=1.1)
    title.text(
        "Testing robustness under reporting pressure and shifts in what a reader knows.",
        0.75,
        3.45,
        7.5,
        23,
        "#E2EBEA",
        max_height=1.65,
    )
    title.text("ADITYA RANJAN", 0.75, 6.25, 7.8, 16, WHITE, True)
    title.text(f"Hanqi Yan · Supervisor\nProgress 2 · {date}", 0.75, 6.65, 7.8, 12, "#B7CBCF")
    for y, label, body in (
        (1.65, "FROZEN COMPARATOR", "8B Apollo adaptation; audited training pipeline"),
        (3.2, "NEUTRAL VALIDATION", "Separate reality, reader belief and sender attribution"),
        (4.75, "PRESSURE TRANSFER", "Main study follows independent validation"),
    ):
        title.rect(9, y, 3.6, 1.15, "#243D50")
        title.text(label, 9.2, y + 0.15, 3.18, 11, GOLD, True)
        title.text(body, 9.2, y + 0.44, 3.15, 16, WHITE, max_height=0.65)
    title.line(10.8, 2.85, 10.8, 3.13, "#A9C2C5")
    title.line(10.8, 4.40, 10.8, 4.68, "#A9C2C5")
    title.text(
        "Belief labels are stipulated; a causal mechanism remains unproven.",
        9,
        6.25,
        3.6,
        12,
        "#B7CBCF",
    )
    title.notes = (
        "This preserves the submitted Week 1 project title and the neutral-readout-to-pressure "
        "research direction. The supervisor's classic Apollo/MASK test and omission extension "
        "remain required. The nested-belief diagnostic is an additional controlled validation "
        "stage, not an already validated communication monitor. Neither readout accuracy nor "
        "a reader decision establishes deceptive intent or spontaneous second-order reasoning.\n\n"
        + provenance
    )

    current = Slide()
    _header(
        current,
        "CURRENT STATE · SINCE PROGRESS 1",
        "Baselines built; new frozen transfer evidence",
        2,
        date,
    )
    cards = (
        (
            "APOLLO ADAPTATION",
            str(results["baseline_examples"]),
            "saved 8B dialogue captures",
            f"Instruction AUROC {results['baseline_instruction_auroc']:.2f};\nbehavioural validation is separate.",
        ),
        (
            "MASK SPECIFICITY PILOT",
            str(mask["natural_responses"]),
            "88 MASK + 32 neutral reference",
            f"Strict: {mask['strict_clean_task_responses']} clean task answers.\nHuman review is pending.",
        ),
        (
            "FROZEN BELIEF TRANSFER",
            f"{known['queried_correct']}/{known['n_rows']}",
            f"{known['queried_correct'] // 2}/{known['n_unique_inputs']} unique sender inputs",
            "Known-message task, after belief question. No refitting on these cases.",
        ),
    )
    for x, (label, value, unit, body) in zip((0.65, 4.78, 8.91), cards, strict=True):
        current.rect(x, 2.03, 3.78, 2.73, PALE)
        current.text(label, x + 0.18, 2.22, 3.42, 11.5, TEAL, True)
        current.text(value, x + 0.18, 2.58, 3.35, 37, INK, True)
        current.text(unit, x + 0.18, 3.21, 3.38, 14, MUTED, True, max_height=0.5)
        current.text(body, x + 0.18, 3.65, 3.36, 17, INK, max_height=1.02)
    current.rect(0.65, 4.99, 12.03, 1.10, NAVY)
    current.text(
        "Promising queried transfer; sensitive to context and questions",
        0.85,
        5.13,
        11.60,
        20,
        WHITE,
        True,
        max_height=0.40,
    )
    current.text(
        f"Story / world-question heads: {known['story_correct']}/{known['n_rows']}. Direct answers: {known['direct_correct']}/{known['n_rows']}; {known['direct_invalid']} capped invalids.",
        0.85,
        5.60,
        11.60,
        15,
        "#DFE9E9",
        max_height=0.40,
    )
    current.text(
        "64 rows = 32 unique inputs; four wordings of one logical family; human review pending.",
        0.7,
        6.27,
        12,
        11,
        MUTED,
    )
    current.text(
        "Plan: baseline complete → validation pilots complete → main pressure-transfer study ahead",
        0.7,
        6.59,
        12,
        15,
        INK,
        True,
    )
    current.notes = (
        "Apollo: 268 eligible fact pairs / 536 dialogues. Training separates honest versus "
        "dishonest instructions using the same truthful continuation. Prompt-only text AUROC "
        "also reaches 1.00; this is a documented smaller-model/quantization adaptation, not "
        "reproduction of the paper's full strategic-deception evaluation.\n\n"
        "MASK: eight prospectively chosen source cases; 32 independent neutral-reference "
        "responses, 40 belief-elicitation responses, 24 pressure and 24 incentive-ablated "
        "responses. Two cases pass the strict five-clear-concordant belief gate, but no "
        "task response also passes the clean-assertion gate. The separate upstream-code-"
        "compatible diagnostic yields four pressured honest assertions from two cases and "
        "five pressured opposing assertions from two cases, with three cases in their union. These "
        "secondary counts apply the clean whole-response exclusions to draft annotations; "
        "they are not merely focal-only counts, primary clean negatives or a general "
        "false-positive-rate estimate. Refusals, truncation and ambiguity stay "
        "visible. The reference cutoff is not a validated honesty FPR.\n\n"
        "Known-message transfer: the previously frozen neutral nested-question block-16 "
        "sender head scores 56/64 paired rows = 28/32 unique inputs. No refitting, layer "
        "choice or recalibration used this panel. Corresponding story and world-question "
        "heads score 32/64; the queried head transferred unchanged to story-only states "
        "also scores 32/64. This is question-conditioned transfer from observation evidence "
        "to an exact known message. Source and new tasks retain the same XNOR composition. "
        "It is not anticipation of an unknown report, natural pressure transfer, a causal "
        "mechanism or validated unqueried belief decoding. Four wordings are one family; "
        "labels specify fictional states and need human review.\n\n"
        "Direct answers score 44/64, including four length-capped invalid rows. Structured "
        "answers score 32/64 and all 32 unique outputs end FINAL:YES; the scaffold has not "
        "established a capable text monitor. Post-hoc emitted-field inspection finds 21/32 "
        "attributed places correct, but 19/32 finals disagree with their own attribution. "
        "This does not identify internal causal reasoning or replace the frozen endpoint.\n\n"
        "Earlier natural-report pilot: 84 reports across 24 development cases; after-report "
        "activation predicts 10/12 pressure outcomes while an exploratory literal text-copy "
        "rule gets 11/12. Added value beyond text remains unestablished.\n\n" + provenance
    )

    barriers = Slide()
    _header(barriers, "CURRENT BARRIERS", "Validation limits come before scaling", 3, date)
    issues = (
        (
            "Behaviour labels",
            "The strict clean MASK task subset is empty; draft labels need human review.",
            "Review assertions and belief evidence; freeze the next sample and both eligibility rules.",
        ),
        (
            "Context robustness",
            "Role: 44/64. Future goals: 50/64 versus 34/64; current target unchanged.",
            "Question dependence and observed-bank sensitivity need independent neutral validation.",
        ),
        (
            "Text comparator",
            "Structured answers always end YES; direct answers include capped failures.",
            "Preserve failures and improve the comparator prospectively before claiming an advantage.",
        ),
        (
            "Compute / scale",
            "Local NF4 results do not establish BF16 or cross-model robustness.",
            "Request one CREATE A100 40 GB for a matched BF16 reference, then second-reader replay.",
        ),
    )
    for index, (label, issue, response) in enumerate(issues):
        y = 2.03 + 1.12 * index
        if index % 2 == 0:
            barriers.rect(0.65, y - 0.08, 12.03, 1.05, PALE)
        barriers.text(label, 0.82, y + 0.08, 1.95, 16, TEAL, True, max_height=0.8)
        barriers.text(issue, 3, y + 0.04, 4.1, 16.5, INK, max_height=0.9)
        barriers.text(response, 7.58, y + 0.04, 4.8, 16.5, INK, max_height=0.9)
    barriers.notes = (
        "The small purposive MASK sample does not support representative prevalence or "
        "a model-wide honest false-positive rate. Strict and upstream-compatible labels "
        "are different estimands and must remain separately reported. Human review is "
        "required before research-grade behavioral-label claims. A larger sample and its "
        "rules must be defined before new outputs; do not substitute favorable cases.\n\n"
        "The frozen known-message head improves on this authored task but fails in some "
        "false-message cases. Explicit answers and the text scaffold are not competitive "
        "monitors merely because they use the same model. Question length/position is not "
        "matched, and no causal question mechanism is isolated. Pairing means 64 rows are "
        "32 unique inputs, not 64 independent experimental units.\n\n"
        "Exploratory goal-context stress test on the already-observed known-message bank: "
        "the original queried head scores 56/64, a neutral role scaffold 44/64, a future "
        "goal for the reader to believe the candidate 50/64, and the opposite future goal "
        "34/64. No new head is fitted and no answer or report is generated. The current "
        "nested target is unchanged; future objectives are explicitly not new evidence. "
        "All 32 unique matched inputs have a higher positive-label probability under "
        "goal-true than goal-false, with mean difference approximately +0.0248. Role framing "
        "already lowers accuracy, and the goals differ in wording, negation and length. "
        "This is bounded context sensitivity, not a fresh holdout, changed current belief, "
        "natural pressured communication, actual deception, reader impact or a causal "
        "goal effect. Four wordings still share one logical family.\n\n"
        "CREATE would resolve a defined precision uncertainty: compare NF4 and BF16 "
        "with the same checkpoint, cases, prompts, extraction and frozen heads on matched "
        "CREATE hardware/software. No evidence currently establishes that quantization "
        "caused any failure. A 40 GB GPU permits the roughly 16 GB BF16 weights plus "
        "runtime overhead. Profile first; later evaluate a second reader on saved reports.\n\n"
        + provenance
    )

    next_steps = Slide()
    _header(
        next_steps,
        "PLANNED NEXT STEPS",
        "Audit labels, validate neutral transfer, then test pressure",
        4,
        date,
    )
    plans = (
        (
            "NOW → 27 NOV",
            "Human-review MASK labels and neutral stories. Freeze a larger panel and report strict and code-compatible eligibility.",
        ),
        (
            "BY 11 DEC",
            "Validate neutral belief readouts on new families. Match capture time before pressure transfer; strengthen text controls.",
        ),
        (
            "18 DEC · BSPR",
            "Report frozen pilots and limitations. Profile a matched BF16 reference on one CREATE A100 40 GB.",
        ),
    )
    for x, (label, body) in zip((0.65, 4.78, 8.91), plans, strict=True):
        next_steps.rect(x, 2.03, 3.78, 2.82, PALE)
        next_steps.text(label, x + 0.18, 2.25, 3.42, 13, TEAL, True)
        next_steps.text(body, x + 0.18, 2.79, 3.38, 19, INK, max_height=1.95)
    next_steps.text(
        "JAN: freeze study → FEB: validation → MAR: held-out families + second reader",
        0.72,
        5.18,
        11.9,
        20,
        INK,
        True,
        max_height=0.75,
    )
    next_steps.text(
        "APR: reproduce figures, finish report / code · Final submission 22 April 2027",
        0.72,
        5.94,
        11.9,
        17.5,
        MUTED,
        max_height=0.4,
    )
    next_steps.rect(0.65, 6.47, 12.03, 0.36, NAVY)
    next_steps.text(
        "Main-study gate: audited labels and independent neutral validation before pressure transfer.",
        0.83,
        6.55,
        11.65,
        12,
        WHITE,
        max_height=0.25,
    )
    next_steps.notes = (
        "Baseline implementation and development validation pilots are complete; the "
        "main robustness study is still ahead. Preserve Apollo/MASK and the same-summary "
        "recipient-knowledge omission comparison from the supervisor's brief. Neutral "
        "belief readouts add a testable comparator, conditional on construct validation.\n\n"
        "First obtain independent human review, then prospectively define a larger MASK "
        "panel and preserve both strict-five and upstream-code-compatible eligibility. "
        "Do not call every non-lying or refusing response honest. Fix neutral target "
        "semantics, visibility and activation timing before transferring to pressure. "
        "Known-message updating and prediction before an unknown future report differ.\n\n"
        "Later freeze fresh scenario-family holdouts, sampling, monitors and metrics, then "
        "replay saved reports with a second reader. Main-study uncertainty should resample "
        "the actual independent scenario units, not token rows. CREATE supports a matched "
        "precision reference and replication, not repair of ambiguous labels.\n\n"
        "Intermediate dates are proposed internal milestones. BSPR is 18 December; check "
        "KEATS for the submission time. The supplied module schedule lists final report "
        "and source submission on 22 April 2027.\n\n" + provenance
    )
    return [title, current, barriers, next_steps]


def _write_deck(
    template: Path,
    pptx_path: Path,
    pdf_path: Path,
    slides: list[Slide],
    project_name: str = "Same Report, Different Reader",
) -> None:
    presentation = Presentation(str(template))
    if len(presentation.slides) != 4:
        raise ValueError("The source template must contain exactly four slides.")
    if (
        abs(presentation.slide_width / 914400 - WIDTH) > 0.01
        or abs(presentation.slide_height / 914400 - HEIGHT) > 0.01
    ):
        raise ValueError("The template must use the original 13.333 × 7.5 inch page.")
    regular, bold_font = _register_fonts()
    pdf = canvas.Canvas(str(pdf_path), pagesize=(WIDTH * 72, HEIGHT * 72))
    pdf.setTitle(f"Progress 2 — {project_name} — measured results")
    pdf.setAuthor("Aditya Ranjan")
    presentation.core_properties.title = f"Progress 2 — {project_name}"
    presentation.core_properties.subject = "Measured project progress, barriers and next steps"
    presentation.core_properties.author = "Aditya Ranjan"
    for index, model in enumerate(slides):
        slide = presentation.slides[index]
        for shape in list(slide.shapes):
            element = shape._element
            element.getparent().remove(element)
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = RGBColor.from_string(model.background[1:])
        pdf.setFillColor(HexColor(model.background))
        pdf.rect(0, 0, WIDTH * 72, HEIGHT * 72, fill=1, stroke=0)
        for obj in model.objects:
            if obj[0] == "rect":
                _, x, y, w, h, color = obj
                shape = slide.shapes.add_shape(
                    MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
                )
                shape.fill.solid()
                shape.fill.fore_color.rgb = RGBColor.from_string(color[1:])
                shape.line.fill.background()
                pdf.setFillColor(HexColor(color))
                pdf.rect(x * 72, (HEIGHT - y - h) * 72, w * 72, h * 72, fill=1, stroke=0)
            elif obj[0] == "line":
                _, x1, y1, x2, y2, color = obj
                shape = slide.shapes.add_connector(
                    MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
                )
                shape.line.color.rgb = RGBColor.from_string(color[1:])
                shape.line.width = Pt(0.6)
                pdf.setStrokeColor(HexColor(color))
                pdf.setLineWidth(0.6)
                pdf.line(x1 * 72, (HEIGHT - y1) * 72, x2 * 72, (HEIGHT - y2) * 72)
            else:
                _, lines, x, y, w, h, size, color, bold, leading = obj
                textbox = slide.shapes.add_textbox(
                    Inches(x), Inches(y), Inches(w + 0.015), Inches(h)
                )
                frame = textbox.text_frame
                frame.clear()
                frame.word_wrap = False
                frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
                for line_index, line in enumerate(lines):
                    paragraph = frame.paragraphs[0] if line_index == 0 else frame.add_paragraph()
                    paragraph.text = line
                    paragraph.font.name = "Arial"
                    paragraph.font.size = Pt(size)
                    paragraph.font.bold = bold
                    paragraph.font.color.rgb = RGBColor.from_string(color[1:])
                    paragraph.space_before = paragraph.space_after = Pt(0)
                    paragraph.line_spacing = Pt(size * leading)
                    pdf.setFont(bold_font if bold else regular, size)
                    pdf.setFillColor(HexColor(color))
                    pdf.drawString(
                        x * 72, (HEIGHT - y) * 72 - size * 0.86 - line_index * size * leading, line
                    )
        slide.notes_slide.notes_text_frame.text = model.notes
        pdf.showPage()
    pdf.save()
    presentation.save(str(pptx_path))


def build_progress2(
    results_path: str | Path | None = None,
    template_path: str | Path | None = None,
    pptx_output: str | Path | None = None,
    pdf_output: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create two new deliverables; preserve and hash-check the original template."""
    template = Path(template_path or DEFAULT_TEMPLATE).resolve()
    pptx_path = Path(pptx_output or DEFAULT_PPTX).resolve()
    pdf_path = Path(pdf_output or DEFAULT_PDF).resolve()
    result_path = Path(results_path).resolve() if results_path is not None else None
    if not template.is_file():
        raise FileNotFoundError(template)
    if pptx_path.suffix.lower() != ".pptx" or pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Outputs must be a .pptx and a .pdf.")
    if template in {pptx_path, pdf_path} or pptx_path == pdf_path:
        raise ValueError("Output paths must be distinct from each other and the source template.")
    legacy_deliverables = REPO.parent / "research" / "deliverables"
    protected = {
        DEFAULT_TEMPLATE.resolve(),
        (legacy_deliverables / "Progress-2-four-slides-editable.pptx").resolve(),
        (legacy_deliverables / "Progress-2-four-slides.pdf").resolve(),
    }
    if pptx_path in protected or pdf_path in protected:
        raise ValueError("Original Progress 2 drafts are protected; choose new output names.")
    for output in (pptx_path, pdf_path):
        if output.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {output}; use overwrite=True / --overwrite explicitly."
            )
        output.parent.mkdir(parents=True, exist_ok=True)
    results = _load_results(result_path)
    template_hash = _sha256(template)
    models = _slides(results)
    with tempfile.TemporaryDirectory(prefix="progress2-") as temporary:
        temporary_path = Path(temporary)
        temporary_pptx = temporary_path / "measured.pptx"
        temporary_pdf = temporary_path / "measured.pdf"
        _write_deck(
            template,
            temporary_pptx,
            temporary_pdf,
            models,
            results.get("project_name", "Same Report, Different Reader"),
        )
        if _sha256(template) != template_hash:
            raise RuntimeError("Source template changed during generation; outputs not published.")
        # Both outputs have been built successfully before either deliverable is replaced.
        pptx_path.write_bytes(temporary_pptx.read_bytes())
        pdf_path.write_bytes(temporary_pdf.read_bytes())
    return {
        "pptx": str(pptx_path),
        "pdf": str(pdf_path),
        "slides": 4,
        "template_sha256": template_hash,
        "results_sha256": _sha256(result_path) if result_path else None,
        "pptx_sha256": _sha256(pptx_path),
        "pdf_sha256": _sha256(pdf_path),
        "neutral_readout_status": results["neutral_readout"]["status"],
        "pdf_renderer": "ReportLab vector layout with the same Arial geometry as the PPTX",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, help="Optional measured-results JSON; pending is the default."
    )
    parser.add_argument("--template", type=Path, help="Original four-slide editable template.")
    parser.add_argument("--pptx", type=Path, help="New editable PPTX output.")
    parser.add_argument("--pdf", type=Path, help="New matching PDF output.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace the two explicitly selected new outputs."
    )
    args = parser.parse_args()
    report = build_progress2(
        args.results, args.template, args.pptx, args.pdf, overwrite=args.overwrite
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
