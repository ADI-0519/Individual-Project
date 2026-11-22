"""Build exactly four editable empirical slides from a hash-bound real report.

The original Progress 2 deck, template, renderer and report are preserved. This
builder introduces no model calls or scientific analyses. Fixture inputs are
never accepted by the public build/CLI; layout tests mark their slides as such.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import tempfile
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / "presentations" / "progress2-main-study-2026-11-22"
RENDERER = Path(__file__).with_name("update_progress2.py")
TEMPLATE = REPO / "presentations" / "templates" / "progress2-original.pptx"
PROJECT = "Latent Probes and Listener Beliefs"
MODEL_IDS = {"Llama": "meta-llama/Llama-3.1-8B-Instruct", "Qwen": "Qwen/Qwen2.5-7B-Instruct"}
SEEDS = [11, 23, 37, 53, 71]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _layout():
    spec = importlib.util.spec_from_file_location("main_study_progress_style", RENDERER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finite(value, minimum=0, maximum=1):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError("Expected a finite measured score in its permitted range")
    return value


def load_sources(summary_path: Path, manifest_path: Path) -> tuple[dict, dict]:
    """Check report bindings and the exact real exploratory comparison inventory."""
    raw, manifest_raw = summary_path.read_bytes(), manifest_path.read_bytes()
    summary, manifest = json.loads(raw), json.loads(manifest_raw)
    if (
        summary.get("schema") != "main-study-report-v1"
        or manifest.get("schema") != "main-study-report-manifest-v1"
    ):
        raise ValueError("Expected the unchanged reporter's summary and manifest schemas")
    if manifest.get("outputs", {}).get(summary_path.name) != hashlib.sha256(raw).hexdigest():
        raise ValueError("Summary hash is not bound by the report manifest")
    if manifest.get("report_script_sha256") != _hash(
        Path(__file__).with_name("report_main_study.py")
    ):
        raise ValueError("Report manifest does not bind the unchanged reporter version")
    for data in (summary, manifest):
        if (
            data.get("test_fixture") is not False
            or data.get("research_status") != "agent_audited_exploratory"
            or data.get("human_review_complete") is not False
            or data.get("confirmatory_result") is not False
        ):
            raise ValueError(
                "Only real agent-audited exploratory, human-incomplete reports accepted"
            )
    bank = manifest.get("bank_sha256", "")
    if len(bank) != 64 or any(c not in "0123456789abcdef" for c in bank):
        raise ValueError("Report requires the canonical bank SHA256")
    models = summary.get("models", [])
    if len(models) != 2 or {m.get("model") for m in models} != set(MODEL_IDS):
        raise ValueError("Exactly one Llama and one Qwen result are required")
    for model in models:
        if model.get("model_id") != MODEL_IDS[model["model"]]:
            raise ValueError("Model identity differs from its label")
        primary = model.get("primary", {})
        if (
            primary.get("status") != "complete"
            or primary.get("allocation_seeds") != SEEDS
            or primary.get("independent_scenario_groups") != 30
        ):
            raise ValueError("Deck requires all thirty primary groups and five allocations")
        restricted, full = (_finite(primary.get(arm)) for arm in ("restricted", "full"))
        delta = _finite(primary.get("full_minus_restricted"), -1)
        if not math.isclose(delta, full - restricted, abs_tol=1e-10):
            raise ValueError("Primary difference contradicts the saved arm scores")
        interval = primary.get("percentile_95_interval")
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("Primary requires its saved interval")
        if _finite(interval[0], -1) > _finite(interval[1], -1):
            raise ValueError("Primary interval endpoints are reversed")
        for mechanism in ("observation", "message", "correction"):
            row = model.get("learning_and_mechanisms", {}).get(f"activation:512:{mechanism}", {})
            if (
                row.get("scenario_groups") != 30
                or row.get("complete_allocations") != 5
                or row.get("planned_allocations") != 5
            ):
                raise ValueError("Complete held-out mechanism and allocation counts required")
            for arm in ("restricted", "full"):
                _finite(row.get(arm))
        if model.get("prose_monitor", {}).get("status") not in ("complete", "not_provided"):
            raise ValueError("Explicit prose monitor status required")
    if len(manifest.get("inputs", [])) != 2:
        raise ValueError("Two saved analysis source bindings are required")
    for item in manifest["inputs"]:
        digest = item.get("sha256", "")
        if (
            not isinstance(item.get("path"), str)
            or not item["path"]
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("Each analysis source must retain its path and valid SHA256")
    bindings = {
        "summary": {"path": str(summary_path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()},
        "report_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": hashlib.sha256(manifest_raw).hexdigest(),
        },
    }
    return summary, bindings


def _slides(summary: dict, bindings: dict) -> list:
    style = _layout()
    fixture = summary.get("test_fixture") is True
    status = (
        "SYNTHETIC FIXTURE — NOT MODEL RESULTS"
        if fixture
        else "AGENT-AUDITED EXPLORATORY · HUMAN REVIEW INCOMPLETE"
    )
    models = {item["model"]: item for item in summary["models"]}
    missing = [
        label for label, data in models.items() if data["prose_monitor"]["status"] != "complete"
    ]
    provenance = "\n\nSource-bound summary and report manifest:\n" + json.dumps(bindings, indent=2)
    retention = "\n\nSaved N512 one-step quartet scores (restricted/full):\n" + "\n".join(
        f"{label}: "
        + "; ".join(
            f"{mechanism} "
            + "/".join(
                f"{100 * data['learning_and_mechanisms'][f'activation:512:{mechanism}'][arm]:.2f}%"
                for arm in ("restricted", "full")
            )
            for mechanism in ("observation", "message")
        )
        for label, data in models.items()
    )

    def header(section, title, number):
        slide = style.Slide()
        slide.text("KING’S COLLEGE LONDON / 6CCSAPRJ", 0.65, 0.35, 12, 10, style.MUTED, True)
        slide.text(section, 0.65, 0.85, 12, 11, style.TEAL, True)
        slide.text(title, 0.65, 1.18, 12.03, 28, style.INK, True, max_height=0.65)
        slide.line(0.65, 6.92, 12.68, 6.92, "#C3D0D2")
        slide.text("PROGRESS 02 · 22 NOVEMBER 2026 · " + status, 0.65, 7.08, 11, 8.3, style.MUTED)
        slide.text(f"{number} / 4", 11.95, 7.08, 0.7, 9, style.MUTED)
        return slide

    title = style.Slide(style.NAVY)
    title.text("KING’S COLLEGE LONDON / 6CCSAPRJ", 0.7, 0.4, 12, 10, "#B7CBCF", True)
    title.text(
        "VARIANT 02 · LATENT PROBES UNDER DISTRIBUTION SHIFT", 0.7, 0.95, 12, 11, style.GOLD, True
    )
    title.text(
        "Latent Probes and\nListener Beliefs", 0.7, 1.6, 8.2, 38, style.WHITE, True, leading=1.1
    )
    title.text(
        "Testing whether a linear readout of what a sender thinks a reader believes "
        "remains reliable when evidence and message structure change.",
        0.75,
        3.35,
        7.5,
        23,
        "#E2EBEA",
        max_height=1.8,
    )
    title.text("ADITYA RANJAN", 0.75, 6.1, 7.8, 16, style.WHITE, True)
    title.text(
        "6CCSAPRJ Individual Project · Progress 2\n22 November 2026", 0.75, 6.47, 7.8, 12, "#B7CBCF"
    )
    for y, heading, body in (
        (1.65, "MATCHED FITTING SUPPORT", "Restricted and full coverage"),
        (3.2, "HELD-OUT CORRECTIONS", "Four linked judgments"),
        (4.75, "TWO MODEL FAMILIES", "Llama 8B + Qwen 7B"),
    ):
        title.rect(9, y, 3.6, 1.15, "#243D50")
        title.text(heading, 9.2, y + 0.15, 3.18, 10, style.GOLD, True)
        title.text(body, 9.2, y + 0.44, 3.15, 16, style.WHITE, max_height=0.65)
    title.text(status, 9, 6.18, 3.6, 10, "#B7CBCF", max_height=0.65)
    title.notes = (
        "The original Variant 2 concerns deception-probe robustness. Apollo remains an "
        "8B recipe adaptation, not the full published 70B reproduction. This current "
        "study has a different, narrower target: fictional sender-attributed reader "
        "belief content. Discuss that scope change with the supervisor; no prior "
        "approval, spontaneous anticipation or causal belief mechanism is claimed. "
        "This deck does not assert a date for the last meeting." + provenance
    )

    progress = header("CURRENT STATE", "Two-model held-out evaluation complete", 2)
    for x, label in ((0.65, "Llama"), (6.8, "Qwen")):
        result = models[label]["primary"]
        progress.rect(x, 2.0, 5.85, 2.55, style.PALE)
        progress.text(label + " · N=512", x + 0.2, 2.16, 5.4, 20, style.TEAL, True)
        progress.text(
            f"Restricted  {100 * result['restricted']:.2f}%", x + 0.2, 2.65, 5.4, 24, bold=True
        )
        progress.text(
            f"Full coverage  {100 * result['full']:.2f}%", x + 0.2, 3.12, 5.4, 24, bold=True
        )
        progress.text(
            f"Full − restricted: {100 * result['full_minus_restricted']:+.2f} pp",
            x + 0.2,
            3.67,
            5.4,
            16,
            max_height=0.4,
        )
        low, high = result["percentile_95_interval"]
        progress.text(
            f"95% group interval: [{100 * low:+.2f}, {100 * high:+.2f}] pp",
            x + 0.2,
            4.02,
            5.4,
            14,
            style.MUTED,
            max_height=0.35,
        )
    progress.text(
        "Primary score: all four matched correction judgments correct.\n"
        "30 correction contexts/model · five paired allocations · 90 test contexts/model.",
        0.7,
        4.82,
        11.9,
        17,
        max_height=0.8,
    )
    progress.text(
        "Restricted: prior belief differs from incoming information.\n"
        "Full coverage also includes agreement; fitting budgets and groups match.",
        0.7,
        5.7,
        11.9,
        15,
        style.MUTED,
        max_height=0.65,
    )
    progress.notes = (
        "Both models have completed saved train/development fits and full held-out "
        "analyses. Scores and intervals are read from the unchanged report; there "
        "is no fresh fitting or statistical calculation here. N512 is unique "
        "fitting inputs per arm. Both arms see full-support development labels "
        "for selection; this tests fitting support, not zero exposure. Quartets "
        "require four linked predictions. The interval resamples thirty authored "
        "correction contexts with paired allocations retained; it is conditional "
        "on this bank and shared renderer. An interval including zero is not "
        "equivalence. These are exploratory outcomes without human validation. "
        "Full-bank capture totals are omitted because the report summary does "
        "not independently bind those counts." + provenance
    )

    barriers = header("CURRENT BARRIERS", "Interpretation, validation and scope", 3)
    baseline_heading = (
        "Fixed prose baseline pending" if missing else "Complete baseline interpretation"
    )
    baseline_body = (
        "The shared fixed monitor is not yet attached to the " + ", ".join(missing) + " reports."
        if missing
        else "Report text/output comparisons and all invalid outputs together."
    )
    points = [
        (
            "Joint consistency remains limited",
            "Qwen is already weak on one-message quartets; this is not a clean composition-only failure.",
        ),
        (
            "Human semantic review incomplete",
            "Independent agent checks support consistency; results remain exploratory.",
        ),
        (baseline_heading, baseline_body),
        (
            "Construct and distribution-shift limits",
            "Correction adds length and events; attributed belief is distinct from deception or anticipation.",
        ),
    ]
    for index, (heading, body) in enumerate(points):
        y = 2.0 + index * 1.04
        barriers.rect(0.7, y, 0.42, 0.42, style.TEAL)
        barriers.text(str(index + 1), 0.81, y + 0.04, 0.2, 15, style.WHITE, True)
        barriers.text(heading, 1.35, y, 11.1, 19, bold=True, max_height=0.4)
        barriers.text(body, 1.35, y + 0.43, 10.9, 15, style.MUTED, max_height=0.55)
    barriers.notes = (
        "Keep each model's absolute scores visible. The result concerns this fixed "
        "layer/precision/readout and these correction accounts, not absence of "
        "beliefs in the model. Human semantic review is still incomplete. Agent "
        "review cannot authenticate human agreement, and all narratives share a "
        "renderer. The fixed prose monitor must retain invalid outputs in its "
        "denominator. No practical latent advantage or spontaneous anticipation "
        "is established. Scope remains distinct from the original natural "
        "omission/recipient-outcome extension." + retention + provenance
    )

    next_steps = header(
        "PLANNED NEXT STEPS", "Finish the fixed comparison and agree the contribution", 4
    )
    items = [
        (
            "Finish the frozen prose monitor" if missing else "Report the frozen prose comparison",
            "Preserve invalid answers, failed attempts and full denominators.",
        ),
        (
            "Complete the baseline table and write-up",
            "Both models, calibration, one-message retention and correction transfer.",
        ),
        (
            "Arrange human semantic review",
            "A future revised study needs fresh evaluation cases; exposed results stay exploratory.",
        ),
        (
            "Agree the Variant 2 scope with the supervisor",
            "Position belief-readout robustness alongside the separate Apollo adaptation.",
        ),
    ]
    for index, (heading, body) in enumerate(items):
        y = 2.0 + index * 0.98
        next_steps.text(f"0{index + 1}", 0.7, y, 0.55, 22, style.TEAL, True)
        next_steps.text(heading, 1.48, y, 10.9, 18, bold=True, max_height=0.4)
        next_steps.text(body, 1.48, y + 0.4, 10.85, 14, style.MUTED, max_height=0.5)
    next_steps.rect(0.7, 6.12, 11.9, 0.56, style.PALE)
    next_steps.text(
        "PLAN: materials + fits + both test analyses complete → baselines + interpretation + report",
        0.9,
        6.25,
        11.5,
        13,
        style.TEAL,
        True,
        max_height=0.35,
    )
    next_steps.notes = (
        "Finish the declared comparison before choosing a publication claim. "
        "Do not change prompts, cases, thresholds or layers to obtain a positive "
        "outcome. The next deliverable is the complete empirical argument and "
        "source-bound results package. Later human review cannot retroactively "
        "make exposed outcomes confirmatory. New hypothesis-driven follow-ups "
        "would use a separately frozen fresh cohort. Publication, full marks "
        "and admissions are not guaranteed." + provenance
    )
    return [title, progress, barriers, next_steps]


def build(summary_path: Path, manifest_path: Path, output: Path = DEFAULT_OUTPUT) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError("Preserve earlier decks; output directory already exists")
    summary, bindings = load_sources(summary_path, manifest_path)
    template_hash, renderer_hash = _hash(TEMPLATE), _hash(RENDERER)
    slides = _slides(summary, bindings)
    if len(slides) != 4:
        raise ValueError("Progress update must contain exactly four slides")
    with tempfile.TemporaryDirectory(prefix="main-study-progress2-") as temporary:
        root = Path(temporary)
        pptx, pdf = root / "Progress-2-main-study.pptx", root / "Progress-2-main-study.pdf"
        _layout()._write_deck(TEMPLATE, pptx, pdf, slides, PROJECT)
        notes = "# Four-slide speaker notes\n\n" + "\n\n".join(
            f"## Slide {i}\n\n{slide.notes}" for i, slide in enumerate(slides, 1)
        )
        notes = notes.rstrip("\r\n") + "\n"
        if _hash(TEMPLATE) != template_hash or _hash(RENDERER) != renderer_hash:
            raise RuntimeError("Original renderer or template changed during generation")
        for binding in bindings.values():
            if _hash(Path(binding["path"])) != binding["sha256"]:
                raise RuntimeError("Source report changed during deck generation")
        output.mkdir(parents=True, exist_ok=False)
        for source in (pptx, pdf):
            (output / source.name).write_bytes(source.read_bytes())
        (output / "speaker-notes.md").write_text(notes, encoding="utf-8", newline="\n")
    receipt = {
        "schema": "main-study-progress2-receipt-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "slides": 4,
        "test_fixture": False,
        "research_status": "agent_audited_exploratory",
        "human_review_complete": False,
        "confirmatory_result": False,
        "inputs": bindings,
        "template_sha256": template_hash,
        "renderer_sha256": renderer_hash,
        "builder_sha256": _hash(Path(__file__)),
        "outputs": {p.name: _hash(p) for p in sorted(output.iterdir())},
    }
    (output / "source-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        receipt = build(args.summary, args.manifest, args.output_directory)
    except (ValueError, FileExistsError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": str(args.output_directory), "slides": receipt["slides"]}))


if __name__ == "__main__":
    main()
