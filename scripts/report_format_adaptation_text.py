"""Export the complete audited lexical diagnostic without fitting or scoring.

The original compressed outputs are copied byte for byte. Fitted artifacts are
archived under six fixed names, and independent reconstruction JSON is losslessly
compressed. Existing latent results appear only as a bound display comparison.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import math
import shutil
import zipfile
from pathlib import Path

from latent_probes.storage import file_sha256, read_json, utc_now

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ("full_messages", "system_only")
ARMS = ("A512", "Mixed512", "Ahalf256", "Offset")
FIT_NAMES = tuple(
    f"fit/{view}/{filename}"
    for view in VIEWS
    for filename in ("settings.json", "arrays.npz", "manifest.json")
)
SOURCE_NAMES = (
    "rows.json.gz",
    "predictions.npz",
    "summary.json.gz",
    "metrics.json",
    "preparation.json",
    "fit_seal.json",
    "analysis_complete.json",
    "fit_started.json",
    "analysis_started.json",
)
AUDITOR = "scripts/audit_format_adaptation_text.py"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_compressed(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def verify_pipeline(source):
    """Use the sealed runner's source/input/fit/result verification unchanged."""
    spec = importlib.util.spec_from_file_location(
        "format_adaptation_text_report_runner", ROOT / "scripts/run_format_adaptation_text.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner.verify(source)


def _validate_audit(source, audit, preparation):
    receipt = read_json(audit / "audit_receipt.json")
    require(
        receipt.get("schema") == "format-adaptation-text-independent-metric-audit-v1"
        and receipt.get("status") == "pass"
        and Path(receipt.get("source", "")).resolve() == source,
        "Passing independent audit for this source is required",
    )
    require(
        receipt.get("model_calls") == receipt.get("classifier_fits") == 0,
        "Metric audit must not fit or infer",
    )
    expected = {"rows.json.gz", "summary.json.gz", "predictions.npz"}
    require(set(receipt.get("input_sha256", {})) == expected, "Incomplete audit input bindings")
    for name in expected:
        require(
            file_sha256(source / name) == receipt["input_sha256"][name],
            "Audited input changed: " + name,
        )
    auditor_sha = preparation.get("source_sha256", {}).get(AUDITOR)
    require(
        isinstance(auditor_sha, str)
        and receipt.get("auditor_sha256") == auditor_sha == file_sha256(ROOT / AUDITOR),
        "Independent auditor differs from the prepared source snapshot",
    )
    require(set(receipt.get("views", {})) == set(VIEWS), "Audit omitted a lexical view")
    for view in VIEWS:
        stats = receipt["views"][view]
        require(
            isinstance(stats.get("numeric"), int)
            and stats["numeric"] > 0
            and isinstance(stats.get("exact"), int)
            and stats["exact"] > 0
            and math.isfinite(stats.get("maximum_numeric_difference", math.inf))
            and 0 <= stats["maximum_numeric_difference"] <= 1e-10,
            "Invalid independent metric comparison receipt",
        )
    seal = read_json(audit / "reconstruction_seal.json")
    names = {view + "_reconstruction.json" for view in VIEWS}
    require(
        seal.get("schema") == "format-adaptation-text-metric-reconstruction-seal-v1"
        and seal.get("saved_summary_opened") is False
        and set(seal.get("views_reconstructed", [])) == set(VIEWS)
        and set(seal.get("output_sha256", {})) == names,
        "Both independent reconstructions must be sealed before summary access",
    )
    for name in names:
        require(
            file_sha256(audit / name) == seal["output_sha256"][name],
            "Independent reconstruction changed: " + name,
        )
    return receipt, seal


def _panel(metric):
    correct, total, accuracy = (
        metric["complete_quartets"],
        metric["quartets"],
        metric["quartet_accuracy"],
    )
    require(
        type(correct) is int
        and type(total) is int
        and total == 192
        and 0 <= correct <= total
        and math.isfinite(accuracy)
        and math.isclose(accuracy, correct / total, rel_tol=1e-12, abs_tol=1e-12),
        "Invalid displayed hard-quartet endpoint",
    )
    return f"{correct}/{total} ({100 * accuracy:.2f}%)"


def _auc(value):
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1, "Bad AUROC")
    return f"{value:.4f}"


def _validate_compact(metrics, summary):
    expected = {
        **{key: value for key, value in summary.items() if key != "views"},
        "views": {
            view: {
                "arms": {
                    arm: {scope: entry["aggregate"] for scope, entry in scopes.items()}
                    for arm, scopes in result["arms"].items()
                },
                "factor_diagnostics": {
                    arm: {scope: entry["aggregate"] for scope, entry in scopes.items()}
                    for arm, scopes in result["factor_diagnostics"].items()
                },
                "comparisons": result["comparisons"],
            }
            for view, result in summary["views"].items()
        },
    }
    require(metrics == expected, "Compact display metrics differ from the audited full summary")


def _latent_display(directory, *, allow_test_fixture):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "manifest.json")
    require(
        manifest.get("schema") == "format-adaptation-report-manifest-v1"
        and manifest.get("status") == "complete"
        and file_sha256(directory / "summary.json")
        == manifest.get("output_sha256", {}).get("summary.json"),
        "Latent display summary differs from its completed report manifest",
    )
    summary = read_json(directory / "summary.json")
    require(
        summary.get("schema") == "format-adaptation-report-v1"
        and (summary.get("test_fixture") is False or allow_test_fixture)
        and set(summary.get("models", {})) == {"llama", "qwen"},
        "Invalid latent display report or unauthorized fixture",
    )
    display = {
        "scope": "Previously completed latent result, displayed only; no new latent analysis or primary",
        "source_sha256": {
            "summary.json": file_sha256(directory / "summary.json"),
            "manifest.json": file_sha256(directory / "manifest.json"),
        },
        "models": {},
    }
    for model in ("llama", "qwen"):
        display["models"][model] = {}
        for arm in ARMS:
            display["models"][model][arm] = {}
            for renderer in ("A", "B"):
                metric = summary["models"][model]["arms"][arm][renderer + "/correction_hard"][
                    "aggregate"
                ]
                _panel(metric)
                display["models"][model][arm][renderer] = {
                    key: metric[key]
                    for key in ("complete_quartets", "quartets", "quartet_accuracy")
                }
    return display


def _markdown(metrics, latent, fixture):
    lines = [
        "# Matched lexical shortcut controls: audited exposed-bank diagnostic",
        "",
        "This posthoc lexical shortcut control uses the same 16 fitting and 24 test contexts as "
        "the completed latent study. It is one shared text comparator, not an independent "
        "replication per checkpoint. Human semantic review remains incomplete. "
        "The original latent primary and earlier null results remain unchanged.",
        "",
        "Each arm predicts four labelled factors and uses the same supplied external update "
        "rule. Every lexical vocabulary and IDF is fitted on that arm's authorized observations "
        "only. Mixed fitting changes lexical features and coefficients. Offset preserves "
        "Ahalf256's feature map and coefficients and learns only four B intercepts.",
        "",
        "| Instrument / format | A512 hard quartets | Mixed512 | Ahalf256 | Offset |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    if fixture:
        lines.insert(0, "TEST FIXTURE — NO SCIENTIFIC RESULT\n")
    for view in VIEWS:
        for renderer in ("A", "B"):
            cells = [
                _panel(metrics["views"][view]["arms"][arm][renderer + "/correction_hard"])
                for arm in ARMS
            ]
            lines.append(f"| Text {view} / {renderer} | " + " | ".join(cells) + " |")
    for model in ("llama", "qwen"):
        for renderer in ("A", "B"):
            cells = [_panel(latent["models"][model][arm][renderer]) for arm in ARMS]
            lines.append(f"| Previous latent {model} / {renderer} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Complete-quartet counts must be read alongside row accuracy, class recalls and "
            "nuisance sensitivity. A larger quartet count alone does not establish recovery of "
            "the attributed-belief target. The format-B hard controls for every lexical arm follow.",
            "",
            "| Text view | Arm | B-hard correct rows | Recall 0 | Recall 1 | "
            "World flips / pairs | Name-map flips / pairs | Both correct after name-map change |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for view in VIEWS:
        for arm in ARMS:
            metric = metrics["views"][view]["arms"][arm]["B/correction_hard"]
            world = metric["nuisance_pairs"]["world"]
            names = metric["nuisance_pairs"]["name_map"]
            lines.append(
                f"| {view} | {arm} | {metric['correct']}/{metric['rows']} "
                f"({100 * metric['row_accuracy']:.2f}%) | "
                f"{100 * metric['class_recalls']['0']:.2f}% | "
                f"{100 * metric['class_recalls']['1']:.2f}% | "
                f"{world['verdict_flips']}/{world['pairs']} | "
                f"{names['verdict_flips']}/{names['pairs']} | "
                f"{names['both_correct']}/{names['pairs']} |"
            )
    lines.extend(
        [
            "",
            "The two receipt-factor AUROCs below pool correction-easy and correction-hard "
            "rows in format B. Full per-stratum, class, context and nuisance metrics are retained.",
            "",
            "| Text view | Arm | B1 AUROC | B2 AUROC |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for view in VIEWS:
        for arm in ARMS:
            factors = metrics["views"][view]["factor_diagnostics"][arm]["B/correction_all"][
                "factors"
            ]
            lines.append(
                f"| {view} | {arm} | {_auc(factors['B1']['raw_auroc'])} | "
                f"{_auc(factors['B2']['raw_auroc'])} |"
            )
    lines.extend(
        [
            "",
            "The full-messages baseline sees roles and all supplied prose, including the final "
            "answer-format instruction; the latent site precedes that instruction. These are "
            "not identical causal token prefixes. The lexical baseline omits chat special tokens. "
            "System-only discards user content and is an instruction/template negative control.",
            "",
            "N-grams are a shallow text control, not a capable semantic judge. Failure of this "
            "baseline cannot establish incremental activation value, model-internal composition, "
            "spontaneous belief attribution or natural deception detection. Equal numerical C "
            "does not equalize lexical and latent feature geometry or capacity.",
            "",
            "## Complete evidence",
            "",
            "[metrics.json](metrics.json) retains all arms, formats, conditions, factor strata "
            "and fixed descriptive comparisons. [summary.json.gz](summary.json.gz) also retains "
            "all 24 context outcomes. [rows.json.gz](rows.json.gz) and "
            "[predictions.npz](predictions.npz) preserve formal test metadata and saved margins. "
            "These are exact original output bytes, not new fits or predictions.",
            "",
            "The two B-hard comparisons use the original paired whole-context bootstrap policy. "
            "Their intervals condition on the exposed bank and fixed fitted instruments and "
            "exclude fitting-cohort uncertainty. They are posthoc descriptive comparisons, "
            "with no new primary or multiplicity guarantee. One-step and asserted conditions "
            "are out-of-training-construct controls; factor truth is evaluated on correction only.",
            "",
            "[audit/audit_receipt.json](audit/audit_receipt.json) verifies independently "
            "reconstructed metrics from saved margins. Both reconstruction JSON files are "
            "losslessly compressed here; their decompressed hashes match the original "
            "[reconstruction seal](audit/reconstruction_seal.json). This is not an independent "
            "refit, feature reconstruction or human semantic certification.",
            "",
            "## Restore the fitted text instruments",
            "",
            "The trusted [fitted-text-readouts.zip](fitted-text-readouts.zip) contains exactly "
            "six fixed-path JSON/NPZ files. No pickle or executable files are included. Extract "
            "into a new directory from this report folder:",
            "",
            "```console",
            "python -m zipfile -e fitted-text-readouts.zip restored",
            "```",
            "",
            "From the locked project environment, use `load_text('restored/fit/full_messages')` "
            "or `load_text('restored/fit/system_only')` from "
            "`latent_probes.format_adaptation_text`. Score records must contain exactly "
            "`messages` and `renderer`; no labels or formal-state metadata enter the text map. "
            "The archive members are bound to [fit_seal.json](fit_seal.json), and every exported "
            "file is bound to [manifest.json](manifest.json). CREATE BF16 replication and a "
            "capable text baseline remain separate follow-ups.",
            "",
        ]
    )
    return "\n".join(lines)


def report(source, audit, output, *, latent_root=None, figures=(), allow_test_fixture=False):
    source, audit, output = (Path(p).resolve() for p in (source, audit, output))
    require(source.is_dir() and audit.is_dir(), "Existing source and audit directories required")
    require(not output.exists(), "Output already exists; preserve all attempts")
    require(
        not output.is_relative_to(source) and not output.is_relative_to(audit),
        "Report output must be outside source and audit directories",
    )
    require(verify_pipeline(source).get("status") == "pass", "Sealed pipeline verification failed")
    preparation = read_json(source / "preparation.json")
    fixture = preparation.get("test_fixture", False)
    require(type(fixture) is bool and (not fixture or allow_test_fixture), "Unauthorized fixture")
    require(
        preparation.get("schema") == "adaptation-lexical-preparation-v1"
        and preparation.get("human_review_complete") is False,
        "Wrong prepared diagnostic status",
    )
    receipt, seal = _validate_audit(source, audit, preparation)
    fit_seal = read_json(source / "fit_seal.json")
    require(set(fit_seal["files_sha256"]) == set(FIT_NAMES), "Unexpected fitted archive inventory")
    for name in FIT_NAMES:
        require(
            file_sha256(source / name) == fit_seal["files_sha256"][name], "Changed fit: " + name
        )
    metrics = read_json(source / "metrics.json")
    summary = _read_compressed(source / "summary.json.gz")
    for data in (metrics, summary):
        require(
            data.get("schema") == "adaptation-lexical-summary-v1"
            and data.get("research_status") == "posthoc_exposed_bank_diagnostic"
            and data.get("human_review_complete") is False
            and data.get("shared_text_comparator_not_checkpoint_replications") is True
            and data.get("new_language_model_forwards") == 0
            and set(data.get("views", {})) == set(VIEWS),
            "Wrong lexical result scope or unauthorized claim",
        )
    _validate_compact(metrics, summary)
    latent_root = Path(latent_root or ROOT / "docs/results/format-adaptation-2026-11-27")
    latent = _latent_display(latent_root, allow_test_fixture=allow_test_fixture)
    markdown = _markdown(metrics, latent, fixture)
    figure_paths, figure_names = [], set()
    for figure in figures:
        figure = Path(figure).resolve()
        require(
            figure.is_file() and figure.suffix.lower() in {".pdf", ".svg", ".png"}, "Bad figure"
        )
        name = "figure/lexical-controls" + figure.suffix.lower()
        require(name not in figure_names, "Duplicate figure extension")
        figure_names.add(name)
        figure_paths.append((figure, name))
    inputs = {
        str(path): file_sha256(path)
        for path in (
            *(source / name for name in (*SOURCE_NAMES, *FIT_NAMES)),
            audit / "audit_receipt.json",
            audit / "reconstruction_seal.json",
            *(audit / name for name in seal["output_sha256"]),
            latent_root / "summary.json",
            latent_root / "manifest.json",
            *(path for path, _ in figure_paths),
        )
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit").mkdir()
    for name in SOURCE_NAMES:
        shutil.copyfile(source / name, output / name)
    for name in ("audit_receipt.json", "reconstruction_seal.json"):
        shutil.copyfile(audit / name, output / "audit" / name)
    reconstruction_exports = {}
    for name, digest in seal["output_sha256"].items():
        destination = output / "audit" / (name + ".gz")
        destination.write_bytes(gzip.compress((audit / name).read_bytes(), mtime=0))
        reconstruction_exports[destination.relative_to(output).as_posix()] = {
            "decompressed_original_sha256": digest,
        }
    with zipfile.ZipFile(output / "fitted-text-readouts.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in FIT_NAMES:
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o100600 << 16
            archive.writestr(member, (source / name).read_bytes(), compresslevel=9)
    for path, name in figure_paths:
        (output / "figure").mkdir(exist_ok=True)
        shutil.copyfile(path, output / name)
    (output / "latent-comparison.json").write_text(
        json.dumps(latent, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(markdown, encoding="utf-8")
    # Close the copy window before claiming the output is complete.
    for path, digest in inputs.items():
        require(file_sha256(path) == digest, "Input changed during report export: " + path)
    for name in SOURCE_NAMES:
        require(file_sha256(output / name) == file_sha256(source / name), "Copy changed: " + name)
    for name in ("audit_receipt.json", "reconstruction_seal.json"):
        require(
            file_sha256(output / "audit" / name) == file_sha256(audit / name), "Audit copy changed"
        )
    manifest = {
        "schema": "format-adaptation-text-report-manifest-v1",
        "status": "complete",
        "created_at": utc_now(),
        "test_fixture": fixture,
        "research_status": "posthoc_exposed_bank_lexical_shortcut_diagnostic",
        "human_review_complete": False,
        "new_language_model_forwards": 0,
        "classifier_fits": 0,
        "source_sha256": file_sha256(__file__),
        "input_sha256": inputs,
        "independent_audit_numeric_checks": {view: receipt["views"][view] for view in VIEWS},
        "fitted_archive_members_sha256": fit_seal["files_sha256"],
        "lossless_reconstruction_exports": reconstruction_exports,
        "output_sha256": {
            p.relative_to(output).as_posix(): file_sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "audit", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--latent-root", type=Path)
    parser.add_argument("--figure", action="append", default=[], type=Path)
    options = parser.parse_args()
    result = report(
        options.source,
        options.audit,
        options.output,
        latent_root=options.latent_root,
        figures=options.figure,
    )
    print(json.dumps({"status": result["status"], "exported_files": len(result["output_sha256"])}))


if __name__ == "__main__":
    main()
