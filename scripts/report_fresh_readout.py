"""Export two completed fresh analyses without fitting, selection or model calls."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np

CORE = ("S0", "S1", "C1", "C2", "output")
CONDITIONS = ("one_step", "correction_easy", "correction_hard", "asserted")
STRATA = tuple(f"{r}/{c}" for r in ("A", "B") for c in CONDITIONS)
BLOCKS = {"llama": (8, 16, 24, 30), "qwen": (7, 14, 21, 26)}
POSITIONS = ("account_last", "query_location_last", "prompt_last")
MODELS = {
    "llama": ("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659"),
    "qwen": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
}
STATUS = "agent_audited_exploratory"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    value = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def close(a, b):
    return (
        math.isfinite(float(a))
        and math.isfinite(float(b))
        and math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
    )


def fraction(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def check_metric(record, rows, quartets):
    require(
        record["rows"] == rows
        and type(record["correct"]) is int
        and 0 <= record["correct"] <= rows
        and close(record["row_accuracy"], record["correct"] / rows),
        "Row metric arithmetic differs",
    )
    require(
        set(record["class_recalls"]) == {"0", "1"}
        and all(fraction(v) for v in record["class_recalls"].values()),
        "Invalid class recalls",
    )
    require(record["quartets"] == quartets, "Four-case panel inventory differs")
    histogram = record["quartet_correct_cells_histogram"]
    require(
        set(histogram) == set(map(str, range(5)))
        and all(type(v) is int and v >= 0 for v in histogram.values())
        and sum(histogram.values()) == quartets,
        "Quartet histogram differs",
    )
    require(record["complete_quartets"] == histogram["4"], "Complete quartet count differs")
    if quartets:
        require(
            close(record["quartet_accuracy"], histogram["4"] / quartets)
            and sum(int(k) * v for k, v in histogram.items()) == record["correct"],
            "Quartet arithmetic differs",
        )
    else:
        require(record["quartet_accuracy"] is None, "Assertions cannot acquire quartet metrics")
    for field in ("nuisance_pairs", "receipt_pairs"):
        require(isinstance(record[field], dict), "Missing intervention controls")
    require(
        set(record["nuisance_pairs"]) == {"world", "name_map"}
        and set(record["receipt_pairs"]) == {"first_receipt", "receipt"},
        "Control inventory differs",
    )
    pairs = list(record["nuisance_pairs"].values())
    for value in record["receipt_pairs"].values():
        require(set(value) == {"changing", "preserving"}, "Receipt control strata differ")
        pairs.extend(value.values())
    for value in pairs:
        require(
            set(value) == {"pairs", "verdict_flips", "both_correct"}
            and all(type(v) is int and 0 <= v <= rows for v in value.values())
            and value["both_correct"] <= value["pairs"]
            and value["verdict_flips"] <= value["pairs"],
            "Invalid pair control counts",
        )
    diagnostics = record.get("unrecalibrated_score_diagnostics")
    if diagnostics is not None:
        require(
            fraction(diagnostics["auroc"])
            and fraction(diagnostics["brier"])
            and fraction(diagnostics["equal_width_ece10"])
            and math.isfinite(diagnostics["log_loss"])
            and diagnostics["log_loss"] >= 0,
            "Invalid raw score diagnostics",
        )


def check_primary(result, groups):
    primary = result["primary"]
    require(
        primary["contrast"] == "S1_minus_S0"
        and primary["renderer"] == "A"
        and primary["condition"] == "correction_hard"
        and primary["endpoint"] == "complete_four_case_quartet_accuracy"
        and primary["groups"] == 24
        and primary["quartets"] == 192,
        "Wrong primary endpoint/inventory",
    )
    require(
        primary["bootstrap_samples"] == 10000 and primary["bootstrap_seed"] == 20261122,
        "Wrong saved conditional bootstrap policy",
    )
    values = primary["group_values"]
    require([v["group_id"] for v in values] == sorted(groups), "Wrong primary paired contexts")
    by_method = {
        m: {g["group_id"]: g for g in result["methods"][m]["strata"]["A/correction_hard"]["groups"]}
        for m in ("S0", "S1")
    }
    for value in values:
        require(
            all(
                fraction(value[m])
                and close(value[m], by_method[m][value["group_id"]]["quartet_accuracy"])
                for m in ("S0", "S1")
            )
            and close(value["difference"], value["S1"] - value["S0"]),
            "Primary group difference arithmetic differs",
        )
    for method in ("S0", "S1"):
        require(
            close(primary[method], np.mean([v[method] for v in values]))
            and close(
                primary[method],
                result["methods"][method]["strata"]["A/correction_hard"]["aggregate"][
                    "quartet_accuracy"
                ],
            ),
            "Primary mean differs from frozen method metrics",
        )
    require(
        close(primary["difference"], np.mean([v["difference"] for v in values]))
        and close(primary["difference"], primary["S1"] - primary["S0"]),
        "Primary difference arithmetic differs",
    )
    interval = primary["conditional_percentile_interval_95"]
    require(
        isinstance(interval, list)
        and len(interval) == 2
        and all(type(v) in (int, float) and math.isfinite(v) and -1 <= v <= 1 for v in interval)
        and interval[0] <= interval[1],
        "Invalid saved conditional primary interval",
    )


def check_strict_first_token(diagnostics):
    require(
        isinstance(diagnostics, dict) and set(diagnostics) == set(STRATA),
        "Need all eight strict first-token diagnostic strata",
    )
    for stratum, value in diagnostics.items():
        n = 192 if stratum.endswith("/asserted") else 768
        counts = (
            "candidate_first_tokens",
            "outside_candidate_first_tokens",
            "correct_full_denominator",
            "full_vocab_argmax_ties",
        )
        require(
            isinstance(value, dict)
            and set(value)
            == {
                "rows",
                *counts,
                "accuracy_full_denominator",
                "candidate_first_token_rate",
                "mean_candidate_token_mass",
                "scope",
            },
            "Wrong strict first-token diagnostic fields",
        )
        require(
            value["rows"] == n
            and all(type(value[k]) is int and 0 <= value[k] <= n for k in counts),
            "Invalid strict first-token counts",
        )
        require(
            value["candidate_first_tokens"] + value["outside_candidate_first_tokens"] == n
            and value["correct_full_denominator"] <= value["candidate_first_tokens"]
            and close(value["accuracy_full_denominator"], value["correct_full_denominator"] / n)
            and close(value["candidate_first_token_rate"], value["candidate_first_tokens"] / n),
            "Strict first-token full-denominator arithmetic differs",
        )
        mass = value["mean_candidate_token_mass"]
        require(
            type(mass) in (int, float)
            and math.isfinite(mass)
            and 0 <= mass <= 1 + 1e-12
            and isinstance(value["scope"], str)
            and value["scope"],
            "Invalid strict first-token probability mass/scope",
        )


def _fit_metadata(result, model):
    paths = [Path(p) for p in result["source_sha256"] if Path(p).name == "settings.json"]
    require(len(paths) == 1, "Analysis must bind one frozen fitting settings file")
    settings = read(paths[0])
    require(
        settings["schema"] == "fresh-readout-fit-v2"
        and settings["status"] == "frozen"
        and settings["test_fixture"] is result["test_fixture"]
        and settings["research_status"] == STATUS
        and settings["human_review_complete"] is False
        and settings["confirmatory"] is False,
        "Invalid upstream fit status",
    )
    provenance = settings["capture_provenance"]
    model_settings = provenance["model_settings"]
    require(
        provenance["model_key"] == model
        and (model_settings["model_id"], model_settings["revision"]) == MODELS[model],
        "Wrong pinned model identity",
    )
    sites = [f"layer_{b}_{p}" for b in BLOCKS[model] for p in POSITIONS]
    require(
        provenance["sites"] == sites and [p["site"] for p in settings["source_panel"]] == sites,
        "Wrong source site panel",
    )
    require(
        all(
            len(settings[p + "_ids"]) == 512
            for p in ("source_fit", "source_dev", "correction_fit", "correction_dev")
        ),
        "Fresh pipelines must retain separate N512 partitions",
    )
    heads, pipelines = settings["heads"], settings["pipelines"]
    original, selected = (heads[pipelines[m]["head_key"]] for m in ("S0", "S1"))
    require(
        original["site"] == f"layer_{16 if model == 'llama' else 14}_prompt_last"
        and result["original_source_site"] == original["site"]
        and result["selected_source_site"] == selected["site"]
        and selected["site"] in sites,
        "Selected/original source metadata differs",
    )
    bank_path = Path(settings["fit_context"]["bank_path"])
    require(
        result["source_sha256"].get(str(bank_path))
        == digest(bank_path)
        == provenance["bank_file_sha256"],
        "Analysis does not bind its exact fresh bank",
    )
    bank = read(bank_path)
    require(
        bank["schema"] == "fresh-readout-bank-v2" and canonical(bank) == result["bank_sha256"],
        "Wrong fresh-bank semantic hash",
    )
    require(
        Counter(g["split"] for g in bank["groups"])
        == Counter(correction_fit=16, correction_dev=8, test=24),
        "Wrong 48-context fresh bank",
    )
    groups = sorted(g["group_id"] for g in bank["groups"] if g["split"] == "test")
    require(len(set(groups)) == 24, "Wrong paired test context identities")
    return (
        {
            "model_id": model_settings["model_id"],
            "original_source_site": original["site"],
            "original_source_C": original["C"],
            "selected_source_site": selected["site"],
            "selected_source_C": selected["C"],
            "source_panel": settings["source_panel"],
        },
        sites,
        groups,
    )


def _compact_methods(result, sites, groups):
    expected = (*CORE, *("source_" + s for s in sites))
    require(set(result["methods"]) == set(expected), "Must retain all seventeen frozen methods")
    compact = {}
    for name in expected:
        method = result["methods"][name]
        require(set(method["strata"]) == set(STRATA), "Missing renderer/condition stratum")
        expected_kind = (
            "synthetic_hard_fold"
            if name == "C2"
            else "conditional_YES_NO_logitdiff"
            if name == "output"
            else "logistic_decision"
        )
        require(method["margin_kind"] == expected_kind, "Wrong margin interpretation")
        strata = {}
        for stratum in STRATA:
            record = method["strata"][stratum]
            asserted = stratum.endswith("/asserted")
            n, q = (192, 0) if asserted else (768, 192)
            check_metric(record["aggregate"], n, q)
            group_records = record["groups"]
            require(
                [g["group_id"] for g in group_records] == groups,
                "Wrong/shared contextual group inventory",
            )
            for group in group_records:
                check_metric(group, n // 24, q // 24)
            for field in ("correct", "complete_quartets"):
                require(
                    sum(g[field] for g in group_records) == record["aggregate"][field],
                    "Aggregate differs from all context values",
                )
            for field in ("row_accuracy", "quartet_accuracy"):
                if not asserted or field != "quartet_accuracy":
                    require(
                        close(
                            record["aggregate"][field], np.mean([g[field] for g in group_records])
                        ),
                        "Equal-context summary differs",
                    )
            strata[stratum] = record["aggregate"]
        compact[name] = {
            "margin_kind": method["margin_kind"],
            "strata": strata,
            "renderer_A_hard_group_values": [
                {
                    "group_id": g["group_id"],
                    "row_accuracy": g["row_accuracy"],
                    "quartet_accuracy": g["quartet_accuracy"],
                }
                for g in method["strata"]["A/correction_hard"]["groups"]
            ],
            "renderer_differences": {
                c: {
                    field + "_B_minus_A": None
                    if strata["A/" + c][field] is None
                    else strata["B/" + c][field] - strata["A/" + c][field]
                    for field in ("row_accuracy", "quartet_accuracy")
                }
                for c in CONDITIONS
            },
        }
    return compact


def _predictions(path, method_names, metadata):
    with np.load(path, allow_pickle=False) as saved:
        require(
            set(saved.files)
            == {
                "row_ids",
                "methods",
                "verdicts",
                "margins",
                "factor_predictions",
                "factor_margins",
            },
            "Wrong safe prediction inventory",
        )
        data = {key: saved[key].copy() for key in saved.files}
    require(
        data["row_ids"].dtype.kind in "US"
        and data["row_ids"].shape == (4992,)
        and len(set(data["row_ids"].tolist())) == 4992
        and data["methods"].dtype.kind in "US"
        and data["methods"].tolist() == list(method_names),
        "Prediction identities differ",
    )
    verdicts, margins = data["verdicts"], data["margins"]
    require(
        verdicts.shape == margins.shape == (4992, 17)
        and np.isin(verdicts, (0, 1)).all()
        and np.isfinite(margins).all()
        and np.array_equal(verdicts, margins > 0),
        "Predictions violate frozen positive orientation/strict threshold",
    )
    bits, factor_margins = data["factor_predictions"], data["factor_margins"]
    require(
        bits.shape == factor_margins.shape == (4992, 4)
        and np.isfinite(factor_margins).all()
        and np.isin(bits, (0, 1)).all()
        and np.array_equal(bits, factor_margins > 0),
        "Invalid factor predictions",
    )
    i, u, b1, b2 = bits.T
    require(
        np.array_equal(
            verdicts[:, list(method_names).index("C2")], np.where(b2, 1 - u, np.where(b1, u, i))
        ),
        "C2 fold differs from saved factors",
    )
    for method, site_field in (("S0", "original_source_site"), ("S1", "selected_source_site")):
        reference = "source_" + metadata[site_field]
        columns = [list(method_names).index(name) for name in (method, reference)]
        require(
            np.array_equal(verdicts[:, columns[0]], verdicts[:, columns[1]])
            and np.array_equal(margins[:, columns[0]], margins[:, columns[1]]),
            "Named source pipeline differs from its frozen panel head",
        )
    return data["row_ids"].tolist()


def load_report(llama_analysis, qwen_analysis, *, allow_test_fixture=False):
    directories = {"llama": Path(llama_analysis).resolve(), "qwen": Path(qwen_analysis).resolve()}
    # Establish BOTH completed exports before reading either set of model results.
    manifests = {k: read(p / "manifest.json") for k, p in directories.items()}
    for manifest in manifests.values():
        require(
            manifest.get("schema") == "fresh-readout-analysis-manifest-v2"
            and manifest.get("status", "complete") == "complete"
            and set(manifest.get("output_sha256", {})) == {"results.json", "predictions.npz"},
            "Both analyses must be complete before reporting",
        )
    for model, directory in directories.items():
        for name, expected in manifests[model]["output_sha256"].items():
            require(
                digest(directory / name) == expected, "Completed analysis output changed: " + name
            )
    results = {k: read(p / "results.json") for k, p in directories.items()}
    summary = {
        "schema": "fresh-readout-report-v2",
        "created_at": datetime.now(UTC).isoformat(),
        "research_status": STATUS,
        "human_review_complete": False,
        "confirmatory_result": False,
        "new_language_model_calls": 0,
        "classifier_fits": 0,
        "new_selection": False,
        "paired_test_contexts": 24,
        "test_contexts_shared_between_models": True,
        "independent_models_pooled": False,
        "hard_quartets_per_renderer_per_model": 192,
        "input_sha256": {
            str(directory / name): digest(directory / name)
            for directory in directories.values()
            for name in ("manifest.json", "results.json", "predictions.npz")
        },
        "reporter_source_sha256": {str(Path(__file__).resolve()): digest(__file__)},
        "upstream_source_sha256": {},
        "models": {},
    }
    group_inventories, row_inventories, bank_hashes, fixtures = [], [], [], []
    for model, result in results.items():
        require(
            result.get("schema") == "fresh-readout-analysis-v2"
            and result.get("model_key") == model
            and result.get("research_status") == STATUS
            and result.get("human_review_complete") is False
            and result.get("confirmatory_result") is False
            and result.get("fresh_test") is True,
            "Wrong model, fresh status or exploratory research status",
        )
        fixture = result.get("test_fixture")
        require(
            type(fixture) is bool and (not fixture or allow_test_fixture),
            "Fixture analyses require explicit allow_test_fixture",
        )
        require(
            result["primary_role"]
            == ("primary" if model == "llama" else "predeclared_replication"),
            "Wrong predeclared model role",
        )
        require(
            result["source_sha256"] == manifests[model]["source_sha256"],
            "Analysis source manifest differs",
        )
        for path, expected in result["source_sha256"].items():
            require(
                Path(path).is_file() and digest(path) == expected,
                "Analysis upstream source changed: " + path,
            )
            require(
                summary["upstream_source_sha256"].get(path, expected) == expected,
                "Shared upstream source bindings differ between models",
            )
            summary["upstream_source_sha256"][path] = expected
        metadata, sites, groups = _fit_metadata(result, model)
        methods = _compact_methods(result, sites, groups)
        check_primary(result, groups)
        strict_first_token = result.get("strict_first_token_diagnostic")
        check_strict_first_token(strict_first_token)
        factors = result["factor_diagnostics"]
        require(
            set(factors)
            == {
                "scope",
                *(f"{r}/{c}" for r in ("A", "B") for c in ("correction_easy", "correction_hard")),
            },
            "Factor truth diagnostics must remain on two-message rows",
        )
        for stratum, value in factors.items():
            if stratum == "scope":
                continue
            require(
                value["rows"] == 768
                and value["factor_order"] == ["I", "M1", "B1", "B2"]
                and len(value["accuracy"]) == 4
                and all(fraction(v) for v in value["accuracy"])
                and fraction(value["all_four_correct"])
                and value["all_four_correct"] <= min(value["accuracy"]) + 1e-12,
                "Invalid four-factor diagnostic",
            )
        row_inventories.append(
            _predictions(directories[model] / "predictions.npz", methods, metadata)
        )
        group_inventories.append(groups)
        bank_hashes.append(result["bank_sha256"])
        fixtures.append(fixture)
        summary["models"][model] = {
            **metadata,
            "primary_role": result["primary_role"],
            "primary": result["primary"],
            "methods": methods,
            "factor_diagnostics": factors,
            "strict_first_token_diagnostic": strict_first_token,
            "source_sha256": result["source_sha256"],
        }
    require(
        group_inventories[0] == group_inventories[1]
        and row_inventories[0] == row_inventories[1]
        and bank_hashes[0] == bank_hashes[1]
        and fixtures[0] is fixtures[1],
        "Both models must retain the same bank and paired 24 contexts/4992 cases",
    )
    summary.update(
        test_fixture=fixtures[0],
        fresh_test=not fixtures[0],
        bank_sha256=bank_hashes[0],
        report_kind="engineering_fixture_not_scientific_result"
        if fixtures[0]
        else "completed_fresh_pipeline_comparison",
    )
    return summary


def pct(value):
    return "—" if value is None else f"{100 * value:.2f}"


def pair_pct(value, key):
    return "—" if not value["pairs"] else pct(value[key] / value["pairs"])


def markdown(summary):
    fixture = summary["test_fixture"]
    title = (
        "ENGINEERING FIXTURE — no scientific result"
        if fixture
        else "Fresh readout study: frozen pipeline comparison"
    )
    p = summary["models"]["llama"]["primary"]
    lo, hi = p["conditional_percentile_interval_95"]
    lines = [
        f"# {title}",
        "",
        "Agent-audited exploratory; human semantic review remains incomplete. "
        "All measurements below are synthetic placeholders."
        if fixture
        else "Agent-audited exploratory; human semantic review remains incomplete. This is not a confirmatory result.",
        "",
        f"Llama renderer-A hard complete-quartet S1−S0: **{100 * p['difference']:+.2f} percentage points**; "
        f"saved conditional 95% interval **[{100 * lo:+.2f}, {100 * hi:+.2f}]**. "
        "The saved bootstrap uses 10,000 whole-context draws, seed 20261122, over 24 paired contexts "
        "and 192 hard quartets per renderer/model. The interval conditions on authored contexts, two "
        "renderers, selected sites and fitted heads; it does not cover their construction uncertainty.",
        "",
    ]
    q = summary["models"]["qwen"]["primary"]
    lines += [
        f"Qwen is the predeclared replication, retained with S1−S0 **{100 * q['difference']:+.2f} percentage points**. "
        "Both models share the same 24 contexts; they do not provide 48 independent contextual units. "
        "No model or method was removed on the basis of its performance.",
        "",
        "## Frozen source selection and controls",
        "",
        "| Model | S0 site | S0 C | S1 site | S1 C |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for model, data in summary["models"].items():
        lines.append(
            f"| {model.title()} | {data['original_source_site']} | {data['original_source_C']:g} | "
            f"{data['selected_source_site']} | {data['selected_source_C']:g} |"
        )
    lines += [
        "",
        "Source fitting and source-development selection use separate original N512 partitions. "
        "Each of the twelve source sites has its own frozen development-selected C. S0 is the original "
        "site; S1 was selected using source development only. The twelve-site panel remains descriptive: "
        "no fresh-test winner is selected. C1 and C2 use the same S1 site, separate correction N512 "
        "fitting and N512 development partitions, and no development absorption.",
        "The source fit covers 32 existing observation/message groups; correction fitting covers 16 "
        "new correction groups. Equal row count does not hold story count or mechanism mix fixed, so "
        "this comparison does not isolate a causal effect of adding one correction operator.",
        "",
        "C2 predicts four separately supervised factors and applies an externally supplied update fold. "
        "Those extra labels and computation are privileged; C1 versus C2 is not an equal-capacity "
        "intervention. Earlier exposed cached factor results used 1,536 outer-training rows, whereas "
        "these fresh controls fit 512 rows. Cached success does not certify this smaller instrument.",
        "",
        "![Frozen hard-quartet comparison](fresh_readout.svg)",
        "",
        "## Every method and stratum",
        "",
        "Values are percentages. R0/R1 are class recalls; world/name are verdict-flip rates. "
        "B1/B2 changing and preserving entries are both-counterparts-correct rates. “—” indicates "
        "an inapplicable metric or no eligible pairs. Four-case panels in one-step/easy strata retain "
        "their own label structure and are not assigned an independent-cell chance rate.",
        "",
    ]
    for stratum in STRATA:
        lines += [
            f"### {stratum}",
            "",
            "| Model | Method | Row | Panel/quartet | R0 | R1 | World flip | Name flip | B1 changing | B1 preserving | B2 changing | B2 preserving | Raw AUROC |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for model, data in summary["models"].items():
            for name, method in data["methods"].items():
                metric = method["strata"][stratum]
                controls = [
                    pair_pct(metric["nuisance_pairs"][n], "verdict_flips")
                    for n in ("world", "name_map")
                ]
                controls += [
                    pair_pct(metric["receipt_pairs"][n][c], "both_correct")
                    for n in ("first_receipt", "receipt")
                    for c in ("changing", "preserving")
                ]
                auc = metric.get("unrecalibrated_score_diagnostics", {}).get("auroc")
                values = [
                    pct(metric["row_accuracy"]),
                    pct(metric["quartet_accuracy"]),
                    pct(metric["class_recalls"]["0"]),
                    pct(metric["class_recalls"]["1"]),
                    *controls,
                    "—" if auc is None else f"{auc:.3f}",
                ]
                lines.append(f"| {model.title()} | {name} | " + " | ".join(values) + " |")
        lines.append("")
    lines += [
        "## Strict first-token elicitation diagnostic",
        "",
        "The recorded full-vocabulary argmax uses the lowest token index for exact ties. "
        "Argmax tokens outside YES/NO count incorrect in the full stratum denominator. "
        "Coverage and total YES/NO probability mass are separate from conditional YES/NO accuracy. "
        "This diagnostic does not observe a complete generated response and never replaces the fixed "
        "conditional output baseline or primary endpoint.",
        "",
        "| Model | Stratum | Rows | YES/NO argmax % | Outside YES/NO | Correct / all rows | Accuracy % | Argmax ties | Mean YES/NO mass % |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, data in summary["models"].items():
        for stratum in STRATA:
            value = data["strict_first_token_diagnostic"][stratum]
            lines.append(
                f"| {model.title()} | {stratum} | {value['rows']} | "
                f"{pct(value['candidate_first_token_rate'])} | {value['outside_candidate_first_tokens']} | "
                f"{value['correct_full_denominator']} / {value['rows']} | "
                f"{pct(value['accuracy_full_denominator'])} | {value['full_vocab_argmax_ties']} | "
                f"{pct(value['mean_candidate_token_mass'])} |"
            )
    lines += [
        "",
        "## Four-factor diagnostics",
        "",
        "| Model | Stratum | I | M1 | B1 | B2 | All four |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, data in summary["models"].items():
        for stratum, value in data["factor_diagnostics"].items():
            if stratum != "scope":
                lines.append(
                    f"| {model.title()} | {stratum} | "
                    + " | ".join(pct(v) for v in [*value["accuracy"], value["all_four_correct"]])
                    + " |"
                )
    lines += [
        "",
        "Factor truth is defined here only on the two-message conditions. C2 always uses its "
        "own predicted second receipt, including attempted transfer to other conditions. Moderate "
        "individual factor accuracy does not establish reliable joint folding.",
        "",
        "## Interpretation and provenance",
        "",
        "Failures and invalid controls remain in their original denominators. A zero-crossing "
        "interval is not equivalence. A source-selected null cannot rule out unselected correction "
        "accessibility. Poor one-step/easy controls limit a hard-composition interpretation; a renderer-B "
        "drop can reflect sensitivity to these two fixed formats. No result establishes absent belief "
        "information everywhere, causal use, spontaneous anticipation or deceptive intent.",
        "",
        "Margins keep their frozen positive-target orientation; raw AUROC is never sign-flipped or "
        "replaced by absolute AUROC. Sigmoid scores are unrecalibrated; YES/NO output probability is "
        "conditional on those two tokens and is not probability of generating a valid answer. Final-prefix "
        "output has later textual context than account/query-position probes. C2's hard fold has no "
        "calibrated target probability.",
        "",
        "[Compact machine-readable results](summary.json) preserve every method/stratum, class recall, "
        "control count, quartet histogram, raw Brier/log loss/ECE, factor diagnostic, renderer contrast "
        "and all 24 renderer-A hard context scores. The full source analyses retain other contextual "
        "details. [The manifest](manifest.json) binds both completed analyses, safe predictions, upstream "
        "sources, reporting code and all exports. This reporting step made zero classifier fits and zero "
        "language-model calls.",
        "",
    ]
    return "\n".join(lines)


def figures(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    x, width = np.arange(len(CORE)), 0.34
    for axis, (model, data) in zip(axes, summary["models"].items(), strict=True):
        for offset, renderer, color in ((-width / 2, "A", "#296b9c"), (width / 2, "B", "#d59136")):
            values = [
                100
                * data["methods"][m]["strata"][renderer + "/correction_hard"]["quartet_accuracy"]
                for m in CORE
            ]
            bars = axis.bar(
                x + offset, values, width=width, label=f"Renderer {renderer}", color=color
            )
            for bar, value in zip(bars, values, strict=True):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 1.2,
                    f"{value:.0f}" if float(value).is_integer() else f"{value:.1f}",
                    ha="center",
                    fontsize=8,
                )
        axis.set(
            title=model.title() + (" — primary" if model == "llama" else " — named replication"),
            xticks=x,
            xticklabels=["S0", "S1", "C1", "C2", "YES/NO"],
            ylim=(0, 106),
        )
        axis.set_yticks(np.arange(0, 101, 20))
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    p = summary["models"]["llama"]["primary"]
    lo, hi = p["conditional_percentile_interval_95"]
    fig.text(
        0.5,
        0.91,
        f"Llama A: S1−S0 {100 * p['difference']:+.1f} pp; saved conditional 95% interval [{100 * lo:+.1f}, {100 * hi:+.1f}]",
        ha="center",
        fontsize=9,
    )
    axes[0].set_ylabel("Hard complete-quartet accuracy (%)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=2, frameon=False
    )
    title = (
        "ENGINEERING FIXTURE — synthetic, not a result"
        if summary["test_fixture"]
        else "Frozen fresh readout comparison"
    )
    fig.suptitle(title, fontsize=13)
    fig.text(
        0.5,
        0.005,
        "24 shared paired contexts; 192 hard quartets per renderer/model. No test-selected winner or panel confidence intervals.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.88))
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(Path(output) / f"fresh_readout.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def export_report(llama_analysis, qwen_analysis, output, *, allow_test_fixture=False):
    output = Path(output).resolve()
    require(not output.exists(), "Use a fresh report directory; existing artifacts are preserved")
    summary = load_report(llama_analysis, qwen_analysis, allow_test_fixture=allow_test_fixture)
    output.mkdir(parents=True, exist_ok=False)
    # Bind the renderer before rendering; preserve a partial attempt on any failure.
    write_new(
        output / "preparation.json",
        {
            "schema": "fresh-readout-report-preparation-v2",
            "input_sha256": summary["input_sha256"],
            "upstream_source_sha256": summary["upstream_source_sha256"],
            "reporter_source_sha256": summary["reporter_source_sha256"],
            "test_fixture": summary["test_fixture"],
        },
    )
    write_new(output / "summary.json", summary)
    with (output / "results.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(markdown(summary))
    figures(summary, output)
    for path, expected in {
        **summary["input_sha256"],
        **summary["reporter_source_sha256"],
        **summary["upstream_source_sha256"],
    }.items():
        require(digest(path) == expected, "Report input/code changed during export")
    write_new(
        output / "manifest.json",
        {
            "schema": "fresh-readout-report-manifest-v2",
            "status": "complete",
            "report_kind": summary["report_kind"],
            "test_fixture": summary["test_fixture"],
            "research_status": STATUS,
            "human_review_complete": False,
            "confirmatory_result": False,
            "input_sha256": summary["input_sha256"],
            "upstream_source_sha256": summary["upstream_source_sha256"],
            "reporter_source_sha256": summary["reporter_source_sha256"],
            "output_sha256": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
        },
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-analysis", type=Path, required=True)
    parser.add_argument("--qwen-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-test-fixture",
        action="store_true",
        help="Engineering fixtures only; exports are marked synthetic",
    )
    args = parser.parse_args()
    summary = export_report(
        args.llama_analysis,
        args.qwen_analysis,
        args.output,
        allow_test_fixture=args.allow_test_fixture,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "report_kind": summary["report_kind"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
