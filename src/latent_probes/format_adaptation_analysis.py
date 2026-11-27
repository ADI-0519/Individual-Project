"""Held-out analysis for the sealed matched-budget format-adaptation protocol.

This module scores fixed safe arrays only. It contains no fitting, calibration,
site selection or checkpoint forward. Bootstrap draws resample paired contexts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from .format_adaptation_fit import (
    ARMS,
    FACTOR_FIELDS,
    FACTORS,
    check_bindings,
    fold_factors,
    load_capture,
    require,
    score_bundle,
    state_key,
    validate_bundle,
    validate_state,
)
from .storage import atomic_json, atomic_npz, canonical_hash, file_sha256, read_json, utc_now

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("one_step", "correction_easy", "correction_hard", "asserted")
POLICY_FILES = (
    "src/latent_probes/format_adaptation_analysis.py",
    "scripts/analyse_format_adaptation.py",
)
BOOTSTRAP_SEED = 20261127
BOOTSTRAP_SAMPLES = 10000


def validate_test_rows(rows, training_groups=()):
    require(isinstance(rows, list) and len(rows) == 4992, "Need exact 4992 held-out rows")
    ids = [r.get("row_id") for r in rows]
    require(
        all(isinstance(x, str) and x for x in ids) and len(ids) == len(set(ids)),
        "Duplicate/missing test row IDs",
    )
    require(
        all(r.get("partition") == "test" and r.get("split", "test") == "test" for r in rows),
        "Non-test or unknown split in analysis",
    )
    groups = {r["group_id"] for r in rows}
    require(
        len(groups) == 24 and groups.isdisjoint(training_groups),
        "Need 24 held-out contexts disjoint from fitting",
    )
    pairs, rosters = defaultdict(dict), defaultdict(set)
    for index, row in enumerate(rows):
        validate_state(row)
        renderer = row.get("renderer")
        require(renderer in ("A", "B"), "Unknown test renderer")
        key = state_key(row)
        require(renderer not in pairs[key], "Duplicate full-state/renderer test key")
        pairs[key][renderer] = index
        state = row["state"]
        if row["condition"] == "asserted":
            formal = (state["asserted"], state["world"], state["name_map"])
            family = "asserted"
        elif row["condition"] == "one_step":
            formal = tuple(state[f] for f in ("initial", "update", "receipt", "world", "name_map"))
            family = "one_step"
        else:
            formal = tuple(state[f] for f in (*FACTOR_FIELDS, "world", "name_map"))
            family = "correction"
        roster = rosters[(row["group_id"], renderer, family)]
        require(formal not in roster, "Duplicate formal test cell")
        roster.add(formal)
    require(
        len(pairs) == 2496 and all(set(pair) == {"A", "B"} for pair in pairs.values()),
        "Test population lacks exact A/B state counterparts",
    )
    for pair in pairs.values():
        require(rows[pair["A"]]["gold"] == rows[pair["B"]]["gold"], "A/B truth differs")
    for group, renderer, family in product(
        sorted(groups), ("A", "B"), ("one_step", "correction", "asserted")
    ):
        dimension = {"one_step": 5, "correction": 6, "asserted": 3}[family]
        require(
            rosters[(group, renderer, family)] == set(product((0, 1), repeat=dimension)),
            "Missing formal test roster cells",
        )
    expected = Counter(
        {
            (renderer, condition): count
            for renderer in ("A", "B")
            for condition, count in (
                ("one_step", 768),
                ("correction_easy", 768),
                ("correction_hard", 768),
                ("asserted", 192),
            )
        }
    )
    require(
        Counter((r["renderer"], r["condition"]) for r in rows) == expected,
        "Wrong held-out condition inventory",
    )
    return pairs


def class_summary(labels, predictions):
    truth, predicted = np.asarray(labels), np.asarray(predictions)
    require(
        truth.shape == predicted.shape
        and truth.ndim == 1
        and truth.size > 0
        and np.isin(truth, (0, 1)).all()
        and np.isin(predicted, (0, 1)).all(),
        "Invalid binary metric arrays",
    )
    return {
        "rows": len(truth),
        "correct": int(np.sum(truth == predicted)),
        "row_accuracy": float(np.mean(truth == predicted)),
        "class_counts": {str(k): int(np.sum(truth == k)) for k in (0, 1)},
        "class_recalls": {
            str(k): float(np.mean(predicted[truth == k] == k)) if np.any(truth == k) else None
            for k in (0, 1)
        },
    }


def verdict_metrics(rows, predictions, margins=None):
    predicted = np.asarray(predictions)
    labels = np.array([r["gold"] for r in rows])
    result = class_summary(labels, predicted)
    panels, nuisances = defaultdict(dict), {f: defaultdict(dict) for f in ("world", "name_map")}
    for row, value in zip(rows, predicted, strict=True):
        state = row["state"]
        group = (row["group_id"], row["renderer"], row["condition"])
        if row["condition"] != "asserted":
            key = (*group, state["update"], state["world"], state["name_map"])
            cell = (
                (state["initial"], state["receipt"])
                if row["condition"] == "one_step"
                else (state["first_receipt"], state["receipt"])
            )
            require(cell not in panels[key], "Duplicate quartet cell")
            panels[key][cell] = bool(value == row["gold"])
        for name in nuisances:
            key = (*group, tuple(sorted((k, v) for k, v in state.items() if k != name)))
            require(state[name] not in nuisances[name][key], "Duplicate nuisance counterpart")
            nuisances[name][key][state[name]] = (int(value), row["gold"])
    require(
        all(set(panel) == set(product((0, 1), repeat=2)) for panel in panels.values()),
        "Incomplete receipt quartet",
    )
    correct = [all(panel.values()) for panel in panels.values()]
    result.update(
        quartets=len(panels),
        complete_quartets=int(sum(correct)),
        quartet_accuracy=float(np.mean(correct)) if correct else None,
        quartet_correct_cells_histogram={
            str(k): sum(sum(p.values()) == k for p in panels.values()) for k in range(5)
        },
    )
    result["nuisance_pairs"] = {}
    for name, pairs in nuisances.items():
        require(
            all(set(pair) == {0, 1} and pair[0][1] == pair[1][1] for pair in pairs.values()),
            "Incomplete or target-changing nuisance pair",
        )
        result["nuisance_pairs"][name] = {
            "pairs": len(pairs),
            "verdict_flips": sum(p[0][0] != p[1][0] for p in pairs.values()),
            "both_correct": sum(p[0][0] == p[0][1] and p[1][0] == p[1][1] for p in pairs.values()),
        }
    if margins is not None:
        values = np.asarray(margins, dtype=np.float64)
        require(
            values.shape == labels.shape and np.isfinite(values).all(), "Invalid output margins"
        )
        result["conditional_yes_no_score"] = {
            "raw_auroc": float(roc_auc_score(labels, values)),
            "mean_log_loss": float(np.mean(np.logaddexp(0, values) - labels * values)),
            "brier": float(np.mean((expit(values) - labels) ** 2)),
            "scope": "Final-prefix YES-minus-NO logits; sigmoid conditional on these two tokens",
        }
    return result


def factor_metrics(rows, margins):
    require(
        all(r["condition"].startswith("correction_") for r in rows),
        "Factor truth is defined here only on correction states",
    )
    truth = np.array([[r["state"][f] for f in FACTOR_FIELDS] for r in rows])
    values = np.asarray(margins)
    require(
        values.shape == truth.shape and np.isfinite(values).all(), "Invalid factor score inventory"
    )
    bits = (values > 0).astype(int)
    result = {
        "rows": len(rows),
        "all_four_correct": float(np.mean(np.all(truth == bits, axis=1))),
        "factor_order": list(FACTORS),
        "factors": {},
    }
    for j, factor in enumerate(FACTORS):
        entry = class_summary(truth[:, j], bits[:, j])
        entry["raw_auroc"] = float(roc_auc_score(truth[:, j], values[:, j]))
        entry["class_margin_quantiles_10_50_90"] = {
            str(k): np.quantile(values[truth[:, j] == k, j], [0.1, 0.5, 0.9]).tolist()
            for k in (0, 1)
        }
        entry["positive_minus_negative_mean_margin"] = float(
            values[truth[:, j] == 1, j].mean() - values[truth[:, j] == 0, j].mean()
        )
        result["factors"][factor] = entry
    return result


def paired_difference(group_values, comparison, *, seed=BOOTSTRAP_SEED, samples=BOOTSTRAP_SAMPLES):
    require(
        comparison in ("A512", "Offset")
        and len(group_values) == 24
        and seed == BOOTSTRAP_SEED
        and samples == BOOTSTRAP_SAMPLES,
        "Need fixed 24-context contrast/bootstrap policy",
    )
    groups = sorted(group_values)
    mixed = np.array([group_values[g]["Mixed512"] for g in groups], dtype=np.float64)
    reference = np.array([group_values[g][comparison] for g in groups], dtype=np.float64)
    require(
        np.isfinite(mixed).all()
        and np.isfinite(reference).all()
        and np.isin(mixed * 8, np.arange(9)).all()
        and np.isin(reference * 8, np.arange(9)).all(),
        "Need eight complete hard quartets per context",
    )
    delta = mixed - reference
    draws = np.random.default_rng(seed).integers(0, 24, size=(samples, 24))
    interval = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975])
    return {
        "contrast": "Mixed512_minus_" + comparison,
        "renderer": "B",
        "condition": "correction_hard",
        "endpoint": "complete_four_case_quartet_accuracy",
        "groups": 24,
        "quartets": 192,
        "Mixed512": float(mixed.mean()),
        comparison: float(reference.mean()),
        "difference": float(delta.mean()),
        "conditional_percentile_interval_95": interval.tolist(),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "uncertainty_scope": "Paired whole contexts conditional on frozen authored bank, old feature map, site and fitted heads; excludes coefficient/checkpoint uncertainty",
        "group_values": [
            {"group_id": g, "Mixed512": float(a), comparison: float(b), "difference": float(a - b)}
            for g, a, b in zip(groups, mixed, reference, strict=True)
        ],
    }


def score_arrays(rows, arm_scores, output_margins, strict_correct, *, model_key):
    validate_test_rows(rows)
    require(set(arm_scores) == set(ARMS), "All four fixed arms are required")
    require(
        np.asarray(output_margins).shape == np.asarray(strict_correct).shape == (4992,)
        and np.isfinite(output_margins).all()
        and np.asarray(strict_correct).dtype == bool,
        "Invalid output comparator inventory",
    )
    result = {"arms": {}, "output_baseline": {}, "factor_diagnostics": {}, "paired_format": {}}
    group_values = defaultdict(dict)
    for arm in ARMS:
        score = arm_scores[arm]
        require(
            score["verdicts"].shape == (4992,)
            and score["factor_margins"].shape == score["factor_predictions"].shape == (4992, 4),
            "Invalid arm prediction inventory",
        )
        require(
            np.isfinite(score["factor_margins"]).all()
            and np.array_equal(
                score["factor_predictions"], (score["factor_margins"] > 0).astype(int)
            )
            and np.array_equal(score["verdicts"], fold_factors(score["factor_predictions"])),
            "Factor decisions/external fold differ from saved margins",
        )
        result["arms"][arm] = {}
        result["factor_diagnostics"][arm] = {}
        for renderer in ("A", "B"):
            for condition in CONDITIONS:
                ix = [
                    i
                    for i, r in enumerate(rows)
                    if r["renderer"] == renderer and r["condition"] == condition
                ]
                subset = [rows[i] for i in ix]
                aggregate = verdict_metrics(subset, score["verdicts"][ix])
                groups = []
                for group in sorted({r["group_id"] for r in subset}):
                    within = [i for i in ix if rows[i]["group_id"] == group]
                    entry = verdict_metrics([rows[i] for i in within], score["verdicts"][within])
                    groups.append({"group_id": group, **entry})
                    if renderer == "B" and condition == "correction_hard":
                        require(entry["quartets"] == 8, "Need eight hard quartets/context")
                        group_values[group][arm] = entry["quartet_accuracy"]
                result["arms"][arm][f"{renderer}/{condition}"] = {
                    "aggregate": aggregate,
                    "groups": groups,
                    "scope": "correction-trained four-factor external fold"
                    if condition.startswith("correction_")
                    else "out_of_training_construct_control; no factor-truth claim",
                }
            for stratum in ("correction_all", "correction_easy", "correction_hard"):
                ix = [
                    i
                    for i, r in enumerate(rows)
                    if r["renderer"] == renderer
                    and (
                        r["condition"].startswith("correction_")
                        if stratum == "correction_all"
                        else r["condition"] == stratum
                    )
                ]
                result["factor_diagnostics"][arm][f"{renderer}/{stratum}"] = {
                    "aggregate": factor_metrics([rows[i] for i in ix], score["factor_margins"][ix]),
                    "groups": [
                        {
                            "group_id": g,
                            **factor_metrics(
                                [rows[i] for i in ix if rows[i]["group_id"] == g],
                                score["factor_margins"][
                                    [i for i in ix if rows[i]["group_id"] == g]
                                ],
                            ),
                        }
                        for g in sorted(group_values) or sorted({r["group_id"] for r in rows})
                    ],
                }
        # Pair by complete state rather than array adjacency/order.
        pairs = validate_test_rows(rows)
        correction = [p for key, p in sorted(pairs.items()) if key[1].startswith("correction_")]
        ai, bi = ([p[renderer] for p in correction] for renderer in ("A", "B"))
        a, b = score["factor_margins"][ai], score["factor_margins"][bi]
        result["paired_format"][arm] = {
            "correction_pairs": len(ai),
            "factor_decision_flips": np.sum((a > 0) != (b > 0), axis=0).tolist(),
            "factor_margin_delta_B_minus_A_quantiles_10_50_90": np.quantile(
                b - a, [0.1, 0.5, 0.9], axis=0
            ).T.tolist(),
        }
    for renderer, condition in product(("A", "B"), CONDITIONS):
        ix = [
            i
            for i, r in enumerate(rows)
            if r["renderer"] == renderer and r["condition"] == condition
        ]
        subset = [rows[i] for i in ix]
        result["output_baseline"][f"{renderer}/{condition}"] = {
            "conditional": verdict_metrics(
                subset,
                (np.asarray(output_margins)[ix] > 0).astype(int),
                np.asarray(output_margins)[ix],
            ),
            "strict_first_token": {
                "rows": len(ix),
                "correct_full_denominator": int(np.sum(strict_correct[ix])),
                "accuracy_full_denominator": float(np.mean(strict_correct[ix])),
                "scope": "Full-vocabulary first token; outside YES/NO counted incorrect",
            },
        }
    result["primary"] = paired_difference(group_values, "A512")
    result["primary"]["role"] = "primary" if model_key == "llama" else "named_secondary_replication"
    result["named_diagnostic"] = paired_difference(group_values, "Offset")
    result["named_diagnostic"]["interpretation"] = (
        "Difference between fixed fitting procedures at equal observed A/B budget; not a unique geometric or mechanistic attribution"
    )
    result["factor_truth_scope"] = (
        "Four correction factors only; no invented one-step or asserted factor truth"
    )
    return result


def validate_seal(seal_path, fit_dir, model_key):
    seal = read_json(seal_path)
    require(
        seal.get("schema") == "format-adaptation-test-seal-v1"
        and set(seal.get("models", {})) == {"llama", "qwen"},
        "Both-model fit seal is required",
    )
    entry = seal["models"][model_key]
    for filename, field in (
        ("coefficients.npz", "coefficients_sha256"),
        ("settings.json", "settings_sha256"),
        ("fit_manifest.json", "fit_manifest_sha256"),
    ):
        require(
            file_sha256(Path(fit_dir) / filename) == entry.get(field),
            "Analysis fit differs from both-model seal",
        )
    check_bindings(seal.get("source_sha256"))
    for relative in POLICY_FILES:
        path = str((ROOT / relative).resolve())
        require(
            seal["source_sha256"].get(path) == file_sha256(path),
            "Analysis implementation must be sealed before test capture",
        )
    return seal


def verify_test_plan(data, capture, seal):
    """Verify original prepared test inputs and the aggregate's row-journal origin."""
    manifest, payload = data["manifest"], data["payload"]
    digest = manifest.get("protocol_sha256")
    protocol_paths = [
        Path(path)
        for path, value in manifest.get("input_artifact_sha256", {}).items()
        if Path(path).name == "protocol.json" and value == digest
    ]
    require(
        len(protocol_paths) == 1 and isinstance(digest, str),
        "Test capture must bind exactly one original protocol",
    )
    protocol = read_json(protocol_paths[0])
    require(
        protocol.get("schema") == "format-adaptation-capture-protocol-v1"
        and protocol.get("model_key") == payload["model_key"]
        and protocol.get("site") == payload["site"]
        and protocol.get("bank_sha256") == seal["bank_sha256"]
        and protocol.get("source_sha256") == seal["source_sha256"],
        "Test protocol model/site/bank/source differs from original seal",
    )
    planned = [r for r in protocol["planned_rows"] if r["partition"] == "test"]
    require(
        len(planned) == 4992
        and [r["row_id"] for r in planned] == [r["row_id"] for r in data["rows"]],
        "Missing/reordered prepared test rows",
    )
    for captured, original in zip(data["rows"], planned, strict=True):
        require(
            all(captured.get(name) == value for name, value in original.items()),
            "Test input differs from pre-inference prepared metadata",
        )
    tokens = protocol["tokenizer"]["answer_token_ids"]
    require(payload.get("answer_token_ids") == tokens, "Test candidate token identities differ")
    data["input_artifact_sha256"].update(protocol["source_sha256"])
    journal = Path(capture) / "journal"
    if payload["test_fixture"] and not journal.exists():
        return {"verified_rows": 0, "scope": "explicit_synthetic_fixture_without_model_journals"}
    require(
        journal.is_dir()
        and {p.name for p in journal.iterdir()} == {r["row_id"] for r in data["rows"]},
        "Missing/unexpected test journals",
    )
    index = protocol["sites"].index(payload["site"])
    sealed_at = datetime.fromisoformat(seal["created_at"])
    for i, row in enumerate(data["rows"]):
        directory = journal / row["row_id"]
        started, complete = (
            read_json(directory / name) for name in ("started.json", "complete.json")
        )
        require(
            complete == row
            and complete.get("status") == "complete"
            and complete.get("protocol_sha256") == digest
            and started.get("protocol_sha256") == digest
            and started.get("row_id") == row["row_id"]
            and not (directory / "failure.json").exists(),
            "Test aggregate differs from its journal",
        )
        require(
            datetime.fromisoformat(started["started_at"]) > sealed_at,
            "Test forward preceded both-model fit seal",
        )
        for name in ("arrays", "raw_forward"):
            require(
                file_sha256(directory / f"{name}.npz") == complete[f"{name}_sha256"],
                "Test journal measurement changed",
            )
        with np.load(directory / "arrays.npz", allow_pickle=False) as saved:
            require(
                np.array_equal(saved["features"][index], data["features"][i]),
                "Aggregate feature differs from selected journal site",
            )
            require(
                saved["output_summary_sha256"].item() == canonical_hash(row["output_logits"]),
                "Journal output summary differs",
            )
        summary = row["output_logits"]
        require(
            summary["yes_minus_no"] == data["output_margins"][i]
            and bool(data["strict_first_token_correct"][i])
            == (summary["argmax_token_id"] == tokens["YES" if row["gold"] else "NO"]),
            "Aggregate output comparator differs from row journal",
        )
    return {
        "verified_rows": len(data["rows"]),
        "scope": "Selected S1 features and output comparator, original row metadata, raw-array hashes and pre-forward seal timing",
    }


def analyse(capture, fit, seal, output, *, allow_test_fixture=False):
    settings, arrays = validate_bundle(fit, allow_test_fixture=allow_test_fixture)
    model = settings["model_key"]
    seal_record = validate_seal(seal, fit, model)
    data = load_capture(capture, "test", model, allow_test_fixture=allow_test_fixture)
    require(
        data["payload"]["test_fixture"] is settings["test_fixture"], "Capture/fit fixture differs"
    )
    validate_test_rows(data["rows"], settings["training_group_ids"])
    # The runner enforces timing per forward; analysis also binds the captured seal.
    manifest = data["manifest"]
    captured_seal = manifest.get("seal_sha256", manifest.get("test_seal_sha256"))
    require(
        captured_seal == file_sha256(seal), "Test capture does not bind the both-model fit seal"
    )
    require(
        manifest.get("bank_sha256") == seal_record.get("bank_sha256")
        and bool(manifest.get("bank_sha256")),
        "Test capture bank differs from fit seal",
    )
    journal_check = verify_test_plan(data, capture, seal_record)
    target = Path(output).resolve()
    require(not target.exists(), "Analysis output exists; preserve previous attempt")
    target.mkdir(parents=True)
    bindings = {
        **data["input_artifact_sha256"],
        str(Path(seal).resolve()): file_sha256(seal),
        **{
            str((Path(fit) / name).resolve()): file_sha256(Path(fit) / name)
            for name in ("coefficients.npz", "settings.json", "fit_manifest.json")
        },
        **{str((ROOT / name).resolve()): file_sha256(ROOT / name) for name in POLICY_FILES},
    }
    attempt = {
        "schema": "format-adaptation-analysis-manifest-v1",
        "status": "analysing",
        "created_at": utc_now(),
        "input_artifact_sha256": bindings,
    }
    atomic_json(target / "manifest.json", attempt)
    try:
        with threadpool_limits(limits=4):
            scored = score_bundle(
                settings, arrays, data["features"], [r["renderer"] for r in data["rows"]]
            )
            result = score_arrays(
                data["rows"],
                scored,
                data["output_margins"],
                data["strict_first_token_correct"],
                model_key=model,
            )
        result.update(
            schema="format-adaptation-analysis-v1",
            model_key=model,
            created_at=utc_now(),
            research_status="agent_audited_exploratory",
            confirmatory=False,
            test_fixture=settings["test_fixture"],
            site=settings["site"],
            primary_model="llama",
            named_secondary_model="qwen",
            source_artifact_sha256=bindings,
            journal_verification=journal_check,
            interpretation="Supervised accessibility under a fixed external fold; no model-internal computation or invariant representation claim",
        )
        if all("output_logits" in r for r in data["rows"]):
            tokens = data["payload"].get("answer_token_ids")
            if tokens:
                result["output_first_token_detail"] = output_token_detail(data["rows"], tokens)
        check_bindings(bindings)
        atomic_npz(
            target / "predictions.npz",
            row_ids=np.array([r["row_id"] for r in data["rows"]]),
            arms=np.array(ARMS),
            factor_order=np.array(FACTORS),
            verdicts=np.column_stack([scored[a]["verdicts"] for a in ARMS]),
            factor_margins=np.stack([scored[a]["factor_margins"] for a in ARMS], axis=1),
            factor_predictions=np.stack([scored[a]["factor_predictions"] for a in ARMS], axis=1),
            output_margins=data["output_margins"],
            strict_first_token_correct=data["strict_first_token_correct"],
        )
        atomic_json(target / "results.json", result)
        attempt.update(
            status="complete",
            output_sha256={
                name: file_sha256(target / name) for name in ("results.json", "predictions.npz")
            },
        )
        atomic_json(target / "manifest.json", attempt)
        return result
    except BaseException as error:
        attempt.update(status="failed_preserved", error_type=type(error).__name__, error=str(error))
        atomic_json(target / "manifest.json", attempt)
        raise


def output_token_detail(rows, tokens):
    result = {}
    for renderer, condition in product(("A", "B"), CONDITIONS):
        subset = [r for r in rows if r["renderer"] == renderer and r["condition"] == condition]
        summaries = [r["output_logits"] for r in subset]
        valid = [s["argmax_token_id"] in (tokens["YES"], tokens["NO"]) for s in summaries]
        mass = [float(np.exp(s["yes_logprob"]) + np.exp(s["no_logprob"])) for s in summaries]
        require(
            np.isfinite(mass).all()
            and np.all(np.array(mass) >= 0)
            and np.all(np.array(mass) <= 1 + 1e-12),
            "Invalid candidate token mass",
        )
        result[f"{renderer}/{condition}"] = {
            "candidate_first_token_rate": float(np.mean(valid)),
            "outside_candidate_first_tokens": len(subset) - sum(valid),
            "full_vocab_argmax_ties": sum(s["argmax_tie_count"] > 1 for s in summaries),
            "mean_candidate_token_mass": float(np.mean(mass)),
        }
    return result
