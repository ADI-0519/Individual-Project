"""Frozen fresh-test scoring and conditional whole-context uncertainty.

No fitting, site selection, calibration or model inference occurs here. The
only primary is S1 minus S0 on renderer-A hard correction complete quartets.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from .fresh_readout_bank import canonical, iter_rows, load_bank
from .fresh_readout_fit import score_bundle, validate_fit_bundle
from .storage import atomic_json, atomic_npz, file_sha256, read_json, utc_now

CONDITIONS = ("one_step", "correction_easy", "correction_hard", "asserted")
ROOT = Path(__file__).resolve().parents[2]
POLICY_FILES = ("src/latent_probes/fresh_readout_analysis.py", "scripts/analyze_fresh_readout.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_aggregate_measurement(feature, journal_features, row, complete):
    require(
        np.array_equal(feature, journal_features), "Aggregate feature differs from its row journal"
    )
    require(
        row["output_yes_no_logitdiff"] == complete["output_logits"]["yes_minus_no"],
        "Aggregate output margin differs from its row journal",
    )


def strict_first_token(rows, tokens):
    """Recorded full-vocabulary argmax, separate from conditional YES/NO scores."""
    valid, correct, tied, mass = 0, 0, 0, []
    for row in rows:
        summary = row["output_logits"]
        token = summary["argmax_token_id"]
        accepted = token in (tokens["YES"], tokens["NO"])
        valid += int(accepted)
        correct += int(accepted and int(token == tokens["YES"]) == row["gold"])
        tied += int(summary["argmax_tie_count"] > 1)
        mass.append(float(np.exp(summary["yes_logprob"]) + np.exp(summary["no_logprob"])))
    return {
        "rows": len(rows),
        "candidate_first_tokens": valid,
        "outside_candidate_first_tokens": len(rows) - valid,
        "correct_full_denominator": correct,
        "accuracy_full_denominator": correct / len(rows),
        "candidate_first_token_rate": valid / len(rows),
        "full_vocab_argmax_ties": tied,
        "mean_candidate_token_mass": float(np.mean(mass)),
        "scope": "Recorded lowest-index full-vocabulary argmax, outside YES/NO counted incorrect; no complete response generation or validity claim",
    }


def metrics(rows, predictions, margins=None):
    """Complete linked panels, row performance and target-preserving controls."""
    predicted = np.asarray(predictions)
    labels = np.asarray([r["gold"] for r in rows])
    require(
        len(rows) and predicted.shape == labels.shape and np.isin(predicted, (0, 1)).all(),
        "Invalid score inventory",
    )
    result = {
        "rows": len(rows),
        "correct": int(np.sum(predicted == labels)),
        "row_accuracy": float(np.mean(predicted == labels)),
        "class_recalls": {
            str(k): float(np.mean(predicted[labels == k] == k)) if np.any(labels == k) else None
            for k in (0, 1)
        },
    }
    panels, nuisance, receipts = (
        defaultdict(list),
        {k: defaultdict(dict) for k in ("world", "name_map")},
        {k: defaultdict(dict) for k in ("first_receipt", "receipt")},
    )
    for row, pred in zip(rows, predicted, strict=True):
        state = row["state"]
        group = (row["group_id"], row["condition"], row["renderer"])
        if row["condition"] != "asserted":
            panel = (*group, state["update"], state["world"], state["name_map"])
            panels[panel].append(bool(pred == row["gold"]))
        for name in (*nuisance, *receipts):
            if name not in state or state[name] is None:
                continue
            signature = (
                *group,
                tuple(
                    sorted(
                        (k, v) for k, v in state.items() if k not in (name, "group_id", "mechanism")
                    )
                ),
            )
            target = nuisance if name in nuisance else receipts
            require(state[name] not in target[name][signature], "Duplicate counterpart state")
            target[name][signature][state[name]] = (int(pred), row["gold"])
    require(all(len(v) == 4 for v in panels.values()), "Incomplete four-case panel")
    result.update(
        quartets=len(panels),
        complete_quartets=sum(all(v) for v in panels.values()),
        quartet_accuracy=float(np.mean([all(v) for v in panels.values()])) if panels else None,
        quartet_correct_cells_histogram={
            str(k): sum(sum(v) == k for v in panels.values()) for k in range(5)
        },
    )
    result["nuisance_pairs"] = {}
    for name, pairs in nuisance.items():
        require(
            all(set(pair) == {0, 1} and pair[0][1] == pair[1][1] for pair in pairs.values()),
            "Incomplete or target-changing nuisance pair",
        )
        result["nuisance_pairs"][name] = {
            "pairs": len(pairs),
            "verdict_flips": sum(p[0][0] != p[1][0] for p in pairs.values()),
            "both_correct": sum(p[0][0] == p[0][1] and p[1][0] == p[1][1] for p in pairs.values()),
        }
    result["receipt_pairs"] = {}
    for name, pairs in receipts.items():
        require(all(set(pair) == {0, 1} for pair in pairs.values()), "Incomplete receipt pair")
        result["receipt_pairs"][name] = {}
        for changes in (False, True):
            subset = [p for p in pairs.values() if (p[0][1] != p[1][1]) == changes]
            result["receipt_pairs"][name]["changing" if changes else "preserving"] = {
                "pairs": len(subset),
                "verdict_flips": sum(p[0][0] != p[1][0] for p in subset),
                "both_correct": sum(p[0][0] == p[0][1] and p[1][0] == p[1][1] for p in subset),
            }
    if margins is not None:
        margins = np.asarray(margins, dtype=float)
        require(
            margins.shape == labels.shape and np.isfinite(margins).all(),
            "Invalid continuous margins",
        )
        p = expit(margins)
        bins = np.minimum((p * 10).astype(int), 9)
        ece = sum(
            np.mean(bins == b)
            * abs(float(np.mean(p[bins == b])) - float(np.mean(labels[bins == b])))
            for b in range(10)
            if np.any(bins == b)
        )
        result["unrecalibrated_score_diagnostics"] = {
            "auroc": float(roc_auc_score(labels, margins)) if len(set(labels)) == 2 else None,
            "brier": float(np.mean((p - labels) ** 2)),
            "log_loss": float(np.mean(np.logaddexp(0, margins) - labels * margins)),
            "equal_width_ece10": float(ece),
            "bins": 10,
            "interpretation": "Raw sigmoid margin; output baseline is conditional on YES/NO. No calibration claim.",
        }
    return result


def primary_difference(group_values, *, seed=20261122, samples=10000):
    groups = sorted(group_values)
    require(
        len(groups) == 24 and samples == 10000 and seed == 20261122,
        "Need fixed 24-context primary/bootstrap policy",
    )
    s0, s1 = (np.array([group_values[g][m] for g in groups], dtype=float) for m in ("S0", "S1"))
    require(
        np.isfinite(s0).all()
        and np.isfinite(s1).all()
        and ((s0 >= 0) & (s0 <= 1) & (s1 >= 0) & (s1 <= 1)).all(),
        "Invalid primary group scores",
    )
    delta = s1 - s0
    indices = np.random.default_rng(seed).integers(0, len(groups), size=(samples, len(groups)))
    draws = delta[indices].mean(axis=1)
    return {
        "contrast": "S1_minus_S0",
        "renderer": "A",
        "condition": "correction_hard",
        "endpoint": "complete_four_case_quartet_accuracy",
        "groups": 24,
        "quartets": 192,
        "S0": float(s0.mean()),
        "S1": float(s1.mean()),
        "difference": float(delta.mean()),
        "conditional_percentile_interval_95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "uncertainty_scope": "Whole paired contexts, conditional on frozen authoring/renderers, selected sites and fitted heads; no independent-model pooling",
        "group_values": [
            {"group_id": g, "S0": float(a), "S1": float(b), "difference": float(b - a)}
            for g, a, b in zip(groups, s0, s1, strict=True)
        ],
    }


def score_arrays(rows, scores):
    require(
        len(rows) == 4992 and len({r["group_id"] for r in rows}) == 24,
        "Need all fresh paired test cases",
    )
    methods = {name: np.asarray(scores[name]) for name in ("S0", "S1", "C1", "C2", "output")}
    margins = {name: np.asarray(scores[name + "_margin"]) for name in ("S0", "S1", "C1", "output")}
    for index, site in enumerate(scores["source_sites"]):
        name = "source_" + site
        methods[name] = scores["source_all12"][:, index]
        margins[name] = scores["source_all12_margins"][:, index]
    result = {"methods": {}, "primary": None, "factor_diagnostics": {}}
    selected = {}
    for name, predicted in methods.items():
        strata = {}
        for renderer in ("A", "B"):
            for condition in CONDITIONS:
                indices = [
                    i
                    for i, r in enumerate(rows)
                    if r["renderer"] == renderer and r["condition"] == condition
                ]
                subset = [rows[i] for i in indices]
                margin = margins[name][indices] if name in margins else None
                aggregate = metrics(subset, predicted[indices], margin)
                groups = []
                for group in sorted({r["group_id"] for r in subset}):
                    within = [i for i in indices if rows[i]["group_id"] == group]
                    values = metrics(
                        [rows[i] for i in within],
                        predicted[within],
                        margins[name][within] if name in margins else None,
                    )
                    groups.append({"group_id": group, **values})
                    if renderer == "A" and condition == "correction_hard" and name in ("S0", "S1"):
                        selected.setdefault(group, {})[name] = values["quartet_accuracy"]
                strata[f"{renderer}/{condition}"] = {"aggregate": aggregate, "groups": groups}
        result["methods"][name] = {
            "margin_kind": "synthetic_hard_fold"
            if name == "C2"
            else "conditional_YES_NO_logitdiff"
            if name == "output"
            else "logistic_decision",
            "strata": strata,
        }
    result["primary"] = primary_difference(selected)
    bits = np.asarray(scores["factor_predictions"])
    require(
        bits.shape == (4992, 4) and np.isin(bits, (0, 1)).all(),
        "Invalid predicted factor inventory",
    )
    for renderer in ("A", "B"):
        for condition in ("correction_easy", "correction_hard"):
            indices = [
                i
                for i, r in enumerate(rows)
                if r["renderer"] == renderer and r["condition"] == condition
            ]
            truth = np.array(
                [
                    [rows[i]["state"][k] for k in ("initial", "update", "first_receipt", "receipt")]
                    for i in indices
                ]
            )
            correct = bits[indices] == truth
            result["factor_diagnostics"][f"{renderer}/{condition}"] = {
                "rows": len(indices),
                "factor_order": ["I", "M1", "B1", "B2"],
                "accuracy": correct.mean(axis=0).tolist(),
                "all_four_correct": float(correct.all(axis=1).mean()),
            }
    result["factor_diagnostics"]["scope"] = (
        "Four-factor truth only on two-message conditions; no invented factors for asserted states or one-step B2. C2 always uses its own predicted B2."
    )
    return result, methods, margins


def analyze(
    test_capture, fit_dir, protocol_path, bank_path, seal_path, output, *, allow_test_fixture=False
):
    from .fresh_readout_capture import _completed, verify_protocol

    target, directory = Path(output).resolve(), Path(test_capture).resolve()
    require(not target.exists(), "Use a new analysis directory")
    protocol, seal, bank = read_json(protocol_path), read_json(seal_path), load_bank(bank_path)
    require(
        not target.is_relative_to(Path(__file__).resolve().parents[2])
        and (
            protocol["test_fixture"]
            or ("onedrive" not in str(target).lower() and not str(target).startswith("/mnt/"))
        ),
        "Use a fresh native analysis directory outside the workspace",
    )
    verify_protocol(protocol)
    require(
        all(
            protocol["source_sha256"].get(str((ROOT / p).resolve())) == file_sha256(ROOT / p)
            for p in POLICY_FILES
        ),
        "Analysis implementation and CLI must be bound before test capture",
    )
    require(
        not protocol["test_fixture"] or allow_test_fixture,
        "Native analysis rejects fixture captures",
    )
    settings, arrays = validate_fit_bundle(fit_dir, allow_test_fixture=allow_test_fixture)
    model = protocol["model_key"]
    require(
        seal["schema"] == "fresh-readout-fit-seal-v2"
        and seal["status"] == "frozen"
        and seal["model_protocol_sha256"][model] == file_sha256(protocol_path),
        "Wrong both-model fit seal",
    )
    for path, digest in seal["fitted_artifact_sha256"].items():
        require(file_sha256(path) == digest, "Frozen fitted artifact changed")
    for name in ("settings.json", "heads.npz", "manifest.json", "candidates.json"):
        path = str((Path(fit_dir) / name).resolve())
        require(
            seal["fitted_artifact_sha256"].get(path) == file_sha256(path),
            "Analysis fit omitted from both-model freeze",
        )
    expected = [r for r in iter_rows(bank, "test")]
    payload, manifest = read_json(directory / "rows.json"), read_json(directory / "manifest.json")
    rows = payload["rows"]
    require(
        manifest["status"] == "complete"
        and manifest["partition"] == "test"
        and manifest["completed_rows"] == len(rows) == 4992,
        "Incomplete held-out capture",
    )
    require(
        payload["provenance"]["protocol_sha256"]
        == file_sha256(protocol_path)
        == settings["capture_provenance"]["protocol_sha256"]
        and payload["provenance"]["bank_file_sha256"] == file_sha256(bank_path),
        "Test bank/protocol mismatch",
    )
    require(
        [r["record_id"] for r in rows] == [r["record_id"] for r in expected],
        "Missing/reordered held-out cases",
    )
    for row, original in zip(rows, expected, strict=True):
        require(all(row.get(k) == v for k, v in original.items()), "Held-out prose/state changed")
    for name, field in (("rows.json", "rows_sha256"), ("features.npz", "features_sha256")):
        require(
            file_sha256(directory / name) == manifest[field], "Held-out aggregate bytes changed"
        )
    for path, digest in manifest["authorization"]["sha256"].items():
        require(file_sha256(path) == digest, "Capture authorization source changed")
    require(
        all(
            manifest["authorization"]["sha256"].get(p) == d
            for p, d in seal["fitted_artifact_sha256"].items()
        ),
        "Test capture omitted frozen fits",
    )
    planned = {r["record_id"]: r for r in protocol["planned_rows"]}
    for row in rows:
        journal = directory / "row_journal" / row["row_id"]
        completed = _completed(
            journal, planned[row["record_id"]], protocol, file_sha256(protocol_path)
        )
        require(
            completed is not None and all(row.get(k) == v for k, v in completed.items()),
            "Held-out journal differs",
        )
        require(
            datetime.fromisoformat(seal["created_at"])
            < datetime.fromisoformat(read_json(journal / "started.json")["started_at"]),
            "Test forward preceded both-model fit freeze",
        )
    with np.load(directory / "features.npz", allow_pickle=False) as saved:
        require(
            set(saved.files) == {"features", "row_ids", "sites"}
            and saved["row_ids"].tolist() == [r["row_id"] for r in rows]
            and saved["sites"].tolist() == protocol["sites"],
            "Test feature identity differs",
        )
        features = saved["features"].copy()
    require(
        features.dtype == np.float32
        and features.shape == (4992, 12, settings["hidden_size"])
        and np.isfinite(features).all(),
        "Invalid held-out aggregate feature tensor",
    )
    for index, row in enumerate(rows):
        journal = directory / "row_journal" / row["row_id"]
        complete = read_json(journal / "complete.json")
        with np.load(journal / "arrays.npz", allow_pickle=False) as saved:
            verify_aggregate_measurement(features[index], saved["features"], row, complete)
    with threadpool_limits(limits=4):
        scored = score_bundle(settings, arrays, features)
        require(
            [p["site"] for p in settings["source_panel"]] == protocol["sites"],
            "Source panel score order differs",
        )
        scored["source_sites"] = protocol["sites"]
        scored["output_margin"] = np.asarray([r["output_yes_no_logitdiff"] for r in rows])
        scored["output"] = (scored["output_margin"] > 0).astype(int)
        result, methods, margins = score_arrays(rows, scored)
    result["strict_first_token_diagnostic"] = {
        f"{renderer}/{condition}": strict_first_token(
            [r for r in rows if r["renderer"] == renderer and r["condition"] == condition],
            protocol["tokenizer"]["answer_token_ids"],
        )
        for renderer in ("A", "B")
        for condition in CONDITIONS
    }
    result.update(
        schema="fresh-readout-analysis-v2",
        model_key=model,
        created_at=utc_now(),
        test_fixture=protocol["test_fixture"],
        research_status="agent_audited_exploratory",
        human_review_complete=False,
        confirmatory_result=False,
        fresh_test=True,
        construct="Fictional sender-attributed reader belief after an explicit question; no spontaneous anticipation or deceptive-intent claim",
        primary_role="primary" if model == "llama" else "predeclared_replication",
        selected_source_site=settings["heads"][settings["pipelines"]["S1"]["head_key"]]["site"],
        original_source_site=settings["heads"][settings["pipelines"]["S0"]["head_key"]]["site"],
        output_baseline_scope="Final complete-prefix YES/NO logits have later context than account/query-position heads; probability is conditional on these two tokens",
        bank_sha256=canonical(bank),
        source_sha256={
            str(Path(p).resolve()): file_sha256(p)
            for p in (
                protocol_path,
                bank_path,
                seal_path,
                Path(fit_dir) / "settings.json",
                Path(fit_dir) / "heads.npz",
                directory / "rows.json",
                directory / "features.npz",
                Path(__file__),
            )
        },
    )
    target.mkdir(parents=True, exist_ok=False)
    method_names = list(methods)
    atomic_npz(
        target / "predictions.npz",
        row_ids=np.array([r["row_id"] for r in rows]),
        methods=np.array(method_names),
        verdicts=np.column_stack([methods[m] for m in method_names]),
        margins=np.column_stack(
            [margins[m] if m in margins else methods[m] * 2 - 1 for m in method_names]
        ),
        factor_predictions=scored["factor_predictions"],
        factor_margins=scored["factor_margins"],
    )
    atomic_json(target / "results.json", result)
    atomic_json(
        target / "manifest.json",
        {
            "schema": "fresh-readout-analysis-manifest-v2",
            "source_sha256": result["source_sha256"],
            "output_sha256": {
                name: file_sha256(target / name) for name in ("results.json", "predictions.npz")
            },
        },
    )
    return result
