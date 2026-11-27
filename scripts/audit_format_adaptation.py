"""Independent, no-fit reconstruction of a completed format-adaptation campaign.

Both reconstructions are written and sealed before any saved analysis artifact
is opened. This file deliberately imports no production experiment modules.
Ordinary full-vocabulary argmax/tie summaries are checked for consistency, not
recomputed from logits which the scientific rows do not retain in full.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

import numpy as np

ARMS = ("A512", "Mixed512", "Ahalf256", "Offset")
FACTORS = ("I", "M1", "B1", "B2")
FIELDS = ("initial", "update", "first_receipt", "receipt")
CONDITIONS = ("one_step", "correction_easy", "correction_hard", "asserted")
SITES = {"llama": "layer_16_query_location_last", "qwen": "layer_21_query_location_last"}
ROOT = Path(__file__).resolve().parents[1]
CODE = (Path(__file__).resolve(), ROOT / "tests/test_audit_format_adaptation.py")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def now():
    return datetime.now(UTC).isoformat()


def bound_files(bindings):
    require(isinstance(bindings, dict) and bindings, "Missing hash bindings")
    for filename, expected in bindings.items():
        path = Path(filename)
        require(path.is_absolute() and path.is_file(), "Missing bound file: " + filename)
        require(digest(path) == expected, "Changed bound file: " + filename)


def sealed_sources(seal):
    sources = seal.get("source_sha256")
    bound_files(sources)
    for path in CODE:
        require(sources.get(str(path)) == digest(path), "Auditor source absent or changed in seal")


def bit(value):
    require(type(value) is int and value in (0, 1), "Invalid formal bit")
    return value


def identity(row):
    return (row["group_id"], row["condition"], json.dumps(row["state"], sort_keys=True))


def gold(row):
    state = row["state"]
    require(state.get("group_id") == row["group_id"], "State/context mismatch")
    bit(state["world"])
    bit(state["name_map"])
    if row["condition"] == "asserted":
        require(state["mechanism"] == "asserted", "Wrong asserted mechanism")
        answer = bit(state["asserted"])
    else:
        i, u, receipt = (bit(state[f]) for f in ("initial", "update", "receipt"))
        if row["condition"] == "one_step":
            require(
                state["mechanism"] == "message" and state["first_receipt"] is None,
                "Wrong one-step factors",
            )
            answer = u if receipt == 1 else i
        else:
            require(
                row["condition"] in ("correction_easy", "correction_hard")
                and state["mechanism"] == "correction",
                "Wrong correction mechanism",
            )
            first = bit(state["first_receipt"])
            require(
                row["condition"] == ("correction_easy" if i == u else "correction_hard"),
                "Wrong correction stratum",
            )
            answer = (1 - u) if receipt == 1 else (u if first == 1 else i)
    require(type(row.get("gold")) is int and row["gold"] == answer, "Wrong formal gold")
    return answer


def test_roster(rows, fitting_groups):
    require(len(rows) == 4992, "Need all 4992 test rows")
    identifiers = [r["row_id"] for r in rows]
    require(len(set(identifiers)) == len(identifiers), "Duplicate test row ID")
    groups = {r["group_id"] for r in rows}
    require(len(groups) == 24 and groups.isdisjoint(fitting_groups), "Test context overlap/count")
    pairs, cells = defaultdict(dict), defaultdict(set)
    for index, row in enumerate(rows):
        require(row["partition"] == row.get("split", "test") == "test", "Non-test row")
        gold(row)
        fmt, state = row["renderer"], row["state"]
        require(fmt in ("A", "B"), "Unknown renderer")
        key = identity(row)
        require(fmt not in pairs[key], "Duplicate formal A/B row")
        pairs[key][fmt] = index
        family = row["condition"]
        if family.startswith("correction_"):
            family, formal = "correction", tuple(state[f] for f in (*FIELDS, "world", "name_map"))
        elif family == "one_step":
            formal = tuple(state[f] for f in ("initial", "update", "receipt", "world", "name_map"))
        else:
            formal = tuple(state[f] for f in ("asserted", "world", "name_map"))
        require(
            formal not in cells[(row["group_id"], fmt, family)], "Duplicate test factorial cell"
        )
        cells[(row["group_id"], fmt, family)].add(formal)
    require(
        len(pairs) == 2496 and all(set(v) == {"A", "B"} for v in pairs.values()),
        "Incomplete exact A/B pairs",
    )
    for group, fmt, family in product(
        sorted(groups), ("A", "B"), ("correction", "one_step", "asserted")
    ):
        dimensions = {"correction": 6, "one_step": 5, "asserted": 3}[family]
        require(
            cells[(group, fmt, family)] == set(product((0, 1), repeat=dimensions)),
            "Incomplete test factorial roster",
        )
    return pairs


def fit_membership(rows, settings):
    require(len(rows) == 768 and len({r["row_id"] for r in rows}) == 768, "Fit roster size/IDs")
    expected = {name: [] for name in (*ARMS[:3], "B_offset")}
    states, support = defaultdict(dict), Counter()
    groups = set()
    for row in rows:
        require(row["partition"] == row.get("split", "fit") == "fit", "Held-out row in fit roster")
        gold(row)
        require(row["condition"].startswith("correction_"), "Noncorrection fit row")
        state, fmt = row["state"], row["renderer"]
        index = int(row["group_id"].rsplit("_", 1)[1])
        require(0 <= index < 16 and fmt in ("A", "B"), "Wrong fitting context/format")
        require(
            (state["initial"] ^ state["update"] ^ state["world"] ^ state["name_map"]) == index % 2,
            "Wrong fixed fitting XOR roster",
        )
        assigned = "A" if state["world"] == (index // 2) % 2 else "B"
        require(row["mixed_format"] == assigned, "Wrong preassigned fitting format")
        groups.add(row["group_id"])
        key = identity(row)
        require(fmt not in states[key], "Duplicate fitting state/format")
        states[key][fmt] = row
        if fmt == "A":
            expected["A512"].append(row["row_id"])
        if fmt == assigned:
            expected["Mixed512"].append(row["row_id"])
            expected["Ahalf256" if fmt == "A" else "B_offset"].append(row["row_id"])
            support[(fmt, *(state[f] for f in (*FIELDS, "world", "name_map")))] += 1
    require(
        len(groups) == 16
        and len(states) == 512
        and len(support) == 128
        and set(support.values()) == {4},
        "Wrong fitting support",
    )
    require(groups == set(settings["training_group_ids"]), "Settings fitting contexts differ")
    for formats in states.values():
        require("A" in formats, "Missing A fitting counterpart")
        assigned = formats["A"]["mixed_format"]
        require(
            set(formats) == ({"A"} if assigned == "A" else {"A", "B"}),
            "Missing fitting replacement",
        )
    require(settings["training_ids"] == expected, "Saved fitting sample membership/order differs")
    for arm, n in (("A512", 512), ("Mixed512", 512), ("Ahalf256", 256)):
        require(
            len(expected[arm]) == settings["arms"][arm]["n"] == n
            and settings["arms"][arm]["C"] == settings["old_C2_C"] * 512 / n,
            "Wrong observation or regularization budget",
        )
    return groups


def output_scores(rows, tokens, protocol_tokens):
    require(
        tokens == protocol_tokens and set(tokens) == {"YES", "NO"}, "Candidate token IDs differ"
    )
    require(
        all(type(v) is int and v >= 0 for v in tokens.values()) and tokens["YES"] != tokens["NO"],
        "Invalid candidate token IDs",
    )
    margins, strict = [], []
    for row in rows:
        summary = row["output_logits"]
        require(row["output_candidate_order"] == ["YES", "NO"], "Candidate order differs")
        y, n = float(summary["yes_logit"]), float(summary["no_logit"])
        margin = y - n
        require(
            np.isfinite([y, n, summary["yes_logprob"], summary["no_logprob"]]).all()
            and margin == summary["yes_minus_no"],
            "Invalid candidate logit margin",
        )
        require(
            np.isclose(summary["yes_logprob"] - summary["no_logprob"], margin, atol=1e-10, rtol=0),
            "Candidate log probabilities differ from logits",
        )
        vocab, winner, ties = (
            summary["vocab_size"],
            summary["argmax_token_id"],
            summary["argmax_tie_count"],
        )
        require(
            type(vocab) is int
            and type(winner) is int
            and type(ties) is int
            and max(tokens.values()) < vocab
            and 0 <= winner < vocab
            and 1 <= ties <= vocab,
            "Invalid full-vocabulary argmax summary",
        )
        require(
            winner not in tokens.values() or (y >= n if winner == tokens["YES"] else n >= y),
            "Candidate argmax contradicts candidate logits",
        )
        if y == n and winner in tokens.values():
            require(ties >= 2, "Unrecorded candidate argmax tie")
            require(winner == min(tokens.values()), "Candidate argmax tie-break differs")
        mass = float(np.exp(summary["yes_logprob"]) + np.exp(summary["no_logprob"]))
        require(0 <= mass <= 1 + 1e-12, "Invalid candidate probability mass")
        margins.append(margin)
        strict.append(winner == tokens["YES" if row["gold"] else "NO"])
    return np.asarray(margins, dtype=np.float64), np.asarray(strict, dtype=bool)


def load_inputs(campaign, model, seal, seal_path, *, allow_fixture=False):
    fit, capture = campaign / (model + "_fit"), campaign / model / "test"
    settings, manifest = read(fit / "settings.json"), read(fit / "fit_manifest.json")
    require(
        settings.get("schema") == "format-adaptation-fit-v1"
        and manifest.get("schema") == "format-adaptation-fit-manifest-v1"
        and settings.get("status") == manifest.get("status") == "frozen",
        "Unfrozen fitting bundle",
    )
    require(
        settings["model_key"] == model and settings["site"] == SITES[model],
        "Wrong frozen model/site",
    )
    require(
        type(settings["test_fixture"]) is bool and (allow_fixture or not settings["test_fixture"]),
        "Native audit rejects fixtures",
    )
    for name, field in (
        ("settings.json", "settings_sha256"),
        ("coefficients.npz", "coefficients_sha256"),
        ("fit_manifest.json", "fit_manifest_sha256"),
    ):
        require(digest(fit / name) == seal["models"][model][field], "Fit differs from test seal")
    for name, expected in manifest["output_sha256"].items():
        require(digest(fit / name) == expected, "Fitting output hash differs")
    require(
        settings["coefficients_sha256"] == digest(fit / "coefficients.npz"),
        "Coefficient hash differs",
    )
    bound_files(settings["input_artifact_sha256"])
    require(
        manifest["input_artifact_sha256"] == settings["input_artifact_sha256"],
        "Fit input hash maps differ",
    )
    fit_rows_path = campaign / model / "fit/rows.json"
    require(
        settings["input_artifact_sha256"].get(str(fit_rows_path)) == digest(fit_rows_path),
        "Fitting sample roster is not hash bound",
    )
    groups = fit_membership(read(fit_rows_path)["rows"], settings)
    with np.load(fit / "coefficients.npz", allow_pickle=False) as saved:
        parameters = {name: saved[name].copy() for name in saved.files}
    expected_arrays = {"mean", "scale", "Offset__B_offset"} | {
        arm + "__" + name for arm in ARMS for name in ("coef", "intercept")
    }
    require(set(parameters) == expected_arrays, "Wrong coefficient array inventory")
    h = settings["hidden_size"]
    for name, values in parameters.items():
        shape = (h,) if name in ("mean", "scale") else (4, h) if name.endswith("__coef") else (4,)
        require(
            values.dtype == np.float64 and values.shape == shape and np.isfinite(values).all(),
            "Invalid coefficient/map array",
        )
    require(np.all(parameters["scale"] > 0), "Nonpositive frozen scale")
    require(
        all(
            np.array_equal(parameters["Offset__" + f], parameters["Ahalf256__" + f])
            for f in ("coef", "intercept")
        ),
        "Offset changed frozen Ahalf coefficients/intercept",
    )
    old_heads = [
        Path(p)
        for p in settings["input_artifact_sha256"]
        if Path(p).name == "heads.npz"
        and (Path(p).parent / "settings.json").is_file()
        and digest(Path(p).parent / "settings.json") == settings["old_settings_sha256"]
    ]
    require(len(old_heads) == 1, "Need one hash-bound old map archive matching old settings")
    old = read(old_heads[0].parent / "settings.json")
    require(old["pipelines"]["C2"]["shared_C"] == settings["old_C2_C"], "Old C changed")
    with np.load(old_heads[0], allow_pickle=False) as archive:
        for factor in FACTORS:
            head = old["heads"][old["pipelines"]["C2"]["factor_heads"][factor]]
            require(
                head["site"] == SITES[model] and head["C"] == settings["old_C2_C"],
                "Old site/C differs",
            )
            for name in ("mean", "scale"):
                require(
                    np.array_equal(parameters[name], archive[head["array_prefix"] + "__" + name]),
                    "Frozen map differs from old C2",
                )
    payload, captured = read(capture / "rows.json"), read(capture / "manifest.json")
    require(
        payload.get("schema") == "format-adaptation-capture-rows-v1"
        and payload["model_key"] == model
        and payload["site"] == SITES[model]
        and payload["test_fixture"] is settings["test_fixture"],
        "Wrong test capture schema/model",
    )
    require(
        captured.get("schema") == "format-adaptation-capture-manifest-v1"
        and captured["status"] == "complete"
        and captured["partition"] == "test"
        and captured["completed_rows"] == 4992,
        "Incomplete test capture",
    )
    require(
        captured.get("seal_sha256", captured.get("test_seal_sha256")) == digest(seal_path)
        and captured["bank_sha256"] == seal["bank_sha256"],
        "Capture seal/bank differs",
    )
    for name, field in (("rows.json", "rows_sha256"), ("features.npz", "features_sha256")):
        require(digest(capture / name) == captured[field], "Capture artifact hash differs")
    bound_files(captured["input_artifact_sha256"])
    protocol_path = campaign / model / "protocol.json"
    require(
        captured["input_artifact_sha256"].get(str(protocol_path)) == digest(protocol_path),
        "Capture protocol is not hash bound",
    )
    protocol = read(protocol_path)
    require(
        protocol["site"] == SITES[model] and protocol["bank_sha256"] == seal["bank_sha256"],
        "Protocol model/site/bank differs",
    )
    require(
        protocol["source_sha256"] == seal["source_sha256"], "Protocol source map differs from seal"
    )
    rows = payload["rows"]
    pairs = test_roster(rows, groups)
    with np.load(capture / "features.npz", allow_pickle=False) as archive:
        require(
            set(archive.files)
            == {"features", "row_ids", "output_margins", "strict_first_token_correct"},
            "Wrong capture feature archive inventory",
        )
        require(
            archive["row_ids"].dtype.kind in "US"
            and archive["row_ids"].tolist() == [r["row_id"] for r in rows],
            "Capture feature row ordering differs",
        )
        features = archive["features"].copy()
        stored_margins, stored_strict = (
            archive["output_margins"].copy(),
            archive["strict_first_token_correct"].copy(),
        )
    require(
        features.dtype == np.float32
        and features.shape == (4992, h)
        and np.isfinite(features).all(),
        "Invalid test features",
    )
    output, strict = output_scores(
        rows, payload["answer_token_ids"], protocol["tokenizer"]["answer_token_ids"]
    )
    require(
        stored_margins.dtype == np.float64
        and np.array_equal(stored_margins, output)
        and stored_strict.dtype == bool
        and np.array_equal(stored_strict, strict),
        "Capture output baseline differs",
    )
    paths = [fit / n for n in ("settings.json", "fit_manifest.json", "coefficients.npz")]
    paths += [capture / n for n in ("rows.json", "manifest.json", "features.npz")]
    paths += [campaign / model / "fit/rows.json", protocol_path]
    inputs = {str(p): digest(p) for p in paths}
    inputs.update(settings["input_artifact_sha256"])
    return rows, features, parameters, pairs, output, strict, payload["answer_token_ids"], inputs


def rank_auc(labels, scores):
    y, x = np.asarray(labels), np.asarray(scores)
    require(
        y.ndim == x.ndim == 1 and y.shape == x.shape and np.isfinite(x).all(),
        "Invalid AUROC arrays",
    )
    positive = int(np.sum(y == 1))
    negative = int(np.sum(y == 0))
    require(positive and negative and positive + negative == len(y), "AUROC needs both classes")
    order = np.argsort(x, kind="stable")
    sorted_x, sorted_y = x[order], y[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_x[1:] != sorted_x[:-1]) + 1, len(x)]
    rank_sum = 0.0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        rank_sum += float(np.sum(sorted_y[start:end])) * ((start + 1 + end) / 2)
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def binary_metrics(truth, predictions):
    truth, predictions = np.asarray(truth), np.asarray(predictions)
    require(
        truth.shape == predictions.shape and len(truth) > 0 and np.isin(predictions, (0, 1)).all(),
        "Invalid binary predictions",
    )
    return {
        "rows": len(truth),
        "correct": int(np.sum(truth == predictions)),
        "row_accuracy": float(np.mean(truth == predictions)),
        "class_counts": {str(k): int(np.sum(truth == k)) for k in (0, 1)},
        "class_recalls": {
            str(k): float(np.mean(predictions[truth == k] == k)) if np.any(truth == k) else None
            for k in (0, 1)
        },
    }


def verdict_metrics(rows, predictions, output=None):
    truth = np.array([r["gold"] for r in rows])
    result = binary_metrics(truth, predictions)
    panels, nuisance = (
        defaultdict(dict),
        {name: defaultdict(dict) for name in ("world", "name_map")},
    )
    for row, predicted in zip(rows, predictions):
        state = row["state"]
        prefix = (row["group_id"], row["renderer"], row["condition"])
        if row["condition"] != "asserted":
            panel = (*prefix, state["update"], state["world"], state["name_map"])
            cell = (
                state["initial"] if row["condition"] == "one_step" else state["first_receipt"],
                state["receipt"],
            )
            require(cell not in panels[panel], "Duplicate reconstructed panel cell")
            panels[panel][cell] = int(predicted == row["gold"])
        for name in nuisance:
            key = (*prefix, tuple(sorted((k, v) for k, v in state.items() if k != name)))
            require(state[name] not in nuisance[name][key], "Duplicate nuisance cell")
            nuisance[name][key][state[name]] = (int(predicted), row["gold"])
    require(
        all(set(v) == set(product((0, 1), repeat=2)) for v in panels.values()),
        "Incomplete reconstructed panel",
    )
    totals = [sum(panel.values()) for panel in panels.values()]
    result.update(
        quartets=len(totals),
        complete_quartets=totals.count(4),
        quartet_accuracy=totals.count(4) / len(totals) if totals else None,
        quartet_correct_cells_histogram={str(k): totals.count(k) for k in range(5)},
    )
    result["nuisance_pairs"] = {}
    for name, pairs in nuisance.items():
        require(
            all(set(v) == {0, 1} and v[0][1] == v[1][1] for v in pairs.values()),
            "Nuisance changed gold",
        )
        result["nuisance_pairs"][name] = {
            "pairs": len(pairs),
            "verdict_flips": sum(v[0][0] != v[1][0] for v in pairs.values()),
            "both_correct": sum(v[0][0] == v[0][1] and v[1][0] == v[1][1] for v in pairs.values()),
        }
    if output is not None:
        probability = np.empty_like(output)
        positive = output >= 0
        probability[positive] = 1 / (1 + np.exp(-output[positive]))
        probability[~positive] = np.exp(output[~positive]) / (1 + np.exp(output[~positive]))
        result["conditional_yes_no_score"] = {
            "raw_auroc": rank_auc(truth, output),
            "mean_log_loss": float(np.mean(np.logaddexp(0, output) - truth * output)),
            "brier": float(np.mean((probability - truth) ** 2)),
        }
    return result


def factor_metrics(rows, margins):
    truth = np.array([[r["state"][name] for name in FIELDS] for r in rows])
    predictions = (margins > 0).astype(np.int64)
    result = {
        "rows": len(rows),
        "all_four_correct": float(np.mean(np.all(predictions == truth, axis=1))),
        "factor_order": list(FACTORS),
        "factors": {},
    }
    for column, name in enumerate(FACTORS):
        entry = binary_metrics(truth[:, column], predictions[:, column])
        entry["raw_auroc"] = rank_auc(truth[:, column], margins[:, column])
        entry["class_margin_quantiles_10_50_90"] = {
            str(k): np.quantile(
                margins[truth[:, column] == k, column], [0.1, 0.5, 0.9], method="linear"
            ).tolist()
            for k in (0, 1)
        }
        entry["positive_minus_negative_mean_margin"] = float(
            margins[truth[:, column] == 1, column].mean()
            - margins[truth[:, column] == 0, column].mean()
        )
        result["factors"][name] = entry
    return result


def contrast(contexts, comparator):
    groups = sorted(contexts)
    require(len(groups) == 24, "Need all 24 paired bootstrap contexts")
    delta = np.array([contexts[g]["Mixed512"] - contexts[g][comparator] for g in groups])
    # Direct multinomial frequencies provide the same paired resamples while
    # calculating each replicate independently of production's indexed mean.
    rng = np.random.Generator(np.random.PCG64(20261127))
    samples = rng.integers(24, size=(10000, 24))
    values = np.array([np.dot(np.bincount(draw, minlength=24), delta) / 24 for draw in samples])
    return {
        "contrast": "Mixed512_minus_" + comparator,
        "renderer": "B",
        "condition": "correction_hard",
        "endpoint": "complete_four_case_quartet_accuracy",
        "groups": 24,
        "quartets": 192,
        "Mixed512": float(np.mean([contexts[g]["Mixed512"] for g in groups])),
        comparator: float(np.mean([contexts[g][comparator] for g in groups])),
        "difference": float(delta.mean()),
        "conditional_percentile_interval_95": np.quantile(
            values, [0.025, 0.975], method="linear"
        ).tolist(),
        "bootstrap_samples": 10000,
        "bootstrap_seed": 20261127,
        "group_values": [
            {
                "group_id": g,
                "Mixed512": contexts[g]["Mixed512"],
                comparator: contexts[g][comparator],
                "difference": contexts[g]["Mixed512"] - contexts[g][comparator],
            }
            for g in groups
        ],
    }


def reconstruct(rows, features, parameters, pairs, output, strict, tokens, model):
    standardized = (features.astype(np.float64) - parameters["mean"]) / parameters["scale"]
    lengths = np.sqrt(np.einsum("ij,ij->i", standardized, standardized))
    lengths[lengths < 10 * np.finfo(np.float64).eps] = 1
    mapped = standardized / lengths[:, None]
    margins = np.stack(
        [mapped @ parameters[arm + "__coef"].T + parameters[arm + "__intercept"] for arm in ARMS],
        axis=1,
    )
    margins[:, 3] += (
        np.array([r["renderer"] == "B" for r in rows])[:, None] * parameters["Offset__B_offset"]
    )
    bits = (margins > 0).astype(np.int64)
    verdicts = np.empty(bits.shape[:2], dtype=np.int64)
    for row_index in range(len(rows)):
        for arm_index in range(4):
            initial, first, read_first, read_second = bits[row_index, arm_index]
            verdicts[row_index, arm_index] = (
                1 - first if read_second else first if read_first else initial
            )
    arrays = {
        "row_ids": np.array([r["row_id"] for r in rows]),
        "arms": np.array(ARMS),
        "factor_order": np.array(FACTORS),
        "factor_margins": margins,
        "factor_predictions": bits,
        "verdicts": verdicts,
        "output_margins": output,
        "strict_first_token_correct": strict,
    }
    metrics = {
        "arms": {},
        "factor_diagnostics": {},
        "output_baseline": {},
        "paired_format": {},
        "output_first_token_detail": {},
    }
    groups = sorted({r["group_id"] for r in rows})
    contexts = {g: {} for g in groups}
    scopes = {
        (fmt, condition): np.array(
            [i for i, r in enumerate(rows) if r["renderer"] == fmt and r["condition"] == condition]
        )
        for fmt, condition in product(("A", "B"), CONDITIONS)
    }
    for arm_index, arm in enumerate(ARMS):
        metrics["arms"][arm], metrics["factor_diagnostics"][arm] = {}, {}
        for (fmt, condition), indices in scopes.items():
            aggregate = verdict_metrics([rows[i] for i in indices], verdicts[indices, arm_index])
            entries = []
            for group in groups:
                within = np.array([i for i in indices if rows[i]["group_id"] == group])
                entry = verdict_metrics([rows[i] for i in within], verdicts[within, arm_index])
                entries.append({"group_id": group, **entry})
                if fmt == "B" and condition == "correction_hard":
                    require(entry["quartets"] == 8, "Wrong hard context denominator")
                    contexts[group][arm] = entry["quartet_accuracy"]
            metrics["arms"][arm][fmt + "/" + condition] = {
                "aggregate": aggregate,
                "groups": entries,
            }
        for fmt, stratum in product(
            ("A", "B"), ("correction_all", "correction_easy", "correction_hard")
        ):
            ix = np.array(
                [
                    i
                    for i, r in enumerate(rows)
                    if r["renderer"] == fmt
                    and (
                        r["condition"].startswith("correction_")
                        if stratum == "correction_all"
                        else r["condition"] == stratum
                    )
                ]
            )
            entries = []
            for group in groups:
                within = np.array([i for i in ix if rows[i]["group_id"] == group])
                entries.append(
                    {
                        "group_id": group,
                        **factor_metrics([rows[i] for i in within], margins[within, arm_index]),
                    }
                )
            metrics["factor_diagnostics"][arm][fmt + "/" + stratum] = {
                "aggregate": factor_metrics([rows[i] for i in ix], margins[ix, arm_index]),
                "groups": entries,
            }
        correction_pairs = [pair for key, pair in pairs.items() if key[1].startswith("correction_")]
        aa = margins[[p["A"] for p in correction_pairs], arm_index]
        bb = margins[[p["B"] for p in correction_pairs], arm_index]
        metrics["paired_format"][arm] = {
            "correction_pairs": len(aa),
            "factor_decision_flips": np.count_nonzero((aa > 0) != (bb > 0), axis=0).tolist(),
            "factor_margin_delta_B_minus_A_quantiles_10_50_90": np.quantile(
                bb - aa, [0.1, 0.5, 0.9], axis=0, method="linear"
            ).T.tolist(),
        }
    for (fmt, condition), ix in scopes.items():
        subset = [rows[i] for i in ix]
        metrics["output_baseline"][fmt + "/" + condition] = {
            "conditional": verdict_metrics(subset, (output[ix] > 0).astype(np.int64), output[ix]),
            "strict_first_token": {
                "rows": len(ix),
                "correct_full_denominator": int(strict[ix].sum()),
                "accuracy_full_denominator": float(strict[ix].mean()),
            },
        }
        summaries = [r["output_logits"] for r in subset]
        valid = [s["argmax_token_id"] in tokens.values() for s in summaries]
        metrics["output_first_token_detail"][fmt + "/" + condition] = {
            "candidate_first_token_rate": float(np.mean(valid)),
            "outside_candidate_first_tokens": len(ix) - sum(valid),
            "full_vocab_argmax_ties": sum(s["argmax_tie_count"] > 1 for s in summaries),
            "mean_candidate_token_mass": float(
                np.mean([np.exp(s["yes_logprob"]) + np.exp(s["no_logprob"]) for s in summaries])
            ),
        }
    metrics["primary"] = {
        **contrast(contexts, "A512"),
        "role": "primary" if model == "llama" else "named_secondary_replication",
    }
    metrics["named_diagnostic"] = contrast(contexts, "Offset")
    return arrays, metrics


def compare_tree(expected, actual, path="metrics", counts=None):
    counts = (
        counts
        if counts is not None
        else {"numeric": 0, "exact": 0, "maximum_absolute_difference": 0.0}
    )
    if isinstance(expected, dict):
        require(
            isinstance(actual, dict) and set(expected).issubset(actual),
            "Missing metric keys at " + path,
        )
        for key, value in expected.items():
            compare_tree(value, actual[key], path + "/" + key, counts)
    elif isinstance(expected, list):
        require(
            isinstance(actual, list) and len(expected) == len(actual),
            "Metric length differs at " + path,
        )
        for i, value in enumerate(expected):
            compare_tree(value, actual[i], path + "/" + str(i), counts)
    elif isinstance(expected, float):
        require(
            isinstance(actual, (int, float))
            and np.isfinite(actual)
            and np.isclose(expected, actual, atol=2e-10, rtol=2e-10),
            "Numeric metric differs at " + path,
        )
        counts["numeric"] += 1
        counts["maximum_absolute_difference"] = max(
            counts["maximum_absolute_difference"], abs(expected - actual)
        )
    else:
        require(expected == actual, "Exact metric differs at " + path)
        counts["exact"] += 1
    return counts


def compare_saved(analysis, arrays, metrics, model, seal_path, fit):
    manifest = read(analysis / "manifest.json")
    require(
        manifest.get("schema") == "format-adaptation-analysis-manifest-v1"
        and manifest["status"] == "complete",
        "Incomplete saved analysis",
    )
    for name in ("results.json", "predictions.npz"):
        require(
            manifest["output_sha256"].get(name) == digest(analysis / name),
            "Analysis artifact hash differs",
        )
    bound_files(manifest["input_artifact_sha256"])
    require(
        manifest["input_artifact_sha256"].get(str(seal_path)) == digest(seal_path),
        "Analysis seal differs",
    )
    for name in ("settings.json", "coefficients.npz", "fit_manifest.json"):
        require(
            manifest["input_artifact_sha256"].get(str(fit / name)) == digest(fit / name),
            "Analysis used another fit",
        )
    maximum, scalar_count = 0.0, 0
    with np.load(analysis / "predictions.npz", allow_pickle=False) as saved:
        require(set(saved.files) == set(arrays), "Saved prediction array inventory differs")
        for name, expected in arrays.items():
            actual = saved[name]
            require(actual.shape == expected.shape, "Saved prediction shape differs: " + name)
            if name in ("factor_margins", "output_margins"):
                require(
                    actual.dtype == np.float64
                    and np.isfinite(actual).all()
                    and np.allclose(expected, actual, atol=2e-10, rtol=2e-10),
                    "Saved margins differ: " + name,
                )
                maximum = max(maximum, float(np.max(np.abs(expected - actual))))
            else:
                require(np.array_equal(expected, actual), "Saved decisions/order differ: " + name)
            scalar_count += expected.size
    result = read(analysis / "results.json")
    require(
        result["schema"] == "format-adaptation-analysis-v1"
        and result["model_key"] == model
        and result["site"] == SITES[model]
        and result["confirmatory"] is False,
        "Saved result identity differs",
    )
    checks = compare_tree(metrics, result)
    return {
        "prediction_scalars_compared": scalar_count,
        "maximum_prediction_margin_difference": maximum,
        "metric_comparisons": checks,
        "analysis_artifact_sha256": {
            str(analysis / name): digest(analysis / name)
            for name in ("manifest.json", "results.json", "predictions.npz")
        },
    }


def audit(campaign, output, seal_path=None, *, allow_fixture=False):
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    require(
        not output.exists() and not output.is_relative_to(campaign),
        "Use fresh audit output outside campaign",
    )
    seal_path = Path(seal_path).resolve() if seal_path else campaign / "test_seal.json"
    output.mkdir(parents=True)
    receipt = {
        "schema": "format-adaptation-independent-audit-v1",
        "status": "reconstructing",
        "campaign": str(campaign),
        "started_at": now(),
        "model_calls": 0,
        "classifier_fits": 0,
        "production_functions_imported": False,
        "saved_analysis_opened_before_both_reconstructions_frozen": False,
    }
    try:
        seal = read(seal_path)
        require(
            seal.get("schema") == "format-adaptation-test-seal-v1"
            and set(seal.get("models", {})) == {"llama", "qwen"},
            "Wrong both-model seal",
        )
        sealed_sources(seal)
        reconstructed, bindings = {}, {str(seal_path): digest(seal_path)}
        for model in ("llama", "qwen"):
            rows, features, parameters, pairs, scores, strict, tokens, inputs = load_inputs(
                campaign, model, seal, seal_path, allow_fixture=allow_fixture
            )
            arrays, metrics = reconstruct(
                rows, features, parameters, pairs, scores, strict, tokens, model
            )
            del features
            np.savez_compressed(output / (model + "_reconstructed_predictions.npz"), **arrays)
            write_new(output / (model + "_reconstructed_metrics.json"), metrics)
            reconstructed[model] = arrays, metrics
            bindings.update(inputs)
        bound_files(bindings)
        sealed_sources(seal)
        reconstruction_files = sorted(output.glob("*_reconstructed_*"))
        frozen = {
            "schema": "format-adaptation-independent-reconstruction-seal-v1",
            "frozen_at": now(),
            "both_models_reconstructed": ["llama", "qwen"],
            "saved_analysis_artifacts_opened": 0,
            "source_sha256": {str(p): digest(p) for p in CODE},
            "input_sha256": bindings,
            "output_sha256": {p.name: digest(p) for p in reconstruction_files},
        }
        write_new(output / "reconstruction_seal.json", frozen)
        receipt.update(
            status="comparing_saved_analysis",
            reconstruction_seal_sha256=digest(output / "reconstruction_seal.json"),
        )
        # This is the first point at which any saved analysis may be opened.
        receipt["models"] = {}
        for model in ("llama", "qwen"):
            arrays, metrics = reconstructed[model]
            receipt["models"][model] = compare_saved(
                campaign / (model + "_analysis"),
                arrays,
                metrics,
                model,
                seal_path,
                campaign / (model + "_fit"),
            )
        bound_files(bindings)
        sealed_sources(seal)
        require(
            all(digest(output / name) == value for name, value in frozen["output_sha256"].items()),
            "Frozen reconstruction changed during comparison",
        )
        receipt.update(
            status="pass",
            finished_at=now(),
            analysis_output_sha256={
                str(campaign / (model + "_analysis") / name): digest(
                    campaign / (model + "_analysis") / name
                )
                for model in ("llama", "qwen")
                for name in ("results.json", "predictions.npz")
            },
            source_sha256=frozen["source_sha256"],
            input_sha256=bindings,
            scope="All saved prediction arrays; arm row/class/quartet/nuisance metrics aggregate and by context; four correction factors in pooled/easy/hard strata aggregate and by context; exact A/B correction pairing diagnostics; conditional/strict output summaries; primary and named diagnostic with all24 context values and fixed10000 resamples. No classifier refitting, no raw full-vocabulary argmax reconstruction, no semantic certification, no uncertainty over new fitting cohorts.",
        )
    except BaseException as error:
        receipt.update(
            status="failed_preserved",
            failed_at=now(),
            error_type=type(error).__name__,
            error=str(error),
        )
        write_new(output / "audit_receipt.json", receipt)
        raise
    write_new(output / "audit_receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seal", type=Path)
    args = parser.parse_args()
    result = audit(args.campaign, args.output, args.seal)
    print(
        json.dumps({"status": result["status"], "receipt": str(args.output / "audit_receipt.json")})
    )


if __name__ == "__main__":
    main()
