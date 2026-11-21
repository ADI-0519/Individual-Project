"""Prospective CPU audit and descriptive analysis of the neutral crossed assay.

Scores are reconstructed from raw arrays, without the runner's scoring helpers.
No fitting, threshold selection, resampling of dependent cells, or gate rescue.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .crossed_belief_bank import build_messages, load_bank
from .storage import canonical_hash, file_sha256, read_json

HEAD = "nested_question__sender__nested__layer_16"
CONDITIONS = ("story", "nested_question")


def require(value, message):
    if not value:
        raise ValueError(message)


def snapshot(root):
    return {
        p.relative_to(root).as_posix(): file_sha256(p)
        for p in sorted(Path(root).rglob("*"))
        if p.is_file()
    }


def local_path(value):
    """Resolve recorded WSL paths on the original Windows host, if necessary."""
    path = Path(value)
    if path.exists() or not str(value).startswith("/") or Path("/mnt/c").exists():
        return path
    if str(value).startswith("/mnt/c/"):
        return Path("C:/" + str(value)[7:])
    return Path("//wsl.localhost/Ubuntu" + str(value))


def auc(labels, scores):
    positive = [s for y, s in zip(labels, scores, strict=True) if y == 1]
    negative = [s for y, s in zip(labels, scores, strict=True) if y == 0]
    require(positive and negative, "AUROC requires both classes")
    return sum(float(a > b) + 0.5 * float(a == b) for a in positive for b in negative) / (
        len(positive) * len(negative)
    )


def reconstruct_probe(vector, metadata, parameters):
    head = metadata["heads"][HEAD]
    require(head["layer"] == "layer_16", "Frozen layer differs")
    vector = np.asarray(vector, dtype=np.float64)
    require(
        vector.shape == (metadata["feature_widths"]["layer_16"],) and np.isfinite(vector).all(),
        "Invalid saved feature",
    )
    prefix = head["activation_transform"]
    centered = (vector - parameters[prefix + "__mean"]) / parameters[prefix + "__scale"]
    norm = float(np.sqrt(np.sum(centered * centered)))
    if norm < 10 * np.finfo(np.float64).eps:
        norm = 1.0
    coefficient = np.asarray(parameters[head["parameter_prefix"] + "__coef"])[0]
    margin = float(
        np.dot(centered / norm, coefficient)
        + parameters[head["parameter_prefix"] + "__intercept"][0]
    )
    probability = float(np.exp(-np.logaddexp(0.0, -margin)))
    return {
        "head_id": HEAD,
        "margin": margin,
        "probability": probability,
        "prediction": int(probability > 0.5),
    }


def reconstruct_logits(vector, tokens):
    require(
        vector.ndim == 1 and vector.dtype == np.dtype("float32") and np.isfinite(vector).all(),
        "Need saved finite full float32 vocabulary logits",
    )
    require(
        set(tokens) == {"YES", "NO"}
        and tokens["YES"] != tokens["NO"]
        and all(type(v) is int and 0 <= v < len(vector) for v in tokens.values()),
        "Invalid answer token identities",
    )
    values = vector.astype(np.float64)
    yes, no = (float(values[tokens[word]]) for word in ("YES", "NO"))
    log_z = float(np.logaddexp.reduce(values))
    return {
        "yes_logit": yes,
        "no_logit": no,
        "yes_logprob": yes - log_z,
        "no_logprob": no - log_z,
        "yes_minus_no": yes - no,
        "p_yes_given_yes_no": float(np.exp(-np.logaddexp(0.0, no - yes))),
        "full_vocab_logsumexp": log_z,
        "vocab_size": len(vector),
        "argmax_token_id": int(np.argmax(vector)),
        "logits_values_sha256": canonical_hash(vector.tolist()),
    }


def verify_answer(row, protocol, tokenizer=None):
    generation = row["generation"]
    if row["condition"] == "story":
        require(generation is None and row["answer"] is None, "Unqueried answer was generated")
        return
    ids = generation["generated_token_ids"]
    settings = protocol["generation"]
    require(
        generation["prompt_token_ids"] == row["prompt_token_ids"]
        and isinstance(ids, list)
        and 0 < len(ids) <= settings["max_new_tokens"]
        and all(type(t) is int and t >= 0 for t in ids)
        and ids[0] == row["output_logits"]["argmax_token_id"],
        "Greedy generation prefix/first token differs",
    )
    eos = set(settings["eos_token_ids"])
    stopped = ids[-1] in eos
    require((generation["finish_reason"] == "eos") == stopped, "EOS stop differs")
    unexpected = any(
        t in eos | set(protocol["tokenizer"]["all_special_ids"])
        for t in (ids[:-1] if stopped else ids)
    )
    match = re.fullmatch(r"\s*(YES|NO)[.!]?\s*", generation["text"], re.IGNORECASE)
    prediction = int(match.group(1).upper() == "YES") if match else None
    if not stopped or unexpected:
        prediction = None
    require(
        row["answer"]["prediction"] == prediction
        and row["answer"]["capped"] == (not stopped)
        and row["answer"]["unexpected_special"] == unexpected,
        "Saved strict answer parse differs",
    )
    if tokenizer is not None:
        require(
            tokenizer.decode(ids, skip_special_tokens=True).strip() == generation["text"],
            "Generated token/text decoding differs",
        )


def metrics(rows, *, queried):
    y = [r["gold"] for r in rows]
    confusion = [[0, 0], [0, 0]]
    for row in rows:
        confusion[row["gold"]][row["probe"]["prediction"]] += 1
    result = {
        "n": len(rows),
        "mapping_paired_cells": len({r["sender_input_id"].rsplit("_map", 1)[0] for r in rows}),
        "class_counts": {str(k): v for k, v in sorted(Counter(y).items())},
        "probe_correct": sum(r["gold"] == r["probe"]["prediction"] for r in rows),
        "probe_confusion": confusion,
        "probe_auroc": auc(y, [r["probe"]["margin"] for r in rows]),
        "probe_brier": sum((r["probe"]["probability"] - r["gold"]) ** 2 for r in rows) / len(rows),
        "shortcut_correct": {
            key: sum(r["source"][key] == r["gold"] for r in rows)
            for key in ("initial_slot", "message_slot", "receipt_reports_read", "world_slot")
        },
        "constant_zero_correct": sum(v == 0 for v in y),
        "symbolic_rule_correct": len(rows),
    }
    if queried:
        answer_confusion = [[0, 0], [0, 0]]
        for row in rows:
            if row["answer"]["prediction"] is not None:
                answer_confusion[row["gold"]][row["answer"]["prediction"]] += 1
        result.update(
            output_score_auroc=auc(y, [r["output_logits"]["yes_minus_no"] for r in rows]),
            output_fixed_correct=sum(
                int(r["output_logits"]["yes_minus_no"] > 0) == r["gold"] for r in rows
            ),
            answer_correct=sum(r["answer"]["prediction"] == r["gold"] for r in rows),
            answer_invalid=sum(r["answer"]["prediction"] is None for r in rows),
            answer_confusion=answer_confusion,
            answer_text_counts=dict(sorted(Counter(r["generation"]["text"] for r in rows).items())),
        )
    return result


def counterfactual_pairs(rows, factor):
    """Pairs change one authored source bit or only the physical-name mapping."""
    keys = ("initial_slot", "message_slot", "receipt_reports_read", "world_slot")
    groups = {}
    for row in rows:
        identity = (row["template_id"],) + tuple(row["source"][k] for k in keys if k != factor)
        if factor != "mapping_id":
            identity += (row["mapping_id"],)
        value = row["mapping_id"] if factor == "mapping_id" else row["source"][factor]
        require(value not in groups.setdefault(identity, {}), "Duplicate paired cell")
        groups[identity][value] = row
    result = []
    for group in groups.values():
        require(set(group) == {0, 1}, "Incomplete counterfactual pair")
        low, high = group[0], group[1]
        delta = high["probe"]["margin"] - low["probe"]["margin"]
        result.append(
            {
                "low_id": low["id"],
                "high_id": high["id"],
                "gold_low": low["gold"],
                "gold_high": high["gold"],
                "margin_delta": delta,
                "gold_oriented_delta": (high["gold"] - low["gold"]) * delta,
                "prediction_changed": low["probe"]["prediction"] != high["probe"]["prediction"],
                "both_correct": all(r["probe"]["prediction"] == r["gold"] for r in group.values()),
            }
        )
    return result


def summarize_pairs(pairs):
    require(bool(pairs), "Need nonempty pair roster")
    return {
        "n_pairs": len(pairs),
        "positive_gold_oriented": sum(r["gold_oriented_delta"] > 0 for r in pairs),
        "prediction_flips": sum(r["prediction_changed"] for r in pairs),
        "both_correct": sum(r["both_correct"] for r in pairs),
        "mean_margin_delta": sum(r["margin_delta"] for r in pairs) / len(pairs),
        "mean_abs_margin_delta": sum(abs(r["margin_delta"]) for r in pairs) / len(pairs),
        "mean_gold_oriented_delta": sum(r["gold_oriented_delta"] for r in pairs) / len(pairs),
        "records": pairs,
    }


def hidden_reader_control(rows):
    require(
        all(r["source"]["initial_slot"] != r["source"]["message_slot"] for r in rows),
        "Hidden-reader exact chance applies only to I != M",
    )
    labels, scores, predictions = [], [], []
    for row in rows:
        for actual in (0, 1):
            labels.append(row["source"]["message_slot" if actual else "initial_slot"])
            scores.append(row["probe"]["margin"])
            predictions.append(row["probe"]["prediction"])
    accuracy = sum(a == b for a, b in zip(labels, predictions, strict=True)) / len(labels)
    area = auc(labels, scores)
    require(accuracy == 0.5 and area == 0.5, "Hidden-reader information leaked")
    return {
        "fictional_rows": len(labels),
        "unique_sender_inputs": len(rows),
        "accuracy": accuracy,
        "auroc": area,
    }


def condition_analysis(rows):
    queried = rows[0]["condition"] == "nested_question"
    primary = [r for r in rows if r["source"]["initial_slot"] != r["source"]["message_slot"]]
    invariant = [r for r in rows if r not in primary]
    require(len(rows) == 128 and len(primary) == len(invariant) == 64, "Subset roster differs")
    primary_metrics = metrics(primary, queried=queried)
    per_mapping = {
        str(m): metrics([r for r in primary if r["mapping_id"] == m], queried=queried)
        for m in (0, 1)
    }
    per_template = {
        t: metrics([r for r in primary if r["template_id"] == t], queried=queried)
        for t in sorted({r["template_id"] for r in rows})
    }
    control_metrics = metrics(invariant, queried=queried)
    control_mapping = {
        str(m): metrics([r for r in invariant if r["mapping_id"] == m], queried=queried)
        for m in (0, 1)
    }
    control_template = {
        t: metrics([r for r in invariant if r["template_id"] == t], queried=queried)
        for t in sorted({r["template_id"] for r in rows})
    }
    pairs = {
        "receipt_primary": summarize_pairs(counterfactual_pairs(primary, "receipt_reports_read")),
        "receipt_invariant": summarize_pairs(
            counterfactual_pairs(invariant, "receipt_reports_read")
        ),
        "world_all": summarize_pairs(counterfactual_pairs(rows, "world_slot")),
        "world_primary": summarize_pairs(counterfactual_pairs(primary, "world_slot")),
        "mapping_all": summarize_pairs(counterfactual_pairs(rows, "mapping_id")),
        "mapping_primary": summarize_pairs(counterfactual_pairs(primary, "mapping_id")),
    }
    probe_failed = []
    if primary_metrics["probe_correct"] < 48:
        probe_failed.append("primary/probe_correct<48/64")
    for label, group, floor in (("mapping", per_mapping, 24), ("template", per_template, 10)):
        for key, cell in group.items():
            if cell["probe_correct"] < floor:
                probe_failed.append(f"{label}/{key}/probe_correct<{floor}/{cell['n']}")
    if pairs["receipt_primary"]["positive_gold_oriented"] < 24:
        probe_failed.append("receipt_primary/positive_gold_oriented<24/32")
    control_cells = [
        ("invariant", control_metrics, 48),
        *[("invariant/mapping/" + k, v, 24) for k, v in control_mapping.items()],
        *[("invariant/template/" + k, v, 10) for k, v in control_template.items()],
    ]
    for label, cell, floor in control_cells:
        if cell["probe_correct"] < floor:
            probe_failed.append(f"{label}/probe_correct<{floor}/{cell['n']}")
    result = {
        "primary": primary_metrics,
        "invariant": control_metrics,
        "overall": metrics(rows, queried=queried),
        "per_mapping_primary": per_mapping,
        "per_template_primary": per_template,
        "per_mapping_invariant": control_mapping,
        "per_template_invariant": control_template,
        "pairs": pairs,
        "hidden_reader_primary": hidden_reader_control(primary),
        "probe_qualification": {"passed": not probe_failed, "failed_checks": probe_failed},
    }
    if queried:
        rank_failed, answer_failed = [], []
        for label, cell, floor, auc_floor in [
            ("primary", primary_metrics, 48, 0.75),
            *[("mapping/" + k, v, 24, 0.75) for k, v in per_mapping.items()],
            *[("template/" + k, v, 10, 0.625) for k, v in per_template.items()],
        ]:
            if cell["output_score_auroc"] < auc_floor:
                rank_failed.append(f"{label}/output_score_auroc<{auc_floor}")
            if cell["answer_correct"] < floor:
                answer_failed.append(f"{label}/answer_correct<{floor}/{cell['n']}")
        if primary_metrics["answer_invalid"]:
            answer_failed.append("primary/answer_invalid>0")
        for label, cell, floor in control_cells:
            if cell["answer_correct"] < floor:
                answer_failed.append(f"{label}/answer_correct<{floor}/{cell['n']}")
        if control_metrics["answer_invalid"]:
            answer_failed.append("invariant/answer_invalid>0")
        result.update(
            output_ranking_qualification={"passed": not rank_failed, "failed_checks": rank_failed},
            behavioral_qualification={"passed": not answer_failed, "failed_checks": answer_failed},
        )
    return result


def analyse(run, *, tokenizer=None, native=None):
    run = Path(run)
    protocol, manifest = read_json(run / "protocol.json"), read_json(run / "manifest.json")
    require(protocol["schema"] == "crossed_belief_validation_protocol_v1", "Wrong protocol")
    require(
        manifest["status"] == "complete_fresh_development_validation"
        and manifest["n_completed_rows"] == manifest["model_measurement_forwards"] == 256
        and manifest["generated_answers"] == 128,
        "Need a complete single-attempt neutral validation",
    )
    require(
        file_sha256(run / "protocol.json") == manifest["protocol_sha256"]
        and read_json(run / "attempt_started.json")["protocol_sha256"]
        == manifest["protocol_sha256"],
        "Attempt/protocol binding differs",
    )
    all_hashes = snapshot(run)
    require(
        {k: v for k, v in all_hashes.items() if k != "manifest.json"} == manifest["output_sha256"],
        "Saved output ledger differs",
    )
    for key, field in (
        ("source_run", "source_artifact_sha256"),
        ("known_run", "known_artifact_sha256"),
        ("fit_dir", "fitted_artifact_sha256"),
    ):
        require(
            snapshot(local_path(protocol[key])) == protocol[field], "Source binding differs: " + key
        )
    for path, digest in protocol["policy_sha256"].items():
        require(file_sha256(local_path(path)) == digest, "Frozen policy differs: " + path)
    for key in ("bank", "config"):
        require(
            file_sha256(local_path(protocol[key + "_path"])) == protocol[key + "_sha256"],
            key + " bytes differ",
        )
    require(protocol["primary_head"] == HEAD and protocol["no_refit"], "Instrument differs")
    require(protocol["threshold"] == 0.5 and protocol["threshold_tie"] == 0, "Threshold differs")
    bank = load_bank(local_path(protocol["bank_path"]))
    visible = {r["id"]: r for r in bank["sender_inputs"]}
    expected = {r["id"]: r for r in protocol["planned_rows"]}
    require(len(expected) == 256, "Planned row roster differs")
    rows = [read_json(p) for p in sorted((run / "row_outputs").glob("*.json"))]
    require(
        len(rows) == 256
        and len({r["id"] for r in rows}) == 256
        and {r["id"] for r in rows} == set(expected),
        "Observed row roster differs",
    )
    fit = local_path(protocol["fit_dir"])
    metadata = read_json(fit / "fit_metadata.json")
    with np.load(fit / "parameters.npz", allow_pickle=False) as data:
        parameters = {k: data[k].copy() for k in data.files}
    errors = {"probe_margin": 0.0, "probe_probability": 0.0, "output_scalar": 0.0}
    reference = read_json(run / "reference_check.json")
    require(
        all(reference[k] is True for k in ("ids_exact", "feature_exact", "score_exact")),
        "Reference failed",
    )
    with np.load(run / "reference_forward.npz", allow_pickle=False) as data:
        require(
            np.array_equal(
                data["layer_16"], np.asarray(protocol["reference"]["feature"], np.float32)
            ),
            "Saved reference vector differs",
        )
        require(
            canonical_hash(data["layer_16"].tolist())
            == reference["feature_values_sha256"]
            == protocol["reference"]["feature_values_sha256"],
            "Reference feature hash differs",
        )
        original_score = reconstruct_probe(data["layer_16"], metadata, parameters)
        original_logits = reconstruct_logits(
            data["next_token_logits"], protocol["answer_token_ids"]
        )
    for recorded in (reference["probe"], protocol["reference"]["probe"]):
        require(
            recorded["head_id"] == HEAD
            and recorded["prediction"] == original_score["prediction"]
            and all(
                abs(recorded[k] - original_score[k]) < 1e-12 for k in ("margin", "probability")
            ),
            "Reference score reconstruction differs",
        )
    for key, value in original_logits.items():
        require(
            abs(value - reference["output_logits"][key]) < 1e-10
            if isinstance(value, float)
            else value == reference["output_logits"][key],
            "Reference output reconstruction differs: " + key,
        )
    layer_check = manifest["layer_index_check"]
    require(
        layer_check["status"] == "passed"
        and layer_check["n_prompts"] == 1
        and set(layer_check["checks"]) == {"8", "16", "24"}
        and all(
            layer_check["checks"][str(layer)] == {"hf_hidden_state_index": layer + 1, "exact": True}
            for layer in (8, 16, 24)
        ),
        "Layer/index exact checks failed",
    )
    for row in rows:
        planned = expected[row["id"]]
        require(
            row["status"] == "complete"
            and row["protocol_sha256"] == manifest["protocol_sha256"]
            and all(row[k] == v for k, v in planned.items()),
            "Observed row differs from frozen prefix",
        )
        source_row = visible[row["sender_input_id"]]
        queried = row["condition"] == "nested_question"
        require(row["condition"] in CONDITIONS, "Unexpected condition")
        for key in ("source", "template_id", "mapping_id", "gold", "stratum", "place_a", "place_b"):
            require(row[key] == source_row[key], "Bank/observed metadata differs: " + key)
        s = row["source"]
        gold = (
            s["receipt_reports_read"] * s["message_slot"]
            + (1 - s["receipt_reports_read"]) * s["initial_slot"]
        )
        require(row["gold"] == gold and "actual_delivery" not in row, "Gold or visibility differs")
        require(
            row["messages"] == build_messages(source_row, queried)
            and canonical_hash(row["messages"]) == row["messages_sha256"]
            and canonical_hash(row["prompt_token_ids"]) == row["prompt_token_ids_sha256"]
            and row["will_generate_answer"] == queried,
            "Exact visible prefix differs",
        )
        if tokenizer is not None:
            require(
                tokenizer.apply_chat_template(
                    row["messages"], tokenize=True, add_generation_prompt=True
                )
                == row["prompt_token_ids"],
                "Pinned tokenizer prefix differs",
            )
        capture = row["capture"]
        require(
            capture["layer"] == 16
            and capture["hf_hidden_state_index"] == 17
            and capture["selected_position"] == len(row["prompt_token_ids"]) - 1
            and capture["selected_token_id"] == row["prompt_token_ids"][-1]
            and capture["generated_answer_tokens_supplied"] == 0
            and capture["perspective_question_supplied"] == queried
            and capture["same_forward_as_logits"] is True
            and capture["prompt_token_ids_sha256"] == row["prompt_token_ids_sha256"],
            "Capture boundary differs",
        )
        path = run / capture["feature_path"]
        require(file_sha256(path) == capture["feature_sha256"], "Raw forward archive differs")
        with np.load(path, allow_pickle=False) as data:
            require(set(data.files) == {"layer_16", "next_token_logits"}, "Archive keys differ")
            feature, logits = data["layer_16"], data["next_token_logits"]
        require(
            canonical_hash(feature.tolist()) == capture["feature_values_sha256"],
            "Feature values differ",
        )
        scored = reconstruct_probe(feature, metadata, parameters)
        require(
            row["probe"]["head_id"] == HEAD and scored["prediction"] == row["probe"]["prediction"],
            "Probe prediction differs",
        )
        for key in ("margin", "probability"):
            error = abs(scored[key] - row["probe"][key])
            require(math.isfinite(error) and error < 1e-12, "Reconstructed probe score differs")
            errors["probe_" + key] = max(errors["probe_" + key], error)
        reconstructed = reconstruct_logits(logits, protocol["answer_token_ids"])
        for key, value in reconstructed.items():
            if isinstance(value, float):
                error = abs(value - row["output_logits"][key])
                require(
                    math.isfinite(error) and error < 1e-10,
                    "Reconstructed output scalar differs: " + key,
                )
                errors["output_scalar"] = max(errors["output_scalar"], error)
            else:
                require(
                    value == row["output_logits"][key],
                    "Reconstructed output identity differs: " + key,
                )
        verify_answer(row, protocol, tokenizer)
    if native is not None:
        require(snapshot(Path(native)) == all_hashes, "Native/mirror complete bytes differ")
    compact_keys = (
        "id",
        "sender_input_id",
        "condition",
        "template_id",
        "mapping_id",
        "source",
        "gold",
        "probe",
        "answer",
    )
    return {
        "schema": "crossed_belief_validation_analysis_v1",
        "status": manifest["status"],
        "execution_commit": protocol["git"]["commit"],
        "protocol_sha256": manifest["protocol_sha256"],
        "bank_sha256": protocol["bank_sha256"],
        "started_at": manifest["started_at"],
        "finished_at": manifest["finished_at"],
        "human_review_complete": protocol["human_review_complete"],
        "model_measurement_forwards": 256,
        "generated_answers": 128,
        "n_visible_inputs": 128,
        "old_goal_conditions_collected": 0,
        "old_goal_qualification_stays_failed": True,
        "tokenizer_verified": tokenizer is not None,
        "native_mirror_verified": native is not None,
        "raw_files": len(all_hashes),
        "raw_bytes": sum(p.stat().st_size for p in run.rglob("*") if p.is_file()),
        "maximum_reconstruction_errors": errors,
        "conditions": {
            condition: condition_analysis([r for r in rows if r["condition"] == condition])
            for condition in CONDITIONS
        },
        "records": [
            {
                **{k: row[k] for k in compact_keys},
                "output_yes_minus_no": row["output_logits"]["yes_minus_no"],
                "answer_text": row["generation"]["text"] if row["generation"] else None,
            }
            for row in rows
        ],
    }
