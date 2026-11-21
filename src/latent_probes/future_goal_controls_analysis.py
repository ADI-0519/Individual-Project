"""Source-bound descriptive analysis for the unexecuted future-goal assay.

No model loading, fitting, calibration, or outcome-dependent selection occurs here.
The neutral gate is recomputed before any non-neutral estimates are returned.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .storage import canonical_hash, file_sha256, read_json

ROOT = Path(__file__).resolve().parents[2]
BANK = ROOT / "data/belief_diagnostic/future_goal_content_controls_draft_v1.json"
BANK_SHA256 = "8c1b64638d12a93585e1bef8260388437ca2404feff6809a214f18a598e78a79"
HEAD = "nested_question__sender__nested__layer_16"
CHANNELS = ("observation", "message")
CONDITIONS = (
    "neutral",
    "own_token_a",
    "own_token_b",
    "own_disk_a",
    "own_disk_b",
    "archive_token_a",
    "archive_token_b",
)
BOOTSTRAP_SEED = 20261121
BOOTSTRAP_DRAWS = 10_000


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def reviewed_bank(path=BANK):
    _require(file_sha256(path) == BANK_SHA256, "Reviewed bank SHA256 differs")
    return read_json(path)


def _bank(bank):
    reviewed = reviewed_bank()
    if bank is not None:
        _require(bank == reviewed, "Analysis bank differs from reviewed source")
    return reviewed


def _answer(row):
    generation = row["generation"]
    text = generation.get("text")
    _require(isinstance(text, str), "Missing saved answer text")
    capped = generation.get("finish_reason") == "length"
    match = re.fullmatch(r"(YES|NO)[.!]?", text.strip(), flags=re.IGNORECASE)
    valid = (
        generation.get("finish_reason") == "eos"
        and not generation.get("unexpected_special_tokens", [])
        and match is not None
    )
    prediction = int(match.group(1).upper() == "YES") if valid else None
    return {
        "prediction": prediction,
        "invalid": not valid,
        "length_capped": capped,
        "correct": valid and prediction == row["gold"],
    }


def _validated_rows(rows, bank):
    expected = {row["id"]: row for row in bank["planned_conditions"]}
    _require(isinstance(rows, list) and rows, "Need saved condition rows")
    _require(len({row.get("id") for row in rows}) == len(rows), "Duplicate condition ID")
    validated = []
    for row in rows:
        _require(row.get("id") in expected, "Unexpected condition ID")
        plan = expected[row["id"]]
        _require(all(row.get(k) == v for k, v in plan.items()), "Condition differs from bank")
        probe = row.get("probe", {})
        margin, probability = probe.get("margin"), probe.get("probability")
        _require(
            type(margin) in (int, float)
            and math.isfinite(margin)
            and type(probability) in (int, float)
            and math.isfinite(probability)
            and 0 <= probability <= 1,
            "Invalid probe score",
        )
        p = float(np.exp(-np.logaddexp(0.0, -margin)))
        _require(
            math.isclose(probability, p, rel_tol=1e-7, abs_tol=1e-10),
            "Probability differs from raw margin",
        )
        _require(probe.get("head_id") == HEAD, "Unexpected probe head")
        _require(
            type(probe.get("prediction")) is int and probe["prediction"] == int(probability > 0.5),
            "Probe threshold differs",
        )
        answer = _answer(row)
        if "answer" in row:
            _require(row["answer"] == answer, "Saved answer disposition differs")
        validated.append({**row, "answer": answer})
    return validated


def _counts(rows):
    return {
        "n": len(rows),
        "probe_correct": sum(row["probe"]["prediction"] == row["gold"] for row in rows),
        "answer_correct": sum(row["answer"]["correct"] for row in rows),
        "invalid": sum(row["answer"]["invalid"] for row in rows),
    }


def qualify_neutral(rows, *, bank=None):
    """Recompute the fixed gate using exactly the 32 source-bound neutral rows."""
    bank = _bank(bank)
    rows = _validated_rows(rows, bank)
    neutral = [row for row in rows if row["condition"] == "neutral"]
    expected = {row["id"] for row in bank["planned_conditions"] if row["condition"] == "neutral"}
    _require(
        len(neutral) == 32 and {row["id"] for row in neutral} == expected,
        "Qualification requires all 32 exact neutral conditions",
    )
    result = {
        "overall": _counts(neutral),
        "per_mapping": {
            str(m): _counts([r for r in neutral if r["mapping_id"] == m]) for m in (0, 1)
        },
        "per_channel": {c: _counts([r for r in neutral if r["channel"] == c]) for c in CHANNELS},
        "crossed": {
            f"{c}/map{m}": _counts(
                [r for r in neutral if r["channel"] == c and r["mapping_id"] == m]
            )
            for c in CHANNELS
            for m in (0, 1)
        },
        "per_label": {str(y): _counts([r for r in neutral if r["gold"] == y]) for y in (0, 1)},
        "per_label_mapping_channel": {
            f"{c}/map{m}/label{y}": _counts(
                [r for r in neutral if (r["channel"], r["mapping_id"], r["gold"]) == (c, m, y)]
            )
            for c in CHANNELS
            for m in (0, 1)
            for y in (0, 1)
        },
        "floors": {"overall": 24, "per_mapping": 12, "per_channel": 10},
        "failed_checks": [],
    }
    checks = [("overall", result["overall"], 24)]
    checks += [(f"mapping/{k}", v, 12) for k, v in result["per_mapping"].items()]
    checks += [(f"channel/{k}", v, 10) for k, v in result["per_channel"].items()]
    for name, counts, floor in checks:
        for measure in ("probe_correct", "answer_correct"):
            if counts[measure] < floor:
                result["failed_checks"].append(f"{name}/{measure}<{floor}")
    result["passed"] = not result["failed_checks"]
    return result


def _metrics(rows):
    counts = _counts(rows)
    positive = [r["probe"]["margin"] for r in rows if r["gold"] == 1]
    negative = [r["probe"]["margin"] for r in rows if r["gold"] == 0]
    auc = (
        (
            sum((a > b) + 0.5 * (a == b) for a in positive for b in negative)
            / (len(positive) * len(negative))
        )
        if positive and negative
        else None
    )
    recalls = {}
    for y in (0, 1):
        cell = [r for r in rows if r["gold"] == y]
        recalls[str(y)] = {
            "n": len(cell),
            "probe": sum(r["probe"]["prediction"] == y for r in cell) / len(cell) if cell else None,
            "answer": sum(r["answer"]["correct"] for r in cell) / len(cell) if cell else None,
        }
    return {
        **counts,
        "probe_accuracy": counts["probe_correct"] / len(rows),
        "answer_accuracy": counts["answer_correct"] / len(rows),
        "invalid_ids": [r["id"] for r in rows if r["answer"]["invalid"]],
        "score_auroc": auc,
        "fixed_threshold_class_recall": recalls,
    }


def _summaries(rows):
    return {
        "overall_descriptive_conditions": _metrics(rows),
        "per_condition": {
            c: _metrics([r for r in rows if r["condition"] == c])
            for c in CONDITIONS
            if any(r["condition"] == c for r in rows)
        },
        "per_channel": {c: _metrics([r for r in rows if r["channel"] == c]) for c in CHANNELS},
        "per_mapping": {
            str(m): _metrics([r for r in rows if r["mapping_id"] == m]) for m in (0, 1)
        },
        "per_condition_channel_mapping": {
            f"{c}/{channel}/map{m}": _metrics(
                [
                    r
                    for r in rows
                    if (r["condition"], r["channel"], r["mapping_id"]) == (c, channel, m)
                ]
            )
            for c in CONDITIONS
            for channel in CHANNELS
            for m in (0, 1)
            if any((r["condition"], r["channel"], r["mapping_id"]) == (c, channel, m) for r in rows)
        },
    }


def _contrast_summary(scenes, field):
    return {
        "n_scenes": len(scenes),
        "mean": float(np.mean([s[field] for s in scenes])) if scenes else None,
        "channel_means": {
            channel: float(np.mean([s[field] for s in scenes if s["channel"] == channel]))
            if any(s["channel"] == channel for s in scenes)
            else None
            for channel in CHANNELS
        },
    }


def _uncertainty(scenes, field):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.zeros(BOOTSTRAP_DRAWS)
    groups = sorted({s["wording_group"] for s in scenes})
    for channel in CHANNELS:
        channel_groups = [
            g
            for g in groups
            if any(s["wording_group"] == g and s["channel"] == channel for s in scenes)
        ]
        _require(len(channel_groups) == 4, "Bootstrap requires four wording groups per channel")
        means = np.array(
            [np.mean([s[field] for s in scenes if s["wording_group"] == g]) for g in channel_groups]
        )
        draws += means[rng.integers(0, 4, size=(BOOTSTRAP_DRAWS, 4))].mean(axis=1) / 2
    return {
        "seed": BOOTSTRAP_SEED,
        "draws": BOOTSTRAP_DRAWS,
        "unit": "four wording groups resampled within each channel; both scenes and mappings intact",
        "percentile_95_interval": np.quantile(draws, [0.025, 0.975]).tolist(),
        "leave_one_wording_group_out": {
            g: _contrast_summary([s for s in scenes if s["wording_group"] != g], field)
            for g in groups
        },
    }


def _specificity(scenes, indexed, condition_names, field):
    included, excluded = [], []
    for scene in scenes:
        required = [indexed[(scene["scene_id"], m, c)] for m in (0, 1) for c in condition_names]
        failed = [r["id"] for r in required if not r["answer"]["correct"]]
        if failed:
            excluded.append({"scene_id": scene["scene_id"], "incorrect_or_invalid_ids": failed})
        else:
            included.append(scene)
    return {
        **_contrast_summary(included, field),
        "required_correct_answers_per_scene": 2 * len(condition_names),
        "total_scenes": len(scenes),
        "included_scene_ids": [s["scene_id"] for s in included],
        "per_channel_n": {c: sum(s["channel"] == c for s in included) for c in CHANNELS},
        "excluded": excluded,
        "interpretation": "selected descriptive subset; not an unbiased population effect",
    }


def _transitions(rows, indexed):
    report = {}
    comparisons = [
        (f"{stem}_a", f"{stem}_b") for stem in ("own_token", "archive_token", "own_disk")
    ]
    comparisons += [("neutral", c) for c in CONDITIONS if c != "neutral"]
    for left, right in comparisons:
        records = []
        for row in rows:
            if row["condition"] != left:
                continue
            other = indexed[(row["scene_id"], row["mapping_id"], right)]

            def answer(r):
                return (
                    "INVALID"
                    if r["answer"]["invalid"]
                    else "YES"
                    if r["answer"]["prediction"]
                    else "NO"
                )

            records.append(
                {
                    "scene_id": row["scene_id"],
                    "mapping_id": row["mapping_id"],
                    "from": answer(row),
                    "to": answer(other),
                }
            )
        report[f"{left}->{right}"] = {
            "n": len(records),
            "counts": dict(sorted(Counter(f"{r['from']}->{r['to']}" for r in records).items())),
            "records": records,
        }
    return report


def _raw_scores(rows):
    return [
        {
            **{
                k: r[k]
                for k in (
                    "id",
                    "scene_id",
                    "channel",
                    "wording_group",
                    "mapping_id",
                    "condition",
                    "gold",
                    "probe",
                    "answer",
                )
            },
            "answer_text": r["generation"]["text"],
            "finish_reason": r["generation"]["finish_reason"],
        }
        for r in rows
    ]


def analyse_rows(rows, *, bank=None):
    """Analyse complete saved results, returning no non-neutral estimates after failure."""
    bank = _bank(bank)
    rows = _validated_rows(rows, bank)
    qualification = qualify_neutral(rows, bank=bank)
    neutral = [r for r in rows if r["condition"] == "neutral"]
    report = {
        "schema": "future_goal_content_controls_analysis_v1",
        "bank_sha256": BANK_SHA256,
        "qualification": qualification,
        "n_saved_conditions": len(rows),
        "neutral_metrics": _summaries(neutral),
        "neutral_raw_scores": _raw_scores(neutral),
        "limits": [
            "Authored scene units, not 224 independent examples; no population significance claim.",
            "Probe probabilities are not validated confidence estimates.",
            "Archive control tests this operative framing; it does not establish an internal mechanism.",
        ],
    }
    if not qualification["passed"]:
        report["status"] = "qualification_failed"
        report["nonneutral_conditions_present_after_failed_gate"] = len(rows) - 32
        return report
    expected = {r["id"] for r in bank["planned_conditions"]}
    if {r["id"] for r in rows} != expected:
        report["status"] = "qualified_neutral_only" if len(rows) == 32 else "incomplete_non_neutral"
        report["missing_condition_ids"] = sorted(expected - {r["id"] for r in rows})
        return report
    indexed = {(r["scene_id"], r["mapping_id"], r["condition"]): r for r in rows}
    scenes = []
    for source in bank["scenes"]:
        mappings = []
        for mapping in (0, 1):
            scores = {c: indexed[(source["id"], mapping, c)]["probe"]["margin"] for c in CONDITIONS}
            own = scores["own_token_b"] - scores["own_token_a"]
            archive = scores["archive_token_b"] - scores["archive_token_a"]
            disk = scores["own_disk_b"] - scores["own_disk_a"]
            mappings.append(
                {
                    "mapping_id": mapping,
                    "own_token_b_minus_a": own,
                    "archive_token_b_minus_a": archive,
                    "own_disk_b_minus_a": disk,
                    "primary": own - archive,
                    "secondary": own - disk,
                }
            )
        scenes.append(
            {
                "scene_id": source["id"],
                "channel": source["channel"],
                "wording_group": source["wording_group"],
                "gold": source["gold"],
                "mappings": mappings,
                **{f: float(np.mean([m[f] for m in mappings])) for f in ("primary", "secondary")},
            }
        )
    report["status"] = "complete"
    report["per_scene"] = scenes
    for field in ("primary", "secondary"):
        report[field] = {
            **_contrast_summary(scenes, field),
            **_uncertainty(scenes, field),
            "per_wording_group": {
                g: _contrast_summary([s for s in scenes if s["wording_group"] == g], field)
                for g in sorted({s["wording_group"] for s in scenes})
            },
        }
    own = ("own_token_a", "own_token_b")
    report["specificity"] = {
        "primary": _specificity(
            scenes, indexed, (*own, "archive_token_a", "archive_token_b"), "primary"
        ),
        "secondary": _specificity(scenes, indexed, (*own, "own_disk_a", "own_disk_b"), "secondary"),
        "joint_primary": _specificity(scenes, indexed, CONDITIONS[1:], "primary"),
        "joint_secondary": _specificity(scenes, indexed, CONDITIONS[1:], "secondary"),
    }
    primary, subset = report["primary"], report["specificity"]["primary"]
    report["larger_study_descriptive_criteria"] = {
        "positive_in_both_channels": all(v > 0 for v in primary["channel_means"].values()),
        "primary_interval_above_zero": primary["percentile_95_interval"][0] > 0,
        "no_leave_group_sign_reversal": all(
            v["mean"] > 0 for v in primary["leave_one_wording_group_out"].values()
        ),
        "behavior_correct_subset": subset["n_scenes"] >= 8
        and subset["mean"] > 0
        and all(n >= 3 for n in subset["per_channel_n"].values()),
        "human_validation_and_runtime_binding_also_required": True,
    }
    report["metrics"] = _summaries(rows)
    report["answer_transitions"] = _transitions(rows, indexed)
    report["raw_scores"] = _raw_scores(rows)
    return report


def _inside(root, name):
    path = (root / name).resolve()
    _require(path.is_relative_to(root.resolve()), "Artifact path escapes its bound directory")
    return path


def _saved_tokens(ids, message):
    _require(isinstance(ids, list) and ids and all(type(i) is int and i >= 0 for i in ids), message)


def _margin(vector, metadata, parameters):
    from .belief_query_readout import _row_normalize

    head = metadata["heads"][HEAD]
    _require(head["layer"] == "layer_16", "Frozen head layer differs")
    prefix = head["activation_transform"]
    x = np.asarray(vector, dtype=np.float64).reshape(1, -1)
    _require(
        x.shape == (1, metadata["feature_widths"]["layer_16"]) and np.isfinite(x).all(),
        "Saved feature width/values differ",
    )
    z = _row_normalize((x - parameters[prefix + "__mean"]) / parameters[prefix + "__scale"])
    return float(
        (z @ parameters[head["parameter_prefix"] + "__coef"].T)[0, 0]
        + parameters[head["parameter_prefix"] + "__intercept"][0]
    )


def _generation_binding(answer, ids, protocol, *, tokenizer=None):
    _require(answer["prompt_token_ids"] == ids, "Answer prefix differs from captured prefix")
    generated = answer["generated_token_ids"]
    _saved_tokens(generated, "Invalid saved output tokens")
    _require(len(generated) <= 12, "Generated token count differs")
    eos = protocol["generation"]["eos_token_ids"]
    finish = "eos" if generated[-1] in eos else "length"
    _require(
        answer["finish_reason"] == finish and (finish == "eos" or len(generated) == 12),
        "EOS/length disposition differs",
    )
    special = protocol["tokenizer"]["all_special_ids"]
    unexpected = [
        t
        for i, t in enumerate(generated)
        if t in special and not (i == len(generated) - 1 and t in eos)
    ]
    _require(
        answer.get("unexpected_special_tokens") == unexpected,
        "Saved special-token disposition differs",
    )
    _require(
        isinstance(answer.get("raw_decoded_text"), str)
        and canonical_hash(answer["raw_decoded_text"]) == answer["raw_text_sha256"],
        "Raw decoded text hash differs",
    )
    if tokenizer is not None:
        _require(
            tokenizer.decode(
                generated, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            == answer["raw_decoded_text"]
            and tokenizer.decode(generated, skip_special_tokens=True).strip() == answer["text"],
            "Saved answer tokens/text differ",
        )


def _terminal_study(manifest):
    _require(
        manifest.get("status") in {"complete", "neutral_failed_stopped"},
        "Completed-study analysis requires a successful terminal study status; "
        "incomplete attempts are preserved unscored",
    )


def analyse_run(output, *, verify_tokenizer=True, tokenizer=None):
    """Verify persisted evidence on CPU, then run the fixed descriptive analysis.

    With ``verify_tokenizer=False``, saved text hashes and prefix identities are
    checked, but tokenization and decoding are explicitly not independently proved.
    """
    from .belief_query_readout import _load_fit, _snapshot

    output = Path(output).resolve()
    protocol = read_json(output / "protocol.json")
    manifest = read_json(output / "manifest.json")
    _terminal_study(manifest)
    digest = file_sha256(output / "protocol.json")
    _require(
        protocol.get("schema") == "future_goal_controls_protocol_v1"
        and manifest.get("schema") == "future_goal_controls_run_v1",
        "Wrong run schema",
    )
    _require(manifest.get("protocol_sha256") == digest, "Protocol hash differs")
    _require(
        protocol.get("bank_sha256") == manifest.get("bank_sha256") == BANK_SHA256,
        "Run bank binding differs",
    )
    bank = reviewed_bank(protocol["bank_path"])
    _require(
        protocol.get("primary_head") == HEAD
        and protocol.get("threshold") == 0.5
        and protocol.get("threshold_tie") == 0
        and protocol.get("layers") == [16]
        and protocol.get("hf_hidden_state_index") == 17
        and protocol.get("no_refit") is True
        and protocol.get("no_hyperparameter_selection") is True,
        "Frozen readout contract differs",
    )
    generation = protocol["generation"]
    _require(
        generation["settings"]["max_new_tokens"] == 12
        and generation["settings"]["do_sample"] is False
        and generation["seed"] == BOOTSTRAP_SEED,
        "Generation policy differs",
    )
    source_backend = protocol["source_backend"]
    _require(
        protocol["tokenizer"]["revision"]
        == source_backend["resolved_revision"]
        == protocol["model_settings"]["revision"]
        and len(source_backend["resolved_revision"]) == 40,
        "Model/tokenizer revision binding differs",
    )
    eos = generation["eos_token_ids"]
    _saved_tokens(eos, "Missing frozen EOS IDs")
    special = protocol["tokenizer"]["all_special_ids"]
    _saved_tokens(special, "Missing frozen special IDs")
    _require(set(eos) <= set(special), "EOS not bound to special-token metadata")
    source_eos = source_backend["generation_defaults"].get("eos_token_id")
    if source_eos is None:
        source_eos = protocol["tokenizer"]["special_token_ids"]["eos"]
    _require(
        eos == (source_eos if isinstance(source_eos, list) else [source_eos]),
        "EOS differs from frozen source backend",
    )
    _require(
        file_sha256(protocol["config_path"]) == protocol["config_sha256"], "Config hash differs"
    )
    _require(read_json(protocol["config_path"]) == protocol["config"], "Frozen config differs")
    _require(
        {
            "scripts/run_future_goal_controls.py",
            "scripts/analyse_future_goal_controls.py",
            "src/latent_probes/future_goal_controls.py",
            "src/latent_probes/future_goal_controls_analysis.py",
            "src/latent_probes/belief_query_readout.py",
            "src/latent_probes/storage.py",
        }
        <= set(protocol["policy_sha256"]),
        "Incomplete frozen policy roster",
    )
    for name, expected_hash in protocol["policy_sha256"].items():
        _require(
            file_sha256(_inside(ROOT, name)) == expected_hash, "Frozen policy differs: " + name
        )
    for directory, hashes in (
        ("source_run", "source_artifact_sha256"),
        ("known_run", "known_artifact_sha256"),
        ("fit_dir", "fitted_artifact_sha256"),
    ):
        _require(
            _snapshot(Path(protocol[directory])) == protocol[hashes],
            "Frozen source artifact set/hash differs: " + directory,
        )
    for directory in ("source_run", "known_run"):
        source_manifest = read_json(Path(protocol[directory]) / "manifest.json")
        _require(source_manifest["status"] == "complete", "Incomplete frozen source run")
        if directory == "source_run":
            _require(source_manifest["backend"] == source_backend, "Source backend binding differs")
    bound_outputs = manifest.get("output_sha256")
    _require(isinstance(bound_outputs, dict), "Missing finalized output hash ledger")
    for name, expected_hash in bound_outputs.items():
        _require(
            file_sha256(_inside(output, name)) == expected_hash, "Output hash differs: " + name
        )
    actual = _snapshot(output)
    _require(set(actual) == set(bound_outputs) | {"manifest.json"}, "Unbound output files present")
    planned = protocol["planned_rows"]
    expected = {r["id"]: r for r in bank["planned_conditions"]}
    _require(
        len(planned) == 224
        and len({r["id"] for r in planned}) == 224
        and {r["id"] for r in planned} == set(expected),
        "Frozen plan roster differs",
    )
    if verify_tokenizer and tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            protocol["tokenizer"]["model_id"],
            revision=protocol["tokenizer"]["revision"],
            local_files_only=True,
            trust_remote_code=False,
        )
    if verify_tokenizer:
        _require(
            canonical_hash(tokenizer.chat_template) == protocol["tokenizer"]["chat_template_sha256"]
            and list(tokenizer.all_special_ids) == special,
            "Tokenizer identity differs",
        )
    backend = manifest["backend"]
    for key in (
        "backend",
        "is_test_fixture",
        "model_id",
        "resolved_revision",
        "model_type",
        "dtype",
        "quantization",
        "decoder_blocks",
        "hidden_size",
        "capture_layers",
        "chat_template_sha256",
        "generation_defaults",
        "special_token_ids",
        "attention_implementation",
    ):
        expected_value = [16] if key == "capture_layers" else protocol["source_backend"].get(key)
        _require(backend.get(key) == expected_value, "Runtime/source backend differs: " + key)
    layer_check = manifest["layer_index_check"]
    _require(
        layer_check.get("status") == "passed"
        and layer_check["checks"]["16"] == {"hf_hidden_state_index": 17, "exact": True},
        "Runtime layer-index check differs",
    )
    lookup = {}
    for row in planned:
        _require(
            all(row.get(k) == v for k, v in expected[row["id"]].items()),
            "Planned condition differs from bank",
        )
        ids = row["prompt_token_ids"]
        _saved_tokens(ids, "Invalid planned prefix tokens")
        _require(
            row["messages_sha256"] == canonical_hash(row["messages"])
            and row["prompt_token_ids_sha256"] == canonical_hash(ids),
            "Planned hashes differ",
        )
        _require(
            len(ids) <= protocol["model_settings"]["max_input_tokens"],
            "Planned prefix exceeds frozen input limit",
        )
        if verify_tokenizer:
            _require(
                tokenizer.apply_chat_template(
                    row["messages"], tokenize=True, add_generation_prompt=True
                )
                == ids,
                "Planned tokens differ from pinned tokenizer",
            )
        lookup[row["id"]] = row
    metadata, parameters, _texts, fitted_hashes = _load_fit(Path(protocol["fit_dir"]))
    _require(fitted_hashes == protocol["fitted_artifact_sha256"], "Loaded fitted source differs")
    reference = protocol["reference"]
    check = read_json(output / "reference_check.json")
    _require(
        all(check.get(k) is True for k in ("ids_exact", "feature_exact", "score_exact")),
        "Source reference runtime check failed",
    )
    _require(
        canonical_hash(reference["feature"])
        == reference["feature_values_sha256"]
        == check["feature_values_sha256"],
        "Reference feature identity differs",
    )
    with np.load(output / "reference_feature.npz", allow_pickle=False) as archive:
        reference_vector = archive["layer_16"]
    _require(
        np.array_equal(reference_vector, np.asarray(reference["feature"], dtype=np.float32)),
        "Reference feature differs",
    )
    _require(
        math.isclose(
            _margin(reference_vector, metadata, parameters),
            reference["probe"]["margin"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        "Reference frozen score differs",
    )
    rows = []
    for path in sorted((output / "row_outputs").glob("*.json")):
        row = read_json(path)
        _require(row.get("status") == "complete", "Incomplete saved condition")
        _require(
            row.get("id") in lookup and path.stem == row["id"], "Saved row filename/ID differs"
        )
        plan = lookup[row["id"]]
        _require(
            all(row.get(k) == v for k, v in plan.items()) and row.get("protocol_sha256") == digest,
            "Saved row/plan binding differs",
        )
        ids, capture = plan["prompt_token_ids"], row["capture"]
        _require(
            capture["prompt_token_ids_sha256"] == canonical_hash(ids)
            and capture["selected_position"] == len(ids) - 1
            and capture["selected_token_id"] == ids[-1]
            and capture["layer"] == 16
            and capture["hf_hidden_state_index"] == 17
            and capture["generated_answer_tokens_supplied"] == 0,
            "Capture convention differs",
        )
        feature_path = _inside(output, capture["feature_path"])
        _require(file_sha256(feature_path) == capture["feature_sha256"], "Feature hash differs")
        with np.load(feature_path, allow_pickle=False) as archive:
            _require(archive.files == ["layer_16"], "Saved feature layers differ")
            vector = archive["layer_16"].copy()
        _require(vector.ndim == 1 and vector.dtype.kind == "f", "Invalid saved feature array")
        _require(
            math.isclose(
                _margin(vector, metadata, parameters),
                row["probe"]["margin"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ),
            "Saved frozen margin differs",
        )
        _generation_binding(
            row["generation"], ids, protocol, tokenizer=tokenizer if verify_tokenizer else None
        )
        rows.append(row)
    _require(
        len(rows) == manifest["n_completed_rows"] == manifest["generated_answers"],
        "Manifest row/answer counts differ",
    )
    result = analyse_rows(rows, bank=bank)
    saved_qualification = read_json(output / "qualification.json")
    for field in ("passed", "overall", "per_mapping", "per_channel", "crossed"):
        _require(
            saved_qualification[field] == result["qualification"][field],
            "Saved neutral qualification differs: " + field,
        )
    _require(
        manifest.get("neutral_qualification_passed") == result["qualification"]["passed"],
        "Manifest qualification differs",
    )
    if manifest["status"] == "complete":
        _require(result["status"] == "complete", "Complete manifest has incomplete evidence")
    if manifest["status"] == "neutral_failed_stopped":
        _require(
            result["status"] == "qualification_failed" and len(rows) == 32,
            "Failed-gate run did not stop at 32",
        )
    result["evidence_verification"] = {
        "run_directory": str(output),
        "protocol_sha256": digest,
        "manifest_status": manifest["status"],
        "source_artifact_hashes_verified": True,
        "saved_feature_margins_recomputed": True,
        "all_224_planned_prefixes_bound": True,
        "pinned_tokenization_and_decoding_repeated": verify_tokenizer,
        "human_review_complete": protocol["human_review_complete"],
        "model_calls": 0,
        "fits": 0,
    }
    if not verify_tokenizer:
        result["limits"].append(
            "Tokenization/decoding not independently repeated; saved bindings only."
        )
    return result
