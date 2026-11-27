"""Audit cached prose-monitor generations and metrics without project imports."""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value, newline=False):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256((text + ("\n" if newline else "")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def native(path):
    if sys.platform == "win32" and str(path).startswith("/mnt/"):
        parts = str(path).split("/", 3)
        return Path(parts[2].upper() + ":/" + parts[3])
    if sys.platform == "win32" and str(path).startswith("/"):
        return Path("//wsl.localhost/Ubuntu" + str(path))
    return Path(path)


def unique_object(pairs):
    keys = [key for key, _ in pairs]
    require(len(keys) == len(set(keys)), "Duplicate JSON key")
    return dict(pairs)


def reject_constant(value):
    raise ValueError("Nonfinite JSON constant: " + value)


def parse_and_fold(text, messages, finish_reason="eos"):
    """Apply the specified strict grammar, using prose only and no gold/state."""
    invalid = {
        "valid": False,
        "prediction": None,
        "score": None,
        "extraction": None,
        "final_location": None,
        "invalid_reason": None,
    }
    try:
        require(isinstance(text, str), "Generated output must be text")
        require(finish_reason == "eos", "Generation did not finish at EOS")
        value = json.loads(
            text.strip(), object_pairs_hook=unique_object, parse_constant=reject_constant
        )
        require(
            isinstance(value, dict)
            and set(value) == {"initial_location", "updates", "queried_location"},
            "Extraction must have exactly the three required keys",
        )
        updates = value["updates"]
        require(
            isinstance(updates, list) and len(updates) in {1, 2},
            "Need one or two ordered candidate updates",
        )
        locations = [value["initial_location"], value["queried_location"]]
        for update in updates:
            require(
                isinstance(update, dict)
                and set(update) == {"location", "accepted_access"}
                and type(update["accepted_access"]) is bool,
                "Update requires exactly location and boolean accepted_access",
            )
            locations.append(update["location"])
        prose = "\n".join(message["content"] for message in messages)
        for location in locations:
            require(
                isinstance(location, str)
                and 0 < len(location) <= 160
                and location == location.strip()
                and all(ord(char) >= 32 for char in location)
                and re.search(r"(?<!\w)" + re.escape(location) + r"(?!\w)", prose) is not None,
                "Location must occur verbatim in source prose",
            )
        require(len(set(locations)) <= 2, "Binary study permits at most two physical names")
        accepted = [u["location"] for u in updates if u["accepted_access"]]
        final = accepted[-1] if accepted else value["initial_location"]
        prediction = int(final == value["queried_location"])
        return {
            "valid": True,
            "prediction": prediction,
            "score": float(prediction),
            "extraction": value,
            "final_location": final,
            "invalid_reason": None,
        }
    except (ValueError, TypeError, RecursionError) as error:
        return {**invalid, "invalid_reason": str(error)}


def belief(state):
    before = state["initial"]
    if state["mechanism"] == "correction":
        before = state["update"] if state["first_receipt"] else before
        final = 1 - state["update"]
    else:
        final = state["update"]
    return final if state["receipt"] else before


def mean(values):
    return sum(values) / len(values)


def binary(rows, records):
    selected = [
        (belief(r["state"]), records[r["case_id"]]["prediction"])
        for r in rows
        if records[r["case_id"]]["valid"]
    ]
    if not selected:
        return None
    labels, scores = zip(*selected)
    correct = sum(a == b for a, b in selected)
    recall = {
        value: mean([b == value for a, b in selected if a == value]) if value in labels else None
        for value in (0, 1)
    }
    positive = [b for a, b in selected if a == 1]
    negative = [b for a, b in selected if a == 0]
    auc = (
        mean([float(a > b) + 0.5 * (a == b) for a in positive for b in negative])
        if positive and negative
        else None
    )
    return {
        "rows": len(selected),
        "correct": correct,
        "accuracy": correct / len(selected),
        "recall_0": recall[0],
        "recall_1": recall[1],
        "auroc": auc,
        "brier": mean([(a - b) ** 2 for a, b in selected]),
    }


def group_metrics(rows, records):
    group = rows[0]["group_id"]
    mechanism = rows[0]["mechanism"]
    require(len(rows) == (64 if mechanism == "correction" else 32), "Incomplete scenario")
    states = {}
    for row in rows:
        state = row["state"]
        key = tuple(
            state[k] for k in ("initial", "update", "receipt", "world", "name_map", "first_receipt")
        )
        require(key not in states, "Duplicate factorial state")
        require(row["gold"] == belief(state), "Gold differs from independent update formula")
        states[key] = row["case_id"]
    valid = {r["case_id"]: records[r["case_id"]]["valid"] for r in rows}
    correct = {
        r["case_id"]: valid[r["case_id"]]
        and records[r["case_id"]]["prediction"] == belief(r["state"])
        for r in rows
    }
    quartets = []
    for update, world, names in itertools.product((0, 1), repeat=3):
        if mechanism == "correction":
            keys = [
                (1 - update, update, receipt, world, names, first)
                for first in (1, 0)
                for receipt in (0, 1)
            ]
        else:
            keys = [
                (initial, update, receipt, world, names, None)
                for initial in (1 - update, update)
                for receipt in (0, 1)
            ]
        ids = [states[key] for key in keys]
        quartets.append(
            {
                "quartet_id": f"{group}__u{update}_w{world}_map{names}",
                "case_ids": ids,
                "all_four_correct": all(correct[c] for c in ids),
            }
        )
    pairs = []
    buckets = collections.defaultdict(dict)
    for key, case in states.items():
        buckets[key[:2] + key[3:]][key[2]] = case
    by_id = {r["case_id"]: r for r in rows}
    for bucket in buckets.values():
        require(set(bucket) == {0, 1}, "Incomplete receipt pair")
        ids = [bucket[0], bucket[1]]
        both_valid = all(valid[c] for c in ids)
        pairs.append(
            {
                "case_ids": ids,
                "changes_if_received": belief(by_id[ids[0]]["state"])
                != belief(by_id[ids[1]]["state"]),
                "both_valid": both_valid,
                "both_correct": all(correct[c] for c in ids),
                "verdict_flip": records[ids[0]]["prediction"] != records[ids[1]]["prediction"]
                if both_valid
                else None,
            }
        )
    result = {
        "group_id": group,
        "mechanism": mechanism,
        "rows": len(rows),
        "valid": sum(valid.values()),
        "invalid": len(rows) - sum(valid.values()),
        "correct": sum(correct.values()),
        "accuracy": sum(correct.values()) / len(rows),
        "complete_quartet_accuracy": mean([q["all_four_correct"] for q in quartets]),
        "quartets": quartets,
        "pairs": pairs,
        "valid_only_binary_metrics": binary(rows, records),
        "invalid_reasons": dict(
            collections.Counter(
                str(records[r["case_id"]]["invalid_reason"])
                for r in rows
                if not valid[r["case_id"]]
            )
        ),
    }
    for changing, name in ((True, "changing"), (False, "preserving")):
        selected = [p for p in pairs if p["changes_if_received"] == changing]
        valid_pairs = [p for p in selected if p["both_valid"]]
        result[name] = {
            "pairs": len(selected),
            "both_valid_pairs": len(valid_pairs),
            "both_correct_fraction": mean([p["both_correct"] for p in selected]),
            "verdict_flip_fraction_of_valid_pairs": mean([p["verdict_flip"] for p in valid_pairs])
            if valid_pairs
            else None,
        }
    if mechanism == "correction":
        secondary = [r["case_id"] for r in rows if r["state"]["initial"] == r["state"]["update"]]
        result["secondary_correction_cells"] = {
            "rows": len(secondary),
            "invalid": sum(not valid[c] for c in secondary),
            "correct": sum(correct[c] for c in secondary),
            "accuracy": mean([correct[c] for c in secondary]),
        }
    return result


def compare(expected, actual, path="metrics"):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(expected) == set(actual), path + " keys differ")
        for key in expected:
            compare(expected[key], actual[key], path + "." + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(expected) == len(actual), path + " length differs")
        for i, value in enumerate(expected):
            compare(value, actual[i], path + f"[{i}]")
    elif isinstance(expected, float):
        require(
            type(actual) in {int, float} and abs(expected - actual) <= 1e-12,
            path + " value differs",
        )
    else:
        require(type(expected) is type(actual) and expected == actual, path + " value differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "analysis", "gold-rows", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--partition", choices=("development", "test"), default="development")
    args = parser.parse_args()
    require(not args.output.exists(), "Use a fresh audit output")
    root = args.directory
    inputs = {name: root / name for name in ("records.json", "run_manifest.json", "protocol.json")}
    hashes = {name: digest(path) for name, path in inputs.items()}
    payload, manifest, protocol = (read(inputs[name]) for name in inputs)
    analysis = read(args.analysis)
    require(analysis["source_sha256"] == hashes, "Analysis source hashes differ")
    require(
        manifest["status"] == "complete" and manifest["records_sha256"] == hashes["records.json"],
        "Incomplete or changed monitor records",
    )
    require(
        manifest["protocol_sha256"] == payload["protocol_sha256"] == hashes["protocol.json"],
        "Protocol binding differs",
    )
    require(
        payload["partition"] == manifest["partition"] == analysis["partition"] == args.partition,
        "Partition differs",
    )
    require(
        payload["test_fixture"] is False and analysis["test_fixture"] is False,
        "Fixture is not empirical evidence",
    )
    require(
        analysis["scientific_result"] is True
        and analysis["real_empirical_result"] is True
        and not analysis["is_symbolic_oracle"],
        "Empirical method markers differ",
    )
    for value in (payload, analysis, manifest["authorization"]):
        require(
            value["research_status"] == "agent_audited_exploratory"
            and value["human_review_complete"] is False
            and value["confirmatory_result"] is False,
            "Exploratory provenance differs",
        )
    require(
        payload["model_settings"] == protocol["model_settings"] == analysis["model_settings"],
        "Model settings differ",
    )
    require(
        payload["monitor_protocol_sha256"]
        == protocol["monitor_protocol_sha256"]
        == digest(native(protocol["monitor_protocol_path"])),
        "Extraction policy differs",
    )
    require(digest(native(protocol["bank_path"])) == protocol["bank_file_sha256"], "Bank changed")
    for bindings in (
        protocol["policy_sha256"],
        manifest["source_bindings"],
        manifest["authorization"]["source_bindings"],
    ):
        for path, expected in bindings.items():
            require(digest(native(path)) == expected, "Frozen source changed: " + path)
    gold_payload = read(args.gold_rows)
    require(
        gold_payload["provenance"]["bank_sha256"]
        == payload["bank_sha256"]
        == protocol["bank_sha256"]
        == analysis["bank_sha256"],
        "Gold source bank differs",
    )
    gold = {r["case_id"]: r for r in gold_payload["rows"] if r["split"] == args.partition}
    planned = [r for r in protocol["planned_rows"] if r["split"] == args.partition]
    records = payload["records"]
    ids = [r["case_id"] for r in records]
    require(
        ids == [r["case_id"] for r in planned]
        and len(ids) == len(set(ids)) == len(gold) == manifest["completed_rows"],
        "Record inventory differs",
    )
    require(set(ids) == set(gold), "Gold/monitor case inventory differs")
    require(
        {p.name for p in (root / "row_journal").iterdir()} == set(ids), "Journal inventory differs"
    )
    parsed_records = {}
    inventory = []
    extraction_lengths = collections.Counter()
    for plan, record in zip(planned, records):
        case_id = record["case_id"]
        source = gold[case_id]
        require(
            record["messages"] == source["messages"] == plan["messages"]
            and canonical(source["messages"], newline=True) == record["prompt_sha256"],
            "Source prose differs",
        )
        require(all(record[k] == v for k, v in plan.items()), "Prepared monitor input differs")
        require(
            canonical(plan["monitor_messages"]) == plan["monitor_messages_sha256"]
            and canonical(plan["prompt_token_ids"]) == plan["prompt_token_ids_sha256"],
            "Prepared hashes differ",
        )
        item = root / "row_journal" / case_id
        require(
            {p.name for p in item.iterdir()}
            == {"started.json", "raw_generation.json", "complete.json"},
            "Incomplete/failed journal",
        )
        complete, started, raw = (
            read(item / name) for name in ("complete.json", "started.json", "raw_generation.json")
        )
        require(
            complete == record
            and record["status"] == "complete"
            and record["protocol_sha256"] == hashes["protocol.json"],
            "Journal/export metadata differs",
        )
        require(
            started["case_id"] == case_id and started["protocol_sha256"] == hashes["protocol.json"],
            "Started metadata differs",
        )
        require(
            digest(item / "raw_generation.json") == record["raw_generation_sha256"],
            "Raw generation bytes changed",
        )
        generation = raw["generation"]
        require(
            raw["raw_response"] == record["raw_response"] == generation["text"],
            "Cached generation text differs",
        )
        require(
            generation["prompt_token_ids"] == plan["prompt_token_ids"]
            and 0 < len(generation["generated_token_ids"]) <= 384,
            "Generated token metadata differs",
        )
        parsed = parse_and_fold(
            raw["raw_response"], source["messages"], generation["finish_reason"]
        )
        compare(parsed, {k: record[k] for k in parsed}, case_id)
        parsed_records[case_id] = parsed
        if parsed["valid"]:
            extraction_lengths[len(parsed["extraction"]["updates"])] += 1
        inventory.append(
            {
                "case_id": case_id,
                "raw_generation_sha256": record["raw_generation_sha256"],
                "complete_sha256": digest(item / "complete.json"),
                "started_sha256": digest(item / "started.json"),
                "raw_text_sha256": hashlib.sha256(raw["raw_response"].encode()).hexdigest(),
            }
        )
    grouped = collections.defaultdict(list)
    for case_id in ids:
        grouped[gold[case_id]["group_id"]].append(gold[case_id])
    reconstructed = [group_metrics(grouped[g], parsed_records) for g in sorted(grouped)]
    compare(reconstructed, analysis["groups"])
    total = sum(g["rows"] for g in reconstructed)
    overall = {
        "rows": total,
        "valid": sum(g["valid"] for g in reconstructed),
        "invalid": sum(g["invalid"] for g in reconstructed),
        "correct": sum(g["correct"] for g in reconstructed),
        "accuracy": sum(g["correct"] for g in reconstructed) / total,
        "macro_scenario_accuracy": mean([g["accuracy"] for g in reconstructed]),
    }
    compare(overall, analysis["overall"])
    mechanisms = {}
    for mechanism in sorted({g["mechanism"] for g in reconstructed}):
        subset = [g for g in reconstructed if g["mechanism"] == mechanism]
        mechanisms[mechanism] = {
            "scenario_groups": len(subset),
            "macro_complete_quartet_accuracy": mean(
                [g["complete_quartet_accuracy"] for g in subset]
            ),
            "macro_accuracy": mean([g["accuracy"] for g in subset]),
            "invalid": sum(g["invalid"] for g in subset),
        }
    compare(mechanisms, analysis["by_mechanism"])
    require(
        all(digest(path) == hashes[name] for name, path in inputs.items()),
        "Inputs changed during audit",
    )
    report = {
        "schema": "main-study-monitor-independent-audit-v1",
        "reviewer_type": "agent",
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "status": "passed",
        "partition": args.partition,
        "research_status": "agent_audited_exploratory",
        "human_review_complete": False,
        "confirmatory_result": False,
        "source_sha256": hashes,
        "analysis_sha256": digest(args.analysis),
        "gold_rows_source_sha256": digest(args.gold_rows),
        "auditor_source_sha256": digest(__file__),
        "coverage": {
            "raw_generation_texts_and_hashes_checked": len(ids),
            "strict_parses_and_folds_checked": len(ids),
            "unique_cases": len(ids),
            "scenario_groups": len(grouped),
            "quartets_checked": sum(len(g["quartets"]) for g in reconstructed),
            "receipt_pairs_checked": sum(len(g["pairs"]) for g in reconstructed),
            "all_group_overall_mechanism_metrics_match": True,
            "all_source_prefix_and_journal_export_bindings_match": True,
        },
        "independent_overall": overall,
        "independent_by_mechanism": mechanisms,
        "strict_quartets_correct": sum(
            q["all_four_correct"] for g in reconstructed for q in g["quartets"]
        ),
        "extracted_update_count_distribution": dict(extraction_lengths),
        "raw_journal_inventory_sha256": canonical(inventory),
        "new_model_calls": 0,
        "new_fits_or_selections": 0,
        "project_modules_imported": any(n.startswith("latent_probes") for n in sys.modules),
        "test_result_files_accessed": args.partition == "test",
        "limitations": [
            "Strict validity is syntactic/verbatim-name validity, not proof the extracted events or access beliefs are semantically correct.",
            "The parsed hard score is a symbolic fold of model-extracted prose; this method is neither an oracle nor a probability estimate.",
            "Raw cached generation texts, token metadata and hashes were verified; generated token IDs were not independently decoded with a tokenizer.",
            "Only the selected monitor partition was evaluated. Development observations do not authorize prompt/parser/model/code changes to the already fixed test.",
            "Scenario groups, not factorial rows, are the distinct authored narrative units; no confirmatory or human-review status is created.",
        ],
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": digest(args.output),
                "overall": overall,
                "strict_quartets_correct": report["strict_quartets_correct"],
            }
        )
    )


if __name__ == "__main__":
    main()
