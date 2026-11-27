"""Full-denominator evaluation of the prose monitor, including invalid outputs.

Nullable predictions are never filled using gold. Invalids count incorrect for
accuracy, pairs and quartets. Ranking/calibration metrics explicitly use only
valid outputs and retain their smaller denominator.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .main_study_analysis import binary_metrics
from .main_study_bank import bank_digest, iter_rows, load_bank, prompt_digest
from .main_study_design import BeliefCase, primary_quartets
from .main_study_monitor import parse_monitor_output
from .storage import file_sha256, read_json


def monitor_group_metrics(rows: list[dict], records: list[dict]) -> dict:
    if not rows or len({r["group_id"] for r in rows}) != 1:
        raise ValueError("Need one complete scenario group")
    by_id = {r["case_id"]: r for r in records}
    if len(by_id) != len(records) or set(by_id) != {r["case_id"] for r in rows}:
        raise ValueError("Monitor predictions must exactly cover the scenario")
    cases = {r["case_id"]: BeliefCase(**r["state"]) for r in rows}
    mechanism = rows[0]["mechanism"]
    if len(rows) != (64 if mechanism == "correction" else 32):
        raise ValueError("Missing primary/secondary scenario cells")
    quartets = primary_quartets(tuple(cases.values()))
    if len(quartets) != 8:
        raise ValueError("Need all eight primary quartets")
    validity, correctness = {}, {}
    valid_labels, valid_scores = [], []
    reasons = Counter()
    for row in rows:
        record = by_id[row["case_id"]]
        valid = record.get("valid")
        if type(valid) is not bool:
            raise ValueError("Monitor validity must be boolean")
        prediction, score = record.get("prediction"), record.get("score")
        if valid:
            if (
                type(prediction) is not int
                or prediction not in (0, 1)
                or score != float(prediction)
            ):
                raise ValueError("The fixed monitor exports hard binary scores only")
            valid_labels.append(row["gold"])
            valid_scores.append(score)
        elif prediction is not None or score is not None:
            raise ValueError("Invalid predictions and scores must stay null")
        else:
            reasons[str(record.get("invalid_reason"))] += 1
        validity[row["case_id"]] = valid
        correctness[row["case_id"]] = valid and prediction == row["gold"]
    quartet_rows = [
        {
            "quartet_id": q.quartet_id,
            "case_ids": [c.case_id for c in q.cases],
            "all_four_correct": all(correctness[c.case_id] for c in q.cases),
        }
        for q in quartets
    ]
    pair_groups = defaultdict(list)
    for case in cases.values():
        pair_groups[
            (case.initial, case.update, case.world, case.name_map, case.first_receipt)
        ].append(case)
    pair_rows = []
    for pair in pair_groups.values():
        pair.sort(key=lambda c: c.receipt)
        if len(pair) != 2 or [c.receipt for c in pair] != [0, 1]:
            raise ValueError("Incomplete receipt pair")
        ids = [c.case_id for c in pair]
        both_valid = all(validity[i] for i in ids)
        pair_rows.append(
            {
                "case_ids": ids,
                "changes_if_received": pair[0].changes_if_received,
                "both_valid": both_valid,
                "both_correct": all(correctness[i] for i in ids),
                "verdict_flip": (by_id[ids[0]]["prediction"] != by_id[ids[1]]["prediction"])
                if both_valid
                else None,
            }
        )
    result = {
        "group_id": rows[0]["group_id"],
        "mechanism": mechanism,
        "rows": len(rows),
        "valid": sum(validity.values()),
        "invalid": len(rows) - sum(validity.values()),
        "correct": sum(correctness.values()),
        "accuracy": sum(correctness.values()) / len(rows),
        "complete_quartet_accuracy": sum(q["all_four_correct"] for q in quartet_rows) / 8,
        "quartets": quartet_rows,
        "pairs": pair_rows,
        "valid_only_binary_metrics": binary_metrics(np.array(valid_labels), np.array(valid_scores))
        if valid_labels
        else None,
        "invalid_reasons": dict(reasons),
    }
    for name, changing in (("changing", True), ("preserving", False)):
        selected = [p for p in pair_rows if p["changes_if_received"] == changing]
        valid_pairs = [p for p in selected if p["both_valid"]]
        result[name] = {
            "pairs": len(selected),
            "both_valid_pairs": len(valid_pairs),
            "both_correct_fraction": sum(p["both_correct"] for p in selected) / len(selected),
            "verdict_flip_fraction_of_valid_pairs": sum(p["verdict_flip"] for p in valid_pairs)
            / len(valid_pairs)
            if valid_pairs
            else None,
        }
    if mechanism == "correction":
        secondary_ids = [i for i, c in cases.items() if not c.primary_eligible]
        result["secondary_correction_cells"] = {
            "rows": len(secondary_ids),
            "invalid": sum(not validity[i] for i in secondary_ids),
            "correct": sum(correctness[i] for i in secondary_ids),
            "accuracy": sum(correctness[i] for i in secondary_ids) / len(secondary_ids),
        }
    return result


def analyse_monitor_run(
    bank_path: str | Path, run_directory: str | Path, *, allow_fixture=False
) -> dict:
    """Reparse saved raw text, check provenance, and retain every planned row."""
    bank = load_bank(bank_path)
    root = Path(run_directory)
    manifest = read_json(root / "run_manifest.json")
    payload = read_json(root / "records.json")
    protocol = read_json(root / "protocol.json")
    if manifest.get("status") != "complete":
        raise ValueError("Monitor run must be complete")
    if payload.get("schema") != "main-study-text-monitor-v1":
        raise ValueError("Unknown monitor record schema")
    if manifest.get("records_sha256") != file_sha256(root / "records.json"):
        raise ValueError("Monitor records hash mismatch")
    protocol_hash = file_sha256(root / "protocol.json")
    if (
        payload.get("protocol_sha256") != protocol_hash
        or manifest.get("protocol_sha256") != protocol_hash
    ):
        raise ValueError("Prepared monitor protocol hash mismatch")
    if payload.get("bank_sha256") != bank_digest(bank) or protocol.get(
        "bank_sha256"
    ) != bank_digest(bank):
        raise ValueError("Monitor and authored bank differ")
    fixture = payload.get("test_fixture")
    if type(fixture) is not bool or fixture != protocol.get("test_fixture"):
        raise ValueError("Monitor fixture provenance differs")
    if fixture and not allow_fixture:
        raise ValueError("Fixture outputs cannot be presented as scientific results")
    if payload.get("research_status") == "agent_audited_exploratory":
        authorization = manifest.get("authorization", {})
        if (
            authorization.get("research_status") != payload["research_status"]
            or payload.get("human_review_complete") is not False
            or authorization.get("human_review_complete") is not False
        ):
            raise ValueError("Monitor agent exploratory provenance differs")
    if payload.get("monitor_protocol_sha256") != protocol.get("monitor_protocol_sha256"):
        raise ValueError("Monitor extraction policy binding differs")
    if payload.get("model_settings") != protocol.get("model_settings"):
        raise ValueError("Monitor checkpoint/precision settings differ")
    split = payload.get("partition")
    if split not in {"development", "test"} or manifest.get("partition") != split:
        raise ValueError("Unknown or inconsistent monitor partition")
    rows = {r["case_id"]: r for r in iter_rows(bank, split)}
    records = payload["records"]
    by_id = {r["case_id"]: r for r in records}
    if len(by_id) != len(records) or set(by_id) != set(rows):
        raise ValueError("All planned scenario variants must be retained exactly once")
    grouped_rows, grouped_records = defaultdict(list), defaultdict(list)
    planned = {r["case_id"]: r for r in protocol["planned_rows"]}
    for case_id, row in rows.items():
        record = by_id[case_id]
        if record.get("prompt_sha256") != prompt_digest(row["messages"]):
            raise ValueError("Monitor source prompt hash differs")
        if record.get("messages") != row["messages"]:
            raise ValueError("Monitor source prose differs")
        raw_path = root / "row_journal" / case_id / "raw_generation.json"
        if record.get("raw_generation_sha256") != file_sha256(raw_path):
            raise ValueError("Raw monitor generation hash differs")
        raw = read_json(raw_path)
        if raw.get("raw_response") != record["raw_response"]:
            raise ValueError("Raw monitor response and exported response differ")
        generation = raw["generation"]
        if (
            case_id not in planned
            or generation.get("prompt_token_ids") != planned[case_id]["prompt_token_ids"]
        ):
            raise ValueError("Generated monitor prefix differs from prepared token IDs")
        if record.get("protocol_sha256") != protocol_hash or record.get("test_fixture") != fixture:
            raise ValueError("Monitor row protocol/fixture provenance differs")
        parsed = parse_monitor_output(
            record["raw_response"], row["messages"], finish_reason=generation["finish_reason"]
        )
        if any(record.get(key) != value for key, value in parsed.items()):
            raise ValueError("Saved monitor prediction does not match raw response parsing")
        grouped_rows[row["group_id"]].append(row)
        grouped_records[row["group_id"]].append(record)
    groups = [
        monitor_group_metrics(grouped_rows[g], grouped_records[g]) for g in sorted(grouped_rows)
    ]
    total, correct = sum(g["rows"] for g in groups), sum(g["correct"] for g in groups)
    result = {
        "schema": "main-study-text-monitor-analysis-v1",
        "bank_sha256": bank_digest(bank),
        "partition": split,
        "test_fixture": fixture,
        "scientific_result": not fixture,
        "real_empirical_result": not fixture,
        "research_status": payload.get("research_status", "unspecified_legacy"),
        "human_review_complete": payload.get("human_review_complete", False),
        "confirmatory_result": False,
        "method": "prose_extractor_then_symbolic_fold",
        "is_symbolic_oracle": False,
        "model_settings": payload["model_settings"],
        "groups": groups,
        "overall": {
            "rows": total,
            "valid": sum(g["valid"] for g in groups),
            "invalid": sum(g["invalid"] for g in groups),
            "correct": correct,
            "accuracy": correct / total,
            "macro_scenario_accuracy": float(np.mean([g["accuracy"] for g in groups])),
        },
        "invalid_policy": "Incorrect in full-denominator accuracy/pairs/quartets; null scores, valid-only AUC/Brier with explicit denominator",
        "source_sha256": {
            name: file_sha256(root / name)
            for name in ("records.json", "run_manifest.json", "protocol.json")
        },
    }
    for mechanism in ("observation", "message", "correction"):
        subset = [g for g in groups if g["mechanism"] == mechanism]
        if subset:
            result.setdefault("by_mechanism", {})[mechanism] = {
                "scenario_groups": len(subset),
                "macro_complete_quartet_accuracy": float(
                    np.mean([g["complete_quartet_accuracy"] for g in subset])
                ),
                "macro_accuracy": float(np.mean([g["accuracy"] for g in subset])),
                "invalid": sum(g["invalid"] for g in subset),
            }
    return result
