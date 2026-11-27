"""NEW posthoc query-anchoring control on the SAME saved monitor responses.

No inference, fitting, response repair or original-artifact changes occur.
Only the final comparison operand changes: existing extracted final_location
is compared with the actual physical name copied from the source question.
Original validity, extraction and all source/evaluation labels are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(
    "//wsl.localhost/Ubuntu/home/adirj/kcl-latent-probes-runs/main_v1_agent_20261122_recovery_v1"
)
QUESTION = re.compile(
    r"^At the end of the scene, does [^\r\n?]+? think that [^\r\n?]+? "
    r"believes that the token is in the (?P<location>[^\r\n?]+)\? "
    r"Answer only YES or NO\.\s*\Z",
    flags=re.MULTILINE,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_binding(path, logical_root, root_path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "root": logical_root,
        "relative_path": path.relative_to(Path(root_path).resolve()).as_posix(),
        "sha256": sha(path),
    }


def resolve_bound_source(binding, roots):
    """Map logical roots portably; recorded native paths are provenance only."""
    if binding["root"] not in roots:
        raise ValueError("No mapping for source logical root: " + binding["root"])
    relative = PurePosixPath(binding["relative_path"])
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in str(relative)
        or any(":" in part for part in relative.parts)
    ):
        raise ValueError("Source binding must have a safe relative POSIX path")
    root = Path(roots[binding["root"]]).resolve()
    target = root.joinpath(*relative.parts).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Source binding escapes its logical root")
    return target


def verify_bound_sources(sources, roots):
    for name, binding in sources.items():
        if sha(resolve_bound_source(binding, roots)) != binding["sha256"]:
            raise ValueError("Bound source hash differs: " + name)
    return len(sources)


def actual_question_location(messages):
    """Parse original source TEXT ONLY; fail closed, never accept state/gold.

    This is a declared exact-template parser, with no anaphora/fuzzy fallback.
    Every message's full context is searched; exactly one terminal question
    across the supplied original message list must match.
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("Original source messages must be a nonempty list")
    matches = []
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("Source messages may contain role/content only")
        if message["role"] not in {"system", "user", "assistant"} or not isinstance(
            message["content"], str
        ):
            raise ValueError("Invalid source role or text")
        matches.extend(match.group("location") for match in QUESTION.finditer(message["content"]))
    if len(matches) != 1 or not matches[0] or matches[0] != matches[0].strip():
        raise ValueError(f"Need exactly one explicit final source question; found {len(matches)}")
    return matches[0]


def unique_by_id(values, name):
    result = {}
    for value in values:
        case_id = value.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in result:
            raise ValueError(f"Duplicate or missing {name} case ID")
        result[case_id] = value
    return result


def anchor_records(rows, records):
    """Produce new predictions; never mutate original rows or records."""
    by_id = unique_by_id(rows, "evaluation")
    monitor = unique_by_id(records, "monitor")
    if set(by_id) != set(monitor):
        raise ValueError("Monitor/evaluation ID sets differ")
    anchored = []
    for row in rows:
        record = monitor[row["case_id"]]
        if record.get("messages") != row.get("messages"):
            raise ValueError("Monitor/evaluation source prose differs")
        query = actual_question_location(record["messages"])
        valid = record.get("valid")
        if type(valid) is not bool:
            raise ValueError("Frozen monitor validity must be boolean")
        if valid:
            prediction = record.get("prediction")
            extraction = record.get("extraction")
            if type(prediction) is not int or prediction not in (0, 1):
                raise ValueError("Frozen valid prediction must be a binary integer")
            if record.get("score") != float(prediction) or not isinstance(extraction, dict):
                raise ValueError("Frozen valid score/extraction differs")
            current = extraction["initial_location"]
            for update in extraction["updates"]:
                if update["accepted_access"]:
                    current = update["location"]
            if (
                current != record.get("final_location")
                or int(current == extraction["queried_location"]) != prediction
            ):
                raise ValueError("Frozen final location/prediction differs from its extraction")
            anchored_prediction = int(record["final_location"] == query)
            extracted_query = extraction["queried_location"]
        else:
            if record.get("prediction") is not None or record.get("score") is not None:
                raise ValueError("Frozen invalid response must retain null prediction/score")
            prediction = anchored_prediction = extracted_query = None
        anchored.append(
            {
                "case_id": row["case_id"],
                "group_id": row["group_id"],
                "mechanism": row["mechanism"],
                "split": row["split"],
                "gold": row["gold"],
                "state": row["state"],
                "valid": valid,
                "invalid_reason": record.get("invalid_reason"),
                "source_question_location": query,
                "extracted_query_location": extracted_query,
                "frozen_extracted_final_location": record.get("final_location"),
                "original_prediction": prediction,
                "anchored_prediction": anchored_prediction,
                "query_exact": extracted_query == query if valid else None,
            }
        )
    return anchored


def primary(row):
    return row["mechanism"] != "correction" or row["state"]["initial"] == 1 - row["state"]["update"]


def row_metrics(rows, field):
    if not rows:
        return None
    correct = sum(r["valid"] and r[field] == r["gold"] for r in rows)
    return {
        "rows": len(rows),
        "valid": sum(r["valid"] for r in rows),
        "invalid": sum(not r["valid"] for r in rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "class_recalls_full_denominator": {
            str(label): sum(r["valid"] and r[field] == label for r in rows if r["gold"] == label)
            / sum(r["gold"] == label for r in rows)
            if any(r["gold"] == label for r in rows)
            else None
            for label in (0, 1)
        },
    }


def comparison_counts(rows):
    counts = Counter()
    for row in rows:
        if not row["valid"]:
            counts["invalid_retained"] += 1
            continue
        before = row["original_prediction"] == row["gold"]
        after = row["anchored_prediction"] == row["gold"]
        counts["valid"] += 1
        counts["prediction_changed"] += row["original_prediction"] != row["anchored_prediction"]
        counts[
            f"original_{'correct' if before else 'wrong'}_anchored_{'correct' if after else 'wrong'}"
        ] += 1
        if not row["query_exact"]:
            counts["wrong_extracted_query"] += 1
            counts[
                f"wrong_query_original_{'correct' if before else 'wrong'}_anchored_{'correct' if after else 'wrong'}"
            ] += 1
        elif row["original_prediction"] != row["anchored_prediction"]:
            raise ValueError("Prediction changed despite identical query")
    return dict(counts)


def summarize(rows):
    blocks = defaultdict(list)
    for row in rows:
        if primary(row):
            s = row["state"]
            blocks[(row["group_id"], s["update"], s["world"], s["name_map"])].append(row)
    if any(len(block) != 4 for block in blocks.values()):
        raise ValueError("Incomplete primary quartet")
    result = {"query_error_cancellation": comparison_counts(rows), "original": {}, "anchored": {}}
    for label, field in (("original", "original_prediction"), ("anchored", "anchored_prediction")):
        result[label]["all_rows"] = row_metrics(rows, field)
        result[label]["primary_rows"] = row_metrics([r for r in rows if primary(r)], field)
        result[label]["secondary_rows"] = row_metrics([r for r in rows if not primary(r)], field)
        hist = Counter(
            sum(r["valid"] and r[field] == r["gold"] for r in block) for block in blocks.values()
        )
        result[label]["primary_quartets"] = {
            "quartets": len(blocks),
            "complete_correct": hist[4],
            "accuracy": hist[4] / len(blocks) if blocks else None,
            "correct_cells_histogram": {str(k): hist[k] for k in range(5)},
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/validation/2026-11-22-main-study-query-anchoring-posthoc-v2.json",
    )
    parser.add_argument(
        "--measurement-audit-report",
        type=Path,
        default=ROOT / "docs/validation/2026-11-22-main-study-measurement-forensics.json",
        help="Explicitly select another measurement JSON; default preserves the historical v1 binding.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("New control refuses to overwrite an existing artifact")
    protected = [
        ROOT / "scripts/audit_main_study_measurement_forensics.py",
        ROOT / "tests/test_main_study_measurement_forensics.py",
        ROOT / "docs/MAIN_STUDY_MEASUREMENT_FORENSICS_2026-11-22.md",
        args.measurement_audit_report,
    ]
    protected_hashes = {str(path): sha(path) for path in protected}
    source_paths = {
        "monitor_records": args.run / "monitor_test/records.json",
        "monitor_protocol": args.run / "monitor_test/protocol.json",
        "monitor_run_manifest": args.run / "monitor_test/run_manifest.json",
        "monitor_original_analysis": args.run / "monitor_test_analysis.json",
        "evaluation_rows": args.run / "qwen_test/rows.json",
        "bank": ROOT / "data/belief_main/main_v1/main_bank_v1.json",
        "frozen_parser_source": ROOT / "src/latent_probes/main_study_monitor.py",
        "audit_code": Path(__file__),
        "audit_tests": ROOT / "tests/test_main_study_query_anchoring.py",
        "current_measurement_code": protected[0],
        "current_measurement_tests": protected[1],
        "immutable_measurement_doc": protected[2],
        "selected_measurement_report": protected[3],
        "historical_source_snapshot_index": ROOT
        / "docs/validation/audit-source-snapshots-2026-11-22/index.json",
    }
    run_keys = {
        "monitor_records",
        "monitor_protocol",
        "monitor_run_manifest",
        "monitor_original_analysis",
        "evaluation_rows",
    }
    sources = {
        key: source_binding(path, "run", args.run)
        if key in run_keys
        else source_binding(path, "workspace", ROOT)
        for key, path in source_paths.items()
    }
    payload = read(source_paths["monitor_records"])
    capture = read(source_paths["evaluation_rows"])
    if payload.get("test_fixture") is not False or capture.get("test_fixture") is not False:
        raise ValueError("Native posthoc control requires explicit nonfixture sources")
    if payload.get("partition") != "test":
        raise ValueError("Expected the completed test monitor responses")
    rows = capture["rows"]
    anchored = anchor_records(rows, payload["records"])
    # Coverage validation uses full original contexts from all 5,888 bank rows.
    # The parser itself receives messages only, never symbolic factors or gold.
    from latent_probes.main_study_bank import iter_rows

    bank_rows = list(iter_rows(read(source_paths["bank"])))
    for row in bank_rows:
        actual_question_location(row["messages"])
    result = {
        "schema": "main-study-query-anchoring-posthoc-v2",
        "audit_version": 2,
        "version_note": "Formatting/provenance refresh; same numerical policy; historical source bytes preserved via snapshots.",
        "research_status": "posthoc_same_responses_query_anchoring_control",
        "new_heldout_result": False,
        "human_review_complete": False,
        "confirmatory_result": False,
        "models_run": 0,
        "heads_fitted": 0,
        "new_responses": 0,
        "policy": "Preserve frozen validity, extracted initial/updates/access/final location; compare final location with source-question TEXT ONLY; invalids remain null and incorrect.",
        "bank_full_contexts_query_parsed": len(bank_rows),
        "source_files": sources,
        "protected_measurement_artifact_sha256_before_and_after": protected_hashes,
        "measurement_report_selection": sources["selected_measurement_report"],
        "overall": summarize(anchored),
        "by_mechanism": {
            m: summarize([r for r in anchored if r["mechanism"] == m])
            for m in ("observation", "message", "correction")
        },
        "by_group": {
            g: summarize([r for r in anchored if r["group_id"] == g])
            for g in sorted({r["group_id"] for r in anchored})
        },
        "rows": anchored,
    }
    if any(sha(path) != digest for path, digest in protected_hashes.items()):
        raise ValueError("Protected measurement artifact changed during audit")
    verify_bound_sources(sources, {"workspace": ROOT, "run": args.run})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "overall": result["overall"],
                "by_mechanism": result["by_mechanism"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
