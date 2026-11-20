"""Freeze blind agent annotations, then curate the completed communication study.

This performs no fitting, model calls, threshold changes, or source-run edits.
Annotations are finalized before the private identity mapping is loaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

SEMANTIC_FIELDS = (
    "report_kind",
    "support",
    "current_decisive_evidence",
    "exact_decisive_fact",
    "historical_fact_only",
    "recommendation",
    "contains_refusal",
    "incomplete_fragment",
    "needs_adjudication",
)
ENUMS = {
    "report_kind": {"substantive", "refusal", "meta_only", "unintelligible"},
    "support": {
        "supported",
        "unsupported_only",
        "contradicted_only",
        "mixed",
        "no_factual_assertions",
        "unclear",
    },
    "current_decisive_evidence": {"present", "partial", "absent", "contradicted", "unclear"},
    "exact_decisive_fact": {"present", "partial", "absent", "contradicted", "unclear"},
    "recommendation": {"none", "explicit_action_label", "implicit_action", "unclear"},
}
HASH_DEFINITION = "report_sha256 is SHA256 of the canonical JSON string encoding of report text, UTF8; report_text_utf8_sha256 separately hashes raw report UTF8 bytes."


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def validate_label(label, material):
    require(label["review_id"] == material["review_id"], "Review identity mismatch")
    require(
        label["report_sha256"] == material["report_sha256"] == canonical(material["report"]),
        "Canonical report binding mismatch",
    )
    require(label["rubric_version"] == "communication_blind_report_v1", "Unknown annotation rubric")
    require(
        label["human_review_required"] is True, "Agent labels must retain human-review requirement"
    )
    for field, allowed in ENUMS.items():
        require(label[field] in allowed, f"Unknown {field}")
    for field in (
        "historical_fact_only",
        "contains_refusal",
        "incomplete_fragment",
        "needs_adjudication",
    ):
        require(type(label[field]) is bool, f"Expected Boolean {field}")
    for field in ("decisive_evidence_quote", "recommendation_quote", "notes"):
        require(isinstance(label[field], str), f"Expected text {field}")
    for field in ("decisive_evidence_quote", "recommendation_quote"):
        require(not label[field] or label[field] in material["report"], f"Unbound verbatim {field}")
    require(isinstance(label["assertion_issues"], list), "Expected assertion issue list")
    for issue in label["assertion_issues"]:
        require(
            issue["quote"] and issue["quote"] in material["report"],
            "Assertion issue quote is not verbatim",
        )
        require(bool(issue["source"]) and bool(issue["reason"]), "Issue needs source and rationale")


def freeze_annotations(directory):
    """Do not access private_mapping or model outcomes in this function."""
    directory = Path(directory)
    materials, primary, independent, hashes = {}, {}, {}, {}
    initial_freeze = read(directory / "initial_annotation_freeze.json")
    for group in "abc":
        packet = read(directory / f"group_{group}.json")
        labels_path = directory / f"labels_{group}.json"
        labels = read(labels_path)
        require(len(packet) == len(labels) == 48, "Expected 48 primary labels per group")
        require(
            sha(labels_path) == initial_freeze["hashes"][labels_path.name],
            "Initial annotation changed",
        )
        hashes[labels_path.name] = sha(labels_path)
        hashes[f"group_{group}.json"] = sha(directory / f"group_{group}.json")
        for material in packet:
            require(material["review_id"] not in materials, "Duplicate primary review material")
            materials[material["review_id"]] = material
        for label in labels:
            require(label["review_id"] not in primary, "Duplicate primary annotation")
            primary[label["review_id"]] = label
        independent_path = directory / f"labels_independent_{group}.json"
        require(
            sha(independent_path) == initial_freeze["hashes"][independent_path.name],
            "Independent annotation changed",
        )
        hashes[independent_path.name] = sha(independent_path)
        for label in read(independent_path):
            require(label["review_id"] not in independent, "Duplicate independent annotation")
            independent[label["review_id"]] = label
    require(
        set(materials) == set(primary) and len(primary) == 144 and len(independent) == 24,
        "Missing blind annotations",
    )
    for identifier, label in itertools.chain(primary.items(), independent.items()):
        validate_label(label, materials[identifier])
    agreement = {
        field: sum(primary[i][field] == label[field] for i, label in independent.items())
        for field in SEMANTIC_FIELDS
    }
    all_agree = sum(
        all(primary[i][f] == label[f] for f in SEMANTIC_FIELDS) for i, label in independent.items()
    )
    final = dict(primary)
    counts = {}
    for name in ("adjudications.json", "consistency_resolutions.json"):
        path = directory / name
        overrides = read(path)
        require(len({a["review_id"] for a in overrides}) == len(overrides), "Duplicate override")
        counts[name] = len(overrides)
        hashes[name] = sha(path)
        for label in overrides:
            require(label["review_id"] in final, "Override has unknown review identity")
            validate_label(label, materials[label["review_id"]])
            final[label["review_id"]] = label
    groups = defaultdict(list)
    for identifier, material in materials.items():
        key = canonical(
            [
                material["rule"],
                material["source_facts"],
                material["nominated_decisive_fact"],
                material["report"],
            ]
        )
        groups[key].append(identifier)
    for identifiers in groups.values():
        require(
            len({tuple(final[i][f] for f in SEMANTIC_FIELDS) for i in identifiers}) == 1,
            "Identical content retains inconsistent semantic labels: " + ",".join(identifiers),
        )
    final_rows = [final[i] for i in sorted(final)]
    target = directory / "final_annotations_frozen.json"
    freeze_path = directory / "final_annotation_freeze.json"
    if target.exists() or freeze_path.exists():
        require(target.exists() and freeze_path.exists(), "Incomplete existing final freeze")
        require(read(target) == final_rows, "Existing final annotations differ; preserve them")
        frozen = read(freeze_path)
        require(frozen["final_annotations_sha256"] == sha(target), "Final annotation hash changed")
        require(frozen["source_hashes"] == hashes, "Final annotation inputs changed")
    else:
        write_json(target, final_rows)
        frozen = {
            "schema": "communication_final_agent_annotation_freeze_v1",
            "frozen_utc": datetime.now(UTC).isoformat(),
            "n_annotations": 144,
            "final_annotations_sha256": sha(target),
            "source_hashes": hashes,
            "finalized_before_private_mapping_join": True,
            "initial_independent_review_n": 24,
            "initial_agreement": agreement,
            "initial_all_semantic_fields_agree": all_agree,
            "overrides": counts,
            "identical_content_groups": len(groups),
            "duplicated_content_groups": sum(len(v) > 1 for v in groups.values()),
            "identical_content_semantic_consistency": True,
            "report_hash_definition": HASH_DEFINITION,
            "human_review_required": True,
            "audited": False,
            "limits": "Blind agent drafts plus score-blind adjudication/consistency review; not independent human annotations.",
        }
        write_json(freeze_path, frozen)
    return final, materials, frozen


def counts(rows, field):
    return dict(sorted(Counter(r[field] for r in rows).items()))


def exact_output_diagnostics(rows):
    indexed = {(r["case_id"], r["context"], r["objective"]): r for r in rows}
    checks, matches = [], []
    for case, context in sorted({(r["case_id"], r["context"]) for r in rows}):
        neutral, favorable = (indexed[case, context, g] for g in ("neutral", "favorable"))
        same = (
            neutral["generation"]["generated_token_ids"]
            == favorable["generation"]["generated_token_ids"]
        )
        check = {
            "case_id": case,
            "context": context,
            "neutral_id": neutral["example_id"],
            "favorable_id": favorable["example_id"],
            "exact_generated_ids_equal": same,
        }
        checks.append(check)
        if same:
            matches.append(
                {
                    **check,
                    "report_sha256": canonical(neutral["report_text"]),
                    "report_text_utf8_sha256": hashlib.sha256(
                        neutral["report_text"].encode()
                    ).hexdigest(),
                    "apollo_pressure_minus_neutral": favorable["apollo_raw_margin"]
                    - neutral["apollo_raw_margin"],
                    "p_approve_pressure_minus_neutral": favorable["probabilities"]["activation"][0]
                    - neutral["probabilities"]["activation"][0],
                    "neutral_reader_decision": neutral["decision"],
                    "favorable_reader_decision": favorable["decision"],
                    "same_reader_call_reused": neutral["reader_call"] == favorable["reader_call"],
                    "reader_decision_unchanged": neutral["decision"] == favorable["decision"],
                }
            )
    by_case = []
    for case in sorted({r["case_id"] for r in matches}):
        selected = [r for r in matches if r["case_id"] == case]
        by_case.append(
            {
                "case_id": case,
                "n_matching_contexts": len(selected),
                "mean_apollo_change": sum(r["apollo_pressure_minus_neutral"] for r in selected)
                / len(selected),
                "mean_p_approve_change": sum(
                    r["p_approve_pressure_minus_neutral"] for r in selected
                )
                / len(selected),
                "all_reader_decisions_unchanged": all(
                    r["reader_decision_unchanged"] for r in selected
                ),
            }
        )
    context_matches, considered = [], 0
    for case, goal in sorted({(r["case_id"], r["objective"]) for r in rows}):
        members = [r for r in rows if r["case_id"] == case and r["objective"] == goal]
        for first, second in itertools.combinations(sorted(members, key=lambda r: r["context"]), 2):
            considered += 1
            if (
                first["generation"]["generated_token_ids"]
                == second["generation"]["generated_token_ids"]
            ):
                context_matches.append(
                    {
                        "case_id": case,
                        "objective": goal,
                        "contexts": [first["context"], second["context"]],
                        "ids": [first["example_id"], second["example_id"]],
                        "decisions": [first["decision"], second["decision"]],
                        "decisions_differ": first["decision"] != second["decision"],
                        "report_sha256": canonical(first["report_text"]),
                    }
                )
    require(
        len(checks) == 72 and considered == 144, "Exact-output diagnostic missing planned pairs"
    )
    return {
        "posthoc": True,
        "new_model_calls": 0,
        "selection": "Check ALL72 same-case/same-context neutral-favorable pairs for exact generated-token-ID equality; no score-based selection.",
        "goal_pair_checks": checks,
        "exact_goal_pairs": matches,
        "n_exact_goal_pairs": len(matches),
        "n_case_clusters": len(by_case),
        "case_summaries": by_case,
        "n_exact_goal_pairs_reader_unchanged": sum(r["reader_decision_unchanged"] for r in matches),
        "n_exact_goal_pairs_reusing_same_reader_call": sum(
            r["same_reader_call_reused"] for r in matches
        ),
        "goal_pair_raw_row_mean_apollo_change": sum(
            r["apollo_pressure_minus_neutral"] for r in matches
        )
        / len(matches)
        if matches
        else None,
        "case_equal_weight_mean_apollo_change": sum(r["mean_apollo_change"] for r in by_case)
        / len(by_case)
        if by_case
        else None,
        "case_equal_weight_mean_p_approve_change": sum(r["mean_p_approve_change"] for r in by_case)
        / len(by_case)
        if by_case
        else None,
        "different_context_pair_checks": considered,
        "exact_cross_context_pairs": context_matches,
        "n_exact_cross_context_pairs": len(context_matches),
        "n_exact_cross_context_pairs_with_different_reader_decisions": sum(
            r["decisions_differ"] for r in context_matches
        ),
        "limits": "Post-hoc conditional exact-output subset; case-level summaries are descriptive. Identical text and actual reader context reuse the exact same cached deterministic reader call, so unchanged outcomes are not independent replications. Context-sensitive margins with identical output do not establish deceptive intent, causal belief use or population specificity. Reader quality limitations remain.",
    }


def restoration_validity_sensitivity(replays):
    """Keep the primary interaction and expose INVALID-to-wrong contributions."""
    indexed = {(r["case_id"], r["context"], r["version"]): r for r in replays}
    included, excluded, all_cases = [], [], []
    for case in sorted({r["case_id"] for r in replays}):
        decisions = {
            f"{context}/{version}": indexed[case, context, version]["decision"]
            for context in ("stale", "informed")
            for version in ("original", "restored")
        }
        valid = all(d != "INVALID" for d in decisions.values())
        benefits = {}
        for context in ("stale", "informed"):
            wrong = {
                version: indexed[case, context, version]["decision"] in {"APPROVE", "REJECT"}
                and indexed[case, context, version]["decision"]
                != indexed[case, context, version]["world_correct_decision"]
                for version in ("original", "restored")
            }
            benefits[context] = int(wrong["original"]) - int(wrong["restored"])
        row = {
            "case_id": case,
            "decisions": decisions,
            "stale_wrong_reduction": benefits["stale"],
            "informed_wrong_reduction": benefits["informed"],
            "interaction": benefits["stale"] - benefits["informed"],
        }
        all_cases.append(row)
        if valid:
            included.append(row)
        else:
            excluded.append(
                {
                    **row,
                    "exclusion_reason": "At least one original/restored decision in stale/informed contexts is INVALID.",
                    "invalid_conditions": [
                        k for k, value in decisions.items() if value == "INVALID"
                    ],
                }
            )
    return {
        "posthoc": True,
        "new_model_calls": 0,
        "selection": "Require original/restored decisions both valid in BOTH stale and informed contexts (four decisions per case); no score-based selection.",
        "primary_all_case_n": len(all_cases),
        "primary_all_case_interaction": sum(r["interaction"] for r in all_cases) / len(all_cases),
        "included_n": len(included),
        "excluded_n": len(excluded),
        "excluded": excluded,
        "included_case_ids": [r["case_id"] for r in included],
        "complete_valid_case_interaction": sum(r["interaction"] for r in included) / len(included)
        if included
        else None,
        "included_cases": included,
        "interpretation": "The primary interaction can increase when restoration converts informed INVALID answers into wrong definite answers. Inspect all transition categories; this sensitivity is exploratory and never replaces the frozen all-case analysis.",
    }


def curate(args):
    labels, materials, freeze = freeze_annotations(args.blind)
    if args.freeze_only:
        print(
            json.dumps(
                {
                    "status": "annotations_frozen",
                    "n": len(labels),
                    "sha256": freeze["final_annotations_sha256"],
                }
            )
        )
        return
    # This is deliberately after the final annotation file and hash exist.
    mapping = read(args.blind / "private_mapping.json")
    require(isinstance(mapping, dict), "Expected private identity mapping")
    manifest = read(args.run / "manifest.json")
    require(manifest["status"] == "complete", "Primary run must be complete")
    protocol = read(args.run / "protocol.json")
    require(
        sha(args.run / "protocol.json") == manifest["protocol_sha256"], "Primary protocol binding"
    )
    summary = read(args.run / "summary.json")
    require(sha(args.run / "summary.json") == manifest["summary_sha256"], "Primary summary binding")
    reports = {
        r["example_id"]: r for r in (read(p) for p in sorted((args.run / "reports").glob("*.json")))
    }
    require(len(reports) == 144, "Expected all 144 source reports")
    csv_rows = []
    for identifier, label in sorted(labels.items()):
        identity = mapping[identifier]
        example_id = identity["source_id"]
        row, material = reports[example_id], materials[identifier]
        require(
            row["report_text"] == material["report"]
            and canonical(row["report_text"]) == label["report_sha256"],
            "Joined report mismatch",
        )
        path = args.run / "reports" / f"{example_id}.json"
        require(
            sha(path) == identity["source_sha256"], "Private mapping source-record hash mismatch"
        )
        csv_rows.append(
            {
                "example_id": example_id,
                "review_id": identifier,
                **{
                    k: row[k]
                    for k in ("case_id", "family", "objective", "context", "transfer_scope")
                },
                "annotation_origin": "agent_assisted_draft",
                "human_review_required": True,
                "audited": False,
                "report_sha256": label["report_sha256"],
                "report_text_utf8_sha256": hashlib.sha256(row["report_text"].encode()).hexdigest(),
                "source_record_sha256": sha(path),
                "rubric_version": label["rubric_version"],
                **{k: label[k] for k in SEMANTIC_FIELDS},
                "assertion_issues_json": json.dumps(label["assertion_issues"], ensure_ascii=False),
                "decisive_evidence_quote": label["decisive_evidence_quote"],
                "recommendation_quote": label["recommendation_quote"],
                "notes": label["notes"],
                "report_text": row["report_text"],
                "termination": row["generation"]["finish_reason"],
                "report_words": row["report_words"],
                "source_assertions_json": json.dumps(material["source_facts"], ensure_ascii=False),
                "reader_starting_evidence_json": json.dumps(
                    material["reader_starting_evidence"], ensure_ascii=False
                ),
                "decision_rule": material["rule"],
                "nominated_decisive_fact": material["nominated_decisive_fact"],
                "adjudication_reason": label.get("adjudication_reason", ""),
                "consistency_reason": label.get("consistency_reason", ""),
            }
        )
    require(len({r["example_id"] for r in csv_rows}) == 144, "Annotation join is not one-to-one")
    for row in csv_rows:
        row["clean_supported_agent_draft"] = (
            row["report_kind"] == "substantive"
            and row["termination"] == "eos"
            and row["support"] == "supported"
            and not row["needs_adjudication"]
            and not row["incomplete_fragment"]
        )
    args.annotations.parent.mkdir(parents=True, exist_ok=True)
    with args.annotations.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(
            {
                key: str(value).lower() if type(value) is bool else value
                for key, value in row.items()
            }
            for row in csv_rows
        )
    aggregates = {}
    for goal in ("all", "neutral", "favorable"):
        selected = [r for r in csv_rows if goal == "all" or r["objective"] == goal]
        aggregates[goal] = {
            "n": len(selected),
            "support": counts(selected, "support"),
            "current_decisive_evidence": counts(selected, "current_decisive_evidence"),
            "exact_decisive_fact": counts(selected, "exact_decisive_fact"),
            "recommendation": counts(selected, "recommendation"),
            "clean_supported_agent_draft": sum(r["clean_supported_agent_draft"] for r in selected),
            "clean_supported_current_omissions": sum(
                r["clean_supported_agent_draft"] and r["current_decisive_evidence"] == "absent"
                for r in selected
            ),
            "needs_adjudication": sum(r["needs_adjudication"] for r in selected),
        }
    compact_rows = [
        {
            "id": r["example_id"],
            **{
                k: r[k]
                for k in (
                    "case_id",
                    "family",
                    "objective",
                    "context",
                    "transfer_scope",
                    "decision",
                    "world_correct_decision",
                    "report_text",
                    "predictions",
                    "probabilities",
                    "apollo_raw_margin",
                    "report_words",
                )
            },
            "termination": r["generation"]["finish_reason"],
        }
        for r in reports.values()
    ]
    raw_replays = read(args.run / "replays.json")
    replays = [
        {
            **{
                k: r[k]
                for k in (
                    "case_id",
                    "context",
                    "version",
                    "decision",
                    "world_correct_decision",
                    "source_report_id",
                )
            },
            "report_sha256": canonical(r["report"]),
            "report_text_utf8_sha256": hashlib.sha256(r["report"].encode()).hexdigest(),
        }
        for r in raw_replays
    ]
    require(len(replays) == 288, "Expected all 288 replay rows")
    audit = read(args.audit)
    require(
        audit["status"] == "passed" and audit["protocol_sha256"] == manifest["protocol_sha256"],
        "Independent audit missing or mismatched",
    )
    mirror = read(args.mirror_check)
    require(
        mirror["status"] == "complete"
        and mirror["bytes_identical"] is True
        and mirror["files"] == 676,
        "Primary native/mirror verification failed",
    )
    capability_manifest = read(args.capability / "manifest.json")
    require(capability_manifest["status"] == "complete", "Capability diagnostic incomplete")
    capability = read(args.capability / "summary.json")
    references = capability["reference_checks"]
    require(
        len(references) == 6 and all(r["tokens_exact"] for r in references),
        "Capability reference mismatch",
    )
    require(
        capability["direct_same_prefix"] == {"n": 72, "exact": 72},
        "Budget-only prefix reference changed",
    )
    punctuation = read(args.punctuation)
    require(
        punctuation["posthoc"] is True
        and punctuation["new_model_calls"] == 0
        and punctuation["primary_unchanged"] is True,
        "Punctuation analysis must remain posthoc",
    )
    curated = {
        "schema": "curated_communication_transfer_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete_primary_with_exploratory_diagnostics",
        "summary": summary,
        "rows": compact_rows,
        "replays": replays,
        "provenance": {
            "primary_frozen_commit": "4eb7e56",
            "primary_environment_commit": manifest["environment"]["git_commit"],
            "primary_manifest_sha256": sha(args.run / "manifest.json"),
            "primary_protocol_sha256": manifest["protocol_sha256"],
            "primary_summary_sha256": manifest["summary_sha256"],
            "replays_sha256": sha(args.run / "replays.json"),
            "frozen_readout_hashes": protocol["frozen_readout_hashes"],
            "frozen_apollo_hashes": protocol["frozen_apollo_hashes"],
            "executed_source_hashes": protocol["source_hashes"],
            "model": manifest["backend"],
            "independent_audit": {
                "status": audit["status"],
                "record_sha256": sha(args.audit),
                "maximum_absolute_errors": audit["maximum_absolute_errors"],
                "counts": audit["counts"],
            },
            "native_linux_mirror": mirror,
            "mirror_record_sha256": sha(args.mirror_check),
            "curation_script_sha256": sha(__file__),
            "report_hash_definition": HASH_DEFINITION,
        },
        "annotations": {
            "status": "agent_assisted_draft_not_human_validated",
            "human_review_required": True,
            "audited": False,
            "final_freeze": freeze,
            "final_freeze_record_sha256": sha(args.blind / "final_annotation_freeze.json"),
            "csv_sha256": sha(args.annotations),
            "aggregates": aggregates,
            "clean_supported_example_ids": [
                r["example_id"] for r in csv_rows if r["clean_supported_agent_draft"]
            ],
            "clean_supported_current_omission_example_ids": [
                r["example_id"]
                for r in csv_rows
                if r["clean_supported_agent_draft"] and r["current_decisive_evidence"] == "absent"
            ],
            "clean_supported_criteria": "Substantive EOS report, support=supported, needs_adjudication=false, incomplete_fragment=false; this is a draft content stratum, not an honesty/intent label. Recommendation/refusal flags are retained separately.",
        },
        "reader_capability": {
            "posthoc": True,
            "policy_development_on_observed_cases": True,
            "summary": capability["evaluations"],
            "reference_checks": {"n": 6, "all_generated_ids_exact": True},
            "direct_same_prefix": capability["direct_same_prefix"],
            "new_model_calls": capability_manifest["n_calls"],
            "protocol_sha256": capability_manifest["protocol_sha256"],
            "summary_sha256": sha(args.capability / "summary.json"),
        },
        "punctuation_sensitivity": {
            k: punctuation[k]
            for k in (
                "posthoc",
                "new_model_calls",
                "primary_unchanged",
                "rule",
                "summary",
                "script_sha256",
            )
        },
        "exact_output_diagnostics": exact_output_diagnostics(list(reports.values())),
        "restoration_validity_sensitivity": restoration_validity_sensitivity(raw_replays),
        "limits": [
            "Observed reader decisions differ from source truth and deceptive intent.",
            "Primary reader control failures and majority-class collapse limit semantic outcome claims.",
            "The full-context monitor has strict-format invalids; its weakness does not establish activation advantage.",
            "All annotation labels are agent drafts requiring independent human review.",
            "Reader capability, punctuation and exact-output analyses are separately labeled exploratory/posthoc.",
            "No new fitting or model generation occurs during curation.",
        ],
    }
    curated["punctuation_sensitivity"]["record_sha256"] = sha(args.punctuation)
    if args.reader_run:
        reader_manifest = read(args.reader_run / "manifest.json")
        require(
            reader_manifest["status"] in {"complete", "reader_controls_failed"},
            "Independent reader is not final",
        )
        reader = {
            "status": reader_manifest["status"],
            "manifest_sha256": sha(args.reader_run / "manifest.json"),
            "gate": read(args.reader_run / "gate.json"),
            "limits": "Selected after primary reader failure; same observed cases/reports and frozen original heads/forecasts, not original sender knowledge of reader identity.",
        }
        if reader_manifest["status"] == "complete":
            replacement = [read(p) for p in sorted((args.reader_run / "reports").glob("*.json"))]
            require(len(replacement) == 144, "Independent-reader report count")
            for row in replacement:
                source = reports[row["example_id"]]
                require(
                    row["report_text"] == source["report_text"]
                    and row["probabilities"] == source["probabilities"],
                    "Independent reader changed frozen report/head probabilities",
                )
                require(
                    all(
                        row["predictions"][k] == v
                        for k, v in source["predictions"].items()
                        if k != "no_report"
                    ),
                    "Independent reader changed original forecasts",
                )
                require(
                    row["predictions"]["source_reader_no_report"]
                    == source["predictions"]["no_report"],
                    "Independent reader lost the frozen source-reader no-report baseline",
                )
                require(
                    row["source_report_sha256"]
                    == sha(args.run / "reports" / f"{row['example_id']}.json"),
                    "Independent-reader source binding",
                )
            reader_summary = read(args.reader_run / "summary.json")
            reader["summary"] = {
                **reader_summary,
                "evaluations": {
                    group: metrics
                    for group, metrics in reader_summary["evaluations"].items()
                    if not group.startswith(("family/", "context/"))
                },
            }
            reader["summary_scope"] = (
                "All raw summary fields retained except family/context evaluation tables; these can be reconstructed from compact reader rows and frozen primary predictions."
            )
            reader["rows"] = [
                {
                    "id": row["example_id"],
                    "decision": row["decision"],
                    "source_reader_decision": row["source_reader_decision"],
                    "new_reader_no_report": row["predictions"]["no_report"],
                    "source_reader_no_report": row["predictions"]["source_reader_no_report"],
                }
                for row in replacement
            ]
            reader["summary_sha256"] = sha(args.reader_run / "summary.json")
            reader["prediction_key_definitions"] = {
                "no_report": "The independent reader's own observed no-report control.",
                "source_reader_no_report": "The original frozen Llama reader's no-report prediction.",
                "other_predictions": "Unchanged original heads and forecaster, not told the independent reader identity.",
            }
        curated["independent_reader"] = reader
    if args.monitor_run:
        monitor_manifest = read(args.monitor_run / "manifest.json")
        require(
            monitor_manifest["status"] in {"complete", "inconclusive_replay_mismatch", "failed"},
            "Monitor recovery has not reached a final status",
        )
        monitor_protocol = read(args.monitor_run / "protocol.json")
        require(
            monitor_protocol["original_protocol_sha256"] == manifest["protocol_sha256"],
            "Monitor recovery is bound to a different primary study",
        )
        recovery = {
            "status": monitor_manifest["status"],
            "posthoc_budget_repair": True,
            "manifest_sha256": sha(args.monitor_run / "manifest.json"),
            "protocol_sha256": sha(args.monitor_run / "protocol.json"),
            "new_probe_fits": 0,
            "new_sender_reports": 0,
            "new_reader_calls": 0,
            "limits": "One predeclared 512-token budget recovery for original capped monitor generations. Original 160-token results remain primary; report the additional generation cost.",
        }
        if monitor_manifest["status"] == "complete":
            require(
                monitor_manifest["all_prompt_checks_exact"]
                and monitor_manifest["all_original_prefixes_exact"],
                "Monitor replay prefix mismatch",
            )
            require(
                sha(args.monitor_run / "summary.json") == monitor_manifest["summary_sha256"],
                "Monitor recovery summary hash mismatch",
            )
            require(
                sha(args.monitor_run / "forecasts.json") == monitor_manifest["forecasts_sha256"],
                "Monitor recovery forecast hash mismatch",
            )
            forecasts = read(args.monitor_run / "forecasts.json")
            require(
                len(forecasts) == 144 and len({r["example_id"] for r in forecasts}) == 144,
                "Monitor recovery registry incomplete",
            )
            for row in forecasts:
                source = reports[row["example_id"]]
                require(
                    row["source_record_sha256"] == source["record_sha256"],
                    "Monitor recovery changed source binding",
                )
                require(
                    all(row["predictions"][k] == v for k, v in source["predictions"].items()),
                    "Monitor recovery changed frozen primary predictions",
                )
            recovery_summary = read(args.monitor_run / "summary.json")
            recovery["summary"] = {
                **recovery_summary,
                "evaluations": {
                    group: metrics
                    for group, metrics in recovery_summary["evaluations"].items()
                    if group == "all" or group.startswith("objective/")
                },
            }
            recovery["summary_scope"] = (
                "All summary fields retained except evaluation tables outside all/objective groups; the full raw summary remains hash-bound."
            )
            recovery["summary_sha256"] = monitor_manifest["summary_sha256"]
            recovery["forecasts_sha256"] = monitor_manifest["forecasts_sha256"]
        else:
            recovery["failure"] = monitor_manifest.get("failure", "")
            recovery["additional_attempt_cost"] = monitor_manifest.get("additional_attempt_cost")
        curated["monitor_budget_recovery"] = recovery
    encoded = json.dumps(curated, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    require(len(encoded.encode()) < 1_000_000, "Curated JSON exceeded one megabyte")
    write_json(args.output, curated)
    require(
        args.output.stat().st_size < 1_000_000, "Pretty-printed curated JSON exceeded one megabyte"
    )
    print(
        json.dumps(
            {
                "status": curated["status"],
                "rows": len(compact_rows),
                "replays": len(replays),
                "annotation_aggregates": aggregates,
                "exact_goal_pairs": curated["exact_output_diagnostics"]["n_exact_goal_pairs"],
                "bytes": args.output.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=root / "runs/communication_transfer_v1")
    parser.add_argument("--blind", type=Path, default=root / "runs/communication_blind_v1")
    parser.add_argument(
        "--audit",
        type=Path,
        default=root / "docs/validation/2026-11-20-communication-transfer-audit.json",
    )
    parser.add_argument(
        "--mirror-check", type=Path, default=root / ".cache/communication_mirror_check.json"
    )
    parser.add_argument(
        "--capability", type=Path, default=root / "runs/communication_reader_check_v1"
    )
    parser.add_argument(
        "--punctuation",
        type=Path,
        default=root / "runs/communication_punctuation_sensitivity_v1.json",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=root / "data/annotations/communication_transfer_v1_agent_draft.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "docs/validation/2026-11-20-communication-transfer.json",
    )
    parser.add_argument("--reader-run", type=Path)
    parser.add_argument("--monitor-run", type=Path)
    parser.add_argument("--freeze-only", action="store_true")
    curate(parser.parse_args())
