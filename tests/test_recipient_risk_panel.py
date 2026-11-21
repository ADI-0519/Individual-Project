"""Check fresh scientific controls, information boundaries and complete job plans."""

import copy
import json
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from latent_probes.recipient_risk_panel import (
    CONTEXTS,
    PANEL_CANONICAL_SHA256,
    RULE_FAMILIES,
    actor_jobs,
    build_actor_messages,
    build_reader_messages,
    control_plan,
    load_panel,
    normative_decision,
    panel_metadata,
    policy_metadata,
    reader_plan,
    reader_prefix,
    render_record_text,
    source_text,
    validate_panel,
    worked_examples,
)
from latent_probes.storage import canonical_hash

DATA = (
    Path(__file__).resolve().parents[1]
    / "data/communication/recipient_risk_validation_v2_draft.json"
)


@pytest.fixture
def panel():
    return load_panel(DATA)


def oracle(case, ids):
    """Independent simple oracle for the authored records, without rendering helpers."""
    records = [
        f["record"] for f in case["source_assertions"] if f["fact_id"] in ids and "record" in f
    ]
    spec = case["decision_spec"]
    common_type = "authorization" if spec["type"] == "authorization_expiry" else "check"
    critical_type = {
        "latest_checks": "check",
        "numeric_threshold": "measurement",
        "authorization_expiry": "expiry",
    }[spec["type"]]

    def latest(kind, field):
        candidates = [r for r in records if r["record_type"] == kind and r["field"] == field]
        return (
            max(candidates, key=lambda r: date.fromisoformat(r["record_date"]))["value"]
            if candidates
            else None
        )

    common = latest(common_type, spec["common_field"])
    critical = latest(critical_type, spec["critical_field"])
    if common in ("FAIL", "DENIED"):
        return "REJECT"
    if critical is not None:
        if spec["type"] == "latest_checks":
            satisfied = critical == "PASS"
        elif spec["type"] == "numeric_threshold":
            satisfied = (
                critical >= spec["threshold"]
                if spec["operator"] == ">="
                else critical <= spec["threshold"]
            )
        else:
            satisfied = date.fromisoformat(critical) >= date.fromisoformat(spec["decision_date"])
        if not satisfied:
            return "REJECT"
        if common in ("PASS", "GRANTED"):
            return "APPROVE"
    return "DEFER"


def fact(case, key):
    return next(f for f in case["source_assertions"] if f["fact_id"] == case[key])


def reword(case, item):
    item["text"] = render_record_text(case["entity"], item["record"], item["text_style"])


def test_bank_is_lf_bound_development_draft(panel):
    assert b"\r" not in DATA.read_bytes()
    assert canonical_hash(panel) == PANEL_CANONICAL_SHA256
    assert panel["human_review"] is False
    assert panel["agent_draft"] is True
    assert panel["fitting_authorized"] is False
    assert panel["observed_labels"] is None
    assert len(panel["cases"]) == 12


def test_independent_oracle_checks_all_packet_and_full_controls(panel):
    for case in panel["cases"]:
        complete = [f["fact_id"] for f in case["source_assertions"]]
        assert (
            oracle(case, complete)
            == normative_decision(case, complete)
            == case["complete_source_gold"]
        )
        for context, ids in case["recipient_packets"].items():
            assert (
                oracle(case, ids)
                == normative_decision(case, ids)
                == case["packet_expected_decisions"][context]
            )
        assert normative_decision(case, []) == "DEFER"


def test_world_positions_and_mention_chronology_are_independent(panel):
    assert Counter((c["rule_family"], c["complete_source_gold"]) for c in panel["cases"]) == {
        (r, y): 2 for r in RULE_FAMILIES for y in ("APPROVE", "REJECT")
    }
    positions, ordering = Counter(), Counter()
    for case in panel["cases"]:
        ids = [f["fact_id"] for f in case["source_assertions"]]
        current, prior = (
            ids.index(case[k]) for k in ("current_decisive_fact_id", "prior_critical_fact_id")
        )
        positions[current + 1, case["complete_source_gold"]] += 1
        ordering[case["rule_family"], case["complete_source_gold"], current < prior] += 1
    assert positions == {(p, y): 2 for p in (2, 5, 8) for y in ("APPROVE", "REJECT")}
    assert ordering == {
        (r, y, before): 1
        for r in RULE_FAMILIES
        for y in ("APPROVE", "REJECT")
        for before in (True, False)
    }


def test_irrelevant_words_cannot_determine_the_world_label(panel):
    for case in panel["cases"]:
        text = source_text(case)
        assert "PASS" in text and "FAIL" in text
        common = [case["common_fact_id"]]
        nuisance = [
            f["fact_id"]
            for f in case["source_assertions"]
            if f["role"] in ("irrelevant_positive", "irrelevant_negative", "irrelevant_addition")
        ]
        assert normative_decision(case, common + nuisance) == "DEFER"
        assert normative_decision(case, nuisance) == "DEFER"


def test_specific_restoration_and_placebo_preserve_expected_semantics(panel):
    for case in panel["cases"]:
        current = case["current_decisive_fact_id"]
        placebo = case["irrelevant_addition"]["fact_id"]
        for context, ids in case["recipient_packets"].items():
            assert (
                normative_decision(case, ids + [placebo])
                == case["packet_expected_decisions"][context]
            )
            assert (
                normative_decision(case, ids if current in ids else ids + [current])
                == case["complete_source_gold"]
            )


@pytest.mark.parametrize("kind", RULE_FAMILIES)
def test_available_failure_rejects_even_with_other_requirement_missing(panel, kind):
    case = next(
        c
        for c in panel["cases"]
        if c["rule_family"] == kind and c["complete_source_gold"] == "REJECT"
    )
    assert normative_decision(case, [case["current_decisive_fact_id"]]) == "REJECT"
    good = next(
        c
        for c in panel["cases"]
        if c["rule_family"] == kind and c["complete_source_gold"] == "APPROVE"
    )
    assert normative_decision(good, [good["current_decisive_fact_id"]]) == "DEFER"


@pytest.mark.parametrize("operator", (">=", "<="))
def test_numeric_boundary_is_inclusive_and_directional(panel, operator):
    case = copy.deepcopy(
        next(
            c
            for c in panel["cases"]
            if c["rule_family"] == "numeric_threshold"
            and c["decision_spec"]["operator"] == operator
        )
    )
    current = fact(case, "current_decisive_fact_id")
    threshold = case["decision_spec"]["threshold"]
    ids = [case["common_fact_id"], case["current_decisive_fact_id"]]
    for value, expected in (
        (threshold, "APPROVE"),
        (threshold + (1 if operator == ">=" else -1), "APPROVE"),
        (threshold + (-1 if operator == ">=" else 1), "REJECT"),
    ):
        current["record"]["value"] = value
        assert normative_decision(case, ids) == expected


def test_expiry_uses_issue_date_not_largest_expiration_and_equality_is_valid(panel):
    case = copy.deepcopy(
        next(
            c
            for c in panel["cases"]
            if c["rule_family"] == "authorization_expiry" and c["complete_source_gold"] == "REJECT"
        )
    )
    current = fact(case, "current_decisive_fact_id")
    assert normative_decision(case, case["recipient_packets"]["stale"]) == "APPROVE"
    assert normative_decision(case, case["recipient_packets"]["informed"]) == "REJECT"
    assert (
        normative_decision(
            case,
            [
                case["common_fact_id"],
                case["prior_critical_fact_id"],
                case["current_decisive_fact_id"],
            ],
        )
        == "REJECT"
    )
    current["record"]["value"] = case["decision_spec"]["decision_date"]
    assert normative_decision(case, case["recipient_packets"]["informed"]) == "APPROVE"
    fact(case, "common_fact_id")["record"]["value"] = "DENIED"
    assert normative_decision(case, [case["common_fact_id"]]) == "REJECT"


def test_prose_and_rule_must_match_the_executable_facts(panel):
    broken = copy.deepcopy(panel)
    fact(broken["cases"][0], "current_decisive_fact_id")["text"] += " It passed."
    with pytest.raises(ValueError, match="model-visible prose"):
        validate_panel(broken, check_hash=False)
    broken = copy.deepcopy(panel)
    broken["cases"][0]["reader_visible_decision_rule"] += " Always approve."
    with pytest.raises(ValueError, match="executable semantics"):
        validate_panel(broken, check_hash=False)


def test_wrong_authored_labels_are_rejected_without_hash_check(panel):
    broken = copy.deepcopy(panel)
    broken["cases"][0]["complete_source_gold"] = "DEFER"
    with pytest.raises(ValueError, match="normative controls"):
        validate_panel(broken, check_hash=False)
    broken = copy.deepcopy(panel)
    broken["cases"][0]["packet_expected_decisions"]["absent"] = "APPROVE"
    with pytest.raises(ValueError, match="normative controls"):
        validate_panel(broken, check_hash=False)


def test_current_must_be_newer_and_ids_unique(panel):
    broken = copy.deepcopy(panel)
    case = broken["cases"][0]
    current, prior = (fact(case, k) for k in ("current_decisive_fact_id", "prior_critical_fact_id"))
    current["record"]["record_date"] = prior["record"]["record_date"]
    reword(case, current)
    with pytest.raises(ValueError, match="strictly newer"):
        validate_panel(broken, check_hash=False)
    with pytest.raises(ValueError, match="unique sequence"):
        normative_decision(panel["cases"][0], [panel["cases"][0]["common_fact_id"]] * 2)


def test_conflicting_same_date_records_are_rejected(panel):
    case = copy.deepcopy(next(c for c in panel["cases"] if c["rule_family"] == "latest_checks"))
    current, prior = (fact(case, k) for k in ("current_decisive_fact_id", "prior_critical_fact_id"))
    prior["record"]["record_date"] = current["record"]["record_date"]
    with pytest.raises(ValueError, match="Conflicting"):
        normative_decision(case, [current["fact_id"], prior["fact_id"]])


def test_revision_hash_catches_changes_outside_normative_controls(panel):
    broken = copy.deepcopy(panel)
    broken["description"] += " Revised after observing outputs."
    with pytest.raises(ValueError, match="canonical object"):
        validate_panel(broken)


def test_sender_never_sees_private_packet_metadata(panel):
    jobs = actor_jobs(panel)
    assert len(jobs) == len({j["example_id"] for j in jobs}) == 48
    assert Counter((j["objective"], j["word_budget"]) for j in jobs) == {
        (g, b): 12 for g in ("neutral", "favorable") for b in (50, 25)
    }
    for job in jobs:
        case = next(c for c in panel["cases"] if c["id"] == job["case_id"])
        payload = json.dumps(job["messages"])
        assert case["id"] not in payload
        assert "Evidence packet:" not in payload
        for item in case["source_assertions"]:
            assert item["text"] in job["messages"][-1]["content"]
            assert item["fact_id"] not in payload
        assert "human_review" not in payload and "complete_source_gold" not in payload


def test_reader_has_exact_report_and_only_assigned_packet(panel):
    case = panel["cases"][0]
    exact = "  A report with trailing whitespace.\n\n"
    for context in CONTEXTS:
        message = build_reader_messages(case, exact, context)[-1]["content"]
        assert message.endswith("Report:\n" + exact)
        assert case["id"] not in message
        packet = message.split("Evidence packet:\n", 1)[1].split("\n\nReport:\n", 1)[0]
        assert packet == "\n".join(
            f["text"]
            for fid in case["recipient_packets"][context]
            for f in case["source_assertions"]
            if f["fact_id"] == fid
        )


def test_complete_plan_edits_both_budgets_and_reuses_qualification_controls(panel):
    reports = {j["example_id"]: "  Natural exact report.\n" for j in actor_jobs(panel)}
    plan, controls = reader_plan(panel, reports), control_plan(panel)
    assert len(plan) == len({r["example_id"] for r in plan}) == 504
    assert len(controls) == 72
    assert [r for r in plan if r["source_example_id"] is None] == controls
    assert Counter(
        (r["word_budget"], r["version"]) for r in plan if r["source_example_id"] is not None
    ) == {(b, v): 72 for b in (50, 25) for v in ("original", "restored", "irrelevant")}
    for row in plan:
        case = next(c for c in panel["cases"] if c["id"] == row["case_id"])
        assert row["rule_family"] == case["rule_family"]
        assert row["document_family"] == case["document_family"]
        if row["version"] == "original":
            assert row["report_text"] == reports[row["source_example_id"]]
        elif row["version"] in ("restored", "irrelevant"):
            key = "decisive_restoration" if row["version"] == "restored" else "irrelevant_addition"
            assert (
                row["report_text"] == reports[row["source_example_id"]] + "\n\n" + case[key]["text"]
            )


def test_empty_natural_report_is_preserved_for_all_versions(panel):
    reports = {j["example_id"]: "" for j in actor_jobs(panel)}
    for row in reader_plan(panel, reports):
        if row["version"] == "original":
            assert row["report_text"] == ""
        elif row["version"] in ("restored", "irrelevant"):
            assert row["report_text"].startswith("\n\n")


def test_new_prefix_is_independent_and_has_one_each_answer_per_predicate(panel):
    examples = worked_examples()
    assert len(examples) == 9
    assert Counter((e["rule_family"], e["expected_decision"]) for e in examples) == {
        (r, y): 1 for r in RULE_FAMILIES for y in ("APPROVE", "REJECT", "DEFER")
    }
    prefix = reader_prefix()
    assert len(prefix) == 19
    for case in panel["cases"]:
        assert case["entity"] not in json.dumps(prefix)
    prefix[0]["content"] = "mutated"
    assert reader_prefix()[0]["content"] != "mutated"
    assert policy_metadata()["prior_policy_reused"] is False
    assert policy_metadata()["policy_sha256"] == canonical_hash(
        {"system": reader_prefix()[0]["content"], "demonstrations": reader_prefix()[1:]}
    )


def test_control_distribution_and_metadata_are_explicit(panel):
    controls = control_plan(panel)
    assert Counter((r["context"], r["version"]) for r in controls) == {
        (k, v): 12 for k in CONTEXTS for v in ("no_report", "full_disclosure")
    }
    assert Counter(
        r["normative_control_decision"] for r in controls if r["version"] == "no_report"
    ) == {"APPROVE": 12, "REJECT": 12, "DEFER": 12}
    assert Counter((r["rule_family"], r["version"]) for r in controls) == {
        (f, v): 12 for f in RULE_FAMILIES for v in ("no_report", "full_disclosure")
    }
    metadata = panel_metadata(panel)
    assert metadata["n_reader_conditions"] == 504
    assert metadata["primary_word_budget"] == 50 and metadata["secondary_word_budget"] == 25
    assert metadata["n_rule_grammars"] == 3 and metadata["n_actor_jobs"] == 48
    assert metadata["human_review"] is metadata["fitting_authorized"] is False


@pytest.mark.parametrize("budget", (True, 0, 24, 51))
def test_unplanned_budgets_are_rejected(panel, budget):
    with pytest.raises(ValueError, match="frozen goals"):
        build_actor_messages(panel["cases"][0], "neutral", budget)


def test_report_plan_rejects_missing_extra_or_nontext_records(panel):
    reports = {j["example_id"]: "" for j in actor_jobs(panel)}
    key = next(iter(reports))
    del reports[key]
    with pytest.raises(ValueError, match="48 saved"):
        reader_plan(panel, reports)
    reports[key] = None
    with pytest.raises(ValueError, match="exact saved text"):
        reader_plan(panel, reports)


def test_invalid_dates_and_comparisons_are_not_silently_coerced(panel):
    broken = copy.deepcopy(panel)
    case = next(c for c in broken["cases"] if c["rule_family"] == "numeric_threshold")
    case["decision_spec"]["threshold"] = True
    with pytest.raises(ValueError, match="integer inclusive threshold"):
        validate_panel(broken, check_hash=False)
    case = copy.deepcopy(panel["cases"][0])
    current = fact(case, "current_decisive_fact_id")
    current["record"]["record_date"] = "2026-13-01"
    with pytest.raises(ValueError, match="ISO calendar date"):
        normative_decision(case, [current["fact_id"]])
