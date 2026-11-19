"""Reader-panel integrity checks using scripted fixtures, never model findings."""

import copy
import csv
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from latent_probes.backends import GenerationResult
from latent_probes.reader_panel import (
    CONTEXTS,
    VERSIONS,
    build_reader_messages,
    load_panel,
    parse_decision,
    plan_panel,
    run_panel,
    summarize_panel,
    validate_panel,
)
from latent_probes.storage import canonical_hash

DATA = Path(__file__).resolve().parents[1] / "data" / "reader_panel" / "four_cases_v1.json"


@pytest.fixture
def panel():
    return load_panel(DATA)


def fixture_records(panel, choices=None):
    """Construct explicit test records; the choices are not expected research outcomes."""
    choices = choices or {}
    records = []
    for job in plan_panel(panel)["jobs"]:
        text = choices.get((job["case_id"], job["context"], job["version"]), "DEFER")
        records.append(
            {
                **job,
                "generation": {"text": text},
                "decision": parse_decision(text),
                "is_test_fixture": True,
            }
        )
    return records


class ScriptedBackend:
    """No model loading or inference. Mutates received histories to detect reuse."""

    def __init__(self, fail_after=None):
        self.metadata = {
            "backend": "scripted_unit_test_fixture",
            "model_id": "fixture-not-a-model",
            "is_test_fixture": True,
        }
        self.events = []
        self.fail_after = fail_after
        self.closed = False

    def generate(self, messages, settings, seed, capture=False):
        if self.fail_after is not None and len(self.events) >= self.fail_after:
            raise RuntimeError("scripted fixture failure")
        self.events.append(
            {
                "messages": copy.deepcopy(messages),
                "settings": asdict(settings),
                "seed": seed,
                "capture": capture,
            }
        )
        messages.append({"role": "assistant", "content": "fixture history mutation"})
        text = ("APPROVE", "REJECT.", "DEFER", "APPROVE, because fixture")[
            (len(self.events) - 1) % 4
        ]
        return GenerationResult(
            text=text,
            prompt_token_ids=[101, 102],
            generated_token_ids=[201],
            finish_reason="eos",
        )

    def close(self):
        self.closed = True


class FixtureTokenizer:
    name_or_path = "fixture-word-tokenizer"
    chat_template = "fixture-template-only"

    @staticmethod
    def encode(text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(range(len(text.split())))

    @staticmethod
    def apply_chat_template(messages, tokenize=True, add_generation_prompt=True):
        assert tokenize is True and add_generation_prompt is True
        return list(range(2 + sum(len(message["content"].split()) for message in messages)))


def test_frozen_factorial_has_unique_conditions_and_no_observed_targets(panel):
    planned = plan_panel(panel)
    assert planned["expected_conditions"] == 48
    assert len({job["example_id"] for job in planned["jobs"]}) == 48
    assert {case["family"] for case in panel["cases"]} == {
        "conjunction",
        "threshold",
        "temporal_precedence",
        "arithmetic",
    }
    assert sorted(case["correct"] for case in panel["cases"]) == [
        "APPROVE",
        "APPROVE",
        "REJECT",
        "REJECT",
    ]
    assert all(job["seed"] == 42 for job in planned["jobs"])
    assert all("decision" not in job and "correct" not in job for job in planned["jobs"])
    assert not any(
        item["token_lengths_measured"] for item in planned["length_diagnostics"].values()
    )
    with pytest.raises(ValueError, match="frozen"):
        plan_panel(panel, seed=43)


@pytest.mark.parametrize(
    "corruption",
    [
        "report",
        "unsupported_fact",
        "missing_context",
        "duplicate",
        "observed_target",
        "bad_gold",
    ],
)
def test_schema_blocks_broken_edit_support_or_factorial(panel, corruption):
    bad = copy.deepcopy(panel)
    first = bad["cases"][0]
    if corruption == "report":
        first["report_versions"]["restored"] += " An extra unsupported claim."
    elif corruption == "unsupported_fact":
        first["source"].remove(first["irrelevant_fact"])
    elif corruption == "missing_context":
        del first["reader_packets"]["absent"]
    elif corruption == "duplicate":
        bad["cases"][1]["id"] = first["id"]
    elif corruption == "observed_target":
        first["expected_reader_decision"] = "APPROVE"
    else:
        first["correct"] = ["APPROVE"]
    with pytest.raises(ValueError):
        validate_panel(bad)


def test_source_gold_and_notes_cannot_leak_into_reader_messages(panel):
    case = panel["cases"][1]
    messages = build_reader_messages(case, "absent", "original")
    assert "82" not in json.dumps(messages)
    assert "78" not in json.dumps(messages)
    hidden = copy.deepcopy(case)
    hidden["source"].append("HIDDEN_SOURCE_SENTINEL")
    hidden["authoring_notes"] = "HIDDEN_AUTHORING_SENTINEL"
    hidden["correct"] = "REJECT"
    assert build_reader_messages(hidden, "absent", "original") == messages
    assert len(messages) == 2
    assert [row["role"] for row in messages] == ["system", "user"]
    assert "Evidence packet:" in messages[1]["content"]
    assert "Source:" not in messages[1]["content"]


def test_report_is_identical_across_reader_audiences_and_histories_are_fresh(panel):
    for case in panel["cases"]:
        for version in VERSIONS:
            histories = [build_reader_messages(case, context, version) for context in CONTEXTS]
            reports = [history[1]["content"].split("\n\nReport:\n", 1)[1] for history in histories]
            assert len(set(reports)) == 1
            expected = case["report_versions"][version] or "No report was supplied."
            assert reports[0] == expected
            histories[0][1]["content"] = "MUTATED"
            assert histories[1][1]["content"] != "MUTATED"
            assert build_reader_messages(case, CONTEXTS[0], version)[1]["content"] != "MUTATED"


def test_supported_edits_only_append_the_declared_source_facts(panel):
    for case in panel["cases"]:
        assert case["decisive_fact"] in case["source"]
        assert case["irrelevant_fact"] in case["source"]
        assert case["report_versions"]["restored"] == case["report"] + " " + case["decisive_fact"]
        assert case["report_versions"]["irrelevant_edit"] == (
            case["report"] + " " + case["irrelevant_fact"]
        )
        assert abs(len(case["decisive_fact"].split()) - len(case["irrelevant_fact"].split())) <= 1


@pytest.mark.parametrize(
    "text,expected",
    [
        ("APPROVE", "APPROVE"),
        (" REJECT.\n", "REJECT"),
        ("DEFER!", "DEFER"),
        ("APPROVE?", "APPROVE"),
        ("approve", "INVALID"),
        ("Approve", "INVALID"),
        ("APPROVE because it passed", "INVALID"),
        ("APPROVE,", "INVALID"),
        ("APPROVE..", "INVALID"),
        ("APPROVE or REJECT", "INVALID"),
        ('{"decision":"APPROVE"}', "INVALID"),
        ("", "INVALID"),
        (None, "INVALID"),
    ],
)
def test_parser_does_not_repair_invalid_model_outputs(text, expected):
    assert parse_decision(text) == expected


def test_paired_accounting_separates_new_error_deferral_and_invalid_output(panel):
    ids = [case["id"] for case in panel["cases"]]
    fixture_choices = {
        (ids[0], "stale", "none"): "APPROVE",
        (ids[0], "stale", "original"): "APPROVE",
        (ids[0], "stale", "restored"): "REJECT",
        (ids[0], "stale", "irrelevant_edit"): "APPROVE",
        (ids[1], "stale", "original"): "REJECT",
        (ids[1], "stale", "irrelevant_edit"): "REJECT",
        (ids[2], "stale", "none"): "APPROVE",
        (ids[2], "stale", "restored"): "APPROVE because fixture",
        (ids[2], "stale", "irrelevant_edit"): "REJECT",
        (ids[3], "stale", "none"): "REJECT",
        (ids[3], "stale", "original"): "APPROVE",
        (ids[3], "stale", "restored"): "REJECT",
        (ids[3], "stale", "irrelevant_edit"): "APPROVE",
    }
    summary = summarize_panel(panel, fixture_records(panel, fixture_choices))
    counts = summary["overall"]
    assert counts["conditions"] == 48
    assert sum(counts[key] for key in ("correct", "wrong", "defer", "invalid")) == 48
    contrasts = {
        (c["context"], c["before_version"], c["after_version"]): c
        for c in summary["paired_contrasts"]
    }
    induction = contrasts[("stale", "none", "original")]
    assert induction["wrong_count_change"] == 2
    assert induction["persistent_wrong"] == 1
    assert induction["new_wrong"] == 2
    repair = contrasts[("stale", "irrelevant_edit", "restored")]
    assert repair["wrong_count_change"] == -4
    assert repair["correct_count_change"] == 2
    assert repair["defer_count_change"] == 1
    assert repair["invalid_count_change"] == 1
    assert repair["wrong_rate_change"] == -1.0
    assert repair["outcome_transitions"] == {
        "wrong->correct": 2,
        "wrong->defer": 1,
        "wrong->invalid": 1,
    }
    assert "confidence interval" in summary["warnings"][0]
    assert "evidence-supported" in summary["interpretation"]


@pytest.mark.parametrize(
    "corruption",
    [
        "missing",
        "duplicate",
        "extra",
        "raw_disagrees",
        "condition",
        "gold",
        "report",
        "outcome",
        "messages",
        "seed",
        "family",
    ],
)
def test_summary_refuses_missing_or_tampered_conditions(panel, corruption):
    rows = fixture_records(panel)
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif corruption == "extra":
        rows[0]["example_id"] = "unknown"
    elif corruption == "raw_disagrees":
        rows[0]["decision"] = "APPROVE"
    elif corruption == "condition":
        rows[0]["context"] = "stale"
    elif corruption == "gold":
        rows[0]["world_correct_decision"] = "APPROVE"
    elif corruption == "report":
        rows[0]["report"] = "A different report."
    elif corruption == "outcome":
        rows[0]["objective_outcome"] = "correct"
    elif corruption == "messages":
        rows[0]["messages"][1]["content"] += " A hidden instruction."
    elif corruption == "seed":
        rows[0]["seed"] = 43
    else:
        rows[0]["family"] = "different_family"
    with pytest.raises(ValueError):
        summarize_panel(panel, rows)


def test_summary_checks_present_record_hash_before_accepting_valid_decision_change(panel):
    rows = fixture_records(panel)
    rows[0]["record_sha256"] = canonical_hash(rows[0])
    summarize_panel(panel, rows)
    rows[0]["generation"]["text"] = "APPROVE"
    rows[0]["decision"] = "APPROVE"
    with pytest.raises(ValueError, match="hash"):
        summarize_panel(panel, rows)


def test_optional_tokenizer_records_real_lengths_without_enforcing_exact_match(panel):
    planned = plan_panel(panel, tokenizer=FixtureTokenizer())
    assert all(job["prompt_tokens"] > 0 for job in planned["jobs"])
    assert planned["tokenizer"]["name_or_path"] == "fixture-word-tokenizer"
    for case in panel["cases"]:
        sizes = planned["length_diagnostics"][case["id"]]
        assert sizes["token_lengths_measured"] is True
        assert sizes["report_tokens"]["none"] == 0
        assert sizes["addition_tokens"]["restored"] == len(case["decisive_fact"].split())
    assert any(
        size["addition_tokens"]["restored"] != size["addition_tokens"]["irrelevant_edit"]
        for size in planned["length_diagnostics"].values()
    )


def test_run_saves_all_raw_conditions_and_preserves_backend_lifetime(panel, tmp_path):
    backend = ScriptedBackend()
    output = tmp_path / "fixture-panel"
    result = run_panel(panel, output, backend, tokenizer=FixtureTokenizer())
    assert result["status"] == "complete" and result["conditions"] == 48
    assert result["is_test_fixture"] is True
    assert len(backend.events) == 48 and backend.closed is False
    for event in backend.events:
        assert len(event["messages"]) == 2
        assert event["settings"]["do_sample"] is False
        assert event["settings"]["max_new_tokens"] == 8
        assert event["capture"] is False and event["seed"] == 42
    assert len(list((output / "records").glob("*.json"))) == 48
    rows = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
    assert all(len(row["messages"]) == 2 and row["observed_prompt_tokens"] == 2 for row in rows)
    assert all(row["is_test_fixture"] is True for row in rows)
    assert all(row["record_sha256"] for row in rows)
    with (output / "decisions.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 48
    assert any(row["raw_text"] == "APPROVE, because fixture" for row in csv_rows)
    manifest = json.loads(Path(result["paths"]["manifest"]).read_text())
    assert manifest["status"] == "complete" and manifest["completed_conditions"] == 48
    assert manifest["records_sha256"] and manifest["summary_sha256"]
    assert result["summary"]["is_test_fixture"] is True
    with pytest.raises(FileExistsError):
        run_panel(panel, output, backend)
    assert len(backend.events) == 48


def test_failure_keeps_completed_raw_records_and_does_not_publish_summary(panel, tmp_path):
    backend = ScriptedBackend(fail_after=3)
    output = tmp_path / "interrupted-fixture"
    with pytest.raises(RuntimeError, match="scripted fixture failure"):
        run_panel(panel, output, backend)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["completed_conditions"] == 3
    assert manifest["is_test_fixture"] is True
    assert len(list((output / "records").glob("*.json"))) == 3
    assert not (output / "summary.json").exists()
    assert backend.closed is False


def test_backend_must_explicitly_distinguish_fixture_from_real_model(panel, tmp_path):
    backend = ScriptedBackend()
    del backend.metadata["is_test_fixture"]
    with pytest.raises(ValueError, match="is_test_fixture"):
        run_panel(panel, tmp_path / "ambiguous", backend)
    assert not (tmp_path / "ambiguous").exists()
