"""Synthetic analysis and fake-backend guards; no language model is executed."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from latent_probes.config import load_config
from latent_probes.evidence_role_panel import authored_panel, jobs
from latent_probes.storage import read_json

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evidence_role_runner", ROOT / "scripts/run_evidence_role_diagnostic.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def analysis_rows(pattern="value_following"):
    rows = []
    decisive = {
        (job["case_id"], job["side"]): job["expected_decision"]
        for job in jobs(authored_panel())
        if job["condition"] == "decisive"
    }
    for job in jobs(authored_panel()):
        if job["condition"] == "decisive":
            decision = job["expected_decision"]
        elif pattern == "value_following" and job["condition"] == "metadata":
            decision = decisive[job["case_id"], job["side"]]
        elif pattern == "fixed_fallback":
            decision = "REJECT"
        else:
            decision = "DEFER"
        rows.append({**job, "decision": decision})
    return rows


def test_all_96_conditions_and_opposing_date_score_polarities():
    rows = analysis_rows()
    assert len(rows) == len({row["example_id"] for row in rows}) == 96
    assert len({row["case_id"] for row in rows}) == 16
    for family, low, high in (("date", "APPROVE", "REJECT"), ("score", "REJECT", "APPROVE")):
        decisive = [
            row for row in rows if row["family"] == family and row["condition"] == "decisive"
        ]
        assert len(decisive) == 16
        assert {row["expected_decision"] for row in decisive if row["side"] == "low"} == {low}
        assert {row["expected_decision"] for row in decisive if row["side"] == "high"} == {high}
    summary = runner.summarise(rows)
    assert summary["n_calls"] == 96 and summary["n_cases"] == 16
    assert not summary["qualifies_reader"]
    assert summary["new_probe_fits"] == summary["untouched_transfer_calls"] == 0


def test_value_following_differs_from_explicit_missing_deferral():
    summary = runner.summarise(analysis_rows())
    for family in ("date", "score"):
        paired = summary["paired"][family]
        assert paired["n_cases"] == 8
        assert (
            paired["metadata_definite_flip_cases"] == paired["metadata_value_following_cases"] == 8
        )
        assert paired["explicit_missing_value_following_cases"] == 0
        assert paired["deferral_intervention"]["mean"] == 1.0
        assert paired["deferral_intervention"]["case_bootstrap_percentile_95"] == [1.0, 1.0]
        assert all(case["pairs"]["explicit_missing"]["both_defer"] for case in paired["cases"])


def test_constant_definite_fallback_is_not_value_substitution():
    summary = runner.summarise(analysis_rows("fixed_fallback"))
    for family in ("date", "score"):
        assert summary["paired"][family]["metadata_definite_flip_cases"] == 0
        assert summary["paired"][family]["metadata_value_following_cases"] == 0
        assert summary["paired"][family]["deferral_intervention"]["mean"] == 0.0
        for case in summary["paired"][family]["cases"]:
            assert not case["pairs"]["metadata"]["any_flip"]
            assert not case["pairs"]["metadata"]["both_defer"]


def test_invalid_and_defer_changes_remain_distinct_from_definite_flips():
    rows = analysis_rows("both_defer")
    changed = next(
        row
        for row in rows
        if row["family"] == "date" and row["condition"] == "metadata" and row["side"] == "low"
    )
    changed["decision"] = "INVALID"
    summary = runner.summarise(rows)
    assert summary["groups"]["date/metadata"]["invalid"] == 1
    pair = next(
        case for case in summary["paired"]["date"]["cases"] if case["case_id"] == changed["case_id"]
    )
    assert pair["pairs"]["metadata"]["any_flip"]
    assert not pair["pairs"]["metadata"]["definite_flip"]
    assert pair["explicit_missing_minus_metadata_deferral"] == 0.5


@pytest.mark.parametrize(
    "corruption", ["missing", "duplicate_id", "missing_side", "mixed_case_family"]
)
def test_analysis_rejects_incomplete_or_cross_family_pairings(corruption):
    rows = analysis_rows()
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate_id":
        rows[-1]["example_id"] = rows[0]["example_id"]
    elif corruption == "missing_side":
        rows[0]["side"] = "high"
    else:
        # Preserve all six family/condition row totals while corrupting case membership.
        date_row = next(
            row for row in rows if row["family"] == "date" and row["condition"] == "decisive"
        )
        score_row = next(
            row for row in rows if row["family"] == "score" and row["condition"] == "decisive"
        )
        date_row["family"], score_row["family"] = "score", "date"
    with pytest.raises(ValueError):
        runner.summarise(rows)


@pytest.mark.parametrize(
    "text,finish,expected",
    [
        ("APPROVE", "eos", "APPROVE"),
        (" REJECT. ", "eos", "REJECT"),
        ("DEFER!", "eos", "DEFER"),
        ("APPROVE", "length", "INVALID"),
        ("DEFER", "length", "INVALID"),
        ("APPROVE because it passed", "eos", "INVALID"),
        ("approve", "eos", "INVALID"),
        ("APPROVE or REJECT", "eos", "INVALID"),
    ],
)
def test_original_strict_parser_retains_caps_and_format_failures(text, finish, expected):
    assert runner.strict_decision({"text": text, "finish_reason": finish}) == expected


def run_fixture(
    tmp_path, monkeypatch, *, source_change=False, oversized=False, prompt_mismatch=False
):
    panel = authored_panel()
    original_settings = load_config(ROOT / "configs/qwen7b_reader_4bit.json")
    project = tmp_path / "project"
    paths = [
        "src/latent_probes/evidence_role_panel.py",
        "src/latent_probes/calibrated_reader.py",
        "src/latent_probes/forecast_panel.py",
        "src/latent_probes/reader_panel.py",
        "src/latent_probes/backends.py",
        "configs/qwen7b_reader_4bit.json",
        "docs/EVIDENCE_ROLE_DIAGNOSTIC_PROTOCOL.md",
    ]
    for relative in paths:
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("explicit synthetic source fixture\n", encoding="utf-8")
    panel_path = project / "panel.json"
    panel_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", project)
    monkeypatch.setattr(runner, "load_panel", lambda _: copy.deepcopy(panel))
    monkeypatch.setattr(runner, "load_config", lambda _: original_settings)

    class TokenizerFixture:
        def apply_chat_template(self, *_args, **_kwargs):
            return {"input_ids": np.ones((1, 2049 if oversized else 20), dtype=np.int64)}

    class BackendFixture:
        """Exercise runner guards in a temp directory; this is not real inference."""

        def __init__(self, settings):
            self.calls, self.closed = 0, False
            self.tokenizer = TokenizerFixture()
            self.metadata = {"is_test_fixture": False, "resolved_revision": settings.revision}
            if source_change:
                (project / paths[0]).write_text("changed after freeze\n", encoding="utf-8")

        def generate(self, _messages, settings, seed):
            self.calls += 1
            assert settings.max_new_tokens == 8 and not settings.do_sample and seed == 42
            record = {
                "text": "DEFER",
                "finish_reason": "eos",
                "prompt_token_ids": [2 if prompt_mismatch else 1] * 20,
                "generated_token_ids": [17, 18],
            }
            return SimpleNamespace(record=lambda: record)

        def close(self):
            self.closed = True

    backend = BackendFixture(original_settings.model)
    # Source mutation must happen inside construction, after protocol hashing.
    if source_change:
        (project / paths[0]).write_text("explicit synthetic source fixture\n", encoding="utf-8")

    def factory(settings):
        if source_change:
            (project / paths[0]).write_text("changed after freeze\n", encoding="utf-8")
        return backend

    return SimpleNamespace(panel=panel_path, output=tmp_path / "attempt"), factory, backend


def test_source_mutation_fails_before_any_generation_and_preserves_failed_manifest(
    tmp_path, monkeypatch
):
    args, factory, backend = run_fixture(tmp_path, monkeypatch, source_change=True)
    with pytest.raises(ValueError, match="changed before generation"):
        runner.run(args, backend_factory=factory)
    assert backend.calls == 0 and backend.closed
    assert read_json(args.output / "manifest.json")["status"] == "failed"
    assert not read_json(args.output / "manifest.json")["source_unchanged"]


@pytest.mark.parametrize("mode", ["oversized", "prompt_mismatch"])
def test_context_and_exact_prompt_guards_stop_an_invalid_attempt(tmp_path, monkeypatch, mode):
    args, factory, backend = run_fixture(tmp_path, monkeypatch, **{mode: True})
    with pytest.raises(ValueError):
        runner.run(args, backend_factory=factory)
    assert backend.closed
    assert backend.calls == (0 if mode == "oversized" else 1)
    assert read_json(args.output / "manifest.json")["status"] == "failed"


def test_complete_fixture_attempt_covers_every_condition_and_cannot_overwrite(
    tmp_path, monkeypatch
):
    args, factory, backend = run_fixture(tmp_path, monkeypatch)
    result = runner.run(args, backend_factory=factory)
    assert result["status"] == "complete" and result["source_unchanged"]
    assert backend.calls == 96 and backend.closed
    assert len(read_json(args.output / "rows.json")) == 96
    assert len(read_json(args.output / "prompt_lengths.json")) == 96
    summary = read_json(args.output / "summary.json")
    assert not summary["qualifies_reader"] and summary["new_probe_fits"] == 0
    with pytest.raises(FileExistsError):
        runner.run(args, backend_factory=factory)
