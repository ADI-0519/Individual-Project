"""Fixed-text factorial controls and paired accounting; no model claims."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from latent_probes.apollo import capture_signature
from latent_probes.context_probe import context_jobs, paired_contrasts, run_context_probe
from latent_probes.reader_panel import load_panel


def panel():
    return load_panel(Path(__file__).resolve().parents[1] / "data/reader_panel/four_cases_v1.json")


def test_report_stays_identical_across_goals_and_attributed_knowledge():
    data = panel()
    jobs = context_jobs(data)
    assert len(jobs) == 48 == len({job["example_id"] for job in jobs})
    for case in data["cases"]:
        for version in ("original", "restored"):
            selected = [j for j in jobs if j["case_id"] == case["id"] and j["version"] == version]
            assert len(selected) == 6
            assert {j["response"] for j in selected} == {case["report_versions"][version]}
            for job in selected:
                assert case["decisive_fact"] in job["messages"][-1]["content"]
                assert all(fact in job["messages"][-1]["content"] for fact in case["source"])
    assert "label" not in jobs[0] and "correct" not in jobs[0]


def test_contrasts_hold_the_other_factors_fixed_and_reject_incomplete_data():
    jobs = context_jobs(panel())
    # Distinct known offsets make cross-condition mix-ups observable.
    scores = {
        j["example_id"]: 10 * (j["goal"] == "favorable")
        + {"informed": 0, "absent": 2, "stale": 5}[j["context"]]
        + 100 * (j["version"] == "restored")
        for j in jobs
    }
    contrasts = paired_contrasts(jobs, scores)
    for row in contrasts:
        expected = {
            "favorable_minus_neutral": 10,
            "absent_minus_informed": 2,
            "stale_minus_informed": 5,
        }[row["contrast"]]
        assert row["raw_margin_difference"] == expected
    incomplete = dict(scores)
    incomplete.pop(jobs[0]["example_id"])
    with pytest.raises(ValueError, match="one score"):
        paired_contrasts(jobs, incomplete)
    duplicate = copy.deepcopy(jobs)
    duplicate.append(duplicate[0])
    with pytest.raises(ValueError, match="one score"):
        paired_contrasts(duplicate, scores)


def test_failed_capture_keeps_auditable_partial_records_and_frozen_probe(tmp_path):
    class FailingFixture:
        metadata = {"is_test_fixture": True, "capture_layers": [0]}
        calls = 0

        def forced_token_activations(self, messages, response, trim_last_words):
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("scripted capture failure")
            return {"layer_0_tokens": np.ones((2, 1))}, {"text": response}

    backend = FailingFixture()
    probe = tmp_path / "probe"
    probe.mkdir()
    (probe / "probe.npz").write_bytes(b"immutable test-only artifact; never scored")
    (probe / "manifest.json").write_text(
        json.dumps({"feature_signature": capture_signature(backend.metadata, 0)}), encoding="utf-8"
    )
    output = tmp_path / "context"
    with pytest.raises(RuntimeError, match="scripted capture failure"):
        run_context_probe(panel(), output, probe, backend)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["backend"]["is_test_fixture"] is True
    assert len(list((output / "records").glob("*.json"))) == 2
    assert (output / "plan.json").exists()
    assert (probe / "probe.npz").read_bytes() == b"immutable test-only artifact; never scored"
