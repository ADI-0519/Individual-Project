"""Prevent semantic mismatches, stale captures and invalid causal controls."""

import importlib.util
import json
from copy import deepcopy

import pytest

from latent_probes import format_bridge as bridge
from latent_probes.fresh_readout_bank import ROOT, iter_rows, load_bank


@pytest.fixture(scope="module")
def source_rows():
    bank = load_bank(ROOT / "data/belief_main/fresh_readout_v2/compiled_v1/bank.json")
    return list(iter_rows(bank, "test"))


@pytest.fixture
def pair(source_rows):
    return deepcopy(bridge.paired_donors(source_rows)[0])


def test_all_exposed_correction_states_have_eight_complete_matched_inputs(source_rows):
    rows = bridge.build_rows(source_rows)
    assert len(rows) == 12288
    assert len({r["record_id"] for r in rows}) == 12288
    assert len({r["group_id"] for r in rows}) == 24
    assert all(r["row_id"] == r["case_id"] == r["record_id"] for r in rows)
    assert all(not r["human_review_complete"] for r in rows)


def test_diagonals_exact_and_query_changes_only_question(pair):
    rows = {r["renderer"]: r for r in bridge.cross_pair(pair)}
    for variant in ("AAA", "BBB"):
        donor = pair[variant[0]]
        assert rows[variant]["messages"] == donor["messages"]
        assert rows[variant]["user_content_char_spans"] == donor["user_content_char_spans"]
    for system in "AB":
        for narrative in "AB":
            a, b = (rows[system + narrative + q] for q in "AB")
            assert a["state"] == b["state"] and a["gold"] == b["gold"]
            assert a["messages"][0] == b["messages"][0]
            end = a["user_content_char_spans"]["account"][1]
            assert a["messages"][1]["content"][:end] == b["messages"][1]["content"][:end]


def test_stale_capture_metadata_is_removed_and_new_journal_ids_are_unique(pair):
    for row in pair.values():
        row.update(
            row_id="old_journal",
            prompt_token_ids=[1, 2],
            output_score=99,
            token_positions={"account_last": 0},
            messages_sha256="old",
        )
    rows = bridge.cross_pair(pair)
    assert len({r["row_id"] for r in rows}) == 8
    for row in rows:
        assert (
            not {"prompt_token_ids", "token_positions", "output_score", "messages_sha256"}
            & row.keys()
        )


@pytest.mark.parametrize("change", ["gold", "state", "span", "location", "duplicate", "missing"])
def test_mismatches_fail_before_crossing(pair, change):
    if change == "gold":
        pair["B"]["gold"] = 1 - pair["B"]["gold"]
    elif change == "state":
        pair["B"]["state"]["world"] = 1 - pair["B"]["state"]["world"]
    elif change == "span":
        pair["B"]["user_content_char_spans"]["query_location"][1] = 10**8
    elif change == "location":
        row = pair["B"]
        left, right = row["user_content_char_spans"]["query_location"]
        text = row["messages"][1]["content"]
        row["messages"][1]["content"] = text[:left] + "X" * (right - left) + text[right:]
    rows = list(pair.values())
    if change == "duplicate":
        rows.append(deepcopy(pair["A"]))
    elif change == "missing":
        rows.pop()
    with pytest.raises(ValueError):
        bridge.build_rows(rows)


def test_binary_state_and_label_types_are_strict(pair):
    for row in pair.values():
        row["gold"] = float(row["gold"])
    with pytest.raises(ValueError, match="integer binary gold"):
        bridge.build_rows(pair.values())


def test_complete_token_prefix_not_just_text_is_checked(pair):
    rows = bridge.cross_pair(pair)
    for row in rows:
        row.update(prompt_token_ids=[1, 2, 3, 4], token_positions={"account_last": 2})
    assert bridge.check_query_prefix_invariance(rows) == 4
    rows[1]["prompt_token_ids"] = [1, 9, 3, 4]
    with pytest.raises(ValueError, match="causal account prefix"):
        bridge.check_query_prefix_invariance(rows)
    with pytest.raises(ValueError, match="No query-prefix"):
        bridge.check_query_prefix_invariance([])


def _preparation_script():
    spec = importlib.util.spec_from_file_location(
        "bridge_preparation_test", ROOT / "scripts/prepare_format_bridge.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preparation_records_exposure_and_refuses_to_overwrite(tmp_path):
    script = _preparation_script()
    output = tmp_path / "new_attempt"
    result = script.prepare(
        ROOT / "data/belief_main/fresh_readout_v2/compiled_v1/bank.json", output
    )
    assert result["prompt_variants"] == 12288 and result["review_items"] == 128
    assert result["model_calls"] == 0 and result["classifier_fits"] == 0
    assert result["exposed_contexts"] and not result["fresh_confirmation"]
    assert result["tokenizer_checks"] == {}
    before = (output / "preparation.json").read_bytes()
    with pytest.raises(FileExistsError):
        script.prepare(ROOT / "data/belief_main/fresh_readout_v2/compiled_v1/bank.json", output)
    assert (output / "preparation.json").read_bytes() == before


def test_failed_input_gate_preserves_its_attempt(tmp_path, monkeypatch, pair):
    script = _preparation_script()
    monkeypatch.setattr(script, "iter_rows", lambda *_args: iter([pair["A"]]))
    output = tmp_path / "invalid_attempt"
    with pytest.raises(ValueError, match="Missing paired donor"):
        script.prepare(ROOT / "data/belief_main/fresh_readout_v2/compiled_v1/bank.json", output)
    assert (output / "started.json").exists()
    failure = json.loads((output / "failure.json").read_text())
    assert failure["exception_type"] == "ValueError" and failure["model_calls"] == 0
    assert not (output / "preparation.json").exists()


def test_completed_reference_uses_source_bindings_not_diagnostic_input_schema():
    script = _preparation_script()
    manifest = {
        "schema": "fresh-readout-analysis-manifest-v2",
        "source_sha256": {"/native/llama/test/rows.json": "original_digest"},
    }
    assert script.reference_rows_digest(manifest, "llama") == "original_digest"
    with pytest.raises(ValueError, match="source bindings"):
        script.reference_rows_digest({"schema": manifest["schema"], "input_sha256": {}}, "llama")
    with pytest.raises(ValueError, match="Ambiguous"):
        script.reference_rows_digest(manifest, "qwen")
