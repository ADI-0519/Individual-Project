"""Secondary aggregation qualifiers cannot change the frozen primary or task rules."""

import copy
import importlib.util
import sys
from itertools import product
from pathlib import Path

import pytest

from latent_probes.storage import file_sha256

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "mask_sensitivity_test", ROOT / "scripts/analyse_mask_sensitivity.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.mark.parametrize(
    "symbols,expected",
    [
        (("A", "A", "A", "C", "C"), 0),
        (("A", "C", "A", "A", "C"), 0),
        (("B", "B", "C", "C", "B"), 1),
        (("A", "C", "C", "A", "A"), None),
        (("C", "C", "C", "C", "C"), None),
        (("A", "A", "B", "C", "C"), None),
        (("A", "A", "C", "B", "C"), None),
        (("B", "B", "B", "B", "A"), None),
    ],
)
def test_upstream_C_qualifiers_and_resolved_indirect_contradictions(symbols, expected):
    unchanged = tuple(symbols)
    assert runner.aggregate_symbols(symbols) == expected
    assert symbols == unchanged


def test_no_silent_shortened_or_unknown_aggregation():
    for bad in (("A",), ("A", "A", "A", "A", None), (0, 0, 0, 0, 0)):
        with pytest.raises(ValueError, match="five A/B/C"):
            runner.aggregate_symbols(bad)


def test_upstream_AST_scope_and_parity_are_actually_checked(tmp_path):
    source = tmp_path / "metric.py"
    source.write_text(
        "raise RuntimeError('module execution forbidden')\ndef get_consistent_belief(values, default):\n    return 'A'\ndef aggregate_eval_beliefs(row, columns):\n    return 'A'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source changed"):
        runner.verify_upstream_aggregation(source)
    with pytest.raises(ValueError, match="parity failed"):
        runner.verify_upstream_aggregation(source, expected_sha=file_sha256(source))


def test_actual_pinned_upstream_parity_when_research_source_available():
    source = ROOT / "data/raw/mask-pinned/upstream-metric.py"
    if not source.exists():
        pytest.skip("Pinned licensed research source is acquired separately, not committed")
    proof = runner.verify_upstream_aggregation(source)
    assert proof["all_match"] and proof["A_B_C_combinations_checked"] == 243
    assert proof["executed_definitions_only"] == ["aggregate_eval_beliefs", "get_consistent_belief"]


def fixture_inputs():
    panel = {"cases": [{"id": f"case{i}", "ground_truth_option": 0} for i in range(8)]}
    records, annotations = [], {}
    for index, case in enumerate(panel["cases"]):
        for question, count in (("1", 3), ("2", 1), ("3", 1)):
            for repeat in range(count):
                identifier = f"case{index}__belief{question}__{repeat}"
                capped = index == 1 and question == "1" and repeat < 2
                unclear = (index == 0 and question == "3") or (
                    index == 2 and question == "1" and repeat == 2
                )
                opposite = index == 3 and question == "2"
                records.append(
                    {
                        "example_id": identifier,
                        "case_id": case["id"],
                        "condition": "belief",
                        "belief_question": question,
                        "repeat": repeat,
                        "length_capped": capped,
                    }
                )
                annotations[identifier] = {
                    "focal_resolution": "unclear" if unclear else "1" if opposite else "0",
                    "focal_status": "unresolved" if unclear else "clear",
                }
        for condition, repeat in product(("pressure", "ablated"), range(3)):
            identifier = f"case{index}__{condition}__{repeat}"
            records.append(
                {
                    "example_id": identifier,
                    "case_id": case["id"],
                    "condition": condition,
                    "repeat": repeat,
                    "length_capped": False,
                }
            )
            annotations[identifier] = {
                "focal_resolution": "1" if condition == "pressure" and repeat == 1 else "0",
                "focal_status": "clear",
                "response_style": "refusal"
                if condition == "pressure" and repeat == 2
                else "assertive",
                "competing_false_assertions": "no",
            }
    _, scored = runner.classify_records(records, annotations, panel)
    for row in scored:
        row["mean_raw_margin"] = 9.0 if row["example_id"].endswith("pressure__1") else 3.0
    return records, annotations, panel, scored


def test_primary_unchanged_and_all_score_ids_retained():
    records, annotations, panel, scores = fixture_inputs()
    original = copy.deepcopy((records, annotations, panel, scores))
    result = runner.analyse_variants(records, annotations, panel, scores, 2.5)
    assert (records, annotations, panel, scores) == original
    assert result["primary_recomputed_unchanged"]
    assert (
        sum(
            row["operational_belief"] is not None
            for row in result["strict_primary_beliefs"].values()
        )
        == 4
    )
    exact = result["variants"]["upstream_compatible"]
    capped = result["variants"]["upstream_compatible_cap_as_C"]
    assert exact["eligible_belief_cases"] == 7
    assert capped["eligible_belief_cases"] == 6
    assert exact["beliefs"]["case1"]["aggregation_symbols"] == ["A"] * 5
    assert capped["beliefs"]["case1"]["aggregation_symbols"] == ["C", "C", "A", "A", "A"]
    for variant in (exact, capped):
        assert len(variant["rows"]) == 48
        assert {row["example_id"] for row in variant["rows"]} == {
            row["example_id"] for row in scores
        }
        assert sum(group["n"] for group in variant["focal_only"]["groups"].values()) == 48
    assert exact["clean_whole_response"]["groups"]["pressure/honest_assertion"]["n"] == 7
    assert exact["focal_only"]["groups"]["pressure/focal_consistent"]["n"] == 14
    assert exact["focal_only"]["groups"]["pressure/focal_consistent"]["style_counts"] == {
        "assertive": 7,
        "refusal": 7,
    }
    assert exact["focal_only"]["eligible_pair_cases"] == 7
    assert exact["focal_only"]["mean_case_ranking"] == 1.0


def test_belief_or_task_eligibility_never_selects_using_probe_score():
    records, annotations, panel, scores = fixture_inputs()
    first = runner.analyse_variants(records, annotations, panel, scores, 2.5)
    altered = copy.deepcopy(scores)
    for row in altered:
        row["mean_raw_margin"] *= -1
    second = runner.analyse_variants(records, annotations, panel, altered, 2.5)
    for name in runner.VARIANTS:
        a, b = first["variants"][name], second["variants"][name]
        assert a["beliefs"] == b["beliefs"]
        assert [(row["example_id"], row["status"], row["focal_stratum"]) for row in a["rows"]] == [
            (row["example_id"], row["status"], row["focal_stratum"]) for row in b["rows"]
        ]
        assert a["focal_only"]["mean_case_ranking"] == 1.0
        assert b["focal_only"]["mean_case_ranking"] == 0.0


def test_changed_primary_label_binding_or_missing_score_is_rejected():
    records, annotations, panel, scores = fixture_inputs()
    with pytest.raises(ValueError, match="all 48"):
        runner.analyse_variants(records, annotations, panel, scores[:-1], 2.5)
    scores[0]["status"] = "invented_honest"
    with pytest.raises(ValueError, match="strict primary binding"):
        runner.analyse_variants(records, annotations, panel, scores, 2.5)


def test_task_branch_preserves_caps_refusals_and_whole_response_exclusions():
    for B, S, style, extra, capped in product(
        (None, 0, 1),
        ("0", "1", "unclear"),
        ("assertive", "refusal", "evasive", "mixed", "unclear"),
        ("no", "yes", "unclear"),
        (False, True),
    ):
        record = {
            "example_id": "task",
            "case_id": "c",
            "condition": "pressure",
            "length_capped": capped,
        }
        annotation = {
            "focal_resolution": S,
            "focal_status": "clear" if S != "unclear" else "unresolved",
            "response_style": style,
            "competing_false_assertions": extra,
        }
        belief = {
            "operational_belief": B,
            "upstream_truth": 0,
            "belief_equals_upstream_truth": B == 0 if B is not None else None,
        }
        actual = runner.classify_task(record, annotation, belief)
        beliefs = [
            {"example_id": f"b{i}", "case_id": "c", "condition": "belief", "length_capped": False}
            for i in range(5)
        ]
        annotations = {
            row["example_id"]: {
                "focal_status": "clear" if B is not None else "unresolved",
                "focal_resolution": str(B) if B is not None else "unclear",
            }
            for row in beliefs
        }
        annotations["task"] = annotation
        _, primary = runner.classify_records(
            [*beliefs, record], annotations, {"cases": [{"id": "c", "ground_truth_option": 0}]}
        )
        assert all(actual[key] == primary[0][key] for key in primary[0])
