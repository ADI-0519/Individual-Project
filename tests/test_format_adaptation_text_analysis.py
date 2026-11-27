"""Metric inventory and controls for a fixed posthoc lexical comparison."""

import copy

import numpy as np
import pytest
from test_format_adaptation_analysis import scores_for, test_rows

from latent_probes.format_adaptation_text_analysis import UNCERTAINTY_SCOPE, score_text_arrays


def test_all_conditions_factors_contexts_and_paired_contrasts_remain():
    rows = test_rows()
    scores = scores_for(rows)
    result = score_text_arrays(rows, scores)
    assert set(result) == {"arms", "factor_diagnostics", "comparisons"}
    assert "output_baseline" not in result and "primary" not in result
    assert len(result["arms"]) == 4
    assert len(result["arms"]["Mixed512"]) == 8
    assert len(result["factor_diagnostics"]["Mixed512"]) == 6
    metric = result["arms"]["Mixed512"]["B/correction_hard"]
    assert metric["aggregate"]["complete_quartets"] == 192
    assert len(metric["groups"]) == 24
    control = result["arms"]["Mixed512"]["B/asserted"]
    assert control["aggregate"]["quartets"] == 0
    assert control["scope"].startswith("out_of_training_construct_control")
    contrast = result["comparisons"]["Mixed512_minus_A512"]
    assert contrast["difference"] == 1
    assert contrast["conditional_percentile_interval_95"] == [1, 1]
    assert contrast["role"] == "posthoc_descriptive_comparison"
    assert contrast["uncertainty_scope"] == UNCERTAINTY_SCOPE
    assert result["comparisons"]["Mixed512_minus_Offset"]["difference"] == 0


@pytest.mark.parametrize("damage", ["missing_arm", "nonfinite", "bits", "fold"])
def test_refuse_missing_or_inconsistent_predictions(damage):
    rows = test_rows()
    scores = scores_for(rows)
    if damage == "missing_arm":
        del scores["Offset"]
    elif damage == "nonfinite":
        scores["Offset"]["factor_margins"][0, 0] = np.nan
    elif damage == "bits":
        scores["Offset"]["factor_predictions"][0, 0] ^= 1
    else:
        scores["Offset"]["verdicts"][0] ^= 1
    with pytest.raises(ValueError):
        score_text_arrays(rows, scores)


def test_test_inventory_still_requires_all_counterparts():
    rows = test_rows()
    scores = scores_for(rows)
    damaged = copy.deepcopy(rows)
    damaged[0]["partition"] = "fit"
    with pytest.raises(ValueError, match="Non-test"):
        score_text_arrays(damaged, scores)
