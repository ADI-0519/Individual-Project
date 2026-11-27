"""Fixed lexical-control metrics on the already exposed adaptation bank.

No model outputs are invented to satisfy the original analysis API. The text
comparator is shared across checkpoints and has no newly designated primary.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from .format_adaptation_analysis import (
    CONDITIONS,
    factor_metrics,
    paired_difference,
    validate_test_rows,
    verdict_metrics,
)
from .format_adaptation_fit import ARMS, fold_factors, require

UNCERTAINTY_SCOPE = (
    "Paired whole contexts conditional on exposed authored bank, fixed lexical policies "
    "and fitted heads; excludes fitting-cohort uncertainty; posthoc diagnostic"
)


def score_text_arrays(rows, scores):
    """Retain all conditions, factors, context failures and nuisance controls."""
    validate_test_rows(rows)
    require(set(scores) == set(ARMS), "All four text arms are required")
    result = {"arms": {}, "factor_diagnostics": {}, "comparisons": {}}
    groups = sorted({r["group_id"] for r in rows})
    context_values = {g: {} for g in groups}
    for arm in ARMS:
        score = scores[arm]
        values = np.asarray(score["factor_margins"])
        bits = (values > 0).astype(int)
        require(
            values.shape == (4992, 4)
            and np.isfinite(values).all()
            and np.array_equal(score["factor_predictions"], bits)
            and np.array_equal(score["verdicts"], fold_factors(bits)),
            "Invalid text prediction inventory or fold",
        )
        result["arms"][arm] = {}
        result["factor_diagnostics"][arm] = {}
        for renderer, condition in product(("A", "B"), CONDITIONS):
            indices = [
                i
                for i, r in enumerate(rows)
                if r["renderer"] == renderer and r["condition"] == condition
            ]
            entries = []
            for group in groups:
                within = [i for i in indices if rows[i]["group_id"] == group]
                metric = verdict_metrics([rows[i] for i in within], score["verdicts"][within])
                entries.append({"group_id": group, **metric})
                if renderer == "B" and condition == "correction_hard":
                    context_values[group][arm] = metric["quartet_accuracy"]
            result["arms"][arm][f"{renderer}/{condition}"] = {
                "aggregate": verdict_metrics(
                    [rows[i] for i in indices], score["verdicts"][indices]
                ),
                "groups": entries,
                "scope": "correction-trained four-factor external fold"
                if condition.startswith("correction_")
                else "out_of_training_construct_control; no factor-truth claim",
            }
        for renderer, stratum in product(
            ("A", "B"), ("correction_all", "correction_easy", "correction_hard")
        ):
            indices = [
                i
                for i, r in enumerate(rows)
                if r["renderer"] == renderer
                and (
                    r["condition"].startswith("correction_")
                    if stratum == "correction_all"
                    else r["condition"] == stratum
                )
            ]
            result["factor_diagnostics"][arm][f"{renderer}/{stratum}"] = {
                "aggregate": factor_metrics([rows[i] for i in indices], values[indices]),
                "groups": [
                    {
                        "group_id": group,
                        **factor_metrics(
                            [rows[i] for i in indices if rows[i]["group_id"] == group],
                            values[[i for i in indices if rows[i]["group_id"] == group]],
                        ),
                    }
                    for group in groups
                ],
            }
    for comparison in ("A512", "Offset"):
        entry = paired_difference(context_values, comparison)
        entry["role"] = "posthoc_descriptive_comparison"
        entry["uncertainty_scope"] = UNCERTAINTY_SCOPE
        result["comparisons"][entry["contrast"]] = entry
    return result
