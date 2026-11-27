# POSTHOC pooled-correction addendum, fixed before pooled computation

26 November 2026. The original v1 specification and initial easy/hard/control
diagnostic outputs remain unchanged in the sibling
[v1 result directory](../20261126T232649Z_90868d74/). Their specification hash is
`3abbc410cbc760a4d5cb81c65902f80f604e6001754ab9292b3223421d70c157`.
The initial source bytes are preserved there as
[diagnostic_source_v1.py](../20261126T232649Z_90868d74/diagnostic_source_v1.py),
SHA256 `2953dc1c4a7101ec664237a6635187bf7ed9dd7ce6a180250e2be06473b513ef`.

This addendum is written **after initial stratum-level outcomes were seen**,
and **before computing any pooled easy+hard outcomes**. An independent design
critic identified the algebraic dependency before seeing those numeric outcomes:
hard rows enforce I=1-M1, while easy rows enforce I=M1. Consequently, high I/M1
AUROC in either stratum alone does not establish separately accessible targets.
Combining all easy and hard correction states yields independent variation of
I/M1 and is the principal descriptive population for interpreting factor
readout accessibility. This does not change the original study's primary
endpoint and does not create a new confirmatory primary analysis.

Analyze all correction rows, both easy and hard without selection, for both
models/renderers and all 24 contexts. Call this fixed population
`correction_all`: 1,536 rows/renderer/model, 3,072 rows/model, and 1,536 exact
formal A/B pairs/model. Preserve original full states and exact row IDs and
prompt hashes; combine conditions only as a diagnostic grouping label. All
original input hash, row-ID, A/B state, target, coverage, frozen prediction and
fold checks still apply. The full 4,992-row metadata/array bundles are verified
before selecting this algebraically determined population.

For I/M1/B1/B2 and S0/S1/C1/output, compute exactly the v1 summaries, using the
same positive-is-target-1 orientation and strict margin>0 boundary: pooled and
every-context raw AUROC, accuracy, recalls, class-conditional margins, same-state
A/B shifts/correlations/transitions, within-context positive-negative order
transitions, and world/name-map nuisance pairs. Compute all-four joint factor
correctness and the unchanged C2 fold's row accuracy. Do not invent or pool
factor targets for one-step/asserted rows. No model calls, refitting, threshold
search, inferred optimal thresholds, calibration, group selection, significance
testing or new uncertainty intervals.

Write a new supplement only. Preserve all bytes of previously generated v1
artifacts. Seal this addendum hash before this new grouping is computed. Report
its timing honestly, retain the original easy/hard/control tables, and describe
all findings as exposed-bank POSTHOC diagnostics. The initial ranking-loss
findings and high-AUROC/poor-threshold cases motivated no factor filtering.
