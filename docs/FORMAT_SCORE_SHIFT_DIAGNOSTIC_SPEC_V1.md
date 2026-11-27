# POSTHOC format score-shift diagnostic: fixed specification v1

Written 26 November 2026 before opening any saved numerical margin arrays for
this diagnostic. The published 23 November interpretation and source schema
were already read. This is an exposed-test, descriptive POSTHOC analysis, not
a new held-out experiment or a basis for changing this bank's conclusions.

## Question and locked scope

Does renderer B mainly move frozen readout scores across the fixed decision
boundary while retaining target ordering, or also damage target ordering and
ingredient accuracy? Analyze **both Llama and Qwen**, **all 24 fresh contexts**,
both renderers, and every row in the fixed conditions `correction_easy`,
`correction_hard`, `one_step`, and `asserted`. Report the three frozen verdict
heads S0/S1/C1 and the conditional YES/NO output-score control in all four
conditions. Report each frozen C2 ingredient head (I=`initial`, M1=`update`,
B1=`first_receipt`, B2=`receipt`) only for the two correction conditions, where
all four formal targets exist and are valid. Do not invent factor targets for
one-step/asserted cases. C2's synthetic folded verdict score is not continuous
evidence and will not receive an AUROC.

## Frozen choices and exclusions

Use saved raw logistic decision margins and conditional YES-minus-NO output
logit difference with the original orientation: **larger means target 1**.
Predict 1 iff margin **strictly greater than zero**; zero predicts 0. Never use
absolute AUROC, invert a head, change its site, fit any classifier or mapping,
select a subset based on results, search/calibrate a threshold, call a model,
or load GPU/model tensors. No p-values, confidence intervals, new bootstrap,
or confirmatory inference. The authored contexts are paired across models,
not 48 independent contexts. Magnitudes across separately fitted heads need
not share a scale; comparisons are within a frozen head.

## Inputs, integrity, and joins

Native input root is
`//wsl.localhost/Ubuntu/home/adirj/kcl-latent-probes-runs/fresh_readout_v2_20261123_recovery_v2`.
Read each model's final analysis manifest, `predictions.npz`, and test
`rows.json`; verify prediction bytes against the completed analysis manifest.
Hash every consumed input, this specification, and diagnostic source. Seal the
specification hash in a newly created unique output directory **before**
parsing rows or numerical arrays; refuse overwrite. Preserve all originals.

Index saved arrays by unique `row_ids` and join metadata by exact row ID; do
not trust row ordering. Require no missing, extra, or duplicate IDs. A/B pairing
uses `(group_id, condition, canonical full state)` excluding renderer, and
checks unique A and B rows, identical formal labels/state, and distinct prompt
identities (row ID plus any recorded prompt hash). Require each model's formal
inventory to match, 4,992 rows/model, and 24 contexts in every condition and
renderer. Keep a pairing inventory with A/B row IDs and recorded prompt hashes
when present. Nuisance pairing removes exactly `world` or `name_map` from the
full-state key and requires exactly states 0/1, with invariant target labels.
Missing metadata or ambiguous pairings fail rather than silently sort or drop.

## Fixed summaries

For every valid model/head/condition/renderer, report pooled and every-context
raw AUROC (Mann-Whitney probability positive margin exceeds negative, exact
ties half credit), fixed-zero accuracy, target-1/target-0 recalls, predicted
positive rate, and class counts. Summarize within-context AUROC and accuracy
over all 24 contexts by mean/min/max; retain the full per-context table so
pooled ranking cannot hide between-context effects. A one-class target has
undefined AUROC (`null`) and is not silently omitted from row counts.

For target classes 0 and 1 separately, report raw margin mean, population SD,
minimum, quartiles (linear interpolation), median, maximum, and fraction above
zero, plus mean positive-minus-negative class separation. Report corresponding
A-to-B changes with same-state pairs, including class-specific B-minus-A mean,
SD, quartiles, median, extrema, and fraction increasing.

For each formal A/B pairing set (pooled and each context), report frozen-verdict
agreement, correct/correct, correct/wrong, wrong/correct, wrong/wrong counts,
raw-score Pearson and Spearman correlations (undefined constants => `null`),
and B-minus-A and absolute-difference summaries. Correlations describe paired
scores, not target correctness or recoverable information by themselves.

Also enumerate **all positive-negative row pairs within each context and
condition** for each valid target. Report fractions where A and B both rank
positive above negative, A alone does, B alone does, neither does, or either
renderer ties. Report A/B pairwise AUROC and the fraction of non-tied pairs
retaining their order. Aggregate pair counts over contexts, never compare rows
from different contexts in this diagnostic. Retain per-context values. This
distinguishes preserved wrong ordering from preserved useful ordering.

For every fixed world/name-map nuisance counterpart in each valid scope,
report pair counts, mean/median/max absolute margin gap, signed state-1 minus
state-0 mean gap, and frozen-verdict flip fraction; retain all contexts.

Report joint correctness of all four frozen factors and frozen C2 fold
accuracy for easy/hard, per renderer and context. This describes ingredient
failure versus the supplied fold without fitting or attributing model internals.

## Interpretation and output

Use descriptive comparisons of AUROC, class-conditioned margin changes,
fixed-zero recalls, factor correctness, and same-state/order concordance.
High B AUROC alongside poor fixed-zero accuracy supports a boundary/score-shift
description for that specific head/scope; reduced AUROC supports damaged
target ordering. Both can coexist. High A/B correlation alone does not prove
that a common offset explains the failures. Do not claim information absence,
causal isolation of a bundled format change, or that a repaired threshold
would transport to fresh data.

Write a machine-readable summary, all-context tables, pairing inventory,
manifest with hashes and interpreter/version, and a concise Markdown receipt
under a new unique child of `docs/results/format-score-shift-2026-11-26/`.
Implementation tests must cover score orientation/ties/zero, order-comparison
math, and joins robust to permutation while rejecting duplicate/missing/formally
mismatched records. Record unavailable Pylance/environment tools explicitly;
verify the supplied `.venv/Scripts/python.exe` runtime (Python 3.11.9) and run
focused tests. Do not edit frozen scientific modules, outputs, or the user's
dirty review prompt file. Any necessary deviation must be named in the receipt.
