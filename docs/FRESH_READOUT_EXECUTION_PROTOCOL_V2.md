# Fresh readout study: prospective execution protocol v2

22 November 2026. This resolves the [earlier draft](FRESH_READOUT_STUDY_SPECIFICATION_V2.md)
before any fresh-study model forward. The completed original study and exposed-bank
diagnostic remain unchanged. This is **agent-audited exploratory research**;
human semantic review is incomplete and no confirmatory or publication claim is made.

## Fixed question and branch

Use the full twelve-site branch. The single primary is **Llama N512, renderer A,
hard-correction complete-quartet accuracy, S1 minus S0**. S0 is the original
midpoint/template site with independently source-development-selected C. S1 is
the source-development-selected site/C. Source row IDs, N and fitting policy
are identical. Qwen repeats the design as a named secondary replication. Its
outcomes cannot select the retained model or the primary. A null primary does
not establish that unselected sites lack information.

C2 is **four-factor logistic readouts plus the supplied Boolean fold**, fixed
from the cached-diagnostic decision before fresh authoring. Neither model's
direct LR passed the proposed adequacy condition; RBF did not qualify in both
models. C2 is a strong Llama diagnostic and an imperfect Qwen control. Its four
labels and external algorithm are privileged supervision/computation.

## Inputs and exact inventory

The compiled [bank](../data/belief_main/fresh_readout_v2/compiled_v1/bank.json)
binds the two independently authored context sources, exact old source IDs,
actor-assignment policy and group splits. All new texts are fictional. There
are 48 new contextual groups, not 48 independent reasoning families. Two
renderers share formal task rules; arbitrary-language generality is not claimed.

| Partition | Groups | Rows/model | Renderer |
| --- | ---: | ---: | --- |
| Source fitting | 32 old groups, full-support seed11 allocation | 512 | Original A |
| Source development | 16 old groups | 512 | Original A |
| Correction fitting | 16 fresh groups | 512 | A |
| Correction development | 8 fresh groups | 512 | A |
| Fresh paired test | 24 fresh groups | 4,992 | A and B |
| Total | | 7,040 | |

The original source messages, IDs and states are unchanged. New correction
fitting uses 32 cells/group satisfying `I XOR M1 XOR W XOR name_map = offset`,
with complementary offsets within predetermined pairs. Both receipts remain
free. Each group covers all sixteen I/M1/B1/B2 paths twice, with 16 hard and
16 easy rows and complete receipt panels. This balances marginals; it does
not make nuisance factors independent conditional on each within-group path.

Each fresh test context contributes 104 cases/renderer: 32 one-step message,
32 easy two-message, 32 hard correction, eight directly asserted final nested
attributions. All 208 variants stay together. There are **192 hard quartets per
renderer/model**. Both models share the same contexts and are never pooled as
independent narrative samples. The source and correction training cohorts match
row budget but differ in narrative count and mechanism mix.

Actor-pair pools from both authors are shuffled deterministically with seed
20261122 and alternated within each cohort. This prevents the authors' name
pools from identifying the cohort. Contexts retain their authored splits;
actors and physical location entities are not reused across fresh contexts or
the old bank. No demographics are measured or inferred from names.

## Semantic review and engineering

The blind agent read all 48 base contexts and all 108 complete representative
prompts using only the opaque review packet. Its saved answers were frozen
before comparison. All 108 agree with the formal key; all 48 contexts are
acceptable, with minor idiomatic storage-location notes preserved. The
[agent review receipt](../data/belief_main/fresh_readout_v2/compiled_v1/agent_review_receipt_v1.json)
binds packet, answers and private key. The execution gate independently
regenerates the sample and checks every answer and coverage field. This is
agent review, with possible correlated blind spots, not human acceptance.

Both pinned local tokenizers passed all 14,080 prefix/span checks. The historical
preflight's exact sources are preserved in its native source snapshot. Formal
preparation repeats prefix validation under the final execution-source seal.
Actual tiny randomly initialized Llama/Qwen models checked all twelve hook/HF
positions exactly. These are engineering checks, not checkpoint task results.

One fixed excluded engineering account per checkpoint verifies real hook/HF
alignment before cohort capture. Its verbal correctness is never an inclusion
gate. Its vectors and full logits are archived separately; it is outside every
cohort and fitting/selection source.

## Model and capture contract

Retain Llama-3.1-8B-Instruct revision
`0e9e39f249a16976918f6564b8830bc894c89659` and Qwen2.5-7B-Instruct revision
`a09a35458c702b33eeacc393d103063234e8bc28`, NF4 weights, BF16 compute, SDPA,
batch one and the 2,048-token fail-closed input limit. No generation, truncation,
CPU offload fallback, fine tuning or precision change is introduced.

Blocks are zero-based outputs: Llama **8,16,24,30**; Qwen **7,14,21,26**.
For each block retain account_last, query_location_last, prompt_last in that
order. The account span includes its paragraph separator and ends strictly
before the explicit question. Query spans name the location in that question,
not the earlier candidate proposition. A declared leading-whitespace token
overlap is permitted; right-boundary crossings are rejected. Exact full-template
IDs, offsets, selected token IDs and causal availability are checked.

All twelve vectors come from one forward. Earlier positions have not processed
later question text, but have seen a candidate statement, explicit rules and a
repeated target structure; they are not a query-free anticipation assay. Final
output logits have later context than earlier-position readouts. Per cohort row
retain twelve float32 vectors, both YES/NO logits, vocabulary summary and hashes;
full vocabulary arrays are retained only for the excluded engineering row.

## Fitting and frozen comparisons

Every scaler uses its 512 fitting rows only, float64 StandardScaler then row-L2.
All heads use L2 logistic regression, lbfgs, intercept, tol1e-4, max_iter2000,
no class/sample weighting, deterministic random_state11 and at most four CPU
threads. **Development is never absorbed into coefficient/scaler fitting.**

S0/S1 and C1 C-grid is `[.001,.01,.1,1,10]`. Each of twelve source sites chooses
C by minimum equally weighted source-development group log loss. S1 chooses
the minimum across those twelve retained winners. Exact floating ties prefer
smaller C, then original site, then lower block and declared position order.
S0 is the retained original-site winner. Correction labels do not enter source
selection. S0 and S1 may be identical; retain that result.

C1 uses correction fitting/development only, at S1's frozen site. C2 uses that
same site and training cohort, four separate I/M1/B1/B2 logistic heads with
shared C in `[.01,.1,1,10,100]`; select minimum mean of four equally weighted
correction-development factor log losses, then smaller C. Verdicts always use
the learned factor outputs and `1-M1 if B2 else M1 if B1 else I`. No oracle
condition override is applied. Threshold is margin>0, with ties predicting zero.
There are 85 individual candidate fits/model, with no selected-model refit.

Retain every candidate/failure, all twelve source winners and the C1/C2 heads.
Convergence warnings/nonfinite values invalidate candidates. An all-failed
selection stops with retained evidence; grids, signs, thresholds and iteration
limits cannot change after observed failures. Store safe JSON/NPZ, no estimator
pickles. Bind settings, arrays, candidates and manifest for **both** model
bundles before **any test forward or test output logit** is captured.

## Analysis and reporting

All seventeen named output columns are fixed: twelve source panel heads,
S0/S1/C1/C2 and the final-prefix YES-minus-NO logit baseline. Source-panel test
values are descriptive; no test-selected winner becomes the instrument.
Raw output sigmoid is conditional on the two candidate tokens, not probability
of a valid generated answer. There is no new calibration fitting or generation.
Separately report the recorded full-vocabulary first-token argmax in every
stratum: YES/NO token coverage, correctness on the full denominator, argmax ties
and mean total YES/NO probability mass. An argmax outside the two candidates
counts incorrect for this strict first-token diagnostic. The recorded argmax
uses the lowest index for exact ties. This does not observe a complete generated
response or establish its validity; it is an elicitation diagnostic, especially
on directly asserted attributions. It never replaces the fixed conditional
output baseline or the primary endpoint.

Report row accuracy, both classes, full quartet histograms, world/name verdict
flips, receipt-changing/preserving pairs, every context and A/B stratum. The
asserted condition has no receipt quartet. Four-factor truth/joint accuracy is
reported only on the easy/hard two-message conditions; no initial/report/receipt
facts are fabricated for directly asserted beliefs. C2 still predicts on every
condition with its learned B2 output. Correctness filtering is prohibited.

Continuous-head diagnostics are raw AUROC with frozen orientation, Brier/log
loss and ten fixed equal-width ECE bins. These are unrecalibrated diagnostics,
not certification of semantic confidence. C2 has a hard folded verdict and no
continuous target-margin/probability claim.

Average primary quartet accuracy within each of 24 contexts. Resample whole
contexts 10,000 times with seed20261122, retaining method/renderers together,
and report the conditional percentile95% interval for S1-S0. This conditions on
fixed authoring, two renderers, selected sites and fitted heads. A zero-crossing
interval is not equivalence. Qwen's corresponding interval is a named replication,
not another opportunity to define the primary.

Before scoring, verify all source/fit/capture hashes, exact test IDs/states/prose,
fit-freeze chronology before each test journal, and aggregate vectors/output
margins against their per-row raw records. Preserve all outcomes. No retry,
prompt repair, case replacement, model-correct subset, extra site or revised
primary is allowed to obtain a favorable result. A concrete software defect
requires a versioned preserved correction record, never silent replacement.

## Execution and stopping

Use native Linux output storage and one sequential GPU worker. Source/development
capture for both models precedes frozen fitting; both fits precede either test
capture. Output-wide exclusive locks prevent overlapping aliases. Explicit
resume can retain completed rows; any failed or started-only unknown row stops
the job for inspection. No automatic retry hides an unknown outcome.

The complete fixed inventory, analyses and report finish this study regardless
of sign or size of the primary result. No further outcome-informed expansion is
part of this protocol. The construct is fictional sender-attributed reader
belief under an explicit question. Spontaneous anticipation, actual private
reader beliefs, deceptive intent, causal use and arbitrary-language robustness
remain outside its conclusions.
