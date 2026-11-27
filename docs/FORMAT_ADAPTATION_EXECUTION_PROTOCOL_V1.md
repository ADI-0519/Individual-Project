# Fresh matched-format adaptation: fixed execution protocol v1

27 November 2026. Written before any new-study model forward or fit. This is
agent-audited exploratory research; human semantic review is incomplete. The
old study and exposed-bank diagnostics stay unchanged. No positive result,
publication novelty, marks or admissions outcome is promised.

## Question, scope and primary

Does allocating a fixed fitting budget to both prompt formats improve a
four-factor belief readout on new contexts? How does that procedure compare
with adjusting only the intercepts of a half-budget source readout using the
same observations? The primary is **Llama, format B, hard complete-quartet
accuracy, Mixed512 minus A512**. Qwen is a named secondary replication.
Mixed512 minus Offset on that endpoint is a named diagnostic, not a new primary.
Each checkpoint has 192 hard quartets in 24 shared paired test contexts.

This tests transport of a fictional sender's accepted reading records and
attributed reader belief after an explicit question. It is separate from the
Apollo adaptation, natural deceptive behavior and spontaneous anticipation.
The four factor labels and supplied fold are privileged. Comparisons within
these C2 arms have matched factor supervision; no direct-probe or language-model
capability advantage is claimed.

## Fixed instrument and observations

Retain the previous pinned Llama-3.1-8B-Instruct revision
`0e9e39f249a16976918f6564b8830bc894c89659` and Qwen2.5-7B-Instruct revision
`a09a35458c702b33eeacc393d103063234e8bc28`. Runtime is the unchanged WSL
NF4/double-quantized configuration, BF16 compute, SDPA, batch one, no generation,
no CPU offload and no truncation. Maximum input is 2,048 tokens. Use the previous
Llama block-16 and Qwen block-21 query-location site, before final normalization.
The audited capture collects its normal twelve-site raw journal for provenance;
only the fixed single site reaches fitting. No new site search is allowed.

Copy the exact old C2 training means/scales and row L2 feature mapping. Do not
fit a new scaler. Both old C2 regularization constants are 100, fixed from their
old development selection. Those old feature maps are shared historical training
information, not new test information. No new development cohort or selection
is needed. No new test threshold, sign, regularization, iteration limit or
candidate family may be chosen after seeing results.

| Inventory per checkpoint | Groups | Captured prefixes |
| --- | ---: | ---: |
| New fitting states, all A plus B replacements used by Mixed | 16 | 768 |
| New test: all one-step, easy correction, hard correction, asserted; A and B | 24 | 4,992 |
| Cohort total | 40 | 5,760 |

There are 11,520 cohort forwards across checkpoints. One excluded engineering
forward at each model/cohort reload checks all twelve hook/HF alignments;
these four checks are not cohort evidence. No eight-package bridge is run here.
That separate prepared exposed-context diagnostic remains unexecuted. The
present comparison identifies adaptation procedures on A/B, not which bundled
system/narrative/query change caused the original failure.

The input [bank](../data/belief_main/format_adaptation_v1/compiled_v1/bank.json)
contains 40 new contextual settings from one explicit author pool, not new
reasoning families. Eight broad domains each provide two fit and three test
contexts; four neutral framing templates occur at equal split proportions.
Actors are deterministically shuffled across cohorts with seed tag 20261127.
No old actor-name component or complete location entity is reused under the
recorded roster checks. Narrative content and entities are disjoint across fit
and test; formal rules and format templates are shared.

## Equal-observation fitting arms

For fit context index g, retain the 32 cells satisfying
`I XOR M1 XOR world XOR name_map = g % 2`; both reading-record bits remain free.
Mixed uses A iff `world == (g//2)%2`, otherwise B. Every context has one A and
one B occurrence of each of the sixteen I/M1/B1/B2 paths. Across the cohort,
all 128 format-by-path-by-world-by-name-map cells have count four. Within each
context format still depends on world; this conditional dependency is retained
as a limitation and tested with full crossed nuisance conditions at evaluation.

| Arm | Exact fitting information |
| --- | --- |
| A512 | All 512 underlying states rendered in A. Fit four coefficients/intercepts. |
| Mixed512 | The same 512 states: 256 A and complementary 256 B. Fit four coefficients/intercepts. |
| Ahalf256 | Only the same 256 A observations used by Mixed512. |
| Offset | Ahalf256 coefficients and feature map frozen. Fit one B-specific additive intercept per factor on the same 256 B observations as Mixed512; A keeps Ahalf256. |

All arms know the experimental format; only Offset explicitly branches its
intercept by that known format. No state is counted twice within Mixed512.
Mixed versus Offset compares these learning procedures with the same total
observations. It cannot uniquely establish geometric rotation: Mixed jointly
refits on A+B whereas Offset freezes finite-sample Ahalf256 coefficients.

Use float64 logistic regression, lbfgs, L2, unpenalized intercept, tol 1e-4,
max_iter 2,000, random_state 11, class_weight/sample_weight None. Fix the penalty
per observation using `C_n = 512 * 100 / n`, hence C=100 at n512 and C=200 at
n256. Record installed sklearn version. Restrict BLAS/OpenMP to four threads.
Any nonfinite value or convergence warning stops and preserves the attempt;
no rescue or refit grid is allowed. Offset minimizes mean logistic loss by
monotone derivative bisection without regularizing its intercept. Balanced
factor labels ensure finite unique optima. Store safe JSON/NPZ, no pickles.

Predict each factor as margin strictly greater than zero (zero predicts zero).
Always fold the four learned predictions as `1-M1 if B2 else M1 if B1 else I`.
Never substitute an oracle receipt or condition. Factor truth is evaluated
only for easy/hard corrections, including their union; one-step/asserted verdicts
are explicitly out-of-training-construct controls for this four-factor pipeline.

## Semantic, source, capture and test gates

A separate agent read all 118 opaque complete prompts and froze its answers
before comparison with the key: all answers agree and all sampled cases are
acceptable. Minor article errors remain. This is not independent human review.
A different source reviewer independently checks formal labels, all allocation
contingencies, cohort/entity isolation, complete panels, and exact regeneration.
Raw authoring, compiled bank, review packet/key/answers and receipts are immutable.

Before weights: validate every tokenizer prefix/span, exact old tokenizer/model
revision, frozen source map, old fitted-map-to-protocol identity and environmental
package versions. Archive exact scientific sources and prepared inputs on native
Linux storage. Fitting gets a separately bound fit-only plan and exact rows;
no test observation is opened by the fitting code. Known authored labels are
outside the model token input.

One sequential worker captures both fitting cohorts, fits both bundles, then
seals all coefficients/settings/manifests for **both models before any new test
forward or test output logit**. Test capture rejects a missing/changed seal.
Every row has an exclusive started/raw/completed journal and exact prompt/token
hashes; partial or failed attempts are retained and block automatic repetition.
One model-call failure never authorizes repeating an unknown row. A source or
recording defect requires a versioned documented recovery, not silent rewriting.

## Fixed analysis, uncertainty and completion

Report every arm/model/format/condition: row accuracy, both class recalls,
complete four-case receipt-panel counts and histograms, world/name verdict
flips and both-correct pairs. Asserted cases have no receipt panel. Keep all
24 contexts and all failures, without model-answer correctness filtering.
Correction factors additionally retain raw signed-orientation AUROC,
fixed-zero accuracy, both recalls and all-four correctness; do not invent
a continuous probability for the deterministic fold.

Retain conditional YES/NO output margins and strict full-vocabulary first-token
correctness on the full denominator, including out-of-candidate errors, ties
and available candidate-mass diagnostics. This is not a free-generation monitor.

For the primary and named diagnostic, use 10,000 paired whole-context bootstrap
draws with seed 20261127, percentile 95% intervals. Qwen repeats the same named
contrasts. These intervals condition on authored groups, formats, old map/site,
checkpoint and fitted heads; they exclude fitting-cohort/checkpoint uncertainty.
The models share the same contexts and do not provide 48 independent units.
Only the Llama Mixed-minus-A512 contrast is primary; no familywise guarantee is
claimed for other descriptive/diagnostic comparisons.

After both analyses, an independently authored presealed checker reconstructs
both models before opening either saved outcome. It verifies predictions and
the declared metrics without fitting or model calls. The source checker and
its fixture tests are sealed before test capture. A failing audit blocks final
report interpretation. Finish the full fixed inventory regardless of score;
retain null, mixed and negative results. A useful adaptation result still does
not establish a universal belief representation or publication firstness.
