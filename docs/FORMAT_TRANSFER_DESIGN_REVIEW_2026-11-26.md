# Format-transfer design review

26 November 2026. Independent source and measurement review. This document is a
post-result proposal, not a change to the completed study's protocol or result.
No checkpoint forward, classifier fit, calibration, sweep or outcome recomputation
was performed for this review. The existing dirty human-review packet was not
edited. The existing graph was queried for dependency context; it has no
fresh-readout nodes, so the claims below were checked against source directly.

## Decision

Execute one finite, descriptive analysis of the saved, exactly paired A/B margins
at the already frozen sites. It can identify whether fixed-threshold failure
coexists with useful ordering, and whether the shift resembles a common offset.
It cannot establish calibration generalization or isolate which component of B
caused the shift. Do not launch another story pilot, select another site, repair
thresholds on these test outcomes, or relabel the exposed bank as fresh.

A later experiment should cross the system, narrative package and query, and
compare A-only with mixed-format training on fresh reviewed contextual groups.
Include an intercept-only adaptation control: improvement from mixed training
alone does not distinguish a change in accessible features from a score offset.

The concurrently prepared [pure bridge module](../src/latent_probes/format_bridge.py)
crosses both easy and hard corrections from the existing 24 contexts. That plan
contains 1,536 paired states and 12,288 package variants per checkpoint if ever
captured, including 3,072 exact diagonal reproductions and 9,216 hybrids. It is
input preparation only. Its token-prefix gate checks causal attention
availability, not semantic equivalence or unchanged hidden features empirically.

## Source findings and remaining confounds

The checked sources are [bank rendering](../src/latent_probes/fresh_readout_bank.py),
[capture](../src/latent_probes/fresh_readout_capture.py),
[fitting](../src/latent_probes/fresh_readout_fit.py),
[analysis](../src/latent_probes/fresh_readout_analysis.py), the original
[renderer](../src/latent_probes/main_study_bank.py), and the formal
[state design](../src/latent_probes/main_study_design.py). They are consistent with
the [frozen protocol](FRESH_READOUT_EXECUTION_PROTOCOL_V2.md) and the principal
limits in the [completed interpretation](FRESH_READOUT_RESULTS_INTERPRETATION_2026-11-23.md).

1. **No concrete label-order, fold or question-location bug was found in the
   examined path.** C2 targets `initial`, `update`, `first_receipt`, `receipt` in
   that order and folds `1-M1 if B2 else M1 if B1 else I`. The gold state uses the
   same rule. This is source inspection, not an independent reconstruction of
   the saved arrays or proof that every input is semantically valid.
2. **The locations are relative to the proposition being queried.**
   `locations(group, name_map)[1]` is both the candidate location and the explicit
   question location. `name_map` swaps physical strings while preserving the
   abstract state and gold. A factor bit is therefore a relation to this mapped
   candidate, not a persistent physical-location identifier. A follow-up that
   changes the queried candidate must recompute relational labels and gold;
   reusing the old label would create a real semantic error.
3. **The capture spans are explicit and causally ordered.** The account span
   includes the paragraph separator immediately before the final question.
   `query_location_last` selects the last token of the location in that question,
   not the earlier candidate proposition. Full-template IDs are compared with
   offset tokenization; right-boundary crossings fail. A permitted left crossing
   must contain whitespace only. The stored token index, ID and offsets should
   still be summarized by format: a correct semantic span does not imply the
   same physical token, local suffix, absolute position or prefix length.
4. **The account anchor is a format-specific closure.** A ends with “End of
   account. A question may follow.”; B ends with “Account closed. The question
   follows separately.” The selected account token may include the separator.
   The same anchor name does not mean the same lexical carrier. The selected
   S1 sites are Llama `layer_16_query_location_last` and Qwen
   `layer_21_query_location_last`; changing the account marker alone cannot
   explain those sites without a demonstrated downstream effect.
5. **B is a bundled prompt intervention.** It changes the system instruction,
   narrative wording, paragraph/event presentation, candidate/closure wording,
   and explicit query wording. A gives the baseline before the report; B gives
   a 09:05 report before the 09:00 baseline and explicitly instructs ordering by
   event time. The formal intended update remains the same in the fixed
   chronology. The present result does not distinguish a readout-domain shift,
   different task elicitation, confusion over temporal order, or a correlated
   semantic misunderstanding. Agent review is not independent human acceptance.
6. **Hard-only factor diagnostics have a dependency.** Hard corrections satisfy
   `I = 1-M1`, so excellent hard-only decoding of both bits cannot establish that
   they are independently represented. Use all easy plus hard corrections for
   the principal factor diagnostics, and keep the hard stratum as a fixed
   secondary breakdown. Even in the full correction bank, factor prediction is
   supervised accessibility, not evidence that the model itself runs the fold.
7. **The endpoint and baseline have different meanings.** Every hard receipt
   quartet has a 3:1 verdict pattern. A rule can obtain 75% row accuracy while
   missing every complete quartet. Conditional YES-minus-NO logits concern the
   complete prompt, after the query and answer instruction; query-location
   features have not seen the suffix. Strict first-token accuracy and candidate
   mass diagnose elicitation, but neither is free-generation behavior. C2 has
   hard folded outputs; its four raw factor margins are the appropriate
   continuous diagnostic, not an invented C2 probability.

## Analysis that is meaningful with the existing arrays

Freeze the analysis definition and input hashes before running it. Mark every
output `posthoc_exposed_bank_descriptive`; the original test was held out for the
original fit, but its results have now been inspected. No new confirmatory
holdout can be created by subdividing it after that exposure.

The minimum inputs are the two saved test row inventories, saved prediction
archives containing factor margins, fit settings/manifests and the bank. Full
feature archives are required only to independently reconstruct the frozen
margins. The 14,080 captured rows are an inventory, not the denominator of this
paired test: the A/B test population is 4,992 rows per checkpoint, comprising
2,496 exact semantic pairs. Correction factor analysis has 1,536 pairs, and the
hard endpoint has 768 pairs or 192 quartets per renderer/checkpoint. Training and
development rows have no saved B counterparts in this study.

### Identity and measurement gate

- Verify source and archive hashes against the existing seals; never overwrite
  an original report or prediction archive.
- Join A/B on group, condition and the entire canonical formal state, after
  removing only renderer and record identity. Require one A and one B per key,
  equal gold and factor truth, and complete world/name/receipt counterparts.
  Never join by array order or a partially stripped case ID.
- Reconstruct the 192 hard panels from formal state and require all four
  receipt cases in each. Retain every context and every outcome.
- Audit recorded selected-token IDs, offsets, positions and lengths separately
  for A/B and checkpoint. This checks exact measurement identity; it does not
  normalize positions or move the probe.
- Stop the new analysis on a mismatch and preserve the failing record. Any
  concrete defect needs a versioned correction record, not silent repair of the
  completed experiment.

### Fixed diagnostic population and outputs

Use only the frozen S1/C1/C2 instruments and the recorded output comparator.
The principal diagnostic is the four C2 margins at S1 on **all correction
rows**, reported separately for A and B. Do not search twelve sites for a
retrospectively favorable instrument. Analyze Llama first as the named model;
report Qwen with the same definitions without pooling their contexts.

For each factor, retain its original orientation and threshold `margin > 0` and
report class counts, both class recalls, raw AUROC, and fixed score quantiles
(10th, 50th, 90th percentile) by true class and renderer. For exact A/B pairs,
report decision flips, margin delta `mB-mA`, the same delta quantiles by true
class, and the difference between positive- and negative-class mean margins
within each renderer. Also retain all-four-factor accuracy and frozen folded
hard-quartet accuracy. Repeat the factor tables for easy and hard strata as
declared descriptive breakdowns. No orientation reversal, threshold search,
fitted translation, fitted rotation, or favorable-row filtering is permitted.

For B1/B2, report the direction and size of the frozen margin change on their
own receipt counterparts, holding the other formal variables fixed. For I/M1,
these independent counterpart contrasts require the pooled easy/hard bank;
hard-only counterparts cannot vary either bit independently. Report each
context's diagnostics, so a pooled score gap cannot hide context offsets.

Reproduce the previously reported frozen row, quartet, nuisance and first-token
results as a consistency check. New differences or intervals must be labeled
posthoc; a whole-context bootstrap, if supplied, describes this fixed bank only.
No new significance tests or selection are needed for this decision.

### Decision gates

These are diagnostic routing rules, not evidence of a general mechanism:

| Observed pattern | Next decision | Claim still unavailable |
| --- | --- | --- |
| B factor ranking remains useful, while class recalls or the zero threshold collapse; paired deltas resemble a common shift | Include the intercept-only adaptation arm before claiming feature decoding failed | A threshold estimated from exposed test labels will generalize |
| Any factor loses ordering substantially, has reversed raw orientation, or shows strongly class-dependent/context-dependent changes | Test the crossed prompt packages; a single shared additive offset cannot account for all of that pattern | Belief information is absent, or the model's reasoning failed |
| Frozen factor accuracy remains strong but quartet/fold failures remain | Inspect exact factor-error combinations and fold sensitivity, with no relabeling | The model internally composes those factors |
| Human semantic review rejects the intended reading or a span/hash/label identity check fails | Preserve the original result, issue a versioned defect/ambiguity record, and freeze the correction before any new inference | The original outcome can be silently replaced |
| Results are mixed | Report the mixed diagnosis and proceed only with the finite crossed design below | A single preferred explanation has been identified |

High raw AUROC is compatible with a threshold problem, but does not by itself
show that the same offset works across contexts. A within-class correlation or
paired-margin correlation is insufficient: it can be high despite a harmful
shift and low despite recoverable information.

## Finite fresh follow-up

This is a proposed next protocol, not authorization to execute one in this
review. Fix both checkpoints, NF4 runtime, revisions and each checkpoint's S1
site from the completed study. Keep Llama primary and Qwen a named replication.
Do not select another site from either old or new test outcomes.

### Cross the prompt packages

Define three independently selectable packages: system S, narrative/account N
(including candidate and closure), and final question Q, each at its A or B
version. Compile all eight S/N/Q combinations on the same formal states. Do not
claim that this further separates narrative wording from event order: N still
bundles those choices. Every crossed prompt must preserve the same attributed
belief task and be reviewed before checkpoint inference.

Use 24 **new** test contexts, all 32 hard states/context and all eight packages:
6,144 test prefixes/checkpoint, with 192 hard quartets/package. The frozen old
heads can score all eight combinations. A system effect can reach every site;
a query-only change is later than account_last and therefore must leave an
identical account-prefix feature unchanged. This is a strong causal-availability
engineering control. It is not evidence that question wording has no effect at
query_location_last or prompt_last. Preserve exact span metadata for every
package; do not force absolute-token alignment by padding.

For frozen-head bridge diagnosis, report all eight package cells and paired
context contrasts for S, N and Q averaged over the other packages, plus their
interactions. Treat them as secondary, finite package contrasts, not a procedure
for choosing a new best prompt. A bridge on the existing 24 contexts would be
cheaper (six additional packages, 4,608 new hard prefixes/checkpoint), but would
remain an exposed-context mechanism diagnostic and cannot replace the fresh
test in this proposed follow-up.

### Match the training budget and include an offset control

Author 16 fresh fitting and eight fresh development contexts, separate from
the 24 test contexts. Retain 32 formal correction cells/fitting context and
all 64 cells/development context. All identities, states, receipt panels and
format assignments must be frozen before inference. The A-only and mixed arms
must use the **same 512 underlying fitting states and contextual groups**:

- A-only: render all 512 states in A.
- Mixed: render exactly 256 of those states in A and the complementary 256 in
  B. Do not duplicate each state in A and B and call 1,024 rows a matched budget.
- Allocate format before observations with exact factor/path balance. With the
  original 32-cell roster, each of the sixteen I/M1/B1/B2 paths occurs twice per
  group: assign one occurrence to each format, rotate nuisance assignments over
  paired groups, and audit both within-format and pooled support. Preserve the
  unavoidable within-group conditional nuisance dependence in the description.

For C2, use the same four targets and external rule in both arms; for a secondary
C1 comparison use the same single verdict target in both arms. The primary
contrast is **mixed minus A-only C2 B-hard complete-quartet accuracy** on the
fresh 24 contexts. Retain A performance, factor truth, both class recalls,
world/name invariance, and strict/conditional output diagnostics. Match available
development formats, labels and objective across arms, or avoid development
selection by fixing regularization prospectively from the old experiment.

To distinguish offset adaptation from changing readout directions, add two
finite diagnostic arms on the **same 256 A + 256 B mixed observations**:

- A-half: fit coefficients/intercepts on the 256 A observations only.
- Offset: start from A-half, freeze its feature map and coefficients, and fit
  only one additive B intercept per factor on the 256 B observations. For A,
  retain A-half's predictions. No sign changes, per-context offsets, rotations,
  or test-dependent thresholds.

Mixed versus Offset asks whether allowing feature directions to use B labels
helps beyond an intercept change with exactly the same observation budget.
A-half supplies the unadapted reference. A-only512 versus Mixed512 remains the
primary budget comparison. If sample size differs between candidate fits, fix
the penalty per observation rather than inadvertently changing effective
regularization. For the sklearn summed-loss convention, fixing
`lambda = 1/(512*C_fixed)` implies `C_n = 512*C_fixed/n`. Choose and seal one
feature-map policy for these diagnostic arms; a common previously frozen map
avoids conflating newly estimated normalization with new directions. Any such
choice must be stated as part of the instrument, not selected on test gains.

With 512 A fitting captures plus the 256 B replacements needed by Mixed, 1,024
development captures (eight contexts, 64 states, two diagonal formats), and the
6,144 crossed hard test captures, the minimal new inventory is **7,936
prefixes/checkpoint**. It omits fresh one-step/asserted/easy test controls; if
these are required, predeclare the additional diagonal A/B inventory: 1,536
one-step, 384 asserted, and 1,536 easy-correction prefixes/checkpoint, for
**11,392 total**. A protocol must select its full inventory before inference,
not expand after an attractive or disappointing result.

### Freeze and interpretation gates

Human semantic review must accept the newly compiled S/N/Q combinations,
including chronology and queried-location labels, before model inference. If
that is unavailable, label the entire study agent-audited exploratory and do
not claim independent semantic validation. Exact tokenizer/span checks precede
capture. Fit/scaler/offset settings and both checkpoint bundles must be sealed
before any new test capture or output logits. Development never becomes a
coefficient-fitting cohort by accident; a dedicated calibration partition, if
chosen instead of the fitting-B offset above, must be budgeted and named as
training data.

Analyze complete contexts together, retain all 192 B quartets and all training
failures, and give a paired whole-context uncertainty interval for the primary.
Do not pool checkpoints as independent contexts. Mixed improvement plus Offset
improvement supports cheap adaptation within this instrument; Mixed improvement
beyond Offset supports benefit from fitting directions on B in this fixed
family. Neither proves an invariant internal belief representation or the
model's causal computation. A null comparison is not evidence of absence of
belief information. Finish and preserve the fixed inventory regardless of score.
