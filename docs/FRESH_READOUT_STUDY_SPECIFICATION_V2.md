# Fresh readout study v2 — DRAFT, not an execution freeze

22 November 2026. This draft specifies a bounded follow-up to the completed
study and the [cached adequacy diagnostic](CACHED_READOUT_ADEQUACY_PROTOCOL_V1.md).
It authorizes neither a sealed run nor a claim of completed measurements.
No 48-case bank has been authored for this draft. The cached diagnostic should
inform the expense and the final protocol before fresh cases or outcomes are
used. Human semantic review remains incomplete; the intended research status is
agent-audited exploratory, with no invented human approval gate.

The primary-source methods and appendix reading, exact versions and limitations
are preserved in the [independent next-experiment review](MAIN_STUDY_NEXT_EXPERIMENT_REVIEW_2026-11-22.md).
This specification adds no new literature claim or API measurement.

## Decision and primary question

Use one fresh held-out cohort to test whether the original midpoint/template
capture explains the source-trained linear readout's correction failure. Compare
a source-development-selected site with the original fixed site, holding the
source row IDs, sample size and fitting procedure constant. The bounded primary
is **Llama N512, renderer A, hard correction complete-quartet accuracy:
selected site minus original site**. Qwen repeats the frozen design with its
own features and fitted heads. It is a named replication, not a model-selection
pool or an opportunity to discard the weaker result.

This asks whether a practical source-only selection rule improves transfer. A
negative result does not exclude every unselected site or nonlinear encoding.
The source selector may itself be insufficient; target-trained controls and the
fixed panel make that limitation visible.

There is a cheaper alternative if the cached diagnostic has already established
strong correction accessibility at the original site. Choose the branch before
fresh authoring, freeze it in a new execution document, and do not change the
primary after viewing fresh outcomes.

| Cached pattern | Final study branch to prepare |
| --- | --- |
| Direct correction-trained LR meets the adequacy criterion in both models | Keep the original site. The fresh primary becomes correction-trained LR minus source-trained LR at that same site on renderer-A hard correction quartets. Recapturing the old source material at new layers is unnecessary for this question. |
| Only nonlinear or factorized decoding meets the criterion, or the two models disagree | Use the four-depth by three-position panel and the source-selected-minus-original primary above. An answer may be linearly accessible later or elsewhere despite accessible ingredients at the original midpoint. |
| No cached method meets the criterion | Prepare the same bounded panel as a diagnosis of capture/task inadequacy only if that remaining question justifies its cost; the draft does not silently launch it. A failed cached instrument is not evidence of absent model capacity. |

Proposed operational adequacy criterion: mean primary quartet accuracy at least
80%, mean secondary-row accuracy at least 90%, and no outer fold below 60%
primary quartet accuracy. Apply it separately to each method/model using all
declared cached out-of-fold predictions. These are prospective engineering
cutoffs proposed for scope selection, not statistically validated reliability
thresholds or confidence bounds; the exposed overlapping-fold result remains
descriptive. Record the criterion and decision before fresh authoring. If the
criterion is revised after cached results, disclose that revision as exploratory
design development; it does not license adapting to the later fresh test.

## Exact cohort and prefix budget

Reuse exactly the 512 full-support source fitting rows from the first declared
allocation seed, 11, at N512, and all 512 original one-step development rows.
The seed is chosen by its fixed ordering, not its measured performance. Preserve
the original observation/message mix and the exact row IDs. Source material is
already known; the new evaluation narratives and renderer B are fresh.

Author **48 new contextual groups**, keeping every related construction in its
assigned cohort. New contexts must not be renamings of old stories or each other.

| Cohort | Contextual groups | Cells/context | Renderers | Prefixes/model |
| --- | ---: | ---: | --- | ---: |
| Reused source fitting | Existing allocated groups | Exact allocation | A | 512 |
| Reused source development | 16 | 32 | A | 512 |
| New correction fitting | 16 | 32 | A | 512 |
| New correction development | 8 | 64 | A | 512 |
| New paired test | 24 | 104 | A and B | 4,992 |
| **Total inventory** | **48 new groups** | | | **7,040** |

The 512 source rows and 512 correction-fitting rows match unique row budget.
Their narrative counts and mechanism mixture differ: the source allocation uses
32 existing groups with 16 rows/group, while correction fitting uses 16 new
groups with 32 rows/group. Consequently the alternate support comparison is a
comparison of fitted pipelines with equal row counts, not a causal estimate of
adding a single operator while holding story coverage constant.

In the full panel branch there are 7,040 new prefills/model, including recapture
of the 1,024 old source/development prefixes at all sites. In the fixed-midpoint
branch, reuse the verified 1,024 old feature vectors and capture **6,016 new
prefixes/model**. Both branches have the same 7,040-prefix evaluation inventory.
There is no 12-fold multiplier in model passes for the full panel.

For each correction fitting group retain the 32 cells satisfying

```
I XOR M1 XOR W XOR name_map = group_offset.
```

Assign complementary offsets to predetermined pairs of groups. Both receipts
remain unconstrained. Every `I,M1,B1,B2` path occurs within each fitting group;
its nuisance combinations are sampled deterministically. Each group contains
16 hard and 16 easy cells and complete receipt panels. Publish the exact masks,
group pairs, row IDs and marginals before capture. Correction development uses
all 64 cells/group.

## Test cells and controls

Each of the 24 test contexts supplies the following **104 cells in A and the
same 104 symbolic cells in B**, totaling 208 prefixes/context:

| Condition | Cells/renderer/context | Target and purpose |
| --- | ---: | --- |
| One-step message | 32 | Full `I,U,B,W,name_map` factorial; establishes fresh one-step transfer with a message mechanism. |
| Easy two-message, `I=M1` | 32 | `Y=M2 if B2 else M1`; same correction prose, event count and broad length as hard cases, with no dependency on B1. |
| Hard correction, `I=M2` | 32 | `Y=M2 if B2 else M1 if B1 else I`; eight complete four-receipt quartets/context. |
| Explicit final nested attribution | 8 | Balance the directly asserted attributed value, private world value and name mapping; a direct-output reading/elicitation control. |

The correction operator stays `M2=1-M1`. No bare retraction or unknown-valued
state is introduced. The private world value remains independent of attributed
belief. Actual private delivery is not disclosed: the target follows the
fictional sender's accepted records and the explicit update rules.

Renderer A retains the familiar chronological account structure. Renderer B is
an independently written semantic equivalent with different syntax, receipt
phrasing and clause order, while preserving explicit timestamps and rules.
Neither B nor B labels are used in fitting, development, calibration or site
selection. Both sequence conditions occur in each renderer so that a renderer
failure can be detected on one-step cases too. Two fixed renderers do not
establish arbitrary natural-language robustness.

World/name changes supply label-preserving counterfactual pairs. Receipt
interventions supply changing and preserving pairs. Keep explicit conflicts
between attributed belief and world, both mappings, and both candidate values
in the controls. The easy four-receipt panels have a different label pattern
from hard quartets; report their own heuristic behavior rather than imposing a
shared independent-cell chance estimate such as `0.5^4`.

An inference-trained probe is not guaranteed to transfer to an asserted-belief
template. Therefore explicit-attribution probe accuracy is a transfer diagnostic;
direct YES/NO output on that condition is the reading/elicitation control.
Failure never causes filtering or replacement. The easy two-message condition
helps distinguish a broad format/event failure from the harder state dependency;
it does not perfectly isolate an abstract faculty of composition.

Before model outcomes, an author-independent agent interprets all 48 base
contexts and a recorded blinded sample spanning every normalized path in A/B.
Record disagreements and exact review coverage, revise before capture, then bind
all prose, factors, masks, splits and review records by hash. Validate other
expansions mechanically. This supplies agent semantic review, with correlated
blind spots possible, not human validation. Keep the research status explicit.

## Capture panel and safeguards

Retain the pinned Llama3.1-8B and Qwen2.5-7B checkpoints, existing NF4 settings,
BF16 compute and input-limit policy. Precision is held fixed; this study does
not attribute failures to quantization or require a new full-precision campaign.

For the full panel use four zero-based block outputs, resolving them from the
pinned configurations before outcomes:

- Llama: blocks **8, 16, 24, 30**.
- Qwen: blocks **7, 14, 21, 26**.

The existing original midpoint is included. The final block is excluded to
avoid conflating the ordinary block output with final decoder normalization.
For each block capture three renderer-defined positions:

1. Last token of the account before the explicit question.
2. Last subtoken of the queried location within the explicit question.
3. Last complete chat-template token, including the assistant header, before
   any answer token—the exact original capture position.

Use explicit character/token spans from the renderer, not a global substring
search or guessed negative offsets. Check selected token IDs and causal-prefix
availability. Earlier positions have not processed later question text. Their
success would not establish spontaneous anticipation in a task with repeated
fixed targets, a candidate statement and explicit rules.

Take all 12 vectors from the same forward with hooks that copy only requested
positions. Preserve float32 vectors, row/token identities and model/environment
metadata. A 7,040 × 12 × 4,096 float32 Llama matrix occupies approximately
1.38 GB before metadata; Qwen is smaller. Do not retain every layer at every
token merely to obtain this panel.

Use a declared engineering example outside every cohort to verify hook/HF
hidden-state alignment and position selection. Verify that recaptured original
sites correspond to the old configuration; record any environment or numeric
differences rather than silently mixing incompatible caches. Truth-table tests
must reproduce all correction labels and quartet memberships independently of
model behavior. A gold-factor oracle is a software/logic ceiling, not a prose
baseline or evidence of model capability.

## Separate source and correction selection

Only source fitting rows may estimate source scalers and source coefficients.
Only the original one-step source development rows may select a source site,
regularization or source calibration. No correction-development label may enter
the source selection path, even though such labels are permitted for the
separately named correction-trained controls.

At each of the 12 sites, fit training-only StandardScaler, row L2 normalization
and L2 logistic regression with `lbfgs`, intercept, `tol=1e-4`, `max_iter=2000`,
and `C ∈ [.001,.01,.1,1,10]`. Use the same 512 row IDs throughout. Select one
source site/C by lowest equally weighted source-development group log loss.
Resolve ties by smaller C, then original site, then lower block index and
declared position order. Independently choose C at the original site using the
same rule. Preserve every candidate and failure. The original archived head is
a separately named historical reference, not silently substituted for this
same-policy comparator.

The fresh comparison has four named pipelines in the panel branch:

| Pipeline | Capture | Fitting/selection | Role |
| --- | --- | --- | --- |
| S0 | Original midpoint/template site | Source 512/source development | Fixed primary comparator. |
| S1 | Source-selected site | Source 512/source development | Primary candidate. |
| C1 | Same site as S1 | New correction 512/correction development | Direct correction-trained LR accessibility control. |
| C2 | Same site as S1 | New correction 512/correction development | One predeclared stronger reference, chosen by the cached rule below. |

Choose C2 **before fresh authoring**: use RBF SVC if it meets the cached adequacy
criterion in both models; otherwise use four-factor LR plus the supplied fold.
This prefers an equal-single-label nonlinear control when demonstrated adequate.
If the factor method also failed the criterion, retain it as an attempted
ingredient diagnostic and state that the cached analysis identified no strong
positive control. The two models use the same C2 method. Freeze the choice;
do not switch C2 after fresh results.

C1 uses the same LR settings/C grid, selected by correction-development group
log loss. RBF C2 uses the cached protocol's fixed C/gamma grid and margin rule,
selected on correction-development primary quartet accuracy, then all-row
accuracy and the declared smaller-parameter ties. Factor C2 uses the four
targets `I,M1,B1,B2`, shared-C selection by mean development factor log loss,
hard factor predictions and `1-M1 if B2 else M1 if B1 else I`. Its four labels
per row and supplied algorithm are privileged supervision/computation. The C1
versus C2 difference is a whole-pipeline diagnostic; differing selection rules
prevent treating it as a pure intervention on capacity.

In the fixed-midpoint branch set the selected site to the original site and
fit only S0, C1 and C2. S1 would be an identical duplicate and is not reported
as another method. The primary is C1 minus S0, as declared above. Source-trained
and correction-trained methods stay distinctly labeled. No target-selected
site, extra pooling scheme or post-answer probe is added in this bounded study.

Freeze all fitting policies, selected settings and trained heads before fresh
test scoring. Store safe arrays and exact source IDs, never estimator pickles.
Nonconvergence, nonfinite transformations/margins or a failed selected refit
are preserved failures; do not increase iterations or substitute a configuration
after seeing held-out performance.

## Analysis and computation contract

Primary endpoint: Llama mean fraction of the eight hard correction quartets
with all four verdicts correct, on renderer A, at N512. The paired difference
uses S1−S0 in the panel branch or C1−S0 in the fixed-midpoint branch. The final
execution document must contain exactly one of these primaries.

Give absolute scores, both classes, one-step/easy/hard strata, A/B differences,
receipt pairs, world/name invariance, and the distribution of zero through four
correct quartet judgments. Report C1/C2 and Qwen outcomes as predefined
secondary results. If factorized C2 is used, include individual factor accuracy
and the rate all four factors are simultaneously correct. Moderate marginal
accuracy does not establish a reliable fold.

Average within each of the 24 fresh contexts, then bootstrap whole contexts
10,000 times with fixed seed 20261122, retaining both renderers, all states and
all pipelines together. Report a conditional percentile 95% interval for the
single primary difference. It conditions on the frozen selected/fitted models
and two authored renderers; it does not capture uncertainty over their
construction or model families. The two models share contexts and are not
pooled as independent observations. Twenty-four units offer limited precision
for small gaps; a zero-crossing interval is not equivalence.

Publish the complete 12-site source-selected panel as descriptive secondary
material. Do not claim the largest fresh-test cell as a new selected winner.
No model-correctness filtering is permitted. Invalid output, failed controls and
poor verbal behavior remain in their original denominators. A behavior-correct
subset can be separately labeled and can never replace the full cohort.

Same-model final-prefix YES/NO probabilities accompany ordinary prefills. Keep
orientation and any source-development calibration fixed, and count invalid
generated outputs incorrect if generation is separately introduced. This draft
does **not** include a new 384-token prose-monitor campaign or a reasoning-token
sweep. The existing monitor's extra generation and external fold are already
documented; they cannot be described as matched compute. C2's own computational
and supervision advantages must remain explicit.

## Fixed stopping rule and permissible conclusions

There is one fresh cohort, one selected primary and the predetermined pipelines.
Run the chosen branch to its full fixed inventory once. Do not inspect partial
test scores to stop early, add contexts, choose layers, revise wording, change
thresholds or demand a positive effect. Failed controls limit interpretation;
they do not trigger data replacement or new general qualification gates.

If a verified source-binding, capture or software defect prevents valid scoring,
preserve the attempt and correct that concrete defect in a versioned record.
Do not call the affected attempt a model failure. Any outcome-informed scientific
revision requires another explicitly new cohort and lies outside this draft.

Interpret the final pattern in terms of measured pipelines. A selected site's
advantage implicates the original capture choice for this transfer; target-only
success implicates support/encoding transfer; success only through RBF or the
factor fold supports nonlinear or ingredient accessibility under their additional
training assumptions. Failure across one-step/easy controls prevents an isolated
hard-composition claim. A renderer-B drop already on one-step cases indicates
format sensitivity within these two renderers. None establishes absent belief
information everywhere, causal use, spontaneous recipient anticipation or
intentional deception.

The Variant 2 contribution is a reliability test of a candidate latent monitor
under controlled shifts, with a clear record of when its readout is adequate.
It remains distinct from Apollo reproduction and from natural omission tests
with independently measured recipient behavior. After the chosen bounded study,
report the result and the instrument's limits rather than expanding the search
until a favorable outcome appears.
