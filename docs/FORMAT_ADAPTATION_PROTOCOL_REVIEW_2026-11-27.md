# Independent review of the finite format-adaptation proposal

27 November 2026. This is a skeptical design and primary-source review of a
**new, agent-audited exploratory experiment**, following the
[26 November result](FORMAT_TRANSFER_RESULTS_2026-11-26.md),
[design review](FORMAT_TRANSFER_DESIGN_REVIEW_2026-11-26.md), and
[literature review](FORMAT_TRANSFER_RESEARCH_REVIEW_2026-11-26.md).
It is not human semantic acceptance, a new approval requirement, or an amendment
to the completed study. No checkpoint forward, classifier fitting, or outcome
analysis was performed for this review. Implementation and opaque-packet review
are separate checks whose actual scope must be recorded when completed.

## Assessment and finite scope

The proposal can answer a useful, narrow question: with a common frozen
measurement map and the same 512 underlying fitting states, does replacing
256 A observations with their B counterparts improve fresh B correction
quartets? Its intercept comparator further tests a particular cheap adaptation
procedure against joint coefficient refitting. Neither comparison uniquely
identifies a changed internal representation, the causal use of a belief, or
the component of B responsible for a difference.

The proposed inventory is coherent: 16 new fitting contexts and 24 new test
contexts; 512 unique correction fitting states, with 512 A and the complementary
256 B captures; and 24 x 104 x 2 = 4,992 test captures per checkpoint. Total new
scientific captures are 5,760 per checkpoint, plus separately excluded
engineering material. The fitting inventory is 768 captures; no arm fits on
all 768. Test includes one-step, easy correction, hard correction, and asserted
attribution controls. No fresh development cohort is needed when sites, old
scaler, regularization, solver policy, endpoints, and reporting are fixed before
these observations. Absence of a development set precludes subsequent tuning;
it does not create a validity defect by itself.

The eight-package bridge is a different component-attribution experiment. Its
omission does not block this matched adaptation comparison. Existing exposed
bridge materials remain diagnostic materials and cannot supply a new holdout.
The new study must retain the statement that A/B changes system instructions,
narrative presentation and chronology, closure/candidate wording, and query
wording together. Success would be adaptation to a known target format on new
contexts, not transport to an unseen format.

## What each arm actually compares

| Arm | Fitting observations | Fitted parameters |
| --- | --- | --- |
| A512 | All 512 A states | Four coefficient vectors and intercepts |
| Mixed512 | 256 A states and 256 complementary B states | The same four kinds of coefficient vectors and intercepts |
| Ahalf256 | The 256 A observations used by Mixed | Four coefficient vectors and intercepts |
| Offset512 | Ahalf256 fit, then the same 256 B observations used by Mixed | Freeze Ahalf coefficients; fit one additive B intercept per factor; retain Ahalf on A |

The primary Mixed512-minus-A512 comparison matches formal-state identities,
contextual groups, unique-state count, fitting-row count, model site, feature
map, targets, and external fold. It deliberately changes which format is
observed for half those states. It does not match training-format coverage;
that is the intervention.

Mixed512-minus-Offset512 matches the available 256 A plus 256 B observations,
but compares two learning procedures. Mixed jointly estimates its coefficients
and shared intercept using both formats. Offset inherits coefficients estimated
from only 256 A rows and then changes its B boundary. Finite-sample coefficient
error in Ahalf can contribute to a Mixed advantage. Offset also has distinct
A/B intercept behavior, whereas Mixed has one shared fitted intercept per
factor. These are substantive procedure differences, not a purified
one-parameter causal contrast.

Keep the named diagnostic and Ahalf reference, but describe a positive result
as **benefit from joint coefficient refitting beyond this Ahalf-based intercept
procedure**. Do not call it proof that rotation was necessary, that a best
possible offset failed, or that B information was absent from the A512 score.
Offset is fitted by logistic loss, not by maximizing complete-quartet accuracy;
its failure does not exhaust all boundary choices. Ahalf-to-Offset directly
measures the frozen-direction boundary intervention. A512-to-Ahalf shows the
cost of the smaller A fitting cohort. None of these diagnostics should replace
the primary after results are seen.

## Allocation and nuisance dependence

The 32-state/context roster retains two nuisance variants of every
I/M1/B1/B2 path. Giving one variant to A and the other to B can give exact
factor/path support per format. It cannot make format independent of the
nuisance variant conditional on a particular context and path: only one of the
two variants is observed in each format. This remains true when global name
and world marginals balance. Complementary assignments over predetermined
context pairs reduce a systematic pooled association, but the paired contexts
are different texts and do not remove all conditional dependence.

Before capture, audit exact state sets and contingency counts by context,
format, all sixteen factor paths, easy/hard stratum, world, name mapping, and
verdict. Check pooled support and complementary pair support, plus the marginal
target-by-nuisance tables for each of the four heads. Publish the deterministic
assignment rule and remaining structural dependencies. This audit must inspect
the actual generated rows, rather than rely only on a verbal description or a
count of 256 rows/format. Do not choose a different assignment after observing
hidden-state scores.

The full fresh test crosses world/name counterparts within A and B, which can
expose an adaptation that exploits the fitting assignment. Report both
class-conditional factor performance and counterpart changes. A low flip rate
from an almost constant classifier is not useful invariance. Do not infer
nuisance independence from high AUROC or complete marginal balance.

Entity disjointness and new contextual texts prevent literal reuse of the old
bank, but templates, formal rules and format grammars remain shared. The
generalization population is new authored contexts in this task family. It is
not arbitrary language, independent reasoning families, naturally occurring
belief attribution, or spontaneous deception. All contexts and failed controls
remain visible in reporting.

## Feature map, regularization, and offsets

Every arm should use the exact old C2 scaler at the old source-selected S1 site,
including the existing standardization followed by row L2 normalization.
Freezing only standardization while silently changing the subsequent
normalization would change the instrument. Hash and identify the checkpoint,
site, old map arrays, old selected C for each factor, and software environment.
The common map contains historical training information and is shared
historical preprocessing, not a scaler newly estimated on the 512 states.

For unit sample weights and no class weighting, sklearn's L2 logistic objective
can be written as

`mean_i loss(y_i, x_i w + b) + ||w||^2 / (2 n C_n)`.

Consequently, fixing `lambda = 1/(512 C_old)` gives
`C_n = 512 C_old / n`: A512 and Mixed use C_old, while Ahalf uses 2 C_old.
This preserves the coefficient penalty relative to average observation loss;
it does not give equal sampling precision to 256 and 512 observations. If old
selected C differs by factor, apply the rule separately. Adding class/sample
weights requires reconsidering the weight sum rather than substituting row
count mechanically. The official
[sklearn objective and solver table](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
support this calculation and show that lbfgs leaves the intercept unpenalized.
Record the installed implementation version; current online documentation is
not itself a runtime receipt.

For Offset, minimize B fitting logistic loss over a single additive delta with
the Ahalf scores held fixed. Both label classes, finite scores, a numerical
solution, and the fitted derivative/residual tolerance must be checked. With
fixed finite scores and both classes, the scalar logistic loss has a finite
unique optimum. No C rescaling is needed for this unpenalized scalar boundary
fit: the frozen coefficient penalty is constant. Do not permit per-context
offsets, sign reversal, a multiplicative slope, or test labels. Preserve fitting
failures and predeclare how a failed required fit halts or invalidates the
comparison; do not quietly replace it with a favorable fallback.

An additive offset leaves within-format ranking unchanged. Thus Offset B
AUROC should equal Ahalf B AUROC, apart from exact floating-point tie handling;
this is a useful implementation check. Joint refitting can change ranking, but
an improved quartet score alone does not establish improved factor ordering.
Retain signed raw AUROC, zero-boundary recalls and margins for each head.

## Endpoint and uncertainty

The fixed primary is Llama B-hard complete-quartet accuracy,
Mixed512 minus A512. Each test context contributes eight hard receipt quartets,
giving 192 quartets per format/checkpoint. A hard quartet's four rows have a
3:1 verdict pattern; row accuracy and quartet accuracy therefore answer
different questions. The supplied fold remains identical across all arms:
use the replacement when B2=1, otherwise the first report when B1=1, otherwise I.
All four factor heads use their original bit orientations and `margin > 0`.

For primary uncertainty, compute each context's paired difference across its
eight quartets, then bootstrap the 24 paired contexts together with replacement
10,000 times at seed 20261127. Seal the percentile levels, quantile convention,
and random generator. Preserve arm pairing in every resample. Do not resample
rows or quartets as independent cases, or treat the two checkpoints as 48
independent contexts. Qwen is a named secondary replication, and
Mixed-minus-Offset is a named diagnostic rather than a second selectable
primary. An interval conditional on these fitted heads and authored contexts
does not include variation from regenerating the fitting cohort or
retraining under alternative allocations. With 24 shared-template contexts,
its precision should not be presented as population-wide certainty.

Factor interpretation must use the easy-plus-hard correction population, where
I and M1 vary independently. Hard-only I/M1 results cannot demonstrate separate
decoding because I=1-M1 there. One-step and asserted controls do not possess the
same four-factor correction semantics; do not invent factor gold for them.
Final-fold behavior can still be reported on controls under an explicitly
declared policy. Preserve the strict output comparator and its invalid/capped
outcomes with denominators; output logits are elicited behavior at the full
prompt, whereas selected query-location features have not seen the later
suffix.

Freeze both checkpoints' fitted bundles and analysis source before any new
test captures or output logits. This prevents one checkpoint's test outcomes
from influencing the other's fitting or reporting policy. Engineering smoke
accounts must be outside every scientific context and must not serve as
performance qualification or prompt selection.

## Primary-source verification and contribution limits

This reviewer independently opened the following versioned primary texts on
27 November 2026 and read the named methods/results passages. This is targeted
reading, not full-paper or code reproduction, and does not inherit the previous
reviewer's PDF coverage. Abstracts and a page index were not treated as methods
verification.

| Source and reading scope | Consequence for this study |
| --- | --- |
| [Kumar v1](https://arxiv.org/html/2605.27958v1), Sections 4.3-4.5, 5.3/5.3.1, Tables 10/11, and limitations on prompt-regime recognition, labels, and layer choice | Already trains clean plus four styles and tests four withheld styles. Table 11 reports .916 to .979 and .921 to .983 withheld-style AUROC. The 27B augmentation comparison uses layer 34; the original collapse table uses 27. Equal total fitting rows are not established by these passages. The proposed fixed site, same states/budget and Offset comparator sharpen an established augmentation question; they do not establish firstness. |
| [Bortoletto et al. v3](https://arxiv.org/html/2406.17513v3), Sections 3.1-3.4, results on controls/prompt sensitivity, and A.1.1-A.1.2 | Already studies protagonist/oracle logistic probes, random/misleading tokens, time specification and explicit initial belief. Appendix A.1.2 specifies L-BFGS, L2 and inverse regularization 10. The inspected prompt comparisons do not establish this proposal's matched frozen-map Offset decomposition. Their control successes do not license concluding that every possible confound is excluded. |
| [Buerger et al. v2](https://arxiv.org/html/2407.12831v2#A4), Appendix D and neighboring cross-dataset definition | Explicitly separates a transferred bias failure despite separation from loss of direction separation; its cities example has AUROC 1 with about 200 errors in one class. Logistic regression shares both possible failure modes. This is direct precedent for the boundary/direction distinction, not a new conceptual result here. |
| [Ichmoukhamedov and Martens v1](https://arxiv.org/html/2505.09807v1), Sections 2-4, including centering, topic splits, keyphrase comparison/control and conclusion | Already documents conversation-format/position fragility and improvement from a shared ending. Its preprocessing centers each format/topic, and its conclusion leaves single-instance bias unresolved. Keeping one old map and fitting an explicit B intercept addresses a practical distinction that its centered comparison intentionally removes. |
| [Klein v1](https://arxiv.org/html/2611.07805v1), Section 2 definitions, Section 3 setup/edit validation, and Section 4 nuisance results/development selection | Target AUROC can coexist with differing nuisance sensitivity. Correlation-controlled fitting, target-preserving edits, placebo edits and family-level splits matter. Its test-selected equivalence comparisons are explicitly descriptive; its development-selected comparisons are separate. This supports preserving world/name diagnostics and careful selection language, rather than claiming that repair validates the belief construct. |

These checks support the bounded rationale, not an exhaustive novelty search or
publication prediction. The defensible contribution would be a transparent
fixed-budget comparison of adaptation procedures on dependent correction
judgments, with fresh contexts, retained failed controls, and no test-time
calibration. Either a null or a positive result can resolve that finite
measurement question. A positive result would not establish internal belief
invariance, intentional deception detection, causal composition by the model,
or general repair beyond the two known formats.

## Remaining concrete checks

No scientific scope blocker was identified in the stated finite design.
Actual implementation must still demonstrate the state/format allocation,
exact common feature map, finite offset optimization, fit/test ordering seals,
correct quartet joins and context bootstrap. An opaque-input agent semantic
review should independently answer the supplied prompts before seeing their
formal key; it remains correlated agent evidence and cannot become human
acceptance through agreement. Any concrete semantic or identity defect should
produce a preserved, versioned correction before inference. The known absence
of human review is already represented by the exploratory status and does not
justify inventing an additional authorization step.

## Completed opaque semantic and generated-row checks

The independent reviewer subsequently read every system and user message in
the [opaque packet](../data/belief_main/format_adaptation_v1/compiled_v1/review_packet.json):
118 complete prompts, 353,540 prompt characters, and 40 distinct context
introductions. The reviewer manually answered the explicit questions and wrote
the [frozen answers](../data/belief_main/format_adaptation_v1/blind_review_answers_v1.json)
before opening the new bank, formal states, bank code, or any review key.
The packet SHA256 is
`3adf3a22c5d4dce219498b5e0ad6f915c939b8fbeaf19b0eb4a8ecb0505b3fa0`;
the frozen answer-file SHA256 is
`4eaad489ab019fe8795733291d6b68e0de91f8fa9122d7ce1122f2f9b2fc4753`.
Key comparison is a separate root-reviewer step; agreement is not assumed here.

All 118 sampled prompts were judged acceptable. Minor article errors in
introductions ("a endleaf", "a index", "a invitation", "a envelope") are
retained in the item notes. They do not change the evidence, locations,
chronology, or queried attribution. The review found no sampled contradiction
requiring rejection. This is direct agent reading of the representative
packet, not human acceptance or exhaustive reading of all 5,760 renderings.

After freezing those answers, the reviewer inspected the
[new bank implementation](../src/latent_probes/format_adaptation_bank.py) and
independently counted the generated rows. The exact pooled support is all
128 format x I/M1/B1/B2 x world x name-map cells, each occurring four times.
Within each of the sixteen fitting contexts, each format covers all sixteen
factor paths exactly once. A512 and Mixed512 contain the identical 512 unique
formal states; the capture inventory is 512 A plus 256 B. Test has 192 complete
hard receipt quartets per format. Full actor names, actor name components and
physical location strings do not overlap either the old main bank or the old
fresh-readout bank.

The implementation makes the remaining within-context dependence particularly
clear: Mixed format is determined by world in each fitting context, while the
world-to-format direction rotates over predetermined contexts. The exact
pooled conditional balance is stronger than marginal balance, but it does not
remove that within-context association. Report both facts.

The reviewer also read the new
[fitting implementation](../src/latent_probes/format_adaptation_fit.py),
[analysis implementation](../src/latent_probes/format_adaptation_analysis.py),
their synthetic tests, and the initial
[capture implementation](../src/latent_probes/format_adaptation_capture.py),
plus the imported exact-prefix/measurement helpers. The inspected map-copy,
sample-size penalty, fixed-coefficient scalar offset, formal-state joins,
hard-quartet grouping and paired 24-context bootstrap agree with the stated
design. The capture source policy includes the imported rendering, formal
design, backend, configuration and storage dependencies. A misleading
synthetic-test comment/control was reported to the implementation author:
constant zero scores 50% over the balanced hard population; predicting I gives
the intended 75%-row/zero-complete-quartet counterexample. The author corrected
the synthetic control and added the explicit row-accuracy assertion.

This source review precedes the final campaign/protocol freeze. It does not
certify an uninspected later version or replace actual native preparation and
capture receipts. A suggested additional provenance assertion was sent to the
root author: the copied old protocol should match the old fitted bundle's
recorded capture-provenance protocol hash and model settings, in addition to
being included in the new source seal. No scientific observations were read
or produced during these checks.
