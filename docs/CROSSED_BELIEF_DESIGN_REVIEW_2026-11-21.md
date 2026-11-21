# Independent review of the crossed neutral belief assay

**21 November 2026. Prospective scientific review, before observing this assay's model outputs.**
This review concerns a new, bounded test of the existing frozen belief readout.
It does not authorize a new fit or resume the failed future-goal assay.
The earlier [known-message result](KNOWN_MESSAGE_RESULTS.md) and
[neutral-control result](FUTURE_GOAL_CONTENT_CONTROLS_RESULTS.md) remain development
evidence with their original limitations and failed gates intact.

## Scientific decision

The proposed crossing is useful because it removes the association between
wording groups and evidence settings in the previous small bank. Making the
reader's initial belief independent of the message also introduces a genuine
no-change control. This improves validation of the readout as an instrument.
It does not establish spontaneous anticipation, strategic deception, a causal
belief mechanism, or a previously unstudied reasoning family.

The useful prospective question is whether the unchanged source readout tracks
the sender-attributed reader state when the visible receipt changes, while
remaining appropriately insensitive to irrelevant world changes. Asking an
explicit nested-belief question and measuring before/after that insertion gives
a transfer comparison; it does not isolate the causal effect of belief-question
semantics from additional tokens, task specification or capture position.

## Variables, gold and information boundaries

Use independently crossed binary variables:

| Variable | Meaning | Visible in sender input? |
| --- | --- | --- |
| I | Reader's initial believed location | Yes |
| M | Exact location assertion in the message | Yes |
| B | Full-reading receipt trusted by the sender | Yes |
| W | Actual current location | Yes |
| A | Whether the reader actually read the message | No |

The stipulated sender-attributed reader location is `T = M if B else I`.
The stipulated actual reader location is `R = M if A else I`.
Class 1 means that the target token is at metadata `place_b` under the relevant
perspective. It does not mean that the belief is accurate, deceptive or strong.
The physical-name mapping must preserve these slot semantics. `place_b` is not
necessarily the second place first mentioned in the natural-language story.

The prose must explicitly establish that the reader adopts the exact message if
read, retains the initial belief otherwise, and obtains no other location
evidence. The sender knows this rule and the initial state, trusts the receipt,
and cannot independently observe actual delivery. Knowledge of W must not imply
that the reader observes W or that the sender corrects the reader outside the
specified message channel. These are fictional stipulated beliefs, not measured
beliefs of an independently instantiated reader model.

A must not enter the model-visible prose, IDs inserted into prompts, chat
metadata, filenames included in inputs, or example selection. Cases differing
only in A must have byte-identical sender messages and token IDs. Cache/dedup by
the complete actual model input. Do not make two model calls and treat their
identical inputs as independent experimental units.

## Exact denominators and necessary baselines

Four fresh wording templates, two physical-name mappings and all `I,M,B,W,A`
settings produce **256 fictional rows but only 128 unique sender inputs**.
Each capture condition therefore needs 128 unique forward inputs, not 256.
With two capture conditions there are 256 unique prefixes; only the queried
condition has a meaningful YES/NO output baseline. A pre-question output
distribution does not answer a question that has not been asked.

The primary subset `I != M` has **64 unique sender inputs**, 32 per mapping and
16 per template. It contains 32 mapping-paired cells. The `I = M` control subset
also has 64 unique inputs. Within the primary subset, the target switches when
B switches. Within the control subset, the target must remain fixed when B
switches.

**The overall bank has a 75% shortcut baseline.** Predicting I alone or M alone
gets 96/128 correct because initial belief and message agree on half the bank.
Both shortcuts are exactly 50% on the primary `I != M` subset. W-only and
B-only baselines are 50% overall and within the primary subset. Every template
and mapping has balanced primary labels. Report these actual counts rather
than labeling every baseline "chance" without a denominator.

The hidden-reader control also changes from the earlier bank. For `I = M`, R
is determined even without A. Consequently, a sender-input method can achieve
75% actual-reader accuracy over the whole A-paired bank. **Only `I != M` requires
exactly 50% actual-reader accuracy and AUROC** for any A-independent score, since
the identical sender input has both opposing R labels. In the full bank, 75% is
the balanced-design accuracy ceiling; full-bank AUROC has no universal fixed
value and must not be required to equal 50%. This control is an information-limit
check, not a separate demonstration of theory of mind.

A structured rule oracle can calculate T perfectly. This must be acknowledged:
the assay tests transfer of a particular latent instrument on controlled inputs,
not whether activations beat an optimized text monitor on this synthetic task.

## Prospective metrics and qualification

Keep the source layer, scaler, normalization, coefficients, intercept, score
orientation, threshold and exact nested question fixed. There is no need for a
new calibration split when no calibration is performed. Any later threshold,
sign, prompt or head selection would require a separately designated development
or calibration set and untouched subsequent evaluation.

The adopted readout qualification floors are reasonable feasibility rules:
at least 48/64 correct primary inputs, at least 24/32 in each mapping, at least
10/16 in each template, and positive gold-oriented receipt-pair movement in at
least 24/32 informative pairs. They are not a power calculation, significance
test, or proof of robust generalization.

**A primary-only pass is insufficient for the expanded conditional-update claim.**
The primary subset retains the old XNOR form. A head that assumes `I = 1-M` could
pass it and fail the newly introduced `I = M` controls. Following this review,
the prospective protocol and analyzer also require at least 48/64 correct
control inputs, 24/32 in each mapping and 10/16 in each template. The same two
subset accuracy floors apply to behavioral qualification, which additionally
requires zero invalid answers. Both capture conditions' probe qualifications
are reported separately.

The separate output-ranking qualification is deliberately **primary-only**:
YES-minus-NO AUROC at least .75 overall and in each mapping, and at least .625 in
every template. The control ranking is still reported. A passed primary ranking
gate must not be described as whole-task likelihood qualification. If verbal
answers remain constant while ranking passes, the failed behavioral endpoint
cannot be relabeled as successful behavior. The
[execution protocol](CROSSED_BELIEF_VALIDATION_PROTOCOL_V1.md) is the authority
for the exact frozen rules.

Report all queried and pre-question results, not just the best condition. Report
confusion matrices and unchanged-threshold accuracy alongside AUROC; retain
invalid/capped generated answers in the full denominator. Freeze the strict
parser before generation. No prompt repair, retry, layer change, sign reversal,
threshold tuning or favorable-template selection may rescue a failure.

The paired diagnostics are important regardless of aggregate qualification:

| Counterfactual pair | Number | Expected target relation |
| --- | --- | --- |
| B flip, I != M | 32 | T flips |
| B flip, I = M | 32 | T remains fixed |
| W flip, all sender inputs | 64 | T remains fixed |
| Physical-name mapping swap | 64 | Slot-valued T remains fixed |

For informative B pairs, report the signed score difference
`(2*M - 1) * (score[B=1] - score[B=0])`, pairwise ranking correctness, and both
members correct at the frozen threshold. For B-invariant and W-invariant pairs,
report prediction agreement and absolute score changes. These are different
diagnostics: small score movement is not guaranteed merely by correct binary
answers. Name swaps test physical-word invariance, not complementary propositions
or logical negation coherence. The queried proposition stays `place_b`.

All quantities are descriptive on this small factorial bank. Mapping and
counterfactual rows are paired, and four authored templates are not a large
sample of independent wording or reasoning families. Do not apply an ordinary
binomial test to all 128 rows as though they were independent stories. No
population-level significance claim is needed for this instrument qualification.

## What success or failure would mean

- A qualified queried readout would strengthen evidence for transfer of
  explicitly queried attributed-belief information under independently crossed
  wording and evidence settings. It would not establish a spontaneous
  communication monitor.
- A queried-success/pre-question-failure pattern would show a limit of this
  frozen instrument across capture conditions. It would not show that the model
  has no other representation of the belief before the question.
- A qualified pre-question readout would be encouraging transfer evidence, but
  alone would not demonstrate that the model uses that information to choose a
  future report. Communication prediction and controlled interventions are
  separate follow-up questions.
- A failure concentrated on `I = M` or B-invariant controls would expose a
  potentially important shortcut: carrying over the old assumption that every
  message reverses the initial belief. Report that failure without changing the
  gold or removing those controls.
- A failure of the primary subset would limit the old transfer claim under new
  wording and initial-state control. It would not invalidate all possible
  second-order-belief research, and would not justify a larger future-goal run.

The primary subset retains the old binary XOR/XNOR update geometry. The new
`I = M` conditions extend it to a conditional selector; the four templates remain
wordings of that same rule. This distinction belongs in any dissertation or
paper. Existing results cannot justify claiming a first discovery of latent
belief representation, prompt sensitivity, or a belief/behavior gap.

Even the pre-question condition includes a candidate proposition, an instruction
that it is a statement to assess, and a system saying questions will follow.
It is therefore a pre-perspective-question capture in an assessment context,
not unconstrained spontaneous monitoring during natural report generation.

## Primary-paper review and implications

The following sections were opened and read for this review; their concrete
methods, rather than only abstracts, inform the design. These are focused
comparisons, not a claim of an exhaustive novelty search.

**Zhu, Zhang and Wang (2024), Sections 3.1-3.2 and 4.1-4.3.** They use BigToM
story/belief pairs, final-token attention-head activations, linear probes and
held-out stories. They separately evaluate protagonist and oracle perspectives,
then test activation interventions and transfer across forward-belief,
forward-action and backward-belief tasks. Our frozen transfer and separation
from world truth have precedent; causal use requires evidence beyond decoding.
Our two capture conditions and fictional sender-attributed target differ from
their first-order protagonist/oracle setup, without establishing firstness.
[Primary paper](https://arxiv.org/html/2402.18496v3).

**Bortoletto et al. (2025), Sections 3.1-3.5 and results on pages 5-7.** They cache
final-token residual streams on BigToM story/belief pairs, compare multiple
model sizes and training regimes, use random-label and PCA controls, and test
random tokens, competing statements, time specification and explicit initial
belief. Others' belief readouts are prompt-sensitive; oracle readouts are more
robust. They also test contrastive activation addition. This makes independent
wording/evidence crossing and precise capture definitions necessary; generic
prompt brittleness or decoding better than elicited answers is already occupied.
Their controls are informative precedents, not a guarantee that any probe
passing analogous controls must represent belief.
[Primary paper](https://aclanthology.org/2025.findings-emnlp.1226.pdf).

**Herrmann and Levinstein (2025 version), Sections 5.1-5.4.1.** They distinguish
accuracy, coherence, uniformity and use. Dataset correlations can make accurate
decoding ambiguous; paraphrase consistency and transfer matter. Use concerns
whether the identified representation contributes appropriately to outputs,
which can be investigated through interventions. This assay mainly tests
accuracy and limited invariance. It does not establish logical coherence,
domain-general uniformity, or behavioral use. The paper provides methodological
criteria, not an empirical validation of our head.
[Primary paper](https://arxiv.org/html/2405.21030).

## Review status

The independent checker read the final
[draft bank](../data/belief_diagnostic/crossed_belief_validation_v1.json) directly
with PowerShell, without importing the project's builder, label helper or Python
analysis code. No model outputs were inspected. Its SHA256 was:

`3fc31d095b75ac948cd04bb5eb2bafdad62a2bf2a872243657e4367effa8cc6c`.

The audit found no discrepancy in:

- 128 unique queried inputs, 128 unique pre-question inputs and 256 private A
  records; each template has all 16 visible-factor combinations and both mappings.
- All 128 I/M/W settings independently recovered from the physical names in
  the initial, outgoing-message and world paragraphs; all 128 B settings recovered
  from independent positive/negative receipt text patterns; all 128 consequent
  T labels, and all 256 private R/T/W labels recomputed directly.
- All 128 exact source-format nested-question wrappers and the relation
  `queried user text = pre-question user text + question wrapper`.
- All 64 complete global physical-name swaps, 64 W counterfactual pairs,
  32 sensitive B pairs and 32 invariant B pairs, including their target relations.
- No sender row ID or private `actual_delivery` field name in model-visible
  messages. There is one shared system message.
- The expected primary label balance in all four templates and both mappings,
  and direct baseline counts 96/128 for I/M and 32/64 on the primary subset.
- Sixteen fresh full entity/place strings absent from all five older diagnostic
  JSON banks. This is lexical disjointness, not semantic independence.
- All 64 rows in the [human annotation sheet](review/crossed_belief_validation_v1_human/annotations.csv)
  remain blank apart from anonymous review IDs.

Two issues were corrected during authoring, before any new model observations:
the source question's `Question: ` wrapper was restored, and overlapping old
physical-place strings were replaced. This was pre-execution input review, not
an outcome-driven rescue or an edit to an executed old bank.

The independent review also read the updated
[CPU analyzer](../src/latent_probes/crossed_belief_analysis.py), including the
control-floor addendum, before inference. The direct pairwise AUROC/tie rule,
conditional-label formula, subset selection, counterfactual grouping and signed
orientation, mapping pairs, and restricted hidden-reader check agree with this
review. The analyzer recomputes margins and the complete saved vocabulary's
argmax/normalizer. Source code review is not a substitute for the later raw-run
audit; no new run outcomes were available or inspected for this review.

Independent agent review is not independent human semantic validation. The
[human packet](review/crossed_belief_validation_v1_human/review.md) remains
incomplete. The renderer and exact JSON audits support a bounded exploratory
instrument test; they do not convert these authored materials into a validated
natural-language benchmark or establish a causal mental-state representation.
