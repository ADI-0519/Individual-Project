# Fresh readout v2: independent design audit

22 November 2026. This is an author-independent agent review of the design
before new cases, native features or fitted heads were available to this
reviewer. It reviews the complete [draft specification](FRESH_READOUT_STUDY_SPECIFICATION_V2.md),
the [cached interpretation](CACHED_READOUT_ADEQUACY_INTERPRETATION_2026-11-22.md),
the [cached protocol](CACHED_READOUT_ADEQUACY_PROTOCOL_V1.md), relevant current
source/capture contracts, and the [research evidence map](MAIN_STUDY_RESEARCH_EVIDENCE_MAP_2026-11-22.md)
and [next-experiment review](MAIN_STUDY_NEXT_EXPERIMENT_REVIEW_2026-11-22.md).
No new primary-paper claim is made or independently reverified here. The
existing 727-node Graphify map was used for navigation and is stale with respect
to these latest documents; current files supplied the substantive evidence.

No model was called, no native-data classifier was fitted, no package was
installed, and no old bank, frozen protocol, capture or head was changed.
Inventory and mask checks below are deterministic Boolean enumeration, using
the existing Python 3.11 environment. This is a design audit, not an audit of
the not-yet-written fresh implementation or a claim of human semantic review.

## Assessment

The bounded full-panel branch is implementable and answers a useful, narrower
question: does a site selected for one-step source performance transfer better
than the original site on fresh correction quartets? Its source-only selector
does not promise to select the best correction site. The design is ready to
make concrete once the execution document resolves the blockers below. These
are software and specification requirements, not a request for human approval.

The cached evidence fixes the branch and C2 choice before fresh authoring.
Direct LR fails the proposed adequacy criterion in both models. RBF does not
meet the criterion in both; use all 12 sites and four-factor LR plus the fixed
fold as C2 for both models. Llama's cached 95.00% factor-fold quartets do not
qualify an N512 fresh instrument: those outer refits used 1,536 rows. Qwen's
45.42% factor-fold result is an attempted ingredient diagnostic, not a strong
positive control. The exposed cached bank remains descriptive development
evidence and cannot become the fresh confirmation cohort.

## Blockers to resolve before fitting or test capture

The execution author has accepted the resolutions recorded here during this
review. Their appearance in a final execution contract and implementation still
needs checking; this audit does not certify unwritten code.

| Issue in the draft or its implementation boundary | Required resolution |
| --- | --- |
| The draft says heads are frozen before test *scoring*. Test prefills also reveal features and output logits. | Freeze the complete source-selected settings and safe head/transform hashes **before any test model forward**, for either model. Tokenizer-only preparation of all 7,040 prefixes is permitted beforehand. A training/development stage must be unable to execute a test row. A later test stage must verify both models' fitted bindings before its first forward. |
| New source position extraction could silently alter the old prefix or use a different notion of account/question end. | Reuse the exact seed-11 N512 source messages and all original development messages and IDs. Add span metadata without changing text. Resolve all three semantic positions on every observation/message source row and every fresh condition, independently for each tokenizer. |
| Calibration and output policies are not fully specified by “keep orientation and any calibration fixed.” | Use no new calibration or threshold tuning. Source/C1 positive margin means symbolic target 1; margin `>0` predicts 1 and a tie predicts 0. Output uses bare distinct YES/NO token IDs, `logit(YES)-logit(NO)`, the same strict threshold and a stable two-logit conditional probability. No test sign reversal, absolute AUROC selection or correction-based source calibration. |
| “Publish the complete 12-site panel” could leave a test-time C choice. | Select and freeze one source-development C at **each** site, as well as S1 and the independent original-site S0 comparator. Every panel entry is that site's source-selected head. Do not pick a fresh-test winner. |
| C2's grid and fitting support could be confused with direct LR or the larger cached refit. | S0/S1/C1 use `[.001,.01,.1,1,10]`; C2 factor heads use the cached `[.01,.1,1,10,100]`. Use exactly 512 fitting rows per support, with no absorption of development into fitting. C2 uses one shared correction-training scaler/normalizer and four heads, selected by mean factor development log loss with equal group weight. |
| Floating ties, candidate failures and selected-refit behavior need deterministic meaning. | Use exact floating-score equality and the declared lexicographic ties: smaller C, then original site, then lower block and declared position order for source site selection. Preserve unsuccessful candidates. Do not change iterations, grids, sites or thresholds after outcomes; a failed required selected fit remains a recorded failure. |
| The assertion condition has no meaningful I/M1/B1/B2 gold vector; one-step accounts have no second receipt. | Leave inapplicable symbolic fields explicitly null. C2 may produce a folded target verdict on every condition as attempted transfer, always using its learned B2 prediction. Do not substitute a gold or condition-dependent B2. Report four-factor and all-four-factor truth metrics only on easy/hard two-message rows. One-step I/U/B1 metrics may be separately named where genuinely defined; no B2 truth metric there, and no factor truth metrics on assertions. |
| A semantic sample can be nominally blinded while its ID exposes the logical bits. | Give the fresh reviewer rendered text and opaque sample IDs, hiding symbolic state, gold, gold-encoding IDs and author implementation. Freeze its interpreted answers before comparison, preserve disagreements and revisions, and bind the final review coverage. The reviewer must be distinct from the case author. |

Do not silently import the original study's human-review gate into this fresh
agent-audited exploratory run. Preserve that status and the incomplete human
semantic-validation disclosure.

## Capture positions and matching

The existing [renderer](../src/latent_probes/main_study_bank.py) supplies the
candidate proposition, then `End of account. A question may follow.`, followed
by the explicit nested question. The existing
[capture routine](../src/latent_probes/main_study_run.py) hooks an ordinary
decoder block output at the final complete template token, with zero answer
tokens, and checks HF hidden-state index `block+1`. It currently captures only
one position/layer and must not be treated as an already implemented panel.

The panel must retain exactly Llama blocks `[8,16,24,30]` and Qwen
`[7,14,21,26]`, checked against pinned configurations. S0 is Llama block 16 or
Qwen block 14 at the final template token. At every block, the positions are:

1. The final token belonging to the rendered account, before question text.
2. The last subtoken overlapping the designated location occurrence inside the
   question, rather than another occurrence in a report or candidate statement.
3. The last complete chat-template token, including the assistant header.

Use renderer-owned character spans, exact chat-template alignment and tokenizer
offsets or an equivalently verified alignment. Save the selected index, token
ID, decoded token/context and causal prefix. Boundary-merging tokens must not
silently introduce later question characters at the account site. Reject or
explicitly resolve such an alignment before native capture. Check full token
IDs against direct chat-template tokenization, not only against decoded strings.

Candidate proposition text must occur before the account endpoint in A and B,
including the assertion condition. The account site therefore already has the
candidate value, actors and update rules. It lacks later explicit question text,
but is not query-free or evidence of spontaneous anticipation. At the queried
location site, later question punctuation/instructions and the assistant header
are causally unavailable. Final-prefix output has more textual context than an
early-position probe and should be labeled accordingly.

Collect all 12 float32 vectors in one untruncated forward, selecting only the
requested positions in four hooks. On a noncohort engineering example, verify
every hook/site against the corresponding HF hidden-state index and token
selection. The excluded final decoder block avoids final-normalization
ambiguity. Preserve environment/checkpoint/tokenizer/template bindings and
numeric differences when recapturing the old site; do not silently mix old and
new feature caches. A recapture discrepancy is an engineering issue to diagnose
from bound inputs, not a reason to alter the scientific comparison after test.

## Inventory and correction masks

The arithmetic is correct:

| Partition | Groups | Rows/group | Renderers | Prefixes/model |
| --- | ---: | ---: | --- | ---: |
| source_fit | 32 existing selected groups | 16 | A | 512 |
| source_dev | 16 existing groups | 32 | A | 512 |
| correction_fit | 16 fresh groups | 32 | A | 512 |
| correction_dev | 8 fresh groups | 64 | A | 512 |
| test | 24 fresh groups | 104 | A and B | 4,992 |
| Total | 48 fresh groups plus old source | | | 7,040 |

Existing allocation code verifies seed 11 N512 full support uses 16 observation
and 16 message groups, 256 rows of each mechanism. Preserve that mix and exact
IDs. In this chosen branch all 7,040 prefixes are recaptured once per model;
12 sites multiply stored vectors, not model passes. There are 192 hard test
quartets per renderer/model, nested in 24 contexts. Across both models the
planned inventory is 14,080 model prefills, excluding the separately declared
engineering checks. Source/development recapture is part of the inventory.

Enumerating the proposed mask `I XOR M1 XOR W XOR name_map = offset` for either
offset produces exactly:

- 32 rows, 16 hard and 16 easy, with 16 target labels of each class.
- Eight complete four-receipt panels: four hard quartets and four easy panels.
- All 16 `I,M1,B1,B2` paths, each appearing twice.
- Balanced 16/16 marginals for each of I, M1, B1, B2, W and name_map.

The mask is independent of receipts, preserving joint receipt tests. It does
couple W/name parity to I/M1 within a group. Complementary offsets in fixed
pairs restore the full 64-cell symbolic factorial across the pair, but those
rows belong to two narratives. Marginal balance is not within-group conditional
independence, and paired narratives do not create a new independent test unit.
Publish exact pair/offset assignments, masks, IDs, marginals and complete
quartet membership before capture. Independently reproduce gold truth tables,
including hard `[I,M1,I,I]` in `(B1,B2)=(00,10,01,11)` order and easy
`[M1,M1,1-M1,1-M1]` in that order. Never impose one shared heuristic chance rate
on these different patterns.

The equal 512-row source/correction budget changes story count, mechanism mix
and supervision as well as support. C1 versus S1 is a fitted-pipeline comparison,
not a causal estimate of adding one correction operator with everything else
held fixed.

## Semantic controls and independence

The eight asserted cells must directly state the sender's final attribution
about the reader at 09:20. Merely stating the reader's actual belief, the world
location, the sender's own belief, or a quoted message does not assert the
nested target. Balance asserted target, independent private world and global
name map. Harmless context must not later overwrite the asserted attribution
or imply that an unshared private check changes it. Retain explicit belief/world
conflicts and both physical candidate locations through global mapping.

The candidate statement and question must refer to the same symbolic slot after
every mapping in every condition. Candidate text is an assessment target, not
an extra report received by either person. B may reorder clauses, but must keep
timestamps, trust, accepted-record scope, no-notification rules and replacement
priority unambiguous. “Correction” is a new location-bearing replacement,
never an unknown-valued retraction. Actual private delivery remains undisclosed.

An inference-trained head's assertion failure is a transfer diagnostic. Direct
YES/NO output on assertions is the reading/elicitation control. Neither that
control nor model behavior may filter cases or trigger replacements. Easy
two-message performance is necessary context for interpreting hard failures;
it does not isolate one abstract compositional faculty.

Assign all conditions, factor variants, paraphrases and renderings of a context
to its one fixed cohort. Compare new contexts against all old banks and each
other for copied skeletons, not just reused entity names or exact byte strings.
Preserve the declared agent semantic sample and its gaps; all unreviewed
expansions still require mechanical binding checks. Review all 48 base contexts,
with condition/path coverage in A and in test-only B. B must remain absent from
all model fitting, development, calibration and selection.

Twenty-four authored test contexts are the analysis units. Shared event rules,
renderer code and narrative conventions mean they are not 24 independent
reasoning families. A/B are two fixed renderers; paired renderers and shared
contexts across models are dependent observations. The proposed whole-context
paired bootstrap is appropriate as conditional uncertainty for frozen pipelines
on this construction, with limited precision. It does not quantify uncertainty
over authoring, renderers, model families or training/site selection.

## Interpretation and reporting recommendations

Keep the single primary: Llama N512 renderer-A hard complete-quartet accuracy,
S1 minus S0. If source development selects the original site and same C, the
two heads and the primary difference may be identical; retain that result.
Selecting a one-step winner may miss a correction-accessible location. A null
primary cannot establish that the original site was optimal for correction,
that every panel site lacks linear information, or that the model lacks the
capacity. C1/C2 at S1 do not establish accessibility at every other site.

Present all frozen per-site source heads descriptively, retaining failures;
do not promote the largest fresh-test score to a selected instrument. C1/C2
and Qwen are named secondary outcomes, with Qwen retained whatever its result.
Four-label C2 plus a supplied algorithm has privileged supervision/computation,
and a different C grid/selection objective. It is not a pure classifier-capacity
intervention or evidence the language model itself executes the fold.

Raw output probability is conditional on YES/NO, not probability of emitting
a valid answer or a calibrated semantic confidence. AUROC must retain source
orientation; Brier/log loss may describe the actual two-logit conditional
probability without claiming calibration. C2's hard fold is a binary verdict,
not a calibrated target probability. No new generation or prose monitor is
required for this bounded study.

Retain absolute scores, class recalls, one-step/easy/hard/assertion strata,
A/B gaps, world/name counterfactuals, receipt-changing/preserving pairs, and
quartet zero-to-four histograms. Failed controls limit conclusions and remain
in denominators. Preserve one fixed inventory, a versioned concrete-defect
correction record when needed, and the stopping rule. Further scientific
adaptation after fresh outcomes would require a separate new cohort; this
audit supplies no publication promise or open-ended search mandate.
