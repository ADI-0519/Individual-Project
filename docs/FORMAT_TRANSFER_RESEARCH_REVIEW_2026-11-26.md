# Format-transfer research review: what the next experiment could establish

26 November 2026. Targeted primary-source review and proposed research decision.
No checkpoint forward, classifier fit, calibration, case authoring, source/data
change, or test execution was performed for this document. The proposal below
has **not been executed**. It does not amend the completed study's seals or
reinterpret an exposed bank as a new holdout.

## Recommendation

First finish the finite, descriptive analysis of the saved frozen margins in
the [score-shift specification](FORMAT_SCORE_SHIFT_DIAGNOSTIC_SPEC_V1.md).
Distinguish loss of target ordering from scores crossing the old boundary.
Neither a failed threshold nor a successful supervised replacement identifies
the absence or causal use of a belief representation.

Do not launch a generic “mixed formats repair brittle probes” study as the
contribution. A directly relevant 2026 paper already tests style augmentation
and withheld-style evaluation. The narrower useful question is whether, at a
fixed capture and fixed observation budget, **changing feature coefficients
improves linked correction judgments beyond fitting an intercept alone**.
The fresh comparison in the [independent design review](FORMAT_TRANSFER_DESIGN_REVIEW_2026-11-26.md)
addresses that question, while crossing prompt packages can locate which part
of the bundled A/B change matters. This is an extension of established probe
transfer work, with a more controlled target and adaptation comparison.

The completed [fresh interpretation](FRESH_READOUT_RESULTS_INTERPRETATION_2026-11-23.md)
already supports a useful descriptive result: the factor-plus-rule pipeline
achieves 176/192 Llama and 118/192 Qwen hard quartets in A, versus 17/192 and
12/192 in B. It does not identify why. B changes system instructions, narrative
wording/presentation, event order and query wording together. Human semantic
review remains incomplete. The four-factor diagnostic is privileged supervision
plus an external rule; success is not evidence that the model computes that rule.

## Five closest inspected works

The five entries below are selected for relevance to this next decision, not as
an exhaustive ranking of the literature. Statements concern the inspected
versions and sections. Dates and identities were checked against official
records on 26 November; the coverage ledger records the actual reading.

### 1. Style augmentation already addresses the broad proposed repair

Kumar, [Pressure-Testing Deception Probes in LLMs](https://arxiv.org/html/2605.27958v1),
[official record](https://arxiv.org/abs/2605.27958): v1, 27 May 2026.

This is the closest intervention precedent. It uses Gemma 3, instructed
honesty/deception, last-response-token probes, eight style changes, and training
on clean data plus four styles while withholding four others. Table 11 gives
withheld-style AUROC .916 to .979 at 4B and .921 to .983 at 27B. The 27B .485
in Table 10 uses layer 27; Table 11 uses reselected layer 34. A .485-to-.983
comparison therefore bundles layer choice with augmentation. Section 6.2 allows
robust prompt-regime recognition as the explanation, and Appendix F retains
many instructed-deception cases whose answers contain the correct answer.
The inspected augmentation description does not establish equal total fitting
rows. Broad format fragility, augmentation and withheld-style rescue are already
covered. Our narrower extension would fix capture, formal states and row budget,
compare intercept versus coefficient adaptation, and evaluate dependent
correction quartets. [Methods/results/limits, PDF pp. 4–10; Appendices D–H, pp. 13–18](https://arxiv.org/pdf/2605.27958v1).

### 2. Belief-probe prompt brittleness is established

Bortoletto et al., [Brittle Minds, Fixable Activations](https://arxiv.org/html/2406.17513v3),
[official record](https://arxiv.org/abs/2406.17513): v3, 19 May 2025.
The v1 title was *Benchmarking Mental State Representations in Language Models*.

Their residual-stream logistic probes classify protagonist and oracle beliefs
in BigToM across 12 models. Random tokens, distracting belief statements,
time specification and an explicit initial belief affect probing accuracy;
protagonist probes are more brittle than oracle probes. They include label
permutation and PCA controls. Thus prompt-sensitive belief decoding, depth
comparisons and simple capacity controls are substantial prior overlap.
The inspected methods/results do not specify a frozen source-format-to-target
transport matrix with intercept adaptation isolated from coefficient refitting.
Do not infer that such an experiment was performed merely from curves comparing
prompt variants. Our target also concerns sender-attributed correction updates
and complete counterfactual panels. Their controls do not by themselves prove
that every possible nuisance is excluded. [Sections 3–4 and Appendix A.1–A.4, PDF pp. 3–7, 12–18](https://arxiv.org/pdf/2406.17513v3).

### 3. Bias transfer and direction transfer are already separated

Bürger et al., [Truth is Universal](https://arxiv.org/html/2407.12831v2),
[official record](https://arxiv.org/abs/2407.12831): v2, 21 October 2024.

They train using affirmative and negated factual statements and evaluate other
topics, compound statements and German. This already tests training coverage
as a repair for transfer. Most directly, Appendix D distinguishes two failures:
the transferred bias misclassifies one class despite good separation, and the
learned direction no longer separates classes. Its cities example has about
200 errors, all false statements, despite AUROC 1.0. Other datasets lose
separation. This is direct precedent for interpreting the project's high-AUROC,
low-threshold-accuracy control, so that distinction is not a new discovery.
Their target is factual statement truth, their polarity supervision differs
from our four factors, and their endpoint is not a linked correction quartet.
The useful extension is a fresh, fixed-site and budget-matched test of adaptation,
not another generic claim that threshold accuracy can hide retained ordering.
[Sections 3, 5.1; Appendices D–E, PDF pp. 5–6, 8–9, 15–16](https://arxiv.org/pdf/2407.12831v2).

### 4. Conversation position and standardised suffixes are close precedents

Ichmoukhamedov and Martens,
[Exploring the generalization of LLM truth directions on conversational formats](https://arxiv.org/html/2505.09807v1),
[official record](https://arxiv.org/abs/2505.09807): v1, 14 May 2025.

They test Llama-3-8B and Ministral-8B on 5,202 factual statements embedded in
short and longer conversations. Transfer worsens when the falsehood occurs
earlier; a common closing phrase improves it. A letter-counting control phrase
also helps, which limits a semantic-attention explanation. Crucially, they
centre activations separately for every format/topic, intentionally removing
format-dependent bias. Their conclusion explicitly leaves single-instance
centering/bias estimation unresolved. This is therefore close evidence for
format-sensitive directions and a shared-suffix intervention, not a demonstration
that one frozen raw-score threshold transports. Our fixed preprocessing and
intercept-only comparator would retain precisely that practical transfer
question. Their prefabricated falsehoods also do not establish intentional
deception. [Sections 2–4, especially centering and conclusion](https://arxiv.org/html/2505.09807v1#S2).

### 5. High ranking accuracy still does not establish nuisance selectivity

Klein, [Decodability Is Not Validity](https://arxiv.org/html/2611.07805v1),
[official record](https://arxiv.org/abs/2611.07805): v1, 5 November 2026.

Across sentiment/occupation targets, two models and multiple readouts, similar
target AUROCs coexist with different sensitivity to named nuisances. The design
varies target–nuisance training correlation, checks nuisance-only edits against
placebos, and separates descriptive test-selected comparisons from
development-selected methods. Decorrelated fitting usually reduces systematic
nuisance dependence; development nuisance diagnostics can improve selection.
Thus augmentation success or retained AUROC alone would not validate a belief
measurement. Preserve world/name counterfactuals and class-conditional scores.
The project's formal factor labels and linked update judgments differ from
these text-concept targets. Appendix A also documents imperfect judge agreement
and residual semantic changes, supporting the need for actual human review
without treating agent agreement as certification. [Sections 2–5; Appendices A–B, PDF pp. 2–5, 7–9](https://arxiv.org/pdf/2611.07805v1).

## Supporting work and checks on inherited citations

Ying et al., [The Truthfulness Spectrum Hypothesis](https://arxiv.org/html/2602.20273v1),
[record](https://arxiv.org/abs/2602.20273): v1, 23 February 2026. Sections 2–4
and Appendix A show multiple truth/deception domains and joint-domain training
recovering performance despite weak pairwise transfer. Layer, architecture and
token pooling are chosen using FLEED cross-domain performance; evaluation uses
five-fold cross-validation. Joint success therefore cannot establish that a
previously failed source direction was the only accessible representation, nor
does it establish unseen-format transport. It further narrows the novelty of
“mixed training finds a more general truth direction.” [PDF pp. 3–6, 12–15](https://arxiv.org/pdf/2602.20273v1).

Orgad et al., [LLMs Know More Than They Show, v4](https://arxiv.org/html/2410.02707v4),
[record](https://arxiv.org/abs/2410.02707): 18 May 2025. The inspected implementation
selects layers per token on validation data and probes generated-answer
positions; its cross-task results compare against a strong logit baseline.
Retain the project's output comparator and distinguish its pre-answer sites
from those generated-answer measurements. [Section 3.2, Section 4 result passage, Appendices A.2/A.4; PDF pp. 4, 7, 19, 22](https://arxiv.org/pdf/2410.02707v4).

The official records also confirm the inherited [Lookbacks v3](https://arxiv.org/abs/2505.14685)
date of 24 February 2026 and [Steele et al. v2](https://arxiv.org/abs/2607.11945)
date of 20 July 2026. Their substantive sections were not reread for this
document; their earlier coverage remains in the [evidence map](MAIN_STUDY_RESEARCH_EVIDENCE_MAP_2026-11-22.md).
The [Goldowsky-Dill et al. PMLR record](https://proceedings.mlr.press/v267/goldowsky-dill25a.html)
confirms ICML 2025 and pages 19755–19786. Its final PDF was downloaded and
identified here, but a page index is not substantive rereading; use the earlier
evidence map for its inspected methods and limitations.

## A defensible contribution, without firstness

**Completed observation:** under these two checkpoints, NF4 runtime, selected
sites, formal groups and renderers, correction-supervised factor decoding plus
a supplied update rule performs much better in A than its frozen A-trained
transport in B. Ingredient errors and nuisance failures also change. This is
a controlled reliability observation, not a validated deception detector,
absence-of-belief result, or proof of isolated composition failure.

**Candidate additional claim, conditional on a new result:** an equal-budget,
fixed-site comparison identifies whether mixed-format fitting improves fresh
linked correction judgments, and whether that gain exceeds an intercept-only
adaptation using the same observations. The contribution is the specific
adaptation decomposition, formal counterfactual endpoint and transparent
failure reporting. Existing literature already motivates each general idea.
No inspected work reports this exact combination; that bounded search finding
does not establish firstness, publication novelty or a guaranteed publishable
positive outcome.

Avoid translating a failed source head into “features disappeared.” Retained
ranking concerns one scalar readout; a successful replacement concerns access
under its supervision. Neither identifies how the model uses information.
Likewise, failure with one PCA component only tests that projection: every full
linear classifier itself defines a scalar direction, regardless of how many
PCA coordinates it needs.

## One discriminating fresh experiment — proposal, not executed

Use the [design review's finite fresh follow-up](FORMAT_TRANSFER_DESIGN_REVIEW_2026-11-26.md)
as the concrete starting point. Its central test should remain small enough to
finish and report regardless of the result. Complete human review of the
intended chronology and attributed-belief interpretation first. If this cannot
be done, explicitly retain exploratory, agent-audited status.

Fix each checkpoint's previous S1 capture, runtime, source preprocessing policy,
four factor targets and supplied fold. Keep Llama primary and Qwen secondary;
do not pool their shared contexts as independent evidence. Use disjoint new
fitting, development and paired test contexts. The old exposed bank can diagnose
the old instrument but cannot supply the fresh endpoint.

Use the same 512 underlying fitting states/groups, balanced before inference:

| Arm | Observations and adjustable parameters | Purpose |
| --- | --- | --- |
| A-only512 | All 512 states rendered A; fit four factor heads. | Full-budget source reference. |
| Mixed512 | Exactly 256 A and complementary 256 B renderings of those same states; fit four heads. | Primary matched-budget coverage intervention. |
| A-half256 | Fit only the 256 A observations used by Mixed512. | Diagnostic adaptation starting point. |
| Offset512 | Start from A-half256; freeze feature map and coefficients; fit one additive B intercept per factor on the same 256 B observations used by Mixed512. | Same-observation intercept-only comparator. |

Do not count rendering every state twice as a matched 512-row intervention.
Freeze regularization per observation, the scaler policy, format allocation,
selection objectives and calibration role. Intercept-fitting data are labeled
training information and must be budgeted as such. The Mixed512 versus
Offset512 diagnostic is especially useful because both have the same 256 A plus
256 B observations. A-only512 versus Mixed512 remains the primary budget
comparison. Use a common previously frozen feature map for the diagnostic
arms, so newly estimated normalization is not bundled with changing
coefficients. Match supervision *within* C2; comparisons with a one-label direct
head are secondary and do not have equal supervision.

Declare B hard complete-quartet accuracy primary, including every context and
panel. Report paired whole-context uncertainty for Mixed512 minus A-only512.
Keep A results and Mixed512 minus Offset512 as named diagnostics. Also report
raw, source-oriented factor AUROC, both class recalls, frozen-zero accuracy,
class-specific margins, all-four-factor correctness and world/name effects.
Use **easy plus hard** corrections for the principal factor diagnostics:
hard-only states satisfy `I = 1-M1`, so their two successful readouts do not
establish independent ingredient access. Do not invent a continuous C2 verdict
probability from its deterministic fold.

The format intervention is still bundled. If the intended claim concerns its
source, cross the A/B system, narrative package and query into eight fixed
packages, on the same fresh formal test states, as the design review specifies.
Keep all cells and interactions; do not choose a winning prompt retrospectively.
That crossing isolates packages, while narrative wording and event order still
remain bundled within the narrative package. All six new hybrid packages need
semantic review. A query-only edit occurs after an account capture and must not
change that identical earlier prefix's features; use this as an engineering
control rather than a claim about later query-position reasoning.

| Fresh pattern | Bounded interpretation |
| --- | --- |
| A-half loses fixed-boundary accuracy but retains ordering; Offset substantially restores panels | The tested B offset adaptation is sufficient for much of this instrument's failure. Generality across new shifts is untested. |
| Mixed exceeds Offset on B, including factor ordering and complete panels | These B observations help beyond one intercept per factor under this fixed feature map/family. It does not prove representation rotation or internal composition. |
| Mixed improves B but worsens A or nuisance invariance | Adaptation trades off measured reliability; do not advertise an invariant belief monitor. |
| Both adapted arms remain weak | The prescribed repair did not suffice. Information absence, universal inaccessibility and a causal reasoning defect remain unestablished. |
| Results differ across factors, contexts or checkpoints | Preserve the mixed result; a single global offset or single mechanism is inadequate as an account. |

If training sees A and B, success on B is seen-format adaptation. An untouched
third renderer would be needed for an unseen-format claim, but should be added
only for that question and sealed before inference. A/B alone is adequate for
the narrower declared adaptation comparison. Do not add formats, sites or
budgets after seeing whether the repair works.

## Coverage ledger and search limits

Primary reading used official arXiv HTML and downloaded versioned PDFs, parsed
locally without model calls. PDF page numbers below are 1-based physical pages.
Tables, prose and figure captions were inspected; plot pixels were not
independently digitised. This is not an independent reproduction of any paper's
code or reported measurements. Downloads reside outside the repository in a
temporary research cache. The earlier graph was queried for repository context;
the two named interpretation/evidence documents were then read directly.

| Work | Actual coverage for this review | Explicit limits |
| --- | --- | --- |
| Kumar v1 | PDF pp. 1–10 and 12–18: main text, methods/results/limitations, Appendices A–I; particular attention to Tables 10/11 and E/F construction. | Bibliography p. 11 was not substantively read. Augmented row count and split units remain insufficiently explicit; no external code audit. |
| Bortoletto v3 | PDF pp. 3–7, p. 9 limitations; Appendix A.1–A.4 and related text/captions pp. 12–18; prompt examples pp. 21–23. Also v1 HTML setup/results were initially consulted. | No attribution of v3 reading to the old evidence map. Did not reproduce probes or infer a frozen cross-format protocol from variant curves. |
| Bürger v2 | PDF pp. 5–6, 8–9, 14–16: Sections 3/5.1 and Appendices B–E; HTML setup and surrounding results. | Other-model Appendix G and real-world Appendix F were not read systematically here. |
| Ichmoukhamedov/Martens v1 | Root reviewer read HTML Sections 1–4, including methods/results/limits; this reviewer independently checked HTML Sections 2–4, including the centering and control-phrase passages. | Team reading contribution; no claim of independent PDF page-by-page reading or plot digitisation. |
| Klein v1 | PDF pp. 1–5, 7 and 9; Appendix B readout passage and p. 12 selection tables. | Appendix tables not listed were not comprehensively read; no human revalidation of edits. |
| Ying et al. v1 | PDF pp. 3–6, p. 9 discussion passage, Appendix A pp. 12–15; official record and HTML contents. | Does not claim complete reading of causal sections or supplementary plots. PDF extraction warned that some repeated graphical form content was skipped. |
| Orgad v4 | PDF pp. 4, 7, 19, 22; official record and HTML section index. | Limited targeted rereading, not full-paper coverage. |
| Lookbacks v3 / Steele v2 / Apollo final | Official identity/date checks; Apollo final PDF download and page index. | Substantive reading remains the earlier map's dated record. Downloading/indexing is not reading. |

Downloaded PDF SHA256 identities, recorded to disambiguate versions:

| PDF | SHA256 |
| --- | --- |
| Kumar v1 | `43f55118822b378d950255b10aa68379fdb24a1431bba7e8721aacaf44d856e0` |
| Bortoletto v3 | `dd0169539d0a82cedbcc4a346fa69303dd852eab02a85a06555c3afffd090763` |
| Bürger v2 | `badf34f0ae9eddd6c159ac8b637d66af5b3a939f296a78cc32e9755584dd1506` |
| Ichmoukhamedov/Martens v1 | `91976ace0be002436bf87933bad37fd41d3c65992768bfef0d0e8849ac54551f` |
| Klein v1 | `2ec45491ed7562b0a026711b9b17372ef3c9b9d12c2ed26f1c5d6e4dc551ddf4` |
| Ying et al. v1 | `917ade43effae49b4ee70257961cb76a9603f702985098691dc0c5f760cfdaf7` |
| Orgad v4 | `be0940142de147a298b4252a2e74f38c78f8401f92ad5ec429ae2e8ed8ac64a6` |
| Apollo final PMLR | `eac88802bf477cf722c028027d810a348b570f11370708d019fcc888ba88fa47` |

Search covered truth/deception/belief probes with prompt/format/distribution
shift, transferred bias, calibration, mixed/style-augmented training and joint
truth-domain fitting. Queries surfaced newer work missing from the old map,
especially Kumar and the current Bortoletto revision. Search results and
abstracts were discovery aids; substantive claims above use primary
methods/results/appendices. Other leads, including *Testing the Limits of Truth
Directions in LLMs* and off-policy probe generalisation, were not deeply reviewed
and support no substantive claim here. The former's record reports a newer v2
than the v1 downloaded during discovery, so that download was not treated as
current evidence. This finite search cannot establish absence of closer work,
firstness or saturation through 26 November 2026.
