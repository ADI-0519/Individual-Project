# What a lexical control can establish after the adaptation result

27 November 2026. Targeted primary-source review supporting a new, explicitly
post-outcome control analysis. This document changes no completed study's
inputs, fitting policy, primary endpoint or interpretation. The
[completed adaptation interpretation](FORMAT_ADAPTATION_RESULTS_INTERPRETATION_2026-11-27.md)
remains the record of the measured result. The proposed lexical analysis is
additional evidence on an already exposed test bank, not a newly untouched
confirmation.

## Recommendation

Fit a simple text classifier on the exact rendered fitting observations used by
the latent arms, with the same four factor labels and supplied update rule.
Evaluate complete correction quartets and the same world/name substitutions.
Add a system-only input restriction as an instruction/template shortcut control.
Freeze the complete lexical policy before opening its new scores, record its
post-outcome origin, and report every prespecified condition regardless of score.
This costs no new language-model forward passes and directly addresses a
remaining weakness: the current output-token comparator has different access
and supervision from the four-factor latent pipeline.

The text comparator is a **lexical shortcut control**. A word/character TF-IDF
classifier is not a capable neural text monitor or the strongest possible
text-only system. Success shows that this shallow text representation suffices
under the measured training and renderer conditions. Failure cannot establish
that activations contain unique information, that no text-only monitor could
match them, or that the model internally computes the externally supplied fold.
Even a numerical latent advantage concerns these fitted procedures, not all
text representations or private belief understanding.

## Three primary methodological sources

### Control tasks measure a particular source of probe capacity

Hewitt and Liang, *Designing and Interpreting Probes with Control Tasks*, final
[EMNLP-IJCNLP 2019 paper](https://aclanthology.org/D19-1275.pdf), DOI
[10.18653/v1/D19-1275](https://doi.org/10.18653/v1/D19-1275).

Their control labels are deterministic random functions of **word types**, so
repeated tokens retain the same arbitrary output. Selectivity subtracts control
accuracy from linguistic-task accuracy. Sections 3.2 and 3.5 show that ordinary
generalization control, dropout and early stopping do not reliably control this
memorization capacity. Section 5.1 expressly distinguishes type-level structure
from completely random example labels. Read Sections 1-3.2, 3.5-3.6 and 5.1-6;
the full intervening hyperparameter figures were not exhaustively digitized.

An example-level row-label permutation would consequently be a leakage/capacity
sanity check, **not their canonical selectivity experiment**. The current
system-only restriction asks whether instructions/templates predict labels;
it does not implement their word-type control. Their ELMo tagging results do not validate this
fictional belief bank's semantics. Their warning motivates the control without
establishing that the current latent result is an artifact.

### MDL is a later sample-efficiency question, not a substitute for controls

Voita and Titov, *Information-Theoretic Probing with Minimum Description Length*,
final [EMNLP 2020 paper](https://aclanthology.org/2020.emnlp-main.14.pdf), DOI
[10.18653/v1/2020.emnlp-main.14](https://doi.org/10.18653/v1/2020.emnlp-main.14).

Online coding first sends a block under a uniform distribution, fits on previous
blocks, and uses its predictive probabilities to encode the next block. Equation
3 includes the initial block cost; final-fit test log loss alone is not MDL.
Section 2.2.2 fixes seeds, training algorithm and block boundaries; its published
schedule runs from 0.1% through 100%. Sections 3.1-3.3 compare real and word-type
control tasks and probe settings. Read Sections 1-3.3, Appendix A tables and the
Appendix B.1 optimization passage; remaining edge-task results were not read
comprehensively.

An extension here would require whole-context blocks, predetermined context
orders, strictly preceding-block preprocessing and a clearly defined label
code. Repeated counterfactual rows cannot be shuffled across earlier/later
blocks as if independent. Four factor code lengths concern four supervised
labels, not the deterministic fold's unmeasured probability. With only sixteen
fitting contexts, block choice would be a substantial design decision. Defer
this extension until after the simpler comparator and human review; do not
label the current endpoint a compression result.

### Quantization stability needs a task-specific measurement

Singh and Sajjad, *Interpreting the Effects of Quantization on LLMs*,
[arXiv:2508.16785v3](https://arxiv.org/html/2508.16785v3), 20 November 2025;
the [official record](https://arxiv.org/abs/2508.16785) identifies AACL 2025
acceptance. The initially discovered v1 is not the version summarized here.

Section 3.3 compares bitsandbytes NF4/double quantization with FP16 compute,
8-bit loading and FP16 weights. Sections 4.1-4.5 report task/model exceptions
alongside broadly modest changes; correlation analysis is limited to Phi-2 and
Llama-2-7B on BoolQ. Section 7 limits the configurations and single-token tasks.
Read the current HTML Sections 1-4.5 and 6-7, Appendix B thresholds, and targeted
related-work passages; this was not an external code replication. Their
activation/attribution/calibration findings do **not** establish preservation of
our belief heads, NF4-to-BF16 transport or adaptation effect. They support a
task-specific precision check, not an expectation that its answer must be
positive.

The [CREATE proposal](CREATE_ADAPTATION_REPLICATION_PLAN_V1.md) therefore
distinguishes matched per-profile refitting from transporting a frozen NF4
head, and requests a same-hardware NF4 reference. Moving simultaneously from a
local GPU/NF4 model to CREATE/BF16 cannot isolate precision by itself.

## Required implementation and reporting safeguards

The following are project-specific recommendations, not procedures claimed to
have been executed by the three papers.

1. Build inputs exclusively from the model-visible rendered messages. Exclude
   formal states, labels, case IDs, split names and review answers from text
   features. Store input hashes and explicit feature-policy settings.
2. Retain the exact A512 and complementary Mixed512 state allocation. Fit
   vocabulary and IDF using the declared fitting observations only. Test text
   must never influence vocabulary, IDF, regularization, feature-family choice,
   thresholds or orientation. Disclose any shared unlabeled fitting information.
3. Use the same factor targets and external fold for the main text comparator.
   A final-label classifier would answer a different supervision question and
   must be named separately if later studied. Preserve all cases, including
   one-step/asserted out-of-training-construct controls.
4. Fix word/character feature definitions, optimizer, regularization and any
   random seeds prospectively for this new analysis. A new setting chosen after
   looking at scores is an exploratory follow-up, not a rescue replacement.
5. Score complete quartets, factor discrimination, both class recalls and
   world/name flips with both-correct counts. Lower nuisance flips alone can
   come from a constant answer. Keep context clusters intact for uncertainty;
   shared text produces one comparator, not two independent model replications.
6. Report the system-only restriction on the full denominator. Low accuracy
   under that restriction cannot exclude other shortcuts in full messages.
   A training-label shuffle could provide an additional future sanity check,
   but is not included in the current fixed diagnostic. Neither a failed shuffle
   nor low restricted-input accuracy proves absence of all leakage. Do not assign
   a naive independent row chance rate to dependent quartet outcomes.

The existing new-bank allocation is balanced in aggregate but format depends on
world within each fitting context. Fully crossed test controls diagnose
consequences; they do not erase that dependency. Human semantic review also
remains incomplete. These limits apply regardless of the lexical result.

## Contribution boundary for Hanqi

The established result is partial recovery of an attributed-belief instrument
under a known format shift, with source-format cost and failures retained. The
lexical analysis can clarify whether a simple surface learner can achieve the
same operational judgments. The BF16 proposal can test robustness to a further
implementation change. Neither establishes spontaneous reader anticipation,
real communicative deception, unique belief information or a causal internal
algorithm. Broad format fragility and augmentation already have close precedents
in the [earlier review](FORMAT_TRANSFER_RESEARCH_REVIEW_2026-11-26.md).
This finite search establishes no firstness or publication guarantee.
