# Main-study decision: training coverage and compositional transfer

**21 November 2026. Recommended design, not an executed or frozen protocol.**
This brief responds to the supplied external critique. Three independent agents
checked original training support and four close primary papers; the root review
also inspected the evidence-weight and prompt-dependence papers and the supplied
assessment guidance. No new model calls, fits, threshold changes or software
frameworks were commissioned for this decision.

## Recommendation

Keep the project, end the exploratory pilot sequence, and make **matched
training coverage followed by untouched compositional evaluation** the next
study. The legacy head is historical context, not the sole main-study subject.
This revises the earlier frozen-head-only scaling recommendation.

The main question is:

> At a fixed training budget, does including belief-preserving examples improve
> second-order belief prediction on unfamiliar compositions of information
> updates, while preserving performance when beliefs genuinely change?

This is a predictive robustness/generalization question. It does not by itself
identify a causal belief mechanism or demonstrate spontaneous anticipation.
Apollo deception-probe adaptation remains a separate reproduction component;
deliberately misleading natural communication remains motivation and a possible
extension. Make that emphasis change explicit to the supervisor.

## What the audit changes

The saved `nested_question__sender__nested__layer_16` head used **32 training
rows but only 16 distinct sender inputs across four wordings**. Validation used
16 rows/eight distinct inputs across two wordings; test used the same counts
with two different wordings. Hidden actual attendance creates duplicate sender
inputs with identical nested labels. The unique training labels are balanced,
eight per class. This was a legitimate development fit, not broad coverage.

The [original bank](../data/belief_diagnostic/development_v1.json) always moves
the object to the opposite location. Its target is `W if B else 1-W`. The new
[crossed assay](CROSSED_BELIEF_VALIDATION_RESULTS.md) changes update channel,
initial/message relation, world/message agreement and framing together.
It establishes imperfect transfer of this particular pipeline. It does not
establish failure of well-supported belief probes in general.

Training-only scaling and row normalization, validation-only C selection and
absence of test fitting were checked in the persisted fit artifacts and
[fitting code](../src/latent_probes/belief_query_readout.py). No numerical
discrepancy was found. The issue is explanatory scope, not invented results.

## Closest work and the remaining uncertainty

| Primary source inspected | What it already establishes | Design implication |
| --- | --- | --- |
| [Klein, 2026, Sections 2-5 and appendices](https://arxiv.org/html/2611.07805v1) | Accuracy and nuisance sensitivity can differ; controlled training correlations and decorrelation already matter. | Training balance repairing a shortcut is insufficient novelty alone. |
| [Steele, Wen and Han, 2026, Sections 3-4; Appendices A-B](https://arxiv.org/pdf/2607.11945v2) | Access manipulations can change inferred beliefs while preserving asserted beliefs; preliminary nested-belief interventions exist. | Selective sensitivity and second-order decoding are not new general ideas. |
| [Li et al., ICML 2025, Sections 3-5; Appendices B, G](https://proceedings.mlr.press/v267/li25bj.html) | GridToM includes second-order true/false-belief probing; overreliance on false-belief patterns is discussed. | A change-only head failing no-change examples is a weak standalone contribution. |
| [Prakash et al., ICLR 2026, Sections 2-6; Appendices D, M](https://arxiv.org/html/2505.14685v3) | Query-time retrieval and distributed story bindings; capability differs strongly across checkpoints. | One pre-question final-token failure cannot establish absent belief information. |
| [Li et al., 2026, Sections 2-6](https://arxiv.org/html/2611.07913v2) | Matched immediate predictions can have different responses to later evidence in a probabilistic task. | Correct unchanged YES/NO answers cannot prove identical internal beliefs. |
| [Noel, 2026, Sections 3-9](https://arxiv.org/html/2608.13329v1) | Contrastive framing changes probe statistics; item count cannot substitute for prompt coverage. | Cross wording independently and avoid a one-framing scale claim. |

These checks support a candidate **coverage-by-composition** study, not a
firstness claim. The inspected papers do not establish the exact proposed
experiment's outcome. An application to second-order labels is not automatically
a paper contribution; a substantive result and independent replication matter.

## Principal comparison

| Arm | Role | New fitting data |
| --- | --- | --- |
| Legacy head | Historical reference | Existing 16-input development fit; unchanged |
| Restricted coverage | Main comparator | Fresh inputs where new information differs from the prior attributed belief |
| Full coverage | Main treatment | Fresh inputs containing both differing and redundant information |

Both new arms must share the model/precision, capture layer and token position,
input perspective, source channels, scenario groups, wording inventory, training
N, class balance, receipt distribution, world-truth balance and selection policy.
The intended initial/message relationship is the manipulated joint distribution;
do not attempt to match away that treatment. Match other factors and audit the
allocation before feature collection. Do not compensate with duplicate rows or
extra restricted-arm wordings.

Use genuinely different, balanced training-subset allocations for repeated fits,
paired between arms. Repeating only a deterministic classifier's random seed
would not establish training-data robustness. A predeclared small learning curve
on the same captured feature pool can separate sample size from coverage; it
requires extra classifier fits, not extra language-model runs. Choose feasible
N values and draws before new outcomes.

For the coverage comparison, preserve identical operating-point and regularizer
procedures. Fit preprocessing only on each arm's training data. Separate scalers
make the treatment a change to the whole training pipeline, not just its weights.
Record all selected settings. Never tune on the held-out composition.

## Data and split

Use observation and message-access mechanisms in training/development. Test on
fresh scenarios from both and a held-out two-step correction mechanism. The
correction must require composition absent from both training and development;
new nouns around the same one-step rule are not a mechanism holdout. One held-out
composition supports a claim about that composition, not arbitrary new families.

Keep whole narratives, variants, entities and related wording groups within
their assigned partition. Author independent scenario groups rather than
counting name swaps as new situations. A state tracker supplies gold only after
the prose's transition and information-access rules are reviewed.

The attachment's 48 training, 16 development and 90 test scenario groups are
planning suggestions, not a power calculation. Its 4,928-prefix count assumes
32 variants for every mechanism. For example, a correction design with
`I, M1, B1, B2, W` and two mappings has 64 combinations when M2 is fixed as the
opposite of M1; independently varying M2 gives 128. Hidden delivery duplicates
are metadata, not extra calls. Resolve the exact transition graph, legitimate
pair/quartet membership and allocation before freezing counts or compute.

The student should review every base narrative and transformation specification.
An independent human should blindly review a documented stratified sample
covering all mechanisms and logical paths, plus disputed cases. Record actual
review coverage and disagreements. Agent agreement or a blank review packet
does not complete human validation. Do not collect held-out outputs before this
audit and the analysis policy are frozen.

## Endpoints and baselines

The recommended primary endpoint is the **paired full-minus-restricted
difference in complete-quartet accuracy on the held-out composition**, with
quartet construction and the primary training N/model fixed in advance. A valid
quartet includes a belief-changing receipt pair and a belief-preserving pair,
and requires all four predictions correct. Initial-only/message-only shortcuts
cannot solve a complete valid quartet.

For multi-step cases the relevant prior is the sender-attributed belief
immediately before the final update. Not every factorial cell necessarily has
all four counterparts. Specify eligibility prospectively, retain remaining
cells in declared secondary evaluations, and never invent missing counterparts.

Report both kinds of pair jointly: both endpoints correct on changing pairs,
both correct on preserving pairs, preserving-pair verdict flips, oriented score
changes, per-class accuracy and threshold-free discrimination. Confidence may
change under repeated evidence; identical scores are not a necessary validity
condition. Resample independent scenario groups with variants intact, and show
training-allocation variability separately. Do not treat thousands of variants
or optimizer seeds as independent scenarios.

Baselines: constants; initial/message/world shortcuts; a simple text classifier;
same-model output candidate probabilities; a preselected capable full-context
text monitor; and a clearly identified symbolic oracle. Give practical output
and probe comparisons equivalent development-only calibration opportunities,
and preserve uncalibrated scores too. Do not claim a latent advantage by
comparing a trained probe with an obviously biased raw YES/NO threshold.

Retain all reviewed test cases for robustness evaluation. Poor verbal answers
are reportable evidence, not a reason to stop this study. Correct-answer-only
analysis can be a predeclared secondary diagnostic; it must not replace the
full evaluation or prove unchanged internal belief.

## Scope, compute and success criteria

Use existing Llama 3.1 8B software for the primary study. Preselect Qwen2.5-7B
Instruct as a second-family replication, fitting its own probes under the same
design. Prior results motivate that choice but do not guarantee success on this
task. No cross-model coefficient reuse, language-model fine-tuning or new MLOps
platform is required. Capture activations once per model and reuse them.

CREATE can supply a separately declared same-hardware NF4/BF16 comparison or
larger reference. It should not delay the core study, and current failures do
not identify quantization as their cause. Existing submitted/unsubmitted compute
plans are not silently enlarged by this brief.

Useful outcomes include coverage repairing familiar cases but not composition,
transferable repair across both models, or both latent arms failing where text
methods succeed. If latent and text methods both fail, attribute the limit to
the evaluated system/task rather than the probe alone. A negative result needs
these discriminating comparisons; another isolated low score adds little.

If adding the missing training cells only fixes familiar examples, that is a
valid dissertation observation but likely a weak standalone paper. The more
substantive question is the limit or success of transfer under composition,
after controlling training quantity, wording and practical baselines. Evaluate
every prespecified outcome before choosing the eventual publication claim.

Keep future-goal contamination and spontaneous communication anticipation as
extensions after the core result. Any extension needs its own frozen protocol
and valid instrument; no old failed gate is relabeled or resumed.

## Immediate deliverables

1. A one-page supervisor scope/contribution statement and a small representative
   set showing one-step change, redundancy, and a genuinely composed correction.
2. One reviewed data specification, split manifest and matched training-allocation
   policy, including exact prompt counts and a measured feasible review budget.
3. A frozen main protocol, then one complete execution, group-level comparison
   and results/methods draft. Existing cases remain development evidence.

Against the supplied assessment guidance, excellence comes from justified
design, critical comparison, systematic evaluation and clear writing. The
[deliverables guide](../../research/_work/extracted/BSc%20Individual%20Project%20Guide%20to%20Deliverables.txt)
explicitly warns that good software alone is insufficient. No mark or paper
acceptance is guaranteed by this proposal. The next work should complete an
empirical argument rather than add another pilot or general framework.
