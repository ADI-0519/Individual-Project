# Next experiment review: readout adequacy before a composition claim

22 November 2026. Independent agent review, using the completed results and
primary methods/appendices. This document proposes work; it does not freeze a
new experiment, run a model, fit a head, change the completed study or certify
human validation. All proposed empirical work can proceed as explicitly
agent-audited exploratory research under the existing authorization. No new
human approval gate is introduced here.

## Recommendation

Do a cheap, prospectively specified diagnostic on the already exposed correction
features first. Compare correction-trained linear, bounded nonlinear and
factor-decoding-plus-fold methods with grouped outer holdouts. Then run one
fresh study that crosses a small development-selected layer/token panel with
one-step, easy two-message and harder correction cases in two fixed renderers.

The first step asks whether the saved midpoint features support the target after
direct correction supervision. The fresh step distinguishes the original capture
choice, sequence support and format sensitivity. Neither step needs a positive
result to be useful. Do not rerun progressively larger coverage comparisons or
keep choosing locations until correction succeeds.

The completed study remains a valid test of its frozen fitted pipelines. Its
primary result does not identify a general absence of belief information, or an
isolated composition failure. This distinction is especially important because
Qwen also has weak one-step joint consistency, and the easier same-format Llama
correction cells are imperfect. See the unchanged
[final interpretation](MAIN_STUDY_FINAL_INTERPRETATION_2026-11-22.md) and
[generated results](results/main-study-2026-11-22-complete/results.md).

## What the existing implementation does and does not establish

I inspected the [specification](MAIN_STUDY_SPECIFICATION_V1.md),
[novelty audit](MAIN_STUDY_NOVELTY_AUDIT_2026-11-22.md),
[result context](MAIN_STUDY_RESULT_CONTEXT_2026-11-22.md),
[final interpretation](MAIN_STUDY_FINAL_INTERPRETATION_2026-11-22.md), and the
relevant capture, fitting, rendering and scoring code:
[main_study_run.py](../src/latent_probes/main_study_run.py),
[main_study_fit.py](../src/latent_probes/main_study_fit.py),
[main_study_bank.py](../src/latent_probes/main_study_bank.py),
[main_study_analysis.py](../src/latent_probes/main_study_analysis.py), and
[main_study_monitor.py](../src/latent_probes/main_study_monitor.py).
The existing Graphify graph was queried first; its older, truncated results were
used for navigation, not as authority for the current code or papers.

The capture hook actually takes the final complete chat-template token at one
decoder-block output, before an answer: block 16 for Llama and block 14 for Qwen.
The first-row hook/HF equality check addresses implementation alignment. It does
not validate the scientific adequacy of that position or depth. No layer search
was performed, so there is no selection bias from correction labels in the
original result; equally, there is little empirical basis for extrapolating its
null beyond that readout.

The learned pipeline is a training-only StandardScaler, row normalization and
regularized logistic regression. Development selects regularization; it does not
teach a nonlinear correction rule. A source-trained linear classifier can fail
because the required facts are unavailable at its location, because their
encoding changes, because its decision surface is inadequate, or because the
source support does not identify the new task. These explanations require
different controls.

There is a simple expressivity issue independent of any particular transformer.
Writing binary belief content as a slot index, the one-step target is

```
Y = U if B else I.
```

For primary correction cases, put `m = M2 = I = 1 - M1`. Then

```
Y = m XOR (B1 AND NOT B2).
```

On the slice `B2=0`, this contains XOR of `m` and `B1`. An affine classifier over
separately represented raw facts cannot solve that slice perfectly. This is an
algebraic property of the label rule, not evidence that a transformer hidden
state has that factorized geometry. The model could already encode an integrated
answer linearly. Consequently, finding decodable ingredients without a good
linear final-label probe would be informative, but would not show absent belief
content. The one-step full-support rule is also a multiplexer: linear
inadequacy is not unique to correction.

The prose monitor supplies an important solvability comparison, with a different
computational contract. It extracts several fields, can generate 384 tokens and
receives an external update algorithm. Its correction advantage does not isolate
text access, additional inference or symbolic composition. A latent factor
decoder followed by the same rule is therefore a particularly useful comparison,
with its additional factor supervision stated plainly.

## Immediate diagnostic using saved features

This is an explicitly posthoc analysis of exposed material. It does not create
an untouched test set, recover confirmatory status or change the original
coverage result. Freeze its complete policy and source hashes before any new
fits. Use both models, all 30 correction groups, and all 64 cells per group.

1. Assign whole groups deterministically to five outer folds of six groups.
   Within each outer training set of 24 groups, deterministically reserve six
   for inner development and fit on 18. Preserve every factor, mapping and
   quartet from a group in its assigned split. Use the same folds for every
   method and model. Do not split rows at random.
2. Compare correction-trained logistic regression; one fixed RBF-kernel
   classifier family with a small declared regularization/width grid; and four
   logistic ingredient heads for `I`, `M1`, `B1`, `B2`, followed by the exact
   declared fold (`M2=1-M1`). Use the same feature vector and outer training
   groups. Factor labels are visible-story facts, not the final belief label.
3. Estimate each scaler, distance scale and other fitted transformation on the
   inner training groups only. Select hyperparameters using the declared inner
   development criterion. If refitting on all 24 outer training groups, refit
   transformations there and retain the selected settings. Never expose outer
   holdout labels to these operations.
4. Use a fixed decision convention. Logistic heads can retain strict `p>.5`,
   tie zero; a kernel margin can use strict positive margin, tie zero. Preserve
   orientation from training labels. Do not enable an undocumented internal
   random-row calibration, reverse signs using holdout AUROC, or manufacture
   Brier scores from uncalibrated kernel margins.
5. Retain failed candidates and nonconvergence. Define the fallback or recorded
   failure policy before fitting. Add one fixed training-label permutation as
   an implementation sanity check. It is not a reproduction of a linguistic
   word-type control task or a complete construct-validity test.

Report primary `I=M2` quartets and secondary `I=M1` cells separately, along with
all-row and per-class accuracy, signed AUROC where defined, each ingredient's
accuracy, all-four-ingredients-correct rate, and final folded-belief performance.
High individual ingredient accuracy can still give poor joint composition.
The factor method uses four labeled targets and a supplied algorithm; describe
that advantage rather than calling it a matched single-label linear probe.

Give the pooled out-of-fold results, all group scores and the five fold scores
as descriptive diagnostics. The five outer models share training data. A naive
bootstrap of saved out-of-fold predictions over groups does not capture that
training dependence, and a five-fold standard error does not supply five
independent replications. Do not attach an ordinary independent-groups confidence
interval to a comparison of these overlapping fitted models. A later fresh
fixed-model cohort supports a cleaner conditional group bootstrap.

| Diagnostic pattern | What it supports | What remains unresolved |
| --- | --- | --- |
| Correction-trained linear succeeds | The saved midpoint features can predict correction with target support. | Why the source head failed; fresh distribution transfer. |
| Nonlinear succeeds substantially beyond linear | Nonlinear target accessibility at this saved location under correction supervision. | Whether the model uses that computation or a stable belief representation. |
| Ingredient heads plus fold succeed | The location preserves usable evidence ingredients, given extra supervision and a supplied rule. | Whether an integrated final belief is linearly available or internally used. |
| Every method is weak | These particular data, location, capacity and fitting budgets do not recover the target well. | Other locations, pooling, formats, precision and model computation. |

This diagnostic costs CPU fitting and saved-array access, with zero model calls.
It is worth doing before paying to recapture thousands of prefixes. Its result
may inform a subsequently frozen fresh protocol; it cannot be used to rename
the old correction bank as fresh evaluation.

## Fresh study: a bounded design with discriminating controls

The following is an affordable concrete option, not a completed power analysis.
Reuse a fixed 512-row full-support source allocation and the existing 512
one-step development rows. Choose the allocation by an arbitrary recorded rule,
such as the first declared seed, 11, rather than its observed performance. The
old source material can legitimately train/select a new pipeline for evaluation
on new stories; disclose that the source task and original outcomes are already
known. Preserve both observation and message source cases if reusing that exact
allocation. Treat fresh message-only transfer as the closest sequence control.

Add 48 genuinely new contextual groups, disjoint from all previous banks and
from each other as follows:

| New cohort | Groups | Captured cells per group | Prefixes |
| --- | ---: | ---: | ---: |
| Correction support, renderer A | 16 | 32 | 512 |
| Correction development, renderer A | 8 | 64 | 512 |
| Fresh paired test, renderers A and B | 24 | 104 per renderer | 4,992 |

Together with 512 source fitting and 512 source development prefixes this is
**7,040 prefills per model**. Rendering variants do not create additional
independent groups: the fresh test has 24 contextual units. With sufficient
budget, author fresh source fitting/development groups too, but this is not
necessary for a fresh evaluation claim. Do not substitute renamed copies for
new contexts.

Before any corresponding outcomes, have an author-independent agent interpret
every new base context and a recorded sample spanning every normalized state
path in both renderers, with symbolic labels hidden during interpretation.
Record disagreements, revisions and exactly which expanded prompts were read;
validate all remaining expansions mechanically against the frozen renderer.
Bind final prose, splits, masks and review records by hash. This is agent
semantic review, with possible correlated blind spots, not human validation or
a reason to require a new human gate before authorized exploratory execution.

For correction fitting, the 32-cell half factorial must contain both initial
strata and all receipt paths, with a predetermined nuisance-balanced mask. One
concrete mask retains cells where `I XOR M1 XOR W XOR name_map` equals a fixed
group offset, independently of both receipts; paired contexts use complementary
offsets. This covers every `I,M1,B1,B2` path within each story while sampling its
world/mapping combinations. Publish the mask and marginals. Sampling it after
model outcomes would invalidate the comparison.

Each test context has these **104 cells per renderer**:

- 32 fresh one-step message cells, using the original full state factorial.
- 64 two-message cells, keeping the original correction operator. Their 32
  `I=M1` cells are the same-format, lower-dependency control; their 32 `I=M2`
  cells are the harder correction target. This holds replacement wording,
  event count and broad length fixed between these two strata.
- Eight explicit nested-attribution cells, balancing the asserted value,
  private world value and location mapping. The attributed final belief is
  stated directly and the rest of the account is harmless context. Evaluate
  direct output and transfer to this condition as a positive control for
  reading/elicitation. A head trained only on inferred beliefs is not guaranteed
  to transfer to assertions, so its failure here is not by itself a failed
  implementation test or grounds to exclude data.

Define renderer A before capture as the familiar syntax. Define B as an
independently authored semantic equivalent with different sentence construction,
receipt phrasing and clause order, retaining explicit timestamps and the same
rules. B is absent from fitting and all development. Both renderers contain
one-step and two-message conditions; this permits a format failure to appear
without a sequence change. These are two fixed renderers, not a sample of
arbitrary natural language. The explicit-assertion control and the easy
correction stratum must also preserve world/belief conflicts and mappings so
simple copying or reality defaults can be observed.

The easier two-message target is a function of `B2` and the two message values;
it does not require `B1`. Its four-receipt panels consequently have a different
label pattern from the harder primary quartets. Report that geometry and
corresponding heuristic baselines; do not invent a common `0.5^4` chance floor.
Comparing the strata is a useful diagnostic, not a perfect intervention on a
single abstract faculty of composition. A further length-matched padding
condition could distinguish length from event structure, but is not required
for this affordable first design.

### Capture and selection

Before outcomes, fix four depths: approximately one quarter, one half, three
quarters and the penultimate block, including the exact original midpoint.
For the pinned configurations, an example is Llama blocks `[8,16,24,30]` and
Qwen `[7,14,21,26]`. Resolve and record the block count, zero-based indices and
HF alignment from the actual pinned configuration rather than assuming them.

Fix three meaningful pre-answer positions using verified renderer spans:
the end of the account before the question; the final subtoken of the queried
location in the explicit question; and the original final complete chat-template
token including the assistant header. Causal masking matters: earlier positions
have not read later question text. They test different information availability;
an earlier successful prediction does not establish spontaneous communication
anticipation in a task with repeated fixed targets and explicit update rules.

The 12 vectors are cached during the same prefill, by selecting only the needed
positions in hooks. They do not require 12 independent model passes. Preserve
float32 vectors and token IDs; approximately 1.38 GB stores the Llama feature
matrix for 7,040 prefixes, 12 sites and 4,096 dimensions before metadata. Avoid
retaining every layer at every token merely to obtain these vectors.

At each site fit the same source full-support logistic procedure, using the
same 512 source row IDs. Choose regularization and one source site by a fixed
group-weighted source-development log-loss rule with declared ties. Correction
support/development and all fresh test labels are excluded from this choice.
Freeze every selected site/head before fresh test capture or scoring. Report
the original midpoint/token under the same fitting policy as a fixed comparator;
the archived original head may also be replayed as a separately named historical
reference. Their fitting policies should not be silently conflated.

At the frozen source-selected site compare:

1. Source-trained linear readout, versus the original fixed-site comparator.
2. One predeclared bounded nonlinear readout trained on the same source rows.
3. Correction-trained linear and nonlinear readouts using the new correction
   support and its separate development cohort.
4. Correction-trained ingredient heads plus the declared fold, using that same
   correction support/development.

The latter methods are target-adapted diagnostics, not source-only transfer.
As a secondary ceiling, allow a correction-development-selected linear site
from the same 12 cached sites, explicitly labeled as target adaptation. This
helps distinguish a source-selected site's inadequacy from the tested panel's
limitations. It cannot supply an unadapted-transfer claim. Publish all panel
results without choosing a new winner from test labels; any claim based on the
best test cell needs a multiplicity-aware analysis and fresh confirmation.

Do not silently add generated answer tokens to this panel. A separately frozen
post-answer probe would address a different computational stage and needs its
own output comparator. In particular, supplying the correct answer as a token
would create a trivial information leak.

### Outcomes and computation

The main fresh diagnostic contrast is source-selected versus original fixed-site
linear correction-quartet performance. Publish absolute one-step, easy and hard
two-message performance in both renderers, regardless of whether positive
controls work. Quantify the easy-versus-hard and A-versus-B drops, with all
methods retained. Failure of the one-step or easy controls limits the
interpretation; it does not cancel or hide the experiment.

Average within context first. Use a paired bootstrap over the 24 fresh contexts
with every renderer, state variant and method retained together, conditional on
the frozen fitted models. This does not include uncertainty from selecting
training stories, authoring renderers or sampling model families. Report the
two models separately; they share the test contexts. Twenty-four contexts give
limited precision for small differences, so avoid equivalence or universal-null
claims from a non-significant result.

Same-model YES/NO output probabilities are available from each ordinary final
prefill and remain a cheap comparator. Keep source orientation and any
source-development calibration frozen. A renewed 384-token text monitor is
additional generation, not free or compute-matched to the probe. If rerunning
it, predetermine a stratified panel and keep invalid outputs incorrect. An
optional equal-cap direct-reasoning versus evidence-extraction comparison can
test two prompted pipelines; report actual generated tokens, latency and the
external fold. Equal token caps alone do not isolate the fold's benefit or
guarantee equal realized compute. The latent ingredient-plus-fold method is
already a useful zero-generation scaffold comparator.

Retain the same pinned models and NF4 configuration for this diagnosis. A
full-precision comparison changes another factor and is not necessary to test
whether the current capture is adequate. Both successful and unsuccessful
results have a valid interpretation. Estimate wall time from measured prefill
profiles; the prefix count and cache estimate above are not a promised runtime.

## Exact primary-source reading and its design implications

All links below were reopened on 22 November 2026. These are targeted section
readings, not claims to have read every page or exhaustively searched the field.

**Prakash et al., [Language Models Use Lookbacks to Track Beliefs, v3](https://arxiv.org/html/2505.14685v3).**
Inspected Sections 3–6, Appendix H, M and O; verified
[metadata](https://arxiv.org/abs/2505.14685), v3 dated 24 February 2026.
Section 3.1 conditions the mechanistic experiments on correct behavior. Section
5.1 distinguishes a final-token pointer at intermediate depth from the later
answer payload. Appendix H describes separate mask fitting/validation; Appendix
M reports different final-token subspaces across distributions despite related
lookback structure. Appendix O's added-visibility condition does not reliably
work. These details motivate multiple fixed sites and full-cohort outcomes;
they do not establish the mechanism in our smaller NF4 models. Our fresh
behavior-correct subset can be secondary, never the primary denominator.

**Steele et al., [Belief–reality separation lives in routing over a shared value slot, v2](https://arxiv.org/pdf/2607.11945v2).**
Inspected Sections 3.1–3.4 and 4.1–4.4, and Appendices A–C; verified
[metadata](https://arxiv.org/abs/2607.11945), v2 dated 20 July 2026.
Clause-order and two-value controls matter to their readout claims. Section
4.4 explicitly distinguishes frame information recoverable by a probe from the
frame-agnostic subspace identified by intervention. Appendix B's nested slot
transfer is on Qwen7B-Instruct, while the nested router uses another substrate.
Appendix C's scaffolds distinguish elicitation from capacity; Appendix A reports
Llama8B sensitivity to task framing. Thus explicit assertions, query positions
and held-out wording are useful controls. Neither generic nested probing nor
scaffolded composition is a new contribution here.

**Klein, [Decodability Is Not Validity, v1](https://arxiv.org/html/2611.07805v1).**
Inspected Sections 2–5, Appendix A's validation passages and Appendix B's readout
ablation; verified [metadata](https://arxiv.org/abs/2611.07805), submitted
5 November 2026. Section 3 uses several depths and mean pooling, with a last-token
ablation. It explicitly distinguishes descriptive test-selected equivalence
from development-selected methods. Section 4.1 selects using nuisance-aware
development labels; the discussion limits claims to named nuisances and linear
readouts. Our next site selection must use development only, and world/mapping
controls must accompany target accuracy. A new correlated-versus-decorrelated
fit alone would repeat an occupied question. Our potential contribution is the
specific sequence/format diagnosis with a frozen instrument.

**Orgad et al., [LLMs Know More Than They Show, v4](https://arxiv.org/html/2410.02707v4).**
Inspected Sections 3.1–3.3 and 4, Appendix A.2 and the A.4 baseline passages.
The current [record](https://arxiv.org/abs/2410.02707) identifies v4, dated
18 May 2025; the previous project audit explicitly used v2. The latest methods
select layers per token on validation data and distinguish generated exact-answer
positions from prompt positions. Section 4 compares transfer with logits and
uses absolute AUROC, permitting sign reversal. A.2's answer-extraction filtering
is an additional selection operation. We should preserve our source orientation,
include the output comparator, and count invalid generated methods in full
denominators. The paper's generated-answer findings do not imply that our
pre-answer belief label should be found in the same place.

**Goldowsky-Dill et al., [Detecting Strategic Deception with Linear Probes, ICML 2025](https://proceedings.mlr.press/v267/goldowsky-dill25a.html).**
Verified the publication and inspected the actual
[32-page PMLR PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/goldowsky-dill25a/goldowsky-dill25a.pdf),
Sections 3–4.2, 5.1–5.2 and Appendices C, D.1–D.5. The web reader rejected the
octet-stream PDF, but it was downloaded into memory and read with pypdf; the
earlier accessible arXiv-v1 text was also checked. Their instructed-pair examples
are off-policy true text, their main monitor aggregates response-token scores,
and D.2 reports sensitivity of low-FPR recall to neighboring layers. D.5 refits
last-position probes rather than merely changing aggregation. Context-sensitive
false alarms and aggregation failures are already documented. Our new work can
inform monitor validation, but it is neither a 70B reproduction nor evidence of
intentional deception.

**Bortoletto et al., [Benchmarking Mental State Representations in Language Models, v1](https://arxiv.org/html/2406.17513v1).**
Inspected Sections 3.1–3.4, 4.2–4.3, and Appendix A.3–A.4. The paper tests
prompt variants and probe capacity, including PCA. This directly precedes a
layer/prompt adequacy diagnosis. Merely recovering higher belief accuracy at a
different site, or showing prompt sensitivity, is insufficient novelty.

**Bürger et al., [Truth is Universal, v2](https://arxiv.org/html/2407.12831v2).**
Inspected Section 5.1 and Appendices D–E. Their affirmative/negated fitting data
are tested on conjunctions, disjunctions and German; Appendix D distinguishes
direction transfer from transferred-bias failures. Coverage followed by logical
composition transfer already exists. Our diagnosis must concern this controlled
attributed-belief pipeline, with frozen thresholds and a new cohort.

**Hewitt and Liang, [Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/).**
Verified the ACL publication and inspected the
[paper](https://aclanthology.org/D19-1275.pdf)'s Sections 2–3.2 and introductory
selectivity discussion.
Their word-type random mappings distinguish task performance from a probe's
memorization capacity. Our proposed nonlinear classifier is an accessibility
diagnostic, not proof of the model's computation; preserve held-out groups and a
sanity control. Their selectivity measure should not be imported unchanged into
our different task.

## Variant 2 and stopping criteria

This is meaningful for Variant 2 as a reliability study of a candidate latent
monitor under controlled support, sequence and wording shifts. The bridge is
measurement validity: whether a frozen readout survives a known shift, and
which kinds of additional supervision or scaffolding repair it. It is narrower
than strategic-deception detection or predicting actual recipient outcomes.

The most useful possible result is a discriminating pattern, including a
negative one: strong one-step and easy two-message controls with a remaining
hard-sequence deficit; failure across both two-message strata; format-specific
failure already on one-step cases; or success only after correction supervision
or a supplied fold. Each materially sharpens the completed study's interpretation.
No pattern establishes the sender model's spontaneous anticipation, the actual
reader's private beliefs or deceptive intent.

After the declared diagnostic and fresh cohort, write the result and limitations.
Do not expand the panel, keep changing wrappers, add pressure or demand a
positive effect as a condition of finishing. A later natural-communication
extension would need independently assigned reader evidence and measured
behavior; the present review supplies no substitute for those labels. Preserve
Apollo as the separately documented reproduction anchor, and explain this
material scope change in the supervisor-facing account.
