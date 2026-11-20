# Before and after the belief question: measured comparison

**20 November 2026.** The controlled comparison and a frozen fresh-wording
evaluation are complete on the real cached Llama 3.1 8B Instruct model.
Introducing the nested-belief question improves the primary sender-view
readout from **16/32 to 24/32** on the fresh cases; a world-question control
gets **16/32**. However, the question-conditioned head gets **16/32** when
transferred unchanged to story-only states. This is evidence of conditional
decodability in this small panel, not a validated unqueried belief decoder.

The original neutral-belief-to-pressure research direction is preserved.
This experiment diagnoses its measurement gate; it does not complete the
main study. See the [previous diagnostic](./BELIEF_DIAGNOSTIC_RESULTS.md),
[target/methods audit](./BELIEF_TARGET_AUDIT.md),
[study gates](./MAIN_STUDY_GATES.md), and
[machine-readable measured record](./validation/2026-11-20-belief-query-comparison.json).

## What was fixed before execution

Source and fresh materials were committed at
`98ca1eb1182abf897d09aeb5c7113ca7b1bae3b2` before this run. Each story is
captured in three conditions, at the last chat-prompt token before any answer:

1. **Story:** the original scene without a perspective question.
2. **Nested question:** the same scene followed by the original question asking
   whether the sender thinks the reader believes the candidate location claim.
3. **World question:** the same scene followed by the original world-truth
   question. This tests whether any question produces the same readout gain.

The final token ID is 271 in all conditions. Its position and preceding text
change. The world question is not length-matched to the nested question, so this
does not isolate question semantics from length, actor mentions, position or
general task conditioning. No answer tokens, generated explanations or reader
decisions enter these captures. **No new answers were generated.**

The targets stay separate: world location, actual reader-attributed location,
and the sender's attribution of the reader's location belief. Gold labels are
stipulated by the fictional scene. Actual reader attendance and the attendance
record trusted by the sender vary independently. In the sender view, private
actual attendance is hidden; in the omniscient view it is visible. For the
sender view, pairs with different actual attendance have identical model inputs.
The actual-reader target must therefore remain exactly 50% in those pairs.

The legacy panel retains 64 prior-state cases across eight wording groups:
four train, two select regularization, and two are already-known test groups.
Its 16 delivered-current controls are excluded from fitting and selection.
The new panel contains **32 prior-state cases across four new wording/entity
groups**, with every eight-cell logical combination represented in each group.
Names, places, case IDs and complete prompts are disjoint from legacy material.
The system instruction and question stems stay fixed. These are new wordings
within the same logical task, not new reasoning families or human-validated data.

The primary target is nested belief and the primary decoder block is **16**
(zero-based; HF hidden-state index 17). Blocks 8 and 24 are diagnostic, not
alternatives selected after seeing fresh performance. The model revision is
`0e9e39f249a16976918f6564b8830bc894c89659`, using NF4 double quantization and
BF16 compute in the existing local WSL GPU environment.

## Fitting, freezing and fresh scoring

Each condition/view/target/layer has its own linear logistic readout. Scaling
uses only legacy training rows, followed by per-row L2 normalization.
Regularization is selected from `0.001, 0.01, 0.1, 1, 10` using legacy validation
negative log loss, with the smaller value breaking exact ties. Matched text
controls use train-only unigram/bigram TF-IDF. Controls, known test rows and
fresh rows never select a setting. There are 54 activation and 18 text heads.

All fitted coefficients, intercepts, means/scales, vocabularies, IDF values and
selected settings were saved and hashed at **01:45:14.293975 UTC**. Fresh capture
started at **01:45:14.302577 UTC**, after the freeze. Fresh evaluation loads these
arrays and does not fit or select anything; a test explicitly disables `.fit()`.
All fit hashes remain unchanged after scoring.

Two comparisons answer different questions:

- **Equal-condition refits** test whether the target is more decodable when the
  question has been introduced. Each head trains on its own legacy condition.
- **Frozen cross-condition scoring** tests whether a trained readout survives a
  changed condition. It retains the source condition's complete transformation
  and weights. The four declared directions are story to nested, story to world,
  nested to story, and world to story. This is not a full three-by-three matrix.

## Primary results

Counts refer to block-16 nested-target predictions. Every fresh condition/view
has balanced labels, 16 negative and 16 positive. Each sender-view condition
has **16 unique prompts represented by 32 rows**; the omniscient view has 32
unique prompts. The wording group, not each repeated row, is the main unit of
generalization. There are only four fresh groups.

| Capture condition | Legacy omniscient test | Legacy sender test | Fresh omniscient | Fresh sender |
|---|---:|---:|---:|---:|
| Story | 9/16 | 8/16 | 16/32 | 16/32 |
| Nested question | 9/16 | 12/16 | **22/32** | **24/32** |
| World question | 9/16 | 8/16 | 17/32 | 16/32 |

| Fresh condition | Omniscient negative log loss | Sender negative log loss |
|---|---:|---:|
| Story | 0.693148 | 0.694196 |
| Nested question | **0.680129** | **0.676828** |
| World question | 0.692546 | 0.693147 |

The proper-loss improvement is modest; the uninformative balanced loss is
approximately 0.693147. These measurements do not establish calibrated
confidence. Every matched nested-target TF-IDF control gets **16/32** on fresh
cases. This is one text baseline, not evidence of superiority to capable text
reasoning or a full-context LLM monitor.

All four frozen cross-condition directions get **16/32** for the primary nested
head in both views. In particular, the nested-question head's 24/32 sender
result becomes 16/32 when applied to unqueried states. The successful condition
cannot be treated as a decoder already suitable for unqueried communication.

### Errors and heterogeneity

| Fresh wording group | Story omniscient | Nested-question omniscient | Story sender | Nested-question sender |
|---|---:|---:|---:|---:|
| bq_t01 | 4/8 | 5/8 | 4/8 | 6/8 |
| bq_t02 | 4/8 | 6/8 | 4/8 | 4/8 |
| bq_t03 | 4/8 | 6/8 | 4/8 | 6/8 |
| bq_t04 | 4/8 | 5/8 | 4/8 | 8/8 |

The sender nested-question head predicts positive on 24/32 rows. Positive
recall is 100%, but negative recall is only 50%; its confusion matrix, true
rows and predicted columns in `[0, 1]` order, is `[[8, 8], [0, 16]]`.
Macro F1 is 0.7333. The story head predicts positive throughout. The omniscient
nested-question confusion is `[[10, 6], [4, 12]]`, macro F1 0.6863.

Where the nested label differs from world truth, the nested-question heads
get 14/16 sender and 15/16 omniscient; where the labels agree, they get 10/16
and 7/16 respectively. Simple copying of world truth cannot explain the result,
but this imbalance is not a causal mechanism test or a robust false-belief claim.
The sender legacy validation accuracy is only **6/16** despite 12/16 known-test
accuracy. Validation selected loss, not accuracy; this is not a selection error,
but it reinforces the generalization limitation.

The hidden actual-reader heads remain exactly 50% in the sender view, as
required by identical inputs with opposing labels. Delivered-current controls
reach 16/16 for the nested-question nested heads in both views, but all three
targets coincide in those controls. They cannot validate perspective separation
or general post-message belief updating.

## Independent verification

The run records 672 aligned feature rows across three blocks, from 504 unique
prefixes: 480 legacy rows/360 prefixes and 192 fresh rows/144 prefixes. There
are 18 exact repeated-prefix checks. A genuine reference prompt in each bank
checks hooks against HF hidden-state indices 9, 17 and 25. All 160 legacy story
rows reproduce the original vectors exactly at all three blocks. Fresh scenes
have no previous capture; the empty fresh legacy-replay record is not an
additional replication test.

Independent agents checked source/output hashes, prefix bindings, validation
selection, fresh label/probability consistency, per-group errors and the frozen
timeline. A separate implementation reproduced all **18,816 prediction rows**
from the saved parameters and reconstructed TF-IDF, with maximum probability
difference 1.44e-15. It recomputed all 840 metric cells, all 72 validation-only
regularization selections and all 112 hidden-reader metric cells. All 18
train-only activation scalers match exactly. Numeric parameters and text
transformations are stored without pickle;
row-level predictions retain source and evaluation conditions. Raw runs and
weights stay ignored by Git; the curated measured record preserves their hashes.
Software checks pass **361 tests plus 48 subtests**, lint and formatting.

## Research decision and next gate

There is now a positive result to investigate: asking the nested question makes
its stipulated target more decodable on fresh wordings. The claim is narrower
than spontaneous second-order belief. Refitting can exploit a question-induced
representation; it does not show the model uses that representation to answer
correctly, anticipate an unknown report's effect, or communicate deceptively.
The previous explicit-answer failures remain part of the evidence.

Do not transfer this queried head directly into unqueried pressured prompts.
Before that transfer, freeze a neutral validation protocol with human-reviewed
cases, more independent reasoning families and question formulations, balanced
negative cases, and a competitive text reasoning control. If question dependence
is the target of a stronger causal claim, add matched-position/length controls.
Retain this four-group bank as an already-observed validation result, not a new
test set to tune against.

Then transfer only a validated readout with a matched target and capture time.
Prior-state nested attribution, updating after an exact delivered message, and
forecasting an unknown future natural message require separate labels. A
post-message study first needs independently varied delivery and trusted receipt;
the current all-delivered controls are insufficient. Natural communication,
pressure perturbations, Apollo comparisons and independent held-out evaluation
remain the main project after this validity gate.

## Reproduction

Use a new output directory and the existing cached model environment:

```bash
HF_HOME=.cache/huggingface-wsl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
HF_HUB_DISABLE_TELEMETRY=1 TRITON_CACHE_DIR=.cache/triton-wsl \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv-wsl/bin/python scripts/run_belief_query_comparison.py \
  --config configs/apollo_llama8b_4bit.json \
  --source-run runs/belief_diagnostic_development_v1 \
  --fresh-panel data/belief_diagnostic/fresh_query_validation_v1.json \
  --output runs/belief_query_comparison_new
```

The completed local run is `runs/belief_query_comparison_v1`. Its original
manifest, frozen-fit record, per-prefix arrays, fitted numeric sidecars and
fresh predictions are preserved. This result used the existing GPU environment
and three agents; no connected-app plugin or additional model download was needed.
