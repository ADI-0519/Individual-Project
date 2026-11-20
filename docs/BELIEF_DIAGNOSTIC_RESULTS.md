# What the reader-forecast failure says about the belief project

20 November 2026. **The project remains an open research question. A reliable neutral nested-belief decoder has not yet been established.** Failure of one reader-outcome forecast does not establish that a language model lacks reader-perspective information. It also does not validate the intended stronger claim by itself.

The [curated machine record](./validation/2026-11-20-belief-diagnostic.json) binds the runs, exact metrics, primary numeric exports and all comparison conditions by hash.

## Correct the target and timing interpretation

The earlier readout learned observed APPROVE/REJECT/DEFER outcomes. It was not trained on a direct second-order label such as “what S thinks R believes.” Its block-8, before-report failure therefore concerns that outcome forecast, feature position, fitting procedure and pressure shift. The [primary-methods audit](./BELIEF_TARGET_AUDIT.md) distinguishes perspective decoding, explicit nested questions, query-time retrieval and future-output prediction in the closest papers.

An after-report activation still precedes the reader's answer. It can legitimately forecast that later answer. What the earlier pilot fails to establish is anticipation **before composing the report**, or added recipient-perspective information beyond the report text. The literal report-word control already gets 11/12 under pressure. A known-message forecast and an unknown-future-message forecast are different tasks.

## Saved-array diagnosis of the original collapse

The original C=1 before-report model fits all 47 valid training outcomes, with log loss .0059, but gets only 8/12 neutral validation outcomes right. There are 4,096 features and only 47 training vectors, with centered rank 46. The validation set contains no REJECT examples. These are substantial fitting/selection limitations.

Under pressure, the median norm of the standardized feature vector rises from 63.47 to 114.58, while the mean REJECT-versus-APPROVE logit margin shifts by +15.87. A separately declared, **post-hoc** 21-candidate grid chooses scaling and C using neutral validation loss only. The selected row-normalized C=10 model scores 9/12 on validation, 7/12 on neutral test and 8/12 on pressure test. That pressure accuracy merely matches the APPROVE-majority baseline; REJECT and DEFER test recall remain zero.

This sensitivity narrows the interpretation of 1/12. It is not evidence for a working forecast, a recovered belief mechanism or global information absence. All original weights and outcomes remain frozen. The [exact CPU record](./validation/2026-11-20-prereport-shift-diagnostic.json) contains the candidate selection, class counts, feature shifts and hashes.

## A fresh, explicitly nested target

The [new authored panel](../data/belief_diagnostic/development_v1.json) has 64 prior-state stories: eight complete combinations per wording group, across eight groups. World location, actual reader attendance and the attendance record trusted by the sender vary independently. From those histories, the renderer computes three distinct labels:

- World truth: where the token actually is.
- Reader perspective: where R believes it is, given R's observation history.
- Nested perspective: where S thinks R believes it is, given S's information.

For example, the token moves from amber to teal; R misses the move; S's trusted attendance display incorrectly says R watched. R retains amber, while S attributes teal to R. Source truth, actual reader perspective and the nested label cannot be substituted for one another.

An omniscient view contains both histories. A sender view excludes actual private attendance. Its matched attendance pairs have **identical messages, token IDs and features**, with opposite actual-reader labels. Any deterministic actual-reader classifier must therefore score exactly 50% on this balanced negative control. The nested label remains available from the sender's information.

There are also 16 trusted-current-message controls. The reader receives the correct location and the sender sees confirmation; all three labels then become world truth. These controls are excluded from fitting and selection. The labels are stipulated fictional states, not direct observations of an LLM's hidden beliefs. Human semantic review is still required.

## What ran and what it found

The same pinned Llama-3.1-8B-Instruct ran locally in NF4/BF16. Four wording groups train, two validate and two test. Each group contains all eight logical cells. This holds out wording and entities within one logical task, not new story families.

Features come from the last story-plus-candidate prompt token **before any perspective question or answer**. Block 16 was declared primary before this run; blocks 8 and 24 were diagnostics. Each head uses train-only standardization and per-row normalization; C is selected only by neutral validation loss. Text controls use training-only TF-IDF and the same validation rule.

| Query-free block-16 head | Omniscient test | Sender-view test |
| --- | ---: | ---: |
| World truth | 12/16 | 14/16 |
| Actual reader perspective | 12/16 | 8/16, structural negative control |
| Nested sender perspective | **9/16** | **8/16** |
| Nested TF-IDF control | 8/16 | 8/16 |

Both nested primary losses are approximately log(2): .69310 and .69374. Nine correct rows are not a meaningful nested-decoding success here. Each test has only **two wording groups**; the repeated factorial cells are not independent templates. The diagnostic layers also fail to establish a robust nested readout. The world-copy control scores exactly 8/16 on both perspective targets because their labels are balanced independently of world truth.

Separate explicit YES/NO questions yield:

| Prior-state question | Omniscient | Sender view |
| --- | ---: | ---: |
| World | 48/64 | 46/64 |
| Reader perspective | 32/64 | Not asked: actual private information unavailable |
| Nested perspective | 32/64, three invalids | 28/64, eight invalids |

Every prior reader answer and every prior nested answer begins NO. All 17 invalid rows across prior and update conditions begin with NO followed by an explanation and reach the six-token cap. The strict primary parser retains them as invalid. Accepting the leading answer would bring sender prior nested accuracy only to 32/64; it would not improve omniscient prior nested accuracy. By contrast, the delivered-current reader control is 16/16, showing that YES can be generated in another condition.

The run saves 160 feature rows from 120 unique query-free prefixes, plus 400 behavior rows from 320 unique generation calls. Identical sender prompts reuse identical features and deterministic answers; duplicates do not increase independent sample size.

## Equivalent location-choice check

A single separately frozen follow-up retains every scene, candidate statement, model and decoding budget, and replaces the YES/NO question with an A/B choice between the two locations. Option order alternates by wording group. This is a post-hoc question-format diagnostic on already seen development cases, not a new held-out confirmation. Its comparison concerns question wording and response format jointly; it does not isolate a single token-level cause.

It completes another 400 rows from 320 unique calls. There are **217 strict-invalid outputs**. Every one already begins with A: 184 echo the selected location and 33 add an explanation. Of these, 135 reach the token limit and 82 end normally. None lacks a leading choice. Strict results remain frozen; a separately reported **post-hoc leading-letter extraction** distinguishes choice content from compliance with the one-letter instruction:

| Prior question | Strict correct | Strict invalid | Supplemental leading-choice correct |
| --- | ---: | ---: | ---: |
| Omniscient world | 47/64 | 17 | 64/64 |
| Omniscient reader | 11/64 | 51 | 40/64 |
| Omniscient nested | 5/64 | 56 | 32/64 |
| Sender world | 52/64 | 8 | 60/64 |
| Sender nested | 10/64 | 42 | 32/64 |

Both prior nested conditions choose **A on every row**. The alternative format therefore does not rescue nested reasoning, even after the supplemental extraction. It reveals format-sensitive world judgments and a first-option response pattern. The delivered-current supplemental reader and nested results also fall to 9/16 and 8/16 in the omniscient view. Do not describe the format change as a general capability improvement or the strict failure as evidence that no semantic answer was present.

All 400 new mappings and records, 320 cache groups, 403 run files and 528 original source files were independently checked. The saved plan and source guards support ordering before new calls; there is no independent external certification of generation-start times.

## Are the ingredients readable even when the direct nested head fails?

In these authored prior scenes, the nested bit is world truth when the sender's attendance record says the reader watched, and the opposite when it says the reader missed the move. This permits a transparent compositional control: separately decode **world truth** and **recorded attendance**, then apply that logical rule outside the model.

The [CPU composition diagnostic](./validation/2026-11-20-belief-composition.json) reuses the already selected block-16 world head and fits only the new attendance head, with the same training-only transform and validation-only C grid. It does not choose a layer or tune the composition on test performance. This is an exploratory analysis of already observed cases.

| View / split | World head | Attendance head | Direct nested head | External logical combination |
| --- | ---: | ---: | ---: | ---: |
| Omniscient validation | 12/16 | 10/16 | 7/16 | 10/16 |
| Omniscient test | 12/16 | 12/16 | 9/16 | 12/16 |
| Sender validation | 12/16 | 10/16 | 8/16 | **6/16** |
| Sender test | 14/16 | 16/16 | 8/16 | **14/16** |

The sender test shows that relevant ingredients can be recovered despite the direct head's 8/16. **The external rule is ours, not evidence that the model spontaneously uses a nested-belief representation.** The sender validation deterioration prevents a robust-composition claim. Two omniscient test successes also come from cancelling errors in both component heads; a correct composed label need not mean correct factual ingredients.

No composed probabilities or calibration are invented from the hard rule. Delivered-current controls are outside this prior-state rule and are excluded from the composition comparison. Existing source weights, labels and all original files remain unchanged.

The second persistence version adds all 128 prior-row predictions and a numeric parameter sidecar so another implementation can reproduce the component and composed decisions. Its aggregates, selected settings and fingerprints exactly match the preserved first version.

An independent implementation scores all 128 rows from those arrays without refitting. Every factual, direct and composed prediction, all 30 aggregate classification cells and all error-cancellation counts match. This verifies the computation; it does not strengthen the limited generalization claim.

## Verification and research decision

All three decoder hooks exactly match HF hidden-state indices 9, 17 and 25 in a genuine reference forward. Four prespecified view/phase repeated captures match exactly. An independent agent verifies 527 output hashes, all 160 capture bindings, all 40 hidden-attendance pairs, 400 behavioral records, 168 classification-metric cells and all 24 validation selections. Portable numeric exports of both primary nested heads reproduce all eight split-level probability losses and confusion results; independent test probabilities agree within 1.11e-16.

The neutral nested-belief gate has **not passed**. Positive world/first-order controls and the hidden-information negative control help localize the problem; they do not establish spontaneous nested reasoning or a causal mechanism. The immediate decision is to diagnose elicitation/readout timing and, if needed, use a targeted model/precision reference before scaling belief transfer. The natural-report pilot remains an operational baseline. No original report, label, failed output or frozen fit is replaced.

The bounded comparison of story-only states with states after a belief question
has now run, with the same normative targets, a world-question control and a
separately frozen fresh-wording evaluation. The [measured query comparison](./QUERY_COMPARISON_RESULTS.md)
finds sender nested decodability of 16/32 before versus 24/32 after the question,
but frozen transfer back to story-only states remains 16/32. This conditional
result does not validate spontaneous use or an unqueried communication decoder.
Factored factual readouts remain controls. If elicited perspective behavior
still collapses, check an established ToMi task and a targeted model/precision
reference before scaling this panel. A successful neutral gate must precede a
new natural-communication transfer claim; the existing four-case results are
not its final test set.

Before testing a specifically post-message decoder, add a separate balanced delivery/receipt block: the reader initially misses the move; a truthful current-location message is actually delivered or not, independently of the delivery receipt trusted by the sender. This separates post-message world, reader and nested perspectives. The present 16 fully delivered controls collapse all three labels and cannot validate that distinction. Transfer to natural reports must preserve the chosen prior or post-message target.

Prior nested tracking, post-message belief updating with the exact message known, and predicting the effect of an unknown future natural message remain separate targets. A successful prior decoder would support only its matching transfer claim. The [study gates](./MAIN_STUDY_GATES.md) and [target audit](./BELIEF_TARGET_AUDIT.md) retain that boundary.

## Reproduction

Use the established cached WSL environment; no reinstall or authentication change is needed. Choose new output paths:

```bash
HF_HOME=.cache/huggingface-wsl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TRITON_CACHE_DIR=.cache/triton-wsl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv-wsl/bin/python scripts/run_belief_diagnostic.py \
  --config configs/apollo_llama8b_4bit.json \
  --panel data/belief_diagnostic/development_v1.json \
  --output runs/belief_diagnostic_new

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv-wsl/bin/python scripts/export_belief_readout.py \
  --run runs/belief_diagnostic_development_v1 \
  --output runs/belief_diagnostic_heads_new

HF_HOME=.cache/huggingface-wsl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TRITON_CACHE_DIR=.cache/triton-wsl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv-wsl/bin/python scripts/run_belief_choice_check.py \
  --config configs/apollo_llama8b_4bit.json \
  --source-run runs/belief_diagnostic_development_v1 \
  --output runs/belief_choice_new

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv-wsl/bin/python scripts/diagnose_belief_composition.py \
  --run runs/belief_diagnostic_development_v1 \
  --output runs/belief_composition_new.json
```

Executed diagnostic sources are preserved at `3185e3c`; exporter and choice-check sources at `b61a2b1`. The comparison keeps the original binary outcomes, normative gold and trained query-free weights frozen.
