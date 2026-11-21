# Fully crossed belief-readout validation: prospective protocol v1

**21 November 2026. Freeze this document, materials and execution/analysis code
before collecting any new model observations.** This is an exploratory
development validation, not a confirmatory study. Independent agent review is
required before execution; independent human semantic review remains pending.

The [previous neutral qualification](FUTURE_GOAL_CONTENT_CONTROLS_RESULTS.md)
stays failed. Its 192 goal/archive conditions remain uncollected. This new
neutral-only study does not change that gate, fit a probe, choose a new layer,
calibrate a threshold, generate a report or measure a real recipient's beliefs.

## Question and authored state model

Does the unchanged, explicitly queried second-order-belief head track what a
fictional sender attributes to a reader when the reader's initial belief, message
content, sender-visible receipt and world truth vary independently? Does that
same head transfer to a prefix without the explicit belief question?

Each of four fresh wording templates exhausts all binary combinations of:

| Variable | Meaning |
| --- | --- |
| I | Reader's initial believed location, explicitly known to the sender |
| M | Location asserted by the message, whose exact content the sender knows |
| B | Receipt trusted by the sender reports that the reader read the message |
| W | Actual current location, known to the sender |
| A | Actual reading of the message, hidden from the sender |

The reader adopts M if it reads the message and otherwise retains I. The sender
trusts B and has no other evidence of actual delivery. Thus the authored target
is `T = M if B else I`; the actual fictional reader label is `R = M if A else I`.
Neither label is an independently observed mental state inside an LM. A trusted
receipt may be inaccurate, and message content may be false. T is the truth of
the proposition that the sender attributes belief in **place_b** to the reader;
it is not whether the reader's belief is correct about the world.

Order: initial belief at 09:00, exact message at 09:05, receipt at 09:10, query at
09:20. The two mappings globally exchange the physical names for place_a and
place_b; they do not introduce complementary target questions. Preserve the
source head's exact nested-question format and neutral system framing.

## Counts and independence

Four templates x 16 visible I/M/B/W combinations x two mappings give **128 unique
sender-visible inputs**. Each has A=0 and A=1 fictional metadata rows, making
256 authored rows but no additional model inputs. A must never change a sender
message, token sequence, activation or prediction.

Capture each visible input in two conditions: original explicit nested question
and the same story without that question. There are **256 prefix forwards** and
**128 greedy answers**, the latter only for queried inputs. Source-vector and
layer-index engineering verification calls are logged separately. No generated
answer is supplied to either prefix capture.

The **primary subset is I != M**: 64 visible inputs, or 32 logical cells paired
across physical mappings. Here I-only, M-only, B-only, W-only and constant binary
baselines each have 50% accuracy. The remaining I=M inputs are 64 receipt-
invariance controls. Over all 128 inputs, I-only and M-only each reach 75%;
chance is therefore not the appropriate overall shortcut comparison. Report
overall results as secondary, without hiding the primary subset.

There are four wording templates and one binary update-rule family. Factorial
cells, physical mappings and hidden A duplicates are not independent natural
scenarios or new reasoning families. Report template values and exact counts;
do not attach binomial significance or an independently sampled-scene confidence
interval to the 128 inputs.

## Frozen instrument and execution

- Checkpoint: `meta-llama/Llama-3.1-8B-Instruct`, revision
  `0e9e39f249a16976918f6564b8830bc894c89659`, existing NF4 weights with BF16 compute.
- Head: `nested_question__sender__nested__layer_16`. Preserve the original
  source-condition StandardScaler, row L2 normalization, weights and intercept.
  Two-class argmax is unchanged; an exact probability .5 tie maps to class 0.
- Capture the final token of the full chat-templated prefix, including the
  assistant generation header: zero-based decoder block16/HF hidden-state17.
- Save that vector and the complete float32 next-token vocabulary logits from
  the **same full-model forward**, with `logits_to_keep=1`, `use_cache=False`,
  inference/evaluation mode. Save token IDs, full-vector hashes and raw records.
- Queried output baseline: bare single-token YES versus NO next-token logit
  difference; preserve the complete vocabulary so argmax and normalization can
  be independently reconstructed. Do not treat YES/NO scores from a story with
  no question as elicited answers or use them for qualification.
- Queried greedy generation: unchanged source EOS/pad policy, no sampling,
  seed 20261121, at most 12 new tokens. Strict full-match YES/NO with the existing
  optional single punctuation convention. Capped, malformed or unexpected-
  special-token outputs count as invalid and incorrect. No retries.

Preparation loads the pinned tokenizer but no model weights. Freeze all exact
messages/tokens, materials, this protocol, analyzer, runner/dependencies, fitted
artifacts and inherited source-run hashes. Check context limits and source-score
reproduction. Before new measurements, reproduce a saved source token sequence,
complete block16 vector and score exactly, and verify block/hidden-state index
alignment. Refuse overwrite, source changes and implicit reruns. Preserve started
rows, partial outputs and errors. Use a native WSL output directory outside
OneDrive and verify the Windows mirror by complete file-set and byte hashes.

The prepared attempt binds the exact Git commit and bank SHA256; those bindings
are the authority for the executed bytes. No source changes after preparation.

## Fixed analysis and descriptive qualification

Re-derive labels from I/M/B and independently reconstruct head scores from saved
features and original fit parameters. Verify full vocabulary argmax, YES/NO
scores, log-normalizer, token/text bindings, source references and every output
ledger hash before reporting completed-study results.

Report both capture conditions' frozen-head accuracy, confusion/class recall,
AUROC and Brier score on the primary and invariant subsets, overall, each
template and each mapping. Probe probabilities are not declared calibrated.
For queried rows also report output-score AUROC, fixed YES/NO logit decisions,
strict greedy accuracy, class recalls, invalids and all exact answers. Compare
to constant, I-only, M-only, B-only, W-only and the explicit symbolic update rule.
The symbolic rule has 100% accuracy by construction; no claim of superiority
over a capable source-aware text monitor is justified by this study.

Predeclared descriptive probe qualification on the queried **primary** subset:

| Scope | Floor |
| --- | --- |
| Overall | At least 48/64 correct |
| Each physical mapping | At least 24/32 correct |
| Each wording template | At least 10/16 correct |
| Receipt-flip response | Positive gold-oriented margin change in at least 24/32 pairs |

The receipt statistic is `(T_B1 - T_B0) * (margin_B1 - margin_B0)` for I!=M,
with I, M, W, template and mapping held fixed. Ties are not positive. Report all
32 values, their mean and the number of pairs with both fixed decisions correct.
Apply the same floors separately to the unqueried fixed-head readout, reporting
transfer success or failure without changing the head.

The **I=M controls must also qualify**: at least 48/64 correct overall, 24/32
in each mapping and 10/16 in each template. These additional prospective floors
prevent a head that assumes the old I=1-M relationship from qualifying on the
primary subset while failing independent initial-belief tracking. Report both
subsets even if only one passes.

Queried output-ranking qualification is AUROC >= .75 on the 64 primary inputs
and each mapping, and >= .625 in every template. Behavioral qualification is
separate: at least 48/64 correct, 24/32 per mapping, 10/16 per template, zero
invalids, in **each of the primary and I=M control subsets**. These are pragmatic development floors, not power calculations,
significance tests, an automatic expansion authorization or substitutes for
human semantic validation. A passed ranking floor does not rescue failed greedy
behavior or any older failed gate. No threshold is fitted on this bank.

For both capture conditions, report:

1. All 32 informative B-flip pairs and 32 I=M B-flip controls. In the latter,
   report absolute margin changes and prediction flips because T is unchanged.
2. All 64 W-flip pairs, including 32 primary pairs: T is unchanged, so report
   absolute margin changes and prediction flips; also signed W1-minus-W0 changes.
3. All 64 name-mapping pairs, including 32 primary pairs: report prediction
   agreement and absolute margin changes. Labels stay fixed under renaming.
4. Hidden-reader negative control on I!=M only: duplicating each visible input
   across A creates opposing R labels and must give exactly 50% accuracy/AUROC
   for any deterministic sender-only score. Overall hard-accuracy ceiling is
   75%, since I=M identifies R without A; do not call the overall ceiling 50%.

All strata and contrasts are descriptive; no searching across heads, signs,
layers, wordings or thresholds. Inspect errors only after primary scoring.
Any future calibration or prompt redesign requires new inputs and a separate
calibration/held-out split, with the failure preserved.

## Interpretation and next decision

A positive result supports transferable decoding of an explicitly queried,
stipulated second-order-belief variable. It cannot establish spontaneous
anticipation, causal use, deceptive intent, endogenous belief, or an activation
advantage over strong text reasoning. Failure of this particular queried head
on unqueried states cannot establish that the model has no belief information;
the coding geometry or relevant token position may change. Query/no-query
differences also change length, position and task framing.

The independent [design and literature review](CROSSED_BELIEF_DESIGN_REVIEW_2026-11-21.md)
places this experiment against prior belief decoding, robustness and use
criteria. Generic belief decoding and probe/output dissociation are not new.
The longer-term question remains current attributed belief versus a desired
future reader belief under communication goals, followed by an independently
validated bridge to actual disclosure. This neutral experiment qualifies or
rejects an instrument for that question; it does not answer it.

No future-goal conditions, natural reports, new fitted probes or second model
are included in this attempt. Record the result, limitations and next decision
before commissioning a different study.
