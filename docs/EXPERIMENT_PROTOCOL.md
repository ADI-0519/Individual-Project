# Initial experiment protocol

Status: **development protocol**, 18 November 2026. The supplied scenarios are
draft synthetic materials. Software tests establish properties of the pipeline;
they do not constitute evidence about a real language model's behaviour.

## Research questions

1. Under similar task pressure, do activation-based monitoring scores distinguish
   different communications, or mainly reflect the instructions?
2. Is there a useful sender representation of the recipient's expected
   conclusion, and can it be distinguished from both the attributed audience
   information and the recipient's actual conclusion?

This implements the assigned Variant 2 topic: latent probes under perturbation
and distribution shifts. The first aim is a controlled, reproducible evaluation.
Belief-readout transfer and causal interventions require additional validation.

## Variables

| Variable | Operational definition | What it does not establish |
| --- | --- | --- |
| Objective | Neutral support or pressure for a favourable answer | That the sender actually misleads |
| Attributed reader information | What the sender is told the reader has received | That the sender internally represents the intended belief |
| Actual reader information | Evidence placed in the recipient's fresh context | Information available to the sender unless explicitly supplied |
| Critical-fact coverage | Whether a generated message communicates the relevant fact | Malicious intent or successful deception |
| Literal factuality | Whether the message's assertions are supported by the source | Whether its implications are harmless |
| Recipient conclusion | Parsed YES, NO, UNKNOWN, or invalid response | A human belief or an infallible inference |
| Elicited forecast | Sender's answer to a subsequent question about the reader | Its spontaneous internal anticipation during the original message |
| Probe score | A fitted function of recorded activations | A direct measurement of conscious intention |

The core comparison keeps these fields separate. A sender can omit something
the reader already knows; that may be appropriate summarisation. A recipient
can be wrong even before receiving any message. Neither case should be silently
labelled as intentional deception.

## Initial design

Start by inspecting four cases across more than one setting, then review the
24-case development set. The set includes multiple scenario families and both
positive and negative complete-information answers. All related paraphrases,
conditions and repetitions must keep their scenario and family identifiers.

For the full pilot, use two sender objectives, two attributed-information
conditions and three sender-generation seeds:

`24 × 2 × 2 × 3 = 288 sender messages`.

This does not create 288 independent scenarios. Inference must account for the
underlying scenario/family clustering. The supplied set is too small for precise
low-false-positive-rate claims. Its purpose is to establish validity and debug
the procedure before collecting separate confirmatory cases.

The sender receives the complete source, rule, question, objective and its
description of the reader's information. It is asked for a short factual
communication. It is allowed to produce a complete report even under favourable
pressure. Do not silently drop unfavourable outputs, noncompliant messages or
literal falsehoods; record their rates and decide their analysis treatment
before the final study.

The recipient receives only the public rule/question, its own information and
the message. It does not receive the sender's objective, complete private
report, attributed-reader condition, annotations or future outcomes.

## What is observed and when

1. Construct the sender input without the recipient's hidden actual-information
   condition.
2. Generate the message normally and save its exact input/output token IDs.
3. Extract the chosen passive activations from that original trace, before any
   forecast question or recipient response is added.
4. Replay the saved message in independent recipient contexts with each actual
   information condition.
5. Run no-message and complete-disclosure controls for the recipient.
6. Optionally elicit the sender's forecast in a separate continuation. Keep its
   inputs, outputs and measurements separate from the passive record.
7. Audit message factuality and disclosure before training a behavioural target.

The same sender checkpoint can initially also be used as the recipient in fresh
contexts. Record this choice. Later recipient replication can hold the sender
messages fixed while changing the recipient checkpoint.

Changing hidden recipient information after generation cannot change the
already-saved sender activations. This is a deliberate information-boundary
control; failure to predict the hidden change is not a detector failure.

Re-extracting activations from the exact tokens a model naturally generated is
trace replay. In contrast, inserting a hand-written or otherwise forced message
is a controlled off-policy diagnostic. Keep such diagnostics separate from
evidence about what the sender naturally chose to say.

## Recipient controls and outcomes

Keep the recipient answer space **YES / NO / UNKNOWN**. UNKNOWN is a distinct
outcome, not an incorrect binary answer. The source establishes the full-world
answer, while the recipient's available evidence determines whether a definite
answer is justified. Preserve invalid answers for inspection and report their
denominator separately.

The starter recipient prompt explicitly instructs the model to avoid assuming
that unmentioned conditions hold. This is a **careful-reader condition**. Report
it as part of the task definition. If selective messages have no effect here,
the result does not establish that ordinary conversational recipients are
equally robust. Any later change to recipient instructions is a new experimental
condition, with its own no-message and full-disclosure checks, not a silent
prompt adjustment until misleading outcomes appear.

For each actual-information condition compare:

- No message: recipient behaviour before communication.
- Full disclosure: ability to use all relevant facts.
- Selective control: a supplied truthful partial report.
- Natural sender message: the model's own communication.

A message-related increase in false conclusions is evidence about the effect
of that communication in this setting. It is not sufficient evidence of sender
intent. Report uncertainty, no-message error and complete-information error
alongside any false-conclusion rate.

## Annotation

The generated-message annotations require human research judgement about the
synthetic text. Distinguish supported assertions, contradicted assertions,
unsupported assertions and ambiguity. For fact coverage, distinguish explicit
coverage, equivalent paraphrase, absence and ambiguity. Do not infer coverage
from a string match alone or treat a missing phrase as proof of omission.

Record the annotation version and any changes. Prefer reviewing messages
without displaying the experimental objective or probe score to the annotator.
Document whether a second researcher checked a subset and how disagreements
were resolved. Using researchers to audit these synthetic materials is distinct
from recruiting human recipients as study participants; discuss the precise
ethics scope with the supervisor.

## Probe development

The supplied probe utility is a general regularised linear-classification
baseline. It does not automatically reproduce Apollo's datasets or numerical
results. The first probe-training data should be independent of the recipient
pilot. Reproduce or adapt a published training recipe with explicit provenance
and labels, then evaluate the resulting probe on the communication scenarios.

Required controls include:

- Group-separated training, validation and test examples.
- Preprocessing fitted on training data only.
- Hyperparameters/layers chosen using validation data; final test untouched.
- A documented threshold rule; no claim of a 1% false-positive rate from a tiny
  negative sample.
- Instruction-only and output/context text comparisons given the appropriate
  information for their prediction target.
- Shuffled-label checks during development and inspection of major errors.

For larger datasets, estimate uncertainty by resampling scenario/family groups,
not individual repeated messages. Compare effect sizes and error patterns, not
only an aggregate AUROC.

### Belief-readout extension

Use a separate controlled belief task with known perspective-specific labels.
First show that the readout generalises to unseen examples of that task. Then
test transfer to the communication setting with the extraction point and
proposition format defined explicitly.

Decoding whether the reader has seen a fact is not the same as decoding what
the sender expects that reader to conclude after this particular message.
Likewise, a readable direction need not be a mechanism the model uses. Causal
intervention experiments belong after transfer validation and require matched
controls for factual knowledge, fluency, instruction following and intervention
magnitude.

## Decision gates

| Observation in the pilot | Next action |
| --- | --- |
| Recipient cannot use full disclosure | Repair comprehension/task format before interpreting selective messages |
| Pressure perfectly predicts disclosure | Broaden the development conditions; the natural data cannot yet separate pressure from behaviour |
| Recipient behaves the same with no message | Avoid attributing its error to the message; examine the contrast |
| Selective messages lead to UNKNOWN | Preserve that result; do not force binary answers to manufacture success |
| Nearly all generated messages are false or malformed | Repair the task and report the original failure rate |
| Neutral belief task works but transfer fails | Report the transfer limitation; inspect format and extraction assumptions |
| Readout transfers and exceeds appropriate baselines | Consider a second checkpoint and a carefully controlled intervention |

All development changes should be logged before evaluating on new held-out
cases. A null or negative finding can still be useful if the experiment rules
out a credible interpretation and the limitations are explained.

## Research records

Retain prompts, raw generations, token IDs, model revisions, software versions,
precision, seeds, extraction metadata, scenario hashes, annotation versions and
split assignments. Keep an explicit record of adopted external code and AI
assistance according to the module's rules. Start a limitations section as soon
as the first result is available.

This initial protocol uses synthetic scenarios and model recipients. Confirm
the module's ethics declaration with the supervisor. Human-recipient experiments
or private personal data would change the scope and need separate assessment.

## Primary methodological starting points

- Goldowsky-Dill et al. (2025), [Detecting Strategic Deception with Linear
  Probes](https://proceedings.mlr.press/v267/goldowsky-dill25a.html).
- [Apollo's public implementation](https://github.com/ApolloResearch/deception-detection).
- Zhu et al. (2024), [Language Models Represent Beliefs of Self and
  Others](https://proceedings.mlr.press/v235/zhu24o.html).

These motivate parts of the method. This document does not assert that their
results establish the proposed communication hypothesis or its novelty.
