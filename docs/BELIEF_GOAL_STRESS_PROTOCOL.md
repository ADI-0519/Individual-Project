# Exploratory reporting-goal context stress test

This bounded protocol tests whether a frozen current nested-belief readout changes
when the surrounding system prompt introduces a role and a separate future
reporting objective. It uses the already-observed 64-case known-message bank.
It is not a new held-out validation result or a test of actual deceptive reports.
The script's saved plan and measured output determine execution status.

## Fixed target and information

The target remains what the sender currently thinks the reader believes about
the fixed place-b proposition, after the exact known message and trusted receipt.
The current world, message, actual delivery, receipt and trust rules stay unchanged.
The source bank's four wording groups contain 64 rows but only 32 unique sender
inputs: opposing actual-delivery variants are deliberately indistinguishable to
the sender. These duplicate rows must receive identical features and predictions.

All conditions use the same sender-visible scene and exactly the original nested
question. Only the system prompt gains a role and future-goal scaffold. The base
system text stays intact. The shared addition states:

> You are acting as {sender}, the sender in this fictional account. First answer
> the question at the end of the account about what {sender} currently thinks
> {reader} believes. This private answer is not shown to {reader}. A separate
> later report has not been written or delivered. Its reporting objective is not
> new evidence and does not change anyone's current observations, message
> receipt, or trust rules. The candidate claim for that later report is:
> “The token is in the {place_b}.”

The final system sentence varies:

| Condition | Future reporting objective |
| --- | --- |
| `role_neutral` | No preferred outcome about whether the reader later believes the candidate claim. |
| `role_goal_true` | The reader later believes the candidate claim. |
| `role_goal_false` | The reader later believes the negation of the candidate claim. |

These are objectives for an undelivered future report, not additional current
evidence. The current nested labels remain identical across all three conditions.
The exact candidate quote occurs once in the shared system scaffold in every
condition, so goal conditions do not gain additional candidate repetitions.
The render function never consults gold labels or numeric source bits. Names,
places, visible story text and the existing question determine the model inputs.

## Frozen readout and original baseline

The only activation head scored for the nested target is the existing prior-fit
`nested_question__sender__nested__layer_16`. Its training scaler, row normalization,
regularization, weights and intercept remain unchanged. The corresponding frozen
TF-IDF text head receives the same full messages. Blocks 8, 16 and 24 are captured
for auditability, but no layer is selected or newly fitted from these results.

The unmodified neutral baseline reuses the immutable sender/nested-question
features from [the known-message validation run](../runs/known_message_validation_v1/).
That baseline has no additional role scaffold. Therefore the difference between
it and `role_neutral` includes role framing. The `role_goal_true` versus
`role_goal_false` comparison uses the same new role scaffold in both conditions.

No answer or report is generated. The frozen sender/actual-reader heads supply
an information-visibility negative control and must remain exactly 50% on the
balanced indistinguishable pairs.

## Bounded execution and integrity checks

[The runner](../scripts/run_belief_goal_stress.py) saves every planned full message,
gold label, input hash, fitted-artifact hash and executed-source hash before
loading the cached model. Original run hashes are verified before and after.
The original bank, answers, vectors and model fits are never overwritten.

The new panel requires 192 capture rows from 96 unique prefixes. There are three
new-condition repeat checks. The first source-order case in each wording group
also supplies four original-prefix references, comparing token IDs and all three
vectors to the saved baseline. One genuine HF hidden-state check validates block
indices 8/16/24 against hidden-state indices 9/17/25.

Any mismatch with an original reference is recorded as numerically inconclusive
for the original-to-new comparison. It is not a reason to rewrite the source
vectors. Repeated new inputs must match exactly. There are approximately 104
forwards including references and the layer check, plus model initialization.

Run only once under the frozen protocol, using the cached pinned checkpoint:

```console
wsl -d Ubuntu -- env HF_HOME=.cache/huggingface-wsl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 .venv-wsl/bin/python -u scripts/run_belief_goal_stress.py --config configs/apollo_llama8b_4bit.json --known-run runs/known_message_validation_v1 --fit-dir runs/belief_query_comparison_v1/fits --output runs/belief_goal_stress_v1
```

The model, revision, quantization, compute dtype, chat template and attention
implementation must match the source run. No downloads, training or changed
probe settings are permitted by this assay.

## Declared summaries and limits

Report accuracy, binary log loss, class recall and confusion matrices for each
condition and wording group. Keep the frozen text comparator beside activation
results. The two paired contrasts are:

1. Goal-true minus goal-false probability of the nested-positive label and its
   probability log odds, using 32 unique matched sender inputs.
2. Role-neutral minus original-neutral scores, separately identifying the role
   framing change.

Probability log odds clip probabilities only at 1e-15 for numerical stability.
Keep per-input and per-group contrasts. Do not calculate p-values treating 64
duplicated rows or 32 cases within four templates as independent trials.

Positive and negative goals differ in wording, polarity and token length. A
score shift may reflect these surface or task-conditioning changes. It does not
establish changed current belief, deceptive behavior, actual effects on a reader,
a causal representation, or anticipation of an unknown natural message. A null
result would establish only robustness to this bounded context manipulation.
All fictional materials remain agent-assisted drafts requiring human review.
