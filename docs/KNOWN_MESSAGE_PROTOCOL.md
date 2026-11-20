# Known-message nested-perspective diagnostic

20 November 2026. **Prospective author-assisted protocol; no result is asserted.**
This evaluates frozen prior-state readouts after an exact known message. It is
neither unknown-message anticipation nor the main pressured-communication study.
It follows the [query comparison](./QUERY_COMPARISON_RESULTS.md), whose successful
queried sender readout did not transfer to unqueried states.

## Why this diagnostic

The earlier delivered-current controls give the reader and sender an accurate
message and observed delivery. World, reader and nested gold then coincide.
Those controls cannot establish that a monitor distinguishes these perspectives.
Here the message can disagree with reality, and actual delivery can disagree
with the sender's trusted receipt. The message text is known before capture.

The four wording/entity groups in the [saved panel](../data/belief_diagnostic/known_message_validation_v1.json)
are new relative to both previous banks. They share **one new message-update
logical family**. Four wordings do not constitute four independent reasoning
families, human validation or broad story-family generalization.

## Data and stipulated labels

Every group contains all sixteen combinations of four independent binary values:

| Bit | Meaning |
| --- | --- |
| `world_bit`, W | The actual final token location, known to the sender |
| `message_bit`, M | The location claimed by the exact outgoing message |
| `delivered`, A | Whether the recipient actually fully reads the message |
| `receipt`, B | Whether the sender's trusted receipt reports full reading |

The jointly observed initial location is `1-M`. The reader sees no subsequent
world evidence. If delivered, the service message is explicitly trusted and its
assertion replaces the reader's remembered location, including when it is false.
Non-delivery includes no notification or partial content. The sender knows the
initial location, actual current location, exact message, update rule and receipt;
it cannot independently observe actual delivery. Possible receipt error is not
information available to the sender.

For the fixed proposition that the token is in `place_b`, the normative labels are:

```text
world  = W
reader = M if A else 1-M
nested = M if B else 1-M
```

These are stipulated fictional contents, not measured human or model beliefs.
All 64 cases are agent-assisted drafts, `needs_human_review=true`. Mathematical
and visibility checks do not replace independent human review of trust, chronology,
reference resolution or the meaning of each source paragraph.

| W | M | A | B | Reader label | Nested label | Why informative |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 0 | 1 | 0 | 1 | Sender attributes a false update that never reached the reader |
| 1 | 0 | 1 | 0 | 0 | 1 | Reader accepts false content; sender attributes unchanged initial belief |
| 1 | 1 | 1 | 0 | 1 | 0 | Accurate message reaches reader while sender attributes non-delivery |
| 0 | 0 | 0 | 1 | 1 | 0 | Accurate message fails while sender attributes successful reading |

Every target is balanced within each group. All eight `(world,reader,nested)`
triples occur twice per group. The sender view excludes actual delivery, leaving
32 distinct sender prompts across 64 rows. Identical private-delivery pairs have
opposing actual-reader labels, so a deterministic reader-target head must get
exactly 50%. The omniscient view contains the narrator's explicitly private account.

## Frozen comparisons and timing

Use the previously saved story and nested-question block-16 heads with their
complete source-condition scaler, normalization, coefficients and intercepts.
Do not fit, recalibrate, select regularization or choose a layer from this panel.
Other heads may be reported as declared diagnostics. This is out-of-condition
evaluation of prior-state heads, not evidence that their semantic mapping is valid.

Capture each view in the existing story, nested-question and world-question
conditions at the final prompt token before any answer. The system instruction
and three question stems are exactly reused. Known message and receipt are present
in every condition. The primary target is sender-view nested attribution, block 16.
The old question-length/position difference remains: this panel does **not** isolate
a causal semantic effect of asking a belief question.

Separately record direct explicit answers and a prospectively fixed structured
text-reasoning baseline with the same permitted scene information. Retain strict
format invalids; do not repair or selectively regenerate failed outputs. The
text monitor's own output must never generate or alter the normative labels.
Strong text reasoning is a comparator, not established merely by calling a prompt
structured. Save its complete prompt, parser, checkpoint and decoding settings.

Report correct/incorrect/invalid counts, class recalls, log loss for numeric
readouts, per-group results and world/nested/reader disagreement strata. Keep
paired sender rows together and identify 32 unique sender inputs. Do not claim
population precision or probability calibration from the four groups.

## Software API and invariants

[message_belief_panel.py](../src/latent_probes/message_belief_panel.py) provides:

- `build_panel()`: deterministic complete draft, without model calls.
- `load_panel(path)` and `validate_panel(panel)`: strict frozen-content and visibility checks.
- `render_messages(case, view)`: detached `system,user` messages for `sender` or `omniscient`.
- `labels(case)`: derive targets from the four source bits, independently of stored gold.

Cases use `id`, `template_id`, `split="fresh"`, `phase="known_message"`, the four
source bits, `reader_name`, `sender_name`, `place_a`, `place_b`, `claim_place_bit=1`,
`needs_human_review=true`, `labels`, `messages` and `questions`. Source metadata
and gold never enter rendered messages. Tests check exact entity disjointness
against both observed banks, all sixteen source combinations, gold independence,
opposing-label identical sender inputs, corruption rejection and saved JSON parity.

## Decision boundary

Failure of explicit behavior calls for a narrower task or a stronger reference,
not an absence-of-belief claim. If behavior succeeds but frozen decoding fails,
the result is a prior-to-message readout transfer limitation. If queried states
alone work, queried communication must remain an explicit experimental condition.
Pressure transfer still requires independent neutral validation with matching
target and capture time. Any observed version of this panel becomes development
evidence and cannot subsequently be reused as an untouched holdout.
