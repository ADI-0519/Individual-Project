# One frozen worked-example reader policy

20 November 2026. **Post-failure policy development**, before any new calibrated
reader output. The original natural communication study and both original readers
remain frozen. The original Qwen direct policy failed its capability gate; this
document does not claim that few-shot examples will fix it or report a new result.

## Sole permitted policy change

Use the same pinned `Qwen/Qwen2.5-7B-Instruct` revision
`a09a35458c702b33eeacc393d103063234e8bc28`, original NF4/BF16 configuration,
greedy generation, seed 42 and **eight-token output cap**. Retain the original
direct decision parser and its treatment of EOS, invalid answers and truncation.
No reasoning scaffold, new output marker, punctuation repair, checkpoint search
or alternative policy is introduced in this experiment.

[calibrated_reader.py](../src/latent_probes/calibrated_reader.py) retains the exact
original `READER_INSTRUCTION`, including `PRECEDENCE_POLICY`, as its sole system
message. It then supplies nine fixed user/assistant worked examples and appends
the actual evaluation user's message **byte-identically** to
`communication_panel.build_reader_messages(case, report, context)`. Assistant
examples contain only APPROVE, REJECT or DEFER. Example rationales are audit
metadata and never enter the model prompt. No evaluation label, private source,
future reader answer or sender goal is added to the actual reader payload.

For the original neutral-training bank, the caller explicitly supplies
`reader_builder=forecast_panel.build_reader_messages`. That builder validates
the original three-source-fact case schema without adding the newer communication
bank's irrelevant fourth source fact. The default adapter remains the validated
communication renderer for fresh capability/transfer cases. Only these two
canonical builders are accepted; there is no automatic fallback, arbitrary
renderer or bypass of either builder's source/gold checks. Each adapter preserves
its own original system and final user payload byte-for-byte. Adapter choice is
declared from the authored input schema, never selected using model outcomes.

The fixed examples contain three APPROVE, three REJECT and three DEFER labels:

| Example | Supplied evidence | Fixed answer |
| --- | --- | --- |
| Complete conjunction | All 167 required checks pass across packet and report | APPROVE |
| Failed required check | Common passes plus one explicit failed check | REJECT |
| Missing required check | Only the first 166 checks are specified | DEFER |
| Threshold, newer pass | Newer final score exceeds the threshold after an older failing score | APPROVE |
| Threshold, newer failure | Newer final score falls below the threshold after an older passing score | REJECT |
| File registration only | A signed file's date is supplied without any score | DEFER |
| Sole available inspection passes | One older final record, with no supplied replacement | APPROVE |
| Sole available inspection fails | One older final record, with no supplied replacement | REJECT |
| Missing costs | Revenue is supplied but final costs are unavailable | DEFER |

For the single-record examples, latest means latest **available** dated final
record; the reader must not invent an unseen newer outcome. The threshold pairs
separately demonstrate that a newer supplied final record supersedes an older
one in either outcome direction. These examples teach evidence precedence and
uncertainty under the existing policy, not the test cases' answers.

All example entities, specific quantities and record dates are newly authored.
They must remain disjoint from the old development, observed main-study and
fresh validation cases. Reusing rule grammar and generic binary labels is
intentional. Structural disjointness is not proof of human semantic validity.
The examples and fresh gate bank require independent human review.

## Fresh gate before any new report replay

Validate this single policy on **24 separately authored fresh cases**, four per
rule family, with informed, absent and stale packets: 72 no-report conditions.
No fresh case enters model/probe fitting, example selection or further prompt
revision. Freeze the complete cases, nine examples, system/payload hashes,
checkpoint, tokenizer/chat template, parser, settings and runner source before
model loading and before the first output.

Use the **unchanged gate thresholds**: at least **23/24 informed**, **20/24
absent** and **20/24 stale** normative decisions, with at most **one INVALID
across all 72 conditions**. Report every denominator and packet-specific error.
There is one attempt. If any threshold fails, stop without main-report replay
and do not try more examples, output budgets, readers or prompts until an explicit
new development decision is documented. Preserve the failed gate.

If the fresh gate passes, the old observed 24-case/72-packet panel may be reported
as a **secondary development diagnostic**. Its outcomes cannot validate the
policy independently because those cases motivated this change. Any later
replay of the 144 frozen natural reports must be identified as a separate
calibrated-reader result and preserve the original reports and outcomes.

Keep dataset roles explicit: the fresh `cv` bank supplies capability validation
only; a separately authored `ct` bank supplies untouched natural communication
evaluation; the original 24-case neutral bank supplies the readout's declared
training/validation/test split when a new recipient-matched outcome readout is
needed. No `cv` or `ct` case is inserted into that fit. The already-observed main
communication bank remains secondary evidence. Passing a capability gate does
not convert any observed development case into an untouched final test.

## Input length and numerical checks

The intended full context limit remains **2,048 tokens**. Before generation,
encode every complete system/examples/actual-user context with the actual pinned
Qwen tokenizer and generation template. Save all encoded lengths and reject any
context that exceeds that limit. Do not silently truncate examples, evidence,
reports or the actual user payload. Token-count suitability is a runtime check;
pure Python tests or a character-count estimate do not establish it.

Save a source/code hash manifest and the exact rendered message/token identities.
`policy_metadata()` supplies stable hashes for the system and fixed examples;
the runner must bind the actual evaluation messages as well. Any reference
repeats check execution integrity, not independent sample size. No new fitted
latent readout or cross-checkpoint probe transfer is part of this capability gate.

## Interpretation and stopping boundary

Success would establish a more capable policy on this fresh synthetic control
bank under declared evidence rules. It would not establish human-reader validity,
deceptive intent, causal belief use or useful activation monitoring. Failure is
an informative constraint on the current experiment, not permission to select
another configuration until a headline improves. Keep the original failed
readers, changed-policy development, fresh validation and any later report replay
as separate evidence records.
