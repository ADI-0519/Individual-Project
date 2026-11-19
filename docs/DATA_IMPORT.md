# Importing independent probe examples

`data/probe_examples_template.jsonl` shows the input format for independently
specified examples. It contains **four invented format examples from only two
related template families**. It is not a research dataset, a reproduction of an
existing benchmark, or enough material to train and evaluate a probe. The probe
fitter intentionally requires at least three groups for train/validation/test,
so fitting this template alone should fail.

Use it to check the import, generation, activation capture and packing path.
Replace it with a larger, separately designed and audited dataset before a
research experiment. More paraphrases of the same four stories do not create
more independent families.

## One JSON object per line

Each row includes:

| Field | Meaning |
| --- | --- |
| `example_id` | Unique identifier for this exact example. |
| `scenario_id` | Underlying paired situation; related variants stay together. |
| `family_id` | Coarsest related story/template family used for splitting. |
| `messages` | Exact `system`/`user` messages passed to the model. |
| `label_target` | Explicit definition of the supervised target. |
| `label` | Mechanically assigned binary target, 0 or 1. |
| `expected_answer` | Human-auditable expected YES/NO answer, not a model result. |
| `notes` | Why the label follows and what the example does not establish. |

Only `messages` should reach the model. The target, expected answer and notes
are annotation/provenance fields and must not be appended to its input.

For these examples, `recipient_prior_belief_true` means:

> The named fictional recipient, under the story's stated observation and memory
> assumptions, believes the supplied proposition is true before any new
> communication to that recipient.

It does **not** mean the proposition is actually true. For example, Niko can
believe the key is in the left locker after an unobserved move to the right
locker: the belief-content label is 1 although the proposition is false in the
world. The observed/unobserved pair keeps the physical outcome fixed.

## Choose the capture position deliberately

For these explicit belief questions, start with **`prompt_last`**. The label is
defined from the story before the model generates an answer. A `response_last`
vector can trivially contain the model's explicit YES/NO answer; high decoding
accuracy there would not demonstrate a latent prior-belief representation.
Inspect the generated answer as a separate behavioural check, including cases
where it disagrees with the expected answer. Do not silently discard those cases
to make the probe's dataset easier.

The capture signature identifies the model, precision, layer and position.
Train and predict with a matching signature. A probe trained at `prompt_last`
cannot be quietly applied at `response_last`. Changing that convention is a
separate transfer experiment that requires a justified implementation and
validation.

## What this starting task can and cannot support

The model is explicitly asked about another agent's perspective. This is an
**elicited belief task**. It can provide a candidate readout for a later transfer
test, but it does not show that a sender spontaneously anticipated its reader's
conclusion while composing a report.

Three questions remain distinct:

1. What information does the sender attribute to the reader before a message?
2. What conclusion does the sender expect this particular message to produce?
3. What does the actual reader conclude?

These four examples illustrate the first type of target. They supply no labels
for the second or third. A later message-conditioned dataset would need explicit
message effects, alternative reader states and independently justified target
labels. It should not reuse an elicited forecast as unquestioned ground truth
for spontaneous anticipation.

## Before collecting a proper training dataset

- Add genuinely different story and reasoning families; keep all variants of a
  source scenario/template in the same partition.
- Independently vary world truth and the named agent's belief content.
- Balance which physical locations and answer words correspond to each label.
- Include self/world versus other-agent perspective controls.
- Include irrelevant events and distractor agents, and vary whose observation
  matters rather than simply whether the story contains "does not see".
- Include true-belief cases as well as false-belief cases.
- Separate lack of knowledge from a definite belief that a proposition is false.
  This four-row binary template deliberately avoids unknown states; a larger
  study must define how uncertainty is represented and scored.
- Check that the underlying model can answer the task sufficiently well for the
  proposed interpretation. Record failures rather than using expected answers as
  evidence that the model actually formed those beliefs.
- Keep communication evaluation scenarios separate from neutral probe-training
  data. Reserve fresh families for final evaluation after development.

Choose dataset size using pilot behaviour, available independent families and
learning curves. There is no arbitrary example count that guarantees a valid
belief probe or a novel result.

For the general fitting and evaluation contract, read [PROBES.md](PROBES.md).
