# Progress 2 explanation of the completed format study

Suggested spoken explanation for the
[four-slide deck](../presentations/progress2-format-adaptation-2026-11-27/Progress-2-format-adaptation.pptx).
Use the [full interpretation](FORMAT_ADAPTATION_RESULTS_INTERPRETATION_2026-11-27.md)
for questions about the controls and limitations.

## Slide 1: the question

My project evaluates how reliable latent probes are under distribution shifts.
The current experiment asks whether a supervised readout of what a fictional
sender thinks a reader believes survives a change in prompt format. The belief
question is explicit. Four classifiers read factual ingredients from the model's
activations, and a supplied rule combines them into the final judgment.

## Slide 2: what was achieved

We completed 11,520 measurements across Llama and Qwen, with 16 fitting contexts
and 24 test contexts. Both fitted bundles were frozen before testing. The
comparison holds the fitting budget at 512 observations and compares using
only format A with allocating half the observations to each format.

On the harder format-B cases, complete four-case panels improve from 2 out of
192 to 151 for Llama, and from 3 to 125 for Qwen. An intercept-only adaptation
using the same observations reaches 20 and 11. Llama's primary gain is 77.60
percentage points, with a conditional 95% interval from 68.23 to 85.42.
A panel counts only when all four related reading-record judgments are correct.

## Slide 3: what still limits the result

The repair has a cost. In format A, Llama falls from 185 to 166 complete panels,
and Qwen from 162 to 131. One-step controls also remain weak. This is partial
repair of this supervised measurement, rather than a generally reliable belief
monitor. The supplied rule and extra factor labels also prevent interpreting it
as proof of the model's own reasoning process. Human semantic review is still
incomplete, and the measurements use NF4 weights and two known formats.

## Slide 4: how this changes the plan

The controlled comparison is now complete. The next deliverables are the scoped
research argument, human review and supervisor feedback, followed by a separately
specified BF16 replication if CREATE access is available. Full-precision weights
and runtime overhead exceed the practical capacity of the local 16 GB GPU.
CREATE would enable that numerical reference; it is not an established remedy
for the remaining failures. The original Apollo adaptation and unfinished
natural-communication experiments remain separate parts of the project.

## Short answers for discussion

- **Is second-order belief itself new?** No. The contribution concerns this
  fixed-budget transfer comparison, its intercept control and its linked panels.
- **Did we prove anticipation or deception?** No. This assay measures explicitly
  queried fictional attribution under supervised readout and supplied computation.
- **Are 11,520 measurements independent stories?** No. There are 24 shared test
  contexts; uncertainty resamples those contexts and conditions on the fixed fits.
- **Can someone check the numbers without a GPU?** Yes. The saved-score bundle
  replays the metrics. A separate native audit additionally reconstructs the
  predictions from saved features.
