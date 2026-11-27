# Progress 2: four-slide empirical update

Aditya Ranjan · 22 November 2026 · Draft content and speaker notes.

**Snapshot:** Both models' held-out analyses and independent numerical audits
are complete; the fixed prose monitor is pending. Update the monitor status
immediately before presenting. Earlier checkpoints in
the [README](../README.md), [continuation state](CURRENT_RESEARCH_STATE.md) and
[20 November brief](SUPERVISOR_PROGRESS_2_BRIEF.md) describe earlier stages;
their “no held-out result” wording does not describe this snapshot.
This draft does not infer the last meeting's date or claim that every item below
was completed after it. Identify the relevant interval from the actual meeting
record before presenting “since our last meeting.”

---

## Slide 1 — Latent Probes and Listener Beliefs

**Aditya Ranjan · 6CCSAPRJ Individual Project**

**Short description:** Testing whether a linear readout of what a sender thinks
a reader believes remains reliable when the evidence and message structure change.

**Progress 2 · 22 November 2026**

**Speaker notes — about 25 seconds**

“The original Variant 2 asks whether latent probes generalize beyond their
training conditions. I started with an Apollo-style deception-probe adaptation
on Llama 8B. The current controlled study asks a narrower, different question:
can a readout track a fictional sender's attribution of a reader's belief when
we change the evidence? This is an explicit scope change from the earlier
deception and omission proposal, which I want to discuss with you.”

---

## Slide 2 — Current progress: completed two-model comparisons

- Built **154 scenario groups**, with **5,888 distinct inputs per model** and
  separate training, development and test groups.
- Completed both models' training/fitting and independently audited their
  **full held-out evaluations**.
- Compared equal training budgets with different coverage of prior beliefs.

| N=512: all four correction judgments correct | Llama | Qwen |
| --- | ---: | ---: |
| Restricted training | **3.75%** | **1.83%** |
| Full coverage training | **3.00%** | **1.00%** |
| Full minus restricted, percentage points | **−0.75** | **−0.83** |
| 95% group-bootstrap interval, points | **−1.92 to +0.33** | **−2.00 to +0.25** |

**Neither primary comparison supports a coverage benefit.**

**Speaker notes — about 60 seconds**

“Restricted training uses cases where the earlier belief differs from the
incoming information; full training also includes agreement. Both arms have
the same number of training examples and the same scenario groups. Both also
use full-support development labels for selection, so the contrast concerns
fitting support, not complete absence of agreement examples anywhere.

“Llama performs substantially better on fresh one-message cases: about 89–90%
individual-judgment accuracy for observation and 84–85% for messages. Qwen is
weaker already on those cases: about 66–72% row accuracy, with only 2–9% of
complete matched quartets correct. We therefore cannot describe both models
as passing one-message reasoning and failing only composition.

“The strict metric requires all four matched judgments to be correct: a shortcut
can get three out of four right yet fail it. Each primary analysis uses the
same 30 correction groups and five paired allocations; groups are the
uncertainty units. Independent audits reconstructed all fitted-head scores,
group/pair/quartet summaries and both bootstraps. This supports measurement
integrity, not a claim about an internal causal belief mechanism.”

---

## Slide 3 — Current barriers: interpretation and scope

- **Strong baseline pending:** the fixed prose-only monitor is needed for the
  complete comparison.
- **Model differences complicate interpretation:** Qwen struggles with complete
  one-message contrasts too; this is not an isolated composition failure.
- **Semantic validation is agent-only:** human review remains incomplete;
  these are exploratory results.
- **The shift changes more than reasoning structure:** correction accounts
  are longer and contain more events.
- **Scope needs discussion:** attributed-belief robustness is different from
  detecting deceptive behaviour or anticipating a real reader.

**Speaker notes — about 50 seconds**

“The negative primary result is a finding, rather than a reason to change the
test until it becomes positive. The present barrier is completing the planned
comparisons and making the interpretation defensible. A separate author-independent
agent's blinded answers agree with 736 representative prompts, but this was
grouped agent checking, not human approval or independent reading of every
expanded prompt. The material also shares a common renderer.

“The earlier communication experiments encountered reader-validity problems;
their dependent studies remain stopped. This controlled task separates a
precisely defined belief-attribution target from those problems, but materially
departs from the proposed misleading-omission experiment. It cannot establish
spontaneous anticipation, deceptive intent or a causal belief mechanism.
The Apollo component is an audited 8B adaptation of the recipe, not a reproduction
of the full published 70B study.”

---

## Slide 4 — Next steps: finish the fixed evaluation and agree the contribution

1. Finish the **fixed prose monitor**, preserving invalid answers and failures.
2. Produce the complete comparison: both models, text/output baselines,
   calibration, one-message retention and correction transfer.
3. Discuss the **revised main-study scope** with the supervisor; arrange human
   semantic review and document the work's limitations.
4. Turn the evidence into clear figures, a reproducible results package and
   the background/specification/design write-up.

**Plan position:** materials, fitting and both model evaluations complete → strong
baseline in progress → complete interpretation and write-up remain.

**Speaker notes — about 45 seconds**

“I will finish the frozen study before deciding its contribution. Both primary
contrasts are negative and their intervals include zero. The immediate question
is whether the capable prose monitor solves accounts on which the latent
readouts fail. That comparison, together with the difference between the two
models, will determine the conclusion. I will not add models or change
thresholds to find a positive result.

“I propose presenting this as a controlled distribution-shift study alongside
the separate Apollo adaptation, with deception and natural communication as
motivation and future evaluation rather than completed claims. I want your
view on whether that should be the main empirical scope. Later human review
will not retrospectively make these exposed outcomes confirmatory. This is
the current stage of the overall plan; I have not claimed that it meets an
unverified calendar milestone.”

---

## Supporting evidence — not additional slides

- [Llama interpretation and complete secondary summaries](MAIN_STUDY_LLAMA_INTERIM_INTERPRETATION_2026-11-22.md).
- [Two-model interpretation and Qwen secondary results](MAIN_STUDY_TWO_MODEL_INTERIM_INTERPRETATION_2026-11-22.md).
- [Independent held-out numerical audit](validation/2026-11-22-main-study-llama-heldout-independent.json).
- [Independent Qwen held-out numerical audit](validation/2026-11-22-main-study-qwen-heldout-independent.json).
- [Main-study specification and declared limitations](MAIN_STUDY_SPECIFICATION_V1.md).
- [Prospective agent-only exploratory amendment](MAIN_STUDY_AGENT_EXECUTION_AMENDMENT_2026-11-22.md).
- [Apollo method and 8B adaptation](APOLLO_METHOD.md).
- [Assessment/scope evidence](MAIN_STUDY_ASSESSMENT_EVIDENCE_2026-11-22.md).

**If asked about the interruption:** WSL stopped during Llama test capture.
All 2,758 completed measurements were preserved and independently checked.
Recovery reused the existing fitted heads and frozen policy, then completed
the remaining cases in a new directory. One attempt had no recorded outcome
and was explicitly repeated; it is not another independent case. See the
[recovery record](MAIN_STUDY_INTERRUPTION_RECOVERY_2026-11-22.md).

**If asked about CREATE:** local NF4 inference has been sufficient for this
core study. A full-precision or larger-checkpoint comparison would be a separate,
justified experiment with a measured resource request. CREATE access would not
guarantee a positive effect or rescue an unsupported conclusion.
