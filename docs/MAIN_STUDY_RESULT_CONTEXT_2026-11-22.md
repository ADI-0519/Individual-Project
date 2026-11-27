# Interpreting the first matched-study result

**22 November 2026, 16:10 UTC. Llama result only; Qwen and the fixed prose monitor
are not included in this snapshot. Agent-audited exploratory study; human review
incomplete; confirmatory result false.** This document explains saved results
and primary literature. It changes no running capture, fit, baseline, endpoint
or analysis policy. The [prospective amendment](MAIN_STUDY_AGENT_EXECUTION_AMENDMENT_2026-11-22.md)
and [novelty audit](MAIN_STUDY_NOVELTY_AUDIT_2026-11-22.md) remain the scope record.

## What the Llama result says

Adding belief-preserving fitting examples **did not improve the prespecified
correction-quartet outcome** in the evaluated Llama pipeline. Restricted coverage
scored **3.75%**, full coverage **3.00%**; full minus restricted was **−0.75
percentage points**, with the saved conditional group-bootstrap 95% interval
**[−1.92, +0.33] percentage points**. Both arms have very low absolute joint
success. The interval includes zero, so this is not evidence of a reliably
negative effect or a formal equivalence result. It provides no support for the
expected positive transfer benefit at this fixed N/layer/precision/task design.

This is a completed empirical comparison, rather than a prediction or a
software-fixture score. It has **30 authored correction contexts**, eight
quartets per context and **five paired training allocations**. The interval
resamples contexts with their dependent variants and allocations retained; it
does not generalize over arbitrary reasoning mechanisms, prompt renderers or
models. The five allocation differences include one positive and four negative
values, ranging from +1.25 to −2.08 percentage points. They are not five new
datasets or independently sampled language-model populations.

The saved source is:

```text
/home/adirj/kcl-latent-probes-runs/main_v1_agent_20261122_recovery_v1/llama_analysis_without_monitor.json
SHA256 32793a3e5bfbb31355dcacb9638ab17c43d99bd4a8947732d6acdfd9aae104e7
```

Its exact bytes were read and hashed independently for this note. Its flags are
`test_fixture=false`, `research_status=agent_audited_exploratory`,
`human_review_complete=false`, and `confirmatory_result=false`. The source has
all five allocations complete at each of N128/256/512 for activation and TF-IDF
fits; `prose_monitor.status=not_provided`. This note does not independently
reauthenticate the underlying activation arrays; that is the separate capture
audit's role. No model or paid API call was made for this interpretation.

## Joint consistency, row accuracy and the same-format control

The following values are **descriptive macro averages**, equally weighting
authored contexts and the five allocations at N512. Only the primary paired
contrast above has the saved interval; no new intervals are invented here.

| Existing saved outcome | Restricted | Full |
| --- | ---: | ---: |
| Fresh observation: complete primary quartets | 60.75% | 59.75% |
| Fresh message: complete primary quartets | 41.17% | 40.67% |
| Correction: complete primary quartets | 3.75% | 3.00% |
| Correction: row accuracy across all 64 cells/context | 68.53% | 66.81% |
| Correction: secondary I=M1 row accuracy, 32 cells/context | 68.33% | 63.63% |
| Correction: calibrated complete primary quartets | 4.00% | 3.33% |

A quartet succeeds only when **all four linked counterfactual decisions are
correct**. Moderate row accuracy can coexist with failure of almost every
quartet; it is not evidence that the joint metric is wrong. Constant,
initial-only and final-update-only rules solve zero complete quartets here.
An independence calculation such as 0.5^4 is not an appropriate tested chance
floor for these correlated predictions. Neither 50% nor a binomial calculation
over factorial rows is the inferential baseline for the quartet result.

The secondary correction cells share the two-message format, but I=M1 makes
the target independent of the first receipt. Their modest accuracy and decline
with full coverage mean the observed weakness is **not cleanly isolated to the
harder primary dependence**. Length, extra events, replacement wording and
representation changes remain plausible contributors. The design tests transfer
to these sequences; it does not identify which contributor caused failure.

There is a concrete tradeoff in a familiar mechanism. On observation cases,
preserving-pair verdict flips fall from **4.67% to 0.75%**, while changing-pair
both-correct success falls from **63.50% to 59.92%**. This is useful evidence that
an invariance improvement need not improve the combined task. It should be
reported alongside, rather than substituted for, the disappointing primary
outcome. The other budgets and mechanisms remain in the full results tables.

Development-only calibration also leaves correction quartet success low. The
calibrated output-probability comparator scores **2.92%**; the calibrated probes
score 4.00% and 3.33%. These small descriptive differences, without the fixed
prose monitor and a comparator uncertainty analysis, do not establish a useful
latent-monitor advantage. TF-IDF is also a limited comparator: its inability to
solve conditional updates would not prove a privileged belief representation.

## What the closest primary methods imply

These are targeted re-readings of the methods/results relevant to this result,
not additional novelty claims. The longer audit records earlier reading scope.

- **Orgad et al., [Sections 3.4–4 and Appendix A.4](https://arxiv.org/html/2410.02707v2#S4):**
  their transfer analysis subtracts a strong logit comparator and distinguishes
  token locations. It also permits test-set direction reversal through absolute
  AUROC. Our frozen source orientation must remain unchanged. Their findings
  motivate completing output/prose comparisons, not rescuing our test score by
  reversing its sign or selecting a new token. Their target is output
  correctness, not sender-attributed belief.
- **Prakash et al., [Sections 5.1–5.2 and Appendix M](https://arxiv.org/html/2505.14685v3#S5):**
  the final-token representation can carry a retrieval pointer before an answer
  value is retrieved in later layers. Appendix M reports different subspaces
  across distributions despite related lookback structure. Consequently a fixed
  midpoint linear readout can fail without showing that the model lacks relevant
  information everywhere. Those mechanisms were studied on larger models and
  different tasks; they are hypotheses for interpretation, not demonstrated
  explanations of our Llama errors.
- **Klein, [Sections 4–5 and Appendix B](https://arxiv.org/html/2611.07805v1#S4):**
  decodability and nuisance selectivity can dissociate, and training correlations
  affect both. Our reduced preserving-pair flips alongside weak joint success
  is compatible with that distinction. It is not a new proof that probes are
  generally invalid, nor a replication of the paper's numerical effects.
- **Steele et al., [Section 4.4 and Appendices A–C](https://arxiv.org/pdf/2607.11945v2):**
  value storage, query routing and elicitation are distinct tests. Appendix A
  specifically reports framing sensitivity for Llama3.1-8B while Qwen7B is more
  robust in their separate task. Thus a Qwen disagreement would be informative,
  rather than an excuse to discard Llama. Their causal interventions and
  scaffolding results do not license causal or absent-capacity claims from our
  single readout.

## Decisions once the declared comparisons finish

| Completed pattern | Defensible interpretation and next deliverable |
| --- | --- |
| Qwen also shows little benefit; a capable prose monitor succeeds | Evidence that the specified one-step fitting coverage does not produce reliable correction readouts in both selected models, despite text-based solvability. Write the bounded transfer/measurement failure with both models and baselines. |
| Qwen benefits materially while Llama does not | Model-specific transfer, not a universal support effect. Report both unchanged and inspect the predeclared same-format and learning-curve diagnostics. |
| Prose monitor and output methods also fail | Failure cannot be localized to the latent probe or its fitting coverage. Report joint task/elicitation/extraction limitations, including all invalid outputs. |
| Prose monitor produces many invalid extractions | Its failed extraction process is part of its performance. Report full denominators and error types; do not silently retry or call a valid-only subset the main baseline. |
| A source-binding, capture or scoring defect is found | Preserve the attempt and correction record. Resolve that concrete defect before interpreting it as model behavior; do not reframe an implementation error as a scientific finding. |

Do not change the running pipeline, search layers/prompts for a positive result,
or restart goal/pressure pilots while these outcomes remain unfinished. A
successful second model or baseline is useful only if it is reported under the
same declared rules as an unsuccessful one.

After the core study is complete, a **fresh, separately frozen cohort** could
distinguish a general two-event/format shift from state-update composition.
Alternatively, a development-selected layer/token comparison or directly trained
correction readout could distinguish readout-location from training-support
limits. These are possible subsequent discriminating experiments, not changes
to this study and not promises of a positive outcome. Existing correction test
cases are now exposed and cannot be rebranded as untouched confirmation.

## Immediate dissertation and supervisor deliverable

Complete Qwen, the fixed prose monitor and the independent saved-array checks;
then generate the prewritten report and figures from the final saved analyses.
The next written asset should contain the question, matched design, primary
figure, same-format diagnostic, practical baseline table, uncertainty scope,
failed attempts/recovery provenance, and one paragraph explaining which initial
expectation the result did not support. An accurate completed empirical argument
is stronger than another unqualified promising pilot.

For the supervisor's **Variant 2**, explain the material scope shift: Apollo
remains the deception-probe reproduction anchor, while this experiment evaluates
the distribution-shift reliability of a separate attributed-belief readout.
It has not completed the original natural omission/recipient-outcome extension.
A bridge back to misleading communication requires valid behavior labels and
independently specified reader evidence/outcomes; adding pressure to a weak
readout alone would not provide that bridge.

If the final full comparison remains a bounded null, its dissertation value is
the controlled disconfirmation, diagnosis and reproducibility. A standalone
paper would need a substantive differentiating argument beyond the already
known fact that probes can fail transfer. Current evidence neither guarantees
publication/full marks nor establishes that the project is doomed. For an
application, present precisely what the student identified, designed, tested
and learned, with the supervisor able to verify that contribution.
