# Two-model interim interpretation: no primary coverage benefit; different failure profiles

22 November 2026. Both frozen held-out analyses and independent numerical audits
are complete. The fixed prose monitor remains pending. This is an interim
interpretation of real agent-audited exploratory data, not a complete baseline
comparison, human-reviewed confirmation or evidence that belief information
does not exist. No model, source policy, fitting or result output was changed
for this note. The earlier [Llama-only note](MAIN_STUDY_LLAMA_INTERIM_INTERPRETATION_2026-11-22.md)
retains its original dated scope.

The [unchanged generated report](results/main-study-2026-11-22-interim/results.md),
[primary figure](results/main-study-2026-11-22-interim/primary.pdf),
[learning curves](results/main-study-2026-11-22-interim/learning_curves.pdf) and
[output/source manifest](results/main-study-2026-11-22-interim/report_manifest.json)
are byte-verified exports of the two saved analyses. Read this interpretation
and its recovery disclosure alongside those figures.

## Primary comparisons

| N512 raw activation readout | Restricted quartet % | Full quartet % | Full minus restricted, points | 95% group-bootstrap interval, points |
| --- | ---: | ---: | ---: | ---: |
| Llama 3.1 8B | 3.7500 | 3.0000 | −0.7500 | [−1.9167, +0.3333] |
| Qwen2.5 7B | 1.8333 | 1.0000 | −0.8333 | [−2.0000, +0.2500] |

Both point estimates are negative and both intervals include zero. Neither
provides positive evidence for the proposed coverage benefit. This is not an
equivalence test or a universal conclusion about training coverage. Each analysis
uses 30 correction groups, eight quartets/group and five paired allocations:
Llama solves 45 versus 36 of 1,200 repeated quartet evaluations; Qwen solves
22 versus 12. The same authored groups are reused across models, so they must
not be pooled as 60 independent correction narratives. Qwen fits its own heads;
this is replication with a second checkpoint, not transfer of Llama weights.

## The models do not have the same one-step starting point

All percentages below are means over the fixed five allocations and equal-sized
groups. The target is balanced, so row and balanced accuracy coincide here.

| N512 raw activation | One-step observation row %, R/F | Observation quartet %, R/F | One-step message row %, R/F | Message quartet %, R/F |
| --- | ---: | ---: | ---: | ---: |
| Llama | 89.3750 / 89.7292 | 60.7500 / 59.7500 | 84.6250 / 84.3750 | 41.1667 / 40.6667 |
| Qwen | 68.3333 / 71.8958 | 9.1667 / 8.1667 | 65.9167 / 72.1875 | 2.4167 / 3.0833 |

Llama's stronger one-step results followed by very poor correction consistency
support a pronounced shift-specific problem in this pipeline. **Qwen already
struggles with complete one-step contrasts.** It cannot be described as a clean
replication of “one-step reasoning succeeds, composition alone fails.” Full
fitting improves Qwen's individual-judgment accuracy, while complete matched
reliability remains weak. Its raw class-0 recall on one-step cases is only
35.7083–47.7500%, versus class-1 recall of 89.8750–96.6250% across arms/mechanisms.
Within-group AUROC of 0.8261–0.9122 nevertheless shows nontrivial score ranking.
Poor thresholded consistency is not an absence test for representational content.

Development calibration does not resolve this: Qwen's N512 calibrated one-step
quartets are **7.2500/9.5833% R/F observation**, **1.8333/4.0833% message**.
These are practical secondary comparisons, not replacements for raw primary
decisions.

## Qwen's correction diagnostics

These are descriptive summaries. Most are predeclared saved fields; primary
I=M2 row accuracy is an additional regrouping after the outcomes were known,
not a separately prespecified endpoint.

| N512 Qwen raw diagnostic | Restricted | Full |
| --- | ---: | ---: |
| All correction row accuracy | 52.4583% | 54.1354% |
| Primary I=M2 row accuracy | 54.1667% | 58.4792% |
| Same-format secondary I=M1 row accuracy | 50.7500% | 49.7917% |
| Correction class-0 recall | 72.0208% | 39.0625% |
| Correction class-1 recall | 32.8958% | 69.2083% |
| Both endpoints correct, changing receipt pairs | 9.0000% | 4.4444% |
| Both endpoints correct, preserving receipt pairs | 49.0833% | 60.8333% |
| Preserving-pair verdict flips | 18.2500% | 12.5000% |
| Macro within-group AUROC | 0.5556 | 0.5757 |
| Brier score | 0.2643 | 0.2899 |

The simpler same-format secondary cells remain near 50%. Together with weak
one-step quartet scores, this supports a broad generalization/reliability
problem rather than isolating one abstract composition operation. The full arm
improves row accuracy and some preserving-pair measures, but loses changing-pair
correctness and reverses the class imbalance. No internal causal explanation
follows from these descriptive secondary metrics.

Raw correction-quartet success remains low at all three training budgets:

| Qwen training N | Correction quartet %, R/F | All-correction row %, R/F |
| --- | ---: | ---: |
| 128 | 1.5833 / 0.0833 | 51.6458 / 50.8333 |
| 256 | 1.3333 / 0.6667 | 51.4688 / 52.1458 |
| 512 | 1.8333 / 1.0000 | 52.4583 / 54.1354 |

All 30 Qwen comparisons and 60 arm calibrations completed; no C candidate was
excluded. Every activation comparison again selected the predeclared grid's
upper endpoint C=10. Extending that grid now would be a new exploratory
intervention, not a correction to the frozen result.

## Calibration and available baselines

Qwen's N512 calibrated correction row accuracy is **52.5417/54.4792% R/F** and
quartet accuracy **2.2500/1.9167%**. Correction Brier scores become
**0.2812/0.2809**: calibration worsens the restricted score and improves the full
score slightly. Both remain worse on Brier score than the constant probability
0.5 reference (0.25). Brier alone does not isolate calibration error from
discrimination error.

Its same-model output baseline reaches **58.7500% raw** and **56.8750%
development-calibrated** correction row accuracy; complete-quartet accuracy is
**0/2.0833%**. Thus the latent readouts do not have a clear practical advantage
on Qwen. Calibrated output reaches **72.2917% observation/76.5625% message** row
accuracy, compared with **72.8750/72.8125%** for the full calibrated latent
readout. This is a mixed comparison. A pointwise ranking does not by itself
establish statistical superiority or general monitoring value.

Paired TF-IDF correction row accuracy is **50.0000/50.0521% R/F**, with zero
complete quartets. Its weakness is not a substitute for the missing capable
prose comparison. The privileged symbolic oracle solves all quartets by
construction and must not be presented as a text-only monitor.

## Remaining interpretation and provenance

Finish the unchanged fixed prose monitor and report its invalid-output rate,
complete decisions and accuracy on all cases. It may clarify whether a capable
text-only evidence extractor succeeds where these readouts fail. Do not infer
the outcome while development/test generation is pending. Do not change cases,
models, layers, regularization or thresholds to seek a positive primary effect.

Shared renderer and synthetic framing, longer correction accounts, additional
events, model-specific midpoint readouts and NF4 weights limit generalization.
The target remains fictional sender-attributed reader belief under explicit
update/trust rules; it is not actual private reading, deceptive intent,
recipient-outcome forecasting or spontaneous anticipation.

Source files under `/home/adirj/kcl-latent-probes-runs/`:

- `main_v1_agent_20261122_recovery_v1/llama_analysis_without_monitor.json`, SHA256
  `32793a3e5bfbb31355dcacb9638ab17c43d99bd4a8947732d6acdfd9aae104e7`.
- `main_v1_agent_20261122_recovery_v1/qwen_analysis_without_monitor.json`, SHA256
  `a9c500ecf13572f02f70886a96d85d213f8e5cf393abd444bfea104175d61b52`.
- Original fitted bundles in `main_v1_agent_20261122_campaign/{llama,qwen}_fit/`.

The [Llama independent audit](validation/2026-11-22-main-study-llama-heldout-independent-v2.json)
and [Qwen independent audit](validation/2026-11-22-main-study-qwen-heldout-independent.json)
reconstruct all 60 heads/model, raw/calibrated group/pair/quartet metrics and both
primary bootstraps without refitting. The original interrupted Llama attempt
and [explicit recovery](MAIN_STUDY_INTERRUPTION_RECOVERY_2026-11-22.md) remain
disclosed. Human semantic validation is still incomplete; neither numerical
agreement nor later human checking can retrospectively make exposed outcomes
confirmatory.
