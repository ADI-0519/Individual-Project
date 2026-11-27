# Completed exploratory study: coverage repair did not improve correction consistency

22 November 2026. The fixed campaign completed at **19:14:37.404004 UTC**,
including both model evaluations, the fixed prose monitor, integrated analyses
and the generated report. This completes the specified exploratory experiment,
not the dissertation or a publication claim. Human semantic review remains
incomplete. No model, fitting rule, threshold, parser or saved outcome was
changed for this interpretation.

Read the [completed generated report](results/main-study-2026-11-22-complete/results.md),
[primary figure](results/main-study-2026-11-22-complete/primary.pdf) and
[learning curves](results/main-study-2026-11-22-complete/learning_curves.pdf)
alongside this interpretation and its recovery disclosure.

The result is mixed but clear: **neither model shows the proposed primary
training-coverage benefit. The capable text pipeline is much stronger on
correction consistency, while Llama's latent readouts remain stronger on the
one-step quartets.** This does not establish absence of belief information,
spontaneous anticipation or deceptive intent.

## Primary result remains unchanged

| N512 raw activation readout | Restricted quartet % | Full quartet % | Full minus restricted, points | 95% group-bootstrap interval, points |
| --- | ---: | ---: | ---: | ---: |
| Llama 3.1 8B | 3.7500 | 3.0000 | −0.7500 | [−1.9167, +0.3333] |
| Qwen2.5 7B | 1.8333 | 1.0000 | −0.8333 | [−2.0000, +0.2500] |

Both intervals include zero; neither comparison supplies positive evidence for
the proposed benefit. This is not an equivalence test. Each model's interval
resamples 30 authored correction groups with five paired allocations retained.
The same groups are used across models; factorial rows, quartets and allocations
must not be counted as additional independent narratives. Both models fit
their own heads rather than sharing probe weights.

## What the strong text baseline changes

The monitor is one fixed **Qwen2.5-7B prose evidence extractor followed by the
declared symbolic update rule**. It receives the actual account and query, not
the bank's structured state or gold label. It is not the privileged symbolic
oracle. Its one saved run is attached to both model reports; it is not two
independent baseline replications.

| Complete-quartet accuracy % | Observation | Message | Correction |
| --- | ---: | ---: | ---: |
| Llama raw restricted / full, N512 | 60.7500 / 59.7500 | 41.1667 / 40.6667 | 3.7500 / 3.0000 |
| Llama dev-calibrated restricted / full | 60.5833 / 55.5000 | 42.2500 / 38.9167 | 4.0000 / 3.3333 |
| Qwen raw restricted / full, N512 | 9.1667 / 8.1667 | 2.4167 / 3.0833 | 1.8333 / 1.0000 |
| Qwen dev-calibrated restricted / full | 7.2500 / 9.5833 | 1.8333 / 4.0833 | 2.2500 / 1.9167 |
| Fixed prose extractor + symbolic fold | **41.2500** | **32.9167** | **33.7500** |

On correction, the text pipeline solves **81/240** unique quartets, compared
with much lower average head scores. It exceeds even the calibrated latent
comparators numerically. However, it still fails 159/240 complete correction
quartets; it is not a reliable general detector. Llama's one-step quartet
performance exceeds this text baseline, whereas Qwen is weak on one-step
matched consistency too. The two models therefore do not provide a clean
replication of “one-step success, composition alone fails.”

The practical comparison uses the same underlying prose but **different
computation**: the text pipeline gets up to 384 generated tokens and an external
symbolic fold, while the probe reads one fixed midpoint activation before an
answer. Its better correction score does not isolate the value of text access,
the extraction prompt, additional inference or the symbolic scaffold. No paired
confidence interval for these baseline gaps is produced by the frozen analysis;
the table is a descriptive comparison, not a new primary significance test.

## Row accuracy, invalid outputs and confidence

The fixed test monitor gets **2,962/3,840 rows correct (77.1354%)**. Equal-group
accuracy is **78.6111%**: these differ because correction groups contain twice
as many rows. There are **44 invalid outputs**, all counted incorrect in full
row, pair and quartet metrics. The independent audit attributes them to 34
non-verbatim locations, nine wrong update counts and one malformed JSON answer.
These are failures under the declared exact parser, not a fresh human assessment
of each response's semantic equivalence. No invalid response was repaired or
retried to improve the score.

| Fixed monitor test stratum | Correct / all rows | Accuracy % | Invalid | Correct quartets / all quartets |
| --- | ---: | ---: | ---: | ---: |
| Observation | 797 / 960 | 83.0208 | 2 | 99 / 240 |
| Message | 769 / 960 | 80.1042 | 30 | 79 / 240 |
| Correction | 1,396 / 1,920 | 72.7083 | 12 | 81 / 240 |

Correction's primary I=M2 rows score **710/960 (73.9583%)**, versus **686/960
(71.4583%)** on same-format secondary I=M1 rows. The primary stratum contains
all 12 correction invalids; the secondary has none. The simpler secondary
format therefore remains imperfect even with this monitor.

For comparison, Llama's correction row accuracy is **68.5313/66.8125% R/F raw**
and **68.5104/67.0313% calibrated**. Qwen's is **52.4583/54.1354% raw** and
**52.5417/54.4792% calibrated**. The text pipeline's improvement is much larger
on the joint quartet metric than on Llama's individual judgments. That is
consistent with the importance of evaluating all linked decisions rather than
reporting only average accuracy; it is not a causal account of the hidden states.

The calibrated same-model output comparator reaches **2.9167% Llama** and
**2.0833% Qwen** correction-quartet accuracy. Matched TF-IDF remains at zero
correction quartets. Neither provides a stronger correction comparator than
the completed prose pipeline.

Development calibration does not rescue the latent transfer result. Llama's
correction Brier scores change from **0.2265/0.2362 raw** to **0.2402/0.2494
calibrated**; Qwen's change from **0.2643/0.2899** to **0.2812/0.2809**.
Brier combines calibration and discrimination; it does not independently
identify a calibration mechanism. The prose monitor supplies hard class scores,
not continuous confidence forecasts. Its AUROC/Brier fields use only the 3,796
valid rows and must not be presented as full-denominator calibrated confidence
comparisons.

## Development was not the held-out result

The fixed monitor's development run completed **512** one-step rows across
16 groups: **430/512 correct (83.9844%)**, zero invalids, and observation/message
quartet scores **42.1875/45.3125%**. There are no development correction groups.
The matched one-step part of test is **1,566/1,920 correct (81.5625%)**. Comparing
83.9844% development directly with 77.1354% all-test accuracy would confound
fresh narratives with the addition and increased row weight of correction.

Development generation ran **16:22:23.523361–16:41:49.968558 UTC**. Test
generation ran **16:43:53.308975–19:12:19.252853 UTC**. Both use the same
prospectively frozen monitor policy hash and greedy 384-token cap. The monitor
prompt, parser and selection were not retuned after development or test output.

## Evidence boundaries and remaining work

- Independent audits reconstructed both models' 60 fitted heads, all saved
  raw/calibrated scores and primary bootstraps, plus every monitor raw response,
  parse/fold and group/pair/quartet result. These establish saved measurement
  integrity, not human semantic approval or causal faithfulness.
- Semantics were reviewed by an author-independent agent through 154 narrative
  setups and 16 normalized paths covering 736 representative prompts. This was
  grouped agent checking, not 736 independent human judgments or a full separate
  reading of every expanded input. Human records remain incomplete. Later
  checking cannot retrospectively make exposed results confirmatory.
- The shared renderer, explicit fictional update/trust rules, correction length
  and event count, selected midpoint layers and NF4 weights limit transfer
  claims. Qwen's weak one-step consistency particularly limits a composition-only
  interpretation. A failed readout does not imply the model lacks belief content.
- WSL interruption required an explicit recovery: 2,758 completed Llama rows
  were preserved and reused; one started-only attempt had no recorded outcome
  and was explicitly repeated among 1,082 pending cases. Original fits, freezes
  and sources were unchanged. That repeat is not an extra independent case.
- The current target differs materially from the original natural-deception
  and misleading-omission extension. The Apollo 8B adaptation remains separate
  from this experiment and from a full published 70B reproduction. Discuss the
  empirical scope with the supervisor; finish the argument, reproduction
  package and dissertation rather than treating a positive effect as required.

## Exact final sources

Native root:
`/home/adirj/kcl-latent-probes-runs/main_v1_agent_20261122_recovery_v1/`.
The unchanged report was generated at **19:14:37.185814 UTC on 22 November 2026**.
All eight report output hashes and both integrated analysis bindings were checked
against the final manifest for this note.

| Relative source | SHA256 |
| --- | --- |
| `llama_analysis_with_monitor.json` | `bfa3d7fefa3016b114f1b6de9931804f815c984ea8c5b702c6d12c0cafe26b19` |
| `qwen_analysis_with_monitor.json` | `02f66b9a15854e6c9209b3da6d9753d2c49ae2ff44b3e0ec07c77e162dcdff5d` |
| `monitor_test_analysis.json` | `6c7d5fba54c2e49170b1f5abcd4e32ebc1788bfb1f0ce134c018f5c9d02aa215` |
| `monitor_development_analysis.json` | `b477a10cfef1f06f41002a75f7b1e41a1afa83b066339236e6048ac8db005b62` |
| `report/results_summary.json` | `b08f5ae3b25e0d567230f984984a32af0e7c1d6e9cbc93db4808bf492a3d6f73` |
| `report/report_manifest.json` | `ac0482be0993919d7d2c21f9806abd2cd168764242df07efc5fae42b9ca39923` |
| `report/results.md` | `7632b3f887fe4776a4a80d015ce90b7f40954b6f4772387c2fd7c5d2baee1e44` |
| `recovery_receipt.json` | `11cde2ad6373423170c2057820eefa3cc8491f74d0f3c886c993f4f95802ebe3` |

Monitor policy SHA256:
`bcefd89009250413e7759991188e94f83169d7a7943345255c6a7b24c192d936`.
See the independent [development](validation/2026-11-22-main-study-monitor-development-independent.json)
and [test](validation/2026-11-22-main-study-monitor-test-independent.json) audits,
the earlier [two-model interpretation](MAIN_STUDY_TWO_MODEL_INTERIM_INTERPRETATION_2026-11-22.md),
and the [recovery account](MAIN_STUDY_INTERRUPTION_RECOVERY_2026-11-22.md).
