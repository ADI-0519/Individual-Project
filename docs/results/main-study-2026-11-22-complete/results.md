# Matched belief-study results

**agent_audited_exploratory | human review INCOMPLETE | confirmatory result: false**

Target: fictional sender-attributed reader belief content. These results do not establish deception, spontaneous anticipation, or absence of belief information.

## Primary correction-quartet comparison (N=512)

Scores are percentages; differences and saved intervals are percentage points. Arm-score intervals are unavailable and are not fabricated.

| Model | Restricted | Full | Full − restricted | Saved 95% group interval | Correction groups | Allocations |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| Llama | 3.8 | 3.0 | -0.8 | [-1.9, +0.3] | 30 | 5/5 |
| Qwen | 1.8 | 1.0 | -0.8 | [-2.0, +0.3] | 30 | 5/5 |

Intervals are copied from the saved analysis: authored correction groups are resampled with five paired allocations retained. They are conditional on this bank and allocation design. Factorial rows and allocations are not extra narratives.

### Paired allocation variability

Each row averages the retained correction groups; no allocation is an extra narrative.

| Model | Allocation | Status | Restricted | Full | Difference (pp) |
| --- | ---: | --- | ---: | ---: | ---: |
| Llama | 11 | complete | 2.5 | 3.8 | +1.2 |
| Llama | 23 | complete | 5.0 | 2.9 | -2.1 |
| Llama | 37 | complete | 3.3 | 2.5 | -0.8 |
| Llama | 53 | complete | 4.6 | 4.2 | -0.4 |
| Llama | 71 | complete | 3.3 | 1.7 | -1.7 |
| Qwen | 11 | complete | 1.7 | 0.8 | -0.8 |
| Qwen | 23 | complete | 0.8 | 1.2 | +0.4 |
| Qwen | 37 | complete | 2.9 | 0.4 | -2.5 |
| Qwen | 53 | complete | 2.9 | 1.2 | -1.7 |
| Qwen | 71 | complete | 0.8 | 1.2 | +0.4 |

## Learning curves and mechanism retention

Equal-group, equal-allocation raw complete-quartet scores; no intervals were saved for these summaries. If any allocation failed, the aggregate remains unavailable.

| Model | Method | N | Mechanism | Restricted | Full | Groups | Allocations |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| Llama | activation | 128 | observation | 47.0 | 51.6 | 30 | 5/5 |
| Llama | activation | 128 | message | 30.9 | 29.8 | 30 | 5/5 |
| Llama | activation | 128 | correction | 4.8 | 3.5 | 30 | 5/5 |
| Llama | activation | 256 | observation | 49.5 | 52.6 | 30 | 5/5 |
| Llama | activation | 256 | message | 37.5 | 35.8 | 30 | 5/5 |
| Llama | activation | 256 | correction | 4.3 | 4.7 | 30 | 5/5 |
| Llama | activation | 512 | observation | 60.8 | 59.8 | 30 | 5/5 |
| Llama | activation | 512 | message | 41.2 | 40.7 | 30 | 5/5 |
| Llama | activation | 512 | correction | 3.8 | 3.0 | 30 | 5/5 |
| Llama | tfidf | 128 | observation | 0.0 | 0.1 | 30 | 5/5 |
| Llama | tfidf | 128 | message | 0.0 | 0.0 | 30 | 5/5 |
| Llama | tfidf | 128 | correction | 0.0 | 0.0 | 30 | 5/5 |
| Llama | tfidf | 256 | observation | 0.0 | 0.1 | 30 | 5/5 |
| Llama | tfidf | 256 | message | 0.0 | 0.0 | 30 | 5/5 |
| Llama | tfidf | 256 | correction | 0.0 | 0.0 | 30 | 5/5 |
| Llama | tfidf | 512 | observation | 0.0 | 0.2 | 30 | 5/5 |
| Llama | tfidf | 512 | message | 0.0 | 0.0 | 30 | 5/5 |
| Llama | tfidf | 512 | correction | 0.0 | 0.0 | 30 | 5/5 |
| Qwen | activation | 128 | observation | 7.3 | 11.1 | 30 | 5/5 |
| Qwen | activation | 128 | message | 6.7 | 4.2 | 30 | 5/5 |
| Qwen | activation | 128 | correction | 1.6 | 0.1 | 30 | 5/5 |
| Qwen | activation | 256 | observation | 9.4 | 10.3 | 30 | 5/5 |
| Qwen | activation | 256 | message | 7.3 | 4.3 | 30 | 5/5 |
| Qwen | activation | 256 | correction | 1.3 | 0.7 | 30 | 5/5 |
| Qwen | activation | 512 | observation | 9.2 | 8.2 | 30 | 5/5 |
| Qwen | activation | 512 | message | 2.4 | 3.1 | 30 | 5/5 |
| Qwen | activation | 512 | correction | 1.8 | 1.0 | 30 | 5/5 |
| Qwen | tfidf | 128 | observation | 0.0 | 0.1 | 30 | 5/5 |
| Qwen | tfidf | 128 | message | 0.0 | 0.0 | 30 | 5/5 |
| Qwen | tfidf | 128 | correction | 0.0 | 0.0 | 30 | 5/5 |
| Qwen | tfidf | 256 | observation | 0.0 | 0.1 | 30 | 5/5 |
| Qwen | tfidf | 256 | message | 0.0 | 0.0 | 30 | 5/5 |
| Qwen | tfidf | 256 | correction | 0.0 | 0.0 | 30 | 5/5 |
| Qwen | tfidf | 512 | observation | 0.0 | 0.2 | 30 | 5/5 |
| Qwen | tfidf | 512 | message | 0.0 | 0.0 | 30 | 5/5 |
| Qwen | tfidf | 512 | correction | 0.0 | 0.0 | 30 | 5/5 |

## Comparators

Rule and output rows below use the same groups. TF-IDF appears above under its matched fitting allocations. Symbolic oracle uses source factors and is a logic ceiling, not a prose monitor. Calibrated readouts use saved development calibration at N=512, while the primary contrast above remains raw. Missing comparators prevent an advantage claim.

| Model | Comparator | Observation | Message | Correction |
| --- | --- | ---: | ---: | ---: |
| Llama | constant_0 | 0.0 | 0.0 | 0.0 |
| Llama | initial | 0.0 | 0.0 | 0.0 |
| Llama | final_update | 0.0 | 0.0 | 0.0 |
| Llama | world | 0.0 | 0.0 | 0.0 |
| Llama | symbolic_oracle | 100.0 | 100.0 | 100.0 |
| Llama | output_raw | 0.0 | 3.3 | 1.7 |
| Llama | output_calibrated | 14.6 | 5.8 | 2.9 |
| Llama | activation_restricted_calibrated_N512 | 60.6 | 42.2 | 4.0 |
| Llama | activation_full_calibrated_N512 | 55.5 | 38.9 | 3.3 |
| Llama | tfidf_restricted_calibrated_N512 | 0.0 | 0.0 | 0.0 |
| Llama | tfidf_full_calibrated_N512 | 2.2 | 0.2 | 0.0 |
| Llama | prose_monitor | 41.2 | 32.9 | 33.8 |

Llama prose-monitor invalid outputs: 44/3840; counted incorrect in quartet accuracy.
| Qwen | constant_0 | 0.0 | 0.0 | 0.0 |
| Qwen | initial | 0.0 | 0.0 | 0.0 |
| Qwen | final_update | 0.0 | 0.0 | 0.0 |
| Qwen | world | 0.0 | 0.0 | 0.0 |
| Qwen | symbolic_oracle | 100.0 | 100.0 | 100.0 |
| Qwen | output_raw | 21.7 | 9.2 | 0.0 |
| Qwen | output_calibrated | 9.2 | 9.2 | 2.1 |
| Qwen | activation_restricted_calibrated_N512 | 7.2 | 1.8 | 2.2 |
| Qwen | activation_full_calibrated_N512 | 9.6 | 4.1 | 1.9 |
| Qwen | tfidf_restricted_calibrated_N512 | 0.0 | 0.0 | 0.0 |
| Qwen | tfidf_full_calibrated_N512 | 2.2 | 0.2 | 0.0 |
| Qwen | prose_monitor | 41.2 | 32.9 | 33.8 |

Qwen prose-monitor invalid outputs: 44/3840; counted incorrect in quartet accuracy.

## Failed comparisons

Llama: 0 failed comparisons in the saved inventory.
Qwen: 0 failed comparisons in the saved inventory.

## Interpretation limits

Read the paired difference together with both absolute scores, preservation and genuine-update performance, allocation variability and comparators. An interval including zero does not demonstrate equivalence. A readout failure does not show that the model has no beliefs. Correction adds event count, length and wording changes; these results alone do not isolate an abstract composition mechanism.

This tool renders saved analysis JSON and checks its consistency; it does not reauthenticate model captures. Source file hashes are recorded in the report manifest.
