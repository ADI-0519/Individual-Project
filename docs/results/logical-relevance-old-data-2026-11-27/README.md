# Old-data logical relevance and privileged oracle substitutions

This exploratory diagnostic uses only the completed November 23 old campaign. The [scope specification](../../LOGICAL_RELEVANCE_OLD_DATA_SPEC_2026-11-27.md) was [sealed](spec_seal.json) at 2026-11-27 01:31:42 UTC before raw margins were loaded. Every factor, both relevance populations, both models/renderers, all 24 contexts, both correction strata and all 16 oracle masks were retained. The running fresh study was neither read nor changed.

Logical relevance does not generally explain away renderer-B receipt-score degradation. On the principal easy+hard population, Llama B1 relevant/irrelevant AUC is 0.7491/0.6854, whereas Qwen B1 is 0.6484/0.6760. The context comparison matters: the corresponding mean within-context AUC differences are +0.1052 and +0.0015. Pooled cross-context ordering and within-context ordering answer different questions; the Qwen pooled negative contrast does not reproduce as a negative mean context contrast.

Oracle replacements expose interactions in the external fold. They are privileged evaluations of a supplied algorithm, not a deployable repair or an intervention in the language model. Gains are neither additive nor guaranteed monotone: on Qwen B-hard, replacing only B2 truth raises row accuracy from 64.06% to 65.23% but reduces all-correct quartets from 12/192 to 3/192. Replacing true bits can remove compensating prediction errors.

## Definition and denominators

Factors are I (initial location), U (the old M1 head, first report location), B1 (first accepted receipt), and B2 (final accepted receipt). Location bits are relative to the candidate. The supplied rule is `f = 1-U if B2 else U if B1 else I`. Relevance intervenes on one true bit while holding the other true bits fixed. It differs from relevance with the other predicted bits held fixed. No factor margin was reversed, recentered or refitted; strictly positive scores predict 1 and zero predicts 0.

Each model/renderer contributes 1,536 principal correction rows and 384 complete receipt quartets; easy and hard each contribute 768 rows and 192 quartets. Every context contributes 64 principal rows, or 32 within each stratum. All nonempty relevance populations have balanced target classes. Easy B1-relevant and easy B2-irrelevant populations are necessarily empty and retain explicit null metrics. One-step and asserted controls are excluded from this specific diagnostic.

## Every factor, pooled populations

AUC is raw positive-is-class-1 AUC. Accuracy columns are percentages at the original zero threshold. Both class counts and both recalls, including null reasons, are in [all factor metrics](factor_metrics.csv).

### union

| Model | Renderer | Factor | Relevant n | Relevant AUC | Relevant accuracy % | Irrelevant n | Irrelevant AUC | Irrelevant accuracy % |
|---|---|---|---:|---:|---:|---:|---:|---:|
| llama | A | I | 384 | 1.0000 | 100.00 | 1152 | 1.0000 | 100.00 |
| llama | A | U | 1152 | 1.0000 | 99.91 | 384 | 1.0000 | 100.00 |
| llama | A | B1 | 384 | 0.9981 | 98.70 | 1152 | 0.9981 | 98.26 |
| llama | A | B2 | 1152 | 0.9985 | 97.83 | 384 | 0.9998 | 98.44 |
| llama | B | I | 384 | 0.9899 | 84.90 | 1152 | 0.9856 | 85.07 |
| llama | B | U | 1152 | 0.9903 | 87.59 | 384 | 0.9903 | 89.84 |
| llama | B | B1 | 384 | 0.7491 | 67.97 | 1152 | 0.6854 | 60.85 |
| llama | B | B2 | 1152 | 0.9162 | 78.30 | 384 | 0.9083 | 75.26 |
| qwen | A | I | 384 | 1.0000 | 100.00 | 1152 | 1.0000 | 100.00 |
| qwen | A | U | 1152 | 1.0000 | 99.91 | 384 | 1.0000 | 100.00 |
| qwen | A | B1 | 384 | 0.9863 | 92.19 | 1152 | 0.9829 | 93.06 |
| qwen | A | B2 | 1152 | 0.9786 | 88.28 | 384 | 0.9514 | 87.50 |
| qwen | B | I | 384 | 0.9997 | 93.75 | 1152 | 0.9993 | 92.36 |
| qwen | B | U | 1152 | 0.9901 | 80.99 | 384 | 0.9890 | 83.33 |
| qwen | B | B1 | 384 | 0.6484 | 50.52 | 1152 | 0.6760 | 50.17 |
| qwen | B | B2 | 1152 | 0.6858 | 57.47 | 384 | 0.6611 | 59.64 |

### correction_easy

| Model | Renderer | Factor | Relevant n | Relevant AUC | Relevant accuracy % | Irrelevant n | Irrelevant AUC | Irrelevant accuracy % |
|---|---|---|---:|---:|---:|---:|---:|---:|
| llama | A | I | 192 | 1.0000 | 100.00 | 576 | 1.0000 | 100.00 |
| llama | A | U | 576 | 1.0000 | 100.00 | 192 | 1.0000 | 100.00 |
| llama | A | B1 | 0 | null | null | 768 | 0.9988 | 98.18 |
| llama | A | B2 | 768 | 0.9989 | 98.18 | 0 | null | null |
| llama | B | I | 192 | 1.0000 | 100.00 | 576 | 1.0000 | 100.00 |
| llama | B | U | 576 | 1.0000 | 95.49 | 192 | 1.0000 | 100.00 |
| llama | B | B1 | 0 | null | null | 768 | 0.6679 | 57.81 |
| llama | B | B2 | 768 | 0.9209 | 79.17 | 0 | null | null |
| qwen | A | I | 192 | 1.0000 | 100.00 | 576 | 1.0000 | 100.00 |
| qwen | A | U | 576 | 1.0000 | 100.00 | 192 | 1.0000 | 100.00 |
| qwen | A | B1 | 0 | null | null | 768 | 0.9864 | 93.49 |
| qwen | A | B2 | 768 | 0.9811 | 89.19 | 0 | null | null |
| qwen | B | I | 192 | 1.0000 | 97.40 | 576 | 1.0000 | 97.40 |
| qwen | B | U | 576 | 0.9997 | 81.60 | 192 | 0.9992 | 85.42 |
| qwen | B | B1 | 0 | null | null | 768 | 0.6512 | 50.26 |
| qwen | B | B2 | 768 | 0.6795 | 54.95 | 0 | null | null |

### correction_hard

| Model | Renderer | Factor | Relevant n | Relevant AUC | Relevant accuracy % | Irrelevant n | Irrelevant AUC | Irrelevant accuracy % |
|---|---|---|---:|---:|---:|---:|---:|---:|
| llama | A | I | 192 | 1.0000 | 100.00 | 576 | 1.0000 | 100.00 |
| llama | A | U | 576 | 1.0000 | 99.83 | 192 | 1.0000 | 100.00 |
| llama | A | B1 | 384 | 0.9981 | 98.70 | 384 | 0.9985 | 98.44 |
| llama | A | B2 | 384 | 0.9975 | 97.14 | 384 | 0.9998 | 98.44 |
| llama | B | I | 192 | 0.9599 | 69.79 | 576 | 0.9427 | 70.14 |
| llama | B | U | 576 | 0.9693 | 79.69 | 192 | 0.9685 | 79.69 |
| llama | B | B1 | 384 | 0.7491 | 67.97 | 384 | 0.7569 | 66.93 |
| llama | B | B2 | 384 | 0.9073 | 76.56 | 384 | 0.9083 | 75.26 |
| qwen | A | I | 192 | 1.0000 | 100.00 | 576 | 1.0000 | 100.00 |
| qwen | A | U | 576 | 1.0000 | 99.83 | 192 | 1.0000 | 100.00 |
| qwen | A | B1 | 384 | 0.9863 | 92.19 | 384 | 0.9800 | 92.19 |
| qwen | A | B2 | 384 | 0.9802 | 86.46 | 384 | 0.9514 | 87.50 |
| qwen | B | I | 192 | 0.9987 | 90.10 | 576 | 0.9974 | 87.33 |
| qwen | B | U | 576 | 0.9752 | 80.38 | 192 | 0.9737 | 81.25 |
| qwen | B | B1 | 384 | 0.6484 | 50.52 | 384 | 0.7509 | 50.00 |
| qwen | B | B2 | 384 | 0.7151 | 62.50 | 384 | 0.6611 | 59.64 |

## Within-context principal contrasts

Entries are unweighted means across the 24 contexts of relevant-minus-irrelevant metrics. Every principal comparison is defined in all 24 contexts. The [full comparison table](factor_comparisons.csv) includes each context and both separate strata; [summary JSON](summary.json) records defined-context counts for every metric.

| Model | Renderer | Factor | Mean AUC difference | Mean accuracy difference (pp) |
|---|---|---|---:|---:|
| llama | A | I | 0.0000 | 0.00 |
| llama | A | U | 0.0000 | -0.09 |
| llama | A | B1 | 0.0002 | 0.43 |
| llama | A | B2 | -0.0002 | -0.61 |
| llama | B | I | 0.0033 | -0.17 |
| llama | B | U | 0.0009 | -2.26 |
| llama | B | B1 | 0.1052 | 7.12 |
| llama | B | B2 | -0.0003 | 3.04 |
| qwen | A | I | 0.0000 | 0.00 |
| qwen | A | U | 0.0000 | -0.09 |
| qwen | A | B1 | 0.0023 | -0.87 |
| qwen | A | B2 | -0.0004 | 0.78 |
| qwen | B | I | 0.0001 | 1.39 |
| qwen | B | U | -0.0004 | -2.34 |
| qwen | B | B1 | 0.0015 | 0.35 |
| qwen | B | B2 | 0.0109 | -2.17 |

## All 16 oracle substitutions

Mask bits 0, 1, 2, 3 replace I, U, B1, B2 respectively with truth. Every other bit retains its original frozen zero-threshold prediction. Tables show row accuracy / complete-quartet all-correct percentage. Mask 0 is the unchanged external fold; mask 15 reproduces formal truth exactly. [All oracle metrics](oracle_metrics.csv) additionally retain every context, all five quartet correctness counts and changes from mask 0.

### llama: union

| Mask | True factors supplied | A row / quartet % | B row / quartet % |
|---:|---|---:|---:|
| 0 | none | 97.98 / 92.71 | 69.21 / 26.56 |
| 1 | I | 97.98 / 92.71 | 71.74 / 27.86 |
| 2 | U | 98.05 / 92.97 | 71.68 / 29.95 |
| 3 | I+U | 98.05 / 92.97 | 74.22 / 32.55 |
| 4 | B1 | 98.31 / 93.75 | 73.37 / 35.16 |
| 5 | I+B1 | 98.31 / 93.75 | 78.12 / 40.36 |
| 6 | U+B1 | 98.37 / 94.01 | 78.97 / 41.93 |
| 7 | I+U+B1 | 98.37 / 94.01 | 83.72 / 50.26 |
| 8 | B2 | 99.61 / 98.44 | 83.98 / 57.81 |
| 9 | I+B2 | 99.61 / 98.44 | 85.35 / 59.90 |
| 10 | U+B2 | 99.67 / 98.70 | 90.62 / 63.28 |
| 11 | I+U+B2 | 99.67 / 98.70 | 91.99 / 67.97 |
| 12 | B1+B2 | 99.93 / 99.74 | 86.91 / 71.61 |
| 13 | I+B1+B2 | 99.93 / 99.74 | 90.69 / 81.77 |
| 14 | U+B1+B2 | 100.00 / 100.00 | 96.22 / 84.90 |
| 15 | I+U+B1+B2 | 100.00 / 100.00 | 100.00 / 100.00 |

### llama: correction_easy

| Mask | True factors supplied | A row / quartet % | B row / quartet % |
|---:|---|---:|---:|
| 0 | none | 98.18 / 93.75 | 77.86 / 44.27 |
| 1 | I | 98.18 / 93.75 | 77.86 / 44.27 |
| 2 | U | 98.18 / 93.75 | 79.17 / 47.40 |
| 3 | I+U | 98.18 / 93.75 | 79.17 / 47.40 |
| 4 | B1 | 98.18 / 93.75 | 76.04 / 40.10 |
| 5 | I+B1 | 98.18 / 93.75 | 76.04 / 40.10 |
| 6 | U+B1 | 98.18 / 93.75 | 79.17 / 47.40 |
| 7 | I+U+B1 | 98.18 / 93.75 | 79.17 / 47.40 |
| 8 | B2 | 100.00 / 100.00 | 98.44 / 94.79 |
| 9 | I+B2 | 100.00 / 100.00 | 98.44 / 94.79 |
| 10 | U+B2 | 100.00 / 100.00 | 100.00 / 100.00 |
| 11 | I+U+B2 | 100.00 / 100.00 | 100.00 / 100.00 |
| 12 | B1+B2 | 100.00 / 100.00 | 96.61 / 88.54 |
| 13 | I+B1+B2 | 100.00 / 100.00 | 96.61 / 88.54 |
| 14 | U+B1+B2 | 100.00 / 100.00 | 100.00 / 100.00 |
| 15 | I+U+B1+B2 | 100.00 / 100.00 | 100.00 / 100.00 |

### llama: correction_hard

| Mask | True factors supplied | A row / quartet % | B row / quartet % |
|---:|---|---:|---:|
| 0 | none | 97.79 / 91.67 | 60.55 / 8.85 |
| 1 | I | 97.79 / 91.67 | 65.62 / 11.46 |
| 2 | U | 97.92 / 92.19 | 64.19 / 12.50 |
| 3 | I+U | 97.92 / 92.19 | 69.27 / 17.71 |
| 4 | B1 | 98.44 / 93.75 | 70.70 / 30.21 |
| 5 | I+B1 | 98.44 / 93.75 | 80.21 / 40.62 |
| 6 | U+B1 | 98.57 / 94.27 | 78.78 / 36.46 |
| 7 | I+U+B1 | 98.57 / 94.27 | 88.28 / 53.12 |
| 8 | B2 | 99.22 / 96.88 | 69.53 / 20.83 |
| 9 | I+B2 | 99.22 / 96.88 | 72.27 / 25.00 |
| 10 | U+B2 | 99.35 / 97.40 | 81.25 / 26.56 |
| 11 | I+U+B2 | 99.35 / 97.40 | 83.98 / 35.94 |
| 12 | B1+B2 | 99.87 / 99.48 | 77.21 / 54.69 |
| 13 | I+B1+B2 | 99.87 / 99.48 | 84.77 / 75.00 |
| 14 | U+B1+B2 | 100.00 / 100.00 | 92.45 / 69.79 |
| 15 | I+U+B1+B2 | 100.00 / 100.00 | 100.00 / 100.00 |

### qwen: union

| Mask | True factors supplied | A row / quartet % | B row / quartet % |
|---:|---|---:|---:|
| 0 | none | 89.71 / 66.41 | 59.90 / 5.21 |
| 1 | I | 89.71 / 66.41 | 59.90 / 5.21 |
| 2 | U | 89.78 / 66.67 | 62.17 / 5.73 |
| 3 | I+U | 89.78 / 66.67 | 62.17 / 5.73 |
| 4 | B1 | 91.15 / 71.88 | 63.41 / 10.16 |
| 5 | I+B1 | 91.15 / 71.88 | 64.00 / 10.94 |
| 6 | U+B1 | 91.21 / 72.14 | 67.51 / 14.06 |
| 7 | I+U+B1 | 91.21 / 72.14 | 68.10 / 14.84 |
| 8 | B2 | 97.98 / 91.93 | 73.89 / 38.80 |
| 9 | I+B2 | 97.98 / 91.93 | 73.89 / 38.80 |
| 10 | U+B2 | 98.05 / 92.19 | 87.63 / 50.52 |
| 11 | I+U+B2 | 98.05 / 92.19 | 87.63 / 50.52 |
| 12 | B1+B2 | 99.93 / 99.74 | 84.18 / 70.83 |
| 13 | I+B1+B2 | 99.93 / 99.74 | 85.74 / 77.08 |
| 14 | U+B1+B2 | 100.00 / 100.00 | 98.44 / 93.75 |
| 15 | I+U+B1+B2 | 100.00 / 100.00 | 100.00 / 100.00 |

### qwen: correction_easy

| Mask | True factors supplied | A row / quartet % | B row / quartet % |
|---:|---|---:|---:|
| 0 | none | 89.19 / 71.35 | 55.73 / 4.17 |
| 1 | I | 89.19 / 71.35 | 55.73 / 4.17 |
| 2 | U | 89.19 / 71.35 | 54.95 / 4.69 |
| 3 | I+U | 89.19 / 71.35 | 54.95 / 4.69 |
| 4 | B1 | 89.19 / 71.35 | 55.99 / 4.17 |
| 5 | I+B1 | 89.19 / 71.35 | 55.99 / 4.17 |
| 6 | U+B1 | 89.19 / 71.35 | 54.95 / 4.69 |
| 7 | I+U+B1 | 89.19 / 71.35 | 54.95 / 4.69 |
| 8 | B2 | 100.00 / 100.00 | 82.55 / 76.04 |
| 9 | I+B2 | 100.00 / 100.00 | 82.55 / 76.04 |
| 10 | U+B2 | 100.00 / 100.00 | 100.00 / 100.00 |
| 11 | I+U+B2 | 100.00 / 100.00 | 100.00 / 100.00 |
| 12 | B1+B2 | 100.00 / 100.00 | 85.55 / 73.44 |
| 13 | I+B1+B2 | 100.00 / 100.00 | 86.20 / 76.04 |
| 14 | U+B1+B2 | 100.00 / 100.00 | 99.35 / 97.40 |
| 15 | I+U+B1+B2 | 100.00 / 100.00 | 100.00 / 100.00 |

### qwen: correction_hard

| Mask | True factors supplied | A row / quartet % | B row / quartet % |
|---:|---|---:|---:|
| 0 | none | 90.23 / 61.46 | 64.06 / 6.25 |
| 1 | I | 90.23 / 61.46 | 64.06 / 6.25 |
| 2 | U | 90.36 / 61.98 | 69.40 / 6.77 |
| 3 | I+U | 90.36 / 61.98 | 69.40 / 6.77 |
| 4 | B1 | 93.10 / 72.40 | 70.83 / 16.15 |
| 5 | I+B1 | 93.10 / 72.40 | 72.01 / 17.71 |
| 6 | U+B1 | 93.23 / 72.92 | 80.08 / 23.44 |
| 7 | I+U+B1 | 93.23 / 72.92 | 81.25 / 25.00 |
| 8 | B2 | 95.96 / 83.85 | 65.23 / 1.56 |
| 9 | I+B2 | 95.96 / 83.85 | 65.23 / 1.56 |
| 10 | U+B2 | 96.09 / 84.38 | 75.26 / 1.04 |
| 11 | I+U+B2 | 96.09 / 84.38 | 75.26 / 1.04 |
| 12 | B1+B2 | 99.87 / 99.48 | 82.81 / 68.23 |
| 13 | I+B1+B2 | 99.87 / 99.48 | 85.29 / 78.12 |
| 14 | U+B1+B2 | 100.00 / 100.00 | 97.53 / 90.10 |
| 15 | I+U+B1+B2 | 100.00 / 100.00 | 100.00 / 100.00 |

## Relation to internal routing claims

Prakash et al. distinguish a pointer from its retrieved payload: exchanging the pointer retrieves the recipient story's indexed state rather than copying the donor answer. Their high-level algorithm is tested against intervened model outputs, with 80 behaviorally correct cases selected for the mechanism experiments. That provides causal evidence that this frozen-score diagnostic does not supply. See [v3, sections 3.1, 3.3 and 5.1](https://arxiv.org/html/2505.14685v3#S5.SS1).

Steele et al. explicitly allow frame information to be linearly decodable at a value token while the value-controlling subspace remains frame-agnostic. Their routing claim uses recipient-value versus donor-color controls and clean-pass projection ablations, with preservation-control and sample-sensitivity limits. The present truth-conditioned partition and oracle fold have no corresponding internal interventions. See [v2, section 4.4, PDF pages 9-11, and section 7, page 13](https://arxiv.org/pdf/2607.11945v2#page=9).

Our inference is limited to external-readout error structure under the given Boolean rule. Neither successful factor decoding nor successful oracle composition establishes that the model internally uses, gates, routes or combines those factors. The diagnostic cannot distinguish a misplaced readout, correlated errors, missing internal computation or a different internal algorithm. [Reading coverage](source_coverage.json) identifies the versions and sections actually inspected.

## Validation and artifacts

Thirteen synthetic tests pass, including exhaustive fold truth and relevance, all masks, strict zero threshold, undefined metrics, exact identity joins and complete quartet checks. Ruff passes. The analysis verifies all 4,992 saved test rows per model before selecting the declared 3,072 correction rows, checks old completion-manifest hashes, exact factor-head target/site mapping, saved zero-threshold predictions and the frozen C2 fold. All input hashes are rechecked after analysis. No feature tensor, new model fit, threshold search or model call was used.

An independent implementation checked all 2,400 factor rows, 1,200 comparison rows and 4,800 oracle rows: 77,600 numeric scalars and 10,400 explicit null cells agreed, with maximum absolute difference 2.22e-16. Its [review receipt](independent_review_receipt.json) records the scope and limitations.

- [Analysis receipt and hashes](analysis_receipt.json)
- [Independent reconstruction receipt](independent_review_receipt.json)
- [Specification seal](spec_seal.json)
- [All factor populations and contexts](factor_metrics.csv)
- [All paired population differences](factor_comparisons.csv)
- [All oracle conditions and contexts](oracle_metrics.csv)
- [Structured pooled results and context means](summary.json)

This is a post-hoc hypothesis about an old dataset, prospectively bounded before its new raw-margin analysis. The relevance populations and oracle truth are privileged. Fresh-study selection, capture and reporting remain unchanged regardless of these results.
