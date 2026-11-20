# Reader decisions follow values attached to the wrong attribute

20 November 2026. Completed **reader-only failure analysis**, prospectively frozen at commit `17e8df762900e94576c3294ff36c749ba7d5ac1a`. This is an appendix to the [communication results](COMMUNICATION_TRANSFER_RESULTS.md), not a passed reader qualification or a new latent-probe result. The original failed gates, reports and policies remain unchanged.

**Qwen answered all 32 decisive-evidence controls correctly, yet deferred on none of the 32 metadata-only conditions.** Changing an irrelevant administrative value from low to high changed the definite decision in the direction justified by that value *if it had been decisive evidence* in all eight date pairs and six of eight score pairs. Explicitly stating that the decisive attribute was missing did not change the date responses. For scores it produced DEFER only on the high-value side, leaving every low-value response as REJECT.

**A subsequent fixed-message Llama replication differed substantially and failed its own decisive controls: 19/32 correct.** It deferred on every score-metadata condition and every explicit-missing condition, while date metadata usually produced a constant APPROVE. The controlled Qwen value-following pattern is therefore **model-specific in the current evidence**, rather than a demonstrated general failure across models. Full results from both runs appear below.

This is a clear, controlled behavioral failure in these authored cases. It is consistent with using irrelevant values as decision evidence; it does not establish an internal substitution mechanism, deceptive intent, a general property of language models or priority over previous research.

## What was held fixed and what changed

The [protocol](EVIDENCE_ROLE_DIAGNOSTIC_PROTOCOL.md) and [16-case input bank](../data/communication/evidence_role_v1.json) specified one 96-call batch: eight date cases and eight score cases, each with low/high values in three evidence roles. The [runner](../scripts/run_evidence_role_diagnostic.py) retained the existing Qwen2.5-7B-Instruct checkpoint, NF4 weights with BF16 compute, nine worked examples, strict EOS parser, greedy decoding, seed 42 and eight-token output cap. No policy, budget, example or checkpoint search followed these outputs.

Each case states a decision boundary. In the decisive condition, the supplied value describes the completion date or evaluation score, making APPROVE or REJECT warranted. In the metadata condition, the same value instead describes a generic administrative registration date or file-ID number. Neither determines the requested attribute, so both low and high require DEFER. The explicit-missing condition retains that metadata and adds a sentence saying that the completion date or evaluation score is not provided.

There is no hidden source or hidden-world answer to guess. For either metadata value, two compatible worlds permit opposite decisions because the completion date or score remains unconstrained. An unwarranted definite answer is therefore an evidence-use error; it is not necessarily false in whichever world one might imagine.

## Qwen: all condition groups

Each row below contains eight calls, one per case in that family. The two value sides were authored before outputs. Date comparisons and score comparisons have opposite decision directions, so the results are shown separately.

| Family | Evidence role | Value side | Warranted answer | APPROVE | REJECT | DEFER | INVALID | Correct |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Date | Decisive completion date | Low | APPROVE | 8 | 0 | 0 | 0 | 8/8 |
| Date | Decisive completion date | High | REJECT | 0 | 8 | 0 | 0 | 8/8 |
| Date | Registration metadata | Low | DEFER | 8 | 0 | 0 | 0 | 0/8 |
| Date | Registration metadata | High | DEFER | 0 | 8 | 0 | 0 | 0/8 |
| Date | Metadata + explicit missingness | Low | DEFER | 8 | 0 | 0 | 0 | 0/8 |
| Date | Metadata + explicit missingness | High | DEFER | 0 | 8 | 0 | 0 | 0/8 |
| Score | Decisive evaluation score | Low | REJECT | 0 | 8 | 0 | 0 | 8/8 |
| Score | Decisive evaluation score | High | APPROVE | 8 | 0 | 0 | 0 | 8/8 |
| Score | File-ID metadata | Low | DEFER | 0 | 8 | 0 | 0 | 0/8 |
| Score | File-ID metadata | High | DEFER | 6 | 2 | 0 | 0 | 0/8 |
| Score | Metadata + explicit missingness | Low | DEFER | 0 | 8 | 0 | 0 | 0/8 |
| Score | Metadata + explicit missingness | High | DEFER | 0 | 0 | 8 | 0 | 8/8 |

Thus the six family-by-role groups have normative accuracy **16/16, 0/16, 0/16** for date decisive/metadata/explicit-missing conditions, and **16/16, 0/16, 8/16** for the corresponding score conditions. Across the 64 insufficient-evidence calls, there were **56 unwarranted definite answers and eight correct deferrals**. This mixed-condition total is descriptive; the authored mixture is not a prevalence estimate for real communication.

All 96 completions reached EOS and were exactly one of the three permitted labels. There were **zero capped or invalid outputs**, and no exclusions or parser relaxations. This failure cannot be attributed to a truncated answer in this run.

## Qwen: every paired case

The following tables include all 16 cases, including the two score cases that did not follow the irrelevant value. Each answer cell is **low → high**. `A` = APPROVE, `R` = REJECT, `D` = DEFER. The warranted pair is A→R for decisive dates, R→A for decisive scores, and D→D for both metadata roles in every case.

| Date case | Entity suffix | Low / deadline / high | Decisive | Metadata | Explicit missing |
| --- | --- | --- | --- | --- | --- |
| er_date_01 | Aster | 2022-01-13 / 2022-01-15 / 2022-01-17 | A→R | A→R | A→R |
| er_date_02 | Birch | 2022-02-13 / 2022-02-15 / 2022-02-17 | A→R | A→R | A→R |
| er_date_03 | Clover | 2022-03-13 / 2022-03-15 / 2022-03-17 | A→R | A→R | A→R |
| er_date_04 | Dune | 2022-04-13 / 2022-04-15 / 2022-04-17 | A→R | A→R | A→R |
| er_date_05 | Elder | 2022-05-13 / 2022-05-15 / 2022-05-17 | A→R | A→R | A→R |
| er_date_06 | Flint | 2022-06-13 / 2022-06-15 / 2022-06-17 | A→R | A→R | A→R |
| er_date_07 | Grove | 2022-07-13 / 2022-07-15 / 2022-07-17 | A→R | A→R | A→R |
| er_date_08 | Heath | 2022-08-13 / 2022-08-15 / 2022-08-17 | A→R | A→R | A→R |

| Score case | Entity suffix | Low / minimum / high | Decisive | Metadata | Explicit missing |
| --- | --- | --- | --- | --- | --- |
| er_score_01 | Aster | 900199 / 900230 / 900267 | R→A | R→R | R→D |
| er_score_02 | Birch | 900410 / 900441 / 900478 | R→A | R→A | R→D |
| er_score_03 | Clover | 900621 / 900652 / 900689 | R→A | R→A | R→D |
| er_score_04 | Dune | 900832 / 900863 / 900900 | R→A | R→R | R→D |
| er_score_05 | Elder | 901043 / 901074 / 901111 | R→A | R→A | R→D |
| er_score_06 | Flint | 901254 / 901285 / 901322 | R→A | R→A | R→D |
| er_score_07 | Grove | 901465 / 901496 / 901533 | R→A | R→A | R→D |
| er_score_08 | Heath | 901676 / 901707 / 901744 | R→A | R→A | R→D |

Metadata produced definite low/high flips that matched the decisive direction in **8/8 date cases and 6/8 score cases**. The two R→R score pairs establish non-abstention, but do not establish value-following. After explicit missingness, all eight date cases still matched the decisive direction. No score case did: all eight became R→D, an asymmetric partial improvement rather than two correct deferrals. **No case in either metadata role had both low and high correctly answered DEFER.**

The frozen case-level bootstrap for explicit-missing minus metadata deferral rate is **0.00, 95% percentile interval [0.00, 0.00]** for dates, and **+0.50, interval [0.50, 0.50]** for scores, each from eight case clusters and 10,000 resamples. Every case within a family has the same improvement, hence the degenerate intervals. These express no variation among these eight authored instances; they do not imply zero uncertainty on new templates, models or tasks.

## A concrete observed trace

The first date case, **Evidence Date Aster**, requires completion on or before **2022-01-15**. The saved actual user payload contains this metadata sentence in the low condition:

> The administrative file registration date for Evidence Date Aster is 2022-01-13.

The response was **APPROVE**. Changing only that registration value to **2022-01-17** produced **REJECT**. Both require DEFER: neither states when completion occurred, and no relation between registration and completion is given. Adding the exact sentence **“The completion date for Evidence Date Aster is not provided.”** left those responses unchanged. Supplying actual completion dates instead produced the same A→R pair, correctly. This trace is the first authored date case and represents the pattern in all eight, rather than an exceptional selected success.

For contrast, **Evidence Score Aster** answered R→R from the low/high file IDs, despite correct R→A answers for actual evaluation scores. **Evidence Score Birch** did answer R→A from file IDs. Both became R→D when explicitly told that the score was absent. Reporting both preserves the distinction between a constant definite fallback and value-sensitive misuse.

## Interpretation against the primary literature

[Shi et al., ICML 2023](https://proceedings.mlr.press/v202/shi23a.html), already show that irrelevant sentences, entity overlap and numerical distractors can change solutions to answerable arithmetic. [Mirzadeh et al., GSM-Symbolic version 2](https://arxiv.org/html/2410.05229v2), already document plausible but irrelevant quantities being used as operations in GSM-NoOp. [Joren et al., Sufficient Context version 3](https://arxiv.org/html/2411.06037v3), already distinguish relevant from sufficient evidence, document failures to abstain, and explicitly reject inferring an unstated relation from a related attribute. Exact methods/limitations reading coverage is recorded in the [literature audit](EVIDENCE_ROLE_LITERATURE.md).

This experiment combines an absent decisive attribute with a controlled low/high role contrast under a fixed three-way decision policy. The decisive controls and opposing date/score directions make the observed failure more specific than simply “the model guessed.” The measured contribution at this stage is a **bounded replication and refinement of established evidence-use failures**, supporting diagnosis of our failed reader. The evidence does not justify a claim of the first such benchmark or a newly identified internal mechanism.

## What the experiment does not resolve

- **Generalization:** the Qwen discovery uses two grammars, eight instantiations each, one model, one quantized runtime and one fixed policy. The subsequent Llama run reuses the same cases and does not reproduce the same pattern; it also fails decisive controls. There are 16 case clusters, not 96 independent scenarios or 16 independent domains.
- **A causal date-versus-score effect:** the families differ in grammar, comparison operator and worked-example coverage; the existing examples contain no deadline case. Their response difference cannot be assigned solely to data type.
- **The effect of adding any metadata:** there is no otherwise identical metadata-free condition. We can observe value-sensitive changes within metadata pairs, but cannot estimate a distraction increase over an empty-evidence baseline.
- **A unique missingness mechanism:** the added sentence changes length, negation, salience and an epistemic cue. Its one-sided score benefit is an observed behavior, not a validated general repair.
- **Confidence or deception:** categorical completions provide no confidence calibration, entropy measurement, sender intention or second-order belief evidence. This batch contains no natural sender reports and fits no probes.
- **Reader qualification:** the fresh nine-shot gate remains **FAIL: 24/24 informed, 17/24 absent, 24/24 stale**, below the required absent-evidence threshold. This diagnostic does not rerun or replace that gate. There were no new dependent training labels, probe fits or calls on the untouched recipient-transfer bank.

## Provenance and verification

The [independently curated evidence](validation/2026-11-20-evidence-role-diagnostic.json) preserves every actual user payload, raw answer, grouped count and paired pattern, with provenance. Its separate implementation reconstructed the numeric summaries, pairs and bootstrap results without importing the project's analysis helper; all matched. It also verified nine source hashes, 96 record files, and every policy, prompt-preflight and final-user binding.

Native output directory: `/home/adirj/kcl-latent-probes-runs/evidence_role_diagnostic_v1`. Manifest status is **complete**, with `source_unchanged: true`. In a further read for this report, all 96 saved messages matched the frozen protocol and every generation's prompt IDs matched preflight. All raw user payloads and completions, all six grouped outcomes and all 16 paired summaries were inspected; no partial-run result was used.

| Artifact | SHA-256 |
| --- | --- |
| Frozen protocol | `6f630f2ab2e194e808e9220b154b68dd5349607314c11e2b14693e61762e8778` |
| Terminal manifest | `cb77fe2a3d26512765367e1f9e418b6bb92201f8f8b72a6300622c5bf47601d1` |
| Complete rows | `3d59b54ea6a8a4042a573fc6104a8742ca018e49f47e36ca93cf0cc135e345f6` |
| Complete summary | `88d0067edaa19aab96909a0e324120a6b6cdcf895a28c0a62ff93e6e1512dec1` |
| Input bank | `a162f02af9ee87e880dc939bd7601968401b449d2668966e1a0ce185dae46fd1` |
| Reader policy | `6dda261f5d5c7667ea29885bddd1c47410f34d59e1807819a317008da570c779` |
| Independent curation | `c486ce39a1fd7e3ee434cc6f2c1e17a359d2bf37ec19db98adacc39168e781c6` |

Model revision: `a09a35458c702b33eeacc393d103063234e8bc28`. Actual complete prompts were 1,023–1,044 tokens, below the 2,048-token limit without truncation. Generation records total **98,924 prompt tokens and 326 generated tokens including EOS**, with **36.87 seconds summed generation time**; terminal generation-stage wall time was approximately **39.24 seconds**, excluding model loading. Those costs describe this local run, not HPC requirements.

## Llama: completed replication after the Qwen discovery

The [separate replication protocol](EVIDENCE_ROLE_REPLICATION_PROTOCOL.md) was frozen at commit `d0d0bd5892f919f81af452b7b20163321f0bbe3e` **after observing Qwen's results and before generating Llama's outputs**. The [replication runner](../scripts/replicate_evidence_role.py) used Llama-3.1-8B-Instruct revision `0e9e39f249a16976918f6564b8830bc894c89659`, NF4 weights with BF16 compute, the exact same 96 message sequences, greedy decoding, seed 42, strict EOS parsing and the eight-token cap. Llama used its own native tokenizer and chat template. This is a post-discovery model replication on the same cases, not an originally preregistered two-model study, fresh held-out content or a precision-controlled experiment.

| Family / role | Qwen correct | Llama correct | Llama APPROVE | Llama REJECT | Llama DEFER | Llama INVALID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Date / decisive | 16/16 | 11/16 | 13 | 3 | 0 | 0 |
| Date / metadata | 0/16 | 0/16 | 15 | 1 | 0 | 0 |
| Date / explicit missing | 0/16 | 16/16 | 0 | 0 | 16 | 0 |
| Score / decisive | 16/16 | 8/16 | 16 | 0 | 0 | 0 |
| Score / metadata | 0/16 | 16/16 | 0 | 0 | 16 | 0 |
| Score / explicit missing | 8/16 | 16/16 | 0 | 0 | 16 | 0 |

Llama's date controls approved all eight low dates correctly, but approved five of the eight high completion dates incorrectly. Its score controls answered APPROVE for all 16 values, missing every below-threshold case. Consequently, **19/32 decisive controls were correct, with only three of 16 case pairs having both controls correct**. Llama is not a comparison-capable alternative reader under this fixed policy simply because it abstains on more insufficient-evidence conditions.

Every Llama pair is shown below with the same A/R/D legend and low→high orientation as the Qwen tables. Values and boundaries are identical to the corresponding Qwen cases above.

| Case | Decisive | Metadata | Explicit missing |
| --- | --- | --- | --- |
| er_date_01 | A→A | A→A | D→D |
| er_date_02 | A→A | A→A | D→D |
| er_date_03 | A→A | A→A | D→D |
| er_date_04 | A→A | A→A | D→D |
| er_date_05 | A→A | A→A | D→D |
| er_date_06 | A→R | A→A | D→D |
| er_date_07 | A→R | A→A | D→D |
| er_date_08 | A→R | A→R | D→D |
| er_score_01 | A→A | D→D | D→D |
| er_score_02 | A→A | D→D | D→D |
| er_score_03 | A→A | D→D | D→D |
| er_score_04 | A→A | D→D | D→D |
| er_score_05 | A→A | D→D | D→D |
| er_score_06 | A→A | D→D | D→D |
| er_score_07 | A→A | D→D | D→D |
| er_score_08 | A→A | D→D | D→D |

Only **1/8 Llama date metadata pairs** followed the irrelevant value in the decisive direction, compared with Qwen's 8/8. No score metadata pair did, compared with Qwen's 6/8. Llama correctly deferred on both sides in all eight score metadata pairs and all 16 explicit-missing pairs; it did so in none of the date metadata pairs. Explicit missingness increased Llama's date deferral rate by **1.00, case-bootstrap interval [1.00, 1.00]**, and score deferral rate by **0.00, interval [0.00, 0.00]**. As for Qwen, identical within-family effects make these conditional intervals degenerate, not broadly certain.

All **96 Llama outputs reached EOS**, with zero invalid or capped completions. All 96 message sequences matched the Qwen discovery and the frozen Llama plan byte-for-byte; all generated prompt IDs matched Llama's own preflight. Complete prompts were 939–960 tokens without truncation. Records total **90,860 prompt tokens, 332 generated tokens including EOS, and 35.82 seconds summed generation time**; generation-stage wall time was approximately **38.17 seconds**, excluding model loading. Terminal manifest status is complete, with discovery and source hashes unchanged.

Native directory: `/home/adirj/kcl-latent-probes-runs/evidence_role_llama_replication_v1`.

| Llama artifact | SHA-256 |
| --- | --- |
| Frozen protocol | `7de4f3fd1aa137d83502989565c127db0abfc764ffbe02ab4e689f62f252acc4` |
| Terminal manifest | `0b7071a8bc8c6a4c6d14fb7b7c9f6aabf4d46b54e0421e094a1c4cc13b19baaf` |
| Complete rows | `cbec03ae16f7a8043bb0849b6f0578e4fa66808f25dcb290d405c5027b4a40aa` |
| Complete summary | `a21dc5e65c039cadaa41ee1016cac278da6acd305e6b3f95f63dd7792601626d` |
| Independent Llama curation | `966d336f314e2f8dc14ceb002858efb285a66a4c65e79009421b220a335a53b5` |

The [independent Llama record](validation/2026-11-20-evidence-role-llama-replication.json)
reconstructs all summaries and pairs without importing the analysis helper. It
verifies all 96 shared message sequences, model-specific prompt IDs and records,
and the 101 unchanged Qwen discovery files. Both native/mirrored runs are
byte-identical. The [combined figure](figures/2026-11-20/evidence_role_diagnostic.pdf)
shows both models' decisive failures, warranted deferrals and paired switches.

The replication narrows the finding: the Qwen configuration exhibits the specific value-following failure despite passing decisive controls; the Llama configuration exhibits a different mixture of definite-answer bias, comparison failures and appropriate abstention. It does **not** establish a shared internal mechanism, demonstrate Qwen's pattern across models, or show that Llama is a qualified replacement. The fixed-policy differences cannot isolate architecture, training, tokenization, generation defaults or quantization as their cause.

Both runs belong in the failure-analysis appendix. No new reports, dependent labels, probe fits, policy changes, qualification passes or untouched recipient-transfer calls followed either diagnostic. The unsupported pressure-transfer pipeline remains stopped. BF16 or CREATE access is not established as a solution by these measurements.
