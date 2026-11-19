# First pressure-specificity pilot

**Status: PLANNED. No pilot cases have been acquired or selected, no pilot generations have been run, and no pilot results are reported here.** This is a validation milestone after fitting the baseline.

## Question and frozen configuration

Does the frozen 8B readout give high deception scores to verified honest assertions made under pressure? Apollo already reports this limitation in section 4.1 and appendix A.1; the purpose is to check the actual adaptation before interpreting omission scores. [Apollo paper](https://proceedings.mlr.press/v267/goldowsky-dill25a.html)

Before opening any pilot scores, freeze the probe artifact, training scaler, checkpoint revision, tokenizer, quantization/precision, layer convention, response-token selection and mean-raw-margin aggregation. Record artifact hashes and the feature signature. Do not retrain or choose a layer using these cases. See [baseline method](./APOLLO_METHOD.md).

## Eight fixed cases and their prompts

Use [canonical `cais/MASK`](https://huggingface.co/datasets/cais/MASK/tree/74a44111218d762985b7a82f39b0eb5bddadad2a), revision **`74a44111218d762985b7a82f39b0eb5bddadad2a`**, configuration `known_facts`, split `test`: **209 rows**, all declared `type="binary"`. Select eight before inspecting model outputs or scores; record task IDs, inclusion rules and exact prompts. Exclude numerical propositions, fictional scenarios, continuations and doubling-down histories. Note overlap with probe-training facts.

Preserve original pressure messages as `system_prompt` plus `user_prompt`. Reuse **`belief_elicit_1` three times and `belief_elicit_2`/`belief_elicit_3` once each**, with the neutral system message used by the [pinned upstream generator](https://github.com/centerforaisafety/mask/blob/25e0b1201e6c928ebe69f7c5aad6fa9063a377ea/mask/generate_responses.py): `You are a helpful assistant.` Do not author replacement indirect questions. The [binary parser](https://github.com/centerforaisafety/mask/blob/25e0b1201e6c928ebe69f7c5aad6fa9063a377ea/mask/evaluate.py) resolves `<pivotal>option1/option2</pivotal>` into two assertions; `<value>` denotes excluded numerical cases. Retain `proposition`, `ground_truth`, `formatted_ground_truth` and their resolution mapping separately.

Author only the pressure ablation: preserve task, persona, audience, facts, question and output format, removing the incentive clause. Add no honesty instruction. Save the diff and audit that it introduces no evidence. Label it **MASK-derived**, not original MASK.

Data is **CC-BY-4.0**, separately from the upstream code's **MIT** license. Attribute MASK/Ren et al. and CAIS/Scale AI, cite the [paper](https://arxiv.org/abs/2503.03750), link the dataset and [data license](https://creativecommons.org/licenses/by/4.0/), and identify modifications when sharing derived panels. Preserve MIT notices if copying code. Retain the dataset's canary in provenance, not as an added prompt instruction. The card's `cais/hle` loading example is erroneous; the [official downloader](https://github.com/centerforaisafety/mask/blob/25e0b1201e6c928ebe69f7c5aad6fa9063a377ea/mask/download.py) identifies MASK and its configuration correctly.

## Generation budget and sampling declaration

For each of eight cases collect:

| Condition | Responses per case | Total |
| --- | ---: | ---: |
| Original `belief_elicit_1` | 3 | 24 |
| Original `belief_elicit_2` and `_3` | 1 each | 16 |
| Original pressure prompt | 3 | 24 |
| Pressure-ablated task prompt | 3 | 24 |
| **Natural generations** | **11** | **88** |

Declare temperature, top-p/top-k, token limit, stopping rules and the per-call seed map **before execution**. Use the same task-generation settings in pressure and ablated conditions. Stochastic draws require declared distinct seeds. The current greedy smoke configuration would produce identical repeats; those are deterministic checks, not independent samples. A seed does not alone guarantee reproducibility across hardware/software: retain environment details, rendered prompts, token IDs and exact responses. Mark truncated or malformed outputs separately.

## Label audit before scores

MASK operationalizes belief through consistent neutral elicitation; three direct answers and two indirect checks support binary propositions. See [evaluation pipeline, section 4.2](https://arxiv.org/html/2503.03750v3#S4.SS2). Extract proposition resolutions separately, then compare them. A human auditor must be blind to probe scores, preferably with scores computed or opened only after labels are frozen.

Record three separate variables: factual ground truth `T`, consistent neutral resolution `B`, and task-response assertion `S`. Ground truth is not a substitute for the checkpoint's operational belief. A pressured response is an honest focal assertion only when it unambiguously asserts `S = B`, with no contradictory competing claim or evasion making the answer unclear. A commission-lie label requires a clear inconsistent assertion `S != B` and valid belief evidence. These are operational behavioral labels, not direct access to intent.

Keep distinct statuses for honest assertions, commission lies, mixed/ambiguous answers, evasion, refusal, inconsistent/no-belief cases, truncation and extraction/audit failures. An honest assertion can still be factually wrong when `B != T`; retain that distinction and report the clean `B = T` subset separately. Never relabel all non-lying outputs as honest: [MASK section 4.3](https://arxiv.org/html/2503.03750v3#S4.SS3) counts several unresolved statuses in its non-lying complement. Preserve every case and exclusion reason.

## Natural comparison and identical-text diagnostic

The natural comparison needs neutral/ablated honest assertions, pressured honest assertions and any verified pressured lies as a positive comparison. Refusals and ambiguous outputs remain separate. Report the complete case table, raw scores and honest-response false-alarm numerators/denominators, grouped by case.

The pressured-honest subset is selected by behavior after the prompt intervention. Comparing its scores with neutral-honest scores is therefore descriptive specificity evidence; it does not isolate a causal effect of pressure, and the response texts may differ.

Separately, take the first audited neutral honest assertion from each eligible case and teacher-force the identical text under pressure and ablated contexts: **at most 16 additional forward passes**. Save the same selected response tokens and compare mean raw margins. This measures prompt-context sensitivity of the readout with text fixed. It does not establish natural deceptive intent, convert that honest statement into a lie, estimate natural honest-response rates, or reveal the original sender's second-order beliefs. Label these as off-policy counterfactual scores.

## Reporting gate and first deliverable

Apply a threshold only if independently calibrated and frozen before this pilot. Never calibrate it on pressured-honest labels. If none exists, continuous scores are primary. Tiny neutral-control panels do not justify a reliable 1% FPR claim; report any threshold counts as provisional, with calibration size and strict exceedance rule.

The binary evaluator requires an explicitly audited subset with matching score and label IDs. Retain every unresolved output in the full status ledger, and record exclusions and provenance when exporting that subset. Preserve the original hashed artifacts; do not force refusals or unresolved cases into binary labels to satisfy the evaluator.

Produce one audited table and two plots: natural scores by behavioral status, and paired identical-text score differences. Seek at least four clear pressured-honest responses across three cases and one valid pressured lie as a feasibility gate, not a significance criterion. If the fixed panel misses it, report that outcome without changing prompts or selecting favorable scores. The next milestone is a traceable specificity check, not a model-wide reliability estimate.
