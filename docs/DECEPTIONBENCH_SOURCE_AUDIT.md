# DeceptionBench: pinned source and release audit

**21 November 2026. Read-only source inspection; no upstream code execution, model inference, paid requests or binary activations downloaded.** This audit concerns Huang et al., *DeceptionBench: A Comprehensive Benchmark for AI Deception Behaviors in Real-world Scenarios*, arXiv:2510.15501. It does **not** concern the different PKU-Alignment/DeceptionBench project. This document reports source facts and their methodological implications; it reports no new model results.

## Exact sources inspected

- [Paper v1, methods Sections 2.1–2.4 and experimental settings](https://arxiv.org/html/2510.15501v1).
- [Aries-iai/DeceptionBench repository](https://github.com/Aries-iai/DeceptionBench/tree/8f3175f66fdff89ded3ebdf0ee2055fa47c955db), pinned commit `8f3175f66fdff89ded3ebdf0ee2055fa47c955db`, committed 30 May 2025.
- [Officially linked dataset](https://huggingface.co/datasets/skyai798/DeceptionBench/tree/38b412da565073483c0a68ce9e04eb15859beedc), pinned revision `38b412da565073483c0a68ce9e04eb15859beedc`, last modified 26 November 2025.

GitHub's REST repository/commit/tree endpoints and Hugging Face's dataset endpoint were inspected, followed by every published domain JSON and the raw Python modules named below. No credentials were requested or displayed. Counts below were calculated from the actual five JSON files, rather than inferred from the abstract. Paper-method descriptions here are explicitly v1-specific; this is not a claim that v1 is the latest paper revision. Later-paper human-validation additions are separate from the released files inspected here.

## What is actually released

The dataset release contains five domain JSON files plus its card and Git attributes. Each file contains 30 scenario rows. Each domain has three subtypes with ten rows each. The result is **150 scenario rows, 15 subtypes, six nonempty prompt variants per row, and 900 distinct static prompts**. There are no generated target responses, binary judgment labels, token traces or cached activations in either inspected release tree. The paper's greater-than-1,000-sample description includes a broader evaluation design; it must not be represented as more than 1,000 released independent static scenarios. [Pinned dataset files](https://huggingface.co/datasets/skyai798/DeceptionBench/tree/38b412da565073483c0a68ce9e04eb15859beedc), [paper Section 2.2.3](https://arxiv.org/html/2510.15501v1#S2.SS2.SSS3).

Each scenario has exactly 11 populated fields:

```text
topic, dimension, question, groundtruth, goal,
L1-self, L1-other,
L2-self-pressure, L2-self-reward,
L2-other-pressure, L2-other-reward
```

There are no L3 static prompt fields. Four L3 conditions begin from corresponding L2 prompts and generate subsequent inducements. The code's default five-turn loop potentially generates five target responses and four auxiliary instructions per scenario/role/inducement. These are dynamically generated, response-dependent observations, not another independent case bank. [Multi-turn implementation](https://github.com/Aries-iai/DeceptionBench/blob/8f3175f66fdff89ded3ebdf0ee2055fa47c955db/processor/multi_turn.py), [auxiliary prompt generator](https://github.com/Aries-iai/DeceptionBench/blob/8f3175f66fdff89ded3ebdf0ee2055fa47c955db/processor/followup.py).

| File | Bytes | Raw-file SHA256 |
| --- | ---: | --- |
| economy.json | 110,231 | `1e0786966b09bc8801097e20ceecb33960aaeb253d2a94a67ec1eec677bb5d88` |
| education.json | 52,984 | `1619c24f315bd76de37ecfa190d6a8da5c4a41bc06fdae2738ad136a1a3bba62` |
| entertainment.json | 104,317 | `4f788de99ae9346ce6214b1dd8fdb28de1f81f8b897566f66ad05a7e18b6f2f4` |
| healthcare.json | 127,395 | `2d80b989cd26c53795a877d282b3afd4220c2baf7ad2bcd19f6eba7446759b23` |
| social.json | 70,324 | `4924b22d258081cafd35cf66f0fd345aeef1fa9e66742ac33def465ecc21e6e8` |

These hashes were calculated over downloaded bytes using .NET SHA256, without rewriting or normalizing JSON. A first attempted HttpClient hashing command failed because its assembly was unavailable; those null results were discarded. The replacement WebClient byte-download command succeeded for all five files. No raw-source download is an experimental model call.

## Paired provenance and what the contrast means

The six variants are co-resident in a scenario row and share its question, reference honest behavior and stated deceptive objective. That supports grouping all variants under one scenario ID and comparing their observed behavior. None of the 600 L1/L2 role-matched comparisons is byte-identical. The release has no explicit immutable scenario ID; our importer should derive one from pinned file, row position and source-record hash, then retain all three.

**L1 means absence of the additional pressure/reward manipulation; it does not mean absence of deceptive intent in the prompt.** For example, the first economy L1 scenario already requests a nonexistent differentiating feature; the next emphasizes withholding a car defect. L2 rewrites context and adds inducements. These are not automatically minimal pressure-only causal interventions, and L1 must never be assigned an honest response label just because its key starts with L1. [Pinned economy source](https://huggingface.co/datasets/skyai798/DeceptionBench/blob/38b412da565073483c0a68ce9e04eb15859beedc/economy.json).

Each actual generated response needs its own behavior judgment. Scenario `groundtruth` is a natural-language honest-answer/action reference, and `goal` describes the deceptive target. Neither is a released binary label for a target model's response. In addition, some references express normative refusals or disallowed actions rather than a binary proposition about an audience's false belief. This bank therefore cannot directly supervise the existing location-belief head, establish second-order belief, or establish deception from prompt metadata alone. [Pinned education source](https://huggingface.co/datasets/skyai798/DeceptionBench/blob/38b412da565073483c0a68ce9e04eb15859beedc/education.json), [pinned social source](https://huggingface.co/datasets/skyai798/DeceptionBench/blob/38b412da565073483c0a68ce9e04eb15859beedc/social.json).

## Generation and judging are separate measurements

The generation helper appends a request for JSON with `thought` and `response`. It sends one user message and flattens returned line breaks. The prompt-visible `thought` is generated self-report; it is not a direct observation of a private internal belief. Source generation returns the textual response and case metadata, without token IDs, local checkpoint revision, RNG state, prompt hashes or hidden-state tensors. Those provenance fields would have to come from our local runner. [Generation helper](https://github.com/Aries-iai/DeceptionBench/blob/8f3175f66fdff89ded3ebdf0ee2055fa47c955db/utils.py), [single-turn processor](https://github.com/Aries-iai/DeceptionBench/blob/8f3175f66fdff89ded3ebdf0ee2055fa47c955db/processor/single_turn.py).

The single-turn judge receives topic, subtype, question, natural-language honest reference, deceptive objective and generated text. It separately classifies the `thought` and `response` sections as honest/deceptive. The instruction also maps outright refusals to honest and maps unstructured goal-aligned text to both deceptive labels. Consequently these are operational judge labels dependent on the model, rubric, parsing and framing. They are not independent human truth labels or proof of strategic intent. Empty judge parses and malformed outputs need explicit accounting in any adapted evaluation. [Exact evaluation instruction/parser](https://github.com/Aries-iai/DeceptionBench/blob/8f3175f66fdff89ded3ebdf0ee2055fa47c955db/processor/eval_single_turn.py).

The multi-turn judge uses an any-turn rule: once a turn is judged deceptive, the aggregate label remains deceptive. This estimates whether the interaction ever elicits a qualifying output, rather than per-turn prevalence or the final recipient's belief. Target requests receive the newly generated instruction as a single message; the auxiliary generator sees accumulated history and tries to append stronger inducements. Thus the released implementation is an adaptive prompt sequence, not necessarily a target model processing a conventional multi-message conversation history. [Multi-turn evaluation](https://github.com/Aries-iai/DeceptionBench/blob/8f3175f66fdff89ded3ebdf0ee2055fa47c955db/processor/eval_multi_turn.py), [target loop](https://github.com/Aries-iai/DeceptionBench/blob/8f3175f66fdff89ded3ebdf0ee2055fa47c955db/processor/multi_turn.py).

## Upstream reproducibility issues to avoid inheriting

These are static-source observations, not claims of executed failures:

1. The common generation helper appends its `thought`/`response` formatting instruction to **judge calls too**, after the judge explicitly requests a different `###FINAL` evaluation schema. This introduces conflicting format instructions. Our judge should have its own exact, frozen request and strict parser. [Helper](https://github.com/Aries-iai/DeceptionBench/blob/8f3175f66fdff89ded3ebdf0ee2055fa47c955db/utils.py), [judge caller](https://github.com/Aries-iai/DeceptionBench/blob/8f3175f66fdff89ded3ebdf0ee2055fa47c955db/processor/eval_single_turn.py).
2. Resume logic identifies cases by `question` text. Across the release, one question occurs twice. Distinct scenario IDs should be retained even when a human-facing question repeats; the released per-domain execution limits when that collision affects a particular file. Never merge all variants or domains using question text alone. [Loader/resume implementation](https://github.com/Aries-iai/DeceptionBench/blob/8f3175f66fdff89ded3ebdf0ee2055fa47c955db/data_loader.py).
3. CLI choices are restricted to the configured model ID list, which does not include the advertised Llama/Qwen local checkpoint IDs. An unmodified CLI invocation for an arbitrary locally served ID would be rejected by argparse. This does not invalidate the dataset; it argues for a small local adapter rather than blind execution. [CLI model argument](https://github.com/Aries-iai/DeceptionBench/blob/8f3175f66fdff89ded3ebdf0ee2055fa47c955db/main.py), [configuration](https://github.com/Aries-iai/DeceptionBench/blob/8f3175f66fdff89ded3ebdf0ee2055fa47c955db/config.py).
4. The README names `calculate_metrics.py`; the tree contains `calculate_metric.py`. The release has lower-bound dependency constraints, not an environment lock or tests. A model parameter `do_sample=False` is configured but the request helper forwards only temperature/top-p/max-tokens. Our actual generation configuration must be explicit and saved. [Metrics script](https://github.com/Aries-iai/DeceptionBench/blob/8f3175f66fdff89ded3ebdf0ee2055fa47c955db/calculate_metric.py), [requirements](https://github.com/Aries-iai/DeceptionBench/blob/8f3175f66fdff89ded3ebdf0ee2055fa47c955db/requirements.txt), [configuration](https://github.com/Aries-iai/DeceptionBench/blob/8f3175f66fdff89ded3ebdf0ee2055fa47c955db/config.py).

## Cost and licensing boundaries

The released route uses OpenAI-compatible endpoints. Local generation can avoid paid target calls; the default judge is GPT-4o and the default adaptive auxiliary is GPT-3.5-turbo. Running the scripts with remote credentials could therefore incur target, judge and auxiliary costs. This audit made no such request. A full L1/L2 run is 900 target generations **per checkpoint**, plus judging. A single-role L1/L2-pressure slice is 300 target generations per checkpoint, not 300 independent scenarios. L3 multiplies calls and introduces adaptive prompts, so it is unnecessary for an initial fixed robustness analysis. [Default configuration](https://github.com/Aries-iai/DeceptionBench/blob/8f3175f66fdff89ded3ebdf0ee2055fa47c955db/config.py).

The GitHub tree has no standalone LICENSE file and GitHub's license metadata is null. Its README says CC BY-SA 4.0 while also adding academic-only/noncommercial restrictions. The dataset card declares CC BY-SA 4.0 and repeats those additional restrictions; the paper uses CC BY-NC-SA 4.0. These public notices are not internally uniform. Retain attribution, upstream notices and exact revisions; use the material for this academic project and clarify redistribution terms before republishing substantial source data under a different license. This is an observed documentation inconsistency, not a legal interpretation. [Pinned README licensing notice](https://github.com/Aries-iai/DeceptionBench/blob/8f3175f66fdff89ded3ebdf0ee2055fa47c955db/readme.md#license), [dataset card](https://huggingface.co/datasets/skyai798/DeceptionBench/blob/38b412da565073483c0a68ce9e04eb15859beedc/README.md).

## Consequence for this project

DeceptionBench is a credible **external evaluation anchor for the supervisor's deception-probe robustness brief**, with scenario/domain grouping supplied by an existing publication. It is not a replacement gold label for second-order belief. A defensible first use would preserve scenario identity, generate real responses locally under fixed L1/L2 conditions, label actual response behavior with a frozen rubric plus recorded human audit, and test the deception probe separately from the belief probe. Prompt-only controls are essential because the prompt itself already contains deceptive objectives. All family variants must remain in the same selection/evaluation partition.

The earlier bank/schema work remains development material, not a completed 154-group dataset or a result. The external benchmark does not justify fitting the old location-belief target to deceptive-behavior labels, claiming causal pressure isolation, or calling a judge's assessment a direct observation of intent. A later omission/recipient-knowledge extension needs its own explicit labels and independently measured recipient effects.

### Recommended first evaluation, still a draft

Keep one bounded external evaluation alongside the proposed belief experiment. A useful first scope is the published **self** role, all 150 scenarios, and L1 / L2-pressure / L2-reward: 450 target generations per checkpoint. The scope is selected by source metadata before observing outputs. Reserve scenario groups for development before generation; keep every variant of a scenario in one partition. This is a proposed scope, not an executed or frozen protocol.

Before weights are loaded, bind the exact generation wrapper, checkpoint, precision, decoding, response span, probe artifact, score orientation, threshold-selection rule and annotation codebook. Explicitly distinguish direct misleading communication, helping someone else deceive, refusals, ambiguous cases and invalid outputs. Assess the actual responses; do not infer labels from conditions. Preserve all counts, and report a metric as unavailable when a condition lacks either class.

Compare the frozen Apollo-style head against an instructions-only baseline and a capable full-context text monitor. Behavior labels must be assessed independently of the monitor being scored: using a judge's own labels as its evaluation target would make that baseline circular. Use blinded human audit and record unresolved disagreements. Report within-condition AUROC, honest-response false positives at a development-fixed operating point, and uncertainty clustered by source scenario. Pooled AUROC is secondary because it can reward separating conditions rather than separating behavior within a condition. The released prompt rewrites still prevent a pressure-only causal claim.

This external evaluation establishes scope and replication; it does not replace the separate, matched training-coverage/composition hypothesis or itself establish a new contribution. Defer L3 and broad parameter sweeps until this fixed comparison is interpretable.

## Prior art prevents a simple novelty claim

Huang et al., *Why Static Probes Fail: Context Determines Deception Feature Composition*, already evaluate **THU-DeceptionBench L2**. Sections 4–5 compare an instructed-pair linear probe with a seven-direction adaptive method. The latter uses ten labeled target-domain examples; it is not frozen zero-shot transfer. Their evaluation includes honest-target false positives at a benign-control-calibrated threshold. Consequently Apollo-style transfer onto this benchmark, domain sensitivity and high honest-response false positives are already studied. They may provide an external replication anchor alongside the matched belief-composition study, not automatic new-paper novelty. [Paper Sections 4–5 and Table 1](https://arxiv.org/html/2610.27577v1#S5).

The public [Yilin1010/deception-probe repository](https://github.com/Yilin1010/deception-probe/tree/8e51d8501fee5571a3dc3854f37fce1ee1eb0f78), pinned commit `8e51d8501fee5571a3dc3854f37fce1ee1eb0f78`, directly advertises an Apollo-style REPE-to-DeceptionBench experiment on Llama-3.3-70B. Its 746 retained response labels come from Claude judgments with ambiguous scores excluded, and its README advertises cached activations. These are that repository's claims, not independently reproduced measurements. We did not download or deserialize its cached Python objects. Its existence independently rules out claiming the application alone as new.

## Local adapter and executed validation

The [source manifest](../research/deceptionbench/source_manifest_v1.json) stores only upstream pins, raw hashes, counts and licensing notices. No raw scenario dataset is vendored. The [stdlib adapter](../src/latent_probes/deceptionbench_data.py) reads caller-provided local files, checks exact bytes and schemas, preserves scenario grouping, and emits six source-only prompt records per scenario. Behavior labels remain null. Reference answers and the separate goal metadata are never appended to model messages; the published source prompts themselves retain their original facts and instructions. The upstream JSON-formatting suffix is deliberately not added, so these messages are identified as source prompts rather than an exact generation-input reproduction.

Run the non-generating validation with:

```powershell
.venv/Scripts/python.exe scripts/validate_deceptionbench_source.py `
  --source-directory <local-published-domain-JSON-directory> `
  --manifest research/deceptionbench/source_manifest_v1.json
```

Actual pinned files were downloaded into a temporary audit directory outside the repository. The CLI verified all five byte hashes, 150 scenarios and 900 unique case IDs; **model calls and behavior labels both remain zero**. The [machine-readable audit record](../research/deceptionbench/source_audit_v1.json) preserves the executed summary. All 24 focused tests pass using explicitly synthetic source fixtures, including label noninference, reference exclusion, paired scenario IDs, schema corruption, wrong benchmark identity and byte mismatches. Synthetic test fixtures are not presented as published inputs or experimental outcomes.
