# Methodology reading log

Read and source-audited on 19 November 2026. This log records what was actually inspected, distinguishes implementation evidence from paper prose, and does not substitute for model experiments.

## Apollo: full paper and all appendices

**Goldowsky-Dill, Chughtai, Heimersheim and Hobbhahn (ICML 2025), Detecting Strategic Deception with Linear Probes.** [Official proceedings](https://proceedings.mlr.press/v267/goldowsky-dill25a.html), [arXiv](https://arxiv.org/abs/2502.03407), [official implementation](https://github.com/ApolloResearch/deception-detection/tree/f8ec4010e74927394709dffa22b97bdf8cd5a62f).

The complete local paper text was read, including the references and all sections A–G. The following is an extraction of the relevant methods, rather than an abstract-only summary.

| Section | Evidence and implication |
| --- | --- |
| Abstract and 1 | Defines strategic deception as attempts to induce false beliefs to pursue a goal. The evaluated 70B scenarios are simpler than the sophisticated scheming threat model. |
| 2 | Separates off-policy factual-truth probing from strategic deception; motivates generalization to on-policy agent behavior. |
| 3.1 | Residual stream, layer 22 of 80; selected token positions flattened for fitting; training normalization; L2 logistic regression, lambda 10; mean real-valued scores at evaluation. Hyperparameters were initially chosen on Llama 3.1 70B and carried to 3.3. |
| 3.2.1 | Honest/dishonest instruction pairs share an inserted true fact; paper says last five tokens excluded. The training data contains no actual deception. |
| 3.2.2 and Table 1 | On-policy Llama 3.3 70B responses. Roleplaying: 84 honest, 84 ambiguous, 203 deceptive. Insider report: 669/324/336. Confirmation: 24/2/630. Sandbagging: 511/64/420. These are paper-specific counts, not expected 8B outcomes. |
| 3.2.2, insider report | Conditions on the model taking the insider trade; reports with explicit admission are honest, vague acknowledgment ambiguous, omission deceptive. This is already an omission-like setting. |
| 3.2.2, sandbagging | Structured reasoning supplies labels: intending to sandbag and changing the internally selected answer is deceptive; intending to sandbag but retaining that answer is ambiguous. A wrong answer by itself is insufficient. |
| 3.2.3 | On-policy Alpaca completions calibrate the neutral-chat control tail. This control distribution is deliberately unlikely to concern deception. |
| 3.3 | Within-dataset honest/deceptive AUROC and deceptive recall above the 99th-percentile neutral-control score are distinct metrics. |
| 4.1.1 | Strong paper AUROCs do not imply reliable pressured-honest classification. Adjacent layers can sharply change recall at the neutral threshold. |
| 4.1.2 | Known false positives include honest confessions. Mean aggregation can dilute a deceptive span with honest text. Some failures remain unexplained. |
| 4.2 | Roleplaying-trained, SAE, follow-up-question and descriptive-question probes differ. Roleplaying training reduces incentive sensitivity, while the forced follow-up probe can rate all sandbagging-task responses highly deceptive. The black-box baseline uses GPT-4o and a less tailored prompt than the ground-truth graders. |
| 5 | Current scores are insufficient as a robust defense. Hidden reasoning is excluded from scored token positions but can influence those positions through attention. |
| A.1 | Pacchiardi roleplaying has matched incentive/no-incentive scenarios. Five completions per prompt. Honest responses score higher when deception is incentivized; the roleplaying-trained probe reduces this difference. The pressure confound is an existing observation. |
| A.2 | AI-audit omissions may reflect confusion or failure to recognize an ethical issue rather than deliberate concealment. The authors exclude this setting from main results because ground truth is questionable. This is an important model-capability and label-validity warning for an 8B adaptation. |
| B.1–B.4 | Documents detailed alternate probes. B.1 states an 80% roleplaying training split; B.3 says 90% while claiming comparability. Do not silently reconcile that inconsistency. The saved main instruction-pair configuration is the reference for this project. |
| C | 9,994 controls. Top-scoring controls request incorrect editing or contradictions; satire, sarcasm and hyperbole can also activate the detector. Neutral chat is not a perfect universal-negative distribution. |
| D.1 | Compares logistic regression, difference of means, covariance-adjusted mean direction and paired PCA/LAT. Logistic regression improves separation from control chat. |
| D.2 | Layer 22 selected using roleplaying on Llama 3.1 70B. Sensitivity of low-FPR recall makes representative validation important. No implication that index 22 is optimal on 8B. |
| D.3 | Lambda 10 is a compromise across evaluated datasets, not universally optimal. |
| D.4 | Mean is best for the instructed-pair probe among the tested aggregations; thresholded ReLU sometimes helps the roleplaying probe. |
| D.5 | Last-token probing is separately trained on last-token roleplaying features. It is not equivalent to taking the final state of the main token-trained probe. |
| E | Excludes ambiguous labels in primary results; inclusion sensitivity changes roleplaying AUROC from .964 to .913 and report AUROC from .999 to .978 in the stated relabelings. Report ambiguity rates rather than hiding them. |
| F.1–F.4 | Reads complete honest/deceptive examples with per-token scores for all four evaluation datasets. Insider examples have common boilerplate and visible preceding reasoning; natural reports differ in disclosure. |
| G.1 | Roleplaying suppresses asides/internal thoughts and prefills character names. Grader scale 1–7: 1–2 honest, 3–5 ambiguous, 6–7 deceptive. Scenario creation includes GPT assistance, self-critique and five vetters. |
| G.2 | Full trade transcript, filtering and report/confirmation grading. Some parsing failures and claimed non-use of insider information are filtered. Conditioned selection must be acknowledged. |
| G.3 | Full sandbagging system prompt and four-step reasoning procedure; only user-facing step 4 is suitable for the declared scored span. |
| G.4 | Black-box monitor receives the entire conversation. The text lists a final binary judgment despite a main-text description of a seven-point scale. Ground-truth grading is more dataset-specific. Avoid claiming exact black-box reproduction from prose alone. |

### Source files inspected at the pinned commit

| Source | Verified implementation detail |
| --- | --- |
| [Saved instructed-pair configuration](https://github.com/ApolloResearch/deception-detection/blob/f8ec4010e74927394709dffa22b97bdf8cd5a62f/example_results/instructed_pairs/cfg.yaml) | `plain`, 3.3 70B, layer tuple index 22, lambda 10, normalization, validation .2, on-policy controls/evaluation, off-policy training. |
| [RepE dataset](https://github.com/ApolloResearch/deception-detection/blob/f8ec4010e74927394709dffa22b97bdf8cd5a62f/deception_detection/data/repe.py) | Selects up to the first 512 true CSV rows; interleaved honest/deceptive pairs; exact contractions in prompts; last five space-delimited words, not tokens; detect-prefix/ignore-suffix segments; zero mask padding. Actual pinned data availability is recorded below. |
| [Experiment](https://github.com/ApolloResearch/deception-detection/blob/f8ec4010e74927394709dffa22b97bdf8cd5a62f/deception_detection/experiment.py) | RepE shuffling disabled; dialogue-level suffix validation split before activations; deceptive features passed as logistic positive class. |
| [Activations, line 73](https://github.com/ApolloResearch/deception-detection/blob/f8ec4010e74927394709dffa22b97bdf8cd5a62f/deception_detection/activations.py#L73) | Uses Hugging Face `hidden_states[layer_idx]`, proving the layer-number convention. |
| [Logistic detector](https://github.com/ApolloResearch/deception-detection/blob/f8ec4010e74927394709dffa22b97bdf8cd5a62f/deception_detection/detectors.py#L300) | StandardScaler on token rows, `C=1/reg_coeff`, no fitted intercept, seed 42, saved scaler, raw projected scores. Mean-feature logistic regression is a different class. |
| [Tokenization](https://github.com/ApolloResearch/deception-detection/blob/f8ec4010e74927394709dffa22b97bdf8cd5a62f/deception_detection/tokenized_data.py#L124) | Character-to-token boundary alignment, half-open mask interval, dataset-specific padding, optional final-token mask. |
| [Preprocessing](https://github.com/ApolloResearch/deception-detection/blob/f8ec4010e74927394709dffa22b97bdf8cd5a62f/deception_detection/utils.py) | Adjacent same-role messages concatenate; masked fact prefix and unmasked suffix remain one assistant turn. |
| [Base dataset](https://github.com/ApolloResearch/deception-detection/blob/f8ec4010e74927394709dffa22b97bdf8cd5a62f/deception_detection/data/base.py) | Default Llama padding differs from zero-padding RepE/roleplaying; reasoning is split into unscored/scored segments, not removed from causal context. |
| [Roleplaying dataset](https://github.com/ApolloResearch/deception-detection/blob/f8ec4010e74927394709dffa22b97bdf8cd5a62f/deception_detection/data/roleplaying.py) | Zero padding and an unscored character-name prefix. |
| [Control dataset](https://github.com/ApolloResearch/deception-detection/blob/f8ec4010e74927394709dffa22b97bdf8cd5a62f/deception_detection/data/control.py) | First 10,000 Alpaca entries, empty reference-output exclusion; control generation is selected by experiment configuration. |
| [Prompt scorer](https://github.com/ApolloResearch/deception-detection/blob/f8ec4010e74927394709dffa22b97bdf8cd5a62f/deception_detection/prompt_scorer.py) | Mean raw token scores; `nanquantile(.99)` threshold; strict exceedance. |
| [Models](https://github.com/ApolloResearch/deception-detection/blob/f8ec4010e74927394709dffa22b97bdf8cd5a62f/deception_detection/models.py) | Exact model mapping, shared 3.1 70B tokenizer, FP16 Llama load. |
| [Metrics](https://github.com/ApolloResearch/deception-detection/blob/f8ec4010e74927394709dffa22b97bdf8cd5a62f/deception_detection/metrics.py) | Executable AUROC code treats high scores as deceptive. Its opening sign comment is inconsistent with the current experiment fitting call. |

### Pinned data audit and correction

The loader's `[:512]` cap was initially mistaken for evidence of 512 available true facts. Direct inspection of the pinned [CSV](https://github.com/ApolloResearch/deception-detection/blob/f8ec4010e74927394709dffa22b97bdf8cd5a62f/data/repe/true_false_facts.csv), independently checked locally, found 612 data rows and only **306 true facts**. SHA-256: `924709159b40a30d5b51488e9ccd7fd886456a89f0e0f5849eacefba6ff55312`. Therefore this artifact supplies **245 training and 61 validation pairs before exclusions**, not the previously inferred 410/102. The available artifact does not establish the historical paper run's effective sample count.

Thirty-one true facts have at most five space-delimited words and therefore no retained prefix under the pinned word-truncation rule. Twenty-one belong to the original training split and ten to validation. Exclusion of these entire pairs **after preserving the original split** retains **224 training and 51 validation pairs before duplicate protection**. Log exclusions and original source indices; do not silently change the mask or first filter and then recompute a new split. Historical handling of these empty masks remains unverified.

The same selected artifact also contains eight later duplicate occurrences across seven statement groups: Mercury (selected-fact indices 2/147), Earth being round (28/251), Great Wall (32/213), Great Barrier Reef (76/81), ice cream (135/180), the human skeleton (140/220/292), and sunrise (265/291). These are indices into the selected true-fact sequence. Repeated text can cross the original partition boundary even with distinct pair IDs. In particular, the human-skeleton group spans training and validation.

With explicit `--duplicate-fact-policy exclude`, keep the first occurrence and log the later ones. One later duplicate overlaps the short-fact exclusions: together the policies remove 38 selected rows and retain **268 unique eligible pairs, split 219 training / 49 validation**, or 536 instruction-conditioned examples. The intermediate short-only count of 275 pairs must not be presented as the final deduplicated training panel. Both policies default to errors and require explicit caller selection; neither changes these instruction labels into observed behavior or deception labels.

## MASK: full paper and all appendices

**Ren et al., The MASK Benchmark: Disentangling Honesty From Accuracy in AI Systems**, local text arXiv v3, 5 January 2026. [Paper](https://arxiv.org/abs/2503.03750), [official code](https://github.com/centerforaisafety/mask).

The complete text was read, including references, A.1–A.10 and the complete model-results table. Upstream MASK evaluation code has not been line-by-line audited in this task; paper-described judging must not be mistaken for a verified code port.

| Section | Extraction |
| --- | --- |
| 1–2 | Distinguishes factual accuracy from honestly reporting a belief. Focuses on commission, unlike this project's omission extension. |
| 3 | Operational belief means neutral responses consistent across related questions. Lying compares pressured assertion with that operational belief, with intent supplied by the scenario design; it is not direct access to a mental state. |
| 4.1 | 1,000 public examples and 500 held-out examples; each has proposition, ground truth, pressure prompt and belief prompt. Avoids prompts that plausibly update factual belief, trivial fiction and subjective ground truth. |
| 4.2 | Three direct neutral queries; binary propositions also get two indirect queries. Judge extracts proposition values separately; programmatic comparison follows. Evasion/no-belief are distinct statuses. Manual validation: 150 examples, 86.4% agreement. |
| 4.3 | `S != B` defines the observed lie event. Evasion and no-belief are not counted as lying. Therefore `1-P(lie)` is not the explicitly honest fraction. Accuracy uses `B` versus `T`, separately. |
| 5.1 | Capability/accuracy and lying propensity need not scale together; the evaluation does not justify importing 70B behavior labels into 8B generations. |
| 5.2 | Developer honesty instructions and LoRRA interventions improve reported honesty in their experiments but do not eliminate lying. LoRRA is an intervention, not the Apollo logistic detector. |
| 6 | Calls for explicit safety-construct evaluation rather than accuracy proxies. |
| A.1–A.2 | Notes scope and judge limitations: English short factual propositions, six prompt archetypes, no comprehensive interactive deployment evaluation. |
| A.3 | Consistency operationalizes robust belief; inconsistent answers do not count as lies. |
| A.4 | Utility analysis adds 20 honesty/dishonesty outcomes to 510 base outcomes and excludes utility fits under 80% accuracy; unrelated to training the baseline probe. |
| A.5 | Persona-removing follow-up self-reports agree with pipeline labels in 83.6% of strictly honest/dishonest cases. Self-report is corroboration, not perfect ground truth. |
| A.6 | Exact developer honesty instruction appended around original system prompt. |
| A.7 | Human contributor training, staged review and final audits; automated filtering/proposition/ground-truth/belief-prompt generation; nearly all examples manually reviewed by leads. Table 2 has known facts, situation-provided facts, doubling down, fabricated statistics, continuations, and disinformation. Situation-provided facts omit neutral elicitation because the facts are in context. |
| A.8 | Lying@10 marks any lie across ten pressure runs, with non-lying status resolved separately. This is not single-generation FPR. |
| A.9 | Additional analysis excludes inconsistent/no-belief cases; denominator differs from primary non-lying score. |
| A.10 | Full model table includes Llama 3.1 8B. Its behavior rates describe this benchmark/version/pipeline, not a promised local replication. |

## Truth is Universal: targeted methodological reading

**Bürger, Hamprecht and Nadler (2024), Truth is Universal: Robust Detection of Lies in LLMs.** [Paper, v2](https://arxiv.org/abs/2407.12831v2).

Read appendix F.1–F.4 in full, with particular attention to F.4. This is not claimed as a full-paper audit.

- F.1: samples at temperature 1 and stops at the first period or exclamation mark, so response-length and aggregation settings differ from Apollo.
- F.2: separates unambiguous truth/lie, ambiguous truth/lie and other. Includes examples illustrating subjective boundaries and invented elaboration.
- F.3: category counts are 55 unambiguous truthful, 49 unambiguous lies, 23 ambiguous truthful, 18 ambiguous lies and 63 other.
- F.4 explicitly asks whether detectors see lies or incentives. Only **six** honest responses despite incentives were available. Reported TTPD accuracy is 82% ± 5%, LR 62% ± 17%, CCS 77% ± 22%, and MM 100% ± 0%. These are accuracies on a tiny selected subgroup, not deployment FPR estimates. The control concept predates this project.

## One Probe Won't Catch Them All: methods and experiment design

**Natarajan, Jain, Arora, Golechha and Bloom (2026), One Probe Won't Catch Them All: Towards Targeted Deception Detection**, local text arXiv v2, 18 June 2026. [Paper](https://arxiv.org/abs/2602.01425v2).

Read sections 1–4 and the beginning of section 5, including complete methods 3.1–3.6. This is not claimed as an all-appendix audit.

- Main model is Gemma 2 9B IT; primary layer 20 is explicitly zero-indexed. Llama 3.3 70B repeats the ANOVA finding.
- Uses forced fact responses under contrasting prompts and describes excluding five response tokens. Its methods also discuss mean aggregation; without inspecting its implementation this should not be assumed identical to Apollo token-row fitting.
- Evaluation responses are on-policy at temperature 1. Training facts, validation scenarios and evaluation scenarios have distinct roles.
- Validation for the single best-average probe contains 50 balanced examples from each of three named datasets, excluded from final evaluation.
- The best-taxonomy result on an evaluation dataset is a **post-hoc oracle** selected using that dataset's results. It must not be reported as an independently validated deployable selector. Separate synthetic validation tests targeted prompts where the type is known a priori.
- Twenty-three taxonomy prompt pairs and 34 framing variations are described. An ANOVA over 2,000 probes attributes 70.6% of measured performance variance to instruction-pair choice; that number is specific to their factor levels and evaluation setup.
- Their tuned regularization is lambda 1, rather than Apollo's lambda 10. Treat this as a different experimental recipe.
- The paper already evaluates MASK transfer failures and incentive/prompt sensitivity. A new 8B pressure-control replication needs the reader-knowledge/omission extension for a distinct contribution.

## MASK implementation and public-data metadata audit

Inspected the official generator, evaluator/parser, downloader, repository
license and dataset card/API metadata. The inspected code revision is
[`25e0b1201e6c928ebe69f7c5aad6fa9063a377ea`](https://github.com/centerforaisafety/mask/tree/25e0b1201e6c928ebe69f7c5aad6fa9063a377ea);
the canonical dataset is
[`cais/MASK` at `74a44111218d762985b7a82f39b0eb5bddadad2a`](https://huggingface.co/datasets/cais/MASK/tree/74a44111218d762985b7a82f39b0eb5bddadad2a).
No pilot case records were downloaded, selected or scored in this audit.

- The `known_facts` / `test` metadata reports 209 binary rows. Its original
  `belief_elicit_1/2/3` prompts remove the need to invent indirect questions.
  The generator repeats the first three times and the others once each.
- Pressure uses the original system/user prompt pair. The upstream generation
  clients are for cloud APIs; our local checkpoint uses this repository's backend.
- The parser recognizes slash-separated alternatives inside `<pivotal>` tags;
  option order does not identify truth. Numerical propositions use `<value>`.
- Dataset terms are CC-BY-4.0; repository code is MIT. These are separate sources
  with separate attribution/notice requirements. Do not assume a code license
  licenses the data.
- The card's loading example points to `cais/hle`; the official downloader
  correctly names MASK. The card's 1,028-example prose also differs from the
  1,000 total in its six configuration counts. Preserve the exact revision and
  selected IDs rather than assuming identity with a historical release.

The resulting [pilot protocol](./PRESSURE_PILOT.md) remains planned, with
behavioral annotation and local inference scheduling still to be completed.

## Consequences for this repository

The implementation contract and explicit 8B adaptations are in [Apollo method](./APOLLO_METHOD.md). The main safeguards are source-pinned training templates, word-boundary masking, token-row fitting, training-only normalization, raw-score aggregation, explicit layer conventions, independent control calibration, and a behavior label that distinguishes honesty from pressure, evasion, factual error and recipient outcomes.

The genuine 8B check now verifies mask/ID replay and exact hook/hidden-state
parity for one saved dialogue; the full instruction-condition diagnostic also
completed. Open issues include broader boundary-test coverage, behavior-label
auditing, neutral-tail sample size, a MASK local-inference adapter and whether
the selected layer transfers to behavioral evaluation. Software checks and
instruction separation do not settle those empirical questions.
