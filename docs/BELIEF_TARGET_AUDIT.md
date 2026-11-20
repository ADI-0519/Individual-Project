# Belief targets: primary-methods audit and the next diagnostic

20 November 2026. **The existing reader-forecast failure does not establish absence of anticipation or second-order belief representations.** It evaluates one linear readout, representation location, task and distribution shift. A second-order-belief decoder has not yet been validated in this project. This note records the primary sources actually inspected and the construct boundaries for the next diagnostic; it reports no new model result.

The current natural-report study and its failures remain preserved in [NEUTRAL_FORECAST_RESULTS.md](./NEUTRAL_FORECAST_RESULTS.md) and [RESEARCH_CHECKS_RESULTS.md](./RESEARCH_CHECKS_RESULTS.md). The broader remaining work is in [MAIN_STUDY_GATES.md](./MAIN_STUDY_GATES.md).

## Primary sources and reading scope

### Zhu, Zhang and Wang: Language Models Represent Beliefs of Self and Others

Primary methods inspected: [arXiv 2402.18496v3, 30 May 2024](https://arxiv.org/html/2402.18496v3), §§3–5 and Appendices A–D. [ICML 2024/PMLR metadata](https://proceedings.mlr.press/v235/zhu24o.html) was verified. The PMLR PDF endpoint failed to fetch, so exact final-PDF parity was not checked. This was not an exhaustive reading of all 44 pages/appendices.

The main experiment uses BigToM story–candidate-statement pairs and attention-head activations at the final token. Binary probes predict protagonist and omniscient-oracle labels; a four-class probe predicts their joint status. This separates perspectives but does not by itself validate the nested target “S believes R believes p.” Heads are selected using validation accuracy for intervention experiments. The paper also tests cross-task intervention, so transferring a belief-related direction is not a new idea alone.

**Important qualification:** Appendix D explicitly extends to ToMi, which contains second-order questions absent from BigToM. It transfers a BigToM intervention and reports aggregate ToMi performance. Do not say the paper never tests second-order ToM; equally, do not treat its main oracle/protagonist probes or aggregate ToMi table as a separate, fully dissociated nested-belief decoder result. Appendix C's nonlinear probes also show that failure of a linear readout is not exhaustive evidence about available information.

### Bortoletto et al.: Brittle Minds, Fixable Activations

Primary source inspected: [official Findings EMNLP 2025 PDF](https://aclanthology.org/2025.findings-emnlp.1226.pdf), §§3–5, limitations, and targeted Appendices A.1–A.3 and A.5. The official 23-page publication was used; all appendix figures/examples were not exhaustively audited. [Publication metadata](https://aclanthology.org/2025.findings-emnlp.1226/) and the authors' implementation link were checked, but implementation details for every prompt experiment were not verified.

The study uses BigToM Forward Belief labels from protagonist and omniscient perspectives. It probes final-token residual states across layers and 12 models, with label-permutation and PCA controls, prompt variations and activation editing. Appendix A.1.2 specifies L-BFGS, L2 regularisation, C=10 and at most 1,000 iterations. Appendix A.2 discusses below-chance early-layer fits and explicitly declines to clip them to chance.

The results motivate testing representation location, fitting choices and prompting. Permutation/PCA controls do not remove every lexical shortcut or prove mental-state semantics. Do not describe every prompt comparison as a verified frozen-probe transfer experiment: reuse versus refitting across all variants was not independently established in this audit. Its task is also not natural pressured communication with measured recipients.

### Prakash et al.: Language Models Use Lookbacks to Track Beliefs

Primary source inspected: [arXiv 2505.14685v3, 24 February 2026](https://arxiv.org/html/2505.14685v3), §§2–3, targeted binding/visibility mechanism passages, and Appendix D. The [ICLR 2026 listing](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0d1941f73833eb107939f62d323d6cc0-Abstract-Conference.html) was verified. This was not an exhaustive audit of every causal-patching appendix.

Their account stores reference/binding information at earlier tokens and retrieves it when the relevant question arrives. Consequently, a final query-free story token need not expose every possible later answer through a simple linear readout. The mechanism experiments select 80 correctly answered samples. Appendix D reports Llama-3.1-8B-Instruct accuracy .722±.044 without visibility and .310±.035 with explicit visibility; Qwen2.5-14B-Instruct scores .962/.912. These figures concern CausalToM and its execution settings, not our task or NF4 run.

The practical implication is a behavioral capability gate and a targeted stronger-model reference if needed. The paper does not imply our 8B model cannot represent beliefs, or that a 14B model will solve our proposed diagnostic.

### Le, Boureau and Nickel: Revisiting the Evaluation of Theory of Mind through Question Answering

Primary source inspected: [EMNLP-IJCNLP 2019 PDF](https://aclanthology.org/D19-1598.pdf), §§2–4, including the shallow-rule analysis and question/control construction. The [authors' ToMi repository](https://github.com/facebookresearch/ToMi) confirms separate first-order, second-order, reality and memory question categories. The complete generator implementation was not audited.

Their second-order target explicitly concerns one agent's belief about another agent's belief. They show that earlier datasets permit simple lexical/ordinal shortcuts, randomise generation to weaken those biases, and ask reality, memory and both orders about the same story. Joint story correctness is useful alongside marginal accuracy. These are direct precedents for our control structure, not novelty claims.

### Kim, Lee and Lee: Telling Incomplete Truths

Primary material inspected for this question: the locally extracted [official EMNLP 2026 PDF](https://aclanthology.org/2026.emnlp-main.656.pdf), §5.8, limitations and Appendix O; [publication metadata](https://aclanthology.org/2026.emnlp-main.656/) was verified. This was a targeted probe-method reading, not a new complete-paper audit.

Their pre-generation label records whether the subsequently generated answer entails decision-critical facts. Appendix O specifies hidden states, NLI-derived output labels and cross-validated layerwise classifiers. This is future-output decodability, not a demonstrated belief about a reader. The paper explicitly distinguishes its decodability finding from knowing and withholding a fact.

## Four quantities that the project must keep separate

| Quantity | Operational meaning | What it does not establish |
| --- | --- | --- |
| World truth, W(p) | The proposition's value under the authored complete record | A model's belief or a recipient's decision |
| Reader perspective, B_R(p) | What follows from R's fictional observation history | What an actual reader LM internally believes |
| Nested sender perspective, B_S(B_R(p)) | What S's information supports about R's belief | The actual R's state, especially when S is mistaken |
| Observed reader outcome, Y | A saved answer/decision from a fresh reader conversation | A normative perspective label or direct mental-state measurement |

Asking “what does R believe?” is a first-order perspective task. A genuinely nested task must ask what S believes R believes, and include cases in which S is wrong about R's information. Jointly decoding world truth and R's belief is not sufficient. Nor should the LLM itself be counted as an extra agent merely to inflate the stated order.

For a model playing the sender, recipient modelling is first-order mentalising from its own role. A narrator-level diagnostic can explicitly test the nested fictional proposition B_S(B_R(p)). Transfer between these task/role formats remains an empirical question; matching notation alone does not demonstrate a shared representation.

## Planned fresh diagnostic and visibility checks

At the time of this note, the proposed panel is **64 fresh prior-state stories**, eight wording groups with all eight crossed world/reader/sender-attribution combinations, split by complete wording group 4/2/2. There are also 16 proposed trusted-current-update controls. This is a construct/development diagnostic, not family-held-out evidence. The final saved plan, rather than this prose, must specify the exact behavioral call count and all extraction settings.

The crucial manipulated factors are world p, R's actual observation/access, and S's view of R's observation/access. For example, R secretly sees a current failed inspection while S thinks R still has the old pass. The nested fictional label differs from R's actual perspective even though S knows the current world state. Counterbalance proposition polarity and event wording so that one receipt word, date position or last-mentioned status cannot solve every label.

Two views have different information boundaries:

- **Omniscient narrator view:** includes both agents' histories. It can legitimately support world, first-order and nested fictional labels.
- **Sender view:** excludes R's actual private attendance/access when S did not observe it. Matched actual-access conditions must be byte-identical when everything visible to S is fixed. Those identical inputs cannot support identification of the hidden actual-access condition; this is a structural negative control, not model failure.

The proposed primary capture is block 16 at a query-free story-plus-candidate prefix, with blocks 8 and 24 declared as diagnostics before results. Perspective questions belong in separate continuations. A query-free candidate prefix is not the same representation as one after an explicit nested-belief question: report that distinction, since query-time retrieval is a plausible reason for different decoding performance. The proposed fixed locations avoid selecting a winner after reading test scores; any later layer search needs new validation and an explicit exploratory label.

Fictional belief gold is **normative under the authored observation rules**, not “the LM's true mental belief.” Validate world, memory, reader and nested questions jointly on each story, and separately measure actual reader answers. Keep UNKNOWN, inconsistency and INVALID visible. Human review must verify the observation histories and update assumptions. The wording-group test has only two group clusters; do not turn 16 dependent condition rows into a population-confidence claim.

## Connecting this diagnostic back to the natural-report project

The original natural study's target is observed reader decision Y under a sender-attributed packet. The new prior nested task has a different, explicitly annotated target. It can establish whether the model/readout distinguishes perspectives before trying to explain transfer failure. It does not retroactively relabel the current forecast as a belief decoder.

Keep separate:

1. **Prior nested state:** B_S(B_R(p)) before a message. A decoder transferred at this time tests perspective tracking under reporting pressure.
2. **Known-message update:** B_S(B_R(p after M)) with the exact M available in both training and evaluation. Trusted-update controls test this limited update assumption; actual reader answers remain separate measurements.
3. **Unknown-message forecast:** the reader outcome expected before a natural report is generated. This averages over the frozen sender policy and reader policy. With greedy generation it concerns one deterministic policy; identical repeats add no observations.

A decoder trained with M available cannot silently become a forecast of its effect before M exists. Similarly, a prior-belief decoder cannot be scored as a post-message decision predictor without an explicit bridge and validation. Natural summaries remain the main communication setting; fixed truthful reports and controlled stories supply construct checks.

A failed neutral probe can indicate inadequate features, a mismatched target, fitting limitations or insufficient information at the selected token. A frozen probe that fails under pressure can indicate readout misalignment or changed task processing. None of these alone establishes information absence. Conversely, successful decoding alone does not establish spontaneous anticipation, deceptive intent or a causal belief mechanism. Fresh matched targets, strong text controls, independent splits and later targeted interventions are what would distinguish those explanations.
