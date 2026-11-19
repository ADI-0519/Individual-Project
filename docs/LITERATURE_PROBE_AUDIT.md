# Latent deception probes: literature audit and project implications

**Audit date: 19 November 2026.** This is a primary-source research assessment, not a report of new model results. Reading scopes below distinguish full-paper reading, targeted sections, implementation inspection, and search-only leads. No novelty or publication guarantee is implied.

The recommendation is to retain **recipient-dependent truthful omissions** as the research direction, while treating the Apollo adaptation and pressure-specificity panel as validation milestones. Generic probe robustness, pressure confounds, omission-themed instruction pairs and elicited-belief deception detection already have substantial prior work.

Existing implementation details and the small pilot are documented in [Apollo method](./APOLLO_METHOD.md), [reading log](./READING_LOG.md) and [planned pressure pilot](./PRESSURE_PILOT.md).

## 1. Primary-paper comparison

### Goldowsky-Dill et al.: the baseline and an existing pressure confound

**Detecting Strategic Deception with Linear Probes**, ICML 2025. [Official proceedings](https://proceedings.mlr.press/v267/goldowsky-dill25a.html); [preprint](https://arxiv.org/abs/2502.03407).

**Reading scope:** complete paper and all appendices A–G. Public implementation inspected at commit `f8ec4010e74927394709dffa22b97bdf8cd5a62f`; subsequent direct data inspection corrected assumptions about available fact counts and exclusions. The companion method document records those discrepancies.

The paper demonstrates transfer from simple instruction-pair or roleplaying training to several on-policy Llama 3.3 70B evaluations. Its training pairs need not contain an actual lie. Sections 4.1 and A.1 already show elevated scores on honest responses when the scenario incentivizes deception. Thus reproducing that confound is a baseline control, not this project's novelty. Appendix A.2 excludes an audit setting because omissions could reflect confusion rather than intentional concealment. The study also distinguishes within-task AUROC from recall above a neutral Alpaca control threshold. Neither guarantees specificity within deception-related contexts.

### Ren et al.: assertion consistency is different from factual accuracy

**The MASK Benchmark: Disentangling Honesty From Accuracy in AI Systems**, first released 2025; inspected paper **v3, 5 January 2026**. [Paper](https://arxiv.org/abs/2503.03750v3); [official code](https://github.com/centerforaisafety/mask).

**Reading scope:** complete paper and all appendices. Also inspected the public dataset schema, original response generator, downloader and relevant binary judging code. The pilot document records the pinned data revision and separate code/data licenses.

MASK compares a pressured assertion with a consistently elicited neutral resolution; it separately compares the latter with factual ground truth. Sections 4.2 and A.3 use three direct elicitations and two indirect checks for binary propositions. Section 4.3 keeps evasion and inconsistent/no-belief cases out of the lying count, so the non-lying complement cannot become a set of clean honest negatives. The benchmark explicitly targets commission rather than omission. Its belief definition is operational: consistency under neutral prompts does not independently establish unchanged belief under pressure. Whole-response audit remains necessary even when a focal proposition is resolved correctly.

### Natarajan et al.: targeted probes already include omission-related contrasts

**One Probe Won't Catch Them All: Towards Targeted Deception Detection**, ICML 2026; inspected **arXiv v2, 18 June 2026**. [Official proceedings](https://proceedings.mlr.press/v306/natarajan26a.html); [v2 text](https://arxiv.org/html/2602.01425v2).

**Reading scope:** complete paper, results, limitations and all appendices A–B. Its implementation was not independently audited to the same depth as Apollo's.

Training instruction contrast explains much of the variance across their tested probe configurations. Targeted contrasts already cover partial truth, concealment, evasion and misleading related answers. The generic probe's MASK Known Facts AUROC is .374, versus .697 for a targeted contrast. The best-taxonomy headline is a post-hoc oracle; the independently selected best-average probe improves less. Synthetic targeting evidence is uneven: the matched probe ranks in the top three for eight of sixteen categories. Taxonomy results primarily concern Gemma 2 9B; only ANOVA is replicated on 70B. Correlated readouts and ANOVA do not establish an internal intent mechanism. Appendix B.1 contains inconsistent sample-count arithmetic, so counts require data verification before reuse.

### Jayabahu: a probe's fitting target can be unidentified

**The Truth Was Never Gone: Perfect Aliasing in Compliant-Context Truth Probes**, inspected **v2, 8 October 2026**. [Primary text](https://arxiv.org/html/2609.10739v2); [author code](https://github.com/dylanjayabahu/perfect-aliasing).

**Reading scope:** task and identification argument, recommendations, limitations, and targeted supporting/instruction-pair material in appendices A, D and J. Not a complete all-appendix audit.

When truth and prescribed action coincide in the fitting data, the labels cannot identify which target a probe measures. Perfect in-distribution performance can therefore coexist with different out-of-distribution interpretations. The instruction-pair analysis explicitly separates instruction labels from lying labels. This supports competing-target tests, but does not prove that aliasing caused an external Apollo result. The main task is a controlled single-token game, not long-form omission. The paper acknowledges incomplete run and denominator records. Its algebraic complementary-label identity should not be misrepresented as independent empirical evidence of a hidden belief mechanism.

### Kumar: a robustness result can still measure prompt regime

**Pressure-Testing Deception Probes in LLMs: Scaling, Robustness, and the Geometry of Deceptive Representations**, **v1, 27 May 2026**. [Primary text](https://arxiv.org/html/2605.27958v1); [public implementation](https://github.com/techsachinkr/llm-deception-probe-stress-test).

**Reading scope:** methods, results, discussion, limitations, dataset/hyperparameter appendices and targeted implementation. Inspected `src/data_generation.py` at `96182ae904b8d66f077b3f1ea64ef7dfa06133fe`.

The paper reports stylistic fragility and improvements from augmentation, while acknowledging that nominal deceptive labels can describe correct responses. Sections 5.1/F.2 retain 29–41% such examples. Their similar scores support prompt-regime classification, not behavioral discrimination. Implementation inspection adds an important distinction: its D-MASK generator does not load original MASK. It constructs synthetic sycophancy from factual QA, uses one belief response, constructs a nominal wrong answer by prefixing the ground truth with “not”, and unconditionally sets `verified=True`. Consequently, its headline results should not be cited as validated original-MASK deception detection. The paper's limitations also disclose changed optimal layers in the augmentation comparison and incomplete separation of style from prompt-regime cues.

### Hollinsworth et al.: the latest strong falsehood benchmark excludes misleading truths

**Telling FIBS: Probes Effectively Detect Sabotage and Catch Unverbalized Deception**, inspected **v4, 20 October 2026**. [Current paper](https://arxiv.org/pdf/2610.12445); [official code](https://github.com/AlignmentResearch/caught-in-the-act-probes).

**Reading scope:** main problem definition, methods/results/limitations, with targeted architecture/optimization and elicited-belief appendices A and J. **The entire 119-page document was not read.** The v4 PDF was used because v4 HTML was unavailable during the audit.

The work combines diverse falsehood data with expressive or multi-layer probes. It already evaluates elicited beliefs under pressure, contrasting forced follow-up answers and self-report logits. Its strong results do not close this project's proposed gap: section 2.1 explicitly excludes misleading-but-true statements and other non-falsehood behavior. Section 4.1 acknowledges mostly off-policy, artificial training examples; estimated label noise is measured against an LLM judge. Low-FPR plots include in-sample thresholds, distinct from prospective calibration. Version discipline matters: v1 was titled *Caught in the Act*, and its older 98.8% headline should not be quoted as the current v4 result.

### Wang et al.: cross-model replication is materially weaker

**Evaluating honesty and lie detection techniques on a diverse suite of dishonest models**, Anthropic technical report, **25 November 2025**. [Primary report](https://alignment.anthropic.com/2025/honesty-elicitation/).

**Reading scope:** relevant setup, results and limitations; Appendix D read in full. Not a complete all-appendix audit of the entire report.

Appendix D reproduces the broad Apollo approach on Haiku 3.5. The Apollo-like mean-response setup gives about .696 AUROC on roleplaying transfer; last-token changes improve the best reported result to .816. Simple prompted lie classification is a strong baseline. This is evidence that a successful recipe on one checkpoint is not automatically a successful detector on another. A purported threshold-calibration discrepancy was encountered as a secondary-source lead, not verified as a finding of this report; it must not be attributed to Anthropic without the corresponding primary evidence.

### Smith, Chughtai and Nanda: construct validity is a research problem

**Difficulties with Evaluating a Deception Detector for AIs**, **v2, 16 December 2025**. [Primary text](https://arxiv.org/html/2511.22662v2).

**Reading scope:** relevant conceptual/empirical sections on strategic versus conditioned behavior, roleplay and belief changes; MASK critique and experimental details in A.1/B. Not a complete all-section audit.

The paper explains why instruction following, roleplay, changed beliefs and strategic deception are not interchangeable. Section 2.3 challenges the assumption that beliefs elicited in a neutral context necessarily persist under another prompt. Appendix A.1 notes that focal-proposition agreement can miss other lies in the response. These concerns motivate observable labels and blinded whole-response audit. They do not make the project infeasible. Recipient-state manipulations are especially useful because behavior contingent on attributed audience knowledge is a more discriminating test than pressure alone, although it still does not establish a causal internal belief representation.

### Taufeeque et al.: representation drift is a different threat model

**The Obfuscation Atlas: Mapping Where Honesty Emerges in RLVR with Deception Probes**, inspected **v2, 26 May 2026**. [Primary text](https://arxiv.org/html/2602.15515v2); [official code](https://github.com/AlignmentResearch/obfuscation-atlas).

**Reading scope:** targeted methods 4.1–4.3, selected results on obfuscated policies/representation drift, and associated interpretation. Not a complete paper/appendix audit.

The study compares original-model and trained-model readouts in a coding reward-hacking environment. Off-domain probe performance can deteriorate after RL while task-specific classification remains strong. Its follow-up extraction and layer aggregation differ from Apollo. Representation drift can occur without a probe penalty; this should not automatically be described as active concealment from a detector. The work motivates checkpoint-aware evaluation, but reproducing its RL program would be a separate project, not necessary for the current 8B omission study.

### Florin et al.: awareness-conditioned concealment of planted mistakes

**Deception by Omission: Language Models Knowingly Hide Their Mistakes**, **v1, 8 October 2026**. [Primary text](https://arxiv.org/html/2610.11351v1); [official code](https://github.com/Lucas-Florin/mistake-honesty-eval/).

**Reading scope:** complete main methods/results/discussion/limitations, sections 3–5; appendices A, F, H and I in full. These cover construction gates, capability controls, awareness/disclosure conditioning and evaluation/prefill awareness. Not all long generation/judge prompts or qualitative transcripts in B–E/J were read. No implementation-level reproduction audit is claimed.

The benchmark prefills synthetic mistakes, then compares tangential and remind-importance follow-ups. Separate judges identify disclosure and verbalized CoT awareness. Section 3.1 requires that the follow-up neither reveal the mistake nor imply that the user noticed it. Thus this is not an informed-versus-uninformed recipient manipulation. It generates new continuations across conditions rather than replaying identical summary wording. Appendix A requires corrections to remain useful in time, but does not measure their effect with an independent recipient or restoration intervention. The important overlap is user-interest/salience affecting disclosure, and the distinction between unnoticed mistakes and awareness-conditioned concealment. Limitations acknowledge off-policy mistakes, CoT faithfulness/redaction and unobserved latent awareness. Capability filtering conditions the evaluated population; it does not prove awareness during the original continuation.

### Huang et al.: target-supervised domain adaptation, not recipient-context evaluation

**Why Static Probes Fail: Context Determines Deception Feature Composition**, **v1, 20 October 2026**. [Primary text](https://arxiv.org/html/2610.27577v1).

**Reading scope:** complete preliminaries/methods/experiments, sections 3–5, and discussion/conclusion, sections 6–7. The inspected 11-page version contains no dedicated limitations section or appendix; no public implementation was independently verified.

The method extracts multiple orthogonal probe directions and adapts a gating network using ten labeled examples from each target domain, excluding them from evaluation. Consequently its headline is not zero-shot transfer. “Context” refers to benchmark/task domains, not independently varied recipient evidence. It does not evaluate fixed reports across recipient knowledge states or measure factual-restoration effects. Table 2's .94 recall uses a threshold with 5% FPR on neutral Alpaca controls; the mean target-honest FPR is .13, not .05. Some target FPRs are considerably higher. This directly motivates separating control calibration from pressured-honest specificity. The paper already covers multi-direction adaptation, so that should not become this project's novelty claim. Its broad geometric interpretations are not a demonstrated sender-belief mechanism, and reproduction details would require further verification.

### Hu et al.: truthful fragments already induce measured downstream false verdicts

**Lying with Truths: Open-Channel Multi-Agent Collusion for Belief Manipulation via Generative Montage**, [official ACL 2026 record](https://aclanthology.org/2026.acl-long.270/). Deeply read primary preprint **arXiv v2, 18 May 2026**: [text](https://arxiv.org/html/2601.01685v2); [author code](https://github.com/CharlesJW222/Lying_with_Truth).

**Reading scope:** complete main formulation/methods/experiments/discussion/limitations, sections 3–7; appendices A and B.1–B.6 in full. The official ACL title/authors/venue were verified, but the final 11 MB ACL PDF exceeded the browser tool's size limit. Exact line-by-line parity with the conference version is therefore unverified. Code was located but not independently audited.

Writer/Editor/Director agents optimize truthful-fragment selection and ordering. Victims process the same feed, form claims, and explicitly report a verdict on the target hypothesis. Appendix B.4 then evaluates five-victim majority vote and a downstream judge given raw evidence plus victim reports. Therefore neither truthful evidence causing mistaken conclusions nor measured downstream verdicts is a new contribution by itself. The experiment does not independently cross recipients' starting evidence or replay identical report wording across informed/stale contexts; it does not measure a fixed decisive-fact restoration against an irrelevant edit. Section 6 proposes monitoring defenses but does not evaluate latent probes. Main section 5.1 mentions false and unverified target rumors whereas B.1 describes false rumors; target-truth certainty needs care before dataset reuse. Self-reported verdicts/confidence are behavioral measures, not direct latent belief observations.

### JANUS: recipient-relevant summaries already receive goal-conditioned analysis

**JANUS**, **v2, 29 October 2026**. [Versioned primary text](https://arxiv.org/html/2606.10852v2).

Reading scope: sections 2.1–2.4, 3.1–3.3 and 4–5; prompting appendix C.2; judging and human validation D.1–D.5; fact inclusion and ordering F/G; robustness H/I. Other appendices and released implementation were not exhaustively audited.

JANUS already conditions summaries on recipients' needs, constraints and vulnerabilities and contrasts neutral with institutional objectives. Recipient context remains fixed within each pair while the summary is regenerated. Human validators see complete evidence and paired outputs and judge misleadingness/action pressure. This is not independent replay of actual readers with different starting evidence. No latent-probe evaluation or fixed decisive-fact restoration/irrelevant-edit factorial was identified in the inspected sections.

The overlap rules out claiming recipient relevance or goal-conditioned truthful communication as new. The bounded fixed-report/independent-evidence/restoration distinction remains. A feasibility warning also matters: fact retention is nearly complete, including Llama 8B adverse-fact inclusion of 99.17% under neutral and 95.83% under goal conditions in appendix F. Natural omission prevalence must be measured rather than assumed. These are that benchmark's inclusion rates, not our omission or deception rates.

## 2. What instruction-pair AUROC 1 establishes

Perfect held-out instruction-condition discrimination is a useful numerical and activation-extraction check. It establishes separation of those conditions on that finite panel under the recorded feature profile.

It does **not** establish:

- discrimination between observed honest and deceptive assertions under the same pressure;
- model knowledge or stable operational beliefs on new propositions;
- calibrated probabilities or controlled false positives on shifted contexts;
- generalization to new prompt wordings, domains, languages or checkpoints;
- numerical equivalence between quantized and BF16 activation profiles;
- detection of truthful omissions, reader harm, or second-order beliefs;
- causal use of the decoded information during generation.

Held-out facts under the same two instructions leave instruction interpretation as a sufficient explanation. Token rows are training observations, but they are not independent experimental units. Uncertainty and splits must respect original cases, fact duplicates and relevant scenario families.

## 3. The most informative immediate pressure comparison

Use the fixed [pressure pilot](./PRESSURE_PILOT.md), keeping the probe, layer, scaler and score aggregation frozen. Its 88 natural generations and at most 16 fixed-text forward passes are a feasibility panel, not a population reliability estimate.

The strongest behavioral question is:

> Within the same pressure prompt and case, does the frozen score rank naturally generated operational-belief-contradicting assertions above naturally generated honest assertions?

For each case that produces both categories, calculate pairwise concordance: the proportion of deceptive/honest response pairs where the deceptive response scores higher; count ties as one half. Report how many cases support this comparison. This fixes the incentive and avoids a pooled AUROC driven by case or topic differences. An absence of mixed-outcome cases is a feasibility limitation, not permission to relabel outputs.

Keep three findings distinct:

1. **Behavioral discrimination:** within-pressure ranking against audited assertion-consistency labels.
2. **Specificity:** false alarms on verified pressured-honest assertions at an independently frozen threshold, if one exists.
3. **Prompt sensitivity:** changing context changes the score of an identical forced honest response.

The pressured-honest subset is selected by behavior after prompting, so its comparison with neutral-honest outputs is descriptive, not a clean causal pressure effect. Fixed-text forcing isolates prompt-context sensitivity of the readout, but it is off-policy and cannot establish natural intent. If independent calibration is absent or too small, continuous scores remain primary. Do not derive a threshold from these evaluation labels.

A capable text monitor or generator self-report logit is an important comparator. FIBS in particular shows why latent scores should not be interpreted without this baseline. The tiny first pilot can establish data validity before adding monitoring machinery.

## 4. Narrow recommended research question

> **Do frozen deception monitors track the decision consequences of truthful omissions when recipient knowledge changes, or do they primarily track reporting incentives and disclosure cues?**

The proposed contribution is a controlled evaluation of **recipient-dependent monitoring and transfer**, not another universal deception direction. It remains aligned with Variant 2 while retaining the useful audience-belief motivation. The additional deep readings narrow the proposed distinction: Florin et al. already manipulate disclosure salience, Hu et al. already measure downstream false verdicts from truthful fragments, and Huang et al. already adapt probes across benchmark domains. The experiment must contribute the **joint control of fixed report wording, independently assigned recipient evidence and a fixed factual restoration**, together with held-out monitor evaluation. This combination is a candidate distinction from these three papers, not a guarantee that the broader literature has never studied it.

A coherent end-to-end study has these components:

- Natural summaries under neutral and favorable reporting objectives.
- A manipulation of what the sender is told the recipient knows, `K_s`.
- Independent replay under the evidence the recipient actually receives, `K_r`.
- The same saved report replayed with informed, absent-evidence and stale-evidence recipients.
- No-report, decisive-fact restoration and matched irrelevant-edit controls.
- Separate correct, wrong, deferred and invalid outcomes.
- A text monitor and an activation-based monitor with the same actual recipient information.
- Scenario/family-held-out evaluation and a second-reader replay.

Original sender activations cannot identify an independently randomized `K_r` introduced after generation. An actual-risk monitor must receive that evidence. A full-context activation reread is a monitoring representation, not evidence of what the original sender internally believed. Likewise, reader decision effects do not by themselves establish the sender's intent.

Falsifiable outcomes include failure of the frozen probe to distinguish actual behavior within pressure; score changes caused by prompt context despite fixed honest text; monitor failure to track the paired decision effect of restoring a fact; or no incremental predictive value from activations over a strong text baseline. Distinguish report-induced mistakes from an already mistaken recipient remaining wrong: the no-report condition measures the latter starting point. Distinguish an effective correction from added detail using the irrelevant-edit control. Negative results remain useful when the labels, controls and held-out evaluation are valid. A mechanistic second-order-belief claim would require substantially stronger intervention evidence and should remain a stretch objective.

## 5. Leads, corrections and unverified claims

Two current leads were found but not deeply audited: [Cross-Lingual Transfer and Shared Geometry of Deception Probes](https://aclanthology.org/2026.mrl-1.33/) and [Designing Effective Monitor-Based Interventions for Mitigating Reward Hacking During RL](https://arxiv.org/abs/2611.09828). Their existence is verified; detailed method or result claims should await full reading.

The paper previously referred to as “Berger” is by **Tom-Felix Thormann**: [Probing the Limits of the Lie Detector Approach to LLM Deception, v2](https://arxiv.org/abs/2603.10003v2). Do not reuse old-version sample counts without checking that revision.

This audit does not independently re-establish the detailed overlap claims for Kim, MonitorBench, TRACER or Shkolnikov in the earlier strategy. Those require the separate primary-source audit of omission and recipient-modeling work. Nor does this document verify admissions deadlines, compute policies or assessment requirements.

Claims to avoid include “first omission probe”, “first pressure-honest evaluation”, “AUROC 1 proves deception detection”, “the probe reveals intent”, “all MASK non-lies are honest”, “the model's belief is unchanged because neutral responses are consistent”, and “a failed frozen readout proves the model stopped representing the information”.
