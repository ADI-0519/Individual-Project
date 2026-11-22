# Main-study novelty audit: 22 November 2026

This is an independent agent's primary-source literature review and design
critique. It contains no new model measurements, human annotations or accepted
main-study results. It does not change the case bank, analysis policy or
execution gates. Read it alongside the [specification](MAIN_STUDY_SPECIFICATION_V1.md),
[decision brief](MAIN_STUDY_DECISION_2026-11-21.md) and
[DeceptionBench source audit](DECEPTIONBENCH_SOURCE_AUDIT.md).

## Decision

Finish the matched fitting-support study. The inspected papers do not report
its exact comparison: equal-budget one-step belief probes trained with versus
without belief-preserving cases, followed by untouched two-message correction
quartets and a second model with separately fitted heads. That is a bounded
search finding, not proof of firstness or a publication prediction.

The broad ideas are already established: decoding other agents' beliefs,
second-order probing, conditional sensitivity to information access, probe
dependence on training correlations, coverage followed by logical-composition
transfer, and failures of belief transfer. Merely
adding missing cases and repairing familiar examples would be weak standalone
novelty. A substantive, reproducible result about the limits or success of the
specified transfer is the candidate contribution.

The defensible question is:

> At a fixed fitting budget and fixed activation capture, does coverage of both
> belief-changing and belief-preserving evidence improve prediction of
> sender-attributed reader belief on unseen correction sequences, and how does
> that result compare with fresh one-step cases and strong text/output methods?

The target is fictional attributed belief content under explicit update rules.
It is not the actual reader's private belief, deceptive intent, a model's
spontaneous anticipation, or a causal neural mechanism. Those distinctions are
part of the contribution's scope, not minor caveats.

## Primary sources actually inspected

The reading below covered the relevant methods, results and appendix passages,
not just abstracts. Coverage is stated per item; it is not a claim to have read
every page of every paper. Public source versions were reopened on 22 November.
No upstream experiment code or cached Python objects were executed.

### Training correlation and measurement validity

**Ayal Klein (2026), [Decodability Is Not Validity: Evaluating Nuisance Sensitivity
in Language Model Concept Probes](https://arxiv.org/html/2611.07805v1).**
Read Sections 2–5, Appendix A's edit validation, A.1's human-label checks, and
Appendix B's fixed-method/last-token comparisons. The official
[metadata](https://arxiv.org/abs/2611.07805) verifies a 5 November 2026 submission;
this is a real current preprint, not a future placeholder.

Correlated versus decorrelated training, nuisance-erased probes, and
development-only nuisance-aware selection are already evaluated. High AUROC
does not establish selective measurement. Important methodological distinctions
are the paper's descriptive test-selected equivalence comparisons versus its
development-selected method comparisons, and its acknowledgement that accepted
LLM edits may still change the target's strength. Our formal transition rules
avoid that particular sentiment-edit ambiguity only if the prose expresses the
rules correctly. The new question must concern held-out correction transfer;
the generic result that changing training correlations changes a probe is
already occupied.

### Second-order probing already exists

**Xinyang Li et al. (ICML 2025), [From Black Boxes to Transparent Minds:
Evaluating and Enhancing the Theory of Mind in Multimodal Large Language
Models](https://proceedings.mlr.press/v267/li25bj.html), GridToM.**
Read the [full-text](https://arxiv.org/html/2506.14224v1) Sections 3.2–4.3,
Appendices B.1–B.2, C.2, F, G and I. The PMLR record confirms the ICML paper.

GridToM includes second-order true/false-belief tasks and logistic probes on
final-token attention-head activations, with second-order probing results in
Appendix G. Its nested task compares correct and incorrect belief pairs;
therefore neither second-order labels nor their linear decodability is new.
Appendix I distinguishes generated factorial instances from underlying maps and
describes human checks. Similarly, our 154 contextual groups and 5,888 prefixes
must not be described as thousands of independent reasoning situations.
The inspected experiment does not supply our matched fitting-support treatment
and correction-sequence holdout.

### Nested interventions, access sensitivity and elicitation

**Oliver Steele, Jiangtao Wen and Yuxing Han (2026), [Belief–reality separation
lives in routing over a shared value slot in language models](https://arxiv.org/pdf/2607.11945v2).**
Read Sections 3–4.3, relevant Section 4.4 passages, and Appendices A–C.
[Metadata](https://arxiv.org/abs/2607.11945) verifies v2 dated 20 July 2026.
The HTML v2 endpoint returned 404; the 21-page PDF was accessible.

This preprint already distinguishes asserted from visibility-derived belief,
tests access-changing and access-irrelevant manipulations, and reports nested
belief interchange results. Appendix B's shared first-/second-order value slot
is a Qwen7B result; its distinct nested router uses a different 14B substrate.
These are not all cross-architecture nested results. Clause-order randomization,
two-value copy controls and chat/completion/scaffold comparisons matter. A
single explicit-question probe in our fixed chronological format cannot support
a broad account of when belief is computed. The paper's task is controlled
frame tracking, not natural communicative action; ours has the same limitation.

### Distribution transfer can change a belief subspace

**Nikhil Prakash et al. (ICLR 2026), [Language Models Use Lookbacks to Track
Beliefs](https://arxiv.org/html/2505.14685v3).**
Read Sections 3–6, Appendix H's fitting procedure and Appendices M–O; inspected
the query-time information-flow results rather than inferring a mechanism from
the abstract. Version 3 is dated 24 February 2026.

The work traces bindings, query-time answer retrieval and visibility-mediated
updates. Section 3 restricts the mechanistic experiments to behaviorally correct
cases. Appendix M reports similar lookback structure across CausalToM and
BigToM but significantly different subspaces at the final token. Appendix O
also reports failure on the three-character visibility condition. Thus neither
distribution-dependent belief subspaces nor a generic composition failure is
new. Our full-cohort predictive evaluation should retain failed verbal cases,
with a behavior-correct subset only secondary. Failure at one token/layer says
that this readout fails, not that no relevant information exists elsewhere.

### Earlier prompt-robustness and task-transfer precedents

**Matteo Bortoletto et al. (2024), [Benchmarking Mental State Representations
in Language Models](https://arxiv.org/html/2406.17513v1).**
Read Sections 3.1–3.4, 4.2–4.3 and relevant Appendix A.1/A.3/A.4 passages.
The study already compares residual-stream belief probing under random-token,
misleading-statement, explicit-initial-belief and time-specification prompts,
and investigates probe capacity using PCA. Prompt sensitivity of a belief
readout is therefore established precedent. Its results do not imply that a
failed fixed probe identifies absence of belief information. Our change in
fitting support and held-out update sequence must carry the empirical claim.

**Wentao Zhu, Zhining Zhang and Yizhou Wang (ICML 2024), [Language Models
Represent Beliefs of Self and Others](https://proceedings.mlr.press/v235/zhu24o.html).**
Read [arXiv v2](https://arxiv.org/html/2402.18496v2), Sections 3.1–3.2 and
4.1–4.3. This reading is explicitly v2, not an assertion that it is the latest
version. The paper already fits protagonist/oracle and joint belief probes and
transfers intervention directions between social-reasoning tasks. Comparing
belief and reality or applying a readout beyond its fitting task is not itself
our novelty.

### Prompt coverage is not replaced by more stories

**Valentin Noël (2026), [A Probe Direction Is a Property of Its
Prompt](https://arxiv.org/html/2608.13329v1).**
Read Sections 3–9 and 11. The
[official record](https://arxiv.org/abs/2608.13329) verifies 13 August 2026.

The paper crosses contrastive framings and distinguishes prompt variation,
model variation, implementation choices and surface-form controls. Its
evaluation-awareness estimator differs from our supervised belief classifier;
its numerical variance shares or required wrapper counts cannot be imported
into our design. The applicable lesson is that a common renderer defines a
conditional evaluation. Thirty narrative wrappers around one correction rule
do not justify arbitrary wording, model-scale or mental-state-generalization
claims. A second model tests replication of the stated effect; it does not
establish a general ranking of model capability.

### Apollo and the supervisor's deception brief

**Lennart Bürger, Fred A. Hamprecht and Boaz Nadler (2024), [Truth is Universal:
Robust Detection of Lies in LLMs](https://arxiv.org/html/2407.12831v2).**
Reopened Section 5.1, Appendix D and F.4; the root agent additionally read
Section 5.2 and Appendices E/F. The accessible revision is v2, 21 October 2024.

This is especially close prior art: classifiers trained on English affirmative
and negated statements are tested on unseen conjunctions, disjunctions and
German statements. Therefore coverage followed by composition transfer is
already an established research design. Our specific target, matched budgets
and narrative groups, redundant-evidence treatment, and correction quartets
distinguish the proposed experiment; the general method is not new. Appendix D
also separates ranking from a failed transferred bias. F.4 directly tests
incentivized-but-honest responses, but contains only six such responses; its
reported TTPD accuracy is 82±5%, LR 62±17%, and MM 100±0%. That small truth-probe
analysis neither settles Apollo's behavior nor establishes our desired
deception-monitor result.

**Benjamin A. Levinstein and Daniel A. Herrmann, [Still No Lie Detector for
Language Models: Probing Empirical and Conceptual Roadblocks](https://arxiv.org/pdf/2307.00175).**
Read Sections 4.4.1 and 4.6 in the accessible 21-page PDF. Its header is arXiv
v1, 30 June 2023; retain that version identity when discussing the supplied
2024 reference. Expanded training including negated examples still transfers
poorly to other negations in Section 4.4.1. Section 4.6 distinguishes information
in embeddings from its use by the underlying model and from the model's
credences. These objections are directly relevant to calling our predictive
classifier a belief measurement. The frozen study tests generalization of an
operational readout, and its labels do not establish the model's own beliefs.

**Hadas Orgad et al., [LLMs Know More Than They Show: On the Intrinsic
Representation of LLM Hallucinations](https://arxiv.org/html/2410.02707v2).**
Reopened Section 3.4's table, Section 4 and Appendix A.2; the root agent read
Sections 3–6 and A.4. These passages use the accessible October 2024 v2, while
the supplied brief cites the later publication year. Section 4 compares transfer
against a strong logit baseline and finds much apparent probe transfer is
already explained by logits. It also reports absolute AUROC, max(AUC,1−AUC),
allowing a reversed direction to look transferable. Our held-out evaluation
must preserve source orientation and frozen thresholds, with no test-label sign
choice. Same-model output probabilities and equivalent development calibration
are substantive controls, not optional finishing touches. This is correctness
probing; it does not itself validate a deception or nested-belief target.

**Nicholas Goldowsky-Dill et al. (ICML 2025), [Detecting Strategic Deception
with Linear Probes](https://proceedings.mlr.press/v267/goldowsky-dill25a.html).**
Read [arXiv v1](https://arxiv.org/html/2502.03407v1), Sections 3–4.2 and 5.2,
Appendix C, and D.5. The PMLR PDF endpoint was served as an unsupported
octet-stream and OpenReview required a browser challenge; method details here
are explicitly from the accessible v1, with publication verified by PMLR.

The original recipe uses Llama3.3-70B, tokenwise normalized activations and
response aggregation; instructed-pair training contains true off-policy text
under contrasting instructions, not behavior labels for generated deception.
The paper itself reports elevated scores on honest responses in deceptive
scenarios, morality/context confounds, aggregation failures, and sensitivity
of low-FPR recall to layer choice. Therefore pressured-but-honest false
positives are a reproduction/robustness question, not an undiscovered weakness.
Our 8B adaptation and separate belief-content heads must retain their distinct
targets. An explicit nested-question final-prefix readout is not a reproduction
of Apollo's response-level monitor or evidence of pre-communication anticipation.

### DeceptionBench helps as an external anchor, not a novelty shortcut

**Huang et al. (2025), [DeceptionBench: A Comprehensive Benchmark for AI
Deception Behaviors in Real-world Scenarios](https://arxiv.org/html/2510.15501v1).**
Read Sections 2.1–2.4, 3's human-validation passage and 4. The currently fetched
v1 HTML reports 420 human-checked interactions, three annotators each, 95.7%
unanimous agreement and 97.1% judge/majority agreement. These are the paper's
reported measurements, not an independently reproduced audit and not labels
for our future model outputs. Do not imply that the paper lacks human checks.

Its generative task and elicited `thought`/`response` judgments differ from our
fictional belief-content task. Generated `thought` text is a self-report, not a
direct observation of hidden belief. The pinned release audit establishes
150 scenarios and 900 static prompts, without saved responses/behavior labels.
L1 is not automatically honest, and L1/L2 rewrites are not a minimal
pressure-only intervention. New outputs need independently assessed labels.
The correct project is the Huang/Aries-iai benchmark, distinct from PKU's.

**Yao Huang et al. (2026), [Why Static Probes Fail: Context Determines Deception
Feature Composition](https://arxiv.org/html/2610.27577v1).**
Read Sections 4.1–4.2 and 5.1–5.3, including Table 1 and target-support policy.
The official page dates v1 to 20 October 2026. It already evaluates an
instructed-pair static probe on THU-DeceptionBench L2 and MASK, alongside
target-domain adaptation using ten labeled examples. Apollo-to-DeceptionBench
transfer and honest-target false positives are consequently occupied.
Target-adapted performance is not frozen zero-shot performance. The empirical
precedent is enough to constrain our novelty claim; its theoretical claims
are not assumptions of this project, and its reported numbers have not been
reproduced here.

## Controls to preserve before inference

These are recommendations about the existing study, not permission to tune its
held-out results. Any added analysis must be versioned before its relevant
outcomes are observed.

1. **Match fitting support, not just a headline row count.** Preserve the
   current paired narrative groups, unique N, nuisance counts and five genuine
   data allocations. Repeated copies or optimizer seeds cannot substitute for
   data coverage. Because each arm fits its own scaler, estimate the effect on
   the complete fitted pipeline rather than claiming a classifier-weight-only
   or neural-mechanism intervention.
2. **Keep absolute and joint outcomes.** Report both arms, complete-quartet
   accuracy, changing-pair success and preserving-pair failures. A 75% row score
   can coexist with a shortcut that solves no quartet. Confidence movement
   alone is not an erroneous belief verdict, and equal binary answers do not
   establish equal internal evidence strength.
3. **Diagnose the correction-format shift.** Primary I=M2 and secondary I=M1
   cases share two-message format but require different dependencies. Use that
   already specified contrast to assess whether failure is broad to correction
   format or concentrated in the harder state dependence. Event count, length,
   correction language and composition still change together versus training;
   this is not a pure causal isolation of composition.
4. **Run the strong prose baseline.** Keep Qwen's frozen evidence extraction
   and deterministic update calculation, plus same-model output probabilities
   with equivalent development calibration. Invalid extractions remain wrong
   in full-denominator accuracy. A symbolic oracle consuming source factors
   is a logic ceiling, not an independent text baseline. A frozen parser that
   extracts the standardized prose would additionally reveal how much can be
   solved without latent access; a perfect parser would defeat a necessity
   claim, not the matched-support research question.
5. **Avoid a word-count victory.** The paired TF-IDF baseline should retain the
   same fitting allocations. A low TF-IDF score alone would not demonstrate
   mental-state representation, because conditional update composition already
   exceeds a bag-of-words classifier's direct feature interactions. Conversely,
   high TF-IDF performance is evidence against a privileged latent interpretation.
6. **Keep the test untouched and denominators honest.** Freeze source bytes,
   thresholds, selection, parsing, calibration and exclusions before inference.
   Bootstrap authored groups with all dependent variants intact. Thirty
   correction groups are thirty contextual groups around one held-out mechanism,
   not thirty held-out reasoning mechanisms. Record incomplete/failed runs and
   do not select only verbal successes for the primary result.
7. **Separate precision and layer scope.** The fixed NF4 midpoint readout is a
   valid bounded system to study. A failure does not identify quantization as
   its cause or eliminate later-layer/nonlinear information. A CREATE BF16
   comparison needs a separately frozen policy and comparable hardware/run
   conditions; more compute is not a substitute for a valid contrast.

## What would and would not support the candidate claim

| Observed pattern after the frozen evaluation | Interpretation |
| --- | --- |
| Full coverage improves fresh one-step cases but not correction quartets | Coverage repairs familiar support; transfer remains limited in this pipeline. A useful dissertation result, with modest standalone novelty. |
| Full coverage improves correction quartets, remains accurate on genuine updates, and the effect repeats on Qwen | Supports transferable benefit of the specified fitting coverage on this correction sequence. Stronger candidate empirical contribution; still conditional on renderer, models, precision and cohorts. |
| Both arms are indistinguishable with useful absolute scores | No demonstrated benefit of the support intervention at these budgets. The expected improvement is unsupported. |
| Both arms fail but the prose/output methods succeed | Supports a limitation of the selected readout and fitting policy, without establishing absence of model competence. |
| Probe and prose/output methods all fail | Does not localize failure to probe support. Report task/elicitation/system limits and extraction errors. |
| An apparent gain disappears after matching allocation, correcting labels or applying full-denominator scoring | The claimed effect is invalidated; retain the correction and do not present the earlier gain as evidence. |

The prespecified group interval and all five allocation results should determine
the strength of evidence. Do not invent a post-hoc significance/size threshold
or call a null result equivalence without an equivalence design. Good software,
many tests and an attractive figure do not by themselves supply an empirical
research contribution.

## Agent-audited exploratory amendment review

The separately proposed [execution amendment](MAIN_STUDY_AGENT_EXECUTION_AMENDMENT_2026-11-22.md)
correctly preserves the blank human ledger, names the agent audit, uses a
different freeze schema and marks its results exploratory/non-confirmatory.
Its matched estimand, conditional uncertainty, development exposure and
fixed-renderer limitations are appropriately explicit. One agent's independent
prose interpretation remains useful evidence, with correlated author/reviewer
blind spots possible.

Subsequent human review cannot retroactively make already observed exploratory
test outcomes confirmatory. Replaying those same exposed cases under the human
route does not restore an untouched test set. If observations motivate changes,
any subsequent confirmatory claim requires separately frozen fresh evaluation
material or must retain its exploratory designation. The existing original
human-review route's continued availability must not imply otherwise.

## Relationship to the allocated Variant 2

The supervisor's brief asks for deception-probe reliability under shifts, then
an omission experiment conditioned on recipient knowledge. The new core study
is a narrower construct-validity and sequence-transfer investigation of a
separate attributed-belief readout. It is scientifically coherent but is a
material scope change, and the supervisor should see that clearly in the next
progress report. Keep Apollo as the reproduction anchor and DeceptionBench as
a possible externally labeled transfer anchor. Neither becomes a belief label
source just because it concerns deception.

A later communication extension would need a validated current-belief
instrument, sender decisions and independently assigned actual reader evidence
or outcomes. Merely adding pressure or desired-future-belief text to this query
would not establish spontaneous anticipation or intent. Completing the current
discriminating comparison is more valuable than restarting another such pilot.

## Masters relevance: verified expectations, not guarantees

[Imperial's 2027 Computing AI/ML MSc listing](https://www.imperial.ac.uk/study/courses/postgraduate-taught/computing-artificial-intelligence-msc/)
states a First-Class degree with a substantial computing component and a
substantial individual AI research project. Its
[current department route](https://www.imperial.ac.uk/computing/prospective-students/courses/pg/msc-specialist-degrees/)
specifies computing study equivalent to two undergraduate years. The central
listing's indexed official text gives Round 2's 6 January 2027 deadline; direct
fetch returned 403. Older departmental routes contain stale intake text, so
do not mix their dates with the current central cycle.

[UCL's 2027 Machine Learning MSc](https://www.ucl.ac.uk/study/prospective-students/graduate/courses/machine-learning-msc)
requires at least a quantitative upper second-class degree, comfort with linear
algebra/calculus, and two references. It asks applicants to explain preparation,
programming experience, course fit and professional plans. The fetched page
inconsistently shows an open date range and a later closed notice; application
status must be checked in the live portal when applying.

Neither inspected course lists an accepted paper or HPC usage as an entry
requirement. A concise methods/results report, auditable figure, clear statement
of the student's own decisions, and a supervisor able to discuss that work are
credible application evidence. Claiming inevitable publication, full marks or
admission would exceed the evidence. Grades, mathematical preparation, references
and course fit remain important alongside this project.

## Audit limits and provenance

The existing Graphify graph was queried first for repository relationships; it
contains 727 nodes and returned truncated older material. It was not used as
evidence of current source code or the literature's conclusions. No semantic
graph rebuild, paid API or new app plugin was used. Repository documents were
read directly and primary paper pages were browsed; source conclusions above
are distinguished from this audit's design recommendations.

The November and July 2026 references were checked against actual arXiv records,
not rejected merely because their identifiers were unfamiliar. The separate
[Same Answer, Different Future record](https://arxiv.org/abs/2611.07913) names
Weihan Li, Junhao Wu, Xinlei Chen and Tianshi Zheng, v2 11 November 2026;
it must not be confused with Baishi Li et al.'s different paper cited by Klein.
Only its metadata was newly checked here; no new result from it is asserted.
This remains a targeted prior-art search, not an exhaustive literature census.
