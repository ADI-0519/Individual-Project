# Complementary-candidate readouts: methods audit

20 November 2026. This note records primary-source reading, the rationale for the proposed measurement change, and its limits. It reports **no results from the complementary-candidate experiment**. The execution protocol and saved manifests determine the actual implementation, frozen choices, data boundaries and run status. Earlier failed and exploratory results remain evidence; this experiment does not replace them.

The intended question is whether matched alternative propositions improve direct linear measurement of a fictional perspective, particularly the normative nested target **S thinks R believes p**. It is not a test of deception intent, anticipation of an unknown future message, or spontaneous belief computation. The broader natural-report forecasting project remains separate from this diagnostic.

## Primary sources actually inspected

The scopes below describe passages read for this audit or the preceding methods audit, rather than claiming exhaustive readings of every appendix. Primary methods and limitations were inspected; these are not abstract-only summaries. Descriptions stay below 200 derived words per source.

### Prakash et al.: Language Models Use Lookbacks to Track Beliefs

[arXiv 2505.14685v3, 24 February 2026](https://arxiv.org/html/2505.14685v3). Read main Sections 3.3–6.1, including causal mediation, the hypothesized causal model, binding/answer lookbacks and visibility interventions; also the relevant setup and Appendix D model evaluation. Not every causal-patching appendix was audited. [ICLR 2026 publication listing](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0d1941f73833eb107939f62d323d6cc0-Abstract-Conference.html).

The experiments localize reference/binding information at earlier tokens and retrieval at later question tokens. They include matched token values, reordered stories and interventions that separate pointers from retrieved payloads. Section 6.1 is especially relevant: patching either address or pointer alone can fail, while patching both recovers an effect. A missing effect at one location need not establish absent information. The mechanistic study selects correctly answered samples and mainly studies much larger capable models; its mechanism is not established in our NF4 8B model. This motivates localization and access controls, not borrowing its causal conclusion for our probes.

### Orgad et al.: LLMs Know More Than They Show

[arXiv 2410.02707v4, 18 May 2025](https://arxiv.org/html/2410.02707v4). Revisited Sections 3.2–3.3 and 4, with Appendix A.2 exact-answer extraction details. This is a targeted methods/results reading, not a new exhaustive audit of every error-analysis appendix.

The study compares prompt, response and exact-answer token positions for predicting generated-answer correctness. Position matters materially: selecting the semantic answer span can outperform generic response positions. Its labels concern correctness, not deception or nested beliefs. Some cross-task analyses choose position/layer using the target task and report orientation-insensitive AUROC, so those diagnostic results should not be equated with a fully frozen deployment test. Our prospective experiment should fix extraction positions and orientation before fresh outcomes. Exact-answer localization also uses extraction procedures and exclusions, rather than providing a universally label-free location rule.

### Burns et al.: Discovering Latent Knowledge in Language Models Without Supervision

[arXiv 2212.03827v2, 2 March 2024; ICLR 2023 paper](https://arxiv.org/pdf/2212.03827). Inspected Section 2.2, experimental setup and Section 5.1 limitations in this version. An older OpenReview reference PDF was also accessed; the current arXiv version is the citation used here. This was not a complete implementation audit.

Contrast-Consistent Search constructs mutually exclusive candidate answers, normalizes their feature distributions separately, and optimizes confidence and complement consistency without labels. Inference averages the two implied answer probabilities. Its reported evaluation selects the better test-set orientation; we must instead fix orientation using training. The paper explicitly notes that a successful readout assumes the model both can and actively does evaluate the relevant input. Complementary location propositions therefore have clear precedent. Our supervised paired-margin/difference approach is related to CCS, but does not reproduce its unsupervised objective or establish a new consistency principle.

### Farquhar et al.: Challenges with Unsupervised LLM Knowledge Discovery

[arXiv 2312.10029v2, 18 December 2023](https://arxiv.org/html/2312.10029v2). Read Sections 2–4, discussion/limitations and relevant Appendix B training and prompt-construction details; not every supplementary result figure.

The theoretical results show that CCS consistency alone does not identify knowledge. Controlled experiments introduce random-word features, explicit character opinions and implicit opinions; unsupervised methods can recover these instead of the intended label. Prompt-template sensitivity is tested separately. This directly motivates distinguishing world truth, reader perspective and sender attribution, rather than treating complementary outputs as evidence of the intended construct. Our supervised target avoids the particular unsupervised identification objective, but still needs independent validation of perspective specificity and resistance to lexical shortcuts. Antisymmetry is a property of the estimator, not evidence that the model uses the decoded information.

### Klein: Decodability Is Not Validity

[arXiv 2611.07805v1, 5 November 2026](https://arxiv.org/html/2611.07805v1). **Recent preprint.** Read Sections 2–5 and targeted Appendices A–B on edit validation, robustness, selection and last-token versus mean-pooled readouts. The complete released implementation was not independently audited.

The study separates target AUROC from nuisance sensitivity conditional on the target, and evaluates target-changing, nuisance-changing and placebo edits. It compares train-only nuisance projection with other linear extractors and separates descriptive test-selected equivalence from development-selected methods. Accuracy-matched probes can differ in what else their scores track. Limits include named-nuisance coverage, selective judge-filtered edits and imperfect edit validation; score-class-gap normalization is unstable for weak probes. Our factorial checks adapt this measurement logic. They do not inherit a guarantee of construct validity, and should use raw score changes or a frozen training-scale normalization when the target separation is weak.

## Why complementary candidates are a reasonable next measurement

The current diagnostic uses one candidate location. Presenting both locations in otherwise identical scenes lets us compare a proposition with an explicit mutually exclusive alternative without introducing ambiguous negation. The candidates must remain propositions for later assessment, not new observations or messages received by either fictional agent.

For the frozen existing head, let its raw decision margin be f(h). The paired score is:

`s_pair = f(h_B) - f(h_A)`

Orientation must be fixed from training. Preserve the original candidate margins, feature identities and transformations, and apply the same procedure to world, reader and nested heads. Do not reverse the score after looking at fresh labels. Using raw margins keeps the contrast interpretable on the fitted head's scale; it does not provide calibrated belief probabilities.

**The existing row-L2 transformation limits the cancellation argument.** If the head first standardizes each activation and independently normalizes each row to unit length, the A and B denominators can differ. Subtracting margins cancels the fitted intercept exactly, but does not guarantee cancellation of a common additive activation/story component. The correct description is *paired relative scoring*, not isolation of a pure belief component. Candidate-specific lexical effects and scene-by-candidate interactions can remain.

## Strictly linear supervised difference readout

A separately identified method can use raw activation differences:

`d = h_B - h_A`

`g(d) = w^T (d / scale_train)`

Use training-only per-dimension scaling, no per-row L2 normalization and no intercept. Symmetric training augmentation with `(-d, 1 - y_B)` preserves complementary orientation and a zero difference mean; zero-variance dimensions require an explicit fixed handling rule. Validation may select regularization from the prespecified grid. No fresh examples may determine scaling, orientation, regularization or a new threshold.

This score is linear in the two activation vectors. If each representation contained only a shared additive scene component plus a candidate-only component, subtraction would remove the shared scene component. It cannot externally reconstruct XOR by applying a nonlinear classifier to separately decoded world and receipt bits. However, successful prediction can still reflect a learned scene-by-candidate interaction or a task shortcut. It is evidence about available representation, not proof of model reasoning or use.

Antisymmetry, `g(-d) = -g(d)`, is enforced by this design. Swapping the candidate order is an implementation check. It is **not an independent successful prediction**, an empirical discovery of logical consistency, or evidence of a nested-belief mechanism. Count one paired decision per scene and account for repeated/private-access-identical inputs.

## What the factorial controls can establish

Use the authored labels separately: world location W, actual reader access a, and the trusted access record b available to the sender. For the place-B proposition, the prior target is W when access is present and the initial location otherwise. Thus reader content depends on a, while the normative nested content depends on b.

The relevant contrasts are:

- Change b at fixed W and a: the nested target changes.
- Change a at fixed W and b: the nested target stays fixed. In the sender view the two inputs must be byte-identical; the hidden actual-reader target is structurally unidentifiable there.
- Change W and b together: the nested target stays fixed despite changes to both ingredients.
- Evaluate disagreement cells separately: nested versus world, and nested versus actual reader. Overall accuracy alone cannot identify the intended perspective.

World and receipt are necessary ingredients of the nested target, so indiscriminately projecting them away as nuisance variables would remove legitimate information. Actual private access is a nuisance for that target. Report both signed and absolute score changes; opposite changes can cancel in a signed mean. Candidate-only, world-copy and train-fitted paired text controls are needed alongside the activation readout.

## The two evidence channels and their limits

The fresh [panel construction](../src/latent_probes/belief_contrast_panel.py) includes observation-access and trusted-message-access channels. The latter is **not** a condition in which every person necessarily receives a truthful current-state update. The sender knows the reader missed the transfer, sends a truthful current-location note, and then accepts an independently varied receipt. Actual private receipt/read/trust and the receipt supplied to the sender can disagree.

When b=0, the sender accepts a failed-delivery/no-reading receipt and attributes the initial location to the reader; nested content can therefore differ from current world truth. When b=1 but a=0, a false positive receipt makes the sender attribute the current location even though the actual reader remains uninformed. These cells retain the world/reader/nested distinctions. The attempted note exists regardless of delivery success: do not describe b=0 as proving that no note was sent.

The channels share the same Boolean gold function. Transfer between them tests fresh wording and evidence-channel variation, not broad reasoning-family generalization or a newly established algorithm. Different entities and wording also vary across channels, so a difference cannot be attributed solely to the evidence operation. These are agent-assisted authored drafts requiring human review; no independent human validation is claimed.

## What a positive or negative result would mean

A promising result would be frozen or train-only paired readouts working on genuinely new wording, including disagreement cells, with the expected target-changing contrast and limited nuisance sensitivity. All observed wording banks are development evidence for subsequent choices. Fresh scoring must follow a written freeze; complete cubes stay together, and uncertainty should acknowledge wording groups and duplicate sender-view inputs rather than treating all rows as independent.

A positive result supports **direct perspective-selective decodability under the stated candidate context**. The candidate itself is an implicit semantic query. It does not establish spontaneous nested reasoning, causal belief tracking, deception intent, an unknown-future-message forecast, or improved generated behavior. A null still bounds this representation location, model, readout and task; it does not establish absent beliefs.

Token localization remains a sensible later comparison if needed: use a small prespecified set of semantically meaningful positions after all necessary facts are available, rather than selecting the best of a large token/layer scan on fresh data. Transport/alignment is less attractive as the immediate repair because paired high-dimensional mappings can overfit scene identities, and successful alignment alone would not distinguish a rotated code from additional question-induced computation.

BeliefScope and roleplay papers are omitted here: they were not independently verified in this audit. No new-mechanism or novelty guarantee is claimed.
