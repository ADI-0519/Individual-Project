# Paper direction after the November priority search

20 November 2026. **Keep neutral-trained recipient-outcome monitoring as the core question.** Treat Apollo as a frozen comparator and the evidence-role assay as a supporting diagnosis. A candidate paper contribution concerns how recipient-state changes affect the reliability of communication monitors; generic prompt sensitivity, omission detection and evidence-sufficiency probing are already well occupied.

This is a literature and design decision record. It reports **no new experiment results**, changes no qualification gate and authorizes no stopped dependent fit. It supplements the [current research direction](RESEARCH_DIRECTION.md), [communication results](COMMUNICATION_TRANSFER_RESULTS.md) and [evidence-role interpretation](EVIDENCE_ROLE_DIAGNOSTIC_RESULTS.md).

## Three additional close papers

Versions and dates below were checked on the primary arXiv pages on 20 November 2026. Reading coverage identifies inspected passages; it does not claim a complete code audit or an all-appendix reading.

### Sadhu et al.: rule-conditioned monitoring

[What Do Compliance Detectors Read? An Audit of Activation Probes and Guard Models](https://arxiv.org/html/2608.16852v1), arXiv:2608.16852v1, 17 August 2026.

**Read:** methods in §2; benchmark validity in §3; generalization and detector comparisons in §§4–5; rule interventions and crossed tests in §6; Appendix A model, baseline and statistics setup; Appendix B.2 fixed-scenario counterfactuals and B.3 crossed construction.

The paper tests rule removal, substitution and scenario-fixed label flips. Its 200-template, 800-row crossed benchmark balances rules and scenarios so neither alone predicts the label. Cheap detectors struggle; fitting on the crossed data gives limited recovery. Lexical floors and failed superiority criteria remain reported.

**Collision:** neither a balanced 2×2 control nor showing that a monitor reads a nuisance variable is sufficient novelty here.

### Mohsin and Rahman: instructions versus decisions

[What Do Collusion Probes Read? Three Pitfalls in Evaluating Multi-Agent Collusion Monitors](https://arxiv.org/html/2611.10620v1), arXiv:2611.10620v1, 7 November 2026.

**Read:** task/contrast definitions in §§3–4; probe, baseline and calibration methods in §6; outcome leakage in §7; instructed-refuser and brief-removal tests in §§8.1–8.2; public-record ceiling results in §9; free-choice decision-probe results in §10.1. Relevant main-text tables were inspected; no appendix or implementation audit.

It separates instruction receipt, action and visible outcome. Refuser negatives, instruction removal, pre-vote scoring and information-matched text comparisons substantially change apparent monitoring success. A decision-trained probe retains some separation after removing the brief, with limited low-false-alarm recall.

**Collision:** pressure-versus-behavior specificity and pre-outcome timing controls are established requirements, not our new discovery.

### Han et al.: sufficiency scores and construction artifacts

[ChainCheck: Near-Perfect Evidence-Sufficiency Scores Can Conceal Sensitivity to Construction Artifacts](https://arxiv.org/html/2611.11965v1), arXiv:2611.11965v1, 9 November 2026.

**Read:** motivation and priority positioning in §§1–2; counterfactual cells and conditional metrics in §§3.1–3.2; independent-confirmation results and crossed edit-trace construction in §§11.1–11.2, including their tables. No checkpoint or private-data reproduction.

The audit separates operational chain sensitivity from edit sensitivity and tests artifact-only scorers. Nominal ceilings and positive crossed statistics can survive simple shortcuts; earlier patterns fail independent confirmation. Its three-cell contrast is explicitly conditional, while the later crossed construction removes additive passage effects without proving semantic reasoning.

**Collision:** another sufficiency probe, paired edit test or near-perfect AUROC would not establish the desired research contribution.

## The narrower seam with omission and reader studies

### Kim, Lee and Lee: transmitted evidence and downstream decisions

[Telling Incomplete Truths: Measuring Omission-Induced Misleadingness in Aligned Language Models](https://aclanthology.org/2026.emnlp-main.656.pdf), final EMNLP 2026 proceedings, 24–29 October 2026.

**Read:** stimulus construction and three-stage instrument in §§3.1–3.3; format and robustness analyses in §§5.4–5.6; coverage-versus-decision analysis in §5.7; alternative-reader and pre-generation-probe details in Appendices N–O.

Natural answers feed frozen readers, and pre-generation probes predict critical-fact transmission. Coverage and decision success are explicitly different endpoints. The design does not independently cross the same saved report with assigned recipient starting evidence, nor distinguish a report-induced mistake from one already present without the report.

**Collision:** natural communication, reader replay, coverage probes and their dissociation are already present. Our candidate difference must concern recipient-conditioned effects and monitor transfer.

### Thormann: misleading truths and truth probes

[Probing the Limits of the Lie Detector Approach to LLM Deception](https://arxiv.org/html/2603.10003v2), arXiv:2603.10003v2, 2 November 2026.

**Read:** manual-entry construction/validation in §2.1; activation, probe and held-out-type methods in §2.4; results in §§3.1–3.2; discussion and limitations in §§4–4.1.

Truth probes miss many authored misleading non-falsities; including misleading examples improves detection across held-out types. Listener-belief probing is explicitly proposed. Downstream recipient decisions are not measured, and §4.1 explicitly excludes intent-based Apollo probes from the experiment.

**Collision:** neither truthful misleadingness nor the idea of decoding a listener's belief is new. Evaluating intent probes against measured recipient-dependent effects is an empirical extension worth testing, not a proven unoccupied field.

## What the project should establish

The core question remains:

> Can a recipient-outcome readout trained under neutral communication transfer to reporting pressure on fresh cases, beyond prompt, report-text, full-context text and source-state controls?

The strongest supporting specificity test asks whether a frozen monitor responds appropriately when the same factual report has different consequences for recipients with different starting evidence. Preserve the distinctions among source truth, reporting incentive, sender-attributed recipient evidence, actual recipient evidence, saved report and observed decision.

Three different endpoints must remain explicit:

1. **Actual behavior:** predicting the particular reader's APPROVE/REJECT/DEFER choice. Erroneous choices are legitimate behavioral targets; predicting them does not establish that the sender anticipated them.
2. **Warranted decision:** what follows from the rule and evidence actually visible to the recipient. This requires independently checked evidence semantics; a lucky guess is not warranted knowledge.
3. **Communication harm:** whether receiving the report causes or maintains an error, and whether restoring omitted evidence corrects it. This needs counterfactual recipient measurements and separate validation. A wrong answer or missing fact alone is insufficient.

A separate, explicitly scoped behavioral study could forecast an imperfect reader without describing it as a reliable semantic judge. That distinction does **not** retroactively pass the existing qualification gate or license the currently stopped pipeline. Any change in target or protocol must be recorded before dependent data collection and fitting.

For recipient-effect analysis, retain no-report, original-report, decisive-restoration and matched irrelevant-addition conditions. Record correct, wrong, deferred and invalid transitions separately. Replaying one report in multiple contexts gives paired conditions, not independent documents. A monitor must receive any actual recipient context needed by its stated target; sender activations captured before a later independent context assignment cannot identify that assignment.

An output-only system necessarily cannot distinguish identical reports with different recipient conditions. That is an information limit built into the design, not evidence that activation monitoring is superior. Include recipient-context-only and matched-information text methods, and report the cost of direct reader simulation as a computational reference.

An intent detector need not predict a deceptive attempt's success. A privately assigned recipient packet cannot change the sender's earlier intent. Apollo-to-recipient-outcome evaluation is therefore cross-task transfer: packet invariance alone does not establish an Apollo false positive or invalidate its original target. Any claimed recipient sensitivity belongs to an explicitly contextual monitoring pass, not to sender states that never received that context.

New source families, paraphrases and report edits should not make the label recoverable from their construction. Audit source and split provenance, compare lexical and artifact-only controls, and preserve unsupported natural reports separately from verified truthful omissions. A genuinely new held-out result needs whole-case or whole-family separation and a frozen analysis, rather than another post-hoc slice of the current panel.

## Ranked recommendation

1. **Primary: neutral-trained recipient-outcome monitoring under pressure.** Establish valid, diverse behavioral targets and strong matched-information baselines before fitting or claiming an activation advantage. Keep the future communication set untouched under its existing protocol. A failed advantage is reportable if the comparison is informative and adequately validated.
2. **Supporting specificity: frozen Apollo and recipient-state contrasts.** Test behavior within the same pressure condition, rather than using pressure as a deception label. Separate natural sender-state scoring from a later forced-message reread; the latter is a monitoring representation, not evidence of original anticipation. A small Apollo evaluation on Thormann's released cases could bridge an explicit limitation, but would not by itself establish natural recipient harm.
3. **Appendix: evidence-role assay and reader failures.** Use these to explain measurement limits and motivate independent semantic validation. Further metadata variants or a generic sufficiency decoder should not displace the communication study.

The candidate contribution is a recipient-conditioned robustness evaluation whose targets and information access survive these controls. It is falsified as an activation-superiority claim if matched text controls perform equally well; it is uninformative about harm if recipient interventions do not yield reliable outcome differences. A null result should narrow the claim rather than trigger outcome-driven prompt or model selection.

This search finds important collisions and a plausible narrow seam. It is **not an exhaustive priority search, proof of firstness, novelty guarantee or prediction of a publishable result**.
