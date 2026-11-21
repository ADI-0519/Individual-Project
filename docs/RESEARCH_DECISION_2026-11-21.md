# Research decision: current belief versus a future communication goal

**21 November 2026. Design recommendation, not a completed experiment or a frozen execution protocol.** No new model calls, fits or CI runs were made for this reassessment. Two independent agents reviewed competing paper questions; a targeted follow-up checked the October belief/action paper. Graphify's existing local map was queried without rebuilding it. No Obsidian, Playwright or other app integration was used.

**Later design refinement, before any execution:** the [content-control design](FUTURE_GOAL_CONTENT_CONTROLS_DESIGN.md) and its exact draft bank supersede the 160-condition candidate-reversal proposal below. The revised assay keeps the source head's slot-b target, globally counterbalances physical names and adds identical archived goal content as a salience control. It has 224 possible conditions, with the same 32-condition neutral-first stop. No new model outcomes exist. “Place_b” is a metadata slot, not necessarily the second location mentioned in prose.

## What we actually have

| Evidence | Measured result | Interpretation boundary |
| --- | --- | --- |
| [Known-message transfer](KNOWN_MESSAGE_RESULTS.md) | Frozen neutral-trained belief readout: 28/32 unique sender inputs correct; 16/32 without the belief question. | Question-conditioned decoding across evidence channels, sharing one logical structure. Not spontaneous anticipation. The 64-row report duplicates hidden actual-delivery variants. |
| [Future-goal stress](BELIEF_GOAL_STRESS_RESULTS.md) | Positive future goal scores above negative future goal on all 32 unique inputs; mean probability difference +0.024799. AUROC .914063 versus .757813. | Current fictional target fixed, but negation, wording and length remain confounded. No answers were generated under these goals. |
| [Natural omission pilot](NATURAL_SELECTION_PILOT_RESULTS.md) | Four agent-agreed supported omissions across three cases; one omitted report received the restoration/placebo comparison. | One compelling controlled example; across the entire addition panel, decisive edits changed five decisions and irrelevant edits four. Human review is incomplete. |
| [Fresh reader qualification](RECIPIENT_RISK_V2_RESULTS.md) | Full-source answers correct in 19/36 conditions. Dependent reports and fits stopped. | A valid control failure. It prevents the planned reader-risk inference; it does not invalidate all belief-probe experiments. |

There is no established latent advantage on natural omissions, no full reproduction of Apollo's original 70B study, and no demonstrated publishable contribution yet. Software checks cannot establish those claims. Some work went into debugging and repeated engineering review; that should not displace the next scientific comparison.

One additional **post-hoc, CPU-only** reanalysis of the saved goal scores finds mean positive-minus-negative-goal log-odds changes of .056593 for current-label-0 inputs and .142983 for current-label-1 inputs (16 unique inputs each). The difference is positive within all four wording groups. The observed shift is therefore not numerically identical across labels; this does not isolate a semantic goal effect. Raw predictions SHA256: `1b02bcfcedf1db18152cc57c538eb3bf1922a783137d7b213c5ac7e445cec68f`. These are development observations, not confirmation or independent replications.

## The literature narrows the possible contribution

These comparisons use primary methods/results and stated limitations, rather than treating abstracts as evidence of an unoccupied gap.

| Primary work and reading coverage | Already established | Consequence for our claim |
| --- | --- | --- |
| [Goldowsky-Dill et al., Apollo](https://arxiv.org/html/2502.03407v1), §§4.1–4.2, 5.1–5.2 and Appendix A.1 | Honest responses in deception-related settings can score highly. Roleplaying training reduces this association relative to instructed pairs. | Pressure sensitivity alone is a replication, not our novelty; our instructed-pairs adaptation cannot stand for every Apollo probe. |
| [Bortoletto et al., *Brittle Minds, Fixable Activations*](https://aclanthology.org/2025.findings-emnlp.1226.pdf), §§3.1–3.4, prompting results pp. 5–7, Appendix A.1 | Other-belief readouts are brittle under random tokens, competing belief statements and temporal/initial-belief prompting. | Generic belief-probe prompt brittleness is occupied. |
| [Schouten et al., *Truth-value judgment*](https://arxiv.org/html/2404.18865v1), §§3.2, 4.1–4.2 | Irrelevant/corrupted premises can move truth probes substantially. | Adding a goal sentence is insufficient: control for comparable irrelevant context. |
| [Zhu et al., *Language Models Represent Beliefs of Self and Others*](https://arxiv.org/html/2402.18496v3), §§3.1–3.2 and task-transfer/intervention results | Belief decoding and transfer across social reasoning tasks. | Our 87.5% transfer is supporting evidence, not a first conceptual result. |
| [*The Belief–Action Gap in Language Model Agents*](https://arxiv.org/html/2610.17444), §§2–3, 5–6, Appendices D.3/F.2 | State decoding, explicit reports and action consistency dissociate in DoorKey; decoded information is not automatically a belief. | Neither decoding/behavior dissociation nor available-versus-used information is a sufficient novelty claim. |
| [Smolin and Wilder, *Beliefs and Behavior in Language Models*](https://arxiv.org/pdf/2609.07943), latent-variable methods and instructed-payoff/frame results | A latent variable inferred from outputs predicts decisions; framing changes inferred decision thresholds. | Distinguish an activation readout from a behavior-inferred belief or utility. |
| [Slater et al., NCP-ExploreToM](https://arxiv.org/html/2606.31916), §§2.1–2.5, discussion and §5 | Agents pursue first/second-order belief goals through simulated actions. | Merely adding a goal to a second-order-belief story is not novel. |

The independent reviews also checked [Kwon's inferred-intent preprint](https://arxiv.org/html/2607.03598) and [Kamdar's dynamic-ToM author report](https://harshilkamdar.github.io/2025/12/10/dynamic-tom.html). The latter is a research blog, not a peer-reviewed paper. The separate [omission literature review](FRESH_FEASIBILITY_DESIGN_REVIEW_2026-11-21.md) remains relevant; restoration and recipient-information interventions already have close precedents.

## One question to test next

> Does a frozen readout of what a sender thinks a reader currently believes systematically follow the sender's desired future reader belief, beyond matched lexical/context effects, including cases where the model correctly answers the current-belief question?

The inspected papers did not contain this exact controlled test. That is a candidate gap, not proof of firstness. The proposed contribution is **specific failure under a controlled distribution shift**, not discovering second-order beliefs or showing deception. This preserves the original interest while avoiding dependence on an unreliable separate reader as gold.

This is a proposed change in emphasis from the supervisor's deception-probe/omission extension. Describe it explicitly when discussing scope with her. A belief readout is not the Apollo deception probe. Keep the existing Apollo adaptation as the classic component; do not claim this small assay alone completes the assigned extension.

## Bounded next assay

Prepare **16 fresh scene units, two candidate locations, five contexts: 160 conditions maximum**. Use eight scenes per evidence channel and multiple fresh wording groups; disclose shared logical structure. Include the same queried object and a distractor object in every version. Counterbalance names, location order, world state, attributed current belief and which object is queried.

The five contexts are role-neutral, literal future goal A/B for the queried object, and future goal A/B for the distractor object. Use direct location names, avoiding the old negation contrast. Match the scaffold, target-location mentions and sentence structure. Document residual tokenizer-length differences before scoring; do not call imperfect matching exact. No future report has been delivered, and the future goal supplies no present evidence. Independently inspect every prompt and stipulated label before execution; human review is still required before a confirmatory interpretation.

Keep the source readout, scaler, normalization, zero-based block 16 and threshold .5 fixed. Capture the original trained feature position before generating a greedy current YES/NO answer, capped at 12 tokens. Freeze the exact prefix, answer instruction, parser and model settings before collecting anything. Do not move capture position or fit a replacement head after inspecting scores. The old head was trained on a proposition about the second named location: first-location candidate questions impose an additional, unvalidated transfer requirement. Failing that reversal would not invalidate the existing second-location result.

**First collect only 32 neutral conditions.** Proposed feasibility floor: at least 24/32 correct for both the frozen probe and the explicit answer, with each candidate orientation at least 12/16 and each evidence channel at least 10/16 on each measure. Report their crossed cells too. Invalid answers count as failures. These are practical qualification thresholds, not statistical evidence. Finalize them with the exact materials before inference. The earlier direct-answer result was only 44/64; this gate could genuinely fail. If it fails, stop this assay without collecting the remaining 128 or altering the same bank to rescue it.

If qualification passes, collect the remaining 128 conditions once. No additional reader model, free-form report, new fit, layer sweep, threshold search or automatic expansion is needed.

For a candidate claim about A let orientation s=+1, and for B let s=-1. With m the frozen score logit, calculate per scene the candidate-averaged contrast:

`D = mean_candidate s * [(m_target-goal-A - m_target-goal-B) - (m_distractor-goal-A - m_distractor-goal-B)]`.

This asks whether semantic relevance adds goal-aligned movement beyond the location-word/control effect. Report all scenes, each wording/channel, raw margins, AUROC, fixed-threshold class recalls, correct answers and invalids. Estimate case variation over 16 scenes, never 160 independent examples; shared wording limits wider generalization. Predeclare the exact uncertainty calculation before execution.

For the behavior-correct specificity diagnostic, include only scenes whose explicit current answers are correct in all four contributing goal contexts under both candidate orientations: eight correct answers per qualifying scene. This selected subset is not an unbiased population effect. Report its size and excluded cases; too few qualifying scenes cannot support the proposed dissociation. A nonsignificant behavioral difference does not establish equivalent behavior. The 160 conditions each require a capture plus an answer, not necessarily one computational forward pass.

## What would justify continuing

A goal-aligned effect surviving candidate reversal, distractor subtraction, fresh wording and correct-answer pairs would justify a larger held-out study and an equivalent neutral-trained readout on a second checkpoint. Cross-model confirmation requires the appropriate model-specific readout; do not copy incompatible head weights. It would still not establish an internal causal mechanism or spontaneous deception.

If controls remove the effect, report a lexical/context explanation. If it mainly accompanies wrong answers, investigate task-reasoning failure rather than claiming probe-specific leakage. If only the threshold changes while ranking survives, report that narrower sensitivity. Do not keep searching this development bank for a favorable alternative endpoint.

Keep the failed reader-dependent pipelines stopped. CREATE is useful later for controlled precision comparisons or justified replication, but **is not a prerequisite for this local screening assay**. No broad HPC allocation is justified merely by wanting a publishable result.
