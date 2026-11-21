# Literature addendum: belief readouts and future communication goals

**21 November 2026.** This addendum records a bounded primary-literature audit
completed while the crossed neutral validation ran. It does not amend any frozen
protocol. The two earlier [content-control design](FUTURE_GOAL_CONTENT_CONTROLS_DESIGN.md)
and [research decision](RESEARCH_DECISION_2026-11-21.md) were read first. The
three additional papers below were inspected beyond their abstracts, including
their methods and relevant appendices. No firstness claim follows from this search.

## Second-order belief as a communication control signal

**Zhu, Rim and Pustejovsky, August 2026,
[Mind the Gap: Theory-of-Mind-Grounded Friction for Epistemic Alignment](https://arxiv.org/html/2608.30719v1).**
Reading coverage: Sections 3–7 and Appendices B–D.

Their schema separately records speaker intent, addressee interpretation and
each participant's model of the other. Removing second-order channels reduces
misunderstanding recall from about 65% to 26%; second-order supervision also
improves intervention F1 over lower-order alternatives. Thus, the broad claim
that second-order belief improves communication monitoring is already occupied.

Their representation consists of GPT-5-extracted structured annotations, rather
than a frozen linear hidden-state readout. Section 4.1 and Appendix B.1 explicitly
allow later repairs or contradictions to reconstruct an earlier attributed
state. The human pilot uses one author annotator on 60 expressions; it does not
establish inter-annotator reliability. These distinctions motivate prospective
input restrictions and separation of attribution from actual reader outcomes.

## Belief isolation under contextual pressure

**Xu et al., August 2026 revision,
[When Should Models Change Their Minds? Contextual Belief Management in Large Language Models](https://arxiv.org/html/2605.30219v2).**
Reading coverage: Sections 3–6 and Appendices C–E.

They hold formal evidence fixed while adding sycophancy, authority or stress
cues; symbolic oracles determine exact candidate-hypothesis sets. Pressure-driven
belief errors, disagreement between elicited rankings and final answers, and
representation steering therefore have substantial precedent.

Their Section 6.3 probing asks the model to rank hypotheses; it is not a
neutral-trained linear belief probe. The target is the model's own supported
hypothesis set, rather than a sender's attributed reader belief. Appendix C
selects steering layers and scales using the same Rule Discovery cases used
for steering evaluation, then transfers those choices unchanged to Circuit
Diagnosis. The held-out-task transfer is consequently more informative than
the tuned within-task improvement. Our evaluation should preserve its existing
readout and distinguish fixed-threshold performance from output-score ranking.

## Desire-conditioned belief reports

**Yongsatianchot and Marsella, WiNLP 2025,
[Investigating Motivated Inference in Large Language Models](https://aclanthology.org/2025.winlp-main.30.pdf).**
Reading coverage: Sections 3–5 and Appendix A, including exact prompt examples.

They vary outcome desirability and numerical uncertainty across four domains,
then test explicit emotional framing. Desirability sometimes changes stated
probabilities, particularly with character roleplay or explicit hope. The broad
claim that what an agent wants influences what it reports believing is therefore
already occupied.

Their measurements are generated numerical estimates, without hidden-state
probes or recipient-belief targets. They leave token-probability comparisons
to future work and acknowledge missing prompt-variation robustness. Their
simulation-count uncertainty manipulation also did not produce the intended
sensitivity. Desire-conditioned answers alone would consequently offer a weak
novelty claim for this project.

## The remaining falsifiable measurement question

The narrower candidate question is:

> When current sender-attributed reader belief is fixed by evidence, does
> adopting a goal about the reader's future belief move a frozen neutral belief
> readout beyond the effect of quoting the identical goal sentence, while
> explicit current-belief answers remain correct?

This is an inference about a possible gap in the inspected experiments, not
proof that no other paper tests it. It preserves the original distinction
between current attributed belief and desired future reader belief without
claiming to discover second-order belief or communicative intent.

The active-minus-archived contrast must be declared before observing new
outcomes. A contrast near zero defeats sensitivity to operative goal status
beyond matched content. Shifts concentrated in incorrect current-belief answers
support changed inference, rather than a readout-specific failure. A surviving
contrast supports failure of readout invariance under goal framing; it does not
alone establish belief–desire entanglement, strategic intent, spontaneous
anticipation or a causal mechanism.

## Present execution decision

The new crossed neutral study gives **48/64** primary probe classifications
correct, with the expected score direction in **29/32** receipt-sensitive
pairs. However, the receipt-invariant control scores **43/64**, below its
prospective 48/64 floor; **13/32** control pairs change the fixed-threshold
verdict when the receipt cannot change the fictional attributed belief. The
unqueried fixed head predicts class 1 throughout, and greedy answers include
**37 invalid outputs**. These observations must be interpreted using the
[frozen execution protocol](CROSSED_BELIEF_VALIDATION_PROTOCOL_V1.md) and the
bound results, rather than the encouraging primary cell alone.

**Full neutral qualification failed. No future-goal conditions or new fitting
follow.** Commissioning a goal study requires a fresh, qualified instrument
and behavioral measure under a separately frozen protocol. Its present
obstacle is measurement validity, rather than missing engineering. The broader
communication bridge remains prospective pre-report captures, independently
assigned actual reader evidence and observed disclosure choices; this study
has not established that bridge.
