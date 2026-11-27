# Research evidence map: contribution and next decision

22 November 2026. This standalone map synthesizes primary sections already read
and verified in the [novelty audit](MAIN_STUDY_NOVELTY_AUDIT_2026-11-22.md) and
[next-experiment review](MAIN_STUDY_NEXT_EXPERIMENT_REVIEW_2026-11-22.md).
It contains no new fit, inference, case authoring or cached-diagnostic outcome.
It changes no sealed source or protocol.

## Closest inspected work

The coverage column states what was inspected, not an assertion of full-paper
reading. Versioned links identify the actual text used. The first four arXiv
version identities were checked against their official records on 22 November;
the remaining older version links identify inspected versions without claiming
they are the latest revision.

| Primary work and inspected sections | What is already established; implication for this project |
| --- | --- |
| Prakash et al., [Language Models Use Lookbacks to Track Beliefs, v3](https://arxiv.org/html/2505.14685v3), Sections 3–6; Appendices H, M, O. [Record](https://arxiv.org/abs/2505.14685): v3, 24 February 2026. | Query-time binding/retrieval and visibility-mediated updates already have mechanistic tests. Section 3.1 uses behaviorally correct cases; Section 5.1 separates intermediate pointer from later answer payload. Appendix M reports distribution-dependent final-token subspaces; O reports a harder visibility failure. A failed fixed midpoint head cannot establish absent belief information or a new generic composition failure. Retain our full cohort rather than inheriting their behavior-correct denominator. |
| Steele et al., [Belief–reality separation lives in routing over a shared value slot, v2](https://arxiv.org/pdf/2607.11945v2), Sections 3–4.3 and relevant 4.4 routing/probe passages; Appendices A–C. [Record](https://arxiv.org/abs/2607.11945): v2, 20 July 2026. | Asserted/derived belief, query routing, access controls and nested interchange already exist. Appendix B's nested shared-slot result is on Qwen7B-Instruct; its nested router uses a different 14B substrate. Appendix A reports Llama8B framing sensitivity; C separates elicitation from capacity using scaffolds. Neither second-order labels nor scaffolded recursion is new here. Our explicit-question task does not measure spontaneous communicative anticipation. |
| Klein, [Decodability Is Not Validity, v1](https://arxiv.org/html/2611.07805v1), Sections 2–5; Appendix A's validation and Appendix B's readout ablation. [Record](https://arxiv.org/abs/2611.07805): v1, 5 November 2026. | Correlated/decorrelated fitting, nuisance-erased probes and development-only nuisance-aware selection are already tested. Section 3 distinguishes descriptive test-selected equivalence from development-selected methods. Good target AUROC is insufficient evidence of selectivity. The new claim must concern this specific correction/format transfer and its controls, rather than discovery that training correlations affect probes. |
| Orgad et al., [LLMs Know More Than They Show, v4](https://arxiv.org/html/2410.02707v4), Sections 3.1–3.3 and 4; Appendix A.2 and A.4 baseline passages. [Record](https://arxiv.org/abs/2410.02707): v4, 18 May 2025. | Token/layer choice materially affects correctness probing; layers are validation-selected per token. Section 4 finds limited transfer beyond a strong logit comparator and permits direction reversal through absolute AUROC. Keep our source orientation, output baseline and full invalid-output denominator. Their strongest generated-answer position is a different computational stage from our pre-answer target. |
| Goldowsky-Dill et al., [Detecting Strategic Deception with Linear Probes, ICML 2025](https://proceedings.mlr.press/v267/goldowsky-dill25a.html), actual [32-page PMLR PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/goldowsky-dill25a/goldowsky-dill25a.pdf): Sections 3–4.2, 5.1–5.2; Appendices C and D.1–D.5. | Their instructed-pair training uses off-policy true text; their main monitor aggregates response-token scores. Honest-target false alarms, aggregation failures and low-FPR sensitivity to layer choice are already documented. D.5 refits last-position probes, rather than merely changing aggregation. An 8B nested-belief classifier is neither their 70B reproduction nor a validated deception detector. |
| Bortoletto et al., [Benchmarking Mental State Representations in Language Models, v1](https://arxiv.org/html/2406.17513v1), Sections 3.1–3.4 and 4.2–4.3; Appendix A.3–A.4. | Belief probing under prompt changes and PCA-based capacity checks already exist. A site/wording sensitivity result alone has substantial precedent. The empirical distinction must come from the controlled transfer question, joint counterfactual outcomes and fresh evaluation. |
| Bürger et al., [Truth is Universal, v2](https://arxiv.org/html/2407.12831v2), Section 5.1; Appendices D–E. | Affirmative/negated training followed by unseen conjunction/disjunction and German testing already instantiates coverage-to-composition transfer. Appendix D distinguishes direction failure from transferred-bias failure. Our specific belief target, matched allocations and correction quartets distinguish the experiment; the broad design is not new. |
| Hewitt and Liang, [Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/), [paper](https://aclanthology.org/D19-1275.pdf), Sections 2–3.2 and introductory selectivity discussion. | Probe capacity and task performance must be distinguished: their control maps assign random outputs to word types. A stronger nonlinear classifier supplies an accessibility diagnostic, not proof that the model performs its learned computation. Our grouped narrative evaluation is a different task; a label-permutation check is not their selectivity measure. |

Two earlier access/version limits are resolved in the newer review: the original
audit explicitly read Orgad v2, while the current review read v4; the original
Apollo audit could not parse the final PDF, while the newer review downloaded
and inspected it. These are dated reading records, not grounds to silently
attribute final-version reading to the earlier audit. This is a targeted map,
not an exhaustive search or proof of firstness.

## Completed evidence and the narrower contribution

The [completed exploratory study](MAIN_STUDY_FINAL_INTERPRETATION_2026-11-22.md)
compares restricted versus full one-step fitting support at equal N, common
groups and fixed capture, with separately fitted Llama and Qwen heads. Neither
model shows the hypothesized primary correction-quartet benefit. Llama has much
better one-step consistency than Qwen; the results do not provide a clean
two-model replication of one-step competence followed by isolated composition
failure. The same-format easier correction cells are also imperfect.

The single shared Qwen prose extractor plus symbolic fold reaches 33.75% correction quartets,
numerically above the latent comparators, but it receives extra generation and
an external algorithm and still fails most complete quartets. This records
higher measured correction consistency, not a matched-compute explanation or a
generally reliable detector. It is not two independent baseline replications.
Human semantic validation remains incomplete.

The defensible completed contribution is a **controlled, reproducible test
showing no primary benefit from a specified fitting-support repair for a fictional
sender-attributed-belief readout**, conditional on the chosen models, NF4
configuration, midpoint/template position, authored groups and renderer. It
combines equal-budget allocations, dependent-state quartets, full-denominator
outcomes and meaningful comparator limits. No inspected paper in this map
reports that exact comparison; that bounded observation does not establish
publication novelty or guarantee a paper.

The candidate additional contribution is a diagnosis of **which readout and
transfer assumptions fail**: correction supervision, nonlinear accessibility,
ingredient extraction, capture location or a held-out wording. The algebraic
target contains interactions, so successful factor decoding with a supplied
fold would not establish an internally integrated belief representation. Kernel
success would likewise establish accessibility under its training assumptions,
not causal use. Both positive and negative outcomes can sharpen the completed
result. Neither establishes deception, private recipient belief or spontaneous
anticipation; Variant 2's bridge is monitor reliability under controlled shifts.

## Decision rules after the cached diagnostic

The [sealed cached protocol](CACHED_READOUT_ADEQUACY_PROTOCOL_V1.md) uses the same
saved midpoint features, whole-group outer holdouts, inner development and
correction supervision. Its overlapping out-of-fold results are descriptive;
they are not a new untouched test or independent-fold confidence interval.

The [fresh specification](FRESH_READOUT_STUDY_SPECIFICATION_V2.md) remains a
**draft**. Its proposed scope criterion is at least 80% mean primary quartets,
90% mean secondary-row accuracy and no outer-fold quartet score below 60%, in
both models. This is an engineering decision rule, not a validated reliability
threshold; the fresh execution policy must record its final version before new
test outcomes.

| Cached outcome | Bounded next test | Interpretation limit |
| --- | --- | --- |
| Direct correction-trained LR satisfies the proposed criterion in both models | Retain the original site; fresh primary is correction-trained LR minus source-trained LR at N512. Reuse old source features: 6,016 new prefills/model, 7,040-prefix total inventory. | Cached outer refits use 1,536 correction rows, versus 512 fresh fitting rows. Cached success qualifies accessibility at that larger budget, not a frozen N512 instrument. The fresh result must establish its own budget/generalization performance. |
| Only RBF or factor-plus-fold succeeds, or models disagree | Use the fixed four-depth by three-position panel. Source development alone chooses one site; primary is that source-trained LR minus same-policy original-site LR on fresh renderer-A hard quartets. | RBF is a fixed regularized family, not inherently low capacity; factor folding uses four labels and a supplied algorithm. Neither can be called an equal-supervision improvement or absence-of-belief test. |
| All cached methods are weak | Decide whether the same single panel still resolves an important capture/task question; otherwise finish the bounded negative result and dissertation argument. | Do not infer absence everywhere, or keep expanding sites, prompts and budgets until a favorable result appears. |

The full fresh draft has 48 new contexts, including **24 paired test units**,
two fixed renderers and 7,040 prefixes/model. The 12 site vectors come from each
same prefill. Easy two-message, fresh one-step, explicit-attribution, world/name
and output controls remain in full denominators; model-correctness filtering is
forbidden. Source and correction selection stay separate. Both models and every
failed control are reported, with one chosen primary frozen before test scoring.
No new case authoring or fresh inference is implied by this evidence map.
