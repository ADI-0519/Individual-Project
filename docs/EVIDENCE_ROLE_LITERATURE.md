# Evidence roles, irrelevant information and warranted abstention

20 November 2026. Primary-paper review for the **prospective 96-call reader-only diagnostic** in the [frozen protocol](EVIDENCE_ROLE_DIAGNOSTIC_PROTOCOL.md). No outputs from that batch were consulted for this review. This document does not change the frozen reader policy, qualification thresholds, datasets or stopped communication pipeline.

**Irrelevant-context failures and answering from insufficient evidence are established findings.** The useful question for this small experiment is narrower: does a definite decision follow an irrelevant value as if that value described the decisive attribute, despite successful comparisons when the decisive attribute is actually supplied? Merely obtaining another failure to answer DEFER would not establish that distinction or a new research contribution.

## Closest primary evidence and exact reading scope

| Primary source | What it already establishes | Difference from this diagnostic |
| --- | --- | --- |
| Shi et al., **Large Language Models Can Be Easily Distracted by Irrelevant Context**, ICML 2023. [Proceedings and paper](https://proceedings.mlr.press/v202/shi23a.html) | GSM-IC adds an irrelevant sentence to still-answerable arithmetic; it varies topic, entity overlap and number range. An easy 100-problem base generates many related variants. Prompting mitigates but does not eliminate distraction. | Here the decisive attribute is absent in metadata conditions, so the warranted response changes to abstention. The same low/high value is crossed with its attribute role. |
| Mirzadeh et al., **GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models**, ICLR 2025; [version 2, 27 August 2025](https://arxiv.org/html/2410.05229v2) | GSM-NoOp adds plausible but operationally irrelevant clauses. Models sometimes turn these into arithmetic operations; demonstrations do not generally remove the deficit. This already covers numerical distractors being used as relevant quantities. | Our dependent variable is selective APPROVE/REJECT/DEFER under missing evidence, rather than recovering a still-available arithmetic answer. |
| Joren et al., **Sufficient Context: A New Lens on Retrieval Augmented Generation Systems**, ICLR 2025; [version 3, 23 April 2025](https://arxiv.org/html/2411.06037v3) | Separates relevance from enough information to answer, documents non-abstention under insufficient context, and combines a sufficiency rater with confidence for selective generation. Explicitly warns against inferring an unstated relation from a related fact. | Our fictional rules admit a direct two-compatible-world audit of DEFER, without retrieval, an LLM sufficiency judge, confidence fitting or real-world parametric knowledge. |

The following coverage records are deliberately narrower than “read every page.” They identify the methods, result interpretation and limits that support the comparison above.

- **Shi:** read dataset construction and metrics (§§3.1–3.2), prompting setup (§§4.1–4.2), factor analysis (§5.2.1), additional exemplars (§5.4), DROP extension (§5.5), discussion (§6), and relevant Appendix A templates and ambiguity repairs. Their “macro accuracy” requires every variant of a base case to be correct; it is not class macro-F1. Models are principally Codex/GPT-3.5. Appendix A repairs pronoun ambiguities that would otherwise compromise answer preservation. No code reproduction. [Full paper](https://proceedings.mlr.press/v202/shi23a/shi23a.pdf)
- **Mirzadeh:** read template construction/setup (§§3.1–3.2), name/value and difficulty experiments (§§4.2–4.3), complete GSM-NoOp intervention (§4.4), conclusion, arithmetic ablation (Appendix A.6), and statistical discussion (A.3/B). The version 2 discussion acknowledges a statistical dispute; the authors' broader interpretation about genuine reasoning is not a mechanistic result we can inherit. No all-appendix or code audit. [Version 2](https://arxiv.org/html/2410.05229v2)
- **Joren:** read sufficiency definition/labeling (§3), datasets, setup and error analysis (§4), selective generation and fine-tuning (§5), limitations, Appendix A model/retrieval details, and relevant evaluator/metric appendices B.3/C. The main context labels are model-rated and imperfect; the stated application is QA. Its relation example already makes the conceptual point that a related attribute is not the requested attribute. No code reproduction. [Version 3](https://arxiv.org/html/2411.06037v3)

## What the frozen experiment can distinguish

The [authored panel](../data/communication/evidence_role_v1.json) has 16 cases: eight date comparisons and eight score comparisons. Each case crosses two values with three evidence roles, producing 96 calls. It uses the existing nine-shot policy, checkpoint, parser and output cap. These are **16 case clusters under two authored grammars**, not 96 independent scenarios.

For a date case, a completion date before the deadline supports APPROVE and one after it supports REJECT. A generic administrative registration date supports neither conclusion about completion. For a score case, an evaluation score below the minimum supports REJECT and one above it supports APPROVE. A file-ID number supports neither conclusion about the score. Both metadata values therefore require DEFER. The explicit-missing condition adds a statement that the decisive attribute was not provided.

The key paired observation is not simply “a model answered definitively when it should abstain.” It is whether **changing only irrelevant metadata from low to high makes the model switch between the same two definite decisions that the actual decisive values justify**, while the actual-evidence controls pass. That pattern is consistent with confusing an attribute's evidential role. It is stronger evidence for that hypothesis than a constant fallback decision, but it still does not locate an internal mechanism.

Date and score directions are opposite: increasing a completion date crosses from APPROVE to REJECT, while increasing a score crosses from REJECT to APPROVE. Analyze value-following within case and domain before any combined statistic. A pooled APPROVE proportion would obscure the distinction. Domain differences also bundle grammar, comparison operator and demonstration coverage; they do not isolate a causal effect of “dates versus numbers.”

| Observed pattern | Bounded interpretation |
| --- | --- |
| Decisive controls fail | The reader may have a comparison, instruction-following or format problem. Metadata errors do not isolate relevance handling. |
| Decisive controls pass; metadata low/high follow their corresponding definite directions | Evidence consistent with value-sensitive misuse of the irrelevant attribute in these cases. |
| Metadata low/high always give the same definite decision | Failure to abstain or a decision bias; not evidence that the irrelevant value was substituted. |
| Metadata low/high change, but do not track the decisive directions | Sensitivity to irrelevant input; the proposed substitution account remains unsupported. |
| Explicit missingness restores DEFER | The combined missingness wording changes behavior. It does not prove a particular internal reasoning repair. |
| All metadata cases give DEFER | This diagnostic does not reproduce the proposed failure, even though the earlier capability gate failed. |

No model probability or entropy is measured. “Unwarranted definite answer” is the appropriate description; “high confidence,” “miscalibration” and “semantic uncertainty” would require additional measurements. Likewise, a wrong decision is not evidence of deception, a sender's intention or anticipation of a recipient's beliefs.

## Limits to retain before interpreting results

1. **No metadata-free condition.** This batch cannot estimate how much metadata increases non-abstention relative to an otherwise identical empty-evidence prompt. Value-sensitive flips can still be informative; a general distraction effect cannot be cleanly isolated.
2. **Explicit missingness is bundled.** It changes length, negation, salience and an epistemic cue. An improvement is not uniquely attributable to recognition of the missing attribute.
3. **The worked examples are fixed and uneven across families.** The current policy has no deadline example. A date/score contrast could reflect demonstration coverage as well as task semantics. This batch does not search for a better policy.
4. **The sample is narrow.** Report exact paired cases and case-cluster summaries, preserve invalid/capped outcomes, and describe any interval as conditional on these authored grammars. Repeating greedy calls would not produce new independent cases.
5. **Labels must refer only to visible evidence.** The two-compatible-world audit establishes why metadata leaves the decision unresolved; those worlds are not hidden labels with which to reward a lucky guess. A registration date asserts no relation to completion, and a file-ID asserts no relation to score.

## Relation to the dissertation and a defensible contribution

The immediate role is **instrument diagnosis**. The fresh reader failed its predeclared qualification gate, so further probe training or pressure-transfer conclusions would lack a reliable behavioral target. This small batch can make that failure more precise while the original results and failed gates remain reported. Passing some diagnostic controls does not qualify the reader or authorize the stopped pipeline.

A defensible result, if the measurements support it, would read: “Under a fixed reader policy, irrelevant attribute values changed otherwise unwarranted definite decisions in the direction predicted by the same values when they described decisive evidence.” The denominator, failed controls and domain differences must accompany it. It would be a bounded replication and refinement of established distractor/sufficiency failures, **not a first discovery that language models confuse irrelevant evidence or fail to abstain**.

A potential later contribution would require independently held-out templates and another reader, a direct comparison with strong evidence-sufficiency text methods, and a demonstrated reason this behavior matters to monitoring pressured communication. Showing that a frozen probe fails or succeeds specifically across these evidence-role changes could connect back to Variant 2. None of that has been established by a 96-call reader-only assay. Whether the complete combination is publishably novel remains an open literature and empirical question; these three closely related papers are not an exhaustive priority search.

More compute can support a matched precision or model comparison, but these behavioral papers do not establish that BF16, a larger GPU or CREATE access would repair the observed reader. Such a claim needs its own controlled comparison.
