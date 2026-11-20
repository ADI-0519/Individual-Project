# One evidence-role substitution failure diagnostic

20 November 2026. **Prospective reader-only failure analysis**, after the fixed
nine-shot reader failed its fresh capability gate. This is not a reader rescue,
a new qualification attempt, probe fitting or access to the untouched `ct`
outcomes. Earlier failures remain unchanged.

The [fixed inputs](../data/communication/evidence_role_v1.json) comprise eight
fresh date cases and eight fresh numeric-score cases. Each has six conditions,
giving **one 96-call batch**. The [panel module](../src/latent_probes/evidence_role_panel.py)
retains the exact original system and nine worked-example message pairs from
the frozen reader policy, followed by a newly authored controlled user payload.
No old case renderer or integrity guard is modified. Condition IDs, expected
answers and low/high tags are analysis metadata, never model input labels.

Each case fixes its entity and decision boundary. Change only the supplied
value within a role pair:

| Condition | Evidence supplied | Normative answer |
| --- | --- | --- |
| Decisive, low/high | Completion date, or evaluation score | Opposite definite decisions |
| Metadata, low/high | Generic administrative file registration date, or file-ID number | DEFER in both |
| Explicit missing, low/high | Same administrative metadata plus an explicit statement that completion date/score is not provided | DEFER in both |

Date low/high values fall before/after the deadline, hence APPROVE/REJECT.
Score low/high values fall below/above a minimum, hence REJECT/APPROVE. This
counterbalances directional definite answers, but date and score grammars also
differ in comparison operator and wording. Differences between those families
are not an isolated causal effect of data type.

There is **no hidden source**. Every actual-reader assertion is mutually
consistent. The administrative date refers to registration of a generic file,
never a "completion file" or final inspection: registration is not asserted to
occur after completion. For a fixed registration date, a completion date before
the deadline and a completion date after it are both compatible with the visible
evidence. Registration may occur before or after either event; the problem states
no functional or temporal relationship. A file-ID number likewise imposes no
arithmetic relation on an evaluation score. The tests retain these two compatible
worlds as an audit of why metadata-only conditions require DEFER. They are not
extra facts shown to the reader or latent world labels used as outcomes.

Entities, specific score quantities and dates are disjoint from the original,
main, fresh-reader-validation and untouched-recipient-test banks. Shared rule
grammar, generic class labels and the original worked examples are intentional.
There are only 16 case clusters and two authored grammars; 96 rows do not provide
96 independent scenarios or broad task generalization.

Freeze the input JSON, all 96 exact messages, source hashes, pinned
Qwen2.5-7B-Instruct revision, existing NF4/BF16 runtime, original nine-shot policy,
strict EOS parser, seed 42 and eight-token output cap before calls. Encode every
complete context with the actual pinned tokenizer and reject overflow without
truncation. Preserve every raw output, token ID, invalid answer and cap. There is
one batch and no later policy, budget, checkpoint, example or case search.

Report all six conditions in each family, normative accuracy, definite-answer
counts, DEFER and INVALID separately. For metadata pairs, report how often
changing only the administrative value changes the answer, and how often both
answers are valid definite decisions that follow the counterfactual decisive
value's direction. Compare those changes with the fully supplied decisive
controls. A constant definite fallback is different from low/high-sensitive
attribute substitution. Include exact paired cases and denominator counts.

Separately compare metadata with explicit-missing wording. That intervention
jointly changes length, negation and an epistemic cue; an improvement does not
uniquely establish a causal internal mechanism. Use case-cluster summaries and
case-level bootstrap intervals, conditional on these two grammars; retain all
invalid/capped outcomes in the main denominators. If a valid-only sensitivity is
shown, label it exploratory and report every exclusion.

This batch may clarify whether irrelevant values influence the failed reader's
decision. Success on any condition does not pass a capability gate, authorize
the stopped training pipeline, establish deception or anticipation, or repair
the original experiment retroactively. Its results belong in the failure-analysis
appendix with the earlier gate and baseline failures, regardless of direction.
