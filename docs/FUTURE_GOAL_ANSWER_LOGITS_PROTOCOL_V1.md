# Posthoc output-head diagnostic: exact neutral prefixes

**21 November 2026, specified after observing the stopped neutral assay and
before collecting output logits.** This is development diagnosis, not a rescue
of the [failed qualification](FUTURE_GOAL_CONTENT_CONTROLS_PROTOCOL_V1.md).
The 192 non-neutral conditions remain uncollected.

The frozen belief head scored 25/32 while all 32 greedy answers were NO, yielding
16/32 answer correctness. The question is whether relative YES-versus-NO output
scores nevertheless discriminate the stipulated current-belief label.

Use only the same 32 exact neutral message prefixes and saved token IDs. Bind
the entire old attempt, original config/bank/fit/code hashes, pinned tokenizer,
installed model-forward source and new diagnostic code before loading weights.
Keep the same Llama checkpoint/revision, NF4/BF16 settings and backend identity.
Require bare YES and NO to each encode to one distinct ordinary token with exact
round-trip decoding; the NO token must equal every saved first generated token.
Stop on ambiguity rather than selecting alternative verbalizers.

For each prefix, perform one full-model forward with final-position logits only,
no cache, no sampling and no generated response. In that same forward, capture
zero-based block16 and require its full float32 vector to equal the saved neutral
vector exactly before accepting logits. No fitting or threshold adjustment.
Preserve any started/incomplete records and refuse overwrite/rerun.

Save YES and NO logits, their difference, full-vocabulary log probabilities,
log normalizer, argmax token and two-choice-normalized p(YES). The latter is a
conditional token probability, not a calibrated belief probability. Record all
32 feature comparisons and compare argmax IDs with the saved greedy first token.

Report descriptive AUROC and class-wise score distributions overall and by
channel, alongside the existing frozen-probe AUROC and constant-NO baseline.
Report ordinary fixed .5/argmax behavior; do not pick a new threshold on these
labels. Physical-name mappings are paired, leaving 16 scene units rather than
32 independent experiments. No p-value or broad generalization claim is planned.

If output scores discriminate despite constant greedy answers, output bias is
a candidate explanation requiring fresh validation. If they do not discriminate,
the existing belief probe contains more label-predictive information under this
measurement. Neither outcome establishes spontaneous belief, a causal mechanism,
novelty, or latent superiority in natural communication. Both leave the original
neutral gate failed. New generated answers: zero; planned full-model passes: 32.
