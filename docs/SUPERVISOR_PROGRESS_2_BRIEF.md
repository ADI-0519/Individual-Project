# Progress 2: what we measured and what it changes

Aditya Ranjan · 20 November 2026 · Latent Probes and Listener Beliefs

The project remains within Variant 2: evaluate a frozen latent probe under
reporting-pressure and recipient-evidence shifts. We now have real natural-report
results. They do **not** establish an activation advantage or a novel positive
finding. The immediate barrier is the validity of the reader-based measurement,
alongside the scarcity of clean natural omissions.

## Work completed since Progress 1

The Apollo-style 8B adaptation is implemented and numerically checked. Its 1.00
held-out AUROC distinguishes paired instruction conditions; it does not measure
natural deception detection. This is a smaller-model, NF4 adaptation rather than
an exact reproduction of every experiment in the original paper. The
[method audit](./APOLLO_METHOD.md) records the differences.

The new communication experiment generated **144 natural reports on 24 fresh
synthetic cases**, six decision rules and two reporting goals. It applied the
unchanged neutral-trained outcome readout and Apollo comparator, collected
actual recipient decisions, and replayed the same reports across recipient
evidence and factual-restoration controls. Three agents conducted score-blind
draft content review; independent human review is still outstanding. The primary
run and follow-ups together made **1,031 actual generation calls**, including
cached-call accounting and retained failures.

| Outcome forecast | Neutral | Approval pressure |
| --- | ---: | ---: |
| Frozen activation readout | 62/72 | 66/71 |
| Always predict APPROVE | 63/72 | 66/71 |
| Frozen full-context TF-IDF | 63/72 | 66/71 |
| Full-context LLM, original 160-token budget | 30/72 | 21/71 |
| Same LLM, separately frozen 512-token recovery | 43/72 | 42/71 |

One pressured reader answer is invalid; it remains in the record and is excluded
only from valid-target forecast denominators. Activation REJECT recall is zero
under both goals. There are no DEFER targets in the primary report outcomes.
High accuracy therefore reflects the dominant answer rather than useful
recipient-sensitive discrimination. The budget recovery reproduced every
original prompt and all 66 capped token prefixes exactly; it does not replace
the original comparison or erase its additional cost.

The supporting specificity result is narrower: **13 exact-output neutral/pressure
pairs, across six cases, all have higher Apollo scores under pressure**. The
source, report and reader packet are identical within each pair, so the reader
input is unchanged. This is a post-hoc replication of a known incentive-confound
concern. Cached reader answers are not 13 independent replications of unchanged
behavior. Raw margins are uncalibrated; no false-positive-rate claim follows.

## Why the next training experiment stopped

The original reader failed basic no-report evidence controls. An independently
pinned Qwen reader also failed. We then froze **one** balanced nine-example
policy and evaluated it on a separately authored 24-case qualification bank.

| Fresh reader controls | Measured | Required |
| --- | ---: | ---: |
| Informed | 24/24 | At least 23/24 |
| Missing decisive evidence | 17/24 | At least 20/24 |
| Stale evidence | 24/24 | At least 20/24 |
| Invalid answers | 0/72 | At most 1/72 |

The fresh gate failed. All seven errors were unwarranted definite answers;
four were deadline cases where a file-registration date did not establish the
completion date. A later registration is compatible with both an on-time and a
late completion. We have inspected the compatible-world controls, but human
semantic review remains necessary. These outputs do not establish the model's
internal reasoning mechanism.

No dependent source-label collection, real probe fitting or natural-report
evaluation on the untouched `ct` bank followed. A strictly gated implementation
and tests exist for those steps. Synthetic-fixture tests establish software
behavior, not research outcomes. The [terminal evidence](./validation/2026-11-20-reader-followups.json)
retains the failed gate and all seven exact prompts and answers.

## Scope and the possible contribution

The auxiliary story/world/belief-question work helped distinguish targets; it
does not replace the communication experiment. Our operational target is the
**observed recipient decision under a declared policy**, not a claim that a
sender spontaneously anticipated a reader, possessed deceptive intent or used
a causal second-order-belief mechanism.

Recent work already studies contextual truth probes, misleading non-false
statements, omission coverage and downstream decisions. Other-agent belief
probing has also been explicitly proposed. The [paper-by-paper reassessment](./COMMUNICATION_LITERATURE_UPDATE.md)
records methods, results and the overlap. We should not claim that combining
omissions with listener beliefs is inherently novel.

The narrower prospective question is whether a recipient-conditioned outcome
readout, fitted on source-supported neutral excerpts, transfers to fresh natural
neutral and pressured reports **beyond capable text and source controls**.
Excerpt training is off-policy; natural reports are on-policy outputs. Both
style transfer and reporting-goal transfer must be measured separately. That
study remains conditional on a defensible reader and adequate target diversity.

## Next work and the reason for CREATE

1. Human-review the draft report labels and seven fresh missing-evidence errors.
   There are only four clean current-fact omissions among the 144 draft labels,
   so this batch cannot support the proposed omission-specific study at scale.
2. Profile a behavior-only NF4/BF16 comparison on the same CREATE hardware and
   runtime, preserving cases, prompts and parsing. Quantization is a hypothesis
   to check, not an established explanation or promised fix.
3. Resolve the target and reader-validity problem before collecting the new
   source-excerpt labels. Keep the future natural-transfer bank untouched until
   preprocessing, hyperparameters and fitted artifacts are frozen.

The [CREATE request](./CREATE_COMPUTE_REQUEST.md) asks for an initial two GPU-hour
cap on one A100 40 GB for a paired precision reference. The local 16 GB RTX 5080
already runs quantized experiments. More memory enables the controlled BF16
reference; prestige alone is not a computational justification. No request has
been sent and no allocation is claimed. Existing NF4 probe weights must not be
treated as validated monitors on BF16 activation coordinates.

The [full measured report](./COMMUNICATION_TRANSFER_RESULTS.md),
[reproducible figures](./figures/2026-11-20/communication_transfer.pdf) and updated
[four-slide deck](../presentations/progress2/Progress-2-measured-results.pptx)
support this discussion. Graphify was actually built and queried; other app
plugins, Obsidian and Playwright were not used.
