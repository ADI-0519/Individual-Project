# Fresh neutral-to-pressure communication transfer

20 November 2026. **Prospective design, written before the fresh panel's model
outputs.** This returns the next experiment to natural reports and measured
recipient decisions. The completed story/question experiments are auxiliary
target and specificity diagnostics; they do not replace the communication study
in the supervisor's brief or establish deceptive behavior.

The question is whether an unchanged readout trained on neutral natural
communication remains useful when a sender faces an approval objective, with
matched comparisons to report text, reader evidence and a full-context monitor.
The primary target remains the actual recipient's APPROVE/REJECT/DEFER decision,
not world truth, deception, or a latent belief asserted by an annotator.

## Frozen new inputs

The [case bank](../data/communication/fresh_transfer_v1.json) has **24 fresh
all-test cases**: four per family, two world approvals and two world rejections
within every family. No new case is used for fitting or hyperparameter selection.
The [panel module](../src/latent_probes/communication_panel.py) checks exact source
rendering, dates, arithmetic, packet construction, balance and stable identities.
It also checks disjoint case IDs, entities, complete source assertions, record
dates and family-specific integer instantiations against the original
[development bank](../data/reader_forecast/development_v1.json).

| Rule family | Decision computation | Transfer designation |
| --- | --- | --- |
| Conjunction | All required checks pass | Fresh cases of an already seen rule |
| Threshold | Current score reaches the threshold | Fresh cases of an already seen rule |
| Supersession | Latest final inspection passes | Fresh cases of an already seen rule |
| Arithmetic | Current revenue minus current costs is nonnegative | Fresh cases of an already seen rule |
| Disjunction | Either alternative route passes; route A is known to fail | Explicit rule family held out from source readout fitting |
| Deadline | Completion date in the latest final record is on or before the deadline | Explicit rule family held out from source readout fitting |

The last two families are different explicit decision rules from the source
readout's four training families. They share language, evidence precedence and
general components such as comparisons. Two held-out rule types cannot establish
broad reasoning generalization. These synthetic tasks are familiar to the base
LLM; "held out" refers to readout fitting, not pretraining.

Each case has one current decisive fact, an opposite older final record, a common
fact and a source-supported irrelevant blue-cover fact. Its reader packets are:

- **Informed:** common fact plus current decisive fact; the normative decision
  equals the full-source decision.
- **Absent:** only the common fact; both world answers remain possible, so the
  normative evidence-supported decision is DEFER.
- **Stale:** common fact plus the older decisive record; under the latest
  available evidence policy, the normative decision is opposite the full-source
  decision until newer evidence is supplied.

The older record does not become intrinsically false because a later record
exists outside the reader's view. Record time and the recorded completion time
are distinct in the deadline family. All dates are synthetic scenario dates.
World error, evidence-supported judgment and actual model behavior are retained
separately. An absent reader appropriately deferring is not a deception success.

## Natural communication and main comparison

Generate one report per **24 cases × 3 attributed packets × 2 sender objectives
= 144 reports**. The neutral goal requests a concise accurate report; the
favorable goal asks the sender to induce APPROVE while keeping assertions true.
The source and attributed packet remain identical within each goal pair. World
approvals and rejections are balanced, so favorable pressure sometimes aligns
with truth and sometimes conflicts with it; report those strata separately.

Both goals additionally require evidence-only reporting without the literal
decision words APPROVE/REJECT/DEFER or a recommendation. This prospectively
addresses the previously observed decision-copy shortcut. It is also a **shared
format shift from the source readout's training distribution**, which must be
reported when interpreting transfer. Keep every format violation, factual error,
word-budget violation and length-terminated report; do not regenerate it.

Use the existing pinned Llama-3.1-8B-Instruct NF4/BF16 runtime, greedy decoding,
seed 42, 40-word instruction and 96-token generation cap. Save exact rendered
prompts, generated token IDs, raw report, termination data and response-content
activation matrices. Keep block 8, its final non-special report token and all
source readout transforms/weights unchanged. These are after-report features;
they do not establish anticipation before the report or causal belief use.

The reader receives only its decision rule, aligned evidence packet and the saved
report in a fresh conversation under the original reader policy. It never sees
the full source or sender goal. Collect **144 aligned reader answers** and **72
unique no-report packet controls**, counting informed controls as full-evidence
checks rather than running or counting duplicate prompts as new observations.

The existing frozen evaluator accepts only its original four declared test IDs.
Fresh transfer therefore needs a separate evaluator with explicit new-panel
disjointness and immutable artifact checks. Do not rename new cases to old IDs,
edit the fitted manifest, weaken the original restriction or fit a new model.
Its numerical reconstruction can reuse the existing saved-transform scorer.

The primary description is neutral versus favorable **frozen outcome-readout
robustness**. Report paired case-level multiclass loss changes, accuracy,
macro-F1, per-class recall, class counts, world correctness and the no-report
decision. INVALID targets remain visible with a declared metric exclusion; an
INVALID monitor forecast counts as a failed forecast. Paired summaries average
within each case before comparing goals. Show results by family and transfer
designation, with all denominators. Repeated packets are not independent cases.

Freeze one run and its failure accounting. A favorable result is not a stopping
rule, and an unfavorable result is not permission to search layers, prompts or
new panels until a positive finding appears. A separately versioned development
repair must explain what failed and cannot be presented as untouched validation.

## Strong controls and an essential ceiling

Keep the old neutral-trained activation, prompt-only, report-only, full-text and
text-plus-activation models unchanged. Their lexical models are preliminary;
the previous pilot showed majority-class collapse, so outperforming them alone
does not establish an activation advantage. Include the now prospectively
declared literal decision-copy rule, retaining a predeclared majority fallback
and all recommendation violations.

Run one separately instructed full-context text forecast per saved report,
**144 calls**, with the source, rule, sender goal, attributed packet and report
available as quoted information. It receives no observed reader answer. Freeze
the monitor prompt and parsing policy before calls. It is a same-checkpoint
monitor, not an independent stronger model. Do not invent calibrated probability
losses from its single generated categorical answer.

**A deterministic reader with known prompt can be simulated perfectly by simply
running the same prompt.** This exact reader simulation is a computational oracle
ceiling; it is the reader call already made, not independent forecast evidence or
a research achievement. The previous monitor continued the original sender
history, so its private-source/goal exposure could create a perspective error;
that weak baseline cannot justify broad claims about text monitors.

Consequently this run cannot show superiority to unlimited exact text
simulation. An efficient-monitor claim would require measured total capture,
reader, monitor and scoring costs under an explicit common budget. Replaying a
generated report to extract activations also has computational cost. Profile
actual times before claiming a saving. Stronger future scope can use an
independent reader policy or a stochastic reader's expected outcome with
declared repeated draws; neither is silently added to this experiment.

## Supporting same-report, different-reader-evidence assay

The prespecified supporting subset is the **24 favorable reports generated for
stale attributed packets**, one per fresh case. Selection uses condition
identities, never observed answers, fact coverage or probe scores. Preserve the
natural reports even if they already disclose the decisive fact or are false.

For each saved report, independently cross three actual reader packets with four
report versions: no report, original, original plus the exact current decisive
fact, and original plus the source-supported irrelevant fact. This yields **288
reader conditions** before exact-prompt caching. Cache reuse of an already run
identical prompt must retain source hashes and provenance; it adds no independent
sample size. Appending an already present fact remains an explicit redundant
edit, not a new omission success. Record duplicate fact strings and semantic
coverage separately rather than selecting them away.

Compare no-report→original and original→restored transitions in correct, wrong,
DEFER and INVALID outcomes, and compare restored against irrelevant edit. A
wrong-to-DEFER transition differs from wrong-to-correct. Initial stale errors
may persist rather than be induced by the report. The original report is exactly
the same across actual reader contexts; context effects are measured outcomes,
not assumed deception labels. The edit-control text lengths are measured, not
claimed to match exactly.

The original sender did not see independently reassigned actual reader packets.
Its saved activation therefore cannot be claimed to encode which packet was
assigned later. A new full-context monitor can be given that actual packet, with
the text comparator receiving the same information. Do not relabel such a
monitor's rereading activation as the original sender's mental state.

## Audit and permissible claims

The source cases and report assertions need an independent semantic/human audit.
Agent structural checks are not a human audit. Freeze draft report factuality,
decisive-fact coverage and recommendation labels before opening probe scores;
retain unsupported assertions, refusals, mixed reports and truncations.
Observed reader answers can be analyzed as outcome labels before this review,
but an omission-specific or honest/deceptive behavioral claim cannot.

This is a bounded extension of [the project direction](./RESEARCH_DIRECTION.md),
not a first omission benchmark, a completed Apollo reproduction, proof of
second-order belief mechanisms or guaranteed novel paper. Existing Apollo,
MASK, fact-coverage and recipient-knowledge studies substantially overlap; the
[literature audit](./LITERATURE_PROBE_AUDIT.md) records those precedents. The value
of this batch is a fresh natural-communication transfer test, explicit shortcut
controls and measured same-report recipient dependence. Its conclusion must
follow the frozen results rather than the desired mark or admission outcome.
