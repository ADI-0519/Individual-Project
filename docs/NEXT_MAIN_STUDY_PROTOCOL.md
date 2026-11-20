# Next main study: recipient-conditioned monitoring under reporting pressure

20 November 2026. **Proposed new development study. Not a preregistration and
not a report of new model results.** The implementation, case bank, prompts,
model revisions, analysis and provenance must be frozen before this study's
first model output. This document sets the intended design and feasibility
criteria; it does not certify that the materials or implementation meet them.

The core question remains:

> Can an outcome monitor trained only on neutral communication predict an
> actual recipient's decision on fresh natural communication under pressure,
> beyond source-state and appropriately informed text controls?

The next task is a small natural-communication feasibility pilot. It fits no
new probe. It tests whether longer, realistic sources and short reports supply
the truthful omissions and recipient-dependent outcomes the main question needs.

## What remains stopped

The earlier nine-example Qwen qualification scored 24/24 informed, 17/24
missing-evidence and 24/24 stale controls. Its missing-evidence requirement was
20/24, so it failed. The [source-excerpt study](NEUTRAL_RECIPIENT_PROBE_PROTOCOL.md)
and its dependent collection/fitting remain stopped. Its gate, protocol,
historical results and fitted artifacts are unchanged. This proposal does not
reinterpret that failure as a pass or resume its conditional pipeline.

The [untouched recipient-transfer bank](../data/communication/fresh_recipient_test_v1.json)
remains untouched. Do not use it for this proposal's pilot, prompt development,
thresholds, training or feasibility selection. Author separate cases for both
the proposed pilot and the eventual main study.

The [measured communication results](COMMUNICATION_TRANSFER_RESULTS.md) and
[reader failure analysis](EVIDENCE_ROLE_DIAGNOSTIC_RESULTS.md) motivate this
change. The earlier source documents contain few substantive facts, and a
40-word report can often include their decisive evidence. A favorable reporting
goal did not reliably create clean omissions. More compression may create
information selection; it does not guarantee misleading communication.

## Targets and the role of reader errors

Keep three different quantities in the data and analysis:

| Quantity | Meaning |
| --- | --- |
| Actual recipient decision, Y | The pinned reader's observed APPROVE, REJECT or DEFER answer under its fixed policy, actual packet and exact report |
| Evidence-supported decision | What the stated rule and the recipient's available packet plus report warrant, under the declared precedence policy |
| Complete-source decision | What the complete source and rule establish, independently of what the recipient sees or answers |

INVALID remains a separately recorded outcome. Preserve raw answers and report
its rate. Declare exclusions from forecast-loss denominators before execution;
never replace an invalid answer with a normative label.

A reader need not follow every normative rule to be a valid target for
predicting its actual behavior. Its mistakes are observable outcomes, not
automatically annotation errors. Accordingly, normative perfection is not a
blanket gate for this new behavioral forecast study. This is an explicit new
target policy after the failed qualification, not a change to the old gate.

The distinction constrains interpretation. Predicting a biased recipient's
answer does not establish correct reasoning, a human belief, deceptive intent
or a useful general-purpose deception detector. Complete-disclosure and
missing-evidence failures remain prominent results. Semantic claims about
misleading omissions require the separately declared stratum below.

Intent and recipient effect are also different constructs. A detector of a
sender's deceptive attempt need not predict whether that attempt succeeds,
whether an informed recipient resists it, or whether a later private packet
changes its effect. Applying frozen Apollo scores to recipient outcomes is a
cross-task transfer and scope audit. Invariance across privately assigned
packets is not, by itself, a false positive or a failure of Apollo's original
intent target. The independently assigned packet cannot change earlier intent.

## Eight new source dossiers

Author eight pilot dossiers in four document settings, with one complete-source
APPROVE and one REJECT case per setting. Use practical settings such as software
procurement, equipment deployment, research-resource eligibility and service
contracts. The [current draft](NATURAL_SELECTION_PILOT_CASES.md) uses one shared
two-condition decision grammar across those four settings. This is a
within-grammar communication feasibility pilot, not a rule-family transfer
study. Record the actual rule and logical structure. The later main bank needs
separately defined families; this pilot cannot establish that broader validity.

Each dossier contains eight atomic, source-supported assertions in approximately
160-220 words. Include one nominated decisive fact, several other useful facts
and neutral background. The source must support the nominated fact explicitly.
Avoid unstated administrative-to-event relationships, ambiguous time precedence
or outside specialist knowledge. These are authored realistic scenarios, not
real-world documents or an estimate of natural omission prevalence.

Align the dossier rule with the unchanged reader system: select the latest
dated final assessment separately for each required check before applying the
decision rule. Use an older final and a newer final for the critical check,
not an undated preliminary/final fallback that conflicts with that system.
Informed packets contain the newer final; stale packets contain only the older
final; missing packets contain neither. The older final remains evidence the
recipient may legitimately use when no newer record is supplied.

For every case prepare, before generation:

- The complete-source decision and an independent derivation from its rule.
- Three actual recipient packets: informed, missing the decisive fact, and
  stale evidence. Specify available-evidence judgments independently of future
  model answers. Historical claims must remain explicitly historical.
- One concise decisive-fact restoration, copied or faithfully restated from the
  source, and one comparably sized irrelevant factual addition. Record exact
  lengths rather than assume that they match.
- Fact IDs, the nominated decisive fact ID and exact packet visibility.
- An authoring and semantic-review record. Agent checking is not independent
  human validation; retain any outstanding human review explicitly.

Check that opposite source conclusions are balanced and that no original pilot,
qualification or future-transfer item has been reused. Resolve input errors
before model generation. Once outputs are observed, do not replace unsuccessful
cases to satisfy feasibility criteria.

## Natural reporting and the bounded pilot

Use the pinned Llama-3.1-8B-Instruct actor and its established NF4/BF16 runtime.
Freeze greedy decoding, seed, model/tokenizer revisions and a common adequate
token ceiling. Preserve the full generated report, including EOS/length
metadata; do not truncate to enforce a word count or regenerate violations.

The actor receives the complete source and public decision task. It does not
receive the recipient's actual packet. Ask for a factual evidence note without
an APPROVE/REJECT/DEFER recommendation. Do not nominate which fact to omit or
instruct misleading content. Change only the declared objective between neutral
support and favorable-APPROVE pressure. Ask for a maximum of either 25 or 50
words in the two fixed budget conditions. The 50-word arm checks whether effects
are specific to compression; it is not a budget selected after reading outputs.

Generate one report per case x goal x word budget, then save it before assigning
the actual packet. Replay that exact saved text in independent recipient calls.
Use the existing pinned nine-example Qwen policy, strict parser and decoding;
retain its failed qualification status. Do not search additional reader policies
or silently change its handling of missing information during this pilot.

Before any generation, build a separate standalone pilot renderer and dry-run
manifest. Reuse the exact system/demonstration prefix and verify its policy
hash; do not loosen the old gated runner's guards or pass this bank off as an
old validated case format. Keep IDs, analysis roles, expected answers and audit
metadata outside model messages. Preflight every planned prompt against its
tokenizer and freeze an adequate input limit: no source, example, packet or
report may be silently truncated. The standalone [execution protocol](NATURAL_SELECTION_RUN_PROTOCOL.md)
and [runner](../scripts/run_natural_selection_pilot.py) implement these stages.
Fixture tests and tokenizer preflight establish engineering checks; actual
model outcomes and semantic review are separate evidence.

| Component | Planned conditions |
| --- | ---: |
| Natural reports: 8 cases x 2 goals x 2 word budgets | 32 |
| Original reports: all 32 saved reports x 3 actual packets | 96 |
| Additions: 16 reports from the 25-word arm x 3 packets x decisive/irrelevant addition | 96 |
| Controls: 8 cases x 3 packets x no report/complete disclosure | 48 |
| Total recipient conditions | 240 |

The complete-disclosure control supplies the complete source as the reader's
evidence report. It checks comprehension; its length is not a matched natural
report condition. Apply the prewritten additions even if a report already
includes that fact, recording this redundancy. Keep the original report
unchanged. Record every correct, wrong, deferred and invalid transition.

These are 32 sender generations and 240 recipient conditions, not necessarily
272 unique model generation calls: identical recipient messages may share an
explicitly recorded cache result. There are only eight independent case
clusters. All report, packet, budget and goal variants of a case stay together.

## What the latent monitor can know

An original sender activation saved before a private packet assignment cannot
contain that later assignment. Do not score its inability to predict that
private change as an informative monitoring failure.

Save the original natural-generation activations for a separately labeled
diagnostic. For the central same-source/same-report recipient comparison, use
an explicitly contextual monitoring pass containing:

1. The source and rule.
2. The reporting goal and fixed recipient-policy specification.
3. The actual recipient packet.
4. The exact saved report.

Freeze the concrete message template and its information visibility before
execution. The reader itself receives only the rule, its packet and report,
under its fixed policy; it never receives the sender's private full source or
objective. No monitor receives the eventual reader answer.

The contextual pass may teacher-force the saved report in the declared context.
Its report text originated naturally, while its combination with a subsequently
assigned packet is a controlled contextual replay. Call the extracted states
**monitoring/reread states**, not spontaneous sender beliefs or pre-report
anticipation. This distinction applies under both neutral and pressured goals.
It preserves a concrete evaluation of latent monitoring under context and goal
shifts without adding a belief-question task.

Use the established decoder-block-8 extraction convention. The prospective
outcome readout uses the final non-special report-content token after the report
is available; explicitly verify token boundaries and exclude chat terminators.
The frozen Apollo comparator retains its own response-span/raw-margin pooling
recipe and training scaler. Do not refit Apollo, reinterpret its margins as
deception probabilities, or choose a new layer from pilot scores.

All matched-information text comparators receive the same source, rule, goal,
actual packet, saved report and recipient-policy specification. Contextual
activation extraction is additional computation. Record it separately from
original sender generation; do not claim that reread features are free.

The deterministic recipient's output is fixed by its own visible policy, rule,
packet and report. Once those are held exact, the sender's private source and
goal cannot change its answer. Include a strong reader-visible-only forecaster,
not just a report-only or packet-only control. Compare goal/source-stripped
monitoring states and matched text under that same visibility. Extra private
information may correlate with outcomes; it is not evidence of an additional
causal route into the recipient. Any analysis relying on it must confront this
invariance rather than call it privileged knowledge of the reader.

## Annotation and semantic interpretation

Review all 32 natural reports blind to probe scores and actual reader outputs.
Record factual support, contradiction, unsupported claims, ambiguity,
decisive-fact coverage, recommendation leakage, refusal, truncation and word
count. Distinguish faithful paraphrase from a literal string match. Preserve
agent-draft and human-review status separately.

The all-output behavioral analysis is primary. Also define, before opening
outcomes, a semantic omission stratum requiring a fully source-supported
report, an absent nominated decisive fact, an unambiguous rule/packet analysis
and a correct complete-disclosure control for that same case and actual packet.
Report its denominator
and exclusions. Do not turn it into the whole dataset or select only strata
with favorable probe scores.

Within this stratum, keep these outcomes separate:

- A new wrong definite decision after the original report, relative to no report.
- An existing wrong decision that the report leaves unchanged.
- An evidence-warranted stale decision that differs from the complete-source
  decision.
- An unwarranted definite decision when the available evidence supports DEFER.
- A correct decision, a deferral and an invalid answer.

A decisive restoration can produce wrong-to-correct, wrong-to-defer,
defer-to-correct or other transitions. Report the full transition table and
compare the irrelevant addition. A world-wrong answer that was already present
without the report is not evidence that the report induced it. Neither a wrong
answer nor an omitted fact independently establishes intentional deception.
These interventions estimate the effects of the specified added sentences;
they do not isolate an internal relevance or belief mechanism from wording,
format and answer cues.

## Pilot advancement criteria

The aggregate criteria in the initial unexecuted design were tightened during
independent review before any pilot output. The 50-word arm could otherwise
qualify an uninformative 25-word arm, or packet differences alone could supply
all outcome variation. Freeze these five revised criteria before the first
generation. Advance the proposed 25-word training study only if all five hold:

| Requirement | Minimum |
| --- | --- |
| Eligible fully source-supported 25-word reports | At least 12 of 16 |
| Eligible supported 25-word reports omitting the nominated decisive fact | At least 4 reports across at least 4 cases and at least 2 document settings |
| Same original 25-word report produces different valid actual decisions across actual packets | At least 4 case clusters |
| Actual original-report decision support within each goal of the 25-word arm | At least 22 valid answers of 24; APPROVE, REJECT and DEFER each occur in at least 2 rows from at least 2 cases |
| Original 25-word report changes a valid decision relative to its matched no-report call | At least 2 distinct cases |

An eligible report is nonempty, source-supported, EOS-complete and within its
requested word budget, with no refusal or action recommendation. A capped
fragment, an over-budget response or an empty refusal cannot supply omission
feasibility. Preserve every such output in the all-output behavioral analysis.
Semantic eligibility remains a separate blind annotation, with agent and human
review status explicit. Do not infer it from downstream answers or keywords.

The natural report-dependence criterion requires an original-report effect;
restoration alone cannot qualify it. Show original-versus-no-report changes
and restoration-versus-original changes separately. Changed decisions are not
automatically harm, benefit or evidence of intent. The 50-word results and
pooled counts remain descriptive and cannot rescue a failed 25-word criterion.

These are development feasibility thresholds, not statistical power claims or
evidence of a monitor's success. Repeated decisions from one case do not add
independent cases. Show counts by goal, budget, family and packet even when an
aggregate criterion passes. Complete-disclosure and invalid rates remain visible
and constrain semantic interpretation; passing these five thresholds does not
certify an ideal reader.

Fit no new probe during this pilot. Apply existing frozen comparators only as
descriptive diagnostics; do not tune thresholds or select cases from their
scores. If a criterion fails, retain the whole pilot and identify which part of
the proposed target it could not support. Do not extend the batch, swap cases,
tighten the report budget or search policies until the criterion passes. Any
further redesign requires its own explicit development decision and fresh data.

## Proposed main-study scale after feasibility

The candidate main bank has 160 new dossiers in eight explicitly defined rule
families, with twenty dossiers per family. The pilot cases are excluded. Split
the complete case bank before observing outputs:

| Partition | Cases |
| --- | ---: |
| Neutral training: 12 cases from each of 6 development families | 72 |
| Neutral validation: 3 cases from each of those 6 families | 18 |
| Fresh seen-family test: remaining 5 cases from each of those 6 families | 30 |
| Held-out-family test: all 20 cases from each of 2 reserved families | 40 |
| Total | 160 |

The final evaluation therefore has seventy case clusters. Shared logical
components across families must be disclosed; a new domain label is not broad
unseen-reasoning generalization. No pilot or qualification cases enter this bank.

Use the 25-word natural-report condition for the proposed main study. Keep
50-word reporting as the pilot's declared compression comparison rather than
adding another main-study sweep. Collect neutral training/validation reports
and actual recipient labels. Fit train-only transforms and independently
selected text/activation readouts using only these case partitions. Fix block 8;
select regularization from C = {0.01, 0.1, 1, 10, 100} on validation loss only,
breaking an exact tie in favor of the smaller C.
All three target classes must be observed in the neutral training data for the
proposed three-class fit; otherwise record it as untrainable. Never manufacture
missing classes from normative labels.

Freeze fitted artifacts, preprocessing, hyperparameters, prompts and parsers
before generating or opening the seventy test cases' natural neutral and
pressured outcomes. Generate each natural report once per case/goal, then cross
its exact text with the three actual packets for contextual monitoring and
reader replay. Preserve the same report/context construction during neutral
training and final evaluation. The new readout is a contextual outcome monitor;
it is not the earlier source-excerpt head or an original sender-state belief
probe under a new name.

## Comparisons, metrics and a useful result

Include, with frozen information and tuning rules:

- The unchanged Apollo probe, retaining raw margins and its training scaler.
- A newly neutral-trained activation outcome monitor and a text-plus-activation
  monitor.
- Independently selected full-context TF-IDF, packet-only and report-only text
  controls; source-world and neutral-training-majority controls.
- Goal-only and fact-coverage controls, plus a format-only control containing
  report length, report/packet layout and edit markers without assessment values
  or substantive evidence. Define these features before model outputs and keep
  case IDs, fact roles, gold labels and audit metadata outside every payload.
- A capable full-context LLM forecaster with the same permitted information,
  prospectively fixed prompt, adequate output budget and strict parser.
- A capable reader-visible-only forecaster seeing the exact policy, rule, packet
  and report, with goal/source-stripped activation and text controls.
- Direct execution of the known deterministic recipient as an oracle-cost
  ceiling. Executing the target itself is not an independent forecasting result.

Primary outcomes are multiclass log loss and Brier score, reported separately
for neutral and pressured test reports and as their paired change. Also report
macro-F1, classwise recall, class counts, accuracy and invalids. Use case-level
paired uncertainty and show held-out families separately; do not treat all
packet/report variants as independent samples. Calibration bins and any
probability calibration must be fixed using development data, never test labels.

Analyze specificity **within the pressured goal**, alongside the neutral-to-
pressure shift. A score that separates objectives can still fail to distinguish
different recipient outcomes under the same objective. Evaluate same-report
packet-sensitive cases and source/recipient disagreements explicitly. Report
unavailable or degenerate metrics rather than invent a second class to compute
an AUROC. Actual wrong-decision labels are downstream behavior, not deception
intent labels. Do not claim a calibrated false-positive rate from uncalibrated
Apollo margins or a handful of negative examples.

Frozen Apollo has no three-class recipient forecast and receives no new
calibrator. Its raw-margin associations with declared downstream outcomes are
descriptive transfer diagnostics, not entries in the multiclass log-loss or
Brier comparison. Those proper losses apply only to methods supplying a
prospectively defined APPROVE/REJECT/DEFER probability vector.

Record unique model calls, cached conditions, input/output tokens, original
generation time, contextual feature-extraction time, forecast time and model
loading separately. A computational advantage requires a common measured
budget. This local feasibility pilot does not require HPC; a later matched
BF16 reference has the separate justification in the [CREATE request](CREATE_COMPUTE_REQUEST.md).

A useful result would establish a reproducible recipient-dependent target and
show whether a neutral-trained contextual readout retains predictive value
under pressure beyond competent text/source controls. A failure to add value,
or a specificity failure concealed by objective-level discrimination, remains
an informative result. Neither pilot feasibility nor a favorable point estimate
is a publication, novelty or academic-mark guarantee.

## Prior work and the proposed contribution's limits

[Kim, Lee and Lee (2026)](https://aclanthology.org/2026.emnlp-main.656.pdf)
already combine natural answers, downstream reader decisions, critical-fact
measurements and pre-generation coverage probes. Their methods and appendices
also test criticality, coherence, judges, reader choice and parametric knowledge.
They distinguish transmitting evidence from transmitting a usable conclusion.
Therefore longer sources, short reports, reader replay and linear coverage
decoding are not novelty claims.

[Apollo (2025)](https://proceedings.mlr.press/v267/goldowsky-dill25a.html)
already reports honest outputs scoring highly under deceptive incentives.
[Thormann (2026)](https://arxiv.org/html/2603.10003v2) studies contextual truth
probes and misleading non-falsities, and proposes other-agent belief readouts.
[Sufficient Context](https://arxiv.org/html/2411.06037v3) already separates
evidence sufficiency from model abstention and correctness. The
[existing literature reassessment](COMMUNICATION_LITERATURE_UPDATE.md) also
records instructed/spontaneous omission transfer, goal-conditioned summaries,
recipient-information interventions and counterfactual omitted-fact analysis.
The [20 November priority update](PAPER_NOVELTY_UPDATE_2026-11-20.md) adds
compliance-rule counterfactuals, instruction-versus-decision audits and evidence-
sufficiency artifact controls. These narrow the candidate contribution further.

The candidate contribution is the controlled transfer evaluation: a monitor
trained only on neutral communication, evaluated on fresh natural pressured
reports, with actual recipient evidence varied for the same source/report and
with normative warrant, source truth, factual support and observed behavior
kept separate. Its value must come from a clear empirical result and the
information-matched comparisons. Combining these established ingredients does
not establish firstness, causal second-order beliefs or a new mechanism.
