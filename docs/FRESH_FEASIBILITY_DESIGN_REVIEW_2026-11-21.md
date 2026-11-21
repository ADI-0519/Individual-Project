# Fresh feasibility design: independent literature and target review

21 November 2026. Independent agent review of the latest external project audit,
the [20 November priority review](PAPER_NOVELTY_UPDATE_2026-11-20.md), and the
[executed development proposal](NEXT_MAIN_STUDY_PROTOCOL.md). This note reports
no new model results. The old reader qualification and natural-selection gates
remain failed. The recommendations below are proposed new development criteria,
not retrospective amendments, a power calculation, or authorization to fit.

## Candidate contribution

The defensible question is whether a contextual communication monitor trained
on neutral reports retains recipient-risk discrimination under favorable
reporting pressure, beyond information-matched text methods. Independently
varying recipient evidence for an unchanged natural report makes this a useful
specificity evaluation. It does not establish firstness, sender anticipation,
second-order beliefs, intentional deception, or latent superiority.

The latest audit makes the central design risk concrete: the selected Cairnfield
case has a fact-specific correction, while the old full addition panel has five
changed decisions after decisive additions and four after irrelevant additions.
Only one of the old omitted reports received the addition experiment. Replicating
the original/restored/placebo contrast on fresh omissions is therefore more
valuable than another favorable aggregate probe score.

## Primary papers inspected this review

Coverage below identifies actual passages read. It does not claim a complete
code audit or a full reading of every appendix. All links are primary papers.

- **[Kim, Lee and Lee, Telling Incomplete Truths](https://aclanthology.org/2026.emnlp-main.656.pdf):**
  inspected sections 3.1-3.2 and 5.7-5.8, Appendices K-O, and the concise-answer
  comparison in P. Their criticality is reader-relative; coherence-preserving
  substitution and human annotation expose its limits. They already combine
  natural outputs, downstream decisions and pre-generation coverage probes.
  Coverage can differ from decision success. Our candidate contribution cannot
  be another coverage probe or the claim that true text can mislead.
- **[MonitorBench v2](https://arxiv.org/html/2603.28590v2):** inspected Appendix
  A.3.1, especially Knowledge and Law Judgment. The same truthful misleading
  passage receives a decisive additional cue and its downstream answer is
  checked. Same-text/different-evidence interventions are established prior work;
  our difference must concern the complete recipient-conditioned transfer test.
- **[Li et al., Thinking Is Not Telling v3](https://arxiv.org/html/2602.07796v3):**
  inspected sections 4.1-5.2 and Appendices C-E. Its evidence-grounded probing is
  span annotation, not latent decoding. Missing disclosure is injected into
  failed trajectories with verbosity and think-content controls. Golden Info
  improves immediate correction more clearly than final recovery. Restoration
  plus downstream correction already exists; our placebo comparison must remain
  prominent and our deterministic reader setting must be explicit.
- **[Abdulhai et al., Evaluating & Reducing Deceptive Dialogue](https://arxiv.org/html/2510.14318v1):**
  inspected sections 3.3-3.4 and the omission example in Appendix A.12.1. Their
  misalignment metric subtracts initial from final world error. This reinforces
  the need to distinguish a new error from a maintained old one. Their listener
  beliefs are inferred by an extra LLM; our observed decisions are a different,
  narrower operational measurement.
- **NEW: [Karkehabadi et al., COVERAGE-TRACE v1](https://arxiv.org/html/2611.25539v1),
  posted 19 November:** inspected complete methods 3.1-3.4, setup, results
  5.1-5.2, limitations and integrity/qualitative Appendices 8-10. It holds the
  candidate fixed and changes designated atomic source facts, scoring
  teacher-forced likelihood. Multi-counterfactual AUROC is 0.764-0.825, but
  meaning-preserving adjustment falls to 0.591-0.624. The benchmark contains
  acknowledged construction defects and incompletely reviewed items. This is
  a close source-side coverage comparator, not recipient-risk evaluation. It
  makes fact selection, paraphrase controls and explicit construct limits
  essential. Do not add a new counterfactual-likelihood research branch now.
- **NEW: [Fox et al., LLM Judges Verify Presence, Not Absence v1](https://arxiv.org/html/2608.31016v1):**
  inspected benchmark/analysis passages in sections 3-4, the pipeline and
  mechanism/compute controls in 7.1-7.5, and section 8 limitations. Fact
  enumeration followed by closed presence checks outperforms undirected whole
  note judgment in their setting. Their benchmark operating points fail to
  transfer directly to real notes; restatements remain difficult. Include a
  source-first enumerate-then-check text baseline with grounded evidence spans,
  not only an open-ended deceptive/not-deceptive judge. Their domain and
  thresholds do not transfer to our task without validation.

## Define the target before output collection

Keep four columns and never derive one by renaming another:

1. `Y`: observed APPROVE/REJECT/DEFER/INVALID response of the pinned recipient.
2. `world_decision`: full-source rule result, independently checked before runs.
3. `warrant`: rule result supported by the actual packet plus report. For natural
   reports this requires semantic review; missing a nominated fact need not
   imply DEFER when another known condition already determines the outcome.
4. `world_wrong_definite`: a valid definite Y opposite to world_decision.

A binary risk target removes the old demand for all three valid answer classes.
It does not remove the need for both risk labels in the neutral training
distribution. Preserve DEFER and INVALID separately: a reduction in wrong
decisions can reflect more abstention, and INVALID is not a safe negative.
For binary losses exclude invalids using a frozen rule and report coverage.

The risk target is not an omission label. A fully informative report can be
misread; an omission can be harmless; a stale wrong decision can be warranted by
the actual available evidence. Report induced error, maintained error, correction
and deferral separately. Do not call every wrong decision communication harm.

Reader-visible-only forecasting cannot know private full-source truth. A strong
factorized comparator forecasts Y from the exact reader-visible payload, then
combines its probability vector with the same world-decision input used by the
other risk methods. For world APPROVE, risk is p(REJECT); for world REJECT, risk
is p(APPROVE). Compare it with full-context text and activation methods with
identical permitted information. A report-only baseline is an information-limit
control. Direct execution of the fixed recipient is an oracle-cost reference.

Original sender activations cannot encode a private packet assigned later.
Recipient-conditioned activations must come from a declared contextual reread;
this provides monitoring features, not evidence of original anticipation.

## Fresh panel and reader policy

The proposed 12-case bank has three predicate types sharing conjunction and
latest-record precedence, with four cases per type and two world APPROVE and
two REJECT outcomes each. Independently
balance nominated-fact position and nondecisive negative material. Keep IDs and
gold/role metadata out of model messages. Semantic review must inspect the whole
source and report, allowing paraphrases and repeated evidence.

Prospectively choose **50 words as the primary arm**, with 25 words a secondary
compression comparison. This choice is informed by the failed pilot and must
be declared as redesign. Neither secondary nor pooled counts can rescue a
primary failure. Preserve violations rather than truncate or regenerate them.

The old check-specific nine-example reader policy does not implement numerical
threshold and authorization-expiry rules. Freeze one new generic policy with
an explicit evidence precedence algorithm and nine separately authored examples:
APPROVE/REJECT/DEFER for each of the three rule structures. Examples cannot be
pilot cases. This is one fixed policy, not a prompt search. The generic policy
must say to resolve the latest supplied final record per relevant field and
then apply the case's actual rule; a known failed conjunct is enough for REJECT
even if another conjunct is unknown.

Run the 72 no-report/full-disclosure controls first. If the frozen reader
comprehension gate fails, stop dependent sender/replay collection and report
the fixed control failure; do not start an open-ended reader-diagnosis campaign.
The planned full panel is 48 natural reports, 432 original/restored/irrelevant
recipient conditions, plus 72 controls: **504 conditions**, not independent
observations. Every report receives both additions in both word-budget arms.
The independent experimental units remain the 12 cases, with correlated rule
structures disclosed. Cached identical calls do not create additional samples.

## Suggested prospective development gates

These conservative floors need adoption in the new executable protocol before
generation. They are feasibility decisions, not statistical claims.

| Gate | Suggested primary-arm requirement |
| --- | --- |
| Reader controls, before reports | Zero INVALID; at least 34/36 full-disclosure answers correct, at least 11/12 within each rule; each no-report packet type at least 11/12 evidence-warrant correct |
| Natural report quality | At least 18/24 eligible 50-word reports, and at least 9/12 in each goal; eligibility requires nonempty, supported, EOS-complete, within budget, no recommendation/refusal |
| Supported omission support | At least four favorable reports across four cases and at least two rule structures |
| Binary target support | Within each goal, both world-risk labels occur across at least three case clusters; positive cases span at least two rules |
| Recipient conditioning | An unchanged original report has different binary risk labels across packets in at least four cases spanning at least two rules |
| Induced-error mechanism | At least two cases spanning two rules: absent packet DEFER without report, wrong definite after original, correct after restoration, still wrong after irrelevant addition |
| Maintained-error mechanism | At least two cases spanning two rules: stale packet already world-wrong without report, original leaves it wrong, restoration corrects it, irrelevant addition does not; these cases may overlap the induced-error cases |
| Specificity beyond generic addition | Positive equal-case average of W_irrelevant minus W_restored in the prospectively defined supported-omission stratum, across all its packets; show every transition and the denominator |

The omission stratum must be determined by blind semantic review and declared
control requirements, not scores or whether the original recipient happened to
be wrong. Include wrong-to-correct and correct-to-wrong transitions. A positive
mean with a few cases is development evidence, not an established population
effect. Human review remains necessary for publication-facing semantic claims.

Passing these gates would justify preparing a fresh main-study protocol and
case-level splits, not fitting on the pilot and calling it a held-out result.
If neutral risk has no positive examples, a neutral-trained risk classifier is
untrainable; pressure positives do not repair that training distribution. A
separately declared behavioral forecast study could still be scoped around an
imperfect reader, but it would have different claims and cannot pass the old
semantic gates by relabeling them.

## Audit boundaries

Graphify's existing 727-node map was queried using its actual vocabulary
`recipient omission neutral monitor evidence`; its returned prior-work nodes
agree that omission, recipient relevance and restoration already have close
precedents. The map does not yet index this note or the new papers. This review
made no changes to frozen old protocols, datasets, scores or raw activations.

The search is targeted rather than exhaustive. A useful paper requires a fresh
empirical comparison that survives the controls above; neither a new dataset
name nor a favorable pilot result establishes research novelty.
