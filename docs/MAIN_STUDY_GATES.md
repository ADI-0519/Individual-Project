# What is finished, and what would finish the main study?

19 November 2026. **The engineering milestone and a small development pilot are finished. The main research study and the individual project are not finished.** “Pilot complete” means that a frozen, reproducible first run exists, including its failures. It does not mean the proposed generalisation, belief-transfer or mechanistic claims have been established.

The main question remains: **can a latent reader-outcome forecast trained during neutral communication transfer under reporting pressure, and add value beyond appropriately informed text and source controls?** Natural sender reports remain the primary communication setting. The fixed-report/recipient-evidence/restoration experiment is a supporting diagnostic. See the [research decision](./RESEARCH_DIRECTION.md), [natural-pilot results](./NEUTRAL_FORECAST_RESULTS.md), [earlier reader controls](./PILOT_RESULTS.md) and [primary-paper audit](./LITERATURE_PROBE_AUDIT.md).

## The present boundary

**Natural communication update, 20 November:** the separately frozen
[fresh communication study](./COMMUNICATION_TRANSFER_RESULTS.md) now includes
144 reports over 24 new cases, six explicit task families, 144 aligned reader
conditions, 72 no-report controls and 288 same-report replay conditions. The
old neutral-trained block-8 outcome readouts remain unchanged. The pressured
66/71 activation accuracy matches an APPROVE constant, with zero REJECT recall;
this is not an activation-advantage result. All 144 reports received score-blind
agent draft reviews, with human review still required. Only four clean supported
current-fact omissions remain, so the omission-specific dataset gate is open.

The next empirical priority is a validated reader and an adequately completed
text comparator. The direct reader passes only 4/24 missing-evidence deferral
controls, while 66/144 full-context forecasts hit their original token cap.
The completed 512-token recovery produces 43/72 neutral and 42/71 pressured
forecasts, with no invalid predictions and every original token prefix intact.
The instruction-only Qwen reader failed (22/24 informed, 12/24 absent, 15/24
stale). One fixed balanced nine-example policy, frozen before fresh `cv` outputs,
then scored 24/24, 17/24 and 24/24, with no invalids. Its absent threshold was
20/24; **the fresh qualification failed**. New recipient-specific label
collection/fitting and all untouched `ct` natural-report calls remain unexecuted.
The [source-excerpt protocol](./NEUTRAL_RECIPIENT_PROBE_PROTOCOL.md) is conditional
software/design work, not a result. Review the seven missing-evidence errors and
the task semantics before any new policy-development decision. Further belief-story variants do not substitute for these
communication-validity gates. The [compute request](./CREATE_COMPUTE_REQUEST.md)
specifies a paired NF4/BF16 behavioral reference without claiming that precision
caused the failures or that HPC will remove them.

| Workstream | What actually exists | What is still missing |
| --- | --- | --- |
| Engineering | Pinned 8B NF4/BF16 execution; 536 Apollo-style forced dialogues; numerical, artifact, masking and actual GPU replay checks | A BF16 precision reference, if needed for the final scope; validation of subsequent changes |
| Natural development pilot | 84 reports and 84 reader labels: 72 neutral, followed by 12 pressured; 16/4/4 case split; 47 valid training labels across three classes | Independent target audit, capable baselines selected before fresh results, family-held-out evaluation |
| Frozen after-report readout | 10/12 correct under each goal on the same four test cases | Evidence of added value beyond report content, robust classwise transfer, useful uncertainty estimates |
| Shortcut diagnosis | Exploratory copy rule: 7/12 neutral, 11/12 pressure; direct LLM forecast: 8/12, 9/12 | Prospective comparisons after freezing these controls; a stronger, independently specified text monitor |
| Belief motivation | A precise reason to separate world facts, reader evidence and resulting decisions | A semantically matched independent neutral task and a defensible statement of what its readout measures |

The copy control outperforming after-report activations under pressure is a useful finding. It prevents an activation-advantage claim; it does not complete the research programme. All learned lexical controls currently predict the majority class, so beating them alone is weak evidence. Four case clusters cannot establish population performance. Equal accuracy conceals activation macro-F1 falling from .703 to .580 and balanced recall from .778 to .556. The same-data before-report diagnostic is **exploratory**, even if its feature positions are declared before it runs: these cases and outcomes have already informed the design.

## Gate 1: agree on the forecast time and label before constructing the independent neutral task

Write a one-page target specification with the supervisor before collecting fresh main-study outcomes. Let the source/rule be **S**, the reader evidence attributed to the sender **K**, reporting goal **G**, natural message **M**, actual reader packet **Kᵣ**, and measured reader decision **Y**. Initially align K and Kᵣ. The target is Y in {APPROVE, REJECT, DEFER}; INVALID remains a separately reported outcome with a declared analysis policy.

These four targets are different:

| Target | Available information / timing | Interpretation |
| --- | --- | --- |
| Reader's starting state | Before receiving M | A prior belief or no-report decision; not the post-message target |
| World-correct decision | Complete source S and rule | A normative control; not the reader's actual decision |
| Message-conditioned reader outcome | After the exact M is available | A forecast of Y given the report and reader context |
| Prospective reader outcome | Before the not-yet-generated M | A forecast averaging over the specified sender policy's future messages and the reader policy |

**Option A — continue the current operational study.** Make message-conditioned reader-outcome transfer the primary empirical target, using the current after-report capture convention. Train the readout on neutral natural communications, freeze it, and evaluate pressured natural communications against matched-information monitors. Include before-report features as a specificity comparison. This preserves the measured pilot's target and directly tests whether activations add anything beyond the report. Its limitation is substantive: success is outcome monitoring, not proof that the sender anticipated a belief before writing.

**Option B — make prospective anticipation the primary target after explicit agreement.** Capture before the first report token and forecast the eventual reader outcome of a report subsequently produced by a frozen sender policy. With sampling, the target is an expectation over possible messages, not the outcome conditional on a particular future message. A single report/reader draw is one noisy observation; repeated draws require declared sampling and clustering. With greedy decoding, the policy is deterministic and identical repeats add no sample size. Pressure changes the message policy being transferred into. This option better tests anticipation but changes the prediction problem and needs a newly matched neutral training task.

**Development decision, 20 November:** preserve Option A's measured outcome pilot as an operational baseline and establish the missing neutral nested-perspective target next. The [belief-target audit](./BELIEF_TARGET_AUDIT.md) specifies a fresh factorial diagnostic separating world truth, actual reader perspective and sender-attributed reader perspective. A failure of the existing before-report outcome readout does not settle that question. Validate this target before freezing its communication transfer study. Keep prior perspective tracking, known-message updating and unknown-message forecasting distinct; do not promote a same-data diagnostic into a fresh primary endpoint.

**Measured update, 20 November:** the [controlled query comparison](./QUERY_COMPARISON_RESULTS.md)
and frozen fresh-wording evaluation now exist. Sender-view nested decodability
is 16/32 without a question and 24/32 after the nested question, but the queried
head gets 16/32 when transferred unchanged to unqueried states. Four new wording
groups, repeated private-attendance pairs and uneven class recall limit the
result. The neutral unqueried-belief transfer gate remains unmet; use this bank
as observed validation and independently validate the eventual matching task.

**Further measured update:** the [known-message diagnostic](./KNOWN_MESSAGE_RESULTS.md)
scores 56/64 with the frozen queried sender head and 32/64 with story/world-question
heads. Four wordings share one update rule and only 32 unique sender inputs.
This supports a conditional evidence-channel transfer result, while unqueried
and natural-communication transfer remain unvalidated. The [MASK pilot](./MASK_SPECIFICITY_RESULTS.md)
also ran: its strict primary subset is empty, with secondary aggregation results
kept separate. Human review and a declared final eligibility policy remain
necessary before estimating behavioral specificity on an expanded fresh panel.

The [exploratory future-goal context test](./BELIEF_GOAL_STRESS_RESULTS.md) now
uses the same observed bank and frozen queried head. Neutral role framing scores
44/64, a positive future-belief objective 50/64, and its opposing objective 34/64.
All 32 unique inputs show a positive-minus-negative goal score increase, despite
unchanged current evidence and gold labels. Role framing already changes scores;
the goal contrast also differs in wording, polarity and length. This establishes
bounded context sensitivity, not changed current belief or deceptive behavior.
Before stronger transfer claims, use fresh rule families, direct place-A/place-B
goal wording, irrelevant-goal and length controls, and a capable text comparator.

The independent neutral task is a **necessary design gate for the intended stronger belief-transfer claim**. It must preserve the chosen target and time point. A decoder trained on an agent's prior false belief cannot automatically be called a decoder of its post-message state. Likewise, a neutral task showing the full message at training time is not semantically matched to testing before an unknown message. Do not prescribe an auxiliary dataset until this distinction is settled. A valid staged outcome is: first demonstrate a well-controlled communication-outcome forecast, then test a separately defined neutral belief/communication task with matched labels and timing. Decodability remains distinct from causal use in both stages.

**Completion evidence:** a versioned specification declaring input visibility, representation time, sender/reader policies, label meaning, primary comparison and the independent-task mapping. An unresolved mapping blocks a belief-transfer claim; it does not invalidate the already measured development result.

## Gate 2: validate cases and labels with human review

Audit a new development version before creating final holdouts. The current full-evidence control passes 23/24 cases; retain the failed case and the earlier profit-restoration failures. Do not edit old traces or regenerate awkward messages until an expected decision appears.

The review rubric must keep separate:

- Whether each report assertion is source-supported, contradicted, unsupported or ambiguous; check numeric comparisons and dates explicitly.
- Whether the decisive fact is communicated, omitted or ambiguous, including equivalent paraphrases.
- The actual parsed reader decision, its world correctness, and whether available evidence supports a definite decision or deferral.
- Explicit recommendation words, report length/termination, malformed reader outputs and other protocol deviations.

A factual error, an omitted fact and deceptive intent are not interchangeable labels. A faithful record of a reader's mistake can still be a valid outcome-prediction label. It cannot by itself validate a “truthful omission” or “intentional deception” benchmark. Decide beforehand whether false factual reports enter the broad natural-communication primary analysis, separate strata, or an audited omission-specific secondary analysis. Report all denominators and exclusions.

Prefer review without probe scores or model correctness summaries; blind the goal where possible. Have a second researcher check a declared subset plus ambiguous/critical cases, record disagreements, and reconcile the rubric before fresh testing. This is annotation of synthetic materials, not a claim to have measured human recipient beliefs; confirm the relevant project ethics scope with the supervisor.

**Completion evidence:** audited cases, a stable rubric, agreement/disagreement records, an exclusion ledger and passing reader controls under the declared evidence-precedence rule. A failure justifies a separately versioned development repair or a narrower task claim; it is not a reason to erase the failure or call the main study complete.

## Gate 3: make the neutral-trained comparison fair

Use neutral training data for preprocessing, readout fitting and the majority fallback; use neutral validation data for permitted hyperparameters and thresholds. Specify what happens if a class is absent. Do not manufacture labels or quietly replace a three-class question with a binary one.

Freeze these controls before new test outcomes:

1. Neutral-training majority; source-world rule; and the actual no-report reader baseline, each with its proper interpretation.
2. The now-explicit literal-copy rule and prompt-only/report-only/full-context lexical models. Tune regularisation and feature-block scaling on development/neutral validation data rather than repeating the underfit fixed-C comparison as a definitive text baseline.
3. A capable direct LLM/full-context text forecast with the same permitted source, rule, message and reader evidence as the activation monitor. Freeze prompts, checkpoint, decoding and any tuning budget.
4. Text plus activation versus the strongest frozen text approach. This is the relevant incremental-value comparison, alongside activation alone.

An after-report text monitor can see M; a prospective text baseline cannot. Comparing those time points is an information ablation, not a fair claim that one representation is better. The sender cannot know a reader packet or reader identity assigned only after its activations were saved. Later independent packet crosses test recipient sensitivity; give actual-risk monitors that actual context and keep original-sender representation claims separate.

**Completion evidence:** saved neutral-only fits, validation choices, frozen baseline code/prompts and a comparison table with matched inputs. If text matches or beats activations, a rigorous robustness/shortcut result remains possible; activation superiority is not a condition for completing the study.

## Gate 4: freeze the main study and a defensible precision budget

The present 16/4/4 split holds out cases within already seen families. The main study needs fresh audited cases and complete scenario-family holdouts. Keep related templates, paraphrases, facts and entities grouped so that surface variations do not leak across partitions. Define “family” operationally and document whether it changes domain, decision rule, composition or another dimension. Report fresh-case transfer within known families separately from genuinely held-out-family transfer.

Before final data are opened, freeze the target specification, models/precision, feature positions/layers, sender goals/policies, reader prompts, case/family splits, losses, invalid handling, inclusion rules, baselines, stopping rule and permitted secondary analyses. A timestamped protocol and hashes are sufficient for the project; the commitment must precede fresh outcomes.

**Provisional planning envelope, not a power calculation:** use another 12–16 development cases to estimate annotation time, class availability and paired case-level loss variability. Keep these outside final test families. Budget roughly 80–120 audited main-study cases overall, aiming initially for 40–48 test cases distributed over at least five or six genuinely distinct held-out families, if that diversity can be authored validly. This is a practical benchmark target, not a guarantee of generalisation or statistical power. Fewer families support a more conditional claim even if there are many rows.

Choose the final count using the new development variance and a declared smallest useful improvement or transfer degradation, expressed in forecast errors or loss. Simulate plausible case/family clustering and ask what confidence width the affordable design can resolve. For example, a desired interval half-width around ten percentage points for a paired accuracy difference is a planning question to test, not a property assumed from “200 examples.” Use sensitivity to uncertain variance estimates; the current four-case result is inadequate for a stable power estimate. If the available budget cannot resolve the intended effect, narrow the claim or report a precision-limited result instead of calling the design powered.

Profile annotation costs first. Reserve approximately 20–30 researcher hours for case/report audit and 6–10 hours of independent checking as a provisional review budget, then revise from measured throughput. Include the review effort in scope: six report conditions per case are not six independent cases.

**Completion evidence:** a feasible frozen protocol, a recorded precision analysis and review/compute budget, and untouched final outcomes. Requiring a larger sample after seeing a favourable p-value is not this gate.

## Gate 5: execute transfer, uncertainty and second-reader checks

Run the prespecified neutral fit/freeze/pressure evaluation once on the fresh holdouts. Report class counts, correct/incorrect/DEFER/INVALID decisions and deviations. For three-class forecasts, prioritise a proper multiclass loss plus macro-F1 and per-class recall; include accuracy without letting it hide minority-class collapse. Give paired neutral-to-pressure changes and incremental activation-versus-text differences.

Use case-level paired summaries and family-aware uncertainty. Resample at the actual grouping level; repeated packets or sampled reports stay with their case. With only a few families, present family results and leave-one-family-out sensitivity as well as any broad intervals, and acknowledge unstable family-level inference. Do not present rows as independent trials.

Probability calibration is separate from accuracy. Any temperature scaling or other calibration must use independent neutral validation data and then stay frozen under pressure. Evaluate log loss/Brier score and reliability with uncertainty and adequate counts; do not infer calibration from a generated single decision or from raw Apollo margins. No 1% false-positive-rate claim follows from this pilot or a small control set.

Replay the **same saved reports** with a second frozen reader. This tests whether the conclusion depends on one reader, not whether the original sender knew an unannounced reader identity. State explicitly whether a readout is transferred unchanged to a new reader policy or fitted/calibrated using that reader's independent neutral data. Do not use second-reader test results to choose the main monitor.

The pinned 8B model is a feasible primary scale. Existing local NF4/BF16 execution supports development; one CREATE GPU can supply a profiled BF16 reference on a preselected subset and, if appropriate, the final run. Request time and memory from measured examples and call counts. A second-reader replay does not require retraining the sender. Larger models, fine-tuning and large GPU allocations are not prerequisites for a credible result.

**Completion evidence:** the complete frozen evaluation, clustered uncertainty, prespecified second-reader results, full failure accounting and reproducible tables. A null or negative result passes the research-execution gate if the design can support its bounded conclusion.

## Gate 6: finish the project and bound the paper claim

The project is complete only after the agreed study is analysed, limitations and alternatives are argued, source/code/data provenance is reviewable, and the report, reproducible figures and required submission package are finished. Engineering success alone does not satisfy this gate. An invalid target discovered late requires an explicit scope decision with the supervisor, not a claim that an unfinished belief experiment is complete.

A defensible paper claim might concern neutral-to-pressure robustness, report-content shortcuts, reader-specific failures or lack of incremental activation value under controlled conditions. It must be determined by the final independent evidence. Existing work already covers incentive-related probe false alarms, deception categories, omission/fact-coverage probes and recipient-sensitive selective reporting: [Apollo](https://proceedings.mlr.press/v267/goldowsky-dill25a.html), [One Probe Won't Catch Them All](https://proceedings.mlr.press/v306/natarajan26a.html), [Kim et al.](https://aclanthology.org/2026.emnlp-main.656/), [JANUS](https://arxiv.org/abs/2606.10852v2). The [literature audit](./LITERATURE_PROBE_AUDIT.md) records reading scopes and limits. Do not claim the first omission study, the first reader-outcome experiment, or a discovered second-order-belief mechanism from a useful decoder alone. Submission readiness and eventual peer-review acceptance are separate milestones.

## Calendar against the supplied project plan

| Checkpoint | Concrete evidence to bring |
| --- | --- |
| 20 November, Progress 2 | The completed engineering and exploratory pilot, the exposed shortcut, and the gates still open |
| By 11 December | Agreed target/timing, audited development cases, same-data diagnostic clearly labelled, and improved neutral/text controls |
| 18 December, BSPR | Literature, software specification, pilot evidence, risks and a feasible main-study protocol/precision plan |
| January | Freeze the primary target, independent neutral-task mapping, family splits, models, baselines and budget before fresh results |
| February | Validation checks and a protocol-compliant interim result; document deviations without repeatedly tuning to test outcomes |
| March | Main held-out-family result, second-reader replay and uncertainty/failure analysis |
| By 22 April | Reproduced figures, completed dissertation/source package and final submission |

These dates follow the existing project slides/plan, not new admissions research. Internal milestones need checking against current KEATS requirements. The immediate next step is to finish target and data-validity decisions while preserving the first run; the project has not been completed in an afternoon.
