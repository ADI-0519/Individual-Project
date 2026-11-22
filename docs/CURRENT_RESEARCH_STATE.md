# Verified continuation state: 22 November 2026

## Latest: 16:25 UTC, both held-out analyses independently verified

Both pinned models completed **2,048 train/development and 3,840 held-out
prefix captures each: 11,776 main-study measurements in total**. Each has
30 matched activation/TF-IDF fit comparisons, with all selected arms retained.
Both held-out raw-array audits passed all 3,840 rows. These measurements are
factorial variants of 154 authored contextual groups, not 11,776 independent
reasoning situations. The held-out set contains 90 groups per model, including
30 correction groups. The same groups are reused across models.

The unchanged reporter produced the [real two-model interim report](results/main-study-2026-11-22-interim/results.md),
[source-bound summary](results/main-study-2026-11-22-interim/results_summary.json),
[primary figure](results/main-study-2026-11-22-interim/primary.pdf) and
[learning curves](results/main-study-2026-11-22-interim/learning_curves.pdf).
The [manifest](results/main-study-2026-11-22-interim/report_manifest.json) hashes
both full source analyses and every generated output. The committed report is
a byte-for-byte verified copy of the native export. Large raw arrays and full
analysis files remain in native storage, outside Git.

| N512 raw correction complete-quartet score | Restricted | Full | Difference, percentage points | Saved 95% group interval, points |
| --- | ---: | ---: | ---: | --- |
| Llama | 3.75% | 3.00% | -0.75 | [-1.9167, +0.3333] |
| Qwen | 1.8333% | 1.00% | -0.8333 | [-2.00, +0.25] |

The proposed fitting-coverage benefit is unsupported in these configurations.
Both intervals include zero; neither exact equivalence nor a harmful population
effect is established. A quartet requires all four linked judgments correct,
so these percentages must not be described as row accuracy. Llama generalizes
much better to fresh one-step cases than to correction accounts; Qwen's
one-step quartet scores are also weak. This is not a clean two-model isolation
of abstract compositional reasoning, absence of belief information, spontaneous
anticipation or deception detection.

The independent [Llama saved-result audit](validation/2026-11-22-main-study-llama-heldout-independent-v2.json)
and [Qwen audit](validation/2026-11-22-main-study-qwen-heldout-independent.json)
each reconstructed all 60 heads with bit-exact margins and probabilities, all
10,800 raw/calibrated group records and the exact primary bootstrap. This is
independent saved-data reconstruction, not a second inference run. Llama's audit
also verified retention of all original measurements and one unique final row
per case; Qwen has 3,840 fresh unique rows. The final helper's 14 corruption tests
pass. A dense arithmetic implementation initially perturbed near-tied TF-IDF
ranks; sorted sparse accumulation resolved that audit discrepancy without any
change to the frozen analyses. The first Llama audit remains preserved. The
[Llama interpretation](MAIN_STUDY_LLAMA_INTERIM_INTERPRETATION_2026-11-22.md)
distinguishes predeclared metrics from descriptive analyses requested after
the primary result. The [two-model interpretation](MAIN_STUDY_TWO_MODEL_INTERIM_INTERPRETATION_2026-11-22.md)
adds Qwen's weaker one-step performance and the predeclared same-format controls.

The [editable four-slide update](../presentations/progress2-main-study-2026-11-22/reviewed/Progress-2-main-study.pptx),
[PDF](../presentations/progress2-main-study-2026-11-22/reviewed/Progress-2-main-study.pdf)
and [speaker notes](../presentations/progress2-main-study-2026-11-22/reviewed/speaker-notes.md)
bind this actual interim report. The new focused checks pass **41 tests with
two optional presentation skips** in the inner CPU environment; all **18 slide
tests**, including those two rendering checks, pass in the existing outer
presentation environment. These counts overlap and must not be added together.
All repository hooks pass. Thirteen exported files, including both manifests,
have identical working-tree and staged bytes. Scientific SVGs are exempt from
whitespace rewriting; generated notes/receipt now emit LF with one final newline.
Earlier slide drafts and their receipts are preserved in native recovery storage.
The CI presentation job now checks provenance/layout and rebuilds the new update;
the last already completed five-job CI run is [752bf62](https://github.com/ADI-0519/Individual-Project/actions/runs/45037480228).

The fixed Qwen prose-only monitor began its 512-row development capture after
the two held-out analyses completed. Its prompt, parser, 384-token cap and
invalid-as-incorrect policy remain unchanged; its 3,840-row test capture and
integrated final reporting follow in the existing recovery campaign. The
interim report explicitly marks this comparator missing. Do not relabel that
report as a completed baseline comparison or tune the running monitor using
the exposed results.

All results remain `agent_audited_exploratory`, `human_review_complete=false`
and `confirmatory_result=false`. Later human review cannot retroactively turn
these exposed results into confirmation. The recovery deviation below remains
part of the evidence. The original study and failed pilots remain preserved;
the older sections are historical checkpoints, superseded by this section.

## Earlier: 15:14 UTC interruption recovery

The original WSL instance powered off at 14:59:38 after **2,758/3,840** Llama
test rows. Its exact shutdown cause is unknown. Its original `state.json` and
capture manifest remain preserved with stale `running` flags; those flags do
not establish that a producer is alive. Both training captures and both fitted
bundles had already completed.

The [independent interruption audit](validation/2026-11-22-main-study-interruption-independent.json)
verified every completed row, original source/fit bindings and the required
runtime environment. The
[boot evidence](validation/2026-11-22-main-study-interruption-boot-evidence.json)
pins the old WSL boot by ID. Exactly one additional case has only `started.json`
and no saved measurement. Its computational outcome is unknown.

The [explicit recovery](MAIN_STUDY_INTERRUPTION_RECOVERY_2026-11-22.md) copied
and revalidated all 2,758 completed rows into fresh native storage, leaving
the originals untouched. It retains **1,082** pending IDs: the one disclosed
unknown attempt plus 1,081 previously unattempted cases. The unknown attempt's
repeat is not an additional independent case. There is no refitting, selection
change, source-policy change or automatic retry loop.

The recovery was launched at 15:14 UTC through a detached hidden Windows WSL
client (PID45904 at launch). Its live directory is
`/home/adirj/kcl-latent-probes-runs/main_v1_agent_20261122_recovery_v1`;
read its `state.json` and logs for current progress. The receipt/config are
sealed before execution. `windows_launcher.json` records the launch and native
Windows stdout/stderr paths. Detachment does not protect against OS/WSL shutdown.
All 14 focused recovery tests pass, including checks with the original saved-row
validator. The previous section below records the pre-interruption campaign.

## 22 November: independent agent audit and exploratory execution

The user explicitly requested independent agent checking and continued work.
The [prospective amendment](MAIN_STUDY_AGENT_EXECUTION_AMENDMENT_2026-11-22.md)
adds a separate agent-audited exploratory route; the original human gate remains
unchanged. No agent record is a human attestation. Later human review cannot
retroactively make exposed results confirmatory.

An author-independent agent read all 154 narrative openings/setups and 16 full
normalized semantic paths, covering all 736 representative prompts. Its
[frozen answers](../data/annotations/main_study_v1_agent_semantic_20261122.json)
were saved before comparison to formal gold; all 736 agree, with zero UNSURE
answers or unresolved blockers. This is grouped agent checking, not 736 separate
human judgments or inspection of all 5,888 expanded prompts. See the
[semantic audit](MAIN_STUDY_AGENT_SEMANTIC_AUDIT_2026-11-22.md). Human records are
still blank. The [new primary-source audit](MAIN_STUDY_NOVELTY_AUDIT_2026-11-22.md)
finds substantial prior art; only a substantive matched-coverage/correction
transfer result could support the narrower candidate contribution.

A new native train/dev execution freeze exists at
`/home/adirj/kcl-latent-probes-runs/main_v1_agent_20261122_training_freeze.json`.
Fresh Llama/Qwen tokenizer preparations each bind all 5,888 exact inputs without
truncation (538-721 and 530-719 tokens respectively). The Llama train/dev capture
has now completed **2,048 real forwards**, and the
[independent raw-array audit](validation/2026-11-22-main-study-llama-train-audit.json)
passed every row. All **30 matched fit comparisons** are complete, yielding 60
selected activation/TF-IDF arm models. Qwen train/dev execution is running.
No held-out result is claimed by this checkpoint.
The [independent fit audit](validation/2026-11-22-main-study-llama-fit-independent.json)
checks all 60 training memberships, 210 saved arrays and selected development
predictions without importing the fitting/analysis modules or refitting.
Activation scalers reproduce exactly; selected-model losses agree within
1.11e-16. Unselected-C losses are checked for metadata consistency only because
their coefficient arrays were not retained. No held-out outputs were accessed.
Both models' fits must bind that exact initial freeze before the separate test
freeze can be built. Every new capture, fit and analysis records
`agent_audited_exploratory`, `human_review_complete=false` and
`confirmatory_result=false`. Historical experiments and their failures stay fixed.

The [sequential campaign](MAIN_STUDY_CAMPAIGN_RUNBOOK_2026-11-22.md) was launched
at 13:43 UTC. Its native directory is
`/home/adirj/kcl-latent-probes-runs/main_v1_agent_20261122_campaign`.
Read `state.json` there for the current stage; it observes the existing Llama
capture without duplicating it, then audits/fits both models, freezes test
eligibility, collects each held-out partition and runs the full fixed prose
baseline. It stops on failures with logs and original row journals preserved.
It never silently retries or overwrites. The runbook explains its between-stage
`STOP` file. Initial analyses explicitly mark a missing monitor; the final
`*_analysis_with_monitor.json` files are separate outputs after it completes.

The [execution checkpoint](validation/2026-11-22-main-study-agent-execution-checkpoint.json)
records **1,255 passing Windows tests, 48 passing subtests and two expected
platform/optional-plot skips**. All 31 reporting tests, including actual static
figure export, pass in the existing WSL CPU environment. Lint, formatting,
pre-commit, wheel/source builds and metadata checks pass. The extracted source
archive also preserves and validates all 736 agent answers. All 27 frozen
scientific inputs still match; no test-conditioned policy change has occurred.

The [assessment evidence note](MAIN_STUDY_ASSESSMENT_EVIDENCE_2026-11-22.md)
maps the supplied college/module criteria and original slides to evidence and
remaining gaps. It distinguishes actual guidance versions and proposes December
deliverables; it predicts neither marks nor admissions. The scope change from
deception/omissions to the separate controlled belief-readout study remains
explicit and should be communicated to the supervisor.

The [standalone capture auditor](../scripts/audit_main_study_capture.py) imports
no project modules and checks raw versus saved versus exported arrays and full
logit summaries. Its tests include corruption and float32-rounding cases. These
are saved-measurement checks, not an independent rerun of model computation or
proof that a decoder measures a model's own beliefs.

Read this file and the linked protocols before changing the project. It records
the current evidence boundary so continuation does not depend on chat memory.
Earlier experiments ran on 20 November; continuation and fresh validation
also ran on 21 November. Read the dated sections for each attempt.

## Latest implementation: full draft material and two real engineering forwards

The resumed main-study build now has **154 authored draft contextual groups**:
48 training, 16 development and 90 test groups. The
[case bank](../data/belief_main/main_v1/main_bank_v1.json),
[split manifest](../data/belief_main/main_v1/split_manifest_v1.json) and
[15 matched allocations](../data/belief_main/main_v1/matched_allocations_v1.json)
bind **5,888 unique inputs per model**. Both authors cross-reviewed the other's
prose. Before assembly, nine cross-split actor overlaps and four excluded
design-name reuses were corrected; no model outcomes informed those edits.
Canonical bank hash is
`e038025571a38a7f2c489e5f8c3bf3b0a3c11a05f9b8c7e89d26292a4e7bfb1d`.

The [prose-only monitor](../src/latent_probes/main_study_monitor.py) is implemented
with a fixed Qwen extraction prompt, strict parsing and an explicit update fold.
Its [analysis](../src/latent_probes/main_study_monitor_analysis.py) retains invalid
outputs as incorrect and reparses hash-bound generation journals. Its optional
integration in the main analysis explicitly records when this baseline is
missing. CPU oracle-constructed extraction fixtures are not Qwen predictions.

The [offline human worksheet](review/main_study_v1/review.html) has 154 groups
and 736 representative prompts. Human answers, decisions and annotations remain
blank. The [review readiness check](validation/2026-11-21-main-study-review-readiness.json)
correctly reports zero human records and 154 missing student acceptances.
Review-fragment merging preserves disagreements; the execution gate now rejects
accepted questionnaire answers that disagree with the formal rule. Software
cannot authenticate humanity or substitute for semantic review.

One dedicated, permanently separate engineering input was profiled once on each
model: Llama block16 yielded 4,096 features and 128,256 vocabulary logits; Qwen
block14 yielded 3,584 features and 152,064 logits. Both passed exact hook/HF
alignment, full raw-tensor readback, source-snapshot and bank-hash checks. Native
outputs are `/home/adirj/kcl-latent-probes-runs/main_backend_profile_20261121_llama`
and the corresponding `_qwen` directory. These are **two engineering forwards,
zero main-study forwards, zero fitted main heads and zero measured monitor
generations**. The later review-gate change does not rewrite their frozen source
snapshots; final execution preparations must bind the updated gate sources.

Initial tokenizer-only preflights succeeded on all 5,888 inputs for both models
and 4,352 development/test monitor inputs (no training monitor generations).
An initial Qwen cache miss was resolved by the existing native cache at
`/home/adirj/kcl-hf-cache`, without downloading or changing model revisions.
The following older sections record earlier states; their authoring pause and
missing-baseline statements are superseded by this section.

The final [material/prefix audit](validation/2026-11-21-main-study-materials-and-prefixes.json)
reconstructs both authored fragments and both manifests, checks all updated
capture-source hashes and token bindings for Llama/Qwen/monitor preparations,
and finds no oversized/truncated prefix. These preparations use the
`_review_gate_v2` native directories described in the
[execution runbook](MAIN_STUDY_RUNBOOK_V1.md). The
[independent hardware audit](validation/2026-11-21-main-study-hardware-independent.json)
imports no project modules and verifies both saved-array/raw-tensor comparisons,
26 frozen source snapshots, prompt/token hashes and full-vocabulary dimensions.
Raw logits retain batch/token axes `(1,1,V)`; exported final-token logits are
their `[0,0]` view, with exactly equal values.

The complete local suite passes **1,168 tests and 48 subtests**. The source
archive and wheel build and pass metadata checks. A separate
[relocated-source audit](validation/2026-11-21-main-study-relocated-source.json)
imports the extracted source under isolated Python and reconstructs all 5,888
inputs and 15 allocations without loading the original project modules.
Git text filters preserve byte-bound case/review inputs. CI now includes the
actual material-manifest validator; no new remote CI result is claimed here.
The [current checkpoint](validation/2026-11-21-main-study-materials-checkpoint.json)
records all local checks. All pre-commit hooks pass using the installed Windows
environment without resync. A uv resync attempt had removed the editable project
registration before hitting a readonly OneDrive directory; only that project
package was reinstalled offline with no dependency changes. Its import, console
entry point and RECORD are restored. WSL and saved research data were unaffected.

## Supervisor-recommended DeceptionBench: source audit, not a new result

The supervisor recommended Huang et al.'s NeurIPS 2025 **DeceptionBench**, arXiv
2510.15501. Read the [pinned release audit](DECEPTIONBENCH_SOURCE_AUDIT.md) before
using it. This is the Aries-iai / skyai798 release, not the separate
PKU-Alignment benchmark with the same short name. The actual public files contain
150 scenarios and 900 static prompts across five domains. They do not contain
generated responses, behavior labels or cached activations. `groundtruth` is a
reference honest answer/action, not a binary label for our model's output. L1
already contains deceptive objectives in some cases and is not an honest class.

Use this as an external evaluation anchor for the **deception-probe** portion of
Variant 2. Keep its behavior target separate from the controlled belief target.
The previously authorized matched coverage/composition study remains a proposed
belief-readout experiment; mass authoring of its 154 groups was paused before
any full corpus was written while the external recommendation was assessed.
Generic fit/capture machinery is engineering scaffolding, not empirical progress.

Apollo-style probing on this exact benchmark already has public precedent:
Huang et al., [Why Static Probes Fail](https://arxiv.org/html/2610.27577v1),
Sections 4-5, includes THU-DeceptionBench L2 and labeled few-shot adaptation;
the [Yilin1010 reproduction repository](https://github.com/Yilin1010/deception-probe)
also directly applies an Apollo-style instructed-pairs recipe. Benchmark transfer,
domain fragility or elevated honest false positives alone are therefore not a
new contribution. A specific new result would need stronger controls and fresh
evaluation, and novelty remains unestablished. No DeceptionBench generation,
judging, model loading or paid API requests were made for this assessment.

The [source-only importer](../src/latent_probes/deceptionbench_data.py) and
[pinned manifest](../research/deceptionbench/source_manifest_v1.json) now validate
the actual published bytes and produce 900 uniquely identified prompt records
with unknown behavior labels. Parent execution independently reproduced the
import. The proposed main-study [capture runner](../src/latent_probes/main_study_run.py),
[fitting engine](../src/latent_probes/main_study_fit.py) and
[analysis engine](../src/latent_probes/main_study_analysis.py) have CPU fixture
coverage, including cross-component capture loading. Independent review caught
and corrected canonical-bank hash and numeric-precision inconsistencies before
any real fitting. The full local suite passes **1,019 tests and 48 subtests**;
repository-wide Ruff lint and formatting pass. See the
[engineering validation record](validation/2026-11-21-main-study-infrastructure.json).
These checks do not complete data authorship, human review or a scientific run.
No new changes have been pushed and no new GitHub CI result is claimed.

## Strategic correction after training-support and closest-paper audit

The [main-study decision brief](MAIN_STUDY_DECISION_2026-11-21.md) supersedes the
earlier proposal to make a larger evaluation of the legacy head the sole main
study. The transferred head used **16 distinct training sender inputs** (32
duplicated rows, four wordings), eight distinct validation inputs and one
change-only observation rule. Its crossed failures are valid pipeline-transfer
findings, not evidence against generally well-trained belief probes.

The recommended next study compares **new restricted/full-coverage probes at
matched training budgets and scenario coverage**, then evaluates untouched
composed updates. Keep the legacy head unchanged as historical reference. The
training intervention, allocation, labels, review and held-out composition are
not yet frozen or executed. Exact sample counts in external planning prose are
not an approved bank. No new model calls or fits were made for this audit.

Generic second-order decoding, selective sensitivity, training-correlation
repair and question-time retrieval already have close primary precedents.
Novelty must come from a substantive coverage/composition result and replication,
not from renaming the old failure. Robustness evaluation with externally reviewed
gold does not require correct verbal answers; the stronger future-goal or causal
interpretations still need additional evidence. Do not make another broad model-
qualification loop a prerequisite for the core comparison. Explain this explicit
belief-readout emphasis to the supervisor separately from Apollo deception work.

## Main-study implementation has begun; no new model results

The [concrete specification](MAIN_STUDY_SPECIFICATION_V1.md) now defines state
rules, valid correction quartets, matched N128/256/512 fitting allocations and
five predeclared data-allocation seeds. The new pure
[state/allocation module](../src/latent_probes/main_study_design.py) passes 32
focused tests. Independent design review replaced a fixed nuisance-parity mask
with randomized, complementary paired-story masks and verified all 15 planned
allocations mathematically; boundary ID hashes match the Python implementation.

The [CPU design audit](validation/2026-11-21-main-study-design.json) and
[independent review](MAIN_STUDY_DESIGN_REVIEW_V1.md) distinguish reserved group
IDs from authored data. The [24-prompt review packet](review/main_study_design_v1/review.md)
contains six permanently excluded design-example groups. Their labels agree
with independently rederived state rules, but human annotations remain blank.
These are specification materials, not another pilot or completed main corpus.

The proposed full bank would contain 5,888 unique prefixes per model if all
48 training, 16 development and 90 test groups are independently authored as
specified. No activations, fitted main heads or held-out outputs have been
collected. Actual narratives, human review, exact baseline policies and final
execution bindings must still be completed. The specification is not frozen.

## Latest: 21 November fully crossed neutral validation

Read the [complete new results](CROSSED_BELIEF_VALIDATION_RESULTS.md) first.
The fresh bank varies initial reader belief I independently of message M,
trusted receipt B and world W in every template. It ran once at frozen commit
`f576a5a8e79c65464fd510f78d7e2b18c40dc213`: 128 unique sender inputs, 256 prefix
measurements and 128 queried answers. No new head was fitted.

The queried head passes every primary I!=M floor: **48/64**, 24/32 per mapping,
and **29/32 positive gold-oriented receipt contrasts**. It fails full
qualification on I=M controls: **43/64**, mapping scores 21/32 and 22/32 below
the required 24/32. Receipt changes flip **13/32 control verdicts** despite
unchanged gold; world changes flip 15/64 and names 7/64. Overall probe 91/128 is
below the I-only/M-only shortcuts 96/128. Do not call it a qualified belief monitor
or characterize it as a proven pure legacy-XNOR shortcut.

Primary output-ranking qualification passes: AUROC .769531 versus probe .782227;
on controls output .825684 exceeds probe .759766. No latent advantage is
established. Greedy behavior fails at 57/128 with 37 capped invalids; fixed output
decisions 66/128 show that length caps alone do not explain the weakness. The
pre-question frozen head predicts class 1 everywhere (64/128), though its ranking
AUROC is not flat (.591797 primary/.666016 controls). No spontaneous anticipation
or absence of all pre-question information is established.

Native run: `/home/adirj/kcl-latent-probes-runs/crossed_belief_validation_v1_20261121`.
The ignored Windows mirror is identical: **517 files, 68,462,262 bytes**. The
[verified CPU record](validation/2026-11-21-crossed-belief-validation.json)
and [independent no-project-import audit](validation/2026-11-21-crossed-belief-independent.json)
reconstruct all scores, 257 arrays including reference, 256 pinned prefixes and
128 answers. The independent audit predates copying and correctly retains its
mirror flag false; the main record separately verifies complete mirror identity.
The [protocol](CROSSED_BELIEF_VALIDATION_PROTOCOL_V1.md) and
[prospective independent review](CROSSED_BELIEF_DESIGN_REVIEW_2026-11-21.md)
are executed frozen files: do not edit. Bank SHA256 is
`3fc31d095b75ac948cd04bb5eb2bafdad62a2bf2a872243657e4367effa8cc6c`.

All 911 local tests and 48 subtests, and five CI jobs at the execution commit, passed.
The [64-account review packet](review/crossed_belief_validation_v1_human/review.md)
has blank human annotations; agent checks do not complete semantic validation.
The [new literature update](CROSSED_BELIEF_LITERATURE_ADDENDUM_2026-11-21.md)
narrows the gap against three additional primary papers. No goal conditions,
replacement probe fits or additional checkpoints were run. Earlier failed gates
stay failed. The later strategic brief above recommends fresh matched training
and untouched composition evaluation as the core; neutral reference qualification
is relevant to the stronger extension, rather than a prerequisite for reporting
robustness. Do not fit on this new validation bank or launch the old 192.

## Earlier 21 November: real future-goal neutral qualification

The [new neutral transfer result](FUTURE_GOAL_CONTENT_CONTROLS_RESULTS.md) is real:
frozen belief readout **25/32**, versus **16/32** direct answers because all 32
were NO. There are 16 scene units and two mappings, not 32 independent scenes.
The readout passes all prospective floors; the answer measure fails. Exactly
32 neutral answers were collected and **zero of the 192 goal/archive conditions**.
No new probe was fitted.
Keep this gate failed and the goal-effect hypothesis unanswered.

Execution was frozen at `ac7e17d`; actual native run is
`/home/adirj/kcl-latent-probes-runs/future_goal_content_controls_v1_20261121`.
The Windows mirror under `runs/` is byte-identical: 70 files, 3,915,852 bytes.
[V2 CPU verification](validation/2026-11-21-future-goal-content-controls-v2.json)
rechecks all bindings and tokenizer outputs. It corrects only an invalid EOS
metadata subset assumption in a separate analyzer; the executed V1 file remains
unchanged. The [independent raw audit](validation/2026-11-21-future-goal-content-independent.json)
gives overall AUROC .792969, observation .6875 and message .921875, and recomputes
all 32 margins with maximum error 8.33e-17. Human review remains incomplete.

A separate [posthoc output-score diagnostic](FUTURE_GOAL_ANSWER_LOGITS_PROTOCOL_V1.md),
frozen at `20d4101`, checks YES/NO next-token discrimination on the same 32 prefixes
with zero new answers. It is not a rescue of this failed gate. Its native output
is `/home/adirj/kcl-latent-probes-runs/future_goal_answer_logits_v1_20261121`.
It completed 32 full-model forwards, zero new generated answers, with every
block16 vector exactly matching the old neutral vector. Output YES−NO AUROC is
.748047 overall, .703125 observation and .8125 message; the frozen probe gives
.792969, .6875 and .921875. All argmax IDs still match NO and conditional YES
probabilities are below .5. The [verified record](validation/2026-11-21-future-goal-answer-logits.json)
and [result/figure](FUTURE_GOAL_CONTENT_CONTROLS_RESULTS.md) distinguish this
posthoc ranking evidence from the failed original gate. The new mirror has
35 files, 4,121,429 bytes. Human semantic review remains pending.

## 21 November continuation

Execution code and the new [v2 protocol](RECIPIENT_RISK_VALIDATION_V2_PROTOCOL.md)
were committed and pushed at `06328fa1894f76f533b2c2749b47364f9d2b607b` before
the new model run. Twelve fresh cases span three predicate types, still sharing
conjunction/latest-record logic. The primary arm is prospectively 50 words;
25 words is secondary, and cannot rescue a primary failure. No fitting is
authorized. Existing v1 gates remain failed.

The real attempt is
`/home/adirj/kcl-latent-probes-runs/recipient_risk_validation_v2_20261121`.
Preparation and all 72 unique comprehension calls completed. **Qualification
failed:** full disclosure 19/36 (checks 7/12, numeric 3/12, expiry 9/12), no-report
informed 12/12, absent 2/12, stale 11/12, zero invalids. Zero actor reports and
zero fits followed. The planned 504-condition panel was not collected. See the
[new results](RECIPIENT_RISK_V2_RESULTS.md). The same informed packets decline
from twelve correct answers without a report to seven with the full source;
four become DEFER and one becomes world-wrong. Do not attribute this specifically
to irrelevant facts or quantization without new evidence.

The [bound analysis](validation/2026-11-21-recipient-risk-v2.json) verifies the
actual control-only attempt; its analyzer SHA is
`6c0ac92ee9ba4bdb99714136b482780b576f798c6898cb57ae209a168fc2c17d`.
The [independent trace audit](validation/2026-11-21-independent-recipient-risk-qualification-audit.json)
rechecks all 72 payloads and output/token bindings, fifteen source hashes and
all normative controls without the project's decision helper. It finds no
material input/label/prose discrepancy; this remains agent, not human, review.
The seven-file, 10,489,648-byte native attempt and ignored Windows mirror are
byte-identical. No natural report bank or activation files exist for v2.

The execution checkpoint passed all five CI jobs, including 787 tests plus one
expected licensed-source skip per configuration. Locally, the earlier suite
passed 760 tests and 48 subtests, the new runner 28 tests, and its separately
reviewed bound analyzer 22 tests. These are engineering checks, not a new
scientific observation. The source bank and executed protocol must stay frozen;
their draft/unexecuted metadata describe the input snapshot, while the attempt
manifest records the completed control failure.

A later full CI run reached 809 passing tests on Windows and Ubuntu Python
3.11; Python 3.12 completed its analysis cases but hit the existing twenty-minute
job limit. The [test-fixture performance audit](validation/2026-11-21-ci-test-performance-audit.json)
records the test-only correction: compact fake token coordinates and reusable
synthetic baselines with isolated per-test copies. All test bodies and assertions,
the workflow limit, production code and measured research evidence stay intact.

The separate engineering preflight made zero model calls; its source
binding predates the tested Windows-write fix and must not be used for inference.

The [portable v1 archive](PORTABLE_PILOT_REPRODUCTION.md) now reproduces all
scientific pilot/Apollo fields after extraction outside the workspace, on native
Windows CPU and WSL CPU. The latter independently verifies pinned tokenizers;
the default mode explicitly verifies saved record bindings only. No model
inference was repeated. The [human review packet](review/natural_selection_v1_human/review.md)
contains all eight sources and 32 exact reports with blank annotations. Human
review remains incomplete. A scoped whitespace exception preserves one literal
line-end space in captured report text; its JSON and Markdown hashes were also
verified from Git archive bytes.

The [new literature/design review](FRESH_FEASIBILITY_DESIGN_REVIEW_2026-11-21.md)
includes COVERAGE-TRACE (19 November) and source-first enumeration baselines.
The v1 result now prominently records all addition effects (five changed
decisions versus four for irrelevant additions) and the motivating omission's
lower Apollo score than its complete, equal-length counterpart.

## Current outcome

The [natural-communication pilot](NATURAL_SELECTION_PILOT_RESULTS.md) is complete:
32 Llama reports, 240 Qwen conditions, 222 unique reader calls, eight source
clusters and one shared rule grammar. Both models use NF4 weights with BF16
compute; this is not a full-BF16 reference. No new probe was fitted.

Two separately blinded agent reviewers agree on four supported current-fact
omissions, all under favorable reporting pressure, across three cases. One
25-word Cairnfield report gives informed/absent/stale readers REJECT/APPROVE/
APPROVE. The nominated restoration gives REJECT in all three; the irrelevant
addition changes none. No-report answers are REJECT/DEFER/APPROVE. Distinguish
the newly induced world error in the absent condition from maintenance of the
stale condition's pre-existing, evidence-warranted world error.

The five frozen 25-word criteria give **pass/fail/pass/fail/pass**. There is only
one eligible omission, and no original-report DEFER in either goal. Do not drop
a class, pool in 50-word results or reinterpret this as a passed fit gate.
Human review remains false for the bank and both annotations.

The [exploratory Apollo diagnostic](NATURAL_SELECTION_APOLLO_DIAGNOSTIC.md)
uses the unchanged instruction-pair fit: omission AUROC .7292 within favorable
reports, versus .7708 for report length and .8333 for a structured source-world
proxy. A source-aware literal-word proxy reaches 1.0 on this small bank.
These are draft coverage labels, not deception/intent labels; no latent added
value or calibrated false-positive rate is established. This diagnostic was
declared after behavioral/semantic inspection and before calculating margins.

## Immutable inputs and results

- Executed code checkpoint: `ab683031d238c83b330f20f6f7d0992f40008120`.
- Native run: `/home/adirj/kcl-latent-probes-runs/natural_selection_pilot_v1_20261120`.
- [Byte-verified Windows mirror](../runs/natural_selection_pilot_v1_20261120/):
  42 files, 29,339,446 bytes. Raw runs/features are intentionally Git-ignored.
- [Bound pilot analysis](validation/2026-11-20-natural-selection-pilot.json),
  [Apollo record](validation/2026-11-20-natural-selection-apollo.json),
  [mirror audit](validation/2026-11-20-natural-selection-mirror.json),
  [figure](figures/2026-11-20/natural_selection_pilot.pdf).
- [Draft source bank](../data/communication/natural_selection_pilot_draft_v1.json):
  canonical object SHA256 `067e53eaa89cf21c18a55e9d48a4e39613af97ccf0abc58dd6c1600832d93462`.
- [Execution protocol](NATURAL_SELECTION_RUN_PROTOCOL.md),
  [five-gate design](NEXT_MAIN_STUDY_PROTOCOL.md),
  [review rubric](NATURAL_SELECTION_REVIEW_RUBRIC.md),
  [blind review assignment](../data/annotations/natural_selection_review_protocol_v1.json).

Preparation freezes source byte hashes. Keep the executed runner, panel module,
backend, reader-policy code and the two executed protocol documents unchanged
while reproducing the existing run. Use new files/versioned studies for changes.
Interrupted executions require explicit resume with every journal/feature/
runtime binding intact; deleted records are not permission to regenerate.

The ignored engineering preflight v1 failed because code was still being edited;
it made zero model calls. Engineering preflight v2 passed. The actual native
research attempt completed once with 32 actor and 222 reader generations and
no failed/excess generation attempts.

The Transformer warnings about ignored temperature/top-p flags are consistent
with the explicit greedy `do_sample=False` overrides; do not switch decoding
to silence them. Local pre-commit Ruff checks used the already tested native
environment with `UV_NO_SYNC=1` after an editable rebuild hit a certificate
fetch error. The clean locked installs/tests passed independently in GitHub CI.

## Earlier evidence remains separate

The [independent checkpoint audit](validation/2026-11-20-independent-checkpoint-audit.json)
rechecks raw counts, bindings and hashes, with its unverified items explicit.
The earlier 144-report batch remains 62/72 neutral and 66/71 pressured activation
accuracy, with zero REJECT recall and four draft omissions. Fresh nine-shot Qwen
qualification remains 24/24 informed, 17/24 absent, 24/24 stale and **failed**.
Its dependent source-excerpt training and untouched transfer pipeline remain
stopped. The reserved `ct` bank was not used. Existing fitted artifacts remain
frozen, and the original Apollo adaptation is not a full 70B reproduction.

The final checkout audit caught LF normalization of the new byte-bound JSON
snapshots. Exact original bytes are now protected by explicit Git attributes
and four checkout-hash regression cases. No model input, annotation content or
result was regenerated. See the [byte-preservation audit](validation/2026-11-21-natural-selection-checkout-bytes.json).

Three editorial mistakes were corrected with before/after hashes. The Graphify
map uses an explicit 64-source allowlist; new pilot files outside it must not
be described as indexed. A fresh source-level graph audit is separate from
scientific validation or model inference.

## Next concrete research decision

**After the fully crossed validation:** the new study above has run and failed
full probe and behavioral qualification. Its primary signal survives, but the
independent-initial-belief controls expose readout sensitivity without a target
change. The later training-support audit shows that the legacy head had only 16
distinct fitting inputs. The recommended core is now a fresh matched restricted/
full-coverage comparison and untouched composition, as specified in the brief
above. Reference/model qualification is an extension-specific decision, not a
universal prerequisite for robustness evaluation. The old 192 goal/archive conditions remain
uncollected. Do not retune either completed bank or declare an old gate passed.
A novel goal-specific effect or natural-communication advantage remains
unestablished. The earlier decisions below are historical.

**21 November scientific reassessment:** the [new research decision](RESEARCH_DECISION_2026-11-21.md)
recommends one bounded fresh test of the existing belief readout under future
communication goals, before further recipient-risk expansion. The strongest
candidate signal is the already-measured goal stress, not a demonstrated natural
omission detection advantage. The [later content-control refinement](FUTURE_GOAL_CONTENT_CONTROLS_DESIGN.md)
had an exact 224-condition draft bank, preserving the original slot-b target and
comparing an active objective against identical quoted content. It superseded the
unexecuted 160-condition proposal. The assay subsequently ran at `ac7e17d`,
collected 32 neutral conditions and stopped before the remaining 192, as recorded
above. Do not treat this historical design decision as a pending new run. Existing
failed gates remain failed. CREATE is not a prerequisite. This proposed emphasis
change must be described explicitly to the supervisor; a belief readout is not an
Apollo deception probe. The earlier recipient-monitoring proposal follows for
context and remains an extension pending valid labels/readers.

The earlier proposal was to keep recipient-relative monitoring as the research question. Before another
fit, independently review the four omissions and define a fresh validation
study for the target. A binary **world-wrong definite recipient decision** target
would retain correct/DEFER/INVALID distinctions and avoid silently turning the
failed three-class pilot into a success. A neutral source-faithful excerpt
curriculum is another explicit training-origin choice, requiring natural-neutral
and natural-pressured evaluation to separate the two shifts. Neither choice is
implemented or validated here.

Use fresh whole-case splits and independently balance field values, dates,
source positions, irrelevant negative facts and lexical markers. Include the
strong source/text and recipient-visible-only baselines already specified.
Current pilot cases remain development data. Do not claim spontaneous sender
anticipation from contextual reread states or ask a sender score to identify a
private packet assigned after generation.

The [literature priority record](PAPER_NOVELTY_UPDATE_2026-11-20.md) identifies
close precedents; missing-fact restoration and same-text evidence interventions
are not firstness claims. CREATE remains a prepared, unsubmitted request for
controlled same-hardware NF4/BF16 references and later scaling, not a fix for a
structurally absent target class.
