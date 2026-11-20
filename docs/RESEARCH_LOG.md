# Research and implementation log

## 20 November 2026 — Direct nested target, elicitation controls and composition

Corrected an overly broad timing interpretation: an after-report state still
precedes the reader's answer and can forecast it. It does not demonstrate
anticipation before composition or added reader-perspective information beyond
the text shortcut. The original outcome readout was not directly trained on a
nested belief label, so its failure cannot establish belief absence.

A primary-methods review covers Zhu, Bortoletto, Lookbacks, ToMi and Kim, with
versioned links and exact reading limits in the
[target audit](./BELIEF_TARGET_AUDIT.md). The new construct diagnostic contains
64 prior vignettes, eight independent truth/perspective combinations per wording
group, plus16 trusted-message controls. Four groups train, two validate, two test.
Two information views enforce a hidden-attendance impossibility control.

The genuine pinned8B run produces160 feature rows from120 unique prefixes and
400 behavior rows from320 calls. Captures precede every perspective question
and answer. Primary block16 nested heads score9/16 omniscient and8/16 sender;
their losses are near chance. All prior nested answers begin NO. Layer hooks,
prefix repeats, output hashes, C selections and primary exported probabilities
pass independent checks. The numeric heads are frozen as baselines, not promoted
to validated belief decoders.

One frozen A/B question-format diagnostic adds400 rows/320 calls, retaining
the exact scenes. All217 strict-invalid answers already contain a leading A;
supplemental extraction yields64/64 world judgments in the omniscient view,
but both prior nested conditions remain constant-A and32/64. Original strict
scores and format violations remain separately visible.

A post-hoc CPU factual-composition control obtains14/16 on the sender test
versus8/16 for the direct nested head, using independently decoded world and
attendance facts plus an external rule. It falls to6/16 on validation, so
neither robust composition nor spontaneous nested representation is claimed.
The original before-report collapse is also sensitive to scaling: a
neutral-validation-selected alternative gets8/12 under pressure, merely the
majority baseline. All source experiments remain unchanged.

See [full measured interpretation](./BELIEF_DIAGNOSTIC_RESULTS.md),
[fitting diagnosis](./validation/2026-11-20-prereport-shift-diagnostic.json),
and [composition record](./validation/2026-11-20-belief-composition.json).
The next gate is representation timing and reliable elicitation, then fresh
neutral-task validation before communication transfer. Three agents contributed;
no connected-app plugin or new model download was used.

Final local checks pass329 tests and48 subtests, lint/format, editor diagnostics,
repository hygiene, source/wheel builds and package metadata. The source
distribution preserves the frozen case bank exactly and excludes raw runs,
caches and environments. Independent sidecar scoring reproduces all128
composition rows and30 aggregate metric cells without fitting or model calls.

## 19 November 2026 — Before-report replay and sender-policy intervention

Completed two real cached-8B diagnostics while preserving all 377 original
natural-pilot files. Prefix-only block-8 features were captured for all 84
saved sender prompts, with no generated response tokens supplied. New
neutral-only fits get 8/12 neutral decisions correct and 1/12 under pressure;
activation and combined models predict REJECT throughout the pressure set.
This reuses already observed outcomes and remains exploratory. Four test
cases, shared families and fixed model/position/regularization limit inference.

Independent verification reproduced all 120 predictions and metrics and all
recorded artifact hashes. Repeats and all 16 same-shape future-token mutations
are exact. Six of eight prefix/full comparisons fail the original BF16
neighbor bound; numerical status stays inconclusive without widening it.
Only prefix-only vectors enter training.

A separately recorded one-sentence sender-policy intervention generated 24
reports and 24 reader answers on eight training cases. Uppercase decision
words drop from 22 to 11 reports; reader correct/wrong/deferred counts change
from 8/10/6 to 15/8/1. Independent pairing review confirms only the declared
sender sentence changes, with exact reader policy, facts and effective
generation settings. This does not establish general improvement or intent.

A sender-only agent-assisted annotation pass covers all original 84 reports
and all new 24. Draft factual labels include 30 original contradictions and
six new contradictions; three paired previously supported reports become
contradicted. Every row remains flagged for human review. Byte-hashed quoted
evidence is preserved without automatic newline/whitespace rewriting.

Executed scripts are preserved at commit
`084f5ac2d37c07e20fda29881b120b363380197f`. Later safeguards reject nested
outputs before writes and reject changed source plans, reader policies,
job bindings and generation settings before reuse. The completed artifacts
already satisfy these conditions; they were not regenerated or edited.

See [measured diagnostics](./RESEARCH_CHECKS_RESULTS.md),
[machine record](./validation/2026-11-19-prereport-and-policy-checks.json),
[original draft audit](./REPORT_AUDIT_DRAFT.md),
[intervention draft audit](./SENDER_POLICY_AUDIT_DRAFT.md), and
[completion gates](./MAIN_STUDY_GATES.md). The main study, independent
evaluation and final dissertation remain unfinished. Three agents contributed;
no connected-app plugin was used.

Local verification after the guard changes passes 248 tests and 48 subtests,
Ruff lint/format, editor diagnostics, repository hygiene, source/wheel builds
and package metadata checks. The semantic annotations remain drafts regardless
of these software checks.

## 19 November 2026 — Restored slide objective and natural forecast pilot

Checked the original Progress 1 PDF and Progress 2 draft directly. They make
the neutral-trained reader forecast central and natural reports primary;
Apollo and fixed-text replay are comparators/diagnostics. Corrected the earlier
scope recommendation in the current direction and supporting design notes.

Implemented the prospective natural workflow with 24 fresh authored cases,
72 neutral reports and reader replays, neutral-only fitting, frozen artifacts,
then 12 favorable test reports. All three classes occur in training: 29 APPROVE,
10 REJECT and 8 DEFER; one INVALID is explicitly excluded. The saved weights
and preprocessing are frozen before favorable generation. An independent
agent verified all 120 frozen predictions, metric calculations and artifact
hashes; no pressure outcome enters fitting.

Activation accuracy is 10/12 in both test conditions across four case clusters.
Macro F1 falls .703 to .580; neutral loss is worse than full-text TF-IDF.
The three lexical models predict the majority class throughout. A separately
recorded direct LLM text forecast gets 8/12 then 9/12. An exploratory report-word
copy control gets 7/12 then 11/12, so added value over text is not established.

Full-evidence controls are 23/24. Selected reports contain factual/precedence
errors, recommendations and explicit decision words. These are legitimate
observed reader-outcome labels, not audited truthful-omission or deception
labels. Four reports exceed the word instruction; two end at the token cap.
No source reports were regenerated after inspection. See the full
[natural-pilot report](./NEUTRAL_FORECAST_RESULTS.md) and
[curated machine record](./validation/2026-11-19-neutral-reader-forecast.json).

The four-slide measured presentation preserves the original neutral-to-pressure
question and foregrounds the new natural pilot and text-shortcut result. The
baseline, natural reader readout, fixed-text diagnostics and optional causal
mechanistic extensions are now explicitly separated.

## 19 November 2026 — Independent audit and first recipient-context results

Three parallel agents reviewed numerical integrity, current probe literature,
and omission/recipient-context novelty. Read scopes and primary sources are in
the [deep audit](./LITERATURE_PROBE_AUDIT.md) and
[recipient design](./RECIPIENT_CONTEXT_DESIGN.md). The resulting
[scope decision](./RESEARCH_DIRECTION.md) keeps Variant 2 and postpones a new
belief decoder or correction-ranking system until the controls are valid.

Reproduced three integrity gaps with failing tests, then fixed them: incomplete
score sidecars could claim verified provenance; training/scoring did not verify
a present Apollo capture's completion and artifact hashes; and resumed runs
did not compare attention implementation. The historical completed SDPA run
is unaffected. Generic external numeric inputs remain explicitly unverified.

The [independent numeric report](./validation/2026-11-19-independent-baseline-audit.json)
verifies the original artifacts and fitting. Prompt-only text AUROC is 1.0;
response-only is 0.5. Twenty independently seeded pair-flip fits average 0.5183
AUROC (range 0.4661–0.6031). There are 178 shared validation causal histories
and 130 exact validation vector duplicates. Purging pairs with any shared
history leaves eight validation pairs with instruction AUROC 1.0. These
diagnostics establish instruction discrimination, not observed deception.

The genuine expanded GPU audit passed 1,840 recorded checks, many of which
are artifact/hash checks. All 536 stored records were audited; 22 selected
dialogues were actually replayed. Their saved/reloaded values, same-forward
layer comparison, repeated forwards and first/last future-token mutations
agree exactly. CUDA forward-phase Torch allocation peaked at 5.480 GiB,
including resident model allocations and excluding caller load peak and
non-Torch memory. See the
[reader and context report](./validation/2026-11-19-reader-and-context-pilot.json).

The same model load completed 48 reader calls and 48 off-policy fixed-report
captures. Informed readers were correct in all four original and restored
conditions. Absent-evidence readers deferred on all four originals; restoration
yielded three correct decisions and one deferral. Stale-evidence readers went
from two wrong/two deferred on original reports to three correct/one wrong
after restoration. The irrelevant addition preserved original decisions in
all four cases and three contexts. No original report induced a new wrong
decision relative to no report. Net-profit restoration failed in both absent
and stale contexts, so the original pilot stays frozen and the failure receives
a separately marked post-hoc development diagnostic.

For original fixed reports, the favorable reporting goal increased the probe
margin in all 12 case/context comparisons (mean +0.8121). Stale-minus-informed
score differences have mixed signs. These are context-sensitivity results,
not detection rates, calibrated probabilities, or evidence of intent. Four
authored cases cannot support population confidence intervals.

The offline suite passed 176 tests plus 48 subtests, with aggregate branch-aware
coverage 77%; lint, formatting and editor diagnostics passed. The source and
wheel build succeeded. Coverage excludes genuine GPU execution; it does not
prove every runtime path. The old starter validation record remains historical.

The nine-call post-hoc profit diagnostic completed separately. The failed
decisions reproduce at both 8- and 64-token budgets. Standalone subtraction and
a complete informed packet succeed. Removing the absent packet's explicit
"no cost figure" phrase resolves that restoration; adding the computed net
loss resolves both absent and stale cases. These support investigating evidence
integration and prompt interpretation, not a unique causal diagnosis. The
[pilot report](./PILOT_RESULTS.md) includes all outcomes, their limits and the
next data-validity gates. Original four-case content and outputs stay frozen.

## 19 November 2026 — Apollo foundation

Imported the supplied `kcl_latent_probes_starter.zip` into the existing empty,
private `ADI-0519/Individual-Project` Git repository. Original archive SHA-256:

`5CDFF7B5B6D45075852319D64EF043339CF54CE877B9004831FDFD722C7447FE`

The ZIP was inspected before extraction and paths checked against the repository
root. Existing files were not overwritten. Its original acknowledgements and
historical validation record remain intact. The imported source was generated
with ChatGPT assistance; the changes in this entry were developed with Codex
and three parallel agents. The student must understand, evaluate and describe
their independent contribution under the module's actual policy.

### Evidence that changed the implementation

- The original last-token classifier is not the saved Apollo instruction-pair
  baseline. Added a separate token-matrix capture/training/scoring path.
- Full Apollo and MASK papers, including appendices, were read for the baseline
  audit; relevant Bürger and Natarajan methods were checked. Cross-checked the
  saved Apollo configuration and executable source at commit
  `f8ec4010e74927394709dffa22b97bdf8cd5a62f`.
- Pinned source uses five **words**, despite paper prose saying tokens; uses HF
  `hidden_states[22]`, not decoder block 22; and fits `C=0.1` without a separate
  classifier intercept. Evaluation averages raw token margins.
- Actual pinned data: 612 rows, 306 true facts; 31 zero-prefix facts and eight
  later duplicate occurrences overlap once. Explicit exclusions after the
  original split yield 219 train / 49 validation pairs. The earlier hypothetical
  410/102 calculation assumed 512 available facts and does not describe this file.
- Raw data SHA-256:
  `924709159b40a30d5b51488e9ccd7fd886456a89f0e0f5849eacefba6ff55312`.
- Reproduced an existing label-semantics bug: heterogeneous targets were accepted
  as one binary label. Added failing regression tests before fixing validation
  and preserving manual target definitions through annotation export/packing.

See [method audit](./APOLLO_METHOD.md), [reading log](./READING_LOG.md) and
[starter audit](./STARTER_AUDIT.md) for the detailed distinctions.

### Reproducibility and software checks

Added a Python 3.11 development environment, `uv.lock`, explicit CPU/CUDA profiles,
Ruff formatting/lint, offline tests, locked packaging, pinned GitHub Actions,
artifact retention, contribution/citation/security metadata and credential/data
ignore rules. No project or upstream-data licence was invented. No model weights,
tokens, private material or generated runs are source-controlled.

Local offline suite initially passed 113 tests and 48 subtests (72% aggregate
branch-aware coverage), then reached 116 tests after further data-quality and
source-provenance regression checks. All four jobs in the first
[GitHub CI run](https://github.com/ADI-0519/Individual-Project/actions/runs/44525529169)
passed, including Linux Python 3.11/3.12 and Windows Python 3.11. The final CI run
is the authoritative check for the committed version. Tests use marked fixtures;
their passing counts are not model research results.

### Real-model execution status

Local NVIDIA inspection found an RTX 5080. The initially installed PyTorch wheel
was CPU-only. A real small-model capture attempt failed before model loading:
Windows Application Control blocked the `regex` extension DLL imported by
Transformers. The same import was blocked outside the sandbox. No protection was
disabled and no fake response or activation was substituted. The failed local
manifest remains an ignored engineering record under `runs/`.

The [first manual Linux CPU smoke](https://github.com/ADI-0519/Individual-Project/actions/runs/44525570337)
passed on commit `d57dbbe9a02aeb44d7178006b4eb4c15c86b52b8`. It loaded public
Qwen 0.5B revision `7ae557604adf67be50417f59c2c2f167def9a775`, captured 14 paired
examples, fitted the readout and generated/scored six neutral controls. The
real decoder hook matched HF hidden-state index 7. Prefix-only versus full-context
activation maximum absolute error was `1.1324882507324219e-06`, below the declared
`1e-4` absolute/relative tolerance. The small
[original validation summary](./validation/2026-11-19-qwen-cpu-smoke.json) is
preserved here; full trace artifacts were downloaded to ignored local runs.
This is engineering validation, not an 8B detection result or a reliable 1% FPR.

Final independent review then found that arbitrary custom CSVs could inherit a
pinned source-revision claim. A regression reproduced this; preparation now
verifies source bytes, marks custom data explicitly unverified, and propagates
the preparation manifest into captures. Missing calibration sidecars now carry
an explicit unverified-provenance warning. The normal first smoke used the
verified download; its measured activation result is unaffected by these guards.

### Scope that remains

Collect independent control/evaluation responses, audit behavioral labels,
run the pressure-specificity pilot and establish a BF16 reference in a verified
CREATE environment. Freeze the held-out behavioral protocol. The reader
forecast and omission extensions remain research tasks; the 24 draft starter
scenarios must not be promoted to validated deception examples.

### Local WSL GPU validation

The separate Ubuntu environment completed installation with Python 3.11.17,
PyTorch 2.14.1+cu130 and CUDA 13.0. The actual CUDA kernel check passed on the
RTX 5080 (compute capability 12.0). The first Qwen forward pass then exposed a
missing C compiler required by Triton. Installed Ubuntu `build-essential` and
`python3-dev` through the package manager; no application protection was changed.
The failed capture manifest remains in ignored local runs. A fresh run used
the same pinned Qwen checkpoint and the new float32 CUDA config.

The complete local engineering pipeline passed: 14 instruction examples yielded
84 token vectors (78 training / 6 validation), the readout fitted, and six
neutral prompts generated 102 scored token vectors. The probe and controls
shared the recorded feature signature. The validator reloaded the actual GPU
checkpoint and checked one saved dialogue: all selected vectors exactly matched
HF hidden-state index 7, and prefix-only replay of its first and last selected
tokens also had maximum absolute error 0.0. The prespecified absolute/relative
tolerance was 1e-4. These checks cover one dialogue and its selected positions;
broader prompt coverage remains unevaluated.

The [GPU validation record](./validation/2026-11-19-wsl-gpu-smoke.json) preserves
the runtime, checkpoint, layer, dtype and numerical checks. The quickstart now
keeps the WSL CUDA environment separate from Windows development and CPU
profiles. The validator retains its original CPU defaults and accepts explicit
paths for this GPU run. Two agents contributed to the validation implementation
and the remaining 8B readiness review; no connected-app plugin was needed.

These are Qwen 0.5B setup results. The two held-out instruction examples are
not an estimate of deception performance. The six-control 99th percentile is
an explicitly unreliable engineering calibration, not a reliable 1% FPR claim.
A genuine 8B quantized load, its memory profile and independently audited
behavioral evaluation remain to be run.

### Llama access and the first 8B run

The user's Hugging Face login succeeded in Ubuntu. The optional Git credential
helper warning did not prevent authentication; downloading the exact Instruct
repository's `config.json` succeeded through the saved token. No global Git
credential configuration was changed, and no token was read into the chat or
committed. The base and Instruct repositories are distinguished in the quickstart.

Added an explicit small-run config pinned to resolved model revision
`0e9e39f249a16976918f6564b8830bc894c89659`, with NF4 loading, BF16 compute and
decoder block 8. The genuine 14-example extraction was launched with the same
verified seven fact pairs used for the engineering check. The original weight
download is approximately 16 GB even though GPU loading is quantized. **At this
entry's commit, the weight download is still running; no successful 8B load,
activation result or probe fit is claimed.** The continuation process waits for
the existing capture to complete, then fits the instruction probe and runs the
dedicated validator. Its ignored pipeline status and validation outputs record
success or failure independently of this journal entry.

The new Llama validator was reviewed by a second agent; imports/CLI help, Ruff
and editor diagnostics pass. Genuine-model checks remain pending. Its numerical
criteria were fixed before observing any 8B activation result: exact equality
between a block-8 hook and HF hidden-state index 9 in the same forward; exact
unchanged repeats; and exact invariance at first/last selected positions when
only later token IDs change in separate, same-length replays. Reloaded versus
saved BF16 values may differ by at most one adjacent representable BF16 value
per coordinate, with errors and out-of-bound counts reported. No float32
tolerance is reused or widened after seeing the results. The validator also
requires actual bitsandbytes 4-bit modules, finite 4,096-dimensional vectors,
matching masks/IDs, and one generated/scored neutral answer. That answer will
exercise scoring; it cannot establish a detection metric or calibration.

### Verified 8B execution and full instruction baseline

The first transfer was interrupted; a process check showed that neither the
capture nor its continuation was still running. Resumed the identical capture
with unchanged package/code fingerprints and retained provenance. The completed
weight shard was reused; unfinished Xet shards restarted their transfers. All
four original shards subsequently finished. The results below are from the
completed executions, not an inference from a waiting status file.

The [8B validation record](./validation/2026-11-19-llama8b-4bit-smoke.json)
reports **pass** on the RTX 5080: 224 actual bitsandbytes Linear4bit modules,
BF16 native residual outputs, 14 captured dialogues and 84 token vectors with
4,096 dimensions. In the checked saved dialogue, all 28,672 compared values
matched the actual HF hidden-state index 9 and the fresh reload exactly. Two
unchanged same-shape forwards matched exactly. Separate mutations of future
tokens preserved the first and last selected vectors exactly. All observed
errors were zero; the predeclared BF16-neighbor rule was not widened. These
index/replay/causality checks cover that dialogue, not all possible prompts.

The same model generated `17 + 25 = 42.` and its saved response activations
were scored successfully. The validator's maximum PyTorch allocation was
6.701 GiB, including its recorded initialization peak. Runtime imports and
cached model loading took 201.4 seconds in this WSL/OneDrive setup. This is a
local engineering profile, not a CREATE timing prediction. The pinned tokenizer
was also inspected: its default date is the constant `26 Jul 2024`, with no
wall-clock `strftime_now` call; no date-pinning code change was necessary.

The [full instruction baseline record](./validation/2026-11-19-llama8b-full-instruction-probe.json)
then completed on the same checkpoint, NF4/BF16 profile and block-8 signature:

- 268 eligible unique facts, with 219 training / 49 validation pairs.
- 536 dialogues and 3,200 token rows: 2,536 training / 664 validation rows.
- Training-only standardization, `C=0.1`, no fitted intercept and mean raw margins.
- AUROC **1.00 on 98 held-out instruction examples**, comprising 49 paired facts.
- Capture wall time 279.9 seconds including model loading; classifier fit 0.81 seconds.
- Maximum recorded per-example PyTorch allocation 5.466 GiB after peak resets;
  this excludes initialization and other applications' allocations.

The AUROC measures separation of instruction conditions. Both continuations in
each pair are fixed source truths, so it is not a measured deception-detection
accuracy. No operating threshold or calibration is inferred from this result.
The full probe also scored the saved neutral arithmetic answer through the
matching feature-signature check; one answer remains a compatibility smoke.
Both main 8B profiles are now pinned to the verified model revision.

The [pressure pilot](./PRESSURE_PILOT.md) remains planned and unexecuted. The
source audit identified original MASK belief prompts and a separate CC-BY-4.0
data license, avoiding unnecessary rewritten belief questions and a misleading
dataset-card loading example. Case selection and blind behavioral audit still
precede scoring. The new Apollo CREATE template calls the correct extraction
command, checks real CUDA execution and passes local Bash syntax/configuration
guards; it has not been executed on CREATE. CI now checks Bash syntax, and
source distributions include the curated JSON validation evidence while raw
runs, weights and credentials remain excluded.

### Tool and plugin use

Used the graphify skill's inspection guidance and the local Python
fact-grounded-coding skill. No existing graph was present, so focused source
inspection was used. No connected-app plugin, Obsidian integration, browser
automation or Playwright was needed. Python, Git, GitHub CLI and Actions are
development tools in this record, not additional app plugins.
