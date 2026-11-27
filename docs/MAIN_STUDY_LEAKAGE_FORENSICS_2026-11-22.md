# Independent leakage and provenance forensics

22 November 2026. This audit found **no direct train/test leakage in the retained
main-study artifacts under the checks below**. That is narrower than certifying
absence of every possible bias. The experiment remains an agent-audited,
exploratory, task-conditioned comparison on one authored bank with a shared
renderer. No human validation or broad reasoning-family independence follows.

## Audit order and independence

The auditor first read the state/allocation, bank, materials, capture, fitting,
execution and analysis implementations and their input manifests, without reading
held-out performance tables or previous result interpretations. It recorded the
source-level hypotheses before checking native artifacts: group leakage,
unintended wording/entity reuse, preprocessing contamination, full-support
development exposure, label-bearing input metadata, answer-token contamination,
fitted-head replacement and outcome-dependent case inclusion. Earlier statements
that an audit passed were not accepted as evidence.

The existing knowledge graph was queried first, as required by the repository
skill. It lacked the new main-study nodes and led to older probe modules. This
audit therefore used the actual sources and retained bytes. It did not rebuild
the graph or infer new main-study relationships from stale graph entries.

The new [standalone auditor](../scripts/audit_main_study_leakage_forensics.py)
imports no project scientific modules, executes no upstream or cached pickle
code, and reads numeric NPZ with `allow_pickle=False`. It reconstructs allocations,
targets, scaler statistics and document frequencies independently. It does not
fit classifiers, optimize calibrators, call language models, generate new model
observations or read held-out performance tables. Numerical development losses
are used only to validate the retained selection record.

The final machine record is
[validation v2](validation/2026-11-22-main-study-leakage-forensics-v2.json).
The initial [validation](validation/2026-11-22-main-study-leakage-forensics.json)
is preserved; v2 additionally checks the two supplementary archived prefix plans,
reconstructs selected-head development losses and enforces recorded chronology.

## Verified safeguards in the actual artifacts

| Check | Independently verified result |
| --- | --- |
| Group splits | 48 training, 16 development and 90 test groups; each group has one split |
| Exact row inventory | 1,536 training, 512 development and 3,840 test rows per model |
| Complete retained captures | All 11,776 rows across both models match frozen inclusion and order |
| Exact visible prompts | 5,888 distinct prompt hashes; retained messages match the split manifest and prepared protocols |
| Targets | Every retained gold value matches an independent state-rule calculation |
| Entities | No case-insensitive full actor/location entity is reused across splits; actor name components are also disjoint |
| Contextual wording | No exact normalized introductory/event framing is shared across splits |
| Training allocations | All 15 size/seed allocations independently reconstructed, including selected group order and both arms' exact row IDs |
| Fitting and calibration | All 60 comparisons across both models have the prescribed training IDs, development IDs and development-only calibration source IDs |
| Activation preprocessing | All 60 retained arm-specific scaler means and standard deviations equal selected-training-only NumPy reconstructions exactly |
| Text preprocessing | All 60 arm-specific vocabularies contain exactly selected-training terms; smoothed IDFs match independent document-frequency calculations |
| Shared regularization | Every retained selected C equals the prescribed minimum of recorded mean development losses, with lower-C tie breaking |
| Retained head/selection agreement | All 120 selected-head arm losses reproduce their stored development loss within absolute tolerance `1e-10` |
| Fitted artifacts | Both settings/head pairs match the four test-freeze artifact hashes; all retained NPZ array hashes match |
| Source preservation | 28 test-freeze source/input bindings and all 30 source-archive entries, including exact prefix plans, match current retained bytes |
| Answer-token exposure | Every row records zero supplied answer tokens and the last prepared prefix position; visible inputs contain system/user messages only |

The text baseline consumes only message roles and message content through
[row_text](../src/latent_probes/main_study_fit.py:199). It does not ingest state,
gold, split, group IDs, case IDs or output-logit columns. Full identifiers are
absent from every retained visible message. The forward call in
[forward_measurement](../src/latent_probes/main_study_run.py:338) receives token IDs
and an attention mask, rather than the encompassing label-bearing row record.
Keeping gold/state metadata beside an activation is therefore not itself target
leakage into these fitted features. This conclusion combines source inspection
with exact retained-prefix binding; the present auditor did not rerun tokenizers
or the model.

## Deliberate overlap and methodological limits

**The restricted arm receives preserving labels during model selection.** Both
arms use all 512 full-support development rows for shared-C selection and their
own monotone calibration. This is an explicitly disclosed feature of
[fit_comparison](../src/latent_probes/main_study_fit.py:255), rather than
contamination by test data. Describe the manipulation as restricted versus full
**fitting support**. A claim that one arm has never encountered preserving labels
anywhere in the pipeline would be false. Calibration uses the same development
set as C selection, so its development loss is an in-sample diagnostic and not an
unbiased estimate of calibration performance. Retained test scores provide the
held-out comparison. The primary raw-threshold endpoint remains separate.

**The split holds out contextual groups and entities, not the renderer.** The
state-transition prose, access-record structures and nested question are shared
by design in [render_messages](../src/latent_probes/main_study_bank.py:73).
Observation and message mechanisms occur in train/development/test; correction
sequences occur only in test. These are three mechanism families, not 154
independent reasoning families. Exact contextual repeats were absent, and the
largest cross-split Jaccard similarity between contextual sets of five-word
sequences was `0.07216495`. This string check does not establish semantic
independence. A separate retained-prompt word-count calculation found that the
authored introduction/event text supplies about 12.13% of user-prompt word tokens
on average, with a 9.05–16.87% range. Most text is the declared common renderer,
factor substitutions and question. Broad wording, genre or task generalization
has not been tested.

**The activation is conditioned on an explicit answer task.** The candidate
proposition and nested YES/NO question occur in the visible prefix before capture
at the final chat-template token. No answer has been supplied, but the activation
may encode response preparation, lexical associations or the evaluated
proposition. Good decoding would not establish a spontaneously represented
belief, an internal belief mechanism, a deceptive intention or a separately
measured reader model's mental state. The full-causal effect of changed training
support also includes different scalers/vocabularies, rather than isolating
coefficient changes or any causal intervention inside the language model.

**Factorial variants remain dependent.** Name swapping and full-factorial states
prevent simple global name/label confounding. They do not make each row a new
narrative. Finite story-conditional nuisance correlations in the balanced masks
remain possible. The five allocations are paired sensitivity draws, not five
independent experiments or additional reasoning families. Correction sequences
also differ in length and event count; they do not isolate composition from all
format effects.

## Retuning, inclusion and interruption provenance

Both models' train/development captures finished before the test freeze at
`2026-11-22T14:31:04.095530+00:00`. The first recorded Llama test attempt began at
`14:32:27.562516+00:00`; Qwen test began at `15:36:54.364518+00:00`. The unchanged
freeze binds both models' fitted bundles. All frozen source bytes still match
their archived counterparts. The capture inclusion sequence is the entire
requested partition, without a branch on labels, output logits or probe scores.
Every retained row is present; no outcome-conditioned removal was found.

Llama recovery retained 2,758 completed rows and finished the remaining 1,082.
The original unknown attempt is
`test_message_27__i0_u0_b1_w1_map0`; its original directory still contains only
the hash-bound `started.json`. There is no recorded raw outcome available there
to select between repeated outcomes. The recovery receipt specifies the repeated
attempt and reuses the same fitted heads and test freeze. It does not count the
repeat as a new independent case. The separate interruption/recovery audit remains
the place for exhaustive per-journal copy-byte validation; this audit rechecked
the unknown directory and final complete-cache bindings.

These records support **no detected output-conditioned retuning in the retained
main campaign**. They cannot authenticate every unrecorded action, establish
externally timestamped preregistration, rule out undocumented deleted work or
reconstruct foundation-model pretraining contamination. Local timestamps and
hash manifests are useful provenance, not an external registration service.
Prior exploratory studies informed the new design, and the agent execution
amendment expressly retains exploratory status. Later human agreement cannot
make these exposed outcomes prospective confirmation.

## Defects, uncertainty and next-study implications

No definite implementation defect causing label, preprocessing or held-out
membership leakage was found in the audited artifacts. The strongest remaining
constraints are common wording/renderer dependence, explicit query conditioning,
full-support development exposure, absent independent human semantic validation,
and limited family/model coverage. None is repaired by rerunning these exposed
cases until a preferred result appears.

A useful next study should first preserve this campaign unchanged, then freeze a
fresh cohort and a specific hypothesis before new observations. Independently
authored rendering structures and a question-free or separately query-controlled
capture can test wording and response-preparation explanations. A development
design that holds out preserving-label exposure would answer a different support
question and needs its own prospective protocol. These are new estimands, not
post-hoc fixes to the current result.

The [focused corruption tests](../tests/test_main_study_leakage_forensics.py)
passed **11 tests**. They reject test/development IDs inserted into fitting,
test IDs inserted into selection or calibration, duplicated fitting IDs,
development-contaminated scalers, vocabulary/IDF contamination, changed source
bytes and duplicated contextual framing. They also independently check the
replacement-update target's invariance to world state and name mapping. All
frozen scientific modules, cohorts, fits and old results were left untouched.
