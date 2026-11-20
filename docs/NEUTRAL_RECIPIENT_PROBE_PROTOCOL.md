# Neutral recipient-sensitive outcome probe, then natural pressure transfer

20 November 2026. **Prospective, gated development protocol. No model outcomes
are specified here.** This returns to the supervisor's communication task:
train a lightweight monitor during neutral communication, freeze it, then test
its transfer to natural reports produced under reporting pressure.

The earlier frozen readout largely reproduced its original reader's APPROVE
bias. High accuracy therefore did not establish recipient-sensitive monitoring.
The original Llama controls and the first independent Qwen gate failed. Preserve
both failures. A separately fixed nine-shot Qwen reader policy receives **one
fresh capability validation**; this protocol executes only if that policy passes
its frozen gate. Neither changing more checkpoints nor repeatedly revising the
policy until it passes is authorized by this protocol. If the gate fails, stop
this study and take the valid-input/reader problem to the adviser.

## Target, information and representation

The target is the **actual recipient's APPROVE/REJECT/DEFER answer** under a
specified reader policy, packet and report. It is not the complete-source answer,
report factuality, deceptive intent, or proof that the actor anticipates the
reader's response. The reader sees its rule, packet and report; it never receives
the actor's private complete source or reporting objective. Its fixed few-shot
policy must use disjoint demonstration cases, with the fresh validation and final
natural-transfer cases excluded from demonstrations.

The actor remains pinned Llama-3.1-8B-Instruct, NF4 weights/BF16 compute. Use the
unchanged original neutral reporting prompt, then teacher-force a known report
with `forced_token_activations(..., trim_last_words=0)`. The scoring coordinate is
decoder block 8, output before final normalization, using the last selected
non-special report-content token. Independently verify selected token IDs,
character boundaries and content-token status before treating this as the final
report token; do not accidentally capture a chat terminator. Use a newly named
feature signature with `pooling=last_non_special_report_token` and
`prediction_time=after_report`.

These are **off-policy fixed continuations**. The actor did not choose their
content. They are physically comparable report-state measurements, not natural
neutral reports, evidence of sender intention, or a pre-report belief mechanism.
The natural pressure-transfer set remains a different generation distribution.

## Fixed neutral source bank and source-supported reports

The [new excerpt panel](../data/communication/neutral_excerpts_v1.json) binds the
unchanged [original development cases](../data/reader_forecast/development_v1.json):
24 cases in four rule families, with the original 16 training, four validation
and four diagnostic test cases. The original source-bank byte SHA256 is
`a2ac1ba2260c2d09555f740debb2b7c96c1e0bfaa6cd326b4a216b339a9a8bb4`.
The [panel module](../src/latent_probes/neutral_recipient_panel.py) also pins the
canonical complete source object, preventing changes to wording, facts, IDs or
splits while still allowing an auditable standalone panel.

Each source contains an older decisive assertion, a common assertion and a
current decisive assertion. Construct three reports by literal concatenation:

| Report kind | Exact source assertions |
| --- | --- |
| `full` | Common assertion, then current decisive assertion |
| `omission` | Common assertion only |
| `historical` | Common assertion, then older decisive assertion |

Complete source assertions are copied verbatim, joined by a single space.
No model authors or paraphrases these reports. Historical dated assertions remain
supported historical facts; their omission of a later update does not itself
establish dishonesty. Each report is crossed with informed, absent and stale
recipient packets, giving **216 fixed communication states**. The actor's source
is complete in every state; the recipient's packet is explicitly attributed to
it. Record report kind and source-assertion indices as metadata, not extra actor
or reader prompt labels.

The expected partition sizes are 144 training states, 36 validation states and
36 source-test states. These are 16/4/4 **case clusters**, not 144/36/36 independent
scenarios. All report/context variants of a case stay together. The programmatic
normative outcome is a control only. Measure real reader answers; never replace
them with normative labels or invent missing classes to fit a classifier.

Under an ideal rule-following reader, absent packets plus common-only reports
support DEFER. Full reports or informed packets supply current decisive evidence.
Older reports without newer evidence support the older available decision under
the declared precedence policy. The intended crossing makes packet-sensitive
answers possible even with exactly the same source-supported report. Whether the
validated reader actually follows those distinctions must be recorded.

## Reader validation gate and execution boundary

Freeze the nine-shot Qwen policy, checkpoint/precision, demonstrations, parser,
eight-token direct-answer budget, 24 fresh validation cases and gate before
generating its validation answers. The fresh gate requires at least **23/24
informed, 20/24 absent and 20/24 stale** normative controls correct, with at most
one INVALID answer across all 72 controls. Keep per-family results; aggregate
thresholds do not certify each rule family. A failure ends this study without
collecting its labels, fitting probes or evaluating natural reports.

Gate success licenses measurement and analysis, not a claim that every label is
semantically valid. Preserve malformed answers and every reader-policy failure.
Check source-supported reports, exact evidence visibility, numerical reasoning,
date precedence and case-disjoint demonstrations. Agent checks remain distinct
from independent human review.

## Frozen neutral fitting and validation-only selection

Use only valid actual recipient answers on the original 16 training cases to fit
the new readout. INVALID answers remain visible in the full ledger and are
excluded under a declared policy. Require all three answer classes for a valid
three-class fit; otherwise mark it untrainable rather than silently changing the
target or manufacturing DEFER examples.

Fit at least these two independent controls:

1. A linear activation classifier with training-only standardization.
2. A full-context TF-IDF linear classifier that receives the actor's complete
   source, rule, attributed packet and the exact report, with training-only
   vocabulary/IDF.

Independently choose each model's regularization from the fixed grid
**C ∈ {0.01, 0.1, 1, 10, 100}**, using multiclass log loss on the original four
validation cases only. Freeze a deterministic tie rule, solver, maximum
iterations, scaling and class order before fitting. Record every validation
candidate rather than selecting a favorable one after testing. Fit scaler,
vocabulary and IDF only on training cases; no source-test or fresh-transfer row
may affect preprocessing, regularization, thresholds, layer choice or fitting.
Select first by validation loss, then the smaller C on an exact tie.

Use the original four test cases only as an explicitly labeled source-domain
diagnostic after configuration is frozen. Their outcomes are already part of an
observed development case bank; they cannot be advertised as new final research
material. No fitting changes follow a disappointing source-test result.

## Natural neutral-to-pressure transfer

The final transfer inputs are a separately authored **24-case fresh bank** with
new identities, facts and numerical instantiations, not these 24 source cases.
Four task types reuse the source rule grammars; two introduce held-out explicit
task types. Date-threshold reasoning still shares components with numerical
thresholds, so this does not establish broad unseen reasoning generalization.
Keep the fresh capability validation cases distinct from the final transfer
cases; a bank used to repair the reader is not untouched final evaluation.

Generate one natural report per fresh case × three attributed packets × neutral
or favorable-APPROVE goal: **144 reports**. Preserve the shared factuality,
evidence-only and report-budget instructions, every violation, every completion
and all failures. This is simultaneously source-excerpt→natural-report transfer,
fresh-case transfer and a paired goal shift; report those dimensions separately.
Freeze all fitted artifacts before any fresh natural-report outcomes are opened.
Run the same validated recipient policy under the attributed packet and report.

Primary comparison is the neutral-to-pressure change in the frozen activation
readout, with paired case-level log loss, accuracy, macro-F1, per-class recall,
class counts and invalids. Compare against the independently selected full-context
text classifier, frozen neutral-training majority, source-world rule and the
actual recipient's no-report decisions. A capable full-context text forecaster
must have matching information and a frozen prompt/budget; report cost and
parsing failures. Beating a weak or invalid monitor does not establish latent
superiority. The known deterministic recipient can always be run directly as an
oracle-cost ceiling; doing so is not independent forecast evidence.

The reader policy is now intentionally part of the forecast target. The original
source readout and its previously frozen forecasts must not be represented as
having anticipated this later policy. If compared, label those separately as
cross-policy transfer. The new readout is trained against this policy's neutral
communication answers and is the properly matched neutral→pressure test.

## Cost, stopping rule and interpretation

Record the 216 actor forward passes, actual unique recipient calls, cache reuse,
token counts, extraction/generation time, model loading and fitted-model scoring
cost separately. Teacher-forced activation extraction is additional model
computation; it is not inference-free. A same-text cache hit adds no independent
sample size. Do not claim a cost advantage without a common measured budget.

This is one predeclared development study after exposed reader/majority
shortcuts. Preserve earlier failed controls and clearly mark the reader repair
and new neutral-training distribution. Do not extend the C grid, search layers,
replace cases, add more reader policies or stop only when a positive result
appears. A valid negative result is informative; failure of the capability gate
leaves this protocol ready for human/adviser review rather than another repair.

A useful result would show reliable recipient-sensitive neutral learning and a
well-controlled transfer pattern under pressure, including a strong text
comparison. Neither this design nor decodability alone proves novel research,
causal second-order beliefs, intentional deception, publication readiness or a
particular academic mark.
