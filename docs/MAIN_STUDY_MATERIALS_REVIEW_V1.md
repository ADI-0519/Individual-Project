# Main-study materials and fitting review

21 November 2026. Independent agent review of the training/development authoring
and implementation; **not human semantic review or model evidence**.

## Scope and conclusion

This reviewer read all 64 training/development narrative introductions and event
descriptions, all specified physical names, the shared renderer, the material
assembler, the matched allocation rules, the fitting module and the held-out
analysis. One exact expanded prompt was read for each of observation, message
and correction, including a correction whose first report was accepted as read
and whose replacement was accepted as unread. The correction illustration came
from the test fragment authored by this reviewer, so it is a renderer check,
**not independent review of that test narrative**. Another reviewer must inspect
the 90 authored test groups.

No fatal target leakage, allocation mismatch or unsupported access assertion was
found in the reviewed training/development material. One entity-part overlap
issue was reported to the parent before material freezing, with proposed fixes
below. This conclusion is conditional on the reviewed snapshots and on the
explicitly stated fictional update rules.

## Narrative and state checks

- All 48 fitting groups and 16 development groups contain one-step observation
  or message mechanisms. None introduces a second location report or a correction
  operator into fitting or selection.
- Observation contexts provide a viewing opportunity and an equipment image;
  they do not assert that the named reader attended. The renderer separately
  specifies the sender's accepted access record. The image excludes participant
  information, and the true observation at 09:05 precedes the independent private
  world check at 09:15.
- Message contexts describe written claims and separate reading records. They
  do not certify those claims as world truth or certify receipt merely from
  transmission. The renderer explicitly permits inaccurate claims and unread
  messages with no notification.
- Roles such as keeping an equipment log or tracking the token specify a task,
  rather than granting an additional observation channel. The common account
  expressly denies additional location evidence and unobserved-move inference.
- Storage names are physical containers, including drawers, bags and cabinets.
  Contextual accessories explain token substitution; they supply no deterministic
  initial location, update content, accepted access or final world state.
- The narratives are distinct contextual framings of the same stated operators.
  Exact prose uniqueness does not establish independent reasoning families or
  generalization to unseen domains. The shared timeline and rule wording remain
  a substantial part of every prompt.

## Entity issue identified before outputs

The initial draft used nine test given names also appearing as components of
training/development full names: Inez, Cerys, Willem, Gethin, Farid, Umar, Giselle,
Pascal and Zosia. Full-string equality checking did not expose this overlap.
The parent was asked to rename those test actors and strengthen assembly checks
for actor-name tokens before freezing. Generic container words may naturally
recur; the intended container exclusion concerns complete physical place names.

Until the merged validator confirms the repair, claim only that full actor
strings differ. Even after repair, there is no claim of an entirely unseen word
vocabulary or unseen application domains. No model outputs informed this issue
or the proposed repairs.

## Fitting and analysis checks

The restricted and full arms use exactly matched sample counts and story groups.
The complete joint counts of group, update, access, private world and name mapping
are identical. Five prespecified allocations vary both selected story pairs and
coverage masks, while training sizes are nested within allocation. Finite
within-story correlations remain; matching does not remove every possible
shortcut.

The activation scaler is fitted only on selected fitting rows, then row-L2
normalization is applied. TF-IDF vocabulary and IDF are fitted only on the same
selected rows in their respective arm. Its text constructor uses message role
and content; symbolic states, labels, split identifiers and case identifiers do
not enter that input. Each arm has its own transformation, so the estimand is
the fitted pipeline's coverage effect rather than an isolated coefficient effect.

Both arms use the same regularization candidate, selected from the prespecified
grid by their mean group-weighted development log loss. A convergence failure
in either arm excludes that candidate for both. Calibration uses development
only and has nonnegative slope; it is separate from the primary raw threshold.
Because development contains full-support examples, the treatment is fitting
support, not the absence of preserving examples from the entire pipeline.

The fitting API rejects test rows and correction rows before fitting. Real
analysis requires all 90 held-out groups, with 30 per mechanism, and checks the
complete comparison inventory and capture/head provenance. It retains all
counterfactual cells. Primary scoring requires all four members of a quartet
correct, averages the five paired allocations within each correction group,
then averages the 30 groups. Bootstrap draws resample whole groups, not rows,
name swaps or training allocations. The interval remains conditional on these
authored contextual groups and this allocation design. Exact probability .5 or
margin zero predicts class zero.

The symbolic oracle uses the declared source state and is labelled accordingly.
It must not be described as a learned prose monitor. The separate prose monitor
runner accepts only original messages, applies no gold-based repair, and retains
invalid/capped generations. Its full-denominator results still need inclusion in
the final comparison/report; successful runner construction alone supplies no
baseline accuracy result.

Tokenizer-only preparation can expose test input prose and mathematical source
labels to the experimenter, as normal dataset preparation does, but performs no
model forwards and obtains no observed outcomes. Model payload construction uses
only prose. Capture and analysis require review/freeze provenance, and held-out
capture additionally binds the fitted artifacts selected before those outcomes.
No outcome-based filtering or behavior qualification is introduced by this review.

## Focused execution evidence

A CPU-only reviewer check independently verified all 15 allocation combinations:
unique counts, equal groups, balanced classes and identical complete nuisance
joint counts between arms. Mutating label, state, split and case-ID metadata while
holding messages fixed left the TF-IDF input unchanged. Boundary decisions were
`[0, 0, 1]` for probabilities `[.499, .5, .501]`. A deliberately constant-effect
30-group/five-allocation bootstrap fixture returned the expected difference .5
and interval `[.5, .5]`. This fixture checks arithmetic, not scientific uncertainty.

These checks made **zero language-model calls and fitted zero heads**. They do
not replace the parent's focused/full test runs, independent human review, or
the future held-out experiment.

## Reviewed snapshots

| Source | SHA256 |
| --- | --- |
| [Training/development fragment](../data/belief_main/authored_train_dev_v1.json) | `0adcb8fd20811015baba9db429861532324193a3c3eaebae07656158eb5ad221` |
| [Material assembler](../src/latent_probes/main_study_materials.py) | `96ffa02213a9516030dd154741387259fa74aca4ed81821dc4cb9d22fdb8a419` |
| [Fitting implementation](../src/latent_probes/main_study_fit.py) | `8c9f2422fb8834fdcfcb749f5a9a8487cd3d1fa5b4efab63f7b872fb94cb5fc2` |
| [Analysis implementation](../src/latent_probes/main_study_analysis.py) | `6c3c36fca4aa696e64c266e48eefe710f9100991e4c18391e92229b16955cebc` |

Subsequent source changes need their own verification. This record does not mark
the human review ledger accepted and does not certify publication novelty.

## Follow-up: repaired materials and prose-monitor implementation

The original snapshot hashes above are retained as the record of the initial
review. Before any main-study outputs, the parent corrected 13 test actor names:
nine cross-split actor-component overlaps and four names excluded by the design
examples. The strengthened assembler now checks actor-name components across
splits in addition to exact physical-place and excluded-entity checks. This
reviewer independently reran assembly from both source fragments and reproduced
the final bank and both input manifests. All 5,888 cases and 15 mathematical
allocations are present; source state/gold rules remain unchanged. These are
mechanical name repairs, not results-driven case replacement or human approval.

| Rechecked source/material | SHA256 |
| --- | --- |
| [Final assembled bank](../data/belief_main/main_v1/main_bank_v1.json), canonical JSON | `e038025571a38a7f2c489e5f8c3bf3b0a3c11a05f9b8c7e89d26292a4e7bfb1d` |
| Same final bank, stored bytes | `802f1519abe98172366e8381a1fa0dfe0772372991c8e8883c2570c431c5cac1` |
| [Strengthened assembler](../src/latent_probes/main_study_materials.py) | `45c81958df10d924fb8c5d86ef45c206a085bfe11c882643306e3f47bc619526` |
| [Prose-monitor runner](../src/latent_probes/main_study_monitor.py) | `b036ed42776cf83ce8cfec94e9b74cb2a5005b137316d378b094383999aa56fe` |
| [Fixed prose-monitor prompt/policy](../data/belief_main/text_monitor_protocol_v1.json) | `bcefd89009250413e7759991188e94f83169d7a7943345255c6a7b24c192d936` |

The monitor review read its full implementation, fixed prompt, generation
settings, and the actual local `HFBackend`/`GenerationResult` implementations.
The call to `generate(messages, settings, seed=..., capture=False)`, returned
token fields, finish reason, record method and cleanup match that backend API.
Source inspection and Python signature binding establish interface compatibility;
no real GPU generation was performed in this review.

Only original role/content prose reaches the model. The fixed prompt explains
the evidence-extraction task and excludes the sender's private check and final
candidate proposition from reader evidence. The parser accepts exactly one JSON
object, rejects unknown and duplicate keys at every object depth, requires actual
boolean access values and verbatim nonempty strings, and performs a chronological
fold over the extracted updates. Neither model-message construction nor parsing
accepts source state or gold as an argument. The symbolic fold is the declared
baseline method; evidence extraction remains the model's task.

CPU-only review constructed an expected extraction from the declared state for
each of the final bank's 5,888 prefixes, then checked parser/fold agreement and
exact prose-only payload round-trip. All passed. Additional direct checks rejected
nested duplicate location keys, integer access values, NaN access values and
concatenated JSON objects. These extractions were **oracle-constructed engineering
fixtures**, never model predictions, stored research outcomes or evidence of
monitor capability. No fitted head or language-model generation was produced.

Two operational/interpretive limitations remain:

- Development and held-out monitor execution need separate fresh preparation
  directories. A completed directory is immutable and cannot switch partitions.
- Verbatim-occurrence validation is syntactic: it can accept a repeated substring
  such as `token` as a location string. Schema-valid extraction therefore does
  not establish semantic extraction fidelity. Accuracy must be assessed against
  the frozen target on every retained case, with invalid outputs included as
  incorrect. The runner preserves both decoded content and special-token-inclusive
  output; its standard content decode skips special tokens before strict JSON
  parsing. This must not be described as token-level JSON grammar enforcement.

No blocking API, parser or gold-circularity issue was found at these snapshots.
Independent human semantic review and real model validation remain outstanding.

### Integration finding for the separate analysis under construction

While the parent was adding the separate monitor-analysis module, this reviewer
identified a schema mismatch: it expected a top-level `finish_reason` in each
record, whereas the runner stores that field inside each bound raw-generation
journal. This would raise `KeyError` during analysis and must be corrected before
the end-to-end path is considered validated. The finding was reported before
any main-study model calls. The recommended fix is to verify the saved raw file
against `raw_generation_sha256`, check its prompt IDs against preparation, and
reparse with its recorded generation finish reason so capped outputs stay
invalid. A passing generation-only fixture suite does not by itself establish
generation-to-analysis compatibility. This finding concerns the subsequently
added analysis, not the reviewed runner/API snapshots above.

### Integration resolution and independent replay check

The original finding above is retained. The repaired analysis now obtains the
finish reason from the raw-generation journal after checking its saved SHA256,
binds generated prompt IDs to preparation, and reparses with that finish reason.
The reviewer independently ran all five monitor-analysis regression tests: all
passed. The actual runner-shaped fixture includes all 512 development rows and
one parseable but capped generation; the replay retains it as invalid and gives
511 correct out of 512. This is synthetic software evidence, not model accuracy.

Invalid predictions remain null. Correctness explicitly requires validity;
invalid outputs count incorrect in row, pair and complete-quartet denominators.
Valid-only AUROC/Brier metrics retain their own smaller row count. The correction
fixture keeps all 64 cells, including all 32 secondary cells, and an all-invalid
fixture yields zero row/quartet accuracy rather than gold-dependent imputation.

Three additional temporary fixture mutations independently confirmed rejection
of a mismatched bank, inconsistent fixture identity, and a development record set
relabeled as test. Their manifest record hashes were recomputed deliberately, so
these checks reached the semantic provenance/coverage checks rather than merely
failing the first file-hash comparison. Source review of the combined analysis
CLI confirmed that it additionally requires the monitor's test partition, exact
held-out bank hash and matching fixture status after the ordinary capture/head
checks. No model inference, fitted head or human attestation was created here.

| Rechecked integration source | SHA256 |
| --- | --- |
| [Monitor analysis](../src/latent_probes/main_study_monitor_analysis.py) | `58877d25bc56468634005ab5da66eb0d27ffb9b95a759e359fc39358f6761a7d` |
| [Combined analysis CLI](../scripts/analyse_main_study.py) | `87d6f8b66cd5cdbd9339cb2499fe3316e53ffa0bb6c2ce6acb2de34bce9d4112` |
| [Regression tests](../tests/test_main_study_monitor_analysis.py) | `92eb262b00733cf83776e96a237d6a3611837e86ba26819d7bdce65cc4d46a52` |

No remaining integration blocker was found in this bounded review. This finding
does not replace the outstanding human semantic review or main-study execution.
