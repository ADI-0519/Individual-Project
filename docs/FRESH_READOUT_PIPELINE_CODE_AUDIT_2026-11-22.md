# Fresh readout pipeline: final independent pre-execution code review

22 November 2026. This follows the [design audit](FRESH_READOUT_DESIGN_AUDIT_V2_2026-11-22.md)
and the [initial capture findings](FRESH_READOUT_CAPTURE_CODE_AUDIT_2026-11-22.md).
The initial capture note intentionally recorded repairs as pending; the present
review verifies their implementation and adds fitting/analysis review. No fresh
study model outcomes or authored test prose/gold packet were inspected. No
native-data fitting, model call or package installation was performed by this
reviewer.

**No unresolved conceptual or code blocker was found in the reviewed selection,
capture, fitting and analysis paths after the repairs below.** This is a code
and synthetic-arithmetic review, not certification of future native artifacts,
optimizer trajectories, human semantics or model capability. Native preparation,
capture and bundle validation must still enforce the reviewed contracts.

## Independently checked evidence

- The current fitting and capture suites passed **56 tests** in this reviewer's
  existing Windows Python environment.
- The current analysis suite passed **11 tests**, including aggregate feature
  and output-margin corruption checks added following this review.
- An independently constructed, in-memory 4,992-row Boolean fixture used 24
  generic contexts, both renderers and every declared symbolic cell. It supplied
  no experimental features or narratives. All 17 scored methods were retained;
  oracle judgments gave 192 hard quartets per renderer, primary difference zero
  and interval `[0,0]` when S0 and S1 were identical. World/name pair counts were
  384 per 768-row stratum and 96 on assertions. Hard receipt controls had 192
  changing and 192 preserving pairs for each receipt; easy B1 had 384 preserving
  pairs and B2 had 384 changing pairs. Four-factor accuracy was evaluated only
  on genuine two-message rows.
- This reviewer inspected the summary and source bindings of the owner's
  [14,080-prefix tokenizer audit](validation/2026-11-22-fresh-readout-token-spans.json),
  without reading its narrative expansions. It records 7,040 prefixes/model,
  exact full-prefix token-ID equality and checked spans, with zero model
  forwards. All four bound source hashes matched when inspected. A subsequent
  policy-list change added analysis bindings, so that receipt remains historical
  evidence for its dated source snapshot. The execution author reports retaining
  the original bound inputs in the native audit archive. Final official
  preparation must recheck all prefixes under the final code seal.
- The capture owner separately reports actual tiny randomly initialized Llama
  and Qwen HF checks matching all 12 hook/HF positions. This is owner-generated
  engineering evidence, distinct from this reviewer's NumPy-fixture tests and
  from any scientific checkpoint evaluation.

## Capture findings resolved

The real adapter now uses the bank's actual `load_bank` API. Generic supplied
rows are fixture-only; native preparation derives the validated bank inventory.
Bindings include the new renderer, original bank/allocation dependencies and
nested authored/source inputs. Full rendered-chat tokenization is checked
against direct chat-template IDs. Strict right-span boundaries prevent later
question characters entering the account site; the query's optional left
overlap is explicitly restricted to whitespace. The account convention includes
its pre-question paragraph separator.

The output-wide exclusive lock covers all partitions and aliases. Unknown stale
locks and incomplete row attempts block automatic retries. A 2,048-row training
alias preserves all four row partitions, avoiding repeated backend loads while
keeping source and correction supports distinct. Protocol/source/authorization
bytes are rechecked before engineering, before cohort forwards and after export.
Native backend checks include actual configuration dimensions and identity.

Native semantic authorization validates the genuine agent review and regenerates
its exact packet/key bindings, rather than accepting arbitrary file hashes.
Native test authorization validates both genuine model fit bundles and all eight
fitted files, their exact source/development IDs, model/protocol/site identity
and safe complete head inventories before any test forward. There is no human
approval gate. Cohort raw and completed archives retain only two candidate
logits; complete vocabulary arrays are confined to the excluded engineering
example.

## Fitting and source separation

The [fit module](../src/latent_probes/fresh_readout_fit.py) rejects test/unknown
partitions before feature-array I/O. It verifies the 2,048-row training capture
against the bank, prepared plan, exact ordered IDs, hashes and individual row
journals. Source fit/development and correction fit/development retain exact
512-row supports, renderer A and disjoint groups. Test plan/bank metadata may be
read for identity validation; no test feature or test label enters preprocessing,
candidate estimation or selection.

Each source site's scaler and coefficients use source fitting rows only.
Source-development equal-group log loss selects its C and then S1. Exact ties
use smaller C, original site, block and position order. S0 independently uses
the same original-site policy. Correction development cannot affect source
heads/site selection. The code saves the selected fitted candidate itself,
without refitting or absorbing development rows.

C1 uses correction-only scaling/fitting and its own development log-loss rule.
C2 shares that training-only transform, uses the declared factor grid and one
shared C selected by four-factor development log loss. All four predictions,
including learned B2, enter the supplied fold on every test condition. Its four
labels and external computation are explicitly privileged. Factor truth metrics
are restricted to two-message rows; assertions and missing one-step B2 do not
receive invented truth labels.

Safe serialization preserves all 12 source heads, C1 and four factor heads,
candidate histories, strict positive-margin orientation and transformations.
Validation checks the 17-head inventory, selected settings and exact arrays.
The both-model seal binds eight actual fitted files and both prepared protocols;
it does not permit a placeholder JSON/NPZ pair. Failed candidates are recorded,
with no grid changes or convergence rescue.

## Analysis and interpretation

The [analysis module](../src/latent_probes/fresh_readout_analysis.py) is bound,
together with its CLI, before test capture. It validates complete held-out IDs,
prose/state bindings, hashes, per-row records, aggregate-to-journal feature
equality and output-margin equality. It checks that test-row starts follow the
both-model fit seal. It performs no fitting, calibration, site selection or
model inference.

The single scientific primary is Llama S1 minus S0 on renderer-A hard complete
quartets: eight quartets/context, 192 quartets in 24 paired contexts. The fixed
10,000-draw seed-20261122 percentile bootstrap resamples context differences.
It conditions on frozen authoring, renderers, site selection and fitted heads;
models are not pooled as independent units. Qwen is explicitly a predefined
replication. All four conditions, both renderers, all named pipelines and every
frozen source-panel head remain in the results without behavioral filtering or
test winner selection.

Output probabilities are the raw two-logit conditional YES/NO probabilities;
they are not probabilities of emitting a valid answer or guaranteed calibrated
semantic confidence. C2's stored synthetic hard score is labeled accordingly.
The final-prefix output comparator has later textual context than earlier
account/query captures, as already specified in the design audit. Selected
source sites are now exported as actual site names. Easy and hard label patterns
and their receipt controls retain separate interpretations; there is no shared
independent-cell chance floor.

A one-step-selected site may still be poor for correction. Null S1-minus-S0
does not establish absence at every site; C1/C2 success concerns their distinct
supervised pipelines, not causal model computation. Passing controls do not
manufacture human semantic validation, and failed controls remain results that
limit interpretation. These limits apply regardless of later measured outcomes.
