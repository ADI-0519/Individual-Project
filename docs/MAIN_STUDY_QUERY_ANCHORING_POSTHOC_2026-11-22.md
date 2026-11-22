# Same-response query anchoring: posthoc control, 22 November 2026

This is a **new posthoc diagnostic on the same saved monitor responses**. It
is not a new held-out result, a new model run, a repaired primary analysis, or
human semantic review. Original campaign results and the completed
[measurement audit](MAIN_STUDY_MEASUREMENT_FORENSICS_2026-11-22.md) remain unchanged.

The [new helper](../scripts/audit_main_study_query_anchoring.py) and
[11 passing tests](../tests/test_main_study_query_anchoring.py) produce a separate
[validation artifact](validation/2026-11-22-main-study-query-anchoring-posthoc.json).
The JSON retains every case's original prediction, new prediction, frozen
validity, source question name and extracted final/query names, plus full group
summaries and exact source/code hashes.

## Exact intervention

For each valid saved response, retain the model's extracted initial location,
ordered updates, access booleans and existing `final_location`. Change only the
last comparison: compare that existing final location with the physical name
in the actual source question, instead of the model-supplied `queried_location`.
For invalid responses, retain null predictions and count them incorrect.

The question parser receives **only the original role/content message list**.
It searches all full message contexts for exactly one explicit terminal nested
question of the original template. It takes the literal location string in that
question. It never accepts a state, name-map, bank location list or gold label;
there is no anaphora resolution, fuzzy matching or gold fallback. Missing or
ambiguous questions cause an error. It parses all **5,888 original bank contexts**.
Evaluation labels and states enter only subsequent scoring and primary/secondary
grouping. ID sets, source-message equality and consistency of the frozen
extraction/final location/prediction are checked before scoring.

The same **44/3,840 invalid responses** remain invalid: two observation, 30
message and 12 correction responses. No response is regenerated or reparsed
under a more permissive validity policy. This remains the one Qwen hybrid
monitor run, not an independent Llama monitor replication.

## Counts before interpretation

Each cell gives original → query-anchored correctness. All denominators include
invalid responses as incorrect. Observation and message rows are all primary;
the secondary subset exists only for corrections.

| Mechanism | All rows | Primary rows | Secondary rows | Complete primary quartets |
| --- | --- | --- | --- | --- |
| observation | 797 → 803 / 960 | 797 → 803 / 960 | unavailable | 99 → 101 / 240 |
| message | 769 → 781 / 960 | 769 → 781 / 960 | unavailable | 79 → 84 / 240 |
| correction | 1396 → 1545 / 1920 | 710 → 781 / 960 | 686 → 764 / 960 | 81 → 90 / 240 |
| overall | 2962 → 3129 / 3840 | 2276 → 2365 / 2880 | 686 → 764 / 960 | 259 → 275 / 720 |

Correction all-row accuracy changes **72.71% → 80.47%**; primary-row accuracy
changes **73.96% → 81.35%**, secondary-row accuracy **71.46% → 79.58%**, and
complete-quartet accuracy **33.75% → 37.50%**. Overall row accuracy changes
77.14% → 81.48%; overall quartet accuracy changes 35.97% → 38.19%.

The query audit also exposes component-error cancellation:

| Mechanism | Valid responses with wrong extracted query | Wrong → correct after anchoring | Correct → wrong after anchoring | Wrong query, verdict unchanged |
| --- | ---: | ---: | ---: | ---: |
| observation | 10 | 8 | 2 | 0 |
| message | 23 | 16 | 4 | 3 |
| correction | 245 | 197 | 48 | 0 |
| overall | 278 | 221 | 54 | 3 |

In the **48 correction cases** that become wrong, the original wrong query
helped an erroneous extracted final location produce the correct binary answer.
Replacing only the query reveals that upstream error. Thus a correct original
answer is not evidence that every extracted component was correct. Conversely,
197 correction answers improve when only the question binding is supplied
correctly. All cases with an already-correct extracted query keep their verdict.

Correction full-denominator class-0 recall changes **63.44% → 78.54%**, while
class-1 recall changes **81.98% → 82.40%**. This identifies an asymmetric
question-binding contribution to the monitor's errors, without attributing every
remaining error to a single cause. There are still 315 valid correction cases
wrong both before and after anchoring, in addition to the 48 newly exposed
errors and 12 frozen invalids. The new correction quartet histogram contains
2, 8, 7, 133 and 90 quartets with respectively 0, 1, 2, 3 and 4 correct cells.

This diagnostic supports supplying the actual query through a deterministic
text interface in a future prospectively specified monitor. The revised
comparison was chosen after the outcomes and extraction errors were inspected;
37.50% must not replace 33.75% in the original study or be advertised as an
untouched held-out baseline. No interval, significance test or new model
capability claim is attached to this same-response intervention.

## Reproducibility and preservation

Exact SHA256 values for the principal files are:

| Artifact | SHA256 |
| --- | --- |
| New control helper | `c51f0b75dd00ae2739c0678867ca25dd9197762922e1871c411162f76d4a8ec0` |
| New control tests | `881e939eeec7178234c3e1effed003d94efcc6f2e4adf48562fbf806843963fd` |
| Saved monitor records | `aa58b9afaeedfe009236d8ac3f7c1b65a79aca6a733fc15c80bf4a07ce20787a` |
| Original monitor analysis | `6c7d5fba54c2e49170b1f5abcd4e32ebc1788bfb1f0ce134c018f5c9d02aa215` |
| Qwen evaluation rows | `c44b45c558210f556d6dfa9b811e6d60918a22a6b5ff4f32ad20bd6da7fb4515` |
| New validation artifact | `99be90d7a43bb3bb9984b409a45502cb74973ba70846fb6c8c142b596b6fe0b4` |
| Preserved measurement helper | `330873ab913abee2ed04f7cb2df7ba288f0842c583aa65065bb85bd5c3cec0e3` |
| Preserved measurement validation artifact | `968589f4f5c32bcf4ea4f368b2ee4afa513b1039142e1f4bee2ba6b96b894c81` |

The [new JSON](validation/2026-11-22-main-study-query-anchoring-posthoc.json)
also binds the saved monitor protocol and run manifest, bank file, frozen
parser source, prior measurement tests and prior measurement report. Every
source/code hash and every protected prior-artifact hash was checked again
after computation. The helper refuses to overwrite an existing output.
No existing measurement helper, hash, report or validation artifact was edited.

Validation used the known inner virtual environment: all 11 new tests pass,
including full-context coverage, explicit-question failure, ID/prose mismatch,
frozen invalid handling, saved-final-location inconsistency, and a fixture where
wrong final location and wrong query cancel. Ruff passes for both new files.

## Independent pre-fit review of the cached readout protocol

The reviewed [protocol](CACHED_READOUT_ADEQUACY_PROTOCOL_V1.md), SHA256
`364d79156b949398209dc2c4e21e463b3eefb8fd7164e7e244f48f9ad3187bee`,
has no conceptual group-split, factor-target or uncertainty blocker for the
stated **posthoc exposed-bank readout-adequacy diagnostic**:

- Whole narrative groups remain together; inner training, development and outer
  test memberships are disjoint at the appropriate stage. Fresh 24-group outer
  refits and scalers after selection are explicitly declared. The two models
  share group assignments; no outer-test feature controls preprocessing.
- Factor I and U mean location relative to the queried symbolic slot, so they
  remain defined when physical names exchange. B1/B2 are the accepted reading
  records. The four-head target supervision and external fold are privileged,
  and their different selection objective is explicitly labelled a pipeline
  difference rather than a pure classifier-capacity intervention.
- RBF uses the same inner/outer membership and train-only transform as linear
  heads. Its explicit C/gamma grid and positive-class orientation check avoid
  adaptive test selection. Up to 1,536 support vectors can remain, so success
  cannot be presented as a low-capacity or mechanistic explanation.
- The protocol correctly withholds naive bootstrap intervals and p-values for
  the overlapping outer folds. Group/fold/row outputs are descriptive, and
  exposed original test groups are not relabelled as untouched confirmation.

Six inner-development groups provide only **48 primary quartets** per fold,
making quartet selection coarse and potentially unstable. Keeping every
candidate metric is necessary; the declared grid should not be extended after
seeing results. Neither RBF success nor factor-fold success by itself proves
that the language model uses an intrinsically nonlinear belief representation.

One minor implementation issue was raised before fitting: the factor method's
synthetic `2*prediction-1` value should be named a folded binary score, rather
than a continuous target decision margin. Its real four factor margins are
separately retained. The inspected direct and RBF paths retain their real
decision margins. No native-data fit or model inference was run for this review;
execution should still verify the final sealed code and required corruption
tests after implementation is complete.
