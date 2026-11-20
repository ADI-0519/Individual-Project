# Current-belief readout under private future-goal contexts

**20 November 2026.** The frozen nested-belief readout changes its scores when
the sender's desired outcome for a later report changes, although the scene,
question and stipulated current belief stay fixed. The probability of the
nested-positive label is higher under the positive than negative future goal on
**all 32 unique sender inputs**, with mean difference **+0.024799**.

This is an **exploratory robustness result on an already-observed bank**. No
report, answer or reader outcome was generated. It does not establish actual
deception, altered beliefs, a causal incentive mechanism or successful transfer
to pressured communication. Goal polarity, negation and length remain confounded.
The [curated record](./validation/2026-11-20-belief-goal-stress.json) preserves
primary metrics, exact paired contrasts, additional ranking diagnostics and hashes.

## Controlled target and comparison

The [known-message result](./KNOWN_MESSAGE_RESULTS.md) supplies the original
56/64 baseline and immutable activations. The source bank contains four wordings
of one message-update logical family, with **64 rows but 32 unique sender inputs**.
Actual delivery is hidden from the sender; opposing private-delivery variants
therefore repeat the same input. Gold is the current sender-attributed reader
belief after an exact known message and trusted receipt.

The [frozen protocol](./BELIEF_GOAL_STRESS_PROTOCOL.md) adds the same sender role
and private-answer scaffold to the system text in three new conditions. The
original system text remains intact, and the user scene and nested question are
byte-identical. A common statement explicitly says that the later report has
not been written or delivered and supplies no new current evidence. The exact
candidate claim appears once in the shared addition in every new condition.
Only the preferred future outcome varies:

| Condition | Future preference |
| --- | --- |
| Role neutral | No preferred outcome about the reader later believing the claim |
| Goal true | Reader later believes the candidate claim |
| Goal false | Reader later believes its negation |

These preferences do not change the fictional current observation, receipt or
trust rules. The current nested labels remain identical. The original baseline
has no added role scaffold, so original-to-role differences include role framing.
The comparison between the two future goals uses the same new role scaffold.

Every condition uses the **same frozen prior-trained block-16 head** and its
source scaler, normalization, coefficients and intercept. The decision threshold
stays 0.5. There is no fitting, calibration, threshold adjustment or layer search.

## Primary results

| Context | Correct / 64 | Log loss | Negative recall | Positive recall |
| --- | --- | --- | --- | --- |
| Original neutral, reused | 56 | 0.641871 | 81.25% | 93.75% |
| New role neutral | 44 | 0.657305 | 93.75% | 43.75% |
| New goal true | 50 | 0.652018 | 100% | 56.25% |
| New goal false | 34 | 0.676499 | 100% | 6.25% |

The negative future goal almost collapses positive recall at the fixed threshold.
The neutral role scaffold itself reduces accuracy, so the entire difference from
56/64 cannot be assigned to the future goal.

| Wording group | Original | Role neutral | Goal true | Goal false |
| --- | --- | --- | --- | --- |
| km_t01 | 14/16 | 12/16 | 16/16 | 10/16 |
| km_t02 | 12/16 | 8/16 | 8/16 | 8/16 |
| km_t03 | 14/16 | 12/16 | 14/16 | 8/16 |
| km_t04 | 16/16 | 12/16 | 12/16 | 8/16 |

The predeclared paired comparison removes duplicate private-delivery rows:

- Goal true minus goal false: **32 positive changes out of 32 unique inputs**;
  mean positive-label probability difference **+0.02479912286473508**, median
  **+0.020357087192460405**, mean probability-log-odds difference **+0.09978783998681279**.
- Role neutral minus original: **29 negative and three positive changes**;
  mean probability difference **-0.03222398613936936**, mean probability-log-odds
  difference **-0.1293721090281521**.

The positive goal is higher than the negative goal on every input; this does not
mean that it raises every score relative to the neutral-role control. The
curated record preserves individual pairs and group summaries. No p-value treats
these 32 inputs within four wordings as independent draws.

The frozen nested-target TF-IDF comparator stays at 32/64 with zero paired score
changes. Frozen actual-reader activation and text heads remain exactly 50%, as
required by hidden delivery. TF-IDF is a limited comparator, not evidence that a
capable text monitor has been beaten.

## Additional post-hoc ranking diagnostic

After observing the primary results, AUROC was calculated from saved predictions
to distinguish threshold sensitivity from ranking loss. **This is an additional
post-hoc diagnostic**, not a new primary endpoint. No threshold was selected or
moved. The pairwise calculation uses 32 unique inputs with half credit for ties;
uniformly duplicated 64-row AUROC is identical.

| Context | AUROC | Mean score, negative cases | Mean score, positive cases |
| --- | --- | --- | --- |
| Original neutral | 0.957031 | 0.483079 | 0.537062 |
| Role neutral | 0.878906 | 0.458655 | 0.497038 |
| Goal true | 0.914063 | 0.454069 | 0.498030 |
| Goal false | 0.757813 | 0.440083 | 0.462417 |

The negative goal shifts scores downward and weakens pooled positive-versus-negative
ranking. Its 34/64 threshold accuracy therefore supports neither absence of all
predictive information nor a pure constant-offset explanation. Log loss remains
below the balanced uninformative value, while confidence interpretation is unvalidated.

Under goal true, AUROC is **1.0 within every wording**, despite pooled AUROC
0.914063 and only 50/64 threshold accuracy. This exposes between-wording score
offsets. Under goal false, within-wording AUROCs are 0.875, 0.6875, 0.75 and 1.0.
Some ranking deterioration therefore also occurs within wordings. These small
group descriptions do not justify retrospective group-specific thresholds or a
general calibration claim.

## Execution, preserved failure and independent audit

The executed source was committed at
`41a86bbef76fb12c6245d8896a3f2f2ab8989e51` before the assay.
The first run, [`belief_goal_stress_v1`](../runs/belief_goal_stress_v1), failed
after 152 capture rows with `PermissionError` during atomic replacement of its
capture ledger in the OneDrive-hosted directory. It has no report or prediction
file and remains preserved, unscored. The process responsible for that access
failure was not established.

The identical assay was repeated once into a new native WSL directory to avoid
that Windows-hosted write path. **Inputs, model, fitted heads, source code and
settings did not change.** All 82 surviving prefix/reference NPZ files shared
with the failed run are byte-identical to their completed-run counterparts.
This was an infrastructure retry, not selection between scored model results.

The completed protocol froze at **12:08:36.386183 UTC**; model execution began
at **12:11:29.149478** and finished at **12:12:11.264590**. The original native
output is `/home/adirj/kcl-latent-probes-runs/belief_goal_stress_v2`; the local
[`belief_goal_stress_v2` mirror](../runs/belief_goal_stress_v2) is independently
verified byte-identical across all 109 recorded files, including its manifest.
Both raw runs remain ignored by Git.

Independent Node reconstruction, without fitting or model calls, verified:

- All **1,024 prediction rows**, including source-normalized activation scoring
  and reconstructed TF-IDF; maximum probability difference **4.9960e-16**.
- All **80 global/group metric objects**, including 40 actual-reader negative
  controls, and the saved paired differences and primary summaries.
- 192 new rows from 96 unique prefixes; **576 row/layer vector bindings** and
  12 source-reference layer bindings. New inputs preserve the original sender
  scene/question and current gold across the three goals.
- Four exact original-prefix references, three exact new-condition repeats and
  genuine HF block-index checks, with zero answer tokens or generated reports.
- **615 file-hash comparisons**, 14 backend identity fields and all 109 native
  output/mirror file comparisons. Original run and fitted artifacts are unchanged.

The committed [curated JSON](./validation/2026-11-20-belief-goal-stress.json)
records major-file, executed-policy, frozen-fit and failed-run hashes. Mathematical
and software checks do not substitute for independent human review of the
agent-assisted fictional materials.

## Scope and next controls

The useful finding is that a frozen current-belief readout can respond consistently
to a desired future outcome that does not change its stipulated current target.
It identifies a robustness problem worth investigating alongside the earlier
positive known-message transfer. The present wording does not isolate a semantic
goal effect from negation, length or task-conditioning effects.

Before a stronger interpretation, express future goals directly as the first
versus second named location, avoiding the word “negation”; add irrelevant-goal
and matched-length controls. Then freeze fresh human-reviewed reasoning families
and independent neutral calibration before actual pressured communication.
The original natural-report study remains separate. No further GPU run is
required to preserve or report this completed milestone.

## Reproduction

Use a fresh native WSL output directory and the existing cached model/fitted runs:

```bash
HF_HOME=.cache/huggingface-wsl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
HF_HUB_DISABLE_TELEMETRY=1 TRITON_CACHE_DIR=.cache/triton-wsl \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv-wsl/bin/python scripts/run_belief_goal_stress.py \
  --config configs/apollo_llama8b_4bit.json \
  --known-run runs/known_message_validation_v1 \
  --fit-dir runs/belief_query_comparison_v1/fits \
  --output /home/adirj/kcl-latent-probes-runs/belief_goal_stress_reproduction
```

The runner refuses overwrite or implicit retry. This report initiates no rerun.
