# Cached readout adequacy: prospective fitting protocol on an exposed bank

22 November 2026. This is a new **posthoc, exposed-bank diagnostic**. The original
study's test outcomes have already been inspected. Nothing here turns those
cases into an untouched test or changes the frozen study's results. The user
explicitly requested independent rechecking and substantial continued research.
Human semantic review remains incomplete; no human acceptance is fabricated.

## Question and scope

Can correction belief content be decoded from the **same saved midpoint
activations** when correction supervision is supplied, and does nonlinear or
factorized decoding differ from a direct linear head?

This is a readout-adequacy control. It does not establish spontaneous
anticipation, deceptive intent, a causal belief mechanism, general capability,
or an isolated effect of compositional training. A successful correction-trained
head would show accessibility at this site after such supervision; it would not
repair or reinterpret the original one-step-trained probe's failed transfer.

No new language-model inference is needed. Use all 30 correction groups and all
64 captured variants per group from each completed model: 1,920 rows/model.
Original row metadata and `split="test"` stay unchanged. New crossfit roles are
separate metadata. Both models use exactly the same group assignments.

## Frozen grouping

Seed: **20261122**. Sort the 30 group IDs by hexadecimal SHA256 of
`readout-adequacy-cf-v1|20261122|{group_id}`, with group ID breaking any tie.
Assign sorted groups round-robin to five outer test folds (six groups/fold).

For each fold, sort the other 24 groups by SHA256 of
`readout-adequacy-cf-v1|inner|{fold}|{group_id}`. First 18 groups are inner
training, final six are inner development. All variants of a narrative remain
together. No outer-test feature, label or prediction may affect preprocessing,
selection, fitting, stopping or calibration of its predicting model.

After inner selection, refit the selected configuration once on all 24 outer
training groups. This absorption of inner development is declared in advance.
Fit a fresh scaler using those 24 groups only. Predict the six outer-test groups
once. Every original correction row receives one out-of-fold prediction per
method. Outer training sets overlap; these are not five independent studies.

## Preprocessing and methods

Use float64 features, train-only `StandardScaler`, then per-row L2 normalization.
Fit inner and outer transforms separately on their respective training sets.
Each method shares the same transform and membership within a given stage.
Feature caches must retain exact row order, finite values and recorded hashes.
No feature selection, PCA, layer search, label-dependent sign reversal or
additional calibration is allowed in this diagnostic.

1. **Direct linear:** L2 logistic regression, `lbfgs`, intercept, `tol=1e-4`,
   `max_iter=2000`, no class weighting. C grid: **0.01, 0.1, 1, 10, 100**.
2. **Regularized nonlinear:** RBF SVC, `probability=False`, `tol=1e-3`,
   `max_iter=200000`, cache 512 MB, no class weighting. C grid:
   **0.1, 1, 10, 100**; gamma grid **0.1, 1, 10** after the stated normalization.
3. **Factorized linear plus fixed fold:** four logistic heads with the same
   settings as method 1 and one shared C. Targets are initial location I,
   first-message value U, first accepted receipt B1 and second accepted receipt
   B2. On these correction accounts the second value is 1-U. Predict factors
   by margin > 0, then calculate `1-U if B2 else U if B1 else I`.

Method 3 has **four target labels per training row and an external algorithm**.
It is an explicitly privileged diagnostic, not an equal-supervision or
equal-compute improvement over the direct classifier or original probes.
Its hard folded output is not a calibrated target probability.
If a common record format encodes that verdict as a synthetic -1/+1 margin,
mark it explicitly as a synthetic hard score and retain the binary folded
score; it is not a continuous target margin. The four genuine factor margins
remain separately available.
RBF may retain up to all 1,536 outer-training rows as support vectors; it is
not necessarily a low-capacity instrument. Success would show accessibility
under that supervised nonlinear family, not the language model's own mechanism.

For direct linear and RBF, select by highest inner-development primary
complete-quartet accuracy, then highest all-row accuracy, then smallest C and
smallest gamma. For factor heads, select shared C by lowest mean of the four
binary development log losses; tie goes to smaller C. Report the factor-fold
development target metrics, but do not use them to select C.
These different declared selection objectives make the factor comparison a
comparison of whole pipelines, not a pure intervention on classifier capacity.

Keep every candidate's selection metric and failure reason. Convergence
warnings, nonfinite parameters/development scores/losses and failed SVC fit status invalidate that
candidate. If no candidate succeeds, or the selected outer refit fails, stop
and preserve the failure. Do not silently increase iterations, change grids,
exclude cases, retry fits or substitute another selected configuration.

Threshold is margin > 0; ties predict zero. SVC positive-class orientation must
be verified from `classes_`. Do not present sigmoid-transformed SVC margins as
calibrated probabilities. Numerical environment and thread limits are recorded
before the first fit and remain fixed.

## Outcomes and uncertainty

Retain every row's ID, original split, state, group, outer fold, prediction and
decision margin. Preserve factor margins and predictions, all four-factor joint
accuracy, each factor's accuracy, and the final folded verdict. Save safe NPZ
coefficients or support vectors/dual coefficients/intercept/gamma plus exact
training/support IDs; never load or emit estimator pickles.

Report separately for every method/model/fold/group:

- Complete primary correction quartets (I=1-U), all four judgments required.
- All-row, primary-row and same-format secondary I=U row accuracy and class
  recalls. Report histograms of zero through four correct quartet cells.
- Receipt-pair consistency and gold-preserving world/name verdict flips.
- Factor marginal accuracy and all-factors-correct joint accuracy for method 3.
- Every failed candidate/refit and every missing prediction, without dropping
  inconvenient groups from an otherwise available denominator.

Aggregate out-of-fold values are **descriptive**. Do not produce a naive
whole-group bootstrap or p-value that ignores overlapping training folds and
the exposed-bank design. Retain all foldwise and original-group values. The
original one-step-trained readouts may be shown as a fixed descriptive reference
only: they use different training groups, support and selection. Any difference
is not a matched causal estimate of adding correction examples.

## Execution seal and reproducibility

Before any native-data fitting, create a JSON protocol and fold manifest binding
this document, new implementation/runner hashes, both completed source caches,
bank digest, model identities/revisions/layers and exact memberships. Track the
document/code in Git before preparing execution. `prepare` verifies everything
without fitting; `execute` checks the sealed configuration before and after the
job, refuses existing outputs, and retains explicit failure state. New results
use `research_status="posthoc_exposed_bank_group_crossfit"`,
`human_review_complete=false`, and `confirmatory_result=false`.

Tests must cover group leakage, train-only transforms, source/row corruption,
threshold ties, Boolean correction folding and safe serialization. Tiny
synthetic fixture fits are permitted before the native-data seal and never
reported as model results. Existing model captures, fitted heads, protocols,
paper figures, human ledgers and the completed campaign remain untouched.

A separately authored, agent-reviewed fresh cohort and development-selected
layer/token panel are a subsequent study. Existing exposed correction cases
cannot be relabelled as its untouched confirmation.
