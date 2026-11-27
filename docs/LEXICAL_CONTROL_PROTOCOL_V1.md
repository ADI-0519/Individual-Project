# Lexical shortcut controls on the exposed adaptation bank

27 November 2026. This diagnostic is specified after observing the completed
latent study and before any lexical fitting or scoring. It is not a new held-out
confirmation, and it cannot replace the original primary. Human semantic review
remains incomplete. Keep the original study and all its null results unchanged.

## Question and interpretation

Can a fixed, shallow classifier of the supplied text solve the same four-factor
assay, and does allocating its fixed fitting budget to both formats help?
This addresses a missing lexical shortcut control. It does not establish the
incremental value of activations over a capable neural text monitor. In
particular, n-grams cannot generally bind a newly named location across distant
sentences. Failure of this comparator cannot rule out textual explanations.

The full-messages view serializes only model-visible message roles and content
using a simple newline serializer, without checkpoint-specific chat special
tokens. It does not include metadata, group or case IDs, formal-state bits,
labels, spans, or authoring notes. The system-only view discards user content;
it tests an instruction/template shortcut. It is a negative control, not a
surrogate for the complete task. Assert exact within-format factor balance on
each permitted fitting allocation. Use both views regardless of score.

The latent site is at the final token of the queried location, before the final
answer-format instruction and assistant suffix. The lexical full-messages view
also includes that final format-specific instruction, which carries no variable
factor fact. It uses the same supplied source prose, not the identical causal
token prefix or checkpoint-specific chat-template representation.

## Frozen observations and learning policy

Reuse the immutable 40-context adaptation bank: 16 fitting and 24 test contexts,
768 available fitting renderings and 4,992 test renderings. Preserve the original
512-state XOR roster, A512/Mixed512/Ahalf256 arms, and 256 B Offset observations.
Retain the same four factor labels and supplied update rule. There is one text
comparator shared across the two checkpoints, not two independent replications.
Verify every test row ID, state, gold and message hash against both original
model capture projections. Verify all 39 original repository bindings in the
preparation's 50-entry source/input inventory. Preserve the other 11 historical
native artifact identities; a portable replay does not depend on those paths.

For each view and each of A512, Mixed512 and Ahalf256, fit word 1–2 gram and
character 3–5 gram TF-IDF vectorizers on that arm's own authorized fitting
documents only. Each uses lowercase, min_df=1, max_features=20,000,
sublinear_tf=True, float64 and L2 normalization; the word token pattern is the
sklearn default `(?u)\b\w\w+\b`. Concatenate the two equally weighted blocks,
then L2-normalize again. Never learn vocabulary, IDF, document frequency,
normalization, hyperparameters or feature selection on test text. Offset keeps
the entire Ahalf vectorizer, normalization, coefficients and source intercepts;
learn only one additional B intercept per factor from its prescribed 256 B rows.

This per-arm text representation is appropriate for a train-only lexical
control, but differs from the common historical feature map in the latent
experiment. A lexical Mixed gain changes available features/IDF as well as
coefficients. It is not evidence about a uniquely identified geometric mechanism.

Use float64 logistic regression, lbfgs/L2, unpenalized intercept, tol=1e-4,
max_iter=2,000, random_state=11, no class/sample weights, four BLAS threads.
Keep `C_n=512*100/n`, hence C=100 at 512 and 200 at 256. This numerical choice
is fixed for the comparator; equal C does not equalize text and hidden-state
feature geometry or effective capacity. No grid or rescue fit is allowed.
Nonfinite values or convergence warnings stop and preserve the attempt.

Margin strictly greater than zero predicts one. Fold the four predicted factors
as `1-M1 if B2 else M1 if B1 else I`. Exact-zero margins predict zero. Never
reverse a test sign or substitute true factors. Do not characterize a random
row-label permutation as Hewitt–Liang selectivity; no permutation study is
included in this fixed diagnostic.

## Seals, analysis and reporting

Before fitting, hash the protocol, implementation, tests, separate-author
numerical auditor and immutable inputs into an exclusive preparation receipt.
Fitting accepts fit-only rows, freezes both views and all four arms, and saves
safe JSON/NPZ without pickle. A second stage scores text-only projections of
the test inputs. Partial/failed attempts remain and cannot be overwritten or
silently repeated. No GPU or new language-model call is required.

Report every arm/view/format/condition, row accuracy, class recalls, complete
receipt quartets and correct-cell histograms, world/name flips and both-correct
pairs, all context outcomes, correction-factor AUROC/accuracy/recalls/quantiles,
and all-four correctness. One-step and asserted conditions remain
out-of-training-construct controls. Retain all 24 contexts. Numerical failures
stop and preserve the entire attempt; they are not replaced by fabricated
predictions. Do not fabricate output logits for a text classifier.

The fixed descriptive contrasts are B-hard Mixed512−A512 and Mixed512−Offset;
use the original paired whole-context 10,000 bootstrap draws, seed 20261127.
These conditional intervals exclude fitting-cohort uncertainty. They are
posthoc descriptive comparisons with no new primary or familywise guarantee.
The independent auditor reconstructs all declared metric scopes from saved
scores, freezes its reconstruction before opening the saved summary, and checks
the complete result. This is independent metric replay, not independent refitting
or human semantic certification. Publish its failures as well as its successes.

Any follow-up feature choice prompted by these results requires a new version
and an explicit exposed-bank label. A useful next confirmation needs a separately
frozen protocol and semantic review. CREATE precision replication is a separate
proposal; these CPU controls do not establish NF4/BF16 equivalence.
