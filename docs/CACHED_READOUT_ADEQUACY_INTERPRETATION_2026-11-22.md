# Cached adequacy: recoverable ingredients do not imply a reliable linear belief head

22 November 2026. The sealed cached diagnostic completed at
**20:01:16.730978 UTC**. This interpretation reads the unchanged native
out-of-fold results and their
[independent audit](validation/2026-11-22-cached-readout-adequacy-independent.json).
It adds no inference, fitting, case authoring or alteration to the original
study. Read the [generated report](results/cached-readout-adequacy-2026-11-22/results.md)
and [figure](results/cached-readout-adequacy-2026-11-22/readout_adequacy.pdf)
alongside the [frozen diagnostic protocol](CACHED_READOUT_ADEQUACY_PROTOCOL_V1.md).

The useful new finding is model-specific. **Llama's same saved midpoint vectors
support 95.00% correction quartets through four learned ingredient heads and
the supplied update rule, compared with 20.42% for direct logistic regression
and 65.00% for the RBF classifier. Qwen remains much weaker: 45.42%, 2.08% and
1.67%, respectively.** Thus Llama's original failure cannot be explained simply
as absence of recoverable task-relevant evidence at that site. Neither result
establishes an integrated belief representation or the model's use of one.

## What was evaluated

Each model contributes its original 1,920 correction vectors: 30 contextual
groups, all 64 variants/group and 240 primary quartets. Capture stays at block
16 for Llama and block 14 for Qwen, at the final complete chat-template token
after the explicit belief question and before any answer. There are no new
language-model calls or alternative layer captures.

Five outer folds each exclude six whole groups. Within the other 24 groups,
18 groups/1,152 rows fit candidates and six groups/384 rows select settings.
Each selected model is refitted on all 24 outer-training groups/**1,536 rows**,
then predicts its six excluded groups. Transforms and memberships are shared
across methods within each stage and use training rows only. All **220 inner
candidate configurations** completed without a failed candidate or outer refit;
a four-factor configuration contains four separate logistic heads.

These are **posthoc, exposed-bank, descriptive group-crossfit results**. The
original `split=test` metadata remains intact. Outer training sets overlap;
folds are not independent studies, and no naive bootstrap, p-value or
independent-fold standard error is justified. Human semantic review remains
incomplete. The two models share the same contextual groups and renderer.

| Model / correction-trained pipeline | Complete primary quartets | All-row accuracy | Primary-row accuracy | Easier secondary-row accuracy |
| --- | ---: | ---: | ---: | ---: |
| Llama direct LR | 49/240 — 20.42% | 1,514/1,920 — 78.85% | 74.06% | 83.65% |
| Llama RBF | 156/240 — 65.00% | 1,741/1,920 — 90.68% | 89.27% | 92.08% |
| Llama factors + rule | 228/240 — 95.00% | 1,904/1,920 — 99.17% | 98.65% | 99.69% |
| Qwen direct LR | 5/240 — 2.08% | 1,092/1,920 — 56.88% | 63.23% | 50.52% |
| Qwen RBF | 4/240 — 1.67% | 1,097/1,920 — 57.14% | 62.92% | 51.35% |
| Qwen factors + rule | 109/240 — 45.42% | 1,415/1,920 — 73.70% | 72.81% | 74.58% |

Primary cases have `I=M2=1-M1` and require all four receipt decisions correct.
Secondary cases have `I=M1` in the same two-message format, making B1 irrelevant
to the target. Their weakness, particularly on Qwen, prevents attributing all
failure to the harder primary dependency. Direct LR gets exactly three cells
correct in 147/240 Llama and 160/240 Qwen quartets: individual row accuracy is
not a substitute for joint consistency.

## Ingredient accessibility and remaining errors

Four-factor decoding predicts `I`, `M1`, `B1`, `B2` from the hidden vector, then
applies `1-M1 if B2 else M1 if B1 else I`. It receives four labels per training
row and an external algorithm. **It does not receive gold factors at prediction
time.** Its stronger supervision and supplied computation still make it a
privileged diagnostic, not an equal-supervision replacement for a final-label
linear probe. Its hard folded verdict is not a calibrated target probability.

| Model | Initial I | First message M1 | First receipt B1 | Final receipt B2 | All four factors correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama | 99.69% | 99.74% | 98.12% | 99.74% | 1,869/1,920 — 97.34% |
| Qwen | 94.95% | 75.00% | 92.24% | 87.45% | 1,117/1,920 — 58.18% |

Llama has only 6, 5, 36 and 5 ingredient errors, respectively. Qwen has 97,
480, 149 and 241; the first-message value is its weakest measured ingredient.
This identifies a limitation of this readout, not a causal explanation such as
forgetting or confusing the two messages. An ingredient can be wrong when that
ingredient is irrelevant to a row's final target, explaining why folded-target
accuracy can exceed all-ingredients-correct accuracy.

Llama's factor-fold method solves every primary quartet in 24/30 groups, but
not every context: the weakest group solves 3/8 quartets and 59/64 rows. Qwen
has no group with all eight primary quartets correct; its group range is 0/8
to 7/8. Strong aggregate accessibility should not be described as perfect
context-general reliability.

## Invariance, receipt sensitivity and fold spread

World and global name-map changes preserve the symbolic target. Each nuisance
has 960 linked pairs/model; those pairs are dependent variants of 30 groups.
The table counts verdict changes, not small movements in confidence.

| Model / pipeline | World flips | Name-map flips | Primary quartet % in folds 0,1,2,3,4 |
| --- | ---: | ---: | --- |
| Llama direct LR | 170/960 — 17.71% | 180/960 — 18.75% | 22.92, 14.58, 20.83, 27.08, 16.67 |
| Llama RBF | 81/960 — 8.44% | 81/960 — 8.44% | 66.67, 62.50, 64.58, 68.75, 62.50 |
| Llama factors + rule | 12/960 — 1.25% | 12/960 — 1.25% | 91.67, 95.83, 100.00, 89.58, 97.92 |
| Qwen direct LR | 212/960 — 22.08% | 140/960 — 14.58% | 2.08, 0.00, 2.08, 4.17, 2.08 |
| Qwen RBF | 159/960 — 16.56% | 147/960 — 15.31% | 2.08, 0.00, 4.17, 2.08, 0.00 |
| Qwen factors + rule | 167/960 — 17.40% | 193/960 — 20.10% | 47.92, 45.83, 39.58, 50.00, 43.75 |

Llama's nonlinear/factor gains occur across folds rather than coming from one
favorable fold. Factorization reduces both measured nuisance flip rates there.
For Qwen it improves target accuracy while **increasing name-map flips** relative
to direct LR; world invariance also remains imperfect. A better target score
does not guarantee better measurement behavior on every control.

Receipt controls give the same distinction. On Llama, both-endpoints-correct
success for belief-changing B1 pairs rises from 77/240 direct LR to 169/240 RBF
and 229/240 factor-fold. On Qwen factor-fold reaches 138/240, versus 20/240
direct LR. But Qwen factor-fold also changes its verdict on 72/240
belief-preserving B2 pairs (30.00%), compared with 38/240 direct LR (15.83%).
The factor method improves some required dependencies while retaining or
worsening other failures; it is not uniformly selective.

## Interpretation and connection to prior methods

The strongest supported conclusion is that **readout choice materially changes
recoverability from these saved Llama features under correction supervision**.
Direct LR remains limited even with target support; a nonlinear family does
better, and explicitly decoding ingredients plus supplying the rule does much
better. The label function contains interactions, so a linear classifier over
separately encoded facts need not compute it. This does not prove that the
hidden-state geometry is exactly factorized, or that the model never formed an
integrated belief elsewhere.
RBF uses the same final-label supervision as direct LR but changes the fitted
family and grid; it can retain up to all 1,536 outer-training rows as support
vectors. It is not inherently a low-capacity instrument. Factor selection also
uses a different development objective, making its comparison a whole-pipeline
diagnosis rather than an isolated intervention on classifier capacity.

Qwen supplies a contrasting result, not a successful architecture-wide
replication of Llama's pattern. The tested RBF family does not rescue its joint
performance, and ingredient extraction itself remains substantially imperfect.
That leaves location, encoding, task support, formulation and finite fitting
capacity unresolved. It does not justify a global ranking of belief ability or
an explanation in terms of quantization alone.

These readings concern predictive accessibility, not causal use. In
[Steele et al., Section 4.4 and Appendices B–C](https://arxiv.org/pdf/2607.11945v2),
probe-recoverable information, routing interventions and scaffolded elicitation
are separate evidence. Their distinctions remain necessary here. The Qwen
accuracy/invariance tradeoff also illustrates why the named-nuisance diagnostics
in [Klein, Sections 2–4.1 and Appendix B](https://arxiv.org/html/2611.07805v1)
cannot be replaced by one accuracy score; our result is not a replication of
that paper's particular edit metric.

[Prakash et al., Section 5.1 and Appendix M](https://arxiv.org/html/2505.14685v3)
distinguish computation stages and distribution-dependent final-token subspaces,
while [Orgad et al., Sections 3.2–4](https://arxiv.org/html/2410.02707v4)
make validation-selected token/layer choice and output baselines substantive.
These precedents motivate the next capture control; they do not identify the
mechanism of our errors. The broader occupied claims and exact reading scope
remain in the [research evidence map](MAIN_STUDY_RESEARCH_EVIDENCE_MAP_2026-11-22.md).

The [original matched study](MAIN_STUDY_FINAL_INTERPRETATION_2026-11-22.md) remains
unchanged. Comparing its source-trained N512 heads with the present
correction-trained heads changes fitting size, groups, supervision, parameter
grids and selection. The numerical difference is not a causal estimate of
adding correction examples or repairing coverage. The cached results also do
not validate spontaneous anticipation, actual private reader beliefs, deceptive
intent or a deployable deception monitor.

## Fresh-study decision

The [fresh draft](FRESH_READOUT_STUDY_SPECIFICATION_V2.md) proposed retaining the
midpoint only if direct LR reached 80% mean primary quartets, 90% secondary-row
accuracy and no fold below 60% primary quartets, in both models. **Direct LR
fails all three conditions in each model.** Llama factor-fold passes those
operational cutoffs, but Qwen factor-fold does not; neither RBF model meets the
primary cutoff. The draft's cheaper direct-LR branch is therefore unsupported.

Proceed in planning with its **four-depth by three-position branch**, keeping
both models. Source-only one-step development chooses one site; correction
development cannot choose that source site. The bounded primary remains Llama
N512 source-trained LR at the selected site minus same-policy LR at the original
site, on fresh renderer-A hard quartets. Under the draft's common-control rule,
C2 is four-factor LR plus the declared fold, since RBF is not adequate in both
models. This does not certify C2 as a strong Qwen positive control; its limitations
must be retained.

The draft still requires a concrete bank, review record and execution freeze;
this note launches none. Its 48 new contexts include 24 paired test units, with
one-step, easy/hard two-message and explicit-attribution conditions in two fixed
renderers, for 7,040 prefixes/model. The 12 vectors come from each same prefill.
No model-correctness filtering, new test-selected sites or continued expansion
until success is allowed.

The cached heads use **1,536 correction-fitting rows**, while that draft proposes
512. Even Llama's 95% result does not qualify an N512 instrument or guarantee
its new-context/renderer transfer. Future changes to that budget or the draft
criteria must be declared before fresh outcomes, not represented as part of
the completed cached experiment.

## Exact source bindings

The independently reconstructed saved predictions and metrics, not a refit, are
the measurement source for this note. The audit reconstructs 15 method/fold
bundles and 5,760 out-of-fold predictions per model, checks training-only
transforms and reports every group/fold/aggregate metric matching. It performs
zero classifier refits and imports no pipeline. That verifies saved measurement
integrity, not human semantics or causal faithfulness.

| Source | SHA256 |
| --- | --- |
| [Llama out-of-fold results](//wsl.localhost/Ubuntu/home/adirj/kcl-latent-probes-runs/readout_adequacy_cf_v1_20261122/llama/oof_results.json) | `4b597a0395e3f98b6037ebe9be767b91ddad9377553de1897e6dd6ea94fa2d88` |
| [Qwen out-of-fold results](//wsl.localhost/Ubuntu/home/adirj/kcl-latent-probes-runs/readout_adequacy_cf_v1_20261122/qwen/oof_results.json) | `4725df0fa9789af77b54671b56608b57588e01342c810f33f2365c536401f5a8` |
| [Completed manifest](//wsl.localhost/Ubuntu/home/adirj/kcl-latent-probes-runs/readout_adequacy_cf_v1_20261122/completed_manifest.json) | `35d4ce616c456694ec9794f133c9116979bb65d61f91c260347d94c88fc424d1` |
| [Independent audit](validation/2026-11-22-cached-readout-adequacy-independent.json) | `7def38061c54c627526c227f377381bcbf2d78fafeaaae70dae4f20376383b42` |
