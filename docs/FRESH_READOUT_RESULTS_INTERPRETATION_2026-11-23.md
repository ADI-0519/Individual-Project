# Fresh belief-readout study: completed results and limits

23 November 2026. The full campaign, independent saved-data reconstruction,
report export and exact repository copy finished at **02:43 GMT**. All 7,040
planned cohort prefixes per checkpoint are complete: 14,080 across Llama and
Qwen, including 4,992 fresh test prefixes each. The 2,948 measurements completed
before interruption were retained without repetition. Engineering forwards
are separate from these cohort counts.

The [frozen protocol](FRESH_READOUT_EXECUTION_PROTOCOL_V2.md),
[pre-outcome interpretation rules](FRESH_READOUT_INTERPRETATION_RULES_V2.md),
[complete generated tables](results/fresh-readout-2026-11-23/results.md),
[machine-readable summary](results/fresh-readout-2026-11-23/summary.json) and
[figure PDF](results/fresh-readout-2026-11-23/fresh_readout.pdf) remain distinct
from this post-result interpretation. Human semantic review is incomplete;
these are **agent-audited exploratory results**.

## Primary result: source-only site selection has no established benefit

The primary compares source-trained logistic probes with the same training
rows and classifier family. S0 fixes the original capture site; S1 selects a
site using source-development loss alone. The test endpoint requires all four
linked counterfactual judgments in a hard-correction panel to be correct.

| Model, original renderer A | S0 | S1 | S1 minus S0 | Conditional 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Llama, primary | 11/192 (5.73%) | 13/192 (6.77%) | +1.04 pp | [-3.65, +5.73] pp |
| Qwen, named secondary | 1/192 (0.52%) | 2/192 (1.04%) | +0.52 pp | [-1.04, +2.60] pp |

Both intervals include zero. This selection procedure has not demonstrated
better transfer. This does not establish equivalence, absence of belief
information, or an optimal capture site. The twelve-site descriptive panel
cannot supply a replacement test-selected winner. The bootstrap uses the
fixed 10,000 whole-context draws over **24 paired contexts**, shared by both
models. Its intervals condition on this authored bank, selected sites and
fitted heads; the models do not provide 48 independent context samples.

## Fresh descriptive result: factor-readout transfer is format-sensitive

C1 is a direct correction-trained logistic readout. C2 learns four factors
from the same correction fitting rows and combines their predictions with a
supplied update rule. Both operate at S1's site and use separate correction
development data; neither is selected on test results.

| Model / renderer | S0 | S1 | C1: direct | C2: four factors + supplied rule | Conditional YES/NO |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama A | 11/192 (5.73%) | 13/192 (6.77%) | 44/192 (22.92%) | **176/192 (91.67%)** | 5/192 (2.60%) |
| Llama B | 1/192 (0.52%) | 0/192 | 8/192 (4.17%) | **17/192 (8.85%)** | 5/192 (2.60%) |
| Qwen A | 1/192 (0.52%) | 2/192 (1.04%) | 16/192 (8.33%) | **118/192 (61.46%)** | 0/192 |
| Qwen B | 0/192 | 0/192 | 3/192 (1.56%) | **12/192 (6.25%)** | 0/192 |

These are complete hard-quartet counts. A and B express the same formal states
in the same test contexts. B changes the wording and presentation, including
event ordering, question wording and the system instruction. It is a bundled
format shift, not an isolated synonym substitution. All training/development
uses A. The C2 A-to-B drops are **82.81 pp for Llama** and **55.21 pp for Qwen**;
these remain descriptive comparisons, without new posthoc significance tests.

This supports a bounded result: four separately supervised ingredients can
be decoded and combined effectively on fresh contexts in A, especially for
Llama, while the frozen pipeline transfers poorly to B. Extra labels, four
heads and external computation make C2 a privileged diagnostic. It does not
show that the model uses this rule internally, establish spontaneous reader
anticipation, or supply an equal-capacity comparison with direct probing.

The earlier cached 95.00% / 45.42% factor results used exposed-bank crossfit
and 1,536 outer fitting rows. This study uses fresh test contexts and 512
correction fitting rows. The outcomes must not be merged into one replication
estimate or described as the same fitting experiment.

## Controls narrow the explanation

- **Ingredient decoding also shifts.** All four factors are correct on 96.22%
  of Llama hard rows in A versus 30.08% in B; Qwen falls from 80.08% to 20.18%.
  Qwen's B first-receipt readout is 50.26% accurate. Failure therefore cannot
  be assigned solely to the supplied final fold.
- **Nuisance sensitivity rises.** C2 hard world/name verdict-flip rates rise
  from 3.39% / 3.39% to 40.89% / 33.07% for Llama, and from 11.20% / 8.07%
  to 36.46% / 29.17% for Qwen. Intended invariances do not transport reliably.
- **Row accuracy can conceal linked failure.** Llama C1 in B gets 575/768 hard
  rows correct (74.87%) but only 8/192 full panels (4.17%). Exactly 178 panels
  have three of four judgments correct. The panel metric exposes errors that
  an aggregate row score would obscure; this alone does not identify a cause.
- **Simpler controls are imperfect.** S1 one-step panel accuracy in A is
  46.88% for Llama and 34.90% for Qwen, falling to 4.69% / 10.94% in B.
  S1 easy-correction A panels are only 18.75% / 0.52%. There is no clean
  demonstration of isolated composition failure after uniformly competent
  simpler reasoning. Factor validity is defined only for the two-message
  cases; C2 one-step/asserted verdicts are pipeline transfer controls.
- **Threshold accuracy is not information absence.** Qwen S1 B/asserted has
  50% accuracy but raw AUROC about 0.994. This suggests a boundary/score-shift
  issue in that control, without licensing retrospective threshold repair or
  explaining every hard-correction error.
- **Conditional output is an elicitation-limited baseline.** On Llama B hard
  cases, the full-vocabulary first choice falls outside YES/NO on 200/768
  prefixes. Strict first-token accuracy is 339/768 (44.14%), whereas the fixed
  two-token comparator gives 456/768 (59.38%). Neither observes free generation.

Source-versus-correction training differs in narrative count and mechanism
mix despite matching row budget. NF4 is the only evaluated precision. Both
models share formal rules and contexts, and the two renderers are not a sample
of arbitrary language shifts. Agent semantic agreement and numerical audits
cannot certify absence of every bias or substitute for independent human review.

## Completion, audit and the next decision

Both model fitting bundles were sealed before either fresh test capture.
The presealed numerical checker independently reconstructs predictions and
metrics from saved arrays, freezes both reconstructions before reading either
saved outcome, and then compares them. The audit passed and both completion
children exited zero. All 169,728 verdict entries and 39,936 factor predictions
match exactly; the largest reported numerical difference is 3.20e-14 against
the fixed 1e-9 tolerance. All seven repository exports match the native report
bytes. See the [separate-agent completion review](validation/2026-11-23-fresh-readout-completion-review-v1.json)
for final receipt hashes and the explicit scope of those checks. Original
startup/interruption evidence remains preserved; no scientific fit, threshold,
case, site or output was changed after inspecting these test results.

After completion, GitHub's Windows launcher tests exposed a missing
`Get-FileHash` command. A narrow infrastructure repair replaces that dependency
with streaming .NET SHA256. Both regressions reproduced before the patch;
all nine focused launcher tests pass afterward. The
[portability receipt](validation/2026-11-23-fresh-readout-launcher-portability-v1.json)
preserves the exact launcher used by the completed run. This post-completion
repair does not change or repeat scientific execution; its new CI is separate
from the scientific freeze's passing checks.

The immediate research decision is to retain this result, complete human
semantic review, and discuss a **narrow format-transfer follow-up** with the
supervisor. A proposed fresh protocol would compare A-only with mixed-format
training at fixed sites, matching narrative groups, total row count and method
supervision; declare B hard complete-quartet accuracy primary; and retain A,
factor accuracy, class recalls and nuisance controls. Fresh paired contexts
would be reviewed before inference, with all selection restricted to development.
That follow-up is a proposal, not a launched or confirmatory experiment. The
current bank is now exposed and cannot become its fresh held-out test.

The [primary-paper evidence map](MAIN_STUDY_RESEARCH_EVIDENCE_MAP_2026-11-22.md)
already documents prior second-order belief, capture-site, selectivity and
composition work. The candidate contribution is the controlled reliability
comparison and its limits, not first discovery of second-order belief or a
validated deception detector. The dissertation and any publication remain
unfinished even though this experiment is complete.
