# Llama interim interpretation: matched coverage does not improve correction transfer

22 November 2026. Read-only interpretation of the completed Llama analysis;
Qwen replication and the fixed prose monitor are still outstanding. No model,
head, threshold, source policy or analysis output was changed for this note.
These are real **agent-audited exploratory** observations, not human-reviewed
confirmatory results or evidence that second-order beliefs are absent.

Source: `main_v1_agent_20261122_recovery_v1/llama_analysis_without_monitor.json`
under `/home/adirj/kcl-latent-probes-runs/`.
SHA256: `32793a3e5bfbb31355dcacb9638ab17c43d99bd4a8947732d6acdfd9aae104e7`.
All tables average authored groups and the five fixed allocations. Percentages
are descriptive except the single predeclared primary interval. Rows and
allocation repetitions are not additional independent scenarios.

## Primary result

At N512, complete correction-quartet accuracy is **3.75% restricted versus
3.00% full**: full minus restricted **−0.75 percentage points**, with the
predeclared whole-group bootstrap interval **[−1.9167, +0.3333] points**.
There are 30 correction groups, eight quartets per group and five allocations.
The 1,200 quartet evaluations per arm are repeated evaluations of those groups.
This provides no positive evidence for the proposed matched-coverage benefit
in this Llama configuration; it does not establish exact equivalence or absence
of encoded belief information.

| Allocation seed | Restricted quartet % | Full quartet % | Difference, points |
| --- | ---: | ---: | ---: |
| 11 | 2.5000 | 3.7500 | +1.2500 |
| 23 | 5.0000 | 2.9167 | −2.0833 |
| 37 | 3.3333 | 2.5000 | −0.8333 |
| 53 | 4.5833 | 4.1667 | −0.4167 |
| 71 | 3.3333 | 1.6667 | −1.6667 |

## What succeeded and where transfer breaks

The readouts generalize substantially to fresh one-step narratives. They perform
much less reliably on the held-out two-message format, including its simpler
I=M1 secondary cells. Thus the low primary score cannot be attributed solely to
the primary correction path's extra logical demand.

| N512 raw activation readout | Restricted row % | Full row % | Restricted quartet % | Full quartet % |
| --- | ---: | ---: | ---: | ---: |
| Fresh observation, 960 rows | 89.3750 | 89.7292 | 60.7500 | 59.7500 |
| Fresh message, 960 rows | 84.6250 | 84.3750 | 41.1667 | 40.6667 |
| All correction, 1,920 rows | 68.5313 | 66.8125 | 3.7500 | 3.0000 |
| Primary correction, 960 rows | 68.7292 | 70.0000 | Same primary quartets | Same primary quartets |
| Same-format secondary I=M1, 960 rows | 68.3333 | 63.6250 | Not a primary quartet metric | Not a primary quartet metric |

Each stratum has balanced target classes, so its row accuracy equals balanced
accuracy. Correction class-0/class-1 recalls are **54.6667/82.3958% restricted**
and **57.0625/76.5625% full**. Full fitting does not uniformly improve either
generalization or class balance.

The complete learning curve also fails to show a consistent correction benefit:

| Training N | Observation quartet %, R/F | Message quartet %, R/F | Correction quartet %, R/F | All-correction row %, R/F |
| --- | ---: | ---: | ---: | ---: |
| 128 | 47.0000 / 51.5833 | 30.9167 / 29.7500 | 4.8333 / 3.5000 | 67.6875 / 67.2292 |
| 256 | 49.5000 / 52.5833 | 37.5000 / 35.8333 | 4.3333 / 4.6667 | 68.4271 / 66.5313 |
| 512 | 60.7500 / 59.7500 | 41.1667 / 40.6667 | 3.7500 / 3.0000 | 68.5313 / 66.8125 |

All 30 feature-kind/N/allocation comparisons completed, all 60 arm calibrations
completed, and no C candidates were excluded for nonconvergence. Every activation
comparison selected C=10, the declared grid's upper endpoint. This limits claims
about optimal regularization; extending the grid after these test outcomes would
be a new exploratory intervention, not the frozen study.

## Why roughly 69% row accuracy can coexist with 3% quartet accuracy

A quartet requires all four matched judgments, including both changing and
preserving evidence. A 75% initial/final-update shortcut can solve three cells
and fail every quartet. A readout's row score therefore does not establish
consistent attribution across the complete contrast.

The predeclared pair metrics show the tradeoff. At N512, full fitting reduces
preserving-pair verdict flips for observation **4.6667→0.7500%**, message
**0.5833→0.1667%**, and correction **14.8333→11.2500%**. However, both endpoints
of genuinely changing pairs are correct only **63.5000→59.9167%**,
**41.7500→40.7500%**, and **27.1111→20.7778%**, respectively. Better stability
on redundant evidence is not improved complete logical reliability.
Each one-step group has eight changing/eight preserving pairs; correction has
24 changing/eight preserving pairs, including its secondary cells.

A further **descriptive regrouping requested after the primary result was
known**, using saved scores and states only, identifies the difficult primary
cell. With B1/B2 denoting accepted first/second receipts, accuracies R/F are
00: **88.3333/91.1667%**, 01: **89.5000/91.4167%**,
10: **20.8333/15.6667%**, and 11: **76.2500/81.7500%**.
Each cell has 240 distinct prefixes evaluated by five heads. The first accepted,
second unaccepted case is particularly weak. This is consistent with a shortcut
failure, but does not identify the model's internal causal mechanism.

## Baselines, ranking and calibration

At N512, TF-IDF correction row accuracy is **50.0000/50.0521% R/F**, with
**0% complete quartets**. Across all three budgets its correction row scores
remain approximately 50% and its quartet scores remain zero. This baseline's
failure is not a strong text-monitor comparison; the capable prose baseline
has not yet run. The structured oracle solves all quartets by construction and
is explicitly labelled as using privileged state/gold information.

The same-model output baseline's correction accuracy is **59.4271% raw** and
**64.3229% development-calibrated**; quartet accuracy is **1.6667→2.9167%**.
Its raw class recalls **91.6667/27.1875%** become **37.7083/90.9375%** after
calibration: development threshold correction transfers imperfectly to this
shift. Compare calibrated competitors, not a trained readout against an
uncorrected output bias.

Monotone calibration does not rescue the activation result: correction quartet
accuracy becomes **4.0000/3.3333% R/F**. Correction Brier score worsens from
**0.2265/0.2362 raw** to **0.2402/0.2494 calibrated**. Macro within-group AUROC
is **0.7734/0.7458** in both saved N512 raw and calibrated summaries. Nontrivial
ranking alongside poor complete decisions is not a reliable monitor claim.

## Coverage and reporting gaps

The frozen [analysis implementation](../src/latent_probes/main_study_analysis.py)
retains the declared budgets/allocations, all groups and failures, row/class
metrics, pair endpoints and flips, oriented margins, AUROC/Brier, primary
quartets, secondary I=M1 metrics and development-calibrated comparators.
The [specification](MAIN_STUDY_SPECIFICATION_V1.md) and observed outputs agree
on those fields. No planned main comparison was silently dropped.

The saved analysis does **not** directly aggregate matched world/name nuisance
flips or primary-only correction row accuracy. Those are reporting gaps, not
missing underlying captures: states and every head's scores are preserved.
The primary row numbers above use equal-sized primary/secondary cells.
A separate descriptive reaggregation of N512 raw scores gives correction verdict
flips under world exchange **19.6875/20.5000% R/F** and global name exchange
**13.3542/11.3750%** (960 matched pairs per allocation; five repetitions).
These gold-preserving transformations expose instability but do not causally
explain the quartet result. They were computed after observing the primary
result and are not additional preregistered discoveries.

## What the remaining runs can resolve

The unchanged Qwen replication will test whether this pattern recurs with a
different tokenizer, model and separately fitted readout. The fixed prose
monitor will establish how a capable text-only evidence extractor performs on
exactly the same accounts, with malformed answers counted incorrect. Neither
result will retroactively validate the semantics as human-reviewed or remove
the shared-renderer, event-count/length and NF4-precision limitations.

Complete those frozen comparisons before deciding the paper's main claim.
Do not add models, operators or fitting choices to seek a positive outcome.
This is evidence about a particular readout pipeline's distribution shift,
not a test of deceptive intent, pressured communication, spontaneous
anticipation or the general absence of second-order belief representations.
The [interruption recovery](MAIN_STUDY_INTERRUPTION_RECOVERY_2026-11-22.md)
and its one unknown-outcome repeated attempt must accompany final reporting.
