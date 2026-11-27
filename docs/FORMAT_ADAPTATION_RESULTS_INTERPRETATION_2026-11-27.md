# Fresh format adaptation: a substantial result with a measurable cost

27 November 2026. **With the same 512-observation fitting budget, including both
formats substantially improves a four-factor belief readout on new contexts in
format B. Adjusting four intercepts using the same observations recovers much
less. The mixed readout also becomes less reliable in format A.** This is a
completed adaptation result, with a secondary result on another checkpoint,
within a constrained fictional belief assay. Human semantic review is incomplete;
the study remains **agent-audited exploratory research**.

The [frozen execution protocol](FORMAT_ADAPTATION_EXECUTION_PROTOCOL_V1.md),
[generated results](results/format-adaptation-2026-11-27/results.md),
[complete numerical summary](results/format-adaptation-2026-11-27/summary.json)
and [figure](results/format-adaptation-2026-11-27/adaptation.pdf) are the evidence
behind this interpretation. All **11,520 cohort measurements** finished across
Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct. Both fitted bundles were sealed
before either model's test measurements. The
[independent audit](results/format-adaptation-2026-11-27/audit_receipt.json)
passed after reconstructing both models before opening either saved analysis.
Its scope includes predictions, declared metrics and bootstrap comparisons;
it does not certify semantics or independently reconstruct full-vocabulary argmax.

## The comparison and its primary result

Each story describes a fictional sender's accepted records about what a reader
read. Four heads predict the initial location I, first report location M1, and
whether the first and replacement reports were recorded as read, B1 and B2.
A supplied rule combines their predictions. The endpoint requires **all four
judgments in a linked reading-record quartet** to be correct. High row accuracy
can conceal failure on these related cases.

Sixteen new fitting contexts provide 512 underlying states. A512 renders all
states in A. Mixed512 renders complementary halves in A and B, without adding
observations. Ahalf256 uses Mixed512's 256 A observations. Offset freezes those
coefficients and learns one additive B intercept per factor from Mixed512's
same 256 B observations. All arms receive the four factor labels. The old
feature map, capture site and regularization policy stay fixed. Neither model
was fine-tuned.

Twenty-four further contexts supply 192 hard quartets per format and checkpoint:

| Model / format | A512 | Mixed512 | Ahalf256 | Offset |
| --- | ---: | ---: | ---: | ---: |
| Llama / A | 185/192 (96.35%) | 166/192 (86.46%) | 178/192 (92.71%) | 178/192 (92.71%) |
| Llama / B | 2/192 (1.04%) | **151/192 (78.65%)** | 7/192 (3.65%) | 20/192 (10.42%) |
| Qwen / A | 162/192 (84.38%) | 131/192 (68.23%) | 145/192 (75.52%) | 145/192 (75.52%) |
| Qwen / B | 3/192 (1.56%) | **125/192 (65.10%)** | 4/192 (2.08%) | 11/192 (5.73%) |

The fixed **Llama primary**, B hard Mixed512 minus A512, is **+77.60 percentage
points**, with a conditional 95% percentile interval **[+68.23, +85.42]**.
The named secondary Qwen comparison is **+63.54 pp [+54.69, +71.35]**.
The named Mixed512-minus-Offset diagnostic is **+68.23 pp [+58.85, +76.04]**
for Llama and **+59.38 pp [+51.04, +67.19]** for Qwen.

These intervals use the prescribed 10,000 paired whole-context resamples.
They condition on the authored bank, fixed fits, sites and checkpoints; they
exclude uncertainty from new fitting cohorts or checkpoints. The models share
24 test contexts, not 48 independent context samples. Only the Llama
Mixed-minus-A512 contrast is primary.

The B improvement is distributed across contexts: each model improves in 23
and ties in one. Failures remain substantial. Llama gets zero of eight B hard
quartets in `adapt_test_19`; Qwen gets one in `adapt_test_11` and two in
`adapt_test_19`. Every context remains in the denominator.

## What changed in the readout

The factor diagnostics pool easy and hard corrections: 1,536 rows per format
and model. Hard cases alone force I to equal 1−M1; pooling permits their
independent variation. The following values are signed, source-oriented AUROC,
without reversing directions or choosing test thresholds:

| B factor | Llama A512 → Mixed512 | Qwen A512 → Mixed512 |
| --- | ---: | ---: |
| Initial location I | .999 → 1.000 | 1.000 → 1.000 |
| Report location M1 | .962 → 1.000 | .992 → 1.000 |
| First reading record B1 | .684 → .996 | .640 → .961 |
| Replacement reading record B2 | .858 → .993 | .696 → .959 |

The largest ordering improvements concern the reading records. An additive
intercept cannot change ordering within B. Offset therefore retains Ahalf256's
B1/B2 AUROCs: .631/.878 for Llama and .623/.696 for Qwen. Offset does improve
some boundaries: Qwen's I/M1 accuracies reach 99.61%/96.09%, but its B1/B2
accuracies remain 58.85%/64.52%. A single intercept per factor is insufficient
for this fitted instrument.

All four factors are simultaneously correct on **28.32% → 91.86%** of Llama B
correction rows and **24.02% → 78.58%** of Qwen rows, comparing A512 with Mixed.
The final B hard class recalls also improve rather than merely exchanging a
dominant answer: Llama's class-0/class-1 recalls change from **88.80%/27.08%**
to **95.05%/94.01%**; Qwen's change from **63.54%/22.14%** to
**88.02%/90.36%**. There are 384 rows of each class in this stratum.

Mixed versus Offset compares two finite-sample fitting procedures. Mixed
jointly refits coefficients and intercepts on A+B; Offset freezes coefficients
estimated from only 256 A observations. Their difference does **not** uniquely
identify representation rotation or prove that every possible offset-based
method must fail. It establishes that this prescribed offset adaptation is
much less effective than this prescribed joint fit.

## The cost and the controls

Mixed loses **9.90 pp** of A hard quartet accuracy for Llama and **16.15 pp**
for Qwen. No A context improves: 13 worsen and 11 tie for Llama; 15 worsen and
nine tie for Qwen. Offset leaves Ahalf256's A predictions unchanged by design.
The result is a measured coverage trade-off, not an invariant readout.

The other conditions make its scope clearer:

| Control: complete quartets, A512 → Mixed512 | Llama A / B | Qwen A / B |
| --- | ---: | ---: |
| Easy correction | 184→173 / 66→169 | 178→144 / 29→128 |
| One-step | 55→41 / 22→5 | 45→52 / 0→34 |

All entries are out of 192. Easy corrections show the same B repair and A cost.
One-step transfer remains weak, especially Llama B: Mixed gets only 5/192,
where Offset gets 47/192. Qwen Mixed reaches 34/192, close to Offset's 31/192.
The four-factor pipeline was trained on correction cases; one-step and asserted
cases are out-of-training-construct controls, not additional factor validation.
On B asserted cases, Mixed gives 178/192 correct rows for Llama and 125/192 for
Qwen; Offset gives 132/192 and 142/192 respectively. The repair does not
uniformly improve every control.

World and name substitutions retain the intended belief judgment, making
verdict changes a reliability problem. Each comparison has 384 pairs. On B hard
cases, Llama world flips fall **113 to 30** and name flips **59 to 26**; Qwen
world flips fall **157 to 55** and name flips **119 to 71**. Llama's both-correct
world/name pair counts rise 166 to 348 and 193 to 350; Qwen's rise 86 to 315 and
105 to 307. Lower flips are therefore accompanied by better answers, although
residual sensitivity remains. A hard world/name flips worsen from 7/7 to 27/27
for Llama and 24/28 to 40/56 for Qwen. The full summary preserves all nuisance controls,
class recalls, quartet histograms and per-context results.

On B hard, the conditional YES/NO output comparator completes only **1/192**
quartets for Llama and **0/192** for Qwen. Strict full-vocabulary first-token
accuracy is **422/768 (54.95%)** and **394/768 (51.30%)**. Llama has 37
noncandidate first choices and 43 argmax ties; Qwen has zero and three. These
remain on the full denominator. Conditional Brier scores are .230/.447 and
log losses .651/4.488. They describe a two-token output distribution, not
calibrated probabilities for the deterministic four-factor fold. Extra factor
labels and supplied computation prevent a matched-supervision capability claim
against output elicitation. No free generation was measured.

## What this contributes and what comes next

This strengthens the supervisor's Variant 2 direction: an explicit format shift
breaks a latent readout, and a prospectively fixed adaptation comparison measures
what recovers and what deteriorates. It preserves the second-order **attributed
reader-belief** target. It does not demonstrate spontaneous anticipation before
a question, real reader beliefs, intentional deception, or the model's internal
use of the supplied update rule.

The [literature review](FORMAT_TRANSFER_RESEARCH_REVIEW_2026-11-26.md) records
the substantive prior overlap. [Kumar v1](https://arxiv.org/html/2605.27958v1) already
tests style augmentation; [Bortoletto et al.](https://arxiv.org/abs/2406.17513v3)
already show prompt-sensitive belief probes; [Bürger et al.](https://arxiv.org/abs/2407.12831v2)
already distinguish bias from direction transfer. The bounded contribution here
is the equal-observation, fixed-map comparison of joint and intercept-only
adaptation on fresh linked correction quartets, with a second checkpoint and
explicit costs. This finite literature search establishes no firstness or
publication guarantee.

The [23 November primary](FRESH_READOUT_RESULTS_INTERPRETATION_2026-11-23.md)
remains a null finding: source-only site selection had no established benefit.
This new intervention does not replace that endpoint. The new contexts also
do not justify directly pooling its percentages with this study.

Important limits remain: one agent author pool; shared formal rules and format
templates; just 16 fit and 24 test contexts across eight domains; B already
present during fitting; old shared preprocessing; fixed NF4 checkpoints; and
incomplete human review. Within each fitting context, format depends on world,
despite balanced pooled allocation. Fully crossed evaluation helps expose that
dependency's consequences; it does not remove the training limitation. A/B also
bundles system, narrative/order and question changes. No unseen-format,
independent-author or isolated-component claim follows.

The next step is to consolidate this result. Complete human review of chronology
and attribution, recording disagreements without silently excluding failures.
If changes are needed, version the bank and preserve these results. Then freeze
a BF16 replication protocol before obtaining new measurements: retain every arm,
endpoint and control, prospectively specify hardware and numerical checks, and
report deviations regardless of outcome. That is a concrete CREATE use case
to test whether the finding survives a different precision regime. It is a
proposal, not completed replication. A third renderer or natural communication
study should answer a separately specified question after this core study is
written up, rather than become a search for a still larger headline.
