# Research update for Hanqi: belief-readout reliability under format shift

27 November 2026. Prepared briefing; no message has been sent. The current
study measures **a fictional sender's attributed belief about what a reader
believes after reading reports**, following an explicit question. It now has a
substantial two-checkpoint adaptation result with retained failures and source-
format costs. It remains agent-audited exploratory research: human semantic
review is incomplete. Publication novelty is a question to assess against prior
work, not a conclusion from the size of the effect.

## What the study asks, and which answers have changed

| Research question | Evidence and present interpretation |
| --- | --- |
| Does broader one-step training support transfer to unfamiliar corrections? | The original matched-budget experiment found no established improvement in either checkpoint. Llama's N512 difference was -0.75 percentage points; Qwen's was -0.83. Those original null results remain unchanged. |
| Can source-only capture-site selection fix correction transfer? | A later fresh study found no established improvement. This does not establish that belief information is absent everywhere. |
| Can supervised ingredient readouts support correction judgments? | Four heads can recover useful ingredients after an explicit question. A supplied external rule combines them. This establishes an instrument's predictive performance, not that the model internally uses that rule. |
| How robust is that instrument to a second prompt format, and can it be adapted? | A source-only instrument is highly brittle. Equal-budget mixed-format fitting substantially repairs format B on new contexts, with a measurable cost in A. This is the principal positive result below. |
| Do these activations predict spontaneous reader anticipation or natural deceptive omissions? | Neither has been established. The present explicit-question, fictional-record target does not qualify those stronger claims. Earlier natural communication controls were insufficient for a validated omission/deception result. |

The original [coverage interpretation](MAIN_STUDY_FINAL_INTERPRETATION_2026-11-22.md),
[fresh site-selection interpretation](FRESH_READOUT_RESULTS_INTERPRETATION_2026-11-23.md)
and [query comparison](QUERY_COMPARISON_RESULTS.md) preserve these distinct
questions. The newer positive does not reverse an earlier null: it fits on
correction examples and includes B during training, so it does not demonstrate
unseen-rule or unseen-format transfer.

## Completed adaptation finding

The model weights stayed frozen. Sixteen fitting contexts supply 512 underlying
states. A512 trains on their A renderings; Mixed512 trains on complementary
halves in A and B. Both receive four factor labels and the same supplied rule.
Twenty-four further contexts provide 192 hard reading-record quartets per
format/checkpoint. A quartet succeeds only when all four linked judgments are
correct, making inconsistent partial success visible.

| Checkpoint / format | A512 | Mixed512 | Offset comparator |
| --- | ---: | ---: | ---: |
| Llama-3.1-8B / A | 185/192 (96.35%) | 166/192 (86.46%) | 178/192 (92.71%) |
| Llama-3.1-8B / B | 2/192 (1.04%) | 151/192 (78.65%) | 20/192 (10.42%) |
| Qwen2.5-7B / A | 162/192 (84.38%) | 131/192 (68.23%) | 145/192 (75.52%) |
| Qwen2.5-7B / B | 3/192 (1.56%) | 125/192 (65.10%) | 11/192 (5.73%) |

The fixed Llama primary is **+77.60 pp**, with conditional 95% interval
**[+68.23, +85.42]**. The named Qwen secondary is **+63.54 pp
[+54.69, +71.35]**. Each improves B in 23 of 24 contexts. A costs are 9.90 pp
and 16.15 pp respectively; one-step controls remain weak.

Offset freezes a head fitted on Mixed's 256 A observations and learns four
B intercepts from Mixed's same 256 B observations. The full report also retains
that Ahalf256 arm. Mixed's much larger repair does not uniquely establish a
rotation or failure of all calibration methods: the finite-sample procedures
differ in joint coefficient fitting versus freezing a half-budget direction.
The [complete interpretation](FORMAT_ADAPTATION_RESULTS_INTERPRETATION_2026-11-27.md)
records all controls, confidence-interval conditions and context failures.

All 11,520 cohort forwards finished. Both fitted bundles were sealed before
either checkpoint's test capture. Independent numerical reconstruction checked
predictions and declared metrics. Its scope does not certify semantic validity
or an internal belief mechanism. The checkpoints share the same 24 test contexts;
they do not supply 48 independent authored test units.

## Completed lexical control: receipt extraction and judgment reliability differ

The new word/character TF-IDF control gets the same fitting states, four labels
and supplied rule. Its vocabulary/IDF use training rows only; all arms/views were
sealed before scoring. It required **zero new language-model/GPU calls**.
Independent saved-score reconstruction passed for both views with zero numerical
discrepancy. This is a predeclared **post-outcome diagnostic on an exposed bank**,
not a new primary or fresh confirmation.

| Full-message text arm | A hard quartets | B hard quartets |
| --- | ---: | ---: |
| A512 | 0/192 | 1/192 |
| Mixed512 | 0/192 | 56/192 (29.17%) |
| Ahalf256 | 0/192 | 1/192 |
| Offset | 0/192 | 0/192 |

The text B-hard Mixed-minus-A512 improvement is **+28.65 pp [+24.48, +32.81]**
under the same conditional whole-context bootstrap. Mixed perfectly classifies
both reading-record factors in B across 1,536 correction rows, while initial/
first-report location accuracies are 50.00%/50.46%; the initial-location head
predicts one on every pooled correction row. B-hard row accuracy is only
399/768 (51.95%), with class recalls 38.02%/65.89%. Name substitutions flip
219/384 judgments; the nine world flips also coexist with only 195/384
both-correct pairs. Mixed completes zero one-step quartets in either format.
The 56 complete B-hard quartets therefore accompany substantial unreliability.
System-only inputs complete zero hard quartets in either format in every arm;
their asserted-control accuracy is 96/192. These controls do not eliminate every
possible shortcut.

The [complete lexical results](results/lexical-controls-2026-11-27/README.md)
retain all conditions; the [comparison figure](results/lexical-controls-2026-11-27/lexical-comparison.pdf)
shows both formats and the four factor readouts. The shared text comparator is one experiment, not two
model replications. Its B Mixed score is numerically below the latent 151/192
and 125/192, but a weak shallow comparator cannot establish unique activation
information or superiority over a capable neural text monitor. Its per-arm text
maps also differ from the latent pipeline's shared historical map.

## Research contribution and next decision

Prior work already establishes [belief-probe prompt brittleness](https://arxiv.org/html/2406.17513v3),
[style-augmented probe training](https://arxiv.org/html/2605.27958v1) and the
[distinction between score-boundary and direction transfer](https://arxiv.org/html/2407.12831v2).
The bounded contribution is the fixed-site, equal-observation adaptation
comparison on linked correction judgments, including the prescribed intercept
control, a second checkpoint, source-format costs and complete failures. We
should frame the paper around reliability of this measured instrument and
discuss whether that controlled contribution is sufficient, rather than claim
second-order belief itself is new. The
[literature review](LEXICAL_CONTROL_RESEARCH_REVIEW_2026-11-27.md) records the
new control-method and quantization readings, inspected sections and limits.

The next substantive qualification is human review of attribution and chronology.
For a stronger paper, the next controls should also qualify a capable prose-only
extractor of the same four factors, using the same supplied fold. Its development
controls must test attribution and chronology; freeze its prompt, parser and
compute budget before fresh evaluation, retaining invalids in the denominator.
Matching target labels does not equalize computational capacity.

Then freeze a confirmation on genuinely fresh, human-reviewed contexts, crossed
under A/B and a proposed renderer C withheld from probe fitting. Crossing the
same new contexts under all renderers separates context difficulty from format.
Specify that endpoint before collecting outcomes. Neither another renderer nor
known style augmentation is automatically novel; the paper argument still rests
on the exact controlled comparison and its limitations.

The [CREATE proposal](CREATE_ADAPTATION_REPLICATION_PLAN_V1.md) then calls for a
paired technical NF4/BF16 replication on the same GPU type, with profile-specific
training-only preprocessing and both profiles' fits sealed before evaluation.
The old bank stays labeled reused; a genuinely fresh confirmation needs its own
versioned bank, review and protocol. The precision pipeline has not been
implemented or run. Its proposed full inventory is 25,088 reference/cohort
forwards across two checkpoints and two profiles, with wall time estimated only
after engineering profiling.

That existing-bank precision replay is a useful numerical robustness check; it
does not replace the qualified text comparison or fresh semantic confirmation,
and is not a novelty milestone by itself.

A 40 GB A100 is justified by BF16 weights plus activation/runtime memory, and
by the need for a same-hardware NF4 reference. King's
[student-access rules](https://docs.er.kcl.ac.uk/CREATE/requesting_access/) require
a staff/PI request and, for taught students, contribution to an existing King's
research project. The practical discussion with Hanqi/supervisor is therefore
the bounded core claim, semantic-review support and eligible project sponsorship.
Neither CREATE use nor these results guarantees publication, marks or admission.
