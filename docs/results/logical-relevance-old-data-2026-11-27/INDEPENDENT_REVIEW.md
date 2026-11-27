# Independent review of the old-data logical-relevance diagnostic

27 November 2026. This review concerns only the completed 23 November campaign
and this separately specified old-data analysis. No fresh format-adaptation
features, fitted bundles, analyses or outcomes were opened. No bound source,
test, protocol or author output was changed, and no model call or fit was run.

The numerical results pass a complete, separately implemented reconstruction
within the scope below. They support an external-readout error analysis. They
do not identify whether the language model internally gates or routes belief
information, and the oracle results cannot be interpreted as additive causal
contributions of four model components.

## Independent arithmetic and source coverage

The reviewer read the complete [analysis script](../../../scripts/analyse_logical_relevance_old.py),
[synthetic tests](../../../tests/test_logical_relevance_old.py),
[scope specification](../../LOGICAL_RELEVANCE_OLD_DATA_SPEC_2026-11-27.md),
and [result report](README.md). The report version initially reviewed has SHA256
`744d784cd839a3bca3eb870f0e0af5e61503044cc8d74c6e24bac221d2fa6a7c`.
The saved factor order was checked against the old readout implementation:
I, M1, B1, B2; the new report's U is an explicit alias for M1.

The [separate checker](../../validation/logical_relevance_old_independent_check_2026-11-27.py)
imports no production analysis functions. It reconstructs relevance from closed
forms, oracle outputs from an independently constructed event-sequence lookup
table, and AUC using sklearn rather than the author's rank implementation.
It joins the saved old margins by exact row IDs and checks all old input and
author source/output hashes. Its [receipt](independent_review_receipt.json)
records **77,600 numeric scalar comparisons and 10,400 explicit null cells**;
the maximum absolute difference is **2.22e-16**. All 2,400 factor rows, 1,200
population-comparison rows, and 4,800 oracle rows were covered, together with
the summary's defined-context counts and mean within-context differences.

This independently checks arithmetic on the saved margins. It does not
reconstruct those margins from old hidden-state features, provide new semantic
acceptance of the narratives, or independently establish the author's actual
first-access time. The specification seal records 01:31:42 UTC and the analysis
receipt records first loading at 01:37:47 UTC; the current specification hash
matches the sealed hash. That bounded provenance claim is appropriate for a
new diagnostic on an already exposed old bank, not fresh confirmation.

## Truth-table review

Sequentially retaining I, replacing it with U when B1 is true, then replacing
it with 1-U when B2 is true gives the stated fold. Its Boolean sensitivity to
one true factor, with the other true factors fixed, is:

| Factor | Exactly relevant when |
| --- | --- |
| I | B1=0 and B2=0 |
| U | B1=1 or B2=1 |
| B1 | B2=0 and I differs from U |
| B2 | B1=1 or I equals U |

These conditions match the author's exhaustive intervention calculation. The
expected per-context populations are also exactly reproduced:

| Population | I relevant / irrelevant | U relevant / irrelevant | B1 relevant / irrelevant | B2 relevant / irrelevant |
| --- | ---: | ---: | ---: | ---: |
| Easy plus hard, 64 rows | 16 / 48 | 48 / 16 | 16 / 48 | 48 / 16 |
| Easy, 32 rows | 8 / 24 | 24 / 8 | 0 / 32 | 32 / 0 |
| Hard, 32 rows | 8 / 24 | 24 / 8 | 16 / 16 | 16 / 16 |

Multiply by 24 for a pooled model/renderer denominator. Every nonempty
population has exactly balanced factor labels, including within each context.
There is no class-imbalance explanation for its accuracy differences. The easy
B1-relevant and easy B2-irrelevant cells are mathematically empty and correctly
have null AUC, accuracy and recalls with reasons. General one-class handling is
implemented and tested; it is not a populated exception in this complete bank.

The union is the appropriate principal factor population. Within hard cases,
I=1-U, and within easy cases I=U. Separate high I/U scores within either
stratum therefore cannot establish their independent representation. The
union varies both bits independently while preserving both separate tables.

## Joins, denominators and complete reporting

Each model's 4,992 saved test rows is validated before selecting all 3,072
correction rows across A/B. Each model/renderer contributes 1,536 union rows
and 384 receipt quartets; either stratum contributes 768 rows and 192 quartets.
Each quartet fixes context, renderer, I, U, world and name mapping, and includes
each B1/B2 cell exactly once. All 24 contexts, four factor heads, both relevance
populations, both renderers, both models, three declared strata and masks
0 through 15 occur in the exported tables. Neither the zero threshold nor the
raw score orientation is selected from these results. One-step and asserted
controls are explicitly excluded from this diagnostic.

The exact ID join allows prediction order to differ from capture order and
rejects duplicates or omissions. Full factorial validation prevents a valid
total row count from concealing missing cells. The receipt binds the old
predictions and input metadata, and the source checks the unchanged external
fold against saved C2 verdicts. No pairing or denominator defect was found.

Pooled AUC compares positives and negatives across contexts, whereas the mean
within-context AUC difference averages only context-local comparisons. They
can differ in sign without an arithmetic error. The report correctly retains
this distinction for Qwen B1. Neither statistic has an accompanying
population-general inferential guarantee, and the repeated factorial rows are
not thousands of independent narratives.

## Interpretation issues that must remain explicit

**True relevance is a privileged partition, not an intervention on the model.**
The relevant and irrelevant populations differ in other formal factors by
definition; they are not randomized versions of the same text. Their AUC or
accuracy difference cannot isolate a causal effect of computational relevance,
nor show that the model stores a factor only when needed. The observed receipt
failure also persists on relevant rows, which defeats the simple explanation
that all degradation came from a gold-irrelevant population. It does not prove
the loss of an internal receipt representation.

**True irrelevance and oracle influence are different questions.** For example,
take truth I=U=0, B1=B2=0. B1 is truly irrelevant. Predictions I=1, U=0, B1=1,
B2=0 nevertheless yield the correct final zero. Replacing B1 alone with its
true zero makes the external prediction one and hence wrong. Thus a truth-
irrelevant factor can change the oracle output when other predicted factors
are wrong. This explains why easy B1 oracle changes are not a truth-table
contradiction or evidence of internal routing.

**Oracle gains are nonmonotone and interact.** The independent check finds ten
pooled mask cells whose row or quartet outcome is below mask 0. The report's
Qwen B-hard example is correct: supplying B2 alone increases row accuracy but
reduces complete quartets from 12/192 to 3/192. Error cancellation, changed
gates and the four-row all-correct endpoint prevent adding single-factor gains
or ranking factors as unique causes of failure. Supplying both true receipts
can help the supplied algorithm substantially while leaving content-readout
errors; this is conditional oracle sufficiency for that algorithm, not a
deployable repair. Mask 15's perfect result follows from inserting the same
formal truth used to define gold and is a consistency check, not evidence of
new information extraction or model competence.

## Primary-paper comparison

The reviewer independently checked [Prakash et al. v3](https://arxiv.org/html/2505.14685v3),
Sections 3.1, 3.3 and 5.1: the 80 behaviorally correct cases, internal
interchange interventions and donor-pointer versus retrieved-value distinction
support the report's contrast. This old-data score substitution does not
perform their causal experiment. No figure curves were independently digitized.

The reviewer checked [Steele et al. v2](https://arxiv.org/pdf/2607.11945v2),
Section 4.4 and Section 7: frame decoding can coexist with a frame-agnostic
value-controlling subspace; recipient-versus-donor value controls and clean-pass
projection ablations provide evidence absent from this diagnostic. Their
preservation and sample-sensitivity limitations remain material. This targeted
reading supports the report's bounded comparison, not a claim that its
mechanism has been replicated here or that the two papers share an identified
circuit.

No blocking numerical or scope defect was found in the inspected diagnostic.
Its useful result is a more detailed account of old external-readout errors
under a known Boolean rule. The report's retained full tables, nonmonotone
counterexample, undefined-cell handling and explicit causal limits are
necessary to that interpretation.
