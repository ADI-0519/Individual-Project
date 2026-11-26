# POSTHOC format score-shift diagnostic receipt

26 November 2026. **The A/B failure combines score-boundary mismatch with loss
of target ordering.** Content-factor ordering largely survives; receipt-factor
ordering degrades substantially, especially for Qwen. These exposed-bank
diagnostics do not identify a universal repair or establish information absence.

The [original fixed specification](../../FORMAT_SCORE_SHIFT_DIAGNOSTIC_SPEC_V1.md)
was hashed/sealed before raw margin arrays were opened. Its
[unchanged v1 outputs](./20261126T232649Z_90868d74/receipt.md) cover both models,
all 4,992 saved rows/model, all 24 contexts, easy/hard corrections, one-step and
asserted controls. Exact row-ID indexing and full formal-state A/B joins found
2,496 pairs/model; completed-analysis hashes, symbolic targets, frozen threshold
predictions, and the unchanged factor fold all agree. No model calls, feature
loads, fitting, threshold search, p-values, or new confidence intervals occurred.

An independent design critic identified that hard cases enforce I=1-M1 and easy
cases enforce I=M1. A [separately sealed pooling addendum](./20261126T233413Z_76bd60ed_correction_pool/specification_addendum.md)
was therefore written **after v1 stratum outcomes were seen and before pooled
outcomes were computed**. Its [new supplement](./20261126T233413Z_76bd60ed_correction_pool/receipt.md)
combines every easy/hard correction state: 1,536 rows/renderer/model and all 24
contexts, allowing independent variation of I/M1. This is the principal
descriptive population for interpreting factor accessibility; the original
study primary endpoint remains unchanged.

| Model | Frozen factor | Raw AUROC A → B | Fixed-zero accuracy A → B | Within-context AUROC A → B |
| --- | --- | --- | --- | --- |
| Llama | I: initial location | 1.0000 → 0.9866 | 1.0000 → 0.8503 | 1.0000 → 0.9907 |
| Llama | M1: first-message location | 1.0000 → 0.9902 | 0.9993 → 0.8815 | 1.0000 → 0.9947 |
| Llama | B1: first-message receipt | 0.9981 → 0.6956 | 0.9837 → 0.6263 | 0.9998 → 0.7191 |
| Llama | B2: correction receipt | 0.9987 → 0.9144 | 0.9798 → 0.7754 | 0.9998 → 0.9457 |
| Qwen | I: initial location | 1.0000 → 0.9994 | 1.0000 → 0.9271 | 1.0000 → 0.9998 |
| Qwen | M1: first-message location | 1.0000 → 0.9897 | 0.9993 → 0.8158 | 1.0000 → 0.9969 |
| Qwen | B1: first-message receipt | 0.9838 → 0.6698 | 0.9284 → 0.5026 | 0.9960 → 0.7109 |
| Qwen | B2: correction receipt | 0.9724 → 0.6771 | 0.8809 → 0.5801 | 0.9924 → 0.7122 |

AUROC uses the original target-1 orientation, never absolute/inverted AUROC.
Within-context AUROC enumerates all positive-negative comparisons separately
inside each context, then aggregates counts. The same-state A/B order-retention
fractions for I/M1/B1/B2 are 0.9907/0.9947/0.7193/0.9456 for Llama and
0.9998/0.9969/0.7135/0.7143 for Qwen. Order retention includes jointly wrong
orders and therefore must be read alongside target AUROC.

Class-conditioned score changes show why one common additive offset cannot
fully describe the receipt failures. For pooled Qwen B1, target-0 mean margin
moves from -4.0410 to +2.5287, while target-1 mean moves from +3.4737 to +3.3315:
A-to-B shifts of +6.5697 and -0.1421. Target-0/target-1 recalls in B are
0.0234/0.9818. The separation contracts and many positive-negative orders reverse.
The remaining AUROC above 0.5 also prevents an information-absence claim.

The asserted control supplies a different pattern: Qwen S1 B predicts 0 on all
192 rows, giving 50% accuracy, while raw AUROC remains 0.9936. Both class means
are below zero (-3.3950 and -0.9704); 379/384 within-context class orders remain
correct. This supports a boundary/score-shift description for that particular
control. It does not explain the receipt-factor ranking losses. For the hard
verdict task, S1 raw AUROC changes 0.7893→0.7575 (Llama) and 0.7394→0.7464
(Qwen); C1 changes 0.8638→0.7785 and 0.8157→0.7387. Both boundary and ranking
effects depend on the head and condition.

Joint correctness of all four factors falls from 0.9635 to 0.3724 for Llama and
0.8158 to 0.2174 for Qwen in pooled correction rows. The unchanged C2 fold's row
accuracy falls 0.9798→0.6921 and 0.8971→0.5990. These row diagnostics do not
replace the original complete-quartet endpoint. Low nuisance flip rates can
also reflect saturation: Qwen B1 B world/name flip fractions are only
0.0260/0.0365 while it predicts 1 on 97.92% of pooled correction rows. Those
figures alone do not demonstrate useful invariance.

Every-context metrics, class distributions, A/B score deltas and nuisance
margin gaps remain available in both reports' JSON and CSV tables. No
contexts, factors, or capture sites were selected from outcomes. The bundled
renderer shift does not isolate a causal wording feature or establish what
computation the model performs internally; no retrospective threshold repair
is proposed on this bank.

The independent v1 arithmetic check verified 2,400 score scopes against
scikit-learn AUROC (maximum difference 1.11e-16), 1,200 paired-score scopes
against SciPy/NumPy correlations, 1,152 within-context order scopes by direct
positive-negative enumeration, and 200 joint-factor scopes. See
[verification.json](./20261126T232649Z_90868d74/verification.json). Eight focused
implementation tests cover orientation/ties/zero, nonfinite targets/scores,
permuted and invalid joins, order transitions, nuisance pairing, fold truth
table, and overwrite refusal; Ruff passes. Pylance/environment tools were absent
from the available catalog, so the supplied Python 3.11.9 runtime was verified
directly. Original source bytes and original result hashes are preserved.
