# Matched lexical shortcut controls: audited exposed-bank diagnostic

This posthoc lexical shortcut control uses the same 16 fitting and 24 test contexts as the completed latent study. It is one shared text comparator, not an independent replication per checkpoint. Human semantic review remains incomplete. The original latent primary and earlier null results remain unchanged.

Each arm predicts four labelled factors and uses the same supplied external update rule. Every lexical vocabulary and IDF is fitted on that arm's authorized observations only. Mixed fitting changes lexical features and coefficients. Offset preserves Ahalf256's feature map and coefficients and learns only four B intercepts.

| Instrument / format | A512 hard quartets | Mixed512 | Ahalf256 | Offset |
| --- | ---: | ---: | ---: | ---: |
| Text full_messages / A | 0/192 (0.00%) | 0/192 (0.00%) | 0/192 (0.00%) | 0/192 (0.00%) |
| Text full_messages / B | 1/192 (0.52%) | 56/192 (29.17%) | 1/192 (0.52%) | 0/192 (0.00%) |
| Text system_only / A | 0/192 (0.00%) | 0/192 (0.00%) | 0/192 (0.00%) | 0/192 (0.00%) |
| Text system_only / B | 0/192 (0.00%) | 0/192 (0.00%) | 0/192 (0.00%) | 0/192 (0.00%) |
| Previous latent llama / A | 185/192 (96.35%) | 166/192 (86.46%) | 178/192 (92.71%) | 178/192 (92.71%) |
| Previous latent llama / B | 2/192 (1.04%) | 151/192 (78.65%) | 7/192 (3.65%) | 20/192 (10.42%) |
| Previous latent qwen / A | 162/192 (84.38%) | 131/192 (68.23%) | 145/192 (75.52%) | 145/192 (75.52%) |
| Previous latent qwen / B | 3/192 (1.56%) | 125/192 (65.10%) | 4/192 (2.08%) | 11/192 (5.73%) |

Complete-quartet counts must be read alongside row accuracy, class recalls and nuisance sensitivity. A larger quartet count alone does not establish recovery of the attributed-belief target. The format-B hard controls for every lexical arm follow.

| Text view | Arm | B-hard correct rows | Recall 0 | Recall 1 | World flips / pairs | Name-map flips / pairs | Both correct after name-map change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_messages | A512 | 387/768 (50.39%) | 89.06% | 11.72% | 17/384 | 39/384 | 174/384 |
| full_messages | Mixed512 | 399/768 (51.95%) | 38.02% | 65.89% | 9/384 | 219/384 | 90/384 |
| full_messages | Ahalf256 | 379/768 (49.35%) | 28.65% | 70.05% | 39/384 | 89/384 | 145/384 |
| full_messages | Offset | 381/768 (49.61%) | 5.99% | 93.23% | 7/384 | 21/384 | 180/384 |
| system_only | A512 | 384/768 (50.00%) | 100.00% | 0.00% | 0/384 | 0/384 | 192/384 |
| system_only | Mixed512 | 384/768 (50.00%) | 100.00% | 0.00% | 0/384 | 0/384 | 192/384 |
| system_only | Ahalf256 | 384/768 (50.00%) | 100.00% | 0.00% | 0/384 | 0/384 | 192/384 |
| system_only | Offset | 384/768 (50.00%) | 100.00% | 0.00% | 0/384 | 0/384 | 192/384 |

The two receipt-factor AUROCs below pool correction-easy and correction-hard rows in format B. Full per-stratum, class, context and nuisance metrics are retained.

| Text view | Arm | B1 AUROC | B2 AUROC |
| --- | --- | ---: | ---: |
| full_messages | A512 | 0.9146 | 0.8321 |
| full_messages | Mixed512 | 1.0000 | 1.0000 |
| full_messages | Ahalf256 | 0.9147 | 0.8321 |
| full_messages | Offset | 0.9147 | 0.8321 |
| system_only | A512 | 0.5000 | 0.5000 |
| system_only | Mixed512 | 0.5000 | 0.5000 |
| system_only | Ahalf256 | 0.5000 | 0.5000 |
| system_only | Offset | 0.5000 | 0.5000 |

The full-messages baseline sees roles and all supplied prose, including the final answer-format instruction; the latent site precedes that instruction. These are not identical causal token prefixes. The lexical baseline omits chat special tokens. System-only discards user content and is an instruction/template negative control.

N-grams are a shallow text control, not a capable semantic judge. Failure of this baseline cannot establish incremental activation value, model-internal composition, spontaneous belief attribution or natural deception detection. Equal numerical C does not equalize lexical and latent feature geometry or capacity.

## Complete evidence

[metrics.json](metrics.json) retains all arms, formats, conditions, factor strata and fixed descriptive comparisons. [summary.json.gz](summary.json.gz) also retains all 24 context outcomes. [rows.json.gz](rows.json.gz) and [predictions.npz](predictions.npz) preserve formal test metadata and saved margins. These are exact original output bytes, not new fits or predictions.

The two B-hard comparisons use the original paired whole-context bootstrap policy. Their intervals condition on the exposed bank and fixed fitted instruments and exclude fitting-cohort uncertainty. They are posthoc descriptive comparisons, with no new primary or multiplicity guarantee. One-step and asserted conditions are out-of-training-construct controls; factor truth is evaluated on correction only.

[audit/audit_receipt.json](audit/audit_receipt.json) verifies independently reconstructed metrics from saved margins. Both reconstruction JSON files are losslessly compressed here; their decompressed hashes match the original [reconstruction seal](audit/reconstruction_seal.json). This is not an independent refit, feature reconstruction or human semantic certification.

## Restore the fitted text instruments

The trusted [fitted-text-readouts.zip](fitted-text-readouts.zip) contains exactly six fixed-path JSON/NPZ files. No pickle or executable files are included. Extract into a new directory from this report folder:

```console
python -m zipfile -e fitted-text-readouts.zip restored
```

From the locked project environment, use `load_text('restored/fit/full_messages')` or `load_text('restored/fit/system_only')` from `latent_probes.format_adaptation_text`. Score records must contain exactly `messages` and `renderer`; no labels or formal-state metadata enter the text map. The archive members are bound to [fit_seal.json](fit_seal.json), and every exported file is bound to [manifest.json](manifest.json). CREATE BF16 replication and a capable text baseline remain separate follow-ups.
