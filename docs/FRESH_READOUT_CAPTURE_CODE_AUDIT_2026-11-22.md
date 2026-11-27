# Fresh readout capture: independent code audit, repairs pending verification

22 November 2026. This agent independently inspected the new
[capture module](../src/latent_probes/fresh_readout_capture.py),
[CLI](../scripts/run_fresh_readout_capture.py) and
[capture tests](../tests/test_fresh_readout_capture.py), plus the bank API and
existing backend/capture contracts. It did not inspect authored fresh test
contexts or their gold labels. This is separate from the
[pre-authoring design audit](FRESH_READOUT_DESIGN_AUDIT_V2_2026-11-22.md).

**Status: repairs are being authored by the capture owner; final-code verification
is PENDING.** Critical findings were sent directly to that owner and the execution
author before native capture. The owner acknowledged the findings and is
implementing the changes below. This document does not certify those changes
or authorize a model campaign.

The initial capture suite passed **25 tests** in the Windows Python 3.11
environment. Its forward fixtures emulate tensor operations with NumPy; those
passes do not verify actual torch/HF hooks or real BPE offsets. Torch,
Transformers and Tokenizers are absent from that Windows environment. The
capture owner is separately checking a tiny randomly initialized HF fixture and
already cached tokenizers in the existing WSL environment; this reviewer did
not duplicate those checks. No native study model call, classifier fit, package
installation or frozen-file edit was performed by this reviewer.

## Critical findings and required repairs

All repair statuses below remain **PENDING independent verification**, even when
the owner has reported implementation progress.

| Finding in the initial inspected code | Required repair and verification |
| --- | --- |
| Default preparation imports `load_materials`, while the fresh bank API exports `load_bank`. | Use the actual bank adapter and run a non-fixture preparation integration check. Do not let tests exercise only generic fixture rows. |
| Native preparation accepts arbitrary JSON containing `rows`, or an externally supplied row iterator, while checking only partition counts. That bypasses validated bank grouping, state and source-ID contracts. | Restrict generic row overrides to explicit fixtures. Native preparation must load/validate the genuine fresh bank, derive its exact rows, and bind all nested source/authored inputs. |
| Initial source bindings omit the new bank renderer and its original-bank/allocation dependencies. | Automatically bind the new bank module, original bank/design modules and all existing capture dependencies, plus the bank's source and authored input bindings. Recheck them throughout execution. |
| Test authorization accepts any existing JSON plus NPZ hashes, and the initial fixture explicitly passes dummy heads. | Before any test engineering or cohort forward, validate genuine **both-model** fit bundles, not merely extensions. Require the agreed `fresh-readout-fit-v2` schema/status/fixture identity; exact bank and respective capture-protocol/model identities; all four exact fitting/development ID lists; all 12 frozen source-site heads and C choices; S0/S1/C1/C2 identity; selected site and safe finite numeric arrays. Verify both settings files and both head archives, and their hashes. A missing, mismatched or partial second model must block the first model's test forward. |
| A nonempty map of review files is accepted without checking that it represents the reviewed bank or completed semantic coverage. | Require the genuine `fresh-readout-agent-semantic-review-v1` result bound to the exact bank and packet, all 48 contexts, declared prompt count and exact opaque-ID set, consistent reviewed answers after the separate key comparison, plus the bound review ledger/receipt. Preserve explicit agent review and `human_review_complete=false`; introduce no human gate. Reject placeholder review hashes. |
| Row-directory exclusivity does not protect alias execution. Overlap checks ignore another alias directory until its manifest exists. Concurrent source/source_fit, or future training/individual calls, can duplicate rows in separate journals. | Add an exclusive prepared-output-wide execution lock before inspecting or mutating any partition/alias. Concurrent calls must fail before backend construction. A stale lock after an unknown process death must block automatic retries. Test alias overlap and exclusive-lock behavior. |
| Protocol bytes are rechecked before cohort rows, but initially not immediately before the engineering forward or after final export. | Verify protocol bytes, sources and authorization immediately before engineering and each cohort forward, and after export before completion. A change during weight loading must block the first forward. |
| Backend identity initially relies mostly on metadata and does not compare every actual config dimension. | Check actual config model type, hidden size, vocabulary, layer count and context limit, as well as resolved revision, tokenizer, configured capture layers, quantization/dtype and installed forward identity. Test mismatches before engineering. |

For efficient authorized execution, the owner is also adding one `training`
alias covering source_fit, source_dev, correction_fit and correction_dev: 2,048
rows/model in one backend lifetime. Preserve each row's original partition so
the fit loader can enforce the four exact independent supports. Alias overlap
rules must prevent an individual partition from recapturing those rows.

## Findings supported by the initial implementation

The offset approach uses the **complete rendered chat prefix**, then verifies
its token IDs equal direct chat-template tokenization. This avoids deriving
positions from a separately tokenized account/question whose BPE boundaries may
differ. It searches once for the entire renderer-owned user message, rather
than searching globally for repeated location names.

The account/query spans are explicit end-exclusive character spans. Every
overlapping token must remain within the right span boundary. The query may
explicitly allow left overlap containing whitespace only, accommodating a BPE
token that combines a leading space and the location. A right crossing still
fails. The bank's account span includes the paragraph separator immediately
before the question; strict right-boundary handling therefore retains separator
tokens without accepting later question characters. Real cached-tokenizer
preflight must still verify this convention on all prepared prefixes.

The selected positions are causally ordered, and all 12 features come from
three positions in each of four block hooks during one forward. Cohort forwards
do not request all HF hidden states. The excluded engineering example requests
them and compares each hook to index `block+1` at the same position. Native
actual-HF verification remains pending; NumPy fixtures alone cannot establish
that behavior.

The initial cohort storage is already bounded correctly: both
`raw_forward.npz` and the completed row archive retain features and the **two
YES/NO candidate logits**, rather than a full vocabulary vector per cohort row.
Full vocabulary arrays are retained only for the excluded engineering example.
Output summaries preserve vocabulary normalization, argmax and entropy
diagnostics. This resolves the execution author's concern that raw row archives
were silently retaining complete cohort logits.

Safe NPZ reads use `allow_pickle=False`. Completed rows bind planned fields,
protocol bytes, started/complete records, array hashes, candidate-token IDs,
candidate logits, summary hash and site token indices. Existing incomplete or
failed attempted rows block resumption. A completed partition cannot be rerun;
explicit resume can skip already completed rows without repeating their model
forward. These are useful safeguards, subject to the output-wide lock repair
and final integration verification above.

## Remaining verification scope

After the repairs stabilize, independently rerun the updated capture tests and
inspect the final native authorization path against the actual fit and semantic
review schemas. Include missing/wrong-bank/wrong-model/partial-head/unsafe-NPZ
negative tests that fail before backend construction, and a changed-protocol
during-backend-load check that fails before the first engineering forward.
Check the 2,048-row training alias and preserved per-row partition metadata.
Record the owner's actual-HF and real-tokenizer check evidence separately from
this initial synthetic test pass.

No fresh study outcome is needed to resolve these engineering defects. Until
the listed native gates and actual-tokenizer/HF checks are verified, preserve
this document's PENDING status rather than treating acknowledged repairs or a
passing initial suite as final measurement validation.
