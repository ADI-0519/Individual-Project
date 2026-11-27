# Fresh readout: startup correction and versioned restart

22 November 2026. This supplements the unchanged
[prospective execution protocol](FRESH_READOUT_EXECUTION_PROTOCOL_V2.md).

The first native campaign, `fresh_readout_v2_20261122_campaign`, stopped during
its first Llama training stage. Its capture manifest records zero completed
rows and a tokenizer/backend identity error at 22:04:02 UTC. The first-stage
exit and logs are retained by the campaign coordinator. Independent inspection
found no engineering forward journal, row journal, arrays, fitted heads, test
capture or analysis anywhere in that campaign. The source order places the
failed check before the first model forward. Loading checkpoint weights alone
did not produce a research measurement.

## Exact defect and correction

In preparation, a local variable initially held `tokenizer_binding(tokenizer)`.
A later loop over the source bank and authored-input provenance reused that
variable. Consequently, both prepared protocols stored the last authored-file
reference in their `tokenizer` field. Runtime validation correctly rejected
the malformed identity. Two agents independently located this assignment
defect from the preserved protocol and source; no model outcomes were needed.

The repair gives the tokenizer identity and the provenance-loop item separate
names. A regression test covers a bank with nested authored-input provenance,
unlike the earlier simple capture fixture. Identity checking remains strict.
No original-study source, HF backend, model setting, case, prompt, gold label,
split, candidate fit, site-selection criterion or scientific analysis changes.
This is a software correction before any fresh-study forward, not a repeat
selected on an outcome.

## Restart boundary

Preserve the original campaign without overwriting any file. Save a separate
failure inventory with exact file hashes and the zero-forward evidence. Prepare
both complete 7,040-prefix protocols again in a new native directory,
`fresh_readout_v2_20261122_recovery_v1`, using the repaired, committed source.
Compare every planned row and its engineering row against the original plans:
they must be exactly equal for each model. Verify that the new tokenizer field
has the complete identity and matches a separately loaded pinned tokenizer.

The new preparations bind this supplement, the unchanged scientific protocol,
the repaired source and the original reviewed data. Both models' training and
development capture must finish before fitting; both fitted bundles must be
sealed before either fresh test capture. All subsequent failures remain
preserved, with no automatic retry. The first campaign contributes zero reused
measurements. Agent audit remains exploratory; human semantic review is still
incomplete.

The restart receipt, new prepared protocols and source archive record the exact
hashes and timing. The final interpretation must disclose this correction.
