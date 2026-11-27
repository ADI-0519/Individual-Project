# Portable replay of the format-adaptation metrics

The [exporter](../scripts/export_format_adaptation_replay.py) packages derived
scores and enough test metadata to rerun the unchanged production metrics away
from the original native campaign paths. It performs no model calls, fitting,
tokenizer calls, threshold changes or hidden-vector reconstruction.

This is **portable replay using production analysis functions**, not another
independent audit. The original native audit checks the feature-to-prediction
calculation and retains its full native evidence. Replay checks packaged bytes,
row identities, factor decisions/fold consistency and all prescribed metrics.
It cannot independently re-prove a projection against omitted prompt text or
hidden vectors.

## Export only after the completed independent audit

Keep current-study outcomes locked until both analyses and the independent audit
have completed. Then run from the repository's existing scientific environment:

```sh
python scripts/export_format_adaptation_replay.py export \
  --campaign /home/adirj/kcl-latent-probes-runs/format_adaptation_v1_20261127_campaign \
  --audit /home/adirj/kcl-latent-probes-runs/format_adaptation_v1_20261127_campaign_audit \
  --output docs/results/format-adaptation-2026-11-27/replay
```

The output must be new and outside both native evidence directories. Public
export accepts only complete real test captures, both complete analyses and a
passing both-model audit bound to their exact bytes. Failures are retained;
existing bundles are never overwritten. Explicit private fixture helpers exist
only for tests, stamp the bundle and receipt as synthetic, and add a prominent
README warning. The public APIs and CLI reject fixture export and replay.

The bundle contains:

- Exact original prediction NPZ bytes for both checkpoints: arm verdicts, four
  factor margins/predictions, conditional output margins, strict first-token
  correctness and row/arm/factor order. Archive keys are checked explicitly;
  features, coefficients and model weights are excluded.
- Gzip JSON projections of all 4,992 test rows per checkpoint: complete formal
  states, gold labels, contextual/format/condition identity, original prompt
  hashes, candidate token identities and the scalar output summary needed for
  strict-token, tie and candidate-mass diagnostics. Prompt text and token arrays
  are excluded. Original capture-row hashes are bound to the audit at export.
- Gzip copies preserving the exact original result, analysis-manifest,
  capture-manifest, test-seal, independent-audit receipt and reconstruction-seal
  bytes. Absolute source paths are retained as historical provenance only.
- Exact sealed production analysis, fit-helper and storage source files, minimal
  package glue and a standalone copy of the replay script. The fit-helper module
  supplies constants and pure functions; replay invokes no fitting routines.
- A manifest hashing every local member and recording the original campaign,
  projection provenance, implementation hashes and export dependency versions.

Prediction files remain unchanged, while repetitive JSON is compressed. The
bundle should stay small relative to native captures; no score archive is
reduced by dropping arms, contexts, conditions or factors.

## Replay after copying

Copy the entire directory to another location. With Python 3.11+ and the normal
metric dependencies installed (`numpy`, `scipy`, `scikit-learn`, `threadpoolctl`),
run from the copied bundle:

```sh
python replay.py replay --bundle . --output ../portable-replay-receipt.json
```

`verify` is an alias for `replay`. No repository checkout, model installation,
original native path, activation archive or fitted coefficients are required.
Receipts must be new files outside the immutable bundle. A failed verification
does not rewrite its inputs.

Replay verifies every member hash, both checkpoint identities, exact arm/factor
order, row joins and complete formal/A-B inventory. It loads the copied sealed
metric sources under an isolated package name. It then reruns all production
`score_arrays` scopes and the scalar output-token diagnostic: all arm/model/
format/condition row/class/quartet/nuisance metrics, per-context values, pooled/
easy/hard factor diagnostics, paired-format diagnostics, output baselines,
Llama primary, Qwen named secondary and Mixed-minus-Offset diagnostic. The
unchanged paired-context bootstrap remains 10,000 draws at seed 20261127.

Counts, strings, booleans and inventory are compared exactly. Floating metrics
use fixed absolute and relative tolerances of `2e-10`, matching the numerical
audit convention. Both input and output manifests preserve the distinction
between replay and independent reconstruction. File hashes provide integrity
relative to the retained manifest; publish or record that manifest hash with the
bundle rather than treating an editable manifest as a digital signature.

The [synthetic tests](../tests/test_export_format_adaptation_replay.py) exercise
both-model replay, a relocated standalone copy with native inputs temporarily
unavailable and repository/model imports blocked, corruption and audit-binding
refusals, fixture barriers and preservation of existing outputs. Fixtures are
software evidence, not research results.

## Repository and package inclusion

Raw runs and activations stay ignored. The repository globally ignores NPZ;
after a real audited export, explicitly add only the two small derived score
archives in this replay folder if publishing them. Do not relax the global rule.
The existing distribution manifest also needs a narrowly scoped inclusion for
this bundle's `*.npz`, `*.gz` and hashed `README.txt` files; for example:

```text
recursive-include docs/results/format-adaptation-2026-11-27/replay *.npz *.gz *.txt
```

That packaging change and the actual export are separate completion steps. This
implementation does not alter the sealed scientific source or native evidence.
