# Portable CPU reproduction of the completed pilot

The [export/replay tool](../scripts/portable_natural_selection.py) closes the
completed pilot's machine-local evidence gap. It produces a bounded archive
containing all 42 captured run files, the original frozen Apollo probe and
manifest, both draft reviews, source/protocol snapshots, original analyses and
the 13 package modules required by their imports. Original artifact bytes and
absolute provenance strings are preserved. A separate inventory maps those
strings to archive members; no original path needs to exist during replay.

The archive includes no model weights, Hugging Face token, model cache,
recipient-transfer bank or raw files from another experiment. Tokenizer assets
are acquired separately under their original access/license terms. The inventory
contains 76 payload files and 59 original-path mappings; the ZIP has one
additional inventory member. It is a replay package, not a complete checkout or
an independent replication of model inference.

## Prepared archive

- Local deliverable: [natural_selection_pilot_v1_cpu_v2.zip](../results/reproduction/natural_selection_pilot_v1_cpu_v2.zip).
- ZIP bytes: `12395308`.
- ZIP SHA256: `00d69661b7c5881d4509f55cdd60b84d9bb5d52b73478dd0781d64e9562f3a14`.
- Canonical inventory SHA256: `8a9af1106f8166c4cfd19f52564a2e526ecebed177383a29115d6e54e9731d11`.
- Payload bytes before compression: `30934438`.

The ZIP is intentionally ignored by Git. It has been prepared locally, not
published as a GitHub release. Anyone receiving it needs the ZIP and its
separately communicated checksum, in addition to the committed source code.
Identical byte snapshots produce identical archives: two consecutive exports
produced the same ZIP SHA256. Subsequent source/document changes may produce a
different inventory; do not overwrite or relabel an earlier archive.

## Export, verify and extract

From the project root, with the existing Python environment:

```powershell
.venv/Scripts/python.exe scripts/portable_natural_selection.py export `
  --run runs/natural_selection_pilot_v1_20261120 `
  --probe runs/apollo_llama8b_full_probe_v1 `
  --output results/reproduction/new_pilot_export.zip

.venv/Scripts/python.exe scripts/portable_natural_selection.py verify `
  --archive results/reproduction/new_pilot_export.zip

.venv/Scripts/python.exe scripts/portable_natural_selection.py extract `
  --archive results/reproduction/new_pilot_export.zip `
  --destination C:/Users/adirj/AppData/Local/Temp/new-pilot-replay
```

The archive and extraction destination must not already exist. Verification
checks the full member allowlist, SHA256 and byte lengths before extraction.
It rejects traversal, absolute paths, Windows-reserved names, case collisions,
symlink entries, duplicate/unlisted members, encryption and excessive sizes.
Checksums protect against accidental changes; they do not authenticate an
unknown publisher or make executing untrusted Python code safe.

## Two explicit levels of replay

The default requires Python 3.11+ and the existing project core dependencies.
The tested Windows versions are NumPy 2.4.6, scikit-learn 1.9.1 and SciPy 1.17.1;
the original lockfile is included as a dependency reference.

```powershell
.venv/Scripts/python.exe -I C:/Users/adirj/AppData/Local/Temp/new-pilot-replay/reproduce.py reproduce `
  --bundle C:/Users/adirj/AppData/Local/Temp/new-pilot-replay `
  --output results/reproduction/new_record_replay.json
```

This recomputes every scientific field of the original pilot analysis, including
the review join, eligibility, all five gates, groups, transitions and behavioral
summaries. It also recomputes all 32 Apollo margins, diagnostic proxies, stratified
AUROCs and goal-paired summaries from the saved feature matrices and original
probe. It checks exact integer/count/string agreement and numerical agreement
within `1e-12` absolute and relative tolerance.

**Default replay does not independently retokenize prompts or decode token IDs.**
It verifies the recorded token IDs, text hashes, messages, capture positions and
their bindings. Its output records this boundary explicitly. The tested Windows
environment does not have transformers installed.

For the independent tokenizer check, add both local tokenizer directories:

```text
python -I /relocated/pilot/reproduce.py reproduce \
  --bundle /relocated/pilot --output /outside/pilot-tokenized-replay.json \
  --actor-tokenizer /local/pinned-llama-tokenizer \
  --reader-tokenizer /local/pinned-qwen-tokenizer
```

This mode uses the unchanged original run verifier with only explicit path
relocation. It independently reconstructs prompt IDs and decoded output text,
and requires exact vocabulary, chat-template and special-ID identity matches.
Loading is local-only with `trust_remote_code=False`. The tested transformers
version is 4.57.6. Neither mode loads model weights, generates an answer, fits a
probe, selects a threshold, changes the target or advances a failed fit gate.

## Actual relocated verification

The final ZIP was extracted outside the repository into
`C:/Users/adirj/AppData/Local/Temp/kcl-portable-pilot-20261121-v3`. A fresh native
Windows process with `-I` reproduced the results without transformers. A fresh
WSL process, launched from `/tmp` against the same extraction, used the pinned
local Llama/Qwen tokenizers to repeat full tokenization/decoding verification.
See the [machine-readable audit](validation/2026-11-21-portable-pilot-reproduction.json)
for exact commands, checksums and the comparison boundary.

Both modes compare against existing saved results. They provide portable
computational reproduction, not new observations, human annotation validation,
a BF16 reference, a new Apollo replication or evidence of generalization.
The [25 focused tests](../tests/test_portable_natural_selection.py) exercise
archive integrity and failure boundaries using synthetic bytes, so CI does not
depend on private model assets or ignored raw runs.
