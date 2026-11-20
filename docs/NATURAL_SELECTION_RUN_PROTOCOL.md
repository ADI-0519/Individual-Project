# Natural communication feasibility execution

20 November 2026. New development pilot; no probe fitting and no reopening of
the failed qualification pipeline. This execution implements the bounded
[next-study design](NEXT_MAIN_STUDY_PROTOCOL.md). It does not establish a
deception label, a sender-belief mechanism or broad family generalization.

## Frozen runtime

- Actor: Llama-3.1-8B-Instruct revision
  `0e9e39f249a16976918f6564b8830bc894c89659`, NF4 double quantization with BF16
  compute, decoder block 8, greedy generation, seed 42, 192 new tokens.
- Reader: Qwen2.5-7B-Instruct revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, same NF4/BF16 profile, unchanged
  nine-example system/demonstration policy, greedy generation, seed 42,
  eight new tokens and the existing strict EOS-aware answer parser.
- Both models use a 4096-token prompt allowance. This new input allowance
  accommodates longer sources; it does not retroactively change the old
  2048-token qualification runtime or its failed result. No truncation is allowed.
- The draft bank is unchanged: eight cases, four document settings sharing one
  two-check/latest-available-final grammar. All human-review flags remain false.

## Stages and preservation

Preparation freezes model settings, data/code/protocol hashes, policy identity,
the 32 actor conversations and exact known reader-control/skeleton inputs.
Tokenizer preflight performs no model generation. After the reports exist,
freeze all 240 complete reader conversations and preflight every actual input
before the first reader answer. Unknown future reports cannot have exact token
IDs at preparation; the second preflight is mandatory.

Generate each actor report once. Save its original prompt/generated IDs and
unstripped decoded report before attempting activation capture. Decode using
`skip_special_tokens=True, clean_up_tokenization_spaces=False`; also retain the
decode including special tokens and record removed special IDs. Replay the
unstripped report string verbatim in reader messages. This preserves report
content across recipients; it does not claim identical cross-model token IDs.

Capture the saved original actor sequence at its non-special generated token
positions without regenerating it. If there is no content, preserve the empty
report and mark features unavailable. These features are a separate original
sender-state diagnostic; the pilot fits and scores no new outcome probe.

Close the actor before loading the reader. Cache exact reader calls by full
messages, pinned runtime, generation settings and seed. Preserve 240 condition
rows separately from the number of unique calls. Always append the nominated
sentence in the 25-word addition arms, even when it repeats known information.
Never filter these interventions using observed omissions or outcomes.

Atomic records preserve completed calls and failures. An explicit resume must
verify every source/protocol hash and all saved call bindings before skipping
completed calls. It cannot replace an existing report or change settings.
Fixtures are identified as fixtures in manifests and summaries.

## Interpretation and feasibility

The five advancement criteria in the main design apply to the 25-word arm.
Report all goals, budgets and invalid outputs. Preserve original/no-report and
original/restoration transition counts separately, including wrong-to-defer.
Source-world correctness, packet-only normative decisions and actual observed
decisions remain distinct fields.

Export source/rule/report text for a blind semantic audit without objectives,
outcomes, probe scores or model-visible IDs. Agent draft annotations do not
substitute for human validation. Behavioral execution alone cannot pass the
supported-omission gates; its summary must retain semantic review as pending.

## Commands and verification

The existing WSL environment and native Linux model cache supply the runtime.
Set the same offline Hugging Face cache environment documented for the earlier
research runs. From the inner repository in that environment:

```bash
.venv-wsl/bin/python scripts/run_natural_selection_pilot.py prepare --output /home/adirj/kcl-latent-probes-runs/natural_selection_pilot_v1_20261120
.venv-wsl/bin/python scripts/run_natural_selection_pilot.py execute --stage reports --output /home/adirj/kcl-latent-probes-runs/natural_selection_pilot_v1_20261120
.venv-wsl/bin/python scripts/run_natural_selection_pilot.py execute --stage readers --output /home/adirj/kcl-latent-probes-runs/natural_selection_pilot_v1_20261120
```

Preparation refuses an existing output directory. Execution preserves partial
calls; `--resume` requires unchanged source, journal, feature and runtime bindings.
A journal update interrupted between its atomic file write and manifest write
is conservatively refused rather than treated as permission to regenerate.

Before the first real run, the full offline suite passes 671 tests and 48
subtests. Forty focused new tests cover the panel and runner, including exact
whitespace/empty reports, deleted or modified records, changed runtime, capture
failure, reader failure, cache reuse and feature corruption. Tokenizer-only
preflight checks all 32 sender prompts and 240 reader layouts. Complete reader
inputs still require the second preflight after natural reports exist. These
checks do not certify semantic labels or establish a model result.
