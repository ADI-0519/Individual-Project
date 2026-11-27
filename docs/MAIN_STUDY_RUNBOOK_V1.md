# Matched belief-study execution: current handoff

**21 November 2026. Full draft inputs exist; main inference has not started.**

The question is whether fitting on belief-preserving as well as changing cases
improves transfer to an unseen correction sequence at matched training budgets.
This measures a readout's generalization. It does not establish spontaneous
anticipation, strategic deception or a causal belief mechanism.

## What is ready

- [154-group draft bank](../data/belief_main/main_v1/main_bank_v1.json): 48 fitting,
  16 development and 90 test groups, expanding to 5,888 prefixes per model.
- [Split manifest](../data/belief_main/main_v1/split_manifest_v1.json) and
  [15 allocations](../data/belief_main/main_v1/matched_allocations_v1.json).
- Train-only activation/TF-IDF fitting, shared regularization selection,
  development calibration, same-forward output logits and group-level analysis.
- [Fixed prose monitor](../data/belief_main/text_monitor_protocol_v1.json), with
  invalid outcomes counted incorrect and raw generations preserved.
- [Baseline policy](../data/belief_main/baseline_protocol_v1.json) and a
  review-gated execution-freeze builder. No actual execution freeze exists yet.
- Both pinned models passed one separate engineering forward with exact hook/HF
  alignment. These profiles contain no main-study cases or fitted heads.

Canonical bank SHA256:
`e038025571a38a7f2c489e5f8c3bf3b0a3c11a05f9b8c7e89d26292a4e7bfb1d`.
The groups share three mechanisms and one renderer. They are contextual
framings, not 154 independent reasoning mechanisms. Correction also differs in
length/event count; interpret transfer to these sequences rather than pure
composition independent of surface form.

## Human semantic review

Open [the offline worksheet](review/main_study_v1/review.html) in a browser.
It starts blank and exports partial records. The student reviews every group
and its representative prompts/transformations. A second human reviews a
stratified sample and every disputed group. The
[preselected independent sample](review/main_study_v1/independent_sample_v1.json)
contains seven groups and 32 representative prompts, selected using a declared
SHA256 rule before main outcomes. Sampling does not certify all other groups.

Use an actual name, record independent answers, and export frequently. A dispute
or disagreement with the formal labels requires reconciliation; neither the
merger nor a model fills or changes human answers. Agent reviews and software
fixtures are separate evidence. The current ledger has zero human records.

Merge downloaded fragments into a **new** ledger. Replace the quoted example
fragment paths below with the actual files. Never edit the existing blank ledger
to imply a review occurred.

```bash
.venv-wsl/bin/python scripts/validate_main_study_review.py \
  --bank data/belief_main/main_v1/main_bank_v1.json \
  --fragments "path/to/student-fragment.json" "path/to/independent-fragment.json" \
  --output /home/adirj/kcl-latent-probes-runs/main_v1_reviewed_ledger.json \
  --report /home/adirj/kcl-latent-probes-runs/main_v1_review_readiness.json
```

Incomplete review returns a nonzero exit and cannot authorize capture.

## Existing WSL environment

The GPU stack is already installed and tested in Ubuntu/WSL. Windows' separate
virtual environment is used for lightweight CPU tests. Run GPU commands from
the repository directory inside WSL, using the existing local cache:

```bash
export HF_HOME="$PWD/.cache/huggingface-wsl"
export HF_HUB_CACHE=/home/adirj/kcl-hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
```

For local Windows pre-commit checks on this already installed environment, set
`UV_NO_SYNC=true` and `UV_OFFLINE=true` for that command. The normal hook tried
to rebuild the editable installation in OneDrive and encountered certificate/
readonly-directory errors; that installation has been repaired and its import,
console entry point and RECORD verified. CI uses a fresh locked Linux environment.
No dependency versions or global TLS settings were changed by the repair.

All result directories must use native storage outside the repository and
OneDrive. Each completed directory executes exactly one partition. Preserve
attempts; use explicit resume only when the runner permits it. An incomplete
unknown in-flight row is not automatically retried.

## Training/development capture and fitting

Create the first freeze **after** the review merger reports complete. This
binds the exact bank, human ledger, source files and baseline policies:

```bash
.venv-wsl/bin/python scripts/prepare_main_study_execution.py \
  --bank data/belief_main/main_v1/main_bank_v1.json \
  --review-ledger /home/adirj/kcl-latent-probes-runs/main_v1_reviewed_ledger.json \
  --stage train_dev \
  --output /home/adirj/kcl-latent-probes-runs/main_v1_training_freeze.json
```

The final tokenizer preparations are in:

```text
/home/adirj/kcl-latent-probes-runs/main_v1_preflight_20261121_llama_train_dev_review_gate_v2
/home/adirj/kcl-latent-probes-runs/main_v1_preflight_20261121_qwen_train_dev_review_gate_v2
/home/adirj/kcl-latent-probes-runs/main_v1_preflight_20261121_text_monitor_development_review_gate_v2
```

Their source hashes must still match. If code changes, prepare fresh directories;
do not rewrite their source bindings. For each model, run the following with its
preparation directory and a new model-specific fit directory. Run GPU processes
sequentially on the local GPU:

```bash
.venv-wsl/bin/python scripts/run_main_study.py execute \
  --output "MODEL_PREPARATION_DIRECTORY" --partition train_dev \
  --review-ledger /home/adirj/kcl-latent-probes-runs/main_v1_reviewed_ledger.json \
  --execution-freeze /home/adirj/kcl-latent-probes-runs/main_v1_training_freeze.json

.venv-wsl/bin/python scripts/fit_main_study.py \
  --bank data/belief_main/main_v1/main_bank_v1.json \
  --capture "MODEL_PREPARATION_DIRECTORY" --output "NEW_MODEL_FIT_DIRECTORY"
```

The fitting CLI retains all 30 comparisons per model: two feature kinds,
three fitting budgets and five paired allocations. No test rows enter fitting.
Failed C candidates and unavailable primary allocations remain recorded.

Execute the monitor's development partition with
`scripts/run_main_study_monitor.py execute --partition development`, using its
prepared directory and the same reviewed ledger/training freeze. The fixed
monitor does not generate training rows. Do not select or rewrite its prompt
using correction outcomes; do not turn another qualification loop into a
prerequisite for the core comparison.

## Held-out execution and analysis

After both model-specific fits are saved, create a **new** test freeze:

```bash
.venv-wsl/bin/python scripts/prepare_main_study_execution.py \
  --bank data/belief_main/main_v1/main_bank_v1.json \
  --review-ledger /home/adirj/kcl-latent-probes-runs/main_v1_reviewed_ledger.json \
  --stage test --fit-dir "LLAMA_FIT_DIRECTORY" --fit-dir "QWEN_FIT_DIRECTORY" \
  --output /home/adirj/kcl-latent-probes-runs/main_v1_test_freeze.json
```

Prepare separate fresh test directories with the two capture/monitor `prepare`
commands. Capture each model's test partition using the test freeze and its
`--fit-dir`. Execute the monitor's test partition once using that freeze and a
bound primary fit directory. All test variants remain together; no filtering
based on whether the model answers correctly is allowed for the primary result.

For each model, attach the completed prose-monitor test run to the held-out
analysis:

```bash
.venv-wsl/bin/python scripts/analyse_main_study.py \
  --capture "MODEL_TEST_CAPTURE_DIRECTORY" --fit "MODEL_FIT_DIRECTORY" \
  --bank data/belief_main/main_v1/main_bank_v1.json \
  --monitor-run "TEXT_MONITOR_TEST_DIRECTORY" --output "NEW_ANALYSIS_FILE.json"
```

The primary Llama N512 contrast averages the five paired allocation differences
within each of 30 correction groups, then bootstraps whole groups. Report both
arms' absolute scores, interval and allocation variation. The same-format
secondary correction cells and changing/preserving pair scores remain visible.
The separate Qwen result uses its own heads. Missing baselines, invalid outputs
and failures remain explicit; no positive or negative result is predetermined.
