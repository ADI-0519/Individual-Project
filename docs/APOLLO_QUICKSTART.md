# Begin with the Apollo baseline

This is an adaptation of Apollo's saved `repe_honesty__plain` recipe. Read
[the method audit](./APOLLO_METHOD.md) and [full reading log](./READING_LOG.md)
before interpreting a score. Training labels identify instructions; they do not
demonstrate that the model lied. The older last-token probe remains available
as a separate starter baseline.

## Install a locked environment

Use Python 3.11 and uv 0.13.5. Ordinary offline development needs:

```sh
uv sync --locked --group dev
uv run --locked python -m pytest tests
uv run --locked python -m ruff check .
```

For a real CPU engineering run:

```sh
uv sync --locked --group dev --extra models --extra cpu
uv run --locked --extra models --extra cpu python -m latent_probes doctor
```

The `cpu` and `cu130` extras are mutually exclusive and use explicit official
PyTorch indexes. For a compatible CUDA 13.0 driver/GPU environment use
`--extra models --extra cu130`, plus `--extra quantization` for the 4-bit profile.
A CUDA installation must pass `doctor`'s actual kernel check. The lock selects
different supported PyTorch wheels for the two profiles; they are distinct
recorded numerical environments. Check the [official uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

On machines using corporate/system certificate authorities, `uv --system-certs`
uses trusted system certificates without disabling TLS verification. Windows
Application Control may independently block a native dependency DLL: changing
TLS settings does not resolve that policy. Use an approved Linux/CREATE runtime
or the manual Linux smoke workflow; do not disable application protection.

## Prepare data explicitly

```sh
uv run --locked --extra models --extra cpu python -m latent_probes apollo fetch-facts --output data/raw/apollo-pinned
uv run --locked --extra models --extra cpu python -m latent_probes apollo prepare --facts data/raw/apollo-pinned/true_false_facts.csv --output data/processed/apollo_smoke_pairs.jsonl --limit 8 --short-fact-policy exclude --duplicate-fact-policy exclude
```

The download is pinned to a source commit and verified Git blob, with SHA-256
provenance. It is not bundled or re-licensed. Both raw and prepared data are
excluded from Git.

The actual pinned CSV has **306 true rows**, not 512 available true facts.
31 have empty prefixes after five-word masking. There are eight later duplicate
occurrences; one also has an empty prefix. Explicitly excluding these after
the original split retains **219 training / 49 validation pairs** for a full
preparation. Omitting `--limit 8` prepares that dataset. Neither exclusion is
silent: the manifest records the original index, split and reason. The small
eight-source-fact example retains seven pairs (six train / one validation).

## Capture and fit

```sh
uv run --locked --extra models --extra cpu python -m latent_probes apollo extract --config configs/apollo_smoke_cpu.json --examples data/processed/apollo_smoke_pairs.jsonl --output runs/apollo_smoke_cpu
uv run --locked --extra models --extra cpu python -m latent_probes apollo train --features runs/apollo_smoke_cpu/features.npz --labels runs/apollo_smoke_cpu/metadata.csv --output runs/apollo_smoke_probe
```

Inspect the rendered dialogues, excluded suffixes, exact selected token IDs and
activation matrices under `runs/apollo_smoke_cpu/records` and `tokens`.
The complete true response remains in context. Every selected token contributes
a training row, with training-only standardization, `C=0.1` and no fitted
intercept. Validation is a held-out **instruction-condition diagnostic**.

The NPZ contains repeated `example_ids`, one per token row. This differs from
the starter's single-vector format: use `apollo train`, not `train-probe`.

Use `--resume` only for an identical interrupted extraction. Changed source,
dependencies, examples, config or completed artifacts are rejected. Use a new
directory for changed experiments; runs have one writer.

## Score actual responses and check the control path

```sh
uv run --locked --extra models --extra cpu python -m latent_probes apollo extract --config configs/apollo_smoke_cpu.json --examples data/examples/apollo_smoke_controls.jsonl --output runs/apollo_smoke_controls
uv run --locked --extra models --extra cpu python -m latent_probes apollo score --features runs/apollo_smoke_controls/features.npz --probe runs/apollo_smoke_probe --output runs/apollo_smoke_control_scores.csv
uv run --locked --extra models --extra cpu python -m latent_probes apollo calibrate --scores runs/apollo_smoke_control_scores.csv --output runs/apollo_smoke_threshold.json
uv run --locked --extra models --extra cpu python scripts/validate_model_smoke.py
```

These six original, neutral prompts exercise the software. They are **not** the
paper's Alpaca control set and cannot estimate a reliable 1% tail. Scores are
means of raw standardized projections, not probabilities. The validator also
loads the genuine checkpoint to compare the hook with its actual hidden-state
tuple and to check causal-prefix invariance at a declared tolerance.

The **Real model smoke (manual)** GitHub Action runs these commands only when
explicitly dispatched. PR CI remains offline and never downloads model weights.
Model caches are not uploaded; small manifests/captures are retained as an
engineering artifact. The job is not a deception experiment.

## Move to the main checkpoint

Use `configs/apollo_llama8b_bf16.json` on suitable hardware and
`configs/apollo_llama8b_4bit.json` for a separate quantized development run.
Llama access must be granted to your own Hugging Face account; authenticate on
the machine, never place a token in Git. The 8B profile uses decoder block 8
(HF hidden-state index 9), a declared relative-depth heuristic. A layer sweep
belongs on validation data, followed by one frozen held-out evaluation.

For real evaluation, use separate JSONL examples with `split=evaluation` and
`mode=generate` for actual target-model outputs; `mode=fixed` is explicitly
off-policy. After inspection, complete `label`, one consistent `label_target`
and `label_audited=true` in a separate copy of the metadata CSV. Do not edit the
original run's hashed metadata. Then score the corresponding features and use:

```sh
uv run --locked --extra models --extra cpu python -m latent_probes apollo evaluate --scores runs/evaluation_scores.csv --labels annotations/evaluation.csv --threshold runs/neutral_control_threshold.json --output runs/evaluation_metrics.json
```

The last paths are examples for a dataset you must collect. Labels must not be
inferred from pressure prompts or from MASK's non-lying complement. Separate
honest assertions, lies, evasion, inconsistent/no-belief cases, ambiguity and
judge failures. The evaluator refuses instruction labels as empirical behavior
labels and checks calibration/training overlap when provenance is available.

No 8B detection result, calibrated deception probability, reliable 1% FPR or
novel omission/belief mechanism follows from the engineering smoke.
