# Larger models and King's CREATE

This is a proposed resource plan. No CREATE account, allocation or model run is
implied by this repository. Check the local pilot before requesting a large job.

## What a larger model would help us test

Use the 7–8B experiment to establish that the task, labels and extraction code
work. If resources permit, repeat selected experiments with a stronger model.
The question is whether the observed relationship between pressure, disclosure,
recipient information and probe scores holds across models.

A sensible order is:

1. **Local debugging:** a small checkpoint, used only to verify the software.
2. **Main pilot:** one 7–8B instruction-tuned checkpoint, with precision recorded.
3. **Precision check:** the same checkpoint at higher precision on a subset.
4. **Scale or family replication:** one justified larger checkpoint, potentially
   in the 20–32B range if this is what the allocation can support.
5. **70B replication:** only after the preceding results justify its cost.

The intermediate size is optional. If the lab already has a reliable 70B
environment and sufficient allocation, one carefully chosen 70B replication
may be more useful than several intermediate models. Parameter count alone is
not a measure of task competence, model comparability or scientific quality.

The Apollo study used Llama-3.3-70B-Instruct as its principal evaluation model,
making that checkpoint a relevant possible replication target. Our smaller
model implementation is an adaptation of the method; it is not a reproduction
of the paper's numerical results. See the
[published paper](https://proceedings.mlr.press/v267/goldowsky-dill25a.html).

### Change one source of variation at a time

| Comparison | Hold fixed | Change | Interpretation |
| --- | --- | --- | --- |
| Precision | Checkpoint, cases, prompts, extraction positions | Quantized / BF16 inference | Whether a result depends on numerical precision |
| Sender model | Recipient checkpoint and settings, scenarios | Sender checkpoint | How sender behaviour and monitoring differ |
| Recipient model | Saved sender messages and their provenance | Recipient checkpoint | How the same communications affect different recipients |
| Domain shift | Training procedure, feature extraction, threshold | Entire held-out scenario families/domains | Generalisation beyond the training material |

Changing both sender and recipient at once makes the cause of a changed outcome
ambiguous. Start with the first two model comparisons separately. A later
factorial study can combine them deliberately.

Train a separate probe for every checkpoint and extraction convention. A vector
from one model cannot simply be treated as coordinates in another model's
representation space. Freeze scenario IDs and evaluation splits before these
comparisons. Record checkpoint revision, tokenizer, chat template, precision,
layer, pooling, generation parameters and seeds. Use fresh confirmatory data
after developing the protocol on the supplied pilot.

## Memory planning

These are arithmetic estimates for raw parameter values in **decimal GB**.
They are not measured peak GPU-memory requirements.

| Nominal parameters | 16-bit parameter values | 4-bit parameter values |
| --- | ---: | ---: |
| 8B | 16 GB | 4 GB |
| 32B | 64 GB | 16 GB |
| 70B | 140 GB | 35 GB |

Quantization metadata, non-quantized modules, attention caches, temporary
buffers and activation extraction add memory. The exact parameter count also
differs from the rounded model name. Loading an original checkpoint and
quantizing it can require substantial host RAM and storage even if its final
GPU weights are smaller.

In particular, **35 GB of raw 70B weights does not establish that a 70B run fits
on a 40 GB GPU**. An 80 GB GPU gives more room for a short-context 4-bit run, but
fit and runtime still require an actual measurement. A BF16 70B run typically
requires weights to be distributed across multiple sufficiently large devices;
the aggregate weight estimate alone does not guarantee that the device mapping
and runtime will fit.

Start each new configuration with one short example at batch size 1. Record
peak allocated/reserved device memory and wall-clock time, then run a small
batch before estimating the budget for the full experiment. Extract the few
vectors the analysis needs and move them to CPU/disk. Avoid retaining all
generation steps, layers and token states simultaneously.

## Request CREATE access

King's currently permits taught-student access when contributing to an existing
King's research project, with support from that project's PI. Ask Hanqi whether
this dissertation is eligible under a project she can support. The request
needs your KCL email, k-number and project association. Access and available GPU
partitions must be confirmed; a research proposal is not an allocation.

Primary guidance, checked 18 November 2026:

- [Requesting access](https://docs.er.kcl.ac.uk/CREATE/requesting_access/)
- [GPU jobs](https://docs.er.kcl.ac.uk/CREATE/hpc/running_jobs_gpu/)
- [Storage](https://docs.er.kcl.ac.uk/CREATE/hpc/storage/)

Describe the first request as **single-GPU inference and activation extraction
for a frozen 8B language model**, followed by small classifier fitting. Ask
which tested Python/CUDA environment, GPU constraint, storage location and job
limits the group recommends. Request a larger configuration after recording
the pilot's actual memory and time requirements.

## Submit the provided job script

CREATE uses Slurm. Its GPU guide specifies a GPU resource request and an
explicit GPU partition at submission. The script requests one GPU and leaves
the partition/account choice to the actual allocation. Its time, CPU and host
memory defaults are initial planning values, not benchmark results.

1. Obtain access and establish the account's supported Python/CUDA environment.
2. Put the repository and Python environment on storage visible to compute nodes.
3. Install the project in that environment and test the command-line interface.
4. Download the permitted model checkpoint to an approved cache location before
   submitting the expensive run; check its revision and disk requirements.
5. Run a short GPU smoke test through Slurm, then the initial experiment.

Example shell setup; replace the example paths with real paths on your account:

```bash
export LATENT_PROJECT_DIR="/path/to/kcl-latent-probes"
export LATENT_PYTHON="/path/to/venv/bin/python"
export LATENT_CONFIG="configs/llama8b_4bit.json"
export LATENT_OUTPUT="/path/to/approved/scratch/pilot8b"
cd "$LATENT_PROJECT_DIR"
sbatch --partition=gpu scripts/create_gpu_job.sh sender
```

`gpu` appears in the current King's guide; use it only if your account is
eligible. Add the account or GPU constraint recommended by ARC/the lab when
needed. Do not guess an account name or GPU constraint. After the sender job
finishes, use the same output path for the recipient stage:

```bash
sbatch --partition=gpu scripts/create_gpu_job.sh recipients
```

Set `LATENT_RECIPIENT_CONFIG` to an explicit model configuration to replay the
same sender messages with a different recipient checkpoint. Preserve the
original replay outputs when comparing recipients. To resume an interrupted
stage with the same experiment settings, set `LATENT_RESUME=true` before
submission. The command validates the existing run rather than silently mixing
different settings.

The script does not install packages or submit further jobs. It uses an existing
environment. Check log and output paths before running; use a new output
directory for a different experiment configuration.

King's scratch storage is intended for active computations and is **not backed
up**. Keep another copy of source code, configurations, annotations and
irreplaceable results. Store neither model weights nor access tokens in Git.
Use the facility's acknowledgement wording in outputs where required.

## Evidence needed before a large run

- Four cases have been inspected end to end and the 24-case pilot reviewed.
- Complete-information recipient controls work, or failures are understood.
- Natural sender outputs provide enough variation to address the question.
- The model revision and extraction convention are fixed and recorded.
- Critical-fact labels were audited; pressure labels are not called deception.
- The chosen group split is frozen and does not reuse pilot tuning as final test.
- One representative example has measured GPU memory and runtime.
- The larger run answers a specific comparison, with sender and recipient changes
  separated and a stop condition agreed in advance.
