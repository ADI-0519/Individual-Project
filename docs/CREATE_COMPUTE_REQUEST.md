# CREATE request: paired precision and reader-policy reference

Prepared 20 November 2026 for Aditya Ranjan, **Latent Probes and Listener Beliefs**,
6CCSAPRJ. This is a supervisor-ready request, not a submitted application or an
allocation. No message has been sent to ARC or another recipient.

## Proposed request to the supervisor

I would like PI-supported access to an eligible existing CREATE research project
and an initial cap of **two GPU-hours on one NVIDIA A100 40 GB**. The first job
will test whether our fresh reader-control failures persist when the same pinned
Qwen2.5-7B-Instruct checkpoint and fixed nine-example policy run in BF16 rather than NF4, with both profiles
executed on the same allocated hardware and software stack. The first comparison
is behavior-only; it does not train a language model or reuse a quantized probe
as though it had been validated at another precision.

The local 16 GB RTX 5080 already supports quantized development. CREATE is requested
for a controlled higher-memory reference and later independent-reader replication,
not to make the project appear more computationally ambitious. Approximately
15.2 GB (about 14.2 GiB) of raw 16-bit parameter values for the reader leave limited space on the local card for
model/runtime overhead. Actual peak memory must be measured. King's lists
A100 40 GB devices in its standard GPU-node specifications; this does not guarantee
availability or account eligibility. [Official compute-node specifications](https://docs.er.kcl.ac.uk/CREATE/hpc/compute_nodes/).

## Empirical reason for the request

The fresh communication batch contains **144 natural reports from 24 cases across
six decision rules**, with frozen neutral-trained monitors and a supporting
same-report recipient-evidence assay. Its measured limitation is reader validity,
not a demonstrated activation advantage. The activation forecast gets 128/143
valid reader targets correct, while a majority prediction gets 129/143; the
activation head recalls none of the 14 REJECT targets. High aggregate accuracy
therefore cannot justify a strong-monitor claim. These are actual-reader outcome
labels, not independently established belief or deception labels.

The 72 no-report controls expose the immediate problem. Correctness below means
the decision supported by the reader's available evidence; an absent reader
should defer, and a stale reader can reasonably follow its older evidence.

| Reader packet, 24 cases each | Original direct policy | Same direct policy, 192-token cap | Reasoned policy, strict parser | Reasoned policy, punctuation-tolerant secondary parser |
| --- | ---: | ---: | ---: | ---: |
| Informed / full evidence | 20/24 | 20/24 | 14/24 | 22/24 |
| Absent decisive evidence | 4/24 | 4/24 | 16/24 | 18/24 |
| Stale evidence | 19/24 | 19/24 | 11/24 | 20/24 |

Increasing the budget alone changes no direct-policy control answers; no output
in the longer-budget diagnostic reaches the cap. Reasoned answers improve under
a separately labelled punctuation-tolerant final-line extraction, but do not
fully solve the controls. Those policy/parser diagnostics used already-observed
cases, so they are development evidence, not fresh confirmation. Strict results
and every invalid output remain preserved. See the [curated measurements](./validation/2026-11-20-communication-transfer.json),
especially `reader_capability` and `punctuation_sensitivity`.

An independent instruction-only Qwen2.5-7B reader also failed its frozen gate:
22/24 informed, 12/24 absent and 15/24 stale. A subsequently fixed nine-example
policy was qualified on **24 separate fresh cases**, yielding 24/24, 17/24 and
24/24, with zero invalids. It failed the predefined absent requirement of 20/24;
no dependent fitting or untouched natural-transfer calls followed. The original
and fresh banks have different instances, so those figures are separate
diagnostics, not a paired model/policy effect. See the [terminal follow-ups](./validation/2026-11-20-reader-followups.json).

These findings do **not** show that quantization caused the errors or that BF16
will fix them. They justify a paired precision reference before scaling a study
whose interpretation depends on a reader using supplied evidence correctly.

## Frozen first experiment and resource cap

Use `Qwen/Qwen2.5-7B-Instruct` at revision
`a09a35458c702b33eeacc393d103063234e8bc28`. Freeze the **same 72 fresh `cv`
qualification inputs**, nine-example reader policy, greedy seed 42, eight-token
generation cap and strict parser before CREATE outputs are opened. Evaluate
**72 controls × two precision profiles = 144 conditions**. Re-run NF4 on CREATE
as well as BF16; comparing only local NF4 with cluster BF16 would confound
precision with hardware/runtime. All existing local outcomes remain unchanged.
These already-observed cases diagnose precision sensitivity; they cannot
independently qualify a policy developed using their failures.

| Resource | Initial request and reason |
| --- | --- |
| GPU | One A100 40 GB; run NF4 and BF16 sequentially on the same device/runtime |
| GPU time | At most two allocated GPU-hours initially, including model loading and profiling; a planning cap, not measured runtime |
| CPU / host RAM | Four CPU cores and 64 GiB RAM as initial planning limits; revise from measured loading/inference peaks |
| Storage | One approved cache copy of the permitted pinned checkpoint, code/environment and small text/metadata outputs; no all-layer activation archive in the first job |

Start with model/kernel checks and one short example in each configuration, then
six predetermined representative prompts covering the rule families and packet
types. Record initialization time, generation time, token counts, peak allocated
and reserved GPU memory, host memory, GPU model, driver, PyTorch/CUDA, tokenizer
and chat-template hashes. Profiled calls may form part of the 144 conditions only
when their frozen settings are identical. Repeated references do not add cases.

Project remaining time from the observed configuration-specific runtimes plus
model-loading overhead. If the remaining work cannot fit within the initial cap,
stop and return the profile with a revised estimate; do not silently expand the
allocation or select only successful prompts. Compare paired per-case decisions,
invalids, deferrals and packet-specific errors, rather than treating repeated
conditions as independent scenarios.

The cluster runtime must first pass its actual CUDA/model check. The local CUDA
wheel configuration is not assumed compatible with CREATE. Confirm the supported
environment and eligible GPU constraint with ARC/project administrators. King's
documents Slurm GPU-resource requests and explicit GPU partitions; its examples
are guidance, not a tested submission for this project. [Official GPU-job guide](https://docs.er.kcl.ac.uk/CREATE/hpc/running_jobs_gpu/).

## Interpretation and later replication

This first experiment can distinguish precision-sensitive behavior from errors
that persist under both numeric profiles on matched CREATE execution. It cannot
separate every interaction with prompting or establish a general causal account.
Even an improvement must be followed by a fresh human-reviewed validation bank,
since these control outcomes are now observed and will inform subsequent decisions.

Do not apply the existing NF4 latent weights/scaler directly to BF16 activations
and interpret the result as an equivalent monitor. Any later latent comparison
needs profile-specific matched neutral fitting/calibration, fixed feature
coordinates and layer/mask reference checks, followed by fresh evaluation.
Preserve the original NF4 artifacts and distinguish precision effects from a
newly fitted monitor's effects. The initial behavior-only request avoids this
additional coordinate-identification problem.

After capability controls pass under a frozen policy, a separate measured budget
can support an **independent reader checkpoint** replaying the same saved reports,
then a scaled fresh case bank. Change reader policy/checkpoint separately from
sender generation. That tests whether recipient-dependent findings survive a
new reader, not whether the original sender anticipated an unannounced recipient.
The broader [CREATE/runtime plan](./SCALING_AND_CREATE.md) retains these boundaries.

## Access route to confirm

Official access guidance was checked on **20 November 2026**. For a taught
undergraduate, the documented route is contribution to an existing King's
research project with its PI's support. Supervising King's staff submit the
student request; general taught-student self-service access is not established.
Please confirm the eligible existing project and PI, then provide the student's
King's email, k-number and project association through that route. Neither this
dissertation nor the resource proposal alone establishes eligibility. [Official requesting-access guidance](https://docs.er.kcl.ac.uk/CREATE/requesting_access/).

The proposed deliverable is a paired control table with exact prompts, outputs,
deviations, environment records and measured compute costs, followed by a clear
decision about fresh validation. No new larger-model allocation is requested at
this stage.
