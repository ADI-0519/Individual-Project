# Latent probes under reporting pressure

[![CI](https://github.com/ADI-0519/Individual-Project/actions/workflows/ci.yml/badge.svg)](https://github.com/ADI-0519/Individual-Project/actions/workflows/ci.yml)

Research software for KCL Variant 2, by Aditya Ranjan. Start with an auditable
adaptation of Apollo's instruction-pair deception probe, then evaluate transfer
under reporting pressure and recipient-knowledge shifts.

**Start here:** [Apollo quickstart](docs/APOLLO_QUICKSTART.md) ·
[paper/source audit](docs/APOLLO_METHOD.md) · [reading log](docs/READING_LOG.md) ·
[starter audit](docs/STARTER_AUDIT.md) ·
[planned pressure pilot](docs/PRESSURE_PILOT.md).

**Current research decision:** [direction and scope](docs/RESEARCH_DIRECTION.md),
[deep literature audit](docs/LITERATURE_PROBE_AUDIT.md), and
[recipient-context experiment](docs/RECIPIENT_CONTEXT_DESIGN.md).
Read the [measured pilot results and next empirical gates](docs/PILOT_RESULTS.md)
before scaling the case bank.

The Apollo path trains on selected token rows and scores mean raw projections.
It is separate from the original last-token baseline documented below.
Dependency locks, offline CI on Linux/Windows, packaging artifacts and a manual
real-model CPU smoke workflow are included. Model weights, credentials, raw
third-party data and experiment runs stay outside Git.

This is research in progress. Software fixtures and small-model smoke checks
do not establish deception-detection performance. See [validation](VALIDATION.md)
for the imported starter's historical checks; current checks are recorded in
the repository research log.

The [genuine CPU smoke run](https://github.com/ADI-0519/Individual-Project/actions/runs/44525570337)
and local Qwen GPU check passed. The
[Llama 8B NF4/BF16 validation](docs/validation/2026-11-19-llama8b-4bit-smoke.json)
also passed, including actual quantization, layer indexing, saved-vector replay
and same-shape causal checks. The
[full instruction baseline](docs/validation/2026-11-19-llama8b-full-instruction-probe.json)
captured 536 dialogues and fitted the probe using 219 training / 49 validation
fact pairs. Its validation target is the instruction condition. The
[independent numerical audit](docs/validation/2026-11-19-independent-baseline-audit.json)
reproduces the fit and finds prompt-only AUROC 1.0 and response-only AUROC 0.5.
The [first reader/context pilot](docs/validation/2026-11-19-reader-and-context-pilot.json)
is complete: 48 genuine reader conversations and 48 fixed-report captures,
alongside exact GPU replay checks on a prespecified 22-dialogue subset. The
four development cases include a failed factual-restoration control. These
are feasibility results; natural pressure responses, audited behavior labels,
strong monitor comparisons and held-out family evaluation remain outstanding.

## Original sender–recipient starter

Working implementation for Variant 2: evaluating latent probes under perturbation and distribution shifts.

The experiment separates the sender's objective, the information attributed to a recipient, the actual message, and the recipient's eventual conclusion. It records model activations without treating task pressure, omission or a mistaken recipient as proof of intentional deception.

## Status

- Runnable sender and recipient command-line pipeline, frozen-model activation extraction, grouped linear-probe fitting and a text baseline.
- 24 **draft development scenarios**, six template families, three domains, balanced full-information YES/NO answers. Human semantic review is still needed.
- A four-example independent belief-task format template. It is deliberately too small for valid probe training.
- Software tests use explicit artificial fixtures. They are not LLM research results.
- The imported starter was tested without a model runtime. Its historical record is in [VALIDATION.md](VALIDATION.md); current evidence and remaining hardware limits are recorded in [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md).

The main first milestone is one real message, one saved activation, and a recipient response with a fully traceable input. Keep the submitted project scope and discuss the final protocol and ethics route with Hanqi.

## 1. Install the lightweight package and check the data

Open a terminal in this directory. Python 3.11 or 3.12 is a practical starting choice.

Ubuntu/WSL2:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m latent_probes validate-data
python -m unittest discover -s tests -v
```

On Windows PowerShell, create the environment with `python -m venv .venv`, then activate it with `& .venv/Scripts/Activate.ps1`. The remaining Python commands are the same. If your current environment already works, use a separate project environment rather than changing unrelated projects.

Preview the full experiment **without loading a model**:

```bash
python -m latent_probes plan --config configs/llama8b_4bit.json --output runs/pilot_plan.json
```

Expected plan: 24 scenario clusters, 288 sender generations and 720 recipient generations including the three control conditions. The plan writes prompts, not invented model responses. Repeated calls need a new output path.

## 2. Check GPU software before downloading the main model

```bash
nvidia-smi
python -m latent_probes doctor
```

The doctor's report includes package versions, GPU properties and an actual small CUDA matrix operation when CUDA is available. A GPU name alone is not proof that the installed PyTorch build supports it.

Use a driver compatible with the selected locked GPU profile. The explicit CUDA 13.0 profile is separate from the CPU engineering profile:

```bash
uv sync --locked --extra models --extra cu130 --extra quantization
uv run --locked --extra models --extra cu130 --extra quantization python -m latent_probes doctor
```

Check the official installation/release guidance against your operating system and driver. The package's version ranges express API requirements; they do not independently guarantee driver compatibility. For CPU-only debugging, use the corresponding CPU PyTorch installation instead. Large-model quantized configurations in this starter require CUDA.

Sources: [uv PyTorch profiles](https://docs.astral.sh/uv/guides/integration/pytorch/), [bitsandbytes installation matrix](https://huggingface.co/docs/bitsandbytes/installation), [CUDA in WSL](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl).

## 3. Run one genuine small-model example

The smoke configuration uses Qwen2.5-1.5B-Instruct, one objective, one audience condition and one greedy generation. It is a debugging checkpoint, not the main research model.

```bash
python -m latent_probes run --config configs/smoke.json --limit 1 --output runs/smoke
python -m latent_probes replay-recipients --run runs/smoke
python -m latent_probes summarise --recipients runs/smoke/recipients/default --output runs/smoke_summary
```

The first command downloads the model if it is not cached. Inspect:

- `runs/smoke/senders.jsonl`: exact prompts, generated text, token IDs and capture positions.
- `runs/smoke/features/`: selected activation vectors in portable NPZ files.
- `runs/smoke/annotations.csv`: blank fields for factuality and disclosure review.
- `runs/smoke/manifest.json`: resolved model revision, precision, settings, code fingerprint and environment.
- `runs/smoke/recipients/default/recipients.jsonl`: fresh recipient contexts and strict YES/NO/UNKNOWN/invalid answers.

The smoke test should produce one sender record and eight recipient records. A failure is diagnostic information: resolve it before moving to 8B. No mock-output fallback is provided by the command line.

## 4. Start the 8B pilot

Obtain access to [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) through your own Hugging Face account, and authenticate locally with `hf auth login`. Keep authentication out of source control.

Start with the first four scenarios:

```bash
python -m latent_probes run --config configs/llama8b_4bit.json --limit 4 --output runs/llama8b_four
python -m latent_probes replay-recipients --run runs/llama8b_four
python -m latent_probes summarise --recipients runs/llama8b_four/recipients/default --output runs/llama8b_four_summary
```

Review the messages, memory use and controls. Then use a new directory and omit `--limit 4` for all 24 development scenarios.

This captures decoder block 16 (zero-based), at two positions:

- `prompt_last`: the last token of the prompt, including the assistant generation header.
- `response_last`: the last non-special token of the naturally generated report.

The backend replays the exact generated token IDs through the frozen decoder and retains only the requested block vectors. It does not retain every layer's full sequence activations. Positions are before any elicited-forecast question. Layer 16 is an initial declared setting, not a result claiming it is the best layer.

To check the causal mask on the real installation, compare the prompt-position vector from a prefix-only forward pass to the corresponding position in the full generated-token replay, allowing a stated numerical tolerance. Stub tests cannot establish this property of an actual model runtime.

## 5. Annotation and probe fitting

Do not label every favourable instruction as deception. Inspect actual outputs and complete the relevant fields of `annotations.csv`:

- `factuality`: your audited factuality assessment.
- `critical_fact_disclosed`: 1 if the designated fact is clearly disclosed; 0 if absent; adjudicate ambiguous cases separately.
- `ambiguous`: an audit flag, not a silently excluded category.
- `label`: a binary target only after its meaning is written down.
- `annotation_notes`: reasoning and disagreements.

Omission of the designated fact can be harmless. The dataset contains an explicit example; see data/README.md. Record off-spec or false statements instead of silently dropping them.

Export one feature definition, joining annotation rows by ID:

```bash
python -m latent_probes pack-features --run runs/llama8b_four --layer 16 --position response_last --annotations runs/llama8b_four/annotations.csv --output runs/llama8b_four_features
```

For an adequate, independently collected and audited training dataset, the fitting interface is:

```bash
python -m latent_probes train-probe --features runs/training_features/features.npz --labels runs/training_features/metadata.csv --target label --output runs/probe_v1
python -m latent_probes predict-probe --features runs/evaluation_features/features.npz --probe runs/probe_v1 --output runs/evaluation_predictions.csv
```

The `training_features` and `evaluation_features` paths above are placeholders for runs you create. The 24 repeatedly inspected pilot scenarios are development data, not a final confirmatory test set. Four smoke examples are insufficient for fitting.

For a clearly named instruction-decoding diagnostic on the full pilot, `--target objective_pressure` uses the known 0/1 objective assignment. This tests instruction information. It is not a deception detector or the final behaviour result.

The fitter uses a single fixed family split, train-only standardisation, validation selection of regularisation, and held-out metrics. Its threshold is 0.5, not a calibrated 1% false-positive operating point. It refuses incomplete labels, overlapping family assignments, single-class partitions and incompatible capture signatures. It also fits a simple TF-IDF text baseline on the same split when the `text` column is present. That is an initial baseline; stronger matched-information monitors remain future evaluation work.

For `prompt_last`, the text baseline receives only the prompt. For `response_last`, it receives prompt plus generated response. `output_text` is provided separately. Original annotation exports contain output-only text; the feature export deliberately constructs the appropriate comparison text.

Read docs/PROBES.md before interpreting a score. Do not run many layers and select the one with the best test score; layer selection belongs on validation data.

## 6. Independent belief-task examples

An importer lets us collect separate training tasks on the same checkpoint:

```bash
python -m latent_probes extract-examples --config configs/smoke.json --examples data/probe_examples_template.jsonl --output runs/belief_format_check
python -m latent_probes pack-features --run runs/belief_format_check --layer 14 --position prompt_last --output runs/belief_format_features
```

These four template examples have only two families and cannot support a train/validation/test fit. Expand the design properly before training. Supplied labels concern a fictional agent's prior belief and are not replaced by the model's answer. The importer does not feed supervision labels to the model.

A neutral prior-belief probe does not establish anticipation of the effect of a new message. The latter remains a separate research question. Read docs/DATA_IMPORT.md and docs/EXPERIMENT_PROTOCOL.md.

## 7. Larger models and CREATE

Included configurations:

| Configuration | Intended use |
| --- | --- |
| `smoke.json` | Qwen2.5 1.5B loading and extraction check |
| `llama8b_4bit.json` | Local 8B development pilot |
| `llama8b_bf16.json` | Same-checkpoint higher-precision check on suitable hardware |
| `llama70b_4bit.json` | Conditional 70B replication, after memory and access checks |
| `llama70b_bf16.json` | Higher-memory, potentially multi-GPU 70B run |

No HPC job is submitted automatically. The Slurm template and verified King's access details are in docs/SCALING_AND_CREATE.md and scripts/create_gpu_job.sh. Its one-GPU/64GB-host-RAM/one-hour defaults are an editable 8B starting request, not a measured budget for a 70B study.

Train a new probe for each checkpoint. Equal hidden dimensions do not make two models' probe weights interchangeable. To isolate a sender-size change, replay its messages with the same recipient configuration:

```bash
python -m latent_probes replay-recipients --run runs/llama70b --config configs/llama8b_4bit.json --output runs/llama70b/recipients/fixed8b
```

Conversely, reuse a fixed sender run and change only the recipient configuration to test recipient sensitivity. Avoid describing both changing together as a pure sender-scale effect.

The supplied 8B checkpoint is Llama 3.1 and the 70B checkpoint is Llama 3.3. Their training differs as well as their size. Treat this as cross-checkpoint replication, not a controlled experiment proving that parameter count alone causes a difference. A same-release 70B comparison can be configured if that is the intended question.

## 8. Recovering an interrupted run

Repeat the identical command with `--resume`:

```bash
python -m latent_probes run --config configs/llama8b_4bit.json --limit 4 --output runs/llama8b_four --resume
python -m latent_probes replay-recipients --run runs/llama8b_four --resume
```

Finished records are skipped. Changed prompts, scenario snapshots, source code, package versions or incompatible captures are rejected. Use a new output directory for a changed experiment. Runs are single-process writers: do not launch two jobs against the same output directory. Keep a backup of durable results; cluster scratch storage is not a backup.

After a successful hardware setup, record its package environment:

```bash
python -m pip freeze > environment.freeze.txt
```

The repository excludes model weights, generated runs and credentials from Git. Seed recording aids reproducibility but does not promise bit-identical generation across GPU types or numerical implementations.

## What remains research work

The code is a starting artefact, not completed evidence of a novel detector. Remaining work includes human audit of the scenarios, validating behaviour variation, a sufficiently varied independent training dataset, adapting and checking the published baseline, meaningful held-out domains, stronger text comparisons, a valid message-conditioned belief target and any justified causal interventions. The code keeps these decisions explicit instead of filling them with unverified automatic labels.

See ACKNOWLEDGEMENTS.md for initial development provenance and the source papers. Keep a log of your own changes, decisions and understanding for the dissertation.
