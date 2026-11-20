# Latent probes under reporting pressure

[![CI](https://github.com/ADI-0519/Individual-Project/actions/workflows/ci.yml/badge.svg)](https://github.com/ADI-0519/Individual-Project/actions/workflows/ci.yml)

Research software for KCL Variant 2, by Aditya Ranjan. Start with an auditable
adaptation of Apollo's instruction-pair deception probe, then evaluate transfer
under reporting pressure and recipient-knowledge shifts.

The main research question is whether a latent reader-outcome readout trained
under neutral communication transfers to pressured natural reports and improves
on text/source controls. Apollo and fixed-report replays are comparators and
diagnostics. See [research direction](docs/RESEARCH_DIRECTION.md).

**Project status: early research development.** The engineering baseline and
small pilot are complete; the main study and dissertation are not. Validated
targets, human-reviewed data, competitive frozen controls, independent family
holdouts, a second reader and the final analysis/write-up remain ahead. The
[main-study gates](docs/MAIN_STUDY_GATES.md) define what counts as completion.

**Latest communication milestone:** [144 naturally generated reports on 24 fresh
decision cases](docs/COMMUNICATION_TRANSFER_RESULTS.md), with actual readers,
frozen neutral-trained outcome readouts and score-blind draft content review.
The activation readout gets 62/72 neutral and 66/71 pressured outcomes correct;
always predicting APPROVE gets 63/72 and 66/71. Zero REJECT recall and missing
DEFER outcomes prevent an activation-advantage claim. Reader capability checks,
and exact-prefix monitor recovery are complete. The independent Qwen reader
failed qualification; a single fixed nine-shot policy on 24 separate fresh cases
scored 24/24 informed, 17/24 missing-evidence and 24/24 stale controls. It failed
the predefined missing-evidence threshold, so new recipient-specific training
and untouched natural-transfer evaluation **were not run**. See the
[terminal follow-up record](docs/validation/2026-11-20-reader-followups.json).
The location/belief-query studies
below remain auxiliary diagnostics, rather than the main empirical result.

**Controlled failure finding:** the separate [96-condition evidence-role
diagnostic](docs/EVIDENCE_ROLE_DIAGNOSTIC_RESULTS.md) gets 32/32 decisive-evidence
controls correct, yet 0/32 metadata-only inputs produce warranted deferral.
All eight date pairs follow an irrelevant registration date, even when the
completion date is explicitly unavailable. This is a reproducible behavioral
pattern under one fixed Qwen policy, not a demonstrated internal mechanism or
novel latent-monitor advantage. The failed qualification and stopped fitting
remain unchanged.
The one independent Llama replay differs: explicit missingness produces 32/32
deferrals, but decisive-evidence controls fall to 19/32. The Qwen value-following
pattern is therefore model-specific here; two checkpoints do not establish a
general internal mechanism.

**Start here:** [Apollo quickstart](docs/APOLLO_QUICKSTART.md) ·
[paper/source audit](docs/APOLLO_METHOD.md) · [reading log](docs/READING_LOG.md) ·
[starter audit](docs/STARTER_AUDIT.md) ·
[pressure-pilot protocol](docs/PRESSURE_PILOT.md).

**Current research decision:** [direction and scope](docs/RESEARCH_DIRECTION.md),
[deep literature audit](docs/LITERATURE_PROBE_AUDIT.md), and
[recipient-context experiment](docs/RECIPIENT_CONTEXT_DESIGN.md).
Read the [measured pilot results and next empirical gates](docs/PILOT_RESULTS.md)
before scaling the case bank.

The [first natural neutral-to-pressure reader-forecast pilot](docs/NEUTRAL_FORECAST_RESULTS.md)
is now complete: 84 natural reports and reader labels, a prospectively frozen
three-class readout, and text/source controls. Activation accuracy is 10/12 in
each test condition across four cases, while an exploratory literal-text rule
gets 11/12 under pressure. Added value beyond report text is not established.

The [next empirical checks](docs/RESEARCH_CHECKS_RESULTS.md) are also complete:
a separately fitted **before-report** readout gets 8/12 neutral and 1/12 under
pressure on those same four development-test cases. A paired sender-instruction
ablation changes reports and reader decisions on eight training cases. The
[84-report draft audit](docs/REPORT_AUDIT_DRAFT.md) and
[24-report ablation audit](docs/SENDER_POLICY_AUDIT_DRAFT.md) need human review.
These are exploratory diagnostics, with retained failures, not a finished study.

The [20 November belief diagnostic](docs/BELIEF_DIAGNOSTIC_RESULTS.md) directly
separates world truth, actual reader perspective and nested sender-attributed
perspective. The earlier outcome-forecast failure is not an absence test for
second-order beliefs. The new query-free nested heads remain near chance;
the [primary-methods audit](docs/BELIEF_TARGET_AUDIT.md) and capability controls
explain the next validity checks before communication transfer.

The [controlled belief-question comparison](docs/QUERY_COMPARISON_RESULTS.md)
and frozen fresh-wording evaluation are now complete. On 32 fresh sender-view
rows, the primary nested readout scores 16/32 before a question, 24/32 after the
nested question, and 16/32 after the world-question control. Its unchanged
question-conditioned head scores 16/32 on story-only states. This is conditional
decodability across four wording groups, with uneven class recall; it does not
yet validate an unqueried belief readout for pressured communication.

The [known-message transfer test](docs/KNOWN_MESSAGE_RESULTS.md) now gives a
positive conditional result: the unchanged queried block-16 head gets **56/64**
on a new message/receipt mechanism, versus 32/64 for story/world-question heads.
Direct answers get 44/64 and structured answers 32/64. These are four wording
groups and 32 unique sender inputs, with human semantic review still required.
Unqueried transfer, natural pressure transfer and causal belief use remain open.

The [exploratory future-goal context stress test](docs/BELIEF_GOAL_STRESS_RESULTS.md)
now exposes a robustness limit on those already-observed cases. The same frozen
head scores 44/64 with a neutral sender-role scaffold, 50/64 with a future
positive-belief goal and 34/64 with the opposing goal. All 32 unique matched
inputs shift toward the positive goal in probe probability. No answer or report
was generated, current gold labels stayed fixed, and role, wording and polarity
confounds prevent interpreting this as changed belief or deceptive intent.

The [real MASK specificity pilot](docs/MASK_SPECIFICITY_RESULTS.md) is also
complete: 120 generations and 88 score-blind draft annotations. Strict primary
eligibility leaves no clean behavioral comparison. A documented secondary
aggregation retains four pressured honest and five opposing assertions, with
pooled AUROC 0.50 and only one within-case contrast. These small draft-label
results do not establish a general honest-response false-positive rate.

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
are feasibility results. The newer natural pressure run is documented above;
audited behavior labels, stronger monitor comparisons and held-out family
evaluation remain outstanding.

## Original sender–recipient starter

Working implementation for Variant 2: evaluating latent probes under perturbation and distribution shifts.

The experiment separates the sender's objective, the information attributed to a recipient, the actual message, and the recipient's eventual conclusion. It records model activations without treating task pressure, omission or a mistaken recipient as proof of intentional deception.

## Status

- Runnable sender and recipient command-line pipeline, frozen-model activation extraction, grouped linear-probe fitting and a text baseline.
- 24 **draft development scenarios**, six template families, three domains, balanced full-information YES/NO answers. Human semantic review is still needed.
- A four-example independent belief-task format template. It is deliberately too small for valid probe training.
- Software tests use explicit artificial fixtures. They are not LLM research results.
- The imported starter was tested without a model runtime. Its historical record is in [VALIDATION.md](VALIDATION.md); current evidence and remaining hardware limits are recorded in [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md).

The initial real-message/activation milestone is complete. The current gate is
validating natural reports, recipient outcomes and predictive value beyond text
before a fresh family-held-out study. Keep the submitted project scope and
discuss the final protocol and ethics route with Hanqi.

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
