# Starter validation record

Date: 18 November 2026.

## Checks completed

- Installed the package in editable mode with existing NumPy and scikit-learn, without fetching model dependencies.
- Python byte-compilation of source and tests completed successfully.
- **78 automated tests passed** using Python unittest.
- All five model configuration files loaded and validated.
- Dataset validation returned 24 scenarios, six families, three domains and 12 YES / 12 NO full-information answers.
- The offline full-pilot planner returned 288 sender generations and 720 recipient generations, with elicited forecasts disabled.
- The CLI help and environment inspection commands ran successfully.
- A model-run attempt without the optional model runtime failed clearly with an actionable installation error and produced no model outputs.
- The Slurm script passed bash syntax checking. Its direct-execution guard was checked during development.

## What the automated tests cover

Scenario references and draft controls; sender/recipient information separation; strict answer parsing; selected activation token indices and hook cleanup; exact feature/metadata alignment; missing or altered input records; resumable sender and recipient jobs; preservation of UNKNOWN and invalid outcomes; train-only preprocessing; grouped data splits; validation-only regularisation selection; portable saved-probe predictions; capture-signature mismatch rejection; and independent-task import without annotation leakage.

The sender/recipient integration tests inject an explicitly marked scripted backend. Activation-hook tests use small stubs. Probe tests use artificial numeric data. These are software correctness checks, not experimental evidence about language models.

## Environment actually used

- Python 3.12.14
- NumPy 2.3.5
- scikit-learn 1.8.0
- Linux x86-64
- No CUDA GPU, PyTorch, Transformers, Accelerate or bitsandbytes available in this development environment.

## Still to verify on real hardware

1. Driver / PyTorch / bitsandbytes compatibility on the user's RTX 5080 or allocated CREATE node.
2. Actual checkpoint download access, loading, chat-template handling and generation.
3. Actual decoder-block capture behaviour on the installed Transformers version.
4. Prompt-position invariance under a prefix-only forward pass versus exact generated-token replay, within a declared numerical tolerance.
5. Peak memory, throughput, truncated output rates and practical HPC resource requirements.
6. Semantic human audit of the draft scenarios and actual generated messages.
7. Behavioural and probe results on genuinely independent evaluation material.

No real 1.5B, 8B or 70B model experiment, publication-level baseline reproduction, validated belief readout or causal intervention result is claimed by this starter.
