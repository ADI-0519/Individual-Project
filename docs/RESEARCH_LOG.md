# Research and implementation log

## 19 November 2026 — Apollo foundation

Imported the supplied `kcl_latent_probes_starter.zip` into the existing empty,
private `ADI-0519/Individual-Project` Git repository. Original archive SHA-256:

`5CDFF7B5B6D45075852319D64EF043339CF54CE877B9004831FDFD722C7447FE`

The ZIP was inspected before extraction and paths checked against the repository
root. Existing files were not overwritten. Its original acknowledgements and
historical validation record remain intact. The imported source was generated
with ChatGPT assistance; the changes in this entry were developed with Codex
and three parallel agents. The student must understand, evaluate and describe
their independent contribution under the module's actual policy.

### Evidence that changed the implementation

- The original last-token classifier is not the saved Apollo instruction-pair
  baseline. Added a separate token-matrix capture/training/scoring path.
- Full Apollo and MASK papers, including appendices, were read for the baseline
  audit; relevant Bürger and Natarajan methods were checked. Cross-checked the
  saved Apollo configuration and executable source at commit
  `f8ec4010e74927394709dffa22b97bdf8cd5a62f`.
- Pinned source uses five **words**, despite paper prose saying tokens; uses HF
  `hidden_states[22]`, not decoder block 22; and fits `C=0.1` without a separate
  classifier intercept. Evaluation averages raw token margins.
- Actual pinned data: 612 rows, 306 true facts; 31 zero-prefix facts and eight
  later duplicate occurrences overlap once. Explicit exclusions after the
  original split yield 219 train / 49 validation pairs. The earlier hypothetical
  410/102 calculation assumed 512 available facts and does not describe this file.
- Raw data SHA-256:
  `924709159b40a30d5b51488e9ccd7fd886456a89f0e0f5849eacefba6ff55312`.
- Reproduced an existing label-semantics bug: heterogeneous targets were accepted
  as one binary label. Added failing regression tests before fixing validation
  and preserving manual target definitions through annotation export/packing.

See [method audit](./APOLLO_METHOD.md), [reading log](./READING_LOG.md) and
[starter audit](./STARTER_AUDIT.md) for the detailed distinctions.

### Reproducibility and software checks

Added a Python 3.11 development environment, `uv.lock`, explicit CPU/CUDA profiles,
Ruff formatting/lint, offline tests, locked packaging, pinned GitHub Actions,
artifact retention, contribution/citation/security metadata and credential/data
ignore rules. No project or upstream-data licence was invented. No model weights,
tokens, private material or generated runs are source-controlled.

Local offline suite initially passed 113 tests and 48 subtests (72% aggregate
branch-aware coverage), then received additional data-quality regression checks.
The final CI run is the authoritative check for the committed version. Tests use
marked fixtures; their passing counts are not model research results.

### Real-model execution status

Local NVIDIA inspection found an RTX 5080. The initially installed PyTorch wheel
was CPU-only. A real small-model capture attempt failed before model loading:
Windows Application Control blocked the `regex` extension DLL imported by
Transformers. The same import was blocked outside the sandbox. No protection was
disabled and no fake response or activation was substituted. The failed local
manifest remains an ignored engineering record under `runs/`.

Added a manual Linux CPU smoke workflow with a fixed public Qwen 0.5B checkpoint,
14 paired training examples, six freshly generated neutral controls and real
hidden-state/prefix-invariance checks. Its outcome is recorded by the Action and
`runs/apollo_smoke_validation.json`, not assumed here. Any successful run is an
engineering validation, not an 8B detection result or a reliable 1% FPR estimate.

### Scope that remains

Obtain personal gated access to Llama 3.1 8B, validate the approved GPU runtime,
run a declared precision/depth profile, collect independent control/evaluation
responses, audit behavioral labels and freeze the held-out protocol. The reader
forecast and omission extensions remain research tasks; the 24 draft starter
scenarios must not be promoted to validated deception examples.

### Tool and plugin use

Used the graphify skill's inspection guidance and the local Python
fact-grounded-coding skill. No existing graph was present, so focused source
inspection was used. No connected-app plugin, Obsidian integration, browser
automation or Playwright was needed. Python, Git, GitHub CLI and Actions are
development tools in this record, not additional app plugins.
