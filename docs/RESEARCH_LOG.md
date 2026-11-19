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
branch-aware coverage), then reached 116 tests after further data-quality and
source-provenance regression checks. All four jobs in the first
[GitHub CI run](https://github.com/ADI-0519/Individual-Project/actions/runs/44525529169)
passed, including Linux Python 3.11/3.12 and Windows Python 3.11. The final CI run
is the authoritative check for the committed version. Tests use marked fixtures;
their passing counts are not model research results.

### Real-model execution status

Local NVIDIA inspection found an RTX 5080. The initially installed PyTorch wheel
was CPU-only. A real small-model capture attempt failed before model loading:
Windows Application Control blocked the `regex` extension DLL imported by
Transformers. The same import was blocked outside the sandbox. No protection was
disabled and no fake response or activation was substituted. The failed local
manifest remains an ignored engineering record under `runs/`.

The [first manual Linux CPU smoke](https://github.com/ADI-0519/Individual-Project/actions/runs/44525570337)
passed on commit `d57dbbe9a02aeb44d7178006b4eb4c15c86b52b8`. It loaded public
Qwen 0.5B revision `7ae557604adf67be50417f59c2c2f167def9a775`, captured 14 paired
examples, fitted the readout and generated/scored six neutral controls. The
real decoder hook matched HF hidden-state index 7. Prefix-only versus full-context
activation maximum absolute error was `1.1324882507324219e-06`, below the declared
`1e-4` absolute/relative tolerance. The small
[original validation summary](./validation/2026-11-19-qwen-cpu-smoke.json) is
preserved here; full trace artifacts were downloaded to ignored local runs.
This is engineering validation, not an 8B detection result or a reliable 1% FPR.

Final independent review then found that arbitrary custom CSVs could inherit a
pinned source-revision claim. A regression reproduced this; preparation now
verifies source bytes, marks custom data explicitly unverified, and propagates
the preparation manifest into captures. Missing calibration sidecars now carry
an explicit unverified-provenance warning. The normal first smoke used the
verified download; its measured activation result is unaffected by these guards.

### Scope that remains

Finish genuine Llama 3.1 8B execution and validate its declared
precision/depth profile, collect independent control/evaluation
responses, audit behavioral labels and freeze the held-out protocol. The reader
forecast and omission extensions remain research tasks; the 24 draft starter
scenarios must not be promoted to validated deception examples.

### Local WSL GPU validation

The separate Ubuntu environment completed installation with Python 3.11.17,
PyTorch 2.14.1+cu130 and CUDA 13.0. The actual CUDA kernel check passed on the
RTX 5080 (compute capability 12.0). The first Qwen forward pass then exposed a
missing C compiler required by Triton. Installed Ubuntu `build-essential` and
`python3-dev` through the package manager; no application protection was changed.
The failed capture manifest remains in ignored local runs. A fresh run used
the same pinned Qwen checkpoint and the new float32 CUDA config.

The complete local engineering pipeline passed: 14 instruction examples yielded
84 token vectors (78 training / 6 validation), the readout fitted, and six
neutral prompts generated 102 scored token vectors. The probe and controls
shared the recorded feature signature. The validator reloaded the actual GPU
checkpoint and checked one saved dialogue: all selected vectors exactly matched
HF hidden-state index 7, and prefix-only replay of its first and last selected
tokens also had maximum absolute error 0.0. The prespecified absolute/relative
tolerance was 1e-4. These checks cover one dialogue and its selected positions;
broader prompt coverage remains unevaluated.

The [GPU validation record](./validation/2026-11-19-wsl-gpu-smoke.json) preserves
the runtime, checkpoint, layer, dtype and numerical checks. The quickstart now
keeps the WSL CUDA environment separate from Windows development and CPU
profiles. The validator retains its original CPU defaults and accepts explicit
paths for this GPU run. Two agents contributed to the validation implementation
and the remaining 8B readiness review; no connected-app plugin was needed.

These are Qwen 0.5B setup results. The two held-out instruction examples are
not an estimate of deception performance. The six-control 99th percentile is
an explicitly unreliable engineering calibration, not a reliable 1% FPR claim.
A genuine 8B quantized load, its memory profile and independently audited
behavioral evaluation remain to be run.

### Llama access and the first 8B run

The user's Hugging Face login succeeded in Ubuntu. The optional Git credential
helper warning did not prevent authentication; downloading the exact Instruct
repository's `config.json` succeeded through the saved token. No global Git
credential configuration was changed, and no token was read into the chat or
committed. The base and Instruct repositories are distinguished in the quickstart.

Added an explicit small-run config pinned to resolved model revision
`0e9e39f249a16976918f6564b8830bc894c89659`, with NF4 loading, BF16 compute and
decoder block 8. The genuine 14-example extraction was launched with the same
verified seven fact pairs used for the engineering check. The original weight
download is approximately 16 GB even though GPU loading is quantized. **At this
entry's commit, the weight download is still running; no successful 8B load,
activation result or probe fit is claimed.** The continuation process waits for
the existing capture to complete, then fits the instruction probe and runs the
dedicated validator. Its ignored pipeline status and validation outputs record
success or failure independently of this journal entry.

The new Llama validator was reviewed by a second agent; imports/CLI help, Ruff
and editor diagnostics pass. Genuine-model checks remain pending. Its numerical
criteria were fixed before observing any 8B activation result: exact equality
between a block-8 hook and HF hidden-state index 9 in the same forward; exact
unchanged repeats; and exact invariance at first/last selected positions when
only later token IDs change in separate, same-length replays. Reloaded versus
saved BF16 values may differ by at most one adjacent representable BF16 value
per coordinate, with errors and out-of-bound counts reported. No float32
tolerance is reused or widened after seeing the results. The validator also
requires actual bitsandbytes 4-bit modules, finite 4,096-dimensional vectors,
matching masks/IDs, and one generated/scored neutral answer. That answer will
exercise scoring; it cannot establish a detection metric or calibration.

### Tool and plugin use

Used the graphify skill's inspection guidance and the local Python
fact-grounded-coding skill. No existing graph was present, so focused source
inspection was used. No connected-app plugin, Obsidian integration, browser
automation or Playwright was needed. Python, Git, GitHub CLI and Actions are
development tools in this record, not additional app plugins.
