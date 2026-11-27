# Proposed CREATE precision replication of format adaptation

27 November 2026. **Planning document v1: not an executable protocol, implemented
runner, submitted job or completed experiment.** The completed NF4 study and its
[interpretation](FORMAT_ADAPTATION_RESULTS_INTERPRETATION_2026-11-27.md) remain
unchanged. This proposal supplies a concrete resource justification and design
for supervisor review. It does not claim current CREATE access or reserve GPU
time. The implementation and final execution specification must be versioned
and independently reviewed before measurement.

## Scientific question and cohort labels

Does the measured benefit of mixed-format fitting persist with BF16 weights,
when the comparison uses the same observation allocations, model revisions,
capture sites, fitting policy and fully crossed test cases? Does a reference
NF4 run on the same CREATE hardware recover the previous pattern? A secondary
question asks whether an NF4-fitted instrument can be transported unchanged to
BF16 activations. Refitting within BF16 and transporting an NF4 instrument are
different experiments and must have separate result labels.

The first proposed campaign reuses the existing forty adaptation contexts. Its
twenty-four test contexts are already exposed. Call this a **paired technical
precision replication on the existing bank**, not a new held-out-context
confirmation. New numerical measurements do not make the text or formal states
unseen to the researchers. The old results must remain visible alongside it.

A later confirmation requires a separately versioned, genuinely fresh fit/test
bank, human review, a frozen allocation and all methods/settings fixed before
new outcomes. Independent authorship is a separate property: it cannot be
claimed merely because prompts have new names or an agent wrote another bank.
The current proposal does not create that bank or authorize a new result search.

## Prerequisites before cohort measurement

- Complete human semantic review of the relevant attribution, reading-record
  chronology and correction cases. The existing blind agent agreement is useful
  engineering evidence and cannot be substituted for human judgment. Record
  the reviewer, coverage and disagreements. Preserve failures and original data;
  substantive corrections require a new bank/version and a revised replication
  label. Do not silently delete cases after their outcomes are known.
- Agree the bounded Variant 2 scope with the supervisor: reliability of a
  queried fictional attributed-belief instrument under format/precision shift.
  Real deception and spontaneous anticipation remain unestablished extensions.
- Implement a new versioned precision-aware capture/fitting/audit path. The old
  adaptation code intentionally binds an old NF4 feature map and runtime
  identity. Existing generic job examples or changing a configuration field are
  insufficient to implement this design safely. No new Slurm script is claimed
  ready in this document.
- Freeze the exact bank, old preprocessing-reference IDs, tokenizer/revision
  identities, source commit, package lock, arithmetic profiles, prefill settings,
  attempt policy, endpoints and auditor before new cohort captures. Verify all
  prefixes and spans without consulting new outcomes.

Human review is a proposed scientific qualification gate for this next campaign;
it does not retroactively turn the completed agent-audited study into human-
validated evidence. Infrastructure profiling may proceed on excluded engineering
prompts once access exists. Those prompts cannot become cohort observations.

## Paired hardware and numerical profiles

Run both profiles sequentially on the same allocated CREATE GPU type and pinned
software stack, one model loaded at a time. Prefer one A100 with 40 GB device
memory for initial profiling. Verify the actual allocated GPU/VRAM rather than
assuming all GPU allocations have that specification. Record CUDA, driver,
PyTorch, Transformers, bitsandbytes, attention implementation, GPU model and
device properties in every capture manifest.

| Setting | CREATE NF4 reference | CREATE BF16 replication |
| --- | --- | --- |
| Weights | NF4 with double quantization | BF16 weights, no weight quantization |
| Compute | BF16, with explicit kernel/runtime record | BF16, with explicit kernel/runtime record |
| Attention / batching | SDPA / batch one | Same requested SDPA / batch one |
| Generation / truncation / offload | None / none / none | None / none / none |
| Maximum token input | 2,048 | 2,048 |

Pin Llama-3.1-8B-Instruct revision
`0e9e39f249a16976918f6564b8830bc894c89659` at the existing block-16
query-location site and Qwen2.5-7B-Instruct revision
`a09a35458c702b33eeacc393d103063234e8bc28` at the existing block-21 site.
Keep the same pre-final-normalization hook meaning, tokenizer and rendered input
bytes. Do not introduce a new site search. Validate hook alignment and actual
parameter/compute dtypes on excluded engineering inputs.

The same hardware comparison reduces the confounding caused by moving from a
local RTX 5080 to an A100. It does not isolate an abstract bit-width variable:
quantized and BF16 kernels differ. Describe the measured contrast as a specified
precision/runtime-profile change. If an attention kernel or package cannot be
matched, preserve the failure, version the deviation before outcomes and limit
the claim accordingly.

## Profile-specific training-only feature maps

For each checkpoint and arithmetic profile, recapture **the exact 512 historical
C2 correction-training reference prefixes** used to estimate the old feature
map. The new implementation must resolve their IDs and hashes from the archived
training manifest, verify they are the actual scaler-fitting rows, and establish
that none belong to either adaptation test cohort. Stop if this provenance
cannot be reconstructed; do not choose convenient replacement rows.

Fit mean and population standard deviation in float64 using only that profile's
512 reference vectors, with the existing constant-feature handling. Freeze that
map, then standardize and apply row L2 normalization in the established order.
All four fitting arms within the checkpoint/profile share the same frozen map.
Reference labels are not used to select a new C, site or threshold. They are
shared historical training information and must be reported in addition to the
512-observation adaptation budget.

**Do not apply the saved local NF4 mean/scale to BF16 features and call that an
equivalent BF16 refit.** The main BF16 instrument uses a BF16 training-reference
map. Similarly, the CREATE NF4 reference gets its own map from the same reference
prefixes. Copying an old numerical map would instead introduce an additional
preprocessing-transport question. The proposed per-profile recapture preserves
the original shared-historical-training structure without giving only Mixed
additional unlabeled A renderings from its new fitting states.

## Fitting arms, capture inventory and seals

Retain the exact sixteen fitting and twenty-four evaluation contexts and all
state allocations in the
[original execution specification](FORMAT_ADAPTATION_EXECUTION_PROTOCOL_V1.md).
Keep A512, Mixed512, Ahalf256 and Offset. Mixed uses the same 512 underlying
states as A512, with complementary 256 A/256 B renderings; Offset freezes the
Ahalf256 coefficients and fits B-specific factor intercepts on Mixed's same B
observations. The within-context format/world dependency remains a limitation.

Retain four factor targets, the supplied fold, strict positive-margin decision,
float64 lbfgs fitting, fixed C policy, convergence/failure gates and unpenalized
offset optimization. No new search or test-dependent threshold is allowed.
Profile-specific fits must be sealed for both checkpoints and both profiles
before any new evaluation forward or output-logit inspection. Preserve started,
raw, completed and failed attempt states and prohibit blind retries of unknown
attempts.

| Prefix inventory per checkpoint and profile | Count |
| --- | ---: |
| Historical training-only preprocessing reference | 512 |
| Adaptation fitting captures, all required A plus B replacements | 768 |
| Existing-bank test captures, all conditions and both formats | 4,992 |
| Total reference plus cohort forwards | 6,272 |

Two checkpoints and two arithmetic profiles therefore require **25,088 planned
reference/cohort forwards**, plus separately counted excluded engineering
prefills. This is a count, not a wall-time estimate. Reusing saved local NF4
test features instead would not supply the same-hardware reference. A model's
profiles may share byte-identical token inputs; they do not share activation
vectors or fitted maps.

The named frozen-instrument transport diagnostic can score the new BF16 test
vectors with the CREATE NF4 map and coefficients. Preserve the NF4 map in that
diagnostic explicitly; do not silently replace it with BF16 statistics. It needs
no extra model forwards once both profiles have been captured. It is secondary
and must not replace a weak within-BF16 primary.

## Endpoints, uncertainty and independent reconstruction

The proposed single primary remains **Llama, BF16 format B hard complete-quartet
accuracy: Mixed512 minus A512**. Name Qwen's corresponding comparison secondary.
Preserve Mixed-minus-Offset, source-format A cost, all factor discrimination and
recalls, world/name flips with both-correct pairs, easy/one-step/asserted controls
and full-denominator output-token diagnostics. Keep failures and invalid outputs.
No free-generation or natural deceptive-behavior conclusion follows.

Use paired whole-context resampling, with the same draw used across arms and
profiles. Retain 10,000 draws and the existing seed as a replication convention.
Report the paired difference between the BF16 and CREATE NF4 adaptation effects
as a named precision diagnostic. The twenty-four contexts are shared across
profiles and checkpoints; they are not ninety-six independent context units.
Intervals condition on the bank, fixed fits and checkpoints, and exclude new
training-cohort uncertainty. A nonsignificant profile difference does not
establish equivalence; no equivalence margin has been justified here.

An independently authored checker should reconstruct profile-specific reference
means/scales, all fitted margins and all metrics before opening the saved
analyses. Bind reference IDs, profile identity, coefficients, prediction files
and full denominators. Exact equality is expected within a scored-data replay;
cross-profile activations need not match numerically. A hardware/reference
disagreement is reported and investigated as a deviation, never removed by
selecting a more favorable run.

## Resource request and current King's guidance

Request access for a bounded unquantized replication and paired reference, not
because use of a cluster itself establishes research quality. BF16 8B weights
alone occupy roughly 16 GB in decimal units before activation and runtime
workspace. The local 16 GB GPU leaves little room at the stated input length;
a 40 GB A100 is a reasonable profiling target. Final GPU-hours and host-memory
requests must follow measured peak memory and seconds per representative prefix,
including the longest inputs and both checkpoints. Do not invent a completion
time from the number of cases.

King's [current access guidance](https://docs.er.kcl.ac.uk/CREATE/requesting_access/),
checked 27 November 2026, requires staff to request student access. Taught-student
access is permitted when contributing to an existing King's research project;
the PI requests the account with student details and project affiliation.
Eligibility and affiliation therefore need confirmation by the supervisor/PI;
this proposal cannot guarantee an undergraduate account. Standard individual
allocations are 50 GiB home and 200 GiB scratch. No request or message has been
sent by writing this plan.

The [GPU documentation](https://docs.er.kcl.ac.uk/CREATE/hpc/running_jobs_gpu/)
explains GPU partition requests and querying the available GPU constraints.
Use one GPU at a time for the proposed serial capture; verify its feature and
memory record. The
[scheduler policy](https://docs.er.kcl.ac.uk/CREATE/hpc/scheduler_policy/)
currently gives ordinary CPU/GPU jobs a 48-hour maximum and a per-user ceiling
of eight simultaneous A100s. These are shared service limits, not a requested
allocation or an availability promise. Interruptible resources may be cancelled;
use them only after row-level recovery behavior is validated.

Use allocated scratch for model caches and active journals, with durable copies
of source/input manifests and completed results elsewhere. King's
[storage guidance](https://docs.er.kcl.ac.uk/CREATE/hpc/storage/) states that
scratch is not backed up. Profile cache plus journal volume before selecting a
project-storage request; keep checkpoints subject to their original model
licenses and access permissions. No credentials belong in the repository.

## Concrete next implementation decision

Finish the no-GPU lexical control and human semantic review while agreeing this
resource request. Then implement and test a new profile-aware preparation,
capture, train-only map, sealing and independent-audit path. Freeze the final
execution protocol after a CREATE engineering profile and before cohort
measurement. Preserve this v1 proposal as the decision record; it is not evidence
that the precision replication has already run.
