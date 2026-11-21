# CREATE request: additional control evidence

21 November 2026. Prepared for supervisor discussion; no request submitted,
account provisioned, allocation obtained or message sent by this work.

The [fresh three-predicate validation](RECIPIENT_RISK_V2_RESULTS.md) makes the
measurement problem concrete. The pinned quantized Qwen reader correctly
answers all twelve informed no-report controls, but only seven after the full
source is added. Overall full-disclosure correctness is 19/36; absent-evidence
deferral is 2/12. Zero invalid outputs and preserved exact inputs make parser
failure an inadequate explanation. They do not identify an internal cause.

This strengthens the reason for the already prepared, bounded
[paired precision reference](CREATE_COMPUTE_REQUEST.md). It does not establish
that NF4 caused the errors or that BF16 will improve them. Run both profiles on
the same CREATE GPU, checkpoint, exact prompts, policy, decoding and software;
a local-NF4/cluster-BF16 comparison would also change hardware/runtime.

**The original first experiment stays fixed:** 72 earlier `cv` control prompts
times two profiles, at most two allocated GPU-hours initially. The new v2 bank
does not silently replace those inputs or enlarge the request. Both observed
banks remain development evidence and cannot independently qualify a reader
selected using their failures. A future choice must be checked on fresh cases.

King's official pages, rechecked on 21 November, list A100 40 GB standard GPU
nodes and require student access requests through supervising King's/BRC staff.
For a taught undergraduate, access depends on contribution to an eligible
existing research project with PI support. Confirm eligibility/project and the
supported CUDA environment with the supervisor/ARC. The resource plan is not
evidence of access or hardware availability.
[Compute specifications](https://docs.er.kcl.ac.uk/CREATE/hpc/compute_nodes/)
and [requesting access](https://docs.er.kcl.ac.uk/CREATE/requesting_access/).

The first deliverable is a paired behavior table with complete outputs,
environment identities, invalids, deferrals, decision changes, peak GPU memory
and measured loading/inference time. No training or transfer of the existing
NF4 probe coefficients to BF16 coordinates is proposed. If errors persist,
report that outcome and reconsider the instrument rather than expanding compute
until a preferred result appears.
