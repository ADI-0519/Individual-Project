# Fresh study: interrupted worker and preserved-data continuation

23 November 2026. This records a runtime interruption after the
[startup correction](FRESH_READOUT_STARTUP_RECOVERY_V1.md), under the unchanged
[scientific execution protocol](FRESH_READOUT_EXECUTION_PROTOCOL_V2.md).

## What stopped

At 23:56 UTC on 22 November, a live status check found no experiment producer.
Both earlier execution-session identifiers were gone, and WSL had a fresh boot.
The original campaign state still said `running`, but its last Qwen progress
write was 22:55:45 UTC. The separate completion watcher last polled at 22:55:30.
Windows recorded deletion of the WSL virtual-network port at approximately
22:57:22. These observations establish that execution stopped; they do not
establish the cause of the worker termination.

The earlier finish estimate was based on progress files without a sufficient
live-process check. It must not be reused. A stale `running` field is not
evidence that an experiment continues after a chat turn.

## Preserved measurement boundary

The interrupted native root is
`/home/adirj/kcl-latent-probes-runs/fresh_readout_v2_20261122_recovery_v1`.
It must remain unchanged, including its stale locks, manifests, logs and original
completion-watcher receipt. The initial inventory contains:

| Cohort | Completed journals | Remaining cohort inputs |
| --- | ---: | ---: |
| Llama training/development | 2,048 | 0 |
| Qwen training/development | 900 | 1,148 |
| Llama fresh test | 0 | 4,992 |
| Qwen fresh test | 0 | 4,992 |

There are no started-only journals, partial raw arrays or recorded failed rows
in that inventory. There are no fitted heads, test bindings, test forwards or
analysis outcomes. The continuation therefore requires no repeated measured
or unknown case. These counts are a required boundary, not a substitute for
validation: an independent audit and the recovery preparer must verify every
completed journal, both saved engineering checks and Llama's aggregate cache.

Engineering-logit summaries must be checked under the original four-thread
numerical environment. A preserved first read-only audit exposed a default
OpenBLAS-32 entropy-reduction difference; the unchanged checker validates both
engineering records with the original thread limit of four. No scientific
data, summary or validation tolerance is changed to accommodate that difference.

## Versioned continuation

Use a new native root, `fresh_readout_v2_20261123_recovery_v2`. Preparation must:

1. Establish that no experiment producer remains.
2. Verify the exact 2,048/900 completed-row boundary and reject any unknown,
   failed, unexpected, partial, fitted or test artifact.
3. Copy unchanged protocols, preparation receipts, engineering records,
   completed Llama capture and completed Qwen journals, preserving their bytes.
   Original stale locks remain in the original directory only.
4. Preserve the original Qwen training authorization and protocol hash. Resume
   with the unchanged capture runner's explicit `resume` mode.
5. Record retained/remaining IDs and source/copy hashes before any new forward.

The separate continuation wrapper adopts the verified completed Llama stage,
resumes only Qwen's 1,148 unattempted training/development inputs, and composes
the original ten-stage plan for all later work. Both fitting bundles still
precede either fresh test forward. The original probe-fitting, selection,
analysis, reporting and independent-audit sources remain unchanged. No new
site, prompt, label, model, threshold or outcome-dependent exclusion is allowed.

## Process lifetime and health checks

Use a least-privilege Windows worker independent of the chat and terminal
lifetime. The worker must have no automatic
retry, archive its launch inputs, set the original numerical/cache environment
before Python imports, and keep the foreground WSL command alive through
capture, fitting, analyses, independent audit and report export.

Before launching the study, test that method with a harmless timed heartbeat
outside the chat process tree and without an attached terminal console.
Record task identity, timestamps, parent process chain and exit status. This
checks the chosen lifetime mechanism; it cannot protect against shutdown,
logoff or an explicit task/WSL termination.

A Task Scheduler console canary stopped at 00:02 with a console-interruption
exit code; the user subsequently confirmed accidentally closing a terminal
then. A separate unsigned GUI-host option was blocked by Windows application
control and was not used. Both engineering attempts remain preserved. Neither
is a successful durability test or a reason to weaken a security policy. The
22:55 experiment stop remains unexplained.

Future status reports must check the Windows worker/task, live Linux producer,
and recent progress timestamps together. A stopped task invalidates its ETA.
Failure must preserve all outputs and stop for investigation, without an
automatic scientific retry. The final interpretation remains
[agent-audited exploratory research](FRESH_READOUT_INTERPRETATION_RULES_V2.md).
