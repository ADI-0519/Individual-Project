# Format-adaptation campaign continuation runbook

Written 27 November 2026. This is a new, unbound operational handoff for an
already running campaign. It does not amend the sealed scientific sources,
tests, inputs, execution protocol or analysis definitions. The governing design is the
[fixed execution protocol](FORMAT_ADAPTATION_EXECUTION_PROTOCOL_V1.md).

## Identity and locations

| Item | Fixed value |
| --- | --- |
| Frozen source commit | `40ae42efe73e7cdfa0917eb89a5d4697bee3ca75` (`40ae42e`) |
| Native campaign | `/home/adirj/kcl-latent-probes-runs/format_adaptation_v1_20261127_campaign` |
| Equivalent Windows UNC | `//wsl.localhost/Ubuntu/home/adirj/kcl-latent-probes-runs/format_adaptation_v1_20261127_campaign` |
| Native audit output | Campaign path with `_audit` appended |
| Native report output | Campaign path with `_report` appended |
| Durable Windows evidence | `C:/Users/adirj/AppData/Local/FreshReadoutDurable/20261127T012649Z_b57d6945755e4b53aeada19d617200dd` |
| Windows worker | PID `46824`, `powershell.exe` |
| Linux coordinator | PID `6073`, parent `6072` |
| Distribution | `Ubuntu` |

The Linux and UNC campaign paths name the same native files. The durable Windows
directory contains launcher registration, configuration, heartbeats and final
exit evidence; it is not a second copy of the scientific data. The
[detached-launch receipt](validation/2026-11-27-format-adaptation-detached-launch-v1.json)
records the initial launch. The native shell launcher is
`/home/adirj/kcl-latent-probes-runs/format_adaptation_v1_20261127_launch.sh`.
It executes the coordinator in the foreground. The Windows worker waits for
that invocation and writes its exit receipt; it does not automatically copy
the completed report into the repository.

## Observed snapshot and live producer

Snapshot: **2026-11-27T01:39:11.734770+00:00**, UTC (also London time on this date).
This is historical evidence, not a promise about a later session.

| Stage or cohort | Observed status |
| --- | --- |
| Llama fitting capture | Complete, `768/768`; coordinator stage exit `0` |
| Qwen fitting capture | Complete, `768/768`; coordinator stage exit `0` |
| Llama classifier bundle | Frozen; fit stage exit `0` |
| Qwen classifier bundle | Frozen; fit stage exit `0` |
| Both-model seal | Completed at `01:39:03.895255Z`; seal file exists |
| Llama test capture | Running, `0/4992` at this snapshot |
| Qwen test capture, analyses, audit and report | Not yet reached in the observed coordinator state |

The Linux process listing independently showed coordinator `6073` alive and
current child `15887` alive beneath it. The child's command was the Llama
`capture --partition test` invocation with the both-model seal. Earlier child
`7646` was the Qwen fitting capture and had already exited successfully. Do not
reuse that earlier child PID as a current liveness check.

Windows process inspection found worker `46824` alive. Its heartbeat at
`01:38:22.4071855Z` recorded `waiting_for_foreground_wsl`; the Linux heartbeat at
`01:38:20.320311Z` still described the then-current Qwen fitting capture. These
timestamps precede the later stage transition above. A subsequent continuation
must reread both heartbeats and inspect the current process tree. A saved
`running` status, registration receipt, old PID or old heartbeat alone does not
establish that a producer is still alive.

Only source, byte hashes, process metadata, and count/status manifests were
inspected for this runbook. No fresh feature arrays, fitted parameter arrays,
prediction values or scientific result values were opened.

## Prepared byte identity

Preparation was created at `2026-11-27T01:25:37.544125Z`, before this campaign's
model calls. A read-only native check found `prepared_no_weights`, all **50**
prepared source-file hashes matching their current exact bytes, and both model
protocols and fit-only plans matching their preparation hashes.

| Artifact | SHA256 |
| --- | --- |
| Preparation | `d714ce20a2380192cd1e0859b8b251ac7639646d99b7446cab775e56ade960aa` |
| Bank | `a05737d2b0ceaccbc71588ffdde31e3dfe73a251d3acbacf0dad190c8375aeed` |
| Llama protocol | `f963488e2e088814d2733f8c8972fc4a15b6c57951563a19eeb4e88d62ae68d0` |
| Qwen protocol | `5521868d727a4a6fe36265267ba44512ecf423c5326f59ecdae18ad952837f1e` |
| Llama fit-only plan | `fc924839fdf65cdc926b29b39e08640e6bc6cfa927dbdd88fe3d1e42e6f51783` |
| Qwen fit-only plan | `c7453de3e910899a6a66d731123e936f99c9e0e2bdb48358c0f31e312aaf0653` |

Use the native preparation record as the source-map authority. Commit identity
alone is insufficient: exact hashes also bind reviewed inputs, old instruments,
the launcher and other prepared dependencies. Preserve the original source
snapshots. Do not edit any bound file while the worker is running. This runbook
is deliberately outside that source map. The existing dirty human-review
packet and unrelated work are not part of this continuation task.

## Eleven stages and automatic gates

The [coordinator](../scripts/run_format_adaptation_campaign.py) runs these stages
sequentially, checking frozen source bytes before each stage:

1. `llama_fit_capture`: capture exactly 768 fitting prefixes.
2. `qwen_fit_capture`: capture exactly 768 fitting prefixes.
3. `llama_fit`: fit the four prescribed arms from the Llama fitting capture.
4. `qwen_fit`: fit the same arms from the Qwen fitting capture.
5. `seal_both_fits`: require both complete captures and validated, model-correct
   frozen bundles; reject any existing test-attempt directory; seal both models.
6. `llama_test_capture`: capture all 4,992 test prefixes under that seal.
7. `qwen_test_capture`: capture all 4,992 test prefixes under that seal.
8. `llama_analysis`: score and retain every prescribed Llama result.
9. `qwen_analysis`: score and retain every prescribed Qwen result.
10. `independent_audit`: independently reconstruct both models before opening
    either saved analysis outcome, then compare the prescribed outputs.
11. `report`: export every arm after a passing audit tied to these exact outputs.

The [capture implementation](../src/latent_probes/format_adaptation_capture.py)
preserves per-row starts, raw forwards and completed journals. The
[fit engine](../src/latent_probes/format_adaptation_fit.py) accepts its separately
bound fit-only plan, copies the old map and C, and performs no fresh selection.
The [analysis engine](../src/latent_probes/format_adaptation_analysis.py) checks
prepared test metadata, selected-site journal equality and pre-forward seal
timing. These behaviors are already frozen; do not substitute a new run command.

The [independent audit](../scripts/audit_format_adaptation.py) is invoked as
`--campaign <native campaign> --output <native campaign>_audit`. The
[reporter](../scripts/report_format_adaptation.py) is invoked as
`--campaign <native campaign> --audit <native campaign>_audit --output <native campaign>_report`.
The [stage and interface tests](../tests/test_format_adaptation_campaign.py)
cover this ordering and the actual CLI argument parsing.

## Continue monitoring without reading scores

Until **both analysis stages have completed**, inspect only coordinator stage
status, cohort row counts, fit-manifest status, heartbeats, live process identity
and exit/failure metadata. Do not open features, coefficients, predictions,
scientific results, the first completed model's analysis stdout, or a partial
report. Automated sealed analysis and audit run in their prescribed sequence;
this restriction concerns manual continuation and interpretation.

For a fresh liveness check, read the native coordinator state and heartbeat,
then inspect the coordinator and the current `child_pid` from the running stage:

```powershell
wsl.exe -d Ubuntu --exec ps -p 6073 -o pid,ppid,lstart,etime,stat,args
wsl.exe -d Ubuntu --exec ps --ppid 6073 -o pid,ppid,lstart,etime,stat,args
Get-Process -Id 46824 | Select-Object Id,ProcessName,StartTime
```

Compare process arguments with the expected native campaign, not just a reused
PID. Check that heartbeat timestamps are recent and capture counts advance when
a capture is active. Reload, fitting, array verification and audit stages can
have quiet periods; a single unchanged count does not justify intervention.
Do not start another producer while investigating uncertain liveness.

## Failures and completion

A failed or partial attempt is evidence to preserve. Do not delete or replace
cohort directories, journals, raw arrays, fit attempts, stage logs, seals or
durable-worker receipts. Do not rerun the coordinator, launcher, a capture row or
a fit to obtain a preferred result. Existing exclusive output paths intentionally
block automatic repetition. A missing producer or stale `running` state is an
unresolved attempt, not proof that no forward happened. Any recovery requires a
separate versioned record explaining the defect and the permitted next action.

Before treating the campaign as complete, require all of the following:

- Coordinator status is `complete`; all eleven stages are complete, with zero
  child exit codes. The in-process seal stage has no child exit code.
- Both fitting captures have 768 rows and both test captures have 4,992 rows;
  both fit manifests are frozen and both analysis manifests are complete.
- The independent audit receipt says `pass`, names this campaign, and binds the
  exact result/prediction bytes of both analyses. A failed audit blocks scientific
  interpretation even if report-looking files exist.
- The report manifest is complete. Its audited input bindings and every exported
  output hash match. The report retains the Llama primary, Qwen named replication,
  diagnostic comparison, all arms and all prespecified controls.
- The durable worker has a successful final exit receipt. Preserve its Windows
  registration, starts, heartbeats and logs alongside the native evidence. A
  missing final receipt requires checking the remaining live process; it does not
  authorize a relaunch.

Completion copies are a separate final operation. Keep the raw campaign, audit
and report originals on native Linux storage. If copying reviewable exports into
a fresh folder under [results](results/), first record the passing audit/report
and coordinator receipts. Copy exact report exports and the relevant receipts
and manifests, then verify SHA256 equality between every copied file and its
native source. Verify the native files still match their own manifests, record
source/destination paths and hashes in a copy receipt, and preserve earlier
copies. Do not overwrite prior evidence or replace an audited figure with a
manually regenerated one. A successful filesystem copy alone does not establish
that its scientific inputs passed the audit.
