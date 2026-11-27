# Explicit recovery of the interrupted test capture

22 November 2026. The original campaign stopped during Llama test capture. Its
stale manifest still says `running`, but WSL restarted and no experiment producer
remains. The exact shutdown cause is not established. Both 2,048-row train/dev
captures and both 30-comparison fitted bundles completed before the interruption.
No complete held-out analysis was produced.

The interrupted Llama directory contains 2,758 completed row journals and one
additional directory containing only `started.json`:
`test_message_27__i0_u0_b1_w1_map0`. Its last saved start time is
`2026-11-22T14:58:27.971209+00:00`. There is no raw forward, feature array,
completed outcome or recorded exception for that attempt. A forward may have
occurred without being saved; it would be incorrect to claim it never ran.

## Recovery boundary

The [new recovery launcher](../scripts/recover_main_study_campaign.py) preserves
the entire original campaign and unknown attempt. It copies the protocol,
manifest, attempts and only fully completed journals into a fresh native folder.
Every copied file must match the original hash. The unchanged original
`_complete_rows` checker must validate every retained copied measurement before
the preparation receipt is written or any model continuation is authorized.

The expected inventory is 2,758 reused measurements and 1,082 remaining case IDs:
one explicitly repeated unknown attempt and 1,081 previously unattempted rows.
The repeated attempt is not another independent case. Selection is based only
on interrupted recording status, never on the unknown prediction or score.
If raw artifacts, a recorded failure, a second unknown attempt, a count mismatch
or corrupted completed measurements are found, preparation fails and preserves
the new partial copy for inspection. There is no fallback or retry loop.

The original fitted heads, train/test freezes, bank, methods and frozen sources
remain unchanged. The copied Llama capture uses the original runner's explicit
`resume=True` with exactly its prior authorization, old fitted bundle and old
test freeze. Only the new copy receives a continuation attempt. The original
human-review ledger remains untouched and all outcomes remain agent-audited
exploratory.

## Prepare, inspect, then execute once

From the original repository in its existing WSL environment with the original
offline cache variables, prepare without loading a model:

```bash
.venv-wsl/bin/python scripts/recover_main_study_campaign.py \
  --campaign /home/adirj/kcl-latent-probes-runs/main_v1_agent_20261122_campaign \
  --recovery /home/adirj/kcl-latent-probes-runs/main_v1_agent_20261122_recovery_v1 \
  --attest-no-running-producers
```

The flag attests only that no producer remains; it is not human semantic review.
The script also scans current processes and refuses to duplicate a producer.
Inspect `recovery_receipt.json`, `prepared_manifest.json` and `config.json`.
They record retained/remaining IDs, hashes, the unknown attempt, source bindings
and the fixed continuation commands. Receipt/config hashes are sealed before
execution. The copied manifest and authorization remain byte-identical to the
interrupted originals at this point.

Execute that prepared recovery separately:

```bash
.venv-wsl/bin/python scripts/recover_main_study_campaign.py \
  --recovery /home/adirj/kcl-latent-probes-runs/main_v1_agent_20261122_recovery_v1 \
  --run-prepared
```

Execution verifies the seal, unchanged sources and every copied artifact again,
then creates an exclusive `execution_started.json`. An attempted, failed or
interrupted recovery cannot be automatically relaunched. The original campaign
launcher is not resumed and no fitting/freezing stage is repeated.

The fixed order is remaining Llama test rows, audit and analysis; fresh Qwen test
capture with its old fit and freeze, audit and analysis; the existing prepared
fixed monitor development run; fresh fixed monitor test; both complete analyses;
then the unchanged reporting script. All future outputs use the new recovery
folder, except execution of the explicitly identified old monitor preparation.

A `STOP` file in the recovery folder prevents later stages. It does not kill an
active child mid-row. Logs and failure state are preserved. Launching the driver
outside the interactive terminal improves continuity if that terminal closes;
it cannot survive Windows shutdown, WSL shutdown or a machine reboot. Any new
interruption requires inspecting the preserved artifacts before another explicit
recovery decision. Final reporting must disclose this recovery and its one
unknown-outcome repeated attempt.

## Actual recovery launch

These commands record a preparation that has already completed; the existing
directory is preserved and cannot be prepared again. Its sealed receipt records
2,758 retained and 1,082 pending cases, with one disclosed unknown-outcome repeat.
The run was launched at 15:14 UTC using Windows `Start-Process` with a hidden
window and redirected native Windows logs. The WSL client PID was45904 at launch.
This avoids relying on the interactive tool terminal's lifetime; it is not a
guarantee against WSL shutdown, machine shutdown or another interruption.

Use the new recovery directory's `state.json`, `windows_launcher.json` and logs.
The original campaign and its stale `running` state remain historical evidence.
The previous native report watcher also stopped with WSL; the recovery's fixed
last stage now runs the unchanged reporter after both complete baseline analyses.
Keep the recovery receipt and this deviation note alongside the final report.
