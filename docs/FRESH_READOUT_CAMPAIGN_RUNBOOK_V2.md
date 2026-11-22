# Fresh campaign: running, completion and recovery

22 November 2026. Scientific policy is the unchanged
[execution protocol](FRESH_READOUT_EXECUTION_PROTOCOL_V2.md), with the
[zero-forward startup correction](FRESH_READOUT_STARTUP_RECOVERY_V1.md).
Read the [interpretation rules](FRESH_READOUT_INTERPRETATION_RULES_V2.md) before
describing an outcome. These instructions do not authorize another scientific
configuration or a repeat selected on performance.

## Active paths and processes

The active native Linux campaign is:

```text
/home/adirj/kcl-latent-probes-runs/fresh_readout_v2_20261122_recovery_v1
```

The original failed campaign and its 53-file failure inventory remain preserved.
The new preparation receipt establishes exact equality of both 7,040-row plans,
their excluded engineering rows, settings, spans and token identities. Only
tokenizer identity and declared provenance/environment metadata changed.

The coordinator runs in a persistent foreground WSL execution session. It uses
one sequential GPU worker. Both training/development captures precede fitting;
both bundles precede the test freeze; both fresh test captures precede their
analyses. Do not launch another producer or shut down WSL while it is running.
The first real Llama engineering forward passed all twelve hook/HF alignment
checks. The first 87 training rows gave about 1.25 rows/second, an early estimate
of roughly three hours for 14,080 capture inputs before subsequent work.
Later conditions and Qwen may differ; this is not a completion promise.

## Automatic completion

The separate [completion watcher](../scripts/complete_fresh_readout_campaign.py)
does not launch models or fits. It freezes and archives its own code, the
[independent auditor](../scripts/audit_fresh_readout_saved_data.py) and the
[reporter](../scripts/report_fresh_readout.py). It polls only campaign status
until both analyses complete, then validates final artifact hashes and native
capture provenance. Its fixed subsequent order is:

1. Independently reconstruct both models' predictions and metrics. Freeze the
   reconstruction files before parsing either saved result/prediction file.
2. Compare all seventeen prediction columns, factor margins, context/stratum
   metrics, primary interval, retained-head development objectives and training
   scaler statistics. The unretained candidate scores cannot be reconstructed
   without refitting; their recorded ranking is checked with that limit stated.
3. Require a passing audit before generating Markdown, JSON and PDF/SVG/PNG.
4. Copy the exact seven report files into a new repository result folder,
   verifying every copied hash. It does not commit, push or interpret results.

The watcher uses these distinct native output directories:

```text
fresh_readout_v2_20261122_recovery_v1_completion
fresh_readout_v2_20261122_recovery_v1_audit
fresh_readout_v2_20261122_recovery_v1_report
```

On successful completion, the repository destination is
`docs/results/fresh-readout-2026-11-22/`. It is created only after the report
passes verification. The native completion directory contains
`completion_receipt.json`, source snapshots and child logs. A campaign status
of `complete` alone does not establish that the audit or report passed: check
the separate completion receipt and report manifest.

## Inspect status without inspecting partial scores

Read `campaign_state.json` and the running stage's `manifest.json` under the
native root. The training/test manifests provide `completed_rows` and status.
The stage stdout logs show counts. Avoid parsing partial test output scores,
changing cases or selecting another site. Keep the machine running; its AC/DC
sleep-after settings were checked as zero on 22 November, without modification.

The full campaign and watcher commands must not be repeated against existing
outputs. Their exclusivity checks intentionally refuse an earlier attempt.

## If anything stops

Preserve the state, exact source archive, logs, started rows and all completed
artifacts. A failed stage is not permission to overwrite, suppress an error or
retry an unknown outcome. Identify whether the error concerns model capture,
recording, fitting, scientific analysis, the independent checker or reporting.
Those failures have different recovery boundaries.

A checker/reporting software error does not by itself require repeating model
inference. An authorized, versioned repair can operate on unchanged saved
measurements, preserving the first failed audit and its independent freezes.
Any recovery must state exactly what changed and verify unchanged scientific
inputs. A numerical mismatch must be investigated before publishing a figure.
The first startup correction is a documented precedent, not a general retry
exception. Agent review remains exploratory; human semantic review is incomplete.
