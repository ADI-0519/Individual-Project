# Fixed sequential exploratory campaign

22 November 2026. The [launcher](../scripts/run_main_study_campaign.py) is an
orchestrator outside the scientific policy. It never changes the frozen bank,
methods, prompts, source bindings or failures. All results retain the
[agent exploratory amendment](MAIN_STUDY_AGENT_EXECUTION_AMENDMENT_2026-11-22.md).

## Start once in the existing WSL environment

Use the repository working directory and already configured offline model cache.
The campaign directory must not exist. The three initial preparation directories
and training freeze must already exist and match the declared source bindings.
Only the externally running Llama capture may be adopted while running. The
launcher observes it without launching another process or loading weights.

```bash
.venv-wsl/bin/python scripts/run_main_study_campaign.py \
  --root-run /home/adirj/kcl-latent-probes-runs \
  --campaign /home/adirj/kcl-latent-probes-runs/main_v1_agent_20261122_campaign \
  --llama-train /home/adirj/kcl-latent-probes-runs/main_v1_agent_20261122_llama_train_dev \
  --qwen-train /home/adirj/kcl-latent-probes-runs/main_v1_agent_20261122_qwen_train_dev \
  --monitor-development /home/adirj/kcl-latent-probes-runs/main_v1_agent_20261122_monitor_development \
  --training-freeze /home/adirj/kcl-latent-probes-runs/main_v1_agent_20261122_training_freeze.json \
  --agent-audit data/annotations/main_study_v1_agent_semantic_20261122.json
```

The script uses its own Python executable for every child, argument arrays with
no shell, and native output storage. Both offline flags must equal `1`.
The default external-Llama wait is bounded at six hours, with 15-second polls.
Unknown or interrupted captures stop execution; the launcher never retries.

## Fixed order and preserved outputs

1. Wait for existing Llama train/development capture, validate identity, run the
   standalone raw-array auditor, then fit all fixed comparisons.
2. Execute or adopt an already complete Qwen train/development capture, audit
   it, then fit. A previously attempted partial Qwen run cannot be resumed.
3. Create the separate test freeze only after both fitted bundles exist.
4. Fresh-prepare, capture and independently audit each model's complete test
   partition. Save separate analyses with the missing text monitor explicit.
5. Execute the fixed monitor's development partition; save its analysis.
   Fresh-prepare and execute its full test partition; save its analysis.
6. Save new Llama and Qwen analyses including that completed monitor. The
   earlier analyses remain preserved.

The fresh campaign folder contains immutable `config.json` with source hashes
and every planned command, `state.json` with actual stage status/times, separate
stage logs, fitted bundles, test captures/audits and analyses. Child stdout is
drained and flushed promptly; each saved log is capped at 64 MiB, with discarded
byte count and final 4 KiB recorded in state. Raw model outputs and row journals
are preserved by the scientific runners and are not truncated by this log cap.

The main outputs are `llama_analysis_with_monitor.json` and
`qwen_analysis_with_monitor.json`. Analyses ending `_without_monitor.json` are
earlier incomplete-baseline snapshots. A separate reporting tool can consume
the final JSON files afterward; it is not a prerequisite for model execution.

The current run also has a separate native `report_after_campaign.py` watcher,
its hash-bound configuration and a 48-hour timeout. It waits for the complete
campaign, checks the reporting script's saved hash, then writes a fresh
`final_report/` directory. A stopped/failed campaign produces no report. This is
postprocessing, with no additional model calls or scientific policy changes.
The [reporting script](../scripts/report_main_study.py) exports Markdown tables
and SVG/PDF/PNG figures, retaining missing/failed comparisons and exploratory
provenance. To run the same postprocessing manually on a fresh directory:

```bash
.venv-wsl/bin/python scripts/report_main_study.py \
  --llama-analysis /home/adirj/kcl-latent-probes-runs/main_v1_agent_20261122_campaign/llama_analysis_with_monitor.json \
  --qwen-analysis /home/adirj/kcl-latent-probes-runs/main_v1_agent_20261122_campaign/qwen_analysis_with_monitor.json \
  --output-directory /home/adirj/kcl-latent-probes-runs/main_v1_agent_20261122_campaign/fresh_manual_report
```

## Stop or investigate

Create a file named `STOP` inside the campaign folder to prevent further stages.
The launcher checks between stages and while observing the external Llama run.
It does not force-kill a child in the middle of an unknown model row. During
an active child, stopping takes effect after that child exits.

If a child fails, inspect its saved log and the original row journal. The
campaign records the failure and exits. Preserve the entire folder; restarting
the same campaign directory is refused. No output is deleted, overwritten,
silently resumed or automatically retried. A deliberate recovery requires a
separate documented decision after examining the preserved attempt.
