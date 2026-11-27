# Progress presentations

The completed experiment's latest [four-slide PowerPoint](./progress2-main-study-2026-11-22/completed/Progress-2-main-study.pptx),
[PDF](./progress2-main-study-2026-11-22/completed/Progress-2-main-study.pdf) and
[speaker notes](./progress2-main-study-2026-11-22/completed/speaker-notes.md)
include the finished shared prose comparison: 33.75% correction quartets,
44 invalid test outputs retained, and the computation/semantic-review limits.
Its [receipt](./progress2-main-study-2026-11-22/completed/source-receipt.json)
binds the [completed report summary](../docs/results/main-study-2026-11-22-complete/results_summary.json).
Use the rebuild command below with `main-study-2026-11-22-complete` as both input
folders and a fresh output folder. CI rebuilds both the interim and completed
updates. Neither snapshot claims a meeting date or human-reviewed confirmation.

The earlier interim update is the [22 November main-study PowerPoint](./progress2-main-study-2026-11-22/reviewed/Progress-2-main-study.pptx)
and [PDF](./progress2-main-study-2026-11-22/reviewed/Progress-2-main-study.pdf), with
[speaker notes](./progress2-main-study-2026-11-22/reviewed/speaker-notes.md). It has exactly
four slides: title/name/description, current progress, barriers and next steps.
It presents both completed model analyses, the lack of a measured fitting-coverage
benefit, incomplete human validation, and the pending prose baseline. It does
not claim spontaneous anticipation, deceptive intent, or a complete research
project. This is a dated result snapshot, not an assertion of the meeting date.

The [source receipt](./progress2-main-study-2026-11-22/reviewed/source-receipt.json) binds
the unchanged template/renderer, builder, actual
[two-model report summary](../docs/results/main-study-2026-11-22-interim/results_summary.json)
and its manifest. Rebuild to a fresh output folder with the existing slides
environment; this makes no model calls:

```powershell
python scripts/build_main_study_progress2.py `
  --summary docs/results/main-study-2026-11-22-interim/results_summary.json `
  --manifest docs/results/main-study-2026-11-22-interim/report_manifest.json `
  --output-directory .cache/main-study-progress2-rebuild
```

CI checks source refusals and four-slide layout, rebuilds this empirical update,
and retains it with its receipt. The earlier deck below is a historical pilot
snapshot and has its own unchanged data and rebuild path.

[Progress 2 PowerPoint](./progress2/Progress-2-measured-results.pptx) and
[PDF](./progress2/Progress-2-measured-results.pdf) contain exactly four slides:
title, current state, current barriers, and planned next steps. The main question
remains neutral-trained reader-outcome prediction under reporting pressure.

The measured [input JSON](./progress2/Progress-2-measured-results.json) records
the pilot counts, model comparisons and speaker-note metrics. See the
[natural-pilot report](../docs/NEUTRAL_FORECAST_RESULTS.md) for interpretation,
post-hoc control status, failed reader controls and remaining validation.

The source [template](./templates/progress2-original.pptx) is an unchanged copy
of the original Progress 2 draft, retained to reproduce its editable visual
style. Original external slides were not edited. The template's SHA-256 is
`c63d6eb4be21cfb626ba0b99fd54fb43629d7e7bd89717da4284af42c80e4255`.

Rebuild on Windows with installed Arial regular/bold fonts:

```powershell
uv sync --locked --extra slides --group dev
uv run --locked --extra slides python scripts/update_progress2.py `
  --results presentations/progress2/Progress-2-measured-results.json `
  --overwrite
```

The builder checks source hashes, slide count and text bounds. PowerPoint
shapes remain editable; the PDF uses the same measured Arial layout through
ReportLab. No model or paid service is called. CI rebuilds both formats on
Windows and retains the checked outputs as an artifact. Byte hashes may differ
between builds because document metadata contains creation timestamps.
