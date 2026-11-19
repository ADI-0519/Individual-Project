# Progress presentations

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
