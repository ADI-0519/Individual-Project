# Contributing

Keep changes small enough to review, and explain the problem, implementation and
validation in the pull request. Use the Python versions tested by CI: 3.11 and
3.12. Install the pinned uv version declared in [pyproject.toml](pyproject.toml),
then create the development environment:

```sh
uv sync --locked --group dev
uv run --locked python -m pre_commit install
```

Before submitting a change, run:

```sh
uv run --locked python -m pytest tests -m "not model"
uv run --locked python -m ruff check .
uv run --locked python -m ruff format --check .
uv run --locked python -m pre_commit run --all-files
uv run --locked python -m build --no-isolation
uv run --locked python -m twine check dist/*
```

The default test suite uses local fixtures and must work without model downloads,
GPU hardware, credentials or API calls. Real-model checks belong in an explicitly
requested experiment and must record the checkpoint revision, precision,
configuration and observed result. Passing fixture tests does not establish that
the model runtime or scientific hypothesis works.

For research changes, identify any changes to labels, split membership, prompts,
capture positions, precision or outcome metrics. Keep development cases separate
from held-out evaluation, and do not silently replace a failed or ambiguous run.
Retain enough provenance to reproduce a result from a clean checkout.

Commit small, audited example data and configurations. Keep downloaded datasets,
weights, raw runs, credentials and private annotations out of Git. Check the
source, licence and redistribution permissions before introducing third-party
material. Update the dependency manifest and [uv.lock](uv.lock) together; do not
edit the lockfile by hand.

CI checks the lightweight installation on Linux and Windows, linting, formatting
and package creation. CI package artifacts are validation outputs, not releases
or evidence of completed experiments.
