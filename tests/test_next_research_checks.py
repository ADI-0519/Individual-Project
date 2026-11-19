"""The combined runner must reject source-contained output before writing anything."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("relative_output", [".", "diagnostic", "nested/diagnostic"])
def test_combined_runner_rejects_source_output_without_writes(
    tmp_path, monkeypatch, relative_output
):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "next_checks", scripts / "run_next_research_checks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "frozen"
    source.mkdir()
    sentinel = source / "untouched.txt"
    sentinel.write_text("Original frozen run", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_next_research_checks.py",
            "--run",
            str(source),
            "--output",
            str(source / relative_output),
            "--config",
            str(tmp_path / "config-not-needed.json"),
        ],
    )
    with pytest.raises(ValueError, match="outside the original frozen run"):
        module.main()
    assert list(source.iterdir()) == [sentinel]
    assert sentinel.read_text(encoding="utf-8") == "Original frozen run"
