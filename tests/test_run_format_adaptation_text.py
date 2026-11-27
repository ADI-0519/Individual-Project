"""Source/input/fit seals and durable failure boundaries, without real fits."""

import copy
import importlib.util
from pathlib import Path

import pytest

from latent_probes.storage import atomic_json, file_sha256

PATH = Path(__file__).resolve().parents[1] / "scripts/run_format_adaptation_text.py"
spec = importlib.util.spec_from_file_location("lexical_runner_test", PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_gzip_roundtrip_deterministic_and_exclusive(tmp_path):
    one, two = tmp_path / "one.gz", tmp_path / "two.gz"
    value = {"rows": ["Belief", "世界"], "integer": 1}
    runner.gzip_json(one, value)
    runner.gzip_json(two, value)
    assert runner.read_gzip(one) == value
    assert one.read_bytes() == two.read_bytes()
    with pytest.raises(FileExistsError):
        runner.gzip_json(one, {})


def test_fit_seal_rejects_mutation_extra_or_missing_files(tmp_path):
    atomic_json(tmp_path / "preparation.json", {"fixed": True})
    files = {}
    for view in ("full_messages", "system_only"):
        for filename in ("metadata.json", "coefficients.npz"):
            path = tmp_path / "fit" / view / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
            files[path.relative_to(tmp_path).as_posix()] = file_sha256(path)
    seal = {
        "schema": "adaptation-lexical-fit-seal-v1",
        "preparation_sha256": file_sha256(tmp_path / "preparation.json"),
        "files_sha256": files,
    }
    atomic_json(tmp_path / "fit_seal.json", seal)
    assert runner.check_fit_seal(tmp_path) == seal
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="fitted artifact"):
        runner.check_fit_seal(tmp_path)
    path.write_bytes(b"fixture")
    extra = tmp_path / "fit" / "unexpected.txt"
    extra.write_text("extra")
    with pytest.raises(ValueError, match="fitted artifact"):
        runner.check_fit_seal(tmp_path)
    extra.unlink()
    path.unlink()
    with pytest.raises(ValueError, match="fitted artifact"):
        runner.check_fit_seal(tmp_path)


def test_fit_started_attempt_is_never_silently_repeated(tmp_path):
    runner.exclusive_start(tmp_path, "fit")
    before = (tmp_path / "fit_started.json").read_bytes()
    with pytest.raises(FileExistsError):
        runner.exclusive_start(tmp_path, "fit")
    assert (tmp_path / "fit_started.json").read_bytes() == before


def test_fit_failure_preserves_started_and_cannot_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner, "check_preparation", lambda output: {"fit_input_sha256": runner.canonical_hash([])}
    )
    monkeypatch.setattr(runner, "input_rows", lambda partition: [])

    def fail(rows, view):
        raise ValueError("synthetic convergence failure")

    monkeypatch.setattr(runner, "fit_text", fail)
    with pytest.raises(ValueError, match="synthetic convergence"):
        runner.fit(tmp_path)
    assert (tmp_path / "fit_started.json").exists()
    assert not (tmp_path / "fit_seal.json").exists()
    with pytest.raises(FileExistsError):
        runner.fit(tmp_path)


def test_test_hash_mismatch_is_rejected_for_either_checkpoint(monkeypatch):
    row = {
        "row_id": "r",
        "group_id": "g",
        "partition": "test",
        "split": "test",
        "renderer": "A",
        "condition": "asserted",
        "state": {},
        "gold": 0,
        "messages_sha256": "original",
    }
    rows = [copy.deepcopy(row) for _ in range(4992)]
    monkeypatch.setattr(runner, "read_gzip", lambda path: {"rows": copy.deepcopy(rows)})
    runner.verify_original_test(rows)

    def bad_second(path):
        result = copy.deepcopy(rows)
        if "qwen" in str(path):
            result[100]["messages_sha256"] = "changed"
        return {"rows": result}

    monkeypatch.setattr(runner, "read_gzip", bad_second)
    with pytest.raises(ValueError, match="qwen original"):
        runner.verify_original_test(rows)


def test_preparation_detects_modified_source_and_does_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "check_original_sources", lambda: {})
    source = tmp_path / "source.py"
    source.write_text("original", encoding="utf-8")
    output = tmp_path / "run"
    output.mkdir()
    atomic_json(
        output / "preparation.json",
        {
            "schema": "adaptation-lexical-preparation-v1",
            "source_sha256": {"source.py": file_sha256(source)},
            "input_sha256": {},
        },
    )
    runner.check_preparation(output)
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Changed sealed artifact"):
        runner.check_preparation(output)
    with pytest.raises(ValueError, match="Output exists"):
        runner.prepare(output)


def test_original_replay_binding_is_checked_before_projection_read(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    base = tmp_path / runner.OLD / "replay"
    projection = base / "llama" / "rows.json.gz"
    projection.parent.mkdir(parents=True)
    projection.write_bytes(b"changed projection")
    atomic_json(
        base / "manifest.json",
        {
            "schema": "format-adaptation-portable-replay-v1",
            "status": "complete",
            "test_fixture": False,
            "files": {"llama/rows.json.gz": "different_hash"},
        },
    )
    monkeypatch.setattr(runner, "read_gzip", lambda path: pytest.fail("Must reject before read"))
    with pytest.raises(ValueError, match="differs from replay manifest"):
        runner.verify_original_test([])
