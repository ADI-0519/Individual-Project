"""Archive boundary tests use synthetic bytes, never private pilot artifacts."""

import importlib.util
import json
import stat
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "portable_natural_selection", ROOT / "scripts/portable_natural_selection.py"
)
portable = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(portable)


def make_archive(path, payload=None, *, mutate=None, extra=None):
    payload = payload or {"run/data.json": b'{"saved": true}\r\n', "reproduce.py": b"# fixture\n"}
    inventory = portable.inventory_for(payload, {"/old/run/data.json": "run/data.json"})
    if mutate:
        mutate(inventory)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("inventory.json", portable.json_bytes(inventory))
        for name, data in payload.items():
            archive.writestr(name, data)
        if extra:
            archive.writestr(*extra)
    return inventory


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "C:/absolute",
        "a\\b",
        "a/../b",
        "a/./b",
        "a//b",
        "a/",
        "CON",
        "a/NUL.txt",
        "a/file. ",
        'a/"b',
        "a/\x00b",
    ],
)
def test_rejects_unsafe_or_nonportable_members(name):
    with pytest.raises(ValueError, match="archive member"):
        portable.member_name(name)


def test_roundtrip_preserves_raw_crlf_bytes_and_inventory(tmp_path):
    archive = tmp_path / "bundle.zip"
    expected = make_archive(archive)
    assert portable.verify_archive(archive)["verified"] is True
    target = tmp_path / "relocated"
    assert portable.extract_bundle(archive, target) == expected
    assert (target / "run/data.json").read_bytes().endswith(b"\r\n")
    assert portable.verify_directory(target) == expected


def test_tampered_member_fails_before_any_extraction(tmp_path):
    archive = tmp_path / "bundle.zip"
    make_archive(
        archive,
        mutate=lambda inventory: inventory["files"]["run/data.json"].update(sha256="0" * 64),
    )
    target = tmp_path / "relocated"
    with pytest.raises(ValueError, match="hash differs"):
        portable.extract_bundle(archive, target)
    assert not target.exists()


def test_unlisted_member_is_rejected(tmp_path):
    archive = tmp_path / "bundle.zip"
    make_archive(archive, extra=("secret-token", b"unexpected"))
    with pytest.raises(ValueError, match="Unlisted"):
        portable.verify_archive(archive)


def test_case_collisions_are_rejected_before_cross_platform_extraction(tmp_path):
    archive = tmp_path / "bundle.zip"
    make_archive(archive, extra=("RUN/data.json", b"collision"))
    with pytest.raises(ValueError, match="Duplicate"):
        portable.verify_archive(archive)


def test_symlink_zip_member_is_rejected(tmp_path):
    archive = tmp_path / "bundle.zip"
    make_archive(archive)
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "a") as stream:
        stream.writestr(info, b"/outside")
    with pytest.raises(ValueError, match="Nonregular"):
        portable.verify_archive(archive)


def test_archive_limits_are_enforced_from_headers(tmp_path, monkeypatch):
    archive = tmp_path / "bundle.zip"
    make_archive(archive)
    monkeypatch.setattr(portable, "MAX_FILE_BYTES", 5)
    with pytest.raises(ValueError, match="too large"):
        portable.verify_archive(archive)


def test_existing_destination_is_never_overwritten(tmp_path):
    archive = tmp_path / "bundle.zip"
    make_archive(archive)
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(ValueError, match="must not exist"):
        portable.extract_bundle(archive, target)


def test_rejects_duplicate_json_keys():
    with pytest.raises(ValueError, match="Duplicate JSON"):
        json.loads('{"files": {}, "files": {}}', object_pairs_hook=portable.unique_object)


def test_expanded_evidence_mutation_and_unlisted_files_fail(tmp_path):
    archive = tmp_path / "bundle.zip"
    make_archive(archive)
    target = tmp_path / "expanded"
    portable.extract_bundle(archive, target)
    (target / "run/data.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Expanded hash differs"):
        portable.verify_directory(target)
    (target / "unexpected.txt").write_text("extra")
    with pytest.raises(ValueError, match="Expanded inventory differs"):
        portable.verify_directory(target)


def test_numeric_tolerance_does_not_tolerate_changed_counts_or_fields():
    portable.compare({"margin": 1.0 + 1e-13, "calls": 222}, {"margin": 1.0, "calls": 222})
    for actual in (
        {"margin": 1.1, "calls": 222},
        {"margin": 1.0, "calls": 223},
        {"margin": 1.0},
        {"margin": 1.0, "calls": 222.0},
    ):
        with pytest.raises(ValueError, match="differ"):
            portable.compare(actual, {"margin": 1.0, "calls": 222})


def test_original_path_mapping_is_explicit_not_a_substring_rewrite():
    assert (
        portable._repo_relative("/mnt/c/user/Individual-Project/scripts/example.py")
        == "scripts/example.py"
    )
    with pytest.raises(ValueError, match="Unknown captured path"):
        portable._repo_relative("/unknown/secrets/token")


def test_canonical_inventory_is_order_independent_and_scope_is_fixed():
    first = portable.inventory_for({"b": b"2", "a": b"1"}, {})
    second = portable.inventory_for({"a": b"1", "b": b"2"}, {})
    assert portable.json_bytes(first) == portable.json_bytes(second)
    first["new_model_calls"] = 1
    with pytest.raises(ValueError, match="scope"):
        portable.validate_inventory(first)
