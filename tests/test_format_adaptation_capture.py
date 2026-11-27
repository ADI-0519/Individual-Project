"""CPU-only capture-gate tests; no fixture constructs HFBackend or loads weights."""

import copy
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from latent_probes import format_adaptation_capture as capture
from latent_probes.storage import file_sha256, read_json


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def bindings(*paths):
    return {str(p.resolve()): file_sha256(p) for p in paths}


class BackendConstructionStopped(RuntimeError):
    pass


@pytest.fixture
def gate(tmp_path, monkeypatch):
    directory = tmp_path / "campaign/llama"
    directory.mkdir(parents=True)
    source = tmp_path / "policy.py"
    source.write_text("fixed policy fixture\n", encoding="utf-8")
    bank = tmp_path / "bank.json"
    write(bank, {"fixture": "bank"})
    source_map = bindings(source, bank)
    environment = {"python": "fixture", "platform": "fixture", "packages": {"numpy": "fixture"}}
    plan_path = directory / "fit_plan.json"
    write(plan_path, {"schema": "format-adaptation-fit-plan-v1", "fixture": True})
    protocol = {
        "model_key": "llama",
        "model_settings": {"model_id": "never-loaded-fixture", "local_files_only": True},
        "bank_path": str(bank),
        "bank_sha256": file_sha256(bank),
        "source_sha256": source_map,
        "environment": environment,
        "fit_plan_sha256": file_sha256(plan_path),
    }
    protocol_path = directory / "protocol.json"
    write(protocol_path, protocol)
    preparation = {
        "schema": "format-adaptation-preparation-v1",
        "status": "prepared_no_weights",
        "bank_sha256": file_sha256(bank),
        "source_sha256": source_map,
        "model_protocol_sha256": {"llama": file_sha256(protocol_path)},
        "fit_plan_sha256": {"llama": file_sha256(plan_path)},
    }
    write(directory.parent / "preparation.json", preparation)
    models = {}
    for key in ("llama", "qwen"):
        bundle = directory.parent / (key + "_fit")
        bundle.mkdir()
        entry = {}
        for name, field in (
            ("coefficients.npz", "coefficients_sha256"),
            ("settings.json", "settings_sha256"),
            ("fit_manifest.json", "fit_manifest_sha256"),
        ):
            path = bundle / name
            path.write_bytes((key + "/" + name).encode())
            entry[field] = file_sha256(path)
        models[key] = entry
    seal_path = directory.parent / "test_seal.json"
    seal = {
        "schema": "format-adaptation-test-seal-v1",
        "bank_sha256": file_sha256(bank),
        "source_sha256": source_map,
        "models": models,
    }
    write(seal_path, seal)
    # Native storage and package identity are isolated only for this synthetic gate fixture.
    monkeypatch.setattr(capture, "native", lambda p: Path(p).resolve())
    monkeypatch.setattr(capture, "environment_info", lambda: copy.deepcopy(environment))
    calls = []

    def stop_backend(settings):
        calls.append(settings)
        raise BackendConstructionStopped("CPU gate fixture: stop before any backend exists")

    return SimpleNamespace(
        directory=directory,
        source=source,
        bank=bank,
        protocol=protocol,
        protocol_path=protocol_path,
        preparation=preparation,
        plan=plan_path,
        seal=seal,
        seal_path=seal_path,
        calls=calls,
        backend=stop_backend,
    )


def test_agent_review_requires_exact_bank_and_all_reviewed_input_bytes(tmp_path):
    bank, evidence, receipt_path = (
        tmp_path / name for name in ("bank.json", "input.txt", "review.json")
    )
    write(bank, {"fixture": True})
    evidence.write_text("reviewed fixture bytes", encoding="utf-8")
    receipt = {
        "schema": "format-adaptation-agent-review-v1",
        "status": "pass",
        "human_review_complete": False,
        "bank_sha256": file_sha256(bank),
        "input_sha256": bindings(evidence),
    }
    write(receipt_path, receipt)
    assert capture.validate_agent_review(bank, receipt_path) == receipt
    evidence.write_text("unreviewed replacement", encoding="utf-8")
    with pytest.raises(ValueError, match="Changed input"):
        capture.validate_agent_review(bank, receipt_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "wrong"),
        ("status", "fail"),
        ("human_review_complete", True),
        ("bank_sha256", "0" * 64),
        ("input_sha256", {}),
    ],
)
def test_agent_review_rejects_invalid_metadata_or_missing_bindings(tmp_path, field, value):
    bank, evidence, path = (tmp_path / name for name in ("bank", "evidence", "review"))
    bank.write_bytes(b"bank")
    evidence.write_bytes(b"evidence")
    receipt = {
        "schema": "format-adaptation-agent-review-v1",
        "status": "pass",
        "human_review_complete": False,
        "bank_sha256": file_sha256(bank),
        "input_sha256": bindings(evidence),
    }
    receipt[field] = value
    write(path, receipt)
    with pytest.raises(ValueError):
        capture.validate_agent_review(bank, path)


def test_test_seal_accepts_both_complete_frozen_bundles(gate):
    assert capture.verify_test_seal(gate.directory, gate.seal_path) == gate.seal


@pytest.mark.parametrize("key", ["llama", "qwen"])
@pytest.mark.parametrize("name", ["coefficients.npz", "settings.json", "fit_manifest.json"])
def test_test_seal_rejects_each_model_bundle_file_mutation(gate, key, name):
    path = gate.directory.parent / (key + "_fit") / name
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Fitted bundle changed"):
        capture.verify_test_seal(gate.directory, gate.seal_path)


@pytest.mark.parametrize(
    "mutation", ["one_model", "extra_model", "bank", "source_map", "source_bytes"]
)
def test_test_seal_rejects_unmatched_bank_sources_or_model_inventory(gate, mutation):
    seal = copy.deepcopy(gate.seal)
    if mutation == "one_model":
        del seal["models"]["qwen"]
    elif mutation == "extra_model":
        seal["models"]["third"] = seal["models"]["llama"]
    elif mutation == "bank":
        seal["bank_sha256"] = "0" * 64
    elif mutation == "source_map":
        seal["source_sha256"] = bindings(gate.source)
    else:
        gate.source.write_bytes(b"changed source")
    write(gate.seal_path, seal)
    with pytest.raises(ValueError):
        capture.verify_test_seal(gate.directory, gate.seal_path)


@pytest.mark.parametrize("key", ["llama", "qwen"])
def test_test_seal_requires_both_bundle_directories_to_exist(gate, key):
    (gate.directory.parent / (key + "_fit") / "coefficients.npz").unlink()
    with pytest.raises(FileNotFoundError):
        capture.verify_test_seal(gate.directory, gate.seal_path)


@pytest.mark.parametrize("mutation", ["no_seal", "bad_bank", "qwen_changed", "llama_changed"])
def test_test_capture_rejects_invalid_seal_before_backend_construction(gate, mutation):
    seal_path = gate.seal_path
    if mutation == "no_seal":
        seal_path = None
    elif mutation == "bad_bank":
        gate.seal["bank_sha256"] = "0" * 64
        write(seal_path, gate.seal)
    else:
        key = mutation.split("_")[0]
        (gate.directory.parent / (key + "_fit") / "coefficients.npz").write_bytes(b"changed")
    with pytest.raises(ValueError):
        capture.capture(gate.directory, "test", seal_path, backend_factory=gate.backend)
    assert gate.calls == []
    assert not (gate.directory / "test").exists()


def test_valid_test_seal_reaches_only_injected_backend_and_preserves_failure(gate):
    with pytest.raises(BackendConstructionStopped):
        capture.capture(gate.directory, "test", gate.seal_path, backend_factory=gate.backend)
    assert len(gate.calls) == 1
    status = read_json(gate.directory / "test/manifest.json")
    assert status["status"] == "failed" and status["completed_rows"] == 0
    assert status["seal_sha256"] == file_sha256(gate.seal_path)
    assert status["error_type"] == "BackendConstructionStopped"


@pytest.mark.parametrize("mutation", ["protocol", "source", "environment"])
def test_capture_rejects_changed_preparation_fingerprints_before_backend(
    gate, monkeypatch, mutation
):
    if mutation == "protocol":
        gate.protocol["unprepared_change"] = True
        write(gate.protocol_path, gate.protocol)
    elif mutation == "source":
        gate.source.write_bytes(b"changed policy")
    else:
        monkeypatch.setattr(capture, "environment_info", lambda: {"python": "changed"})
    with pytest.raises(ValueError):
        capture.capture(gate.directory, "fit", backend_factory=gate.backend)
    assert gate.calls == []
    assert not (gate.directory / "fit").exists()


@pytest.mark.parametrize("mutation", ["actual_plan", "preparation_plan"])
def test_fit_capture_rejects_changed_fit_plan_before_backend(gate, mutation):
    if mutation == "actual_plan":
        write(gate.plan, {"unprepared_replacement": True})
    else:
        gate.preparation["fit_plan_sha256"]["llama"] = "0" * 64
        write(gate.directory.parent / "preparation.json", gate.preparation)
    with pytest.raises(ValueError):
        capture.capture(gate.directory, "fit", backend_factory=gate.backend)
    assert gate.calls == []
    assert not (gate.directory / "fit").exists()


def test_valid_fit_capture_preserves_preparation_and_plan_bindings_without_seal(gate):
    with pytest.raises(BackendConstructionStopped):
        capture.capture(gate.directory, "fit", backend_factory=gate.backend)
    assert len(gate.calls) == 1
    manifest = read_json(gate.directory / "fit/manifest.json")
    assert manifest["status"] == "failed" and manifest["seal_sha256"] is None
    assert manifest["fit_plan_path"] == str(gate.plan)
    assert manifest["input_artifact_sha256"][str(gate.plan)] == file_sha256(gate.plan)
    preparation_path = gate.directory.parent / "preparation.json"
    assert manifest["input_artifact_sha256"][str(preparation_path)] == file_sha256(preparation_path)


def test_unknown_capture_partition_stops_before_backend(gate):
    with pytest.raises(ValueError, match="Unknown capture cohort"):
        capture.capture(gate.directory, "development", backend_factory=gate.backend)
    assert gate.calls == []


@pytest.mark.parametrize("system", ["Windows", "Darwin"])
def test_real_native_guard_rejects_non_linux_without_loading_models(tmp_path, monkeypatch, system):
    monkeypatch.setattr(capture.platform, "system", lambda: system)
    with pytest.raises(ValueError, match="native Linux storage"):
        capture.native(tmp_path / "output")


def test_real_native_guard_rejects_output_inside_checkout(monkeypatch):
    monkeypatch.setattr(capture.platform, "system", lambda: "Linux")
    with pytest.raises(ValueError, match="outside the checkout"):
        capture.native(capture.ROOT / "forbidden-capture-fixture")


def test_fresh_json_preserves_existing_artifact_bytes(tmp_path):
    path = tmp_path / "evidence.json"
    capture.fresh_json(path, {"earlier_attempt": True})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        capture.fresh_json(path, {"overwrite": True})
    assert path.read_bytes() == before


@pytest.mark.parametrize("partition", ["fit", "test"])
def test_capture_refuses_existing_cohort_without_overwriting_evidence(gate, partition):
    cohort = gate.directory / partition
    cohort.mkdir()
    marker = cohort / "earlier_attempt.txt"
    marker.write_bytes(b"preserve this failed attempt")
    with pytest.raises(FileExistsError):
        capture.capture(gate.directory, partition, gate.seal_path, backend_factory=gate.backend)
    assert marker.read_bytes() == b"preserve this failed attempt"
    assert gate.calls == []


@pytest.fixture
def preparation_fixture(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "policy.py").write_text("fixed policy", encoding="utf-8")
    bank, review_input, receipt = (tmp_path / name for name in ("bank", "review_input", "review"))
    bank.write_bytes(b"synthetic bank")
    review_input.write_bytes(b"synthetic reviewed input")
    write(
        receipt,
        {
            "schema": "format-adaptation-agent-review-v1",
            "status": "pass",
            "human_review_complete": False,
            "bank_sha256": file_sha256(bank),
            "input_sha256": bindings(review_input),
        },
    )
    tokenizer = {"fingerprint": "fixed fixture tokenizer"}
    old = tmp_path / "old"
    for key, site in (
        ("llama", "layer_16_query_location_last"),
        ("qwen", "layer_21_query_location_last"),
    ):
        previous = dict.fromkeys(capture.MODEL_FIELDS, "fixture")
        previous.update(
            model_key=key,
            model_settings={"model_id": key + "-never-loaded", "revision": "fixed"},
            tokenizer=tokenizer,
            engineering_row={"partition": "engineering", "row_id": "eng"},
        )
        protocol_path = old / key / "protocol.json"
        write(protocol_path, previous)
        heads = {
            factor: factor + "_head" for factor in ("initial", "update", "first_receipt", "receipt")
        }
        fitted = {
            "capture_provenance": {
                "protocol_sha256": file_sha256(protocol_path),
                "model_settings": previous["model_settings"],
            },
            "pipelines": {"C2": {"factor_heads": heads, "shared_C": 0.25}},
            "heads": {head: {"site": site} for head in heads.values()},
        }
        write(old / (key + "_fit") / "settings.json", fitted)
        for name in ("heads.npz", "manifest.json", "candidates.json"):
            (old / (key + "_fit") / name).write_bytes((key + name).encode())
    token_loads = []

    def token_factory(model_id, **kwargs):
        token_loads.append((model_id, kwargs))
        assert kwargs == {"revision": "fixed", "local_files_only": True}
        return object()

    module = ModuleType("transformers")
    module.AutoTokenizer = SimpleNamespace(from_pretrained=token_factory)
    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.setattr(capture, "ROOT", checkout)
    monkeypatch.setattr(capture, "POLICY_FILES", ("policy.py",))
    monkeypatch.setattr(capture, "native", lambda p: Path(p).resolve())
    monkeypatch.setattr(capture, "load", lambda p: {"fixture": True})
    monkeypatch.setattr(capture, "audit_rows", lambda bank: {"rows_per_checkpoint": 5760})
    monkeypatch.setattr(
        capture,
        "iter_rows",
        lambda bank: (
            {"row_id": f"fixture_{i}", "partition": "fit" if i < 768 else "test"}
            for i in range(5760)
        ),
    )
    monkeypatch.setattr(capture, "tokenizer_binding", lambda unused: tokenizer)
    monkeypatch.setattr(capture, "prepare_row", lambda unused, row, **kwargs: copy.deepcopy(row))
    monkeypatch.setattr(capture, "environment_info", lambda: {"fixture": True})
    return SimpleNamespace(
        bank=bank, receipt=receipt, old=old, output=tmp_path / "prepared", token_loads=token_loads
    )


def test_preparation_freezes_both_protocols_and_old_bundle_fit_plan_bindings(preparation_fixture):
    f = preparation_fixture
    capture.prepare_campaign(f.bank, f.old, f.output, f.receipt)
    preparation = read_json(f.output / "preparation.json")
    assert preparation["status"] == "prepared_no_weights"
    assert len(f.token_loads) == 2
    assert preparation["bank_sha256"] == file_sha256(f.bank)
    for path, digest in preparation["source_sha256"].items():
        assert file_sha256(path) == digest
        snapshot = f.output / "source_snapshots" / (digest + "_" + Path(path).name + ".txt")
        assert snapshot.read_bytes() == Path(path).read_bytes()
    for key in ("llama", "qwen"):
        directory = f.output / key
        plan = read_json(directory / "fit_plan.json")
        protocol = read_json(directory / "protocol.json")
        assert preparation["model_protocol_sha256"][key] == file_sha256(directory / "protocol.json")
        assert preparation["fit_plan_sha256"][key] == file_sha256(directory / "fit_plan.json")
        assert protocol["fit_plan_sha256"] == file_sha256(directory / "fit_plan.json")
        assert protocol["old_C2_C"] == 0.25
        assert protocol["test_fixture"] is False
        assert len(protocol["planned_rows"]) == 5760
        assert len(plan["rows"]) == 768 and all(r["partition"] == "fit" for r in plan["rows"])
        assert plan["bank_sha256"] == file_sha256(f.bank)
        assert plan["old_bundle_path"] == str(f.old / (key + "_fit"))
        assert set(plan["old_bundle_sha256"]) == {
            "settings.json",
            "heads.npz",
            "manifest.json",
            "candidates.json",
        }
        for name, digest in plan["old_bundle_sha256"].items():
            assert file_sha256(f.old / (key + "_fit") / name) == digest


def test_preparation_refuses_existing_root_without_tokenizer_or_weights(preparation_fixture):
    f = preparation_fixture
    f.output.mkdir()
    (f.output / "preserve.txt").write_bytes(b"earlier attempt")
    with pytest.raises(ValueError, match="fresh campaign root"):
        capture.prepare_campaign(f.bank, f.old, f.output, f.receipt)
    assert f.token_loads == []
    assert (f.output / "preserve.txt").read_bytes() == b"earlier attempt"


@pytest.mark.parametrize("mutation", ["protocol_binding", "fixed_site"])
def test_preparation_rejects_old_bundle_provenance_or_site_reselection(
    preparation_fixture, mutation
):
    f = preparation_fixture
    settings_path = f.old / "llama_fit/settings.json"
    settings = read_json(settings_path)
    if mutation == "protocol_binding":
        settings["capture_provenance"]["protocol_sha256"] = "0" * 64
    else:
        for head in settings["heads"].values():
            head["site"] = "layer_1_query_location_last"
    write(settings_path, settings)
    with pytest.raises(ValueError):
        capture.prepare_campaign(f.bank, f.old, f.output, f.receipt)
    assert f.token_loads == []
    failure = read_json(f.output / "preparation_failure.json")
    assert failure["model_calls"] == 0
    assert not (f.output / "preparation.json").exists()
