"""Fixed-cohort capture using the previously audited exact-prefix measurement.

No generation, automatic retries, test-dependent selection or source edits.
Raw twelve-site journals are preserved; only the frozen S1 site reaches fitting.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np

from .backends import HFBackend
from .config import ModelSettings
from .format_adaptation_bank import audit_rows, iter_rows, load
from .fresh_readout_capture import (
    _completed,
    _measure_row,
    _validate_backend,
    prepare_row,
    tokenizer_binding,
)
from .main_study_run import ROOT, snapshot_files
from .storage import atomic_json, atomic_npz, environment_info, file_sha256, read_json, utc_now

MODEL_FIELDS = (
    "model_key",
    "model_settings",
    "model_type",
    "decoder_blocks",
    "hidden_size",
    "vocab_size",
    "max_position_embeddings",
    "forward_contract",
    "tokenizer",
    "sites",
    "position_order",
    "original_site",
)
POLICY_FILES = (
    "src/latent_probes/format_adaptation_bank.py",
    "src/latent_probes/format_adaptation_capture.py",
    "src/latent_probes/format_adaptation_fit.py",
    "src/latent_probes/format_adaptation_analysis.py",
    "scripts/run_format_adaptation_capture.py",
    "scripts/run_format_adaptation_campaign.py",
    "scripts/start_fresh_readout_durable.ps1",
    "scripts/fit_format_adaptation.py",
    "scripts/analyse_format_adaptation.py",
    "scripts/audit_format_adaptation.py",
    "scripts/report_format_adaptation.py",
    "tests/test_audit_format_adaptation.py",
    "tests/test_format_adaptation_bank.py",
    "tests/test_format_adaptation_capture.py",
    "tests/test_format_adaptation_fit.py",
    "tests/test_format_adaptation_analysis.py",
    "tests/test_format_adaptation_campaign.py",
    "tests/test_format_adaptation_report.py",
    "src/latent_probes/fresh_readout_capture.py",
    "src/latent_probes/fresh_readout_bank.py",
    "src/latent_probes/main_study_bank.py",
    "src/latent_probes/main_study_design.py",
    "src/latent_probes/main_study_run.py",
    "src/latent_probes/backends.py",
    "src/latent_probes/config.py",
    "src/latent_probes/storage.py",
    "src/latent_probes/crossed_belief_validation.py",
    "docs/FORMAT_ADAPTATION_EXECUTION_PROTOCOL_V1.md",
    "docs/FORMAT_ADAPTATION_PROTOCOL_REVIEW_2026-11-27.md",
    "docs/validation/2026-11-27-format-adaptation-bank-review-v1.json",
    "docs/validation/2026-11-27-format-adaptation-capture-review-v1.json",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def native(path):
    path = Path(path).resolve()
    require(
        platform.system() == "Linux" and not str(path).startswith("/mnt/"),
        "Use native Linux storage for real captures",
    )
    require(not path.is_relative_to(ROOT), "Capture output must be outside the checkout")
    return path


def fresh_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def validate_agent_review(bank_path, receipt_path):
    receipt = read_json(receipt_path)
    require(
        receipt.get("schema") == "format-adaptation-agent-review-v1"
        and receipt.get("status") == "pass"
        and receipt.get("human_review_complete") is False
        and receipt.get("bank_sha256") == file_sha256(bank_path),
        "Need exact-bank exploratory agent review",
    )
    inputs = receipt["input_sha256"]
    snapshot_files(inputs)
    return receipt


def prepare_campaign(bank_path, old_root, output, review_receipt, launch_script=None):
    from transformers import AutoTokenizer

    output, old_root = native(output), Path(old_root).resolve()
    require(not output.exists(), "Use a fresh campaign root; preserve earlier attempts")
    bank_path, review_receipt = Path(bank_path).resolve(), Path(review_receipt).resolve()
    bank = load(bank_path)
    audit = audit_rows(bank)
    review = validate_agent_review(bank_path, review_receipt)
    policy = [ROOT / p for p in POLICY_FILES] + [bank_path, review_receipt]
    if launch_script is not None:
        policy.append(Path(launch_script).resolve())
    policy += [Path(p) for p in review["input_sha256"]]
    for key in ("llama", "qwen"):
        policy += [
            old_root / key / "protocol.json",
            old_root / (key + "_fit") / "settings.json",
            old_root / (key + "_fit") / "heads.npz",
            old_root / (key + "_fit") / "manifest.json",
            old_root / (key + "_fit") / "candidates.json",
        ]
    sources = {str(p.resolve()): file_sha256(p) for p in policy}
    output.mkdir(parents=True, exist_ok=False)
    source_dir = output / "source_snapshots"
    source_dir.mkdir()
    for path, sha in sources.items():
        (source_dir / (sha + "_" + Path(path).name + ".txt")).write_bytes(Path(path).read_bytes())
    fresh_json(
        output / "preparation_started.json", {"created_at": utc_now(), "source_sha256": sources}
    )
    try:
        protocols, fit_plans = {}, {}
        for key in ("llama", "qwen"):
            previous = read_json(old_root / key / "protocol.json")
            fitted = read_json(old_root / (key + "_fit") / "settings.json")
            provenance = fitted["capture_provenance"]
            require(
                provenance["protocol_sha256"] == file_sha256(old_root / key / "protocol.json")
                and provenance["model_settings"] == previous["model_settings"],
                "Old fitted feature map is not bound to the supplied protocol",
            )
            factor_heads = fitted["pipelines"]["C2"]["factor_heads"]
            sites = {fitted["heads"][head]["site"] for head in factor_heads.values()}
            require(len(sites) == 1, "Old factor heads must share a fixed site")
            site = next(iter(sites))
            require(
                site
                == {
                    "llama": "layer_16_query_location_last",
                    "qwen": "layer_21_query_location_last",
                }[key],
                "The previous fixed capture site cannot be reselected",
            )
            tokenizer = AutoTokenizer.from_pretrained(
                previous["model_settings"]["model_id"],
                revision=previous["model_settings"]["revision"],
                local_files_only=True,
            )
            require(tokenizer_binding(tokenizer) == previous["tokenizer"], "Old tokenizer changed")
            planned = [prepare_row(tokenizer, row, max_tokens=2048) for row in iter_rows(bank)]
            require(len(planned) == 5760, "Wrong prepared input inventory")
            protocol = {
                **{field: previous[field] for field in MODEL_FIELDS},
                "schema": "format-adaptation-capture-protocol-v1",
                "created_at": utc_now(),
                "test_fixture": False,
                "research_status": "agent_audited_exploratory",
                "human_review_complete": False,
                "site": site,
                "bank_path": str(bank_path),
                "bank_sha256": file_sha256(bank_path),
                "review_receipt_sha256": file_sha256(review_receipt),
                "old_bundle": str(old_root / (key + "_fit")),
                "old_C2_C": fitted["pipelines"]["C2"]["shared_C"],
                "source_sha256": sources,
                "planned_rows": planned,
                "engineering_row": prepare_row(
                    tokenizer, previous["engineering_row"], max_tokens=2048
                ),
                "environment": environment_info(),
            }
            directory = output / key
            directory.mkdir()
            fresh_json(
                directory / "fit_plan.json",
                {
                    "schema": "format-adaptation-fit-plan-v1",
                    "model_key": key,
                    "site": site,
                    "bank_sha256": file_sha256(bank_path),
                    "old_bundle_path": str(old_root / (key + "_fit")),
                    "old_bundle_sha256": {
                        name: file_sha256(old_root / (key + "_fit") / name)
                        for name in (
                            "settings.json",
                            "heads.npz",
                            "manifest.json",
                            "candidates.json",
                        )
                    },
                    "rows": [r for r in planned if r["partition"] == "fit"],
                },
            )
            fit_plans[key] = file_sha256(directory / "fit_plan.json")
            protocol["fit_plan_sha256"] = fit_plans[key]
            fresh_json(directory / "protocol.json", protocol)
            protocols[key] = file_sha256(directory / "protocol.json")
            print(f"Prepared {key}: 768 fit + 4992 test prefixes; no weights", flush=True)
        snapshot_files(sources)
        fresh_json(
            output / "preparation.json",
            {
                "schema": "format-adaptation-preparation-v1",
                "status": "prepared_no_weights",
                "created_at": utc_now(),
                "bank_sha256": file_sha256(bank_path),
                "model_protocol_sha256": protocols,
                "fit_plan_sha256": fit_plans,
                "source_sha256": sources,
                "audit": audit,
            },
        )
    except BaseException as error:
        fresh_json(
            output / "preparation_failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "created_at": utc_now(),
                "model_calls": 0,
            },
        )
        raise


def verify_test_seal(model_directory, seal_path):
    directory, seal_path = Path(model_directory).resolve(), Path(seal_path).resolve()
    protocol = read_json(directory / "protocol.json")
    seal = read_json(seal_path)
    require(
        seal.get("schema") == "format-adaptation-test-seal-v1"
        and set(seal["models"]) == {"llama", "qwen"}
        and seal["bank_sha256"] == protocol["bank_sha256"],
        "Invalid both-model test seal",
    )
    snapshot_files(seal["source_sha256"])
    require(seal["source_sha256"] == protocol["source_sha256"], "Seal source map changed")
    for key, entry in seal["models"].items():
        bundle = directory.parent / (key + "_fit")
        for name, field in (
            ("coefficients.npz", "coefficients_sha256"),
            ("settings.json", "settings_sha256"),
            ("fit_manifest.json", "fit_manifest_sha256"),
        ):
            require(file_sha256(bundle / name) == entry[field], "Fitted bundle changed before test")
    return seal


def capture(model_directory, partition, seal_path=None, *, backend_factory=HFBackend):
    directory = native(model_directory)
    require(partition in ("fit", "test"), "Unknown capture cohort")
    protocol_path = directory / "protocol.json"
    protocol, digest = read_json(protocol_path), file_sha256(protocol_path)
    preparation = read_json(directory.parent / "preparation.json")
    require(
        preparation["model_protocol_sha256"][protocol["model_key"]] == digest,
        "Prepared protocol bytes changed",
    )
    snapshot_files(protocol["source_sha256"])
    require(
        preparation["fit_plan_sha256"][protocol["model_key"]] == protocol["fit_plan_sha256"],
        "Prepared fit-only plan fingerprint changed",
    )
    require(
        file_sha256(directory / "fit_plan.json") == protocol["fit_plan_sha256"],
        "Prepared fit-only plan bytes changed",
    )
    now = environment_info()
    require(
        all(now[k] == protocol["environment"][k] for k in ("python", "platform", "packages")),
        "Prepared numerical environment changed",
    )
    if partition == "test":
        require(seal_path is not None, "Both fitted bundles must be sealed before testing")
        verify_test_seal(directory, seal_path)
    cohort = directory / partition
    cohort.mkdir(exist_ok=False)
    status = {
        "schema": "format-adaptation-capture-manifest-v1",
        "status": "running",
        "partition": partition,
        "model_key": protocol["model_key"],
        "started_at": utc_now(),
        "completed_rows": 0,
        "bank_sha256": protocol["bank_sha256"],
        "protocol_sha256": digest,
        "input_artifact_sha256": {str(protocol_path): digest},
        "seal_sha256": file_sha256(seal_path) if seal_path else None,
    }
    if partition == "fit":
        status["fit_plan_path"] = str(directory / "fit_plan.json")
        status["preparation_path"] = str(directory.parent / "preparation.json")
        status["input_artifact_sha256"].update(
            {
                str(directory / "fit_plan.json"): protocol["fit_plan_sha256"],
                str(directory.parent / "preparation.json"): file_sha256(
                    directory.parent / "preparation.json"
                ),
            }
        )
    atomic_json(cohort / "manifest.json", status)
    try:
        backend = backend_factory(ModelSettings(**protocol["model_settings"]))
        _validate_backend(backend, protocol)
        engineering = _measure_row(
            backend,
            protocol,
            digest,
            protocol["engineering_row"],
            cohort / "engineering",
            engineering=True,
        )
        require(engineering["capture"]["hook_alignment_checked"], "Engineering alignment missing")
        index = protocol["sites"].index(protocol["site"])
        selected = [row for row in protocol["planned_rows"] if row["partition"] == partition]
        require(len(selected) == (768 if partition == "fit" else 4992), "Wrong cohort size")
        rows, features, margins, strict = [], [], [], []
        journal = cohort / "journal"
        journal.mkdir()
        for planned in selected:
            row_dir = journal / planned["row_id"]
            _measure_row(backend, protocol, digest, planned, row_dir)
            record = _completed(row_dir, planned, protocol, digest)
            with np.load(row_dir / "arrays.npz", allow_pickle=False) as data:
                features.append(data["features"][index].copy())
            rows.append(record)
            margins.append(record["output_logits"]["yes_minus_no"])
            tokens = protocol["tokenizer"]["answer_token_ids"]
            strict.append(
                record["output_logits"]["argmax_token_id"]
                == tokens["YES" if record["gold"] else "NO"]
            )
            status.update(completed_rows=len(rows), last_progress_at=utc_now())
            atomic_json(cohort / "manifest.json", status)
            if len(rows) % 100 == 0 or len(rows) == len(selected):
                print(
                    f"{protocol['model_key']} {partition}: {len(rows)}/{len(selected)}", flush=True
                )
        atomic_npz(
            cohort / "features.npz",
            row_ids=np.asarray([r["row_id"] for r in rows]),
            features=np.stack(features).astype(np.float32),
            output_margins=np.asarray(margins, dtype=np.float64),
            strict_first_token_correct=np.asarray(strict, dtype=bool),
        )
        atomic_json(
            cohort / "rows.json",
            {
                "schema": "format-adaptation-capture-rows-v1",
                "test_fixture": False,
                "model_key": protocol["model_key"],
                "partition": partition,
                "site": protocol["site"],
                "rows": rows,
                "answer_token_ids": protocol["tokenizer"]["answer_token_ids"],
            },
        )
        snapshot_files(protocol["source_sha256"])
        if partition == "test":
            verify_test_seal(directory, seal_path)
        status.update(
            status="complete",
            finished_at=utc_now(),
            rows_sha256=file_sha256(cohort / "rows.json"),
            features_sha256=file_sha256(cohort / "features.npz"),
            engineering_complete_sha256=file_sha256(cohort / "engineering/complete.json"),
        )
        atomic_json(cohort / "manifest.json", status)
    except BaseException as error:
        status.update(
            status="failed", failed_at=utc_now(), error_type=type(error).__name__, error=str(error)
        )
        atomic_json(cohort / "manifest.json", status)
        raise
