"""Fresh-case validation of one fixed nine-shot reader after old-policy failures."""

from __future__ import annotations

import argparse
import copy
import re
import traceback
from collections import Counter
from datetime import date
from pathlib import Path

from replay_communication_reader import GATE, capability_gate
from run_communication_transfer import strict_decision

from latent_probes.backends import HFBackend
from latent_probes.calibrated_reader import build_reader_messages, policy_metadata
from latent_probes.communication_panel import FAMILIES, audit_disjointness, render_case
from latent_probes.config import GenerationSettings, load_config
from latent_probes.storage import (
    atomic_json,
    canonical_hash,
    environment_info,
    file_sha256,
    read_json,
    utc_now,
)

ROOT = Path(__file__).resolve().parents[1]
READER_SETTINGS = GenerationSettings(max_new_tokens=8, do_sample=False)
SEED = 42
CONTEXTS = ("informed", "absent", "stale")
CASE_FIELDS = {
    "id",
    "family",
    "transfer_group",
    "split",
    "world",
    "rule",
    "source",
    "reader_packets",
    "correct",
    "decisive_fact",
    "irrelevant_fact",
    "packet_expected_decisions",
    "authoring_notes",
}


def validate_bank(panel, references, *, purpose="calibration"):
    """Validate new identities and exact facts without pretending they are old cf cases."""
    fields = {
        "schema_version",
        "panel_id",
        "seed",
        "word_budget",
        "max_new_tokens",
        "status",
        "authoring_notes",
        "cases",
    }
    bank_types = {
        "calibration": ("fresh_calibration_validation_v1", "cv", "fresh_policy_validation"),
        "recipient_test": (
            "fresh_recipient_transfer_test_v1",
            "ct",
            "untouched_recipient_transfer_test",
        ),
    }
    if purpose not in bank_types:
        raise ValueError("Unknown fresh-bank purpose")
    panel_id, prefix, transfer_group = bank_types[purpose]
    if set(panel) != fields or panel["schema_version"] != 1 or panel["panel_id"] != panel_id:
        raise ValueError("Expected the new fixed calibration-validation bank")
    if panel["seed"] != SEED or panel["word_budget"] != 40 or panel["max_new_tokens"] != 96:
        raise ValueError("Fresh-bank settings changed")
    if len(panel["cases"]) != 24 or not panel["status"] or not panel["authoring_notes"]:
        raise ValueError("Expected exactly 24 documented all-test cases")
    identifiers, entities = set(), set()
    for case in panel["cases"]:
        family = case["family"]
        if (
            set(case) != CASE_FIELDS
            or family not in FAMILIES
            or not re.fullmatch(rf"{prefix}_{family}_0[1-4]", case["id"])
        ):
            raise ValueError("Fresh case identity/schema does not match its rule")
        if case["id"] in identifiers or case["world"]["entity"] in entities:
            raise ValueError("Duplicate fresh identity or entity")
        identifiers.add(case["id"])
        entities.add(case["world"]["entity"])
        if case["split"] != "test" or case["transfer_group"] != transfer_group:
            raise ValueError("Every new case must remain validation-only")
        expected = render_case(case["world"], family)
        if any(case[key] != value for key, value in expected.items()):
            raise ValueError("Fresh source, packet, normative label or decisive fact changed")
        if not case["authoring_notes"]:
            raise ValueError("Authorship limits are required")
        if family == "deadline" and max(
            date.fromisoformat(case["world"][key])
            for key in ("deadline", "completion_date", "previous_completion_date")
        ) >= date.fromisoformat(case["world"]["previous_date"]):
            raise ValueError("Deadline events must precede the first final record")
    for family in FAMILIES:
        members = [case for case in panel["cases"] if case["family"] == family]
        if [c["id"] for c in members] != [
            f"{prefix}_{family}_{i:02d}" for i in range(1, 5)
        ] or Counter(c["correct"] for c in members) != {"APPROVE": 2, "REJECT": 2}:
            raise ValueError("Each family needs four ordered cases and balanced source decisions")
    for reference in references:
        audit_disjointness(panel, reference)
        old_numbers = {
            value
            for case in reference["cases"]
            for value in case["world"].values()
            if type(value) is int
        }
        new_numbers = {
            value
            for case in panel["cases"]
            for value in case["world"].values()
            if type(value) is int
        }
        if old_numbers & new_numbers:
            raise ValueError("Fresh numeric instantiations overlap an earlier bank")
    return copy.deepcopy(panel)


def control_jobs(panel, scope):
    return [
        {
            "example_id": scope + "__" + case["id"] + "__" + context,
            "case_id": case["id"],
            "family": case["family"],
            "context": context,
            "scope": scope,
            "expected_decision": case["packet_expected_decisions"][context],
            "messages": build_reader_messages(case, "", context),
        }
        for case in panel["cases"]
        for context in CONTEXTS
    ]


def run_validation_batches(backend, fresh_jobs, secondary_jobs, output):
    """A failed fresh gate prevents all old-case diagnostic calls."""
    fresh, secondary = [], []

    def collect(jobs, destination):
        for job in jobs:
            generation = backend.generate(job["messages"], READER_SETTINGS, SEED).record()
            record = {**job, "generation": generation, "decision": strict_decision(generation)}
            atomic_json(output / "records" / f"{job['example_id']}.json", record)
            destination.append(record)

    collect(fresh_jobs, fresh)
    gate = capability_gate(fresh)
    atomic_json(output / "fresh_controls.json", fresh)
    atomic_json(output / "fresh_gate.json", gate)
    print(f"Fresh fixed-policy reader gate: {gate}", flush=True)
    if not gate["passed"]:
        return {
            "status": "fresh_reader_controls_failed",
            "fresh_gate": gate,
            "n_fresh_calls": len(fresh),
            "n_secondary_calls": 0,
            "secondary_status": "not_run_fresh_gate_failed",
        }
    collect(secondary_jobs, secondary)
    atomic_json(output / "secondary_old_controls.json", secondary)
    secondary_gate = capability_gate(secondary)
    atomic_json(output / "secondary_old_gate.json", secondary_gate)
    return {
        "status": "complete_fresh_gate_passed",
        "fresh_gate": gate,
        "n_fresh_calls": len(fresh),
        "n_secondary_calls": len(secondary),
        "secondary_status": "diagnostic_only",
        "secondary_old_gate": secondary_gate,
    }


def run(args, backend_factory=HFBackend):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Preserve this one fixed-policy validation attempt")
    bank_path = Path(args.bank)
    old_path = ROOT / "data/reader_forecast/development_v1.json"
    main_path = ROOT / "data/communication/fresh_transfer_v1.json"
    untouched_path = ROOT / "data/communication/fresh_recipient_test_v1.json"
    references = [read_json(old_path), read_json(main_path), read_json(untouched_path)]
    panel = validate_bank(read_json(bank_path), references, purpose="calibration")
    fresh_jobs = control_jobs(panel, "fresh_validation")
    old_jobs = control_jobs(references[1], "secondary_observed_main")
    settings = load_config(args.config).model
    if (
        settings.model_id != "Qwen/Qwen2.5-7B-Instruct"
        or settings.revision != "a09a35458c702b33eeacc393d103063234e8bc28"
        or settings.quantization != "4bit"
        or settings.dtype != "bfloat16"
        or not settings.local_files_only
    ):
        raise ValueError("Fixed cached Qwen checkpoint and NF4/BF16 profile must remain unchanged")
    source_files = [
        bank_path,
        old_path,
        main_path,
        untouched_path,
        Path(args.config),
        Path(__file__),
        ROOT / "src/latent_probes/calibrated_reader.py",
        ROOT / "src/latent_probes/communication_panel.py",
        ROOT / "src/latent_probes/forecast_panel.py",
        ROOT / "src/latent_probes/backends.py",
        ROOT / "scripts/replay_communication_reader.py",
        ROOT / "scripts/run_communication_transfer.py",
    ]
    hashes = {str(path.resolve()): file_sha256(path) for path in source_files}
    plan = {
        "schema": "fresh_calibrated_reader_validation_v1",
        "created_utc": utc_now(),
        "selection": "One fixed nine-shot policy developed after old-reader failures; new cases before any outputs; no prompt or model search.",
        "fresh_panel": panel,
        "policy": policy_metadata(),
        "model": settings.__dict__,
        "reader_settings": READER_SETTINGS.__dict__,
        "seed": SEED,
        "gate": GATE,
        "fresh_jobs": fresh_jobs,
        "conditional_secondary_jobs": old_jobs,
        "source_sha256": hashes,
        "order": "72 fresh no-report calls; stop on gate failure; only on pass run 72 observed-main no-report diagnostics.",
        "new_reports": 0,
        "new_probe_fits": 0,
        "limits": [
            "Policy development follows earlier failures; these fresh cases validate one fixed policy, not untouched policy development.",
            "Same six synthetic rule grammars; new instances do not establish family generalization.",
            "Agent-authored source materials still need independent human semantic review.",
            "No saved-report replay, new sender generation or probe-fitting claim follows from a passed control gate.",
        ],
    }
    output.mkdir(parents=True)
    atomic_json(output / "protocol.json", plan)
    manifest = {
        "status": "prepared",
        "created_utc": utc_now(),
        "protocol_sha256": file_sha256(output / "protocol.json"),
        "environment": environment_info(),
        "policy": plan["policy"],
    }
    atomic_json(output / "manifest.json", manifest)
    backend = None
    try:
        backend = backend_factory(settings)
        if (
            backend.metadata.get("is_test_fixture") is not False
            or backend.metadata["model_id"] != settings.model_id
            or backend.metadata["resolved_revision"] != settings.revision
        ):
            raise ValueError("A genuine reader with the fixed checkpoint is required")
        manifest.update(status="running", started_utc=utc_now(), backend=backend.metadata)
        atomic_json(output / "manifest.json", manifest)
        lengths = []
        for job in fresh_jobs + old_jobs:
            encoded = backend.tokenizer.apply_chat_template(
                job["messages"],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            n_tokens = int(encoded["input_ids"].shape[1])
            lengths.append(
                {
                    "example_id": job["example_id"],
                    "prompt_tokens": n_tokens,
                    "prompt_sha256": canonical_hash(job["messages"]),
                }
            )
            if n_tokens > settings.max_input_tokens:
                atomic_json(output / "prompt_lengths.json", lengths)
                raise ValueError(
                    "Fixed policy exceeds context allowance; no prompt truncation allowed"
                )
        atomic_json(output / "prompt_lengths.json", lengths)
        summary = run_validation_batches(backend, fresh_jobs, old_jobs, output)
        if any(file_sha256(path) != digest for path, digest in hashes.items()):
            raise ValueError("A frozen policy, case bank or source file changed")
        atomic_json(output / "summary.json", summary)
        manifest.update(
            **summary,
            completed_utc=utc_now(),
            source_unchanged=True,
            summary_sha256=file_sha256(output / "summary.json"),
        )
    except BaseException:
        manifest.update(status="failed", failure=traceback.format_exc())
        raise
    finally:
        if backend is not None:
            backend.close()
        manifest.update(
            finished_utc=utc_now(),
            source_unchanged=all(file_sha256(path) == digest for path, digest in hashes.items()),
        )
        atomic_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bank", type=Path, default=ROOT / "data/communication/fresh_calibration_v1.json"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen7b_reader_4bit.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/adirj/kcl-latent-probes-runs/calibrated_reader_validation_v1"),
    )
    run(parser.parse_args())
