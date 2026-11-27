"""Create review-gated, immutable execution bindings without model operations."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from .main_study_analysis import load_fit_bundle
from .main_study_bank import (
    bank_digest,
    load_bank,
    prompt_digest,
    render_messages,
    representative_cases,
    verify_review_ledger,
)
from .main_study_design import ALLOCATION_SEEDS, TRAINING_SIZES
from .main_study_fit import C_GRID, NUMERICAL_SEED, TFIDF_SETTINGS
from .main_study_monitor import DEFAULT_PROTOCOL, load_monitor_protocol
from .main_study_monitor import POLICY_FILES as MONITOR_POLICY_FILES
from .main_study_review import write_fresh_json
from .main_study_run import MODELS, ROOT, require, snapshot_files
from .main_study_run import POLICY_FILES as CAPTURE_POLICY_FILES
from .storage import environment_info, file_sha256, read_json, utc_now

DEFAULT_BASELINE_PROTOCOL = ROOT / "data/belief_main/baseline_protocol_v1.json"
DEFAULT_REVIEW_PACKET = ROOT / "docs/review/main_study_v1/review_prompts.jsonl"
AGENT_AMENDMENT = ROOT / "docs/MAIN_STUDY_AGENT_EXECUTION_AMENDMENT_2026-11-22.md"
AGENT_FREEZE_SCHEMA = "main-study-agent-exploratory-freeze-v1"
AGENT_RESEARCH_STATUS = "agent_audited_exploratory"
USER_AUTHORIZATION = (
    "well check it yourself please with an independent agent to then keep going and "
    "researching deeply to make sure this is truly research grade and hypercompetitive "
    "for masters applications"
)
EXTRA_POLICY_FILES = (
    "src/latent_probes/main_study_execution.py",
    "scripts/prepare_main_study_execution.py",
    "src/latent_probes/main_study_fit.py",
    "scripts/fit_main_study.py",
    "src/latent_probes/main_study_analysis.py",
    "scripts/analyse_main_study.py",
    "src/latent_probes/main_study_monitor_analysis.py",
    "scripts/analyse_main_study_monitor.py",
    "src/latent_probes/main_study_materials.py",
    "scripts/build_main_study_bank.py",
)


def expected_baseline_policy() -> dict:
    """Return the predeclared numerical/text policy, never observations or fits."""
    return {
        "schema": "main-study-baseline-protocol-v1",
        "status": "specified_unexecuted",
        "target": "fictional sender-attributed reader belief content, not deception",
        "numerical_seed": NUMERICAL_SEED,
        "training_sizes": list(TRAINING_SIZES),
        "allocation_seeds": list(ALLOCATION_SEEDS),
        "fitting": {
            "C_grid": list(C_GRID),
            "solver": "lbfgs",
            "max_iter": 2000,
            "shared_C": "mean of equal-scenario development log losses; lower C breaks ties",
            "nonconvergence": "exclude a C candidate for both arms if either arm fails",
            "activation_pipeline": "float64; train-only StandardScaler then row L2 then logistic",
            "fitting_support": "matched N and story groups; restricted versus full coverage",
            "test_policy": "test and correction rows forbidden to fitting and selection",
        },
        "tfidf": {
            key: "float64" if key == "dtype" else list(value) if isinstance(value, tuple) else value
            for key, value in TFIDF_SETTINGS.items()
        },
        "raw_threshold": 0.5,
        "tie_class": 0,
        "calibration": {
            "loss": "equal-scenario development logistic",
            "slope_min": 0,
            "initial_slope_intercept": [1.0, 0.0],
            "method": "L-BFGS-B",
            "maxiter": 2000,
            "ftol": 1e-12,
            "gtol": 1e-8,
            "penalty": None,
            "scope": "development only; practical comparisons, not primary raw decisions",
        },
        "output_scoring": {
            "candidate_strings": ["YES", "NO"],
            "preflight": "distinct bare single tokens; exact decode; neither special; no truncation",
            "logit_difference": "logit(YES) - logit(NO)",
            "probability": "sigmoid of logit difference",
            "capture": "same forward as the last-prompt-token activation; no generated answer",
        },
        "rule_baselines": ["constant_0", "initial", "final_update", "world"],
        "symbolic_oracle": "labelled separately; may use structured states and gold",
        "text_monitor": {
            "protocol_path": DEFAULT_PROTOCOL.relative_to(ROOT).as_posix(),
            "protocol_sha256": file_sha256(DEFAULT_PROTOCOL),
            "method": "prose evidence extractor then symbolic fold; no gold/state model input",
            "invalid_policy": "null predictions/scores; incorrect in full-denominator metrics",
            "output_directories": "separate fresh preparation for development and test",
        },
        "primary_analysis": {
            "model": "llama",
            "training_n": 512,
            "unit": "correction scenario; paired allocations averaged within group",
            "metric": "full minus restricted complete-quartet accuracy",
            "bootstrap_draws": 10000,
            "bootstrap_seed": NUMERICAL_SEED,
            "bootstrap_unit": "whole correction group, preserving all paired allocation scores",
        },
    }


def _validate_fits(bank: dict, directories: tuple[Path, ...]) -> dict[str, str]:
    expected = {
        (kind, n, seed)
        for kind in ("activation", "tfidf")
        for n in TRAINING_SIZES
        for seed in ALLOCATION_SEEDS
    }
    groups = {g["group_id"]: g["split"] for g in bank["groups"] if g["split"] != "test"}
    bindings, models = {}, set()
    for directory in directories:
        settings, _ = load_fit_bundle(directory)
        require(settings.get("test_fixture") is False, "Test freeze cannot bind fixture heads")
        capture = settings.get("capture_provenance", {})
        require(capture.get("bank_sha256") == bank_digest(bank), "Fitted bank differs")
        require(capture.get("partition") == "train_dev", "Fits must originate from train/dev")
        require(
            settings.get("fitting_group_splits") == groups, "Fitted group/split inventory differs"
        )
        comparisons = settings.get("comparisons", [])
        inventory = [
            (c.get("feature_kind"), c.get("training_n"), c.get("allocation_seed"))
            for c in comparisons
        ]
        require(
            len(inventory) == len(expected) and set(inventory) == expected,
            "Test freeze requires the complete comparison inventory",
        )
        require(
            all(
                c.get("status") in {"complete", "no_converged_shared_candidate"}
                for c in comparisons
            ),
            "Unknown fitted comparison status",
        )
        model = capture.get("model_settings", {}).get("model_id")
        require(
            isinstance(model, str) and model and model not in models,
            "One fit directory per distinct model is required",
        )
        models.add(model)
        for name in ("settings.json", "heads.npz"):
            path = directory / name
            bindings[str(path)] = file_sha256(path)
    return bindings


def build_execution_freeze(
    bank_path: str | Path,
    ledger_path: str | Path,
    baseline_protocol_path: str | Path,
    output_path: str | Path,
    *,
    stage: str = "train_dev",
    fit_directories: tuple[str | Path, ...] = (),
) -> dict:
    """Freeze current-platform paths only after the real human-review gate passes.

    The initial freeze also authorizes monitor development. A separate test-stage
    freeze adds preselected fitted bundles; it never rewrites the initial file.
    No fixture override or unreviewed execution option exists.
    """
    require(stage in {"train_dev", "test"}, "Unknown execution stage")
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError("Preserve the earlier freeze; choose a fresh output path")
    bank_path, ledger_path = Path(bank_path).resolve(), Path(ledger_path).resolve()
    baseline = Path(baseline_protocol_path).resolve()
    bank = load_bank(bank_path)
    review = verify_review_ledger(bank, ledger_path)
    require(review.get("review_complete") is True, "Human semantic review is incomplete")
    require(read_json(baseline) == expected_baseline_policy(), "Baseline policy differs from code")
    load_monitor_protocol(DEFAULT_PROTOCOL)
    directories = tuple(Path(p).resolve() for p in fit_directories)
    require(len(set(directories)) == len(directories), "Duplicate fit directory")
    require(bool(directories) == (stage == "test"), "Only test stage requires fitted directories")
    fitted = _validate_fits(bank, directories)
    require(
        all(
            load_fit_bundle(directory)[0]["capture_provenance"].get("research_status")
            != AGENT_RESEARCH_STATUS
            for directory in directories
        ),
        "Human freeze cannot relabel agent exploratory fits",
    )
    source_paths = {
        ROOT / p for p in (*CAPTURE_POLICY_FILES, *MONITOR_POLICY_FILES, *EXTRA_POLICY_FILES)
    }
    source_paths.update((bank_path, ledger_path, baseline, DEFAULT_PROTOCOL))
    payload = {
        "schema": "main-study-execution-freeze-v1",
        "status": "frozen",
        "stage": stage,
        "created_at": utc_now(),
        "bank_sha256": bank_digest(bank),
        "review": review,
        "review_ledger_path": str(ledger_path),
        "review_ledger_sha256": file_sha256(ledger_path),
        "baseline_protocol_path": str(baseline),
        "baseline_protocol_sha256": file_sha256(baseline),
        "text_monitor_protocol_path": str(DEFAULT_PROTOCOL.resolve()),
        "text_monitor_protocol_sha256": file_sha256(DEFAULT_PROTOCOL),
        "source_sha256": {str(p.resolve()): file_sha256(p) for p in sorted(source_paths)},
        "fitted_artifact_sha256": fitted,
        "environment": environment_info(),
        "model_calls": 0,
        "interpretation": "Execution bindings only; no scientific result or new human attestation",
    }
    write_fresh_json(output, payload)
    return payload


def validate_agent_audit(bank: dict, audit_path: str | Path, packet_path: str | Path) -> dict:
    """Check a saved independent agent audit; never create human acceptances.

    Answers must already exist in an immutable source artifact before this
    function compares them with formal labels. Agreement is a consistency check,
    not authentication of independence, humanity or semantic correctness.
    """
    audit, packet = read_json(audit_path), Path(packet_path).resolve()
    require(audit.get("schema") == "main-study-agent-semantic-audit-v1", "Unknown agent audit")
    require(audit.get("reviewer_type") == "agent", "Agent audit must explicitly identify an agent")
    reviewer = audit.get("reviewer_id")
    require(isinstance(reviewer, str) and reviewer.strip(), "Agent reviewer ID required")
    require(audit.get("author_independent") is True, "Independent agent declaration required")
    require(audit.get("bank_sha256") == bank_digest(bank), "Agent audit bank differs")
    require(audit.get("review_packet_sha256") == file_sha256(packet), "Agent packet hash differs")
    require(
        audit.get("review_scope") == "all_representative_prompts_and_transformations",
        "Agent review scope is incomplete",
    )
    require(
        audit.get("transformation_review")
        == {"location_complement": True, "world_independence": True, "global_name_exchange": True},
        "Agent transformation review is incomplete",
    )
    require(audit.get("unresolved_blockers") == [], "Unresolved agent semantic blockers")
    limitations = audit.get("limitations")
    require(
        isinstance(limitations, list)
        and limitations
        and all(isinstance(value, str) and value.strip() for value in limitations),
        "Agent audit must preserve its limitations",
    )
    try:
        created = datetime.fromisoformat(audit["created_at"].replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Actual agent timestamp required") from exc
    require(created.tzinfo is not None and created <= datetime.now(UTC), "Invalid agent timestamp")
    expected = {}
    for group in bank["groups"]:
        for case in representative_cases(group):
            expected[case.case_id] = {
                "case_id": case.case_id,
                "group_id": group["group_id"],
                "prompt_sha256": prompt_digest(render_messages(group, case)),
                "messages": render_messages(group, case),
            }
    packet_rows = [json.loads(line) for line in packet.read_text(encoding="utf-8").splitlines()]
    require(
        len(packet_rows) == len(expected)
        and {row.get("case_id") for row in packet_rows} == set(expected)
        and all(row == expected[row["case_id"]] for row in packet_rows),
        "Agent packet must exactly match every representative prompt",
    )
    records = audit.get("records")
    require(isinstance(records, list) and len(records) == len(expected), "Incomplete agent answers")
    seen = set()
    gold = {
        case.case_id: "YES" if case.gold else "NO"
        for group in bank["groups"]
        for case in representative_cases(group)
    }
    for record in records:
        require(isinstance(record, dict), "Invalid agent answer record")
        case_id = record.get("case_id")
        require(case_id in expected and case_id not in seen, "Duplicate or unknown agent answer")
        seen.add(case_id)
        require(
            record.get("prompt_sha256") == expected[case_id]["prompt_sha256"],
            "Agent answer prompt hash differs",
        )
        require(
            record.get("answer") == gold[case_id], "Unresolved independent agent answer mismatch"
        )
        require(record.get("issues") == [], "Unresolved agent case issue")
        assumptions = record.get("assumptions")
        require(
            isinstance(assumptions, list)
            and all(isinstance(value, str) and value.strip() for value in assumptions),
            "Agent prose assumptions must be explicit",
        )
    return {
        "reviewer_type": "agent",
        "reviewer_id": reviewer,
        "representative_prompts": len(expected),
        "groups": len(bank["groups"]),
        "answers_agree_with_formal_rule": True,
        "human_review_complete": False,
        "review_complete": False,
        "limitations": limitations,
    }


def _require_versioned_amendment() -> None:
    """Require the prospective amendment in Git's index before authorizing runs."""
    relative = AGENT_AMENDMENT.relative_to(ROOT).as_posix()
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    require(
        result.returncode == 0, "Prospective agent amendment must be versioned before execution"
    )
    require(
        USER_AUTHORIZATION in AGENT_AMENDMENT.read_text(encoding="utf-8"), "Authorization absent"
    )


def build_agent_execution_freeze(
    bank_path: str | Path,
    audit_path: str | Path,
    baseline_protocol_path: str | Path,
    output_path: str | Path,
    *,
    stage: str = "train_dev",
    fit_directories: tuple[str | Path, ...] = (),
    initial_freeze_path: str | Path | None = None,
    review_packet_path: str | Path = DEFAULT_REVIEW_PACKET,
) -> dict:
    """Prospectively bind a separate agent-audited exploratory execution.

    This never calls or relaxes the original human gate. The test stage requires
    both pinned models' train/dev fits and the unchanged initial agent freeze.
    """
    require(stage in {"train_dev", "test"}, "Unknown execution stage")
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError("Preserve the earlier freeze; choose a fresh output path")
    bank_path, audit_path = Path(bank_path).resolve(), Path(audit_path).resolve()
    baseline, packet = Path(baseline_protocol_path).resolve(), Path(review_packet_path).resolve()
    bank = load_bank(bank_path)
    review = validate_agent_audit(bank, audit_path, packet)
    _require_versioned_amendment()
    require(read_json(baseline) == expected_baseline_policy(), "Baseline policy differs from code")
    load_monitor_protocol(DEFAULT_PROTOCOL)
    directories = tuple(Path(path).resolve() for path in fit_directories)
    require(len(set(directories)) == len(directories), "Duplicate fit directory")
    require(bool(directories) == (stage == "test"), "Only test stage requires fitted directories")
    require(
        (initial_freeze_path is not None) == (stage == "test"), "Test needs initial agent freeze"
    )
    fitted = _validate_fits(bank, directories)
    sources = {
        ROOT / p for p in (*CAPTURE_POLICY_FILES, *MONITOR_POLICY_FILES, *EXTRA_POLICY_FILES)
    }
    sources.update((bank_path, audit_path, baseline, packet, DEFAULT_PROTOCOL, AGENT_AMENDMENT))
    if stage == "test":
        models = {
            load_fit_bundle(directory)[0]["capture_provenance"]["model_settings"]["model_id"]
            for directory in directories
        }
        require(
            models == {model["model_id"] for model in MODELS.values()},
            "Both pinned model fits required",
        )
        for directory in directories:
            provenance = load_fit_bundle(directory)[0]["capture_provenance"]
            require(
                provenance.get("research_status") == AGENT_RESEARCH_STATUS
                and provenance.get("human_review_complete") is False,
                "Agent test fits must originate from agent exploratory captures",
            )
        initial_path = Path(initial_freeze_path).resolve()
        initial = read_json(initial_path)
        require(
            initial.get("schema") == AGENT_FREEZE_SCHEMA
            and initial.get("status") == "frozen"
            and initial.get("stage") == "train_dev"
            and initial.get("bank_sha256") == bank_digest(bank)
            and initial.get("agent_audit_sha256") == file_sha256(audit_path)
            and initial.get("research_status") == AGENT_RESEARCH_STATUS
            and initial.get("human_review_complete") is False,
            "Initial agent freeze differs",
        )
        snapshot_files(initial.get("source_sha256"))
        require(
            all(
                load_fit_bundle(directory)[0]["capture_provenance"]
                .get("source_bindings", {})
                .get(str(initial_path))
                == file_sha256(initial_path)
                for directory in directories
            ),
            "Fitted captures must bind this exact initial agent freeze",
        )
        sources.add(initial_path)
    payload = {
        "schema": AGENT_FREEZE_SCHEMA,
        "status": "frozen",
        "stage": stage,
        "created_at": utc_now(),
        "research_status": AGENT_RESEARCH_STATUS,
        "human_review_complete": False,
        "confirmatory_result": False,
        "bank_sha256": bank_digest(bank),
        "review": review,
        "agent_audit_path": str(audit_path),
        "agent_audit_sha256": file_sha256(audit_path),
        "review_packet_path": str(packet),
        "review_packet_sha256": file_sha256(packet),
        "baseline_protocol_path": str(baseline),
        "baseline_protocol_sha256": file_sha256(baseline),
        "text_monitor_protocol_path": str(DEFAULT_PROTOCOL.resolve()),
        "text_monitor_protocol_sha256": file_sha256(DEFAULT_PROTOCOL),
        "authorization": {"source": "user request, 2026-11-22", "statement": USER_AUTHORIZATION},
        "amendment_path": str(AGENT_AMENDMENT.resolve()),
        "amendment_sha256": file_sha256(AGENT_AMENDMENT),
        "initial_freeze_path": str(Path(initial_freeze_path).resolve())
        if initial_freeze_path
        else None,
        "source_sha256": {str(p.resolve()): file_sha256(p) for p in sorted(sources)},
        "fitted_artifact_sha256": fitted,
        "environment": environment_info(),
        "model_calls": 0,
        "interpretation": "Agent-audited exploratory execution; no human approval or confirmatory result",
    }
    write_fresh_json(output, payload)
    return payload


def authorize_agent_execution(
    protocol, audit_path, freeze_path, *, partition, fit_dir, monitor=False
):
    """Validate only the distinct exploratory schema, retaining every binding."""
    freeze = read_json(freeze_path)
    require(
        freeze.get("schema") == AGENT_FREEZE_SCHEMA
        and freeze.get("status") == "frozen"
        and freeze.get("bank_sha256") == protocol["bank_sha256"]
        and freeze.get("research_status") == AGENT_RESEARCH_STATUS
        and freeze.get("human_review_complete") is False
        and freeze.get("confirmatory_result") is False,
        "Invalid agent exploratory freeze",
    )
    require(
        freeze.get("stage") == ("test" if partition == "test" else "train_dev"),
        "Agent freeze stage does not authorize this partition",
    )
    audit = Path(audit_path).resolve()
    require(
        freeze.get("agent_audit_path") == str(audit)
        and freeze.get("agent_audit_sha256") == file_sha256(audit),
        "Agent audit identity changed",
    )
    snapshot_files(freeze.get("source_sha256"))
    required_sources = {
        ROOT / p for p in (*CAPTURE_POLICY_FILES, *MONITOR_POLICY_FILES, *EXTRA_POLICY_FILES)
    }
    required_sources.update(
        (
            Path(protocol["bank_path"]),
            audit,
            AGENT_AMENDMENT,
            Path(freeze["baseline_protocol_path"]),
            Path(freeze["review_packet_path"]),
            DEFAULT_PROTOCOL,
        )
    )
    require(
        all(
            freeze["source_sha256"].get(str(p.resolve())) == file_sha256(p)
            for p in required_sources
        )
        and all(
            freeze["source_sha256"].get(path) == digest
            for path, digest in protocol["policy_sha256"].items()
        ),
        "Agent freeze must bind all source and input policies",
    )
    require(
        freeze.get("amendment_sha256") == file_sha256(AGENT_AMENDMENT)
        and freeze.get("authorization", {}).get("statement") == USER_AUTHORIZATION,
        "Agent amendment or user authorization changed",
    )
    bank = load_bank(protocol["bank_path"])
    review = validate_agent_audit(bank, audit, freeze["review_packet_path"])
    baseline = Path(freeze["baseline_protocol_path"]).resolve()
    require(
        file_sha256(baseline) == freeze.get("baseline_protocol_sha256")
        and read_json(baseline) == expected_baseline_policy(),
        "Agent baseline policy changed",
    )
    if monitor:
        require(
            str(Path(freeze["text_monitor_protocol_path"]).resolve())
            == protocol["monitor_protocol_path"]
            and freeze.get("text_monitor_protocol_sha256") == protocol["monitor_protocol_sha256"],
            "Agent monitor policy differs",
        )
    bindings = {
        str(Path(freeze_path).resolve()): file_sha256(freeze_path),
        **freeze["source_sha256"],
    }
    if partition == "test":
        require(fit_dir is not None, "Agent test capture requires frozen fitted heads")
        directory = Path(fit_dir).resolve()
        _validate_fits(bank, (directory,))
        settings, _ = load_fit_bundle(directory)
        capture = settings["capture_provenance"]
        require(
            capture.get("research_status") == AGENT_RESEARCH_STATUS
            and capture.get("human_review_complete") is False
            and settings.get("test_fixture") is False,
            "Agent fit research status differs",
        )
        if not monitor:
            require(
                capture.get("model_settings") == protocol["model_settings"]
                and capture.get("capture_layer") == protocol["capture_layer"],
                "Agent fitted model/capture layer differs",
            )
        for name in ("settings.json", "heads.npz"):
            path = str(directory / name)
            digest = file_sha256(path)
            require(
                freeze.get("fitted_artifact_sha256", {}).get(path) == digest,
                "Agent fitted artifact changed",
            )
            bindings[path] = digest
    return {
        "review": review,
        "research_status": AGENT_RESEARCH_STATUS,
        "human_review_complete": False,
        "confirmatory_result": False,
        "source_bindings" if monitor else "sha256": bindings,
    }
