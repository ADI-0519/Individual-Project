"""Create review-gated, immutable execution bindings without model operations."""

from __future__ import annotations

from pathlib import Path

from .main_study_analysis import load_fit_bundle
from .main_study_bank import bank_digest, load_bank, verify_review_ledger
from .main_study_design import ALLOCATION_SEEDS, TRAINING_SIZES
from .main_study_fit import C_GRID, NUMERICAL_SEED, TFIDF_SETTINGS
from .main_study_monitor import DEFAULT_PROTOCOL, load_monitor_protocol
from .main_study_monitor import POLICY_FILES as MONITOR_POLICY_FILES
from .main_study_review import write_fresh_json
from .main_study_run import POLICY_FILES as CAPTURE_POLICY_FILES
from .main_study_run import ROOT, require
from .storage import environment_info, file_sha256, read_json, utc_now

DEFAULT_BASELINE_PROTOCOL = ROOT / "data/belief_main/baseline_protocol_v1.json"
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
