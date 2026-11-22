"""Frozen, development-only fresh readout fitting and safe-array inference.

The source selector sees original one-step development labels only. Correction
controls share S1's site and use a separate correction development partition.
No test row is accepted by the fitting loader, even if it has no known ID.
"""

from __future__ import annotations

import json
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize
from threadpoolctl import threadpool_info, threadpool_limits

from .storage import (
    atomic_json,
    atomic_npz,
    canonical_hash,
    environment_info,
    file_sha256,
    read_json,
    utc_now,
)

ROOT = Path(__file__).resolve().parents[2]
FIT_PARTITIONS = ("source_fit", "source_dev", "correction_fit", "correction_dev")
DIRECT_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
FACTOR_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
FACTORS = ("I", "M1", "B1", "B2")
FACTOR_FIELDS = ("initial", "update", "first_receipt", "receipt")
POLICY_FILES = (
    "src/latent_probes/fresh_readout_fit.py",
    "scripts/fit_fresh_readout.py",
)
THREAD_LIMIT = 4


def require(condition, message):
    if not condition:
        raise ValueError(message)


class CandidateFailure(ValueError):
    """A candidate is retained as a failure and cannot become a fallback."""


def fit_policy():
    """The complete policy is bound before fitting and saved with every bundle."""
    return {
        "branch": "twelve_site_source_development_selection",
        "C2": "fourfactor_lr",
        "source_allocation": {"n": 512, "seed": 11, "support": "full"},
        "training_partitions": list(FIT_PARTITIONS),
        "development_absorbed": False,
        "transform": ["float64", "training_only_StandardScaler", "row_L2"],
        "logistic_regression": {
            "solver": "lbfgs",
            "penalty": "l2",
            "fit_intercept": True,
            "max_iter": 2000,
            "tol": 1e-4,
            "class_weight": None,
            "sample_weight": None,
            "random_state": 11,
        },
        "direct_C_grid": list(DIRECT_GRID),
        "factor_C_grid": list(FACTOR_GRID),
        "source_selection": "equal_group_source_dev_log_loss",
        "C1_selection": "equal_group_correction_dev_log_loss",
        "C2_selection": "mean_four_equal_group_correction_dev_factor_log_loss",
        "exact_ties": ["lower_C", "original_site", "lower_block", "declared_position_order"],
        "position_order": ["account_last", "query_location_last", "prompt_last"],
        "factor_targets": list(FACTORS),
        "C2_fold": "1-M1 if B2 else M1 if B1 else I",
        "C2_privilege": "four supervised labels per row and externally supplied fold",
        "equal_capacity_claim": False,
        "margin_orientation": "positive_is_symbolic_target_1",
        "threshold": "margin > 0; exact zero predicts 0",
        "calibration": None,
        "output_baseline": "YES_minus_NO; sigmoid is conditional on {YES,NO} only",
        "selected_refit": False,
        "candidate_failure": "nonconvergence_or_nonfinite_invalidates; no_rescue",
        "numerical_thread_limit": THREAD_LIMIT,
    }


def _fitting_sources():
    return {str((ROOT / p).resolve()): file_sha256(ROOT / p) for p in POLICY_FILES}


def _check_hashes(bindings):
    require(isinstance(bindings, dict) and bindings, "Missing artifact bindings")
    for path, digest in bindings.items():
        require(Path(path).is_file() and file_sha256(path) == digest, f"Changed input: {path}")


def _identity(row):
    return row.get("row_id", row.get("record_id", row.get("case_id")))


def _bank_rows(path, *, fixture):
    payload = read_json(path)
    if fixture and "rows" in payload:
        return payload["rows"]
    from .fresh_readout_bank import iter_rows, load_bank

    return list(iter_rows(load_bank(path)))


def _protocol_metadata(protocol, protocol_path, bank_path, *, allow_test_fixture):
    from .fresh_readout_capture import BLOCKS, panel_sites, verify_protocol
    from .main_study_run import MODELS

    require(protocol.get("schema") == "fresh-readout-capture-protocol-v1", "Wrong protocol schema")
    fixture = protocol.get("test_fixture")
    require(
        type(fixture) is bool and (not fixture or allow_test_fixture),
        "Fixture input requires explicit allow_test_fixture",
    )
    require(
        Path(protocol["bank_path"]).resolve() == Path(bank_path).resolve()
        and file_sha256(bank_path) == protocol["bank_file_sha256"],
        "Wrong bank binding",
    )
    key = protocol.get("model_key")
    require(key in MODELS, "Unknown model")
    settings = protocol["model_settings"]
    require(
        settings["model_id"] == MODELS[key]["model_id"]
        and settings["revision"] == MODELS[key]["revision"]
        and settings["capture_layers"] == list(BLOCKS[key])
        and settings["quantization"] == "4bit"
        and settings["dtype"] == "bfloat16",
        "Wrong pinned model settings",
    )
    require(protocol["sites"] == panel_sites(key), "Wrong twelve-site panel")
    require(
        protocol["original_site"] == f"layer_{16 if key == 'llama' else 14}_prompt_last",
        "Wrong original site",
    )
    verify_protocol(protocol)
    sources = _fitting_sources()
    require(
        all(protocol["source_sha256"].get(p) == d for p, d in sources.items()),
        "Fitting policy must be bound in the protocol before fitting",
    )
    return {
        "bank_file_sha256": protocol["bank_file_sha256"],
        "model_key": key,
        "model_settings": settings,
        "sites": protocol["sites"],
        "protocol_sha256": file_sha256(protocol_path),
    }


def load_training_capture(capture_dir, bank_path, protocol_path, *, allow_test_fixture=False):
    """Validate metadata, IDs and partition exclusion before opening feature arrays."""
    from .fresh_readout_capture import _completed

    directory, bank_path, protocol_path = map(Path, (capture_dir, bank_path, protocol_path))
    protocol = read_json(protocol_path)
    provenance = _protocol_metadata(
        protocol, protocol_path, bank_path, allow_test_fixture=allow_test_fixture
    )
    payload, manifest = read_json(directory / "rows.json"), read_json(directory / "manifest.json")
    rows = payload.get("rows")
    require(isinstance(rows, list) and rows, "Empty training capture")
    # This check precedes ALL feature I/O. Do not filter unknown or held-out rows.
    require(
        all(r.get("partition") in FIT_PARTITIONS for r in rows),
        "Test or unknown partition is forbidden in fitting input",
    )
    fixture, digest = protocol["test_fixture"], provenance["protocol_sha256"]
    require(
        payload.get("schema") == "fresh-readout-capture-v1"
        and payload.get("test_fixture") is fixture
        and manifest.get("test_fixture") is fixture,
        "Capture fixture/schema differs",
    )
    require(
        manifest.get("schema") == "fresh-readout-capture-manifest-v1"
        and manifest.get("status") == "complete"
        and manifest.get("partition") == "training"
        and manifest.get("protocol_sha256") == digest
        and manifest.get("completed_rows") == len(rows),
        "Incomplete/wrong training manifest",
    )
    require(
        payload["provenance"].get("partition") == "training"
        and all(payload["provenance"].get(k) == v for k, v in provenance.items())
        and payload["provenance"].get("original_site") == protocol["original_site"],
        "Capture provenance differs",
    )
    for filename, field in (("features.npz", "features_sha256"), ("rows.json", "rows_sha256")):
        require(
            file_sha256(directory / filename) == manifest.get(field),
            "Capture output hash differs: " + filename,
        )
    _check_hashes(manifest["authorization"]["sha256"])
    require(
        all(
            manifest["authorization"]["sha256"].get(p) == d
            for p, d in protocol["source_sha256"].items()
        ),
        "Capture omitted policy bindings",
    )
    bank_rows = _bank_rows(bank_path, fixture=fixture)
    require(len(bank_rows) == len({_identity(r) for r in bank_rows}), "Duplicate bank IDs")
    expected = [r for r in bank_rows if r.get("partition") in FIT_PARTITIONS]
    require(
        all(r.get("partition") in (*FIT_PARTITIONS, "test") for r in bank_rows),
        "Unknown bank partition",
    )
    planned = protocol["planned_rows"]
    require(len(planned) == len({_identity(r) for r in planned}), "Duplicate planned IDs")
    expected_by_id, planned_by_id = ({_identity(r): r for r in rs} for rs in (expected, planned))
    ids = [_identity(r) for r in rows]
    expected_ids = [_identity(r) for r in planned if r["partition"] in FIT_PARTITIONS]
    require(
        len(ids) == len(set(ids)) and ids == expected_ids and set(ids) == set(expected_by_id),
        "Wrong or reordered training row IDs",
    )
    for row in rows:
        original, plan = expected_by_id[_identity(row)], planned_by_id[_identity(row)]
        require(
            all(plan.get(k) == v for k, v in original.items())
            and all(row.get(k) == v for k, v in plan.items()),
            "Row metadata differs from bank/protocol",
        )
        require(
            row.get("renderer") == "A" and type(row.get("gold")) is int and row["gold"] in (0, 1),
            "Invalid fitting renderer/label",
        )
        require(
            row.get("protocol_sha256") == digest
            and row.get("test_fixture") is fixture
            and row.get("status") == "complete",
            "Row capture binding differs",
        )
    counts = Counter(r["partition"] for r in rows)
    require(set(counts) == set(FIT_PARTITIONS), "Need all four fitting/development partitions")
    require(fixture or counts == dict.fromkeys(FIT_PARTITIONS, 512), "Need exact N512 partitions")
    for fit, dev in (("source_fit", "source_dev"), ("correction_fit", "correction_dev")):
        require(
            not {r["group_id"] for r in rows if r["partition"] == fit}.intersection(
                r["group_id"] for r in rows if r["partition"] == dev
            ),
            "Fitting/development group overlap",
        )
    require(
        not {r["group_id"] for r in rows}.intersection(
            r["group_id"] for r in bank_rows if r["partition"] == "test"
        ),
        "Test contextual group overlap",
    )
    journals = (directory / "row_journal").exists()
    require(fixture or journals, "Native fitting requires verified per-row capture journals")
    with np.load(directory / "features.npz", allow_pickle=False) as saved:
        require(
            set(saved.files) == {"features", "row_ids", "sites"}, "Wrong feature-array inventory"
        )
        require(
            saved["row_ids"].dtype.kind in "US"
            and saved["row_ids"].tolist() == ids
            and saved["sites"].dtype.kind in "US"
            and saved["sites"].tolist() == provenance["sites"],
            "NPZ row/site identities differ",
        )
        features = saved["features"].copy()
    require(
        features.dtype == np.float32
        and features.shape == (len(rows), 12, protocol["hidden_size"])
        and np.isfinite(features).all(),
        "Invalid feature matrix",
    )
    if journals:
        require(
            {p.name for p in (directory / "row_journal").iterdir()} == set(ids),
            "Unexpected/missing row journal",
        )
        for index, row in enumerate(rows):
            journal = directory / "row_journal" / _identity(row)
            complete = _completed(journal, planned_by_id[_identity(row)], protocol, digest)
            require(
                complete is not None and all(row.get(k) == v for k, v in complete.items()),
                "Aggregate row differs from its journal",
            )
            with np.load(journal / "arrays.npz", allow_pickle=False) as saved:
                require(
                    np.array_equal(saved["features"], features[index]),
                    "Aggregate features differ from their row journal",
                )
    inputs = {
        str((directory / n).resolve()): file_sha256(directory / n)
        for n in ("manifest.json", "rows.json", "features.npz")
    }
    inputs.update(
        {str(bank_path.resolve()): file_sha256(bank_path), str(protocol_path.resolve()): digest}
    )
    return {
        "rows": rows,
        "features": features,
        "protocol": protocol,
        "provenance": provenance,
        "input_artifact_sha256": inputs,
        "row_journal_verified": journals,
    }


def transform(features, parameters):
    values = np.asarray(features, dtype=np.float64)
    require(values.ndim == 2 and np.isfinite(values).all(), "Nonfinite/invalid transform input")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        values = normalize((values - parameters["mean"]) / parameters["scale"], norm="l2")
    require(np.isfinite(values).all(), "Nonfinite transformed features")
    return values


def fit_transform(features):
    scaler = StandardScaler().fit(np.asarray(features, dtype=np.float64))
    parameters = {
        "mean": scaler.mean_.copy(),
        "scale": scaler.scale_.copy(),
        "var": scaler.var_.copy(),
    }
    require(
        all(np.isfinite(v).all() for v in parameters.values()) and (parameters["scale"] > 0).all(),
        "Invalid fitted scaler",
    )
    return parameters


def equal_group_log_loss(labels, margins, groups):
    labels, margins, groups = np.asarray(labels), np.asarray(margins), np.asarray(groups)
    require(
        labels.shape == margins.shape == groups.shape
        and labels.ndim == 1
        and len(labels) > 0
        and np.isfinite(margins).all(),
        "Invalid log-loss inputs",
    )
    losses = np.logaddexp(0.0, margins) - labels * margins
    result = float(np.mean([losses[groups == g].mean() for g in sorted(set(groups))]))
    require(np.isfinite(result), "Nonfinite development loss")
    return result


def _fit_candidate(x, labels, C):
    model = LogisticRegression(
        C=C,
        solver="lbfgs",
        penalty="l2",
        fit_intercept=True,
        max_iter=2000,
        tol=1e-4,
        class_weight=None,
        random_state=11,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, labels)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise CandidateFailure("Candidate did not converge")
    if not np.array_equal(model.classes_, [0, 1]):
        raise CandidateFailure("Both ordered binary classes required")
    if not all(np.isfinite(a).all() for a in (model.coef_, model.intercept_)):
        raise CandidateFailure("Nonfinite fitted parameters")
    return model


def _candidate(x, labels, dev_x, dev_labels, groups, C):
    model = _fit_candidate(x, labels, C)
    margins = model.decision_function(dev_x)
    if not np.isfinite(margins).all():
        raise CandidateFailure("Nonfinite development margins")
    loss = equal_group_log_loss(dev_labels, margins, groups)
    return model, loss


def site_selection_key(candidate, original_site):
    match = re.fullmatch(
        r"layer_(\d+)_(account_last|query_location_last|prompt_last)", candidate["site"]
    )
    require(match is not None, "Invalid site name")
    return (
        candidate["dev_group_log_loss"],
        candidate["C"],
        int(candidate["site"] != original_site),
        int(match[1]),
        fit_policy()["position_order"].index(match[2]),
    )


def fold_factors(bits):
    bits = np.asarray(bits)
    require(
        bits.ndim == 2 and bits.shape[1] == 4 and np.isin(bits, (0, 1)).all(),
        "Need four binary factor predictions",
    )
    i, u, b1, b2 = bits.T
    return np.where(b2, 1 - u, np.where(b1, u, i)).astype(np.int64)


def _indices(rows, partition):
    return np.asarray([i for i, r in enumerate(rows) if r["partition"] == partition], dtype=int)


def fit_bundle(capture_dir, bank_path, protocol_path, output_dir, *, allow_test_fixture=False):
    """Write selected actual estimators, never refit or absorb development rows."""
    data = load_training_capture(
        capture_dir, bank_path, protocol_path, allow_test_fixture=allow_test_fixture
    )
    output = Path(output_dir).resolve()
    require(not output.exists(), "Fit output already exists; preserve the attempted fit")
    output.mkdir(parents=True)
    rows, features, protocol = data["rows"], data["features"], data["protocol"]
    ids = {p + "_ids": [_identity(r) for r in rows if r["partition"] == p] for p in FIT_PARTITIONS}
    settings = {
        "schema": "fresh-readout-fit-v2",
        "status": "fitting",
        "created_at": utc_now(),
        "test_fixture": protocol["test_fixture"],
        "research_status": "agent_audited_exploratory",
        "human_review_complete": False,
        "confirmatory": False,
        "capture_provenance": data["provenance"],
        "original_site": protocol["original_site"],
        "hidden_size": features.shape[-1],
        **ids,
        "policy": fit_policy(),
        "policy_sha256": canonical_hash(fit_policy()),
        "fitting_source_sha256": _fitting_sources(),
        "input_artifact_sha256": data["input_artifact_sha256"],
        "fit_context": {
            "scope": "training_development_only",
            "test_rows_loaded": 0,
            "development_absorbed": False,
            "row_journal_verified": data["row_journal_verified"],
            "protocol_path": str(Path(protocol_path).resolve()),
            "bank_path": str(Path(bank_path).resolve()),
            "capture_dir": str(Path(capture_dir).resolve()),
        },
        "environment": environment_info(),
        "heads": {},
        "source_panel": [],
        "pipelines": {},
    }
    history = {"source_candidates": [], "C1_candidates": [], "C2_candidates": []}
    manifest = {
        "schema": "fresh-readout-fit-manifest-v2",
        "status": "fitting",
        "started_at": utc_now(),
        "test_fixture": protocol["test_fixture"],
    }
    atomic_json(output / "manifest.json", manifest)
    arrays = {}

    def retain(key, model, parameters, site, C, target, partition, score):
        for name, value in {
            **parameters,
            "coef": model.coef_[0],
            "intercept": model.intercept_,
        }.items():
            arrays[f"{key}__{name}"] = np.asarray(value, dtype=np.float64).copy()
        settings["heads"][key] = {
            "site": site,
            "C": C,
            "target": target,
            "array_prefix": key,
            "training_partition": partition,
            "training_ids": ids[partition + "_ids"],
            "scaler_n_samples_seen": len(ids[partition + "_ids"]),
            "dev_group_log_loss": score,
            "n_iter": model.n_iter_.tolist(),
        }

    def persist():
        atomic_json(output / "candidates.json", history)

    try:
        with threadpool_limits(limits=THREAD_LIMIT):
            settings["numerical_execution"] = {
                "thread_limit": THREAD_LIMIT,
                "threadpools": threadpool_info(),
            }
            require(
                all(p["num_threads"] <= THREAD_LIMIT for p in threadpool_info()),
                "Numerical thread limit was not applied",
            )
            sf, sd, cf, cd = (_indices(rows, p) for p in FIT_PARTITIONS)
            labels = np.asarray([r["gold"] for r in rows], dtype=np.int64)
            source_groups = np.asarray([rows[i]["group_id"] for i in sd])
            selected_models = {}
            for site_index, site in enumerate(protocol["sites"]):
                successful = []
                try:
                    scaler = fit_transform(features[sf, site_index])
                    x, dx = (
                        transform(features[sf, site_index], scaler),
                        transform(features[sd, site_index], scaler),
                    )
                    transform_error = None
                except (ValueError, FloatingPointError) as error:
                    transform_error = error
                for C in DIRECT_GRID:
                    record = {
                        "site": site,
                        "C": C,
                        "training_partition": "source_fit",
                        "development_partition": "source_dev",
                    }
                    try:
                        if transform_error is not None:
                            raise CandidateFailure(str(transform_error))
                        model, score = _candidate(x, labels[sf], dx, labels[sd], source_groups, C)
                        record.update(
                            status="ok", dev_group_log_loss=score, n_iter=model.n_iter_.tolist()
                        )
                        successful.append((record, model))
                    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                        record.update(
                            status="failed",
                            failure={"type": type(error).__name__, "message": str(error)},
                        )
                    history["source_candidates"].append(record)
                    persist()
                if successful:
                    record, model = min(
                        successful, key=lambda item: (item[0]["dev_group_log_loss"], item[0]["C"])
                    )
                    key = f"source_{site_index:02d}"
                    retain(
                        key,
                        model,
                        scaler,
                        site,
                        record["C"],
                        "Y",
                        "source_fit",
                        record["dev_group_log_loss"],
                    )
                    panel = {**record, "head_key": key}
                    settings["source_panel"].append(panel)
                    selected_models[site] = panel
            require(len(selected_models) == 12, "All source candidates failed at a required site")
            chosen = min(
                settings["source_panel"],
                key=lambda r: site_selection_key(r, protocol["original_site"]),
            )
            settings["pipelines"].update(
                S0={"head_key": selected_models[protocol["original_site"]]["head_key"]},
                S1={"head_key": chosen["head_key"]},
            )
            site, index = chosen["site"], protocol["sites"].index(chosen["site"])
            scaler = fit_transform(features[cf, index])
            x, dx = transform(features[cf, index], scaler), transform(features[cd, index], scaler)
            groups = np.asarray([rows[i]["group_id"] for i in cd])
            successful = []
            for C in DIRECT_GRID:
                record = {
                    "site": site,
                    "C": C,
                    "training_partition": "correction_fit",
                    "development_partition": "correction_dev",
                }
                try:
                    model, score = _candidate(x, labels[cf], dx, labels[cd], groups, C)
                    record.update(
                        status="ok", dev_group_log_loss=score, n_iter=model.n_iter_.tolist()
                    )
                    successful.append((record, model))
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                    record.update(
                        status="failed",
                        failure={"type": type(error).__name__, "message": str(error)},
                    )
                history["C1_candidates"].append(record)
                persist()
            require(successful, "All C1 candidates failed")
            record, model = min(
                successful, key=lambda item: (item[0]["dev_group_log_loss"], item[0]["C"])
            )
            retain(
                "C1",
                model,
                scaler,
                site,
                record["C"],
                "Y",
                "correction_fit",
                record["dev_group_log_loss"],
            )
            settings["pipelines"]["C1"] = {"head_key": "C1"}
            factor_labels = np.asarray(
                [[rows[i]["state"][f] for f in FACTOR_FIELDS] for i in np.r_[cf, cd]],
                dtype=np.int64,
            )
            require(np.isin(factor_labels, (0, 1)).all(), "Invalid correction factor labels")
            fy, dy = factor_labels[: len(cf)], factor_labels[len(cf) :]
            successful = []
            for C in FACTOR_GRID:
                record = {
                    "site": site,
                    "C": C,
                    "training_partition": "correction_fit",
                    "development_partition": "correction_dev",
                    "factors": {},
                }
                models, losses = {}, []
                for j, factor in enumerate(FACTORS):
                    try:
                        model, score = _candidate(x, fy[:, j], dx, dy[:, j], groups, C)
                        models[factor] = model
                        losses.append(score)
                        record["factors"][factor] = {
                            "status": "ok",
                            "dev_group_log_loss": score,
                            "n_iter": model.n_iter_.tolist(),
                        }
                    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                        record["factors"][factor] = {
                            "status": "failed",
                            "failure": {"type": type(error).__name__, "message": str(error)},
                        }
                if len(models) == 4:
                    record.update(status="ok", dev_group_log_loss=float(np.mean(losses)))
                    successful.append((record, models))
                else:
                    record["status"] = "failed"
                history["C2_candidates"].append(record)
                persist()
            require(successful, "All C2 candidates failed")
            record, models = min(
                successful, key=lambda item: (item[0]["dev_group_log_loss"], item[0]["C"])
            )
            for factor in FACTORS:
                retain(
                    f"C2_{factor}",
                    models[factor],
                    scaler,
                    site,
                    record["C"],
                    factor,
                    "correction_fit",
                    record["factors"][factor]["dev_group_log_loss"],
                )
            settings["pipelines"]["C2"] = {
                "factor_heads": {f: f"C2_{f}" for f in FACTORS},
                "shared_C": record["C"],
                "privileged_supervision": True,
                "external_fold": fit_policy()["C2_fold"],
            }
        _check_hashes(data["input_artifact_sha256"])
        _check_hashes(protocol["source_sha256"])
        _check_hashes(settings["fitting_source_sha256"])
        atomic_npz(output / "heads.npz", **arrays)
        settings.update(
            status="frozen",
            frozen_at=utc_now(),
            source_candidates=history["source_candidates"],
            C1_candidates=history["C1_candidates"],
            C2_candidates=history["C2_candidates"],
            output_sha256={n: file_sha256(output / n) for n in ("heads.npz", "candidates.json")},
        )
        atomic_json(output / "settings.json", settings)
        manifest.update(
            status="frozen",
            finished_at=utc_now(),
            output_sha256={
                n: file_sha256(output / n)
                for n in ("heads.npz", "settings.json", "candidates.json")
            },
        )
        atomic_json(output / "manifest.json", manifest)
        validate_fit_bundle(output, allow_test_fixture=allow_test_fixture)
        return settings
    except BaseException as error:
        persist()
        manifest.update(
            status="failed_preserved",
            finished_at=utc_now(),
            error_type=type(error).__name__,
            error=str(error),
        )
        atomic_json(output / "manifest.json", manifest)
        raise


def validate_fit_bundle(path, *, allow_test_fixture=False):
    """Reject incomplete, stale, unsafe or structurally fabricated fit bundles."""
    from .fresh_readout_capture import panel_sites

    path = Path(path)
    require(path.is_dir(), "Fit bundle must be a directory")
    manifest = read_json(path / "manifest.json")
    require(
        manifest.get("schema") == "fresh-readout-fit-manifest-v2"
        and manifest.get("status") == "frozen",
        "Fit manifest is not frozen",
    )
    require(
        set(manifest.get("output_sha256", {})) == {"settings.json", "heads.npz", "candidates.json"},
        "Wrong fit output inventory",
    )
    for name, digest in manifest["output_sha256"].items():
        require(file_sha256(path / name) == digest, "Fit artifact changed: " + name)
    settings = read_json(path / "settings.json")
    require(
        settings.get("schema") == "fresh-readout-fit-v2" and settings.get("status") == "frozen",
        "Settings are not a genuine frozen fresh fit",
    )
    fixture = settings.get("test_fixture")
    require(
        type(fixture) is bool
        and (not fixture or allow_test_fixture)
        and manifest.get("test_fixture") is fixture,
        "Fixture bundle cannot authorize a native run",
    )
    require(
        settings.get("research_status") == "agent_audited_exploratory"
        and settings.get("human_review_complete") is False
        and settings.get("confirmatory") is False,
        "Incorrect research status",
    )
    require(
        settings.get("policy") == fit_policy()
        and settings.get("policy_sha256") == canonical_hash(fit_policy()),
        "Fitting policy differs",
    )
    sources = _fitting_sources()
    require(settings.get("fitting_source_sha256") == sources, "Fitting code changed")
    _check_hashes(sources)
    for name, digest in settings["output_sha256"].items():
        require(file_sha256(path / name) == digest, "Fit array/history hash differs")
    context = settings["fit_context"]
    require(
        context.get("scope") == "training_development_only"
        and context.get("test_rows_loaded") == 0
        and context.get("development_absorbed") is False
        and (fixture or context.get("row_journal_verified") is True),
        "Invalid fitting context",
    )
    _check_hashes(settings["input_artifact_sha256"])
    protocol = read_json(context["protocol_path"])
    provenance = _protocol_metadata(
        protocol,
        context["protocol_path"],
        context["bank_path"],
        allow_test_fixture=allow_test_fixture,
    )
    require(settings["capture_provenance"] == provenance, "Fit provenance differs from protocol")
    require(
        settings["hidden_size"] == protocol["hidden_size"],
        "Fit hidden size differs from bound protocol",
    )
    all_ids = []
    for partition in FIT_PARTITIONS:
        ids = settings.get(partition + "_ids")
        expected = [_identity(r) for r in protocol["planned_rows"] if r["partition"] == partition]
        require(
            isinstance(ids, list)
            and ids
            and ids == expected
            and len(set(ids)) == len(ids)
            and (fixture or len(ids) == 512),
            "Wrong fitting IDs",
        )
        all_ids.extend(ids)
    require(len(set(all_ids)) == len(all_ids), "Overlapping fitting partitions")
    sites = panel_sites(provenance["model_key"])
    panel, heads, pipelines = settings["source_panel"], settings["heads"], settings["pipelines"]
    require(
        [p["site"] for p in panel] == sites and len(heads) == 17,
        "Incomplete twelve-source plus C1/four-factor head inventory",
    )
    require(set(pipelines) == {"S0", "S1", "C1", "C2"}, "Missing named pipeline")
    original = settings["original_site"]
    require(original == protocol["original_site"], "Wrong original site")
    history = read_json(path / "candidates.json")
    for key, grid, count in (
        ("source_candidates", DIRECT_GRID, 60),
        ("C1_candidates", DIRECT_GRID, 5),
        ("C2_candidates", FACTOR_GRID, 5),
    ):
        candidates = settings[key]
        require(
            candidates == history[key]
            and len(candidates) == count
            and all(c["status"] in ("ok", "failed") and c["C"] in grid for c in candidates),
            "Incomplete candidate/failure history",
        )
        for candidate in candidates:
            if candidate["status"] == "ok":
                require(np.isfinite(candidate["dev_group_log_loss"]), "Nonfinite candidate score")
            else:
                require(
                    "failure" in candidate
                    or any(f["status"] == "failed" for f in candidate.get("factors", {}).values()),
                    "Missing candidate failure details",
                )
    for p in panel:
        options = [c for c in history["source_candidates"] if c["site"] == p["site"]]
        require([c["C"] for c in options] == list(DIRECT_GRID), "Wrong source candidate grid")
        selected = min(
            (c for c in options if c["status"] == "ok"),
            key=lambda c: (c["dev_group_log_loss"], c["C"]),
        )
        require(
            all(p.get(k) == v for k, v in selected.items()) and p["head_key"] in heads,
            "Per-site development selection differs",
        )
    selected = min(panel, key=lambda p: site_selection_key(p, original))
    require(
        pipelines["S0"]["head_key"] == next(p["head_key"] for p in panel if p["site"] == original)
        and pipelines["S1"]["head_key"] == selected["head_key"],
        "S0/S1 selection differs",
    )
    factor_keys = pipelines["C2"].get("factor_heads")
    require(
        isinstance(factor_keys, dict)
        and set(factor_keys) == set(FACTORS)
        and pipelines["C2"].get("privileged_supervision") is True
        and pipelines["C2"].get("external_fold") == fit_policy()["C2_fold"],
        "Invalid four-factor pipeline",
    )
    required = (
        {p["head_key"] for p in panel} | {pipelines["C1"]["head_key"]} | set(factor_keys.values())
    )
    require(
        required == set(heads) and len(required) == 17, "Head references are incomplete/overlapping"
    )
    expected_arrays = set()
    with np.load(path / "heads.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    hidden_size = settings["hidden_size"]
    for key, head in heads.items():
        prefix = head["array_prefix"]
        require(prefix == key and head["site"] in sites, "Invalid array prefix/site")
        source = key in {p["head_key"] for p in panel}
        partition = "source_fit" if source else "correction_fit"
        require(
            head["training_partition"] == partition
            and head["training_ids"] == settings[partition + "_ids"]
            and head["scaler_n_samples_seen"] == len(settings[partition + "_ids"]),
            "Scaler/head training support differs",
        )
        require(source or head["site"] == selected["site"], "Correction head moved from S1 site")
        for field in ("mean", "scale", "var", "coef", "intercept"):
            name = prefix + "__" + field
            expected_arrays.add(name)
            value = arrays[name]
            require(
                value.dtype == np.float64
                and value.shape == ((1,) if field == "intercept" else (hidden_size,))
                and np.isfinite(value).all(),
                "Invalid safe head/scaler array",
            )
        require(
            (arrays[prefix + "__scale"] > 0).all() and (arrays[prefix + "__var"] >= 0).all(),
            "Invalid scaler scale/variance",
        )
    require(set(arrays) == expected_arrays, "Unexpected head arrays")
    for p in panel:
        h = heads[p["head_key"]]
        require(
            h["C"] == p["C"]
            and h["site"] == p["site"]
            and h["target"] == "Y"
            and h["dev_group_log_loss"] == p["dev_group_log_loss"],
            "Panel head differs",
        )
    for name in ("C1", "C2"):
        candidates = history[name + "_candidates"]
        grid = DIRECT_GRID if name == "C1" else FACTOR_GRID
        require(
            [c["C"] for c in candidates] == list(grid)
            and all(c["site"] == selected["site"] for c in candidates),
            "Wrong correction candidates",
        )
        winner = min(
            (c for c in candidates if c["status"] == "ok"),
            key=lambda c: (c["dev_group_log_loss"], c["C"]),
        )
        keys = [pipelines[name]["head_key"]] if name == "C1" else list(factor_keys.values())
        require(all(heads[k]["C"] == winner["C"] for k in keys), "Correction C selection differs")
        if name == "C1":
            head = heads[keys[0]]
            require(
                head["target"] == "Y"
                and head["dev_group_log_loss"] == winner["dev_group_log_loss"],
                "C1 target/development score differs",
            )
        else:
            require(pipelines["C2"].get("shared_C") == winner["C"], "C2 shared C differs")
            for candidate in candidates:
                require(
                    set(candidate.get("factors", {})) == set(FACTORS),
                    "Incomplete C2 factor candidates",
                )
                factors = candidate["factors"]
                if candidate["status"] == "ok":
                    require(
                        all(factors[f]["status"] == "ok" for f in FACTORS)
                        and candidate["dev_group_log_loss"]
                        == float(np.mean([factors[f]["dev_group_log_loss"] for f in FACTORS])),
                        "C2 factor development objective differs",
                    )
            require(
                all(
                    heads[factor_keys[f]]["dev_group_log_loss"]
                    == winner["factors"][f]["dev_group_log_loss"]
                    for f in FACTORS
                ),
                "C2 selected factor score differs",
            )
    for factor, key in factor_keys.items():
        require(heads[key]["target"] == factor, "Wrong factor target")
        for field in ("mean", "scale", "var"):
            require(
                np.array_equal(
                    arrays[key + "__" + field], arrays[pipelines["C1"]["head_key"] + "__" + field]
                ),
                "Correction controls must share their training-only transform",
            )
    execution = settings["numerical_execution"]
    require(
        execution["thread_limit"] == THREAD_LIMIT
        and all(p["num_threads"] <= THREAD_LIMIT for p in execution["threadpools"]),
        "Wrong numerical thread limit",
    )
    return settings, arrays


def score_bundle(settings, arrays, features):
    """Return strict-threshold verdicts and margins for every frozen pipeline."""
    features = np.asarray(features)
    sites = settings["capture_provenance"]["sites"]
    require(
        features.ndim == 3
        and features.shape[1:] == (12, settings["hidden_size"])
        and np.isfinite(features).all(),
        "Invalid inference feature matrix",
    )
    margins = {}
    for key, head in settings["heads"].items():
        prefix = head["array_prefix"]
        parameters = {f: arrays[prefix + "__" + f] for f in ("mean", "scale")}
        values = transform(features[:, sites.index(head["site"])], parameters)
        margins[key] = values @ arrays[prefix + "__coef"] + arrays[prefix + "__intercept"][0]
        require(np.isfinite(margins[key]).all(), "Nonfinite inference margin")
    result = {}
    source_margins = np.column_stack([margins[p["head_key"]] for p in settings["source_panel"]])
    result.update(
        source_all12=(source_margins > 0).astype(np.int64), source_all12_margins=source_margins
    )
    for name in ("S0", "S1", "C1"):
        margin = margins[settings["pipelines"][name]["head_key"]]
        result.update(
            {
                name: (margin > 0).astype(np.int64),
                name + "_margin": margin,
                name + "_probability": expit(margin),
            }
        )
    factors = np.column_stack(
        [margins[settings["pipelines"]["C2"]["factor_heads"][f]] for f in FACTORS]
    )
    result.update(
        factor_margins=factors,
        factor_predictions=(factors > 0).astype(np.int64),
        C2=fold_factors((factors > 0).astype(np.int64)),
    )
    return result


def freeze_fits(llama_fit, qwen_fit, protocols, output, *, allow_test_fixture=False):
    """Seal both model bundles before either model is permitted a test forward."""
    require(len(protocols) == 2, "Need both prepared protocol files")
    protocol_by_model = {read_json(p)["model_key"]: Path(p).resolve() for p in protocols}
    require(set(protocol_by_model) == {"llama", "qwen"}, "Need Llama and Qwen protocols")
    fitted, settings_by_model, sources = {}, {}, {}
    for key, path in (("llama", llama_fit), ("qwen", qwen_fit)):
        path = Path(path).resolve()
        settings, _ = validate_fit_bundle(path, allow_test_fixture=allow_test_fixture)
        require(
            settings["capture_provenance"]["model_key"] == key
            and settings["capture_provenance"]["protocol_sha256"]
            == file_sha256(protocol_by_model[key]),
            "Bundle bound to wrong model/protocol",
        )
        settings_by_model[key] = settings
        fitted.update(
            {
                str(path / n): file_sha256(path / n)
                for n in ("settings.json", "heads.npz", "candidates.json", "manifest.json")
            }
        )
        sources.update(read_json(protocol_by_model[key])["source_sha256"])
    llama, qwen = settings_by_model.values()
    require(
        llama["test_fixture"] is qwen["test_fixture"]
        and llama["capture_provenance"]["bank_file_sha256"]
        == qwen["capture_provenance"]["bank_file_sha256"]
        and all(llama[p + "_ids"] == qwen[p + "_ids"] for p in FIT_PARTITIONS),
        "Both model fits must share bank, fixture status and exact fitting IDs",
    )
    require(
        [s.split("_", 2)[2] for s in llama["capture_provenance"]["sites"]]
        == [s.split("_", 2)[2] for s in qwen["capture_provenance"]["sites"]],
        "Site inventories differ",
    )
    seal = {
        "schema": "fresh-readout-fit-seal-v2",
        "status": "frozen",
        "created_at": utc_now(),
        "test_fixture": llama["test_fixture"],
        "research_status": "agent_audited_exploratory",
        "human_review_complete": False,
        "confirmatory": False,
        "fitted_artifact_sha256": fitted,
        "source_sha256": sources,
        "model_protocol_sha256": {k: file_sha256(p) for k, p in protocol_by_model.items()},
        "bank_file_sha256": llama["capture_provenance"]["bank_file_sha256"],
    }
    _check_hashes(fitted)
    _check_hashes(sources)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(seal, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return seal
