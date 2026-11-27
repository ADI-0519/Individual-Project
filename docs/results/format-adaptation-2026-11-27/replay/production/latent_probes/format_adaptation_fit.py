"""Fixed-budget, frozen-map four-factor adaptation; no development selection.

Only fresh fitting rows are accepted. The old C2 map and C are copied from a
hash-verified old bundle. No scaler, site, penalty or threshold is estimated
from the new test bank. Failures preserve their attempt directory and stop.
"""

from __future__ import annotations

import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from .storage import (
    atomic_json,
    atomic_npz,
    canonical_hash,
    environment_info,
    file_sha256,
    read_json,
    utc_now,
)

FACTORS = ("I", "M1", "B1", "B2")
FACTOR_FIELDS = ("initial", "update", "first_receipt", "receipt")
ARMS = ("A512", "Mixed512", "Ahalf256", "Offset")
MODEL_SITES = {"llama": "layer_16_query_location_last", "qwen": "layer_21_query_location_last"}
ROOT = Path(__file__).resolve().parents[2]
POLICY_FILES = (
    "src/latent_probes/format_adaptation_fit.py",
    "scripts/fit_format_adaptation.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fit_policy():
    return {
        "arms": list(ARMS),
        "factor_order": list(FACTORS),
        "factor_fields": list(FACTOR_FIELDS),
        "training_inventory": {"contexts": 16, "states_per_context": 32, "A": 512, "B": 256},
        "feature_map": ["float64", "old_C2_mean_scale", "row_L2"],
        "site": "old_frozen_S1_per_model",
        "C": "old_C2_shared_C; C_n = 512*C_old/n",
        "fresh_development_or_selection": False,
        "logistic_regression": {
            "solver": "lbfgs",
            "penalty": "l2",
            "fit_intercept": True,
            "tol": 1e-4,
            "max_iter": 2000,
            "random_state": 11,
            "class_weight": None,
            "sample_weight": None,
        },
        "offset": {
            "starting_arm": "Ahalf256",
            "fixed_coefficients": True,
            "fit_rows": "same_256_B_replacements_used_by_Mixed512",
            "objective": "unpenalized_mean_binary_logistic_loss",
            "solver": "monotone_derivative_bisection",
            "A": "unchanged_Ahalf256",
            "B": "Ahalf256_intercept_plus_fitted_offset",
        },
        "fold": "1-M1 if B2 else M1 if B1 else I",
        "threshold": "margin > 0; exact zero predicts 0",
        "thread_limit": 4,
        "failure": "stop_and_preserve_attempt; no_fallback_or_rescue",
    }


def source_bindings():
    return {str((ROOT / p).resolve()): file_sha256(ROOT / p) for p in POLICY_FILES}


def check_bindings(bindings):
    require(isinstance(bindings, dict) and bool(bindings), "Missing artifact bindings")
    for name, digest in bindings.items():
        require(Path(name).is_file() and file_sha256(name) == digest, f"Changed input: {name}")


def state_key(row):
    """Full formal-state identity, excluding only format and record identity."""
    return (row["group_id"], row["condition"], canonical_hash(row["state"]))


def validate_state(row):
    state = row.get("state")
    require(isinstance(state, dict), "Missing formal state")
    require(state.get("group_id", row["group_id"]) == row["group_id"], "State group differs")
    for field in ("world", "name_map"):
        require(type(state.get(field)) is int and state[field] in (0, 1), "Invalid nuisance bit")
    condition = row.get("condition")
    if condition == "asserted":
        require(state.get("mechanism") == "asserted", "Wrong asserted mechanism")
        require(
            set(state) == {"group_id", "mechanism", "asserted", "world", "name_map"},
            "Unknown/missing asserted formal-state fields",
        )
        target = state.get("asserted")
        require(type(target) is int and target in (0, 1), "Invalid asserted bit")
    else:
        require(
            condition in ("one_step", "correction_easy", "correction_hard"), "Unknown condition"
        )
        require(
            set(state)
            == {
                "group_id",
                "mechanism",
                "initial",
                "update",
                "first_receipt",
                "receipt",
                "world",
                "name_map",
            },
            "Unknown/missing formal-state fields",
        )
        for field in ("initial", "update", "receipt"):
            require(
                type(state.get(field)) is int and state[field] in (0, 1), "Invalid factor truth"
            )
        if condition.startswith("correction_"):
            require(state.get("mechanism") == "correction", "Wrong correction mechanism")
            require(
                type(state.get("first_receipt")) is int and state["first_receipt"] in (0, 1),
                "Invalid first receipt",
            )
            expected = (
                "correction_hard" if state["initial"] == 1 - state["update"] else "correction_easy"
            )
            require(condition == expected, "Wrong correction stratum")
            target = (
                1 - state["update"]
                if state["receipt"]
                else state["update"]
                if state["first_receipt"]
                else state["initial"]
            )
        else:
            require(
                state.get("mechanism") == "message" and state.get("first_receipt") is None,
                "Wrong one-step mechanism/factors",
            )
            target = state["update"] if state["receipt"] else state["initial"]
    require(type(row.get("gold")) is int and row["gold"] == target, "Gold differs from formal rule")


def training_indices(rows):
    """Validate all membership before any features or model labels are opened."""
    require(isinstance(rows, list) and len(rows) == 768, "Need exact 768 fitting captures")
    require(
        all(r.get("partition") == "fit" and r.get("split", "fit") == "fit" for r in rows),
        "Test or unknown partition is forbidden in fitting input",
    )
    ids = [r.get("row_id") for r in rows]
    require(
        all(isinstance(x, str) and x for x in ids) and len(set(ids)) == len(ids),
        "Duplicate/missing fitting row IDs",
    )
    by_state = defaultdict(dict)
    for index, row in enumerate(rows):
        validate_state(row)
        require(
            row["condition"].startswith("correction_"), "Fitting accepts correction states only"
        )
        renderer, assignment = row.get("renderer"), row.get("mixed_format")
        require(renderer in ("A", "B") and assignment in ("A", "B"), "Invalid format assignment")
        key = state_key(row)
        require(renderer not in by_state[key], "Duplicate full-state/renderer fitting key")
        by_state[key][renderer] = index
        try:
            group_index = int(row["group_id"].rsplit("_", 1)[1])
        except (ValueError, IndexError) as error:
            raise ValueError("Fitting context lacks fixed allocation index") from error
        state = row["state"]
        require(
            0 <= group_index < 16
            and state["initial"] ^ state["update"] ^ state["world"] ^ state["name_map"]
            == group_index % 2,
            "Fitting state is outside the fixed XOR roster",
        )
        expected = "A" if state["world"] == (group_index // 2) % 2 else "B"
        require(assignment == expected, "Format assignment differs from fixed allocation")
    require(len(by_state) == 512, "Need same 512 unique underlying fitting states")
    groups = defaultdict(list)
    a, mixed, ahalf, b = [], [], [], []
    for key, formats in by_state.items():
        require("A" in formats, "Every fitting state needs its A capture")
        ai = formats["A"]
        ar = rows[ai]
        assignment = ar["mixed_format"]
        require(
            set(formats) == ({"A"} if assignment == "A" else {"A", "B"}),
            "B capture must be exactly a complementary mixed replacement",
        )
        a.append(ai)
        groups[key[0]].append(ai)
        if assignment == "A":
            ahalf.append(ai)
            mixed.append(ai)
        else:
            bi = formats["B"]
            require(
                rows[bi]["mixed_format"] == assignment and rows[bi]["gold"] == ar["gold"],
                "Paired fitting assignment/truth differs",
            )
            b.append(bi)
            mixed.append(bi)
    require(
        len(groups) == 16 and all(len(v) == 32 for v in groups.values()),
        "Need 16 fitting contexts with 32 underlying states each",
    )
    require(
        len(a) == len(mixed) == 512 and len(ahalf) == len(b) == 256,
        "Wrong matched observation budgets",
    )
    for indices in groups.values():
        for format_name in ("A", "B"):
            selected = [rows[i] for i in indices if rows[i]["mixed_format"] == format_name]
            paths = Counter(tuple(r["state"][f] for f in FACTOR_FIELDS) for r in selected)
            require(
                len(selected) == 16 and set(paths.values()) == {1} and len(paths) == 16,
                "Each format must contain all 16 factor paths once per context",
            )
    cells = Counter(
        (
            rows[i]["renderer"],
            *(rows[i]["state"][f] for f in FACTOR_FIELDS),
            rows[i]["state"]["world"],
            rows[i]["state"]["name_map"],
        )
        for i in mixed
    )
    require(
        len(cells) == 128 and set(cells.values()) == {4},
        "Format/factor/nuisance support differs from fixed cohort balance",
    )
    return {
        "A512": np.array(sorted(a)),
        "Mixed512": np.array(sorted(mixed)),
        "Ahalf256": np.array(sorted(ahalf)),
        "B_offset": np.array(sorted(b)),
    }


def _capture_metadata(capture_dir, partition, model_key, *, allow_test_fixture=False):
    directory = Path(capture_dir).resolve()
    payload, manifest = (read_json(directory / n) for n in ("rows.json", "manifest.json"))
    require(payload.get("schema") == "format-adaptation-capture-rows-v1", "Wrong capture schema")
    fixture = payload.get("test_fixture")
    require(
        type(fixture) is bool and (not fixture or allow_test_fixture),
        "Fixture capture requires explicit test authorization",
    )
    require(
        payload.get("model_key") == model_key and payload.get("site") == MODEL_SITES[model_key],
        "Capture model/site differs from frozen old S1",
    )
    rows = payload.get("rows")
    # Partition exclusion is deliberately before np.load and feature checksums.
    require(
        isinstance(rows, list) and rows and all(r.get("partition") == partition for r in rows),
        "Test or unknown partition is forbidden in fitting input"
        if partition == "fit"
        else "Non-test row in test capture",
    )
    require(
        manifest.get("schema") == "format-adaptation-capture-manifest-v1"
        and manifest.get("status") == "complete"
        and manifest.get("partition") == partition
        and manifest.get("completed_rows") == len(rows),
        "Incomplete/wrong capture manifest",
    )
    upstream = dict(manifest.get("input_artifact_sha256", {}))
    fit_plan = None
    if partition == "fit":
        fit_plan, prepared_sources = verify_fit_plan(payload, manifest, model_key)
        upstream.update(prepared_sources)
    for filename, field in (("rows.json", "rows_sha256"), ("features.npz", "features_sha256")):
        require(
            file_sha256(directory / filename) == manifest.get(field),
            f"Capture artifact changed: {filename}",
        )
    if manifest.get("input_artifact_sha256"):
        check_bindings(manifest["input_artifact_sha256"])
    return directory, payload, manifest, upstream, fit_plan


def verify_fit_plan(payload, manifest, model_key):
    """Bind original fitting metadata without opening a protocol's test inventory."""
    require(
        isinstance(manifest.get("fit_plan_path"), str)
        and isinstance(manifest.get("preparation_path"), str),
        "Fitting capture lacks its predeclared fit-only plan",
    )
    plan_path = Path(manifest["fit_plan_path"]).resolve()
    preparation_path = Path(manifest["preparation_path"]).resolve()
    inputs = manifest.get("input_artifact_sha256", {})
    for path in (plan_path, preparation_path):
        require(
            inputs.get(str(path)) == file_sha256(path),
            "Fitting capture does not bind its fit-only plan/preparation",
        )
    preparation, plan = read_json(preparation_path), read_json(plan_path)
    require(
        preparation.get("schema") == "format-adaptation-preparation-v1",
        "Wrong pre-inference preparation schema",
    )
    check_bindings(preparation.get("source_sha256"))
    require(
        all(
            preparation["source_sha256"].get(path) == digest
            for path, digest in source_bindings().items()
        ),
        "Fitting source was not bound in pre-inference preparation",
    )
    require(
        preparation.get("fit_plan_sha256", {}).get(model_key) == file_sha256(plan_path),
        "Fit-only plan differs from pre-inference preparation",
    )
    require(
        plan.get("schema") == "format-adaptation-fit-plan-v1"
        and plan.get("model_key") == model_key
        and plan.get("site") == payload["site"]
        and plan.get("bank_sha256") == manifest.get("bank_sha256")
        and bool(plan.get("bank_sha256")),
        "Fit-only plan model/site/bank differs",
    )
    planned = plan.get("rows")
    require(
        isinstance(planned, list)
        and len(planned) == 768
        and all(r.get("partition") == "fit" for r in planned),
        "Fit-only plan must contain exactly 768 fitting rows",
    )
    require(
        [r.get("row_id") for r in planned] == [r.get("row_id") for r in payload["rows"]],
        "Fitting capture rows differ from original planned order",
    )
    for captured, original in zip(payload["rows"], planned, strict=True):
        require(
            all(captured.get(name) == value for name, value in original.items()),
            "Fitting row differs from predeclared fit-only plan",
        )
    return plan, preparation["source_sha256"]


def load_capture(capture_dir, partition, model_key, *, allow_test_fixture=False):
    require(model_key in MODEL_SITES, "Unknown model")
    directory, payload, manifest, upstream, fit_plan = _capture_metadata(
        capture_dir, partition, model_key, allow_test_fixture=allow_test_fixture
    )
    rows = payload["rows"]
    indices = training_indices(rows) if partition == "fit" else None
    with np.load(directory / "features.npz", allow_pickle=False) as archive:
        require(
            set(archive.files)
            == {"features", "row_ids", "output_margins", "strict_first_token_correct"},
            "Wrong capture array inventory",
        )
        require(
            archive["row_ids"].dtype.kind in "US"
            and archive["row_ids"].tolist() == [r["row_id"] for r in rows],
            "Capture row order differs from safe arrays",
        )
        features = archive["features"].copy()
        margins = archive["output_margins"].copy()
        strict = archive["strict_first_token_correct"].copy()
    require(
        features.dtype == np.float32
        and features.ndim == 2
        and features.shape[0] == len(rows)
        and features.shape[1] > 0
        and np.isfinite(features).all(),
        "Invalid feature matrix",
    )
    require(
        margins.dtype == np.float64
        and margins.shape == (len(rows),)
        and np.isfinite(margins).all()
        and strict.dtype == bool
        and strict.shape == (len(rows),),
        "Invalid output comparator arrays",
    )
    bindings = {
        **upstream,
        **{
            str(directory / name): file_sha256(directory / name)
            for name in ("rows.json", "features.npz", "manifest.json")
        },
    }
    return {
        "rows": rows,
        "features": features,
        "output_margins": margins,
        "strict_first_token_correct": strict,
        "payload": payload,
        "manifest": manifest,
        "fit_plan": fit_plan,
        "indices": indices,
        "input_artifact_sha256": bindings,
    }


def load_old_instrument(old_bundle, model_key, *, allow_test_fixture=False):
    """Read only frozen settings/arrays; never reopen old captured test rows."""
    directory = Path(old_bundle).resolve()
    settings, manifest = (read_json(directory / n) for n in ("settings.json", "manifest.json"))
    require(
        settings.get("schema") == "fresh-readout-fit-v2"
        and settings.get("status") == "frozen"
        and manifest.get("schema") == "fresh-readout-fit-manifest-v2"
        and manifest.get("status") == "frozen",
        "Old instrument is not frozen",
    )
    require(
        not settings.get("test_fixture") or allow_test_fixture,
        "Old fixture instrument cannot authorize native fitting",
    )
    for name, digest in manifest["output_sha256"].items():
        require(file_sha256(directory / name) == digest, f"Old fit changed: {name}")
    require(settings["capture_provenance"]["model_key"] == model_key, "Old model differs")
    selected = settings["heads"][settings["pipelines"]["S1"]["head_key"]]["site"]
    require(selected == MODEL_SITES[model_key], "Old selected site differs from fixed S1")
    c2 = settings["pipelines"]["C2"]
    C = float(c2["shared_C"])
    require(np.isfinite(C) and C > 0, "Invalid old C2 C")
    require(set(c2["factor_heads"]) == set(FACTORS), "Incomplete old four-factor instrument")
    parameters = []
    with np.load(directory / "heads.npz", allow_pickle=False) as archive:
        for factor in FACTORS:
            head = settings["heads"][c2["factor_heads"][factor]]
            require(
                head["site"] == selected and head["C"] == C and head["target"] == factor,
                "Old C2 site/C/target differs",
            )
            prefix = head["array_prefix"]
            parameters.append(
                {name: archive[f"{prefix}__{name}"].copy() for name in ("mean", "scale")}
            )
    mean, scale = (parameters[0][name] for name in ("mean", "scale"))
    require(
        mean.dtype == scale.dtype == np.float64
        and mean.ndim == scale.ndim == 1
        and mean.shape == scale.shape
        and mean.size == settings["hidden_size"]
        and np.isfinite(mean).all()
        and np.isfinite(scale).all()
        and (scale > 0).all(),
        "Invalid old frozen feature map",
    )
    require(
        all(
            np.array_equal(p["mean"], mean) and np.array_equal(p["scale"], scale)
            for p in parameters
        ),
        "Old C2 heads do not share the frozen feature map",
    )
    bindings = {
        str(directory / n): file_sha256(directory / n)
        for n in ("settings.json", "heads.npz", "manifest.json", "candidates.json")
    }
    return {
        "mean": mean,
        "scale": scale,
        "C": C,
        "site": selected,
        "hidden_size": len(mean),
        "input_artifact_sha256": bindings,
        "old_settings_sha256": bindings[str(directory / "settings.json")],
    }


def transform(features, mean, scale):
    values = np.asarray(features, dtype=np.float64)
    require(
        values.ndim == 2
        and values.shape[1:] == np.shape(mean)
        and np.shape(mean) == np.shape(scale)
        and np.isfinite(values).all()
        and np.isfinite(mean).all()
        and np.isfinite(scale).all()
        and np.all(scale > 0),
        "Invalid frozen feature-map input",
    )
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        values = (values - mean) / scale
        values = normalize(values, norm="l2")
    require(np.isfinite(values).all(), "Nonfinite frozen-map output")
    return values


def fit_lr(features, labels, C):
    model = LogisticRegression(
        C=C,
        solver="lbfgs",
        penalty="l2",
        fit_intercept=True,
        tol=1e-4,
        max_iter=2000,
        random_state=11,
        class_weight=None,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(features, labels)
    require(
        not any(issubclass(w.category, ConvergenceWarning) for w in caught),
        "Adaptation logistic fit did not converge",
    )
    require(
        np.array_equal(model.classes_, [0, 1])
        and np.isfinite(model.coef_).all()
        and np.isfinite(model.intercept_).all(),
        "Invalid fitted factor parameters",
    )
    return model


def fit_intercept_offset(base_margins, labels):
    """Unique convex intercept optimum; no threshold/sign/orientation selection."""
    margins, labels = np.asarray(base_margins, dtype=np.float64), np.asarray(labels)
    require(
        margins.ndim == 1
        and labels.shape == margins.shape
        and len(labels) > 0
        and np.isfinite(margins).all()
        and np.isin(labels, (0, 1)).all()
        and np.sum(labels) * 2 == len(labels),
        "Offset needs balanced binary fitting labels",
    )

    def derivative(offset):
        return float(np.mean(expit(margins + offset) - labels))

    # At these finite bounds every predicted probability is below/above 1/2.
    left, right = -float(np.max(margins)) - 40.0, -float(np.min(margins)) + 40.0
    require(
        np.isfinite(left) and np.isfinite(right) and derivative(left) < 0 < derivative(right),
        "Cannot bracket unique offset optimum",
    )
    iterations = 0
    for iterations in range(1, 257):
        middle = left + (right - left) / 2
        value = derivative(middle)
        if value > 0:
            right = middle
        else:
            left = middle
        if abs(value) <= 1e-13 or right - left <= 1e-12:
            break
    offset = middle
    loss = float(np.mean(np.logaddexp(0, margins + offset) - labels * (margins + offset)))
    require(
        np.isfinite(offset) and np.isfinite(loss) and abs(derivative(offset)) <= 1e-10,
        "Offset optimum failed convergence",
    )
    return offset, {
        "iterations": iterations,
        "mean_loss": loss,
        "mean_loss_derivative": derivative(offset),
        "positive_labels": int(labels.sum()),
        "negative_labels": int(len(labels) - labels.sum()),
    }


def fold_factors(bits):
    values = np.asarray(bits)
    require(
        values.ndim == 2 and values.shape[1] == 4 and np.isin(values, (0, 1)).all(),
        "Need four binary factors",
    )
    i, m1, b1, b2 = values.T
    return np.where(b2, 1 - m1, np.where(b1, m1, i)).astype(np.int64)


def fit_bundle(capture, old_bundle, model_key, output, *, allow_test_fixture=False):
    data = load_capture(capture, "fit", model_key, allow_test_fixture=allow_test_fixture)
    old = load_old_instrument(old_bundle, model_key, allow_test_fixture=allow_test_fixture)
    plan = data["fit_plan"]
    require(
        plan.get("old_bundle_path") == str(Path(old_bundle).resolve()),
        "Old bundle path differs from predeclared fit-only plan",
    )
    expected_old = {
        Path(path).name: digest for path, digest in old["input_artifact_sha256"].items()
    }
    require(
        plan.get("old_bundle_sha256") == expected_old,
        "Old bundle hashes differ from predeclared fit-only plan",
    )
    require(
        all(
            data["input_artifact_sha256"].get(path) == digest
            for path, digest in old["input_artifact_sha256"].items()
            if Path(path).name in ("settings.json", "heads.npz", "manifest.json")
        ),
        "Old instrument differs from predeclared preparation",
    )
    require(data["features"].shape[1] == old["hidden_size"], "New/old hidden sizes differ")
    target = Path(output).resolve()
    require(not target.exists(), "Fit output already exists; preserve prior attempt")
    target.mkdir(parents=True)
    bindings = {
        **data["input_artifact_sha256"],
        **old["input_artifact_sha256"],
        **source_bindings(),
    }
    rows, indices = data["rows"], data["indices"]
    settings = {
        "schema": "format-adaptation-fit-v1",
        "status": "fitting",
        "model_key": model_key,
        "site": old["site"],
        "hidden_size": old["hidden_size"],
        "created_at": utc_now(),
        "test_fixture": data["payload"]["test_fixture"],
        "research_status": "agent_audited_exploratory",
        "confirmatory": False,
        "policy": fit_policy(),
        "policy_sha256": canonical_hash(fit_policy()),
        "old_C2_C": old["C"],
        "old_settings_sha256": old["old_settings_sha256"],
        "input_artifact_sha256": bindings,
        "source_sha256": source_bindings(),
        "fit_context": {
            "partition": "fit",
            "test_rows_loaded": 0,
            "fresh_development_rows_loaded": 0,
            "scaler_fitted": False,
        },
        "training_ids": {name: [rows[i]["row_id"] for i in ix] for name, ix in indices.items()},
        "training_group_ids": sorted({r["group_id"] for r in rows}),
        "arms": {},
        "environment": environment_info(),
    }
    manifest = {
        "schema": "format-adaptation-fit-manifest-v1",
        "status": "fitting",
        "started_at": utc_now(),
        "model_key": model_key,
    }
    atomic_json(target / "settings.json", settings)
    atomic_json(target / "fit_manifest.json", manifest)
    arrays = {"mean": old["mean"].copy(), "scale": old["scale"].copy()}
    stage = "frozen_transform"
    try:
        with threadpool_limits(limits=4):
            x = transform(data["features"], old["mean"], old["scale"])
            y = np.array([[r["state"][f] for f in FACTOR_FIELDS] for r in rows], dtype=np.int64)
            for arm in ARMS[:3]:
                ix = indices[arm]
                C = 512 * old["C"] / len(ix)
                records, coefficients, intercepts = {}, [], []
                for j, factor in enumerate(FACTORS):
                    stage = f"{arm}/{factor}"
                    model = fit_lr(x[ix], y[ix, j], C)
                    coefficients.append(model.coef_[0])
                    intercepts.append(model.intercept_[0])
                    records[factor] = {"n_iter": model.n_iter_.tolist(), "status": "converged"}
                arrays[arm + "__coef"] = np.array(coefficients, dtype=np.float64)
                arrays[arm + "__intercept"] = np.array(intercepts, dtype=np.float64)
                settings["arms"][arm] = {"n": len(ix), "C": C, "factors": records}
                atomic_npz(target / "coefficients.npz", **arrays)
                atomic_json(target / "settings.json", settings)
            arrays["Offset__coef"] = arrays["Ahalf256__coef"].copy()
            arrays["Offset__intercept"] = arrays["Ahalf256__intercept"].copy()
            offsets, records = [], {}
            ix = indices["B_offset"]
            base = x[ix] @ arrays["Ahalf256__coef"].T + arrays["Ahalf256__intercept"]
            for j, factor in enumerate(FACTORS):
                stage = f"Offset/{factor}"
                offset, record = fit_intercept_offset(base[:, j], y[ix, j])
                offsets.append(offset)
                records[factor] = record
            arrays["Offset__B_offset"] = np.array(offsets, dtype=np.float64)
            settings["arms"]["Offset"] = {
                "n_A": 256,
                "n_B": 256,
                "unpenalized_offset": True,
                "factors": records,
            }
        check_bindings(bindings)
        atomic_npz(target / "coefficients.npz", **arrays)
        settings.update(
            status="frozen",
            frozen_at=utc_now(),
            coefficients_sha256=file_sha256(target / "coefficients.npz"),
        )
        atomic_json(target / "settings.json", settings)
        manifest.update(
            status="frozen",
            frozen_at=settings["frozen_at"],
            input_artifact_sha256=bindings,
            output_sha256={
                name: file_sha256(target / name) for name in ("settings.json", "coefficients.npz")
            },
        )
        atomic_json(target / "fit_manifest.json", manifest)
        validate_bundle(target, allow_test_fixture=allow_test_fixture)
        return settings
    except BaseException as error:
        settings.update(status="failed_preserved", failed_stage=stage)
        atomic_json(target / "settings.json", settings)
        manifest.update(
            status="failed_preserved",
            failed_stage=stage,
            finished_at=utc_now(),
            error_type=type(error).__name__,
            error=str(error),
        )
        atomic_json(target / "fit_manifest.json", manifest)
        raise


def validate_bundle(path, *, allow_test_fixture=False):
    directory = Path(path).resolve()
    settings, manifest = (
        read_json(directory / name) for name in ("settings.json", "fit_manifest.json")
    )
    require(
        settings.get("schema") == "format-adaptation-fit-v1"
        and manifest.get("schema") == "format-adaptation-fit-manifest-v1"
        and settings.get("status") == manifest.get("status") == "frozen",
        "Fit is not frozen",
    )
    require(
        type(settings.get("test_fixture")) is bool
        and (not settings["test_fixture"] or allow_test_fixture),
        "Native run rejects fixture fit",
    )
    require(
        settings.get("model_key") in MODEL_SITES
        and settings["site"] == MODEL_SITES[settings["model_key"]],
        "Fit model/site differs",
    )
    require(
        settings.get("policy") == fit_policy()
        and settings.get("policy_sha256") == canonical_hash(fit_policy()),
        "Fit policy differs",
    )
    require(settings.get("source_sha256") == source_bindings(), "Fit source changed")
    require(
        set(manifest.get("output_sha256", {})) == {"settings.json", "coefficients.npz"},
        "Wrong fit output inventory",
    )
    for name, digest in manifest["output_sha256"].items():
        require(file_sha256(directory / name) == digest, f"Fit artifact changed: {name}")
    require(
        settings["coefficients_sha256"] == file_sha256(directory / "coefficients.npz"),
        "Fit coefficients changed",
    )
    check_bindings(settings["input_artifact_sha256"])
    require(
        manifest["input_artifact_sha256"] == settings["input_artifact_sha256"],
        "Fit input bindings differ",
    )
    require(
        settings.get("fit_context")
        == {
            "partition": "fit",
            "test_rows_loaded": 0,
            "fresh_development_rows_loaded": 0,
            "scaler_fitted": False,
        },
        "Invalid fit context",
    )
    require(set(settings["arms"]) == set(ARMS), "Incomplete four-arm fit")
    C = settings["old_C2_C"]
    require(np.isfinite(C) and C > 0, "Invalid old C")
    ids = settings["training_ids"]
    require(
        set(ids) == {*ARMS[:3], "B_offset"} and len(settings["training_group_ids"]) == 16,
        "Invalid fitting membership",
    )
    for arm, n in (("A512", 512), ("Mixed512", 512), ("Ahalf256", 256)):
        require(
            len(ids[arm]) == len(set(ids[arm])) == settings["arms"][arm]["n"] == n
            and settings["arms"][arm]["C"] == 512 * C / n,
            "Fit sample/penalty budget differs",
        )
    require(
        len(ids["B_offset"]) == len(set(ids["B_offset"])) == 256
        and set(ids["Ahalf256"]).issubset(ids["A512"])
        and set(ids["Mixed512"]) == set(ids["Ahalf256"]) | set(ids["B_offset"])
        and set(ids["A512"]).isdisjoint(ids["B_offset"]),
        "Fit arm membership differs",
    )
    expected = {"mean", "scale", "Offset__B_offset"}
    expected.update(arm + "__" + field for arm in ARMS for field in ("coef", "intercept"))
    with np.load(directory / "coefficients.npz", allow_pickle=False) as archive:
        require(set(archive.files) == expected, "Wrong safe coefficient-array inventory")
        arrays = {name: archive[name].copy() for name in archive.files}
    h = settings["hidden_size"]
    for name, array in arrays.items():
        shape = (4, h) if name.endswith("__coef") else (h,) if name in ("mean", "scale") else (4,)
        require(
            array.dtype == np.float64 and array.shape == shape and np.isfinite(array).all(),
            f"Invalid safe parameter array: {name}",
        )
    require((arrays["scale"] > 0).all(), "Invalid frozen scales")
    for name in ("coef", "intercept"):
        require(
            np.array_equal(arrays["Offset__" + name], arrays["Ahalf256__" + name]),
            "Offset altered A-half parameters",
        )
    return settings, arrays


def score_bundle(settings, arrays, features, renderers):
    require(
        len(features) == len(renderers) and set(renderers).issubset({"A", "B"}),
        "Invalid scoring renderer inventory",
    )
    x = transform(features, arrays["mean"], arrays["scale"])
    result = {}
    for arm in ARMS:
        margins = x @ arrays[arm + "__coef"].T + arrays[arm + "__intercept"]
        if arm == "Offset":
            margins = margins + (np.array(renderers) == "B")[:, None] * arrays["Offset__B_offset"]
        require(np.isfinite(margins).all(), "Nonfinite adaptation score")
        bits = (margins > 0).astype(np.int64)
        result[arm] = {
            "factor_margins": margins,
            "factor_predictions": bits,
            "verdicts": fold_factors(bits),
        }
    return result
