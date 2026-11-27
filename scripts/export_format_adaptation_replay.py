"""Export or replay audited derived scores without native paths or hidden vectors.

Portable replay reuses the unchanged production metric functions. It is not an
independent audit and does not establish the feature-to-prediction calculation.
Public export/replay reject fixtures; explicit private helpers support tests.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import posixpath
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("llama", "qwen")
ARMS = ("A512", "Mixed512", "Ahalf256", "Offset")
FACTORS = ("I", "M1", "B1", "B2")
SITES = {"llama": "layer_16_query_location_last", "qwen": "layer_21_query_location_last"}
SOURCE_NAMES = ("format_adaptation_analysis.py", "format_adaptation_fit.py", "storage.py")
ARRAY_NAMES = {
    "row_ids",
    "arms",
    "factor_order",
    "verdicts",
    "factor_margins",
    "factor_predictions",
    "output_margins",
    "strict_first_token_correct",
}
ROW_FIELDS = ("row_id", "group_id", "partition", "split", "renderer", "condition", "state", "gold")
HASH_FIELDS = ("messages_sha256", "rendered_chat_sha256", "prompt_token_ids_sha256")
OUTPUT_FIELDS = (
    "argmax_token_id",
    "argmax_tie_count",
    "yes_logprob",
    "no_logprob",
    "yes_minus_no",
    "yes_logit",
    "no_logit",
    "argmax_logit",
    "vocab_size",
)
ATOL = RTOL = 2e-10
SCOPE = (
    "Portable replay using unchanged production metrics; not a second independent audit "
    "or a feature-to-prediction reconstruction. No model calls, fits, tokenizer calls, "
    "new thresholds or hidden vectors."
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def canonical_path(value):
    return posixpath.normpath(str(value).replace("\\", "/"))


def hash_binding(mapping, name, digest):
    require(isinstance(mapping, dict), "Missing original hash bindings")
    normalized = {canonical_path(k): v for k, v in mapping.items()}
    require(
        len(normalized) == len(mapping) and normalized.get(canonical_path(name)) == digest,
        "Missing or stale original binding: " + str(name),
    )


def read_json(path):
    return json.loads(Path(path).read_bytes())


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def write_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(json_bytes(value))


def compressed(path, value):
    Path(path).write_bytes(gzip.compress(value, compresslevel=9, mtime=0))


def unpack(path):
    return gzip.decompress(Path(path).read_bytes())


def _audit_identity(audit):
    require(
        audit.get("schema") == "format-adaptation-independent-audit-v1"
        and audit.get("status") == "pass"
        and set(audit.get("models", {})) == set(MODELS)
        and audit.get("model_calls") == audit.get("classifier_fits") == 0
        and audit.get("production_functions_imported") is False
        and audit.get("saved_analysis_opened_before_both_reconstructions_frozen") is False,
        "Passing complete both-model independent audit required",
    )
    identity = canonical_path(audit.get("campaign", ""))
    require(
        identity.startswith("/") or (len(identity) > 2 and identity[1] == ":"),
        "Original campaign identity missing",
    )
    return identity


def _sources(seal, output):
    result = {}
    package = output / "production/latent_probes"
    package.mkdir(parents=True)
    # Empty package glue: all scientific modules below are exact sealed bytes.
    (package / "__init__.py").write_bytes(b'"""Portable production metric source snapshot."""\n')
    for name in SOURCE_NAMES:
        suffix = "/src/latent_probes/" + name
        found = [
            (p, d) for p, d in seal["source_sha256"].items() if canonical_path(p).endswith(suffix)
        ]
        require(len(found) == 1, "Missing/ambiguous sealed production metric source: " + name)
        original, digest = found[0]
        value = (ROOT / "src/latent_probes" / name).read_bytes()
        require(sha(value) == digest, "Production source differs from original test seal")
        (package / name).write_bytes(value)
        result[name] = {"original_path": original, "sha256": digest}
    return result


def _metrics(bundle):
    """Load exact local snapshots under an isolated package, never native paths."""
    package = bundle / "production/latent_probes"
    name = "_portable_format_adaptation_" + sha(str(bundle).encode())[:16]
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                name, package / "__init__.py", submodule_search_locations=[str(package)]
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return importlib.import_module(name + ".format_adaptation_analysis")
    finally:
        sys.dont_write_bytecode = previous


def _arrays(path, rows):
    with np.load(path, allow_pickle=False) as saved:
        require(
            set(saved.files) == ARRAY_NAMES, "Unexpected prediction arrays; derived scores only"
        )
        values = {name: saved[name].copy() for name in saved.files}
    for name, expected in (("arms", ARMS), ("factor_order", FACTORS)):
        require(
            values[name].dtype.kind in "US" and values[name].tolist() == list(expected),
            "Wrong model arm/factor order",
        )
    require(
        values["row_ids"].dtype.kind in "US"
        and values["row_ids"].tolist() == [r["row_id"] for r in rows],
        "Prediction row join/order differs",
    )
    for name in ("verdicts", "factor_predictions"):
        shape = (4992, 4) if name == "verdicts" else (4992, 4, 4)
        require(
            values[name].dtype.kind in "iu"
            and values[name].shape == shape
            and np.isin(values[name], (0, 1)).all(),
            "Invalid binary prediction inventory",
        )
    for name, shape in (("factor_margins", (4992, 4, 4)), ("output_margins", (4992,))):
        require(
            values[name].dtype == np.float64
            and values[name].shape == shape
            and np.isfinite(values[name]).all(),
            "Invalid continuous prediction inventory",
        )
    require(
        values["strict_first_token_correct"].dtype == bool
        and values["strict_first_token_correct"].shape == (4992,),
        "Invalid strict token flags",
    )
    return values


def _project(payload, key, fixture):
    require(
        payload.get("schema") == "format-adaptation-capture-rows-v1"
        and payload.get("model_key") == key
        and payload.get("site") == SITES[key]
        and payload.get("test_fixture") is fixture,
        "Capture identity/fixture mismatch",
    )
    tokens = payload.get("answer_token_ids")
    require(
        isinstance(tokens, dict)
        and set(tokens) == {"YES", "NO"}
        and all(type(v) is int and v >= 0 for v in tokens.values())
        and tokens["YES"] != tokens["NO"],
        "Missing candidate token identities",
    )
    projected = []
    for row in payload["rows"]:
        require(
            all(
                isinstance(row.get(name), str) and row[name]
                for name in ("row_id", "group_id", "partition", "split", "renderer", "condition")
            ),
            "Projected row identities must be scalar strings",
        )
        state = row.get("state")
        fields = {"group_id", "mechanism", "world", "name_map"}
        fields |= (
            {"asserted"}
            if row["condition"] == "asserted"
            else {"initial", "update", "first_receipt", "receipt"}
        )
        require(
            isinstance(state, dict)
            and set(state) == fields
            and state.get("group_id") == row["group_id"]
            and isinstance(state.get("mechanism"), str),
            "Unexpected formal-state fields; vectors cannot be projected",
        )
        require(
            all(
                (value is None and name == "first_receipt" and row["condition"] == "one_step")
                or (type(value) is int and value in (0, 1))
                for name, value in state.items()
                if name not in ("group_id", "mechanism")
            ),
            "Formal state values must be declared scalar bits",
        )
        require(type(row.get("gold")) is int and row["gold"] in (0, 1), "Invalid scalar gold")
        require(
            all(
                isinstance(row[name], str)
                and len(row[name]) == 64
                and all(c in "0123456789abcdef" for c in row[name])
                for name in HASH_FIELDS
                if name in row
            ),
            "Invalid original prompt hash",
        )
        summary = row.get("output_logits")
        require(
            isinstance(summary, dict)
            and all(type(v) in (str, int, float, bool) or v is None for v in summary.values())
            and all(
                name in summary
                for name in (
                    "argmax_token_id",
                    "argmax_tie_count",
                    "yes_logprob",
                    "no_logprob",
                    "yes_minus_no",
                )
            ),
            "Scalar output summary required; no hidden/logit vectors permitted",
        )
        projected.append(
            {
                **{name: row[name] for name in ROW_FIELDS},
                **{name: row[name] for name in HASH_FIELDS if name in row},
                "output_logits": {name: summary[name] for name in OUTPUT_FIELDS if name in summary},
            }
        )
    return {
        "schema": "format-adaptation-portable-rows-v1",
        "model_key": key,
        "site": SITES[key],
        "test_fixture": fixture,
        "answer_token_ids": tokens,
        "rows": projected,
    }


def compare_tree(expected, actual, path="metrics", counts=None):
    counts = counts or {"exact": 0, "numeric": 0, "maximum_absolute_difference": 0.0}
    if isinstance(expected, dict):
        require(
            isinstance(actual, dict) and set(expected) == set(actual), "Metric keys differ: " + path
        )
        for name in expected:
            compare_tree(expected[name], actual[name], path + "/" + name, counts)
    elif isinstance(expected, list):
        require(
            isinstance(actual, list) and len(expected) == len(actual),
            "Metric list differs: " + path,
        )
        for i, (a, b) in enumerate(zip(expected, actual, strict=True)):
            compare_tree(a, b, path + "/" + str(i), counts)
    elif type(expected) is float:
        require(
            type(actual) in (int, float)
            and math.isfinite(expected)
            and math.isfinite(actual)
            and math.isclose(expected, actual, abs_tol=ATOL, rel_tol=RTOL),
            "Numeric metric differs: " + path,
        )
        counts["numeric"] += 1
        counts["maximum_absolute_difference"] = max(
            counts["maximum_absolute_difference"], abs(expected - actual)
        )
    else:
        require(
            type(expected) is type(actual) and expected == actual, "Exact metric differs: " + path
        )
        counts["exact"] += 1
    return counts


def _calculate(bundle, key, saved, projection, metric):
    fixture = saved["test_fixture"]
    require(
        projection.get("schema") == "format-adaptation-portable-rows-v1"
        and projection.get("model_key") == key
        and projection.get("site") == SITES[key]
        and projection.get("test_fixture") is fixture,
        "Projected model/fixture identity differs",
    )
    rows = projection["rows"]
    metric.validate_test_rows(rows)
    arrays = _arrays(bundle / key / "predictions.npz", rows)
    tokens = projection["answer_token_ids"]
    for i, row in enumerate(rows):
        summary = row["output_logits"]
        require(
            summary["yes_minus_no"] == arrays["output_margins"][i]
            and bool(arrays["strict_first_token_correct"][i])
            == (summary["argmax_token_id"] == tokens["YES" if row["gold"] else "NO"]),
            "Saved output comparator differs from projected scalar summary",
        )
    scores = {
        arm: {
            "verdicts": arrays["verdicts"][:, j],
            "factor_predictions": arrays["factor_predictions"][:, j],
            "factor_margins": arrays["factor_margins"][:, j],
        }
        for j, arm in enumerate(ARMS)
    }
    result = metric.score_arrays(
        rows, scores, arrays["output_margins"], arrays["strict_first_token_correct"], model_key=key
    )
    require("output_first_token_detail" in saved, "Original output-token scope missing")
    result["output_first_token_detail"] = metric.output_token_detail(rows, tokens)
    checks = compare_tree(result, {name: saved[name] for name in result})
    return {
        "rows": len(rows),
        "arms": list(ARMS),
        "factor_order": list(FACTORS),
        "replayed_scopes": list(result),
        "comparisons": checks,
    }


def _local_files(bundle, manifest):
    require(
        manifest.get("schema") == "format-adaptation-portable-replay-v1"
        and manifest.get("status") == "complete",
        "Portable bundle is incomplete",
    )
    require(manifest.get("models") == list(MODELS), "Portable model order differs")
    require(isinstance(manifest.get("files"), dict) and manifest["files"], "Missing bundle hashes")
    actual = {
        str(p.relative_to(bundle)).replace("\\", "/") for p in bundle.rglob("*") if p.is_file()
    }
    require(actual == set(manifest["files"]) | {"manifest.json"}, "Unexpected/missing bundle files")
    for name, digest in manifest["files"].items():
        path = bundle / name
        require(
            not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and not path.is_symlink()
            and path.resolve().is_relative_to(bundle)
            and sha(path.read_bytes()) == digest,
            "Changed/unsafe bundle member: " + name,
        )
    require(
        sha(Path(__file__).read_bytes()) == manifest["exporter_sha256"],
        "Use the exact exported replay.py implementation",
    )


def _replay_bundle(bundle, output=None, *, allow_fixture=False):
    bundle = Path(bundle).resolve()
    manifest = read_json(bundle / "manifest.json")
    _local_files(bundle, manifest)
    fixture = manifest.get("test_fixture")
    require(
        type(fixture) is bool and (allow_fixture or fixture is False),
        "Public portable replay rejects synthetic fixtures",
    )
    audit_raw, seal_raw = (
        unpack(bundle / name) for name in ("audit_receipt.json.gz", "test_seal.json.gz")
    )
    audit, seal = json.loads(audit_raw), json.loads(seal_raw)
    identity = _audit_identity(audit)
    require(identity == manifest["original_campaign"], "Portable campaign identity differs")
    require(
        seal.get("schema") == "format-adaptation-test-seal-v1"
        and set(seal.get("models", {})) == set(MODELS),
        "Incomplete both-model test seal",
    )
    hash_binding(audit["input_sha256"], identity + "/test_seal.json", sha(seal_raw))
    reconstruction = unpack(bundle / "reconstruction_seal.json.gz")
    require(
        sha(reconstruction) == audit["reconstruction_seal_sha256"],
        "Independent reconstruction seal changed",
    )
    for name, source in manifest["production_sources"].items():
        hash_binding(seal["source_sha256"], source["original_path"], source["sha256"])
        require(
            sha((bundle / "production/latent_probes" / name).read_bytes()) == source["sha256"],
            "Copied production metric source differs from original seal",
        )
    require(
        set(manifest["production_sources"]) == set(SOURCE_NAMES), "Incomplete metric source closure"
    )
    metric = _metrics(bundle)
    results = {}
    row_identities = None
    for key in MODELS:
        directory = bundle / key
        raw = unpack(directory / "results.json.gz")
        saved, original = (
            json.loads(raw),
            json.loads(unpack(directory / "analysis_manifest.json.gz")),
        )
        require(
            saved.get("schema") == "format-adaptation-analysis-v1"
            and saved.get("model_key") == key
            and saved.get("site") == SITES[key]
            and saved.get("test_fixture") is fixture
            and saved.get("research_status") == "agent_audited_exploratory"
            and saved.get("confirmatory") is False,
            "Original model result provenance differs",
        )
        prediction_hash = sha((directory / "predictions.npz").read_bytes())
        require(
            original.get("schema") == "format-adaptation-analysis-manifest-v1"
            and original.get("status") == "complete"
            and original.get("output_sha256")
            == {"results.json": sha(raw), "predictions.npz": prediction_hash},
            "Original analysis manifest/result binding differs",
        )
        for name, value in (("results.json", sha(raw)), ("predictions.npz", prediction_hash)):
            hash_binding(
                audit["analysis_output_sha256"], identity + "/" + key + "_analysis/" + name, value
            )
        hash_binding(
            audit["models"][key]["analysis_artifact_sha256"],
            identity + "/" + key + "_analysis/manifest.json",
            sha(unpack(directory / "analysis_manifest.json.gz")),
        )
        hash_binding(original["input_artifact_sha256"], identity + "/test_seal.json", sha(seal_raw))
        capture_raw = unpack(directory / "capture_manifest.json.gz")
        capture = json.loads(capture_raw)
        hash_binding(
            audit["input_sha256"], identity + "/" + key + "/test/manifest.json", sha(capture_raw)
        )
        require(
            capture.get("schema") == "format-adaptation-capture-manifest-v1"
            and capture.get("status") == "complete"
            and capture.get("partition") == "test"
            and capture.get("completed_rows") == 4992
            and capture.get("rows_sha256")
            == manifest["row_projection"][key]["original_rows_sha256"]
            and capture.get("seal_sha256", capture.get("test_seal_sha256")) == sha(seal_raw)
            and capture.get("bank_sha256") == seal["bank_sha256"],
            "Original capture provenance differs",
        )
        projected = json.loads(unpack(directory / "rows.json.gz"))
        ids = [
            (r["row_id"], r["group_id"], r["condition"], r["renderer"], r["state"], r["gold"])
            for r in projected["rows"]
        ]
        require(
            row_identities is None or ids == row_identities, "Checkpoint row populations differ"
        )
        row_identities = ids
        hash_binding(
            audit["input_sha256"],
            identity + "/" + key + "/test/rows.json",
            manifest["row_projection"][key]["original_rows_sha256"],
        )
        results[key] = _calculate(bundle, key, saved, projected, metric)
    receipt = {
        "schema": "format-adaptation-portable-replay-receipt-v1",
        "status": "pass",
        "created_at": datetime.now(UTC).isoformat(),
        "test_fixture": fixture,
        "scope": SCOPE,
        "independent_audit": False,
        "feature_to_prediction_verified": False,
        "model_calls": 0,
        "classifier_fits": 0,
        "original_campaign": identity,
        "bundle_manifest_sha256": sha((bundle / "manifest.json").read_bytes()),
        "production_metric_module": Path(metric.__file__).relative_to(bundle).as_posix(),
        "numeric_tolerance": {"absolute": ATOL, "relative": RTOL},
        "models": results,
    }
    if output is not None:
        target = Path(output).resolve()
        require(
            not target.is_relative_to(bundle), "Write replay receipts outside the immutable bundle"
        )
        write_json(target, receipt)
    return receipt


def replay_bundle(bundle, output=None):
    return _replay_bundle(bundle, output)


def _export_bundle(campaign, audit_directory, output, *, allow_fixture=False):
    campaign, audit_directory, output = map(
        lambda p: Path(p).resolve(), (campaign, audit_directory, output)
    )
    require(not output.exists(), "Preserve prior portable bundle; output already exists")
    require(
        not output.is_relative_to(campaign) and not output.is_relative_to(audit_directory),
        "Keep portable outputs outside immutable native evidence directories",
    )
    audit_raw = (audit_directory / "audit_receipt.json").read_bytes()
    audit = json.loads(audit_raw)
    identity = _audit_identity(audit)
    require(canonical_path(campaign) == identity, "Audit belongs to a different campaign")
    seal_raw = (campaign / "test_seal.json").read_bytes()
    seal = json.loads(seal_raw)
    hash_binding(audit["input_sha256"], identity + "/test_seal.json", sha(seal_raw))
    reconstruction = (audit_directory / "reconstruction_seal.json").read_bytes()
    require(
        sha(reconstruction) == audit["reconstruction_seal_sha256"], "Reconstruction seal is stale"
    )
    prepared = {}
    fixture = None
    input_hashes = {
        str(audit_directory / "audit_receipt.json"): sha(audit_raw),
        str(campaign / "test_seal.json"): sha(seal_raw),
        str(audit_directory / "reconstruction_seal.json"): sha(reconstruction),
    }
    for key in MODELS:
        analysis = campaign / (key + "_analysis")
        rows_path = campaign / key / "test/rows.json"
        result_raw, analysis_raw, rows_raw, capture_raw = (
            p.read_bytes()
            for p in (
                analysis / "results.json",
                analysis / "manifest.json",
                rows_path,
                campaign / key / "test/manifest.json",
            )
        )
        saved, original, payload, capture = map(
            json.loads, (result_raw, analysis_raw, rows_raw, capture_raw)
        )
        flag = saved.get("test_fixture")
        require(
            type(flag) is bool
            and (allow_fixture or flag is False)
            and (fixture is None or fixture is flag),
            "Public portable export rejects synthetic/mixed fixtures",
        )
        fixture = flag
        require(
            capture.get("schema") == "format-adaptation-capture-manifest-v1"
            and capture.get("status") == "complete"
            and capture.get("partition") == "test"
            and capture.get("completed_rows") == 4992
            and capture.get("rows_sha256") == sha(rows_raw)
            and capture.get("seal_sha256", capture.get("test_seal_sha256")) == sha(seal_raw)
            and capture.get("bank_sha256") == seal["bank_sha256"],
            "Incomplete/misbound original test rows",
        )
        hash_binding(audit["input_sha256"], identity + "/" + key + "/test/rows.json", sha(rows_raw))
        hash_binding(
            audit["input_sha256"], identity + "/" + key + "/test/manifest.json", sha(capture_raw)
        )
        projection = _project(payload, key, fixture)
        # Refuse unexpected arrays before creating or copying any export member.
        _arrays(analysis / "predictions.npz", projection["rows"])
        prepared[key] = (result_raw, analysis_raw, capture_raw, projection)
        input_hashes.update(
            {
                str(p): sha(value)
                for p, value in (
                    (analysis / "results.json", result_raw),
                    (analysis / "manifest.json", analysis_raw),
                    (rows_path, rows_raw),
                    (campaign / key / "test/manifest.json", capture_raw),
                )
            }
        )
    output.mkdir(parents=True)
    manifest = {
        "schema": "format-adaptation-portable-replay-v1",
        "status": "exporting",
        "created_at": datetime.now(UTC).isoformat(),
        "test_fixture": fixture,
        "scope": SCOPE,
        "original_campaign": identity,
        "original_audit": str(audit_directory),
        "input_sha256": input_hashes,
        "row_projection": {},
        "models": list(MODELS),
        "exporter_sha256": sha(Path(__file__).read_bytes()),
        "export_environment": {
            "python": sys.version.split()[0],
            **{
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")
            },
        },
    }
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    try:
        compressed(output / "audit_receipt.json.gz", audit_raw)
        compressed(output / "test_seal.json.gz", seal_raw)
        compressed(output / "reconstruction_seal.json.gz", reconstruction)
        manifest["production_sources"] = _sources(seal, output)
        (output / "replay.py").write_bytes(Path(__file__).read_bytes())
        banner = "SYNTHETIC FIXTURE - NO RESEARCH RESULT\n\n" if fixture else ""
        (output / "README.txt").write_text(
            banner + SCOPE + "\nRun: python replay.py replay --bundle .\n"
            "Dependencies: Python 3.11+, numpy, scipy, scikit-learn, threadpoolctl.\n"
            "The full independent audit and raw evidence remain at the original native location.\n"
            "Projection integrity is bound at export; this replay cannot independently reconstruct omitted prompts or vectors.\n",
            encoding="utf-8",
        )
        for key, (result_raw, analysis_raw, capture_raw, projection) in prepared.items():
            directory = output / key
            directory.mkdir()
            predictions = campaign / (key + "_analysis/predictions.npz")
            value = predictions.read_bytes()
            (directory / "predictions.npz").write_bytes(value)
            manifest["input_sha256"][str(predictions)] = sha(value)
            compressed(directory / "results.json.gz", result_raw)
            compressed(directory / "analysis_manifest.json.gz", analysis_raw)
            compressed(directory / "capture_manifest.json.gz", capture_raw)
            compressed(directory / "rows.json.gz", json_bytes(projection))
            manifest["row_projection"][key] = {
                "original_rows_sha256": input_hashes[str(campaign / key / "test/rows.json")],
                "projection_fields": list(ROW_FIELDS) + list(HASH_FIELDS) + ["output_logits"],
                "output_summary": "Original scalar summary retained; no vectors or prompt text",
            }
        manifest["files"] = {
            str(p.relative_to(output)).replace("\\", "/"): sha(p.read_bytes())
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        }
        manifest["status"] = "complete"
        (output / "manifest.json").write_bytes(json_bytes(manifest))
        receipt = _replay_bundle(output, allow_fixture=allow_fixture)
        for path, digest in manifest["input_sha256"].items():
            require(sha(Path(path).read_bytes()) == digest, "Original export input changed")
        return receipt
    except BaseException as error:
        manifest.update(
            status="failed_preserved", error_type=type(error).__name__, error=str(error)
        )
        (output / "manifest.json").write_bytes(json_bytes(manifest))
        raise


def export_bundle(campaign, audit_directory, output):
    return _export_bundle(campaign, audit_directory, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    export = modes.add_parser("export")
    export.add_argument("--campaign", type=Path, required=True)
    export.add_argument("--audit", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    for name in ("replay", "verify"):
        command = modes.add_parser(name)
        command.add_argument("--bundle", type=Path, required=True)
        command.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "export":
        result = export_bundle(args.campaign, args.audit, args.output)
    else:
        result = replay_bundle(args.bundle, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "scope": result["scope"],
                "test_fixture": result["test_fixture"],
                "models": list(result["models"]),
            }
        )
    )


if __name__ == "__main__":
    main()
