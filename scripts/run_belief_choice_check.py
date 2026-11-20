"""Post hoc A/B question-format diagnostic; no new holdout or feature capture.

Reuse the completed belief panel, exact query-free scenes and candidate claims.
Only the explicit question/answer format changes. Invalid outputs stay invalid.
"""

from __future__ import annotations

import argparse
import copy
import re
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from latent_probes.backends import HFBackend
from latent_probes.belief_panel import validate_panel
from latent_probes.config import GenerationSettings, load_config
from latent_probes.storage import atomic_json, canonical_hash, file_sha256, read_json, utc_now

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = PROJECT_ROOT / "scripts/run_belief_diagnostic.py"
SEED = 42
GENERATION = GenerationSettings(max_new_tokens=6, do_sample=False)
SIGNATURE_FIELDS = (
    "backend",
    "is_test_fixture",
    "model_id",
    "requested_revision",
    "resolved_revision",
    "model_type",
    "dtype",
    "quantization",
    "device_map",
    "decoder_blocks",
    "hidden_size",
    "capture_layers",
    "chat_template_sha256",
    "special_token_ids",
    "attention_implementation",
    "extraction",
)


def parse_choice(text):
    match = re.fullmatch(r"\s*([AB])[.!]?\s*", text, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def disjoint_paths(source_run, output_dir):
    source, output = Path(source_run).resolve(), Path(output_dir).resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Source and output must be disjoint directory trees")
    if output.exists():
        raise FileExistsError("Use a new choice-check output directory")
    return source, output


def option_order(template_id):
    number = int(template_id.removeprefix("bd_t"))
    if number not in range(1, 9):
        raise ValueError("Unknown template for fixed option ordering")
    return (0, 1) if number % 2 else (1, 0)


def choice_messages(case, view, target):
    """Append a question using names/places only, never gold or private source bits."""
    if target not in ("world", "reader", "nested") or view not in ("omniscient", "sender"):
        raise ValueError("Unknown perspective")
    if view == "sender" and target == "reader":
        raise ValueError("Private actual reader state is unavailable in sender view")
    order = option_order(case["template_id"])
    if target == "world":
        question = "At the end of the scene, where is the token actually located?"
    elif target == "reader":
        question = (
            f"At the end of the scene, where does {case['reader_name']} "
            "believe the token is located?"
        )
    else:
        question = (
            f"At the end of the scene, where does {case['sender_name']} think "
            f"{case['reader_name']} believes the token is located?"
        )
    places = (case["place_a"], case["place_b"])
    question += f"\nA: the {places[order[0]]}\nB: the {places[order[1]]}\nAnswer only A or B."
    messages = copy.deepcopy(case["messages"][view])
    messages[-1]["content"] += "\n\nQuestion: " + question
    return messages


def build_jobs(panel):
    validate_panel(panel)
    jobs = []
    for case in panel["cases"]:
        order = option_order(case["template_id"])
        for view in ("omniscient", "sender"):
            targets = ("world", "reader", "nested") if view == "omniscient" else ("world", "nested")
            for target in targets:
                messages = choice_messages(case, view, target)
                gold = case["labels"][target]
                identifier = case["id"] + "__" + view + "__" + target
                jobs.append(
                    {
                        "example_id": identifier + "__choice",
                        "source_example_id": identifier,
                        "case_id": case["id"],
                        "template_id": case["template_id"],
                        "phase": case["phase"],
                        "split": case["split"],
                        "view": view,
                        "target": target,
                        "option_place_bits": list(order),
                        "gold": gold,
                        "gold_choice": "AB"[order.index(gold)],
                        "messages": messages,
                        "prompt_sha256": canonical_hash(messages),
                    }
                )
    if len(jobs) != 400 or len({row["prompt_sha256"] for row in jobs}) != 320:
        raise ValueError("Expected 400 rows and 320 unique fixed-format prompts")
    return jobs


def _tree_hashes(root):
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _safe_source_file(source, name):
    path = (source / name).resolve()
    if not path.is_relative_to(source) or not path.is_file():
        raise ValueError("Source artifact is missing or escapes the source directory")
    return path


def source_state(source_run):
    source = Path(source_run).resolve()
    manifest = read_json(source / "manifest.json")
    if (
        manifest.get("schema") != "belief_construct_development_run_v1"
        or manifest.get("status") != "complete"
        or manifest.get("n_cases_completed") != 80
        or manifest.get("n_behavior_rows") != 400
    ):
        raise ValueError("Source must be the completed 80-case belief diagnostic")
    protocol = read_json(source / "protocol.json")
    panel = validate_panel(protocol["panel"])
    if canonical_hash(panel) != manifest.get("panel_sha256"):
        raise ValueError("Source panel identity changed")
    if file_sha256(source / "protocol.json") != manifest.get("protocol_sha256"):
        raise ValueError("Source protocol changed")
    if file_sha256(SOURCE_SCRIPT) != manifest.get("script_sha256"):
        raise ValueError("Original behavior policy script changed")
    policies = {str(SOURCE_SCRIPT): manifest["script_sha256"]}
    for relative, digest in manifest.get("source_sha256", {}).items():
        path = (PROJECT_ROOT / relative).resolve()
        if not path.is_relative_to(PROJECT_ROOT) or file_sha256(path) != digest:
            raise ValueError("Source panel/readout policy changed")
        policies[str(path)] = digest
    if set(manifest.get("source_sha256", {})) != {
        "src/latent_probes/belief_panel.py",
        "src/latent_probes/belief_readout.py",
    }:
        raise ValueError("Expected original source-policy fingerprints")
    tree = _tree_hashes(source)
    expected = dict(manifest.get("output_sha256", {}))
    expected["manifest.json"] = file_sha256(source / "manifest.json")
    if tree != expected:
        raise ValueError("Source output ledger or source bytes changed")
    for name in expected:
        _safe_source_file(source, name)
    jobs = build_jobs(panel)
    for job in jobs:
        relative = "behavior/" + job["source_example_id"] + ".json"
        if relative not in expected:
            raise ValueError("A paired source behavior record is missing")
        job["source_behavior_reference"] = relative
        job["source_behavior_sha256"] = expected[relative]
    return {
        "source": source,
        "manifest": manifest,
        "panel": panel,
        "jobs": jobs,
        "tree": tree,
        "policies": policies,
    }


def _assert_policy_unchanged(state, script_digest):
    source = state["source"]
    for name in ("manifest.json", "protocol.json"):
        if file_sha256(source / name) != state["tree"][name]:
            raise ValueError("Source protocol or completion state changed during the check")
    for name, digest in state["policies"].items():
        if file_sha256(name) != digest:
            raise ValueError("Source policy changed during the check")
    if file_sha256(__file__) != script_digest:
        raise ValueError("Choice-check policy changed during the check")
    if GENERATION.max_new_tokens != 6 or GENERATION.do_sample or SEED != 42:
        raise ValueError("Choice generation must stay greedy with six tokens and seed 42")


def assert_backend_matches(actual, expected):
    for field in SIGNATURE_FIELDS:
        if field not in actual or field not in expected or actual[field] != expected[field]:
            raise ValueError(f"Pinned source backend signature differs: {field}")


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[f"{row['view']}/{row['phase']}/{row['target']}"].append(row)
    result = {}
    for key, selected in sorted(groups.items()):
        correct = sum(row["prediction"] == row["gold"] for row in selected)
        classes = {}
        for gold in (0, 1):
            subset = [row for row in selected if row["gold"] == gold]
            classes[str(gold)] = {
                "n": len(subset),
                "correct": sum(row["prediction"] == gold for row in subset),
            }
        result[key] = {
            "n_rows": len(selected),
            "n_templates": len({row["template_id"] for row in selected}),
            "n_unique_prompts": len({row["prompt_sha256"] for row in selected}),
            "correct": correct,
            "invalid": sum(row["prediction"] is None for row in selected),
            "accuracy_including_invalid": correct / len(selected),
            "predicted_choices": dict(
                Counter(row["prediction_choice"] or "INVALID" for row in selected)
            ),
            "by_gold_bit": classes,
        }
    return {
        "perspectives": result,
        "interpretation": (
            "Post hoc question-format diagnostic on previously inspected development cases. "
            "A/B recovery indicates format sensitivity; failure is bounded to tested formats. "
            "This is not new held-out confirmation or evidence of hidden belief validity."
        ),
    }


def run_choice_check(source_run, output_dir, backend, *, progress=print):
    source, output = disjoint_paths(source_run, output_dir)
    state = source_state(source)
    assert_backend_matches(backend.metadata, state["manifest"]["backend"])
    script_digest = file_sha256(__file__)
    _assert_policy_unchanged(state, script_digest)
    output.mkdir(parents=True)
    plan = {
        "schema": "belief_choice_format_plan_v1",
        "created_at": utc_now(),
        "posthoc_development_diagnostic": True,
        "source_run": str(source),
        "source_sha256": state["tree"],
        "source_policy_sha256": state["policies"],
        "script_sha256": script_digest,
        "source_backend": state["manifest"]["backend"],
        "panel_sha256": canonical_hash(state["panel"]),
        "generation": {"max_new_tokens": 6, "do_sample": False, "seed": 42},
        "option_order_by_template": {
            case["template_id"]: list(option_order(case["template_id"]))
            for case in state["panel"]["cases"]
        },
        "primary_parser": "Whole response A or B, optional one terminal period/exclamation; otherwise INVALID",
        "unchanged": "Exact query-free scenes, candidate statements, model signature, six-token cap",
        "changed": "Explicit question asks the location and requests a counterbalanced A/B choice",
        "capture_features": False,
        "expected_rows": 400,
        "expected_unique_generation_calls": 320,
        "controls_excluded_from_fitting": True,
        "jobs": state["jobs"],
    }
    # The complete prompt/label/option-order plan exists before the first model call.
    atomic_json(output / "plan.json", plan)
    plan_digest = file_sha256(output / "plan.json")
    manifest = {
        "schema": "belief_choice_format_run_v1",
        "status": "running",
        "created_at": utc_now(),
        "source_run": str(source),
        "source_manifest_sha256": state["tree"]["manifest.json"],
        "script_sha256": script_digest,
        "plan_sha256": plan_digest,
        "backend": backend.metadata,
        "is_test_fixture": backend.metadata["is_test_fixture"],
        "n_behavior_rows": 0,
        "n_unique_generations": 0,
        "posthoc_development_diagnostic": True,
        "capture_features": False,
    }
    atomic_json(output / "manifest.json", manifest)
    cache, records = {}, []
    try:
        for index, job in enumerate(state["jobs"]):
            key = job["prompt_sha256"]
            reused = key in cache
            if not reused:
                _assert_policy_unchanged(state, script_digest)
                if file_sha256(output / "plan.json") != plan_digest:
                    raise ValueError("Frozen choice-check plan changed")
                generation = backend.generate(
                    copy.deepcopy(job["messages"]), GENERATION, SEED, capture=False
                ).record()
                if len(generation["generated_token_ids"]) > 6:
                    raise ValueError("Backend exceeded the six-token generation cap")
                cache[key] = generation
            generation = cache[key]
            choice = parse_choice(generation["text"])
            prediction = job["option_place_bits"]["AB".index(choice)] if choice else None
            record = {
                **job,
                "generation": generation,
                "prediction_choice": choice,
                "prediction": prediction,
                "reused_identical_generation": reused,
            }
            record["record_sha256"] = canonical_hash(record)
            atomic_json(output / "behavior" / (job["example_id"] + ".json"), record)
            records.append(record)
            manifest.update(n_behavior_rows=len(records), n_unique_generations=len(cache))
            if (index + 1) % 5 == 0:
                atomic_json(output / "manifest.json", manifest)
                if progress:
                    progress(f"Choice check {(index + 1) // 5}/80", flush=True)
        _assert_policy_unchanged(state, script_digest)
        if _tree_hashes(source) != state["tree"]:
            raise ValueError("Original source artifacts changed")
        if file_sha256(output / "plan.json") != plan_digest:
            raise ValueError("Frozen choice-check plan changed")
        if len(cache) != 320 or len(records) != 400:
            raise ValueError("Unexpected choice-check denominator")
        atomic_json(output / "summary.json", summarize(records))
        manifest.update(status="complete", source_unchanged=True)
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        manifest["finished_at"] = utc_now()
        manifest["output_sha256"] = {
            path.relative_to(output).as_posix(): file_sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        atomic_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    source, _output = disjoint_paths(args.source_run, args.output)
    state = source_state(source)
    original = state["manifest"]["backend"]
    settings = replace(
        load_config(args.config).model,
        model_id=original["model_id"],
        revision=original["resolved_revision"],
        dtype=original["dtype"],
        quantization=original["quantization"],
        capture_layers=list(original["capture_layers"]),
        local_files_only=True,
    )
    backend = HFBackend(settings)
    try:
        run_choice_check(source, args.output, backend)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
