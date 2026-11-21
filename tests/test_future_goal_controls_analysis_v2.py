"""Regression and exact-change audit for the versioned CPU EOS correction."""

import ast
from copy import deepcopy

import numpy as np
import pytest

from latent_probes import future_goal_controls_analysis as v1
from latent_probes import future_goal_controls_analysis_v2 as v2
from latent_probes.storage import atomic_json, file_sha256


def staged_run(tmp_path, eos):
    """Reach metadata validation, then stop at an intentionally absent config."""
    protocol = {
        "schema": "future_goal_controls_protocol_v1",
        "bank_sha256": v1.BANK_SHA256,
        "bank_path": str(v1.BANK),
        "primary_head": v1.HEAD,
        "threshold": 0.5,
        "threshold_tie": 0,
        "layers": [16],
        "hf_hidden_state_index": 17,
        "no_refit": True,
        "no_hyperparameter_selection": True,
        "generation": {
            "settings": {"max_new_tokens": 12, "do_sample": False},
            "seed": 20261121,
            "eos_token_ids": eos,
        },
        "source_backend": {
            "resolved_revision": "0" * 40,
            "generation_defaults": {"eos_token_id": [128001, 128008, 128009]},
        },
        "tokenizer": {
            "revision": "0" * 40,
            "all_special_ids": [128000, 128009],
            "special_token_ids": {"eos": 128001},
        },
        "model_settings": {"revision": "0" * 40},
        "config_path": str(tmp_path / "intentionally_absent_config.json"),
        "policy_sha256": {v2.FROZEN_ORIGINAL_NAME: v2.FROZEN_ORIGINAL_SHA256},
    }
    atomic_json(tmp_path / "protocol.json", protocol)
    atomic_json(
        tmp_path / "manifest.json",
        {
            "schema": "future_goal_controls_run_v1",
            "status": "neutral_failed_stopped",
            "bank_sha256": v1.BANK_SHA256,
            "protocol_sha256": file_sha256(tmp_path / "protocol.json"),
        },
    )
    return protocol


def test_model_eos_need_not_be_registered_tokenizer_special(tmp_path):
    staged_run(tmp_path, [128001, 128008, 128009])
    with pytest.raises(ValueError, match="EOS not bound to special-token metadata"):
        v1.analyse_run(tmp_path, verify_tokenizer=False)
    # Reaching the absent config proves that the same source-bound EOS list was
    # accepted despite 128001/128008 not being registered tokenizer specials.
    with pytest.raises(FileNotFoundError, match="intentionally_absent_config"):
        v2.analyse_run(tmp_path, verify_tokenizer=False)


def test_changed_model_eos_and_missing_original_binding_still_fail(tmp_path):
    protocol = staged_run(tmp_path, [128001, 128009])
    with pytest.raises(ValueError, match="EOS differs from frozen source backend"):
        v2.analyse_run(tmp_path, verify_tokenizer=False)
    protocol["policy_sha256"][v2.FROZEN_ORIGINAL_NAME] = "0" * 64
    with pytest.raises(ValueError, match="Executed policy does not bind"):
        v2._correction_provenance(protocol)


def test_version_correction_binds_original_and_records_its_own_bytes():
    provenance = v2._correction_provenance(
        {"policy_sha256": {v2.FROZEN_ORIGINAL_NAME: v2.FROZEN_ORIGINAL_SHA256}}
    )
    assert file_sha256(v1.__file__) == provenance["frozen_original_sha256"]
    assert provenance["corrected_module_sha256"] == file_sha256(v2.__file__)
    assert provenance["corrected_cli_sha256"] == file_sha256(
        v2.ROOT / "scripts/analyse_future_goal_controls_v2.py"
    )
    assert provenance["new_model_calls"] == provenance["new_fits"] == 0


def test_neutral_failure_and_all_metrics_are_identical_to_frozen_v1():
    rows = deepcopy(
        [r for r in v1.reviewed_bank()["planned_conditions"] if r["condition"] == "neutral"]
    )
    for index, row in enumerate(rows):
        prediction = row["gold"] if index >= 7 else 1 - row["gold"]
        margin = 1.0 if prediction else -1.0
        row["probe"] = {
            "head_id": v1.HEAD,
            "margin": margin,
            "probability": float(np.exp(-np.logaddexp(0, -margin))),
            "prediction": prediction,
        }
        row["generation"] = {"text": "YES", "finish_reason": "eos"}
    original, corrected = v1.analyse_rows(rows), v2.analyse_rows(rows)
    assert original == corrected
    assert corrected["status"] == "qualification_failed"
    assert corrected["qualification"]["overall"]["probe_correct"] == 25
    assert corrected["qualification"]["overall"]["answer_correct"] == 16
    assert (
        not {"primary", "secondary", "per_scene", "raw_scores", "metrics", "specificity"}
        & corrected.keys()
    )


def test_exact_ast_change_audit_preserves_every_other_function_and_check():
    original = ast.parse(v1.ROOT.joinpath(v2.FROZEN_ORIGINAL_NAME).read_text(encoding="utf-8"))
    corrected = ast.parse(
        v2.ROOT.joinpath("src/latent_probes/future_goal_controls_analysis_v2.py").read_text(
            encoding="utf-8"
        )
    )
    old_functions = {n.name: n for n in original.body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name: n for n in corrected.body if isinstance(n, ast.FunctionDef)}
    assert set(new_functions) == set(old_functions) | {"_correction_provenance"}
    for name, old in old_functions.items():
        new = deepcopy(new_functions[name])
        old = deepcopy(old)
        if name == "analyse_run":
            old.body = [
                n
                for n in old.body
                if ast.unparse(n)
                != "_require(set(eos) <= set(special), 'EOS not bound to special-token metadata')"
            ]
            new.body = [
                n
                for n in new.body
                if ast.unparse(n)
                not in {
                    "correction = _correction_provenance(protocol)",
                    "result['analysis_correction'] = correction",
                }
            ]
        assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False), (
            name
        )
    old_other = [n for n in original.body[1:] if not isinstance(n, ast.FunctionDef)]
    new_other = [
        n
        for n in corrected.body[1:]
        if not isinstance(n, ast.FunctionDef)
        and not (
            isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in {"FROZEN_ORIGINAL_NAME", "FROZEN_ORIGINAL_SHA256"}
        )
    ]
    assert ast.dump(
        ast.Module(body=old_other, type_ignores=[]), include_attributes=False
    ) == ast.dump(ast.Module(body=new_other, type_ignores=[]), include_attributes=False)
    original_cli = ast.parse((v1.ROOT / "scripts/analyse_future_goal_controls.py").read_text())
    corrected_cli = ast.parse((v1.ROOT / "scripts/analyse_future_goal_controls_v2.py").read_text())
    for node in corrected_cli.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "latent_probes.future_goal_controls_analysis_v2"
        ):
            node.module = "latent_probes.future_goal_controls_analysis"
    original_cli.body.pop(0)
    corrected_cli.body.pop(0)
    assert ast.dump(original_cli, include_attributes=False) == ast.dump(
        corrected_cli, include_attributes=False
    )
