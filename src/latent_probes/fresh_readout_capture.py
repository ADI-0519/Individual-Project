"""Fresh-study exact-span panel capture; preparation never loads model weights.

This module composes the frozen HF backend. A cohort row gets one forward, no
answer tokens, twelve float32 block/position vectors, and exact YES/NO logits.
Execution is explicitly bound and keeps failed/unknown attempts without retries.
"""

from __future__ import annotations

import json
import platform
import re
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .backends import HFBackend
from .config import ModelSettings
from .crossed_belief_validation import summarize_logits
from .main_study_run import MODELS, _forward_contract, _token_map, snapshot_files
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
BLOCKS = {"llama": (8, 16, 24, 30), "qwen": (7, 14, 21, 26)}
POSITIONS = ("account_last", "query_location_last", "prompt_last")
COUNTS = {
    "source_fit": 512,
    "source_dev": 512,
    "correction_fit": 512,
    "correction_dev": 512,
    "test": 4992,
}
PARTITIONS = {**{key: (key,) for key in COUNTS}, "source": ("source_fit", "source_dev")}
PARTITIONS["training"] = ("source_fit", "source_dev", "correction_fit", "correction_dev")
POLICY_FILES = (
    "src/latent_probes/fresh_readout_analysis.py",
    "scripts/analyze_fresh_readout.py",
    "src/latent_probes/fresh_readout_capture.py",
    "src/latent_probes/fresh_readout_bank.py",
    "src/latent_probes/fresh_readout_review.py",
    "src/latent_probes/fresh_readout_fit.py",
    "scripts/run_fresh_readout_capture.py",
    "scripts/fit_fresh_readout.py",
    "src/latent_probes/backends.py",
    "src/latent_probes/config.py",
    "src/latent_probes/storage.py",
    "src/latent_probes/main_study_run.py",
    "src/latent_probes/main_study_bank.py",
    "src/latent_probes/main_study_design.py",
    "src/latent_probes/crossed_belief_validation.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def panel_sites(model_key):
    return [f"layer_{block}_{position}" for block in BLOCKS[model_key] for position in POSITIONS]


def _native_output(output, *, test_fixture):
    path = Path(output).resolve()
    if not test_fixture:
        require(
            platform.system() == "Linux"
            and not str(path).startswith("/mnt/")
            and "onedrive" not in str(path).lower(),
            "Real capture requires native Linux storage outside Windows mounts and OneDrive",
        )
    require(not path.is_relative_to(ROOT), "Capture outputs must be outside the source tree")
    return path


def tokenizer_binding(tokenizer):
    require(getattr(tokenizer, "is_fast", False), "Exact-span capture requires a fast tokenizer")
    require(bool(tokenizer.chat_template), "Require a chat template")
    return {
        "class": type(tokenizer).__name__,
        "chat_template_sha256": canonical_hash(tokenizer.chat_template),
        "backend_sha256": canonical_hash(tokenizer.backend_tokenizer.to_str()),
        "vocab_sha256": canonical_hash(tokenizer.get_vocab()),
        "special_token_ids": {"eos": tokenizer.eos_token_id, "pad": tokenizer.pad_token_id},
        "all_special_ids": list(tokenizer.all_special_ids),
        "answer_token_ids": _token_map(tokenizer),
    }


def _flat_ids(ids):
    require(
        isinstance(ids, list) and ids and all(type(i) is int and i >= 0 for i in ids),
        "Tokenizer must return a flat nonempty integer prefix",
    )
    return ids


def prepare_row(tokenizer, raw, *, max_tokens=2048):
    """Map end-exclusive user character spans to the exact complete chat prefix.

    Content is located exactly once, rather than searching for a location word.
    Token crossings fail closed. A renderer may explicitly declare
    ``span_boundary_policy: {"query_location": "leading_whitespace"}`` to permit
    a token's left overlap consisting solely of whitespace. Right crossings
    always fail: the selected account token can never extend past its span.
    """
    row = dict(raw)
    row_id = row.get("row_id", row.get("record_id", row.get("case_id")))
    require(
        isinstance(row_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", row_id),
        "Row IDs must be safe nonempty journal basenames",
    )
    messages = row["messages"]
    users = [message for message in messages if message.get("role") == "user"]
    require(len(users) == 1, "Require exactly one renderer-defined user message")
    content = users[0]["content"]
    require(isinstance(content, str) and content, "User content must be nonempty text")
    spans = row.get("user_content_char_spans", row.get("user_text_char_spans"))
    require(
        isinstance(spans, dict) and set(spans) == {"account", "query_location"},
        "Need explicit account and query_location user-text character spans",
    )
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    require(isinstance(rendered, str), "Chat template must render text")
    user_start = rendered.find(content)
    require(
        user_start >= 0 and rendered.find(content, user_start + 1) < 0,
        "User content must occur exactly once in the rendered chat template",
    )
    ids = _flat_ids(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    )
    encoded = tokenizer(
        rendered, add_special_tokens=False, return_offsets_mapping=True, truncation=False
    )
    require(
        _flat_ids(encoded["input_ids"]) == ids,
        "Chat-template token IDs differ from full rendered-text tokenization",
    )
    require(len(ids) <= max_tokens, "Prompt exceeds frozen input/context limit; no truncation")
    offsets = encoded["offset_mapping"]
    require(len(offsets) == len(ids), "Offset/token count differs")
    require(
        all(
            len(pair) == 2
            and all(type(x) is int for x in pair)
            and 0 <= pair[0] <= pair[1] <= len(rendered)
            for pair in offsets
        ),
        "Invalid full-template character offsets",
    )
    positions, identities = {}, {}
    absolute = {}
    special_ids = set(tokenizer.all_special_ids)
    policies = row.get("span_boundary_policy", {})
    require(isinstance(policies, dict) and set(policies) <= set(spans), "Unknown span policy")
    for name, span in spans.items():
        require(
            isinstance(span, (list, tuple))
            and len(span) == 2
            and all(type(x) is int for x in span)
            and 0 <= span[0] < span[1] <= len(content),
            "Invalid renderer character span: " + name,
        )
        start, end = user_start + span[0], user_start + span[1]
        absolute[name] = [start, end]
        policy = policies.get(name, "strict")
        require(policy in {"strict", "leading_whitespace"}, "Unknown token-boundary policy")
        candidates = []
        for index, (left, right) in enumerate(offsets):
            if left == right or right <= start or left >= end:
                continue
            allowed_left = left >= start or (
                policy == "leading_whitespace" and rendered[left:start].isspace()
            )
            require(
                allowed_left and right <= end,
                f"Token crosses {name} character boundary; declare proper alignment or repair renderer",
            )
            require(ids[index] not in special_ids, "Span includes a special token")
            candidates.append(index)
        require(bool(candidates), "Span contains no complete token: " + name)
        selected = candidates[-1]
        # All non-whitespace span content must be represented, including its end.
        require(not rendered[offsets[selected][1] : end].strip(), "Unmapped end of span: " + name)
        position = name + "_last"
        positions[position] = selected
        identities[position] = {
            "token_index": selected,
            "token_id": ids[selected],
            "rendered_char_offset": list(offsets[selected]),
            "span_boundary_policy": policy,
        }
    require(
        absolute["account"][1] <= absolute["query_location"][0],
        "Account must precede the explicit query location",
    )
    positions["prompt_last"] = len(ids) - 1
    identities["prompt_last"] = {
        "token_index": len(ids) - 1,
        "token_id": ids[-1],
        "rendered_char_offset": list(offsets[-1]),
    }
    require(
        positions["account_last"] < positions["query_location_last"] <= positions["prompt_last"],
        "Selected tokens are not causally ordered",
    )
    return {
        **row,
        "row_id": row_id,
        "messages_sha256": canonical_hash(messages),
        "rendered_chat_sha256": canonical_hash(rendered),
        "user_content_rendered_char_span": [user_start, user_start + len(content)],
        "rendered_char_spans": absolute,
        "prompt_token_ids": ids,
        "prompt_token_ids_sha256": canonical_hash(ids),
        "token_positions": positions,
        "selected_tokens": identities,
    }


def prepare(
    bank_path,
    output,
    *,
    model_key="llama",
    rows=None,
    engineering_row=None,
    tokenizer=None,
    model_config=None,
    forward=None,
    test_fixture=False,
    source_bindings=None,
):
    """Bind the full inventory using pinned tokenizer/config, without model weights.

    JSON capture-input files containing ``rows`` work without a bank adapter.
    Otherwise the fresh bank's ``load_bank`` and ``iter_rows`` are used.
    """
    require(model_key in MODELS and type(test_fixture) is bool, "Invalid model/fixture choice")
    output = _native_output(output, test_fixture=test_fixture)
    require(not output.exists(), "Use a new preparation directory; no overwrite")
    bank_path = Path(bank_path).resolve()
    payload = read_json(bank_path)
    require(test_fixture or rows is None, "Native preparation must use the validated fresh bank")
    if rows is None:
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            require(test_fixture, "Generic capture-input rows are for engineering fixtures only")
            rows = payload["rows"]
            engineering_row = engineering_row or payload.get("engineering_row")
        else:
            from .fresh_readout_bank import iter_rows, load_bank

            rows = iter_rows(load_bank(bank_path))
    model = MODELS[model_key]
    if tokenizer is None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        common = {
            "revision": model["revision"],
            "local_files_only": True,
            "trust_remote_code": False,
        }
        model_config = AutoConfig.from_pretrained(model["model_id"], **common)
        tokenizer = AutoTokenizer.from_pretrained(model["model_id"], **common)
        forward = AutoModelForCausalLM._model_mapping[type(model_config)].forward
    require(
        model_config is not None
        and model_config._commit_hash == model["revision"]
        and model_config.model_type == model["model_type"],
        "Pinned model config differs",
    )
    blocks = model_config.num_hidden_layers
    require(
        type(blocks) is int and all(0 <= b < blocks - 1 for b in BLOCKS[model_key]),
        "Capture panel must precede final decoder normalization",
    )
    settings = ModelSettings(
        model_id=model["model_id"],
        revision=model["revision"],
        dtype="bfloat16",
        quantization="4bit",
        device="cuda",
        capture_layers=list(BLOCKS[model_key]),
        max_input_tokens=2048,
        local_files_only=True,
    )
    limit = min(settings.max_input_tokens, model_config.max_position_embeddings)
    binding = tokenizer_binding(tokenizer)
    planned = [prepare_row(tokenizer, row, max_tokens=limit) for row in rows]
    require(
        planned and len({r["row_id"] for r in planned}) == len(planned), "Duplicate/empty row IDs"
    )
    require(len({r["messages_sha256"] for r in planned}) == len(planned), "Duplicate exact prompts")
    counts = Counter(r.get("partition") for r in planned)
    require(set(counts) <= set(COUNTS), "Unknown row partition")
    require(test_fixture or counts == COUNTS, "Real preparation requires all 7040 fixed prefixes")
    engineering = None
    if engineering_row is not None:
        engineering = prepare_row(tokenizer, engineering_row, max_tokens=limit)
        require(
            engineering["row_id"] not in {r["row_id"] for r in planned}
            and engineering["messages_sha256"] not in {r["messages_sha256"] for r in planned},
            "Engineering example must be excluded from every cohort",
        )
    sources = {str((ROOT / name).resolve()): file_sha256(ROOT / name) for name in POLICY_FILES}
    # Source narratives and authored packets are independent frozen inputs, even
    # though the fully expanded messages are also preserved in the protocol.
    for binding in (
        [payload["source_bank"]] + payload.get("authored_inputs", [])
        if "source_bank" in payload
        else []
    ):
        relative = Path(binding["path"])
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "Nested bank input escapes the source tree",
        )
        sources[str((ROOT / relative).resolve())] = binding["sha256"]
    sources.update(source_bindings or {})
    snapshot_files(sources)
    protocol = {
        "schema": "fresh-readout-capture-protocol-v1",
        "created_at": utc_now(),
        "test_fixture": test_fixture,
        "research_status": "agent_audited_exploratory",
        "bank_path": str(bank_path),
        "bank_file_sha256": file_sha256(bank_path),
        "bank_sha256": canonical_hash(payload),
        "source_sha256": sources,
        "model_key": model_key,
        "model_settings": asdict(settings),
        "model_type": model["model_type"],
        "decoder_blocks": blocks,
        "hidden_size": model_config.hidden_size,
        "vocab_size": model_config.vocab_size,
        "max_position_embeddings": model_config.max_position_embeddings,
        "forward_contract": _forward_contract(forward),
        "tokenizer": binding,
        "sites": panel_sites(model_key),
        "position_order": list(POSITIONS),
        "original_site": f"layer_{16 if model_key == 'llama' else 14}_prompt_last",
        "planned_rows": planned,
        "engineering_row": engineering,
        "environment": environment_info(),
        "preflight": {
            "model_weights_loaded": False,
            "model_forwards": 0,
            "rows": len(planned),
            "partitions": dict(counts),
            "minimum_tokens": min(len(r["prompt_token_ids"]) for r in planned),
            "maximum_tokens": max(len(r["prompt_token_ids"]) for r in planned),
        },
        "capture": {
            "same_forward_all_sites_and_logits": True,
            "answer_tokens_supplied": 0,
            "feature_dtype": "float32",
            "block_output_before_final_norm": True,
            "truncation": False,
            "generation": False,
        },
    }
    output.mkdir(parents=True)
    atomic_json(output / "protocol.json", protocol)
    atomic_json(
        output / "preparation.json",
        {
            "status": "prepared_no_weights",
            "protocol_sha256": file_sha256(output / "protocol.json"),
            "test_fixture": test_fixture,
        },
    )
    return protocol


def verify_protocol(protocol):
    require(file_sha256(protocol["bank_path"]) == protocol["bank_file_sha256"], "Bank changed")
    snapshot_files(protocol["source_sha256"])
    contract = protocol["forward_contract"]
    require(
        file_sha256(contract["source_path"]) == contract["source_sha256"],
        "Installed model-forward source changed",
    )


def _authorization(protocol, protocol_sha256, bindings_path, partition):
    require(bindings_path is not None, "Execution requires frozen execution bindings")
    path = Path(bindings_path).resolve()
    freeze = read_json(path)
    require(
        freeze.get("schema") == "fresh-readout-execution-bindings-v1"
        and freeze.get("status") == "frozen"
        and freeze.get("protocol_sha256") == protocol_sha256,
        "Execution bindings are not frozen for this exact prepared protocol",
    )
    sources = freeze.get("source_sha256", {})
    require(
        all(sources.get(p) == d for p, d in protocol["source_sha256"].items()),
        "Execution bindings must cover every capture source",
    )
    snapshot_files(sources)
    reviews = freeze.get("review_artifact_sha256")
    require(isinstance(reviews, dict) and reviews, "Need bound agent semantic-review artifacts")
    snapshot_files(reviews)
    if not protocol["test_fixture"]:
        _validate_semantic_review(protocol, reviews)
    fits = freeze.get("fitted_artifact_sha256", {})
    if partition == "test":
        require(
            isinstance(fits, dict) and len(fits) == 8,
            "Test capture requires frozen fitted settings and safe head arrays for both models",
        )
        snapshot_files(fits)
        _validate_fits(protocol, protocol_sha256, freeze, fits)
    return {
        "bindings_path": str(path),
        "bindings_sha256": file_sha256(path),
        "sha256": {**sources, **reviews, **fits, str(path): file_sha256(path)},
    }


def _validate_semantic_review(protocol, reviews):
    from .fresh_readout_bank import load_bank, review_packet
    from .fresh_readout_review import validate_review_receipt

    receipts = [(p, read_json(p)) for p in reviews if Path(p).suffix == ".json"]
    receipts = [
        (p, r) for p, r in receipts if r.get("schema") == "fresh-readout-agent-semantic-review-v1"
    ]
    require(len(receipts) == 1, "Need one genuine agent semantic-review receipt")
    receipt_path, receipt = receipts[0]
    validate_review_receipt(receipt_path, protocol["bank_path"])
    require(
        receipt.get("status") == "passed"
        and receipt.get("reviewer_type") == "agent"
        and receipt.get("human_review_complete") is False
        and receipt.get("research_status") == "agent_audited_exploratory"
        and receipt.get("all_answers_match") is True
        and receipt.get("context_issues") == []
        and receipt.get("bank_file_sha256") == protocol["bank_file_sha256"]
        and receipt.get("bank_sha256") == protocol["bank_sha256"],
        "Semantic-review receipt identity/status differs",
    )
    sources = receipt.get("source_files", {})
    require(set(sources) == {"packet", "answers", "key"}, "Incomplete review source bindings")
    loaded = {}
    for name, binding in sources.items():
        relative = Path(binding["path"])
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "Review source path escapes the source tree",
        )
        path = (ROOT / relative).resolve()
        require(
            reviews.get(str(path)) == binding["sha256"] and file_sha256(path) == binding["sha256"],
            "Review packet/answers/key must be bound by exact source bytes",
        )
        loaded[name] = read_json(path)
    packet, answers, key = loaded["packet"], loaded["answers"], loaded["key"]
    require(
        packet.get("schema") == "fresh-readout-blind-review-v1"
        and answers.get("schema") == "fresh-readout-blind-answers-v1"
        and key.get("schema") == "fresh-readout-review-key-v1"
        and packet.get("bank_file_sha256") == protocol["bank_file_sha256"]
        and packet.get("bank_sha256") == protocol["bank_sha256"]
        and answers.get("reviewer_type") == "agent"
        and answers.get("human_review_complete") is False,
        "Review source schema/bank/agent provenance differs",
    )
    bank = load_bank(protocol["bank_path"])
    expected_packet, expected_key = review_packet(bank)
    require(
        packet["base_contexts"] == expected_packet["base_contexts"]
        and packet["prompts"] == expected_packet["prompts"]
        and key["records"] == expected_key["records"],
        "Review must cover the entire declared blinded sample and all normalized paths",
    )
    groups = {g["group_id"] for g in bank["groups"]}
    context_reviews = answers["context_reviews"]
    require(
        len(groups) == len(context_reviews) == len(packet["base_contexts"]) == 48
        and {r["group_id"] for r in context_reviews} == groups
        and {r["group_id"] for r in packet["base_contexts"]} == groups
        and all(r.get("acceptable") is True for r in context_reviews),
        "All 48 base contexts require recorded acceptable agent review",
    )
    prompts, answer_rows, key_rows = packet["prompts"], answers["answers"], key["records"]
    ids = [r["review_id"] for r in prompts]
    coverage = receipt["coverage"]
    require(
        ids
        and len(set(ids)) == len(ids)
        and len(ids) == len(answer_rows) == len(key_rows) == coverage["prompts"]
        and coverage["base_contexts"] == 48
        and set(coverage["review_ids"]) == set(ids)
        and len(coverage["review_ids"]) == len(ids)
        and {r["review_id"] for r in answer_rows} == set(ids)
        and {r["review_id"] for r in key_rows} == set(ids),
        "Review coverage/opaque IDs differ",
    )
    answer_map = {r["review_id"]: r for r in answer_rows}
    key_map = {r["review_id"]: r for r in key_rows}
    planned = {r["row_id"]: r for r in protocol["planned_rows"]}
    for prompt in prompts:
        target = key_map[prompt["review_id"]]
        answer = answer_map[prompt["review_id"]]["answer"]
        require(
            set(prompt) == {"review_id", "messages"} and target["record_id"] in planned,
            "Review packet must be blinded and contain declared cohort records",
        )
        row = planned[target["record_id"]]
        require(
            prompt["messages"] == row["messages"]
            and target["gold"] == row["gold"]
            and target["condition"] == row["condition"]
            and target["renderer"] == row["renderer"]
            and answer in {"YES", "NO"}
            and (answer == "YES") == bool(row["gold"]),
            "A recorded agent interpretation disagrees with the exact bank record",
        )


def _load_fit_bundle(directory, *, allow_test_fixture):
    from .fresh_readout_fit import validate_fit_bundle

    return validate_fit_bundle(directory, allow_test_fixture=allow_test_fixture)


def _validate_fits(protocol, digest, freeze, fits):
    settings_paths = [Path(p) for p in fits if Path(p).name == "settings.json"]
    require(
        len(settings_paths) == 2
        and len({p.parent for p in settings_paths}) == 2
        and set(fits)
        == {
            str(p.parent / name)
            for p in settings_paths
            for name in ("settings.json", "heads.npz", "manifest.json", "candidates.json")
        },
        "Test capture requires all four exact frozen fit files for both models",
    )
    model_protocols = freeze.get("model_protocol_sha256", {})
    require(
        set(model_protocols) == set(MODELS)
        and model_protocols.get(protocol["model_key"]) == digest,
        "Both model capture protocols must be frozen before test capture",
    )
    observed = set()
    for path in settings_paths:
        settings, _arrays = _load_fit_bundle(
            path.parent, allow_test_fixture=protocol["test_fixture"]
        )
        require(
            settings.get("schema") == "fresh-readout-fit-v2"
            and settings.get("status") == "frozen"
            and settings.get("test_fixture") == protocol["test_fixture"]
            and settings.get("research_status") == "agent_audited_exploratory"
            and settings.get("human_review_complete") is False,
            "Fit schema/status/fixture/research provenance differs",
        )
        provenance = settings["capture_provenance"]
        key = provenance.get("model_key")
        require(key in MODELS and key not in observed, "Need one genuine fitted bundle per model")
        observed.add(key)
        expected_settings = {
            **protocol["model_settings"],
            "model_id": MODELS[key]["model_id"],
            "revision": MODELS[key]["revision"],
            "capture_layers": list(BLOCKS[key]),
        }
        require(
            provenance.get("bank_file_sha256") == protocol["bank_file_sha256"]
            and provenance.get("model_settings") == expected_settings
            and provenance.get("sites") == panel_sites(key)
            and provenance.get("protocol_sha256") == model_protocols[key],
            "Fitted bundle bank/model/site/protocol identity differs",
        )
        for partition in PARTITIONS["training"]:
            expected_ids = [
                r["row_id"] for r in protocol["planned_rows"] if r["partition"] == partition
            ]
            require(
                settings.get(partition + "_ids") == expected_ids,
                "Fitted bundle ordered training row IDs differ: " + partition,
            )
    require(observed == set(MODELS), "Both model fitted bundles must be frozen")


def _summarize_logits(logits, tokens):
    summary = summarize_logits(logits, tokens)
    values = logits.astype(np.float64)
    shifted = values - values.max()
    normalizer = float(np.log(np.exp(shifted).sum()))
    probabilities = np.exp(shifted - normalizer)
    return {
        **summary,
        "full_vocab_all_finite": True,
        "argmax_tie_count": int(np.count_nonzero(values == values.max())),
        "full_vocab_entropy": float(normalizer - np.dot(probabilities, shifted)),
    }


def forward_measurement(backend, planned, tokens, *, check_alignment=False, raw_sink=None):
    """One forward; hooks copy only the three requested positions per block.

    ``check_alignment`` is solely for the excluded engineering example. Cohort
    forwards never materialize all HF hidden states.
    """
    ids, positions = planned["prompt_token_ids"], planned["token_positions"]
    require(0 < len(ids) <= backend.settings.max_input_tokens, "Invalid or oversized prefix")
    require(len(ids) <= backend.model.config.max_position_embeddings, "Context limit exceeded")
    require(
        set(positions) == set(POSITIONS)
        and all(type(p) is int and 0 <= p < len(ids) for p in positions.values()),
        "Invalid selected token positions",
    )
    captured, observations, hooks = {}, Counter(), []
    started = time.perf_counter()
    try:
        for block in backend.settings.capture_layers:

            def capture(_module, _inputs, output, block_index=block):
                hidden = output[0] if isinstance(output, tuple) else output
                observations[block_index] += 1
                for name in POSITIONS:
                    captured[f"layer_{block_index}_{name}"] = (
                        hidden[0, positions[name], :].detach().float().cpu().numpy().copy()
                    )

            hooks.append(backend.base.layers[block].register_forward_hook(capture))
        sequence = backend.torch.tensor([ids], device=backend.input_device)
        with backend.torch.inference_mode():
            result = backend.model(
                input_ids=sequence,
                attention_mask=backend.torch.ones_like(sequence),
                logits_to_keep=1,
                use_cache=False,
                return_dict=True,
                output_hidden_states=check_alignment,
            )
    finally:
        for hook in hooks:
            hook.remove()
    sites = [
        f"layer_{block}_{name}" for block in backend.settings.capture_layers for name in POSITIONS
    ]
    require(
        all(observations[b] == 1 for b in backend.settings.capture_layers)
        and set(captured) == set(sites),
        "Need exactly one hook observation at every site",
    )
    features = np.stack([captured[site] for site in sites])
    values = result.logits.detach().float().cpu().numpy().copy()
    if raw_sink is not None:
        raw_sink(features, values)
    require(
        features.shape == (12, backend.metadata["hidden_size"])
        and features.dtype == np.float32
        and np.isfinite(features).all(),
        "Invalid panel features",
    )
    require(values.shape == (1, 1, backend.model.config.vocab_size), "Unexpected vocabulary shape")
    if check_alignment:
        for block in backend.settings.capture_layers:
            for name in POSITIONS:
                reference = (
                    result.hidden_states[block + 1][0, positions[name], :]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                require(
                    np.array_equal(captured[f"layer_{block}_{name}"], reference),
                    "Hook/HF hidden-state alignment differs",
                )
    logits = values[0, 0].copy()
    return features, logits, _summarize_logits(logits, tokens), time.perf_counter() - started


def _validate_backend(backend, protocol):
    metadata = backend.metadata
    require(
        bool(metadata["is_test_fixture"]) == protocol["test_fixture"], "Fixture identity differs"
    )
    for key in ("model_id", "dtype", "quantization", "capture_layers"):
        require(metadata[key] == protocol["model_settings"][key], "Backend setting differs: " + key)
    require(
        metadata["resolved_revision"] == protocol["model_settings"]["revision"]
        and metadata["decoder_blocks"] == protocol["decoder_blocks"]
        and metadata["hidden_size"] == protocol["hidden_size"]
        and _forward_contract(backend.model.forward) == protocol["forward_contract"]
        and tokenizer_binding(backend.tokenizer) == protocol["tokenizer"],
        "Pinned backend/tokenizer/forward identity differs",
    )
    if not protocol["test_fixture"]:
        config = backend.model.config
        require(
            config.model_type == protocol["model_type"]
            and config.num_hidden_layers == protocol["decoder_blocks"]
            and config.hidden_size == protocol["hidden_size"]
            and config.vocab_size == protocol["vocab_size"]
            and config.max_position_embeddings == protocol["max_position_embeddings"]
            and config._commit_hash == protocol["model_settings"]["revision"],
            "Actual pinned model configuration differs",
        )
    backend.model.eval()
    require(not backend.model.training, "Model must be in evaluation mode")


def _measure_row(backend, protocol, digest, planned, directory, *, engineering=False):
    require(
        prepare_row(
            backend.tokenizer,
            planned,
            max_tokens=min(backend.settings.max_input_tokens, protocol["max_position_embeddings"]),
        )
        == planned,
        "Exact prepared prefix or selected token identities changed",
    )
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(
        directory / "started.json",
        {
            "row_id": planned["row_id"],
            "protocol_sha256": digest,
            "started_at": utc_now(),
            "engineering": engineering,
        },
    )
    try:
        token_map = protocol["tokenizer"]["answer_token_ids"]
        candidate_ids = [token_map["YES"], token_map["NO"]]

        def preserve_raw(features, values):
            saved = {"features": features}
            if engineering:
                saved["full_forward_logits"] = values
            else:
                saved["output_candidate_logits"] = values[0, 0, candidate_ids]
            atomic_npz(directory / "raw_forward.npz", **saved)

        features, logits, summary, elapsed = forward_measurement(
            backend,
            planned,
            protocol["tokenizer"]["answer_token_ids"],
            check_alignment=engineering,
            raw_sink=preserve_raw,
        )
        atomic_npz(
            directory / "arrays.npz",
            features=features,
            output_candidate_logits=logits[candidate_ids],
            output_candidate_token_ids=np.asarray(candidate_ids, dtype=np.int64),
            output_summary_sha256=np.asarray(canonical_hash(summary)),
            site_token_indices=np.asarray(
                [
                    planned["token_positions"][p]
                    for _ in protocol["model_settings"]["capture_layers"]
                    for p in POSITIONS
                ]
            ),
            **({"next_token_logits": logits} if engineering else {}),
        )
        complete = {
            **planned,
            "status": "complete",
            "protocol_sha256": digest,
            "test_fixture": protocol["test_fixture"],
            "arrays_sha256": file_sha256(directory / "arrays.npz"),
            "raw_forward_sha256": file_sha256(directory / "raw_forward.npz"),
            "output_logits": summary,
            "output_candidate_order": ["YES", "NO"],
            "elapsed_seconds": elapsed,
            "capture": {
                "sites": protocol["sites"],
                "same_forward_as_logits": True,
                "answer_tokens_supplied": 0,
                "hook_alignment_checked": engineering,
                "output_hidden_states": engineering,
            },
        }
        atomic_json(directory / "complete.json", complete)
        return complete
    except BaseException as error:
        atomic_json(
            directory / "failure.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at": utc_now(),
                "preserved_files_sha256": {
                    p.name: file_sha256(p) for p in directory.iterdir() if p.is_file()
                },
            },
        )
        raise


def _completed(directory, planned, protocol, digest):
    if not directory.exists():
        return None
    require(
        (directory / "started.json").is_file()
        and (directory / "complete.json").is_file()
        and not (directory / "failure.json").exists(),
        "Incomplete/failed attempted row blocks resumption; preserve it and stop",
    )
    complete = read_json(directory / "complete.json")
    started = read_json(directory / "started.json")
    require(
        all(complete.get(k) == v for k, v in planned.items())
        and complete.get("status") == "complete"
        and complete["protocol_sha256"] == digest
        and started["protocol_sha256"] == digest
        and started["row_id"] == planned["row_id"],
        "Completed row binding changed",
    )
    for name in ("arrays", "raw_forward"):
        require(
            file_sha256(directory / f"{name}.npz") == complete[f"{name}_sha256"],
            "Completed row arrays changed",
        )
    with np.load(directory / "arrays.npz", allow_pickle=False) as data:
        features, candidates = data["features"], data["output_candidate_logits"]
        require(
            features.shape == (12, protocol["hidden_size"])
            and features.dtype == np.float32
            and np.isfinite(features).all(),
            "Invalid saved panel features",
        )
        summary = complete["output_logits"]
        require(
            candidates.shape == (2,)
            and candidates.dtype == np.float32
            and np.isfinite(candidates).all()
            and summary["vocab_size"] == protocol["vocab_size"]
            and [summary["yes_logit"], summary["no_logit"]] == candidates.tolist()
            and summary["yes_minus_no"] == float(candidates[0]) - float(candidates[1])
            and data["output_summary_sha256"].item() == canonical_hash(summary),
            "Saved output measurement changed",
        )
        require(
            complete["output_candidate_order"] == ["YES", "NO"]
            and data["output_candidate_token_ids"].tolist()
            == [protocol["tokenizer"]["answer_token_ids"][word] for word in ("YES", "NO")],
            "Saved YES/NO token identities changed",
        )
        require(
            data["site_token_indices"].tolist()
            == [
                planned["token_positions"][p]
                for _ in protocol["model_settings"]["capture_layers"]
                for p in POSITIONS
            ],
            "Saved token indices changed",
        )
        if "next_token_logits" in data:
            require(
                _summarize_logits(
                    data["next_token_logits"], protocol["tokenizer"]["answer_token_ids"]
                )
                == summary,
                "Engineering vocabulary summary changed",
            )
    return complete


def _execute_locked(
    output, *, partition, bindings_path, resume=False, backend_factory=HFBackend, progress=print
):
    """Capture a fixed partition; only an explicitly completed row may be resumed past."""
    output = Path(output).resolve()
    protocol = read_json(output / "protocol.json")
    digest = file_sha256(output / "protocol.json")
    require(partition in PARTITIONS, "Unknown execution partition")
    require(
        read_json(output / "preparation.json")["protocol_sha256"] == digest,
        "Prepared protocol changed",
    )
    _native_output(output, test_fixture=protocol["test_fixture"])
    verify_protocol(protocol)
    authorization = _authorization(protocol, digest, bindings_path, partition)
    environment = environment_info()
    require(
        all(
            environment[k] == protocol["environment"][k] for k in ("python", "platform", "packages")
        ),
        "Prepared Python/platform/package environment changed",
    )
    if protocol["test_fixture"]:
        require(backend_factory is not HFBackend, "Fixture protocol cannot load real model weights")
    require(protocol["engineering_row"] is not None, "Need a separate excluded engineering example")
    selected = [r for r in protocol["planned_rows"] if r["partition"] in PARTITIONS[partition]]
    require(bool(selected), "No rows in requested partition")
    # Do not permit source/source_fit aliases to run the same row twice.
    for other in output.iterdir():
        if (
            other.name in PARTITIONS
            and other.name != partition
            and (other / "manifest.json").exists()
        ):
            other_manifest = read_json(other / "manifest.json")
            require(
                not set(PARTITIONS[other.name]) & set(PARTITIONS[partition]),
                "Overlapping partition already attempted: " + other_manifest["partition"],
            )
    target = output / partition
    if target.exists():
        require(resume, "Existing partition requires explicit resume")
        manifest = read_json(target / "manifest.json")
        require(manifest["status"] != "complete", "Completed capture cannot be rerun")
        require(
            manifest["protocol_sha256"] == digest and manifest["authorization"] == authorization,
            "Resume requires exact prior protocol and authorization",
        )
    else:
        require(not resume, "New partition does not require resume")
        target.mkdir()
        manifest = {
            "schema": "fresh-readout-capture-manifest-v1",
            "partition": partition,
            "protocol_sha256": digest,
            "authorization": authorization,
            "completed_rows": 0,
            "test_fixture": protocol["test_fixture"],
        }
    journal = target / "row_journal"
    expected = {r["row_id"] for r in selected}
    require(
        not journal.exists() or all(p.is_dir() and p.name in expected for p in journal.iterdir()),
        "Unexpected row journal entry",
    )
    completed = [
        row
        for planned in selected
        if (row := _completed(journal / planned["row_id"], planned, protocol, digest)) is not None
    ]
    # The engineering forward has its own journal and can never be replaced after failure.
    engineering = protocol["engineering_row"]
    engineering_done = _completed(output / "engineering", engineering, protocol, digest)
    if engineering_done is not None:
        require(engineering_done["capture"]["hook_alignment_checked"], "Engineering check missing")
    manifest.update(status="running", environment=environment, completed_rows=len(completed))
    atomic_json(target / "manifest.json", manifest)
    attempts = target / "attempts"
    attempts.mkdir(exist_ok=True)
    with (attempts / f"{len(list(attempts.iterdir())):04d}.json").open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(
            {"started_at": utc_now(), "explicit_resume": resume, "protocol_sha256": digest}, stream
        )
    backend = None
    try:
        pending = [r for r in selected if r["row_id"] not in {c["row_id"] for c in completed}]
        if pending or engineering_done is None:
            backend = backend_factory(ModelSettings(**protocol["model_settings"]))
            _validate_backend(backend, protocol)
            manifest["backend"] = backend.metadata
            atomic_json(target / "manifest.json", manifest)
        if engineering_done is None:
            verify_protocol(protocol)
            snapshot_files(authorization["sha256"])
            require(
                file_sha256(output / "protocol.json") == digest,
                "Protocol changed before engineering forward",
            )
            _measure_row(
                backend, protocol, digest, engineering, output / "engineering", engineering=True
            )
        for planned in pending:
            verify_protocol(protocol)
            snapshot_files(authorization["sha256"])
            require(
                file_sha256(output / "protocol.json") == digest, "Protocol changed during capture"
            )
            _measure_row(backend, protocol, digest, planned, journal / planned["row_id"])
            manifest["completed_rows"] += 1
            atomic_json(target / "manifest.json", manifest)
            if progress:
                progress(
                    f"Fresh {partition}: {manifest['completed_rows']}/{len(selected)}", flush=True
                )
        verify_protocol(protocol)
        snapshot_files(authorization["sha256"])
        rows, features = [], []
        for planned in selected:
            row = _completed(journal / planned["row_id"], planned, protocol, digest)
            require(row is not None, "Cannot export incomplete capture")
            rows.append({**row, "output_yes_no_logitdiff": row["output_logits"]["yes_minus_no"]})
            with np.load(journal / planned["row_id"] / "arrays.npz", allow_pickle=False) as data:
                features.append(data["features"].copy())
        atomic_npz(
            target / "features.npz",
            features=np.stack(features).astype(np.float32),
            row_ids=np.asarray([r["row_id"] for r in selected], dtype=str),
            sites=np.asarray(protocol["sites"], dtype=str),
        )
        atomic_json(
            target / "rows.json",
            {
                "schema": "fresh-readout-capture-v1",
                "rows": rows,
                "test_fixture": protocol["test_fixture"],
                "provenance": {
                    "protocol_sha256": digest,
                    "bank_file_sha256": protocol["bank_file_sha256"],
                    "source_bindings": authorization["sha256"],
                    "partition": partition,
                    "model_key": protocol["model_key"],
                    "model_settings": protocol["model_settings"],
                    "sites": protocol["sites"],
                    "original_site": protocol["original_site"],
                    "research_status": "agent_audited_exploratory",
                },
            },
        )
        verify_protocol(protocol)
        snapshot_files(authorization["sha256"])
        require(file_sha256(output / "protocol.json") == digest, "Protocol changed during export")
        manifest.update(
            status="complete",
            finished_at=utc_now(),
            features_sha256=file_sha256(target / "features.npz"),
            rows_sha256=file_sha256(target / "rows.json"),
        )
        atomic_json(target / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        manifest.update(
            status="interrupted_preserved",
            error_type=type(error).__name__,
            error=str(error),
            finished_at=utc_now(),
        )
        atomic_json(target / "manifest.json", manifest)
        raise
    finally:
        if backend is not None:
            backend.close()


def execute(
    output, *, partition, bindings_path, resume=False, backend_factory=HFBackend, progress=print
):
    """Hold one exclusive output lock across all partitions, aliases and resumes.

    A process death leaves the lock as an unknown attempt. It must be inspected
    explicitly; a later invocation never silently removes or steals that lock.
    """
    output = Path(output).resolve()
    lock = output / "execution.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            json.dump({"partition": partition, "started_at": utc_now(), "resume": resume}, stream)
    except FileExistsError as error:
        raise ValueError(
            "Capture output is locked by an active or unknown attempt; preserve and inspect it"
        ) from error
    try:
        return _execute_locked(
            output,
            partition=partition,
            bindings_path=bindings_path,
            resume=resume,
            backend_factory=backend_factory,
            progress=progress,
        )
    finally:
        lock.unlink()
