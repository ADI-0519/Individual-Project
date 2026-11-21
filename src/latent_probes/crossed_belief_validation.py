"""Bound full-model measurements for a frozen, fully crossed belief assay.

No fitting, threshold selection, or output-based retry occurs in this module.
The same forward supplies the block-16 feature and the complete output logits.
"""

from __future__ import annotations

import re

import numpy as np

from .storage import canonical_hash

HEAD = "nested_question__sender__nested__layer_16"
CONDITIONS = ("story", "nested_question")


def require(value, message):
    if not value:
        raise ValueError(message)


def summarize_logits(values, tokens):
    """Use the preserved complete float32 vocabulary vector, without calibration."""
    raw = np.asarray(values)
    require(
        raw.ndim == 1 and raw.dtype == np.dtype("float32") and np.isfinite(raw).all(),
        "Need a finite complete float32 vocabulary vector",
    )
    require(
        set(tokens) == {"YES", "NO"}
        and all(type(t) is int and 0 <= t < len(raw) for t in tokens.values())
        and tokens["YES"] != tokens["NO"],
        "Invalid distinct YES/NO token IDs",
    )
    logits = raw.astype(np.float64)
    yes, no = (float(logits[tokens[word]]) for word in ("YES", "NO"))
    log_z = float(logits.max() + np.log(np.exp(logits - logits.max()).sum()))
    return {
        "yes_logit": yes,
        "no_logit": no,
        "yes_logprob": yes - log_z,
        "no_logprob": no - log_z,
        "yes_minus_no": yes - no,
        "p_yes_given_yes_no": float(np.exp(-np.logaddexp(0.0, no - yes))),
        "full_vocab_logsumexp": log_z,
        "vocab_size": len(logits),
        "argmax_token_id": int(logits.argmax()),
        "logits_values_sha256": canonical_hash(raw.tolist()),
    }


def forward_measurement(backend, ids, tokens):
    """One untruncated full forward; capture the last prompt token before any answer."""
    require(
        isinstance(ids, list)
        and ids
        and all(type(i) is int and i >= 0 for i in ids)
        and len(ids) <= backend.settings.max_input_tokens,
        "Invalid or oversized exact prompt prefix",
    )
    require(list(backend.settings.capture_layers) == [16], "Capture only frozen block16")
    maximum = getattr(backend.model.config, "max_position_embeddings", None)
    require(maximum is None or len(ids) <= maximum, "Prefix exceeds model context")
    captured = []

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden[0, -1].detach().float().cpu().numpy().copy())

    handle = backend.base.layers[16].register_forward_hook(hook)
    try:
        sequence = backend.torch.tensor([ids], device=backend.input_device)
        with backend.torch.inference_mode():
            result = backend.model(
                input_ids=sequence,
                attention_mask=backend.torch.ones_like(sequence),
                logits_to_keep=1,
                use_cache=False,
                return_dict=True,
            )
    finally:
        handle.remove()
    require(len(captured) == 1, "Expected exactly one block16 hook observation")
    vector = captured[0]
    require(
        vector.shape == (backend.metadata["hidden_size"],)
        and vector.dtype == np.dtype("float32")
        and np.isfinite(vector).all(),
        "Invalid block16 feature",
    )
    values = result.logits.detach().float().cpu().numpy()
    require(
        values.ndim == 3 and values.shape[:2] == (1, 1),
        "Expected only the last prompt token's full vocabulary logits",
    )
    values = values[0, 0].copy()
    return vector, values, summarize_logits(values, tokens)


def answer_record(generation, ids, logits, *, max_new_tokens, eos_ids, special_ids):
    """Strict observed answer; caps/unexpected specials remain invalid, never retried."""
    require(generation["prompt_token_ids"] == ids, "Generated answer prefix differs")
    generated = generation["generated_token_ids"]
    require(
        isinstance(generated, list)
        and 0 < len(generated) <= max_new_tokens
        and all(type(token) is int and token >= 0 for token in generated),
        "Invalid generation token record",
    )
    require(generated[0] == logits["argmax_token_id"], "Greedy first token differs from logits")
    eos_stop = generated[-1] in eos_ids
    require(
        (generation["finish_reason"] == "eos") == eos_stop,
        "Generation finish reason differs from EOS tokens",
    )
    capped = not eos_stop
    unexpected_special = any(
        token in set(special_ids) | set(eos_ids)
        for token in (generated[:-1] if eos_stop else generated)
    )
    match = re.fullmatch(r"\s*(YES|NO)[.!]?\s*", generation["text"], flags=re.IGNORECASE)
    prediction = int(match.group(1).upper() == "YES") if match else None
    if capped or unexpected_special:
        prediction = None
    return {
        "prediction": prediction,
        "capped": capped,
        "unexpected_special": unexpected_special,
        "parser": "strict YES/NO, optional terminal period/exclamation; caps/specials invalid",
    }
