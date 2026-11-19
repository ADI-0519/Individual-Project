"""Frozen Hugging Face generation, with selected residual-stream captures.

The real backend supports Llama and Qwen2 decoder architectures. It never
silently falls back to scripted responses. Torch is imported only on use.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from .config import GenerationSettings, ModelSettings
from .storage import canonical_hash, environment_info


@dataclass
class GenerationResult:
    text: str
    prompt_token_ids: list[int] = field(default_factory=list)
    generated_token_ids: list[int] = field(default_factory=list)
    features: dict[str, np.ndarray] = field(default_factory=dict)
    token_positions: dict[str, int] = field(default_factory=dict)
    finish_reason: str = "unknown"
    elapsed_seconds: float = 0.0
    extraction_seconds: float = 0.0
    peak_allocated_gib: dict[str, float] = field(default_factory=dict)

    def record(self):
        return {k: v for k, v in self.__dict__.items() if k != "features"}


def doctor():
    info = environment_info()
    try:
        import torch
    except ImportError as exc:
        info["cuda"] = {"available": False, "reason": f"PyTorch import failed: {exc}"}
        return info
    info["cuda"] = {
        "available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "devices": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["cuda"]["devices"].append(
                {
                    "index": i,
                    "name": props.name,
                    "total_gib": props.total_memory / 2**30,
                    "compute_capability": list(torch.cuda.get_device_capability(i)),
                }
            )
        # A discovered GPU is insufficient: actually launch a small kernel.
        x = torch.ones((32, 32), device="cuda")
        info["cuda"]["kernel_check_passed"] = bool(torch.all(x @ x == 32).item())
        torch.cuda.synchronize()
    return info


class HFBackend:
    def __init__(self, settings: ModelSettings):
        try:
            import torch
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Model runtime import failed. Install compatible [models,cpu] or "
                f"[models,cu130] dependencies. Underlying error: {exc}"
            ) from exc
        self.torch = torch
        self.settings = settings
        if settings.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable. Run 'latent-probes doctor'.")
        if settings.quantization != "none" and not torch.cuda.is_available():
            raise RuntimeError("Quantized starter configurations require a working CUDA GPU")
        if settings.device == "cpu" and settings.dtype != "float32":
            raise ValueError("Use float32 for this starter's CPU debug configuration")
        common = {
            "revision": settings.revision,
            "trust_remote_code": False,
            "local_files_only": settings.local_files_only,
        }
        config = AutoConfig.from_pretrained(settings.model_id, **common)
        if config.model_type not in {"llama", "qwen2"}:
            raise ValueError(
                "This extractor explicitly supports llama and qwen2; validate other architectures before adding them"
            )
        # Resolve a mutable Hub branch once, then use the same revision for all loads.
        resolved = getattr(config, "_commit_hash", None)
        if resolved:
            common["revision"] = resolved
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_id, **common)
        if not self.tokenizer.chat_template:
            raise ValueError("An instruct checkpoint with a chat template is required")
        dtype = getattr(torch, settings.dtype)
        device_map = (
            {"": "cpu"}
            if settings.device == "cpu"
            else ({"": 0} if settings.device == "cuda" else "auto")
        )
        load_kwargs = {
            **common,
            "config": config,
            "dtype": dtype,
            "device_map": device_map,
            "use_safetensors": True,
            "attn_implementation": "sdpa",
        }
        if settings.quantization != "none":
            quant_args = {
                "load_in_4bit": settings.quantization == "4bit",
                "load_in_8bit": settings.quantization == "8bit",
            }
            if settings.quantization == "4bit":
                quant_args.update(
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            load_kwargs["quantization_config"] = BitsAndBytesConfig(**quant_args)
        if settings.max_memory:
            load_kwargs["max_memory"] = {
                int(k) if k.isdigit() else k: v for k, v in settings.max_memory.items()
            }
        self.model = AutoModelForCausalLM.from_pretrained(settings.model_id, **load_kwargs)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.base = self.model.base_model
        if not hasattr(self.base, "layers"):
            raise RuntimeError("Expected a decoder with a .layers module list")
        for layer in settings.capture_layers:
            if layer >= len(self.base.layers):
                raise ValueError(
                    f"Decoder layer {layer} does not exist; model has {len(self.base.layers)} blocks"
                )
        placement = {str(k): str(v) for k, v in getattr(self.model, "hf_device_map", {}).items()}
        if any(value == "disk" for value in placement.values()):
            raise RuntimeError(
                "Disk offload is not supported by this starter; request more memory or use a smaller checkpoint"
            )
        if (
            torch.cuda.is_available()
            and settings.device != "cpu"
            and not settings.allow_cpu_offload
        ):
            if any(value in {"cpu", "disk"} for value in placement.values()):
                raise RuntimeError(
                    "Model was offloaded to CPU/disk. Reduce model/memory demand or explicitly allow_cpu_offload."
                )
        self.input_device = self.model.get_input_embeddings().weight.device
        if self.input_device.type == "meta":
            raise RuntimeError(
                "Offloaded embeddings have no directly addressable device; this starter cannot use this placement"
            )
        self.metadata = {
            "backend": "huggingface",
            "is_test_fixture": False,
            "model_id": settings.model_id,
            "requested_revision": settings.revision,
            "resolved_revision": resolved,
            "model_type": config.model_type,
            "dtype": settings.dtype,
            "quantization": settings.quantization,
            "device_map": placement,
            "decoder_blocks": len(self.base.layers),
            "hidden_size": config.hidden_size,
            "capture_layers": settings.capture_layers,
            "chat_template_sha256": canonical_hash(self.tokenizer.chat_template),
            "generation_defaults": self.model.generation_config.to_dict(),
            "special_token_ids": {
                "eos": self.tokenizer.eos_token_id,
                "pad": self.tokenizer.pad_token_id,
            },
            "attention_implementation": "sdpa",
            "extraction": "decoder_block_output_before_final_model_norm; 0-based blocks; exact generated token replay",
            "positions": {
                "prompt_last": "last prompt token including assistant generation header",
                "response_last": "last non-special generated token, before any elicited forecast",
            },
        }

    def _capture(self, sequence, prompt_length, response_position):
        torch = self.torch
        captured = {}
        positions = {"prompt_last": prompt_length - 1, "response_last": response_position}
        hooks = []
        for layer in self.settings.capture_layers:

            def capture(_module, _inputs, output, layer_index=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                for name, position in positions.items():
                    vector = (
                        hidden[0, position, :]
                        .detach()
                        .to(device="cpu", dtype=torch.float32)
                        .numpy()
                        .copy()
                    )
                    if not np.isfinite(vector).all():
                        raise RuntimeError("Non-finite activation values")
                    captured[f"layer_{layer_index}_{name}"] = vector

            hooks.append(self.base.layers[layer].register_forward_hook(capture))
        try:
            with torch.inference_mode():
                # Bypass the vocabulary projection: retain only selected vectors.
                self.base(
                    input_ids=sequence,
                    attention_mask=torch.ones_like(sequence),
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
        finally:
            for hook in hooks:
                hook.remove()
        expected = len(self.settings.capture_layers) * len(positions)
        if len(captured) != expected:
            raise RuntimeError("Not all requested activations were captured")
        return captured, positions

    def generate(self, messages, settings: GenerationSettings, seed: int, capture=False):
        torch = self.torch
        from transformers import set_seed

        set_seed(seed)
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        prompt_length = encoded["input_ids"].shape[1]
        if prompt_length > self.settings.max_input_tokens:
            raise ValueError(
                f"Prompt has {prompt_length} tokens, exceeding {self.settings.max_input_tokens}; no facts were silently truncated"
            )
        max_context = getattr(self.model.config, "max_position_embeddings", None)
        if max_context and prompt_length + settings.max_new_tokens > max_context:
            raise ValueError("Prompt plus generation budget exceeds model context")
        encoded = {k: v.to(self.input_device) for k, v in encoded.items()}
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = next((x for x in eos_ids if x is not None), None)
        kwargs = {
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": settings.do_sample,
            "pad_token_id": pad,
            "use_cache": True,
            "return_dict_in_generate": False,
            "num_beams": 1,
            "num_return_sequences": 1,
            "repetition_penalty": 1.0,
            "min_new_tokens": 0,
        }
        if eos is not None:
            kwargs["eos_token_id"] = eos
        if settings.do_sample:
            kwargs.update(temperature=settings.temperature, top_p=settings.top_p, top_k=0)
        gpu_indices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        for i in gpu_indices:
            torch.cuda.reset_peak_memory_stats(i)
        start = time.perf_counter()
        with torch.inference_mode():
            sequences = self.model.generate(**encoded, **kwargs)
        generated = sequences[0, prompt_length:].tolist()
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        features, positions, extraction_seconds = {}, {}, 0.0
        if capture:
            content_positions = [
                prompt_length + i
                for i, token in enumerate(generated)
                if token not in set(self.tokenizer.all_special_ids)
            ]
            if not content_positions:
                raise ValueError("No non-special response tokens; cannot extract response_last")
            extract_start = time.perf_counter()
            features, positions = self._capture(sequences, prompt_length, content_positions[-1])
            extraction_seconds = time.perf_counter() - extract_start
        for i in gpu_indices:
            torch.cuda.synchronize(i)
        return GenerationResult(
            text=text,
            prompt_token_ids=encoded["input_ids"][0].tolist(),
            generated_token_ids=generated,
            features=features,
            token_positions=positions,
            finish_reason="eos" if generated and generated[-1] in eos_ids else "length",
            elapsed_seconds=time.perf_counter() - start,
            extraction_seconds=extraction_seconds,
            peak_allocated_gib={
                str(i): torch.cuda.max_memory_allocated(i) / 2**30 for i in gpu_indices
            },
        )

    def close(self):
        del self.base
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def _capture_token_rows(self, sequence, positions):
        """Capture selected rows only, before the decoder's final normalization.

        Unlike the starter's last-token vectors, these matrices retain every
        selected training/evaluation token for the Apollo-style detector.
        """
        if (
            not positions
            or len(set(positions)) != len(positions)
            or any(type(p) is not int or p < 0 or p >= sequence.shape[1] for p in positions)
        ):
            raise ValueError("Capture positions must be nonempty, unique integer indices in range")
        torch = self.torch
        captured, hooks = {}, []
        for layer in self.settings.capture_layers:

            def capture(_module, _inputs, output, layer_index=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                rows = (
                    hidden[0, positions, :]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .numpy()
                    .copy()
                )
                if rows.ndim != 2 or not np.isfinite(rows).all():
                    raise RuntimeError("Invalid token activation matrix")
                captured[f"layer_{layer_index}_tokens"] = rows

            hooks.append(self.base.layers[layer].register_forward_hook(capture))
        try:
            with torch.inference_mode():
                self.base(
                    input_ids=sequence,
                    attention_mask=torch.ones_like(sequence),
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
        finally:
            for hook in hooks:
                hook.remove()
        if len(captured) != len(self.settings.capture_layers):
            raise RuntimeError("Not all requested token activations were captured")
        return captured

    def forced_token_activations(self, messages, response, trim_last_words=0):
        """Score an explicitly off-policy assistant continuation, not a generation.

        The source-faithful RepE profile excludes five space-delimited words.
        Character-to-token half-open boundaries follow the pinned source. The
        complete response, including the unscored suffix, remains in context.
        """
        if not isinstance(response, str) or not response.strip():
            raise ValueError("A nonempty fixed assistant response is required")
        if type(trim_last_words) is not int or trim_last_words < 0:
            raise ValueError("trim_last_words must be a nonnegative integer")
        if not self.tokenizer.is_fast:
            raise ValueError("Auditable character masks require a fast tokenizer")
        start = time.perf_counter()
        gpu_indices = (
            list(range(self.torch.cuda.device_count())) if self.torch.cuda.is_available() else []
        )
        for i in gpu_indices:
            self.torch.cuda.reset_peak_memory_stats(i)
        prefix = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        rendered = self.tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": response}],
            tokenize=False,
            add_generation_prompt=False,
        )
        if not rendered.startswith(prefix + response):
            raise ValueError("Chat template does not preserve the assistant span exactly")
        if trim_last_words:
            words = response.split(" ")
            retained = " ".join(words[:-trim_last_words])
            if not retained.strip():
                raise ValueError("Fact has no retained prefix after word trimming")
        else:
            retained = response
        encoded = self.tokenizer(
            rendered, add_special_tokens=False, return_tensors="pt", return_offsets_mapping=True
        )
        char_start, char_end = len(prefix), len(prefix) + len(retained)
        token_start = encoded.char_to_token(0, char_start)
        token_end = (
            encoded.char_to_token(0, char_end)
            if char_end < len(rendered)
            else encoded["input_ids"].shape[1]
        )
        if token_start is None or token_end is None or token_end <= token_start:
            raise ValueError("Assistant character boundary has no exact nonempty token mask")
        sequence = encoded["input_ids"].to(self.input_device)
        if sequence.shape[1] > self.settings.max_input_tokens:
            raise ValueError("Forced dialogue exceeds max_input_tokens; nothing was truncated")
        maximum = getattr(self.model.config, "max_position_embeddings", None)
        if maximum and sequence.shape[1] > maximum:
            raise ValueError("Forced dialogue exceeds the model context")
        positions = list(range(token_start, token_end))
        features = self._capture_token_rows(sequence, positions)
        selected_ids = sequence[0, positions].tolist()
        for i in gpu_indices:
            self.torch.cuda.synchronize(i)
        peak_memory = {str(i): self.torch.cuda.max_memory_allocated(i) / 2**30 for i in gpu_indices}
        elapsed = time.perf_counter() - start
        record = {
            "mode": "off_policy_fixed_response",
            "text": response,
            "rendered_dialogue": rendered,
            "input_token_ids": sequence[0].tolist(),
            "selected_token_ids": selected_ids,
            "selected_token_positions": positions,
            "selected_decoded_text": self.tokenizer.decode(selected_ids),
            "character_span": [char_start, char_end],
            "excluded_suffix": response[len(retained) :],
            "trim_last_words": trim_last_words,
            "mask": "char_to_token half-open interval; zero header padding",
            "elapsed_seconds": elapsed,
            "peak_allocated_gib": peak_memory,
        }
        return features, record

    def generated_token_activations(self, messages, settings, seed):
        """Generate naturally, then replay EXACT token IDs and score content rows."""
        result = self.generate(messages, settings, seed, capture=False)
        prompt_ids, response_ids = result.prompt_token_ids, result.generated_token_ids
        specials = set(self.tokenizer.all_special_ids)
        positions = [
            len(prompt_ids) + i for i, token in enumerate(response_ids) if token not in specials
        ]
        if not positions:
            raise ValueError("Generated response contains no scoreable content tokens")
        sequence = self.torch.tensor([prompt_ids + response_ids], device=self.input_device)
        start = time.perf_counter()
        features = self._capture_token_rows(sequence, positions)
        gpu_indices = (
            list(range(self.torch.cuda.device_count())) if self.torch.cuda.is_available() else []
        )
        for i in gpu_indices:
            self.torch.cuda.synchronize(i)
        extraction_seconds = time.perf_counter() - start
        selected_ids = [token for token in response_ids if token not in specials]
        return features, {
            **result.record(),
            "mode": "on_policy_generated_response",
            "generation_seconds": result.elapsed_seconds,
            "extraction_seconds": extraction_seconds,
            "elapsed_seconds": result.elapsed_seconds + extraction_seconds,
            "peak_allocated_gib": {
                str(i): self.torch.cuda.max_memory_allocated(i) / 2**30 for i in gpu_indices
            },
            "selected_token_ids": selected_ids,
            "selected_token_positions": positions,
            "selected_decoded_text": self.tokenizer.decode(selected_ids),
            "mask": "all non-special generated response tokens; no reasoning trimming",
        }
