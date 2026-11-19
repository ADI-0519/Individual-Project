"""Explicit configuration with no model imports or downloads during validation."""

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


def _checked(cls, data):
    if not isinstance(data, dict):
        raise ValueError(f"{cls.__name__} must be a JSON object")
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {sorted(unknown)}")
    return cls(**data)


@dataclass(frozen=True)
class ModelSettings:
    model_id: str
    revision: str = "main"
    dtype: str = "bfloat16"
    quantization: str = "none"
    device: str = "auto"
    capture_layers: list[int] = field(default_factory=lambda: [16])
    max_input_tokens: int = 1024
    max_memory: dict[str, str] | None = None
    allow_cpu_offload: bool = False
    local_files_only: bool = False

    def __post_init__(self):
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if not isinstance(self.revision, str) or not self.revision:
            raise ValueError("revision must be a non-empty string")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16 or bfloat16")
        if self.quantization not in {"none", "4bit", "8bit"}:
            raise ValueError("quantization must be none, 4bit or 8bit")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")
        if self.device == "cpu" and self.quantization != "none":
            raise ValueError("This starter supports quantized loading on CUDA only")
        if not self.capture_layers or any(type(x) is not int or x < 0 for x in self.capture_layers):
            raise ValueError("capture_layers must contain non-negative decoder block indices")
        if len(set(self.capture_layers)) != len(self.capture_layers):
            raise ValueError("capture_layers contains duplicates")
        if type(self.max_input_tokens) is not int or self.max_input_tokens < 32:
            raise ValueError("max_input_tokens must be an integer >= 32")
        if any(type(v) is not bool for v in [self.allow_cpu_offload, self.local_files_only]):
            raise ValueError("allow_cpu_offload and local_files_only must be booleans")
        if self.max_memory is not None and (
            not isinstance(self.max_memory, dict)
            or any(
                not isinstance(k, str) or not isinstance(v, str) for k, v in self.max_memory.items()
            )
        ):
            raise ValueError("max_memory must map string device keys to memory limits")


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int = 96
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.95

    def __post_init__(self):
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if type(self.do_sample) is not bool:
            raise ValueError("do_sample must be boolean")
        if any(
            type(x) not in {int, float} or not math.isfinite(x)
            for x in (self.temperature, self.top_p)
        ):
            raise ValueError("temperature and top_p must be finite numbers, not booleans")
        if self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("temperature must be positive and top_p must be in (0, 1]")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelSettings
    sender: GenerationSettings = field(default_factory=GenerationSettings)
    recipient: GenerationSettings = field(
        default_factory=lambda: GenerationSettings(max_new_tokens=8, do_sample=False)
    )
    seeds: list[int] = field(default_factory=lambda: [17, 29, 43])
    objectives: list[str] = field(default_factory=lambda: ["neutral", "favourable"])
    assumed_knowledge: list[bool] = field(default_factory=lambda: [False, True])
    elicited_forecast: bool = False

    def __post_init__(self):
        if not self.seeds or any(type(x) is not int or not 0 <= x < 2**32 for x in self.seeds):
            raise ValueError("seeds must be integers in [0, 2**32)")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if not self.sender.do_sample and len(self.seeds) > 1:
            raise ValueError(
                "Greedy generation needs one seed; extra seeds are not independent samples"
            )
        if (
            not self.objectives
            or len(set(self.objectives)) != len(self.objectives)
            or any(x not in {"neutral", "favourable"} for x in self.objectives)
        ):
            raise ValueError("objectives must be unique values from neutral, favourable")
        if not self.assumed_knowledge or any(type(x) is not bool for x in self.assumed_knowledge):
            raise ValueError("assumed_knowledge must contain booleans")
        if len(set(self.assumed_knowledge)) != len(self.assumed_knowledge):
            raise ValueError("assumed_knowledge must be unique")
        if type(self.elicited_forecast) is not bool:
            raise ValueError("elicited_forecast must be boolean")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["model"] = _checked(ModelSettings, data["model"])
        for key in ("sender", "recipient"):
            if key in data:
                data[key] = _checked(GenerationSettings, data[key])
        return _checked(cls, data)


def load_config(path):
    return ExperimentConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
