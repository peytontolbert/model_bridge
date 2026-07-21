from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ROOT = Path("/arxiv/models")


@dataclass(frozen=True)
class AudioGenerationRequest:
    model_id: str
    model_root: str = str(DEFAULT_MODEL_ROOT)
    prompt: str = "lo-fi drums and warm bass"
    max_new_tokens: int = 8
    device: str = "cuda:0"
    dtype: str = "auto"
    run_generate: bool = False


@dataclass(frozen=True)
class AudioGenerationResult:
    status: str
    model_id: str
    model_path: str | None
    task: str | None = None
    model_type: str | None = None
    architectures: tuple[str, ...] = ()
    transformers_version: str | None = None
    weight_files: tuple[str, ...] = ()
    processor_files: tuple[str, ...] = ()
    tokenizer_files: tuple[str, ...] = ()
    model_class: str | None = None
    processor_class: str | None = None
    sampling_rate: int | None = None
    generated_audio_shape: tuple[int, ...] = ()
    generated_audio_dtype: str | None = None
    generated_audio_seconds: float | None = None
    artifact_contract: str = "transformers_audio_generation_contract"
    error: str | None = None


class AudioGenerationBridgeError(RuntimeError):
    pass


def resolve_model_path(model_id: str, model_root: str | Path = DEFAULT_MODEL_ROOT) -> Path:
    candidate = Path(model_id)
    if candidate.exists():
        return candidate
    root = Path(model_root)
    candidates = [root / model_id]
    if "/" in model_id:
        candidates.append(root / model_id.split("/", 1)[1])
        candidates.append(root / model_id.replace("/", "--"))
    for item in candidates:
        if item.exists():
            return item
    raise AudioGenerationBridgeError(f"model path not found for {model_id!r} under {root}")


def read_model_config(model_path: str | Path) -> dict[str, Any]:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        raise AudioGenerationBridgeError(f"missing config.json: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def infer_task(config: dict[str, Any]) -> str:
    model_type = config.get("model_type")
    if model_type == "musicgen":
        return "musicgen_text_to_audio"
    if model_type == "musicgen_melody":
        return "musicgen_melody_text_to_audio"
    raise AudioGenerationBridgeError(f"unsupported audio generation model_type={model_type!r}")


def _torch_dtype(dtype: str) -> Any:
    if dtype == "auto":
        return "auto"
    import torch

    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype not in aliases:
        raise AudioGenerationBridgeError(f"unsupported dtype: {dtype}")
    return aliases[dtype]


def _artifact_files(model_path: Path) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    weight_suffixes = {".bin", ".safetensors", ".pt", ".pth", ".ckpt"}
    weights = sorted(path.name for path in model_path.glob("*") if path.suffix in weight_suffixes)
    processors = tuple(
        sorted(path.name for path in model_path.glob("*") if path.name in {"preprocessor_config.json", "processor_config.json"})
    )
    tokenizers = tuple(
        sorted(
            path.name
            for path in model_path.glob("*")
            if path.name in {"tokenizer.json", "tokenizer_config.json", "spiece.model", "special_tokens_map.json"}
        )
    )
    return tuple(weights), tokenizers, processors


def _sampling_rate(model_path: Path) -> int | None:
    path = model_path / "preprocessor_config.json"
    if not path.is_file():
        return None
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("sampling_rate"))
    except Exception:
        return None


def inspect_audio_generation(request: AudioGenerationRequest) -> AudioGenerationResult:
    try:
        model_path = resolve_model_path(request.model_id, request.model_root)
        config = read_model_config(model_path)
        task = infer_task(config)
        weights, tokenizers, processors = _artifact_files(model_path)
        sampling_rate = _sampling_rate(model_path)
        status = "ready"
        model_class = None
        processor_class = None
        shape: tuple[int, ...] = ()
        audio_dtype = None
        seconds = None
        error = None
        if request.run_generate:
            try:
                import torch
                from transformers import AutoProcessor, MusicgenForConditionalGeneration, MusicgenMelodyForConditionalGeneration

                processor = AutoProcessor.from_pretrained(str(model_path))
                model_cls = MusicgenMelodyForConditionalGeneration if task == "musicgen_melody_text_to_audio" else MusicgenForConditionalGeneration
                model = model_cls.from_pretrained(
                    str(model_path),
                    torch_dtype=_torch_dtype(request.dtype),
                    low_cpu_mem_usage=True,
                )
                if request.device and request.device != "cpu":
                    model = model.to(request.device)
                model.eval()
                inputs = processor(text=[request.prompt], padding=True, return_tensors="pt")
                if request.device and request.device != "cpu":
                    inputs = {key: value.to(request.device) for key, value in inputs.items()}
                with torch.no_grad():
                    audio = model.generate(**inputs, max_new_tokens=request.max_new_tokens)
                shape = tuple(int(dim) for dim in audio.shape)
                audio_dtype = str(audio.dtype)
                if sampling_rate and shape:
                    seconds = float(shape[-1]) / float(sampling_rate)
                model_class = type(model).__name__
                processor_class = type(processor).__name__
                status = "ok"
            except Exception as exc:
                status = "generate_blocked"
                error = f"{type(exc).__name__}:{exc}"
        return AudioGenerationResult(
            status=status,
            model_id=request.model_id,
            model_path=str(model_path),
            task=task,
            model_type=config.get("model_type"),
            architectures=tuple(config.get("architectures") or ()),
            transformers_version=config.get("transformers_version"),
            weight_files=weights,
            processor_files=processors,
            tokenizer_files=tokenizers,
            model_class=model_class,
            processor_class=processor_class,
            sampling_rate=sampling_rate,
            generated_audio_shape=shape,
            generated_audio_dtype=audio_dtype,
            generated_audio_seconds=seconds,
            error=error,
        )
    except Exception as exc:
        return AudioGenerationResult(status="failed", model_id=request.model_id, model_path=None, error=f"{type(exc).__name__}:{exc}")


def to_json(obj: Any) -> dict[str, Any]:
    return asdict(obj)
