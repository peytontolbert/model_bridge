from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ROOT = Path("/arxiv/models")
DEFAULT_HF_CACHE_ROOT = Path("/data/huggingface/hub")


@dataclass(frozen=True)
class ClassicTransformersRequest:
    model_id: str
    model_root: str = str(DEFAULT_MODEL_ROOT)
    task: str | None = None
    device: str = "cpu"
    dtype: str = "auto"
    prompt: str = "Legacy model bridge smoke input."
    max_new_tokens: int = 4
    run_synthetic: bool = False
    trust_remote_code: bool = False


@dataclass(frozen=True)
class ClassicTransformersResult:
    status: str
    model_id: str
    model_path: str | None
    task: str | None = None
    config_model_type: str | None = None
    architectures: tuple[str, ...] = ()
    transformers_version: str | None = None
    weight_files: tuple[str, ...] = ()
    tokenizer_files: tuple[str, ...] = ()
    processor_files: tuple[str, ...] = ()
    model_class: str | None = None
    tokenizer_class: str | None = None
    processor_class: str | None = None
    synthetic_outputs: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None
    artifact_contract: str = "transformers_auto_contract"
    error: str | None = None


class ClassicTransformersBridgeError(RuntimeError):
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
    hf_snapshot = resolve_hf_cache_snapshot(model_id)
    if hf_snapshot is not None:
        return hf_snapshot
    raise ClassicTransformersBridgeError(f"model path not found for {model_id!r} under {root}")


def resolve_hf_cache_snapshot(model_id: str, hf_cache_root: str | Path = DEFAULT_HF_CACHE_ROOT) -> Path | None:
    cache_root = Path(hf_cache_root)
    repo_dir = cache_root / f"models--{model_id.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = [path for path in snapshots.iterdir() if (path / "config.json").is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_model_config(model_path: str | Path) -> dict[str, Any]:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        raise ClassicTransformersBridgeError(f"missing config.json: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def infer_task(config: dict[str, Any], requested: str | None = None) -> str:
    if requested:
        return requested
    model_type = config.get("model_type")
    architectures = set(config.get("architectures") or [])
    if model_type == "bart" or "BartForConditionalGeneration" in architectures:
        return "seq2seq_generation"
    if model_type == "gpt2" or any(item.endswith("ForCausalLM") or item == "GPT2LMHeadModel" for item in architectures):
        return "causal_lm_generation"
    if any(item.endswith("ForSequenceClassification") for item in architectures):
        return "sequence_classification"
    if any(item.endswith("ForMaskedLM") for item in architectures):
        return "masked_lm"
    if model_type == "bert" or any(item == "BertModel" for item in architectures):
        return "text_encoder"
    if model_type == "hubert" or any("Hubert" in item for item in architectures):
        return "audio_encoder"
    if model_type == "dinov2" or any("Dinov2" in item for item in architectures):
        return "vision_encoder"
    raise ClassicTransformersBridgeError(f"cannot infer classic Transformers task for model_type={model_type!r}")


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
        raise ClassicTransformersBridgeError(f"unsupported dtype: {dtype}")
    return aliases[dtype]




def _load_transformers_model(loader: Any, model_path: Path, request: ClassicTransformersRequest) -> Any:
    dtype = _torch_dtype(request.dtype)
    kwargs = {"trust_remote_code": request.trust_remote_code}
    try:
        return loader.from_pretrained(str(model_path), dtype=dtype, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
    return loader.from_pretrained(str(model_path), torch_dtype=dtype, **kwargs)


def runtime_provenance(request: ClassicTransformersRequest) -> dict[str, Any]:
    import platform

    payload: dict[str, Any] = {
        "requested_device": request.device,
        "requested_dtype": request.dtype,
        "python_version": platform.python_version(),
        "torch_version": None,
        "transformers_version": None,
        "cuda_available": None,
        "cuda_device_name": None,
    }
    try:
        import torch

        payload["torch_version"] = getattr(torch, "__version__", None)
        payload["cuda_available"] = bool(torch.cuda.is_available())
        if payload["cuda_available"] and request.device and request.device != "cpu":
            device_index = 0
            if request.device.startswith("cuda:"):
                try:
                    device_index = int(request.device.split(":", 1)[1])
                except ValueError:
                    device_index = 0
            payload["cuda_device_name"] = torch.cuda.get_device_name(device_index)
    except Exception as exc:
        payload["torch_error"] = f"{type(exc).__name__}:{exc}"
    try:
        import transformers

        payload["transformers_version"] = getattr(transformers, "__version__", None)
    except Exception as exc:
        payload["transformers_error"] = f"{type(exc).__name__}:{exc}"
    return payload


def _artifact_files(model_path: Path) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    weight_suffixes = {".bin", ".safetensors", ".pt", ".pth", ".ckpt"}
    weights = sorted(path.name for path in model_path.glob("*") if path.suffix in weight_suffixes)
    tokenizers = sorted(
        path.name
        for path in model_path.glob("*")
        if path.name
        in {
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "sentencepiece.bpe.model",
            "spiece.model",
            "vocab.txt",
        }
    )
    processors = sorted(
        path.name
        for path in model_path.glob("*")
        if path.name
        in {
            "preprocessor_config.json",
            "feature_extractor_config.json",
            "image_processor_config.json",
            "processor_config.json",
        }
    )
    return tuple(weights), tuple(tokenizers), tuple(processors)


def _move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    if not device or device == "cpu":
        return batch
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def _cast_floating_batch(batch: dict[str, Any], dtype: Any) -> dict[str, Any]:
    if dtype in {None, "auto"}:
        return batch
    return {
        key: value.to(dtype=dtype) if getattr(value, "is_floating_point", lambda: False)() else value
        for key, value in batch.items()
    }


def _synthetic_seq2seq(model_path: Path, request: ClassicTransformersRequest) -> tuple[Any, Any, dict[str, Any]]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=request.trust_remote_code)
    model = _load_transformers_model(AutoModelForSeq2SeqLM, model_path, request)
    if request.device and request.device != "cpu":
        model = model.to(request.device)
    model.eval()
    encoded = _move_batch(tokenizer(request.prompt, return_tensors="pt"), request.device)
    output_ids = model.generate(**encoded, max_new_tokens=request.max_new_tokens, min_length=0)
    prompt_len = int(encoded["input_ids"].shape[-1])
    return model, tokenizer, {
        "generated_text": tokenizer.decode(output_ids[0], skip_special_tokens=True),
        "prompt_token_count": prompt_len,
        "generated_token_count": int(output_ids.shape[-1]),
    }


def _synthetic_causal_lm(model_path: Path, request: ClassicTransformersRequest) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=request.trust_remote_code)
    model = _load_transformers_model(AutoModelForCausalLM, model_path, request)
    if request.device and request.device != "cpu":
        model = model.to(request.device)
    model.eval()
    encoded = _move_batch(tokenizer(request.prompt, return_tensors="pt"), request.device)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=request.max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    prompt_len = int(encoded["input_ids"].shape[-1])
    total_len = int(output_ids.shape[-1])
    return model, tokenizer, {
        "generated_text": tokenizer.decode(output_ids[0], skip_special_tokens=True),
        "prompt_token_count": prompt_len,
        "generated_token_count": max(total_len - prompt_len, 0),
        "total_token_count": total_len,
    }


def _synthetic_text_encoder(model_path: Path, request: ClassicTransformersRequest) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=request.trust_remote_code)
    model = _load_transformers_model(AutoModel, model_path, request)
    if request.device and request.device != "cpu":
        model = model.to(request.device)
    model.eval()
    encoded = _move_batch(tokenizer(request.prompt, return_tensors="pt"), request.device)
    with torch.no_grad():
        outputs = model(**encoded)
    payload = {"last_hidden_state_shape": list(outputs.last_hidden_state.shape)}
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is not None:
        payload["pooler_output_shape"] = list(pooled.shape)
    return model, tokenizer, payload


def _synthetic_sequence_classification(model_path: Path, request: ClassicTransformersRequest) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=request.trust_remote_code)
    model = _load_transformers_model(AutoModelForSequenceClassification, model_path, request)
    if request.device and request.device != "cpu":
        model = model.to(request.device)
    model.eval()
    encoded = _move_batch(tokenizer(request.prompt, return_tensors="pt"), request.device)
    with torch.no_grad():
        outputs = model(**encoded)
    return model, tokenizer, {
        "logits_shape": list(outputs.logits.shape),
        "num_labels": int(getattr(model.config, "num_labels", outputs.logits.shape[-1])),
    }


def _synthetic_masked_lm(model_path: Path, request: ClassicTransformersRequest) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=request.trust_remote_code)
    model = _load_transformers_model(AutoModelForMaskedLM, model_path, request)
    if request.device and request.device != "cpu":
        model = model.to(request.device)
    model.eval()
    prompt = request.prompt
    mask_token = getattr(tokenizer, "mask_token", None)
    if mask_token and mask_token not in prompt:
        prompt = f"{prompt} {mask_token}"
    encoded = _move_batch(tokenizer(prompt, return_tensors="pt"), request.device)
    with torch.no_grad():
        outputs = model(**encoded)
    return model, tokenizer, {"logits_shape": list(outputs.logits.shape)}


def _synthetic_audio_encoder(model_path: Path, request: ClassicTransformersRequest) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    extractor = AutoFeatureExtractor.from_pretrained(str(model_path), trust_remote_code=request.trust_remote_code)
    model = _load_transformers_model(AutoModel, model_path, request)
    if request.device and request.device != "cpu":
        model = model.to(request.device)
    model.eval()
    sample_rate = int(getattr(extractor, "sampling_rate", 16000) or 16000)
    audio = torch.zeros(sample_rate, dtype=torch.float32).numpy()
    batch = extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
    batch = _move_batch(dict(batch), request.device)
    batch = _cast_floating_batch(batch, next(model.parameters()).dtype)
    with torch.no_grad():
        outputs = model(**batch)
    hidden = outputs.last_hidden_state
    return model, extractor, {"last_hidden_state_shape": list(hidden.shape), "sample_rate": sample_rate}


def _synthetic_vision_encoder(model_path: Path, request: ClassicTransformersRequest) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(
        str(model_path), trust_remote_code=request.trust_remote_code, use_fast=False
    )
    model = _load_transformers_model(AutoModel, model_path, request)
    if request.device and request.device != "cpu":
        model = model.to(request.device)
    model.eval()
    size = getattr(getattr(model, "config", None), "image_size", 224) or 224
    pixel_values = torch.zeros((1, 3, int(size), int(size)), dtype=torch.float32)
    batch = _move_batch({"pixel_values": pixel_values}, request.device)
    batch = _cast_floating_batch(batch, next(model.parameters()).dtype)
    with torch.no_grad():
        outputs = model(**batch)
    hidden = outputs.last_hidden_state
    return model, processor, {"last_hidden_state_shape": list(hidden.shape), "image_size": int(size)}


def inspect_classic_transformers(request: ClassicTransformersRequest) -> ClassicTransformersResult:
    try:
        model_path = resolve_model_path(request.model_id, request.model_root)
        config = read_model_config(model_path)
        task = infer_task(config, request.task)
        weights, tokenizers, processors = _artifact_files(model_path)
        model_class = None
        tokenizer_class = None
        processor_class = None
        synthetic_outputs = None
        status = "ready"
        error = None
        if request.run_synthetic:
            try:
                if task == "seq2seq_generation":
                    model, helper, synthetic_outputs = _synthetic_seq2seq(model_path, request)
                    tokenizer_class = type(helper).__name__
                elif task == "causal_lm_generation":
                    model, helper, synthetic_outputs = _synthetic_causal_lm(model_path, request)
                    tokenizer_class = type(helper).__name__
                elif task == "text_encoder":
                    model, helper, synthetic_outputs = _synthetic_text_encoder(model_path, request)
                    tokenizer_class = type(helper).__name__
                elif task == "sequence_classification":
                    model, helper, synthetic_outputs = _synthetic_sequence_classification(model_path, request)
                    tokenizer_class = type(helper).__name__
                elif task == "masked_lm":
                    model, helper, synthetic_outputs = _synthetic_masked_lm(model_path, request)
                    tokenizer_class = type(helper).__name__
                elif task == "audio_encoder":
                    model, helper, synthetic_outputs = _synthetic_audio_encoder(model_path, request)
                    processor_class = type(helper).__name__
                elif task == "vision_encoder":
                    model, helper, synthetic_outputs = _synthetic_vision_encoder(model_path, request)
                    processor_class = type(helper).__name__
                else:
                    raise ClassicTransformersBridgeError(f"unsupported synthetic task: {task}")
                model_class = type(model).__name__
                status = "ok"
            except Exception as exc:
                status = "synthetic_blocked"
                error = f"{type(exc).__name__}:{exc}"
        return ClassicTransformersResult(
            status=status,
            model_id=request.model_id,
            model_path=str(model_path),
            task=task,
            config_model_type=config.get("model_type"),
            architectures=tuple(config.get("architectures") or ()),
            transformers_version=config.get("transformers_version"),
            weight_files=weights,
            tokenizer_files=tokenizers,
            processor_files=processors,
            model_class=model_class,
            tokenizer_class=tokenizer_class,
            processor_class=processor_class,
            synthetic_outputs=synthetic_outputs,
            runtime=runtime_provenance(request),
            error=error,
        )
    except Exception as exc:
        return ClassicTransformersResult(status="failed", model_id=request.model_id, model_path=None, runtime=runtime_provenance(request), error=f"{type(exc).__name__}:{exc}")


def to_json(obj: Any) -> dict[str, Any]:
    return asdict(obj)
