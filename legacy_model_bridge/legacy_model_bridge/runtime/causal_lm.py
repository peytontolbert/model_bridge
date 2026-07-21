from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from legacy_model_bridge.runtime.transformers_compat import (
    PATCH_ID_MOBILELLM_LEGACY_CACHE,
    PATCH_ID_MOBILELLM_SLOW_TOKENIZER,
    is_non_callable_tokenizer_failure,
    mobilellm_requires_no_cache,
)

DEFAULT_MODEL_ROOT = Path("/arxiv/models")


@dataclass(frozen=True)
class CausalLMRequest:
    model_id: str
    prompt: str = "Hello"
    model_root: str = str(DEFAULT_MODEL_ROOT)
    max_new_tokens: int = 4
    device: str = "cuda:0"
    dtype: str = "auto"
    trust_remote_code: bool | None = None
    use_cache: bool | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class CausalLMResult:
    status: str
    model_id: str
    model_path: str | None
    prompt: str
    generated_text: str | None = None
    prompt_token_count: int | None = None
    generated_token_count: int | None = None
    tokenizer_class: str | None = None
    model_class: str | None = None
    device: str | None = None
    dtype: str | None = None
    trust_remote_code: bool | None = None
    use_cache: bool | None = None
    applied_patches: tuple[str, ...] = ()
    error: str | None = None
    dry_run: bool = False


class CausalLMBridgeError(RuntimeError):
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
    raise CausalLMBridgeError(f"model path not found for {model_id!r} under {root}")


def read_model_config(model_path: str | Path) -> dict[str, Any]:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        raise CausalLMBridgeError(f"missing config.json: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def should_trust_remote_code(config: dict[str, Any], requested: bool | None) -> bool:
    if requested is not None:
        return requested
    return bool(config.get("auto_map"))


def should_use_cache(config: dict[str, Any], requested: bool | None) -> bool:
    if requested is not None:
        return requested
    if mobilellm_requires_no_cache(config):
        return False
    return bool(config.get("use_cache", True))


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
        raise CausalLMBridgeError(f"unsupported dtype: {dtype}")
    return aliases[dtype]


def load_tokenizer_with_fallback(model_path: str | Path, *, trust_remote_code: bool) -> tuple[Any, tuple[str, ...]]:
    from transformers import AutoTokenizer

    patches: list[str] = []
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=trust_remote_code, use_fast=True)
        if callable(tokenizer):
            return tokenizer, tuple(patches)
        if not is_non_callable_tokenizer_failure(type(tokenizer).__name__) and tokenizer is not False:
            raise CausalLMBridgeError(f"AutoTokenizer returned non-callable {type(tokenizer).__name__}")
    except Exception as first_exc:
        tokenizer_model = Path(model_path) / "tokenizer.model"
        if not tokenizer_model.is_file():
            raise first_exc
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained(str(model_path), legacy=True, use_fast=False)
    patches.append(PATCH_ID_MOBILELLM_SLOW_TOKENIZER)
    return tokenizer, tuple(patches)


def generate_causal_lm(request: CausalLMRequest) -> CausalLMResult:
    model_path: Path | None = None
    try:
        model_path = resolve_model_path(request.model_id, request.model_root)
        config = read_model_config(model_path)
        trust_remote_code = should_trust_remote_code(config, request.trust_remote_code)
        use_cache = should_use_cache(config, request.use_cache)
        patches: list[str] = []
        if mobilellm_requires_no_cache(config) and use_cache is False:
            patches.append(PATCH_ID_MOBILELLM_LEGACY_CACHE)
        if request.dry_run:
            return CausalLMResult(
                status="dry_run",
                model_id=request.model_id,
                model_path=str(model_path),
                prompt=request.prompt,
                device=request.device,
                dtype=request.dtype,
                trust_remote_code=trust_remote_code,
                use_cache=use_cache,
                applied_patches=tuple(patches),
                dry_run=True,
            )

        from transformers import AutoModelForCausalLM

        tokenizer, tokenizer_patches = load_tokenizer_with_fallback(model_path, trust_remote_code=trust_remote_code)
        patches.extend(item for item in tokenizer_patches if item not in patches)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=trust_remote_code,
            torch_dtype=_torch_dtype(request.dtype),
        )
        if request.device and request.device != "cpu":
            model = model.to(request.device)
        elif request.device == "cpu":
            model = model.cpu()
        if hasattr(model, "eval"):
            model.eval()
        encoded = tokenizer(request.prompt, return_tensors="pt")
        if request.device and request.device != "cpu":
            encoded = {key: value.to(request.device) for key, value in encoded.items()}
        output_ids = model.generate(**encoded, max_new_tokens=request.max_new_tokens, use_cache=use_cache)
        input_len = int(encoded["input_ids"].shape[-1])
        output_len = int(output_ids.shape[-1])
        return CausalLMResult(
            status="ok",
            model_id=request.model_id,
            model_path=str(model_path),
            prompt=request.prompt,
            generated_text=tokenizer.decode(output_ids[0], skip_special_tokens=True),
            prompt_token_count=input_len,
            generated_token_count=max(0, output_len - input_len),
            tokenizer_class=type(tokenizer).__name__,
            model_class=type(model).__name__,
            device=request.device,
            dtype=request.dtype,
            trust_remote_code=trust_remote_code,
            use_cache=use_cache,
            applied_patches=tuple(patches),
        )
    except Exception as exc:
        return CausalLMResult(
            status="failed",
            model_id=request.model_id,
            model_path=str(model_path) if model_path is not None else None,
            prompt=request.prompt,
            error=f"{type(exc).__name__}:{exc}",
            dry_run=request.dry_run,
        )


def to_json(obj: Any) -> dict[str, Any]:
    return asdict(obj)
