from __future__ import annotations

from pathlib import Path
from typing import Any

from .transformers_compat import ensure_modeling_utils_legacy_helpers


class WorldModelBridgeError(RuntimeError):
    pass


def load_cosmos_embed1(
    source_path: str | Path,
    *,
    dtype: str = "float32",
    device: str | None = None,
    trust_remote_code: bool = True,
    local_files_only: bool = True,
    apply_compat_patches: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    if apply_compat_patches:
        ensure_modeling_utils_legacy_helpers()
    try:
        import torch
        from transformers import AutoConfig, AutoModel, AutoProcessor
    except ImportError as exc:
        raise WorldModelBridgeError("torch and transformers are required to load Cosmos Embed1 models") from exc

    source = Path(source_path)
    torch_dtype = getattr(torch, dtype)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = AutoConfig.from_pretrained(source, trust_remote_code=trust_remote_code, local_files_only=local_files_only)
    processor = AutoProcessor.from_pretrained(source, trust_remote_code=trust_remote_code, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(
        source,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
        **kwargs,
    )
    model = model.to(selected_device)
    if hasattr(model, "eval"):
        model.eval()
    return {"model": model, "processor": processor, "config": config, "device": selected_device, "dtype": torch_dtype}
