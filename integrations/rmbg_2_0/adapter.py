from __future__ import annotations

from pathlib import Path
from typing import Any


REQUIRED_FILES = ("config.json",)
CHECKPOINT_FILES = ("model.safetensors", "pytorch_model.bin")


class RMBGBridgeError(RuntimeError):
    pass


def detect(source_path: str | Path) -> bool:
    source = Path(source_path)
    if not source.exists() or not source.is_dir():
        return False
    if not all((source / name).exists() for name in REQUIRED_FILES):
        return False
    return any((source / name).exists() for name in CHECKPOINT_FILES)


def load_config(source_path: str | Path) -> dict[str, Any]:
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RMBGBridgeError("transformers is required to load RMBG config") from exc

    return AutoConfig.from_pretrained(Path(source_path), trust_remote_code=True, local_files_only=True).to_dict()


def load_model(source_path: str | Path, *, device: str | None = None, **kwargs: Any) -> Any:
    try:
        import torch
        from transformers import AutoModelForImageSegmentation
    except ImportError as exc:
        raise RMBGBridgeError("torch, transformers, timm, and kornia are required to load RMBG") from exc

    model = AutoModelForImageSegmentation.from_pretrained(
        Path(source_path),
        trust_remote_code=True,
        local_files_only=True,
        **kwargs,
    ).eval()
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return model.to(selected_device)
