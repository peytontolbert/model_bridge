from __future__ import annotations

from pathlib import Path
from typing import Any

from legacy_model_bridge.runtime.encoder_classifier import (
    EncoderClassifierBridgeError,
    load_repaired_sequence_classifier,
)


REQUIRED_FILES = ("config.json",)
CHECKPOINT_FILES = ("model.safetensors", "pytorch_model.bin")


def detect(source_path: str | Path) -> bool:
    source = Path(source_path)
    if not source.exists() or not source.is_dir():
        return False
    if not all((source / name).exists() for name in REQUIRED_FILES):
        return False
    return any((source / name).exists() for name in CHECKPOINT_FILES) or any(source.glob("*.safetensors"))


def load_config(source_path: str | Path) -> dict[str, Any]:
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise EncoderClassifierBridgeError("transformers is required to load abstract-repo-planning config") from exc

    return AutoConfig.from_pretrained(Path(source_path)).to_dict()


def load_model(source_path: str | Path, **kwargs: Any) -> Any:
    return load_repaired_sequence_classifier(source_path, **kwargs)
