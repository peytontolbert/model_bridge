from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from legacy_model_bridge.runtime.transformers_compat import ensure_modeling_utils_legacy_helpers
from legacy_model_bridge.runtime.world_model import load_cosmos_embed1


PATCHES_REQUIRED = ("transformers_apply_chunking_to_forward_compat", "transformers_modeling_utils_pruning_helpers_compat")


def detect(source_path: str | Path) -> bool:
    source = Path(source_path)
    if not source.exists() or not source.is_dir():
        return False
    config_path = source / "config.json"
    if not config_path.exists():
        return source.name == "Cosmos-Embed1-448p-anomaly-detection"
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError:
        return False
    return source.name == "Cosmos-Embed1-448p-anomaly-detection" or config.get("model_type") == "cosmos_embed1"


def apply_compatibility_patches() -> list[str]:
    ensure_modeling_utils_legacy_helpers()
    return list(PATCHES_REQUIRED)


def load_config(source_path: str | Path) -> dict[str, Any]:
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError("transformers is required to load Cosmos Embed1 config") from exc

    apply_compatibility_patches()
    return AutoConfig.from_pretrained(Path(source_path), trust_remote_code=True, local_files_only=True).to_dict()


def load_model(source_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    apply_compatibility_patches()
    return load_cosmos_embed1(source_path, dtype="float32", trust_remote_code=True, local_files_only=True, **kwargs)
