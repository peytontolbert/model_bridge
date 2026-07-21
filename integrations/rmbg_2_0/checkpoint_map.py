from __future__ import annotations

from pathlib import Path
from typing import Any


TENSOR_KEY_MAP: dict[str, str] = {}


def map_checkpoint_key(key: str) -> str:
    return TENSOR_KEY_MAP.get(key, key)


def convert_checkpoint(source_path: str | Path, out_path: str | Path) -> dict[str, Any]:
    raise NotImplementedError('rmbg_2_0' + ' checkpoint conversion is not implemented yet')
