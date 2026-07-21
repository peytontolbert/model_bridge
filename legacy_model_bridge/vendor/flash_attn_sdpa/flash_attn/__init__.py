from __future__ import annotations

from .flash_attn_interface import flash_attn_kvpacked_func
from .flash_attn_interface import flash_attn_qkvpacked_func
from .flash_attn_interface import flash_attn_varlen_func
from .flash_attn_interface import flash_attn_varlen_kvpacked_func

__version__ = "0.0.0+legacy-model-bridge-sdpa"

__all__ = [
    "flash_attn_kvpacked_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_varlen_func",
    "flash_attn_varlen_kvpacked_func",
]
