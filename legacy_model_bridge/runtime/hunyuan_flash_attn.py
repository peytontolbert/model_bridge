from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


BRIDGE_ROOT = Path(__file__).resolve().parents[2]
FLASH_ATTN_SDPA_SHIM_ROOT = BRIDGE_ROOT / "legacy_model_bridge" / "vendor" / "flash_attn_sdpa"


@dataclass(frozen=True)
class FlashAttentionCompatibility:
    status: str
    flash_attn_available: bool
    shim_available: bool
    selected_backend: str
    shim_root: str
    detail: str
    blockers: tuple[str, ...] = ()
    versions: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def flash_attn_spec_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def install_sdpa_flash_attn_shim(*, prefer_real: bool = True) -> bool:
    if prefer_real and flash_attn_spec_available():
        return False
    root = str(FLASH_ATTN_SDPA_SHIM_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return True


def inspect_flash_attention_compatibility(*, allow_sdpa_shim: bool = True) -> FlashAttentionCompatibility:
    versions: dict[str, str] = {}
    try:
        import torch

        versions["torch"] = getattr(torch, "__version__", "unknown")
        versions["torch_cuda"] = str(getattr(torch.version, "cuda", None))
        versions["torch_cxx11_abi"] = str(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", "unknown"))
    except Exception as exc:
        return FlashAttentionCompatibility(
            status="missing_torch",
            flash_attn_available=False,
            shim_available=FLASH_ATTN_SDPA_SHIM_ROOT.exists(),
            selected_backend="none",
            shim_root=str(FLASH_ATTN_SDPA_SHIM_ROOT),
            detail="Torch is required before either native flash-attn or the SDPA shim can be used.",
            blockers=(f"torch import failed: {type(exc).__name__}:{exc}",),
            versions=versions,
        )

    if flash_attn_spec_available():
        try:
            import flash_attn

            versions["flash_attn"] = getattr(flash_attn, "__version__", "unknown")
            versions["flash_attn_file"] = str(getattr(flash_attn, "__file__", "unknown"))
            return FlashAttentionCompatibility(
                status="native_flash_attn_available",
                flash_attn_available=True,
                shim_available=FLASH_ATTN_SDPA_SHIM_ROOT.exists(),
                selected_backend="native_flash_attn",
                shim_root=str(FLASH_ATTN_SDPA_SHIM_ROOT),
                detail="Native flash-attn imports in this environment.",
                versions=versions,
            )
        except Exception as exc:
            return FlashAttentionCompatibility(
                status="native_flash_attn_import_error",
                flash_attn_available=True,
                shim_available=FLASH_ATTN_SDPA_SHIM_ROOT.exists(),
                selected_backend="none",
                shim_root=str(FLASH_ATTN_SDPA_SHIM_ROOT),
                detail="A flash_attn module is discoverable but fails to import.",
                blockers=(f"flash_attn import failed: {type(exc).__name__}:{exc}",),
                versions=versions,
            )

    if allow_sdpa_shim and FLASH_ATTN_SDPA_SHIM_ROOT.exists():
        install_sdpa_flash_attn_shim(prefer_real=False)
        return FlashAttentionCompatibility(
            status="sdpa_flash_attn_shim_available",
            flash_attn_available=False,
            shim_available=True,
            selected_backend="torch_sdpa_shim",
            shim_root=str(FLASH_ATTN_SDPA_SHIM_ROOT),
            detail=(
                "Native flash-attn is absent; the bridge can expose a minimal flash_attn_varlen_func "
                "compatibility module backed by torch.nn.functional.scaled_dot_product_attention."
            ),
            versions=versions,
        )

    return FlashAttentionCompatibility(
        status="flash_attn_missing",
        flash_attn_available=False,
        shim_available=FLASH_ATTN_SDPA_SHIM_ROOT.exists(),
        selected_backend="none",
        shim_root=str(FLASH_ATTN_SDPA_SHIM_ROOT),
        detail="Native flash-attn is absent and the SDPA shim is disabled or unavailable.",
        blockers=("flash_attn missing",),
        versions=versions,
    )
