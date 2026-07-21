from __future__ import annotations

import importlib
import importlib.metadata as metadata
import importlib.util
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SageAttentionStatus:
    available: bool
    distribution_version: str | None
    module_version: str | None
    torch_version: str | None
    cuda_version: str | None
    device_name: str | None
    capability: tuple[int, int] | None
    has_fp16_triton: bool
    has_fp16_cuda: bool
    has_fp8_cuda: bool
    has_varlen: bool
    sageattn3_available: bool
    recommended_abot_backend: str
    upgrade_recommendation: str
    detail: str


@dataclass(frozen=True)
class SageAttentionSmoke:
    status: str
    dtype: str
    layout: str
    shape: tuple[int, ...]
    elapsed_sec: float | None = None
    max_abs: float | None = None
    mean_abs: float | None = None
    error: str | None = None


def sageattention_status(device: int | str | None = None) -> SageAttentionStatus:
    module = None
    available = importlib.util.find_spec("sageattention") is not None
    if available:
        try:
            module = importlib.import_module("sageattention")
        except Exception:
            available = False
            module = None
    torch_version = None
    cuda_version = None
    device_name = None
    capability = None
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_version = str(torch.version.cuda)
        if torch.cuda.is_available():
            dev = _resolve_device(device)
            device_name = torch.cuda.get_device_name(dev)
            capability = torch.cuda.get_device_capability(dev)
    except Exception:
        pass
    has_fp16_triton = bool(module is not None and hasattr(module, "sageattn_qk_int8_pv_fp16_triton"))
    has_fp16_cuda = bool(module is not None and hasattr(module, "sageattn_qk_int8_pv_fp16_cuda"))
    has_fp8_cuda = bool(module is not None and hasattr(module, "sageattn_qk_int8_pv_fp8_cuda"))
    has_varlen = bool(module is not None and hasattr(module, "sageattn_varlen"))
    sageattn3_available = importlib.util.find_spec("sageattn3") is not None
    dist_version = _dist_version("sageattention")
    module_version = str(getattr(module, "__version__", "")) or None if module is not None else None
    current_kernel = has_fp16_triton or has_fp16_cuda or has_fp8_cuda
    if available and current_kernel and capability == (8, 6):
        recommendation = "do_not_upgrade_for_rtx3090"
        backend = "sageattn"
        detail = "SageAttention2 sm80/sm86 kernels are installed and appropriate for RTX 3090; sageattn3 is Blackwell-only."
    elif available and current_kernel:
        recommendation = "keep_installed_unless_regression_fails"
        backend = "sageattn"
        detail = "SageAttention kernels are installed; run kernel smokes after any Torch/CUDA change."
    elif available:
        recommendation = "rebuild_sageattention_from_source"
        backend = "sdpa"
        detail = "sageattention imports but lacks expected compiled kernel symbols."
    else:
        recommendation = "install_sageattention_from_source"
        backend = "sdpa"
        detail = "sageattention is not importable."
    return SageAttentionStatus(
        available=available,
        distribution_version=dist_version,
        module_version=module_version,
        torch_version=torch_version,
        cuda_version=cuda_version,
        device_name=device_name,
        capability=capability,
        has_fp16_triton=has_fp16_triton,
        has_fp16_cuda=has_fp16_cuda,
        has_fp8_cuda=has_fp8_cuda,
        has_varlen=has_varlen,
        sageattn3_available=sageattn3_available,
        recommended_abot_backend=backend,
        upgrade_recommendation=recommendation,
        detail=detail,
    )


def smoke_sageattention_kernel(
    *,
    dtype: str = "bfloat16",
    layout: str = "NHD",
    shape: tuple[int, ...] = (1, 512, 16, 128),
) -> SageAttentionSmoke:
    try:
        import time

        import torch
        from torch.nn import functional as F
        from sageattention import sageattn

        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
        torch.manual_seed(0)
        q = torch.randn(shape, device="cuda", dtype=torch_dtype)
        k = torch.randn(shape, device="cuda", dtype=torch_dtype)
        v = torch.randn(shape, device="cuda", dtype=torch_dtype)
        torch.cuda.synchronize()
        started = time.perf_counter()
        out = sageattn(q, k, v, tensor_layout=layout, is_causal=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if layout == "NHD":
            q_ref, k_ref, v_ref, out_ref = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), out.transpose(1, 2)
        elif layout == "HND":
            q_ref, k_ref, v_ref, out_ref = q, k, v, out
        else:
            raise ValueError(f"unsupported layout: {layout}")
        expected = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=False)
        diff = (out_ref.float() - expected.float()).abs()
        return SageAttentionSmoke(
            status="ok",
            dtype=str(torch_dtype),
            layout=layout,
            shape=tuple(shape),
            elapsed_sec=elapsed,
            max_abs=float(diff.max()),
            mean_abs=float(diff.mean()),
        )
    except Exception as exc:
        return SageAttentionSmoke(status="failed", dtype=dtype, layout=layout, shape=tuple(shape), error=f"{type(exc).__name__}:{exc}")


def _resolve_device(device: int | str | None) -> int:
    if device is None:
        return 0
    if isinstance(device, str):
        import torch

        return int(torch.device(device).index or 0)
    return int(device)


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def to_json(obj: Any) -> dict[str, Any]:
    return asdict(obj)
