#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legacy_model_bridge.runtime.hunyuan_flash_attn import inspect_flash_attention_compatibility
from legacy_model_bridge.runtime.hunyuan_flash_attn import install_sdpa_flash_attn_shim


def _smoke_varlen(selected_backend: str) -> dict[str, object]:
    install_sdpa_flash_attn_shim(prefer_real=True)
    import torch
    from flash_attn.flash_attn_interface import flash_attn_varlen_func

    use_cuda = selected_backend == "native_flash_attn" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float16 if use_cuda else torch.float32
    q = torch.randn(5, 2, 8, dtype=dtype, device=device)
    k = torch.randn(5, 2, 8, dtype=dtype, device=device)
    v = torch.randn(5, 2, 8, dtype=dtype, device=device)
    cu = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
    out = flash_attn_varlen_func(q, k, v, cu, cu, 3, 3)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "shape": list(out.shape),
        "dtype": str(out.dtype),
        "device": str(out.device),
        "finite": bool(torch.isfinite(out).all().item()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect Hunyuan Avatar flash-attn compatibility.")
    parser.add_argument("--no-sdpa-shim", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    result = inspect_flash_attention_compatibility(allow_sdpa_shim=not args.no_sdpa_shim).to_dict()
    if args.smoke and result["selected_backend"] != "none":
        result["smoke"] = _smoke_varlen(str(result["selected_backend"]))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["selected_backend"] != "none" else 2


if __name__ == "__main__":
    raise SystemExit(main())
