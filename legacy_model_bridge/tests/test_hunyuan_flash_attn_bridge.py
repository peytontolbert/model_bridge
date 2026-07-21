import importlib
import sys

import pytest
import torch

from legacy_model_bridge.runtime.hunyuan_flash_attn import FLASH_ATTN_SDPA_SHIM_ROOT
from legacy_model_bridge.runtime.hunyuan_flash_attn import inspect_flash_attention_compatibility
from legacy_model_bridge.runtime.hunyuan_flash_attn import install_sdpa_flash_attn_shim



def _require_torch_sdpa():
    pytest.importorskip("torch.nn.functional")


def test_sdpa_flash_attn_shim_exports_varlen_func(monkeypatch) -> None:
    _require_torch_sdpa()
    monkeypatch.setattr("legacy_model_bridge.runtime.hunyuan_flash_attn.flash_attn_spec_available", lambda: False)
    monkeypatch.syspath_prepend(str(FLASH_ATTN_SDPA_SHIM_ROOT))
    sys.modules.pop("flash_attn", None)
    sys.modules.pop("flash_attn.flash_attn_interface", None)

    inserted = install_sdpa_flash_attn_shim(prefer_real=True)

    assert inserted is True
    module = importlib.import_module("flash_attn.flash_attn_interface")
    assert hasattr(module, "flash_attn_varlen_func")


def test_sdpa_flash_attn_varlen_matches_torch_sdpa(monkeypatch) -> None:
    _require_torch_sdpa()
    monkeypatch.syspath_prepend(str(FLASH_ATTN_SDPA_SHIM_ROOT))
    sys.modules.pop("flash_attn", None)
    sys.modules.pop("flash_attn.flash_attn_interface", None)
    from flash_attn.flash_attn_interface import flash_attn_varlen_func

    q = torch.randn(5, 2, 8)
    k = torch.randn(5, 2, 8)
    v = torch.randn(5, 2, 8)
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)

    out = flash_attn_varlen_func(q, k, v, cu, cu, 3, 3)
    expected = torch.cat(
        [
            torch.nn.functional.scaled_dot_product_attention(
                q[:2].unsqueeze(0).transpose(1, 2),
                k[:2].unsqueeze(0).transpose(1, 2),
                v[:2].unsqueeze(0).transpose(1, 2),
            ).transpose(1, 2).squeeze(0),
            torch.nn.functional.scaled_dot_product_attention(
                q[2:].unsqueeze(0).transpose(1, 2),
                k[2:].unsqueeze(0).transpose(1, 2),
                v[2:].unsqueeze(0).transpose(1, 2),
            ).transpose(1, 2).squeeze(0),
        ],
        dim=0,
    )

    assert torch.allclose(out, expected)


def test_inspector_reports_shim_when_native_missing(monkeypatch) -> None:
    _require_torch_sdpa()
    monkeypatch.setattr("legacy_model_bridge.runtime.hunyuan_flash_attn.flash_attn_spec_available", lambda: False)

    result = inspect_flash_attention_compatibility()

    assert result.selected_backend == "torch_sdpa_shim"
    assert result.status == "sdpa_flash_attn_shim_available"
