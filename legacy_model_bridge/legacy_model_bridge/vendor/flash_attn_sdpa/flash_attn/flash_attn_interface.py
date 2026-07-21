from __future__ import annotations

import torch
import torch.nn.functional as F


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, dropout_p: float = 0.0, causal: bool = False) -> torch.Tensor:
    q_bhsd = q.unsqueeze(0).transpose(1, 2)
    k_bhsd = k.unsqueeze(0).transpose(1, 2)
    v_bhsd = v.unsqueeze(0).transpose(1, 2)
    out = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd, dropout_p=dropout_p, is_causal=causal)
    return out.transpose(1, 2).squeeze(0)


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    **_: object,
) -> torch.Tensor | tuple[torch.Tensor, None, None]:
    if softmax_scale is not None:
        raise NotImplementedError("The legacy-model-bridge SDPA shim does not support custom softmax_scale.")
    if window_size != (-1, -1):
        raise NotImplementedError("The legacy-model-bridge SDPA shim does not support sliding-window attention.")
    if alibi_slopes is not None:
        raise NotImplementedError("The legacy-model-bridge SDPA shim does not support ALiBi slopes.")
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    cu_q = cu_seqlens_q.detach().to("cpu", torch.long)
    cu_k = cu_seqlens_k.detach().to("cpu", torch.long)
    if cu_q.numel() != cu_k.numel():
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same number of sequence boundaries.")

    chunks = []
    for idx in range(cu_q.numel() - 1):
        qs, qe = int(cu_q[idx]), int(cu_q[idx + 1])
        ks, ke = int(cu_k[idx]), int(cu_k[idx + 1])
        chunks.append(_sdpa(q[qs:qe], k[ks:ke], v[ks:ke], dropout_p=dropout_p, causal=causal))
    out = torch.cat(chunks, dim=0) if chunks else q.new_empty((0, *q.shape[1:]))
    if return_attn_probs:
        return out, None, None
    return out


def flash_attn_qkvpacked_func(qkv: torch.Tensor, dropout_p: float = 0.0, causal: bool = False, **kwargs: object) -> torch.Tensor:
    q, k, v = qkv.unbind(dim=-3)
    return _sdpa(q, k, v, dropout_p=dropout_p, causal=causal)


def flash_attn_kvpacked_func(q: torch.Tensor, kv: torch.Tensor, dropout_p: float = 0.0, causal: bool = False, **kwargs: object) -> torch.Tensor:
    k, v = kv.unbind(dim=-3)
    return _sdpa(q, k, v, dropout_p=dropout_p, causal=causal)


def flash_attn_varlen_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    **kwargs: object,
) -> torch.Tensor:
    k, v = kv.unbind(dim=-3)
    return flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs)
