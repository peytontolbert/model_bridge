from __future__ import annotations

import torch


def index_first_axis(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.index_select(x, 0, indices.to(device=x.device, dtype=torch.long))
