import importlib.util

import pytest
import torch

from scripts.convert_cosmos25_pt_to_safetensors import convert_checkpoint


@pytest.mark.skipif(not hasattr(torch, "save"), reason="full torch serialization unavailable in this env")
@pytest.mark.skipif(importlib.util.find_spec("safetensors") is None, reason="safetensors unavailable")
def test_convert_checkpoint_filters_and_shards_net_prefix(tmp_path) -> None:
    checkpoint = tmp_path / "tiny.pt"
    torch.save(
        {
            "net.a": torch.zeros((2, 2), dtype=torch.float32),
            "net.b": torch.zeros((2, 2), dtype=torch.float32),
            "net_teacher.a": torch.ones((2, 2), dtype=torch.float32),
        },
        checkpoint,
    )

    payload = convert_checkpoint(
        checkpoint,
        tmp_path / "out",
        include_prefix="net",
        strip_prefix=True,
        max_shard_size="20B",
    )

    assert payload["tensor_count"] == 2
    assert len(payload["shard_files"]) == 2
    assert (tmp_path / "out" / "model.safetensors.index.json").is_file()
