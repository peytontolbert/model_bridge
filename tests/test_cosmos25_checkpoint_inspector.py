import pytest
import torch

from scripts.inspect_cosmos25_checkpoint import inspect_checkpoint


@pytest.mark.skipif(not hasattr(torch, "save"), reason="full torch serialization unavailable in this env")
def test_inspect_checkpoint_reports_prefix_sizes(tmp_path) -> None:
    checkpoint = tmp_path / "tiny.pt"
    torch.save(
        {
            "net.layer.weight": torch.zeros((2, 3), dtype=torch.bfloat16),
            "net.blocks.0.layer.weight": torch.zeros((2, 3), dtype=torch.bfloat16),
            "net_fake_score.layer.weight": torch.zeros((2, 3), dtype=torch.float32),
        },
        checkpoint,
    )

    payload = inspect_checkpoint(checkpoint)

    assert payload["format"] == "pytorch_zip_pt"
    assert payload["prefix_counts"]["net"] == 2
    assert payload["prefix_counts"]["net_fake_score"] == 1
    assert payload["nbytes_by_prefix"]["net"] == 24
    assert payload["nbytes_by_prefix"]["net_fake_score"] == 24
    assert payload["nbytes_by_block"]["net.blocks.0"] == 12
