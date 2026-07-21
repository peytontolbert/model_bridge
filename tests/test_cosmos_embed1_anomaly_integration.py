import json
import sys
import types
from pathlib import Path

from integrations.cosmos_embed1_448p_anomaly_detection.adapter import apply_compatibility_patches, detect
from legacy_model_bridge.runtime.transformers_compat import apply_chunking_to_forward


def test_cosmos_anomaly_detect_accepts_cosmos_embed1_config(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "cosmos_embed1"}))

    assert detect(tmp_path) is True


def test_cosmos_anomaly_detect_rejects_other_model_type(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "bert"}))

    assert detect(tmp_path) is False


def test_cosmos_anomaly_adapter_applies_chunking_patch(monkeypatch) -> None:
    def find_pruneable_heads_and_indices():
        return None

    def prune_linear_layer():
        return None

    transformers = types.ModuleType("transformers")
    modeling_utils = types.ModuleType("transformers.modeling_utils")
    pytorch_utils = types.ModuleType("transformers.pytorch_utils")
    pytorch_utils.apply_chunking_to_forward = apply_chunking_to_forward
    pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    pytorch_utils.prune_linear_layer = prune_linear_layer
    transformers.modeling_utils = modeling_utils
    transformers.pytorch_utils = pytorch_utils
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", modeling_utils)
    monkeypatch.setitem(sys.modules, "transformers.pytorch_utils", pytorch_utils)

    patches = apply_compatibility_patches()

    assert patches == [
        "transformers_apply_chunking_to_forward_compat",
        "transformers_modeling_utils_pruning_helpers_compat",
    ]
    assert modeling_utils.apply_chunking_to_forward is apply_chunking_to_forward
    assert modeling_utils.find_pruneable_heads_and_indices is find_pruneable_heads_and_indices
    assert modeling_utils.prune_linear_layer is prune_linear_layer
