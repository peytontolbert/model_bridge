import sys
import types

import pytest

from legacy_model_bridge.runtime.transformers_compat import (
    TransformersCompatError,
    apply_chunking_to_forward,
    ensure_apply_chunking_to_forward,
    ensure_modeling_utils_legacy_helpers,
    ensure_modeling_utils_pruning_helpers,
)


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values),)

    def chunk(self, chunks: int, dim: int = 0):
        assert dim == 0
        size = len(self.values) // chunks
        return tuple(FakeTensor(self.values[index * size : (index + 1) * size]) for index in range(chunks))

    @classmethod
    def cat(cls, tensors, dim: int = 0):
        assert dim == 0
        values = []
        for tensor in tensors:
            values.extend(tensor.values)
        return cls(values)


def test_apply_chunking_to_forward_chunks_and_concatenates(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", FakeTensor)

    result = apply_chunking_to_forward(lambda x: FakeTensor([value * 2 for value in x.values]), 2, 0, FakeTensor([1, 2, 3, 4]))

    assert result.values == [2, 4, 6, 8]


def test_apply_chunking_to_forward_zero_chunk_calls_directly() -> None:
    result = apply_chunking_to_forward(lambda x: x + 1, 0, 0, 4)

    assert result == 5


def test_apply_chunking_to_forward_rejects_non_divisible_shape() -> None:
    with pytest.raises(TransformersCompatError, match="divisible"):
        apply_chunking_to_forward(lambda x: x, 2, 0, FakeTensor([1, 2, 3]))


def test_ensure_apply_chunking_to_forward_installs_missing_symbol(monkeypatch) -> None:
    transformers = types.ModuleType("transformers")
    modeling_utils = types.ModuleType("transformers.modeling_utils")
    transformers.modeling_utils = modeling_utils
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", modeling_utils)

    installed = ensure_apply_chunking_to_forward()

    assert installed is True
    assert modeling_utils.apply_chunking_to_forward is apply_chunking_to_forward


def test_ensure_apply_chunking_to_forward_preserves_existing_symbol(monkeypatch) -> None:
    def existing():
        return None

    transformers = types.ModuleType("transformers")
    modeling_utils = types.ModuleType("transformers.modeling_utils")
    modeling_utils.apply_chunking_to_forward = existing
    transformers.modeling_utils = modeling_utils
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", modeling_utils)

    installed = ensure_apply_chunking_to_forward()

    assert installed is False
    assert modeling_utils.apply_chunking_to_forward is existing


def test_ensure_pruning_helpers_aliases_from_pytorch_utils(monkeypatch) -> None:
    def find_pruneable_heads_and_indices():
        return None

    def prune_linear_layer():
        return None

    transformers = types.ModuleType("transformers")
    modeling_utils = types.ModuleType("transformers.modeling_utils")
    pytorch_utils = types.ModuleType("transformers.pytorch_utils")
    pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    pytorch_utils.prune_linear_layer = prune_linear_layer
    transformers.modeling_utils = modeling_utils
    transformers.pytorch_utils = pytorch_utils
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", modeling_utils)
    monkeypatch.setitem(sys.modules, "transformers.pytorch_utils", pytorch_utils)

    installed = ensure_modeling_utils_pruning_helpers()

    assert installed == ["find_pruneable_heads_and_indices", "prune_linear_layer"]
    assert modeling_utils.find_pruneable_heads_and_indices is find_pruneable_heads_and_indices
    assert modeling_utils.prune_linear_layer is prune_linear_layer


def test_ensure_modeling_utils_legacy_helpers_reports_patch_ids(monkeypatch) -> None:
    def find_pruneable_heads_and_indices():
        return None

    def prune_linear_layer():
        return None

    transformers = types.ModuleType("transformers")
    modeling_utils = types.ModuleType("transformers.modeling_utils")
    pytorch_utils = types.ModuleType("transformers.pytorch_utils")
    pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    pytorch_utils.prune_linear_layer = prune_linear_layer
    pytorch_utils.apply_chunking_to_forward = apply_chunking_to_forward
    transformers.modeling_utils = modeling_utils
    transformers.pytorch_utils = pytorch_utils
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", modeling_utils)
    monkeypatch.setitem(sys.modules, "transformers.pytorch_utils", pytorch_utils)

    installed = ensure_modeling_utils_legacy_helpers()

    assert installed == [
        "transformers_apply_chunking_to_forward_compat",
        "transformers_modeling_utils_pruning_helpers_compat",
    ]
