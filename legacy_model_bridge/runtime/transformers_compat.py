from __future__ import annotations

from collections.abc import Callable
from typing import Any


PATCH_ID_APPLY_CHUNKING = "transformers_apply_chunking_to_forward_compat"
PATCH_ID_MODELING_UTILS_PRUNING = "transformers_modeling_utils_pruning_helpers_compat"
PATCH_ID_MOBILELLM_LEGACY_CACHE = "transformers_mobilellm_legacy_cache"
PATCH_ID_MOBILELLM_SLOW_TOKENIZER = "transformers_mobilellm_slow_tokenizer"


class TransformersCompatError(RuntimeError):
    pass


def _shape_at(tensor: Any, dim: int) -> int:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        try:
            return int(tensor.size(dim))
        except AttributeError as exc:
            raise TransformersCompatError(f"input tensor has no shape or size(): {type(tensor).__name__}") from exc
    try:
        return int(shape[dim])
    except IndexError as exc:
        raise TransformersCompatError(f"chunk_dim {dim} is out of bounds for shape {tuple(shape)!r}") from exc


def _chunk_tensor(tensor: Any, chunks: int, dim: int) -> tuple[Any, ...]:
    if hasattr(tensor, "chunk"):
        return tuple(tensor.chunk(chunks, dim=dim))
    raise TransformersCompatError(f"input tensor does not implement chunk(): {type(tensor).__name__}")


def apply_chunking_to_forward(
    forward_fn: Callable[..., Any],
    chunk_size: int,
    chunk_dim: int,
    *input_tensors: Any,
) -> Any:
    """Compatibility shim for the Transformers helper removed from modeling_utils.

    This mirrors the behavior remote-code models usually rely on: split every
    input tensor into equal chunks along `chunk_dim`, run `forward_fn` on each
    chunk group, and concatenate the outputs along the same dimension.
    """
    if chunk_size <= 0:
        return forward_fn(*input_tensors)
    if not input_tensors:
        raise TransformersCompatError("apply_chunking_to_forward requires at least one input tensor")

    tensor_shape = _shape_at(input_tensors[0], chunk_dim)
    if any(_shape_at(tensor, chunk_dim) != tensor_shape for tensor in input_tensors):
        raise TransformersCompatError("all input tensors must have the same shape along chunk_dim")
    if tensor_shape % chunk_size != 0:
        raise TransformersCompatError(
            f"input dimension {tensor_shape} must be divisible by chunk_size {chunk_size}"
        )

    num_chunks = tensor_shape // chunk_size
    chunked_inputs = tuple(_chunk_tensor(tensor, num_chunks, chunk_dim) for tensor in input_tensors)
    output_chunks = tuple(forward_fn(*chunk_group) for chunk_group in zip(*chunked_inputs))
    if not output_chunks:
        raise TransformersCompatError("chunked forward produced no outputs")

    first = output_chunks[0]
    cat_owner = None
    try:
        import torch

        cat_owner = torch
    except ImportError:
        cat_owner = getattr(first, "__class__", None)

    if cat_owner is not None and hasattr(cat_owner, "cat"):
        return cat_owner.cat(output_chunks, dim=chunk_dim)
    if hasattr(first, "concatenate"):
        return first.concatenate(output_chunks, axis=chunk_dim)
    raise TransformersCompatError("cannot concatenate chunked outputs without torch.cat or compatible concatenate")


def _modeling_and_pytorch_utils() -> tuple[Any, Any | None]:
    try:
        import transformers.modeling_utils as modeling_utils
    except ImportError as exc:
        raise TransformersCompatError("transformers is required to patch modeling_utils") from exc
    try:
        from transformers import pytorch_utils
    except ImportError:
        pytorch_utils = None
    return modeling_utils, pytorch_utils


def ensure_apply_chunking_to_forward(*, force: bool = False) -> bool:
    """Install the chunking shim into `transformers.modeling_utils` when missing."""
    modeling_utils, pytorch_utils = _modeling_and_pytorch_utils()
    if hasattr(modeling_utils, "apply_chunking_to_forward") and not force:
        return False
    shim = apply_chunking_to_forward
    if pytorch_utils is not None:
        shim = getattr(pytorch_utils, "apply_chunking_to_forward", apply_chunking_to_forward)
    modeling_utils.apply_chunking_to_forward = shim
    return True


def ensure_modeling_utils_pruning_helpers(*, force: bool = False) -> list[str]:
    """Alias legacy pruning helpers back onto `transformers.modeling_utils`."""
    modeling_utils, pytorch_utils = _modeling_and_pytorch_utils()
    if pytorch_utils is None:
        raise TransformersCompatError("transformers.pytorch_utils is required to patch pruning helpers")
    installed: list[str] = []
    for name in ("find_pruneable_heads_and_indices", "prune_linear_layer"):
        if hasattr(modeling_utils, name) and not force:
            continue
        if not hasattr(pytorch_utils, name):
            raise TransformersCompatError(f"transformers.pytorch_utils.{name} is not available")
        setattr(modeling_utils, name, getattr(pytorch_utils, name))
        installed.append(name)
    return installed


def ensure_modeling_utils_legacy_helpers(*, force: bool = False) -> list[str]:
    installed: list[str] = []
    if ensure_apply_chunking_to_forward(force=force):
        installed.append(PATCH_ID_APPLY_CHUNKING)
    if ensure_modeling_utils_pruning_helpers(force=force):
        installed.append(PATCH_ID_MODELING_UTILS_PRUNING)
    return installed


def mobilellm_requires_no_cache(config: dict[str, Any]) -> bool:
    model_type = str(config.get("model_type", "")).lower()
    architectures = {str(item).lower() for item in config.get("architectures", [])}
    return model_type == "mobilellm" or any("mobilellm" in item for item in architectures)


def is_non_callable_tokenizer_failure(error: BaseException | str) -> bool:
    text = str(error).lower()
    return "not callable" in text or "sentencepiece" in text or "tokenizer.model" in text
