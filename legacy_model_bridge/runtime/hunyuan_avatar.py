from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_CKPT_DIR = Path("/arxiv/models/HunyuanVideo-Avatar/ckpts")
DEFAULT_LLAVA_DIR = DEFAULT_CKPT_DIR / "llava_llama_image"
IMAGE_PLACEHOLDER = "<image>"


@dataclass(frozen=True)
class HunyuanAvatarLlavaAlignment:
    status: str
    runnable: bool
    llava_dir: str
    image_token_index: int
    expected_image_tokens: int
    original_image_tokens: int
    patched_image_tokens: int
    original_token_count: int
    patched_token_count: int
    patch_applied: bool
    prompt_suffix: str
    blockers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_llava_config(llava_dir: str | Path = DEFAULT_LLAVA_DIR) -> dict[str, Any]:
    config_path = Path(llava_dir) / "config.json"
    return json.loads(config_path.read_text())


def expected_llava_image_tokens(config: dict[str, Any]) -> int:
    image_seq_length = config.get("image_seq_length")
    if image_seq_length:
        return int(image_seq_length)
    vision = config.get("vision_config") or {}
    image_size = int(vision.get("image_size", 336))
    patch_size = int(vision.get("patch_size", 14))
    return (image_size // patch_size) ** 2


def image_token_index(config: dict[str, Any], tokenizer: Any | None = None) -> int:
    configured = config.get("image_token_index")
    if configured is not None:
        return int(configured)
    if tokenizer is None:
        raise ValueError("image_token_index missing from config and no tokenizer was provided")
    token_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
    if token_id is None or token_id < 0:
        raise ValueError("tokenizer cannot resolve <image> token id")
    return int(token_id)


def expand_llava_image_placeholders(text: str, expected_count: int, marker: str = IMAGE_PLACEHOLDER) -> tuple[str, bool]:
    count = text.count(marker)
    if count == expected_count:
        return text, False
    if count == 1:
        return text.replace(marker, marker * expected_count), True
    if count == 0:
        return text + "\n" + marker * expected_count, True
    if count % expected_count == 0:
        return text, False
    normalized = text.replace(marker * count, marker * expected_count)
    if normalized == text:
        normalized = text.replace(marker, marker * expected_count, 1).replace(marker, "")
    return normalized, True


def build_hunyuan_avatar_llava_prompt(prompt: str, *, name: str = "person") -> str:
    return prompt + f"\nThe {name} looks like{IMAGE_PLACEHOLDER}"


def count_token_id(tokenizer: Any, text: str, token_id: int) -> tuple[int, int]:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=4096,
        padding=False,
        return_attention_mask=True,
        return_tensors=None,
    )
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return int(ids.count(token_id)), len(ids)


def inspect_llava_image_token_alignment(
    *,
    llava_dir: str | Path = DEFAULT_LLAVA_DIR,
    prompt: str = "A person talks naturally to the camera.",
    name: str = "person",
) -> HunyuanAvatarLlavaAlignment:
    from transformers import LlamaTokenizerFast

    selected = Path(llava_dir)
    config = load_llava_config(selected)
    expected = expected_llava_image_tokens(config)
    tokenizer = LlamaTokenizerFast.from_pretrained(selected, padding_side="right", local_files_only=True)
    token_id = image_token_index(config, tokenizer)
    original = build_hunyuan_avatar_llava_prompt(prompt, name=name)
    patched, patch_applied = expand_llava_image_placeholders(original, expected)
    original_count, original_len = count_token_id(tokenizer, original, token_id)
    patched_count, patched_len = count_token_id(tokenizer, patched, token_id)
    blockers: list[str] = []
    if patched_count != expected:
        blockers.append(f"patched prompt image token count {patched_count} != expected {expected}")
    if token_id != tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER):
        blockers.append("config image_token_index differs from tokenizer <image> id")
    status = "verified_llava_image_token_alignment_bridge" if not blockers else "blocked_llava_image_token_alignment_bridge"
    return HunyuanAvatarLlavaAlignment(
        status=status,
        runnable=not blockers,
        llava_dir=str(selected),
        image_token_index=token_id,
        expected_image_tokens=expected,
        original_image_tokens=original_count,
        patched_image_tokens=patched_count,
        original_token_count=original_len,
        patched_token_count=patched_len,
        patch_applied=patch_applied,
        prompt_suffix=IMAGE_PLACEHOLDER * min(expected, 8) + ("..." if expected > 8 else ""),
        blockers=tuple(blockers),
    )

def install_distributed_output_none_patch() -> dict[str, Any]:
    """Make non-output Avatar ranks safe under modern Diffusers BaseOutput.

    The optimized FSDP2 runner decodes/writes video on rank 0 only. Upstream
    returns HunyuanVideoPipelineOutput(videos=None) on other ranks, then its
    sampler indexes ``[0]``. Modern Diffusers drops None fields from the tuple,
    so ``[0]`` raises IndexError. Returning None for index 0 preserves the
    upstream sampler's existing ``if samples is None`` contract.
    """

    from hymm_sp.diffusion.pipelines import pipeline_hunyuan_video_audio as pipeline_mod

    output_cls = pipeline_mod.HunyuanVideoPipelineOutput
    if getattr(output_cls, "_legacy_model_bridge_none_rank_patch", False):
        return {"patched": True, "already_patched": True, "target": "HunyuanVideoPipelineOutput.__getitem__"}

    original_getitem = output_cls.__getitem__

    def getitem_rank_safe(self: Any, key: Any) -> Any:
        if key == 0 and getattr(self, "videos", None) is None:
            return None
        return original_getitem(self, key)

    output_cls._legacy_model_bridge_original_getitem = original_getitem
    output_cls.__getitem__ = getitem_rank_safe
    output_cls._legacy_model_bridge_none_rank_patch = True
    return {"patched": True, "already_patched": False, "target": "HunyuanVideoPipelineOutput.__getitem__"}


def install_torch_fsdp2_mesh_layout_pickle_patch() -> dict[str, Any]:
    """Restore legacy FSDP2 mesh-layout pickle symbols for modern Torch.

    Some rank-local FSDP2 shards were serialized when Torch exposed the private
    ``torch.distributed._mesh_layout._FlatLayout`` class. Torch 2.10 exposes the
    compatible implementation as ``_MeshLayout`` with ``shape``/``stride``
    storage. Old pickles may restore ``sizes``/``strides`` state, so the bridge
    installs a tiny compatibility subclass that normalizes those attributes
    before ``torch.load`` hydrates the rank-local shards.
    """

    import importlib
    import sys

    try:
        mesh_layout = sys.modules.get("torch.distributed._mesh_layout")
        if mesh_layout is None:
            mesh_layout = importlib.import_module("torch.distributed._mesh_layout")
    except (ImportError, ModuleNotFoundError) as exc:
        return {
            "patched": False,
            "already_patched": False,
            "target": "torch.distributed._mesh_layout._FlatLayout",
            "error": str(exc),
        }

    existing = getattr(mesh_layout, "_FlatLayout", None)
    if existing is not None and getattr(existing, "_legacy_model_bridge_flatlayout_patch", False):
        return {"patched": True, "already_patched": True, "target": "torch.distributed._mesh_layout._FlatLayout"}
    current = getattr(mesh_layout, "_MeshLayout", None)
    if current is None:
        return {
            "patched": False,
            "already_patched": False,
            "target": "torch.distributed._mesh_layout._FlatLayout",
            "error": "Torch exposes neither _FlatLayout nor _MeshLayout",
        }

    class _FlatLayout(current):  # type: ignore[misc, valid-type]
        _legacy_model_bridge_flatlayout_patch = True

        @property
        def shape(self) -> Any:
            values = object.__getattribute__(self, "__dict__")
            if "_legacy_shape" in values:
                return values["_legacy_shape"]
            if "sizes" in values:
                return values["sizes"]
            raise AttributeError("_FlatLayout has no shape/sizes state")

        @shape.setter
        def shape(self, value: Any) -> None:
            object.__setattr__(self, "_legacy_shape", value)

        @property
        def stride(self) -> Any:
            values = object.__getattribute__(self, "__dict__")
            if "_legacy_stride" in values:
                return values["_legacy_stride"]
            if "strides" in values:
                return values["strides"]
            raise AttributeError("_FlatLayout has no stride/strides state")

        @stride.setter
        def stride(self, value: Any) -> None:
            object.__setattr__(self, "_legacy_stride", value)

        def __setstate__(self, state: dict[str, Any]) -> None:
            shape = state.get("shape", state.get("sizes"))
            stride = state.get("stride", state.get("strides"))
            if shape is None or stride is None:
                for key, value in state.items():
                    object.__setattr__(self, key, value)
                return
            object.__setattr__(self, "_legacy_shape", shape)
            object.__setattr__(self, "_legacy_stride", stride)

        def __getstate__(self) -> dict[str, Any]:
            return {"shape": self.shape, "stride": self.stride}

    _FlatLayout.__name__ = "_FlatLayout"
    _FlatLayout.__qualname__ = "_FlatLayout"
    _FlatLayout.__module__ = "torch.distributed._mesh_layout"

    def _axis_value(values: dict[str, Any], attr: str) -> Any:
        axes = values.get("axes")
        if not axes:
            raise AttributeError(f"_MeshLayout has no {attr} state")
        derived = tuple(getattr(axis, attr) for axis in axes)
        return derived[0] if len(derived) == 1 else derived

    def _mesh_shape(self: Any) -> Any:
        values = object.__getattribute__(self, "__dict__")
        if "_legacy_shape" in values:
            return values["_legacy_shape"]
        if "shape" in values:
            return values["shape"]
        if "sizes" in values:
            return values["sizes"]
        return _axis_value(values, "shape")

    def _set_mesh_shape(self: Any, value: Any) -> None:
        object.__setattr__(self, "_legacy_shape", value)

    def _mesh_stride(self: Any) -> Any:
        values = object.__getattribute__(self, "__dict__")
        if "_legacy_stride" in values:
            return values["_legacy_stride"]
        if "stride" in values:
            return values["stride"]
        if "strides" in values:
            return values["strides"]
        return _axis_value(values, "stride")

    def _set_mesh_stride(self: Any, value: Any) -> None:
        object.__setattr__(self, "_legacy_stride", value)

    if not getattr(current, "_legacy_model_bridge_meshlayout_state_patch", False):
        current.shape = property(_mesh_shape, _set_mesh_shape)
        current.stride = property(_mesh_stride, _set_mesh_stride)
        current._legacy_model_bridge_meshlayout_state_patch = True

    setattr(mesh_layout, "_FlatLayout", _FlatLayout)
    return {"patched": True, "already_patched": False, "target": "torch.distributed._mesh_layout._FlatLayout"}



def remap_avatar_fp8_fsdp2_state_keys(state: Any) -> Any:
    """Map legacy FP8 linear shard keys to the split holder names used now."""

    from collections import OrderedDict

    scale_suffix = ".fp8_scale"
    weight_suffix = ".weight"
    holder_weight_suffix = ".fp8_weight_holder.weight"
    holder_scale_suffix = ".fp8_weight_holder.scale"
    prefixes = {key[: -len(scale_suffix)] for key in state.keys() if isinstance(key, str) and key.endswith(scale_suffix)}
    if not prefixes:
        return state
    remapped = OrderedDict()
    changed = False
    for key, value in state.items():
        new_key = key
        if isinstance(key, str) and key.endswith(scale_suffix):
            prefix = key[: -len(scale_suffix)]
            new_key = prefix + holder_scale_suffix
        elif isinstance(key, str) and key.endswith(weight_suffix):
            prefix = key[: -len(weight_suffix)]
            if prefix in prefixes:
                new_key = prefix + holder_weight_suffix
        changed = changed or new_key != key
        remapped[new_key] = value
    return remapped if changed else state


def install_avatar_fp8_fsdp2_state_dict_key_patch() -> dict[str, Any]:
    """Patch transformer_10 shard loading for legacy FP8 key names.

    Older shard files store native FP8 linears as ``linear.weight`` plus
    ``linear.fp8_scale``. The optimized latest-env model splits those tensors
    into ``linear.fp8_weight_holder.weight`` and
    ``linear.fp8_weight_holder.scale`` before FSDP2 wrapping. This loader patch
    remaps only linears that have a matching ``.fp8_scale`` key, preserving BF16
    and non-FP8 parameters unchanged.
    """

    import torch
    from pathlib import Path
    from torch.distributed.checkpoint.state_dict import set_model_state_dict
    from runtime import hunyuan_avatar_fsdp as fsdp_mod

    if getattr(fsdp_mod, "_legacy_model_bridge_fp8_key_patch", False):
        return {"patched": True, "already_patched": True, "target": "runtime.hunyuan_avatar_fsdp.load_rank_local_fsdp2_shard"}

    def load_rank_local_fsdp2_shard(model: Any, shard_dir: str | Path, *, rank: int) -> None:
        shard_path = Path(shard_dir) / f"avatar_transformer.rank{rank:02d}.pt"
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing FSDP2 shard for rank {rank}: {shard_path}")
        state = torch.load(shard_path, map_location="cpu", weights_only=False)
        state = remap_avatar_fp8_fsdp2_state_keys(state)
        incompatible = set_model_state_dict(model, state)
        del state
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Avatar FSDP2 shard did not load cleanly after bridge key remap: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )

    fsdp_mod._legacy_model_bridge_original_load_rank_local_fsdp2_shard = fsdp_mod.load_rank_local_fsdp2_shard
    fsdp_mod.load_rank_local_fsdp2_shard = load_rank_local_fsdp2_shard
    fsdp_mod._legacy_model_bridge_fp8_key_patch = True
    return {"patched": True, "already_patched": False, "target": "runtime.hunyuan_avatar_fsdp.load_rank_local_fsdp2_shard"}


def install_llava_llama_model_property_patch() -> dict[str, Any]:
    """Expose ``LlamaModel.model`` for legacy LLaVA loader code.

    Newer Transformers can make ``LlavaForConditionalGeneration.language_model``
    the ``LlamaModel`` itself, while the Avatar loader expects an older nested
    ``language_model.model.norm`` shape. Returning ``self`` for the missing
    ``.model`` attribute preserves that legacy access path without replacing
    Hunyuan's text encoder loader.
    """

    try:
        from transformers.models.llama.modeling_llama import LlamaModel
    except Exception as exc:
        return {"patched": False, "already_patched": False, "target": "transformers.LlamaModel.model", "error": str(exc)}
    if hasattr(LlamaModel, "model"):
        return {"patched": True, "already_patched": True, "target": "transformers.LlamaModel.model"}
    LlamaModel.model = property(lambda self: self)
    LlamaModel._legacy_model_bridge_model_property_patch = True
    return {"patched": True, "already_patched": False, "target": "transformers.LlamaModel.model"}
