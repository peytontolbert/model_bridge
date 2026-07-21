from __future__ import annotations

import os
from typing import Any, Callable, Mapping


def _module_to_device(module: Any, device: str) -> None:
    if module is None:
        return
    target = getattr(module, "model", module)
    if hasattr(target, "to"):
        target.to(device)


def _module_to_cpu(module: Any) -> None:
    _module_to_device(module, "cpu")


def _wrap_text_encoder_cpu_offload(text_encoder: Any) -> None:
    if text_encoder is None or getattr(text_encoder, "_legacy_model_bridge_cpu_offload", False):
        return
    original_compute: Callable[..., Any] = text_encoder.compute_text_embeddings_online

    def compute_with_cpu_offload(*args: Any, **kwargs: Any) -> Any:
        borrowed_modules = tuple(getattr(text_encoder, "_legacy_model_bridge_precompute_cpu_modules", ()))
        for module in borrowed_modules:
            _module_to_cpu(module)

        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            text_encoder.device = "cuda"
            return original_compute(*args, **kwargs)
        finally:
            _module_to_cpu(text_encoder)
            text_encoder.device = "cpu"
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
            for module in borrowed_modules:
                _module_to_device(module, "cuda")

    text_encoder.compute_text_embeddings_online = compute_with_cpu_offload
    text_encoder.device = "cpu"
    _module_to_cpu(text_encoder)
    text_encoder._legacy_model_bridge_cpu_offload = True


def _wrap_tokenizer_cpu_offload(model: Any) -> None:
    if getattr(model, "_legacy_model_bridge_tokenizer_cpu_offload", False):
        return
    original_encode = model.encode
    original_decode = model.decode

    def encode_with_cpu_offload(state: Any) -> Any:
        _module_to_device(model.tokenizer, "cuda")
        try:
            return original_encode(state)
        finally:
            _module_to_cpu(model.tokenizer)

    def decode_with_cpu_offload(latent: Any) -> Any:
        _module_to_device(model.tokenizer, "cuda")
        try:
            return original_decode(latent)
        finally:
            _module_to_cpu(model.tokenizer)

    model.encode = encode_with_cpu_offload
    model.decode = decode_with_cpu_offload
    _module_to_cpu(model.tokenizer)
    model._legacy_model_bridge_tokenizer_cpu_offload = True


def apply_cosmos25_transfer_student_only_patch() -> dict[str, Any]:
    """Patch Cosmos Transfer DMD2 inference to construct only the student net.

    The distilled Transfer 2.5 checkpoint contains `net.*` student weights only,
    but the official DMD2 model builds teacher and fake-score modules before it
    later discards fake-score for inference. This opt-in patch keeps the bridge
    patch local and must be applied before the official inference entrypoint
    instantiates its pipeline.
    """

    from cosmos_transfer2._src.interactive.methods.cosmos2_interactive_model import Cosmos2InteractiveModel
    from cosmos_transfer2._src.interactive.methods.distribution_matching.dmd2 import DMD2Model

    if getattr(DMD2Model, "_legacy_model_bridge_student_only", False):
        return {"patched": True, "already_patched": True, "target": "cosmos_transfer2.DMD2Model"}

    original_build_model = DMD2Model.build_model
    original_load_state_dict = DMD2Model.load_state_dict
    original_state_dict = DMD2Model.state_dict
    original_model_dict = DMD2Model.model_dict
    original_broadcast_split = DMD2Model.broadcast_split_for_model_parallelsim

    def build_model_student_only(self: Any) -> None:
        from cosmos_transfer2._src.interactive.methods import cosmos2_interactive_model as base

        cpu_offload = os.environ.get("LEGACY_MODEL_BRIDGE_COSMOS25_CPU_OFFLOAD", "1") != "0"
        self.text_encoder = None
        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            self.text_encoder = base.TextEncoder(self.config.text_encoder_config)
            if cpu_offload:
                _wrap_text_encoder_cpu_offload(self.text_encoder)

        self.neg_embed = (
            base.easy_io.load(self.config.neg_embed_path) if getattr(self.config, "neg_embed_path", "") else None
        )
        use_neg_prompt_str = getattr(self.config, "use_neg_prompt_str", False)
        neg_prompt_str = getattr(self.config, "neg_prompt_str", None)
        if use_neg_prompt_str and neg_prompt_str:
            assert self.text_encoder is not None, "text_encoder is required when use_neg_prompt_str is enabled"
            caption_key = getattr(self.config, "input_caption_key", "ai_caption")
            neg_data_batch = {caption_key: [neg_prompt_str]}
            neg_embed = self.text_encoder.compute_text_embeddings_online(neg_data_batch, caption_key)
            self.neg_embed = neg_embed[0] if isinstance(neg_embed, base.torch.Tensor) and neg_embed.ndim == 3 else neg_embed

        self.tokenizer = base.lazy_instantiate(self.config.tokenizer)
        assert self.tokenizer.latent_ch == self.config.state_ch, (
            f"latent_ch {self.tokenizer.latent_ch} != state_shape {self.config.state_ch}"
        )
        if cpu_offload:
            _module_to_cpu(self.tokenizer)

        if getattr(self.config, "ema", None) is not None and self.config.ema.enabled:
            self.config.ema.enabled = False

        self.net = self.build_net(self.config.net)
        if cpu_offload and self.text_encoder is not None:
            self.text_encoder._legacy_model_bridge_precompute_cpu_modules = (self.net,)
        self.net_teacher = None
        self.net_fake_score = None
        self.net_discriminator_head = None
        if getattr(self.net, "use_crossattn_projection", False):
            self.net.crossattn_proj.requires_grad_(False)
        if cpu_offload:
            _wrap_tokenizer_cpu_offload(self)
        self.conditioner = base.lazy_instantiate(self.config.conditioner)
        self.condition_postprocessor = (
            base.lazy_instantiate(self.config.condition_postprocessor)
            if getattr(self.config, "condition_postprocessor", None)
            else None
        )
        self.denoiser_nets = {"student": self.net}
        base.torch.cuda.empty_cache()

    def load_state_dict_student_only(
        self: Any,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        return Cosmos2InteractiveModel.load_state_dict(self, state_dict, strict=strict, assign=assign)

    def state_dict_student_only(self: Any) -> dict[str, Any]:
        return Cosmos2InteractiveModel.state_dict(self)

    def model_dict_student_only(self: Any) -> dict[str, Any]:
        return Cosmos2InteractiveModel.model_dict(self)

    def broadcast_split_student_only(self: Any, *args: Any, **kwargs: Any):
        return Cosmos2InteractiveModel.broadcast_split_for_model_parallelsim(self, *args, **kwargs)

    DMD2Model._legacy_model_bridge_originals = {
        "build_model": original_build_model,
        "load_state_dict": original_load_state_dict,
        "state_dict": original_state_dict,
        "model_dict": original_model_dict,
        "broadcast_split_for_model_parallelsim": original_broadcast_split,
    }
    DMD2Model.build_model = build_model_student_only
    DMD2Model.load_state_dict = load_state_dict_student_only
    DMD2Model.state_dict = state_dict_student_only
    DMD2Model.model_dict = model_dict_student_only
    DMD2Model.broadcast_split_for_model_parallelsim = broadcast_split_student_only
    DMD2Model._legacy_model_bridge_student_only = True

    return {
        "patched": True,
        "already_patched": False,
        "target": "cosmos_transfer2.DMD2Model",
        "skipped_modules": ["net_teacher", "net_fake_score", "net_discriminator_head", "net_ema"],
        "cpu_offload": os.environ.get("LEGACY_MODEL_BRIDGE_COSMOS25_CPU_OFFLOAD", "1") != "0",
        "checkpoint_prefix": "net",
    }
