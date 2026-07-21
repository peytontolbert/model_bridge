from __future__ import annotations

import sys
import types
from typing import Any

PATCH_ID_NEMO_RNNT_BPE_PROMPT_CLASS = "nemo_rnnt_bpe_prompt_class_shim"
LEGACY_TARGET = "nemo.collections.asr.models.rnnt_bpe_models_prompt.EncDecRNNTBPEModelWithPrompt"
SHIM_MODULE = "nemo.collections.asr.models.rnnt_bpe_models_prompt"


def install_nemo_rnnt_bpe_prompt_shim() -> bool:
    """Provide NeMo 2.7.x compatibility for legacy RNNT BPE prompt archives.

    Older Nemotron ASR archives target ``rnnt_bpe_models_prompt``, which is not
    shipped by current NeMo. The installed hybrid prompt class is not load
    compatible because it creates CTC decoder parameters. This shim builds the
    missing non-hybrid class from the current RNNT BPE base and prompt projection
    behavior, preserving the archive state_dict contract.
    """
    try:
        __import__(SHIM_MODULE, fromlist=["EncDecRNNTBPEModelWithPrompt"])
        return False
    except Exception:
        pass

    import torch
    from omegaconf import ListConfig, OmegaConf, open_dict

    from nemo.collections.asr.metrics.bleu import BLEU
    from nemo.collections.asr.metrics.wer import WER
    from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding
    from nemo.utils import logging, model_utils

    class EncDecRNNTBPEModelWithPrompt(EncDecRNNTBPEModel):
        def __init__(self, cfg: Any, trainer: Any = None):
            cfg = model_utils.convert_model_config_to_dict_config(cfg)
            cfg = model_utils.maybe_update_config_version(cfg)
            if "tokenizer" not in cfg:
                raise ValueError("`cfg` must have `tokenizer` config to create a tokenizer !")
            if not hasattr(cfg, "get"):
                cfg = OmegaConf.create(cfg)

            self._setup_tokenizer(cfg.tokenizer)
            vocabulary = self.tokenizer.tokenizer.get_vocab()
            with open_dict(cfg):
                cfg.labels = ListConfig(list(vocabulary))
                cfg.num_prompts = cfg.model_defaults.get("num_prompts", 128)
            with open_dict(cfg.decoder):
                cfg.decoder.vocab_size = len(vocabulary)
            with open_dict(cfg.joint):
                cfg.joint.num_classes = len(vocabulary)
                cfg.joint.vocabulary = ListConfig(list(vocabulary))
                cfg.joint.jointnet.encoder_hidden = cfg.model_defaults.enc_hidden
                cfg.joint.jointnet.pred_hidden = cfg.model_defaults.pred_hidden

            super().__init__(cfg=cfg, trainer=trainer)
            self.concat = False
            if self.cfg.model_defaults.get("initialize_prompt_feature", False):
                self.initialize_prompt_feature()

        def initialize_prompt_feature(self) -> None:
            logging.info("RNNT BPE prompt compatibility shim initialized")
            self.concat = True
            self.num_prompts = self.cfg.get("num_prompts", 128)
            proj_in_size = self.num_prompts + self._cfg.model_defaults.enc_hidden
            proj_out_size = self._cfg.model_defaults.enc_hidden
            self.prompt_kernel = torch.nn.Sequential(
                torch.nn.Linear(proj_in_size, proj_out_size * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(proj_out_size * 2, proj_out_size),
            )
            self.decoding = RNNTBPEDecoding(
                decoding_cfg=self.cfg.decoding,
                decoder=self.decoder,
                joint=self.joint,
                tokenizer=self.tokenizer,
            )
            self.wer = WER(
                decoding=self.decoding,
                batch_dim_index=0,
                use_cer=self.cfg.get("use_cer", False),
                log_prediction=self.cfg.get("log_prediction", True),
                dist_sync_on_step=True,
            )
            self._bleu = BLEU(decoding=self.decoding, tokenize=self.cfg.get("bleu_tokenizer", "13a"), log_prediction=True)
            if self.joint.fuse_loss_wer:
                self.joint.set_loss(self.loss)
                self.joint.set_wer(self.wer)
            self.cur_decoder = "rnnt"

        def _default_prompt(self, encoded: Any) -> Any:
            prompt_dict = self.cfg.model_defaults.get("prompt_dictionary")
            prompt_id = int(prompt_dict.get("en-US", prompt_dict.get("en", 0))) if prompt_dict else 0
            prompt = encoded.new_zeros(encoded.shape[0], encoded.shape[1], int(self.num_prompts))
            prompt[:, :, prompt_id] = 1.0
            return prompt

        def forward(
            self,
            input_signal: Any = None,
            input_signal_length: Any = None,
            processed_signal: Any = None,
            processed_signal_length: Any = None,
            prompt: Any = None,
        ) -> tuple[Any, Any]:
            has_input_signal = input_signal is not None and input_signal_length is not None
            has_processed_signal = processed_signal is not None and processed_signal_length is not None
            if (has_input_signal ^ has_processed_signal) is False:
                raise ValueError(
                    "Arguments input_signal/input_signal_length and processed_signal/processed_signal_length are mutually exclusive"
                )
            if not has_processed_signal:
                processed_signal, processed_signal_length = self.preprocessor(
                    input_signal=input_signal,
                    length=input_signal_length,
                )
            if self.spec_augmentation is not None and self.training:
                processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)
            encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
            encoded = torch.transpose(encoded, 1, 2)
            if self.concat:
                if prompt is None:
                    prompt = self._default_prompt(encoded)
                elif prompt.shape[1] > encoded.shape[1]:
                    prompt = prompt[:, : encoded.shape[1], :]
                out_dtype = encoded.dtype
                encoded = self.prompt_kernel(torch.cat([encoded, prompt.to(device=encoded.device, dtype=encoded.dtype)], dim=-1)).to(out_dtype)
            encoded = torch.transpose(encoded, 1, 2)
            return encoded, encoded_len

    EncDecRNNTBPEModelWithPrompt.__module__ = SHIM_MODULE
    module = types.ModuleType(SHIM_MODULE)
    module.EncDecRNNTBPEModelWithPrompt = EncDecRNNTBPEModelWithPrompt
    sys.modules[SHIM_MODULE] = module
    return True


def maybe_install_nemo_asr_compat(target: str | None = None) -> tuple[str, ...]:
    if target not in {None, LEGACY_TARGET}:
        return ()
    patches: list[str] = []
    if install_nemo_rnnt_bpe_prompt_shim():
        patches.append(PATCH_ID_NEMO_RNNT_BPE_PROMPT_CLASS)
    return tuple(patches)
