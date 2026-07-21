# HunyuanVideo-Avatar External Patch Audit

Date: 2026-07-20

Scope: audit the Hunyuan Avatar compatibility logic that currently exists as local modifications in `/data/clone/hunyuanvideo-avatar`, and identify safe bridge-owned coverage in `/data/legacy_model_bridge`.

## Summary

The bridge already owns two low-risk pieces of Hunyuan Avatar compatibility:

- `legacy_model_bridge/runtime/hunyuan_avatar.py` verifies LLaVA image-token alignment before generation by expanding one `<image>` placeholder to the configured image-token count.
- `legacy_model_bridge/runtime/hunyuan_avatar.py` installs `HunyuanVideoPipelineOutput.__getitem__` coverage so non-output distributed ranks can return `None` at tuple index 0 after rank-0-only decode.

The remaining external patches are not safe to express as small monkeypatches without replacing large upstream methods. They alter `HunyuanVideoAudioPipeline.encode_prompt`, the main denoising loop, `HunyuanVideoSampler.predict`, and `hymm_sp.text_encoder.load_text_encoder` behavior. Those should be moved into bridge-owned source patches or a maintained adapter module, not duplicated as fragile runtime monkeypatches.

## External Patch Coverage

| Patch area | Local modified file | Current bridge coverage | Monkeypatch recommendation |
| --- | --- | --- | --- |
| LLaVA image-token expansion | `/data/clone/hunyuanvideo-avatar/hymm_sp/diffusion/pipelines/pipeline_hunyuan_video_audio.py:454` | Preflight coverage only: `expand_llava_image_placeholders` and `inspect_llava_image_token_alignment` verify 576 image tokens. | Next target. Safe if implemented as a helper that expands tokenizer batch encodings, then used by a small wrapper around `encode_prompt_audio_text_base`; avoid replacing the full pipeline method. |
| Stale LLaVA masks replaced with embedding-shape masks | `/data/clone/hunyuanvideo-avatar/hymm_sp/diffusion/pipelines/pipeline_hunyuan_video_audio.py:1064` | Not bridge-owned. | Pair with image-token expansion. Mask repair depends on actual `prompt_embeds_input` shape and should live next to the pipeline encode wrapper. |
| Env-controlled denoise window and unconditional audio window | `/data/clone/hunyuanvideo-avatar/hymm_sp/diffusion/pipelines/pipeline_hunyuan_video_audio.py:1137` | Launch env sets `HUNYUAN_AVATAR_WINDOW_LATENTS`, but the behavior requires local source changes. | Do not monkeypatch independently; this is inside the denoising loop and would require copying most of `__call__`. |
| Rank-local denoiser release and rank-0-only VAE decode | `/data/clone/hunyuanvideo-avatar/hymm_sp/diffusion/pipelines/pipeline_hunyuan_video_audio.py:1373` | Partial bridge-owned guard for `HunyuanVideoPipelineOutput(videos=None)[0]`. | Keep the bridge guard. The actual release/decode branch should remain a source patch or adapter-owned pipeline subclass because it changes distributed synchronization and memory ownership. |
| VAE latent cache and fixed reference frames | `/data/clone/hunyuanvideo-avatar/hymm_sp/sample_inference_audio.py:111` | Launch env sets `HUNYUAN_AVATAR_REFERENCE_FRAMES` and `HUNYUAN_AVATAR_VAE_LATENT_CACHE`, but the behavior requires local source changes. | Do not monkeypatch `HunyuanVideoSampler.predict`; it is too broad and coupled to dataset batch structure, face masks, VAE precision, and rotary embedding length. |
| INT8 LLaVA via bitsandbytes | `/data/clone/hunyuanvideo-avatar/hymm_sp/text_encoder/__init__.py:7` | Launch env sets `HUNYUAN_AVATAR_LLAVA_INT8_GPU=1` and `HUNYUAN_AVATAR_LLAVA_GPU_DEVICE=cuda:0`, but the behavior requires local source changes. | Candidate for a narrow bridge monkeypatch of `hymm_sp.text_encoder.load_text_encoder`, but only after a live import smoke confirms installed `transformers`, `bitsandbytes`, and device-map behavior in `ai`. |

## Risk Notes

- The LLaVA expansion currently hard-codes `576` in the local pipeline patch. Bridge preflight computes the count from `image_seq_length` or `vision_config`; any bridge implementation should reuse that logic rather than hard-coding the current CLIP grid.
- The denoise-window patch changes multiple invariants at once: latent window length, unconditional audio shape, cache tensor shape, and wraparound behavior. Splitting out only `HUNYUAN_AVATAR_WINDOW_LATENTS` would leave the unconditional branch mismatch unfixed.
- Rank-0-only decode is a correctness and memory patch, not just an output-contract patch. The bridge's existing `__getitem__` guard prevents rank cleanup failures after `videos=None`, but it does not make unpatched nonzero ranks skip VAE decode.
- The VAE latent cache patch changes both memory pressure and geometry semantics. It should be validated with the same `reference_frames`, VAE latent temporal length, and face-mask resolution used by the optimized worker.

## Recommended Next Patch Target

Implement bridge-owned LLaVA tokenizer-batch expansion plus embedding-shape mask repair first. It has the smallest behavioral surface, directly addresses modern Transformers compatibility, and can be unit-tested without running distributed generation. The implementation target should be a helper in `legacy_model_bridge.runtime.hunyuan_avatar` plus tests that feed fake tokenizer encodings and fake LLaVA pixel batches, before wiring it into the optimized worker.
