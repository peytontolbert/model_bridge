# transformer_10 Compatibility Seed

This repo should treat transformer_10 as a source of verified bridge decisions,
not as code to import wholesale.

## Source Documents

- `/data/transformer_10/docs/model-stack-conda-envs.md`
- `/data/transformer_10/docs/model-stack-compatibility-patches.md`
- `/data/transformer_10/docs/model-stack-model-verification.md`
- `/data/transformer_10/docs/model-dependency-diagnostics.md`

## Bridge Lanes

| Lane | Default Env | Rule |
| --- | --- | --- |
| `diffusers_cuda_bridge` | `ai` | Start on current Diffusers/Transformers and add component-level placement, cached embeddings, or narrow API patches before adding another env. |
| `video_diffusion_bridge` | `ai` | Treat custom Wan, AnyFlow, ABot, Cosmos, and Hunyuan paths as explicit model bridges with status reports. |
| `transformers_causal_lm_bridge` | `ai` | Prefer modern Transformers with tokenizer/config/cache patches for narrow API drift. |
| `encoder_classifier_bridge` | `ai` | Use checkpoint head shapes as authoritative when stale classifier configs disagree. |
| `peft_adapter_bridge` | `match_base_model_env` | Resolve the base model first; adapters inherit its runtime constraints. |
| `nemo_asr_bridge` | `ai` | Use local `.nemo` archive restore through the warm worker; NeMo Toolkit 2.7.3 ASR is now verified in `ai` for the standard archive lane. |
| `three_d_gen_bridge` | `ai` or `trellis` | Keep conflicting 3D geometry stacks behind worker boundaries and normalize outputs to mesh artifacts. |

## Current Patch IDs To Preserve

- `transformers_mobilellm_legacy_cache`
- `transformers_mobilellm_slow_tokenizer`
- `transformers_classifier_head_num_labels_from_checkpoint`
- `transformers_apply_chunking_to_forward_compat`
- `diffusers_linear_activation_fallback`
- `abot_generator_direct_cuda_device_map_bf16`
- `safetensors_device_map_string_cuda`
- `abot_flash_attention_sdpa_fallback`
- `abot_lazy_t5_prompt_encoder`
- `abot_prompt_embedding_cache`
- `sageattention2_lightx2v_backend_select`
- `diffusers_anyflow_far_return_tuple_padding`
- `nemo_prefer_archive_over_external_transformers_metadata`

## Porting Rule

Every manual model inclusion should produce:

1. a `data/bridge_catalog.json` entry,
2. a model integration profile under `integrations/<model>/profile.json`,
3. a smoke or diagnostic report reference,
4. explicit compatibility patch IDs when runtime behavior differs from current dependencies,
5. a clear env decision: `ai`/`lmb-default`, a temporary named env, or a worker boundary.
