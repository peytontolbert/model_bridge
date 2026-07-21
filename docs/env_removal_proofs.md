# Environment Removal Proofs

This document records the current proof status for legacy environments the bridge would like to remove.

Report:

- `reports/env-removal/trellis-ai-native-stack.proof.json`
- `reports/worker-preflights/three_d_trellis2_ai_pending_geometry.json`
- `reports/world-model-smokes/trellis2.load_only.ai.json`
- `reports/world-model-smokes/trellis2.tiny_generate.real_image.ai.json`
- `reports/world-model-smokes/trellis2.tiny_generate.ai.json` documents the discarded blank-image probe failure.

## Removed And Verified In `ai`

### `trellis`

The `trellis` conda env was intentionally removed to free disk space. `microsoft/TRELLIS.2-4B` is now verified in `ai` for the default official TRELLIS.2 `flex_gemm` path.

Current proof:

- `ai` imports the default TRELLIS.2 core native stack: `nvdiffrast`, `cumesh`, `flex_gemm`, and `o_voxel`.
- `reports/env-removal/trellis-ai-native-stack.proof.json` has status `ok` for core native dependencies.
- `reports/worker-preflights/three_d_trellis2_ai_pending_geometry.json` has status `ok` with Python 3.11, Torch 2.10 CUDA 12.8, native FlashAttention, and three CUDA devices.
- `reports/world-model-smokes/trellis2.load_only.ai.json` verifies official pipeline load in `ai`.
- `reports/world-model-smokes/trellis2.tiny_generate.real_image.ai.json` verifies a real-image one-step GLB generation in `ai` and writes `/data/tmp/model_stack_trellis2_ai_real/mesh.glb`.
- `spconv` and `kaolin` remain optional legacy parity items; local TRELLIS.2 defaults to `flex_gemm` and has no proven core `kaolin` import.
- Worker commands preload system `libstdc++` because CuMesh requires `GLIBCXX_3.4.30`, newer than the conda `ai` copy.

The earlier blank-image smoke in `reports/world-model-smokes/trellis2.tiny_generate.ai.json` failed after sparse sampling produced empty coordinates; it is retained as a bad-probe artifact, not as evidence that the native stack is blocked.

### `nemo_speech`

`nemo_speech` was removed after `ai` replaced it for the standard NeMo ASR archive lane.

Current proof:

- `nemo_toolkit[asr]==2.7.3` is installed in `ai` without replacing Torch, Transformers, or native FlashAttention.
- `ai` imports `nemo`, `nemo.collections.asr.models`, Torch `2.10.0+cu128`, Transformers `4.57.6`, and FlashAttention `2.8.3.post1` with CUDA visible.
- `reports/nemo-asr-smokes/archive-expansion.inspect.ai.json` marks six Parakeet/Nemotron archives ready in `ai`.
- `reports/nemo-asr-smokes/parakeet-tdt_ctc-110m.load_only.cuda1.ai.json` restores the small Parakeet archive in `ai`.
- `reports/nemo-asr-smokes/parakeet-tdt_ctc-110m.transcribe.cuda1.ai.json` passes one-shot transcription in `ai`.
- `reports/nemo-asr-smokes/parakeet-tdt_ctc-110m.warm-jsonl.cuda1.ai.json` passes the warm JSONL service contract in `ai` with cached restore.
- `reports/env-removal/nemo-speech-after-ai.proof.json` records the `nemo_speech` removal decision, accurate `ai` package versions, and before/after freeze paths.

`nemotron-3.5-asr-streaming-0.6b` is now bridged in `ai` with `nemo_rnnt_bpe_prompt_class_shim`; warm JSONL transcription passes in `reports/nemo-asr-smokes/nemotron-3.5-asr-streaming-0.6b.warm-jsonl.cuda1.ai.json`.

Installing NeMo in `ai` changed `fsspec` to `2024.12.0`, `packaging` to `24.2`, and `dill` to `0.3.8`; `/data/tmp/ai.before-nemo.freeze.txt` records the pre-install snapshot.

### `py311build`

`ai` now replaces `py311build` for the registry-level `HunyuanVideo-Avatar` worker preflight and the Wan Animate cached/int8 bridge inspection lane.

Current proof:

- Native `flash_attn 2.8.3.post1` was built inside `ai` for Torch `2.10.0+cu128`, CUDA `12.8`, and the local Ampere GPU target.
- `reports/env-removal/hunyuan-flash-attn.ai.native.json` selects `native_flash_attn` and passes a CUDA `flash_attn_varlen_func` smoke.
- `reports/env-removal/py311build-after-ai-flash-attn.proof.json` has `safe_to_remove_now: true` for the registry-level Hunyuan Avatar import/path requirements.
- `reports/env-removal/py311build-after-ai-worker-migration.proof.json` shows no remaining registered workers own `py311build`.
- `reports/world-model-smokes/Wan-AI--Wan2.2-Animate-14B.bridge-inspect.ai.cuda0.json` reports `verified_wan_animate_cached_int8_bridge_ai` with no blockers in `ai`.
- `reports/env-removal/py311build-after-wan-animate-ai.proof.json` records the Conda env removal and post-removal disk state.
- The bridge still includes a scoped SDPA-backed `flash_attn_varlen_func` shim for import/debug/small synthetic smokes when native flash-attn is absent.
- Hunyuan Avatar LLaVA alignment is bridged; the worker now reaches BF16 FSDP denoising in `ai` with prewarmed VAE cache and min-65 reference-frame cache sizing.
- Local RTX 3090/SM86 cannot run FP8 distributed collectives, so auto mode routes to BF16 FSDP on this host and keeps FP8/FSDP2 for SM90+ hardware.

Full generation still requires a distributed/FSDP generation smoke in `ai` with two nearly empty 24 GiB GPUs, or larger GPUs. That is a runtime-capacity blocker, not a reason to keep `py311build`.

### `abot_world`

`ai` replaces `abot_world` for ABot-World through the bridge-owned generator and prompt-cache path.

Current proof:

- `reports/world-model-smokes/acvlab--ABot-World-0-5B-LF.generator_cuda_bf16.bridge.ai.cuda2.json` loads the BF16 generator in `ai` on CUDA with SageAttention selected.
- `reports/world-model-smokes/acvlab--ABot-World-0-5B-LF.prompt_cache_hit.bridge.ai.json` verifies cache-only prompt embeddings without loading the T5 encoder.
- `reports/world-model-smokes/sageattention.ai.cuda2.probe.json` verifies the local RTX 3090 SageAttention2 backend in `ai`.
- `reports/env-removal/abot-world-after-ai.proof.json` records the Conda env removal and post-removal disk state.
