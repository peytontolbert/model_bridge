# Next Integration Candidates

This review uses `/data/staticpeytonsite/src/research_library/data/model_index.json`,
the current bridge catalogs, and transformer_10 smoke evidence. It is ranked for
the next engineering pass after the implemented 3D and NeMo ASR worker lanes.

The reusable machine-readable version is `data/next_integration_candidates.json`.

## Ranked Targets

1. `nvidia/Cosmos-Transfer2.5-2B`
   - Next bridge: `legacy_model_bridge.runtime.cosmos25` plus `scripts/cosmos25_official_worker.py`.
   - Why next: worker registry already has the `cosmos25_py310` preflight lane, official runtime imports validate, and local checkpoint overrides are known.
   - Current bridge state: student-only worker is verified with a 93-frame/1-step bounded generation on GPU 0. The bridge now skips teacher/fake-score construction, borrows GPU memory for text encoding, and handles tokenizer wrapper device movement. Remaining blocker is `ai` parity for official Cosmos native imports/packages, not Transfer 2.5 runtime OOM.

2. `nvidia/Cosmos-Predict2.5-14B`
   - Next bridge: reuse the Cosmos 2.5 official worker with larger-model offload controls.
   - Why next: same Python 3.10/Torch cu128 native boundary as Transfer 2.5, but with more VRAM risk.
   - Current bridge state: no-generation launch-plan worker is implemented and validates the local 14B checkpoint/input with offload flags. Next smoke is tiny offloaded generation.

3. `MobileLLM family`
   - Next bridge: `legacy_model_bridge.runtime.causal_lm`.
   - Why next: transformer_10 already verified repeated CUDA generation across classic, layer-share, R1.5, and Pro variants in `ai`.
   - First smoke: port the MobileLLM-125M 4-token CUDA smoke with slow-tokenizer fallback and `use_cache=false`.

4. `Audio2Face-3D-v2.3-Mark`
   - Next bridge: `legacy_model_bridge.runtime.audio2face_onnx`.
   - Why next: ONNX Runtime loads `network.onnx` in `ai`; remaining work is bridge-owned preprocessing/postprocessing.
   - First smoke: inspect ONNX tensor names/shapes and run a synthetic audio window.

5. `Audio2Face-3D-v3.0`
   - Next bridge: same Audio2Face ONNX runtime with version-specific tensor metadata.
   - Why next: same validated ONNX load lane as v2.3.
   - First smoke: run the same session/tensor inspection smoke and compare IO metadata.

6. `Wan-AI--Wan2.2-Animate-14B`
   - Bridge: `legacy_model_bridge.runtime.wan_animate`.
   - Status: integrated as an `ai` bridge inspector.
   - Proof: `reports/world-model-smokes/Wan-AI--Wan2.2-Animate-14B.bridge-inspect.ai.cuda0.json` validates shards, cached T5 prompt context, cached VAE control latents, SageAttention2 backend selection, and 40-block INT8/offload artifacts.

7. `HunyuanVideo-Avatar`
   - Bridge: `legacy_model_bridge.runtime.hunyuan_avatar` plus the existing `hunyuan_avatar_ai` FSDP worker lane.
   - Status: LLaVA image-token alignment preflight is verified in `ai`; the bridge expands the single `<image>` marker to 576 tokens matching CLIP features.
   - Next smoke: run bounded distributed/FSDP generation in `ai`.
   - Proof: `reports/world-model-smokes/HunyuanVideo-Avatar.llava-alignment.ai.json`.

8. `DAM-3B`
   - Bridge: `legacy_model_bridge.runtime.dam`, `scripts/inspect_dam.py`, and `scripts/smoke_dam_description.py`.
   - Status: component bridge is verified in `ai` for `DAM-3B`, `DAM-3B-Self-Contained`, and `DAM-3B-Video`; `DAM-3B-Self-Contained` full eager `AutoModel` load is verified on CUDA; the legacy `DescribeAnythingModel` wrapper now produces a bounded image/mask description in `ai`.
   - Next smoke: replace the eager wrapper path with controlled lazy LLM placement/offload for faster optimized inference.
   - Proof: `reports/world-model-smokes/DAM-3B-Self-Contained.description8.cuda1.ai.json`, `reports/world-model-smokes/DAM-3B-Self-Contained.load_only.cuda1.ai.json`, plus DAM component bridge reports.

## Deferred

`MOSS-SoundEffect-v2.0`, `sam-audio-*`, `pe-av-*`, and LingBot video/world
remain important, but they are lower priority until missing runtime packages,
incomplete checkpoints, or bounded generation blockers are resolved.
