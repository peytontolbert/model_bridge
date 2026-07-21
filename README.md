# ModelBridge

Run selected legacy models from one up-to-date runtime instead of manually switching through old Conda environments.

ModelBridge is a curated compatibility layer for model stacks that were built against older PyTorch, Transformers, Diffusers, NeMo, ONNX, CUDA extension, or project-specific APIs. The repo does not try to execute arbitrary old repositories unchanged. For each supported model or model family, it owns a narrow bridge contract: resolve the local artifact, patch the known incompatibilities, load with current libraries, run a bounded smoke, and record the evidence.

A current known-good target stack is:

- Python 3.11 or 3.12 callers
- Recent CUDA-enabled PyTorch
- Recent Transformers, Diffusers, ONNX Runtime, NeMo, and related model libraries
- Native FlashAttention where the model and hardware can use it
- NVIDIA CUDA GPUs for lanes that require GPU inference

## What This Solves

Old model repos often pin mutually incompatible dependencies. This bridge moves the compatibility work into this repository so a user can keep the latest environment and call verified legacy lanes through stable scripts or APIs.

The rule is simple: if a model is marked `latest_env_ready`, users should not need its old environment. If it still needs isolation, the bridge owns that worker boundary and exposes a normalized contract.

`ModelBridge` is the public project name. The Python package currently remains `legacy_model_bridge` for import stability.

## Verified Latest-Env Lanes

The authoritative source is [`data/bridge_catalog.json`](data/bridge_catalog.json). Current verified examples include:

| Lane | Models | Bridge contract |
| --- | --- | --- |
| Classic Transformers | BART, GPT-2, DistilGPT-2, HuBERT base/large/xlarge, DINOv2, webssl-DINO300M, BERT encoders/classifiers | `Auto*` loaders with task inference, HF cache snapshot resolution, dtype compatibility, modality-specific processors |
| MobileLLM causal LM | MobileLLM classic, R1.5, Pro, and ParetoQ representatives | bounded CUDA generation with tokenizer/cache compatibility patches |
| NeMo ASR | Parakeet and Nemotron `.nemo` archives | warm archive worker that restores once and serves repeated transcription requests |
| ONNX Audio Face | Audio2Emotion and Audio2Face v2.3 | ONNX Runtime tensor contracts plus sidecar-driven semantic output parsing |
| Audio Generation | MusicGen/MusicGen Melody and BigVGAN | modern Transformers generation or vendored vocoder load with optional CUDA-kernel bypass |
| 3D Generation | TRELLIS.2 and Hunyuan3D shape paths | latest-env native geometry stack or bridge-owned mesh artifact normalization |
| World / Video Models | ABot World, AnyFlow, FLUX.2-klein, Cosmos3, DAM, Wan Animate inspection paths | model-specific bridge patches for imports, attention, placement, cached prompts, or component loading |

Some large or structurally difficult models remain explicitly tracked as candidates or worker-boundary work. That is intentional: unsupported models should fail with a cataloged reason instead of pushing users into dependency guesswork.

## Quick Start

Run commands from the repository root:

```bash
git clone https://github.com/peytontolbert/model_bridge
cd modelbridge
conda activate <your-current-ml-env>
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD"
```

Point the bridge at your local model storage. The exact paths are user-owned and should not be hardcoded by the repo:

```bash
export MODELBRIDGE_MODEL_ROOT=/path/to/models
export HF_HOME=/path/to/huggingface/cache
```

Inspect the catalog:

```bash
python -m legacy_model_bridge.cli list
python -m legacy_model_bridge.cli consolidation --summary
python -m legacy_model_bridge.cli doctor gpt2
```

Run representative bridge smokes:

```bash
python scripts/inspect_classic_transformers.py --model gpt2 --run-synthetic --device cpu --max-new-tokens 4
python scripts/inspect_classic_transformers.py --model hubert-xlarge-ll60k --run-synthetic --device cuda:0 --dtype float16
python -m legacy_model_bridge.cli nemo-asr run --archive-path "$MODELBRIDGE_MODEL_ROOT/parakeet-tdt_ctc-110m/parakeet-tdt_ctc-110m.nemo" --inspect-only
python -m legacy_model_bridge.cli three-d backends
```

## How A Bridge Works

For each integrated model, this repository records and tests the compatibility contract:

1. Resolve the local artifact from a configured model root, Hugging Face cache, or model-specific path.
2. Read the legacy config, checkpoint, tokenizer, processor, sidecars, and native extension requirements.
3. Apply only the compatibility patches needed for that model family.
4. Load through current maintained APIs when possible.
5. Run bounded validation for load, inference, output shape, dtype, device placement, and worker behavior.
6. Store evidence under `reports/` and promote the model in `data/bridge_catalog.json`.

The project preserves model behavior, not legacy environments.

## Compatibility Policy

Preferred fixes:

- old import path -> local compatibility import
- old checkpoint names -> tensor remapping
- old config schema -> config adapter
- old custom CUDA op -> maintained CUDA/Triton/PyTorch implementation, or an explicit documented fallback
- old Transformers behavior -> local wrapper or task-specific AutoModel route
- old environment pin -> current environment package or bridge-owned worker boundary

Non-goals:

- running every old repository unchanged
- preserving arbitrary pinned dependency stacks
- hiding model-specific limitations
- promising bit-exact parity when modern kernels or numerics differ

## Key Files

```text
legacy_model_bridge/runtime/     bridge runtime code
data/bridge_catalog.json         curated per-model readiness and evidence
data/compatibility_patches.json  named mismatch fixes and validation refs
scripts/                         inspection and smoke entrypoints
docs/                            lane-specific engineering notes
reports/                         generated validation evidence
tests/                           compatibility and smoke-contract tests
```

Useful docs:

- [`docs/environment_consolidation.md`](docs/environment_consolidation.md)
- [`docs/classic_transformers_bridge.md`](docs/classic_transformers_bridge.md)
- [`docs/causal_lm_bridge.md`](docs/causal_lm_bridge.md)
- [`docs/nemo_asr_archive_expansion.md`](docs/nemo_asr_archive_expansion.md)
- [`docs/audio_generation_bridge.md`](docs/audio_generation_bridge.md)
- [`docs/onnx_audio_face_bridge.md`](docs/onnx_audio_face_bridge.md)
- [`docs/env_removal_proofs.md`](docs/env_removal_proofs.md)

## Development Checks

```bash
python -m py_compile legacy_model_bridge/runtime/classic_transformers.py scripts/inspect_classic_transformers.py
pytest -q
```

When adding a model, update the bridge code, add or extend focused tests, run a real smoke in the current target environment, write the report under `reports/`, then promote the model and patch IDs in the catalog.
