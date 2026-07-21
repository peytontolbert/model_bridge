# Older Model Compatibility Targets

Source index: `/data/staticpeytonsite/src/research_library/data/model_index.json`.

This list intentionally favors smaller or older model formats where the bridge can
add compatibility without repeating the Cosmos-scale memory problem.

## 1. NeMo ASR Archive Expansion

Use the hardened warm worker for more `.nemo` ASR archives: `parakeet-tdt_ctc-110m`,
`parakeet-tdt-0.6b-v3`, `parakeet-unified-en-0.6b`,
`nemotron-speech-streaming-en-0.6b`, `nemotron-3.5-asr-streaming-0.6b`, and
`multitalker-parakeet-streaming-0.6b-v1`.

First smoke: inspect archive target metadata, then CPU or CUDA load-only for the
smallest archives, then warm JSONL transcribe where the public API is standard
`ASRModel.transcribe`.

## 2. MobileLLM Family Completion

`MobileLLM-125M`, `MobileLLM-125M-layer-share`, `MobileLLM-350M`, `MobileLLM-350M-layer-share`, `MobileLLM-600M`, `MobileLLM-1B`, `MobileLLM-1.5B`, `MobileLLM-R1.5-140M`, `MobileLLM-R1.5-360M`, `MobileLLM-R1.5-950M`, and `MobileLLM-Pro-base` are cataloged as runnable. The
next pass should generalize that bridge to layer-share, 600M, 1B, 1.5B, R1.5,
and Pro-base variants. Expected shims are slow tokenizer fallback, removed cache
API handling, `use_cache=false`, and remote-code loader controls.

## 3. ONNX Audio/Face Models

`Audio2Emotion-v3.0`, `Audio2Face-3D-v2.3-Mark`, and `Audio2Face-3D-v3.0` have
ONNX graphs plus JSON/NPZ sidecars. Build a lightweight ONNX runtime bridge that
records tensor names/shapes/dtypes first, then adds preprocessing/postprocessing
contracts model by model.

## 4. Classic Transformers

Start with `facebook/bart-large-cnn`, HuBERT base/large, and DINOv2 base/large.
These are older stable formats with `.bin`, config, tokenizer/processor assets,
and occasional safetensors mirrors. First smoke should be load/config plus a tiny
forward or 4-token generation.

## 5. Audio Generation Classics

After the easier probes, cover `audiogen-medium`, MusicGen small/stereo variants,
and BigVGAN. Keep duration bounded and make sample-rate/artifact contracts
explicit.
