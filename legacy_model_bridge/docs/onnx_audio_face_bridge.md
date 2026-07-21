# ONNX Audio/Face Bridge

This bridge covers lightweight ONNX runtime compatibility for local
Audio2Emotion and Audio2Face assets without requiring the original SDK or
TensorRT runtime.

## What The Bridge Does

- Resolves model directories under `/arxiv/models`.
- Loads `network.onnx` with ONNX Runtime CPU provider.
- Records input/output tensor names, dtypes, and dynamic shapes.
- Reads JSON sidecars defensively and records NPZ tensor shapes.
- Can run bounded synthetic zero-input inference using sidecar audio buffer hints.
- Adds Audio2Face v2.3 semantic metadata for audio window, emotion vector width, and geometry output partitions.

## Current Results

- `Audio2Emotion-v3.0`: synthetic CPU run succeeds, output `output` shape `[1, 6]`.
- `Audio2Face-3D-v2.3-Mark`: model-aware synthetic CPU run succeeds with `input` `[1, 1, 8320]`, `emotion` `[1, 1, 26]`, and `result` `[1, 1, 301]`; the bridge records 272 skin blendshapes, 10 tongue blendshapes, 15 jaw values, and 4 eye values.
- `Audio2Face-3D-v3.0`: ONNX contract inspection succeeds; synthetic execution is deferred because the diffusion graph requires a large `noise` tensor with width `88831`.

## Reports

- `reports/onnx-smokes/audio-face.inspect.json`
- `reports/onnx-smokes/audio-face.synthetic-small.json`
- `reports/onnx-smokes/Audio2Face-3D-v2.3-Mark.semantic-synthetic.cpu.ai.json`

## Commands

```bash
python scripts/inspect_onnx_audio_face.py \
  --json-out reports/onnx-smokes/audio-face.inspect.json
```

```bash
python scripts/inspect_onnx_audio_face.py \
  --model Audio2Emotion-v3.0 \
  --model Audio2Face-3D-v2.3-Mark \
  --run-synthetic \
  --max-dynamic-dim 1 \
  --json-out reports/onnx-smokes/audio-face.synthetic-small.json
```

```bash
python scripts/inspect_onnx_audio_face.py \
  --model Audio2Face-3D-v2.3-Mark \
  --run-synthetic \
  --model-aware \
  --json-out reports/onnx-smokes/Audio2Face-3D-v2.3-Mark.semantic-synthetic.cpu.ai.json
```

