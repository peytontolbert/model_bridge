# Audio Generation Bridge

This lane covers legacy audio generation artifacts that should run in the latest `ai` environment without preserving AudioCraft or older Transformers environments.

## MusicGen

The MusicGen family is verified through the bridge with latest Transformers/Torch:

- Loader: `AutoProcessor` + `MusicgenForConditionalGeneration` for standard MusicGen.
- Melody loader: `AutoProcessor` + `MusicgenMelodyForConditionalGeneration` for `musicgen_melody` configs.
- Environment: `ai`
- Device: CUDA fp16
- Smoke: bounded `max_new_tokens=8`
- Output contract: 0.1 seconds at 32kHz; mono models emit `[1, 1, 3200]`, stereo models emit `[1, 2, 3200]`.

Reports:

- `reports/audio-generation-smokes/musicgen.inspect.json`
- `reports/audio-generation-smokes/musicgen-small.generate8.ai.cuda2.json`
- `reports/audio-generation-smokes/musicgen-stereo-small.generate8.ai.cuda1.json`
- `reports/audio-generation-smokes/musicgen-stereo-medium.generate8.ai.cuda1.json`
- `reports/audio-generation-smokes/musicgen-large.generate8.ai.cuda1.json`
- `reports/audio-generation-smokes/musicgen-stereo-large.generate8.ai.cuda1.json`
- `reports/audio-generation-smokes/musicgen-stereo-melody-large.generate8.ai.cuda1.json`

## BigVGAN

`bigvgan_v2_44khz_128band_512x` is verified through the bridge with the latest `ai` environment:

- Loader: vendored `bigvgan.py` from the artifact directory
- Checkpoint: `bigvgan_generator.pt`
- Compatibility setting: `use_cuda_kernel=false`
- Smoke: synthetic mel tensor `[1, 128, 4]`
- Output: `[1, 1, 2048]` float32 waveform, about 0.046 seconds at 44.1kHz

The optional fused CUDA alias-free activation kernel is intentionally disabled. The upstream README says it builds with `nvcc` and `ninja` and was tested with CUDA 12.1; the bridge keeps the pure PyTorch path so users do not need a native extension build just to run the model in `ai`.

Reports:

- `reports/audio-generation-smokes/bigvgan.inspect.json`
- `reports/audio-generation-smokes/bigvgan.synthetic4.ai.cuda2.json`

## AudioGen

`audiogen-medium` is inspected but not yet runnable in `ai`:

- Artifact: AudioCraft solver checkpoint pair, `state_dict.bin` plus `compression_state_dict.bin`.
- Embedded config: 16kHz mono AudioGen, 4 EnCodec codebooks, transformer LM with delay pattern.
- Present in `ai`: `torch`, `torchaudio`, `encodec`.
- Local source probe: `/data/tmp/lmb_audiocraft_src` is present, but importing `audiocraft.models.AudioGen` in `ai` currently fails at `ModuleNotFoundError: No module named 'julius'`.
- Remaining runtime bridge: provide the small AudioCraft dependency surface and override the legacy `memory_efficient=true` attention path to latest Torch SDPA or another compatible backend before generation.

Report:

- `reports/audio-generation-smokes/audiogen-medium.inspect.ai.json`

Next bridge step: vendor or install a latest-compatible AudioCraft loader, satisfy the import-only dependency blockers without downgrading Torch, apply the attention backend compatibility override, and run a bounded 16kHz text-to-audio smoke using the checkpoint pair. Converting this artifact to a Transformers MusicGen-like directory is higher risk because it requires config synthesis, key remapping, EnCodec conversion, and text-encoder handling.

## Remaining Audio Targets

- `audiogen-medium`: implement the AudioCraft-compatible loader/generation adapter described above.
