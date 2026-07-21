# NeMo ASR Archive Expansion

Source reports: `reports/nemo-asr-smokes/archive-expansion.inspect.ai.json` and historical `reports/nemo-asr-smokes/archive-expansion.inspect.json`.

The batch inspector runs each ranked `.nemo` archive through the bridge worker in
inspect-only mode (`restore=false`) inside `ai`. This validates archive
resolution, target metadata, target importability, and backend env readiness
without paying model restore cost.

## Ready In `ai`

- `parakeet-tdt_ctc-110m` -> `EncDecHybridRNNTCTCBPEModel`
- `parakeet-tdt-0.6b-v3` -> `EncDecRNNTBPEModel`
- `parakeet-unified-en-0.6b` -> `EncDecRNNTBPEModel`
- `nemotron-speech-streaming-en-0.6b` -> `EncDecRNNTBPEModel`
- `nvidia/nemotron-speech-streaming-en-0.6b` -> `EncDecRNNTBPEModel`
- `multitalker-parakeet-streaming-0.6b-v1` -> `EncDecMultiTalkerRNNTBPEModel`

These archives have target importability and env readiness in `ai`. `parakeet-tdt_ctc-110m` also has load-only, one-shot transcribe, and warm JSONL service smokes in `ai`; `parakeet-tdt-0.6b-v3` has CUDA load-only proof in `reports/nemo-asr-smokes/parakeet-tdt-0.6b-v3.load_only.cuda1.ai.json`; `nemotron-3.5-asr-streaming-0.6b` has preflight, load-only restore, and warm JSONL transcription proof through the prompt class shim. Expand transcription proof across the remaining ready archives as needed.

## Nemotron 3.5 Prompt Target Shim

`nemotron-3.5-asr-streaming-0.6b` is now bridged in `ai` for preflight, load-only restore, and warm JSONL transcription. Its archive target is from newer NeMo metadata:

`nemo.collections.asr.models.rnnt_bpe_models_prompt.EncDecRNNTBPEModelWithPrompt`

NeMo Toolkit 2.7.3 does not ship that module. The neighboring installed hybrid prompt class was tested and rejected because it expects CTC decoder weights; the archive has `prompt_kernel` keys and no CTC decoder keys. The bridge now installs a narrow non-hybrid RNNT BPE prompt class shim before target import and restore.

Reports:

- `reports/nemo-asr-smokes/nemotron-3.5-asr-streaming-0.6b.inspect.ai.json`
- `reports/nemo-asr-smokes/nemotron-3.5-asr-streaming-0.6b.load_only.cpu.ai.json`
- `reports/nemo-asr-smokes/nemotron-3.5-asr-streaming-0.6b.warm-jsonl.cuda1.ai.json`
- `reports/nemo-asr-smokes/nemotron-3.5-asr-streaming-0.6b.rnnt_prompt_shim_restore.cpu.ai.json`

## Commands

```bash
python scripts/inspect_nemo_asr_archives.py \
  --json-out reports/nemo-asr-smokes/archive-expansion.inspect.ai.json \
  --timeout-sec 180
```

```bash
python -m legacy_model_bridge.cli nemo-asr run \
  --model-id parakeet-tdt_ctc-110m \
  --load-only \
  --device cuda:0 \
  --output-dir /data/tmp/lmb_nemo_parakeet_tdt_ctc_load_ai
```

## Parakeet Unified Compatibility

`parakeet-unified-en-0.6b` now has warm JSONL transcription proof in `ai`: `reports/nemo-asr-smokes/parakeet-unified-en-0.6b.warm-jsonl.cuda1.ai.json`. The bridge applies `nemo_parakeet_unified_context_config_compat` to translate old `chunked_limited_with_rc` encoder context config to modern `chunked_limited`, and `nemo_asr_transcribe_validation_ds_default` to default missing `validation_ds` before transcribe.
