# ABot World Bridge

ABot World is now verified from this repository against the latest `ai` environment instead of relying only on transformer_10 notes.

Validated environment:

- Python 3.11
- Torch 2.10.0+cu128
- CUDA 12.8
- Model: `/arxiv/models/acvlab--ABot-World-0-5B-LF`
- Upstream runtime checkout: `/data/repositories/ABot-World`

## Verified Contract

- The bridge loads the CausalWan generator with `torch_dtype=bfloat16`, `low_cpu_mem_usage=True`, `use_safetensors=True`, and string `device_map={"": "cuda:0"}`.
- The latest environment selected `sageattn`; `flash_attn` and `sageattn3` remained optional and absent.
- Generator load completed in about 8.8 seconds and allocated about 10.1GB VRAM on one RTX 3090.
- Prompt-cache-only mode was verified with `ABOT_WORLD_REQUIRE_PROMPT_CACHE=1`; cached BF16 prompt embeddings returned without loading the T5 encoder.
- SageAttention2 is already installed and verified in `ai` as `sageattention==2.2.0` with sm80/sm86 kernel symbols. On RTX 3090, do not install `sageattn3`; ABot documents that path as Blackwell-only.

Reports:

- `reports/world-model-smokes/acvlab--ABot-World-0-5B-LF.generator_cuda_bf16.bridge.ai.cuda2.json`
- `reports/world-model-smokes/acvlab--ABot-World-0-5B-LF.prompt_cache_hit.bridge.ai.json`
- `reports/world-model-smokes/sageattention.ai.cuda2.probe.json`

Run:

```bash
conda run --no-capture-output -n ai env PYTHONNOUSERSITE=1 PYTHONPATH=/data/legacy_model_bridge CUDA_VISIBLE_DEVICES=2 \
  python scripts/smoke_abot_world.py \
  --repo-path /data/repositories/ABot-World \
  --model-path /arxiv/models/acvlab--ABot-World-0-5B-LF \
  --device cuda:0 \
  --dtype bfloat16
```

Probe SageAttention after any Torch/CUDA change:

```bash
conda run --no-capture-output -n ai env PYTHONNOUSERSITE=1 PYTHONPATH=/data/legacy_model_bridge CUDA_VISIBLE_DEVICES=2 \
  python scripts/probe_sageattention.py --device cuda:0 --run-smoke
```
