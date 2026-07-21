# Causal LM Bridge

The causal-LM bridge provides a small in-process compatibility lane for older
Transformers text-generation models that should run in the latest `ai` env.

## MobileLLM Policies

Classic MobileLLM configs use remote model code and can return a non-callable
`bool` from `AutoTokenizer.from_pretrained` under the latest Transformers stack.
The classic default matrix keeps `MobileLLM-125M` and `MobileLLM-125M-layer-share` as separate entries; this is covered by tests so fresh older-model sweeps do not silently skip those artifacts.

The bridge handles this by:

- enabling `trust_remote_code` when `config.auto_map` is present;
- forcing `use_cache=false` for classic `model_type=mobilellm` configs;
- falling back to slow `LlamaTokenizer` when AutoTokenizer fails or returns a
  non-callable value.

## MobileLLM-ParetoQ Policies

MobileLLM-ParetoQ snapshots are cataloged separately from classic MobileLLM because their local configs are standard `model_type=llama` / `LlamaForCausalLM` artifacts. They do not need remote code, slow-tokenizer fallback, `use_cache=false`, or a runtime quantization package in `ai`; the ParetoQ weights are already baked into `pytorch_model.bin`.

The built-in smoke matrix is available with `scripts/smoke_mobilellm_family.py --family paretoq`. The full local family has dry-run coverage, and representative CUDA generation smokes cover quantized and BF16 checkpoints.

## Commands

Dry-run policy check:

```bash
python -m legacy_model_bridge.cli causal-lm generate \
  --model-id MobileLLM-125M \
  --model-root /arxiv/models \
  --prompt Hello \
  --max-new-tokens 4 \
  --dry-run
```

CUDA smoke:

```bash
conda run --no-capture-output -n ai env PYTHONNOUSERSITE=1 \
  PYTHONPATH=/data/legacy_model_bridge CUDA_VISIBLE_DEVICES=1 \
  python -m legacy_model_bridge.cli causal-lm generate \
  --model-id MobileLLM-125M --model-root /arxiv/models \
  --prompt Hello --max-new-tokens 4 --device cuda:0 --dtype float16
```

Family dry-run matrix:

```bash
conda run --no-capture-output -n ai env PYTHONNOUSERSITE=1 \
  PYTHONPATH=/data/legacy_model_bridge \
  python scripts/smoke_mobilellm_family.py \
  --dry-run --device cpu --cuda-label dryrun \
  --summary-json reports/causal-lm-smokes/MobileLLM-family.dryrun.ai.summary.json
```

CUDA family subset smoke:

```bash
conda run --no-capture-output -n ai env PYTHONNOUSERSITE=1 \
  PYTHONPATH=/data/legacy_model_bridge CUDA_VISIBLE_DEVICES=1 \
  python scripts/smoke_mobilellm_family.py \
  --model MobileLLM-350M-layer-share --model MobileLLM-600M \
  --device cuda:0 --dtype float16 --cuda-visible-devices 1 --cuda-label cuda1 \
  --summary-json reports/causal-lm-smokes/MobileLLM-family.partial.generate4.ai.cuda1.summary.json
```

ParetoQ dry-run matrix:

```bash
conda run --no-capture-output -n ai env PYTHONNOUSERSITE=1 \
  PYTHONPATH=/data/legacy_model_bridge \
  python scripts/smoke_mobilellm_family.py \
  --family paretoq --dry-run --device cpu --cuda-label dryrun \
  --summary-json reports/causal-lm-smokes/MobileLLM-ParetoQ-family.dryrun.ai.summary.json
```

ParetoQ representative CUDA smoke:

```bash
conda run --no-capture-output -n ai env PYTHONNOUSERSITE=1 \
  PYTHONPATH=/data/legacy_model_bridge CUDA_VISIBLE_DEVICES=1 \
  python scripts/smoke_mobilellm_family.py \
  --model MobileLLM-ParetoQ-125M-1-bit \
  --model MobileLLM-ParetoQ-350M-4-bit \
  --model MobileLLM-ParetoQ-125M-BF16 \
  --device cuda:0 --dtype float16 --cuda-visible-devices 1 --cuda-label cuda1 \
  --summary-json reports/causal-lm-smokes/MobileLLM-ParetoQ-family.representative.generate4.ai.cuda1.summary.json
```

## Smoke Reports

- `reports/causal-lm-smokes/MobileLLM-125M.generate4.ai.cuda1.json`
- `reports/causal-lm-smokes/MobileLLM-125M-layer-share.generate4.ai.cuda1.json`
- `reports/causal-lm-smokes/MobileLLM-family.dryrun.ai.summary.json`
- `reports/causal-lm-smokes/MobileLLM-350M.generate4.ai.cuda1.json`
- `reports/causal-lm-smokes/MobileLLM-350M-layer-share.generate4.ai.cuda1.json`
- `reports/causal-lm-smokes/MobileLLM-600M.generate4.ai.cuda1.json`
- `reports/causal-lm-smokes/MobileLLM-family.partial.generate4.ai.cuda1.summary.json`
- `reports/causal-lm-smokes/MobileLLM-family.1b-1_5b.generate4.ai.cuda1.summary.json`
- `reports/causal-lm-smokes/MobileLLM-1B.generate4.ai.cuda1.json`
- `reports/causal-lm-smokes/MobileLLM-1.5B.generate4.ai.cuda1.json`
- `reports/causal-lm-smokes/MobileLLM-family.r1_5.generate4.ai.cuda1.summary.json`
- `reports/causal-lm-smokes/MobileLLM-R1.5-140M.generate4.ai.cuda1.json`
- `reports/causal-lm-smokes/MobileLLM-R1.5-360M.generate4.ai.cuda1.json`
- `reports/causal-lm-smokes/MobileLLM-R1.5-950M.generate4.ai.cuda1.json`
- `reports/causal-lm-smokes/mobilellm-pro.cuda1.summary.json`
- `reports/causal-lm-smokes/MobileLLM-Pro.generate4.ai.cuda1.json`
- `reports/causal-lm-smokes/MobileLLM-family.pro-base.generate4.ai.cuda1.summary.json`
- `reports/causal-lm-smokes/MobileLLM-Pro-base.generate4.ai.cuda1.json`
- `reports/causal-lm-smokes/MobileLLM-ParetoQ-family.dryrun.ai.summary.json`
- `reports/causal-lm-smokes/MobileLLM-ParetoQ-family.representative.generate4.ai.cuda1.summary.json`
- `reports/causal-lm-smokes/MobileLLM-ParetoQ-125M-1-bit.generate4.ai.cuda1.json`
- `reports/causal-lm-smokes/MobileLLM-ParetoQ-350M-4-bit.generate4.ai.cuda1.json`
- `reports/causal-lm-smokes/MobileLLM-ParetoQ-125M-BF16.generate4.ai.cuda1.json`
