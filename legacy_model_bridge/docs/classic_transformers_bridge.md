# Classic Transformers Bridge

This lane covers older standard Transformers artifacts that should run in the latest `ai` environment without preserving their original exporter/runtime environments.

Validated latest environment:

- Python 3.11.11
- Torch 2.10.0+cu128
- Transformers 4.57.6
- CUDA available on three RTX 3090 GPUs

## Covered Models

| Model | Task | Bridge Loader | Smoke |
| --- | --- | --- | --- |
| `facebook/bart-large-cnn` | `seq2seq_generation` | `AutoTokenizer` + `AutoModelForSeq2SeqLM` | CUDA fp16 bounded generate |
| `gpt2` | `causal_lm_generation` | `AutoTokenizer` + `AutoModelForCausalLM` | CPU bounded generate |
| `distilgpt2` | `causal_lm_generation` | `AutoTokenizer` + `AutoModelForCausalLM` | CPU bounded generate |
| `sshleifer/tiny-gpt2` | `causal_lm_generation` | `AutoTokenizer` + `AutoModelForCausalLM` | CPU bounded generate |
| `sentence-transformers/all-MiniLM-L6-v2` | `text_encoder` | `AutoTokenizer` + `AutoModel` | CPU text forward |
| `intfloat/e5-base-v2` | `text_encoder` | `AutoTokenizer` + `AutoModel` | CPU text forward |
| `repository_library/file-embedding` | `text_encoder` | `AutoTokenizer` + `AutoModel` | CPU text forward |
| `repository_library/cross-encoder-reranker` | `sequence_classification` | `AutoTokenizer` + `AutoModelForSequenceClassification` | CPU logits forward |
| `hubert-base-ls960` | `audio_encoder` | `AutoFeatureExtractor` + `AutoModel` | CUDA fp16 zero-audio forward |
| `hubert-large-ll60k` | `audio_encoder` | `AutoFeatureExtractor` + `AutoModel` | CUDA fp16 zero-audio forward |
| `hubert-xlarge-ll60k` | `audio_encoder` | `AutoFeatureExtractor` + `AutoModel` | CUDA fp16 zero-audio forward |
| `dinov2-base` | `vision_encoder` | `AutoImageProcessor` + `AutoModel` | CUDA fp16 zero-image forward |
| `dinov2-large` | `vision_encoder` | `AutoImageProcessor` + `AutoModel` | CUDA fp16 zero-image forward |
| `webssl-dino300m-full2b-224` | `vision_encoder` | `AutoImageProcessor` + `AutoModel` | CUDA fp16 zero-image forward |

## Bridge Mismatches

- BART artifacts were exported by old Transformers versions and carry summarization generation defaults. Tiny bridge smokes override `min_length=0` so `max_new_tokens` can remain bounded.
- HuBERT encoder artifacts advertise tokenizer classes but do not include tokenizer vocab files. The bridge intentionally uses feature-extractor-only loading for encoder forwards.
- Hugging Face cache artifacts resolve from `/data/huggingface/hub/models--*/snapshots/*`, so callers can pass model ids like `gpt2` or `sentence-transformers/all-MiniLM-L6-v2` without snapshot hashes.
- Latest Transformers prefers `dtype=` for model loading. The bridge uses that keyword first and falls back to `torch_dtype=` only for older loaders.
- GPT-2 tokenizers generally have no pad token; bounded generation passes `eos_token_id` as `pad_token_id` when needed.
- BERT-family artifacts route by architecture: plain encoders use `AutoModel`, sequence classifiers use `AutoModelForSequenceClassification`, and masked-LM heads use `AutoModelForMaskedLM`.
- HuBERT and DINOv2 fp16 model loads need floating synthetic inputs cast to the loaded model dtype. Integer token tensors are left unchanged.
- DINOv2 uses `BitImageProcessor`; the bridge routes vision models through `AutoImageProcessor`, with `use_fast=False` to preserve the saved slow-processor behavior across future Transformers defaults.

## Reports

The synthetic reports include requested device/dtype plus runtime Python, Torch, Transformers, CUDA availability, and CUDA device name. BART generated-token counts are decoder output lengths because encoder-decoder outputs do not include prompt tokens.

- `reports/classic-transformers-smokes/classic-transformers.inspect.json`
- `reports/classic-transformers-smokes/classic-transformers.base.synthetic.cuda1.json`
- `reports/classic-transformers-smokes/classic-transformers.large.synthetic.cuda1.json`
- `reports/classic-transformers-smokes/gpt2-family.synthetic.cpu.ai.json`
- `reports/classic-transformers-smokes/bert-family.synthetic.cpu.ai.json`
- `reports/classic-transformers-smokes/hubert-xlarge-webssl-dino.synthetic.cuda1.ai.json`

CLI examples:

```bash
python -m legacy_model_bridge.cli classic-transformers inspect --model-id hubert-base-ls960
python scripts/inspect_classic_transformers.py --model gpt2 --run-synthetic --device cpu --max-new-tokens 4
python scripts/inspect_classic_transformers.py --model dinov2-base --run-synthetic --device cuda:0 --dtype float16
```
