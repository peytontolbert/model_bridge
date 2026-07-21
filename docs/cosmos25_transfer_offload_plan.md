# Cosmos Transfer 2.5 Offload Plan

The bounded generation crawl for `nvidia/Cosmos-Transfer2.5-2B` shows that the
released distilled edge checkpoint is not the source of the 24 GiB OOM by
itself.

## Checkpoint Facts

Local checkpoint:

`/arxiv/models/nvidia/Cosmos-Transfer2.5-2B/distilled/general/edge/41f07f13-f2e4-4e34-ba4c-86f595acbc20_ema_bf16.pt`

Inspection report:

`reports/cosmos25-smokes/Cosmos-Transfer2.5-2B.checkpoint-inspect.json`

Findings:

- Format is PyTorch zip `.pt`, not safetensors.
- The checkpoint contains 796 tensor keys.
- Every key has the `net.` prefix.
- No `net_teacher.`, `net_fake_score.`, or `net_ema.` keys are present.
- Tensor payload is about 4.392 GiB.
- The student network has 28 repeated `net.blocks.*` groups, about 132 MiB each.

This means the released `.pt` is already an EMA-exported student checkpoint. A
student-only safetensors conversion is useful for lazy/indexed loading, but it
does not remove the largest runtime allocation.

## OOM Root Cause

Official distilled inference sets `model.config.load_teacher_weights=False`, so
it avoids loading teacher weights. It still constructs the teacher and fake-score
network modules in `DMD2Model.build_model`:

- `self.net_teacher = self.build_net(self.config.net_teacher)`
- `self.net_fake_score = self.build_net(self.config.net_fake_score)`

The inference wrapper only sets `net_fake_score = None` after pipeline
construction, which is too late for 24 GiB GPUs.

## Verified Bridge Smoke

`reports/cosmos25-smokes/Cosmos-Transfer2.5-2B.student-only.bounded-generation.gpu0-93f.json` records a successful 93-frame, 1-step distilled edge generation through the bridge worker on GPU 0. The run generated `/data/tmp/lmb_cosmos25_transfer_student_only_gpu0_93f/lmb_transfer25_edge_smoke_93f.mp4` and the matching edge control video. This proves the 24 GiB OOM was caused by avoidable runtime construction, not an unavoidable 2B checkpoint footprint.

The bounded smoke needed a 93-frame fixture because Transfer 2.5 uses `state_t=24`, and the tokenizer expects `(24 - 1) * 4 + 1 = 93` pixel frames. The earlier 9-frame shortcut correctly failed official validation after the OOM path was fixed.

## Bridge Work

Implemented bridge pieces:

- `legacy_model_bridge.runtime.cosmos25_student_only` applies an opt-in monkey
  patch before official Transfer inference instantiates `DMD2Model`.
- `scripts/cosmos25_transfer_student_only_infer.py` wraps the official
  `examples/inference.py` entrypoint so the patch can run without editing the
  vendor checkout.
- `Cosmos25Request(student_only=True)` and CLI `--student-only` route Transfer
  launch plans through the wrapper and record the
  `cosmos25_transfer_student_only_dmd2` patch in the launch plan.
- Text encoder CPU offload now temporarily moves the student net to CPU, computes
  prompt embeddings on CUDA, offloads the text encoder, then restores the student
  net before denoising.
- Tokenizer CPU offload now handles official tokenizer wrappers whose inner
  `.model` owns `.to(...)`.

Remaining work:

1. Solve `ai` parity for official Cosmos imports/native packages (`natten`,
   `transformer_engine`, `cosmos_transfer2.config`, and
   `cosmos_transfer2.inference`) or replace `cosmos25_py310` with a bridge-owned
   isolated worker image.
2. Smoke `nvidia/Cosmos-Predict2.5-14B`; Transfer 2.5 is verified, but Predict
   14B remains much larger and should keep the worker boundary until proven.
3. Convert the `.pt` to sharded safetensors only if future profiling needs
   indexed/manual placement; it was not required for the verified Transfer smoke.

## Tooling

Inspect checkpoint metadata without CUDA model construction:

```bash
conda run -n cosmos25_py310 env PYTHONNOUSERSITE=1 PYTHONPATH=. \
  python scripts/inspect_cosmos25_checkpoint.py \
  /arxiv/models/nvidia/Cosmos-Transfer2.5-2B/distilled/general/edge/41f07f13-f2e4-4e34-ba4c-86f595acbc20_ema_bf16.pt \
  --json-out reports/cosmos25-smokes/Cosmos-Transfer2.5-2B.checkpoint-inspect.json
```

Convert student tensors to sharded safetensors:

```bash
conda run -n cosmos25_py310 env PYTHONNOUSERSITE=1 PYTHONPATH=. \
  python scripts/convert_cosmos25_pt_to_safetensors.py \
  /arxiv/models/nvidia/Cosmos-Transfer2.5-2B/distilled/general/edge/41f07f13-f2e4-4e34-ba4c-86f595acbc20_ema_bf16.pt \
  --output-dir /data/tmp/cosmos_transfer25_edge_student_safetensors \
  --include-prefix net \
  --max-shard-size 2GiB
```
