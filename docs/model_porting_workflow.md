# Model Porting Workflow

This is the standard workflow for integrating one legacy model.

## 1. Inventory The Legacy Model

Record:

- original repository URL or local source path
- model family and task
- original Python version
- original PyTorch version
- original Transformers version, if used
- CUDA requirements
- custom extensions or kernels
- checkpoint format
- tokenizer or processor format
- minimal command that used to run

Do not start by recreating the whole old environment unless it is needed to
produce reference outputs.

## 2. Define The Target Runtime

Prefer the latest project-standard runtime. Only add an older runtime when a
model cannot reasonably be ported yet.

Example target:

```text
Python >=3.11
PyTorch current stable
Transformers current stable
safetensors
tokenizers
accelerate
triton
```

## 3. Run Dependency Diagnostics Before Full Load

Before materializing large weights, classify the model with safe probes:

- config and model index files
- checkpoint format and shard completeness
- local custom Python imports
- declared package versions from model metadata and README examples
- installed package versions in candidate envs
- known bridge-lane minimums

This phase should identify whether the problem is a narrow API drift patch,
missing package, missing runtime source, incomplete checkpoint, adapter/base-model
gap, or validation-only gap.

## 4. Create The Integration Skeleton

```text
integrations/<model_name>/
  README.md
  profile.yaml
  adapter.py
  checkpoint_map.py
  tests/
```

The first pull request for a model can be small. It should at least identify the
legacy artifacts and create a failing or skipped validation path for missing
work.

## 5. Port Config Loading

Map legacy config fields into a modern config object. Preserve source config
values in metadata so conversion decisions remain auditable.

Common fixes:

- renamed fields
- removed defaults
- changed enum names
- generation defaults
- dtype and device defaults
- rope, attention, or cache settings

## 6. Port The Model Code

Use current PyTorch APIs. Avoid importing large sections of old repositories
unchanged unless they are reviewed and owned by this project.

When old behavior matters, reimplement it locally and test it.

## 7. Convert Checkpoints

Build `checkpoint_map.py` to translate legacy tensor names and layouts.

The converter should fail loudly on:

- missing required tensors
- unexpected tensor shapes
- unsupported checkpoint variants
- ambiguous mapping rules

## 8. Replace Legacy Ops

For old custom ops, choose one:

- modern PyTorch implementation
- Triton implementation
- current CUDA extension
- maintained third-party equivalent
- slower fallback with documented performance cost

The replacement should match behavior first. Optimization can come after the
model runs correctly.

Native extensions must be optional at import time, discoverable at runtime, and
covered by a Python, eager PyTorch, Triton, or documented slower fallback.

## 9. Validate

Minimum validation:

```bash
legacy-model-bridge doctor <model>
legacy-model-bridge convert <model> --source <legacy-path> --out <modern-path>
legacy-model-bridge validate <model> --model <modern-path>
```

If reference outputs exist, compare with documented tolerances.

Validation commands should set `PYTHONNOUSERSITE=1` when using Conda or worker
envs so user-site packages do not hide dependency blockers.

## 10. Document Support

The integration README should explain:

- what artifact variants are supported
- what has been replaced
- expected runtime requirements
- known limitations
- validation status

