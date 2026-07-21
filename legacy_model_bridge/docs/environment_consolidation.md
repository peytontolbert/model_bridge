# Environment Consolidation

The goal is to make `ai` the latest/default runtime and remove one-off legacy
Conda environments whenever the model can be carried forward with bridge-owned
compatibility behavior.

Do not treat every non-`ai` model as a package-install problem. Some stacks have
structural native dependency conflicts and should move behind explicit worker
boundaries until their upstream runtime supports the latest env cleanly.

## Commands

```bash
python -m legacy_model_bridge.cli consolidation --summary
python -m legacy_model_bridge.cli consolidation --current-env cosmos25_py310
python -m legacy_model_bridge.cli list --env-policy isolated_worker
python -m legacy_model_bridge.cli doctor HunyuanVideo-Avatar
```

## Python Caller Compatibility

The bridge package targets Python 3.11 and 3.12 for callers. Consolidation data
records `caller_python` separately from `backend_python` so a model can be usable
from either caller version even while a backend worker remains pinned to Python
3.10, 3.11, or 3.12. `mismatch_classes` records what still needs to be ported.

Useful commands:

```bash
python -m legacy_model_bridge.cli consolidation --summary
python -m legacy_model_bridge.cli consolidation --caller-python 3.12
python -m legacy_model_bridge.cli consolidation --decision worker_boundary_required
```

## Decision Meanings

| Decision | Meaning |
| --- | --- |
| `latest_env_ready` | Already validated in `ai`; no legacy env is needed. |
| `latest_env_with_patches` | Move to `ai` with registered compatibility patches. |
| `latest_env_patch_candidate` | Narrow patch is known, but still needs `ai` validation. |
| `latest_env_bridge_candidate` | Runtime should live in `ai`, but bridge implementation or bounded generation is pending. |
| `latest_env_needs_packages` | Latest env likely works after normal package additions and validation. |
| `latest_env_needs_packages_or_worker` | Try latest-env package support first; keep worker boundary if package stack is incompatible. |
| `latest_env_needs_runtime_package` | Model runtime package/classes are missing from latest env. |
| `custom_bridge_needed_in_latest_env` | Artifacts are present, but generic loaders are wrong; write a model-specific bridge. |
| `worker_boundary_required` | Do not flatten into one Python process yet; use an isolated worker and shared artifact contract. |

## Current Removal Order

1. Latest-env ready: keep only `ai` paths for MobileLLM, BERT classifiers,
   AnyFlow, Hunyuan3D 2.x shape paths, FLUX.2-klein, CogVideoX, and Cosmos3.
2. Narrow patch candidates: validate Cosmos Embed1 anomaly in `ai` with
   `transformers_apply_chunking_to_forward_compat`.
3. Package candidates: RMBG has been migrated to `ai` after adding `kornia`;
   evaluate NeMo archive restore under latest Python/Torch before deciding worker image.
4. Bridge candidates: Wan Animate now has a bridge-owned `ai` cached/int8
   inspection proof; continue bounded `ai` smokes for LingBot World, DAM, MOSS
   SoundEffect, PixelDiT, HunyuanWorld, HunyuanVideo-I2V, and Hunyuan3D-Omni.
5. Structural worker boundaries: keep TRELLIS, Cosmos 2.5, and Hunyuan Avatar
   behind bridge-owned worker boundaries until their native/runtime stacks are
   validated on the latest env.

## Worker Boundary Rule

A worker boundary is acceptable backward compatibility when the old environment
cannot be collapsed without mixing incompatible native dependency stacks. The
user-facing runtime should still be `ai`; the bridge owns worker dispatch so
users do not manually switch Conda environments. The bridge must still own the
public API and normalize outputs, for example:

- TRELLIS and Hunyuan3D return `mesh_bundle_glb` artifacts.
- NeMo ASR should expose a warm transcription service, not per-request archive
  restores.
- Cosmos 2.5 should launch official examples with local checkpoint path
  overrides and no implicit downloads.

## Latest Env Package Additions

- `kornia==0.8.3` was installed into `ai` so `RMBG-2.0` can run without the
  `trellis` env. `timm==1.0.27` was already present. The validation report is
  `reports/encoder-classifier-smokes/RMBG-2.0.image_segmentation.cuda0.ai.json`.
