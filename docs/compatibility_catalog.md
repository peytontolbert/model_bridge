# Compatibility Catalog

`data/bridge_catalog.json` is the current source of truth for manually included
models. It is intentionally compact and curated. Do not copy the full research
library model index into this repo.

## Entry Fields

| Field | Meaning |
| --- | --- |
| `model_id` | Stable catalog or Hugging Face-style identifier. |
| `lane` | Runtime bridge lane that owns loading and validation behavior. |
| `status` | Current support or blocker state. |
| `preferred_env` | First environment or worker target to use. |
| `runnable` | Whether a validated run path exists today. |
| `compatibility_patches` | Named patches required or applied for current dependencies. |
| `source_refs` | Docs or reports that justify the decision. |
| `notes` | Short operational caveat or next action. |

## Promotion Flow

1. Run `python scripts/derive_bridge_candidates.py --limit 100` against the
   external model index.
2. Pick one candidate family and inspect its local artifacts.
3. Add or update one curated `data/bridge_catalog.json` entry.
4. Add a matching integration profile when implementation begins.
5. Attach validation report paths once smoke tests exist.

## Current Principle

Use `lmb-default` or the latest verified lane env first. Keep separate envs only
when the blocker is structural, such as incompatible native geometry stacks,
NeMo archive runtime requirements, or a model-specific CUDA/toolkit dependency.
