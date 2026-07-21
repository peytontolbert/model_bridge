# Integration Contract

An integration is the unit of support in Legacy Model Bridge. A model is
supported only when it has a registered integration and passing validation.

## Required Files

Each integration should provide:

```text
integrations/<model_name>/
  README.md
  profile.yaml
  adapter.py
  checkpoint_map.py
  tests/
```

Additional files such as `model.py`, `config_map.py`, `tokenizer_bridge.py`, and
`ops.py` should be added when the model needs them.

## Profile

`profile.yaml` describes the model's compatibility envelope.

Example:

```yaml
name: trellis
status: verified
bridge_lane: three_d_gen_bridge
preferred_env: ai
fallback_envs: []
target_runtime:
  python: ">=3.11"
  torch: ">=2.10"
  transformers: ">=4.57"
artifacts:
  config: required
  checkpoint: required
  tokenizer: optional
compatibility:
  patches_required: []
  patches_applied: []
  checkpoint_conversion: optional
  custom_ops: rebuilt_or_wrapped
  numerical_parity: approximate
validation:
  smoke: required
  forward: required
  reference_outputs: optional
  reports:
    - reports/world-model-smokes/trellis2.tiny_generate.real_image.ai.json
blocked_reason: null
```

Required bridge fields:

- `bridge_lane`: one of the shared runtime lanes such as
  `transformers_causal_lm_bridge`, `diffusers_cuda_bridge`,
  `video_diffusion_bridge`, `encoder_classifier_bridge`, `peft_adapter_bridge`,
  `nemo_asr_bridge`, `three_d_gen_bridge`, or a model-specific custom lane.
- `preferred_env`: the first environment or worker target to validate.
- `fallback_envs`: temporary environments to try only when the preferred env has
  a structural blocker.
- `patches_required`: named compatibility patches that must be available.
- `patches_applied`: named patches observed in validation reports.
- `validation.reports`: generated JSON or Markdown evidence paths.
- `blocked_reason`: concrete blocker when status is `blocked` or
  `needs_custom_bridge_or_env`.

## Adapter Responsibilities

The adapter must:

- Detect whether a source model directory is compatible with the integration.
- Load the original config and metadata.
- Build the modern model object.
- Load or convert checkpoint weights.
- Apply tokenizer or processor compatibility when required.
- Expose a standard `load_model` entry point.

## Checkpoint Conversion

Checkpoint conversion should be deterministic and repeatable.

The converter should:

- read original checkpoint files
- map old tensor keys to canonical keys
- transform tensor layouts when needed
- validate missing and unexpected keys
- write `safetensors` when possible
- write conversion metadata

Conversion metadata should include:

- source path
- source checkpoint hashes when practical
- integration name and version
- runtime library versions
- timestamp
- warnings and unsupported fields

## Validation Requirements

A model integration is not considered usable until it has:

- a config load test
- a checkpoint conversion test or checkpoint key-map test
- a minimal model construction test
- a minimal forward pass test

For models where numerical parity matters, include a reference fixture with
documented tolerances.

## Status Levels

Use these statuses:

```text
planned       Integration is listed but not implemented.
experimental  Loads and runs, but API or parity may change.
beta          Validated on representative artifacts.
stable        Validated, documented, and unlikely to change.
blocked       Known missing component prevents support.
```


Bridge triage may also use finer-grained pre-integration statuses in the catalog:

```text
patch_available
patch_candidate
not_needed
needs_env_package
needs_custom_bridge_or_env
incomplete_snapshot
adapter_needs_base_model
validation_pending
runtime_source_gap
checkpoint_gap
```
