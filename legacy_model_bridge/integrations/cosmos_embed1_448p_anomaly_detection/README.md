# cosmos_embed1_448p_anomaly_detection

Model ID: `Cosmos-Embed1-448p-anomaly-detection`

Bridge lane: `world_model_bridge`

Preferred env: `ai`

Catalog status: `verified_text_embedding_smoke_ai`

Runnable: `true`

Compatibility patches: transformers_apply_chunking_to_forward_compat, transformers_modeling_utils_pruning_helpers_compat

## Source References

- /data/transformer_10/docs/model-stack-conda-envs.md
- /data/transformer_10/docs/model-stack-compatibility-patches.md

## Checklist

- [ ] Config load
- [ ] Checkpoint map or conversion
- [ ] Model construction
- [ ] Minimal forward pass
- [ ] Validation report

## Known Limitations

FP32 text embedding smoke passes in ai with modeling_utils legacy helper shims; text_proj_shape=[1, 768].
