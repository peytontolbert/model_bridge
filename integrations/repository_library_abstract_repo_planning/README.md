# repository_library_abstract_repo_planning

Model ID: `repository_library/abstract-repo-planning`

Bridge lane: `encoder_classifier_bridge`

Preferred env: `ai`

Catalog status: `works_cuda_forward_smoke_with_config_patch`

Runnable: `true`

Compatibility patches: transformers_classifier_head_num_labels_from_checkpoint

## Source References

- /data/transformer_10/docs/model-stack-model-verification.md
- /data/transformer_10/docs/model-stack-compatibility-patches.md

## Checklist

- [ ] Config load
- [ ] Checkpoint map or conversion
- [ ] Model construction
- [ ] Minimal forward pass
- [ ] Validation report

## Known Limitations

BERT classifier head shape is authoritative over stale config.num_labels.
