# rmbg_2_0

Model ID: `RMBG-2.0`

Bridge lane: `encoder_classifier_bridge`

Preferred env: `ai`

Catalog status: `verified_image_segmentation_cuda_smoke_ai`

Runnable: `true`

Compatibility patches: none

## Source References

- /data/transformer_10/docs/model-stack-conda-envs.md
- reports/encoder-classifier-smokes/RMBG-2.0.image_segmentation.cuda0.ai.json

## Checklist

- [ ] Config load
- [ ] Checkpoint map or conversion
- [ ] Model construction
- [ ] Minimal forward pass
- [ ] Validation report

## Known Limitations

AutoModelForImageSegmentation remote-code load runs as BiRefNet in ai after adding kornia; 256x256 CUDA FP32 forward smoke passes with output_shape=[1, 1, 8, 8].
