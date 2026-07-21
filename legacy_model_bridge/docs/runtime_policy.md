# Runtime Policy

Legacy Model Bridge should minimize environment sprawl while staying honest
about runtime requirements.

## Default Runtime

The bridge package and CLI support Python 3.11 and Python 3.12 as caller
runtimes. Project metadata is intentionally constrained to `>=3.11,<3.13` until
3.13 has real validation.

The default caller runtime should track current maintained libraries:

```text
Python 3.11 or 3.12
PyTorch current stable
Transformers current stable
safetensors
tokenizers
accelerate
triton
```

Model backends are allowed to use a different Python version only behind a
bridge-owned adapter or worker. That backend version is evidence, not the public
API. For example, a Python 3.12 caller may invoke a TRELLIS worker that currently
runs Python 3.10, while the bridge owns request serialization, worker launch, and
artifact normalization.

The exact default versions should be pinned in a lock file once the project has
code and tests.

## Compatibility Rule

Prefer porting models forward over installing old dependencies. Every crawled
model should record literal mismatch classes in `data/environment_consolidation.json`:

- `bridge_patch_required`: API/import/config drift fixable in this repo.
- `missing_runtime_package`: a modern package or vendored runtime is missing.
- `model_specific_runtime_adapter`: generic loaders are wrong; write an adapter.
- `python_abi_or_native_extension`: compiled extension or Python ABI mismatch.
- `torch_cuda_native_stack`: CUDA/PyTorch kernel stack mismatch.
- `native_or_runtime_worker_boundary`: keep a backend worker until the native stack is ported.

Allowed modernization techniques:

- local adapters
- checkpoint conversion
- config conversion
- tokenizer wrappers
- local model implementations
- modern replacement kernels
- local emulation of legacy behavior

## When To Add A Separate Runtime

Add a separate runtime only when:

- a model is valuable but not yet ported
- a required kernel has no replacement yet
- a validation baseline requires the old stack
- the model depends on hardware behavior that differs across major CUDA or
  PyTorch versions

Separate runtimes should be treated as temporary unless there is a clear reason
to keep them. Even when a backend runtime remains, user-facing invocation should
stay in the Python 3.11/3.12 bridge API.

## Environment Naming

Use names based on compatibility, not project history:

```text
lmb-default
lmb-vllm
lmb-cu124
lmb-reference-old
```

Avoid one-off names that only make sense locally.

## Cache Policy

Large caches should live on data storage, not home directories.

Recommended locations:

```text
/data/cache/huggingface
/data/cache/torch
/data/cache/pip
/data/conda/envs
/data/conda/pkgs
```

Recommended environment variables:

```bash
export HF_HOME=/data/cache/huggingface
export TRANSFORMERS_CACHE=/data/cache/huggingface/transformers
export TORCH_HOME=/data/cache/torch
export PIP_CACHE_DIR=/data/cache/pip
```

