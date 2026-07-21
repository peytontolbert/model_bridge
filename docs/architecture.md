# Architecture

Legacy Model Bridge has three layers:

1. Runtime layer
2. Integration layer
3. Validation layer

## Runtime Layer

The runtime layer contains shared building blocks used across many model ports.
It should stay small, stable, and well tested.

Expected modules:

```text
runtime/
  registry.py
  config.py
  checkpoint.py
  tensor_mapping.py
  tokenizer.py
  ops.py
  validation.py
```

Responsibilities:

- Register supported model integrations.
- Load legacy artifacts from disk.
- Convert legacy configs into modern config objects.
- Convert checkpoint tensor names and tensor layouts.
- Provide reusable compatibility helpers for common PyTorch and Transformers
  behavior changes.
- Provide standard validation utilities.

The runtime layer should not contain model-specific logic unless that logic is
shared by multiple integrations.

## Integration Layer

Each supported model gets its own directory:

```text
integrations/<model_name>/
  README.md
  profile.yaml
  adapter.py
  model.py
  config_map.py
  checkpoint_map.py
  tokenizer_bridge.py
  ops.py
  tests/
```

Responsibilities:

- Define the model's compatibility profile.
- Load original artifacts.
- Rebuild or wrap the model on the modern runtime.
- Convert checkpoints into the canonical format.
- Replace legacy kernels and binary extensions.
- Document known limitations and validation status.

Integrations should be explicit. If a model requires special behavior, encode it
in that model's adapter instead of adding broad global hacks.

## Validation Layer

Every model integration should have at least:

- import smoke test
- config load test
- checkpoint mapping test
- minimal forward pass test
- CPU test when feasible
- GPU test when the model requires CUDA

Numerical parity tests are preferred when reference outputs are available.
Parity tolerances must be documented per model.

## CLI Shape

Initial CLI commands should be:

```bash
legacy-model-bridge list
legacy-model-bridge doctor <model>
legacy-model-bridge convert <model> --source <path> --out <path>
legacy-model-bridge run <model> --model <path> [-- ...]
legacy-model-bridge validate <model> --model <path>
```

The CLI should make compatibility decisions visible. If a model cannot run on
the current runtime, the error should explain exactly what is missing or
unimplemented.

