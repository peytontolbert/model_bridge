from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


CLASSIFIER_WEIGHT_KEY = "classifier.weight"
CLASSIFIER_BIAS_KEY = "classifier.bias"
PATCH_ID = "transformers_classifier_head_num_labels_from_checkpoint"


class EncoderClassifierBridgeError(RuntimeError):
    pass


@dataclass(frozen=True)
class HeadShape:
    num_labels: int
    hidden_size: int
    has_bias: bool


def _shape_tuple(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", value)
    try:
        return tuple(int(dim) for dim in shape)
    except TypeError as exc:
        raise EncoderClassifierBridgeError(f"invalid tensor shape metadata: {shape!r}") from exc


def classifier_head_shape(tensors: Mapping[str, Any]) -> HeadShape:
    if CLASSIFIER_WEIGHT_KEY not in tensors:
        raise EncoderClassifierBridgeError(f"missing required tensor: {CLASSIFIER_WEIGHT_KEY}")
    weight_shape = _shape_tuple(tensors[CLASSIFIER_WEIGHT_KEY])
    if len(weight_shape) != 2:
        raise EncoderClassifierBridgeError(
            f"{CLASSIFIER_WEIGHT_KEY} must be rank 2, got shape {weight_shape}"
        )
    num_labels, hidden_size = weight_shape
    has_bias = CLASSIFIER_BIAS_KEY in tensors
    if has_bias:
        bias_shape = _shape_tuple(tensors[CLASSIFIER_BIAS_KEY])
        if bias_shape != (num_labels,):
            raise EncoderClassifierBridgeError(
                f"{CLASSIFIER_BIAS_KEY} shape {bias_shape} does not match classifier label count {num_labels}"
            )
    return HeadShape(num_labels=num_labels, hidden_size=hidden_size, has_bias=has_bias)


def generic_label_maps(num_labels: int) -> tuple[dict[int, str], dict[str, int]]:
    id2label = {idx: f"LABEL_{idx}" for idx in range(num_labels)}
    label2id = {label: idx for idx, label in id2label.items()}
    return id2label, label2id


def repair_config_from_classifier_head(config: Any, tensors: Mapping[str, Any]) -> Any:
    head = classifier_head_shape(tensors)
    id2label, label2id = generic_label_maps(head.num_labels)
    config.num_labels = head.num_labels
    config.id2label = id2label
    config.label2id = label2id
    return config


def load_safetensors_metadata(path: str | Path) -> dict[str, tuple[int, ...]]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise EncoderClassifierBridgeError("safetensors is required to inspect safetensors checkpoint metadata") from exc

    shapes: dict[str, tuple[int, ...]] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            shapes[key] = tuple(handle.get_tensor(key).shape)
    return shapes


def load_torch_checkpoint_metadata(path: str | Path) -> dict[str, tuple[int, ...]]:
    try:
        import torch
    except ImportError as exc:
        raise EncoderClassifierBridgeError("torch is required to inspect PyTorch checkpoint metadata") from exc

    raw = torch.load(Path(path), map_location="cpu")
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    if not isinstance(state, Mapping):
        raise EncoderClassifierBridgeError("PyTorch checkpoint did not contain a tensor mapping")
    return {str(key): _shape_tuple(value) for key, value in state.items() if hasattr(value, "shape")}


def find_checkpoint_file(source_path: str | Path) -> Path:
    source = Path(source_path)
    candidates = [
        source / "model.safetensors",
        source / "pytorch_model.bin",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    safetensors = sorted(source.glob("*.safetensors"))
    if safetensors:
        return safetensors[0]
    bins = [path for path in sorted(source.glob("*.bin")) if path.name != "training_args.bin"]
    if bins:
        return bins[0]
    raise EncoderClassifierBridgeError(f"no supported checkpoint file found under {source}")


def load_checkpoint_metadata(source_path: str | Path) -> dict[str, tuple[int, ...]]:
    checkpoint = find_checkpoint_file(source_path)
    if checkpoint.suffix == ".safetensors":
        return load_safetensors_metadata(checkpoint)
    return load_torch_checkpoint_metadata(checkpoint)


def load_repaired_sequence_classifier(source_path: str | Path, **kwargs: Any) -> Any:
    try:
        from transformers import AutoConfig, AutoModelForSequenceClassification
    except ImportError as exc:
        raise EncoderClassifierBridgeError("transformers is required to load encoder classifier models") from exc

    source = Path(source_path)
    config = AutoConfig.from_pretrained(source, **kwargs.pop("config_kwargs", {}))
    metadata = load_checkpoint_metadata(source)
    repaired = repair_config_from_classifier_head(config, metadata)
    return AutoModelForSequenceClassification.from_pretrained(source, config=repaired, **kwargs)
