#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from legacy_model_bridge.runtime.transformers_compat import ensure_modeling_utils_legacy_helpers
from legacy_model_bridge.runtime.world_model import load_cosmos_embed1


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return [int(dim) for dim in shape]
    for attr in ("text_proj", "video_proj", "last_hidden_state", "logits"):
        nested = getattr(value, attr, None)
        if nested is not None:
            nested_shape = _shape(nested)
            if nested_shape is not None:
                return nested_shape
    if isinstance(value, dict):
        for key in ("text_proj", "video_proj", "last_hidden_state", "logits"):
            if key in value:
                nested_shape = _shape(value[key])
                if nested_shape is not None:
                    return nested_shape
    if isinstance(value, (tuple, list)) and value:
        return _shape(value[0])
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke Cosmos Embed1 text embedding with compatibility patches.")
    parser.add_argument("model_path")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--text", default="a compatibility smoke test")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--config-only", action="store_true", help="Only validate config/processor construction; do not load weights.")
    args = parser.parse_args()

    started = time.time()
    patches_applied = ensure_modeling_utils_legacy_helpers()

    payload: dict[str, Any] = {
        "model_path": str(Path(args.model_path)),
        "dtype": args.dtype,
        "device": args.device,
        "compatibility_patches_applied": patches_applied,
    }

    try:
        if args.config_only:
            from transformers import AutoConfig, AutoProcessor

            config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
            processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
            payload.update(
                {
                    "status": "ok",
                    "config_class": type(config).__name__,
                    "processor_class": type(processor).__name__,
                }
            )
        else:
            import torch

            artifacts = load_cosmos_embed1(args.model_path, dtype=args.dtype, device=args.device)
            model = artifacts["model"]
            processor = artifacts["processor"]
            selected_device = artifacts["device"]
            dtype = artifacts["dtype"]
            inputs = processor(text=args.text)
            if hasattr(inputs, "to"):
                inputs = inputs.to(selected_device, dtype=dtype)
            with torch.inference_mode():
                if hasattr(model, "get_text_embeddings"):
                    embeddings = model.get_text_embeddings(**inputs)
                else:
                    embeddings = model(**inputs)
            payload.update(
                {
                    "status": "ok",
                    "model_class": type(model).__name__,
                    "processor_class": type(processor).__name__,
                    "text_proj_shape": _shape(embeddings),
                }
            )
    except Exception as exc:
        payload.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})
    payload["elapsed_seconds"] = round(time.time() - started, 3)

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text)
    print(text, end="")
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
