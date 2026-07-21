#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return [int(dim) for dim in shape]
    if isinstance(value, (tuple, list)) and value:
        return _shape(value[0])
    if isinstance(value, dict):
        for nested in value.values():
            result = _shape(nested)
            if result is not None:
                return result
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke RMBG image segmentation in the current env.")
    parser.add_argument("model_path")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    started = time.time()
    payload: dict[str, Any] = {
        "model_path": str(Path(args.model_path)),
        "device": args.device,
        "dtype": args.dtype,
        "size": args.size,
    }
    try:
        import torch
        from transformers import AutoModelForImageSegmentation

        dtype = getattr(torch, args.dtype)
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = AutoModelForImageSegmentation.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            local_files_only=True,
        ).eval().to(device)
        inputs = torch.zeros((1, 3, args.size, args.size), device=device, dtype=dtype)
        with torch.inference_mode():
            output = model(inputs)
        payload.update(
            {
                "status": "ok",
                "model_class": type(model).__name__,
                "output_shape": _shape(output),
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
