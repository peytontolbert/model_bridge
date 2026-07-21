from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from scripts.inspect_cosmos25_checkpoint import _tensor_nbytes


def _parse_size(value: str) -> int:
    raw = value.strip().lower()
    units = {
        "b": 1,
        "kb": 1000,
        "kib": 1024,
        "mb": 1000**2,
        "mib": 1024**2,
        "gb": 1000**3,
        "gib": 1024**3,
    }
    for suffix, multiplier in sorted(units.items(), key=lambda item: len(item[0]), reverse=True):
        if raw.endswith(suffix):
            return int(float(raw[: -len(suffix)]) * multiplier)
    return int(raw)


def _load_state_dict(path: str | Path) -> dict[str, Any]:
    import torch

    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"expected checkpoint to load as dict, got {type(state).__name__}")
    if "model" in state and isinstance(state["model"], dict):
        return state["model"]
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        return state["state_dict"]
    return state


def _selected_items(
    state: dict[str, Any],
    *,
    include_prefix: str,
    strip_prefix: bool,
) -> list[tuple[str, Any]]:
    prefix = include_prefix.rstrip(".") + "."
    selected = []
    for key, value in state.items():
        if not key.startswith(prefix):
            continue
        output_key = key[len(prefix) :] if strip_prefix else key
        selected.append((output_key, value.contiguous() if hasattr(value, "contiguous") else value))
    if not selected:
        raise ValueError(f"no tensors matched prefix {include_prefix!r}")
    return selected


def convert_checkpoint(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    include_prefix: str = "net",
    strip_prefix: bool = False,
    max_shard_size: str = "2GiB",
) -> dict[str, Any]:
    from safetensors.torch import save_file

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = _load_state_dict(checkpoint)
    selected = _selected_items(state, include_prefix=include_prefix, strip_prefix=strip_prefix)
    max_bytes = _parse_size(max_shard_size)
    shards: list[dict[str, Any]] = []
    current: OrderedDict[str, Any] = OrderedDict()
    current_bytes = 0
    weight_map: dict[str, str] = {}

    def flush() -> None:
        nonlocal current, current_bytes
        if not current:
            return
        shard_name = f"model-{len(shards) + 1:05d}-of-PLACEHOLDER.safetensors"
        shards.append({"name": shard_name, "tensors": current, "bytes": current_bytes})
        current = OrderedDict()
        current_bytes = 0

    for key, value in selected:
        nbytes = _tensor_nbytes(value) or 0
        if current and current_bytes + nbytes > max_bytes:
            flush()
        current[key] = value
        current_bytes += nbytes
    flush()

    total = len(shards)
    total_bytes = 0
    shard_files = []
    for index, shard in enumerate(shards, start=1):
        final_name = f"model-{index:05d}-of-{total:05d}.safetensors"
        path = out_dir / final_name
        save_file(shard["tensors"], path)
        total_bytes += int(shard["bytes"])
        shard_files.append(final_name)
        for key in shard["tensors"].keys():
            weight_map[key] = final_name

    index_payload = {
        "metadata": {
            "source_checkpoint": str(checkpoint),
            "include_prefix": include_prefix,
            "strip_prefix": strip_prefix,
            "total_size": total_bytes,
            "format": "safetensors_sharded",
        },
        "weight_map": weight_map,
    }
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps(index_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(out_dir),
        "shard_files": shard_files,
        "tensor_count": len(weight_map),
        "total_bytes": total_bytes,
        "index_path": str(out_dir / "model.safetensors.index.json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-prefix", default="net")
    parser.add_argument("--strip-prefix", action="store_true")
    parser.add_argument("--max-shard-size", default="2GiB")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)

    payload = convert_checkpoint(
        args.checkpoint,
        args.output_dir,
        include_prefix=args.include_prefix,
        strip_prefix=args.strip_prefix,
        max_shard_size=args.max_shard_size,
    )
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
