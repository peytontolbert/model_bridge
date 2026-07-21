from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DTYPE_BYTES = {
    "torch.float64": 8,
    "torch.float32": 4,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.int64": 8,
    "torch.int32": 4,
    "torch.int16": 2,
    "torch.int8": 1,
    "torch.uint8": 1,
    "torch.bool": 1,
}


def _tensor_nbytes(tensor: Any) -> int | None:
    shape = getattr(tensor, "shape", None)
    dtype = str(getattr(tensor, "dtype", ""))
    if shape is None or dtype not in DTYPE_BYTES:
        return None
    count = 1
    for dim in shape:
        count *= int(dim)
    return count * DTYPE_BYTES[dtype]


def _shape(tensor: Any) -> list[int] | None:
    shape = getattr(tensor, "shape", None)
    return [int(dim) for dim in shape] if shape is not None else None


def inspect_checkpoint(path: str | Path, *, sample_limit: int = 25) -> dict[str, Any]:
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    checkpoint_path = Path(path)
    with FakeTensorMode():
        state = torch.load(checkpoint_path, map_location="meta", mmap=True, weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"expected checkpoint to load as dict, got {type(state).__name__}")

    keys = list(state.keys())
    prefix_counts = Counter(key.split(".", 1)[0] for key in keys)
    nbytes_by_prefix: dict[str, int] = {}
    nbytes_by_block: dict[str, int] = {}
    samples = []
    total_bytes = 0
    unknown_bytes = 0
    for key in keys:
        value = state[key]
        nbytes = _tensor_nbytes(value)
        if nbytes is None:
            unknown_bytes += 1
        else:
            total_bytes += nbytes
            prefix = key.split(".", 1)[0]
            nbytes_by_prefix[prefix] = nbytes_by_prefix.get(prefix, 0) + nbytes
            parts = key.split(".")
            if len(parts) > 3 and parts[0] == "net" and parts[1] == "blocks":
                block = f"net.blocks.{parts[2]}"
                nbytes_by_block[block] = nbytes_by_block.get(block, 0) + nbytes
        if len(samples) < sample_limit:
            samples.append(
                {
                    "key": key,
                    "shape": _shape(value),
                    "dtype": str(getattr(value, "dtype", "")),
                    "nbytes": nbytes,
                }
            )
    likely_student_prefixes = [
        prefix
        for prefix in ("net", "net_ema")
        if prefix in prefix_counts
    ]
    return {
        "checkpoint_path": str(checkpoint_path),
        "format": "pytorch_zip_pt",
        "key_count": len(keys),
        "prefix_counts": dict(prefix_counts.most_common()),
        "nbytes_by_prefix": dict(sorted(nbytes_by_prefix.items())),
        "nbytes_by_block": dict(sorted(nbytes_by_block.items(), key=lambda item: int(item[0].rsplit(".", 1)[-1]))),
        "block_count": len(nbytes_by_block),
        "total_tensor_bytes": total_bytes,
        "total_tensor_gib": round(total_bytes / 1024**3, 3),
        "unknown_nbytes_entries": unknown_bytes,
        "likely_student_prefixes": likely_student_prefixes,
        "sample_keys": samples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--json-out")
    parser.add_argument("--sample-limit", type=int, default=25)
    args = parser.parse_args(argv)

    payload = inspect_checkpoint(args.checkpoint, sample_limit=args.sample_limit)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
