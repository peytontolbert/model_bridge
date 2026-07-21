#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legacy_model_bridge.runtime.bigvgan import BigVGANRequest, inspect_bigvgan, to_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect or synthetic-smoke BigVGAN vocoder artifacts.")
    parser.add_argument("--model-id", default="bigvgan_v2_44khz_128band_512x")
    parser.add_argument("--model-root", default="/arxiv/models")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--mel-frames", type=int, default=4)
    parser.add_argument("--run-synthetic", action="store_true")
    parser.add_argument("--use-cuda-kernel", action="store_true")
    parser.add_argument("--keep-weight-norm", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)

    result = inspect_bigvgan(
        BigVGANRequest(
            model_id=args.model_id,
            model_root=args.model_root,
            device=args.device,
            dtype=args.dtype,
            mel_frames=args.mel_frames,
            run_synthetic=args.run_synthetic,
            use_cuda_kernel=args.use_cuda_kernel,
            remove_weight_norm=not args.keep_weight_norm,
        )
    )
    text = json.dumps(to_json(result), indent=2, sort_keys=True) + "\n"
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0 if result.status in {"ready", "ok"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
