#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from legacy_model_bridge.runtime.wan_animate import WanAnimatePaths, to_json, wan_animate_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Wan Animate latest-env bridge readiness.")
    parser.add_argument("--model-path", type=Path, default=WanAnimatePaths().model_path)
    parser.add_argument("--wan-source", type=Path, default=WanAnimatePaths().wan_source)
    parser.add_argument("--transformer10-root", type=Path, default=WanAnimatePaths().transformer10_root)
    parser.add_argument("--int8-artifact-dir", type=Path, default=WanAnimatePaths().int8_artifact_dir)
    parser.add_argument("--cache-smoke-root", type=Path, default=WanAnimatePaths().cache_smoke_root)
    parser.add_argument("--device", default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = wan_animate_status(
        WanAnimatePaths(
            model_path=args.model_path,
            wan_source=args.wan_source,
            transformer10_root=args.transformer10_root,
            int8_artifact_dir=args.int8_artifact_dir,
            cache_smoke_root=args.cache_smoke_root,
        ),
        device=args.device,
    )
    payload = to_json(status)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
