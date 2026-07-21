#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legacy_model_bridge.runtime.abot_world import ABotWorldRequest, to_json, verify_abot_world, write_probe_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke ABot-World through the bridge verification contract.")
    parser.add_argument("--repo-path", default="/data/repositories/ABot-World")
    parser.add_argument("--model-path", default="/arxiv/models/acvlab--ABot-World-0-5B-LF")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--json-out", default="reports/world-model-smokes/acvlab--ABot-World-0-5B-LF.generator_cuda_bf16.json")
    args = parser.parse_args(argv)

    try:
        probe = verify_abot_world(
            ABotWorldRequest(
                repo_path=args.repo_path,
                model_path=args.model_path,
                device=args.device,
                dtype=args.dtype,
                load_generator=not args.metadata_only,
            )
        )
        write_probe_json(probe, args.json_out)
        print(json.dumps(to_json(probe), indent=2, sort_keys=True))
        return 0 if probe.status in {"ready", "generator_cuda_bf16_loaded"} else 1
    except Exception as exc:
        payload = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
