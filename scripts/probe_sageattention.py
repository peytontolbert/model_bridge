#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legacy_model_bridge.runtime.sageattention import sageattention_status, smoke_sageattention_kernel, to_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe SageAttention compatibility for bridge-managed models.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)

    payload = {"status": to_json(sageattention_status(args.device)), "smokes": []}
    if args.run_smoke:
        payload["smokes"] = [
            to_json(smoke_sageattention_kernel(dtype="float16", layout="NHD")),
            to_json(smoke_sageattention_kernel(dtype="bfloat16", layout="NHD")),
            to_json(smoke_sageattention_kernel(dtype="float16", layout="HND", shape=(1, 16, 512, 128))),
            to_json(smoke_sageattention_kernel(dtype="bfloat16", layout="HND", shape=(1, 16, 512, 128))),
        ]
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0 if payload["status"]["available"] and all(item["status"] == "ok" for item in payload["smokes"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
