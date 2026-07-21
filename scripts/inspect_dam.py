#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legacy_model_bridge.runtime.dam import probe_dam_components, probe_to_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect DAM component bridge status.")
    parser.add_argument("--model-id", default="DAM-3B")
    parser.add_argument("--model-root", default="/arxiv/models")
    parser.add_argument("--runtime-source", default="/arxiv/models/DAM-3B-Self-Contained")
    parser.add_argument("--load-components", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)

    model_path = Path(args.model_id)
    if not model_path.exists():
        model_path = Path(args.model_root) / args.model_id
        if not model_path.exists() and "/" in args.model_id:
            model_path = Path(args.model_root) / args.model_id.split("/", 1)[1]

    result = probe_dam_components(
        model_path,
        model_id=args.model_id,
        runtime_source=args.runtime_source,
        load_components=args.load_components,
    )
    text = json.dumps(probe_to_json(result), indent=2, sort_keys=True) + "\n"
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    failed = any(component.status == "failed" for component in result.components)
    return 2 if failed or not result.imports.get("llava_llama") else 0


if __name__ == "__main__":
    raise SystemExit(main())
