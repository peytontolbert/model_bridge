#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legacy_model_bridge.runtime.audiogen import AudioGenRequest, inspect_audiogen, to_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect AudioCraft AudioGen checkpoint-pair artifacts.")
    parser.add_argument("--model-id", default="audiogen-medium")
    parser.add_argument("--model-root", default="/arxiv/models")
    parser.add_argument("--inspect-state", action="store_true")
    parser.add_argument("--audiocraft-source-path")
    parser.add_argument("--probe-source-import", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)

    result = inspect_audiogen(AudioGenRequest(
        model_id=args.model_id,
        model_root=args.model_root,
        inspect_state=args.inspect_state,
        audiocraft_source_path=args.audiocraft_source_path,
        probe_source_import=args.probe_source_import,
    ))
    text = json.dumps(to_json(result), indent=2, sort_keys=True) + "\n"
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0 if result.status in {"blocked_missing_audiocraft_adapter", "ready_for_audiocraft_load_smoke"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
