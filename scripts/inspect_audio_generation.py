#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legacy_model_bridge.runtime.audio_generation import AudioGenerationRequest, inspect_audio_generation, to_json

DEFAULT_MODELS = [
    "musicgen-small",
    "musicgen-stereo-small",
    "musicgen-stereo-medium",
    "musicgen-large",
    "musicgen-stereo-large",
    "musicgen-stereo-melody-large",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect or smoke Transformers audio generation models.")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--model-root", default="/arxiv/models")
    parser.add_argument("--prompt", default="lo-fi drums and warm bass")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--run-generate", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)

    models = args.model or DEFAULT_MODELS
    results = [
        inspect_audio_generation(
            AudioGenerationRequest(
                model_id=model,
                model_root=args.model_root,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                dtype=args.dtype,
                run_generate=args.run_generate,
            )
        )
        for model in models
    ]
    payload = {"models": models, "run_generate": args.run_generate, "results": [to_json(result) for result in results]}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0 if all(result.status in {"ready", "ok"} for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
