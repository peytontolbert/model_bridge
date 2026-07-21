#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legacy_model_bridge.runtime.classic_transformers import (
    ClassicTransformersRequest,
    inspect_classic_transformers,
    to_json,
)

DEFAULT_MODELS = [
    "facebook/bart-large-cnn",
    "gpt2",
    "distilgpt2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/e5-base-v2",
    "hubert-base-ls960",
    "hubert-large-ll60k",
    "hubert-xlarge-ll60k",
    "dinov2-base",
    "dinov2-large",
    "webssl-dino300m-full2b-224",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect classic Transformers artifacts through the bridge contract.")
    parser.add_argument("--model", action="append", default=[], help="Model id to inspect. Can be repeated.")
    parser.add_argument("--model-root", default="/arxiv/models")
    parser.add_argument(
        "--task",
        choices=[
            "seq2seq_generation",
            "causal_lm_generation",
            "text_encoder",
            "sequence_classification",
            "masked_lm",
            "audio_encoder",
            "vision_encoder",
        ],
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--prompt", default="Legacy model bridge smoke input.")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--run-synthetic", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)

    models = args.model or DEFAULT_MODELS
    results = [
        inspect_classic_transformers(
            ClassicTransformersRequest(
                model_id=model,
                model_root=args.model_root,
                task=args.task,
                device=args.device,
                dtype=args.dtype,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                run_synthetic=args.run_synthetic,
                trust_remote_code=args.trust_remote_code,
            )
        )
        for model in models
    ]
    payload = {
        "models": models,
        "run_synthetic": args.run_synthetic,
        "results": [to_json(result) for result in results],
    }
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
