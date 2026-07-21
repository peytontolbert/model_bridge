from __future__ import annotations

import argparse
import json
from pathlib import Path

from legacy_model_bridge.runtime.hunyuan_avatar import DEFAULT_LLAVA_DIR
from legacy_model_bridge.runtime.hunyuan_avatar import inspect_llava_image_token_alignment


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Hunyuan Avatar LLaVA image-token alignment bridge.")
    parser.add_argument("--llava-dir", default=str(DEFAULT_LLAVA_DIR))
    parser.add_argument("--prompt", default="A person talks naturally to the camera.")
    parser.add_argument("--name", default="person")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    result = inspect_llava_image_token_alignment(llava_dir=args.llava_dir, prompt=args.prompt, name=args.name)
    payload = result.to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
    print(text)
    return 0 if result.runnable else 1


if __name__ == "__main__":
    raise SystemExit(main())
