#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legacy_model_bridge.runtime.dam import DEFAULT_RUNTIME_SOURCE


def _synthetic_inputs(workdir: Path) -> tuple[Path, Path]:
    from PIL import Image, ImageDraw

    workdir.mkdir(parents=True, exist_ok=True)
    image_path = workdir / "dam_synthetic_image.png"
    mask_path = workdir / "dam_synthetic_mask.png"

    image = Image.new("RGB", (384, 384), (236, 239, 235))
    draw = ImageDraw.Draw(image)
    draw.rectangle((112, 96, 272, 288), fill=(35, 105, 190))
    draw.ellipse((154, 140, 230, 216), fill=(242, 198, 72))
    draw.line((96, 320, 288, 320), fill=(32, 32, 32), width=8)
    image.save(image_path)

    mask = Image.new("L", (384, 384), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle((112, 96, 272, 288), fill=255)
    mask.save(mask_path)
    return image_path, mask_path


def _load_images(image_path: Path, mask_path: Path):
    from PIL import Image

    return Image.open(image_path).convert("RGB"), Image.open(mask_path).convert("L")


def run_smoke(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    model_path = Path(args.model_path)
    image_path, mask_path = _synthetic_inputs(Path(args.workdir))

    import torch

    runtime_source = Path(args.runtime_source).resolve()
    if str(runtime_source) not in sys.path:
        sys.path.insert(0, str(runtime_source))
    import llava_llama

    load_started = time.monotonic()
    model = llava_llama.DescribeAnythingModel(
        str(model_path),
        conv_mode=args.conv_mode,
        prompt_mode=args.prompt_mode,
        torch_dtype=getattr(torch, args.dtype),
        device=args.device,
        local_files_only=True,
    )
    load_sec = round(time.monotonic() - load_started, 3)
    model.eval()

    image_pil, mask_pil = _load_images(image_path, mask_path)
    describe_started = time.monotonic()
    text = model.get_description(
        image_pil,
        mask_pil,
        args.query,
        streaming=False,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
    )
    describe_sec = round(time.monotonic() - describe_started, 3)

    return {
        "model_id": args.model_id,
        "model_path": str(model_path),
        "status": "ok",
        "env": args.env,
        "device": args.device,
        "dtype": args.dtype,
        "model_class": type(model).__name__,
        "inner_model_class": type(model.model).__name__,
        "runtime_source": str(runtime_source),
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "query": args.query,
        "output_text": text,
        "max_new_tokens": args.max_new_tokens,
        "load_sec": load_sec,
        "describe_sec": describe_sec,
        "elapsed_sec": round(time.monotonic() - started, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a bounded DAM-3B description smoke.")
    parser.add_argument("--model-id", default="DAM-3B-Self-Contained")
    parser.add_argument("--model-path", default="/arxiv/models/DAM-3B-Self-Contained")
    parser.add_argument("--runtime-source", default=str(DEFAULT_RUNTIME_SOURCE))
    parser.add_argument("--workdir", default="/data/tmp/lmb_dam_smoke")
    parser.add_argument("--json-out", default="reports/world-model-smokes/DAM-3B-Self-Contained.description8.cuda1.ai.json")
    parser.add_argument("--env", default="ai")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--conv-mode", default="v1")
    parser.add_argument("--prompt-mode", default="full+focal_crop")
    parser.add_argument("--query", default="<image>\nDescribe the masked region briefly.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.5)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args(argv)

    try:
        result = run_smoke(args)
    except Exception as exc:
        result = {
            "model_id": args.model_id,
            "model_path": args.model_path,
            "status": "failed",
            "error": f"{type(exc).__name__}:{exc}",
            "traceback": traceback.format_exc(),
        }

    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
