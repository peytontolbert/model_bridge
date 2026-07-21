#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_MODEL_INDEX = Path("/data/staticpeytonsite/src/research_library/data/model_index.json")

LANE_BY_LIBRARY = {
    "diffusers": "diffusers_cuda_bridge",
    "transformers": "transformers_causal_lm_bridge",
    "peft": "peft_adapter_bridge",
    "nemo": "nemo_asr_bridge",
    "hunyuan3d-2": "three_d_gen_bridge",
    "cosmos": "world_model_bridge",
}

LANE_BY_PIPELINE = {
    "automatic-speech-recognition": "nemo_asr_bridge",
    "image-to-3d": "three_d_gen_bridge",
    "image-to-video": "video_diffusion_bridge",
    "text-to-video": "video_diffusion_bridge",
    "video-to-video": "video_diffusion_bridge",
    "text-classification": "encoder_classifier_bridge",
    "feature-extraction": "encoder_classifier_bridge",
    "text-generation": "transformers_causal_lm_bridge",
    "image-text-to-text": "world_model_bridge",
}


def bridge_lane(model: dict[str, Any]) -> str:
    if model.get("model_stack", {}).get("lane"):
        return model["model_stack"]["lane"]
    library = model.get("library_name")
    if library in LANE_BY_LIBRARY:
        return LANE_BY_LIBRARY[library]
    pipeline = model.get("pipeline_tag")
    if pipeline in LANE_BY_PIPELINE:
        return LANE_BY_PIPELINE[pipeline]
    model_type = model.get("model_type")
    if model_type in {"bert", "dinov2", "clip"}:
        return "encoder_classifier_bridge"
    if model_type in {"llama", "llama4_text", "mobilellm", "qwen3"}:
        return "transformers_causal_lm_bridge"
    return "custom_bridge_review"


def candidate_status(model: dict[str, Any]) -> str:
    if model.get("model_stack_status"):
        return model["model_stack_status"]
    if model.get("model_stack_runnable") is True:
        return "runnable_from_model_index"
    exts = model.get("tree_file_extensions") or model.get("file_extensions") or {}
    if ".nemo" in exts:
        return "needs_nemo_archive_review"
    if ".onnx" in exts:
        return "needs_onnx_prepost_bridge"
    if ".safetensors" in exts or ".bin" in exts or ".pt" in exts or ".pth" in exts:
        return "needs_bridge_profile"
    return "metadata_only_or_incomplete"


def compact_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": model["id"],
        "lane": bridge_lane(model),
        "status": candidate_status(model),
        "preferred_env": model.get("model_stack_preferred_env") or "lmb-default",
        "runnable": bool(model.get("model_stack_runnable", False)),
        "library_name": model.get("library_name"),
        "pipeline_tag": model.get("pipeline_tag"),
        "model_type": model.get("model_type"),
        "weight_bytes": model.get("tree_weight_size_bytes") or model.get("direct_weight_size_bytes") or 0,
        "file_extensions": sorted((model.get("tree_file_extensions") or model.get("file_extensions") or {}).keys()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Derive first-pass bridge candidates from the research library model index.")
    parser.add_argument("--model-index", type=Path, default=DEFAULT_MODEL_INDEX)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    raw = json.loads(args.model_index.read_text())
    candidates = [compact_model(model) for model in raw["models"]]
    candidates.sort(key=lambda item: (not item["runnable"], item["lane"], -int(item["weight_bytes"]), item["model_id"]))

    payload = {
        "metadata": {
            "source": str(args.model_index),
            "source_generated_utc": raw.get("generated_utc"),
            "source_model_count": raw.get("model_count"),
            "candidate_count": len(candidates),
            "lane_counts": dict(Counter(item["lane"] for item in candidates)),
            "status_counts": dict(Counter(item["status"] for item in candidates)),
        },
        "models": candidates[: args.limit],
    }

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
