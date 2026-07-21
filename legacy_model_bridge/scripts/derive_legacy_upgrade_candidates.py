#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_INDEX = Path("/data/staticpeytonsite/src/research_library/data/model_index.json")


PATTERNS = {
    "transformers": re.compile(r"transformers|AutoModel|AutoProcessor|from_pretrained", re.I),
    "torch": re.compile(r"torch|pytorch|cuda|flash_attn|sageattention|triton", re.I),
    "diffusers": re.compile(r"diffusers|pipeline|safetensors", re.I),
    "onnx": re.compile(r"onnx|tensorrt", re.I),
    "nemo": re.compile(r"nemo|\\.nemo", re.I),
}


def score_model(model: dict[str, Any], catalog_ids: set[str]) -> dict[str, Any] | None:
    model_id = str(model["id"])
    if model_id in catalog_ids:
        return None
    text = json.dumps(model, ensure_ascii=False)
    hits = [name for name, pattern in PATTERNS.items() if pattern.search(text)]
    if not hits:
        return None
    score = len(hits) * 10
    if model.get("model_stack"):
        score += 25
    if model.get("library_name") in {"transformers", "diffusers", "PyTorch"}:
        score += 10
    if "safetensors" in text:
        score += 5
    if any(skip in model_id for skip in {"Cosmos-Transfer2.5", "Cosmos-Predict2.5"}):
        score -= 100
    lane = infer_lane(model, hits)
    return {
        "model_id": model_id,
        "score": score,
        "lane": lane,
        "local_path": f"/arxiv/models/{model.get('path') or model_id}",
        "library_name": model.get("library_name"),
        "model_type": model.get("model_type"),
        "pipeline_tag": model.get("pipeline_tag"),
        "class_name": model.get("class_name"),
        "dependency_hits": hits,
        "example_snippet": (model.get("example") or {}).get("snippet", "")[:600],
        "model_stack": model.get("model_stack"),
        "mismatch_classes": infer_mismatches(model, hits),
    }


def infer_lane(model: dict[str, Any], hits: list[str]) -> str:
    model_id = str(model["id"]).lower()
    model_type = str(model.get("model_type") or "")
    if "musicgen" in model_type or "bigvgan" in model_id or "audiogen" in model_id:
        return "audio_generation_bridge"
    if "onnx" in hits:
        return "onnx_runtime_bridge"
    if model.get("library_name") == "diffusers" or "diffusers" in hits:
        return "diffusers_cuda_bridge"
    if "wav2vec2" in model_type or "hubert" in model_type:
        return "classic_transformers_bridge"
    if "transformers" in hits:
        return "transformers_auto_bridge"
    return "custom_runtime_bridge"


def infer_mismatches(model: dict[str, Any], hits: list[str]) -> list[str]:
    text = json.dumps(model, ensure_ascii=False).lower()
    mismatches = []
    if "git+https://github.com/huggingface/transformers" in text or "transformers_version" in text:
        mismatches.append("transformers_export_drift")
    if "flash_attn" in text or "sageattention" in text:
        mismatches.append("attention_backend_selection")
    if "trust_remote_code" in text:
        mismatches.append("remote_code_loader")
    if ".pt" in text or ".pth" in text:
        mismatches.append("pytorch_checkpoint_contract")
    if "onnx" in hits:
        mismatches.append("onnx_tensor_contract")
    if "processor" in text or "preprocessor_config" in text:
        mismatches.append("processor_contract")
    return mismatches or ["latest_env_validation_needed"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rank legacy model upgrade candidates from model_index.json.")
    parser.add_argument("--model-index", default=str(MODEL_INDEX))
    parser.add_argument("--catalog", default="data/bridge_catalog.json")
    parser.add_argument("--json-out", default="reports/legacy-upgrade-candidates.json")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args(argv)

    index = json.loads(Path(args.model_index).read_text(encoding="utf-8"))
    catalog = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
    catalog_ids = {item["model_id"] for item in catalog.get("models", [])}
    candidates = [score_model(model, catalog_ids) for model in index.get("models", [])]
    selected = sorted((item for item in candidates if item is not None), key=lambda item: (-item["score"], item["model_id"]))
    payload = {
        "source": str(args.model_index),
        "excluded_catalog_entries": len(catalog_ids),
        "candidates": selected[: args.limit],
    }
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
