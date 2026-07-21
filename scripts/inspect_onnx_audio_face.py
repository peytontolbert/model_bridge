#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy_model_bridge.runtime.onnx_runtime import ONNXRequest, audio2face_contract, inspect_onnx_model, to_json

DEFAULT_MODELS = ["Audio2Emotion-v3.0", "Audio2Face-3D-v2.3-Mark", "Audio2Face-3D-v3.0"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect Audio2Emotion/Audio2Face ONNX tensor contracts.")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--model-root", default="/arxiv/models")
    parser.add_argument("--json-out", default="reports/onnx-smokes/audio-face.inspect.json")
    parser.add_argument("--run-synthetic", action="store_true")
    parser.add_argument("--max-dynamic-dim", type=int, default=4)
    parser.add_argument("--model-aware", action="store_true", help="Attach Audio2Face semantic sidecar contract metadata.")
    args = parser.parse_args(argv)

    results = []
    for model_id in args.models or DEFAULT_MODELS:
        result = inspect_onnx_model(
            ONNXRequest(
                model_id=model_id,
                model_root=args.model_root,
                run_synthetic=args.run_synthetic,
                max_dynamic_dim=args.max_dynamic_dim,
            )
        )
        payload = to_json(result)
        if args.model_aware and result.model_path and "Audio2Face" in model_id:
            payload["semantic_contract"] = audio2face_contract(result.model_path, result.sidecars)
        results.append(payload)
    report = {
        "schema_version": 1,
        "artifact_contract": "onnx_audio_face_semantic_contract_matrix" if args.model_aware else "onnx_audio_face_tensor_contract_matrix",
        "model_aware": args.model_aware,
        "run_synthetic": args.run_synthetic,
        "ready_models": [item["model_id"] for item in results if item["status"] in {"ready", "ok"}],
        "blocked_models": [item["model_id"] for item in results if item["status"] not in {"ready", "ok"}],
        "results": results,
    }
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not report["blocked_models"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
