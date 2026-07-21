#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy_model_bridge.runtime.nemo_asr import NemoASRRequest
from legacy_model_bridge.runtime.nemo_asr import NemoASRWarmClient


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke the NeMo ASR warm JSONL service contract.")
    parser.add_argument("--model-id", default="parakeet-tdt_ctc-110m")
    parser.add_argument("--archive-path")
    parser.add_argument("--audio", action="append", dest="audio_paths", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="/data/tmp/lmb_nemo_warm_service")
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--timeout-sec", type=float, default=300.0)
    args = parser.parse_args()

    started = time.perf_counter()
    request = NemoASRRequest(
        model_id=args.model_id,
        archive_path=args.archive_path,
        audio_paths=tuple(args.audio_paths),
        output_dir=args.output_dir,
        device=args.device,
    )
    client = NemoASRWarmClient(request)
    shutdown = None
    payload = {
        "schema_version": 1,
        "artifact_contract": "warm_transcription_service",
        "model_id": args.model_id,
        "audio_paths": args.audio_paths,
        "status": "failed",
        "command": list(client.command),
        "request_path": str(client.request_path),
    }
    try:
        startup = client.start(timeout_sec=args.timeout_sec)
        status = client.status(timeout_sec=args.timeout_sec)
        transcribe = client.transcribe(args.audio_paths, timeout_sec=args.timeout_sec)
        shutdown = client.close(timeout_sec=30.0)
        payload.update({
            "startup": startup,
            "status_payload": status,
            "transcribe": transcribe,
            "shutdown": shutdown,
            "elapsed_sec": time.perf_counter() - started,
            "status": "ok" if startup.get("status") == "loaded" and transcribe.get("status") == "ok" else "failed",
        })
    except Exception as exc:
        payload.update({
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_sec": time.perf_counter() - started,
        })
        if shutdown is None:
            try:
                shutdown = client.close(timeout_sec=10.0)
                payload["shutdown"] = shutdown
            except Exception:
                pass
    _write_json(Path(args.json_out), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
