#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy_model_bridge.runtime.nemo_asr import NemoASRRequest, transcribe_nemo_asr, to_json

DEFAULT_TARGETS_PATH = Path("data/older_compatibility_targets.json")
DEFAULT_OUTPUT = Path("reports/nemo-asr-smokes/archive-expansion.inspect.json")


def _load_default_models(path: Path = DEFAULT_TARGETS_PATH) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for target in payload["targets"]:
        if target["group_id"] == "nemo_asr_archive_expansion":
            return list(target["models"])
    raise ValueError("nemo_asr_archive_expansion target group not found")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def inspect_models(
    models: list[str],
    *,
    output_dir: Path,
    timeout_sec: int,
    dry_run: bool,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for model_id in models:
        model_output = output_dir / model_id.replace("/", "--")
        request = NemoASRRequest(model_id=model_id, output_dir=str(model_output), restore=False, load_only=False)
        result = transcribe_nemo_asr(request, dry_run=dry_run, timeout_sec=timeout_sec)
        payload = to_json(result)
        worker_payload = result.payload or {}
        results.append(
            {
                "model_id": model_id,
                "status": result.status,
                "env": result.env,
                "archive_path": worker_payload.get("nemo_archive"),
                "archive_exists": worker_payload.get("nemo_archive_exists"),
                "nemo_model_target": worker_payload.get("nemo_model_target"),
                "nemo_model_target_importable": worker_payload.get("nemo_model_target_importable"),
                "nemo_model_target_import_detail": worker_payload.get("nemo_model_target_import_detail"),
                "env_ok": worker_payload.get("env_ok"),
                "env_problems": worker_payload.get("env_problems"),
                "request_path": result.request_path,
                "result_path": result.result_path,
                "returncode": result.returncode,
                "error": result.error,
                "dry_run": result.dry_run,
                "command": list(result.command),
                "raw_result": payload,
            }
        )
    ready = [item["model_id"] for item in results if item["status"] == "ready"]
    blocked = [item["model_id"] for item in results if item["status"] != "ready"]
    return {
        "schema_version": 1,
        "artifact_contract": "nemo_archive_inspection_matrix",
        "target_group": "nemo_asr_archive_expansion",
        "dry_run": dry_run,
        "ready_models": ready,
        "blocked_models": blocked,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect ranked NeMo ASR .nemo archives through the bridge worker.")
    parser.add_argument("--model", action="append", dest="models", help="Model id to inspect. Can be repeated.")
    parser.add_argument("--output-dir", default="/data/tmp/legacy_model_bridge_nemo_asr_archive_inspect")
    parser.add_argument("--json-out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    models = args.models or _load_default_models()
    report = inspect_models(models, output_dir=Path(args.output_dir), timeout_sec=args.timeout_sec, dry_run=args.dry_run)
    _write_json(Path(args.json_out), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not report["blocked_models"] or args.dry_run else 2


if __name__ == "__main__":
    raise SystemExit(main())
