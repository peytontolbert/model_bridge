from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

CLASSIC_MODELS = (
    "MobileLLM-125M",
    "MobileLLM-125M-layer-share",
    "MobileLLM-350M",
    "MobileLLM-350M-layer-share",
    "MobileLLM-600M",
    "MobileLLM-1B",
    "MobileLLM-1.5B",
    "MobileLLM-R1.5-140M",
    "MobileLLM-R1.5-360M",
    "MobileLLM-R1.5-950M",
    "MobileLLM-Pro-base",
)

PARETOQ_MODELS = (
    "MobileLLM-ParetoQ-125M-1-bit",
    "MobileLLM-ParetoQ-125M-1.58-bit",
    "MobileLLM-ParetoQ-125M-BF16",
    "MobileLLM-ParetoQ-350M-1-bit",
    "MobileLLM-ParetoQ-350M-1.58-bit",
    "MobileLLM-ParetoQ-350M-4-bit",
    "MobileLLM-ParetoQ-350M-BF16",
    "MobileLLM-ParetoQ-600M-1-bit",
    "MobileLLM-ParetoQ-600M-1.58-bit",
    "MobileLLM-ParetoQ-600M-2-bit",
    "MobileLLM-ParetoQ-600M-3-bit",
    "MobileLLM-ParetoQ-600M-4-bit",
    "MobileLLM-ParetoQ-600M-BF16",
    "MobileLLM-ParetoQ-1B-1-bit",
    "MobileLLM-ParetoQ-1B-1.58-bit",
    "MobileLLM-ParetoQ-1B-4-bit",
    "MobileLLM-ParetoQ-1B-BF16",
    "MobileLLM-ParetoQ-1.5B-BF16",
)

DEFAULT_MODELS = CLASSIC_MODELS
FAMILY_MODELS = {
    "classic": CLASSIC_MODELS,
    "paretoq": PARETOQ_MODELS,
}


def _safe_name(model_id: str) -> str:
    return model_id.replace("/", "--")


def _model_path(model_root: Path, model_id: str) -> Path:
    direct = model_root / model_id
    if direct.exists():
        return direct
    if "/" in model_id:
        for candidate in (model_root / model_id.split("/", 1)[1], model_root / model_id.replace("/", "--")):
            if candidate.exists():
                return candidate
    return direct


def _run_one(args: argparse.Namespace, model_id: str) -> dict[str, Any]:
    model_path = _model_path(Path(args.model_root), model_id)
    report_path = Path(args.report_dir) / f"{_safe_name(model_id)}.generate{args.max_new_tokens}.{args.env_name}.{args.cuda_label}.json"
    if not model_path.exists():
        payload = {
            "status": "skipped_missing_model_path",
            "model_id": model_id,
            "model_path": str(model_path),
            "error": f"model path not found: {model_path}",
            "dry_run": bool(args.dry_run),
        }
        report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"model_id": model_id, "status": payload["status"], "report": str(report_path)}

    cmd = [
        sys.executable,
        "-m",
        "legacy_model_bridge.cli",
        "causal-lm",
        "generate",
        "--model-id",
        model_id,
        "--model-root",
        str(args.model_root),
        "--prompt",
        args.prompt,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
    ]
    if args.dry_run:
        cmd.append("--dry-run")

    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    repo_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    start = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=args.timeout_sec,
        check=False,
    )
    elapsed = time.monotonic() - start
    try:
        payload = json.loads(proc.stdout)
        if not isinstance(payload, dict):
            raise ValueError("CLI output was not a JSON object")
    except Exception as exc:
        payload = {
            "status": "failed_invalid_cli_json",
            "model_id": model_id,
            "model_path": str(model_path),
            "error": f"{type(exc).__name__}:{exc}",
        }
    payload["matrix_returncode"] = proc.returncode
    payload["matrix_elapsed_sec"] = elapsed
    payload["matrix_command"] = cmd
    payload["stderr_tail"] = proc.stderr[-args.output_capture_chars:]
    if payload.get("status") == "failed" and proc.returncode == 0:
        proc_status = 9
    else:
        proc_status = proc.returncode
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "model_id": model_id,
        "status": str(payload.get("status")),
        "returncode": proc_status,
        "report": str(report_path),
        "generated_token_count": payload.get("generated_token_count"),
        "applied_patches": payload.get("applied_patches", []),
        "error": payload.get("error"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run bounded MobileLLM family causal-LM bridge smokes.")
    parser.add_argument("--model", action="append", dest="models", help="Model id to smoke. Repeatable. Overrides --family.")
    parser.add_argument("--family", choices=sorted(FAMILY_MODELS), default="classic", help="Built-in MobileLLM family matrix to smoke when --model is not provided.")
    parser.add_argument("--model-root", default="/arxiv/models")
    parser.add_argument("--report-dir", default="reports/causal-lm-smokes")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--cuda-label", default="cuda")
    parser.add_argument("--env-name", default="ai")
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--output-capture-chars", type=int, default=12000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)

    Path(args.report_dir).mkdir(parents=True, exist_ok=True)
    models = tuple(args.models) if args.models else FAMILY_MODELS[args.family]
    results = []
    for model_id in models:
        result = _run_one(args, model_id)
        results.append(result)
        print(f"{model_id}\t{result['status']}\t{result['report']}", flush=True)
        if args.fail_fast and result.get("status") not in {"ok", "dry_run"}:
            break

    summary = {
        "status": "ok" if all(item.get("status") in {"ok", "dry_run"} for item in results) else "partial_or_failed",
        "models": list(models),
        "results": results,
        "dry_run": bool(args.dry_run),
        "device": args.device,
        "dtype": args.dtype,
        "cuda_visible_devices": args.cuda_visible_devices,
        "patch_contract": [
            "transformers_mobilellm_legacy_cache",
            "transformers_mobilellm_slow_tokenizer",
        ],
    }
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
