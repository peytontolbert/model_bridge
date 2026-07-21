from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path | None, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(text, end="")
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def _bool_flag(name: str, enabled: bool) -> list[str]:
    return [name] if enabled else []


def _import_status(modules: list[str]) -> dict[str, bool]:
    status: dict[str, bool] = {}
    for module in modules:
        try:
            importlib.import_module(module)
            status[module] = True
        except Exception:
            status[module] = False
    return status


def _official_command(request: dict[str, Any]) -> list[str]:
    family = request["family"]
    repo_root = Path(request["repo_root"])
    entrypoint = repo_root / "examples" / "inference.py"
    bridge_root = Path(__file__).resolve().parents[1]
    student_only = bool(request.get("student_only", False))
    if student_only and family != "transfer":
        raise ValueError("student_only is only supported for Cosmos Transfer 2.5")
    nproc = int(request.get("nproc_per_node") or 1)
    if student_only and nproc > 1:
        raise ValueError("student_only currently supports single-process inference only")
    if nproc > 1:
        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc}",
            f"--master_port={int(request.get('master_port') or 12341)}",
            str(entrypoint),
        ]
    elif student_only:
        cmd = [
            "python",
            str(bridge_root / "scripts" / "cosmos25_transfer_student_only_infer.py"),
            "--runtime-root",
            str(repo_root),
            "--",
        ]
    else:
        cmd = ["python", str(entrypoint)]
    for input_file in request.get("input_files") or []:
        cmd.extend(["-i", str(input_file)])
    cmd.extend(["-o", str(request["output_dir"])])
    model = request.get("model")
    if model:
        cmd.append(f"--model={model}")
    checkpoint_path = request.get("checkpoint_path")
    if checkpoint_path:
        cmd.append(f"--checkpoint-path={checkpoint_path}")
    context_parallel_size = request.get("context_parallel_size")
    if context_parallel_size is not None:
        cmd.append(f"--context-parallel-size={int(context_parallel_size)}")
    if family == "predict":
        cmd.append(f"--inference-type={request.get('inference_type') or 'text2world'}")
    cmd.extend(_bool_flag("--disable-guardrails", bool(request.get("disable_guardrails", True))))
    # Cosmos Transfer 2.5 does not expose these as official CLI flags.
    # Student-only offload is handled by legacy_model_bridge.runtime.cosmos25_student_only.
    if family != "transfer":
        cmd.extend(_bool_flag("--offload-diffusion-model", bool(request.get("offload_diffusion_model", False))))
        cmd.extend(_bool_flag("--offload-text-encoder", bool(request.get("offload_text_encoder", False))))
        cmd.extend(_bool_flag("--offload-tokenizer", bool(request.get("offload_tokenizer", False))))
    for key, value in sorted((request.get("extra_args") or {}).items()):
        flag = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            cmd.extend(_bool_flag(flag, value))
        elif value is not None:
            cmd.append(f"{flag}={value}")
    return cmd


def _validate_request(request: dict[str, Any]) -> dict[str, Any]:
    family = request["family"]
    repo_root = Path(request["repo_root"])
    checkpoint_path = Path(request["checkpoint_path"])
    entrypoint = repo_root / "examples" / "inference.py"
    source_paths = [
        str(repo_root),
        str(repo_root / "packages" / "cosmos-oss"),
        str(repo_root / "packages" / "cosmos-cuda"),
        str(Path(__file__).resolve().parents[1]),
    ]
    for source_path in reversed(source_paths):
        if source_path not in sys.path:
            sys.path.insert(0, source_path)
    required_imports = [
        "torch",
        "transformers",
        "diffusers",
        "flash_attn",
        "natten",
        "transformer_engine",
        "cosmos_cuda",
        "cosmos_oss",
        "cosmos_predict2" if family == "predict" else "cosmos_transfer2",
        "cosmos_predict2.config" if family == "predict" else "cosmos_transfer2.config",
        "cosmos_predict2.inference" if family == "predict" else "cosmos_transfer2.inference",
    ]
    imports = _import_status(required_imports)
    paths = {
        "repo_root": repo_root.exists(),
        "cosmos_oss_package": (repo_root / "packages" / "cosmos-oss").exists(),
        "cosmos_cuda_package": (repo_root / "packages" / "cosmos-cuda").exists(),
        "entrypoint": entrypoint.is_file(),
        "checkpoint_path": checkpoint_path.is_file(),
    }
    input_paths = {}
    for item in request.get("input_files") or []:
        input_path = Path(item)
        resolved = input_path if input_path.is_absolute() else repo_root / input_path
        input_paths[str(item)] = resolved.is_file()
    paths.update({f"input:{path}": exists for path, exists in input_paths.items()})
    missing_imports = sorted(name for name, ok in imports.items() if not ok)
    missing_paths = sorted(name for name, ok in paths.items() if not ok)
    return {
        "family": family,
        "repo_root": str(repo_root),
        "source_paths": source_paths,
        "entrypoint": str(entrypoint),
        "checkpoint_path": str(checkpoint_path),
        "imports": imports,
        "paths": paths,
        "missing_imports": missing_imports,
        "missing_paths": missing_paths,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--result-json")
    args = parser.parse_args(argv)

    request = _load_json(args.request_json)
    validation = _validate_request(request)
    launch_command = _official_command(request)
    missing_imports = validation["missing_imports"]
    missing_paths = validation["missing_paths"]
    status = "ready" if not missing_imports and not missing_paths else "blocked"
    payload = {
        "status": status,
        "model_id": request.get("model_id"),
        "family": validation["family"],
        "inspect_only": bool(request.get("inspect_only", True)),
        "launch_plan": {
            "cwd": validation["repo_root"],
            "env": {
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": os.pathsep.join(validation["source_paths"]),
                **({"LEGACY_MODEL_BRIDGE_COSMOS25_STUDENT_ONLY": "1"} if request.get("student_only") else {}),
                **({"LEGACY_MODEL_BRIDGE_COSMOS25_CPU_OFFLOAD": "1"} if request.get("student_only") else {}),
                "HF_HOME": "/data/huggingface",
                "HUGGINGFACE_HUB_CACHE": "/data/huggingface/hub",
                "TRANSFORMERS_CACHE": "/data/huggingface/hub",
                **({"CUDA_VISIBLE_DEVICES": str(request["cuda_visible_devices"])} if request.get("cuda_visible_devices") else {}),
                **({"HF_HUB_OFFLINE": "1"} if request.get("offline_only", True) else {}),
                **({"COSMOS_EXPERIMENTAL_CHECKPOINTS": "1"} if validation["family"] == "transfer" else {}),
            },
            "command": launch_command,
            "checkpoint_path": validation["checkpoint_path"],
            "output_dir": request["output_dir"],
            "artifact_contract": "video_generation_bundle",
            "patches": ["cosmos25_transfer_student_only_dmd2"] if request.get("student_only") else [],
        },
        "validation": validation,
        "error": None,
    }
    if status != "ready":
        payload["error"] = "missing imports or paths"
    _write_json(args.result_json, payload)
    return 0 if status == "ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
