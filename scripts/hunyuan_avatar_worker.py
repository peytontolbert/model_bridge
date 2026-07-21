from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

BRIDGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AVATAR_ROOT = Path("/data/clone/hunyuanvideo-avatar")
DEFAULT_TRANSFORMER10_ROOT = Path("/data/transformer_10")
DEFAULT_MODEL_BASE = Path("/arxiv/models/HunyuanVideo-Avatar")
DEFAULT_FP8_SHARD_DIR = DEFAULT_TRANSFORMER10_ROOT / "checkpoints" / "hunyuan_avatar_fp8_fsdp2"
DEFAULT_BF16_SHARD_DIR = DEFAULT_TRANSFORMER10_ROOT / "checkpoints" / "hunyuan_avatar_bf16_fsdp2"
DEFAULT_SHARD_DIR = DEFAULT_FP8_SHARD_DIR
DEFAULT_FP8_CKPT = DEFAULT_MODEL_BASE / "ckpts" / "hunyuan-video-t2v-720p" / "transformers" / "mp_rank_00_model_states_fp8.pt"
DEFAULT_BF16_CKPT = DEFAULT_MODEL_BASE / "ckpts" / "hunyuan-video-t2v-720p" / "transformers" / "mp_rank_00_model_states.pt"
DEFAULT_CKPT = DEFAULT_FP8_CKPT
DEFAULT_INPUT = DEFAULT_AVATAR_ROOT / "input" / "peyton_avatar_test.csv"
DEFAULT_OUTPUT_DIR = Path("/data/tmp/lmb_hunyuan_avatar_ai_smoke")
DEFAULT_OUTPUT_FILE = "peyton_seated_v1_audio.mp4"


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path | None, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(text, end="")
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def _request_value(request: dict[str, Any], key: str, default: Any) -> Any:
    value = request.get(key)
    return default if value is None else value


def _source_paths(request: dict[str, Any]) -> list[str]:
    return [
        str(Path(_request_value(request, "avatar_root", DEFAULT_AVATAR_ROOT))),
        str(Path(_request_value(request, "transformer10_root", DEFAULT_TRANSFORMER10_ROOT))),
        str(BRIDGE_ROOT),
    ]


def _import_status(modules: list[str], source_paths: list[str]) -> dict[str, bool]:
    for source_path in reversed(source_paths):
        if source_path not in sys.path:
            sys.path.insert(0, source_path)
    status: dict[str, bool] = {}
    for module in modules:
        try:
            importlib.import_module(module)
            status[module] = True
        except Exception:
            status[module] = False
    return status


def _expected_output_path(request: dict[str, Any]) -> Path:
    output_dir = Path(_request_value(request, "output_dir", DEFAULT_OUTPUT_DIR))
    return output_dir / str(_request_value(request, "output_file", DEFAULT_OUTPUT_FILE))


def _gpu_memory_status(request: dict[str, Any]) -> dict[str, Any]:
    visible = str(_request_value(request, "cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES", ""))).strip()
    selected = [item.strip() for item in visible.split(",") if item.strip()]
    if not selected:
        selected = [str(index) for index in range(int(_request_value(request, "nproc_per_node", 2)))]
    min_free_default = 22000 if str(_request_value(request, "precision_mode", "auto")) != "fp8" else 18000
    min_free_mb = int(_request_value(request, "min_free_vram_mb", min_free_default))
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        return {
            "available": False,
            "selected": selected,
            "min_free_vram_mb": min_free_mb,
            "gpus": {},
            "blocked": selected,
            "error": str(exc),
        }
    gpus: dict[str, dict[str, int]] = {}
    if proc.returncode == 0:
        for line in proc.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                continue
            index, used, total = parts
            try:
                used_mb = int(used)
                total_mb = int(total)
            except ValueError:
                continue
            gpus[index] = {"used_mb": used_mb, "total_mb": total_mb, "free_mb": total_mb - used_mb}
    blocked = [index for index in selected if gpus.get(index, {}).get("free_mb", 0) < min_free_mb]
    return {
        "available": proc.returncode == 0,
        "selected": selected,
        "min_free_vram_mb": min_free_mb,
        "gpus": gpus,
        "blocked": blocked,
        "error": None if proc.returncode == 0 else proc.stderr[-2000:],
    }


def _vae_cache_command(request: dict[str, Any]) -> list[str]:
    transformer10_root = Path(_request_value(request, "transformer10_root", DEFAULT_TRANSFORMER10_ROOT))
    return [
        sys.executable,
        str(transformer10_root / "scripts" / "cache_hunyuan_avatar_vae_latents.py"),
        "--input", str(Path(_request_value(request, "input", DEFAULT_INPUT))),
        "--cache-dir", str(Path(_request_value(request, "vae_latent_cache", DEFAULT_AVATAR_ROOT / "cache" / "vae_latents_512_w17"))),
        "--image-size", str(int(_request_value(request, "image_size", 512))),
        "--frames", str(int(_request_value(request, "reference_frames", max(int(_request_value(request, "sample_n_frames", 65)), 65)))),
    ]


def _optimized_command(request: dict[str, Any]) -> list[str]:
    precision_mode = str(_request_value(request, "precision_mode", "auto"))
    default_shard_dir = DEFAULT_FP8_SHARD_DIR if precision_mode == "fp8" else DEFAULT_BF16_SHARD_DIR
    default_ckpt = DEFAULT_FP8_CKPT if precision_mode == "fp8" else DEFAULT_BF16_CKPT
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={int(_request_value(request, 'nproc_per_node', 2))}",
        f"--master_port={int(_request_value(request, 'master_port', 29615))}",
        str(BRIDGE_ROOT / "scripts" / "hunyuan_avatar_optimized_fp8_fsdp2_worker.py"),
        "--avatar-root", str(Path(_request_value(request, "avatar_root", DEFAULT_AVATAR_ROOT))),
        "--transformer10-root", str(Path(_request_value(request, "transformer10_root", DEFAULT_TRANSFORMER10_ROOT))),
        "--model-base", str(Path(_request_value(request, "model_base", DEFAULT_MODEL_BASE))),
        "--precision-mode", precision_mode,
        "--shard-dir", str(Path(_request_value(request, "shard_dir", default_shard_dir))),
        "--ckpt", str(Path(_request_value(request, "ckpt", default_ckpt))),
        "--input", str(Path(_request_value(request, "input", DEFAULT_INPUT))),
        "--save-path", str(Path(_request_value(request, "output_dir", DEFAULT_OUTPUT_DIR))),
        "--infer-steps", str(int(_request_value(request, "infer_steps", 4))),
        "--sample-n-frames", str(int(_request_value(request, "sample_n_frames", 65))),
        "--reference-frames", str(int(_request_value(request, "reference_frames", max(int(_request_value(request, "sample_n_frames", 65)), 65)))),
        "--image-size", str(int(_request_value(request, "image_size", 512))),
        "--seed", str(int(_request_value(request, "seed", 1025))),
        "--cfg-scale", str(float(_request_value(request, "cfg_scale", 7.5))),
        "--flow-shift-eval-video", str(float(_request_value(request, "flow_shift_eval_video", 5.0))),
        "--cpu-offload", "1" if bool(_request_value(request, "cpu_offload", True)) else "0",
    ]
    return cmd


def _validate_request(request: dict[str, Any]) -> dict[str, Any]:
    source_paths = _source_paths(request)
    required_imports = [
        "torch", "torch.distributed", "transformers", "diffusers", "flash_attn", "imageio", "loguru",
        "hymm_sp.config", "hymm_sp.sample_inference_audio", "hymm_sp.data_kits.audio_dataset",
        "hymm_sp.data_kits.face_align", "runtime.hunyuan_avatar_fsdp",
    ]
    imports = _import_status(required_imports, source_paths)
    paths = {
        "avatar_root": Path(_request_value(request, "avatar_root", DEFAULT_AVATAR_ROOT)).exists(),
        "model_base": Path(_request_value(request, "model_base", DEFAULT_MODEL_BASE)).exists(),
        "shard_manifest": (Path(_request_value(request, "shard_dir", DEFAULT_BF16_SHARD_DIR if str(_request_value(request, "precision_mode", "auto")) != "fp8" else DEFAULT_FP8_SHARD_DIR)) / "avatar_transformer.manifest.json").is_file(),
        "rank00_shard": (Path(_request_value(request, "shard_dir", DEFAULT_BF16_SHARD_DIR if str(_request_value(request, "precision_mode", "auto")) != "fp8" else DEFAULT_FP8_SHARD_DIR)) / "avatar_transformer.rank00.pt").is_file(),
        "rank01_shard": (Path(_request_value(request, "shard_dir", DEFAULT_BF16_SHARD_DIR if str(_request_value(request, "precision_mode", "auto")) != "fp8" else DEFAULT_FP8_SHARD_DIR)) / "avatar_transformer.rank01.pt").is_file(),
        "ckpt": Path(_request_value(request, "ckpt", DEFAULT_FP8_CKPT if str(_request_value(request, "precision_mode", "auto")) == "fp8" else DEFAULT_BF16_CKPT)).is_file(),
        "input": Path(_request_value(request, "input", DEFAULT_INPUT)).is_file(),
        "bridge_worker": (BRIDGE_ROOT / "scripts" / "hunyuan_avatar_optimized_fp8_fsdp2_worker.py").is_file(),
    }
    executables = {"ffmpeg": shutil.which("ffmpeg") is not None, "torchrun": shutil.which("torchrun") is not None}
    vram = _gpu_memory_status(request) if bool(_request_value(request, "require_free_vram", True)) else {"blocked": []}
    missing_imports = sorted(name for name, ok in imports.items() if not ok)
    missing_paths = sorted(name for name, ok in paths.items() if not ok)
    missing_executables = sorted(name for name, ok in executables.items() if not ok)
    return {
        "source_paths": source_paths,
        "imports": imports,
        "paths": paths,
        "executables": executables,
        "vram": vram,
        "missing_imports": missing_imports,
        "missing_paths": missing_paths,
        "missing_executables": missing_executables,
        "insufficient_vram_gpus": list(vram.get("blocked", [])),
    }


def _launch_env(request: dict[str, Any], source_paths: list[str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join(source_paths),
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HUNYUAN_AVATAR_ROOT": str(Path(_request_value(request, "avatar_root", DEFAULT_AVATAR_ROOT))),
        "CPU_OFFLOAD": "1" if bool(_request_value(request, "cpu_offload", True)) else "0",
        "MODEL_BASE": str(Path(_request_value(request, "model_base", DEFAULT_MODEL_BASE))),
        "HUNYUAN_AVATAR_LLAVA_INT8_GPU": "1",
        "HUNYUAN_AVATAR_LLAVA_GPU_DEVICE": "cuda:0",
        "HUNYUAN_AVATAR_REFERENCE_FRAMES": str(int(_request_value(request, "reference_frames", max(int(_request_value(request, "sample_n_frames", 65)), 65)))),
        "HUNYUAN_AVATAR_WINDOW_LATENTS": str(int(_request_value(request, "window_latents", 17))),
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "HUNYUAN_AVATAR_VAE_LATENT_CACHE": str(Path(_request_value(request, "vae_latent_cache", DEFAULT_AVATAR_ROOT / "cache" / "vae_latents_512_w17"))),
    })
    if request.get("cuda_visible_devices"):
        env["CUDA_VISIBLE_DEVICES"] = str(request["cuda_visible_devices"])
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--result-json")
    args = parser.parse_args(argv)

    request = _load_json(args.request_json)
    validation = _validate_request(request)
    command = _optimized_command(request)
    output_path = _expected_output_path(request)
    ready = (
        not validation["missing_imports"]
        and not validation["missing_paths"]
        and not validation["missing_executables"]
        and not validation["insufficient_vram_gpus"]
    )
    inspect_only = bool(request.get("inspect_only", True))
    payload: dict[str, Any] = {
        "status": "ready" if ready else "blocked",
        "model_id": "HunyuanVideo-Avatar",
        "inspect_only": inspect_only,
        "artifact_contract": "video_or_avatar_render_bundle",
        "launch_plan": {
            "cwd": str(Path(_request_value(request, "avatar_root", DEFAULT_AVATAR_ROOT))),
            "command": command,
            "vae_cache_command": _vae_cache_command(request),
            "env": {k: v for k, v in _launch_env(request, validation["source_paths"]).items() if k in {
                "PYTHONNOUSERSITE", "PYTHONPATH", "TOKENIZERS_PARALLELISM", "PYTORCH_CUDA_ALLOC_CONF",
                "HUNYUAN_AVATAR_ROOT", "MODEL_BASE", "HUNYUAN_AVATAR_LLAVA_INT8_GPU", "HUNYUAN_AVATAR_LLAVA_GPU_DEVICE",
                "HUNYUAN_AVATAR_REFERENCE_FRAMES", "HUNYUAN_AVATAR_WINDOW_LATENTS", "HUNYUAN_AVATAR_VAE_LATENT_CACHE", "CUDA_VISIBLE_DEVICES", "PYTORCH_ALLOC_CONF", "CPU_OFFLOAD",
            }},
            "patches": [
                "hunyuan_avatar_llava_image_token_alignment",
                "hunyuan_avatar_distributed_rank_output_none_guard",
                "hunyuan_avatar_fp8_fsdp2_int8_llava_optimized_worker",
                "hunyuan_avatar_import_time_cpu_offload_env",
                "hunyuan_avatar_torch_fsdp2_mesh_layout_pickle_state",
                "hunyuan_avatar_fp8_fsdp2_key_remap",
                "hunyuan_avatar_llava_llamamodel_model_property",
                "hunyuan_avatar_sm86_auto_bf16_fsdp_route",
                "hunyuan_avatar_vae_cache_prewarm_before_fsdp",
                "hunyuan_avatar_reference_frames_min_65",
            ],
            "expected_output": str(output_path),
        },
        "validation": validation,
        "error": None,
    }
    if not ready:
        payload["error"] = "missing imports, paths, executables, or free VRAM"
        _write_json(args.result_json, payload)
        return 3
    if not inspect_only:
        env = _launch_env(request, validation["source_paths"])
        prewarm_vae_cache = bool(_request_value(request, "prewarm_vae_cache", True))
        payload["vae_cache_prewarm"] = {"enabled": prewarm_vae_cache}
        if prewarm_vae_cache:
            cache_start = time.monotonic()
            cache_proc = subprocess.run(
                _vae_cache_command(request),
                cwd=Path(_request_value(request, "avatar_root", DEFAULT_AVATAR_ROOT)),
                env=env,
                text=True,
                capture_output=True,
                timeout=int(request["vae_cache_timeout_sec"]) if request.get("vae_cache_timeout_sec") else None,
                check=False,
            )
            payload["vae_cache_prewarm"].update({
                "returncode": cache_proc.returncode,
                "elapsed_sec": time.monotonic() - cache_start,
                "stdout_tail": cache_proc.stdout[-8000:],
                "stderr_tail": cache_proc.stderr[-8000:],
            })
            if cache_proc.returncode != 0:
                payload["status"] = "failed"
                payload["error"] = f"VAE cache prewarm exited {cache_proc.returncode}"
                if "OutOfMemoryError" in (cache_proc.stderr + cache_proc.stdout) or "CUDA out of memory" in (cache_proc.stderr + cache_proc.stdout):
                    payload["failure_class"] = "cuda_oom_or_gpu_contention"
                _write_json(args.result_json, payload)
                return 1
        start = time.monotonic()
        proc = subprocess.run(
            command,
            cwd=Path(_request_value(request, "avatar_root", DEFAULT_AVATAR_ROOT)),
            env=env,
            text=True,
            capture_output=True,
            timeout=int(request["timeout_sec"]) if request.get("timeout_sec") else None,
            check=False,
        )
        payload["returncode"] = proc.returncode
        payload["elapsed_sec"] = time.monotonic() - start
        output_capture_chars = int(_request_value(request, "output_capture_chars", 60000))
        payload["stdout_tail"] = proc.stdout[-output_capture_chars:]
        payload["stderr_tail"] = proc.stderr[-output_capture_chars:]
        combined_error = proc.stderr + "\n" + proc.stdout
        if "Missing key(s)" in combined_error or "Unexpected key(s)" in combined_error:
            payload["failure_class"] = "fsdp2_state_dict_key_layout_mismatch"
        elif "_MeshLayout" in combined_error and "shape" in combined_error:
            payload["failure_class"] = "fsdp2_mesh_layout_pickle_state_mismatch"
        elif "The size of tensor a" in combined_error and "must match the size of tensor b" in combined_error:
            payload["failure_class"] = "avatar_reference_latent_shape_mismatch"
        elif "OutOfMemoryError" in combined_error or "CUDA out of memory" in combined_error:
            payload["failure_class"] = "cuda_oom_or_gpu_contention"
        payload["output_artifact"] = str(output_path)
        payload["output_artifact_exists"] = output_path.is_file()
        payload["output_artifact_bytes"] = output_path.stat().st_size if output_path.is_file() else None
        if proc.returncode == 0 and output_path.is_file():
            payload["status"] = "verified_optimized_fp8_fsdp2_generation_ai"
        elif output_path.is_file():
            payload["status"] = "ok_with_rank_cleanup_warning"
            payload["error"] = f"torchrun exited {proc.returncode} after writing artifact"
        else:
            payload["status"] = "failed"
            payload["error"] = f"torchrun exited {proc.returncode} without expected artifact"
    _write_json(args.result_json, payload)
    return 0 if payload["status"] in {"ready", "verified_optimized_fp8_fsdp2_generation_ai", "ok_with_rank_cleanup_warning"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
