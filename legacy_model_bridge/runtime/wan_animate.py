from __future__ import annotations

import importlib
import importlib.metadata as metadata
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .sageattention import SageAttentionStatus, sageattention_status, to_json as sageattention_to_json


DEFAULT_MODEL_PATH = Path("/arxiv/models/Wan-AI--Wan2.2-Animate-14B")
DEFAULT_WAN_SOURCE = Path("/home/peyton/src/Wan2.2")
DEFAULT_TRANSFORMER10_ROOT = Path("/data/transformer_10")
DEFAULT_INT8_ARTIFACT_DIR = DEFAULT_TRANSFORMER10_ROOT / "checkpoints" / "wan_animate_int8_weightonly_v2"
DEFAULT_CACHE_SMOKE_ROOT = DEFAULT_TRANSFORMER10_ROOT / "tmp" / "wan_animate_teacher_cache_smoke"
MODEL_ID = "Wan-AI--Wan2.2-Animate-14B"


class WanAnimateBridgeError(RuntimeError):
    pass


@dataclass(frozen=True)
class WanAnimatePaths:
    model_path: Path = DEFAULT_MODEL_PATH
    wan_source: Path = DEFAULT_WAN_SOURCE
    transformer10_root: Path = DEFAULT_TRANSFORMER10_ROOT
    int8_artifact_dir: Path = DEFAULT_INT8_ARTIFACT_DIR
    cache_smoke_root: Path = DEFAULT_CACHE_SMOKE_ROOT


@dataclass(frozen=True)
class WanAnimateStatus:
    model_id: str
    status: str
    runnable: bool
    preferred_env: str
    model_path: str
    wan_source: str
    checkpoint_files: tuple[str, ...]
    missing_checkpoint_files: tuple[str, ...]
    diffusion_shard_count: int
    diffusion_index_ok: bool
    cache_root: str
    t5_cache_path: str | None
    control_latent_paths: tuple[str, ...]
    teacher_cache_manifest: str | None
    teacher_cache_steps: int
    teacher_cache_output: str | None
    int8_artifact_dir: str
    int8_manifest: str | None
    int8_format: str | None
    int8_num_blocks: int | None
    int8_safetensor_block_count: int
    int8_quantized_modules: int | None
    int8_storage_formats: tuple[str, ...]
    required_imports: dict[str, str | None]
    sageattention: SageAttentionStatus
    selected_lightx2v_attention_backend: str
    blockers: tuple[str, ...]
    evidence_refs: tuple[str, ...]


def wan_animate_status(paths: WanAnimatePaths | None = None, *, device: int | str | None = None) -> WanAnimateStatus:
    paths = paths or WanAnimatePaths()
    missing: list[str] = []
    checkpoint_files = _required_checkpoint_files()
    for relative in checkpoint_files:
        if not (paths.model_path / relative).is_file():
            missing.append(relative)

    index_ok = _diffusion_index_is_complete(paths.model_path)
    diffusion_shards = tuple(sorted(paths.model_path.glob("diffusion_pytorch_model-*-of-*.safetensors")))
    if not index_ok:
        missing.append("diffusion_pytorch_model.safetensors.index.json:complete_weight_map")

    t5_cache = paths.cache_smoke_root / "input33" / ".t5_context_rank0.pt"
    control_latents = tuple(sorted((paths.cache_smoke_root / "input33" / ".wan_vae_control_latents").glob("*.pt")))
    teacher_manifest = paths.cache_smoke_root / "cache" / "window_0000" / "manifest.json"
    teacher_output = paths.cache_smoke_root / "output" / "wan_animate_int8_33f_4step_teacher_cache.mp4"
    teacher_steps = _teacher_step_count(teacher_manifest)

    int8_manifest = paths.int8_artifact_dir / "manifest.json"
    int8_payload = _read_json(int8_manifest)
    int8_blocks = tuple(sorted((paths.int8_artifact_dir / "blocks").glob("block_*.safetensors")))

    imports = _required_import_versions()
    sage = sageattention_status(device)
    backend = select_lightx2v_attention_backend(sage)

    blockers: list[str] = []
    if not paths.model_path.is_dir():
        blockers.append("Wan Animate model path is missing")
    if not paths.wan_source.is_dir():
        blockers.append("Wan2.2 source checkout is missing")
    if missing:
        blockers.append("Wan Animate checkpoint layout is incomplete")
    if not t5_cache.is_file():
        blockers.append("cached T5 prompt context is missing")
    if not control_latents:
        blockers.append("cached VAE control latents are missing")
    if not teacher_manifest.is_file() or teacher_steps <= 0:
        blockers.append("teacher cache manifest is missing or empty")
    if not teacher_output.is_file():
        blockers.append("bounded Wan Animate output proof is missing")
    if int8_payload.get("format") != "modelstack.wan-animate.int8.blocks.v1":
        blockers.append("Wan Animate INT8 block manifest is missing or has unexpected format")
    if int(int8_payload.get("num_blocks") or 0) != len(int8_blocks):
        blockers.append("Wan Animate INT8 safetensor block count does not match manifest")
    failed_imports = tuple(name for name, version in imports.items() if version is None)
    if failed_imports:
        blockers.append("required ai imports are missing: " + ", ".join(failed_imports))
    if backend == "sdpa":
        blockers.append("SageAttention2 backend is unavailable; Wan Animate can fall back but acceleration is unproven")

    evidence_refs = (
        "/data/transformer_10/tmp/wan_animate_teacher_cache_smoke/output/wan_animate_int8_33f_4step_teacher_cache.mp4",
        "/data/transformer_10/tmp/wan_animate_teacher_cache_smoke/cache/window_0000/manifest.json",
        "/data/transformer_10/checkpoints/wan_animate_int8_weightonly_v2/manifest.json",
        "reports/world-model-smokes/sageattention.ai.cuda2.probe.json",
    )

    return WanAnimateStatus(
        model_id=MODEL_ID,
        status="verified_wan_animate_cached_int8_bridge_ai" if not blockers else "candidate_wan_animate_custom_bridge",
        runnable=not blockers,
        preferred_env="ai",
        model_path=str(paths.model_path),
        wan_source=str(paths.wan_source),
        checkpoint_files=checkpoint_files,
        missing_checkpoint_files=tuple(missing),
        diffusion_shard_count=len(diffusion_shards),
        diffusion_index_ok=index_ok,
        cache_root=str(paths.cache_smoke_root),
        t5_cache_path=str(t5_cache) if t5_cache.is_file() else None,
        control_latent_paths=tuple(str(path) for path in control_latents),
        teacher_cache_manifest=str(teacher_manifest) if teacher_manifest.is_file() else None,
        teacher_cache_steps=teacher_steps,
        teacher_cache_output=str(teacher_output) if teacher_output.is_file() else None,
        int8_artifact_dir=str(paths.int8_artifact_dir),
        int8_manifest=str(int8_manifest) if int8_manifest.is_file() else None,
        int8_format=int8_payload.get("format"),
        int8_num_blocks=int8_payload.get("num_blocks"),
        int8_safetensor_block_count=len(int8_blocks),
        int8_quantized_modules=len(int8_payload.get("quantized_modules", [])) if int8_payload else None,
        int8_storage_formats=tuple(int8_payload.get("storage_formats", [])),
        required_imports=imports,
        sageattention=sage,
        selected_lightx2v_attention_backend=backend,
        blockers=tuple(blockers),
        evidence_refs=evidence_refs,
    )


def select_lightx2v_attention_backend(status: SageAttentionStatus | None = None) -> str:
    status = status or sageattention_status()
    if status.available and (status.has_fp16_triton or status.has_fp16_cuda):
        return "sage_attn2"
    return "sdpa"


def to_json(status: WanAnimateStatus) -> dict[str, Any]:
    payload = asdict(status)
    payload["sageattention"] = sageattention_to_json(status.sageattention)
    return payload


def _required_checkpoint_files() -> tuple[str, ...]:
    return (
        "Wan2.1_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "diffusion_pytorch_model.safetensors.index.json",
        "diffusion_pytorch_model-00001-of-00004.safetensors",
        "diffusion_pytorch_model-00002-of-00004.safetensors",
        "diffusion_pytorch_model-00003-of-00004.safetensors",
        "diffusion_pytorch_model-00004-of-00004.safetensors",
        "xlm-roberta-large/config.json",
        "xlm-roberta-large/model.safetensors",
        "relighting_lora/adapter_model.safetensors",
        "relighting_lora.ckpt",
        "process_checkpoint/det/yolov10m.onnx",
        "process_checkpoint/pose2d/vitpose_h_wholebody.onnx/end2end.onnx",
        "process_checkpoint/sam2/sam2_hiera_base_plus.pt",
    )


def _diffusion_index_is_complete(model_path: Path) -> bool:
    index_path = model_path / "diffusion_pytorch_model.safetensors.index.json"
    payload = _read_json(index_path)
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    expected = {f"diffusion_pytorch_model-{index:05d}-of-00004.safetensors" for index in range(1, 5)}
    return expected.issubset(set(weight_map.values()))


def _teacher_step_count(manifest_path: Path) -> int:
    payload = _read_json(manifest_path)
    steps = payload.get("steps")
    return len(steps) if isinstance(steps, list) else 0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WanAnimateBridgeError(f"invalid JSON: {path}") from exc


def _required_import_versions() -> dict[str, str | None]:
    modules = (
        "accelerate",
        "cv2",
        "decord",
        "diffusers",
        "einops",
        "flash_attn",
        "imageio",
        "PIL",
        "safetensors",
        "sageattention",
        "torch",
        "torchvision",
        "transformers",
    )
    return {name: _module_version(name) for name in modules}


def _module_version(name: str) -> str | None:
    if importlib.util.find_spec(name) is None:
        return None
    distribution = "Pillow" if name == "PIL" else name
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        module = importlib.import_module(name)
        return str(getattr(module, "__version__", "unknown"))
