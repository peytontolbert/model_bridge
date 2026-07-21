from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKER_CWD = REPO_ROOT
DEFAULT_WORKER_ENTRYPOINT = Path("scripts/three_d_gen_worker.py")


@dataclass(frozen=True)
class ThreeDBackend:
    backend: str
    model_id: str
    family: str
    env: str
    model_path: str
    worker_cwd: str
    worker_entrypoint: str
    artifact_contract: str = "mesh_bundle_glb"
    user_env: str = "ai"
    worker_boundary: str = "process"
    expected_python: tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class ThreeDGenRequest:
    backend: str
    image_path: str
    output_dir: str
    prompt: str | None = None
    model_id: str | None = None
    model_path: str | None = None
    output_format: str = "glb"
    seed: int = 42
    variant: str | None = None
    texture: bool = False
    extra_args: dict[str, Any] | None = None


@dataclass(frozen=True)
class ThreeDGenResult:
    backend: str
    status: str
    env: str
    command: tuple[str, ...]
    request_path: str
    output_dir: str
    artifacts: dict[str, str]
    returncode: int | None = None
    elapsed_sec: float | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: str | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class ThreeDPreflightResult:
    backend: str
    env: str
    expected_python: tuple[str, ...]
    command: tuple[str, ...]
    status: str
    actual_python: str | None = None
    returncode: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: str | None = None


@dataclass(frozen=True)
class ThreeDConflictReport:
    status: str
    bridge_strategy: str
    conflicts: tuple[str, ...]
    bridge_patches: tuple[str, ...]
    trellis: ThreeDBackend
    hunyuan3d: ThreeDBackend


class ThreeDGenBridgeError(RuntimeError):
    pass


def resolve_trellis2_model_path(hf_cache: str | Path = "/data/huggingface/hub") -> str:
    repo_cache = Path(hf_cache) / "models--microsoft--TRELLIS.2-4B"
    ref = repo_cache / "refs" / "main"
    if ref.is_file():
        revision = ref.read_text(encoding="utf-8").strip()
        snapshot = repo_cache / "snapshots" / revision
        if snapshot.exists():
            return str(snapshot)
    known_snapshot = repo_cache / "snapshots" / "af44b45f2e35a493886929c6d786e563ec68364d"
    if known_snapshot.exists():
        return str(known_snapshot)
    return str(repo_cache)


BACKENDS: dict[str, ThreeDBackend] = {
    "trellis": ThreeDBackend(
        backend="trellis",
        model_id="microsoft/TRELLIS.2-4B",
        family="trellis2",
        env="ai",
        model_path=resolve_trellis2_model_path(),
        worker_cwd=str(DEFAULT_WORKER_CWD),
        worker_entrypoint=str(DEFAULT_WORKER_ENTRYPOINT),
        expected_python=("3.11", "3.12"),
        detail=(
            "The legacy trellis env was intentionally removed. TRELLIS.2 now has its "
            "default core native stack rebuilt in ai: nvdiffrast, cumesh, flex_gemm, "
            "o_voxel, and flash-attn. spconv and kaolin remain optional legacy parity; "
            "real-image one-step GLB generation smoke passes in ai."
        ),
    ),
    "hunyuan3d": ThreeDBackend(
        backend="hunyuan3d",
        model_id="Hunyuan3D-2mv",
        family="hunyuan3d_hy3dgen",
        env="ai",
        model_path="/arxiv/models/Hunyuan3D-2mv",
        worker_cwd=str(DEFAULT_WORKER_CWD),
        worker_entrypoint=str(DEFAULT_WORKER_ENTRYPOINT),
        worker_boundary="none_or_process",
        expected_python=("3.11",),
        detail=(
            "Hunyuan3D-2mv is latest-env runnable through source-routed official "
            "hy3dgen; the same worker contract is used so 3D outputs stay normalized."
        ),
    ),
}

MODEL_ID_TO_BACKEND = {backend.model_id: name for name, backend in BACKENDS.items()}


def get_3d_backend(name_or_model_id: str) -> ThreeDBackend:
    backend_name = MODEL_ID_TO_BACKEND.get(name_or_model_id, name_or_model_id)
    try:
        return BACKENDS[backend_name]
    except KeyError as exc:
        raise ThreeDGenBridgeError(f"unsupported 3D backend or model: {name_or_model_id}") from exc


def list_3d_backends() -> tuple[ThreeDBackend, ...]:
    return tuple(BACKENDS[name] for name in sorted(BACKENDS))




def build_3d_preflight_command(backend: str) -> tuple[str, ...]:
    spec = get_3d_backend(backend)
    return (
        "conda",
        "run",
        "-n",
        spec.env,
        "env",
        "PYTHONNOUSERSITE=1",
        "python",
        "-c",
        "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
    )


def preflight_3d_backend(backend: str, *, timeout_sec: int = 30) -> ThreeDPreflightResult:
    spec = get_3d_backend(backend)
    cmd = build_3d_preflight_command(backend)
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_sec, check=False)
    except Exception as exc:
        return ThreeDPreflightResult(
            backend=spec.backend,
            env=spec.env,
            expected_python=spec.expected_python,
            command=cmd,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
    actual = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else None
    status = "ok" if proc.returncode == 0 and (not spec.expected_python or actual in spec.expected_python) else "mismatch"
    error = None if status == "ok" else f"expected Python {','.join(spec.expected_python) or 'unknown'}, got {actual or 'unknown'}"
    if proc.returncode != 0:
        status = "failed"
        error = f"preflight exited {proc.returncode}"
    return ThreeDPreflightResult(
        backend=spec.backend,
        env=spec.env,
        expected_python=spec.expected_python,
        command=cmd,
        status=status,
        actual_python=actual,
        returncode=proc.returncode,
        stdout_tail=proc.stdout[-4000:],
        stderr_tail=proc.stderr[-4000:],
        error=error,
    )


def _request_payload(request: ThreeDGenRequest) -> dict[str, Any]:
    backend = get_3d_backend(request.backend)
    payload = asdict(request)
    payload["model_id"] = request.model_id or backend.model_id
    default_model_path = resolve_trellis2_model_path() if backend.backend == "trellis" else backend.model_path
    payload["model_path"] = request.model_path or default_model_path
    return payload


def write_3d_request(request: ThreeDGenRequest, output_dir: str | Path | None = None) -> Path:
    out_dir = Path(output_dir or request.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    request_path = out_dir / f"three_d_gen_request_{request.backend}_{uuid4().hex}.json"
    request_path.write_text(json.dumps(_request_payload(request), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return request_path


def build_3d_worker_command(
    request_path: str | Path,
    backend: str,
    result_path: str | Path | None = None,
    *,
    worker_cwd: str | Path | None = None,
    worker_entrypoint: str | Path | None = None,
    env_overrides: dict[str, str] | None = None,
) -> tuple[str, ...]:
    spec = get_3d_backend(backend)
    entrypoint = str(worker_entrypoint or spec.worker_entrypoint)
    command = [
        "conda",
        "run",
        "-n",
        spec.env,
        "env",
        "PYTHONNOUSERSITE=1",
        "PYTHONPATH=.",
        "HF_HOME=/data/huggingface",
        "HUGGINGFACE_HUB_CACHE=/data/huggingface/hub",
        "TRANSFORMERS_CACHE=/data/huggingface/hub",
    ]
    if spec.backend == "trellis":
        command.append("LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
    if env_overrides:
        for key, value in sorted(env_overrides.items()):
            if not key or "=" in key:
                raise ValueError(f"invalid environment override key: {key!r}")
            command.append(f"{key}={value}")
    command.extend([
        "python",
        entrypoint,
        "--request-json",
        str(request_path),
    ])
    if result_path is not None:
        command.extend(["--result-json", str(result_path)])
    return tuple(command)


def expected_artifacts(request: ThreeDGenRequest) -> dict[str, str]:
    out_dir = Path(request.output_dir)
    extra = request.extra_args or {}
    stem = str(extra.get("output_stem", "mesh"))
    artifacts: dict[str, str] = {request.output_format: str(out_dir / f"{stem}.{request.output_format}")}
    if request.output_format == "glb":
        artifacts = {"glb": str(out_dir / f"{stem}.glb")}
    if request.texture:
        artifacts["textured_glb"] = str(out_dir / f"{stem}_textured.glb")
    return artifacts


def generate_3d(
    request: ThreeDGenRequest,
    *,
    dry_run: bool = False,
    timeout_sec: int | None = None,
    worker_cwd: str | Path | None = None,
    worker_entrypoint: str | Path | None = None,
    env_overrides: dict[str, str] | None = None,
) -> ThreeDGenResult:
    backend = get_3d_backend(request.backend)
    if not request.image_path:
        raise ValueError("image_path is required")

    out_dir = Path(request.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    req_path = write_3d_request(request, out_dir)
    suffix = req_path.stem.rsplit("_", 1)[-1]
    result_path = out_dir / f"three_d_gen_result_{request.backend}_{suffix}.json"
    cmd = build_3d_worker_command(
        req_path,
        request.backend,
        result_path,
        worker_cwd=worker_cwd,
        worker_entrypoint=worker_entrypoint,
        env_overrides=env_overrides,
    )
    artifacts = expected_artifacts(request)
    if dry_run:
        return ThreeDGenResult(
            backend=request.backend,
            status="dry_run",
            env=backend.env,
            command=cmd,
            request_path=str(req_path),
            output_dir=str(out_dir),
            artifacts=artifacts,
            dry_run=True,
        )

    start = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=Path(worker_cwd or backend.worker_cwd),
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )
    elapsed = time.monotonic() - start
    status = "ok" if proc.returncode == 0 else "failed"
    error = None if proc.returncode == 0 else f"worker exited {proc.returncode}"
    if result_path.is_file():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
            error = error or "worker result JSON was invalid"
        if payload:
            status = str(payload.get("status", status))
            artifacts = dict(payload.get("artifacts", artifacts))
            error = payload.get("error", error)
    return ThreeDGenResult(
        backend=request.backend,
        status=status,
        env=backend.env,
        command=cmd,
        request_path=str(req_path),
        output_dir=str(out_dir),
        artifacts=artifacts,
        returncode=proc.returncode,
        elapsed_sec=elapsed,
        stdout_tail=proc.stdout[-4000:],
        stderr_tail=proc.stderr[-4000:],
        error=error,
    )


def compare_trellis_hunyuan3d() -> ThreeDConflictReport:
    return ThreeDConflictReport(
        status="trellis_ai_real_image_tiny_glb_verified",
        bridge_strategy=(
            "Expose one latest-env bridge API and normalize results to mesh_bundle_glb artifacts; "
            "TRELLIS.2 now uses rebuilt native geometry imports in ai and passes a real-image one-step GLB smoke."
        ),
        conflicts=(
            "Legacy trellis env was deleted to free space for latest-env native builds.",
            "TRELLIS.2 core native stack imports in ai after rebuilding nvdiffrast, cumesh, flex_gemm, and o_voxel; spconv and kaolin are optional parity items.",
            "Hunyuan3D-2mv uses source-routed official hy3dgen in ai; TRELLIS.2 uses source-routed official trellis2 in ai with the default flex_gemm backend verified.",
        ),
        bridge_patches=(
            "trellis_official_runtime_worker",
            "trellis_hf_cache_root_resolution",
            "trellis_mesh_artifact_glb_contract",
            "hunyuan3d_hy3dgen_official_loader",
            "hunyuan3d_single_image_to_mv_front_view",
            "hunyuan3d_mesh_artifact_glb_contract",
        ),
        trellis=BACKENDS["trellis"],
        hunyuan3d=BACKENDS["hunyuan3d"],
    )


def to_json(obj: Any) -> dict[str, Any]:
    return asdict(obj)
