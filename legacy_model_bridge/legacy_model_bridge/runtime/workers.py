from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from legacy_model_bridge.registry import REPO_ROOT


DEFAULT_WORKER_REGISTRY_PATH = REPO_ROOT / "data" / "worker_registry.json"


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    lane: str
    models: tuple[str, ...]
    env: str
    expected_python: tuple[str, ...]
    entrypoint: str
    artifact_contract: str
    required_imports: tuple[str, ...] = ()
    source_paths: tuple[str, ...] = ()
    required_paths: tuple[str, ...] = ()
    required_executables: tuple[str, ...] = ()
    min_cuda_devices: int = 0
    env_overrides: dict[str, str] | None = None
    optional_imports: tuple[str, ...] = ()
    mismatch_classes: tuple[str, ...] = ()
    preflight_kind: str = "python_imports"
    status: str = "planned"
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "WorkerSpec":
        return cls(
            worker_id=str(raw["worker_id"]),
            lane=str(raw["lane"]),
            models=tuple(raw.get("models", [])),
            env=str(raw["env"]),
            expected_python=tuple(raw.get("expected_python", [])),
            entrypoint=str(raw.get("entrypoint", "")),
            artifact_contract=str(raw.get("artifact_contract", "")),
            required_imports=tuple(raw.get("required_imports", [])),
            source_paths=tuple(raw.get("source_paths", [])),
            required_paths=tuple(raw.get("required_paths", [])),
            required_executables=tuple(raw.get("required_executables", [])),
            min_cuda_devices=int(raw.get("min_cuda_devices", 0)),
            env_overrides={str(k): str(v) for k, v in raw.get("env_overrides", {}).items()} or None,
            optional_imports=tuple(raw.get("optional_imports", [])),
            mismatch_classes=tuple(raw.get("mismatch_classes", [])),
            preflight_kind=str(raw.get("preflight_kind", "python_imports")),
            status=str(raw.get("status", "planned")),
            notes=str(raw.get("notes", "")),
        )


@dataclass(frozen=True)
class WorkerPreflightResult:
    worker_id: str
    env: str
    expected_python: tuple[str, ...]
    command: tuple[str, ...]
    status: str
    actual_python: str | None = None
    imports: dict[str, bool] | None = None
    missing_imports: tuple[str, ...] = ()
    paths: dict[str, bool] | None = None
    missing_paths: tuple[str, ...] = ()
    executables: dict[str, bool] | None = None
    missing_executables: tuple[str, ...] = ()
    versions: dict[str, str] | None = None
    torch_cuda_available: bool | None = None
    torch_cuda_version: str | None = None
    torch_cuda_device_count: int | None = None
    returncode: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: str | None = None


class WorkerRegistryError(RuntimeError):
    pass


class WorkerRegistry:
    def __init__(self, workers: list[WorkerSpec], metadata: dict[str, Any] | None = None) -> None:
        self.workers = workers
        self.metadata = metadata or {}
        self._by_id = {worker.worker_id: worker for worker in workers}
        self._by_model = {model: worker for worker in workers for model in worker.models}

    def get(self, worker_id: str) -> WorkerSpec:
        try:
            return self._by_id[worker_id]
        except KeyError as exc:
            raise WorkerRegistryError(f"worker is not registered: {worker_id}") from exc

    def for_model(self, model_id: str) -> WorkerSpec:
        try:
            return self._by_model[model_id]
        except KeyError as exc:
            raise WorkerRegistryError(f"model has no registered worker: {model_id}") from exc

    def filter(self, *, lane: str | None = None, env: str | None = None, status: str | None = None) -> list[WorkerSpec]:
        workers = self.workers
        if lane is not None:
            workers = [worker for worker in workers if worker.lane == lane]
        if env is not None:
            workers = [worker for worker in workers if worker.env == env]
        if status is not None:
            workers = [worker for worker in workers if worker.status == status]
        return workers


def load_worker_registry(path: str | Path = DEFAULT_WORKER_REGISTRY_PATH) -> WorkerRegistry:
    raw = json.loads(Path(path).read_text())
    return WorkerRegistry([WorkerSpec.from_dict(item) for item in raw["workers"]], raw.get("metadata", {}))


def build_worker_preflight_command(worker: WorkerSpec) -> tuple[str, ...]:
    probe = {
        "imports": list(worker.required_imports),
        "source_paths": list(worker.source_paths),
        "required_paths": list(worker.required_paths),
        "required_executables": list(worker.required_executables),
        "min_cuda_devices": worker.min_cuda_devices,
    }
    env_overrides = worker.env_overrides or {}
    code = (
        "import importlib,importlib.metadata,importlib.util,json,os,shutil,sys;"
        f"probe={probe!r};"
        "[sys.path.insert(0,p) for p in probe['source_paths'] if p and p not in sys.path];"
        "imports={};"
        "\nfor m in probe[\'imports\']:\n"
        "    try:\n"
        "        imports[m]=(importlib.util.find_spec(m) is not None)\n"
        "    except Exception:\n"
        "        imports[m]=False\n"
        "paths={p:os.path.exists(p) for p in probe['required_paths']};"
        "executables={e:(shutil.which(e) is not None) for e in probe['required_executables']};"
        "versions={};"
        "\nfor n in ['torch','transformers','diffusers','nemo_toolkit','nemo','flash_attn','natten','transformer_engine']:\n"
        "    try:\n        versions[n]=importlib.metadata.version(n)\n    except Exception:\n        versions[n]='missing'\n"
        "torch_cuda_available=None;torch_cuda_version=None;torch_cuda_device_count=None;"
        "\nif imports.get('torch'):\n"
        "    import torch;torch_cuda_available=bool(torch.cuda.is_available());torch_cuda_version=torch.version.cuda;torch_cuda_device_count=torch.cuda.device_count() if torch_cuda_available else 0\n"
        "print(json.dumps({'python':f'{sys.version_info.major}.{sys.version_info.minor}','imports':imports,'paths':paths,'executables':executables,'versions':versions,'torch_cuda_available':torch_cuda_available,'torch_cuda_version':torch_cuda_version,'torch_cuda_device_count':torch_cuda_device_count},sort_keys=True))"
    )
    env_items = tuple(f"{key}={value}" for key, value in sorted(env_overrides.items()))
    return ("conda", "run", "-n", worker.env, "env", "PYTHONNOUSERSITE=1", *env_items, "python", "-c", code)


def preflight_worker(worker: WorkerSpec, *, timeout_sec: int = 30) -> WorkerPreflightResult:
    cmd = build_worker_preflight_command(worker)
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_sec, check=False)
    except Exception as exc:
        return WorkerPreflightResult(
            worker_id=worker.worker_id,
            env=worker.env,
            expected_python=worker.expected_python,
            command=cmd,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
    payload: dict[str, Any] = {}
    if proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
        except json.JSONDecodeError:
            payload = {}
    actual_python = payload.get("python")
    imports = payload.get("imports") if isinstance(payload.get("imports"), dict) else {}
    paths = payload.get("paths") if isinstance(payload.get("paths"), dict) else {}
    executables = payload.get("executables") if isinstance(payload.get("executables"), dict) else {}
    versions = payload.get("versions") if isinstance(payload.get("versions"), dict) else {}
    torch_cuda_available = payload.get("torch_cuda_available")
    torch_cuda_version = payload.get("torch_cuda_version")
    torch_cuda_device_count = payload.get("torch_cuda_device_count")
    missing = tuple(name for name, ok in imports.items() if not ok)
    missing_paths = tuple(name for name, ok in paths.items() if not ok)
    missing_executables = tuple(name for name, ok in executables.items() if not ok)
    status = "ok"
    error = None
    if proc.returncode != 0:
        status = "failed"
        error = f"preflight exited {proc.returncode}"
    elif worker.expected_python and actual_python not in worker.expected_python:
        status = "mismatch"
        error = f"expected Python {','.join(worker.expected_python)}, got {actual_python or 'unknown'}"
    elif missing:
        status = "missing_imports"
        error = "missing imports: " + ", ".join(missing)
    elif missing_paths:
        status = "missing_paths"
        error = "missing paths: " + ", ".join(missing_paths)
    elif missing_executables:
        status = "missing_executables"
        error = "missing executables: " + ", ".join(missing_executables)
    elif worker.min_cuda_devices and (not isinstance(torch_cuda_device_count, int) or torch_cuda_device_count < worker.min_cuda_devices):
        status = "insufficient_cuda_devices"
        error = f"expected at least {worker.min_cuda_devices} CUDA devices, got {torch_cuda_device_count if torch_cuda_device_count is not None else 'unknown'}"
    return WorkerPreflightResult(
        worker_id=worker.worker_id,
        env=worker.env,
        expected_python=worker.expected_python,
        command=cmd,
        status=status,
        actual_python=actual_python,
        imports=imports,
        missing_imports=missing,
        paths=paths,
        missing_paths=missing_paths,
        executables=executables,
        missing_executables=missing_executables,
        versions=versions,
        torch_cuda_available=torch_cuda_available if isinstance(torch_cuda_available, bool) else None,
        torch_cuda_version=torch_cuda_version if isinstance(torch_cuda_version, str) else None,
        torch_cuda_device_count=torch_cuda_device_count if isinstance(torch_cuda_device_count, int) else None,
        returncode=proc.returncode,
        stdout_tail=proc.stdout[-4000:],
        stderr_tail=proc.stderr[-4000:],
        error=error,
    )


def to_json(obj: Any) -> dict[str, Any]:
    return asdict(obj)
