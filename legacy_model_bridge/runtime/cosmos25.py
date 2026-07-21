from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from legacy_model_bridge.registry import REPO_ROOT


COSMOS25_ENV = "cosmos25_py310"
DEFAULT_WORKER_ENTRYPOINT = Path("scripts/cosmos25_official_worker.py")
DEFAULT_OUTPUT_DIR = Path("/data/tmp/legacy_model_bridge_cosmos25")

PREDICT_MODEL_ID = "nvidia/Cosmos-Predict2.5-14B"
TRANSFER_MODEL_ID = "nvidia/Cosmos-Transfer2.5-2B"

DEFAULT_CHECKPOINTS = {
    PREDICT_MODEL_ID: Path(
        "/arxiv/models/nvidia/Cosmos-Predict2.5-14B/base/post-trained/"
        "e21d2a49-4747-44c8-ba44-9f6f9243715f_ema_bf16.pt"
    ),
    TRANSFER_MODEL_ID: Path(
        "/arxiv/models/nvidia/Cosmos-Transfer2.5-2B/distilled/general/edge/"
        "41f07f13-f2e4-4e34-ba4c-86f595acbc20_ema_bf16.pt"
    ),
}

DEFAULT_REPO_ROOTS = {
    PREDICT_MODEL_ID: Path("/data/clone/third_party/cosmos-predict2.5"),
    TRANSFER_MODEL_ID: Path("/data/clone/third_party/cosmos-transfer2.5"),
}


@dataclass(frozen=True)
class Cosmos25Request:
    model_id: str
    input_files: tuple[str, ...] = ()
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    checkpoint_path: str | None = None
    repo_root: str | None = None
    model: str | None = None
    inference_type: str | None = None
    nproc_per_node: int = 1
    master_port: int = 12341
    context_parallel_size: int | None = None
    cuda_visible_devices: str | None = None
    offline_only: bool = True
    disable_guardrails: bool = True
    offload_diffusion_model: bool = False
    offload_text_encoder: bool = False
    offload_tokenizer: bool = False
    student_only: bool = False
    inspect_only: bool = True
    extra_args: dict[str, Any] | None = None


@dataclass(frozen=True)
class Cosmos25Result:
    status: str
    env: str
    command: tuple[str, ...]
    request_path: str
    output_dir: str
    result_path: str | None = None
    returncode: int | None = None
    elapsed_sec: float | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: str | None = None
    payload: dict[str, Any] | None = None
    dry_run: bool = False


def family_for_model(model_id: str) -> str:
    if model_id == PREDICT_MODEL_ID or "Cosmos-Predict2.5" in model_id:
        return "predict"
    if model_id == TRANSFER_MODEL_ID or "Cosmos-Transfer2.5" in model_id:
        return "transfer"
    raise ValueError(f"unsupported Cosmos 2.5 model id: {model_id}")


def default_checkpoint_for_model(model_id: str) -> Path:
    family_for_model(model_id)
    return DEFAULT_CHECKPOINTS[PREDICT_MODEL_ID if "Predict" in model_id else TRANSFER_MODEL_ID]


def default_repo_root_for_model(model_id: str) -> Path:
    family_for_model(model_id)
    return DEFAULT_REPO_ROOTS[PREDICT_MODEL_ID if "Predict" in model_id else TRANSFER_MODEL_ID]


def _request_payload(request: Cosmos25Request) -> dict[str, Any]:
    payload = asdict(request)
    family = family_for_model(request.model_id)
    payload["family"] = family
    payload["checkpoint_path"] = str(request.checkpoint_path or default_checkpoint_for_model(request.model_id))
    payload["repo_root"] = str(request.repo_root or default_repo_root_for_model(request.model_id))
    if request.student_only and family != "transfer":
        raise ValueError("student_only is only supported for nvidia/Cosmos-Transfer2.5-2B")
    if request.model is None:
        payload["model"] = "14B/post-trained" if family == "predict" else "edge/distilled"
    if request.inference_type is None and family == "predict":
        payload["inference_type"] = "text2world"
    return payload


def write_cosmos25_request(request: Cosmos25Request, output_dir: str | Path | None = None) -> Path:
    out_dir = Path(output_dir or request.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    request_path = out_dir / f"cosmos25_request_{uuid4().hex}.json"
    request_path.write_text(json.dumps(_request_payload(request), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return request_path


def build_cosmos25_worker_command(request_path: str | Path, result_path: str | Path | None = None) -> tuple[str, ...]:
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        COSMOS25_ENV,
        "env",
        "PYTHONNOUSERSITE=1",
        "PYTHONPATH=.",
        "python",
        str(DEFAULT_WORKER_ENTRYPOINT),
        "--request-json",
        str(request_path),
    ]
    if result_path is not None:
        cmd.extend(["--result-json", str(result_path)])
    return tuple(cmd)


def plan_or_run_cosmos25(request: Cosmos25Request, *, dry_run: bool = False, timeout_sec: int | None = None) -> Cosmos25Result:
    out_dir = Path(request.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    req_path = write_cosmos25_request(request, out_dir)
    result_path = out_dir / f"cosmos25_result_{req_path.stem.rsplit('_', 1)[-1]}.json"
    cmd = build_cosmos25_worker_command(req_path, result_path)
    if dry_run:
        return Cosmos25Result(
            status="dry_run",
            env=COSMOS25_ENV,
            command=cmd,
            request_path=str(req_path),
            output_dir=str(out_dir),
            result_path=str(result_path),
            dry_run=True,
        )
    start = time.monotonic()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, timeout=timeout_sec, check=False)
    elapsed = time.monotonic() - start
    status = "ok" if proc.returncode == 0 else "failed"
    error = None if proc.returncode == 0 else f"worker exited {proc.returncode}"
    payload: dict[str, Any] | None = None
    if result_path.is_file():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            status = str(payload.get("status", status))
            error = payload.get("error", error)
        except json.JSONDecodeError:
            error = error or "worker result JSON was invalid"
    return Cosmos25Result(
        status=status,
        env=COSMOS25_ENV,
        command=cmd,
        request_path=str(req_path),
        output_dir=str(out_dir),
        result_path=str(result_path),
        returncode=proc.returncode,
        elapsed_sec=elapsed,
        stdout_tail=proc.stdout[-4000:],
        stderr_tail=proc.stderr[-4000:],
        error=error,
        payload=payload,
    )


def to_json(obj: Any) -> dict[str, Any]:
    return asdict(obj)
