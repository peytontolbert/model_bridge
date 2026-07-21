from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

DEFAULT_MODEL_ROOT = Path("/arxiv/models")


@dataclass(frozen=True)
class BigVGANRequest:
    model_id: str = "bigvgan_v2_44khz_128band_512x"
    model_root: str = str(DEFAULT_MODEL_ROOT)
    device: str = "cuda:0"
    dtype: str = "float32"
    mel_frames: int = 4
    run_synthetic: bool = False
    use_cuda_kernel: bool = False
    remove_weight_norm: bool = True


@dataclass(frozen=True)
class BigVGANResult:
    status: str
    model_id: str
    model_path: str | None
    sampling_rate: int | None = None
    num_mels: int | None = None
    hop_size: int | None = None
    upsample_rates: tuple[int, ...] = ()
    checkpoint_files: tuple[str, ...] = ()
    use_cuda_kernel: bool = False
    model_class: str | None = None
    load_seconds: float | None = None
    synthetic_mel_shape: tuple[int, ...] = ()
    waveform_shape: tuple[int, ...] = ()
    waveform_dtype: str | None = None
    waveform_seconds: float | None = None
    artifact_contract: str = "bigvgan_mel_vocoder_contract"
    error: str | None = None


class BigVGANBridgeError(RuntimeError):
    pass


@contextmanager
def _source_context(model_path: Path) -> Iterator[None]:
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    os.chdir(model_path)
    sys.path.insert(0, str(model_path))
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def resolve_model_path(model_id: str, model_root: str | Path = DEFAULT_MODEL_ROOT) -> Path:
    candidate = Path(model_id)
    if candidate.exists():
        return candidate
    root = Path(model_root)
    candidates = [root / model_id]
    if "/" in model_id:
        candidates.append(root / model_id.split("/", 1)[1])
        candidates.append(root / model_id.replace("/", "--"))
    for item in candidates:
        if item.exists():
            return item
    raise BigVGANBridgeError(f"model path not found for {model_id!r} under {root}")


def read_bigvgan_config(model_path: str | Path) -> dict[str, Any]:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        raise BigVGANBridgeError(f"missing config.json: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def inspect_bigvgan(request: BigVGANRequest) -> BigVGANResult:
    try:
        model_path = resolve_model_path(request.model_id, request.model_root)
        config = read_bigvgan_config(model_path)
        checkpoints = tuple(sorted(path.name for path in model_path.glob("bigvgan_generator*.pt")))
        status = "ready"
        model_class = None
        load_seconds = None
        mel_shape: tuple[int, ...] = ()
        waveform_shape: tuple[int, ...] = ()
        waveform_dtype = None
        waveform_seconds = None
        error = None
        if request.run_synthetic:
            try:
                import torch

                started = time.perf_counter()
                with _source_context(model_path):
                    import bigvgan as bigvgan_module

                    model = bigvgan_module.BigVGAN.from_pretrained(
                        str(model_path),
                        use_cuda_kernel=request.use_cuda_kernel,
                        map_location="cpu",
                    )
                if request.remove_weight_norm:
                    model.remove_weight_norm()
                if request.device and request.device != "cpu":
                    model = model.to(request.device)
                model.eval()
                torch_dtype = getattr(torch, request.dtype)
                mel = torch.zeros(
                    (1, int(config["num_mels"]), int(request.mel_frames)),
                    dtype=torch_dtype,
                    device=request.device if request.device else "cpu",
                )
                with torch.inference_mode():
                    wav = model(mel)
                if request.device and request.device != "cpu":
                    torch.cuda.synchronize()
                load_seconds = time.perf_counter() - started
                model_class = type(model).__name__
                mel_shape = tuple(int(dim) for dim in mel.shape)
                waveform_shape = tuple(int(dim) for dim in wav.shape)
                waveform_dtype = str(wav.dtype)
                waveform_seconds = float(waveform_shape[-1]) / float(config["sampling_rate"])
                status = "ok"
            except Exception as exc:
                status = "synthetic_blocked"
                error = f"{type(exc).__name__}:{exc}"
        return BigVGANResult(
            status=status,
            model_id=request.model_id,
            model_path=str(model_path),
            sampling_rate=config.get("sampling_rate"),
            num_mels=config.get("num_mels"),
            hop_size=config.get("hop_size"),
            upsample_rates=tuple(int(item) for item in config.get("upsample_rates", [])),
            checkpoint_files=checkpoints,
            use_cuda_kernel=request.use_cuda_kernel,
            model_class=model_class,
            load_seconds=load_seconds,
            synthetic_mel_shape=mel_shape,
            waveform_shape=waveform_shape,
            waveform_dtype=waveform_dtype,
            waveform_seconds=waveform_seconds,
            error=error,
        )
    except Exception as exc:
        return BigVGANResult(status="failed", model_id=request.model_id, model_path=None, error=f"{type(exc).__name__}:{exc}")


def to_json(obj: Any) -> dict[str, Any]:
    return asdict(obj)
