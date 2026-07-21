from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ROOT = Path("/arxiv/models")


@dataclass(frozen=True)
class ONNXRequest:
    model_id: str
    model_root: str = str(DEFAULT_MODEL_ROOT)
    provider: str = "CPUExecutionProvider"
    run_synthetic: bool = False
    max_dynamic_dim: int = 4


@dataclass(frozen=True)
class ONNXResult:
    status: str
    model_id: str
    model_path: str | None
    onnx_path: str | None = None
    providers: tuple[str, ...] = ()
    inputs: tuple[dict[str, Any], ...] = ()
    outputs: tuple[dict[str, Any], ...] = ()
    sidecars: dict[str, Any] | None = None
    synthetic_outputs: tuple[dict[str, Any], ...] = ()
    artifact_contract: str = "onnx_tensor_contract"
    error: str | None = None


class ONNXBridgeError(RuntimeError):
    pass


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
    raise ONNXBridgeError(f"model path not found for {model_id!r} under {root}")


def find_onnx_path(model_path: str | Path) -> Path:
    root = Path(model_path)
    direct = root / "network.onnx"
    if direct.is_file():
        return direct
    matches = sorted(root.glob("*.onnx")) or sorted(root.glob("**/*.onnx"))
    if not matches:
        raise ONNXBridgeError(f"no .onnx file found under {root}")
    return matches[0]


def _value_info(value: Any) -> dict[str, Any]:
    return {"name": value.name, "type": value.type, "shape": list(value.shape)}


def _json_sidecar(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}:{exc}"}


def inspect_sidecars(model_path: str | Path) -> dict[str, Any]:
    root = Path(model_path)
    sidecars: dict[str, Any] = {"json": {}, "npz": {}}
    for path in sorted(root.glob("*.json")):
        sidecars["json"][path.name] = _json_sidecar(path)
    try:
        import numpy as np

        for path in sorted(root.glob("*.npz")):
            archive = np.load(path)
            sidecars["npz"][path.name] = {name: list(archive[name].shape) for name in archive.files[:16]}
    except Exception as exc:
        sidecars["npz_error"] = f"{type(exc).__name__}:{exc}"
    return sidecars


def _shape_for_synthetic(shape: list[Any], max_dynamic_dim: int) -> tuple[int, ...]:
    concrete = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            concrete.append(dim)
        else:
            concrete.append(max_dynamic_dim)
    return tuple(concrete)


def _synthetic_shape_hint(input_name: str, raw_shape: list[Any], sidecars: dict[str, Any] | None) -> tuple[int, ...] | None:
    if not sidecars:
        return None
    json_sidecars = sidecars.get("json", {})
    network_info = json_sidecars.get("network_info.json", {})
    trt_info = json_sidecars.get("trt_info.json", {})
    audio_params = network_info.get("audio_params", {}) if isinstance(network_info, dict) else {}
    defaults = trt_info.get("defaults", {}) if isinstance(trt_info, dict) else {}
    if input_name == "input_values" and ("OPT_BUFFER_LEN" in defaults or "MIN_BUFFER_LEN" in defaults):
        return (1, int(defaults.get("OPT_BUFFER_LEN", defaults["MIN_BUFFER_LEN"])))
    if input_name in {"input", "window"} and "buffer_len" in audio_params:
        if len(raw_shape) == 3:
            return (1, 1, int(audio_params["buffer_len"]))
        return (1, int(audio_params["buffer_len"]))
    params = network_info.get("params", {}) if isinstance(network_info, dict) else {}
    if input_name == "emotion" and "implicit_emotion_len" in params:
        default_emotion = params.get("default_emotion", [])
        emotion_len = int(params["implicit_emotion_len"]) + len(default_emotion)
        if len(raw_shape) == 3:
            return (1, 1, emotion_len)
        return (1, emotion_len)
    return None


def audio2face_contract(model_path: str | Path, sidecars: dict[str, Any] | None = None) -> dict[str, Any]:
    root = Path(model_path)
    loaded_sidecars = sidecars or inspect_sidecars(root)
    network_info = loaded_sidecars.get("json", {}).get("network_info.json", {})
    params = network_info.get("params", {}) if isinstance(network_info, dict) else {}
    audio_params = network_info.get("audio_params", {}) if isinstance(network_info, dict) else {}
    model_id = network_info.get("id", {}) if isinstance(network_info, dict) else {}
    result_jaw_size = int(params.get("result_jaw_size", 0) or 0)
    result_eyes_size = int(params.get("result_eyes_size", 0) or 0)
    num_shapes_skin = int(params.get("num_shapes_skin", 0) or 0)
    num_shapes_tongue = int(params.get("num_shapes_tongue", 0) or 0)
    output_width = num_shapes_skin + num_shapes_tongue + result_jaw_size + result_eyes_size
    return {
        "contract": "onnx_audio2face_geometry_contract",
        "actor": model_id.get("actor"),
        "version": model_id.get("version"),
        "network_output": model_id.get("output"),
        "sample_rate": audio_params.get("samplerate"),
        "audio_buffer_len": audio_params.get("buffer_len"),
        "audio_buffer_offset": audio_params.get("buffer_ofs"),
        "explicit_emotions": params.get("explicit_emotions", []),
        "implicit_emotion_len": params.get("implicit_emotion_len"),
        "emotion_width": (params.get("implicit_emotion_len") or 0) + len(params.get("default_emotion", [])),
        "output_width": output_width or None,
        "output_partitions": {
            "skin_blendshapes": num_shapes_skin,
            "tongue_blendshapes": num_shapes_tongue,
            "jaw": result_jaw_size,
            "eyes": result_eyes_size,
        },
        "sidecar_files": sorted(path.name for path in root.glob("*.json")) + sorted(path.name for path in root.glob("*.npz")),
    }


def _synthetic_inputs(
    inputs: tuple[dict[str, Any], ...],
    max_dynamic_dim: int,
    sidecars: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import numpy as np

    feed: dict[str, Any] = {}
    for item in inputs:
        if item["type"] not in {"tensor(float)", "tensor(float16)", "tensor(double)"}:
            raise ONNXBridgeError(f"unsupported synthetic input type for {item['name']}: {item['type']}")
        dtype = np.float16 if item["type"] == "tensor(float16)" else np.float32
        shape = _synthetic_shape_hint(item["name"], item["shape"], sidecars) or _shape_for_synthetic(
            item["shape"], max_dynamic_dim
        )
        size = 1
        for dim in shape:
            size *= dim
        if size > 2_000_000:
            raise ONNXBridgeError(f"synthetic input too large for {item['name']}: shape={shape}")
        feed[item["name"]] = np.zeros(shape, dtype=dtype)
    return feed


def inspect_onnx_model(request: ONNXRequest) -> ONNXResult:
    try:
        import onnxruntime as ort

        model_path = resolve_model_path(request.model_id, request.model_root)
        onnx_path = find_onnx_path(model_path)
        providers = [request.provider]
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        inputs = tuple(_value_info(item) for item in session.get_inputs())
        outputs = tuple(_value_info(item) for item in session.get_outputs())
        synthetic_outputs: tuple[dict[str, Any], ...] = ()
        status = "ready"
        error = None
        sidecars = inspect_sidecars(model_path)
        if request.run_synthetic:
            try:
                feed = _synthetic_inputs(inputs, request.max_dynamic_dim, sidecars)
                raw_outputs = session.run(None, feed)
                synthetic_outputs = tuple(
                    {"name": outputs[index]["name"], "shape": list(value.shape), "dtype": str(value.dtype)}
                    for index, value in enumerate(raw_outputs)
                )
                status = "ok"
            except Exception as exc:
                status = "synthetic_blocked"
                error = f"{type(exc).__name__}:{exc}"
        return ONNXResult(
            status=status,
            model_id=request.model_id,
            model_path=str(model_path),
            onnx_path=str(onnx_path),
            providers=tuple(session.get_providers()),
            inputs=inputs,
            outputs=outputs,
            sidecars=sidecars,
            synthetic_outputs=synthetic_outputs,
            error=error,
        )
    except Exception as exc:
        return ONNXResult(status="failed", model_id=request.model_id, model_path=None, error=f"{type(exc).__name__}:{exc}")


def to_json(obj: Any) -> dict[str, Any]:
    return asdict(obj)
