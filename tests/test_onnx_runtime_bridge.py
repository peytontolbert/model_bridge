import sys
import types
from pathlib import Path

import pytest

from legacy_model_bridge.runtime.onnx_runtime import (
    ONNXBridgeError,
    ONNXRequest,
    audio2face_contract,
    _shape_for_synthetic,
    _synthetic_inputs,
    _synthetic_shape_hint,
    find_onnx_path,
    inspect_onnx_model,
    resolve_model_path,
)
from scripts.inspect_onnx_audio_face import DEFAULT_MODELS


def test_resolve_onnx_model_path(tmp_path: Path) -> None:
    model = tmp_path / "Audio2Emotion-v3.0"
    model.mkdir()

    assert resolve_model_path("Audio2Emotion-v3.0", tmp_path) == model


def test_find_onnx_path_prefers_network(tmp_path: Path) -> None:
    model = tmp_path / "m"
    model.mkdir()
    network = model / "network.onnx"
    other = model / "other.onnx"
    other.write_bytes(b"other")
    network.write_bytes(b"network")

    assert find_onnx_path(model) == network


def test_shape_for_synthetic_replaces_dynamic_dims() -> None:
    assert _shape_for_synthetic(["batch", 1, "seq"], 4) == (4, 1, 4)


def test_synthetic_input_rejects_large_arrays() -> None:
    with pytest.raises(ONNXBridgeError, match="too large"):
        _synthetic_inputs(({"name": "x", "type": "tensor(float)", "shape": [9999999]},), 4)


def test_default_audio_face_models_are_ranked_targets() -> None:
    assert DEFAULT_MODELS == ["Audio2Emotion-v3.0", "Audio2Face-3D-v2.3-Mark", "Audio2Face-3D-v3.0"]


def test_inspect_onnx_model_uses_session_metadata(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "Audio2Emotion-v3.0"
    model.mkdir()
    (model / "network.onnx").write_bytes(b"onnx")

    class Value:
        def __init__(self, name, type_, shape):
            self.name = name
            self.type = type_
            self.shape = shape

    class FakeSession:
        def __init__(self, path, providers):
            self.path = path
            self.providers = providers

        def get_inputs(self):
            return [Value("input_values", "tensor(float)", ["batch", "seq"])]

        def get_outputs(self):
            return [Value("output", "tensor(float)", ["batch", 2])]

        def get_providers(self):
            return ["CPUExecutionProvider"]

    fake_ort = types.SimpleNamespace(InferenceSession=FakeSession)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    result = inspect_onnx_model(ONNXRequest(model_id="Audio2Emotion-v3.0", model_root=str(tmp_path)))

    assert result.status == "ready"
    assert result.inputs[0]["name"] == "input_values"
    assert result.outputs[0]["name"] == "output"


def test_synthetic_shape_hint_uses_trt_opt_buffer() -> None:
    sidecars = {"json": {"trt_info.json": {"defaults": {"MIN_BUFFER_LEN": 5000, "OPT_BUFFER_LEN": 30000}}}}

    assert _synthetic_shape_hint("input_values", ["batch", "seq"], sidecars) == (1, 30000)


def test_synthetic_shape_hint_uses_audio_buffer_len() -> None:
    sidecars = {"json": {"network_info.json": {"audio_params": {"buffer_len": 8320}}}}

    assert _synthetic_shape_hint("input", ["batch", 1, 8320], sidecars) == (1, 1, 8320)

def test_synthetic_shape_hint_uses_audio2face_emotion_width() -> None:
    sidecars = {
        "json": {
            "network_info.json": {
                "params": {
                    "implicit_emotion_len": 16,
                    "default_emotion": [0.0] * 10,
                }
            }
        }
    }

    assert _synthetic_shape_hint("emotion", ["batch", 1, 26], sidecars) == (1, 1, 26)


def test_audio2face_contract_summarizes_sidecars(tmp_path: Path) -> None:
    sidecars = {
        "json": {
            "network_info.json": {
                "id": {"actor": "mark", "version": "2.3", "output": "geometry"},
                "audio_params": {"samplerate": 16000, "buffer_len": 8320, "buffer_ofs": 4160},
                "params": {
                    "default_emotion": [0.0] * 10,
                    "explicit_emotions": ["joy", "sadness"],
                    "implicit_emotion_len": 16,
                    "num_shapes_skin": 272,
                    "num_shapes_tongue": 10,
                    "result_jaw_size": 15,
                    "result_eyes_size": 4,
                },
            }
        }
    }

    contract = audio2face_contract(tmp_path, sidecars)

    assert contract["contract"] == "onnx_audio2face_geometry_contract"
    assert contract["actor"] == "mark"
    assert contract["emotion_width"] == 26
    assert contract["output_width"] == 301
    assert contract["output_partitions"]["skin_blendshapes"] == 272

