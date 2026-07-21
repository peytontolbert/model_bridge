import json
import sys
import types
from pathlib import Path

from legacy_model_bridge.runtime.audio_generation import (
    AudioGenerationRequest,
    infer_task,
    inspect_audio_generation,
    resolve_model_path,
)
from scripts.inspect_audio_generation import DEFAULT_MODELS


def _write_model(path: Path, config: dict) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_default_audio_generation_models_are_musicgen_family() -> None:
    assert DEFAULT_MODELS[:2] == ["musicgen-small", "musicgen-stereo-small"]
    assert "musicgen-stereo-melody-large" in DEFAULT_MODELS


def test_audio_generation_resolves_namespace_fallback(tmp_path: Path) -> None:
    _write_model(tmp_path / "musicgen-small", {"model_type": "musicgen"})

    assert resolve_model_path("facebook/musicgen-small", tmp_path) == tmp_path / "musicgen-small"


def test_audio_generation_task_inference() -> None:
    assert infer_task({"model_type": "musicgen"}) == "musicgen_text_to_audio"
    assert infer_task({"model_type": "musicgen_melody"}) == "musicgen_melody_text_to_audio"


def test_audio_generation_inspect_reads_contract(tmp_path: Path) -> None:
    model = tmp_path / "musicgen-small"
    _write_model(model, {"model_type": "musicgen", "architectures": ["MusicgenForConditionalGeneration"]})
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "preprocessor_config.json").write_text(json.dumps({"sampling_rate": 32000}), encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")

    result = inspect_audio_generation(AudioGenerationRequest(model_id="musicgen-small", model_root=str(tmp_path)))

    assert result.status == "ready"
    assert result.task == "musicgen_text_to_audio"
    assert result.sampling_rate == 32000
    assert result.weight_files == ("model.safetensors",)


def test_audio_generation_generate_uses_musicgen_contract(tmp_path: Path, monkeypatch) -> None:
    model_dir = tmp_path / "musicgen-small"
    _write_model(model_dir, {"model_type": "musicgen"})

    class FakeTensor:
        shape = (1, 1, 64)
        dtype = "float32"

        def to(self, device):
            return self

    class FakeProcessor:
        def __call__(self, text, padding=True, return_tensors=None):
            return {"input_ids": FakeTensor()}

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return None

        def generate(self, **kwargs):
            return FakeTensor()

    fake_transformers = types.SimpleNamespace(
        AutoProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeProcessor()),
        MusicgenForConditionalGeneration=types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeModel()),
        MusicgenMelodyForConditionalGeneration=types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeModel()),
    )
    fake_torch = types.SimpleNamespace(no_grad=lambda: _NoOpContext())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result = inspect_audio_generation(
        AudioGenerationRequest(model_id="musicgen-small", model_root=str(tmp_path), run_generate=True)
    )

    assert result.status == "ok"
    assert result.generated_audio_shape == (1, 1, 64)
    assert result.model_class == "FakeModel"


class _NoOpContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False



def test_audio_generation_generate_selects_melody_class(tmp_path: Path, monkeypatch) -> None:
    model_dir = tmp_path / "musicgen-stereo-melody-large"
    _write_model(model_dir, {"model_type": "musicgen_melody"})

    class FakeTensor:
        shape = (1, 2, 64)
        dtype = "float32"

        def to(self, device):
            return self

    class FakeProcessor:
        def __call__(self, text, padding=True, return_tensors=None):
            return {"input_ids": FakeTensor()}

    class FakeMelodyModel:
        def to(self, device):
            return self

        def eval(self):
            return None

        def generate(self, **kwargs):
            return FakeTensor()

    class FailingPlainModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise AssertionError("plain MusicGen class should not load melody configs")

    fake_transformers = types.SimpleNamespace(
        AutoProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeProcessor()),
        MusicgenForConditionalGeneration=FailingPlainModel,
        MusicgenMelodyForConditionalGeneration=types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeMelodyModel()),
    )
    fake_torch = types.SimpleNamespace(no_grad=lambda: _NoOpContext())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result = inspect_audio_generation(
        AudioGenerationRequest(model_id="musicgen-stereo-melody-large", model_root=str(tmp_path), run_generate=True)
    )

    assert result.status == "ok"
    assert result.task == "musicgen_melody_text_to_audio"
    assert result.model_class == "FakeMelodyModel"
