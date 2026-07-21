import json
import subprocess
import sys
import types
from pathlib import Path

from legacy_model_bridge.runtime.classic_transformers import (
    ClassicTransformersRequest,
    infer_task,
    inspect_classic_transformers,
    read_model_config,
    resolve_hf_cache_snapshot,
    resolve_model_path,
)
from scripts.inspect_classic_transformers import DEFAULT_MODELS


def _write_model(path: Path, config: dict) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_default_classic_transformers_models_are_ranked_targets() -> None:
    assert DEFAULT_MODELS == [
        "facebook/bart-large-cnn",
        "gpt2",
        "distilgpt2",
        "sentence-transformers/all-MiniLM-L6-v2",
        "intfloat/e5-base-v2",
        "hubert-base-ls960",
        "hubert-large-ll60k",
        "hubert-xlarge-ll60k",
        "dinov2-base",
        "dinov2-large",
        "webssl-dino300m-full2b-224",
    ]


def test_resolve_model_path_supports_namespace_fallback(tmp_path: Path) -> None:
    _write_model(tmp_path / "bart-large-cnn", {"model_type": "bart"})

    assert resolve_model_path("facebook/bart-large-cnn", tmp_path) == tmp_path / "bart-large-cnn"


def test_resolve_hf_cache_snapshot_supports_model_ids(tmp_path: Path) -> None:
    older = tmp_path / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / "older"
    newer = tmp_path / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / "newer"
    _write_model(older, {"model_type": "bert"})
    _write_model(newer, {"model_type": "bert"})
    newer.touch()

    assert resolve_hf_cache_snapshot("sentence-transformers/all-MiniLM-L6-v2", tmp_path) == newer


def test_infer_classic_transformers_tasks() -> None:
    assert infer_task({"model_type": "bart"}) == "seq2seq_generation"
    assert infer_task({"model_type": "gpt2"}) == "causal_lm_generation"
    assert infer_task({"architectures": ["GPT2LMHeadModel"]}) == "causal_lm_generation"
    assert infer_task({"architectures": ["BertForSequenceClassification"]}) == "sequence_classification"
    assert infer_task({"architectures": ["BertForMaskedLM"]}) == "masked_lm"
    assert infer_task({"model_type": "bert"}) == "text_encoder"
    assert infer_task({"architectures": ["HubertModel"]}) == "audio_encoder"
    assert infer_task({"model_type": "dinov2"}) == "vision_encoder"


def test_inspect_classic_transformers_reads_artifact_contract(tmp_path: Path) -> None:
    model = tmp_path / "dinov2-base"
    _write_model(
        model,
        {
            "model_type": "dinov2",
            "architectures": ["Dinov2Model"],
            "transformers_version": "4.31.0.dev0",
        },
    )
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    result = inspect_classic_transformers(ClassicTransformersRequest(model_id="dinov2-base", model_root=str(tmp_path)))

    assert result.status == "ready"
    assert result.task == "vision_encoder"
    assert result.weight_files == ("model.safetensors",)
    assert result.processor_files == ("preprocessor_config.json",)


def test_synthetic_seq2seq_uses_auto_contract(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "bart"
    _write_model(model, {"model_type": "bart", "architectures": ["BartForConditionalGeneration"]})

    class FakeTensor:
        shape = (1, 2)

        def to(self, device):
            return self

        def __getitem__(self, index):
            return self

    class FakeTokenizer:
        def __call__(self, text, return_tensors=None):
            return {"input_ids": FakeTensor()}

        def decode(self, ids, skip_special_tokens=True):
            return "summary"

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return None

        def generate(self, **kwargs):
            return FakeTensor()

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeTokenizer()),
        AutoModelForSeq2SeqLM=types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeModel()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    result = inspect_classic_transformers(
        ClassicTransformersRequest(model_id="bart", model_root=str(tmp_path), run_synthetic=True)
    )

    assert result.status == "ok"
    assert result.tokenizer_class == "FakeTokenizer"
    assert result.model_class == "FakeModel"
    assert result.synthetic_outputs == {
        "generated_text": "summary",
        "prompt_token_count": 2,
        "generated_token_count": 2,
    }
    assert result.runtime is not None
    assert result.runtime["requested_device"] == "cpu"



def test_synthetic_causal_lm_uses_auto_contract(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "gpt2"
    _write_model(model, {"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]})

    class FakeTensor:
        shape = (1, 3)

        def to(self, device):
            return self

        def __getitem__(self, index):
            return self

    class FakeOutputIds(FakeTensor):
        shape = (1, 7)

    class FakeTokenizer:
        pad_token_id = None
        eos_token_id = 50256

        def __call__(self, text, return_tensors=None):
            return {"input_ids": FakeTensor()}

        def decode(self, ids, skip_special_tokens=True):
            return "Legacy model bridge smoke input. output"

    class NoOpContext:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return None

        def generate(self, **kwargs):
            assert kwargs["pad_token_id"] == 50256
            assert kwargs["do_sample"] is False
            return FakeOutputIds()

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeTokenizer()),
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeModel()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(no_grad=lambda: NoOpContext(), cuda=FakeCuda))

    result = inspect_classic_transformers(
        ClassicTransformersRequest(model_id="gpt2", model_root=str(tmp_path), run_synthetic=True)
    )

    assert result.status == "ok"
    assert result.task == "causal_lm_generation"
    assert result.tokenizer_class == "FakeTokenizer"
    assert result.model_class == "FakeModel"
    assert result.synthetic_outputs == {
        "generated_text": "Legacy model bridge smoke input. output",
        "prompt_token_count": 3,
        "generated_token_count": 4,
        "total_token_count": 7,
    }


def test_cli_classic_transformers_inspect(tmp_path: Path) -> None:
    _write_model(tmp_path / "hubert-base-ls960", {"model_type": "hubert"})

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "legacy_model_bridge.cli",
            "classic-transformers",
            "inspect",
            "--model-id",
            "hubert-base-ls960",
            "--model-root",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["status"] == "ready"
    assert payload["task"] == "audio_encoder"



def test_cast_floating_batch_preserves_integer_tensors(monkeypatch) -> None:
    from legacy_model_bridge.runtime.classic_transformers import _cast_floating_batch

    class FakeTensor:
        def __init__(self, floating: bool):
            self.floating = floating
            self.cast_dtype = None

        def is_floating_point(self):
            return self.floating

        def to(self, dtype=None):
            self.cast_dtype = dtype
            return self

    floating = FakeTensor(True)
    integer = FakeTensor(False)
    result = _cast_floating_batch({"pixel_values": floating, "input_ids": integer}, "float16")

    assert result["pixel_values"].cast_dtype == "float16"
    assert result["input_ids"].cast_dtype is None
