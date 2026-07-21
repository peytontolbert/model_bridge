import json
from pathlib import Path

from legacy_model_bridge.runtime.audiogen import AudioGenRequest, inspect_audiogen, resolve_model_path


def test_audiogen_resolves_model_path(tmp_path: Path) -> None:
    model = tmp_path / "audiogen-medium"
    model.mkdir()

    assert resolve_model_path("facebook/audiogen-medium", tmp_path) == model


def test_audiogen_inspect_reports_checkpoint_pair(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "audiogen-medium"
    model.mkdir()
    (model / "state_dict.bin").write_bytes(b"lm")
    (model / "compression_state_dict.bin").write_bytes(b"codec")

    monkeypatch.setattr("legacy_model_bridge.runtime.audiogen._dependency_status", lambda: {
        "audiocraft": False,
        "encodec": True,
        "torch": True,
        "torchaudio": True,
    })

    result = inspect_audiogen(AudioGenRequest(model_id="audiogen-medium", model_root=str(tmp_path)))

    assert result.status == "blocked_missing_audiocraft_adapter"
    assert result.checkpoint_files == ("state_dict.bin", "compression_state_dict.bin")
    assert result.checkpoint_bytes == {"state_dict.bin": 2, "compression_state_dict.bin": 5}
    assert "audiogen_audiocraft_solver_checkpoint_inspector" in result.compatibility_patches


def test_audiogen_cfg_summary_and_state_examples() -> None:
    from legacy_model_bridge.runtime.audiogen import _cfg_summary, _state_prefix_counts

    cfg = """
  sample_rate: 32000
sample_rate: 16000
channels: 1
lm_model: transformer_lm
compression_model: encodec
card: 2048
n_q: 4
codebooks_pattern:
transformer_lm:
"""

    summary = _cfg_summary(cfg)

    assert summary["sample_rate"] == "16000"
    assert summary["channels"] == "1"
    assert summary["has_transformer_lm"] is True
    assert summary["has_encodec"] is True
    assert summary["has_delay_pattern"] is True
    assert _state_prefix_counts({"transformer.layers.0.weight": object(), "emb.0.weight": object()}) == {
        "emb": 1,
        "transformer": 1,
    }


def test_audiogen_source_probe_reports_missing_path(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "audiogen-medium"
    model.mkdir()
    (model / "state_dict.bin").write_bytes(b"lm")
    (model / "compression_state_dict.bin").write_bytes(b"codec")

    monkeypatch.setattr("legacy_model_bridge.runtime.audiogen._dependency_status", lambda: {
        "audiocraft": False,
        "encodec": True,
        "torch": True,
        "torchaudio": True,
    })

    result = inspect_audiogen(AudioGenRequest(
        model_id="audiogen-medium",
        model_root=str(tmp_path),
        audiocraft_source_path=str(tmp_path / "missing-audiocraft"),
        probe_source_import=True,
    ))

    assert result.status == "blocked_missing_audiocraft_adapter"
    assert result.source_probe == {
        "source_path": str(tmp_path / "missing-audiocraft"),
        "exists": False,
        "importable": False,
        "error": "source_path_missing",
    }


def test_audiogen_source_probe_reports_import_error(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "audiogen-medium"
    model.mkdir()
    (model / "state_dict.bin").write_bytes(b"lm")
    (model / "compression_state_dict.bin").write_bytes(b"codec")
    source = tmp_path / "src"
    pkg = source / "audiocraft"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("raise RuntimeError('old dependency missing')\n", encoding="utf-8")

    monkeypatch.setattr("legacy_model_bridge.runtime.audiogen._dependency_status", lambda: {
        "audiocraft": False,
        "encodec": True,
        "torch": True,
        "torchaudio": True,
    })

    result = inspect_audiogen(AudioGenRequest(
        model_id="audiogen-medium",
        model_root=str(tmp_path),
        audiocraft_source_path=str(source),
        probe_source_import=True,
    ))

    assert result.status == "blocked_missing_audiocraft_adapter"
    assert result.source_probe is not None
    assert result.source_probe["importable"] is False
    assert result.source_probe["error"] == "RuntimeError:old dependency missing"


def test_audiogen_attention_backend_plan_disables_memory_efficient() -> None:
    from legacy_model_bridge.runtime.audiogen import audiogen_attention_backend_plan, disable_audiogen_memory_efficient_attention

    cfg = {"transformer_lm": {"memory_efficient": True, "custom": False}}

    assert disable_audiogen_memory_efficient_attention(cfg) is True
    assert cfg["transformer_lm"]["memory_efficient"] is False
    plan = audiogen_attention_backend_plan()
    assert plan["checkpoint_weight_changes_required"] is False
    assert plan["preferred_runtime_backend"] == "torch_nn_multihead_attention"


def test_xformers_import_shim_installs_import_only_modules(monkeypatch) -> None:
    import importlib.util
    import sys

    from legacy_model_bridge.runtime.audiogen import install_xformers_import_shim

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "xformers" else importlib.util.find_spec(name))
    monkeypatch.delitem(sys.modules, "xformers", raising=False)
    monkeypatch.delitem(sys.modules, "xformers.ops", raising=False)

    assert install_xformers_import_shim() is True
    from xformers import ops

    assert hasattr(ops, "LowerTriangularMask")
    try:
        ops.memory_efficient_attention(None)
    except RuntimeError as exc:
        assert "should disable memory_efficient attention" in str(exc)
    else:
        raise AssertionError("xformers attention shim should raise when called")
