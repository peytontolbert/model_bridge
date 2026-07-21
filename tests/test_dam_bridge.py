from pathlib import Path

from legacy_model_bridge.runtime.dam import dam_status, probe_dam_components
from scripts.smoke_dam_description import _synthetic_inputs


def _write_dam_tree(root: Path) -> None:
    for rel in [
        "config.json",
        "vision_tower/config.json",
        "vision_tower/model.safetensors",
        "mm_projector/config.json",
        "mm_projector/model.safetensors",
        "context_provider/config.json",
        "context_provider/model.safetensors",
        "llm/config.json",
        "llm/model.safetensors.index.json",
        "llm/tokenizer.model",
        "llm/model-00001-of-00001.safetensors",
    ]:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")


def test_dam_status_reports_artifact_contract(tmp_path: Path) -> None:
    model = tmp_path / "DAM-3B"
    runtime = tmp_path / "runtime"
    _write_dam_tree(model)
    runtime.mkdir()
    (runtime / "llava_llama.py").write_text("", encoding="utf-8")

    status = dam_status(model, runtime_source=runtime)

    assert status.status == "candidate_dam_lazy_submodule_bridge"
    assert status.missing_artifacts == ()
    assert "llm/model index" in status.present_artifacts
    assert status.runtime_source == str(runtime)


def test_dam_probe_reports_missing_runtime_import(tmp_path: Path) -> None:
    model = tmp_path / "DAM-3B"
    runtime = tmp_path / "runtime"
    _write_dam_tree(model)
    runtime.mkdir()
    (runtime / "llava_llama.py").write_text("raise RuntimeError('bad runtime')\n", encoding="utf-8")

    result = probe_dam_components(model, runtime_source=runtime, load_components=False)

    assert result.imports["llava_llama"] is False
    assert "RuntimeError:bad runtime" in result.errors["llava_llama"]

def test_dam_synthetic_description_inputs(tmp_path: Path) -> None:
    image_path, mask_path = _synthetic_inputs(tmp_path)

    assert image_path.is_file()
    assert mask_path.is_file()
    assert image_path.name == "dam_synthetic_image.png"
    assert mask_path.name == "dam_synthetic_mask.png"
