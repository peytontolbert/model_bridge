from pathlib import Path

from integrations.rmbg_2_0.adapter import detect


def test_rmbg_detect_accepts_minimal_artifacts(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "pytorch_model.bin").write_bytes(b"")

    assert detect(tmp_path) is True


def test_rmbg_detect_rejects_missing_config(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"")

    assert detect(tmp_path) is False
