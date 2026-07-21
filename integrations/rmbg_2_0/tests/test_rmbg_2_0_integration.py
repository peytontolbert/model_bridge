from pathlib import Path

from integrations.rmbg_2_0.adapter import detect


def test_detect_accepts_minimal_rmbg_artifacts(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"")

    assert detect(tmp_path) is True


def test_detect_rejects_missing_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")

    assert detect(tmp_path) is False
