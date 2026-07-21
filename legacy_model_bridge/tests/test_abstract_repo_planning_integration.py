from pathlib import Path

from integrations.repository_library_abstract_repo_planning.adapter import detect


def test_abstract_repo_planning_detect_accepts_minimal_artifacts(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"")

    assert detect(tmp_path) is True


def test_abstract_repo_planning_detect_rejects_missing_config(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"")

    assert detect(tmp_path) is False
