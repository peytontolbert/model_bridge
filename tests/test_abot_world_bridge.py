import json
from pathlib import Path

import pytest

from legacy_model_bridge.runtime.abot_world import ABotWorldBridgeError, ABotWorldRequest, verify_abot_world


def test_abot_world_metadata_validates_paths(tmp_path: Path) -> None:
    repo = tmp_path / "ABot-World"
    model = tmp_path / "acvlab--ABot-World-0-5B-LF"
    repo.mkdir()
    model.mkdir()

    with pytest.raises(ModuleNotFoundError):
        verify_abot_world(ABotWorldRequest(repo_path=str(repo), model_path=str(model), load_generator=False))


def test_abot_world_missing_repo_is_explicit(tmp_path: Path) -> None:
    model = tmp_path / "acvlab--ABot-World-0-5B-LF"
    model.mkdir()

    with pytest.raises(ABotWorldBridgeError, match="ABot repo not found"):
        verify_abot_world(ABotWorldRequest(repo_path=str(tmp_path / "missing"), model_path=str(model)))


def test_abot_world_report_schema_matches_verified_report() -> None:
    report = Path("reports/world-model-smokes/acvlab--ABot-World-0-5B-LF.generator_cuda_bf16.ai.cuda2.json")
    if not report.is_file():
        pytest.skip("ABot verification report is generated only on the GPU host")
    payload = json.loads(report.read_text(encoding="utf-8"))

    assert payload["status"] == "generator_cuda_bf16_loaded"
    assert payload["model_id"] == "acvlab--ABot-World-0-5B-LF"
    assert "abot_generator_direct_cuda_device_map_bf16" in payload["patches"]
