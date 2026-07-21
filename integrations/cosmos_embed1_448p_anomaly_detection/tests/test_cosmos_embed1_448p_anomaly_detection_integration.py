import json
from pathlib import Path

from integrations.cosmos_embed1_448p_anomaly_detection.adapter import detect


def test_detect_accepts_cosmos_embed1_config(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "cosmos_embed1"}))

    assert detect(tmp_path) is True


def test_detect_accepts_named_anomaly_directory(tmp_path: Path) -> None:
    model = tmp_path / "Cosmos-Embed1-448p-anomaly-detection"
    model.mkdir()

    assert detect(model) is True


def test_detect_rejects_other_model_type(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "bert"}))

    assert detect(tmp_path) is False
