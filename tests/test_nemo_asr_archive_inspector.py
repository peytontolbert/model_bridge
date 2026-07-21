import json
from pathlib import Path

from scripts.inspect_nemo_asr_archives import _load_default_models, inspect_models


def test_load_default_nemo_archive_targets() -> None:
    models = _load_default_models()

    assert "parakeet-tdt_ctc-110m" in models
    assert "nemotron-speech-streaming-en-0.6b" in models


def test_inspect_models_dry_run_records_each_target(tmp_path: Path) -> None:
    report = inspect_models(
        ["parakeet-tdt_ctc-110m", "nemotron-speech-streaming-en-0.6b"],
        output_dir=tmp_path / "out",
        timeout_sec=1,
        dry_run=True,
    )

    assert report["dry_run"] is True
    assert [item["model_id"] for item in report["results"]] == [
        "parakeet-tdt_ctc-110m",
        "nemotron-speech-streaming-en-0.6b",
    ]
    assert all(item["status"] == "dry_run" for item in report["results"])
    assert all(item["command"] for item in report["results"])
