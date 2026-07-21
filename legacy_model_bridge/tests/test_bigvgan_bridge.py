import json
from pathlib import Path

from legacy_model_bridge.runtime.bigvgan import BigVGANRequest, inspect_bigvgan, read_bigvgan_config, resolve_model_path


def _write_model(path: Path, config: dict) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_bigvgan_resolves_model_path(tmp_path: Path) -> None:
    _write_model(tmp_path / "bigvgan_v2_44khz_128band_512x", {"num_mels": 128})

    assert resolve_model_path("bigvgan_v2_44khz_128band_512x", tmp_path) == tmp_path / "bigvgan_v2_44khz_128band_512x"


def test_bigvgan_reads_config(tmp_path: Path) -> None:
    model = tmp_path / "bigvgan"
    _write_model(model, {"num_mels": 128, "sampling_rate": 44100})

    assert read_bigvgan_config(model)["sampling_rate"] == 44100


def test_bigvgan_inspect_records_vocoder_contract(tmp_path: Path) -> None:
    model = tmp_path / "bigvgan"
    _write_model(
        model,
        {
            "num_mels": 128,
            "sampling_rate": 44100,
            "hop_size": 512,
            "upsample_rates": [8, 4, 2, 2, 2, 2],
        },
    )
    (model / "bigvgan_generator.pt").write_bytes(b"stub")

    result = inspect_bigvgan(BigVGANRequest(model_id="bigvgan", model_root=str(tmp_path)))

    assert result.status == "ready"
    assert result.artifact_contract == "bigvgan_mel_vocoder_contract"
    assert result.checkpoint_files == ("bigvgan_generator.pt",)
    assert result.upsample_rates == (8, 4, 2, 2, 2, 2)
