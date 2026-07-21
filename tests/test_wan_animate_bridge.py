import sys
import types
from pathlib import Path

from legacy_model_bridge.runtime.wan_animate import (
    WanAnimatePaths,
    select_lightx2v_attention_backend,
    wan_animate_status,
)


def test_selects_sage_attn2_for_sageattention2_symbols(monkeypatch) -> None:
    fake_status = types.SimpleNamespace(available=True, has_fp16_triton=True, has_fp16_cuda=False)

    assert select_lightx2v_attention_backend(fake_status) == "sage_attn2"


def test_falls_back_to_sdpa_without_sageattention2_symbols() -> None:
    fake_status = types.SimpleNamespace(available=False, has_fp16_triton=False, has_fp16_cuda=False)

    assert select_lightx2v_attention_backend(fake_status) == "sdpa"


def test_wan_animate_status_validates_cached_and_int8_artifacts(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "Wan-AI--Wan2.2-Animate-14B"
    wan_source = tmp_path / "Wan2.2"
    cache_root = tmp_path / "cache-smoke"
    int8_root = tmp_path / "int8"
    model.mkdir()
    wan_source.mkdir()
    for relative in (
        "Wan2.1_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "diffusion_pytorch_model-00001-of-00004.safetensors",
        "diffusion_pytorch_model-00002-of-00004.safetensors",
        "diffusion_pytorch_model-00003-of-00004.safetensors",
        "diffusion_pytorch_model-00004-of-00004.safetensors",
        "xlm-roberta-large/config.json",
        "xlm-roberta-large/model.safetensors",
        "relighting_lora/adapter_model.safetensors",
        "relighting_lora.ckpt",
        "process_checkpoint/det/yolov10m.onnx",
        "process_checkpoint/pose2d/vitpose_h_wholebody.onnx/end2end.onnx",
        "process_checkpoint/sam2/sam2_hiera_base_plus.pt",
    ):
        path = model / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    (model / "diffusion_pytorch_model.safetensors.index.json").write_text(
        '{"weight_map":{"a":"diffusion_pytorch_model-00001-of-00004.safetensors","b":"diffusion_pytorch_model-00002-of-00004.safetensors","c":"diffusion_pytorch_model-00003-of-00004.safetensors","d":"diffusion_pytorch_model-00004-of-00004.safetensors"}}',
        encoding="utf-8",
    )
    (cache_root / "input33" / ".wan_vae_control_latents").mkdir(parents=True)
    (cache_root / "input33" / ".t5_context_rank0.pt").write_bytes(b"x")
    (cache_root / "input33" / ".wan_vae_control_latents" / "control.pt").write_bytes(b"x")
    (cache_root / "cache" / "window_0000").mkdir(parents=True)
    (cache_root / "cache" / "window_0000" / "manifest.json").write_text('{"steps":[{"step_index":0}]}', encoding="utf-8")
    (cache_root / "output").mkdir(parents=True)
    (cache_root / "output" / "wan_animate_int8_33f_4step_teacher_cache.mp4").write_bytes(b"x")
    (int8_root / "blocks").mkdir(parents=True)
    (int8_root / "blocks" / "block_00.safetensors").write_bytes(b"x")
    (int8_root / "manifest.json").write_text(
        '{"format":"modelstack.wan-animate.int8.blocks.v1","num_blocks":1,"quantized_modules":["blocks.0.self_attn.q"],"storage_formats":["safetensors"]}',
        encoding="utf-8",
    )

    fake_module = types.SimpleNamespace(
        sageattn_qk_int8_pv_fp16_triton=object(),
        __version__="2.2.0",
    )
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr("importlib.metadata.version", lambda name: "1.0")
    monkeypatch.setitem(sys.modules, "sageattention", fake_module)
    status = wan_animate_status(
        WanAnimatePaths(
            model_path=model,
            wan_source=wan_source,
            transformer10_root=tmp_path,
            int8_artifact_dir=int8_root,
            cache_smoke_root=cache_root,
        )
    )

    assert status.runnable is True
    assert status.status == "verified_wan_animate_cached_int8_bridge_ai"
    assert status.selected_lightx2v_attention_backend == "sage_attn2"
    assert status.int8_safetensor_block_count == 1
