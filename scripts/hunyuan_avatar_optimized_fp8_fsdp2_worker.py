from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy_model_bridge.runtime.hunyuan_avatar import install_avatar_fp8_fsdp2_state_dict_key_patch
from legacy_model_bridge.runtime.hunyuan_avatar import install_distributed_output_none_patch
from legacy_model_bridge.runtime.hunyuan_avatar import install_llava_llama_model_property_patch
from legacy_model_bridge.runtime.hunyuan_avatar import install_torch_fsdp2_mesh_layout_pickle_patch

DEFAULT_AVATAR_ROOT = Path("/data/clone/hunyuanvideo-avatar")
DEFAULT_TRANSFORMER10_ROOT = Path("/data/transformer_10")
DEFAULT_MODEL_BASE = Path("/arxiv/models/HunyuanVideo-Avatar")
DEFAULT_FP8_SHARD_DIR = DEFAULT_TRANSFORMER10_ROOT / "checkpoints" / "hunyuan_avatar_fp8_fsdp2"
DEFAULT_BF16_SHARD_DIR = DEFAULT_TRANSFORMER10_ROOT / "checkpoints" / "hunyuan_avatar_bf16_fsdp2"
DEFAULT_SHARD_DIR = DEFAULT_FP8_SHARD_DIR
DEFAULT_FP8_CKPT = DEFAULT_MODEL_BASE / "ckpts" / "hunyuan-video-t2v-720p" / "transformers" / "mp_rank_00_model_states_fp8.pt"
DEFAULT_BF16_CKPT = DEFAULT_MODEL_BASE / "ckpts" / "hunyuan-video-t2v-720p" / "transformers" / "mp_rank_00_model_states.pt"
DEFAULT_CKPT = DEFAULT_FP8_CKPT
DEFAULT_INPUT = DEFAULT_AVATAR_ROOT / "input" / "peyton_avatar_test.csv"
DEFAULT_SAVE_PATH = Path("/data/tmp/lmb_hunyuan_avatar_fp8_fsdp2")


def _prepend_path(path: Path) -> None:
    value = str(path.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run optimized Hunyuan Avatar FP8/FSDP2 through bridge patches.")
    parser.add_argument("--avatar-root", type=Path, default=DEFAULT_AVATAR_ROOT)
    parser.add_argument("--transformer10-root", type=Path, default=DEFAULT_TRANSFORMER10_ROOT)
    parser.add_argument("--model-base", type=Path, default=DEFAULT_MODEL_BASE)
    parser.add_argument("--shard-dir", type=Path, default=None)
    parser.add_argument("--precision-mode", choices=["auto", "fp8", "bf16"], default="auto")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--save-path", type=Path, default=DEFAULT_SAVE_PATH)
    parser.add_argument("--infer-steps", type=int, default=4)
    parser.add_argument("--sample-n-frames", type=int, default=65)
    parser.add_argument("--reference-frames", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1025)
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--flow-shift-eval-video", type=float, default=5.0)
    parser.add_argument("--cpu-offload", choices=["0", "1"], default="1")
    args = parser.parse_args(argv)

    _prepend_path(args.avatar_root)
    _prepend_path(args.transformer10_root)
    _prepend_path(Path(__file__).resolve().parents[1])
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["CPU_OFFLOAD"] = str(args.cpu_offload)
    os.environ.setdefault("HUNYUAN_AVATAR_ROOT", str(args.avatar_root))
    os.environ.setdefault("MODEL_BASE", str(args.model_base))
    os.environ.setdefault("HUNYUAN_AVATAR_LLAVA_INT8_GPU", "1")
    os.environ.setdefault("HUNYUAN_AVATAR_LLAVA_GPU_DEVICE", "cuda:0")
    reference_frames = args.reference_frames if args.reference_frames is not None else max(args.sample_n_frames, 65)
    os.environ.setdefault("HUNYUAN_AVATAR_REFERENCE_FRAMES", str(reference_frames))
    if args.sample_n_frames <= 17:
        os.environ.setdefault("HUNYUAN_AVATAR_VAE_LATENT_CACHE", str(args.avatar_root / "cache" / "vae_latents_512_w17"))
    else:
        os.environ.setdefault("HUNYUAN_AVATAR_VAE_LATENT_CACHE", str(args.avatar_root / "cache" / "vae_latents_512_w17"))

    install_torch_fsdp2_mesh_layout_pickle_patch()
    install_avatar_fp8_fsdp2_state_dict_key_patch()
    install_llava_llama_model_property_patch()
    install_distributed_output_none_patch()

    precision_mode = args.precision_mode
    if precision_mode == "auto":
        import torch
        capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
        precision_mode = "fp8" if capability[0] >= 9 else "bf16"
    if args.shard_dir is None:
        args.shard_dir = DEFAULT_FP8_SHARD_DIR if precision_mode == "fp8" else DEFAULT_BF16_SHARD_DIR
    if args.ckpt == DEFAULT_CKPT and precision_mode == "bf16":
        args.ckpt = DEFAULT_BF16_CKPT

    if precision_mode == "fp8":
        sampler = args.transformer10_root / "scripts" / "sample_hunyuan_avatar_fp8_fsdp2.py"
    else:
        sampler = REPO_ROOT / "scripts" / "hunyuan_avatar_bf16_fsdp_worker.py"
    sys.argv = [
        str(sampler),
        "--shard-dir", str(args.shard_dir),
        "--ckpt", str(args.ckpt),
        "--cpu-offload",
        "--input", str(args.input),
        "--save-path", str(args.save_path),
        "--infer-steps", str(args.infer_steps),
        "--flow-shift-eval-video", str(args.flow_shift_eval_video),
        "--seed", str(args.seed),
        "--image-size", str(args.image_size),
        "--sample-n-frames", str(args.sample_n_frames),
        "--cfg-scale", str(args.cfg_scale),
        "--use-deepcache", "1",
        "--infer-min",
    ]
    if precision_mode == "fp8":
        sys.argv.insert(3, "--use-fp8")
    runpy.run_path(str(sampler), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
