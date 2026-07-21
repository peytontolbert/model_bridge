from legacy_model_bridge.runtime.hunyuan_avatar import IMAGE_PLACEHOLDER
from legacy_model_bridge.runtime.hunyuan_avatar import build_hunyuan_avatar_llava_prompt
from legacy_model_bridge.runtime.hunyuan_avatar import expand_llava_image_placeholders
from legacy_model_bridge.runtime.hunyuan_avatar import expected_llava_image_tokens


def test_expected_llava_image_tokens_from_vision_config() -> None:
    config = {"vision_config": {"image_size": 336, "patch_size": 14}}

    assert expected_llava_image_tokens(config) == 576


def test_expected_llava_image_tokens_prefers_configured_seq_length() -> None:
    config = {"image_seq_length": 128, "vision_config": {"image_size": 336, "patch_size": 14}}

    assert expected_llava_image_tokens(config) == 128


def test_expand_llava_image_placeholders_replaces_single_marker() -> None:
    prompt = build_hunyuan_avatar_llava_prompt("hello", name="avatar")

    expanded, changed = expand_llava_image_placeholders(prompt, 4)

    assert changed is True
    assert expanded.count(IMAGE_PLACEHOLDER) == 4
    assert "The avatar looks like" in expanded


def test_expand_llava_image_placeholders_leaves_already_expanded_prompt() -> None:
    prompt = "hello\n" + IMAGE_PLACEHOLDER * 4

    expanded, changed = expand_llava_image_placeholders(prompt, 4)

    assert changed is False
    assert expanded == prompt

def test_distributed_output_none_patch_returns_none_for_rank_output(monkeypatch) -> None:
    import sys
    import types
    class FakeOutput:
        def __init__(self, videos=None) -> None:
            self.videos = videos

        def __getitem__(self, key):
            if key == 0 and self.videos is not None:
                return self.videos
            raise IndexError(key)

    module = types.SimpleNamespace(HunyuanVideoPipelineOutput=FakeOutput)
    monkeypatch.setitem(sys.modules, "hymm_sp", types.ModuleType("hymm_sp"))
    monkeypatch.setitem(sys.modules, "hymm_sp.diffusion", types.ModuleType("hymm_sp.diffusion"))
    monkeypatch.setitem(sys.modules, "hymm_sp.diffusion.pipelines", types.ModuleType("hymm_sp.diffusion.pipelines"))
    monkeypatch.setitem(sys.modules, "hymm_sp.diffusion.pipelines.pipeline_hunyuan_video_audio", module)

    from legacy_model_bridge.runtime.hunyuan_avatar import install_distributed_output_none_patch

    result = install_distributed_output_none_patch()

    assert result["patched"] is True
    assert FakeOutput(videos=None)[0] is None
    assert FakeOutput(videos="video")[0] == "video"

def test_hunyuan_avatar_worker_renders_optimized_fp8_fsdp2_command(tmp_path) -> None:
    from scripts.hunyuan_avatar_worker import _optimized_command

    cmd = _optimized_command({
        "output_dir": str(tmp_path),
        "cuda_visible_devices": "0,1",
        "infer_steps": 4,
        "sample_n_frames": 65,
        "image_size": 512,
    })

    assert cmd[:3] == ["torchrun", "--standalone", "--nproc_per_node=2"]
    assert "scripts/hunyuan_avatar_optimized_fp8_fsdp2_worker.py" in cmd[4]
    assert str(tmp_path) in cmd



def test_hunyuan_avatar_worker_prewarms_minimum_reference_frames(tmp_path) -> None:
    from scripts.hunyuan_avatar_worker import _optimized_command, _vae_cache_command

    request = {
        "input": str(tmp_path / "input.csv"),
        "vae_latent_cache": str(tmp_path / "cache"),
        "sample_n_frames": 33,
        "image_size": 384,
    }

    cache_cmd = _vae_cache_command(request)
    optimized_cmd = _optimized_command(request)

    assert cache_cmd[cache_cmd.index("--frames") + 1] == "65"
    assert optimized_cmd[optimized_cmd.index("--reference-frames") + 1] == "65"


def test_hunyuan_avatar_worker_allows_explicit_reference_frames(tmp_path) -> None:
    from scripts.hunyuan_avatar_worker import _vae_cache_command

    cache_cmd = _vae_cache_command({
        "input": str(tmp_path / "input.csv"),
        "vae_latent_cache": str(tmp_path / "cache"),
        "sample_n_frames": 33,
        "reference_frames": 129,
    })

    assert cache_cmd[cache_cmd.index("--frames") + 1] == "129"

def test_hunyuan_avatar_optimized_wrapper_uses_fp8_fsdp2_flags(monkeypatch, tmp_path) -> None:
    import runpy

    from scripts import hunyuan_avatar_optimized_fp8_fsdp2_worker as worker

    captured = {}

    monkeypatch.setattr(worker, "install_torch_fsdp2_mesh_layout_pickle_patch", lambda: {"patched": True})
    monkeypatch.setattr(worker, "install_avatar_fp8_fsdp2_state_dict_key_patch", lambda: {"patched": True})
    monkeypatch.setattr(worker, "install_llava_llama_model_property_patch", lambda: {"patched": True})
    monkeypatch.setattr(worker, "install_distributed_output_none_patch", lambda: {"patched": True})

    def fake_run_path(path, run_name):
        captured["path"] = path
        captured["run_name"] = run_name
        captured["argv"] = list(worker.sys.argv)
        return {}

    monkeypatch.setattr(runpy, "run_path", fake_run_path)

    assert worker.main([
        "--avatar-root", str(tmp_path),
        "--transformer10-root", str(tmp_path),
        "--model-base", str(tmp_path),
        "--precision-mode", "fp8",
        "--shard-dir", str(tmp_path / "shards"),
        "--ckpt", str(tmp_path / "model.pt"),
        "--input", str(tmp_path / "input.csv"),
        "--save-path", str(tmp_path / "out"),
    ]) == 0

    argv = captured["argv"]
    assert "--use-fp8" in argv
    assert "--cpu-offload" in argv
    assert "--use-deepcache" in argv
    assert "sample_hunyuan_avatar_fp8_fsdp2.py" in captured["path"]


def test_torch_fsdp2_mesh_layout_pickle_patch_aliases_flat_layout(monkeypatch) -> None:
    import sys
    import types

    class MeshLayout:
        pass

    module = types.SimpleNamespace(_MeshLayout=MeshLayout)
    monkeypatch.setitem(sys.modules, "torch.distributed._mesh_layout", module)

    from legacy_model_bridge.runtime.hunyuan_avatar import install_torch_fsdp2_mesh_layout_pickle_patch

    result = install_torch_fsdp2_mesh_layout_pickle_patch()

    assert result["patched"] is True
    assert result["already_patched"] is False
    restored = module._FlatLayout.__new__(module._FlatLayout)
    restored.__setstate__({"sizes": (2,), "strides": (1,)})
    assert restored.shape == (2,)
    assert restored.stride == (1,)


def test_remap_avatar_fp8_fsdp2_state_keys_only_maps_fp8_linears() -> None:
    from collections import OrderedDict

    from legacy_model_bridge.runtime.hunyuan_avatar import remap_avatar_fp8_fsdp2_state_keys

    state = OrderedDict([
        ("block.linear.weight", "fp8_weight"),
        ("block.linear.bias", "bias"),
        ("block.linear.fp8_scale", "scale"),
        ("block.norm.weight", "norm"),
    ])

    remapped = remap_avatar_fp8_fsdp2_state_keys(state)

    assert list(remapped) == [
        "block.linear.fp8_weight_holder.weight",
        "block.linear.bias",
        "block.linear.fp8_weight_holder.scale",
        "block.norm.weight",
    ]
    assert remapped["block.linear.fp8_weight_holder.weight"] == "fp8_weight"
    assert remapped["block.linear.fp8_weight_holder.scale"] == "scale"


def test_llava_llama_model_property_patch(monkeypatch) -> None:
    import sys
    import types

    class FakeLlamaModel:
        pass

    module = types.SimpleNamespace(LlamaModel=FakeLlamaModel)
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    monkeypatch.setitem(sys.modules, "transformers.models", types.ModuleType("transformers.models"))
    monkeypatch.setitem(sys.modules, "transformers.models.llama", types.ModuleType("transformers.models.llama"))
    monkeypatch.setitem(sys.modules, "transformers.models.llama.modeling_llama", module)

    from legacy_model_bridge.runtime.hunyuan_avatar import install_llava_llama_model_property_patch

    result = install_llava_llama_model_property_patch()

    instance = FakeLlamaModel()
    assert result["patched"] is True
    assert instance.model is instance
