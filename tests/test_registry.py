from legacy_model_bridge.registry import BridgeCatalog, load_catalog


def test_catalog_loads_curated_entries() -> None:
    catalog = load_catalog()

    entry = catalog.get("acvlab--ABot-World-0-5B-LF")

    assert entry.runnable is True
    assert entry.lane == "video_diffusion_bridge"
    assert "abot_flash_attention_sdpa_fallback" in entry.compatibility_patches


def test_catalog_filters_by_lane_and_runnable() -> None:
    catalog = load_catalog()

    entries = catalog.filter(lane="transformers_causal_lm_bridge", runnable=True)

    assert {entry.model_id for entry in entries} >= {"MobileLLM-125M", "MobileLLM-350M"}


def test_env_matrix_groups_entries() -> None:
    catalog = load_catalog()

    matrix = catalog.env_matrix()

    assert "ai" in matrix
    assert any(entry.model_id == "AnyFlow-Wan2.1-T2V-1.3B-Diffusers" for entry in matrix["ai"])


def test_catalog_can_be_built_from_dict_entries() -> None:
    catalog = BridgeCatalog(
        entries=[
            load_catalog().get("cwm"),
        ]
    )

    assert catalog.filter(runnable=False)[0].status == "blocked_missing_architecture"


def test_catalog_records_caller_backend_python_policy() -> None:
    catalog = load_catalog()

    trellis = catalog.get("microsoft/TRELLIS.2-4B")
    hunyuan = catalog.get("Hunyuan3D-2mv")
    wan = catalog.get("Wan-AI--Wan2.2-Animate-14B")

    assert trellis.caller_python == ("3.11", "3.12")
    assert trellis.backend_python == ("3.11", "3.12")
    assert "torch_cuda_native_stack" in trellis.mismatch_classes
    assert hunyuan.backend_python == ("3.11",)
    assert wan.runnable is True
    assert wan.preferred_env == "ai"
    assert "sageattention2_lightx2v_backend_select" in wan.compatibility_patches
