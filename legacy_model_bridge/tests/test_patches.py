from legacy_model_bridge.patches import load_patch_registry, validate_catalog_patches
from legacy_model_bridge.registry import load_catalog


def test_patch_registry_loads_known_patch() -> None:
    registry = load_patch_registry()

    patch = registry.get("transformers_classifier_head_num_labels_from_checkpoint")

    assert patch.lane == "encoder_classifier_bridge"
    assert patch.status == "verified"


def test_patch_registry_filters_by_lane() -> None:
    registry = load_patch_registry()

    patch_ids = {patch.patch_id for patch in registry.filter(lane="nemo_asr_bridge")}

    assert "nemo_prefer_archive_over_external_transformers_metadata" in patch_ids
    assert "nemo_warm_archive_resident_worker" in patch_ids


def test_catalog_patch_references_are_registered() -> None:
    assert validate_catalog_patches() == {}


def test_entry_patch_resolution() -> None:
    catalog = load_catalog()
    registry = load_patch_registry()

    patches = registry.for_entry(catalog.get("MobileLLM-125M"))

    assert {patch.patch_id for patch in patches} == {
        "transformers_mobilellm_legacy_cache",
        "transformers_mobilellm_slow_tokenizer",
    }
