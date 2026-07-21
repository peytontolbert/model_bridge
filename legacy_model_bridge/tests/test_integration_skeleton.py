import subprocess
import sys
from pathlib import Path

import pytest

from legacy_model_bridge.integration_skeleton import (
    IntegrationSkeletonError,
    plan_integration_skeleton,
    render_files,
    safe_integration_name,
    write_integration_skeleton,
)


def test_safe_integration_name_normalizes_catalog_ids() -> None:
    assert safe_integration_name("MobileLLM-125M") == "mobilellm_125m"
    assert safe_integration_name("acvlab--ABot-World-0-5B-LF") == "acvlab_abot_world_0_5b_lf"
    assert safe_integration_name("microsoft/TRELLIS.2-4B") == "microsoft_trellis_2_4b"


def test_safe_integration_name_rejects_empty_or_traversal() -> None:
    with pytest.raises(IntegrationSkeletonError):
        safe_integration_name("../bad")
    with pytest.raises(IntegrationSkeletonError):
        safe_integration_name("!!!")


def test_safe_integration_name_prefixes_digit_and_rejects_keywords() -> None:
    assert safe_integration_name("123-model") == "model_123_model"
    with pytest.raises(IntegrationSkeletonError):
        safe_integration_name("class")


def test_generator_maps_catalog_fields(tmp_path: Path) -> None:
    plan = plan_integration_skeleton("repository_library/abstract-repo-planning", out_dir=tmp_path)

    assert plan.integration_name == "repository_library_abstract_repo_planning"
    assert plan.profile["bridge_lane"] == "encoder_classifier_bridge"
    assert plan.profile["preferred_env"] == "ai"
    assert plan.profile["caller_runtime"]["python"] == ">=3.11,<3.13"
    assert plan.profile["backend_runtime"]["preferred_env"] == "ai"
    assert plan.profile["backend_runtime"]["python"] == ["3.11"]
    assert plan.profile["compatibility"]["patches_required"] == [
        "transformers_classifier_head_num_labels_from_checkpoint"
    ]
    assert "bridge_patch_required" in plan.profile["compatibility"]["mismatch_classes"]
    assert plan.profile["catalog"]["status"] == "works_cuda_forward_smoke_with_config_patch"


def test_generator_dry_run_render_does_not_write_files(tmp_path: Path) -> None:
    plan = plan_integration_skeleton("MobileLLM-125M", out_dir=tmp_path)
    rendered = render_files(plan)

    assert plan.files
    assert rendered[plan.root / "profile.yaml"].startswith("name: mobilellm_125m")
    assert not plan.root.exists()


def test_generator_fails_for_uncataloged_model_by_default(tmp_path: Path) -> None:
    with pytest.raises(IntegrationSkeletonError):
        plan_integration_skeleton("not-a-real-model", out_dir=tmp_path)


def test_generator_allows_uncataloged_with_lane_and_env(tmp_path: Path) -> None:
    plan = plan_integration_skeleton(
        "not-a-real-model",
        out_dir=tmp_path,
        allow_uncataloged=True,
        lane="custom_bridge_review",
        preferred_env="lmb-default",
    )

    assert plan.profile["catalog"]["status"] == "uncataloged"


def test_generator_writes_required_and_optional_files(tmp_path: Path) -> None:
    plan = plan_integration_skeleton("MobileLLM-125M", out_dir=tmp_path, include=("model.py", "ops.py"))

    written = write_integration_skeleton(plan)

    assert plan.root / "README.md" in written
    assert plan.root / "profile.yaml" in written
    assert plan.root / "adapter.py" in written
    assert plan.root / "checkpoint_map.py" in written
    assert plan.root / "model.py" in written
    assert plan.root / "ops.py" in written
    assert (plan.root / "tests" / "test_mobilellm_125m_integration.py").exists()


def test_generator_fails_when_directory_exists_without_force(tmp_path: Path) -> None:
    plan = plan_integration_skeleton("MobileLLM-125M", out_dir=tmp_path)
    write_integration_skeleton(plan)

    with pytest.raises(IntegrationSkeletonError):
        write_integration_skeleton(plan)


def test_cli_generate_integration_dry_run(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "legacy_model_bridge.cli",
            "generate-integration",
            "MobileLLM-125M",
            "--out-dir",
            str(tmp_path),
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "mobilellm_125m/profile.yaml" in result.stdout
    assert not (tmp_path / "mobilellm_125m").exists()
