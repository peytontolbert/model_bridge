from __future__ import annotations

from legacy_model_bridge.runtime.trellis_native_stack import BUILD_ORDER
from legacy_model_bridge.runtime.trellis_native_stack import DEPENDENCIES
from legacy_model_bridge.runtime.trellis_native_stack import inspect_trellis_native_stack
from legacy_model_bridge.runtime.trellis_native_stack import report_to_dict


def test_trellis_native_stack_tracks_all_known_native_dependencies() -> None:
    names = {dep.name for dep in DEPENDENCIES}

    assert names == {"spconv", "kaolin", "nvdiffrast", "open3d", "cumesh", "flex_gemm", "o_voxel"}
    assert BUILD_ORDER[-1] == "o_voxel"


def test_trellis_native_stack_records_import_sites_and_build_notes() -> None:
    by_name = {dep.name: dep for dep in DEPENDENCIES}

    assert "conv_spconv.py:5" in " ".join(by_name["spconv"].import_sites)
    assert by_name["spconv"].required_level == "optional_backend_parity"
    assert by_name["kaolin"].required_level == "optional_legacy_parity"
    assert by_name["open3d"].required_level == "optional_data_toolkit"
    assert "trellis2/modules/sparse/config.py:3" in by_name["flex_gemm"].import_sites
    assert by_name["o_voxel"].source_path.endswith("/data/clone/third_party/TRELLIS.2/o-voxel")
    assert "--no-build-isolation" in by_name["nvdiffrast"].build_notes


def test_trellis_native_stack_report_is_json_ready() -> None:
    report = inspect_trellis_native_stack(target_env="ai")
    payload = report_to_dict(report)

    assert payload["target_env"] == "ai"
    assert payload["build_order"] == BUILD_ORDER
    assert {item["name"] for item in payload["dependencies"]} == {dep.name for dep in DEPENDENCIES}
    assert payload["status"] in {"ok", "missing_native_dependencies"}
