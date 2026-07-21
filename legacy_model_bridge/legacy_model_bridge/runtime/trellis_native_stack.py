from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TRELLIS_ROOT = Path("/data/clone/third_party/TRELLIS.2")


@dataclass(frozen=True)
class TrellisNativeDependency:
    name: str
    import_names: tuple[str, ...]
    package_hint: str
    build_kind: str
    source_path: str | None
    required_for: str
    import_sites: tuple[str, ...]
    build_notes: str
    required_level: str = "core"


@dataclass(frozen=True)
class TrellisNativeDependencyStatus:
    name: str
    package_hint: str
    build_kind: str
    source_path: str | None
    required_for: str
    import_sites: tuple[str, ...]
    build_notes: str
    imports: dict[str, bool]
    errors: dict[str, str]
    version: str | None
    installed: bool
    source_exists: bool | None
    required_level: str


@dataclass(frozen=True)
class TrellisNativeStackReport:
    status: str
    trellis_root: str
    target_env: str
    dependencies: tuple[TrellisNativeDependencyStatus, ...]
    missing: tuple[str, ...]
    build_order: tuple[str, ...]
    blockers: tuple[str, ...]


DEPENDENCIES: tuple[TrellisNativeDependency, ...] = (
    TrellisNativeDependency(
        name="spconv",
        import_names=("spconv", "spconv.pytorch"),
        package_hint="spconv-cu12x/spconv source compatible with Torch 2.10+cu128",
        build_kind="wheel_or_source_sparse_conv",
        source_path=None,
        required_for="Optional sparse structure and SLAT convolution backend when SPARSE_CONV_BACKEND=spconv is selected; default is flex_gemm.",
        import_sites=(
            "trellis2/modules/sparse/conv/conv_spconv.py:5",
            "trellis2/modules/sparse/basic.py:373",
        ),
        build_notes="Legacy env used spconv-cu121==2.3.8. ai only needs this for spconv backend parity; default TRELLIS.2 uses flex_gemm.",
        required_level="optional_backend_parity",
    ),
    TrellisNativeDependency(
        name="kaolin",
        import_names=("kaolin",),
        package_hint="kaolin built for the active Torch/CUDA ABI",
        build_kind="torch_cuda_extension_package",
        source_path=None,
        required_for="Documented legacy env parity package; no active TRELLIS.2 source import found in the official runtime tree.",
        import_sites=("documented TRELLIS runtime stack",),
        build_notes="Legacy env used kaolin==0.17.0, but local TRELLIS.2 source has no kaolin import. Defer unless a later runtime path proves it required.",
        required_level="optional_legacy_parity",
    ),
    TrellisNativeDependency(
        name="nvdiffrast",
        import_names=("nvdiffrast", "nvdiffrast.torch"),
        package_hint="NVlabs/nvdiffrast v0.4.0 source install",
        build_kind="source_cuda_extension",
        source_path=None,
        required_for="Mesh/PBR rendering and GLB postprocessing rasterization.",
        import_sites=(
            "trellis2/renderers/mesh_renderer.py:45",
            "trellis2/renderers/pbr_mesh_renderer.py:22",
            "trellis2/pipelines/trellis2_texturing.py:13",
            "o-voxel/o_voxel/postprocess.py:10",
        ),
        build_notes="Official setup clones NVlabs/nvdiffrast at v0.4.0 and installs with --no-build-isolation.",
    ),
    TrellisNativeDependency(
        name="cumesh",
        import_names=("cumesh",),
        package_hint="JeffreyXiang/CuMesh source install",
        build_kind="source_cuda_extension",
        source_path=None,
        required_for="CuMesh mesh simplification, UV unwrap, BVH, remeshing, and GLB postprocessing.",
        import_sites=(
            "trellis2/representations/mesh/base.py:4",
            "trellis2/pipelines/trellis2_texturing.py:12",
            "o-voxel/o_voxel/postprocess.py:11",
        ),
        build_notes="Official setup clones CuMesh recursively and installs with --no-build-isolation. Build before o_voxel.",
    ),
    TrellisNativeDependency(
        name="flex_gemm",
        import_names=("flex_gemm", "flex_gemm.ops.grid_sample", "flex_gemm.ops.spconv"),
        package_hint="JeffreyXiang/FlexGEMM source install",
        build_kind="source_cuda_extension",
        source_path=None,
        required_for="Default TRELLIS sparse convolution backend and 3D grid sampling.",
        import_sites=(
            "trellis2/modules/sparse/config.py:3",
            "trellis2/modules/sparse/conv/conv_flex_gemm.py:6",
            "trellis2/representations/mesh/base.py:5",
            "trellis2/renderers/mesh_renderer.py:163",
            "trellis2/pipelines/trellis2_texturing.py:15",
        ),
        build_notes="Official setup clones FlexGEMM recursively and installs with --no-build-isolation. Build before o_voxel.",
    ),
    TrellisNativeDependency(
        name="o_voxel",
        import_names=("o_voxel", "o_voxel.convert", "o_voxel.io", "o_voxel.postprocess", "o_voxel.rasterize"),
        package_hint="bundled TRELLIS.2/o-voxel source install",
        build_kind="repo_bundled_cuda_extension",
        source_path=str(TRELLIS_ROOT / "o-voxel"),
        required_for="Voxel IO/conversion/rasterization and final GLB export path.",
        import_sites=(
            "app.py:19",
            "example.py:11",
            "trellis2/renderers/voxel_renderer.py:49",
            "trellis2/representations/voxel/voxel_model.py:35",
            "trellis2/pipelines/trellis2_texturing.py:11",
        ),
        build_notes="Install from /data/clone/third_party/TRELLIS.2/o-voxel with --no-build-isolation after cumesh and flex_gemm are present.",
    ),    TrellisNativeDependency(
        name="open3d",
        import_names=("open3d",),
        package_hint="open3d>=0.19 wheel for Python 3.11/3.12",
        build_kind="python_wheel_geometry_runtime",
        source_path=None,
        required_for="Data toolkit geometry IO support and documented env parity; not an active official pipeline import in local TRELLIS.2 source.",
        import_sites=("data_toolkit/setup.sh", "documented TRELLIS runtime stack"),
        build_notes="Open3D 0.19.0 is already present in ai; keep as data-toolkit parity, not a core blocker.",
        required_level="optional_data_toolkit",
    ),

)

BUILD_ORDER: tuple[str, ...] = (
    "open3d",
    "spconv",
    "kaolin",
    "nvdiffrast",
    "cumesh",
    "flex_gemm",
    "o_voxel",
)


def _probe_import(name: str) -> tuple[bool, str | None]:
    try:
        importlib.import_module(name)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _version_for(dep: TrellisNativeDependency) -> str | None:
    candidates = [dep.name, dep.package_hint.split()[0], dep.name.replace("_", "-")]
    if dep.name == "spconv":
        candidates.extend(["spconv-cu121", "spconv-cu120", "spconv-cu12"])
    for candidate in candidates:
        try:
            return importlib.metadata.version(candidate)
        except Exception:
            continue
    return None


def inspect_trellis_native_stack(target_env: str = "ai") -> TrellisNativeStackReport:
    statuses: list[TrellisNativeDependencyStatus] = []
    missing: list[str] = []
    blockers: list[str] = []
    for dep in DEPENDENCIES:
        imports: dict[str, bool] = {}
        errors: dict[str, str] = {}
        for import_name in dep.import_names:
            ok, error = _probe_import(import_name)
            imports[import_name] = ok
            if error:
                errors[import_name] = error
        installed = all(imports.values())
        source_exists = Path(dep.source_path).exists() if dep.source_path else None
        if not installed:
            missing.append(dep.name)
            if dep.required_level == "core":
                blockers.append(f"{dep.name}: {dep.build_notes}")
        statuses.append(
            TrellisNativeDependencyStatus(
                name=dep.name,
                package_hint=dep.package_hint,
                build_kind=dep.build_kind,
                source_path=dep.source_path,
                required_for=dep.required_for,
                import_sites=dep.import_sites,
                build_notes=dep.build_notes,
                imports=imports,
                errors=errors,
                version=_version_for(dep),
                installed=installed,
                source_exists=source_exists,
                required_level=dep.required_level,
            )
        )
    core_missing = tuple(status.name for status in statuses if status.required_level == "core" and not status.installed)
    status = "ok" if not core_missing else "missing_native_dependencies"
    return TrellisNativeStackReport(
        status=status,
        trellis_root=str(TRELLIS_ROOT),
        target_env=target_env,
        dependencies=tuple(statuses),
        missing=core_missing,
        build_order=BUILD_ORDER,
        blockers=tuple(blockers),
    )


def report_to_dict(report: TrellisNativeStackReport) -> dict[str, Any]:
    return asdict(report)


def report_to_json(report: TrellisNativeStackReport) -> str:
    return json.dumps(report_to_dict(report), indent=2, sort_keys=True) + "\n"
