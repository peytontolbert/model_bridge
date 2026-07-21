from __future__ import annotations

import keyword
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .registry import BridgeEntry, load_catalog


OPTIONAL_FILES = {"model.py", "config_map.py", "tokenizer_bridge.py", "ops.py"}
CONTRACT_STATUSES = {"planned", "experimental", "beta", "stable", "blocked"}
_RESERVED_NAMES = set(keyword.kwlist) | {"none", "true", "false"}


@dataclass(frozen=True)
class SkeletonPlan:
    integration_name: str
    root: Path
    files: tuple[Path, ...]
    profile: dict[str, Any]
    entry: BridgeEntry | None


class IntegrationSkeletonError(ValueError):
    pass


def safe_integration_name(raw: str) -> str:
    if any(part == ".." for part in re.split(r"[/\\]+", raw)):
        raise IntegrationSkeletonError("integration name must not contain path traversal")
    slug = re.sub(r"[/\\.\-\s]+", "_", raw.lower())
    slug = re.sub(r"[^a-z0-9_]", "", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        raise IntegrationSkeletonError("integration name is empty after normalization")
    if slug[0].isdigit():
        slug = f"model_{slug}"
    if slug in _RESERVED_NAMES:
        raise IntegrationSkeletonError(f"reserved integration name: {slug}")
    return slug


def _entry_from_args(
    model_id: str,
    *,
    catalog_path: str | Path,
    allow_uncataloged: bool,
    lane: str | None,
    preferred_env: str | None,
) -> BridgeEntry:
    try:
        return load_catalog(catalog_path).get(model_id)
    except KeyError:
        if not allow_uncataloged:
            raise IntegrationSkeletonError(f"model is not in catalog: {model_id}")
        if not lane or not preferred_env:
            raise IntegrationSkeletonError("--lane and --preferred-env are required for uncataloged models")
        return BridgeEntry(
            model_id=model_id,
            lane=lane,
            status="uncataloged",
            preferred_env=preferred_env,
            runnable=False,
            compatibility_patches=(),
            source_refs=(),
            notes="Generated before catalog promotion.",
        )


def _profile(name: str, entry: BridgeEntry, status: str) -> dict[str, Any]:
    return {
        "name": name,
        "model_id": entry.model_id,
        "status": status,
        "bridge_lane": entry.lane,
        "preferred_env": entry.preferred_env,
        "fallback_envs": [],
        "caller_runtime": {"python": ">=3.11,<3.13"},
        "backend_runtime": {
            "preferred_env": entry.preferred_env,
            "python": list(entry.backend_python),
            "strategy": entry.python_strategy,
            "worker_boundary": entry.worker_boundary,
        },
        "artifacts": {"config": "required", "checkpoint": "required", "tokenizer": "optional"},
        "compatibility": {
            "patches_required": list(entry.compatibility_patches),
            "patches_applied": [],
            "checkpoint_conversion": "required",
            "custom_ops": "unknown",
            "numerical_parity": "unknown",
            "mismatch_classes": list(entry.mismatch_classes),
        },
        "validation": {"smoke": "required", "forward": "required", "reference_outputs": "optional", "reports": []},
        "blocked_reason": None,
        "catalog": {
            "status": entry.status,
            "runnable": entry.runnable,
            "source_refs": list(entry.source_refs),
            "notes": entry.notes,
        },
    }


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or any(ch in text for ch in ":#[]{}\n") or text.lower() in {"null", "true", "false"}:
        return repr(text)
    return text


def render_yaml(value: Any, indent: int = 0) -> str:
    spaces = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, list) and not item:
                lines.append(f"{spaces}{key}: []")
            elif isinstance(item, (dict, list)):
                lines.append(f"{spaces}{key}:")
                rendered = render_yaml(item, indent + 2)
                if rendered:
                    lines.append(rendered)
            else:
                lines.append(f"{spaces}{key}: {_yaml_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return f"{spaces}[]"
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{spaces}-")
                lines.append(render_yaml(item, indent + 2))
            else:
                lines.append(f"{spaces}- {_yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{spaces}{_yaml_scalar(value)}"


def plan_integration_skeleton(
    model_id: str,
    *,
    catalog_path: str | Path = Path("data/bridge_catalog.json"),
    out_dir: str | Path = Path("integrations"),
    name: str | None = None,
    status: str = "planned",
    include: tuple[str, ...] = (),
    allow_uncataloged: bool = False,
    lane: str | None = None,
    preferred_env: str | None = None,
) -> SkeletonPlan:
    if status not in CONTRACT_STATUSES:
        raise IntegrationSkeletonError(f"invalid integration status: {status}")
    unknown_optional = set(include) - OPTIONAL_FILES
    if unknown_optional:
        raise IntegrationSkeletonError(f"unknown optional files: {', '.join(sorted(unknown_optional))}")
    entry = _entry_from_args(
        model_id,
        catalog_path=catalog_path,
        allow_uncataloged=allow_uncataloged,
        lane=lane,
        preferred_env=preferred_env,
    )
    integration_name = safe_integration_name(name or model_id)
    root = Path(out_dir) / integration_name
    resolved_out = Path(out_dir).resolve()
    resolved_root = root.resolve()
    if resolved_out not in resolved_root.parents and resolved_root != resolved_out:
        raise IntegrationSkeletonError("integration root resolves outside out-dir")
    files = [
        root / "README.md",
        root / "profile.yaml",
        root / "adapter.py",
        root / "checkpoint_map.py",
        root / "tests" / f"test_{integration_name}_integration.py",
    ]
    files.extend(root / item for item in include)
    return SkeletonPlan(
        integration_name=integration_name,
        root=root,
        files=tuple(files),
        profile=_profile(integration_name, entry, status),
        entry=entry,
    )


def _readme(plan: SkeletonPlan) -> str:
    entry = plan.entry
    assert entry is not None
    patches = ", ".join(entry.compatibility_patches) if entry.compatibility_patches else "none"
    refs = "\n".join(f"- {ref}" for ref in entry.source_refs) or "- none"
    return (
        f"# {plan.integration_name}\n\n"
        f"Model ID: `{entry.model_id}`\n\n"
        f"Bridge lane: `{entry.lane}`\n\n"
        f"Preferred env: `{entry.preferred_env}`\n\n"
        f"Catalog status: `{entry.status}`\n\n"
        f"Runnable: `{str(entry.runnable).lower()}`\n\n"
        f"Compatibility patches: {patches}\n\n"
        f"## Source References\n\n{refs}\n\n"
        "## Checklist\n\n"
        "- [ ] Config load\n"
        "- [ ] Checkpoint map or conversion\n"
        "- [ ] Model construction\n"
        "- [ ] Minimal forward pass\n"
        "- [ ] Validation report\n\n"
        f"## Known Limitations\n\n{entry.notes}\n"
    )


def _adapter(name: str) -> str:
    return (
        "from __future__ import annotations\n\n"
        "from pathlib import Path\n"
        "from typing import Any\n\n\n"
        "def detect(source_path: str | Path) -> bool:\n"
        "    source = Path(source_path)\n"
        "    return source.exists()\n\n\n"
        "def load_config(source_path: str | Path) -> dict[str, Any]:\n"
        f"    raise NotImplementedError({name!r} + ' config loading is not implemented yet')\n\n\n"
        "def load_model(source_path: str | Path, **kwargs: Any) -> Any:\n"
        f"    raise NotImplementedError({name!r} + ' model loading is not implemented yet')\n"
    )


def _checkpoint_map(name: str) -> str:
    return (
        "from __future__ import annotations\n\n"
        "from pathlib import Path\n"
        "from typing import Any\n\n\n"
        "TENSOR_KEY_MAP: dict[str, str] = {}\n\n\n"
        "def map_checkpoint_key(key: str) -> str:\n"
        "    return TENSOR_KEY_MAP.get(key, key)\n\n\n"
        "def convert_checkpoint(source_path: str | Path, out_path: str | Path) -> dict[str, Any]:\n"
        f"    raise NotImplementedError({name!r} + ' checkpoint conversion is not implemented yet')\n"
    )


def _test_file(plan: SkeletonPlan, test_style: str) -> str:
    marker = "skip" if test_style == "skip" else "xfail"
    reason = f"{plan.integration_name} integration skeleton pending implementation"
    return (
        "import pytest\n\n\n"
        f"@pytest.mark.{marker}(reason={reason!r})\n"
        "def test_load_model_placeholder() -> None:\n"
        f"    from integrations.{plan.integration_name}.adapter import load_model\n\n"
        "    load_model('/tmp/nonexistent')\n"
    )


def _optional_file(name: str) -> str:
    return f"'''Optional integration module for {name}.'''\n"


def render_files(plan: SkeletonPlan, *, test_style: str = "skip") -> dict[Path, str]:
    if test_style not in {"skip", "xfail"}:
        raise IntegrationSkeletonError("test_style must be skip or xfail")
    contents = {
        plan.root / "README.md": _readme(plan),
        plan.root / "profile.yaml": render_yaml(plan.profile) + "\n",
        plan.root / "adapter.py": _adapter(plan.integration_name),
        plan.root / "checkpoint_map.py": _checkpoint_map(plan.integration_name),
        plan.root / "tests" / f"test_{plan.integration_name}_integration.py": _test_file(plan, test_style),
    }
    for path in plan.files:
        if path.name in OPTIONAL_FILES:
            contents[path] = _optional_file(plan.integration_name)
    return contents


def write_integration_skeleton(plan: SkeletonPlan, *, force: bool = False, test_style: str = "skip") -> list[Path]:
    if plan.root.exists() and not force:
        raise IntegrationSkeletonError(f"integration directory already exists: {plan.root}")
    contents = render_files(plan, test_style=test_style)
    for path, text in contents.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return sorted(contents)
