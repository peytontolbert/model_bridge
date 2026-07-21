from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .registry import DEFAULT_CATALOG_PATH, REPO_ROOT, BridgeEntry, load_catalog


DEFAULT_PATCH_REGISTRY_PATH = REPO_ROOT / "data" / "compatibility_patches.json"


@dataclass(frozen=True)
class CompatibilityPatch:
    patch_id: str
    model_family: str
    lane: str
    mismatch: str
    mechanism: str
    status: str
    validation_refs: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CompatibilityPatch":
        return cls(
            patch_id=str(raw["patch_id"]),
            model_family=str(raw["model_family"]),
            lane=str(raw["lane"]),
            mismatch=str(raw["mismatch"]),
            mechanism=str(raw["mechanism"]),
            status=str(raw["status"]),
            validation_refs=tuple(raw.get("validation_refs", [])),
        )


class PatchRegistry:
    def __init__(self, patches: list[CompatibilityPatch], metadata: dict[str, Any] | None = None) -> None:
        self.patches = patches
        self.metadata = metadata or {}
        self._by_id = {patch.patch_id: patch for patch in patches}

    def get(self, patch_id: str) -> CompatibilityPatch:
        try:
            return self._by_id[patch_id]
        except KeyError as exc:
            raise KeyError(f"compatibility patch is not registered: {patch_id}") from exc

    def filter(self, *, lane: str | None = None, status: str | None = None) -> list[CompatibilityPatch]:
        patches = self.patches
        if lane is not None:
            patches = [patch for patch in patches if patch.lane == lane]
        if status is not None:
            patches = [patch for patch in patches if patch.status == status]
        return patches

    def for_entry(self, entry: BridgeEntry) -> list[CompatibilityPatch]:
        return [self.get(patch_id) for patch_id in entry.compatibility_patches]

    def missing_for_entry(self, entry: BridgeEntry) -> list[str]:
        return [patch_id for patch_id in entry.compatibility_patches if patch_id not in self._by_id]



def load_patch_registry(path: str | Path = DEFAULT_PATCH_REGISTRY_PATH) -> PatchRegistry:
    registry_path = Path(path)
    raw = json.loads(registry_path.read_text())
    patches = [CompatibilityPatch.from_dict(item) for item in raw["patches"]]
    return PatchRegistry(patches=patches, metadata=raw.get("metadata", {}))



def validate_catalog_patches(
    *,
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
    patch_registry_path: str | Path = DEFAULT_PATCH_REGISTRY_PATH,
) -> dict[str, list[str]]:
    catalog = load_catalog(catalog_path)
    registry = load_patch_registry(patch_registry_path)
    missing: dict[str, list[str]] = {}
    for entry in catalog.entries:
        missing_for_entry = registry.missing_for_entry(entry)
        if missing_for_entry:
            missing[entry.model_id] = missing_for_entry
    return missing
