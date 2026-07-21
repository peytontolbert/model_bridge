from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_PATH = REPO_ROOT / "data" / "bridge_catalog.json"


@dataclass(frozen=True)
class BridgeEntry:
    model_id: str
    lane: str
    status: str
    preferred_env: str
    runnable: bool
    compatibility_patches: tuple[str, ...]
    source_refs: tuple[str, ...]
    notes: str
    target_env: str = ""
    env_policy: str = "review"
    worker_boundary: str = "none"
    worker_entrypoint: str | None = None
    artifact_contract: str | None = None
    consolidation_decision: str = "review"
    consolidation_blockers: tuple[str, ...] = ()
    caller_python: tuple[str, ...] = ("3.11", "3.12")
    backend_python: tuple[str, ...] = ()
    python_strategy: str = "bridge_caller_python_stable_backend_specific"
    mismatch_classes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BridgeEntry":
        return cls(
            model_id=str(raw["model_id"]),
            lane=str(raw["lane"]),
            status=str(raw["status"]),
            preferred_env=str(raw["preferred_env"]),
            runnable=bool(raw.get("runnable", False)),
            compatibility_patches=tuple(raw.get("compatibility_patches", [])),
            source_refs=tuple(raw.get("source_refs", [])),
            notes=str(raw.get("notes", "")),
            target_env=str(raw.get("target_env", raw.get("preferred_env", ""))),
            env_policy=str(raw.get("env_policy", "review")),
            worker_boundary=str(raw.get("worker_boundary", "none")),
            worker_entrypoint=raw.get("worker_entrypoint"),
            artifact_contract=raw.get("artifact_contract"),
            consolidation_decision=str(raw.get("consolidation_decision", "review")),
            consolidation_blockers=tuple(raw.get("consolidation_blockers", [])),
            caller_python=tuple(raw.get("caller_python", ["3.11", "3.12"])),
            backend_python=tuple(raw.get("backend_python", [])),
            python_strategy=str(raw.get("python_strategy", "bridge_caller_python_stable_backend_specific")),
            mismatch_classes=tuple(raw.get("mismatch_classes", [])),
        )


class BridgeCatalog:
    def __init__(self, entries: list[BridgeEntry], metadata: dict[str, Any] | None = None) -> None:
        self.entries = entries
        self.metadata = metadata or {}
        self._by_id = {entry.model_id: entry for entry in entries}

    def get(self, model_id: str) -> BridgeEntry:
        try:
            return self._by_id[model_id]
        except KeyError as exc:
            raise KeyError(f"model is not in the bridge catalog: {model_id}") from exc

    def filter(
        self,
        *,
        lane: str | None = None,
        status: str | None = None,
        runnable: bool | None = None,
        env_policy: str | None = None,
        consolidation_decision: str | None = None,
    ) -> list[BridgeEntry]:
        entries = self.entries
        if lane is not None:
            entries = [entry for entry in entries if entry.lane == lane]
        if status is not None:
            entries = [entry for entry in entries if entry.status == status]
        if runnable is not None:
            entries = [entry for entry in entries if entry.runnable is runnable]
        if env_policy is not None:
            entries = [entry for entry in entries if entry.env_policy == env_policy]
        if consolidation_decision is not None:
            entries = [entry for entry in entries if entry.consolidation_decision == consolidation_decision]
        return entries

    def env_matrix(self) -> dict[str, list[BridgeEntry]]:
        matrix: dict[str, list[BridgeEntry]] = {}
        for entry in self.entries:
            matrix.setdefault(entry.preferred_env, []).append(entry)
        return matrix

    def consolidation_matrix(self) -> dict[str, list[BridgeEntry]]:
        matrix: dict[str, list[BridgeEntry]] = {}
        for entry in self.entries:
            matrix.setdefault(entry.consolidation_decision, []).append(entry)
        return matrix


def load_catalog(path: str | Path = DEFAULT_CATALOG_PATH) -> BridgeCatalog:
    catalog_path = Path(path)
    raw = json.loads(catalog_path.read_text())
    entries = [BridgeEntry.from_dict(item) for item in raw["models"]]
    return BridgeCatalog(entries=entries, metadata=raw.get("metadata", {}))
