from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .registry import REPO_ROOT


DEFAULT_CONSOLIDATION_PATH = REPO_ROOT / "data" / "environment_consolidation.json"
LATEST_ENV_DECISIONS = {
    "latest_env_ready",
    "latest_env_with_patches",
    "latest_env_patch_candidate",
    "latest_env_bridge_candidate",
    "latest_env_needs_packages",
    "latest_env_needs_packages_or_worker",
    "latest_env_needs_runtime_package",
    "custom_bridge_needed_in_latest_env",
}
WORKER_DECISIONS = {"worker_boundary_required"}


@dataclass(frozen=True)
class ConsolidationEntry:
    model_id: str
    lane: str
    current_env: str
    target_env: str
    decision: str
    remove_env_after: str
    required_patches: tuple[str, ...]
    blocker: str
    source_refs: tuple[str, ...]
    caller_python: tuple[str, ...] = ("3.11", "3.12")
    backend_python: tuple[str, ...] = ()
    python_strategy: str = "bridge_caller_python_stable_backend_specific"
    mismatch_classes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ConsolidationEntry":
        return cls(
            model_id=str(raw["model_id"]),
            lane=str(raw["lane"]),
            current_env=str(raw["current_env"]),
            target_env=str(raw["target_env"]),
            decision=str(raw["decision"]),
            remove_env_after=str(raw.get("remove_env_after", "")),
            required_patches=tuple(raw.get("required_patches", [])),
            blocker=str(raw.get("blocker", "")),
            source_refs=tuple(raw.get("source_refs", [])),
            caller_python=tuple(raw.get("caller_python", ["3.11", "3.12"])),
            backend_python=tuple(raw.get("backend_python", [])),
            python_strategy=str(raw.get("python_strategy", "bridge_caller_python_stable_backend_specific")),
            mismatch_classes=tuple(raw.get("mismatch_classes", [])),
        )

    @property
    def can_target_latest(self) -> bool:
        return self.decision in LATEST_ENV_DECISIONS

    @property
    def requires_worker_boundary(self) -> bool:
        return self.decision in WORKER_DECISIONS

    def supports_caller_python(self, version: str) -> bool:
        major_minor = ".".join(version.split(".")[:2])
        return major_minor in self.caller_python


class ConsolidationPlan:
    def __init__(self, entries: list[ConsolidationEntry], metadata: dict[str, Any] | None = None) -> None:
        self.entries = entries
        self.metadata = metadata or {}
        self._by_id = {entry.model_id: entry for entry in entries}

    def get(self, model_id: str) -> ConsolidationEntry:
        try:
            return self._by_id[model_id]
        except KeyError as exc:
            raise KeyError(f"model is not in the consolidation plan: {model_id}") from exc

    def filter(
        self,
        *,
        current_env: str | None = None,
        target_env: str | None = None,
        decision: str | None = None,
        lane: str | None = None,
        caller_python: str | None = None,
    ) -> list[ConsolidationEntry]:
        entries = self.entries
        if current_env is not None:
            entries = [entry for entry in entries if entry.current_env == current_env]
        if target_env is not None:
            entries = [entry for entry in entries if entry.target_env == target_env]
        if decision is not None:
            entries = [entry for entry in entries if entry.decision == decision]
        if lane is not None:
            entries = [entry for entry in entries if entry.lane == lane]
        if caller_python is not None:
            entries = [entry for entry in entries if entry.supports_caller_python(caller_python)]
        return entries

    def decision_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry.decision] = counts.get(entry.decision, 0) + 1
        return counts

    def current_env_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry.current_env] = counts.get(entry.current_env, 0) + 1
        return counts

    def removable_env_candidates(self) -> list[ConsolidationEntry]:
        return [entry for entry in self.entries if entry.can_target_latest]

    def worker_boundaries(self) -> list[ConsolidationEntry]:
        return [entry for entry in self.entries if entry.requires_worker_boundary]

    def caller_python_matrix(self) -> dict[str, list[ConsolidationEntry]]:
        matrix: dict[str, list[ConsolidationEntry]] = {}
        for entry in self.entries:
            for version in entry.caller_python:
                matrix.setdefault(version, []).append(entry)
        return matrix



def load_consolidation_plan(path: str | Path = DEFAULT_CONSOLIDATION_PATH) -> ConsolidationPlan:
    plan_path = Path(path)
    raw = json.loads(plan_path.read_text())
    entries = [ConsolidationEntry.from_dict(item) for item in raw["models"]]
    return ConsolidationPlan(entries=entries, metadata=raw.get("metadata", {}))
