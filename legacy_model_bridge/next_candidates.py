from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NEXT_CANDIDATES_PATH = REPO_ROOT / "data" / "next_integration_candidates.json"


@dataclass(frozen=True)
class NextIntegrationCandidate:
    rank: int
    model_id: str
    local_path: str
    lane: str
    catalog_state: str
    preferred_env: str
    caller_python: tuple[str, ...]
    backend_python: tuple[str, ...]
    artifact_contract: str
    mismatch_classes: tuple[str, ...]
    recommended_bridge: str
    first_smoke: str
    model_index_evidence: dict[str, Any]
    transformer_10_evidence: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "NextIntegrationCandidate":
        return cls(
            rank=int(raw["rank"]),
            model_id=str(raw["model_id"]),
            local_path=str(raw["local_path"]),
            lane=str(raw["lane"]),
            catalog_state=str(raw["catalog_state"]),
            preferred_env=str(raw["preferred_env"]),
            caller_python=tuple(str(item) for item in raw.get("caller_python", [])),
            backend_python=tuple(str(item) for item in raw.get("backend_python", [])),
            artifact_contract=str(raw["artifact_contract"]),
            mismatch_classes=tuple(str(item) for item in raw.get("mismatch_classes", [])),
            recommended_bridge=str(raw["recommended_bridge"]),
            first_smoke=str(raw["first_smoke"]),
            model_index_evidence=dict(raw.get("model_index_evidence", {})),
            transformer_10_evidence=tuple(str(item) for item in raw.get("transformer_10_evidence", [])),
        )


@dataclass(frozen=True)
class NextIntegrationPlan:
    metadata: dict[str, Any]
    candidates: tuple[NextIntegrationCandidate, ...]

    def filter(
        self,
        *,
        lane: str | None = None,
        env: str | None = None,
        catalog_state: str | None = None,
        limit: int | None = None,
    ) -> tuple[NextIntegrationCandidate, ...]:
        selected: Iterable[NextIntegrationCandidate] = self.candidates
        if lane is not None:
            selected = (candidate for candidate in selected if candidate.lane == lane)
        if env is not None:
            selected = (candidate for candidate in selected if candidate.preferred_env == env)
        if catalog_state is not None:
            selected = (candidate for candidate in selected if candidate.catalog_state == catalog_state)
        ordered = tuple(sorted(selected, key=lambda candidate: candidate.rank))
        return ordered[:limit] if limit is not None else ordered

    def get(self, model_id: str) -> NextIntegrationCandidate:
        for candidate in self.candidates:
            if candidate.model_id == model_id:
                return candidate
        raise KeyError(f"Unknown next integration candidate: {model_id}")


def load_next_integration_plan(path: str | Path = DEFAULT_NEXT_CANDIDATES_PATH) -> NextIntegrationPlan:
    raw = json.loads(Path(path).read_text())
    return NextIntegrationPlan(
        metadata=dict(raw.get("metadata", {})),
        candidates=tuple(NextIntegrationCandidate.from_dict(item) for item in raw.get("candidates", [])),
    )


def to_json(value: NextIntegrationPlan | NextIntegrationCandidate) -> dict[str, Any]:
    if isinstance(value, NextIntegrationPlan):
        return {
            "metadata": value.metadata,
            "candidates": [to_json(candidate) for candidate in value.candidates],
        }
    return {
        "rank": value.rank,
        "model_id": value.model_id,
        "local_path": value.local_path,
        "lane": value.lane,
        "catalog_state": value.catalog_state,
        "preferred_env": value.preferred_env,
        "caller_python": list(value.caller_python),
        "backend_python": list(value.backend_python),
        "artifact_contract": value.artifact_contract,
        "mismatch_classes": list(value.mismatch_classes),
        "model_index_evidence": value.model_index_evidence,
        "transformer_10_evidence": list(value.transformer_10_evidence),
        "recommended_bridge": value.recommended_bridge,
        "first_smoke": value.first_smoke,
    }
