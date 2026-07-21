"""Legacy Model Bridge runtime metadata helpers."""

from .consolidation import ConsolidationEntry, ConsolidationPlan, load_consolidation_plan
from .integration_skeleton import plan_integration_skeleton, safe_integration_name
from .next_candidates import NextIntegrationCandidate, NextIntegrationPlan, load_next_integration_plan
from .patches import CompatibilityPatch, PatchRegistry, load_patch_registry
from .registry import BridgeCatalog, BridgeEntry, load_catalog
from .runtime.three_d_gen import ThreeDBackend, ThreeDGenRequest, ThreeDGenResult, generate_3d

__all__ = [
    "BridgeCatalog",
    "BridgeEntry",
    "ThreeDBackend",
    "ThreeDGenRequest",
    "ThreeDGenResult",
    "ConsolidationEntry",
    "ConsolidationPlan",
    "CompatibilityPatch",
    "PatchRegistry",
    "NextIntegrationCandidate",
    "NextIntegrationPlan",
    "generate_3d",
    "load_catalog",
    "load_consolidation_plan",
    "load_next_integration_plan",
    "load_patch_registry",
    "plan_integration_skeleton",
    "safe_integration_name",
]
