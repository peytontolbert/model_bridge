from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ROOT = Path("/arxiv/models")


@dataclass(frozen=True)
class AudioGenRequest:
    model_id: str = "audiogen-medium"
    model_root: str = str(DEFAULT_MODEL_ROOT)
    inspect_state: bool = False
    audiocraft_source_path: str | None = None
    probe_source_import: bool = False


@dataclass(frozen=True)
class AudioGenResult:
    status: str
    model_id: str
    model_path: str | None
    artifact_contract: str = "audiocraft_audiogen_checkpoint_pair"
    checkpoint_files: tuple[str, ...] = ()
    checkpoint_bytes: dict[str, int] | None = None
    dependency_status: dict[str, bool] | None = None
    source_probe: dict[str, Any] | None = None
    attention_backend_plan: dict[str, Any] | None = None
    config_summary: dict[str, Any] | None = None
    state_summary: dict[str, Any] | None = None
    compatibility_patches: tuple[str, ...] = ()
    next_bridge_step: str | None = None
    error: str | None = None


class AudioGenBridgeError(RuntimeError):
    pass


def resolve_model_path(model_id: str, model_root: str | Path = DEFAULT_MODEL_ROOT) -> Path:
    candidate = Path(model_id)
    if candidate.exists():
        return candidate
    root = Path(model_root)
    candidates = [root / model_id]
    if "/" in model_id:
        candidates.append(root / model_id.split("/", 1)[1])
        candidates.append(root / model_id.replace("/", "--"))
    for item in candidates:
        if item.exists():
            return item
    raise AudioGenBridgeError(f"model path not found for {model_id!r} under {root}")


def _checkpoint_files(model_path: Path) -> tuple[str, ...]:
    return tuple(name for name in ("state_dict.bin", "compression_state_dict.bin") if (model_path / name).is_file())


def _dependency_status() -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in ("audiocraft", "encodec", "torch", "torchaudio")}


def _cfg_value(cfg: str, key: str) -> str | None:
    top_level = re.findall(rf"(?m)^{re.escape(key)}:\s*([^\n#]+)", cfg)
    if top_level:
        return top_level[-1].strip()
    match = re.search(rf"(?m)^\s+{re.escape(key)}:\s*([^\n#]+)", cfg)
    return match.group(1).strip() if match else None


def _cfg_summary(cfg: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("sample_rate", "channels", "lm_model", "compression_model", "compression_model_checkpoint", "card", "n_q", "segment_duration"):
        value = _cfg_value(cfg, key)
        if value is not None:
            summary[key] = value
    summary["has_transformer_lm"] = "transformer_lm:" in cfg
    summary["has_encodec"] = "compression_model: encodec" in cfg or "compression_model=encodec" in cfg
    summary["has_delay_pattern"] = "codebooks_pattern" in cfg
    return summary


def _state_prefix_counts(state: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in state:
        prefix = str(key).split(".", 1)[0]
        counts[prefix] = counts.get(prefix, 0) + 1
    return dict(sorted(counts.items()))


def _tensor_examples(state: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    examples = []
    for key, value in state.items():
        examples.append({
            "key": str(key),
            "shape": list(value.shape) if hasattr(value, "shape") else None,
            "dtype": str(getattr(value, "dtype", type(value).__name__)),
        })
        if len(examples) >= limit:
            break
    return examples


def _raise_xformers_called(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("xformers attention shim was called; AudioGen bridge should disable memory_efficient attention")


def install_xformers_import_shim() -> bool:
    """Install a minimal import-only xformers shim for AudioCraft's no-xformers path.

    AudioCraft imports ``from xformers import ops`` at module import time even when
    ``memory_efficient`` attention is disabled. The bridge uses this shim only to
    satisfy imports; any real xformers attention call raises immediately.
    """
    if importlib.util.find_spec("xformers") is not None:
        return False
    import types

    package = types.ModuleType("xformers")
    package.__file__ = "<legacy_model_bridge_xformers_shim>"
    package.__path__ = []
    package.__spec__ = importlib.machinery.ModuleSpec("xformers", loader=None, is_package=True)

    ops = types.ModuleType("xformers.ops")
    ops.__file__ = "<legacy_model_bridge_xformers_ops_shim>"
    ops.__spec__ = importlib.machinery.ModuleSpec("xformers.ops", loader=None)

    class LowerTriangularMask:
        pass

    ops.LowerTriangularMask = LowerTriangularMask
    ops.memory_efficient_attention = _raise_xformers_called
    package.ops = ops
    sys.modules.setdefault("xformers", package)
    sys.modules.setdefault("xformers.ops", ops)
    return True


def disable_audiogen_memory_efficient_attention(cfg: Any) -> bool:
    """Force AudioCraft AudioGen LM config onto nn.MultiheadAttention fallback.

    The upstream checkpoint weights stay unchanged; AudioCraft remaps regular MHA
    keys in ``StreamingMultiheadAttention._load_from_state_dict``.
    """
    try:
        transformer_lm = cfg["transformer_lm"] if isinstance(cfg, dict) else cfg.transformer_lm
    except Exception:
        return False
    try:
        old = transformer_lm["memory_efficient"] if isinstance(transformer_lm, dict) else transformer_lm.memory_efficient
    except Exception:
        old = None
    if isinstance(transformer_lm, dict):
        transformer_lm["memory_efficient"] = False
    else:
        transformer_lm.memory_efficient = False
    return old is not False


def audiogen_attention_backend_plan() -> dict[str, Any]:
    return {
        "preferred_runtime_backend": "torch_nn_multihead_attention",
        "checkpoint_weight_changes_required": False,
        "bridge_overrides": [
            "install_xformers_import_shim_when_xformers_missing",
            "set cfg.transformer_lm.memory_efficient=false before builders.get_lm_model",
        ],
        "why_not_torch_sdpa_first": (
            "AudioCraft imports xformers.ops unconditionally and its memory_efficient=true path still verifies xformers masks before reaching torch SDPA."
        ),
    }


def _probe_audiocraft_source(source_path: str | Path | None) -> dict[str, Any] | None:
    if source_path is None:
        return None
    root = Path(source_path)
    probe: dict[str, Any] = {"source_path": str(root), "exists": root.exists()}
    if not root.exists():
        probe.update({"importable": False, "error": "source_path_missing"})
        return probe

    before_path = list(sys.path)
    before_modules = set(sys.modules)
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("audiocraft")
        from audiocraft.models import AudioGen as _AudioGen  # noqa: F401

        probe.update({
            "importable": True,
            "audiocraft_version": getattr(module, "__version__", None),
            "loader": "audiocraft.models.AudioGen",
        })
    except Exception as exc:
        probe.update({"importable": False, "error": f"{type(exc).__name__}:{exc}"})
    finally:
        sys.path[:] = before_path
        for name in set(sys.modules) - before_modules:
            if name == "audiocraft" or name.startswith("audiocraft."):
                sys.modules.pop(name, None)
    return probe


def inspect_audiogen(request: AudioGenRequest) -> AudioGenResult:
    try:
        model_path = resolve_model_path(request.model_id, request.model_root)
        files = _checkpoint_files(model_path)
        sizes = {name: int((model_path / name).stat().st_size) for name in files}
        deps = _dependency_status()
        source_probe = _probe_audiocraft_source(request.audiocraft_source_path) if request.probe_source_import else None
        source_importable = bool(source_probe and source_probe.get("importable"))
        status = "blocked_missing_audiocraft_adapter" if not deps.get("audiocraft") and not source_importable else "ready_for_audiocraft_load_smoke"
        config_summary: dict[str, Any] = {}
        state_summary: dict[str, Any] = {}
        if request.inspect_state:
            import torch

            for name in files:
                obj = torch.load(model_path / name, map_location="cpu", weights_only=False)
                if not isinstance(obj, dict) or "best_state" not in obj:
                    raise AudioGenBridgeError(f"{name} is not an AudioCraft solver checkpoint")
                cfg = obj.get("xp.cfg")
                state = obj.get("best_state")
                if not isinstance(cfg, str) or not isinstance(state, dict):
                    raise AudioGenBridgeError(f"{name} missing xp.cfg string or best_state dict")
                config_summary[name] = _cfg_summary(cfg)
                state_summary[name] = {
                    "best_state_key_count": len(state),
                    "prefix_counts": _state_prefix_counts(state),
                    "tensor_examples": _tensor_examples(state),
                }
        return AudioGenResult(
            status=status,
            model_id=request.model_id,
            model_path=str(model_path),
            checkpoint_files=files,
            checkpoint_bytes=sizes,
            dependency_status=deps,
            source_probe=source_probe,
            attention_backend_plan=audiogen_attention_backend_plan(),
            config_summary=config_summary or None,
            state_summary=state_summary or None,
            compatibility_patches=("audiogen_audiocraft_solver_checkpoint_inspector",),
            next_bridge_step=(
                "Vendor or install a latest-compatible AudioCraft loader, then build a bounded 16kHz text-to-audio smoke using state_dict.bin plus compression_state_dict.bin."
                if not deps.get("audiocraft") and not source_importable
                else "Run AudioCraft-compatible load-state smoke in ai using the local checkpoint pair, with bridge-owned attention backend overrides as needed."
            ),
        )
    except Exception as exc:
        return AudioGenResult(status="failed", model_id=request.model_id, model_path=None, error=f"{type(exc).__name__}:{exc}")


def to_json(obj: Any) -> dict[str, Any]:
    return asdict(obj)
