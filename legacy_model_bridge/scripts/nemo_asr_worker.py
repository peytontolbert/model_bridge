#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

from legacy_model_bridge.runtime.nemo_asr_prompt_compat import maybe_install_nemo_asr_compat


PATCH_ID_NEMO_PARAKEET_UNIFIED_CONTEXT_CONFIG = "nemo_parakeet_unified_context_config_compat"
PATCH_ID_NEMO_TRANSCRIBE_VALIDATION_DS_DEFAULT = "nemo_asr_transcribe_validation_ds_default"


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _installed_versions() -> dict[str, Any]:
    import importlib.metadata as metadata

    names = {dist.metadata["Name"].lower().replace("-", "_"): dist.version for dist in metadata.distributions()}
    out: dict[str, Any] = {
        "python": sys.version.split()[0],
        "nemo_toolkit": names.get("nemo_toolkit", "missing"),
        "torch": names.get("torch", "missing"),
        "transformers": names.get("transformers", "missing"),
        "pytorch_lightning": names.get("pytorch_lightning", "missing"),
        "hydra_core": names.get("hydra_core", "missing"),
        "omegaconf": names.get("omegaconf", "missing"),
        "soundfile": names.get("soundfile", "missing"),
        "librosa": names.get("librosa", "missing"),
    }
    try:
        import torch

        out["torch_import"] = torch.__version__
        out["torch_cuda"] = bool(torch.cuda.is_available())
        out["torch_cuda_version"] = str(getattr(torch.version, "cuda", None))
        out["torch_cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception as exc:
        out["torch_import"] = f"failed:{type(exc).__name__}:{exc}"
        out["torch_cuda"] = False
        out["torch_cuda_version"] = None
        out["torch_cuda_device_count"] = 0
    return out


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in value.replace("+", ".").split("."):
        if not chunk.isdigit():
            break
        parts.append(int(chunk))
    return tuple(parts)


def _env_status(versions: dict[str, Any]) -> tuple[bool, tuple[str, ...]]:
    problems: list[str] = []
    if _version_tuple(str(versions["python"])) < (3, 11):
        problems.append("python<3.11")
    if versions["torch"] == "missing" or _version_tuple(str(versions["torch"])) < (2, 7):
        problems.append("torch<2.7")
    try:
        import importlib.util

        if versions["nemo_toolkit"] == "missing" or not importlib.util.find_spec("nemo"):
            problems.append("nemo_toolkit_missing")
    except Exception:
        problems.append("nemo_toolkit_missing")
    return not problems, tuple(problems)


def _find_nemo_archive(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    model_path = Path(path)
    if model_path.is_file() and model_path.suffix == ".nemo":
        return model_path
    if not model_path.is_dir():
        return None
    direct = sorted(model_path.glob("*.nemo"))
    if direct:
        return direct[0]
    nested = sorted(model_path.glob("**/*.nemo"))
    return nested[0] if nested else None


def _nemo_model_target(nemo_path: Path | None) -> str | None:
    if nemo_path is None:
        return None
    try:
        with tarfile.open(nemo_path) as archive:
            config_name = next((name for name in archive.getnames() if name.endswith("model_config.yaml")), None)
            if config_name is None:
                return None
            config_file = archive.extractfile(config_name)
            if config_file is None:
                return None
            for raw_line in config_file:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line.startswith("target:"):
                    return line.split(":", 1)[1].strip().strip("'\"")
    except Exception as exc:
        return f"failed:{type(exc).__name__}:{exc}"
    return None


def _target_import_status(target: str | None) -> tuple[bool | None, str]:
    try:
        maybe_install_nemo_asr_compat(target)
    except Exception:
        pass
    if not target:
        return None, "missing_target_metadata"
    if target.startswith("failed:"):
        return False, target
    module_name, _, class_name = target.rpartition(".")
    if not module_name or not class_name:
        return False, "invalid_target"
    try:
        import importlib

        module = importlib.import_module(module_name)
        getattr(module, class_name)
    except Exception as exc:
        return False, f"{type(exc).__name__}:{exc}"
    return True, "ok"


def _exception_chain(exc: BaseException) -> tuple[BaseException, ...]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return tuple(chain)


def _failure_status(exc: BaseException) -> tuple[str, str]:
    detail = " <- ".join(f"{type(item).__name__}:{item}" for item in _exception_chain(exc))
    lowered = detail.lower()
    if any(marker in lowered for marker in ("transformers", "automodel", "generationconfig", "tokenization", "tokenizer")):
        return "needs_nemo_model_specific_env", detail
    return "nemo_restore_failed", detail


def _output_text(item: Any) -> str:
    return str(getattr(item, "text", item))


def _config_text_from_archive(nemo_path: Path) -> str | None:
    try:
        with tarfile.open(nemo_path) as archive:
            config_name = next((name for name in archive.getnames() if name.endswith("model_config.yaml")), None)
            if config_name is None:
                return None
            config_file = archive.extractfile(config_name)
            return config_file.read().decode("utf-8", errors="replace") if config_file is not None else None
    except Exception:
        return None


def _parakeet_unified_restore_override(nemo_path: Path, target: str | None) -> tuple[Any | None, tuple[str, ...]]:
    if target != "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel":
        return None, ()
    text = _config_text_from_archive(nemo_path)
    if not text or "att_chunk_context_size" not in text or "chunked_limited_with_rc" not in text:
        return None, ()
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(text)
    encoder = cfg.get("encoder")
    if encoder is None:
        return None, ()
    old_chunk_context = encoder.get("att_chunk_context_size")
    if "att_chunk_context_size" in encoder:
        del encoder.att_chunk_context_size
    if "conv_context_style" in encoder:
        del encoder.conv_context_style
    if encoder.get("att_context_style") == "chunked_limited_with_rc":
        right_contexts = list(old_chunk_context[-1]) if old_chunk_context is not None else [0]
        encoder.att_context_style = "chunked_limited"
        encoder.att_context_size = [-1, int(max(right_contexts))]
    return cfg, (PATCH_ID_NEMO_PARAKEET_UNIFIED_CONTEXT_CONFIG,)


def _restore_model(nemo_path: Path, map_location: str | None) -> Any:
    target = _nemo_model_target(nemo_path)
    maybe_install_nemo_asr_compat(target)
    override_config, restore_patches = _parakeet_unified_restore_override(nemo_path, target)
    if override_config is not None and target:
        module_name, _, class_name = target.rpartition(".")
        model_cls = getattr(importlib.import_module(module_name), class_name)
        model = model_cls.restore_from(str(nemo_path), override_config_path=override_config, map_location=map_location)
        setattr(model, "_lmb_applied_patches", restore_patches)
        return model
    from nemo.collections.asr.models import ASRModel

    model = ASRModel.restore_from(str(nemo_path), map_location=map_location)
    setattr(model, "_lmb_applied_patches", restore_patches)
    return model


class NemoASRWarmService:
    def __init__(self, req: dict[str, Any]) -> None:
        self.req = req
        self.model = None
        self.restore_payload: dict[str, Any] | None = None
        self.restore_count = 0
        self.transcribe_count = 0
        self.started_at = time.perf_counter()

    def restore(self) -> dict[str, Any]:
        if self.model is not None and self.restore_payload is not None:
            return self._status_payload(self.restore_payload, cached=True)
        payload = inspect_request(self.req)
        if payload["status"] != "ready":
            self.restore_payload = payload
            return payload
        archive = Path(payload["nemo_archive"])
        started = time.perf_counter()
        try:
            map_location = self.req.get("restore_map_location", self.req.get("map_location", "cpu"))
            model = _restore_model(archive, map_location)
            device = self.req.get("device", "cuda:0")
            if device and device != "none":
                model = model.cpu() if device == "cpu" else model.to(device)
            restore_patches = list(getattr(model, "_lmb_applied_patches", ())) + list(_apply_post_restore_compat(model))
            setattr(model, "_lmb_applied_patches", tuple(dict.fromkeys(restore_patches)))
            if hasattr(model, "eval"):
                model.eval()
        except Exception as exc:
            status, detail = _failure_status(exc)
            payload.update({"status": status, "error": detail, "restore_seconds": time.perf_counter() - started})
            self.restore_payload = payload
            return payload
        self.restore_count += 1
        applied_patches = list(dict.fromkeys(list(payload.get("applied_patches", [])) + list(getattr(model, "_lmb_applied_patches", ()))) )
        payload.update({
            "status": "loaded",
            "restore_seconds": time.perf_counter() - started,
            "restored_class": type(model).__name__,
            "device": self.req.get("device", "cuda:0"),
            "worker_pid": os.getpid(),
            "applied_patches": applied_patches,
        })
        self.model = model
        self.restore_payload = payload
        return self._status_payload(payload, cached=False)

    def transcribe(self, audio_paths: list[str]) -> dict[str, Any]:
        if not isinstance(audio_paths, list) or not all(isinstance(path, str) for path in audio_paths):
            return {"status": "bad_request", "error": "audio_paths must be a list of strings"}
        if not audio_paths:
            return {"status": "bad_request", "error": "audio_paths must not be empty"}
        loaded = self.restore()
        if loaded.get("status") != "loaded":
            return loaded
        missing = [path for path in audio_paths if not Path(path).is_file()]
        if missing:
            return {"status": "input_missing", "missing_audio": missing}
        started = time.perf_counter()
        try:
            outputs = self.model.transcribe([str(Path(path)) for path in audio_paths])
        except Exception as exc:
            status, detail = _failure_status(exc)
            return {"status": status, "error": detail, "transcribe_seconds": time.perf_counter() - started}
        self.transcribe_count += 1
        return {
            "status": "ok",
            "transcribe_seconds": time.perf_counter() - started,
            "outputs": [
                {"audio": audio_paths[index], "text": _output_text(item)}
                for index, item in enumerate(outputs)
            ],
            "audio_paths": audio_paths,
            "service": self._service_stats(),
        }

    def _service_stats(self) -> dict[str, Any]:
        return {
            "restore_count": self.restore_count,
            "transcribe_count": self.transcribe_count,
            "uptime_seconds": time.perf_counter() - self.started_at,
            "worker_pid": os.getpid(),
        }

    def _status_payload(self, payload: dict[str, Any], *, cached: bool) -> dict[str, Any]:
        return {**payload, "restore_cached": cached, "service": self._service_stats()}


def _apply_post_restore_compat(model: Any) -> tuple[str, ...]:
    patches: list[str] = []
    cfg = getattr(model, "cfg", None)
    if cfg is not None and getattr(cfg, "validation_ds", None) is None:
        cfg.validation_ds = {}
        patches.append(PATCH_ID_NEMO_TRANSCRIBE_VALIDATION_DS_DEFAULT)
    return tuple(patches)


def inspect_request(req: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    archive = _find_nemo_archive(req.get("archive_path"))
    versions = _installed_versions()
    env_ok, env_problems = _env_status(versions)
    target = _nemo_model_target(archive)
    applied_patches = ()
    try:
        applied_patches = maybe_install_nemo_asr_compat(target)
    except Exception:
        applied_patches = ()
    target_importable, target_import_detail = _target_import_status(target)
    payload: dict[str, Any] = {
        "status": "ready",
        "model_id": req.get("model_id") or (archive.stem if archive else None),
        "nemo_archive": str(archive) if archive else None,
        "nemo_archive_exists": bool(archive and archive.is_file()),
        "nemo_model_target": target,
        "nemo_model_target_importable": target_importable,
        "nemo_model_target_import_detail": target_import_detail,
        "versions": versions,
        "env_ok": env_ok,
        "env_problems": env_problems,
        "preflight_seconds": time.perf_counter() - started,
        "applied_patches": list(applied_patches),
    }
    if archive is None:
        payload["status"] = "missing_nemo_archive"
        payload["error"] = "No .nemo archive found"
    elif not env_ok:
        payload["status"] = "env_not_ready"
        payload["error"] = "NeMo ASR environment is not ready: " + ", ".join(env_problems)
    elif target_importable is False and not req.get("force_restore", False):
        payload["status"] = "needs_nemo_model_specific_env"
        payload["error"] = f"target_not_importable:{target}:{target_import_detail}"
    return payload


def run_request(req: dict[str, Any]) -> dict[str, Any]:
    if not req.get("restore", True):
        return inspect_request(req)
    service = NemoASRWarmService(req)
    loaded = service.restore()
    audio_paths = req.get("audio_paths") or req.get("audio") or []
    if req.get("load_only", False) or not audio_paths:
        return loaded
    result = service.transcribe([str(path) for path in audio_paths])
    return {**loaded, "status": result.get("status", loaded.get("status")), "transcribe": result}


def serve_jsonl(req: dict[str, Any]) -> int:
    service = NemoASRWarmService(req)
    with contextlib.redirect_stdout(sys.stderr):
        startup = service.restore()
    print(json.dumps(startup, sort_keys=True), flush=True)
    for raw in sys.stdin:
        if not raw.strip():
            continue
        try:
            command = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(json.dumps({"status": "bad_json", "error": str(exc)}, sort_keys=True), flush=True)
            continue
        action = command.get("action", "transcribe")
        if action == "shutdown":
            print(json.dumps({"status": "shutdown"}, sort_keys=True), flush=True)
            return 0
        if action == "status":
            with contextlib.redirect_stdout(sys.stderr):
                status_payload = service.restore()
            print(json.dumps(status_payload, sort_keys=True), flush=True)
            continue
        if action == "transcribe":
            audio_paths = command.get("audio_paths") or command.get("audio") or []
            with contextlib.redirect_stdout(sys.stderr):
                transcribe_payload = service.transcribe([str(path) for path in audio_paths])
            print(json.dumps(transcribe_payload, sort_keys=True), flush=True)
            continue
        print(json.dumps({"status": "unsupported_action", "error": f"unsupported action: {action}"}, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="NeMo ASR archive worker for Legacy Model Bridge.")
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--result-json")
    parser.add_argument("--serve-jsonl", action="store_true", help="Restore once, then read JSONL commands from stdin.")
    args = parser.parse_args()
    req = _load_json(args.request_json)
    if args.serve_jsonl:
        return serve_jsonl(req)
    result = run_request(req)
    _write_json(args.result_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2 if result.get("status") in {"env_not_ready", "missing_nemo_archive", "needs_nemo_model_specific_env", "nemo_restore_failed"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
