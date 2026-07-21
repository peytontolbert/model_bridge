#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORKER_REGISTRY = ROOT / "data" / "worker_registry.json"


def _run_python(env: str, code: str, timeout: int = 60) -> dict[str, Any]:
    proc = subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", env, "python", "-c", code],
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return {
        "env": env,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _probe_imports(env: str, imports: list[str]) -> dict[str, Any]:
    code = r"""
import importlib, json, sys
mods = __MODS__
results = {}
for name in mods:
    try:
        mod = importlib.import_module(name)
        results[name] = {"ok": True, "file": getattr(mod, "__file__", None), "version": getattr(mod, "__version__", None)}
    except Exception as exc:
        results[name] = {"ok": False, "error": f"{type(exc).__name__}:{exc}"}
print(json.dumps(results, sort_keys=True))
""".replace("__MODS__", repr(imports))
    raw = _run_python(env, code)
    try:
        raw["imports"] = json.loads(raw["stdout"].strip().splitlines()[-1])
    except Exception:
        raw["imports"] = {}
    return raw


def _load_workers() -> list[dict[str, Any]]:
    return json.loads(WORKER_REGISTRY.read_text(encoding="utf-8"))["workers"]


def prove_env(env: str, latest_env: str = "ai") -> dict[str, Any]:
    workers = [worker for worker in _load_workers() if worker.get("env") == env]
    required_imports = sorted({item for worker in workers for item in worker.get("required_imports", [])})
    required_paths = sorted({item for worker in workers for item in worker.get("required_paths", [])})
    latest_probe = _probe_imports(latest_env, required_imports) if required_imports else {"imports": {}}
    legacy_probe = _probe_imports(env, required_imports) if required_imports else {"imports": {}}
    missing_latest = sorted(name for name, result in latest_probe.get("imports", {}).items() if not result.get("ok"))
    missing_legacy = sorted(name for name, result in legacy_probe.get("imports", {}).items() if not result.get("ok"))
    missing_paths = sorted(path for path in required_paths if not Path(path).exists())
    safe_to_remove = not workers or (not missing_latest and not missing_paths)
    blockers = []
    if workers and missing_latest:
        blockers.append(f"{latest_env} missing imports: {', '.join(missing_latest)}")
    if missing_paths:
        blockers.append(f"missing required paths: {', '.join(missing_paths)}")
    if missing_legacy:
        blockers.append(f"{env} worker imports also failing: {', '.join(missing_legacy)}")
    return {
        "env": env,
        "latest_env": latest_env,
        "safe_to_remove_now": safe_to_remove,
        "workers": [worker["worker_id"] for worker in workers],
        "models": sorted({model for worker in workers for model in worker.get("models", [])}),
        "required_imports": required_imports,
        "required_paths": required_paths,
        "missing_in_latest_env": missing_latest,
        "missing_in_legacy_env": missing_legacy,
        "missing_required_paths": missing_paths,
        "blockers": blockers,
        "latest_probe": latest_probe,
        "legacy_probe": legacy_probe,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prove whether a worker env can be removed.")
    parser.add_argument("env", nargs="+")
    parser.add_argument("--latest-env", default="ai")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)

    payload = {"proofs": [prove_env(env, args.latest_env) for env in args.env]}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0 if all(item["safe_to_remove_now"] for item in payload["proofs"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
