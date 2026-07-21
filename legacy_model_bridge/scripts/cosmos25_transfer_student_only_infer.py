from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def _add_source_paths(runtime_root: Path) -> None:
    paths = [
        runtime_root,
        runtime_root / "packages" / "cosmos-oss",
        runtime_root / "packages" / "cosmos-cuda",
        Path(__file__).resolve().parents[1],
    ]
    for path in reversed(paths):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Cosmos Transfer 2.5 official inference with the bridge student-only DMD2 patch."
    )
    parser.add_argument("--runtime-root", required=True, help="Path to the cosmos-transfer2.5 checkout.")
    parser.add_argument("official_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    runtime_root = Path(args.runtime_root).resolve()
    _add_source_paths(runtime_root)

    from legacy_model_bridge.runtime.cosmos25_student_only import apply_cosmos25_transfer_student_only_patch

    apply_cosmos25_transfer_student_only_patch()
    official_args = list(args.official_args)
    if official_args and official_args[0] == "--":
        official_args = official_args[1:]
    sys.argv = [str(runtime_root / "examples" / "inference.py"), *official_args]
    runpy.run_path(str(runtime_root / "examples" / "inference.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
