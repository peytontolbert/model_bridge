#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legacy_model_bridge.runtime.trellis_native_stack import inspect_trellis_native_stack
from legacy_model_bridge.runtime.trellis_native_stack import report_to_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect TRELLIS.2 native dependency availability.")
    parser.add_argument("--target-env", default="ai")
    args = parser.parse_args(argv)

    report = inspect_trellis_native_stack(target_env=args.target_env)
    sys.stdout.write(report_to_json(report))
    return 0 if report.status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
