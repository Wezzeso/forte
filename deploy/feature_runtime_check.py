"""Check an isolated feature runtime without media or CUDA initialization."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys


def check(lock_path: Path, *, require_av: bool = False, training: bool = False) -> dict:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != "forte.training.runtime.v1":
        raise ValueError("feature_runtime_lock_invalid")
    if sys.platform != "linux" or sys.version_info[:2] != (3, 12):
        raise ValueError("feature_runtime_platform_mismatch")
    expected = dict(lock["core_packages"])
    if require_av:
        expected.update(lock["optional_media_packages"])
    if training:
        expected.update(lock["training_addon_packages"])
    actual = {name: importlib.metadata.version(name) for name in expected}
    if actual != expected:
        raise ValueError("feature_runtime_dependency_mismatch")
    return {"status": "passed", "runtime_packages": actual,
            "cuda_initialized": False, "model_weights_opened": False,
            "private_media_opened": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=Path("deploy/feature-runtime-lock.json"))
    parser.add_argument("--require-av", action="store_true")
    parser.add_argument("--training", action="store_true")
    args = parser.parse_args()
    try:
        result = check(args.lock, require_av=args.require_av, training=args.training)
    except Exception as exc:
        # Do not emit exception text, paths or environment values.
        result = {"status": "blocked", "reason_code": "feature_runtime_check_failed",
                  "error_type": type(exc).__name__}
    print(json.dumps(result, sort_keys=True))
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
