"""Validate the isolated training stack; CUDA is opt-in, never used by default."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

from .guards import TrainingGateError, load_json, require


def verify_runtime(lock_path: Path, *, gpu: bool = False) -> dict:
    lock = load_json(lock_path)
    require(lock.get("schema_version") == "forte.training.runtime.v1", "training_runtime_lock_invalid")
    require(sys.platform == "linux" and sys.version_info[:2] == (3, 12), "training_runtime_platform_mismatch")
    expected = {**lock["core_packages"], **lock["addon_packages"]}
    actual = {}
    for name, version in expected.items():
        try:
            actual[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise TrainingGateError("training_runtime_dependency_missing") from None
        require(actual[name] == version, "training_runtime_dependency_version_mismatch")
    result = {"status": "passed", "runtime_packages": actual, "cuda_initialized": False,
              "model_weights_opened": False, "private_media_opened": False}
    if gpu:
        import torch
        require(torch.version.cuda == lock["cuda_version"], "training_runtime_cuda_version_mismatch")
        require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "training_runtime_cuda_bf16_unavailable")
        tensor = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
        require(bool(torch.isfinite(tensor @ tensor).all().item()), "training_runtime_cuda_bf16_failed")
        torch.cuda.synchronize()
        result.update(cuda_initialized=True, cuda_bf16_verified=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=Path("training/runtime-lock.json"))
    parser.add_argument("--gpu", action="store_true", help="Only after the GPU reservation is granted")
    args = parser.parse_args()
    try:
        print(json.dumps(verify_runtime(args.lock, gpu=args.gpu)))
        return 0
    except TrainingGateError as exc:
        print(json.dumps({"status": "blocked", "reason_code": str(exc)}))
        return 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "reason_code": "training_runtime_check_failed",
                          "error_type": type(exc).__name__}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
