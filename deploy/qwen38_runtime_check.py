"""Inspect candidate package pins by default; synthetic NF4 GPU smoke is opt-in.

This script never opens model files or private inputs. Only the root operator
may request --gpu-smoke in the serialized GPU window. Metadata inspection does
not import torch, bitsandbytes, Transformers or any optional kernel module.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import time


class RuntimeCheckError(ValueError):
    """Safe reason codes only; no raw environment or exception output."""


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeCheckError(reason)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify_metadata(lock: dict, *, platform_name: str | None = None,
                    python_version: tuple[int, int] | None = None,
                    machine: str | None = None) -> dict:
    require(lock.get("schema_version") == "forte.qwen38.training-runtime.v1",
            "qwen38_runtime_lock_invalid")
    require((platform_name or sys.platform) == "linux"
            and (python_version or sys.version_info[:2]) == (3, 12)
            and (machine or platform.machine()).lower() in {"x86_64", "amd64"},
            "qwen38_runtime_platform_mismatch")
    expected = {**lock["core_packages"], **lock["addon_packages"]}
    require(expected.get("bitsandbytes") == "0.50.2"
            and expected.get("torch") == "2.13.0+cu129"
            and expected.get("transformers") == "5.17.0",
            "qwen38_runtime_core_pin_invalid")
    actual = {}
    for name, version in expected.items():
        try:
            actual[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            raise RuntimeCheckError("qwen38_runtime_package_missing") from None
        require(actual[name] == version, "qwen38_runtime_package_version_mismatch")
    require(os.environ.get("BNB_CUDA_VERSION") is None,
            "qwen38_runtime_unplanned_bnb_cuda_override")
    bnb = metadata.distribution("bitsandbytes")
    verified = {}
    pins = lock["bitsandbytes"]["installed_file_sha256"]
    require(bool(pins), "qwen38_runtime_bnb_file_pins_missing")
    for name, expected_hash in pins.items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts
                and relative.parts[0] == "bitsandbytes", "qwen38_runtime_bnb_path_invalid")
        target = Path(bnb.locate_file(name))
        require(target.is_file() and digest(target) == expected_hash,
                "qwen38_runtime_bnb_file_hash_mismatch")
        verified[name] = expected_hash
    optional = {}
    for name in lock.get("optional_kernel_packages", []):
        try:
            optional[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            optional[name] = None
    return {
        "schema_version": "forte.qwen38.runtime-check.v1",
        "status": "passed", "inspection": "package_metadata_and_exact_installed_file_hashes",
        "runtime_packages": actual, "verified_bnb_files": verified,
        "optional_kernel_packages": optional,
        "hybrid_kernel_execution_verified": False,
        "cuda_initialized": False, "gpu_smoke_performed": False,
        "model_weights_opened": False, "private_media_opened": False,
        "full_qwen38_model_verified": False,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def synthetic_nf4_smoke(lock: dict) -> dict:
    """Small generated-input two-step adapter test; requires root's GPU window."""
    import torch
    require(torch.version.cuda == lock["cuda_version"], "qwen38_runtime_cuda_version_mismatch")
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            "qwen38_runtime_cuda_bf16_unavailable")
    require(list(torch.cuda.get_device_capability(0)) == lock["target_gpu_compute_capability"],
            "qwen38_runtime_gpu_capability_mismatch")
    import bitsandbytes as bnb
    from bitsandbytes.cextension import CudaBNBNativeLibrary, lib
    require(isinstance(lib, CudaBNBNativeLibrary), "qwen38_runtime_bnb_native_cuda_not_loaded")
    loaded_library = Path(str(lib._lib._name)).name
    require(loaded_library == lock["bitsandbytes"]["expected_cuda_library"],
            "qwen38_runtime_bnb_loaded_library_mismatch")
    torch.manual_seed(20261017)
    torch.cuda.manual_seed_all(20261017)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    reference = torch.nn.Linear(128, 128, bias=False, dtype=torch.bfloat16)
    quantized = bnb.nn.Linear4bit(128, 128, bias=False, compute_dtype=torch.bfloat16,
                                 compress_statistics=True, quant_type="nf4")
    quantized.load_state_dict(reference.state_dict())
    quantized = quantized.to("cuda")
    quantized.requires_grad_(False)
    require(quantized.weight.quant_type == "nf4" and quantized.weight.quant_state is not None
            and quantized.compute_dtype == torch.bfloat16,
            "qwen38_runtime_nf4_quantization_unverified")
    require(bool(quantized.weight.quant_state.nested), "qwen38_runtime_double_quantization_missing")

    def tensor_hash(value) -> str:
        return hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

    frozen_before = tensor_hash(quantized.weight)
    adapter_a = torch.nn.Parameter(torch.randn((8, 128), dtype=torch.bfloat16, device="cuda") * 0.01)
    adapter_b = torch.nn.Parameter(torch.zeros((128, 8), dtype=torch.bfloat16, device="cuda"))
    adapter_before = tensor_hash(adapter_b)
    optimizer = torch.optim.AdamW([adapter_a, adapter_b], lr=0.01, weight_decay=0.0)
    losses, gradient_norms = [], []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        inputs = torch.randn((4, 128), dtype=torch.bfloat16, device="cuda", requires_grad=True)
        targets = torch.randn((4, 128), dtype=torch.bfloat16, device="cuda")
        outputs = quantized(inputs) + (inputs @ adapter_a.T) @ adapter_b.T
        loss = (outputs.float() - targets.float()).square().mean()
        require(bool(torch.isfinite(loss).item()), "qwen38_runtime_nf4_nonfinite_loss")
        loss.backward()
        grads = [adapter_a.grad, adapter_b.grad, inputs.grad]
        require(all(x is not None and bool(torch.isfinite(x).all().item()) for x in grads),
                "qwen38_runtime_nf4_nonfinite_or_missing_gradient")
        norm = sum(float(x.detach().float().norm().item()) for x in grads)
        require(norm > 0, "qwen38_runtime_nf4_zero_gradients")
        losses.append(float(loss.detach().item()))
        gradient_norms.append(norm)
        optimizer.step()
    torch.cuda.synchronize()
    frozen_after, adapter_after = tensor_hash(quantized.weight), tensor_hash(adapter_b)
    require(frozen_before == frozen_after and adapter_before != adapter_after,
            "qwen38_runtime_nf4_frozen_or_adapter_update_invalid")
    return {
        "synthetic_only": True, "full_qwen38_model_verified": False,
        "private_input_used": False, "model_weights_opened": False,
        "cuda_initialized": True, "gpu_name": torch.cuda.get_device_name(0),
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_cuda_version": torch.version.cuda, "loaded_bnb_library": loaded_library,
        "quant_type": "nf4", "compute_dtype": "bfloat16", "double_quantized": True,
        "optimizer_steps": 2, "losses": losses, "finite_gradient_norms": gradient_norms,
        "frozen_quantized_weight_sha256": frozen_after,
        "frozen_quantized_weight_unchanged": True,
        "synthetic_adapter_changed": True, "synthetic_adapter_sha256": adapter_after,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "wall_seconds": time.monotonic() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=Path("deploy/qwen38-training-runtime-lock.json"))
    parser.add_argument("--gpu-smoke", action="store_true",
                        help="Root only, after backup/idle/exclusive GPU handoff gates; generated inputs only")
    args = parser.parse_args()
    try:
        lock = json.loads(args.lock.read_text(encoding="utf-8-sig"))
        result = verify_metadata(lock)
        result["runtime_lock_sha256"] = digest(args.lock)
        if args.gpu_smoke:
            result["synthetic_nf4"] = synthetic_nf4_smoke(lock)
            result.update(cuda_initialized=True, gpu_smoke_performed=True)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        result = {"status": "blocked", "reason_code": str(exc) if isinstance(exc, RuntimeCheckError)
                  else "qwen38_runtime_check_failed", "error_type": type(exc).__name__}
        print(json.dumps(result, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
