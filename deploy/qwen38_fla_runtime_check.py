"""Exact optional FLA metadata and CPU import resolution; never run a kernel.

Default inspection does not import GPU packages. --cpu-imports is permitted
only in an isolated runc container with no GPU device nodes and no network.
This proves import/selection availability, never correctness or performance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
import sys


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata_check(lock):
    require(lock["schema_version"] == "forte.qwen38.fla-runtime.v1", "fla_lock_schema_invalid")
    require(sys.platform == "linux" and sys.version_info[:2] == (3, 12), "fla_platform_mismatch")
    packages = {}
    for name, expected in {**lock["preserved_packages"], **lock["addon_packages"]}.items():
        actual = metadata.version(name)
        require(actual == expected, "fla_package_version_mismatch")
        packages[name] = actual
    files = {}
    for package, paths in lock["installed_file_sha256"].items():
        distribution = metadata.distribution(package)
        for name, expected in paths.items():
            require(not Path(name).is_absolute() and ".." not in Path(name).parts, "fla_file_path_invalid")
            path = distribution.locate_file(name)
            require(Path(path).is_file() and digest(path) == expected, "fla_installed_file_mismatch")
            files[name] = expected
    require(not any(name.lower().replace("_", "-") == "causal-conv1d"
                    for name in (x.metadata["Name"] for x in metadata.distributions())),
            "fla_unplanned_causal_conv1d_installed")
    return {"schema_version": "forte.qwen38.fla-runtime-check.v1", "status": "passed",
            "runtime_packages": packages, "installed_file_sha256": files,
            "addon_licenses": lock["addon_licenses"], "cpu_imports_performed": False,
            "cuda_initialized": False, "gpu_kernel_calls": 0, "model_instances_created": 0,
            "model_weights_opened": False, "private_media_opened": False,
            "actual_gpu_kernel_correctness_verified": False,
            "actual_gpu_kernel_performance_measured": False,
            "captured_at_utc": datetime.now(timezone.utc).isoformat()}


def closure_implementation(function):
    """Read Transformers' actual chosen original-package function, never call it."""
    current = function
    visited = set()
    for _ in range(8):
        if id(current) in visited:
            break
        visited.add(id(current))
        try:
            values = inspect.getclosurevars(current).nonlocals
        except TypeError:
            values = {}
        implementation = values.get("implementation")
        if callable(implementation):
            return implementation
        wrapped = getattr(current, "__wrapped__", None)
        if wrapped is None:
            break
        current = wrapped
    return current


def cpu_import_check():
    require(not list(Path("/dev").glob("nvidia*")), "fla_cpu_import_gpu_device_present")
    import torch
    require(not torch.cuda.is_initialized(), "fla_cpu_import_cuda_already_initialized")
    from fla.modules import FusedRMSNormGated
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen

    imports = {}
    for name, value in (("FusedRMSNormGated", FusedRMSNormGated),
                        ("chunk_gated_delta_rule", chunk_gated_delta_rule),
                        ("fused_recurrent_gated_delta_rule", fused_recurrent_gated_delta_rule)):
        require(callable(value), "fla_required_symbol_not_callable")
        imports[name] = {"module": value.__module__, "qualname": value.__qualname__,
                         "source_sha256": digest(inspect.getfile(value)), "callable": True}
    selected = {}
    for name, expected in (("torch_chunk_gated_delta_rule", chunk_gated_delta_rule),
                           ("torch_recurrent_gated_delta_rule", fused_recurrent_gated_delta_rule)):
        actual = closure_implementation(getattr(qwen, name))
        require(actual is expected, "fla_qwen_function_resolution_mismatch")
        selected[name] = {"implementation_module": actual.__module__,
                          "implementation_name": actual.__qualname__,
                          "is_exact_imported_fla_function": True}
    conv = {}
    for name in ("causal_conv1d_fn", "causal_conv1d_update"):
        actual = closure_implementation(getattr(qwen, name))
        conv[name] = {"implementation_module": actual.__module__,
                      "implementation_name": actual.__qualname__,
                      "optimized_causal_conv1d_selected": actual.__module__.startswith("causal_conv1d")}
    require(not torch.cuda.is_initialized(), "fla_cpu_import_initialized_cuda")
    require(not list(Path("/dev").glob("nvidia*")), "fla_cpu_import_gpu_device_appeared")
    return {"cpu_imports_performed": True, "gpu_device_nodes_present": False,
            "cuda_initialized": False, "torch_cuda_version_metadata": torch.version.cuda,
            "symbols": imports, "qwen_selected_delta_functions": selected,
            "qwen_causal_convolution": conv,
            "qwen_gated_norm": {"class_module": qwen.Qwen3_5RMSNormGated.__module__,
                "class_name": qwen.Qwen3_5RMSNormGated.__qualname__,
                "fla_fused_norm_available": True,
                "fla_fused_norm_selected_automatically": False,
                "note": "Pinned Qwen norm uses a hub-kernel class decorator; installing FLA alone does not replace its native norm."},
            "gpu_kernel_calls": 0, "model_instances_created": 0,
            "scope": "CPU imports and Python closure identities only; no CUDA tensor, kernel, model or media operation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=Path("deploy/qwen38-fla-runtime-lock.json"))
    parser.add_argument("--cpu-imports", action="store_true")
    args = parser.parse_args()
    try:
        lock = json.loads(args.lock.read_text(encoding="utf-8-sig"))
        result = metadata_check(lock)
        result["runtime_lock_sha256"] = digest(args.lock)
        if args.cpu_imports:
            result["cpu_import_proof"] = cpu_import_check()
            result["cpu_imports_performed"] = True
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        # Only bounded static reason codes/types, never raw imports/env/output.
        code = str(exc) if type(exc) is ValueError and str(exc).startswith("fla_") else "fla_runtime_check_failed"
        print(json.dumps({"status": "blocked", "reason_code": code, "error_type": type(exc).__name__}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
