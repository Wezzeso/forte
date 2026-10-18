"""Exact parameter receipts and quarantined diagnostics for Qwen3.8 training.

These helpers never run a model forward, approve an adapter, or select serving.
NF4 bias initialization changes storage only after an exact reversible-value
check; subsequent frozen snapshots require identical bytes and metadata.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .guards import canonical_sha256, file_sha256, require

BIAS_POLICY = "exact_bf16_before_freeze.v1"


def tensor_sha256(value: Any) -> str:
    import torch
    raw = value.detach().as_subclass(torch.Tensor).contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
    return hashlib.sha256(memoryview(raw).cast("B")).hexdigest()


def quant_state_identity(value: Any) -> Any:
    """Include every tensor and scalar from packed quantization state, recursively."""
    import torch
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_sha256(value)}
    if isinstance(value, dict):
        return {str(key): quant_state_identity(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [quant_state_identity(item) for item in value]
    if hasattr(value, "as_dict"):
        return quant_state_identity(value.as_dict(packed=False))
    require(False, "silver_quantization_state_type_unrecognized")


def parameter_identity(parameter: Any) -> dict:
    state = quant_state_identity(getattr(parameter, "quant_state", None))
    return {"shape": list(parameter.shape), "dtype": str(parameter.dtype),
            "parameter_class": type(parameter).__module__ + "." + type(parameter).__name__,
            "trainable": bool(parameter.requires_grad), "raw_sha256": tensor_sha256(parameter),
            "quant_state": state, "quant_state_sha256": canonical_sha256(state)}


def frozen_snapshot(model: Any) -> dict:
    records = {name: parameter_identity(parameter) for name, parameter in model.named_parameters()
               if not parameter.requires_grad}
    require(bool(records), "silver_frozen_snapshot_empty")
    return {"schema_version": "forte.qwen38.frozen-snapshot.v1", "scope": "all_frozen_named_parameters_and_quant_state",
            "parameters": records, "parameters_sha256": canonical_sha256(records)}


def compare_frozen_snapshots(before: dict, after: dict) -> dict:
    for snapshot in (before, after):
        require(snapshot.get("schema_version") == "forte.qwen38.frozen-snapshot.v1"
                and snapshot.get("scope") == "all_frozen_named_parameters_and_quant_state"
                and isinstance(snapshot.get("parameters"), dict) and bool(snapshot["parameters"])
                and canonical_sha256(snapshot["parameters"]) == snapshot.get("parameters_sha256"),
                "silver_frozen_snapshot_integrity_invalid")
    prior, current = before["parameters"], after["parameters"]
    missing, added = sorted(prior.keys() - current.keys()), sorted(current.keys() - prior.keys())
    changes = []
    for name in sorted(prior.keys() & current.keys()):
        if prior[name] != current[name]:
            changes.append({"name": name, "changed_fields": sorted(key for key in prior[name].keys() | current[name].keys()
                if prior[name].get(key) != current[name].get(key)), "before": prior[name], "after": current[name]})
    return {"schema_version": "forte.qwen38.frozen-difference.v1", "unchanged": not (missing or added or changes),
            "before_parameters_sha256": before["parameters_sha256"], "after_parameters_sha256": after["parameters_sha256"],
            "parameters_checked": len(prior), "missing_names": missing, "added_names": added,
            "changed_parameter_count": len(changes), "changes": changes}


def canonicalize_nf4_bias_storage(model: Any, *, policy: str = BIAS_POLICY) -> dict:
    """Eagerly perform only the pinned, value-preserving bnb BF16 bias cast.

    PEFT can promote BF16 frozen bias to FP32; bitsandbytes then assigns that
    bias back to its BF16 input dtype on first use. Establish that storage before
    the frozen training baseline, checking an exact FP32 value roundtrip. A
    non-reversible cast, trainable bias, unknown dtype or non-BF16 compute fails.
    """
    import torch
    import bitsandbytes as bnb
    require(policy == BIAS_POLICY, "silver_nf4_bias_policy_invalid")
    entries, seen = [], set()
    for name, module in model.named_modules():
        if not isinstance(module, bnb.nn.Linear4bit) or module.bias is None:
            continue
        bias = module.bias
        require(not bias.requires_grad and module.compute_dtype == torch.bfloat16
                and bias.dtype in {torch.float32, torch.bfloat16}, "silver_nf4_bias_cast_scope_invalid")
        if id(bias) in seen:
            continue
        seen.add(id(bias))
        before = parameter_identity(bias)
        before_values = bias.detach().as_subclass(torch.Tensor).float().clone()
        target = bias.detach().as_subclass(torch.Tensor).to(torch.bfloat16)
        require(bool(torch.isfinite(before_values).all().item())
                and torch.equal(before_values, target.float()), "silver_nf4_bias_cast_not_exactly_reversible")
        if bias.dtype != torch.bfloat16:
            with torch.no_grad():
                bias.data = target
        after = parameter_identity(bias)
        require(torch.equal(before_values, bias.detach().as_subclass(torch.Tensor).float()),
                "silver_nf4_bias_logical_values_changed")
        entries.append({"module_name": name, "parameter_name": name + ".bias", "before": before, "after": after,
                        "fp32_logical_sha256": tensor_sha256(before_values), "exact_values_preserved": True,
                        "storage_changed": before != after})
    require(bool(entries), "silver_nf4_bias_initialization_scope_empty")
    result = {"schema_version": "forte.qwen38.nf4-bias-initialization.v1", "policy": policy,
              "bitsandbytes_version": bnb.__version__, "biases_examined": len(entries),
              "biases_storage_changed": sum(item["storage_changed"] for item in entries),
              "all_values_exactly_preserved": True, "entries": entries}
    result["initialization_sha256"] = canonical_sha256(result)
    return result


def save_quarantined_adapter(model: Any, output: Path, config: dict, *, reason_code: str,
                             failure_stage: str, optimizer_steps: int) -> dict:
    """Preserve actual stepped weights without making a selectable candidate."""
    require(type(optimizer_steps) is int and optimizer_steps > 0, "silver_quarantine_zero_step_forbidden")
    directory = output / "quarantine" / "adapter"
    require(not directory.exists(), "silver_quarantine_adapter_already_exists")
    for peft_config in model.peft_config.values():
        peft_config.revision = config["base_revision"]
        peft_config.base_model_name_or_path = config["model_id"]
    model.save_pretrained(directory, safe_serialization=True)
    checksums = {path.relative_to(directory).as_posix(): file_sha256(path)
                 for path in sorted(directory.rglob("*")) if path.is_file()}
    require(bool(checksums), "silver_quarantine_adapter_save_empty")
    return {"schema_version": "forte.qwen38.checkpoint-quarantine.v1", "status": "quarantined",
            "selection": "forbidden", "serving_eligible": False, "training_resume_authorized": False,
            "reason_code": reason_code, "failure_stage": failure_stage, "optimizer_steps": optimizer_steps,
            "base_revision": config["base_revision"], "runtime_image_sha256": config["runtime_image_sha256"],
            "code_archive_sha256": config["code_archive_sha256"], "adapter_checksums": checksums,
            "adapter_tree_sha256": canonical_sha256(checksums), "save_completed": True,
            "frozen_base_verified_by_this_quarantine": False, "reload_forward_verified": False}
