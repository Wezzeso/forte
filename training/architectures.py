"""Explicit offline architecture dispatch and auditable language-only LoRA scope.

The Qwen3.8-27B release uses the Transformers ``qwen3_5`` implementation and
Qwen3VLProcessor. Its DeltaNet projections are not self-attention q/v layers.
No model is loaded, downloaded, or moved to a GPU by importing this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Mapping

from .guards import canonical_sha256, require


QWEN38_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SOURCE_REFERENCES = (
    "https://huggingface.co/Qwen/Qwen3.8-27B/blob/" + QWEN38_REVISION + "/config.json",
    "https://huggingface.co/Qwen/Qwen3.8-27B/blob/" + QWEN38_REVISION + "/model.safetensors.index.json",
    "https://raw.githubusercontent.com/huggingface/transformers/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py",
)


@dataclass(frozen=True)
class ArchitectureProfile:
    model_type: str
    model_class_name: str
    processor_class_names: tuple[str, ...]
    hybrid: bool


_PROFILES = {
    "qwen3_vl": ArchitectureProfile("qwen3_vl", "Qwen3VLForConditionalGeneration", ("Qwen3VLProcessor",), False),
    "qwen3_5": ArchitectureProfile("qwen3_5", "Qwen3_5ForConditionalGeneration", ("Qwen3VLProcessor",), True),
}
_FULL = ("q_proj", "k_proj", "v_proj", "o_proj")
_LINEAR = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
_MLP = ("gate_proj", "up_proj", "down_proj")
_LAYER = re.compile(r"^(?:base_model\.model\.)?(?:model\.)?language_model\.layers\.(\d+)\.(self_attn|linear_attn|mlp)\.([A-Za-z0-9_]+)$")
_ADAPTER = re.compile(r"^(.*)\.(lora_A|lora_B)\.([A-Za-z0-9_-]+)\.weight$")
_FORBIDDEN = frozenset({"visual", "vision", "vision_model", "vision_tower", "aligner", "merger", "projector", "multi_modal_projector"})


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def resolve_architecture(model_type: str) -> ArchitectureProfile:
    require(isinstance(model_type, str) and model_type in _PROFILES, "unsupported_model_architecture")
    return _PROFILES[model_type]


def resolve_training_classes(profile: ArchitectureProfile, transformers_module: Any) -> tuple[Any, Any]:
    """Return explicit model class and AutoProcessor; callers verify its instance."""
    require(profile == resolve_architecture(profile.model_type), "training_architecture_profile_invalid")
    model_class = getattr(transformers_module, profile.model_class_name, None)
    processor_class = getattr(transformers_module, "AutoProcessor", None)
    require(model_class is not None and processor_class is not None, "training_architecture_class_unavailable")
    return model_class, processor_class


def validate_architecture_config(config: Any) -> ArchitectureProfile:
    profile = resolve_architecture(_field(config, "model_type"))
    require(_field(config, "architectures") == [profile.model_class_name], "model_architecture_class_mismatch")
    require(_field(config, "language_model_only", False) is False, "multimodal_model_required")
    _layer_types(config, profile)
    return profile


def _layer_types(config: Any, profile: ArchitectureProfile) -> list[str]:
    text = _field(config, "text_config")
    count = _field(text, "num_hidden_layers")
    require(isinstance(count, int) and not isinstance(count, bool) and count > 0, "lora_language_layer_count_invalid")
    if not profile.hybrid:
        return ["full_attention"] * count
    types = _field(text, "layer_types")
    require(isinstance(types, (list, tuple)) and len(types) == count, "lora_hybrid_layer_types_invalid")
    require(all(value in {"full_attention", "linear_attention"} for value in types), "lora_hybrid_layer_type_unsupported")
    require(set(types) == {"full_attention", "linear_attention"}, "lora_hybrid_both_attention_families_required")
    return list(types)


def _policy(profile: ArchitectureProfile, layer_type: str) -> dict[str, tuple[str, ...]]:
    if not profile.hybrid:
        # Preserve the established Qwen3-VL baseline experiment recipe.
        return {"self_attn": ("q_proj", "v_proj")}
    return {("linear_attn" if layer_type == "linear_attention" else "self_attn"):
            (_LINEAR if layer_type == "linear_attention" else _FULL), "mlp": _MLP}


def _language_target(name: str) -> re.Match[str] | None:
    if _FORBIDDEN.intersection(name.split(".")):
        return None
    return _LAYER.fullmatch(name)


def _is_lora_linear(module: Any) -> bool:
    # bitsandbytes Linear4bit/Linear8bitLt subclass nn.Linear. Conv1d does not.
    from torch import nn
    return isinstance(module, nn.Linear)


def inventory_language_lora_targets(model: Any, model_type: str) -> dict[str, Any]:
    """Enumerate actual modules and require complete declared per-layer coverage.

    This runs before PEFT wrapping. Exact full module paths are passed to PEFT;
    suffix-only targets could accidentally adapt the vision encoder or aligner.
    """
    profile = resolve_architecture(model_type)
    require(_field(model.config, "model_type") == model_type, "lora_model_architecture_mismatch")
    types = _layer_types(model.config, profile)
    found: dict[int, dict[str, dict[str, Any]]] = {}
    for name, module in model.named_modules():
        match = _language_target(name)
        if match is None:
            continue
        index, group, leaf = int(match[1]), match[2], match[3]
        require(index < len(types), "lora_language_layer_out_of_range")
        if leaf not in _policy(profile, types[index]).get(group, ()):
            continue
        require(_is_lora_linear(module), "lora_target_is_not_linear")
        require(leaf not in found.setdefault(index, {}).setdefault(group, {}), "lora_duplicate_language_layer")
        found[index][group][leaf] = {
            "name": name, "module_class": type(module).__module__ + "." + type(module).__name__,
            "in_features": int(module.in_features), "out_features": int(module.out_features),
        }
    require(bool(found), "lora_language_targets_missing")
    layers, targets = [], []
    group_counts: dict[str, int] = {}
    for index, layer_type in enumerate(types):
        policy = _policy(profile, layer_type)
        for group, leaves in policy.items():
            require(set(found.get(index, {}).get(group, {})) == set(leaves), "lora_layer_targets_missing")
            group_counts[group] = group_counts.get(group, 0) + len(leaves)
            targets.extend(found[index][group][leaf] for leaf in leaves)
        layers.append({"layer_index": index, "layer_type": layer_type,
                       "target_groups": {group: list(leaves) for group, leaves in policy.items()}})
    targets.sort(key=lambda value: value["name"])
    receipt = {
        "schema_version": "forte.training.lora-inventory.v1", "model_type": model_type,
        "model_class_name": profile.model_class_name, "policy": "hybrid_language_projections_and_mlp" if profile.hybrid else "baseline_language_qv",
        "target_modules": [value["name"] for value in targets], "targets": targets,
        "target_count": len(targets), "language_layer_count": len(types), "layers": layers,
        "full_attention_layer_count": types.count("full_attention"),
        "linear_attention_layer_count": types.count("linear_attention"), "groups": group_counts,
        "coverage": "all_configured_language_layers", "vision_and_aligner_target_count": 0,
        "source_references": list(SOURCE_REFERENCES) if profile.hybrid else [],
    }
    receipt["inventory_sha256"] = canonical_sha256(receipt)
    return receipt


def _adapter_target(name: str) -> tuple[str, str, str] | None:
    match = _ADAPTER.fullmatch(name)
    if match is None or _language_target(match[1]) is None:
        return None
    return match[1], match[2], match[3]


def _unwrapped_target(name: str) -> str:
    return name.removeprefix("base_model.model.")


def validate_trainable_adapter(model: Any, model_type: str | None = None, *,
                               inventory: dict[str, Any] | None = None) -> list[tuple[str, Any]]:
    """Reject any trainable base, vision, aligner, head or unsupported adapter."""
    model_type = model_type or _field(model.config, "model_type")
    profile = resolve_architecture(model_type)
    require(_field(model.config, "model_type") == model_type, "lora_model_architecture_mismatch")
    types = _layer_types(model.config, profile)
    trainable, pairs = [], {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parsed = _adapter_target(name)
        require(parsed is not None, "non_language_adapter_parameter_trainable")
        target, side, adapter = parsed
        match = _language_target(target)
        assert match is not None
        index, group, leaf = int(match[1]), match[2], match[3]
        require(index < len(types) and leaf in _policy(profile, types[index]).get(group, ()),
                "unsupported_language_adapter_parameter_trainable")
        pairs.setdefault((_unwrapped_target(target), adapter), set()).add(side)
        trainable.append((name, parameter))
    require(bool(trainable), "lora_trainable_parameters_missing")
    require(all(sides == {"lora_A", "lora_B"} for sides in pairs.values()), "lora_trainable_pair_incomplete")
    require(len({adapter for _, adapter in pairs}) == 1, "lora_multiple_trainable_adapters_forbidden")
    actual = {(int(_language_target(target)[1]), _language_target(target)[2], _language_target(target)[3])
              for target, _ in pairs}
    required = {(index, group, leaf) for index, layer_type in enumerate(types)
                for group, leaves in _policy(profile, layer_type).items() for leaf in leaves}
    require(actual == required, "lora_trainable_layer_coverage_incomplete")
    expected = {_unwrapped_target(value) for value in inventory["target_modules"]} if inventory else None
    if expected is not None:
        require({target for target, _ in pairs} == expected, "lora_trainable_inventory_mismatch")
    return trainable


def validate_adapter_gradients(model: Any, *, inventory: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call after backward and before zero_grad; require active hybrid groups."""
    import torch
    trainable = validate_trainable_adapter(model, inventory=inventory)
    nonzero_groups, nonzero_count = set(), 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            require(parameter.grad is None, "frozen_parameter_received_gradient")
    for name, parameter in trainable:
        gradient = parameter.grad
        require(gradient is not None and bool(torch.isfinite(gradient).all().item()), "training_gradient_missing_or_nonfinite")
        if bool(torch.count_nonzero(gradient).item()):
            parsed = _adapter_target(name)
            assert parsed is not None
            nonzero_groups.add(_language_target(parsed[0])[2])
            nonzero_count += 1
    profile = resolve_architecture(_field(model.config, "model_type"))
    expected = {"self_attn", "linear_attn", "mlp"} if profile.hybrid else {"self_attn"}
    require(nonzero_groups == expected, "training_gradient_group_inactive")
    return {"trainable_tensors": len(trainable), "nonzero_gradient_tensors": nonzero_count,
            "nonzero_gradient_groups": sorted(nonzero_groups), "frozen_gradients_absent": True,
            "all_trainable_gradients_finite": True}


def _tensor_sha256(tensor: Any) -> str:
    """Hash every stored byte; BF16 and packed quantized storage stay exact."""
    import torch
    value = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
    return hashlib.sha256(memoryview(value)).hexdigest()


def capture_parameter_proof(model: Any, *, inventory: dict[str, Any] | None = None) -> dict[str, Any]:
    """Full parameter hashing is deliberate; this is not a sampled freeze claim.

    On a 27B model this transfers the full stored base to CPU one tensor at a
    time. Call at controlled smoke boundaries, not at every optimizer step.
    Dynamic caches and non-parameter buffers are outside this declared scope.
    """
    import torch
    validate_trainable_adapter(model, inventory=inventory)
    records = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            require(bool(torch.isfinite(parameter).all().item()), "adapter_parameter_nonfinite")
        records[name] = {"shape": list(parameter.shape), "dtype": str(parameter.dtype),
                         "trainable": bool(parameter.requires_grad), "sha256": _tensor_sha256(parameter)}
    return {"schema_version": "forte.training.parameter-proof.v1", "scope": "all_named_parameters",
            "parameters": records, "parameters_sha256": canonical_sha256(records)}


def verify_parameter_update(model: Any, before: dict[str, Any], *, optimizer_steps: int,
                            minimum_optimizer_steps: int = 2,
                            inventory: dict[str, Any] | None = None) -> dict[str, Any]:
    """Prove changed adapters and byte-identical frozen base after actual steps.

    The caller supplies its executed optimizer-step counter. This helper does
    not infer optimizer execution from weights or run a training step itself.
    """
    require(isinstance(minimum_optimizer_steps, int) and not isinstance(minimum_optimizer_steps, bool)
            and minimum_optimizer_steps >= 1 and isinstance(optimizer_steps, int)
            and not isinstance(optimizer_steps, bool) and optimizer_steps >= minimum_optimizer_steps,
            "optimizer_step_proof_insufficient")
    require(before.get("schema_version") == "forte.training.parameter-proof.v1"
            and before.get("scope") == "all_named_parameters", "parameter_proof_scope_invalid")
    prior = before.get("parameters", {})
    require(bool(prior) and canonical_sha256(prior) == before.get("parameters_sha256"), "parameter_proof_digest_mismatch")
    after = capture_parameter_proof(model, inventory=inventory)
    current = after["parameters"]
    require(set(prior) == set(current), "parameter_proof_name_mismatch")
    changed, frozen_count, changed_groups = [], 0, set()
    for name, record in current.items():
        previous = prior[name]
        require(all(record[key] == previous[key] for key in ("shape", "dtype", "trainable")), "parameter_proof_metadata_changed")
        if not record["trainable"]:
            require(record["sha256"] == previous["sha256"], "frozen_parameter_changed")
            frozen_count += 1
        elif record["sha256"] != previous["sha256"]:
            changed.append(name)
            changed_groups.add(_language_target(_adapter_target(name)[0])[2])
    require(frozen_count > 0, "frozen_parameter_proof_missing")
    profile = resolve_architecture(_field(model.config, "model_type"))
    expected = {"self_attn", "linear_attn", "mlp"} if profile.hybrid else {"self_attn"}
    require(changed_groups == expected, "adapter_parameter_group_unchanged")
    return {"schema_version": "forte.training.optimizer-proof.v1", "optimizer_steps_reported": optimizer_steps,
            "parameter_scope": "all_named_parameters", "frozen_parameters_checked": frozen_count,
            "frozen_parameters_byte_identical": True, "changed_adapter_tensors": changed,
            "changed_adapter_groups": sorted(changed_groups), "adapter_parameters_finite": True,
            "before_parameters_sha256": before["parameters_sha256"],
            "after_parameters_sha256": after["parameters_sha256"]}
