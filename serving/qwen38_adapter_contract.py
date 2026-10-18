"""CPU-only proof of the pinned 27B language adapter, without tensor imports.

PEFT may serialize the reviewed leaf suffixes although the fit receipt records
qualified targets. Equivalence requires the complete actual topology and every
serialized A/B tensor shape; suffix comparison alone is never sufficient.
"""

import hashlib
import json
from math import prod
from pathlib import Path


class AdapterContractError(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise AdapterContractError(code)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


FULL = {"q_proj": (5120, 12288), "k_proj": (5120, 1024),
        "v_proj": (5120, 1024), "o_proj": (6144, 5120)}
LINEAR = {"in_proj_qkv": (5120, 10240), "in_proj_z": (5120, 6144),
          "in_proj_b": (5120, 48), "in_proj_a": (5120, 48), "out_proj": (6144, 5120)}
MLP = {"gate_proj": (5120, 17408), "up_proj": (5120, 17408), "down_proj": (17408, 5120)}
LEAF_TARGETS = frozenset(FULL) | frozenset(LINEAR) | frozenset(MLP)


def expected_targets():
    targets = {}
    for index in range(64):
        groups = {"self_attn": FULL} if index % 4 == 3 else {"linear_attn": LINEAR}
        groups["mlp"] = MLP
        for group, leaves in groups.items():
            for leaf, dimensions in leaves.items():
                targets[f"model.language_model.layers.{index}.{group}.{leaf}"] = dimensions
    return targets


def validate_inventory(inventory):
    require(isinstance(inventory, dict), "adapter_inventory_missing")
    expected = expected_targets()
    names = inventory.get("target_modules")
    require(isinstance(names, list) and all(isinstance(name, str) for name in names)
            and len(names) == 496 and len(set(names)) == 496
            and set(names) == set(expected), "adapter_full_target_topology_mismatch")
    require(inventory.get("model_type") == "qwen3_5"
            and inventory.get("coverage") == "all_configured_language_layers"
            and inventory.get("language_layer_count") == 64
            and inventory.get("linear_attention_layer_count") == 48
            and inventory.get("full_attention_layer_count") == 16
            and inventory.get("vision_and_aligner_target_count") == 0
            and inventory.get("target_count") == 496
            and inventory.get("groups") == {"self_attn": 64, "linear_attn": 240, "mlp": 192},
            "adapter_inventory_coverage_mismatch")
    require(all(type(inventory.get(key)) is int for key in ("language_layer_count", "linear_attention_layer_count",
        "full_attention_layer_count", "vision_and_aligner_target_count", "target_count"))
        and all(type(value) is int for value in inventory["groups"].values()), "adapter_inventory_coverage_mismatch")
    layers = inventory.get("layers")
    require(isinstance(layers, list) and len(layers) == 64, "adapter_layer_inventory_missing")
    for index, layer in enumerate(layers):
        require(isinstance(layer, dict), "adapter_layer_schedule_mismatch")
        full = index % 4 == 3
        expected_groups = {"self_attn" if full else "linear_attn": set(FULL if full else LINEAR), "mlp": set(MLP)}
        groups = layer.get("target_groups", {}) if isinstance(layer, dict) else {}
        require(type(layer.get("layer_index")) is int and layer["layer_index"] == index and layer.get("layer_type") == (
                "full_attention" if full else "linear_attention") and isinstance(groups, dict) and set(groups) == set(expected_groups)
                and all(isinstance(groups[key], list) and len(groups[key]) == len(expected_groups[key])
                    and all(isinstance(value, str) for value in groups[key])
                    and set(groups[key]) == expected_groups[key] for key in expected_groups),
                "adapter_layer_schedule_mismatch")
    rows = inventory.get("targets")
    require(isinstance(rows, list) and len(rows) == 496, "adapter_target_dimensions_missing")
    actual = {}
    for row in rows:
        require(isinstance(row, dict) and row.get("name") in expected and row["name"] not in actual,
                "adapter_target_dimension_name_mismatch")
        dimensions = (row.get("in_features"), row.get("out_features"))
        require(all(type(value) is int for value in dimensions) and dimensions == expected[row["name"]],
                "adapter_target_dimensions_mismatch")
        actual[row["name"]] = dimensions
    digest = inventory.get("inventory_sha256")
    require(isinstance(digest, str) and digest == canonical({key: value for key,value in inventory.items()
            if key != "inventory_sha256"}), "adapter_inventory_digest_mismatch")
    return expected


def _pairs(rows):
    result = {}
    for key, value in rows:
        require(key not in result, "adapter_safetensors_duplicate_key")
        result[key] = value
    return result


def read_tensor_header(path):
    path = Path(path)
    try:
        with path.open("rb") as stream:
            prefix = stream.read(8)
            require(len(prefix) == 8, "adapter_safetensors_header_invalid")
            length = int.from_bytes(prefix, "little")
            require(2 <= length <= 2 * 1024 * 1024, "adapter_safetensors_header_invalid")
            header = json.loads(stream.read(length), object_pairs_hook=_pairs,
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        require(isinstance(header, dict), "adapter_safetensors_header_invalid")
        header.pop("__metadata__", None)
        return header, path.stat().st_size - 8 - length
    except (OSError, ValueError, TypeError, RecursionError) as error:
        if isinstance(error, AdapterContractError):
            raise
        raise AdapterContractError("adapter_safetensors_header_invalid") from None


def verify_adapter_contract(inventory, adapter_config, safetensors_path):
    expected = validate_inventory(inventory)
    require(isinstance(adapter_config, dict) and adapter_config.get("peft_type") == "LORA",
            "adapter_lora_configuration_required")
    names = adapter_config.get("target_modules")
    require(isinstance(names, list) and all(isinstance(name, str) for name in names) and len(names) == len(set(names))
            and (set(names) == LEAF_TARGETS or set(names) == set(expected)),
            "adapter_serialized_target_selector_mismatch")
    rank = adapter_config.get("r")
    empty_selectors = all(adapter_config.get(key) is None
        or (type(adapter_config[key]) in {list, dict} and len(adapter_config[key]) == 0)
        for key in ("target_parameters", "layer_replication", "exclude_modules", "layers_pattern",
                    "trainable_token_indices", "modules_to_save", "rank_pattern", "alpha_pattern"))
    require(type(rank) is int and rank == 8 and empty_selectors
            and type(adapter_config.get("lora_alpha")) is int and adapter_config["lora_alpha"] == 16
            and adapter_config.get("bias", "none") == "none"
            and adapter_config.get("fan_in_fan_out", False) is False
            and adapter_config.get("use_dora", False) is False
            and adapter_config.get("use_rslora", False) is False
            and adapter_config.get("use_qalora", False) is False
            and adapter_config.get("lora_bias", False) is False
            and adapter_config.get("layers_to_transform") is None,
            "adapter_rank_or_auxiliary_parameters_unsupported")
    header, data_bytes = read_tensor_header(safetensors_path)
    shapes = {}
    for name, (incoming, outgoing) in expected.items():
        prefix = "base_model.model." + name
        shapes[prefix + ".lora_A.weight"] = [rank, incoming]
        shapes[prefix + ".lora_B.weight"] = [outgoing, rank]
    require(set(header) == set(shapes), "adapter_serialized_tensor_topology_mismatch")
    ranges, elements = [], 0
    for name, shape in shapes.items():
        row = header[name]
        require(isinstance(row, dict) and set(row) == {"dtype", "shape", "data_offsets"}
                and row.get("shape") == shape and all(type(value) is int for value in row["shape"]),
                "adapter_serialized_tensor_shape_mismatch")
        dtype = row.get("dtype")
        size = {"BF16": 2, "F16": 2, "F32": 4}.get(dtype) if isinstance(dtype, str) else None
        offsets = row.get("data_offsets")
        require(size is not None and isinstance(offsets, list) and len(offsets) == 2
                and all(type(value) is int and value >= 0 for value in offsets)
                and offsets[1] - offsets[0] == prod(shape) * size and offsets[1] <= data_bytes,
                "adapter_serialized_tensor_offsets_mismatch")
        ranges.append(tuple(offsets))
        elements += prod(shape)
    cursor = 0
    for start, stop in sorted(ranges):
        require(start == cursor and stop > start, "adapter_serialized_tensor_storage_overlap_or_gap")
        cursor = stop
    require(cursor == data_bytes, "adapter_serialized_tensor_storage_extra_bytes")
    return {"schema_version": "forte.qwen38.adapter-topology.v1", "verified": True,
        "full_target_count": len(expected), "serialized_leaf_selector_count": len(LEAF_TARGETS),
        "serialized_tensor_count": len(header), "rank": rank, "trainable_parameters": elements,
        "serialized_target_selector_count": len(names),
        "serialized_target_selector_form": "leaf_suffixes" if set(names) == LEAF_TARGETS else "qualified_paths",
        "language_layers": 64, "linear_layers": 48, "full_attention_layers": 16,
        "vision_aligner_head_target_count": 0, "inventory_sha256": inventory["inventory_sha256"],
        "tensor_values_inspected": False, "model_loaded": False}
