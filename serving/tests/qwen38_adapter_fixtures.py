"""Synthetic sparse safetensors metadata; payloads are generated zero bytes."""

from io import BytesIO
import json
import tarfile

from serving.qwen38_adapter_contract import FULL, LINEAR, MLP, LEAF_TARGETS, canonical, expected_targets


def inventory_fixture():
    targets = expected_targets()
    inventory = {"model_type": "qwen3_5", "coverage": "all_configured_language_layers",
        "language_layer_count": 64, "linear_attention_layer_count": 48, "full_attention_layer_count": 16,
        "vision_and_aligner_target_count": 0, "target_count": 496,
        "groups": {"self_attn": 64, "linear_attn": 240, "mlp": 192},
        "target_modules": sorted(targets), "targets": [{"name": key, "in_features": dimensions[0],
            "out_features": dimensions[1]} for key,dimensions in sorted(targets.items())],
        "layers": [{"layer_index": index, "layer_type": "full_attention" if index%4 == 3 else "linear_attention",
            "target_groups": {"self_attn" if index%4 == 3 else "linear_attn": list(FULL if index%4 == 3 else LINEAR),
                              "mlp": list(MLP)}} for index in range(64)]}
    inventory["inventory_sha256"] = canonical(inventory)
    return inventory


def adapter_config_fixture(base="Qwen/Qwen3.8-27B"):
    return {"peft_type": "LORA", "base_model_name_or_path": base,
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "r": 8,
        "target_modules": sorted(LEAF_TARGETS), "bias": "none", "lora_alpha": 16}


def tensor_header_fixture():
    result, cursor = {}, 0
    for name, (incoming, outgoing) in sorted(expected_targets().items()):
        for side, shape in (("A", [8, incoming]), ("B", [outgoing, 8])):
            size = shape[0]*shape[1]*2
            result["base_model.model." + name + ".lora_" + side + ".weight"] = {
                "dtype": "BF16", "shape": shape, "data_offsets": [cursor, cursor+size]}
            cursor += size
    return result, cursor


def write_adapter_fixture(path, header=None, payload_size=None):
    default, default_size = tensor_header_fixture()
    header = default if header is None else header
    payload_size = default_size if payload_size is None else payload_size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    with path.open("wb") as stream:
        stream.write(len(encoded).to_bytes(8, "little"))
        stream.write(encoded)
        stream.seek(payload_size-1, 1)
        stream.write(b"\x00")


def archive_fixture(path, overrides=None):
    files = {name: ("# generated " + name).encode() for name in (
        "training/architectures.py", "training/data.py", "src/forte/pipeline.py",
        "rules/catalog.json", "annotations/taxonomy.v1.json")}
    files.update(overrides or {})
    with tarfile.open(path, "w") as archive:
        for name, raw in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, BytesIO(raw))
