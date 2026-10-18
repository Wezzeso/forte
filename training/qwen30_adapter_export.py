"""Offline Qwen3-VL-30B-A3B LoRA checkpoint inspection and optional BF16 export.

The default command reads metadata and hashes on the CPU. It neither imports
Torch nor loads model tensors. No media, prompts, labels or generated text are
read. A successful inspection is not a runtime or accuracy qualification.

Example on the rented host::

    python -m training.qwen30_adapter_export \
        --adapter-dir /ephemeral/forte/runs/qwen30-public-desk-lora-v1/adapter \
        --base-dir /ephemeral/models/q30 \
        --base-receipt /ephemeral/forte/runs/q30-download.json

vLLM 0.31.0 supports Qwen3VLMoeForConditionalGeneration LoRA directly, so merging
is optional. Add ``--merge-output /new/local/path`` only when a separate BF16
copy is wanted. Merging reloads the original verified BF16 snapshot. All
exports remain disabled and unqualified candidates.
"""
from __future__ import annotations

import argparse
import array
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
from typing import Any


BASE_REPO = "Qwen/Qwen3-VL-30B-A3B-Instruct"
BASE_REVISION = "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
BASE_CONFIG_SHA256 = "6129b3f54d517a18553cf7525ff4886be6831c5806fc094d632b8ef38d859912"
BASE_WEIGHT_MANIFEST_SHA256 = "11c0e92dfe4fe6cd9b534417e90a329ce5c2617397be02a5080c41adb43699df"
BASE_WEIGHT_BYTES = 62141618016

RANK, ALPHA, DROPOUT = 8, 16, 0.05
_TENSOR = re.compile(r"base_model\.model\.(model\.language_model\.layers\.\d+\.self_attn\.(?:q_proj|v_proj))\.lora_([AB])\.weight")
_FLOAT_TYPES = {"BF16": (2, "H", 0x7F80), "F16": (2, "H", 0x7C00),
                "F32": (4, "I", 0x7F800000)}
SOURCE_REFERENCES = (
    "https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/vllm/model_executor/models/qwen3_vl_moe.py",
    "https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/docs/features/lora.md",
    "https://huggingface.co/docs/peft/main/en/developer_guides/checkpoint",
    f"https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/blob/{BASE_REVISION}/config.json",
)


class AdapterExportError(ValueError):
    """A fixed failure code; never includes input text or exception content."""


def require(condition: Any, code: str) -> None:
    if not condition:
        raise AdapterExportError(code)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "json_duplicate_key")
        result[key] = value
    return result


def _invalid_constant(_: str) -> None:
    raise AdapterExportError("json_nonfinite_number")


def _json_bytes(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(data, object_pairs_hook=_unique_object,
                           parse_constant=_invalid_constant)
    except (UnicodeError, json.JSONDecodeError):
        raise AdapterExportError("invalid_json") from None
    require(isinstance(value, dict), "json_object_required")
    return value


def _read_json(path: Path, maximum_bytes: int = 2 * 1024 * 1024) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), "json_file_missing_or_linked")
    require(0 < path.stat().st_size <= maximum_bytes, "json_file_size_invalid")
    return _json_bytes(path.read_bytes())


def write_receipt(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), "receipt_symlink_refused")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _regular_directory(path: Path, code: str) -> Path:
    require(path.is_dir() and not path.is_symlink(), code)
    return path.resolve()


def _validate_base_config(config: dict[str, Any]) -> dict[str, Any]:
    require(config.get("model_type") == "qwen3_vl_moe" and
            config.get("architectures") == ["Qwen3VLMoeForConditionalGeneration"],
            "base_architecture_mismatch")
    require(not config.get("quantization_config") and not config.get("auto_map"),
            "base_must_be_original_bf16")
    text = config.get("text_config")
    require(isinstance(text, dict), "base_text_config_missing")
    expected = {"num_hidden_layers": 48, "hidden_size": 2048,
                "num_attention_heads": 32, "num_key_value_heads": 4,
                "head_dim": 128, "num_experts": 128, "num_experts_per_tok": 8}
    require(all(type(text.get(key)) is type(value) and text[key] == value
                for key, value in expected.items()), "base_text_geometry_mismatch")
    require(text.get("dtype") == "bfloat16" and text.get("model_type") == "qwen3_vl_moe_text",
            "base_text_architecture_or_dtype_mismatch")
    return text


def expected_targets(text: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """Only attention Q/V; no MoE experts, router, vision, embeddings or head."""
    result: dict[str, tuple[int, int]] = {}
    for index in range(text["num_hidden_layers"]):
        prefix = f"model.language_model.layers.{index}.self_attn."
        result[prefix + "q_proj"] = (text["num_attention_heads"] * text["head_dim"], text["hidden_size"])
        result[prefix + "v_proj"] = (text["num_key_value_heads"] * text["head_dim"], text["hidden_size"])
    return result


def validate_base(base_dir: Path, receipt_path: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    base_dir = _regular_directory(base_dir, "base_directory_missing_or_linked")
    require(receipt_path is not None, "base_download_receipt_required")
    receipt = _read_json(receipt_path)
    require(receipt.get("schema") == "forte.local-model-download.v1" and
            receipt.get("complete") is True and receipt.get("repo") == BASE_REPO and
            receipt.get("revision") == BASE_REVISION and receipt.get("license") == "apache-2.0" and
            receipt.get("gated") is False and receipt.get("dtype") == "bfloat16" and
            receipt.get("architectures") == ["Qwen3VLMoeForConditionalGeneration"] and
            receipt.get("weight_hashes_verified_against_upstream_lfs") is True,
            "base_download_identity_mismatch")
    config_path, index_path = base_dir / "config.json", base_dir / "model.safetensors.index.json"
    config = _read_json(config_path)
    require(file_sha256(config_path) == BASE_CONFIG_SHA256 == receipt.get("config_sha256"),
            "base_config_hash_mismatch")
    text = _validate_base_config(config)
    index = _read_json(index_path, maximum_bytes=32 * 1024 * 1024)
    require(file_sha256(index_path) == receipt.get("index_sha256"), "base_index_hash_mismatch")
    manifest = receipt.get("weight_manifest")
    require(isinstance(manifest, list) and len(manifest) == 13, "base_weight_manifest_missing")
    require(canonical_sha256(manifest) == receipt.get("weight_manifest_sha256") == BASE_WEIGHT_MANIFEST_SHA256,
            "base_weight_manifest_hash_mismatch")
    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for row in manifest:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "base_weight_manifest_entry_invalid")
        name, size, digest = row["name"], row["bytes"], row["sha256"]
        require(isinstance(name, str) and re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", name),
                "base_weight_filename_invalid")
        require(name not in hashes and type(size) is int and size > 0 and
                isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
                "base_weight_manifest_entry_invalid")
        hashes[name], sizes[name] = digest, size
    require(receipt.get("weight_file_count") == len(hashes) and
            receipt.get("weight_bytes") == sum(sizes.values()) == BASE_WEIGHT_BYTES,
            "base_weight_count_or_size_mismatch")
    require(list(hashes) == sorted(hashes) and set(hashes) == {
                f"model-{number:05d}-of-{len(hashes):05d}.safetensors" for number in range(1, len(hashes) + 1)},
            "base_shard_sequence_mismatch")
    require({p.name for p in base_dir.glob("*.safetensors")} == set(hashes), "base_weight_inventory_mismatch")
    require(not any(base_dir.glob("*.bin")) and not any(base_dir.glob("*.pt")) and
            not any(base_dir.glob("*.pth")), "unexpected_base_weight_format")
    weight_map = index.get("weight_map")
    require(isinstance(weight_map, dict) and bool(weight_map) and
            set(weight_map.values()) == set(hashes), "base_weight_index_inventory_mismatch")
    for name, expected_hash in hashes.items():
        path = base_dir / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == sizes[name],
                "base_weight_missing_linked_or_wrong_size")
        require(file_sha256(path) == expected_hash, "base_weight_hash_mismatch")
        metadata = base_dir / ".cache/huggingface/download" / (name + ".metadata")
        require(metadata.is_file() and not metadata.is_symlink(), "base_download_metadata_missing")
        lines = metadata.read_text(encoding="utf-8").splitlines()
        require(len(lines) >= 2 and lines[0].strip() == BASE_REVISION and
                lines[1].strip().strip('"') == expected_hash, "base_download_metadata_mismatch")
    return text, {"repo": BASE_REPO, "revision": BASE_REVISION,
                  "config_sha256": BASE_CONFIG_SHA256, "index_sha256": receipt["index_sha256"],
                  "weights": hashes, "weight_manifest_sha256": receipt["weight_manifest_sha256"],
                  "weight_bytes": sum(sizes.values()), "weight_hashes_rechecked": True,
                  "source_chain": "pinned_HF_revision_upstream_LFS_receipt_and_local_download_metadata",
                  "download_receipt_sha256": file_sha256(receipt_path)}


def _validate_adapter_config(config: dict[str, Any], targets: dict[str, tuple[int, int]]) -> None:
    require(config.get("base_model_name_or_path") == BASE_REPO and
            config.get("revision") == BASE_REVISION, "adapter_base_identity_mismatch")
    require(config.get("peft_type") == "LORA" and config.get("task_type") == "CAUSAL_LM",
            "adapter_type_mismatch")
    require(type(config.get("r")) is int and config["r"] == RANK, "adapter_rank_mismatch")
    require(type(config.get("lora_alpha")) in (int, float) and config["lora_alpha"] == ALPHA,
            "adapter_alpha_mismatch")
    require(type(config.get("lora_dropout")) in (int, float) and config["lora_dropout"] == DROPOUT,
            "adapter_dropout_mismatch")
    require(config.get("bias") == "none" and config.get("fan_in_fan_out", False) is False,
            "adapter_bias_or_orientation_refused")
    require(config.get("inference_mode") is True, "adapter_inference_mode_required")
    require(config.get("init_lora_weights", True) is True, "adapter_initialization_variant_refused")
    target_list = config.get("target_modules")
    require(isinstance(target_list, list) and all(isinstance(x, str) for x in target_list) and
            len(target_list) == len(set(target_list)) and
            (set(target_list) == set(targets) or set(target_list) == {"q_proj", "v_proj"}),
            "adapter_target_scope_mismatch")
    for key in ("modules_to_save", "rank_pattern", "alpha_pattern", "layer_replication",
                "layers_to_transform", "layers_pattern", "target_parameters", "exclude_modules",
                "trainable_token_indices", "megatron_config", "loftq_config", "eva_config",
                "corda_config", "arrow_config", "alora_invocation_tokens", "auto_mapping"):
        require(config.get(key) in (None, [], {}), "adapter_extra_trainable_scope_refused")
    for key in ("use_dora", "use_rslora", "lora_bias", "use_qalora"):
        require(config.get(key, False) is False, "adapter_variant_refused")


def _read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    require(path.is_file() and not path.is_symlink(), "adapter_weights_missing_or_linked")
    size = path.stat().st_size
    require(size >= 10, "safetensors_file_too_short")
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        require(2 <= header_size <= 32 * 1024 * 1024 and header_size + 8 <= size,
                "safetensors_header_size_invalid")
        header = _json_bytes(stream.read(header_size))
    metadata = header.pop("__metadata__", {})
    require(isinstance(metadata, dict) and all(isinstance(k, str) and isinstance(v, str)
            for k, v in metadata.items()), "safetensors_metadata_invalid")
    require(metadata.get("format", "pt") == "pt", "safetensors_framework_mismatch")
    require(bool(header), "safetensors_empty")
    return header, header_size + 8


def _tensor_digest(stream: Any, offset: int, length: int, dtype: str) -> str:
    """Check finite IEEE values while hashing; never reconstructs model tensors."""
    digest = hashlib.sha256()
    element_size, array_type, exponent_mask = _FLOAT_TYPES[dtype]
    stream.seek(offset)
    remaining = length
    while remaining:
        block = stream.read(min(remaining, 1024 * 1024))
        require(bool(block) and len(block) % element_size == 0, "safetensors_truncated_tensor")
        digest.update(block)
        bits = array.array(array_type)
        bits.frombytes(block)
        if sys.byteorder != "little":
            bits.byteswap()
        require(all(value & exponent_mask != exponent_mask for value in bits),
                "adapter_nonfinite_weight")
        remaining -= len(block)
    return digest.hexdigest()


def validate_adapter(adapter_dir: Path, text_config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    adapter_dir = _regular_directory(adapter_dir, "adapter_directory_missing_or_linked")
    targets = expected_targets(text_config)
    require(len(targets) == 96, "adapter_target_inventory_mismatch")
    config_path = adapter_dir / "adapter_config.json"
    config = _read_json(config_path)
    _validate_adapter_config(config, targets)
    require({p.name for p in adapter_dir.glob("*.safetensors")} == {"adapter_model.safetensors"} and
            not any(adapter_dir.glob("*.bin")) and not any(adapter_dir.glob("*.pt")) and
            not any(adapter_dir.glob("*.pth")), "unexpected_adapter_weight_inventory")
    weights_path = adapter_dir / "adapter_model.safetensors"
    header, data_start = _read_safetensors_header(weights_path)
    require(len(header) == 2 * len(targets), "adapter_tensor_count_mismatch")
    tensor_metadata: list[dict[str, Any]] = []
    entries: list[tuple[int, int, str, dict[str, Any]]] = []
    seen: set[tuple[str, str]] = set()
    for key, tensor in header.items():
        match = _TENSOR.fullmatch(key)
        require(match is not None, "adapter_unexpected_tensor_key")
        target, side = match.groups()
        require(target in targets and (target, side) not in seen, "adapter_unexpected_tensor_target")
        seen.add((target, side))
        require(isinstance(tensor, dict) and set(tensor) == {"dtype", "shape", "data_offsets"},
                "safetensors_tensor_metadata_invalid")
        dtype, shape, offsets = tensor["dtype"], tensor["shape"], tensor["data_offsets"]
        require(isinstance(dtype, str) and dtype in _FLOAT_TYPES, "adapter_tensor_dtype_refused")
        rows, columns = targets[target]
        expected_shape = [RANK, columns] if side == "A" else [rows, RANK]
        require(isinstance(shape, list) and all(type(value) is int for value in shape) and
                shape == expected_shape, "adapter_tensor_shape_mismatch")
        require(isinstance(offsets, list) and len(offsets) == 2 and
                all(type(value) is int for value in offsets) and 0 <= offsets[0] < offsets[1],
                "safetensors_tensor_offsets_invalid")
        require(offsets[1] - offsets[0] == math.prod(shape) * _FLOAT_TYPES[dtype][0],
                "safetensors_tensor_length_mismatch")
        entries.append((offsets[0], offsets[1], key, tensor))
    require(seen == {(name, side) for name in targets for side in ("A", "B")},
            "adapter_tensor_pair_missing")
    cursor = 0
    with weights_path.open("rb") as stream:
        for begin, end, key, tensor in sorted(entries):
            require(begin == cursor, "safetensors_overlapping_or_unclaimed_data")
            digest = _tensor_digest(stream, data_start + begin, end - begin, tensor["dtype"])
            tensor_metadata.append({"key": key, "dtype": tensor["dtype"],
                                    "shape": tensor["shape"], "sha256": digest})
            cursor = end
    require(data_start + cursor == weights_path.stat().st_size, "safetensors_trailing_or_truncated_data")
    tensor_metadata.sort(key=lambda item: item["key"])
    parameter_count = sum(math.prod(item["shape"]) for item in tensor_metadata)
    require(parameter_count == 3342336, "adapter_parameter_count_mismatch")
    config_hash, weights_hash = file_sha256(config_path), file_sha256(weights_path)
    return {"config_sha256": config_hash, "weights_sha256": weights_hash,
            "tensor_keys_sha256": canonical_sha256([item["key"] for item in tensor_metadata]),
            "tensors_sha256": canonical_sha256(tensor_metadata),
            "parameter_count": parameter_count,
            "tensor_count": len(tensor_metadata), "target_module_count": len(targets),
            "target_declaration_format": "projection_suffixes" if
                set(config["target_modules"]) == {"q_proj", "v_proj"} else "full_module_paths",
            "rank": RANK, "alpha": ALPHA, "dropout": DROPOUT,
            "finite_weights_verified": True, "language_attention_only": True}, tensor_metadata


def _merge_bf16(base_dir: Path, adapter_dir: Path, output: Path, device: str) -> dict[str, Any]:
    require(device in {"cpu", "cuda:0"}, "merge_device_refused")
    output = output.absolute()
    require(not output.exists() and output.parent.is_dir(), "merge_output_must_be_new")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", WANDB_DISABLED="true",
                      TOKENIZERS_PARALLELISM="false")
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

    if device == "cuda:0":
        require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "merge_bf16_cuda_unavailable")
    # This local BF16 load is independent of the training process.
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        str(base_dir), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map={"": device}, attn_implementation="sdpa")
    require(not getattr(model, "is_loaded_in_4bit", False) and
            not getattr(model, "is_loaded_in_8bit", False), "quantized_base_merge_refused")
    adapted = PeftModel.from_pretrained(model, str(adapter_dir), local_files_only=True,
                                       is_trainable=False)
    merged = adapted.merge_and_unload(safe_merge=True)
    require(not any("lora_" in name for name, _ in merged.named_parameters()), "merge_adapter_residue")
    processor = AutoProcessor.from_pretrained(str(base_dir), local_files_only=True,
                                             trust_remote_code=False)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    merged.save_pretrained(str(temporary), safe_serialization=True, max_shard_size="8GB")
    processor.save_pretrained(str(temporary))
    require((temporary / "config.json").is_file() and bool(list(temporary.glob("*.safetensors"))),
            "merge_output_incomplete")
    hashes = {path.name: file_sha256(path) for path in sorted(temporary.iterdir()) if path.is_file()}
    temporary.rename(output)
    return {"status": "exported", "dtype": "bfloat16", "device": device,
            "files": hashes, "files_sha256": canonical_sha256(hashes),
            "runtime_load_verified": False, "qualification_status": "unqualified"}


def inspect_and_export(adapter_dir: Path, base_dir: Path, *, base_receipt: Path | None = None,
                       merge_output: Path | None = None, merge_device: str = "cpu") -> dict[str, Any]:
    text, base = validate_base(base_dir, base_receipt)
    adapter, tensors = validate_adapter(adapter_dir, text)
    root_hash = canonical_sha256({"adapter_config.json": adapter["config_sha256"],
                                  "adapter_model.safetensors": adapter["weights_sha256"]})
    result: dict[str, Any] = {
        "schema": "forte.qwen30-adapter-export.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "validated", "base": base, "adapter": adapter,
        "adapter_sha256": root_hash, "tensor_metadata": tensors,
        "requires_network": False, "local_files_only": True,
        "network_isolation_verified": False, "validation_device": "cpu", "default_enabled": False,
        "runtime_load_verified": False, "qualification_status": "unqualified",
        "heldout_accuracy_claim": False, "cross_validation_claim": False,
        "release_gate": {"independent_40_video_runs": 2, "minimum_correct_each": 37,
                         "false_positives_each": 0, "processed_each": 40,
                         "six_parallel_max_s": 285, "schema_and_evidence_required": True,
                         "user_promotion_approval_required": True},
        "serving": {"vllm_version": "0.31.0", "base_served_name": "q30test", "desk_adapter_model": "q30desk",
                    "adapter_request_scope": "DESK_SCHEMA_crops_only",
                    "adapter_fields_used": ["cash", "id_or_document"],
                    "adapter_served_name": "q30desk", "request_model": "q30test",
                    "adapter_request_model": "q30desk",
                    "additional_arguments": ["--enable-lora", "--max-lora-rank", str(RANK),
                        "--max-loras", "1", "--lora-modules", "q30desk=" + str(adapter_dir.resolve())],
                    "synthetic_schema_and_actual_adapter_load_required": True,
                    "startup_smoke_passed": False},
        "sources": list(SOURCE_REFERENCES), "merged": None,
    }
    if merge_output is not None:
        result["merged"] = _merge_bf16(base_dir.resolve(), adapter_dir.resolve(),
                                       merge_output, merge_device)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--base-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--merge-output", type=Path, help="Explicit optional new local BF16 directory")
    parser.add_argument("--merge-device", choices=("cpu", "cuda:0"), default="cpu")
    args = parser.parse_args(argv)
    output = args.output or args.adapter_dir / "forte-adapter-export.json"
    output_safe = False
    try:
        protected = {args.adapter_dir / "adapter_config.json", args.adapter_dir / "adapter_model.safetensors",
                     *(path for path in args.base_dir.iterdir() if path.is_file())} if args.base_dir.is_dir() else {
                     args.adapter_dir / "adapter_config.json", args.adapter_dir / "adapter_model.safetensors"}
        if args.base_receipt is not None:
            protected.add(args.base_receipt)
        require(output.resolve() not in {path.resolve() for path in protected}, "receipt_overwrites_input_refused")
        output_safe = True
        result = inspect_and_export(args.adapter_dir, args.base_dir, base_receipt=args.base_receipt,
                                    merge_output=args.merge_output, merge_device=args.merge_device)
        write_receipt(output, result)
        print(json.dumps({"status": result["status"], "adapter_sha256": result["adapter_sha256"],
                          "tensor_count": result["adapter"]["tensor_count"],
                          "parameter_count": result["adapter"]["parameter_count"],
                          "qualification_status": result["qualification_status"]}, sort_keys=True))
        return 0
    except AdapterExportError as error:
        failure_code = str(error)
    except Exception:
        # Third-party loader errors may include generated text, paths or secrets.
        failure_code = "adapter_export_operation_failed"
    failure = {"schema": "forte.qwen30-adapter-export.v1", "status": "failed",
               "failure_code": failure_code, "default_enabled": False,
               "qualification_status": "unqualified",
               "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    if output_safe:
        try:
            # Do not leave a stale successful receipt after this validation fails.
            write_receipt(output, failure)
        except Exception:
            pass
    print(json.dumps({"status": "failed", "failure_code": failure_code}, sort_keys=True))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
