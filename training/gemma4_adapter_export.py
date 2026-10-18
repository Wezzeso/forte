"""Offline Gemma 4 LoRA checkpoint inspection and optional BF16 export.

The default command reads metadata and hashes on the CPU. It neither imports
Torch nor loads model tensors. No media, prompts, labels or generated text are
read. A successful inspection is not a runtime or accuracy qualification.

Example on the rented host::

    python -m training.gemma4_adapter_export \
        --adapter-dir /ephemeral/forte/gemma4-ft/adapter \
        --base-dir /ephemeral/models/gemma4 \
        --base-receipt /ephemeral/forte/runs/gemma4-download.json

vLLM 0.31.0 supports Gemma4ForConditionalGeneration LoRA directly, so merging
is optional. Add ``--merge-output /new/local/path`` only when a separate BF16
copy is wanted. Merging always reloads the original BF16 base, never the NF4
training instance. All exports remain disabled and unqualified candidates.
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


BASE_REPO = "google/gemma-4-31B-it"
BASE_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
BASE_CONFIG_SHA256 = "e967dd38bc5cfd38bd09a995a7bf4a754075df2b46aba68f7fbb5a791e6d8dd1"
BASE_WEIGHTS = {
    "model-00001-of-00002.safetensors": "eeef8791537bc04f110967c513149e037d2a9ae97d49add7291ebfa62806bbfa",
    "model-00002-of-00002.safetensors": "018912220f559f7025d60333e0996183cd538aa77ad6f4988a89ce47be681f10",
}
RANK, ALPHA, DROPOUT = 8, 16, 0.05
_TENSOR = re.compile(r"base_model\.model\.(model\.language_model\.layers\.\d+\.self_attn\.(?:q_proj|v_proj))\.lora_([AB])\.weight")
_FLOAT_TYPES = {"BF16": (2, "H", 0x7F80), "F16": (2, "H", 0x7C00),
                "F32": (4, "I", 0x7F800000)}
SOURCE_REFERENCES = (
    "https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/vllm/model_executor/models/gemma4_mm.py",
    "https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/docs/features/lora.md",
    "https://huggingface.co/docs/peft/main/en/developer_guides/checkpoint",
    f"https://huggingface.co/google/gemma-4-31B-it/blob/{BASE_REVISION}/config.json",
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
    require(config.get("model_type") == "gemma4" and
            config.get("architectures") == ["Gemma4ForConditionalGeneration"],
            "base_architecture_mismatch")
    require(config.get("audio_config") is None, "base_audio_architecture_refused")
    require(not config.get("quantization_config"), "base_must_be_original_bf16")
    text = config.get("text_config")
    require(isinstance(text, dict), "base_text_config_missing")
    expected = {"num_hidden_layers": 60, "hidden_size": 5376,
                "num_attention_heads": 32, "num_key_value_heads": 16,
                "num_global_key_value_heads": 4, "head_dim": 256,
                "global_head_dim": 512, "num_kv_shared_layers": 0,
                "attention_k_eq_v": True}
    require(all(type(text.get(key)) is type(value) and text[key] == value
                for key, value in expected.items()), "base_text_geometry_mismatch")
    require(text.get("layer_types") == ["full_attention" if i % 6 == 5 else
            "sliding_attention" for i in range(60)], "base_layer_types_mismatch")
    return text


def expected_targets(text: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """Full-attention layers have K=V and therefore no separate V projection."""
    result: dict[str, tuple[int, int]] = {}
    for index, layer_type in enumerate(text["layer_types"]):
        prefix = f"model.language_model.layers.{index}.self_attn."
        head_dim = text["global_head_dim"] if layer_type == "full_attention" else text["head_dim"]
        result[prefix + "q_proj"] = (text["num_attention_heads"] * head_dim, text["hidden_size"])
        if layer_type != "full_attention":
            result[prefix + "v_proj"] = (text["num_key_value_heads"] * head_dim, text["hidden_size"])
    return result


def validate_base(base_dir: Path, receipt_path: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    base_dir = _regular_directory(base_dir, "base_directory_missing_or_linked")
    config_path = base_dir / "config.json"
    config = _read_json(config_path)
    require(file_sha256(config_path) == BASE_CONFIG_SHA256, "base_config_hash_mismatch")
    text = _validate_base_config(config)
    actual_files = {p.name for p in base_dir.glob("*.safetensors")}
    require(actual_files == set(BASE_WEIGHTS), "base_weight_inventory_mismatch")
    require(not any(base_dir.glob("*.bin")) and not any(base_dir.glob("*.pt")) and
            not any(base_dir.glob("*.pth")), "unexpected_base_weight_format")
    for name, expected_hash in BASE_WEIGHTS.items():
        path = base_dir / name
        require(path.is_file() and not path.is_symlink(), "base_weight_missing_or_linked")
        require(file_sha256(path) == expected_hash, "base_weight_hash_mismatch")
    manifest_hash = canonical_sha256(BASE_WEIGHTS)
    receipt_hash = None
    if receipt_path is not None:
        receipt = _read_json(receipt_path)
        require(receipt.get("schema") == "forte.local-model-download.v1" and
                receipt.get("complete") is True and receipt.get("repo") == BASE_REPO and
                receipt.get("revision") == BASE_REVISION and
                receipt.get("license") == "apache-2.0" and
                receipt.get("config_sha256") == BASE_CONFIG_SHA256 and
                receipt.get("weight_manifest_sha256") == manifest_hash and
                receipt.get("weight_file_count") == len(BASE_WEIGHTS) and
                receipt.get("weight_bytes") == sum((base_dir / name).stat().st_size for name in BASE_WEIGHTS),
                "base_download_receipt_mismatch")
        receipt_hash = file_sha256(receipt_path)
    return text, {"repo": BASE_REPO, "revision": BASE_REVISION,
                  "config_sha256": BASE_CONFIG_SHA256, "weights": dict(BASE_WEIGHTS),
                  "weight_manifest_sha256": manifest_hash,
                  "weight_hashes_rechecked": True, "download_receipt_sha256": receipt_hash}


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
            len(target_list) == len(set(target_list)) and set(target_list) == set(targets),
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
    config_path = adapter_dir / "adapter_config.json"
    _validate_adapter_config(_read_json(config_path), targets)
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
    config_hash, weights_hash = file_sha256(config_path), file_sha256(weights_path)
    return {"config_sha256": config_hash, "weights_sha256": weights_hash,
            "tensor_keys_sha256": canonical_sha256([item["key"] for item in tensor_metadata]),
            "tensors_sha256": canonical_sha256(tensor_metadata),
            "parameter_count": sum(math.prod(item["shape"]) for item in tensor_metadata),
            "tensor_count": len(tensor_metadata), "target_module_count": len(targets),
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
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    if device == "cuda:0":
        require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "merge_bf16_cuda_unavailable")
    # This explicit, local BF16 load is independent of the quantized training process.
    model = Gemma4ForConditionalGeneration.from_pretrained(
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
        "schema": "forte.gemma4-adapter-export.v1",
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
        "serving": {"vllm_version": "0.31.0", "base_served_name": "gemma4",
                    "adapter_served_name": "gemma4-ft", "request_model": "gemma4-ft",
                    "additional_arguments": ["--enable-lora", "--max-lora-rank", str(RANK),
                        "--max-loras", "1", "--lora-modules", "gemma4-ft=" + str(adapter_dir.resolve())],
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
    parser.add_argument("--base-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--merge-output", type=Path, help="Explicit optional new local BF16 directory")
    parser.add_argument("--merge-device", choices=("cpu", "cuda:0"), default="cpu")
    args = parser.parse_args(argv)
    output = args.output or args.adapter_dir / "forte-adapter-export.json"
    output_safe = False
    try:
        protected = {args.adapter_dir / "adapter_config.json", args.adapter_dir / "adapter_model.safetensors",
                     args.base_dir / "config.json", *(args.base_dir / name for name in BASE_WEIGHTS)}
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
    failure = {"schema": "forte.gemma4-adapter-export.v1", "status": "failed",
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
