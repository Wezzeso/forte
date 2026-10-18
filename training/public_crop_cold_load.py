"""Offline CPU binding proof for a completed, pinned public crop adapter.

Run in the configured digest-pinned runtime with no GPU devices and no network.
This loads the original BF16 base, not the training-time NF4 representation. It
does not open a dataset, instantiate a processor, execute a forward pass, or
establish GPU inference, accuracy, promotion eligibility, or NF4 equivalence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import time
from typing import Any

from training import public_crop_lora as crop
from training.guards import TrainingGateError, canonical_sha256
from training.specialists.common import (cli_error, file_sha256, load_json,
    pinned_json, require, sha, write_json)

SCHEMA = "forte.public-crop-lora.cpu-cold-load.v1"
TRAINING_SOURCE_SHA256 = "43a9153abd54d49fef7720a33ed0b3bedbbe65b5fc137c9c6bfdf6f9cf6b93a6"


def hidden_cuda_environment() -> None:
    # Fail a GPU-visible launch; never silently turn it into an asserted proof.
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and os.environ.get("NVIDIA_VISIBLE_DEVICES") in {"", "void", "none"},
            "public_crop_cold_load_gpu_must_be_hidden_before_launch")


def expected_targets() -> list[str]:
    return sorted(f"model.language_model.layers.{layer}.self_attn.{projection}"
                  for layer in range(28) for projection in ("q_proj", "k_proj", "v_proj", "o_proj"))


def verify_inputs(config_path: Path, expected_run_sha256: str) -> dict:
    """Authenticate artifacts only; deliberately never call training preflight."""
    config = crop.resolve_config(config_path)
    output = Path(config["output_dir"])
    run_path = output / "run.json"
    run = pinned_json(run_path, expected_run_sha256)
    require(config.get("schema_version") == crop.SCHEMA
            and config.get("runtime_versions") == crop.PINS
            and run.get("runtime_versions") == crop.PINS
            and run.get("schema_version") == crop.RUN_SCHEMA
            and run.get("state") == "trained_unselected"
            and run.get("training_completed") is True
            and run.get("memory_smoke_only") is False
            and type(run.get("optimizer_steps")) is int
            and run["optimizer_steps"] > 0
            and run["optimizer_steps"] == config.get("optimizer_steps"),
            "public_crop_cold_load_completed_fit_required")
    require(load_json(output / "config.json") == config
            and file_sha256(output / "config.json") == run.get("config_sha256")
            and run.get("model_id") == config.get("model_id") == crop.MODEL_ID
            and run.get("revision") == config.get("revision") == crop.REVISION
            and run.get("model_manifest_sha256") == config.get("expected_model_manifest_sha256")
            and sha(config.get("expected_dataset_lock_sha256"))
            and run.get("dataset_lock_sha256") == config["expected_dataset_lock_sha256"]
            and run.get("runtime_image_digest") == config.get("runtime_image_digest")
            and isinstance(config.get("runtime_image_digest"), str)
            and config["runtime_image_digest"].startswith("sha256:")
            and sha(config["runtime_image_digest"][7:]),
            "public_crop_cold_load_run_config_binding_mismatch")
    require(run.get("memory_recipe") == crop.memory_recipe(config)
            and file_sha256(Path(crop.__file__)) == TRAINING_SOURCE_SHA256,
            "public_crop_cold_load_training_source_changed")
    require(sha(run.get("adapter_before_sha256")) and sha(run.get("adapter_after_sha256"))
            and run["adapter_before_sha256"] != run["adapter_after_sha256"]
            and sha(run.get("frozen_before_sha256"))
            and run["frozen_before_sha256"] == run.get("frozen_after_sha256")
            and run.get("sealed_test_used_for_training_or_selection") is False
            and run.get("promotion_eligible") is False
            and run.get("target_modules") == expected_targets(),
            "public_crop_cold_load_training_proof_invalid")
    for name, field in (("evaluation-protocol.json", "evaluation_protocol_sha256"),
                        ("crop-manifest.json", "crop_manifest_sha256"),
                        ("step-proof.json", "step_proof_sha256")):
        require(sha(run.get(field)) and file_sha256(output / name) == run[field],
                "public_crop_cold_load_run_artifact_changed")
    protocol = load_json(output / "evaluation-protocol.json")
    require(protocol.get("source_sha256") == TRAINING_SOURCE_SHA256
            and protocol.get("dataset_lock_sha256") == config["expected_dataset_lock_sha256"],
            "public_crop_cold_load_protocol_binding_mismatch")
    model_dir, _ = crop.verify_model(config)
    crop.verify_adapter(output, run)
    adapter_config = load_json(output / "adapter" / "adapter_config.json")
    serialized_targets = adapter_config.get("target_modules")
    # PEFT 0.21.2 may minimize the full target list to equivalent suffixes when
    # saving. Only its exact four attention projections are admissible here;
    # verify_state independently checks every live full path and all 224 tensors.
    targets_valid = (isinstance(serialized_targets, list)
        and all(isinstance(target, str) for target in serialized_targets)
        and (sorted(serialized_targets) == expected_targets()
             or sorted(serialized_targets) == ["k_proj", "o_proj", "q_proj", "v_proj"]))
    require(adapter_config.get("peft_type") == "LORA"
            and adapter_config.get("task_type") == "CAUSAL_LM"
            and adapter_config.get("r") == config.get("lora_rank")
            and adapter_config.get("lora_alpha") == config.get("lora_alpha")
            and adapter_config.get("bias") == "none"
            and adapter_config.get("lora_dropout") == 0
            and adapter_config.get("inference_mode") is True
            and not adapter_config.get("modules_to_save")
            and not adapter_config.get("use_dora")
            and not adapter_config.get("use_rslora")
            and not adapter_config.get("rank_pattern")
            and not adapter_config.get("alpha_pattern")
            and targets_valid,
            "public_crop_cold_load_adapter_config_mismatch")
    paths = {"requested_config": config_path.resolve(), "run": run_path,
             "config": output / "config.json", "model_manifest": Path(config["model_manifest"]),
             "evaluation_protocol": output / "evaluation-protocol.json",
             "crop_manifest": output / "crop-manifest.json", "step_proof": output / "step-proof.json",
             "cold_load_source": Path(__file__)}
    for name in run["memory_recipe"]["source_files"]:
        paths["training_source:" + name] = Path(crop.__file__).parent / name
    # Record the helper's shared guard implementation as well as its own source.
    for name in ("guards.py", "specialists/common.py"):
        paths["helper_dependency:" + name] = Path(crop.__file__).parent / name
    for name in load_json(Path(config["model_manifest"]))["files"]:
        paths["snapshot:" + name] = model_dir / name
    for name in run["adapter_files"]:
        paths["adapter:" + name] = output / "adapter" / name
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    return {"config": config, "run": run, "model_dir": model_dir,
            "adapter_dir": output / "adapter", "paths": paths, "hashes": hashes}


def runtime() -> tuple:
    hidden_cuda_environment()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      WANDB_DISABLED="true", TOKENIZERS_PARALLELISM="false")
    for name, expected in crop.PINS.items():
        require(importlib.metadata.version(name) == expected,
                "public_crop_cold_load_runtime_version_mismatch:" + name)
    import torch
    require(not torch.cuda.is_initialized(), "public_crop_cold_load_cuda_already_initialized")
    from transformers import Qwen3VLForConditionalGeneration
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    require(not torch.cuda.is_initialized(), "public_crop_cold_load_import_initialized_cuda")
    return torch, Qwen3VLForConditionalGeneration, PeftModel, get_peft_model_state_dict, load_file


def verify_state(torch: Any, model: Any, saved: dict, actual: dict, run: dict) -> dict:
    keys = {f"base_model.model.{target}.lora_{side}.weight"
            for target in expected_targets() for side in ("A", "B")}
    require(set(saved) == set(actual) == keys, "public_crop_cold_load_adapter_keys_mismatch")
    live = {name for name, _ in model.named_parameters() if ".lora_" in name}
    expected_live = {name.replace(".weight", ".default.weight") for name in keys}
    require(live == expected_live, "public_crop_cold_load_live_adapter_keys_mismatch")
    modules = {name: module for name, module in model.named_modules()
               if hasattr(module, "lora_A") or hasattr(module, "lora_B")}
    require(set(modules) == {"base_model.model." + target for target in expected_targets()}
            and all(set(module.lora_A) == set(module.lora_B) == {"default"}
                    and module.active_adapters == ["default"]
                    and module.disable_adapters is False and not module.merged_adapters
                    for module in modules.values()),
            "public_crop_cold_load_adapter_layer_state_mismatch")
    parameters = list(model.parameters())
    require(bool(parameters) and all(p.device.type == "cpu" for p in parameters)
            and all(b.device.type == "cpu" for b in model.buffers()),
            "public_crop_cold_load_non_cpu_tensor")
    require(all(not p.requires_grad for p in parameters), "public_crop_cold_load_trainable_parameter")
    metadata, count = {}, 0
    for name in sorted(keys):
        expected, loaded = saved[name], actual[name]
        require(expected.device.type == loaded.device.type == "cpu"
                and expected.shape == loaded.shape and expected.dtype == loaded.dtype,
                "public_crop_cold_load_adapter_tensor_metadata_mismatch")
        require(bool(torch.isfinite(expected).all()) and bool(torch.isfinite(loaded).all())
                and torch.equal(expected.detach(), loaded.detach()),
                "public_crop_cold_load_adapter_values_mismatch")
        metadata[name] = {"shape": list(loaded.shape), "dtype": str(loaded.dtype)}
        count += loaded.numel()
    require(count == run.get("trainable_parameters"), "public_crop_cold_load_adapter_parameter_count_mismatch")
    require(not torch.cuda.is_initialized(), "public_crop_cold_load_cuda_initialized")
    return {"adapter_tensor_count": len(keys), "adapter_parameter_count": count,
            "tensor_metadata": metadata, "tensor_metadata_sha256": canonical_sha256(metadata),
            "keys_shapes_dtypes_values_exact": True, "all_parameters_and_buffers_cpu": True,
            "all_parameters_frozen": True, "cuda_initialized": False}


def cold_load(config_path: Path, expected_run_sha256: str, output: Path) -> dict:
    hidden_cuda_environment()
    require(not output.exists(), "public_crop_cold_load_output_exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"schema_version": SCHEMA, "state": "started", "phase": "verify_inputs",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "run_sha256": expected_run_sha256,
        "scope": "cpu_unquantized_bf16_base_and_saved_adapter_binding_only",
        "training_representation": "gpu_nf4_language_base_bf16_vision_with_lora",
        "inference_executed": False, "nf4_equivalence_verified": False,
        "gpu_inference_verified": False, "dataset_accessed": False, "processor_loaded": False,
        "sealed_test_accessed": False, "promotion_eligible": False}
    # Exclusive creation refuses both accidental overwrites and concurrent runs.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True)
    started = time.monotonic()
    try:
        checked = verify_inputs(config_path, expected_run_sha256)
        receipt.update(input_sha256=checked["hashes"], runtime_versions=crop.PINS,
            expected_runtime_image_digest=checked["config"]["runtime_image_digest"],
            image_identity_scope="config_expectation_requires_separate_launcher_image_receipt")
        receipt["phase"] = "runtime_imports"
        torch, base_class, peft_class, state_dict, load_file = runtime()
        receipt["phase"] = "base_cpu_load"
        model, loading = base_class.from_pretrained(checked["model_dir"], local_files_only=True,
            trust_remote_code=False, dtype=torch.bfloat16, device_map={"": "cpu"},
            attn_implementation="sdpa", output_loading_info=True)
        require(model.__class__.__name__ == crop.MODEL_CLASS
                and not getattr(model, "is_loaded_in_4bit", False)
                and not getattr(model, "is_loaded_in_8bit", False)
                and not getattr(model.config, "quantization_config", None),
                "public_crop_cold_load_base_architecture_mismatch")
        require(all(not loading.get(key) for key in
                    ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")),
                "public_crop_cold_load_base_loading_mismatch")
        parameters = list(model.parameters())
        require(bool(parameters) and all(p.device.type == "cpu" and p.dtype == torch.bfloat16
                                        for p in parameters), "public_crop_cold_load_base_dtype_or_device_mismatch")
        require(not torch.cuda.is_initialized(), "public_crop_cold_load_base_initialized_cuda")
        receipt["phase"] = "adapter_cpu_load"
        model = peft_class.from_pretrained(model, checked["adapter_dir"], adapter_name="default",
            is_trainable=False, local_files_only=True, torch_device="cpu", ephemeral_gpu_offload=False,
            autocast_adapter_dtype=True)
        model.eval()
        require(set(model.peft_config) == {"default"} and model.active_adapters == ["default"]
                and model.peft_config["default"].inference_mode,
                "public_crop_cold_load_adapter_not_active_for_inference")
        receipt["phase"] = "exact_adapter_tensor_binding"
        saved = load_file(str(checked["adapter_dir"] / "adapter_model.safetensors"), device="cpu")
        actual = state_dict(model, adapter_name="default", save_embedding_layers=False)
        receipt.update(verify_state(torch, model, saved, actual, checked["run"]))
        receipt["phase"] = "reverify_inputs"
        # Full inventory guards catch added files as well as changed file bytes.
        after = verify_inputs(config_path, expected_run_sha256)
        require(after["hashes"] == checked["hashes"], "public_crop_cold_load_inputs_changed_during_load")
        hidden_cuda_environment()
        require(not torch.cuda.is_initialized(), "public_crop_cold_load_cuda_initialized")
        receipt.update(state="passed", phase="complete", inputs_unchanged=True,
                       elapsed_seconds=time.monotonic() - started)
        write_json(output, receipt)
        return receipt
    except Exception as exc:
        receipt.update(state="failed", elapsed_seconds=time.monotonic() - started,
            error=str(exc) if isinstance(exc, TrainingGateError) else "public_crop_cold_load_failed",
            failure_diagnostic=crop.failure_diagnostic(exc, receipt["phase"]))
        write_json(output, receipt)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-run-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = cold_load(args.config, args.expected_run_sha256, args.output)
        print(json.dumps({"ok": True, "state": result["state"], "scope": result["scope"],
                          "receipt_sha256": file_sha256(args.output)}))
        return 0
    except Exception as exc:
        return cli_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
