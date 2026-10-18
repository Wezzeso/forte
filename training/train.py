"""Small, single-GPU local LoRA experiment. No model downloads or remote logging."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .architectures import (inventory_language_lora_targets, resolve_architecture,
                            resolve_training_classes, validate_adapter_gradients,
                            validate_trainable_adapter)
from .data import VideoCompletionCollator
from .guards import (TrainingGateError, canonical_sha256, file_sha256, load_annotations,
                     load_json, require, validate_target_scope, validate_training_inputs)


def write_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_inputs(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    paths = {key: Path(config[key]).resolve() for key in
             ("train_manifest", "split_manifest", "annotations", "source_manifest", "scenario_groups",
              "exposure_registry", "exposure_inventory_status")}
    for path in paths.values():
        require(path.is_file(), "required_input_missing")
    hashes = {key + "_sha256": file_sha256(path) for key, path in paths.items()}
    split = load_json(paths["split_manifest"])
    manifest = load_json(paths["train_manifest"])
    records = validate_training_inputs(
        manifest, split, load_annotations(paths["annotations"]),
        split_hash=hashes["split_manifest_sha256"], annotations_hash=hashes["annotations_sha256"],
        media_root=Path(config["media_root"]).resolve(),
    )
    validate_target_scope(records, config.get("expected_fact_count", 19))
    require(split.get("source_manifest_sha256") == hashes["source_manifest_sha256"], "source_manifest_hash_mismatch")
    require(split.get("scenario_groups_sha256") == hashes["scenario_groups_sha256"], "scenario_groups_hash_mismatch")
    from scripts.split_gate import validate_locked_training_inputs
    inventory_override = config.get("source_inventory_path_override")
    audit = validate_locked_training_inputs(
        split, load_annotations(paths["source_manifest"]), load_json(paths["scenario_groups"]),
        expected_assignments_sha256=config.get("expected_assignments_sha256", ""),
        registry_path=paths["exposure_registry"], coverage_status_path=paths["exposure_inventory_status"],
        requested_content_hashes={record["content_sha256"] for record in records},
        requested_recording_ids={record["source_recording_id"] for record in records},
        max_snapshot_age_sec=float(config.get("max_exposure_snapshot_age_sec", 300)),
        source_inventory_path_override=Path(inventory_override).resolve() if inventory_override else None,
    )
    require(audit["ok"] is True, "a02_locked_exposure_training_gate_failed")
    return records, hashes


def check_local_model(config: dict[str, Any]) -> tuple[Path, Path]:
    import re

    model_dir, processor_dir = Path(config["model_dir"]).resolve(), Path(config["processor_dir"]).resolve()
    require(model_dir.is_dir() and processor_dir.is_dir(), "local_model_or_processor_missing")
    for field in ("base_revision", "processor_revision"):
        require(bool(re.fullmatch(r"[0-9a-f]{40}", str(config.get(field, "")))), "immutable_model_revision_required")
    require(config["base_revision"] == config["processor_revision"], "processor_revision_mismatch")
    model_config = load_json(model_dir / "config.json")
    profile = resolve_architecture(model_config.get("model_type"))
    require(config.get("model_type", profile.model_type) == profile.model_type, "configured_model_architecture_mismatch")
    require(model_config.get("architectures") == [profile.model_class_name], "model_class_declaration_mismatch")
    require(bool(list(model_dir.glob("*.safetensors"))), "local_model_weights_missing")
    require((processor_dir / "tokenizer_config.json").is_file(), "local_tokenizer_missing")
    require(model_dir == processor_dir, "training_requires_one_verified_snapshot")
    if profile.model_type == "qwen3_5":
        from serving.qwen38_identity import verify_candidate_base
        model_manifest = load_json(Path(config["model_manifest"]))
        require(model_manifest.get("revision") == config["base_revision"], "candidate_manifest_revision_mismatch")
        require(model_manifest.get("processor_revision") == config["processor_revision"],
                "candidate_processor_revision_mismatch")
        verify_candidate_base(model_dir, config["model_manifest"])
    else:
        from forte.model.manifest import ManifestError, load_manifest, verify_local_snapshot
        try:
            model_manifest = load_manifest(config["model_manifest"])
            require(model_manifest["revision"] == config["base_revision"], "baseline_manifest_revision_mismatch")
            require(model_manifest["processor_revision"] == config["processor_revision"], "baseline_processor_revision_mismatch")
            require(model_manifest.get("model_id") == "Qwen/Qwen3-VL-8B-Instruct", "training_model_identity_mismatch")
            verify_local_snapshot(model_dir, config["model_manifest"])
        except ManifestError:
            raise TrainingGateError("baseline_snapshot_verification_failed") from None
    return model_dir, processor_dir


def validate_model_processor(model: Any, processor: Any, model_type: str) -> dict[str, Any]:
    """Validate the instantiated pair, including the shared Qwen3-VL processor.

    Qwen3.8 uses the qwen3_5 model architecture but its pinned public files declare
    Qwen3VLProcessor. A made-up Qwen3_5Processor or a text-only class is not valid.
    """
    profile = resolve_architecture(model_type)
    require(model.__class__.__name__ == profile.model_class_name, "loaded_model_class_mismatch")
    require(getattr(model.config, "model_type", None) == model_type, "loaded_model_architecture_mismatch")
    require(processor.__class__.__name__ in profile.processor_class_names, "processor_architecture_mismatch")
    tokenizer = processor.tokenizer
    token_names = {"image_token_id": "<|image_pad|>", "video_token_id": "<|video_pad|>",
                   "vision_start_token_id": "<|vision_start|>", "vision_end_token_id": "<|vision_end|>"}
    token_ids = {}
    for field, token in token_names.items():
        expected = getattr(model.config, field, None)
        actual = tokenizer.convert_tokens_to_ids(token)
        require(type(expected) is int and type(actual) is int and actual == expected,
                "model_processor_token_mismatch")
        if field in {"image_token_id", "video_token_id"}:
            require(getattr(processor, field, None) == expected, "model_processor_token_mismatch")
        token_ids[field] = expected
    require(len(set(token_ids.values())) == len(token_ids), "model_visual_tokens_not_distinct")
    embeddings = model.get_input_embeddings().num_embeddings
    require(len(tokenizer) <= embeddings and all(0 <= value < embeddings for value in token_ids.values()),
            "model_tokenizer_vocabulary_mismatch")
    vision = model.config.vision_config
    for name in ("image_processor", "video_processor"):
        component = getattr(processor, name, None)
        require(component is not None, "visual_processor_missing")
        for processor_field, model_field in (("patch_size", "patch_size"),
                                             ("temporal_patch_size", "temporal_patch_size"),
                                             ("merge_size", "spatial_merge_size")):
            require(getattr(component, processor_field, None) == getattr(vision, model_field, None)
                    and type(getattr(vision, model_field, None)) is int,
                    "model_processor_patch_geometry_mismatch")
    return {"model_type": model_type, "model_class": profile.model_class_name,
            "processor_class": processor.__class__.__name__, "visual_token_ids": token_ids,
            "embedding_vocabulary_size": embeddings, "tokenizer_vocabulary_size": len(tokenizer),
            "processor_model_compatibility_verified": True}


def load_model_and_processor(config: dict[str, Any], model_dir: Path,
                             processor_dir: Path) -> tuple[Any, Any]:
    """Offline CUDA loader shared by the human-reviewed and separate SILVER runs.

    Calling this function allocates the GPU; callers must first pass their own
    data, runtime, exposure and exclusive-compute gates.
    """
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      WANDB_DISABLED="true", TOKENIZERS_PARALLELISM="false")
    import torch
    import transformers

    profile = resolve_architecture(load_json(model_dir / "config.json").get("model_type"))
    require(config.get("model_type", profile.model_type) == profile.model_type, "configured_model_architecture_mismatch")
    model_class, processor_class = resolve_training_classes(profile, transformers)
    require(torch.cuda.is_available(), "training_cuda_unavailable")
    require(torch.cuda.is_bf16_supported(), "training_bf16_unavailable")
    require(torch.version.cuda == "12.9", "training_cuda_stack_not_locked")
    capability = list(torch.cuda.get_device_capability(0))
    expected_capability = config.get("required_compute_capability")
    if expected_capability is not None:
        require(expected_capability == capability, "training_gpu_compute_capability_mismatch")
    quantization = config.get("quantization", "none")
    require(quantization in {"none", "nf4"}, "training_quantization_unsupported")
    kwargs = {"local_files_only": True, "trust_remote_code": False,
              "dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    bnb = None
    if quantization == "nf4":
        try:
            import bitsandbytes as bnb
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise TrainingGateError("training_nf4_dependency_unavailable") from None
        require(config.get("double_quantization", True) is True, "training_nf4_double_quantization_required")
        kwargs.update(quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True), device_map={"": 0})
    processor = processor_class.from_pretrained(processor_dir, local_files_only=True, trust_remote_code=False)
    model = model_class.from_pretrained(model_dir, **kwargs)
    if quantization == "none":
        model = model.to("cuda")
    else:
        require(getattr(model, "is_loaded_in_4bit", False) is True, "training_nf4_not_loaded")
        quantized = [module for _, module in model.named_modules() if isinstance(module, bnb.nn.Linear4bit)]
        require(bool(quantized), "training_nf4_modules_missing")
        require(all(getattr(module.weight, "quant_type", None) == "nf4"
                    and getattr(module.weight, "quant_state", None) is not None
                    and module.compute_dtype == torch.bfloat16 for module in quantized),
                "training_nf4_runtime_not_verified")
    receipt = validate_model_processor(model, processor, profile.model_type)
    receipt.update(quantization=quantization, nf4_runtime_verified=quantization == "nf4",
                   bitsandbytes_version=getattr(bnb, "__version__", None),
                   cuda_version=torch.version.cuda, gpu_compute_capability=capability,
                   attn_implementation="sdpa", hybrid_kernel_path_verified=False,
                   hybrid_kernel_forward_audit_required=profile.hybrid)
    model._forte_runtime_load_receipt = receipt
    return model, processor


def configure_adapter(model: Any, config: dict[str, Any]) -> tuple[Any, list[tuple[str, Any]]]:
    """Shared by the real run and the tiny random-model gradient smoke check."""
    from peft import LoraConfig, TaskType, get_peft_model

    model_type = getattr(model.config, "model_type", None)
    resolve_architecture(model_type)
    require(config.get("model_type", model_type) == model_type, "configured_model_architecture_mismatch")
    if getattr(model, "is_loaded_in_4bit", False):
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.requires_grad_(False)
    inventory = inventory_language_lora_targets(model, model_type)
    targets = inventory["target_modules"]
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=int(config.get("lora_rank", 8)),
        lora_alpha=int(config.get("lora_alpha", 16)), lora_dropout=float(config.get("lora_dropout", 0.05)),
        target_modules=targets, bias="none",
    ))
    trainable = validate_trainable_adapter(model, model_type, inventory=inventory)
    model._forte_lora_target_inventory = inventory
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model, trainable


def train(config: dict[str, Any], *, preflight_only: bool = False) -> dict[str, Any]:
    # Enforce offline before importing HF libraries. Disable all common experiment exporters.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      WANDB_DISABLED="true", TOKENIZERS_PARALLELISM="false")
    records, hashes = load_inputs(config)
    decision_path = Path(config.get("decision_config", "configs/decision/default.json"))
    from forte.decision import DecisionConfig
    from .alignment import provider_records
    try:
        decision_config = DecisionConfig(**load_json(decision_path))
    except Exception:
        raise TrainingGateError("training_decision_config_invalid") from None
    records, alignment = provider_records(records, config=decision_config,
                                           media_root=Path(config["media_root"]).resolve())
    hashes["decision_config_sha256"] = file_sha256(decision_path)
    hashes["training_alignment_sha256"] = alignment["training_alignment_sha256"]
    model_dir, processor_dir = check_local_model(config)
    hashes["model_manifest_sha256"] = file_sha256(Path(config["model_manifest"]))
    require(config.get("experiment_id") in {"E1", "E2"}, "experiment_id_invalid")
    if config["experiment_id"] == "E2":
        analysis_path = Path(config.get("e2_error_analysis", ""))
        require(analysis_path.is_file() and file_sha256(analysis_path) == config.get("e2_error_analysis_sha256"),
                "e2_requires_validation_error_analysis")
        analysis = load_json(analysis_path)
        require(analysis.get("justifies_e2") is True and analysis.get("baseline_comparison_completed") is True
                and analysis.get("selection_split") == "validation", "e2_not_justified_by_validation")
    require(config.get("microbatch_size", 1) == 1, "microbatch_must_be_one")
    require(config.get("media_format", "image_sequence") == "image_sequence", "training_serving_format_not_aligned")
    require(1 <= config.get("max_steps", 100) <= 500, "max_steps_out_of_bounds")
    require(not Path(config["output_dir"]).resolve().exists(), "output_directory_already_exists")
    require(1 <= config.get("lora_rank", 8) <= 32 and 1 <= config.get("lora_alpha", 16) <= 64,
            "lora_config_out_of_bounds")
    require(0 <= config.get("lora_dropout", 0.05) < 1 and 0 < config.get("learning_rate", 1e-5) <= 1e-3,
            "optimizer_config_out_of_bounds")
    output = Path(config["output_dir"]).resolve()
    require(not any(output.is_relative_to(root) for root in
                    (model_dir, processor_dir, Path(config["media_root"]).resolve())),
            "output_must_not_modify_source_inputs")
    if preflight_only:
        return {"status": "preflight_passed", "training_windows": len(records), **hashes,
                "model_revision": config["base_revision"],
                "model_type": load_json(model_dir / "config.json")["model_type"], "gpu_training_executed": False}

    from .runtime_check import verify_runtime
    runtime_lock = Path(config.get("runtime_lock", "training/runtime-lock.json"))
    verify_runtime(runtime_lock, gpu=False)
    hashes["runtime_lock_sha256"] = file_sha256(runtime_lock)

    import numpy as np
    import torch

    require(torch.cuda.is_available(), "training_cuda_unavailable")
    require(torch.cuda.is_bf16_supported(), "training_bf16_unavailable")
    require(torch.version.cuda == "12.9", "training_cuda_stack_not_locked")
    seed = int(config.get("seed", 20261017))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    model, processor = load_model_and_processor(config, model_dir, processor_dir)
    runtime_load = model._forte_runtime_load_receipt
    model, trainable = configure_adapter(model, config)
    collator = VideoCompletionCollator(processor, Path(config["media_root"]).resolve(),
                                      int(config.get("max_length", 4096)),
                                      media_format=config.get("media_format", "image_sequence"),
                                      max_pixels=int(config.get("max_pixels_per_image", 1048576)),
                                      model_type=runtime_load["model_type"])
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable],
                                 lr=float(config.get("learning_rate", 1e-5)), weight_decay=0.01)
    accumulation = int(config.get("gradient_accumulation_steps", 8))
    require(1 <= accumulation <= 64, "gradient_accumulation_out_of_bounds")
    epochs, max_steps = int(config.get("epochs", 2)), int(config.get("max_steps", 100))
    require(1 <= epochs <= 3, "epochs_out_of_bounds")
    total_microsteps = min(epochs * len(records), max_steps * accumulation)
    require(total_microsteps > 0, "no_training_steps")
    output = Path(config["output_dir"]).resolve()
    require(not output.exists(), "output_directory_already_exists")
    output.mkdir(parents=True)
    versions = {package: importlib.metadata.version(package)
                for package in ("torch", "transformers", "peft", "torchvision", "accelerate")}
    run = {"schema_version": "forte.training.run.v1", "status": "running", "experiment_id": config["experiment_id"],
           "started_at": datetime.now(timezone.utc).isoformat(), "seed": seed, **hashes,
           "config_sha256": canonical_sha256(config), "base_revision": config["base_revision"],
           "processor_revision": config["processor_revision"], "versions": versions,
           "runtime_load": runtime_load, "lora_target_inventory": model._forte_lora_target_inventory,
           "training_windows": len(records), "trainable_parameters": sum(p.numel() for _, p in trainable),
           "vision_encoder_frozen": True, "baseline_selectable": True, "selection": "unselected",
           "media_format": config.get("media_format", "image_sequence"),
           "compact_frame_ids": decision_config.compact_frame_ids,
           "concise_observations": decision_config.concise_observations,
           "target_scope": "full_19_fact_observation",
           "gpu_name": torch.cuda.get_device_name(0), "gpu_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
           "determinism": "seeded; CUDA determinism warnings may indicate non-bitwise kernels"}
    write_json(output / "run.json", run)
    write_json(output / "config.json", config)
    microstep = step = 0
    gradient_checked = False
    optimizer.zero_grad(set_to_none=True)
    try:
        with (output / "metrics.jsonl").open("w", encoding="utf-8") as metrics:
            for epoch in range(epochs):
                indices = list(range(len(records)))
                random.Random(seed + epoch).shuffle(indices)
                for index in indices:
                    if microstep >= total_microsteps:
                        break
                    batch = {name: tensor.to("cuda") for name, tensor in collator([records[index]]).items()}
                    group_start = (microstep // accumulation) * accumulation
                    group_size = min(accumulation, total_microsteps - group_start)
                    loss = model(**batch).loss
                    require(bool(torch.isfinite(loss).item()), "training_loss_nonfinite")
                    (loss / group_size).backward()
                    microstep += 1
                    if microstep % accumulation == 0 or microstep == total_microsteps:
                        gradients = [p.grad for _, p in trainable if p.grad is not None]
                        require(bool(gradients) and all(bool(torch.isfinite(g).all().item()) for g in gradients),
                                "training_gradient_missing_or_nonfinite")
                        require(any(bool(torch.count_nonzero(g).item()) for g in gradients), "training_gradient_zero")
                        require(not any(p.grad is not None for name, p in model.named_parameters() if "visual" in name),
                                "vision_encoder_received_gradient")
                        require(not any(p.grad is not None for _, p in model.named_parameters()
                                        if not p.requires_grad), "frozen_base_received_gradient")
                        gradient_proof = validate_adapter_gradients(model, inventory=model._forte_lora_target_inventory)
                        gradient_checked = True
                        norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
                        require(bool(torch.isfinite(norm).item()), "gradient_norm_nonfinite")
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        step += 1
                        metrics.write(json.dumps({"step": step, "epoch": epoch, "loss_last_microbatch": float(loss.item()),
                                                  "gradient_norm": float(norm.item())}) + "\n")
                        metrics.flush()
                if microstep >= total_microsteps:
                    break
        require(gradient_checked and step > 0, "training_gradient_check_missing")
        adapter = output / "adapter"
        model.save_pretrained(adapter, safe_serialization=True)
        # Verify serialized PEFT configuration and weights can actually be reloaded locally.
        from peft import PeftConfig, load_peft_weights

        PeftConfig.from_pretrained(adapter, local_files_only=True)
        reloaded = load_peft_weights(str(adapter), device="cpu", local_files_only=True)
        require(bool(reloaded) and all(bool(torch.isfinite(t).all()) for t in reloaded.values()),
                "serialized_adapter_invalid")
        model.load_adapter(str(adapter), adapter_name="roundtrip_check", is_trainable=False,
                           local_files_only=True)
        model.set_adapter("roundtrip_check")
        model.eval()
        with torch.no_grad():
            reload_batch = {name: tensor.to("cuda") for name, tensor in collator([records[0]]).items()}
            require(bool(torch.isfinite(model(**reload_batch).loss).item()), "reloaded_adapter_forward_failed")
        checksums = {path.name: file_sha256(path) for path in sorted(adapter.iterdir()) if path.is_file()}
        run.update(status="candidate_unselected", optimizer_steps=step, microsteps=microstep,
                   completed_at=datetime.now(timezone.utc).isoformat(), adapter_checksums=checksums,
                   adapter_tree_sha256=canonical_sha256(checksums), gradients_verified=True,
                   gradient_proof=gradient_proof,
                   peft_load_verified=True, serving_load_verified=False, validation_comparison="pending_A10")
        write_json(output / "run.json", run)
        return {"status": run["status"], "optimizer_steps": step, "training_windows": len(records),
                "adapter_tree_sha256": run["adapter_tree_sha256"], "selection": "unselected"}
    except Exception:
        # No exception string or record data is written; private failures stay inside the service.
        run.update(status="failed", optimizer_steps=step, microsteps=microstep)
        write_json(output / "run.json", run)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true", help="Validate local inputs; do not allocate a GPU")
    args = parser.parse_args()
    try:
        result = train(load_json(args.config), preflight_only=args.preflight)
        print(json.dumps(result, allow_nan=False))
        return 0
    except TrainingGateError as exc:
        print(json.dumps({"status": "blocked", "reason_code": str(exc)}))
        return 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "reason_code": "local_training_error",
                          "error_type": type(exc).__name__}))
        return 3


if __name__ == "__main__":
    sys.exit(main())
