"""LoRA gradient/roundtrip check on tiny RANDOM Qwen models and generated frames.

This is an API/gradient smoke check, never a trained checkpoint or accuracy result.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import tempfile
import time
from pathlib import Path

from .architectures import (capture_parameter_proof, resolve_architecture,
                           resolve_training_classes, validate_adapter_gradients,
                           verify_parameter_update)
from .data import SilverCompletionCollator, VideoCompletionCollator
from .guards import TrainingGateError, canonical_sha256, file_sha256, require
from .train import configure_adapter


def tiny_model_config(model_type: str) -> dict:
    """Two real decoder layers; the hybrid fixture exercises both layer families."""
    resolve_architecture(model_type)
    text = {"vocab_size": 128, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2,
            "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0,
                                "mrope_section": [1, 1, 2], "mrope_interleaved": True}}
    if model_type == "qwen3_5":
        text.update(layer_types=["linear_attention", "full_attention"],
                    linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
                    linear_num_key_heads=2, linear_num_value_heads=4, partial_rotary_factor=1.0)
        text["rope_parameters"]["partial_rotary_factor"] = 1.0
    return {"text_config": text,
            "vision_config": {"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
                              "out_hidden_size": 32, "patch_size": 16, "temporal_patch_size": 2,
                              "spatial_merge_size": 2, "num_position_embeddings": 64,
                              "deepstack_visual_indexes": [] if model_type == "qwen3_5" else [0]},
            "image_token_id": 9, "video_token_id": 10, "vision_start_token_id": 3, "vision_end_token_id": 4}


def check(processor_dir: Path, *, model_type: str = "qwen3_vl", device: str = "cpu",
          runtime_image_sha256: str | None = None) -> dict:
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    import torch
    import transformers
    from PIL import Image

    require(device in {"cpu", "cuda"}, "synthetic_device_invalid")
    require(device == "cpu" or re.fullmatch(r"[0-9a-f]{64}", runtime_image_sha256 or "") is not None,
            "synthetic_runtime_image_pin_required")
    if device == "cuda":
        require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "synthetic_cuda_bf16_unavailable")
        require(torch.version.cuda == "12.9", "synthetic_cuda_stack_mismatch")
        torch.cuda.reset_peak_memory_stats()
    profile = resolve_architecture(model_type)
    model_class, processor_class = resolve_training_classes(profile, transformers)
    torch.set_num_threads(2)
    torch.manual_seed(1)
    processor = processor_class.from_pretrained(processor_dir, local_files_only=True, trust_remote_code=False)
    require(processor.__class__.__name__ in profile.processor_class_names, "processor_architecture_mismatch")
    config_fields = tiny_model_config(model_type)
    config_class = getattr(transformers, "Qwen3_5Config" if model_type == "qwen3_5" else "Qwen3VLConfig")
    config = config_class(**config_fields)
    config._attn_implementation = "sdpa"
    model = model_class(config).to(device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    model, trainable = configure_adapter(model, {"model_type": model_type, "lora_rank": 2,
                                                "lora_alpha": 4, "lora_dropout": 0.0})
    inventory = model._forte_lora_target_inventory
    before = capture_parameter_proof(model, inventory=inventory)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="forte-tiny-random-") as temporary:
        root = Path(temporary)
        frames = []
        for index, pts in enumerate((0.25, 1.25)):
            path = root / f"synthetic-{index}.jpg"
            Image.new("RGB", (64, 64), (index * 100, 50, 150)).save(path)
            frames.append({"path": path.name, "frame_id": f"synthetic-{index}", "pts_sec": pts})
        row = {"frames": frames, "prompt": "Return requested synthetic fact JSON.",
               "target": {"facts": [{"fact_id": "synthetic", "state": "unclear", "supporting_frame_ids": []}],
                          "candidates": []}}
        if model_type == "qwen3_5":
            row.update(split="train", annotation_origin="local_model", human_reviewed=False,
                       target_scope="exact_anchor_frame", focal_frame_id=frames[0]["frame_id"],
                       fact_supervision_mask={"CASH_VISIBLE": True, "CASH_COUNT_VISIBLE": False},
                       target={"facts": [{"fact_id": "CASH_VISIBLE", "state": "present",
                                          "supporting_frame_ids": [frames[0]["frame_id"]]},
                                         {"fact_id": "CASH_COUNT_VISIBLE", "state": "unclear",
                                          "supporting_frame_ids": []}], "candidates": []})
        collator_class = SilverCompletionCollator if model_type == "qwen3_5" else VideoCompletionCollator
        batch = collator_class(processor, root, model_type=model_type, max_pixels=65536)([row])
        supervised_tokens = int((batch["labels"] != -100).sum().item())
        require(supervised_tokens > 0, "synthetic_completion_all_masked")
        # Preserve special visual IDs while mapping the large tokenizer vocabulary to 128
        # synthetic IDs. This model has random weights and cannot produce useful predictions.
        special = {processor.image_token_id: 9, processor.video_token_id: 10,
                   processor.tokenizer.convert_tokens_to_ids("<|vision_start|>"): 3,
                   processor.tokenizer.convert_tokens_to_ids("<|vision_end|>"): 4}
        for name in ("input_ids", "labels"):
            original = batch[name].clone()
            mapped = 11 + original.remainder(100)
            mapped[original == -100] = -100
            for source, replacement in special.items():
                mapped[original == source] = replacement
            batch[name] = mapped
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=1e-4)
        losses, gradient_proofs = [], []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            require(bool(torch.isfinite(loss).item()), "synthetic_loss_nonfinite")
            loss.backward()
            gradient_proofs.append(validate_adapter_gradients(model, inventory=inventory))
            optimizer.step()
            losses.append(float(loss.item()))
        parameter_proof = verify_parameter_update(model, before, optimizer_steps=2, inventory=inventory)
        adapter = root / "adapter"
        model.save_pretrained(adapter, safe_serialization=True)
        from peft import PeftConfig, load_peft_weights
        PeftConfig.from_pretrained(adapter, local_files_only=True)
        weights = load_peft_weights(str(adapter), device="cpu", local_files_only=True)
        require(bool(weights) and all(bool(torch.isfinite(value).all().item()) for value in weights.values()),
                "synthetic_serialized_adapter_invalid")
        model.load_adapter(str(adapter), adapter_name="roundtrip", is_trainable=False, local_files_only=True)
        model.set_adapter("roundtrip")
        model.eval()
        with torch.no_grad():
            require(bool(torch.isfinite(model(**batch).loss).item()), "synthetic_reload_forward_failed")
        return {"schema_version": "forte.training.tiny-visual-smoke.v1", "status": "passed",
                "fixture": "tiny_random_" + model_type + "_synthetic_only", "model_type": model_type,
                "optimizer_steps": 2, "runtime_image_sha256": runtime_image_sha256,
                "runtime_versions": {name: importlib.metadata.version(name)
                                     for name in ("torch", "transformers", "peft", "numpy", "pillow")},
                "collator_class": collator_class.__name__, "supervised_completion_tokens": supervised_tokens,
                "partial_silver_mask_exercised": model_type == "qwen3_5",
                "gpu_executed": device == "cuda", "finite_loss_verified": True,
                "gradients_verified": True, "gradient_proofs": gradient_proofs,
                "frozen_tensors_unchanged": parameter_proof["frozen_parameters_byte_identical"],
                "changed_adapter_tensors": len(parameter_proof["changed_adapter_tensors"]),
                "parameter_update_proof": parameter_proof, "lora_target_inventory": inventory,
                "losses": losses, "elapsed_seconds": time.monotonic() - started,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
                "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
                "gpu_compute_capability": list(torch.cuda.get_device_capability(0)) if device == "cuda" else None,
                "model_config_sha256": canonical_sha256(config_fields),
                "source_sha256": {name: file_sha256(Path(__file__).parent / name)
                                   for name in ("check_gradient.py", "architectures.py", "data.py", "train.py")},
                "processor_metadata_sha256": {path.name: file_sha256(path) for path in sorted(processor_dir.iterdir())
                                              if path.is_file() and path.suffix in {".json", ".jinja", ".txt"}},
                "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
                "vision_frozen": True, "peft_load_verified": True, "real_weights_used": False,
                "reload_forward_verified": True, "serving_load_verified": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--model-type", choices=("qwen3_vl", "qwen3_5"), default="qwen3_vl")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--runtime-image-sha256", help="Actual image digest, supplied by the host operator")
    parser.add_argument("--receipt", type=Path, help="Save the actual completed smoke receipt")
    args = parser.parse_args()
    try:
        result = check(args.processor_dir, model_type=args.model_type, device=args.device,
                       runtime_image_sha256=args.runtime_image_sha256)
        if args.receipt:
            from .train import write_json
            require(not args.receipt.exists(), "synthetic_receipt_already_exists")
            write_json(args.receipt, result)
        print(json.dumps(result))
        return 0
    except TrainingGateError as exc:
        print(json.dumps({"status": "blocked", "reason_code": str(exc)}))
        return 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "reason_code": "tiny_random_gradient_check_failed",
                          "error_type": type(exc).__name__}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
