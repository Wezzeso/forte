"""Offline, human-reviewed crop QA LoRA for the Qwen3-VL-30B-A3B MoE.

Preflight is CPU-only and imports neither torch nor transformers. A successful
preflight is not a training receipt. No model/data downloads or promotion occur.
See training/specialist-lora.md for manifests and commands.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
from typing import Any

from training.data import completion_labels
from training.guards import TrainingGateError
from training.specialists.common import (cli_error, file_sha256,
                                        load_json, number, pinned_json, relative_file,
                                        require, sha, write_json)
from training.specialists.prepare import verify_dataset

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
MODEL_TYPE = "qwen3_vl_moe"
MODEL_CLASS = "Qwen3VLMoeForConditionalGeneration"
PINS = {"torch": "2.13.0+cu129", "transformers": "5.17.0", "peft": "0.21.2",
        "accelerate": "1.15.0"}
PATH_FIELDS = ("model_dir", "model_manifest", "dataset_dir", "crop_manifest", "output_dir")
LINK_FIELDS = ("source_recording_sha256", "scenario_group", "camera_id", "split", "origin")
TARGET_RE = re.compile(r"model\.language_model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$")


def resolve_config(path: Path) -> dict:
    config = load_json(path)
    require(isinstance(config, dict), "crop_lora_config_object_required")
    for field in PATH_FIELDS:
        value = config.get(field)
        require(isinstance(value, str) and bool(value) and "://" not in value,
                "crop_lora_local_paths_required")
        config[field] = str((path.resolve().parent / value).resolve())
    return config


def verify_model(config: dict) -> tuple[Path, dict, dict]:
    """Verify an operator-reviewed immutable snapshot inventory, without fetching.

    The inventory pins locally reviewed bytes; it is not an upstream signature.
    All immediate snapshot files are inventoried so Transformers cannot pick an
    unverified tokenizer/chat template or alternative checkpoint.
    """
    require(config.get("model_id") == MODEL_ID and config.get("model_type") == MODEL_TYPE,
            "crop_lora_only_qwen3_vl_30b_moe_supported")
    revision = config.get("base_revision")
    require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision)
            and config.get("processor_revision") == revision, "crop_lora_immutable_revision_required")
    manifest_path = Path(config.get("model_manifest", "")).resolve()
    manifest = pinned_json(manifest_path, config.get("expected_model_manifest_sha256", ""))
    require(manifest.get("schema_version") == "forte.crop-lora.snapshot.v1"
            and manifest.get("model_id") == MODEL_ID and manifest.get("revision") == revision
            and manifest.get("processor_revision") == revision
            and manifest.get("license") == "apache-2.0" and manifest.get("license_verified") is True
            and manifest.get("reviewed_by") and manifest.get("reviewed_at"),
            "crop_lora_snapshot_identity_mismatch")
    root = Path(config.get("model_dir", "")).resolve()
    require(root.is_dir(), "crop_lora_model_directory_missing")
    files = manifest.get("files")
    require(isinstance(files, dict) and bool(files), "crop_lora_snapshot_inventory_required")
    for name, digest in files.items():
        require(isinstance(name, str) and name == Path(name).name and "\\" not in name
                and ":" not in name and sha(digest), "crop_lora_snapshot_entry_invalid")
        # HF cache snapshots commonly use symlinks into adjacent content-addressed blobs.
        artifact = root / name
        require(artifact.is_file() and file_sha256(artifact) == digest,
                "crop_lora_snapshot_hash_mismatch")
    actual_files = {p.name for p in root.iterdir() if p.is_file()}
    require(actual_files == set(files), "crop_lora_uninventoried_snapshot_file")
    require(not any(p.is_dir() for p in root.iterdir()), "crop_lora_snapshot_subdirectories_unsupported")
    required = {"config.json", "tokenizer_config.json", "tokenizer.json", "preprocessor_config.json",
                "chat_template.json", "model.safetensors.index.json"}
    require(required <= set(files), "crop_lora_snapshot_files_missing")
    model_config = load_json(root / "config.json")
    require(model_config.get("model_type") == MODEL_TYPE
            and model_config.get("architectures") == [MODEL_CLASS]
            and not model_config.get("quantization_config")
            and not model_config.get("auto_map"), "crop_lora_snapshot_architecture_mismatch")
    text = model_config.get("text_config", {})
    require(all(text.get(k) == v for k, v in {"model_type": "qwen3_vl_moe_text",
            "hidden_size": 2048, "num_hidden_layers": 48, "num_experts": 128,
            "num_experts_per_tok": 8}.items()), "crop_lora_snapshot_model_size_mismatch")
    index = load_json(root / "model.safetensors.index.json")
    weight_map = index.get("weight_map", {})
    require(isinstance(weight_map, dict) and bool(weight_map), "crop_lora_weight_index_invalid")
    shards = set(weight_map.values())
    require(all(isinstance(s, str) and s.endswith(".safetensors") for s in shards)
            and shards == {n for n in files if n.endswith(".safetensors")},
            "crop_lora_weight_inventory_mismatch")
    return root, manifest, model_config


def validate_crop_manifest(manifest: dict, lock: dict, expected_lock: str) -> list[dict]:
    require(manifest.get("schema_version") == "forte.crop-lora.data.v1"
            and manifest.get("frozen") is True and manifest.get("reviewed_by")
            and manifest.get("reviewed_at"), "crop_lora_human_data_review_required")
    require(manifest.get("dataset_lock_sha256") == expected_lock, "crop_lora_dataset_link_mismatch")
    sources = {r["sample_id"]: r for r in lock["records"]}
    rows = manifest.get("records")
    require(isinstance(rows, list) and bool(rows), "crop_lora_records_required")
    seen, result = set(), []
    for raw in rows:
        require(isinstance(raw, dict), "crop_lora_record_invalid")
        row = dict(raw)
        identifier = row.get("example_id")
        require(isinstance(identifier, str) and identifier and identifier not in seen,
                "crop_lora_example_id_invalid")
        seen.add(identifier)
        source = sources.get(row.get("source_sample_id"))
        require(source is not None, "crop_lora_source_sample_missing")
        require(source.get("origin") in {"own", "synthetic"}
                and source.get("split") in {"train", "validation"},
                "crop_lora_evaluation_source_forbidden")
        require(all(row.get(field) == source.get(field) for field in LINK_FIELDS),
                "crop_lora_source_lineage_mismatch")
        require(source.get("labels_reviewed") is True and source.get("exhaustive_labels") is True
                and source.get("label_reviewer") and source.get("label_reviewed_at")
                and row.get("human_reviewed") is True and row.get("reviewer") and row.get("reviewed_at"),
                "crop_lora_human_qa_review_required")
        box = row.get("crop_xyxy")
        require(isinstance(box, list) and len(box) == 4 and all(type(v) is int for v in box),
                "crop_lora_crop_box_invalid")
        x1, y1, x2, y2 = box
        require(0 <= x1 < x2 <= source["width"] and 0 <= y1 < y2 <= source["height"],
                "crop_lora_crop_box_outside_image")
        for field in ("question", "answer"):
            value = row.get(field)
            require(isinstance(value, str) and 0 < len(value.strip()) <= 4096
                    and "<|" not in value, "crop_lora_question_or_answer_invalid")
        # Only reviewed frame bytes supply pixels; no extra file or remote URL can enter the collator.
        row["_export_path"] = source["export_path"]
        result.append(row)
    require({r["split"] for r in result} == {"train", "validation"},
            "crop_lora_train_validation_qa_required")
    return result


def preflight(config: dict) -> dict:
    require(config.get("schema_version") == "forte.crop-lora.train.v1", "crop_lora_config_schema")
    require(config.get("runtime_versions") == PINS, "crop_lora_runtime_pins_required")
    for field, maximum in (("optimizer_steps", 100000), ("grad_accum_steps", 128),
                           ("max_length", 16384), ("lora_rank", 128), ("lora_alpha", 512),
                           ("max_pixels", 1048576)):
        require(type(config.get(field)) is int and 1 <= config[field] <= maximum,
                "crop_lora_training_parameter_invalid")
    require(type(config.get("seed")) is int and config["seed"] >= 0, "crop_lora_seed_required")
    require(number(config.get("learning_rate"), 1e-9, .01)
            and number(config.get("lora_dropout"), 0, .5)
            and number(config.get("minimum_free_cuda_gib"), 1, 1024), "crop_lora_training_parameter_invalid")
    require(type(config.get("merge_adapter")) is bool, "crop_lora_merge_choice_required")
    model_dir, snapshot, model_config = verify_model(config)
    geometry = model_config.get("vision_config", {})
    require(all(type(geometry.get(k)) is int and geometry[k] > 0
                for k in ("patch_size", "spatial_merge_size", "temporal_patch_size")),
            "crop_lora_snapshot_patch_geometry_invalid")
    require(config["max_pixels"] >= (geometry["patch_size"] * geometry["spatial_merge_size"]) ** 2,
            "crop_lora_pixel_budget_below_one_visual_token")
    root = Path(config.get("dataset_dir", "")).resolve()
    lock = verify_dataset(root, config.get("expected_dataset_lock_sha256", ""))
    manifest = pinned_json(Path(config.get("crop_manifest", "")).resolve(),
                           config.get("expected_crop_manifest_sha256", ""))
    rows = validate_crop_manifest(manifest, lock, config["expected_dataset_lock_sha256"])
    return {"model_dir": model_dir, "dataset_dir": root, "snapshot": snapshot,
            "model_config": model_config, "lock": lock, "rows": rows,
            "crop_manifest": manifest}


def inventory_targets(model: Any, layers: int) -> list[str]:
    targets = sorted(name for name, _ in model.named_modules() if TARGET_RE.fullmatch(name))
    expected = {f"model.language_model.layers.{i}.self_attn.{p}"
                for i in range(layers) for p in ("q_proj", "k_proj", "v_proj", "o_proj")}
    require(set(targets) == expected, "crop_lora_language_attention_inventory_mismatch")
    return targets


class CropQACollator:
    """One image/QA per microbatch with proven prompt-prefix completion masking."""

    def __init__(self, processor: Any, root: Path, model_config: dict, *, max_length: int,
                 max_pixels: int = 262144, tensor_factory: Any = None):
        self.processor, self.root, self.max_length = processor, root, max_length
        self.max_pixels, self.model_config = max_pixels, model_config
        self.visual_ids = {model_config[k] for k in ("image_token_id", "video_token_id",
                           "vision_start_token_id", "vision_end_token_id")}
        self.tensor_factory = tensor_factory

    def __call__(self, records: list[dict]) -> dict:
        require(len(records) == 1, "crop_lora_one_image_microbatch_required")
        from PIL import Image
        row = records[0]
        with Image.open(relative_file(self.root, row["_export_path"])) as image:
            crop = image.convert("RGB").crop(tuple(row["crop_xyxy"]))
        user = [{"role": "user", "content": [{"type": "image"},
                 {"type": "text", "text": row["question"]}]}]
        full = [*user, {"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]}]
        prompt_text = self.processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
        full_text = self.processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
        require(full_text.startswith(prompt_text), "crop_lora_chat_template_prefix_mismatch")
        kwargs = {"images": [crop], "padding": False, "truncation": False, "return_tensors": "pt",
                  "images_kwargs": {"size": {"longest_edge": self.max_pixels,
                      "shortest_edge": min(self.max_pixels, self.processor.image_processor.size["shortest_edge"])}}}
        prompt = self.processor(text=[prompt_text], **kwargs)
        batch = self.processor(text=[full_text], **kwargs)
        ids, prefix = batch["input_ids"][0].tolist(), prompt["input_ids"][0].tolist()
        require(len(ids) <= self.max_length, "crop_lora_sequence_too_long_no_truncation")
        labels = completion_labels(ids, prefix, self.visual_ids,
            attention_mask=batch["attention_mask"][0].tolist(),
            prompt_attention_mask=prompt["attention_mask"][0].tolist())
        require(bool(set(ids) & self.visual_ids) and "pixel_values" in batch and "image_grid_thw" in batch
                and "pixel_values_videos" not in batch and "video_grid_thw" not in batch,
                "crop_lora_image_only_inputs_required")
        require(all(key in prompt and batch[key].equal(prompt[key]) for key in ("pixel_values", "image_grid_thw")),
                "crop_lora_prompt_full_pixels_mismatch")
        grid, pixels = batch["image_grid_thw"], batch["pixel_values"]
        require(grid.ndim == 2 and tuple(grid.shape) == (1, 3), "crop_lora_image_grid_invalid")
        dimensions = grid.tolist()[0]
        require(all(type(v) is int and v > 0 for v in dimensions) and dimensions[0] == 1,
                "crop_lora_image_grid_invalid")
        _, height, width = dimensions
        visual = self.model_config["vision_config"]
        patch, merge = visual["patch_size"], visual["spatial_merge_size"]
        require(height % merge == 0 and width % merge == 0
                and height * width * patch**2 <= self.max_pixels, "crop_lora_visual_pixel_budget_exceeded")
        require(pixels.ndim == 2 and pixels.shape[0] == height * width
                and pixels.shape[1] == 3 * visual["temporal_patch_size"] * patch**2,
                "crop_lora_pixel_grid_mismatch")
        require(ids.count(self.model_config["image_token_id"]) == height * width // merge**2
                and self.model_config["video_token_id"] not in ids, "crop_lora_visual_token_grid_mismatch")
        if self.tensor_factory is None:
            import torch
            batch["labels"] = torch.tensor([labels], dtype=torch.long)
        else:
            batch["labels"] = self.tensor_factory([labels])
        return batch


def _runtime_model(config: dict, checked: dict) -> tuple[Any, Any]:
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      WANDB_DISABLED="true", TOKENIZERS_PARALLELISM="false")
    for package, expected in PINS.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            raise TrainingGateError("crop_lora_runtime_dependency_missing:" + package) from None
        require(actual == expected, "crop_lora_runtime_version_mismatch:" + package)
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    require(torch.cuda.is_available(), "crop_lora_cuda_unavailable")
    require(torch.cuda.is_bf16_supported(), "crop_lora_cuda_bf16_unavailable")
    free, _ = torch.cuda.mem_get_info(0)
    require(free >= config["minimum_free_cuda_gib"] * 1024**3, "crop_lora_insufficient_free_cuda_memory")
    model = AutoModelForImageTextToText.from_pretrained(checked["model_dir"], local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})
    processor = AutoProcessor.from_pretrained(checked["model_dir"], local_files_only=True,
        trust_remote_code=False, max_pixels=config["max_pixels"])
    require(model.__class__.__name__ == MODEL_CLASS
            and getattr(model.config, "model_type", None) == MODEL_TYPE
            and processor.__class__.__name__ == "Qwen3VLProcessor", "crop_lora_loaded_architecture_mismatch")
    require(all(p.device.type == "cuda" for p in model.parameters()), "crop_lora_model_not_fully_on_cuda")
    visual = checked["model_config"]["vision_config"]
    require(all(getattr(processor.image_processor, attr, None) == visual[key]
                for attr, key in (("patch_size", "patch_size"), ("merge_size", "spatial_merge_size"),
                                  ("temporal_patch_size", "temporal_patch_size"))),
            "crop_lora_processor_patch_geometry_mismatch")
    for field, token in (("image_token_id", "<|image_pad|>"), ("video_token_id", "<|video_pad|>"),
                         ("vision_start_token_id", "<|vision_start|>"), ("vision_end_token_id", "<|vision_end|>")):
        require(processor.tokenizer.convert_tokens_to_ids(token) == checked["model_config"].get(field),
                "crop_lora_processor_visual_token_mismatch")
    require(len(processor.tokenizer) <= model.get_input_embeddings().num_embeddings,
            "crop_lora_processor_vocabulary_mismatch")
    return model, processor


def train(config: dict) -> dict:
    checked = preflight(config)
    output = Path(config["output_dir"]).resolve()
    require(not output.exists(), "crop_lora_output_already_exists")
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "crop-manifest.json", checked["crop_manifest"])
    run = {"schema_version": "forte.crop-lora.run.v1", "state": "preparing", "model_id": MODEL_ID,
           "base_revision": config["base_revision"], "model_type": MODEL_TYPE,
           "model_manifest_sha256": config["expected_model_manifest_sha256"],
           "dataset_lock_sha256": config["expected_dataset_lock_sha256"],
           "crop_manifest_sha256": config["expected_crop_manifest_sha256"],
           "config_sha256": file_sha256(output / "config.json"),
           "approved_types": [], "promotion_eligible": False, "audio_used": False,
           "optimizer_steps": 0}
    write_json(output / "run.json", run)
    try:
        model, processor = _runtime_model(config, checked)
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        model.requires_grad_(False)
        targets = inventory_targets(model, checked["model_config"]["text_config"]["num_hidden_layers"])
        model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM,
            r=config["lora_rank"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
            target_modules=targets, bias="none"))
        trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        require(bool(trainable) and all(".lora_" in name and ".language_model.layers." in name
                and ".self_attn." in name for name, _ in trainable), "crop_lora_unexpected_trainable_parameters")
        baseline = {name: p.detach().cpu().clone() for name, p in trainable}
        model.config.use_cache = False
        model.config.text_config.use_cache = False
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        collator = CropQACollator(processor, checked["dataset_dir"], checked["model_config"],
                                  max_length=config["max_length"], max_pixels=config["max_pixels"])
        train_rows = [r for r in checked["rows"] if r["split"] == "train"]
        validation = [r for r in checked["rows"] if r["split"] == "validation"]
        order, cursor = list(range(len(train_rows))), 0
        random.shuffle(order)
        optimizer = torch.optim.AdamW([p for _, p in trainable], lr=config["learning_rate"])
        run.update(state="training", target_modules=targets,
                   trainable_parameters=sum(p.numel() for _, p in trainable), runtime_versions=PINS,
                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                   qa_counts=dict(Counter(r["split"] for r in checked["rows"])))
        write_json(output / "run.json", run)
        step_proofs = []
        model.train()
        for step in range(config["optimizer_steps"]):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(config["grad_accum_steps"]):
                row = train_rows[order[cursor]]
                cursor += 1
                if cursor == len(order):
                    random.shuffle(order)
                    cursor = 0
                batch = {k: v.to("cuda:0") for k, v in collator([row]).items()}
                loss = model(**batch).loss
                require(bool(torch.isfinite(loss).item()), "crop_lora_nonfinite_training_loss")
                losses.append(float(loss.detach().item()))
                (loss / config["grad_accum_steps"]).backward()
                del loss, batch
            gradients = [p.grad for _, p in trainable if p.grad is not None]
            require(bool(gradients) and all(bool(torch.isfinite(g).all().item()) for g in gradients),
                    "crop_lora_missing_or_nonfinite_gradients")
            require(any(bool(torch.count_nonzero(g).item()) for g in gradients), "crop_lora_all_zero_gradients")
            grad_norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0).item())
            require(math.isfinite(grad_norm), "crop_lora_nonfinite_gradient_norm")
            optimizer.step()
            torch.cuda.synchronize()
            require(all(bool(torch.isfinite(p).all().item()) for _, p in trainable),
                    "crop_lora_nonfinite_adapter_weights")
            step_proofs.append({"step": step + 1, "mean_loss": sum(losses) / len(losses),
                                "gradient_norm": grad_norm})
            run["optimizer_steps"] = step + 1
            write_json(output / "step-proof.json", step_proofs)
            write_json(output / "run.json", run)
        changed = sum(not torch.equal(baseline[name], p.detach().cpu()) for name, p in trainable)
        require(changed > 0, "crop_lora_optimizer_did_not_change_adapter")
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for row in validation:
                batch = {k: v.to("cuda:0") for k, v in collator([row]).items()}
                loss = model(**batch).loss
                require(bool(torch.isfinite(loss).item()), "crop_lora_nonfinite_validation_loss")
                validation_losses.append(float(loss.item()))
                del loss, batch
        checkpoint = output / "adapter"
        model.save_pretrained(checkpoint, safe_serialization=True)
        processor.save_pretrained(output / "processor")
        require((checkpoint / "adapter_model.safetensors").is_file(), "crop_lora_adapter_checkpoint_missing")
        # Recheck all data/model bytes after fitting, before claiming a complete receipt.
        preflight(config)
        run.update(state="trained_unselected", changed_adapter_tensors=changed,
                   validation_mean_loss=sum(validation_losses) / len(validation_losses),
                   step_proof_sha256=file_sha256(output / "step-proof.json"),
                   adapter_files={p.name: file_sha256(p) for p in checkpoint.iterdir() if p.is_file()},
                   processor_files={p.name: file_sha256(p) for p in (output / "processor").iterdir() if p.is_file()})
        # A merge is opt-in, may require another full snapshot of disk space, and
        # does not select the checkpoint for inference or bypass quality gates.
        if config["merge_adapter"]:
            weights_bytes = sum((checked["model_dir"] / p).stat().st_size
                                for p in checked["snapshot"]["files"] if p.endswith(".safetensors"))
            require(shutil.disk_usage(output).free > weights_bytes * 1.2, "crop_lora_merge_disk_space_insufficient")
            free, _ = torch.cuda.mem_get_info(0)
            require(free > 2 * 1024**3, "crop_lora_merge_cuda_margin_insufficient")
            merged = model.merge_and_unload(safe_merge=True)
            merged.save_pretrained(output / "merged", safe_serialization=True, max_shard_size="5GB")
            processor.save_pretrained(output / "merged")
            run["merged_files"] = {p.name: file_sha256(p) for p in (output / "merged").iterdir() if p.is_file()}
        write_json(output / "run.json", run)
        return {"ok": True, "state": run["state"], "optimizer_steps": run["optimizer_steps"],
                "changed_adapter_tensors": changed, "run_sha256": file_sha256(output / "run.json"),
                "promotion_eligible": False}
    except Exception as exc:
        run.update(state="failed", error=str(exc) if isinstance(exc, TrainingGateError) else "crop_lora_training_failed")
        write_json(output / "run.json", run)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true", help="Verify bytes/review/splits without importing CUDA libraries")
    args = parser.parse_args()
    try:
        config = resolve_config(args.config)
        if args.preflight:
            checked = preflight(config)
            print(json.dumps({"ok": True, "state": "preflight_only", "model_id": MODEL_ID,
                "qa_counts": dict(Counter(r["split"] for r in checked["rows"])),
                "training_completed": False, "promotion_eligible": False}))
        else:
            print(json.dumps(train(config)))
        return 0
    except Exception as exc:
        if isinstance(exc, TrainingGateError):
            return cli_error(exc)
        if isinstance(exc, (OSError, ValueError, KeyError, TypeError)):
            return cli_error(TrainingGateError("crop_lora_input_invalid_or_unavailable"))
        return cli_error(TrainingGateError("crop_lora_execution_failed"))


if __name__ == "__main__":
    raise SystemExit(main())
