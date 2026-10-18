"""Offline public-label QLoRA experiment; never selects a production model.

Only published object boxes supply two-class crop labels. Sealed-test image files
are inaccessible to training and development selection. See public-crop-lora.md.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import time
import traceback
from typing import Any

from training.guards import TrainingGateError, canonical_sha256
from training.specialist_lora import CropQACollator, inventory_targets
from training.specialists.common import (cli_error, file_sha256, load_json, number,
    pinned_json, relative_file, require, sha, write_json)

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
WEIGHTS_SHA256 = "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"
MODEL_CLASS = "Qwen3VLForConditionalGeneration"
CLASSES = ["single_banknote", "id_card"]
QUESTION = "Identify the visible object. Answer exactly one label: single_banknote or id_card."
PINS = {"torch": "2.13.0+cu129", "transformers": "5.17.0", "peft": "0.21.2",
        "accelerate": "1.15.0", "bitsandbytes": "0.50.2"}
SCHEMA = "forte.public-crop-lora.train.v1"
SNAPSHOT_SCHEMA = "forte.public-crop-lora.snapshot.v1"
RUN_SCHEMA = "forte.public-crop-lora.run.v1"
SHORT_SIDES = [0, 64, 32, 16]


def snapshot(model_dir: Path, output: Path) -> dict:
    """Inventory an already downloaded official snapshot; performs no fetching."""
    require(not output.exists(), "public_crop_snapshot_output_exists")
    root = model_dir.resolve()
    require(root.is_dir() and not any(p.is_dir() for p in root.iterdir()),
            "public_crop_flat_snapshot_required")
    files = {p.name: file_sha256(p) for p in root.iterdir() if p.is_file()}
    required = {"README.md", "config.json", "tokenizer.json", "tokenizer_config.json",
                "preprocessor_config.json", "chat_template.json", "model.safetensors"}
    require(required <= set(files), "public_crop_snapshot_files_missing")
    require(files["model.safetensors"] == WEIGHTS_SHA256, "public_crop_official_weights_hash_mismatch")
    card = (root / "README.md").read_text(encoding="utf-8")
    require(re.search(r"(?m)^license:\s*apache-2\.0\s*$", card) is not None,
            "public_crop_official_license_declaration_missing")
    result = {"schema_version": SNAPSHOT_SCHEMA, "model_id": MODEL_ID, "revision": REVISION,
              "license": "apache-2.0", "license_evidence": "README.md",
              "source": f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}", "files": files,
              "inventory_scope": "local_bytes_not_upstream_signature"}
    write_json(output, result)
    return {"model_manifest_sha256": file_sha256(output), "model_id": MODEL_ID, "revision": REVISION}


def verify_model(config: dict) -> tuple[Path, dict]:
    require(config.get("model_id") == MODEL_ID and config.get("revision") == REVISION,
            "public_crop_only_pinned_qwen_2b_supported")
    manifest = pinned_json(Path(config["model_manifest"]), config.get("expected_model_manifest_sha256"))
    require(manifest.get("schema_version") == SNAPSHOT_SCHEMA and manifest.get("model_id") == MODEL_ID
            and manifest.get("revision") == REVISION and manifest.get("license") == "apache-2.0",
            "public_crop_snapshot_identity_mismatch")
    root = Path(config["model_dir"]).resolve()
    files = manifest.get("files")
    require(isinstance(files, dict) and bool(files), "public_crop_snapshot_inventory_missing")
    for name, digest in files.items():
        require(name == Path(name).name and "\\" not in name and ":" not in name and sha(digest),
                "public_crop_snapshot_entry_invalid")
        require(file_sha256(relative_file(root, name)) == digest, "public_crop_snapshot_hash_mismatch")
    require({p.name for p in root.iterdir() if p.is_file()} == set(files)
            and not any(p.is_dir() for p in root.iterdir()), "public_crop_uninventoried_snapshot_file")
    required = {"README.md", "config.json", "tokenizer.json", "tokenizer_config.json",
                "preprocessor_config.json", "chat_template.json", "model.safetensors"}
    require(required <= set(files), "public_crop_snapshot_files_missing")
    require(files["model.safetensors"] == WEIGHTS_SHA256, "public_crop_official_weights_hash_mismatch")
    card = (root / "README.md").read_text(encoding="utf-8")
    require(re.search(r"(?m)^license:\s*apache-2\.0\s*$", card) is not None,
            "public_crop_official_license_declaration_missing")
    value = load_json(root / "config.json")
    require(value.get("model_type") == "qwen3_vl" and value.get("architectures") == [MODEL_CLASS]
            and not value.get("auto_map") and not value.get("quantization_config"),
            "public_crop_model_architecture_mismatch")
    require(all(value.get("text_config", {}).get(k) == v for k, v in {
        "model_type": "qwen3_vl_text", "num_hidden_layers": 28, "hidden_size": 2048,
        "num_attention_heads": 16, "num_key_value_heads": 8}.items()),
        "public_crop_model_size_mismatch")
    require(all(value.get("vision_config", {}).get(k) == v for k, v in {
        "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2}.items()),
        "public_crop_vision_geometry_mismatch")
    return root, value


def dataset_metadata(config: dict) -> tuple[Path, dict]:
    """Authenticate the frozen publisher import without decoding held-out pixels.

    The dataset importer's full verifier must run before this hash is authorized.
    Here all split/lineage metadata is checked, and only selected train/dev files
    are opened later. Test images may be mounted separately for final evaluation.
    """
    root = Path(config["dataset_dir"]).resolve()
    lock = pinned_json(root / "dataset-lock.json", config.get("expected_dataset_lock_sha256"))
    require(lock.get("schema_version") == "forte.specialists.public-dataset-lock.v1"
            and lock.get("state") == "frozen" and lock.get("origin") == "public"
            and lock.get("dataset_kind") == "combined_public", "public_crop_combined_public_lock_required")
    require(lock.get("dataset_sha256") == canonical_sha256({k: v for k, v in lock.items()
                                                           if k != "dataset_sha256"}),
            "public_crop_dataset_digest_mismatch")
    require(lock.get("classes") == CLASSES and lock.get("class_names") == {"0": CLASSES[0], "1": CLASSES[1]},
            "public_crop_class_mapping_mismatch")
    rows = lock.get("records")
    require(isinstance(rows, list) and bool(rows), "public_crop_records_missing")
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    hashes: dict[str, set[str]] = defaultdict(set)
    seen = set()
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get("sample_id"), str)
                and row["sample_id"] not in seen, "public_crop_record_identity_invalid")
        seen.add(row["sample_id"])
        require(row.get("dataset_id") in {"dny3dgvvw8", "MIDV-Holo"} and type(row.get("included")) is bool
                and row.get("split") in {"train", "validation", "sealed_test", "quarantine"},
                "public_crop_record_provenance_invalid")
        if not row["included"]:
            continue
        require(row["split"] != "quarantine" and sha(row.get("content_sha256"))
                and lock.get("files", {}).get(row.get("export_path")) == row["content_sha256"],
                "public_crop_record_inventory_invalid")
        hashes[row["content_sha256"]].add(row["split"])
        require(bool(row.get("source_group")) and bool(row.get("template_group")),
                "public_crop_source_groups_missing")
        for field in ("combined_group", "source_group", "template_group", "physical_document_group", "scene_group"):
            if row.get(field):
                groups[(field, row[field])].add(row["split"])
        if sha(row.get("source_content_sha256")):
            hashes[row["source_content_sha256"]].add(row["split"])
    require(all(len(x) == 1 for x in groups.values()) and all(len(x) == 1 for x in hashes.values()),
            "public_crop_partition_leakage")
    return root, lock


def crop_rows(lock: dict, split: str, per_class: int, seed: int) -> list[dict]:
    require(split in {"train", "validation", "sealed_test"}, "public_crop_split_invalid")
    buckets: dict[str, list[dict]] = {key: [] for key in CLASSES}
    for source in lock["records"]:
        if source["split"] != split or not source["included"]:
            continue
        require(type(source.get("width")) is int and type(source.get("height")) is int
                and source["width"] > 0 and source["height"] > 0, "public_crop_dimensions_invalid")
        for index, obj in enumerate(source.get("objects", [])):
            name, box = obj.get("class_name"), obj.get("bbox")
            require(name in CLASSES and isinstance(box, list) and len(box) == 4
                    and all(number(v) for v in box), "public_crop_published_box_invalid")
            x, y, width, height = box
            require(width > 0 and height > 0 and x + width <= source["width"] + .01
                    and y + height <= source["height"] + .01, "public_crop_published_box_outside_image")
            # Tight publisher boxes; no synthetic absence/procedural labels or face/OCR targets.
            xyxy = [math.floor(x), math.floor(y), min(source["width"], math.ceil(x + width)),
                    min(source["height"], math.ceil(y + height))]
            identifier = source["sample_id"] + f":object:{index}"
            buckets[name].append({"example_id": identifier, "source_sample_id": source["sample_id"],
                "split": split, "source_group": source["source_group"], "class_name": name,
                "_export_path": source["export_path"], "content_sha256": source["content_sha256"],
                "crop_xyxy": xyxy, "question": QUESTION, "answer": name,
                "label_source": "published_object_class_and_box"})
    require(all(buckets.values()), "public_crop_both_classes_required_in_split")
    result = []
    for name in CLASSES:
        # Fixed metadata-only ordering declared before any model outcome is seen.
        ordered = sorted(buckets[name], key=lambda r: hashlib.sha256(
            f"{seed}:{r['example_id']}".encode()).hexdigest())
        result.extend(ordered[:per_class])
    return result


def verify_crop_files(root: Path, rows: list[dict]) -> None:
    checked = set()
    for row in rows:
        if row["_export_path"] in checked:
            continue
        require(file_sha256(relative_file(root, row["_export_path"])) == row["content_sha256"],
                "public_crop_image_hash_mismatch")
        checked.add(row["_export_path"])


def time_remaining(config: dict) -> float:
    try:
        stop = datetime.fromisoformat(config["stop_at_utc"].replace("Z", "+00:00"))
        require(stop.tzinfo is not None and stop.utcoffset().total_seconds() == 0,
                "public_crop_utc_deadline_required")
        return (stop - datetime.now(timezone.utc)).total_seconds()
    except (KeyError, TypeError, ValueError):
        raise TrainingGateError("public_crop_utc_deadline_required") from None


def preflight(config: dict) -> dict:
    require(config.get("schema_version") == SCHEMA and config.get("runtime_versions") == PINS,
            "public_crop_schema_or_runtime_pins_invalid")
    require(isinstance(config.get("runtime_image_digest"), str)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", config["runtime_image_digest"]),
            "public_crop_deployment_verified_runtime_image_pin_required")
    require(config.get("recipe_predeclared") is True and config.get("recipe_reference")
            and config.get("verified_dataset_import_reference"), "public_crop_predeclared_recipe_required")
    require(isinstance(config.get("test_exposure_registry_dir"), str)
            and bool(config["test_exposure_registry_dir"]) and "://" not in config["test_exposure_registry_dir"],
            "public_crop_shared_test_registry_required")
    require(config.get("train_short_side_curriculum") == SHORT_SIDES
            and config.get("evaluation_short_sides") == SHORT_SIDES,
            "public_crop_predeclared_resolution_protocol_required")
    for field, low, high in (("optimizer_steps", 1, 1000), ("grad_accum_steps", 1, 16),
        ("seed", 0, 2**31 - 1), ("lora_rank", 1, 8), ("lora_alpha", 1, 32),
        ("max_length", 128, 512), ("max_pixels", 1024, 65536),
        ("train_per_class", 1, 4096), ("validation_per_class", 1, 128),
        ("test_per_class", 1, 1024), ("gpu_memory_budget_mib", 1024, 5632)):
        require(type(config.get(field)) is int and low <= config[field] <= high,
                "public_crop_parameter_invalid:" + field)
    require(number(config.get("learning_rate"), 1e-7, 3e-4)
            and number(config.get("max_fit_seconds"), 1, 3600)
            and number(config.get("evaluation_reserve_seconds"), 300, 1800)
            and number(config.get("minimum_free_margin_mib"), 1024, 8192),
            "public_crop_bounded_budget_required")
    require(time_remaining(config) > config["evaluation_reserve_seconds"], "public_crop_deadline_reserve_exhausted")
    model_dir, model_config = verify_model(config)
    dataset_dir, lock = dataset_metadata(config)
    rows = {split: crop_rows(lock, split, config[split + "_per_class"], config["seed"])
            for split in ("train", "validation")}
    verify_crop_files(dataset_dir, rows["train"] + rows["validation"])
    return {"model_dir": model_dir, "model_config": model_config, "dataset_dir": dataset_dir,
            "lock": lock, "rows": rows}


class _ResizeProcessor:
    def __init__(self, processor: Any):
        self.wrapped, self.short_side = processor, 0

    def __getattr__(self, name: str):
        return getattr(self.wrapped, name)

    def __call__(self, **kwargs):
        # A min area equal to the max can round a non-square crop above the
        # maximum in Qwen's smart_resize. One merged patch is the lower bound.
        image_kwargs = dict(kwargs.get("images_kwargs", {}))
        if "size" in image_kwargs:
            size = dict(image_kwargs["size"])
            size["shortest_edge"] = min(1024, size["longest_edge"])
            image_kwargs["size"] = size
            kwargs["images_kwargs"] = image_kwargs
        if self.short_side:
            from PIL import Image
            images = []
            for image in kwargs["images"]:
                scale = min(1.0, self.short_side / min(image.size))
                size = tuple(max(1, round(side * scale)) for side in image.size)
                images.append(image.resize(size, Image.Resampling.LANCZOS) if size != image.size else image)
            kwargs["images"] = images
        return self.wrapped(**kwargs)


class PublicCropCollator(CropQACollator):
    """Downsample original crops only; the standard processor then resizes them."""
    def __init__(self, processor: Any, *args, **kwargs):
        super().__init__(_ResizeProcessor(processor), *args, **kwargs)

    def __call__(self, records: list[dict]):
        require(len(records) == 1 and records[0].get("_short_side", 0) in SHORT_SIDES,
                "public_crop_resolution_not_predeclared")
        self.processor.short_side = records[0].get("_short_side", 0)
        return super().__call__(records)


def _runtime(config: dict, checked: dict, *, progress=None):
    def phase(value: str) -> None:
        if progress is not None:
            progress(value)

    phase("runtime_authorization_and_versions")
    authorization = config.get("compute_release_authorization", {})
    require(authorization.get("released_by_A18_or_root") is True and authorization.get("release_reference"),
            "public_crop_compute_not_released")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      WANDB_DISABLED="true", TOKENIZERS_PARALLELISM="false")
    require(all(importlib.metadata.version(k) == v for k, v in PINS.items()),
            "public_crop_runtime_version_mismatch")
    phase("runtime_imports")
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    phase("runtime_cuda_budget")
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "public_crop_cuda_bf16_required")
    free, total = torch.cuda.mem_get_info(0)
    budget = config["gpu_memory_budget_mib"] * 1024**2
    require(free >= budget + config["minimum_free_margin_mib"] * 1024**2,
            "public_crop_insufficient_cuda_margin")
    torch.cuda.set_per_process_memory_fraction(budget / total, 0)
    torch.cuda.reset_peak_memory_stats(0)
    torch.manual_seed(config["seed"])
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["model.visual", "lm_head"])
    phase("runtime_quantized_model_load")
    model = Qwen3VLForConditionalGeneration.from_pretrained(checked["model_dir"],
        local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": 0}, quantization_config=quantization)
    phase("runtime_loaded_model_validation")
    require(model.__class__.__name__ == MODEL_CLASS and getattr(model, "is_loaded_in_4bit", False),
            "public_crop_quantized_architecture_mismatch")
    require(all(p.device.type == "cuda" for p in model.parameters()), "public_crop_model_not_all_cuda")
    from bitsandbytes.nn import Linear4bit
    visual = model.model.visual
    require(not any(isinstance(module, Linear4bit) for module in visual.modules())
            and all(p.dtype == torch.bfloat16 for p in visual.parameters()),
            "public_crop_frozen_vision_must_remain_bf16")
    # Freeze explicitly without prepare_model_for_kbit_training's blanket fp32
    # cast of the large frozen vision tower/embeddings. BF16 frozen modules are
    # intentional; only language-attention adapters become trainable below.
    phase("runtime_freeze_model")
    model.requires_grad_(False)
    phase("runtime_processor_load")
    processor = AutoProcessor.from_pretrained(checked["model_dir"], local_files_only=True,
        trust_remote_code=False, max_pixels=config["max_pixels"])
    phase("runtime_processor_validation")
    require(processor.__class__.__name__ == "Qwen3VLProcessor", "public_crop_processor_mismatch")
    for field, token in (("image_token_id", "<|image_pad|>"), ("video_token_id", "<|video_pad|>"),
        ("vision_start_token_id", "<|vision_start|>"), ("vision_end_token_id", "<|vision_end|>")):
        require(processor.tokenizer.convert_tokens_to_ids(token) == checked["model_config"][field],
                "public_crop_visual_token_mismatch")
    return model, processor


def _budget(config: dict, start: float, *, fitting: bool) -> None:
    import torch
    require(torch.cuda.max_memory_reserved(0) <= config["gpu_memory_budget_mib"] * 1024**2,
            "public_crop_allocator_budget_exceeded")
    reserve = config["evaluation_reserve_seconds"] if fitting else 60
    require(time_remaining(config) > reserve, "public_crop_deadline_reached")
    if fitting:
        require(time.monotonic() - start < config["max_fit_seconds"], "public_crop_fit_walltime_reached")


def classify(model: Any, collator: CropQACollator, rows: list[dict], config: dict) -> dict:
    """Greedy exact-label generation; unknown output counts as an error."""
    import torch
    model.eval()
    counts = Counter()
    results = []
    started = time.monotonic()
    generation_seconds = 0.0
    with torch.inference_mode():
        for row in rows:
            _budget(config, 0, fitting=False)
            batch = collator([row])
            labels = batch.pop("labels")[0].tolist()
            prefix = next(i for i, label in enumerate(labels) if label != -100)
            for key in ("input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids", "position_ids"):
                if key in batch:
                    require(batch[key].shape[-1] == len(labels), "public_crop_text_metadata_length_mismatch")
                    batch[key] = batch[key][..., :prefix]
            inputs = {key: value.to("cuda:0") for key, value in batch.items()}
            torch.cuda.synchronize()
            generation_started = time.monotonic()
            generated = model.generate(**inputs, max_new_tokens=12, do_sample=False, use_cache=True)
            torch.cuda.synchronize()
            generation_seconds += time.monotonic() - generation_started
            answer = collator.processor.decode(generated[0, prefix:], skip_special_tokens=True).strip()
            prediction = answer if answer in CLASSES else "invalid_output"
            correct = prediction == row["answer"]
            counts["correct"] += correct
            counts["total"] += 1
            counts[f"true:{row['answer']}:pred:{prediction}"] += 1
            results.append({"example_id": row["example_id"], "class_name": row["answer"],
                            "prediction": prediction, "correct": correct})
            del generated, inputs, batch
    elapsed = time.monotonic() - started
    per_class = {name: {"examples": sum(row["class_name"] == name for row in results),
        "correct": sum(row["class_name"] == name and row["correct"] for row in results)} for name in CLASSES}
    for value in per_class.values():
        value["accuracy"] = value["correct"] / value["examples"]
    return {"accuracy": counts["correct"] / counts["total"], "counts": dict(counts), "results": results,
            "per_class": per_class, "macro_class_accuracy": sum(v["accuracy"] for v in per_class.values()) / len(CLASSES),
            "unique_source_groups": len({row["source_group"] for row in rows}),
            "timing": {"examples": len(rows), "total_seconds": elapsed,
                "generation_seconds": generation_seconds, "crops_per_second": len(rows) / elapsed,
                "mean_generation_seconds": generation_seconds / len(rows),
                "scope": "single_crop_microbatches_includes_preprocessing_not_whole_video_api_latency"},
            "scope": "published_object_crops_two_classes_not_bank_video_verdicts"}


def stress_evaluate(model: Any, collator: PublicCropCollator, rows: list[dict], config: dict) -> dict:
    reports = {str(side): classify(model, collator, [dict(r, _short_side=side) for r in rows], config)
               for side in SHORT_SIDES}
    return {"conditions": reports, "macro_accuracy": sum(r["macro_class_accuracy"] for r in reports.values()) / len(reports),
            "short_side_zero_means_original": True, "pixels_reconstructed": False}


def evaluation_protocol(config: dict) -> dict:
    lock = pinned_json(Path(config["dataset_dir"]) / "dataset-lock.json", config["expected_dataset_lock_sha256"])
    sealed = {path: digest for path, digest in lock["files"].items() if path.startswith("sealed_test/")}
    require(bool(sealed), "public_crop_sealed_inventory_missing")
    return {"schema_version": "forte.public-crop-lora.evaluation-protocol.v1",
        "member_id": "crop_vlm", "sealed_inventory_sha256": canonical_sha256(sealed),
        "selection_scope": "public_train_dev_only", "test_reuse_for_tuning_allowed": False,
        "classes": CLASSES, "question": QUESTION, "crop": "publisher_bbox_floor_start_ceil_end",
        "sampling": "sha256_of_seed_and_source_object_id_per_class", "seed": config["seed"],
        "validation_per_class": config["validation_per_class"], "test_per_class": config["test_per_class"],
        "short_sides": SHORT_SIDES, "degradation": "lanczos_downsample_only_then_standard_processor",
        "max_pixels": config["max_pixels"], "max_length": config["max_length"],
        "processor_min_pixels": 1024,
        "generation": {"do_sample": False, "max_new_tokens": 12, "use_cache": True},
        "accepted_output": "exact_class_name_after_strip", "invalid_output": "incorrect",
        "primary_metric": "equal_class_then_equal_resolution_macro_accuracy; pooled_accuracy_also_reported",
        "development_models": ["base", "adapter"], "sealed_test_models": ["selected_adapter_only"],
        "derived_stress_views_are_independent_examples": False,
        "bank_video_false_alarm_or_accuracy_claim": False,
        "dataset_lock_sha256": config["expected_dataset_lock_sha256"],
        "source_sha256": file_sha256(Path(__file__))}


def memory_recipe(config: dict) -> dict:
    result = {k: config[k] for k in ("max_pixels", "max_length", "lora_rank", "lora_alpha",
        "gpu_memory_budget_mib", "grad_accum_steps", "runtime_image_digest")}
    result["source_files"] = {name: file_sha256(Path(__file__).parent / name) for name in
                              ("public_crop_lora.py", "specialist_lora.py", "data.py")}
    return result


def balanced_draws(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Alternate classes; cycle each class's fixed support without inventing data."""
    buckets = {name: [row for row in rows if row["answer"] == name] for name in CLASSES}
    require(all(buckets.values()), "public_crop_balanced_support_missing")
    rng = random.Random(seed)
    cursors = {name: 0 for name in CLASSES}
    for bucket in buckets.values():
        rng.shuffle(bucket)
    result = []
    for i in range(count):
        name = CLASSES[i % len(CLASSES)]
        result.append(buckets[name][cursors[name]])
        cursors[name] += 1
        if cursors[name] == len(buckets[name]):
            cursors[name] = 0
            rng.shuffle(buckets[name])
    return result


def verify_adapter(output: Path, run: dict) -> None:
    files = run.get("adapter_files", {})
    require(isinstance(files, dict) and "adapter_model.safetensors" in files
            and "adapter_config.json" in files, "public_crop_adapter_inventory_missing")
    root = output / "adapter"
    for name, expected in files.items():
        require(name == Path(name).name and "\\" not in name and ":" not in name and sha(expected)
                and file_sha256(relative_file(root, name)) == expected, "public_crop_adapter_changed")
    require({p.name for p in root.iterdir() if p.is_file()} == set(files)
            and not any(p.is_dir() for p in root.iterdir()), "public_crop_uninventoried_adapter_file")


def failure_diagnostic(exc: Exception, phase: str) -> dict:
    """Bounded technical locations only: no exception text, locals or source lines."""
    def identifier(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.<>-]", "_", value)[:100]

    chain, seen = [], set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(chain) < 3:
        seen.add(id(current))
        frames = traceback.extract_tb(current.__traceback__, limit=-8)
        detail = {"error_class": identifier(type(current).__name__),
            "error_module": identifier(type(current).__module__),
            "frames": [{"file": identifier(frame.filename.replace("\\", "/").rsplit("/", 1)[-1]),
                "function": identifier(frame.name), "line": frame.lineno} for frame in frames]}
        if isinstance(current, (AttributeError, ImportError)) and isinstance(current.name, str):
            detail["symbol"] = identifier(current.name)
        if isinstance(current, OSError) and isinstance(current.errno, int):
            detail["errno"] = current.errno
        chain.append(detail)
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    return {"phase": phase, "exceptions": chain,
            "scope": "bounded_exception_types_and_frame_locations_without_values"}


def train(config: dict, *, memory_smoke: bool = False) -> dict:
    command_started = time.monotonic()
    checked = preflight(config)
    if not memory_smoke:
        smoke = pinned_json(Path(config.get("memory_smoke_run", "")), config.get("expected_memory_smoke_run_sha256"))
        require(smoke.get("state") == "memory_smoke_passed" and smoke.get("memory_smoke_only") is True
                and smoke.get("dataset_lock_sha256") == config["expected_dataset_lock_sha256"]
                and smoke.get("model_manifest_sha256") == config["expected_model_manifest_sha256"]
                and smoke.get("runtime_versions") == PINS
                and smoke.get("memory_recipe") == memory_recipe(config),
                "public_crop_matching_memory_smoke_required")
    output = Path(config["output_dir"]).resolve()
    require(not output.exists(), "public_crop_output_exists")
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "crop-manifest.json", checked["rows"])
    write_json(output / "evaluation-protocol.json", evaluation_protocol(config))
    run = {"schema_version": RUN_SCHEMA, "state": "preparing", "model_id": MODEL_ID,
        "revision": REVISION, "model_manifest_sha256": config["expected_model_manifest_sha256"],
        "dataset_lock_sha256": config["expected_dataset_lock_sha256"],
        "config_sha256": file_sha256(output / "config.json"), "runtime_versions": PINS,
        "runtime_image_digest": config["runtime_image_digest"],
        "crop_manifest_sha256": file_sha256(output / "crop-manifest.json"),
        "evaluation_protocol_sha256": file_sha256(output / "evaluation-protocol.json"),
        "optimizer_steps": 0, "approved_types": [], "promotion_eligible": False,
        "sealed_test_used_for_training_or_selection": False, "audio_used": False,
        "memory_smoke_only": memory_smoke, "training_completed": False,
        "memory_recipe": memory_recipe(config)}
    write_json(output / "run.json", run)
    def phase(value: str) -> None:
        run["phase"] = value
        write_json(output / "run.json", run)

    try:
        model, processor = _runtime(config, checked, progress=phase)
        phase("training_imports_and_collator")
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from training.specialists.public_train import parameter_sha256
        collator = PublicCropCollator(processor, checked["dataset_dir"], checked["model_config"],
            max_length=config["max_length"], max_pixels=config["max_pixels"])
        if memory_smoke:
            phase("memory_smoke_generation")
            smoke_rows = [next(r for r in checked["rows"]["train"] if r["answer"] == name) for name in CLASSES]
            classify(model, collator, smoke_rows, config)
            run["smoke_generation_completed"] = True
        phase("development_baseline")
        baseline = None if memory_smoke else stress_evaluate(model, collator, checked["rows"]["validation"], config)
        if baseline is not None:
            write_json(output / "development-baseline.json", baseline)
        phase("adapter_initialization")
        targets = inventory_targets(model, 28)
        model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM,
            r=config["lora_rank"], lora_alpha=config["lora_alpha"], lora_dropout=0.0,
            target_modules=targets, bias="none"))
        trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        require(bool(trainable) and all(".lora_" in name and ".language_model.layers." in name
                and ".self_attn." in name for name, _ in trainable), "public_crop_unexpected_trainable_parameter")
        phase("initial_parameter_hashes")
        before = parameter_sha256(model, trainable=True)
        frozen_before = parameter_sha256(model, trainable=False)
        phase("optimizer_initialization")
        model.config.use_cache = False
        model.config.text_config.use_cache = False
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = torch.optim.AdamW([p for _, p in trainable], lr=config["learning_rate"])
        steps = 1 if memory_smoke else config["optimizer_steps"]
        order = balanced_draws(smoke_rows if memory_smoke else checked["rows"]["train"],
                               steps * config["grad_accum_steps"], config["seed"])
        cursor, proofs, start = 0, [], time.monotonic()
        run.update(state="training", trainable_parameters=sum(p.numel() for _, p in trainable),
            target_modules=targets, adapter_before_sha256=before, frozen_before_sha256=frozen_before,
            gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
            training_sampling="deterministic_alternating_classes_with_within_class_cycles",
            unique_training_crop_support=dict(Counter(row["answer"] for row in checked["rows"]["train"])),
            planned_training_draw_counts=dict(Counter(row["answer"] for row in order)),
            planned_unique_drawn_crops={name: len({r["example_id"] for r in order if r["answer"] == name})
                                       for name in CLASSES})
        write_json(output / "run.json", run)
        model.train()
        # Keep the entire frozen vision tower in evaluation mode too.
        for name, module in model.named_modules():
            if name.endswith(".visual"):
                module.eval()
        phase("optimizer_steps")
        for step in range(steps):
            _budget(config, start, fitting=True)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for micro in range(config["grad_accum_steps"]):
                _budget(config, start, fitting=True)
                # Both classes see each resolution equally; resolution must not
                # correlate with the alternating class label.
                row = dict(order[cursor], _short_side=SHORT_SIDES[(cursor // len(CLASSES)) % len(SHORT_SIDES)])
                cursor += 1
                batch = {k: v.to("cuda:0") for k, v in collator([row]).items()}
                loss = model(**batch).loss
                require(bool(torch.isfinite(loss).item()), "public_crop_nonfinite_loss")
                losses.append(float(loss.detach().item()))
                (loss / config["grad_accum_steps"]).backward()
                del loss, batch
            grads = [p.grad for _, p in trainable if p.grad is not None]
            require(bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
                    and any(bool(torch.count_nonzero(g)) for g in grads), "public_crop_invalid_gradient")
            norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0).item())
            require(math.isfinite(norm), "public_crop_nonfinite_gradient_norm")
            optimizer.step()
            torch.cuda.synchronize()
            require(all(bool(torch.isfinite(p).all()) for _, p in trainable), "public_crop_nonfinite_adapter")
            proofs.append({"step": step + 1, "mean_loss": sum(losses) / len(losses), "gradient_norm": norm,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2})
            run["optimizer_steps"] = step + 1
            write_json(output / "step-proof.json", proofs)
            write_json(output / "run.json", run)
        optimizer.zero_grad(set_to_none=True)
        fit_seconds = time.monotonic() - start
        phase("final_parameter_hashes")
        after = parameter_sha256(model, trainable=True)
        frozen_after = parameter_sha256(model, trainable=False)
        require(after != before and frozen_before == frozen_after, "public_crop_update_or_freeze_proof_failed")
        model.gradient_checkpointing_disable()
        phase("development_adapter")
        validation = None if memory_smoke else stress_evaluate(model, collator, checked["rows"]["validation"], config)
        if validation is not None:
            write_json(output / "development-adapter.json", validation)
        phase("adapter_save_and_input_reverification")
        model.save_pretrained(output / "adapter", safe_serialization=True)
        require((output / "adapter" / "adapter_model.safetensors").is_file(), "public_crop_checkpoint_missing")
        verify_crop_files(checked["dataset_dir"], checked["rows"]["train"] + checked["rows"]["validation"])
        require(file_sha256(checked["dataset_dir"] / "dataset-lock.json") == config["expected_dataset_lock_sha256"],
                "public_crop_dataset_changed_during_run")
        run.update(state="memory_smoke_passed" if memory_smoke else "trained_unselected", phase="complete",
            training_completed=not memory_smoke, adapter_after_sha256=after, frozen_after_sha256=frozen_after,
            development_baseline_accuracy=None if baseline is None else baseline["macro_accuracy"],
            development_adapter_accuracy=None if validation is None else validation["macro_accuracy"],
            development_improved=None if validation is None else validation["macro_accuracy"] > baseline["macro_accuracy"],
            fit_walltime_seconds=fit_seconds, total_command_walltime_seconds=time.monotonic() - command_started,
            development_inference_timing=None if validation is None else {
                side: report["timing"] for side, report in validation["conditions"].items()},
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
            adapter_files={p.name: file_sha256(p) for p in (output / "adapter").iterdir() if p.is_file()},
            step_proof_sha256=file_sha256(output / "step-proof.json"))
        write_json(output / "run.json", run)
        return {"ok": True, "state": run["state"], "optimizer_steps": run["optimizer_steps"],
            "development_baseline_accuracy": run["development_baseline_accuracy"],
            "development_adapter_accuracy": run["development_adapter_accuracy"],
            "peak_reserved_mib": run["peak_reserved_mib"], "run_sha256": file_sha256(output / "run.json"),
            "promotion_eligible": False}
    except Exception as exc:
        run.update(state="failed", error=str(exc) if isinstance(exc, TrainingGateError) else "public_crop_training_failed",
            failure_diagnostic=failure_diagnostic(exc, run.get("phase", "preparing")))
        write_json(output / "run.json", run)
        raise


def final_evaluate(config: dict, expected_run_sha256: str, *, cohort_receipt: Path | None = None,
                   expected_cohort_sha256: str | None = None) -> dict:
    checked = preflight(config)
    output = Path(config["output_dir"]).resolve()
    run = pinned_json(output / "run.json", expected_run_sha256)
    require(run.get("state") == "trained_unselected" and run.get("schema_version") == RUN_SCHEMA
            and run.get("config_sha256") == file_sha256(output / "config.json")
            and load_json(output / "config.json") == config
            and run.get("dataset_lock_sha256") == config["expected_dataset_lock_sha256"],
            "public_crop_final_evaluation_run_mismatch")
    verify_adapter(output, run)
    require(run.get("evaluation_protocol_sha256") == file_sha256(output / "evaluation-protocol.json")
            and load_json(output / "evaluation-protocol.json") == evaluation_protocol(config),
            "public_crop_evaluation_protocol_changed")
    if cohort_receipt is None:
        from training.specialists.public_train import claim_sealed_test
        exposure = claim_sealed_test(config, checked["lock"], run["adapter_files"]["adapter_model.safetensors"])
    else:
        from training.specialists.public_cohort import claim_cohort_member
        require(sha(expected_cohort_sha256), "public_crop_cohort_hash_required")
        exposure = claim_cohort_member(config, checked["lock"], run["adapter_files"]["adapter_model.safetensors"],
            cohort_receipt=cohort_receipt, expected_cohort_sha256=expected_cohort_sha256,
            member_id="crop_vlm", run_path=output / "run.json", config_path=output / "config.json",
            protocol_path=output / "evaluation-protocol.json")
    marker = output / "sealed-test-access.json"
    # Exclusive creation records test access before pixels are opened. Retry is
    # deliberately blocked, including after partial/failed final evaluation.
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({"run_sha256": expected_run_sha256, "state": "access_started",
                   "opened_at_utc": datetime.now(timezone.utc).isoformat()}, stream)
    rows = crop_rows(checked["lock"], "sealed_test", config["test_per_class"], config["seed"])
    verify_crop_files(checked["dataset_dir"], rows)
    if cohort_receipt is not None:
        from training.specialists.public_cohort import verify_cohort
        verify_cohort(cohort_receipt, expected_cohort_sha256)
    model, processor = _runtime(config, checked)
    collator = PublicCropCollator(processor, checked["dataset_dir"], checked["model_config"],
        max_length=config["max_length"], max_pixels=config["max_pixels"])
    from peft import PeftModel
    if cohort_receipt is not None:
        verify_cohort(cohort_receipt, expected_cohort_sha256)
    adapted = PeftModel.from_pretrained(model, output / "adapter", is_trainable=False, local_files_only=True)
    result = {"schema_version": "forte.public-crop-lora.sealed-evaluation.v1", "run_sha256": expected_run_sha256,
        "adapter": stress_evaluate(adapted, collator, rows, config),
        "promotion_eligible": False, "test_reuse_for_tuning_allowed": False,
        "shared_test_exposure_receipt": str(exposure)}
    write_json(output / "sealed-test-results.json", result)
    if cohort_receipt is None:
        exposure_record = load_json(exposure)
        exposure_record.update(state="complete", metrics_sha256=file_sha256(output / "sealed-test-results.json"))
        write_json(exposure, exposure_record)
    return {"ok": True, "adapter_accuracy": result["adapter"]["macro_accuracy"], "promotion_eligible": False,
        "report_sha256": file_sha256(output / "sealed-test-results.json")}


def cohort_descriptor(config: dict, expected_run_sha256: str) -> dict:
    """Prepare exact artifact pins for the root's joint registration, CPU only."""
    checked = preflight(config)
    output = Path(config["output_dir"]).resolve()
    run = pinned_json(output / "run.json", expected_run_sha256)
    require(run.get("state") == "trained_unselected" and not run.get("memory_smoke_only"),
            "public_crop_completed_fit_required")
    verify_adapter(output, run)
    require(run.get("config_sha256") == file_sha256(output / "config.json")
            and load_json(output / "config.json") == config
            and run.get("evaluation_protocol_sha256") == file_sha256(output / "evaluation-protocol.json")
            and load_json(output / "evaluation-protocol.json") == evaluation_protocol(config),
            "public_crop_frozen_run_inputs_changed")
    def pin(path: Path) -> dict:
        return {"path": str(path.resolve()), "sha256": file_sha256(path)}
    artifacts = {"evaluation_source": pin(Path(__file__)),
        "collator_source": pin(Path(__file__).parent / "specialist_lora.py"),
        "label_mask_source": pin(Path(__file__).parent / "data.py"),
        "adapter_config": pin(output / "adapter" / "adapter_config.json"),
        "model_manifest": pin(Path(config["model_manifest"]))}
    snapshot_manifest = load_json(Path(config["model_manifest"]))
    for name in snapshot_manifest["files"]:
        artifacts["snapshot_" + name.lstrip(".")] = pin(checked["model_dir"] / name)
    checkpoint = output / "adapter" / "adapter_model.safetensors"
    require(file_sha256(checkpoint) == run.get("adapter_files", {}).get(checkpoint.name),
            "public_crop_selected_adapter_changed")
    return {"run": pin(output / "run.json"), "checkpoint": pin(checkpoint),
        "config": pin(output / "config.json"), "protocol": pin(output / "evaluation-protocol.json"),
        "artifacts": artifacts}


def resolve_config(path: Path) -> dict:
    config = load_json(path)
    for key in ("model_dir", "model_manifest", "dataset_dir", "output_dir", "test_exposure_registry_dir"):
        value = config.get(key)
        require(isinstance(value, str) and bool(value) and "://" not in value, "public_crop_local_paths_required")
        config[key] = str((path.resolve().parent / value).resolve())
    if config.get("memory_smoke_run"):
        config["memory_smoke_run"] = str((path.resolve().parent / config["memory_smoke_run"]).resolve())
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--memory-smoke", action="store_true")
    parser.add_argument("--inventory-model", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--final-eval-run-sha256")
    parser.add_argument("--cohort-receipt", type=Path)
    parser.add_argument("--expected-cohort-sha256")
    parser.add_argument("--cohort-descriptor-run-sha256")
    args = parser.parse_args()
    try:
        if args.inventory_model:
            require(args.manifest_output is not None and args.config is None, "public_crop_snapshot_cli_invalid")
            result = snapshot(args.inventory_model, args.manifest_output)
        else:
            require(args.config is not None, "public_crop_config_required")
            config = resolve_config(args.config)
            if args.preflight:
                checked = preflight(config)
                result = {"ok": True, "state": "preflight_only", "training_completed": False,
                    "crop_counts": {k: len(v) for k, v in checked["rows"].items()}, "promotion_eligible": False}
            elif args.cohort_descriptor_run_sha256:
                result = cohort_descriptor(config, args.cohort_descriptor_run_sha256)
            else:
                from training.specialists.public_train import cli_deadline_watchdog
                watchdog = cli_deadline_watchdog(config)
                try:
                    if args.final_eval_run_sha256:
                        result = final_evaluate(config, args.final_eval_run_sha256,
                            cohort_receipt=args.cohort_receipt, expected_cohort_sha256=args.expected_cohort_sha256)
                    else:
                        result = train(config, memory_smoke=args.memory_smoke)
                finally:
                    watchdog.cancel()
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception as exc:
        return cli_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
