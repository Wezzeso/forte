"""Offline Qwen3-VL MoE BF16 DESK LoRA; never reads bank-video labels.

The recipe is fixed before evaluation. Checkpoints are selected by the declared
step/time budget, never by Forte's 40-video scores. Stdout contains JSON metrics
and fixed error codes only; arbitrary library output is suppressed. Images are
decoded only inside this local training process and never saved or displayed.

python -m training.qwen30_lora_train --config CONFIG --mode preflight
python -m training.qwen30_lora_train --config CONFIG --mode smoke
python -m training.qwen30_lora_train --config CONFIG --mode train
python -m training.qwen30_lora_train --config CONFIG --mode train --resume CHECKPOINT
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

from training.data import completion_labels, masked_completion_labels
from training.guards import TrainingGateError, canonical_sha256, file_sha256, require

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
REVISION = "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
MODEL_CLASS = "Qwen3VLMoeForConditionalGeneration"
CONFIG_SHA256 = "6129b3f54d517a18553cf7525ff4886be6831c5806fc094d632b8ef38d859912"
PINS = {"torch": "2.13.0+cu129", "transformers": "5.17.0", "peft": "0.21.2",
        "accelerate": "1.15.0", "bitsandbytes": "0.50.2", "pillow": "12.3.0"}
CONDITIONS = frozenset({"note_only", "id_only", "both", "blank"})
DESK_FIELDS = ("cash", "cash_bag", "coins_or_valuables", "drink", "coat_or_handbag", "id_or_document")
DESK_Q = ("Close-up of a cashier's desk in a bank. Ignore the laptop, phone, counting machine and detector. "
          "Report what lies on the desk surface: banknotes/cash bundles, a cloth cash bag (sack), coin tubes/rolls, "
          "a cup or drink, a coat/jacket or handbag, an ID card or document. JSON booleans.")
SYSTEM = ("You are a precise visual inspector of CCTV footage from a bank cashier room. "
          "Report only what is clearly visible. Never identify people. Answer strictly in JSON.")
SOURCE_DATASETS = frozenset({"dny3dgvvw8", "MIDV-Holo"})
TARGET_RE = re.compile(r"model\.language_model\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)$")
SCHEMA = "forte.qwen30-public-lora.run.v1"
PATH_FIELDS = ("model_dir", "model_download_receipt", "manifest",
               "image_root", "output_dir")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        raise TrainingGateError("qwen30_invalid_or_missing_json") from None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def emit(value: dict) -> None:
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


@contextmanager
def quiet_libraries():
    """Third-party exception/progress text can contain prompts: never forward it."""
    sys.stdout.flush()
    sys.stderr.flush()
    out, err = os.dup(1), os.dup(2)
    with open(os.devnull, "w") as sink:
        try:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(out, 1)
            os.dup2(err, 2)
            os.close(out)
            os.close(err)


def resolve_config(path: Path) -> dict:
    config = read_json(path)
    require(isinstance(config, dict), "qwen30_config_object_required")
    for key in PATH_FIELDS:
        value = config.get(key)
        require(isinstance(value, str) and bool(value) and "://" not in value,
                "qwen30_local_paths_required")
        config[key] = str((path.resolve().parent / value).resolve())
    return config


def validate_config(config: dict) -> None:
    require(config.get("schema_version") == "forte.qwen30-public-lora.config.v1",
            "qwen30_config_schema_invalid")
    require(config.get("model_id") == MODEL_ID and config.get("revision") == REVISION,
            "qwen30_base_identity_mismatch")
    require(config.get("runtime_versions") == PINS, "qwen30_runtime_pins_mismatch")
    require(config.get("training_data_policy") == "public_annotations_and_generated_canvases"
            and config.get("bank_video_labels_used") is False
            and config.get("production_enabled") is False, "qwen30_public_only_staging_required")
    require(config.get("target_loss_policy") == "supported_cash_id_boolean_values_only"
            and config.get("adapter_scope") == "desk_cash_id_refinement_only", "qwen30_desk_only_scope_required")
    require(isinstance(config.get("recipe_principle"), str) and bool(config["recipe_principle"]),
            "qwen30_visual_principle_required")
    for key, low, high in (("optimizer_steps", 1, 200), ("grad_accum_steps", 1, 16),
        ("max_length", 128, 4096), ("max_pixels", 4096, 1048576), ("seed", 0, 2**31-1),
        ("lora_rank", 1, 32), ("lora_alpha", 1, 64), ("checkpoint_every", 1, 100),
        ("pilot_steps", 1, 100), ("pilot_max_seconds", 10, 1000),
        ("warmup_steps", 0, 100), ("max_training_seconds", 30, 3600)):
        require(type(config.get(key)) is int and low <= config[key] <= high,
                "qwen30_training_parameter_invalid")
    for key, low, high in (("learning_rate", 1e-7, .001), ("lora_dropout", 0, .2),
        ("weight_decay", 0, 1), ("max_grad_norm", .01, 10),
        ("minimum_free_cuda_gib", 1, 80), ("gpu_memory_budget_gib", 10, 79)):
        value = config.get(key)
        require(type(value) in {int, float} and math.isfinite(value) and low <= value <= high,
                "qwen30_training_parameter_invalid")
    require(config.get("quantization") == "bf16" and type(config.get("verify_weight_hashes")) is bool,
            "qwen30_exact_bf16_recipe_required")
    require(config["warmup_steps"] < config["optimizer_steps"], "qwen30_warmup_exceeds_steps")
    require(re.fullmatch(r"[0-9a-f]{64}", str(config.get("expected_manifest_sha256"))) is not None,
            "qwen30_manifest_hash_pin_required")
    stop = deadline(config)
    require(stop.tzinfo is not None and stop.utcoffset().total_seconds() == 0,
            "qwen30_utc_deadline_required")


def deadline(config: dict) -> datetime:
    try:
        return datetime.fromisoformat(config["stop_at_utc"].replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        raise TrainingGateError("qwen30_utc_deadline_required") from None


def safe_image_path(root: Path, value: str) -> Path:
    require(isinstance(value, str) and bool(value) and "://" not in value,
            "qwen30_image_path_invalid")
    path = (root / value).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file()
            and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"},
            "qwen30_image_outside_public_root")
    return path


def validate_rows(rows: list[dict]) -> dict:
    require(bool(rows), "qwen30_manifest_empty")
    groups, hashes = defaultdict(set), defaultdict(set)
    counts, facts = Counter(), Counter()
    for row in rows:
        require(isinstance(row, dict) and row.get("split") in {"train", "validation"}
                and row.get("condition") in CONDITIONS, "qwen30_split_or_condition_invalid")
        require(row.get("prompt") == DESK_Q and row.get("system") == SYSTEM,
                "qwen30_exact_pipeline_desk_prompt_required")
        target, fields = row.get("target"), row.get("supervision_fields")
        require(isinstance(target, dict) and set(target) == set(DESK_FIELDS)
                and all(type(value) is bool for value in target.values()), "qwen30_exact_desk_target_required")
        require(isinstance(fields, list) and bool(fields) and len(fields) == len(set(fields))
                and set(fields) <= {"cash", "id_or_document"}, "qwen30_only_supported_desk_values_supervised")
        composition = row.get("composition")
        require(isinstance(composition, dict) and type(composition.get("width")) is int
                and type(composition.get("height")) is int
                and 64 <= composition["width"] <= 1024 and 64 <= composition["height"] <= 1024,
                "qwen30_canvas_geometry_invalid")
        background = composition.get("background")
        require(isinstance(background, list) and len(background) == 3
                and all(type(value) is int and 0 <= value <= 255 for value in background),
                "qwen30_generated_background_invalid")
        components = composition.get("components")
        require(isinstance(components, list) and len(components) <= 2, "qwen30_components_invalid")
        classes, boxes = [], []
        for component in components:
            require(isinstance(component, dict) and component.get("source_dataset") in SOURCE_DATASETS
                    and component.get("object_class") in {"banknote", "id_card"}
                    and isinstance(component.get("source_group"), str) and bool(component["source_group"]),
                    "qwen30_published_component_required")
            classes.append(component["object_class"])
            require(re.fullmatch(r"[0-9a-f]{64}", str(component.get("image_sha256"))) is not None,
                    "qwen30_component_hash_required")
            hashes[component["image_sha256"]].add(row["split"])
            for key in ("source_group", "template_group", "physical_document_group", "scene_group"):
                if component.get(key):
                    groups[(key, component[key])].add(row["split"])
            box = component.get("crop_box_xyxy")
            require(isinstance(box, list) and len(box) == 4 and all(type(v) is int for v in box)
                    and 0 <= box[0] < box[2] and 0 <= box[1] < box[3], "qwen30_component_box_invalid")
            require(all(type(component.get(key)) is int for key in ("x", "y", "width", "height")),
                    "qwen30_placement_integer_required")
            x, y, width, height = (component[key] for key in ("x", "y", "width", "height"))
            require(0 <= x and 0 <= y and width > 0 and height > 0
                    and x + width <= composition["width"] and y + height <= composition["height"],
                    "qwen30_component_outside_canvas")
            for px, py, pw, ph in boxes:
                require(x + width <= px or px + pw <= x or y + height <= py or py + ph <= y,
                        "qwen30_components_occlude_each_other")
            boxes.append((x, y, width, height))
            if component["object_class"] == "id_card":
                quad = component.get("source_quad")
                require(component.get("mask_kind") == "publisher_quadrangle"
                        and isinstance(quad, list) and len(quad) == 4
                        and all(isinstance(point, list) and len(point) == 2
                                and all(type(v) in {int, float} and math.isfinite(v) for v in point) for point in quad),
                        "qwen30_id_publisher_polygon_required")
            else:
                require(component.get("mask_kind") == "bbox_unknown_background",
                        "qwen30_banknote_background_uncertainty_required")
        expected = {"blank": [], "id_only": ["id_card"], "note_only": ["banknote"],
                    "both": ["banknote", "id_card"]}[row["condition"]]
        require(sorted(classes) == sorted(expected), "qwen30_condition_components_mismatch")
        require(target["cash"] == ("banknote" in classes)
                and target["id_or_document"] == ("id_card" in classes), "qwen30_supported_targets_mismatch")
        expected_fields = {"cash"} if row["condition"] == "note_only" else {"cash", "id_or_document"}
        require(set(fields) == expected_fields, "qwen30_unsupported_absence_supervision")
        for field in fields:
            facts[(row["split"], field, target[field])] += 1
        counts[(row["split"], row["condition"])] += 1
    require(all(len(value) == 1 for value in groups.values()) and all(len(value) == 1 for value in hashes.values()),
            "qwen30_source_group_split_leakage")
    require(all(counts[(split, condition)] > 0 for split in ("train", "validation") for condition in CONDITIONS),
            "qwen30_all_compositions_each_split_required")
    return {"rows": len(rows), "source_groups": len({key for kind, key in groups if kind == "source_group"}),
            "lineage_groups": len(groups),
            "condition_counts": {f"{split}/{condition}": count for (split, condition), count in sorted(counts.items())},
            "supervised_value_counts": {f"{split}/{field}/{value}": count for (split, field, value), count in sorted(facts.items())}}


def load_rows(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except (OSError, ValueError):
        raise TrainingGateError("qwen30_manifest_invalid") from None


def preflight(config: dict) -> dict:
    validate_config(config)
    manifest_path = Path(config["manifest"])
    require(file_sha256(manifest_path) == config["expected_manifest_sha256"], "qwen30_manifest_hash_mismatch")
    rows = load_rows(manifest_path)
    stats = validate_rows(rows)
    root = Path(config["image_root"])
    seen = set()
    for row in rows:
        for component in row["composition"]["components"]:
            path = safe_image_path(root, component["image_path"])
            digest = component["image_sha256"]
            if (path, digest) not in seen:
                require(file_sha256(path) == digest, "qwen30_public_image_hash_mismatch")
                seen.add((path, digest))
    model_root = Path(config["model_dir"])
    model_config = read_json(model_root / "config.json")
    require(file_sha256(model_root / "config.json") == CONFIG_SHA256
            and model_config.get("model_type") == "qwen3_vl_moe"
            and model_config.get("architectures") == [MODEL_CLASS]
            and not model_config.get("auto_map") and not model_config.get("quantization_config"),
            "qwen30_pinned_architecture_mismatch")
    receipt = read_json(Path(config["model_download_receipt"]))
    weights = receipt.get("weight_manifest")
    require(receipt.get("repo") == MODEL_ID and receipt.get("revision") == REVISION
            and receipt.get("license") == "apache-2.0" and receipt.get("complete") is True
            and receipt.get("config_sha256") == CONFIG_SHA256
            and receipt.get("weight_manifest_sha256") == canonical_sha256(weights), "qwen30_weight_receipt_mismatch")
    require(isinstance(weights, list) and bool(weights) and all(isinstance(row, dict)
            and row.get("name") == Path(row["name"]).name and row["name"].endswith(".safetensors")
            and re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256")))
            and type(row.get("bytes")) is int and row["bytes"] > 0 for row in weights),
            "qwen30_weight_inventory_invalid")
    require({row["name"] for row in weights} == {p.name for p in model_root.glob("*.safetensors")},
            "qwen30_weight_inventory_mismatch")
    if config["verify_weight_hashes"]:
        for row in weights:
            path = model_root / row["name"]
            require(path.stat().st_size == row["bytes"] and file_sha256(path) == row["sha256"],
                    "qwen30_weight_hash_mismatch")
    metadata_files = {p.name: file_sha256(p) for p in model_root.iterdir()
                      if p.is_file() and not p.name.endswith(".safetensors")}
    return {"rows": rows, "stats": stats, "model_config": model_config, "weight_receipt": receipt,
            "model_metadata_files": metadata_files, "model_metadata_sha256": canonical_sha256(metadata_files),
            "source_sha256": file_sha256(Path(__file__))}


class Qwen30Collator:
    """Public composition; only supported cash/ID boolean values receive loss."""
    def __init__(self, processor: Any, image_root: Path, *, max_length: int = 1024,
                 max_pixels: int = 262144, image_token_id: int = 151655,
                 visual_ids: set[int] | None = None):
        self.processor, self.image_root = processor, Path(image_root)
        self.max_length, self.max_pixels = max_length, max_pixels
        self.image_token_id = image_token_id
        self.visual_ids = visual_ids or {151655, 151656, 151652, 151653}

    def _crop(self, row: dict):
        from PIL import Image, ImageDraw
        comp = row["composition"]
        canvas = Image.new("RGB", (comp["width"], comp["height"]), tuple(comp["background"]))
        for component in comp["components"]:
            with Image.open(safe_image_path(self.image_root, component["image_path"])) as source:
                box = component["crop_box_xyxy"]
                require(box[2] <= source.width and box[3] <= source.height, "qwen30_crop_outside_image")
                crop = source.convert("RGB").crop(tuple(box))
            alpha = None
            if component["mask_kind"] == "publisher_quadrangle":
                alpha = Image.new("L", crop.size, 0)
                points = [(round(x - box[0]), round(y - box[1])) for x, y in component["source_quad"]]
                ImageDraw.Draw(alpha).polygon(points, fill=255)
            size = (component["width"], component["height"])
            crop = crop.resize(size, Image.Resampling.BILINEAR)
            if alpha is not None:
                alpha = alpha.resize(size, Image.Resampling.NEAREST)
            canvas.paste(crop, (component["x"], component["y"]), alpha)
        return canvas

    def _texts(self, row: dict, *, prompt_only: bool = False) -> tuple[str, str]:
        user = [{"role": "system", "content": [{"type": "text", "text": row["system"]}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": row["prompt"]}]}]
        prompt_text = self.processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True,
                                                        enable_thinking=False)
        if prompt_only:
            return prompt_text, ""
        answer, _ = target_text(row)
        full = [*user, {"role": "assistant", "content": [{"type": "text", "text": answer}]}]
        full_text = self.processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False,
                                                      enable_thinking=False)
        require(full_text.startswith(prompt_text), "qwen30_chat_template_prefix_mismatch")
        return prompt_text, full_text

    def _process(self, text: str, crop: Any) -> dict:
        size = self.processor.image_processor.size
        batch = dict(self.processor(text=[text], images=[crop], padding=False, truncation=False,
            return_tensors="pt", images_kwargs={"size": {"longest_edge": self.max_pixels,
                "shortest_edge": min(self.max_pixels, size["shortest_edge"])}}))
        required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
        require(required <= set(batch) and not any("audio" in k or "video" in k for k in batch),
                "qwen30_image_only_processor_required")
        ids = batch["input_ids"][0].tolist()
        require(len(ids) <= self.max_length and 0 < ids.count(self.image_token_id),
                "qwen30_visual_or_sequence_budget_exceeded")
        return batch

    def prompt_inputs(self, row: dict) -> dict:
        prompt, _ = self._texts(row, prompt_only=True)
        return self._process(prompt, self._crop(row))

    def __call__(self, rows: list[dict]) -> dict:
        require(len(rows) == 1, "qwen30_one_image_microbatch_required")
        import torch
        row, crop = rows[0], self._crop(rows[0])
        prompt_text, full_text = self._texts(row)
        prompt, batch = self._process(prompt_text, crop), self._process(full_text, crop)
        answer, spans = target_text(row)
        labels = masked_completion_labels(batch["input_ids"][0].tolist(), prompt["input_ids"][0].tolist(),
            self.visual_ids, tokenizer=self.processor.tokenizer, prompt_text=prompt_text, full_text=full_text,
            target_text=answer, target_spans=spans, attention_mask=batch["attention_mask"][0].tolist(),
            prompt_attention_mask=prompt["attention_mask"][0].tolist())
        require(all(key in prompt and batch[key].equal(prompt[key])
                    for key in ("pixel_values", "image_grid_thw")), "qwen30_prompt_pixels_mismatch")
        batch["labels"] = torch.tensor([labels], dtype=torch.long)
        return batch


def target_text(row: dict) -> tuple[str, list[tuple[int, int]]]:
    """Only annotated boolean VALUE spans carry loss; placeholders are unscored."""
    text, spans = "{", []
    for index, name in enumerate(DESK_FIELDS):
        if index:
            text += ","
        text += json.dumps(name) + ":"
        start = len(text)
        text += "true" if row["target"][name] else "false"
        if name in row["supervision_fields"]:
            spans.append((start, len(text)))
    return text + "}", spans


def inventory_targets(model: Any, model_config: dict) -> list[str]:
    targets = sorted(name for name, module in model.named_modules()
                     if TARGET_RE.fullmatch(name) and hasattr(module, "weight"))
    text = model_config["text_config"]
    layers = text["num_hidden_layers"]
    expected = {f"model.language_model.layers.{i}.self_attn.{projection}"
                for i in range(layers) for projection in ("q_proj", "v_proj")}
    require(set(targets) == expected, "qwen30_language_projection_inventory_mismatch")
    return targets


def completion_loss(model: Any, batch: dict):
    """Compute LM logits only for answer tokens, saving a large vocabulary tensor."""
    import torch
    import torch.nn.functional as F
    labels = batch["labels"]
    positions = torch.nonzero(labels[0] != -100, as_tuple=False).flatten()
    require(positions.numel() > 0 and positions[0].item() > 0,
            "qwen30_supported_boolean_tokens_required")
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    logits = model(**inputs, use_cache=False, logits_to_keep=positions - 1).logits
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels[:, positions].reshape(-1))


def balanced_indices(rows: list[dict], count: int, seed: int) -> list[int]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        buckets[row["condition"]].append(index)
    rng, order, cursors = random.Random(seed), [], Counter()
    classes = sorted(buckets)
    for items in buckets.values():
        rng.shuffle(items)
    for i in range(count):
        name = classes[i % len(classes)]
        items = buckets[name]
        cursor = cursors[name]
        if cursor and cursor % len(items) == 0:
            rng.shuffle(items)
        order.append(items[cursor % len(items)])
        cursors[name] += 1
    return order


def adapter_digest(model: Any) -> str:
    result = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            result.update(name.encode())
            result.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def frozen_digest(model: Any) -> str:
    """Hash all frozen parameter bytes in bounded CPU chunks, without pixel I/O."""
    import torch
    result = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            continue
        result.update(name.encode())
        result.update(str(tuple(parameter.shape)).encode())
        result.update(str(parameter.dtype).encode())
        require(parameter.is_contiguous(), "qwen30_frozen_parameter_noncontiguous")
        raw = parameter.detach().as_subclass(torch.Tensor).view(torch.uint8).reshape(-1)
        for chunk in raw.split(32 * 1024**2):
            result.update(chunk.cpu().numpy().tobytes())
    return result.hexdigest()


def set_offline() -> None:
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      DO_NOT_TRACK="1", WANDB_DISABLED="true", TOKENIZERS_PARALLELISM="false",
                      HF_HUB_DISABLE_PROGRESS_BARS="1", TQDM_DISABLE="1")


def load_training_model(config: dict, checked: dict, resume: Path | None = None):
    set_offline()
    require(all(importlib.metadata.version(k) == value for k, value in PINS.items()),
            "qwen30_runtime_version_mismatch")
    import torch
    from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration
    from peft import LoraConfig, PeftModel, get_peft_model
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1
            and torch.cuda.is_bf16_supported(), "qwen30_one_cuda_bf16_device_required")
    free, total = torch.cuda.mem_get_info(0)
    require(free >= config["minimum_free_cuda_gib"] * 1024**3, "qwen30_insufficient_free_cuda")
    budget = config["gpu_memory_budget_gib"] * 1024**3
    require(budget <= total, "qwen30_gpu_budget_exceeds_device")
    torch.cuda.set_per_process_memory_fraction(budget / total, 0)
    torch.cuda.reset_peak_memory_stats(0)
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(config["model_dir"], local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    require(not getattr(model, "is_loaded_in_4bit", False) and model.__class__.__name__ == MODEL_CLASS
            and all(p.device.type == "cuda" and p.dtype == torch.bfloat16 for p in model.parameters()),
            "qwen30_exact_bf16_cuda_model_required")
    model.requires_grad_(False)
    targets = inventory_targets(model, checked["model_config"])
    if resume:
        model = PeftModel.from_pretrained(model, str(resume), is_trainable=True, local_files_only=True)
    else:
        lora = LoraConfig(task_type="CAUSAL_LM", r=config["lora_rank"], lora_alpha=config["lora_alpha"],
                          lora_dropout=config["lora_dropout"], target_modules=targets, bias="none",
                          base_model_name_or_path=MODEL_ID, revision=REVISION)
        model = get_peft_model(model, lora)
    for adapter in model.peft_config.values():
        adapter.base_model_name_or_path, adapter.revision = MODEL_ID, REVISION
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    require(bool(trainable) and all(".lora_" in name and ".language_model.layers." in name
            and any(f".{projection}." in name for projection in ("q_proj", "v_proj"))
            for name, _ in trainable), "qwen30_unexpected_trainable_parameter")
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    # Non-reentrant checkpointing supports frozen embeddings without image gradients.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    processor = AutoProcessor.from_pretrained(config["model_dir"], local_files_only=True, trust_remote_code=False,
                                             max_pixels=config["max_pixels"])
    require(processor.__class__.__name__ == "Qwen3VLProcessor", "qwen30_native_processor_required")
    return model, processor, trainable, targets


def save_checkpoint(model: Any, optimizer: Any, output: Path, state: dict, config_hash: str,
                    manifest_hash: str) -> Path:
    import torch
    target = output / f"checkpoint-{state['optimizer_steps']:04d}"
    require(not target.exists(), "qwen30_checkpoint_already_exists")
    staging = target.with_name(target.name + ".partial")
    require(not staging.exists(), "qwen30_checkpoint_partial_exists")
    staging.mkdir(parents=True)
    model.save_pretrained(staging, safe_serialization=True)
    torch.save({"optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()}, staging / "optimizer.pt")
    write_json(staging / "trainer_state.json", {"schema_version": SCHEMA, "optimizer_steps": state["optimizer_steps"],
        "config_sha256": config_hash, "manifest_sha256": manifest_hash,
        "metrics": state["steps"], "initial_adapter_sha256": state["initial_adapter_sha256"],
        "frozen_before_sha256": state["frozen_before_sha256"],
        "source_sha256": state["source_sha256"], "weight_manifest_sha256": state["weight_manifest_sha256"],
        "model_metadata_sha256": state["model_metadata_sha256"],
        "cumulative_training_seconds": state.get("training_seconds", 0)})
    write_json(staging / "checkpoint_complete.json", {"schema_version": "forte.qwen30-lora.checkpoint.v1",
        "complete": True, "files": {path.name: file_sha256(path) for path in staging.iterdir() if path.is_file()}})
    staging.rename(target)
    return target


def validate_checkpoint(path: Path) -> dict:
    receipt = read_json(path / "checkpoint_complete.json")
    require(receipt.get("schema_version") == "forte.qwen30-lora.checkpoint.v1" and receipt.get("complete") is True,
            "qwen30_resume_checkpoint_incomplete")
    files = receipt.get("files")
    require(isinstance(files, dict) and {"adapter_config.json", "adapter_model.safetensors", "optimizer.pt",
            "trainer_state.json"} <= set(files), "qwen30_resume_checkpoint_files_missing")
    require(set(files) | {"checkpoint_complete.json"} == {p.name for p in path.iterdir() if p.is_file()},
            "qwen30_resume_checkpoint_inventory_mismatch")
    for name, digest in files.items():
        require(name == Path(name).name and file_sha256(path / name) == digest,
                "qwen30_resume_checkpoint_hash_mismatch")
    state = read_json(path / "trainer_state.json")
    require(state.get("schema_version") == SCHEMA and type(state.get("optimizer_steps")) is int
            and state["optimizer_steps"] > 0, "qwen30_resume_state_invalid")
    return state


def train(config: dict, checked: dict, *, smoke: bool = False, resume: Path | None = None) -> dict:
    output = Path(config["output_dir"])
    if smoke:
        output = output.with_name(output.name + "-smoke")
    require(resume is not None or not output.exists(), "qwen30_output_exists_use_explicit_resume")
    output.mkdir(parents=True, exist_ok=True)
    config_hash = canonical_sha256(config)
    previous = None
    if resume:
        require(resume.resolve().is_relative_to(output.resolve()), "qwen30_resume_outside_run")
        previous = validate_checkpoint(resume)
        require(previous.get("config_sha256") == config_hash
                and previous.get("manifest_sha256") == config["expected_manifest_sha256"]
                and previous.get("source_sha256") == checked["source_sha256"]
                and previous.get("weight_manifest_sha256") == checked["weight_receipt"]["weight_manifest_sha256"]
                and previous.get("model_metadata_sha256") == checked["model_metadata_sha256"],
                "qwen30_resume_provenance_mismatch")
    run = {"schema_version": SCHEMA, "state": "loading", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID, "revision": REVISION, "config_sha256": config_hash,
        "manifest_sha256": config["expected_manifest_sha256"], "source_sha256": checked["source_sha256"],
        "weight_manifest_sha256": checked["weight_receipt"]["weight_manifest_sha256"],
        "model_metadata_sha256": checked["model_metadata_sha256"],
        "model_metadata_files": checked["model_metadata_files"],
        "public_data": checked["stats"], "bank_video_labels_used": False, "production_enabled": False,
        "qualification_state": "not_evaluated", "mode": "smoke" if smoke else "train", "optimizer_steps": 0,
        "steps": [], "runtime_versions": PINS, "quantization": "bf16", "base_dtype": "bfloat16",
        "target_loss_policy": "supported_cash_id_boolean_values_only",
        "adapter_scope": "desk_cash_id_refinement_only",
        "unsupported_desk_values": "false_context_placeholders_loss_masked_never_ground_truth",
        "checkpoint_selection": "predeclared_final_step_or_time_budget_no_bank_scores"}
    run["training_hyperparameters_predeclared"] = ["optimizer_steps", "grad_accum_steps", "learning_rate",
        "warmup_steps", "weight_decay", "max_grad_norm", "lora_rank", "lora_alpha", "lora_dropout",
        "max_length", "max_pixels", "seed"]
    run["training_hyperparameter_count"] = 12
    run["training_parameters_tuned_to_bank_labels"] = 0
    run["pilot"] = {"steps": config["pilot_steps"], "maximum_seconds": config["pilot_max_seconds"],
                    "scope": "optimizer_loop_only_excludes_model_load_and_parameter_digest", "state": "pending"}
    write_json(output / "run.json", run)
    model = optimizer = None
    try:
        require(datetime.now(timezone.utc) < deadline(config), "qwen30_deadline_already_passed")
        emit({"phase": "model_load", "mode": run["mode"]})
        with quiet_libraries():
            model, processor, trainable, targets = load_training_model(config, checked, resume)
        import torch
        collator = Qwen30Collator(processor, Path(config["image_root"]), max_length=config["max_length"],
            max_pixels=config["max_pixels"], image_token_id=checked["model_config"]["image_token_id"])
        optimizer = torch.optim.AdamW([p for _, p in trainable], lr=config["learning_rate"],
                                     weight_decay=config["weight_decay"])
        run.update(state="training", trainable_parameters=sum(p.numel() for _, p in trainable),
                   target_modules=targets, gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                   initial_adapter_sha256=adapter_digest(model))
        emit({"phase": "frozen_parameter_digest"})
        run["frozen_before_sha256"] = frozen_digest(model)
        if previous:
            state = torch.load(resume / "optimizer.pt", map_location="cpu", weights_only=True)
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["torch_rng"])
            torch.cuda.set_rng_state_all(state["cuda_rng"])
            run.update(optimizer_steps=previous["optimizer_steps"], steps=previous["metrics"],
                       initial_adapter_sha256=previous["initial_adapter_sha256"], resumed=True)
            require(run["frozen_before_sha256"] == previous["frozen_before_sha256"],
                    "qwen30_resume_frozen_parameter_mismatch")
        rows = [r for r in checked["rows"] if r["split"] == "train"]
        target_steps = 1 if smoke else config["optimizer_steps"]
        order = balanced_indices(rows, target_steps * config["grad_accum_steps"], config["seed"])
        start = time.monotonic()
        previous_seconds = previous.get("cumulative_training_seconds", 0) if previous else 0
        model.train()
        for name, module in model.named_modules():
            if name.endswith(".visual"):
                module.eval()
        write_json(output / "run.json", run)
        for step in range(run["optimizer_steps"], target_steps):
            if datetime.now(timezone.utc) >= deadline(config) or previous_seconds + time.monotonic() - start >= config["max_training_seconds"]:
                run["state"] = "stopped_budget"
                break
            if step < config["pilot_steps"] and previous_seconds + time.monotonic() - start >= config["pilot_max_seconds"]:
                run["state"], run["pilot"]["state"] = "stopped_throughput_pilot", "too_slow"
                break
            optimizer.zero_grad(set_to_none=True)
            losses, step_start = [], time.monotonic()
            lr = config["learning_rate"] * min(1, (step + 1) / max(1, config["warmup_steps"]))
            for group in optimizer.param_groups:
                group["lr"] = lr
            interrupted = False
            for micro in range(config["grad_accum_steps"]):
                # A partially accumulated step is never committed on deadline.
                elapsed = previous_seconds + time.monotonic() - start
                if datetime.now(timezone.utc) >= deadline(config) or elapsed >= config["max_training_seconds"]:
                    run["state"], interrupted = "stopped_budget", True
                    break
                if step < config["pilot_steps"] and elapsed >= config["pilot_max_seconds"]:
                    run["state"], run["pilot"]["state"], interrupted = "stopped_throughput_pilot", "too_slow", True
                    break
                row = rows[order[step * config["grad_accum_steps"] + micro]]
                with quiet_libraries():
                    batch = {k: v.to("cuda:0") for k, v in collator([row]).items()}
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = completion_loss(model, batch)
                    require(bool(torch.isfinite(loss).item()), "qwen30_nonfinite_loss")
                    losses.append(float(loss.detach().item()))
                    (loss / config["grad_accum_steps"]).backward()
                del loss, batch
            if interrupted:
                optimizer.zero_grad(set_to_none=True)
                break
            gradients = [p.grad for _, p in trainable if p.grad is not None]
            require(bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients)
                    and any(bool(torch.count_nonzero(g)) for g in gradients), "qwen30_invalid_or_zero_gradients")
            norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in trainable], config["max_grad_norm"]).item())
            require(math.isfinite(norm), "qwen30_nonfinite_gradient_norm")
            elapsed = previous_seconds + time.monotonic() - start
            if datetime.now(timezone.utc) >= deadline(config) or elapsed >= config["max_training_seconds"]:
                optimizer.zero_grad(set_to_none=True)
                run["state"] = "stopped_budget"
                break
            if step < config["pilot_steps"] and elapsed >= config["pilot_max_seconds"]:
                optimizer.zero_grad(set_to_none=True)
                run["state"], run["pilot"]["state"] = "stopped_throughput_pilot", "too_slow"
                break
            optimizer.step()
            torch.cuda.synchronize()
            require(all(bool(torch.isfinite(p).all()) for _, p in trainable), "qwen30_nonfinite_adapter")
            metric = {"step": step + 1, "mean_loss": sum(losses) / len(losses), "gradient_norm": norm,
                      "learning_rate": lr, "seconds": time.monotonic() - step_start,
                      "peak_allocated_mib": torch.cuda.max_memory_allocated(0) / 1024**2,
                      "peak_reserved_mib": torch.cuda.max_memory_reserved(0) / 1024**2}
            run["optimizer_steps"] = step + 1
            run["steps"].append(metric)
            run["training_seconds"] = previous_seconds + time.monotonic() - start
            if step + 1 == config["pilot_steps"]:
                run["pilot"].update(state="passed", measured_seconds=run["training_seconds"])
            write_json(output / "run.json", run)
            emit(metric)
            if (step + 1) % config["checkpoint_every"] == 0:
                with quiet_libraries():
                    path = save_checkpoint(model, optimizer, output, run, config_hash, config["expected_manifest_sha256"])
                run["last_checkpoint"] = path.name
        require(run["optimizer_steps"] > 0, "qwen30_no_completed_optimizer_step")
        optimizer.zero_grad(set_to_none=True)
        run["final_adapter_sha256"] = adapter_digest(model)
        require(run["final_adapter_sha256"] != run["initial_adapter_sha256"], "qwen30_adapter_weights_unchanged")
        emit({"phase": "frozen_parameter_digest_final"})
        run["frozen_after_sha256"] = frozen_digest(model)
        require(run["frozen_after_sha256"] == run["frozen_before_sha256"], "qwen30_frozen_parameters_changed")
        last_checkpoint = output / f"checkpoint-{run['optimizer_steps']:04d}"
        if not last_checkpoint.exists():
            with quiet_libraries():
                last_checkpoint = save_checkpoint(model, optimizer, output, run, config_hash, config["expected_manifest_sha256"])
        run["last_checkpoint"] = last_checkpoint.name
        final = output / "adapter"
        if final.exists():
            require(previous is not None, "qwen30_final_adapter_already_exists")
            archived = output / f"adapter-before-resume-{previous['optimizer_steps']:04d}"
            require(not archived.exists(), "qwen30_resume_adapter_archive_exists")
            final.rename(archived)
        with quiet_libraries():
            model.save_pretrained(final, safe_serialization=True)
            processor.save_pretrained(output / "processor")
        run["adapter_files"] = {p.name: file_sha256(p) for p in final.iterdir() if p.is_file()}
        run["processor_files"] = {p.name: file_sha256(p) for p in (output / "processor").iterdir() if p.is_file()}
        if run["state"] == "training":
            run["state"] = "smoke_complete" if smoke else "trained_not_qualified"
        run["training_seconds"] = previous_seconds + time.monotonic() - start
        run["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "run.json", run)
        return {"ok": True, "state": run["state"], "optimizer_steps": run["optimizer_steps"],
                "trainable_parameters": run["trainable_parameters"], "adapter_sha256": run["final_adapter_sha256"]}
    except Exception as error:
        run["state"] = "failed"
        run["error_type"] = type(error).__name__
        if isinstance(error, TrainingGateError):
            run["error_code"] = str(error)
        write_json(output / "run.json", run)
        raise
    finally:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)


def synthetic_check() -> dict:
    """CPU/stdlib checks: contamination, groups, balanced draws, q/v inventory."""
    rows = []
    for split in ("train", "validation"):
        for condition in sorted(CONDITIONS):
            objects = {"blank": [], "note_only": ["banknote"], "id_only": ["id_card"],
                       "both": ["banknote", "id_card"]}[condition]
            components = []
            for index, name in enumerate(objects):
                components.append({"source_dataset": "MIDV-Holo" if name == "id_card" else "dny3dgvvw8",
                    "source_group": split + name, "image_sha256": hashlib.sha256((split + name).encode()).hexdigest(),
                    "image_path": "unused.jpg", "object_class": name, "crop_box_xyxy": [0, 0, 8, 8],
                    "source_quad": [[0, 0], [8, 0], [8, 8], [0, 8]] if name == "id_card" else None,
                    "mask_kind": "publisher_quadrangle" if name == "id_card" else "bbox_unknown_background",
                    "x": index * 128, "y": 0, "width": 100, "height": 100})
            target = {key: False for key in DESK_FIELDS}
            target.update(cash="banknote" in objects, id_or_document="id_card" in objects)
            rows.append({"condition": condition, "split": split, "system": SYSTEM, "prompt": DESK_Q,
                "target": target, "supervision_fields": ["cash"] if condition == "note_only" else ["cash", "id_or_document"],
                "composition": {"width": 448, "height": 448, "background": [127, 127, 127], "components": components}})
    require(validate_rows(rows)["rows"] == 8, "qwen30_synthetic_rows_failed")
    leaked = next(row for row in rows if row["split"] == "train" and row["condition"] == "both")
    bad = [*rows, dict(leaked, split="validation")]
    try:
        validate_rows(bad)
        raise AssertionError("split leakage accepted")
    except TrainingGateError as error:
        require(str(error) == "qwen30_source_group_split_leakage", "qwen30_synthetic_wrong_leakage_gate")
    for target in ({"cash": True, "verdict": "violation"}, {"description": "blocked"}):
        try:
            validate_rows([dict(rows[0], target=target), *rows[1:]])
            raise AssertionError("free-form or verdict target accepted")
        except TrainingGateError:
            pass
    indices = balanced_indices(rows[:4], 20, 0)
    require(Counter(indices) == {0: 5, 1: 5, 2: 5, 3: 5}, "qwen30_synthetic_balance_failed")
    class CharTokenizer:
        is_fast = True
        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            value = {"input_ids": [ord(character) for character in text]}
            if return_offsets_mapping:
                value["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
            return value
    answer, spans = target_text(leaked)
    prompt, full = "Q", "Q" + answer
    labels = masked_completion_labels([ord(c) for c in full], [ord(c) for c in prompt], {9999},
        tokenizer=CharTokenizer(), prompt_text=prompt, full_text=full, target_text=answer, target_spans=spans)
    require("".join(chr(value) for value in labels if value != -100) == "truetrue",
            "qwen30_synthetic_boolean_only_mask_failed")
    note = next(row for row in rows if row["condition"] == "note_only")
    try:
        validate_rows([*rows, dict(note, supervision_fields=["cash", "id_or_document"])])
        raise AssertionError("unsupported note ID absence accepted")
    except TrainingGateError as error:
        require(str(error) == "qwen30_unsupported_absence_supervision", "qwen30_synthetic_wrong_absence_gate")
    class Module:
        weight = True
    class FakeModel:
        def named_modules(self):
            return [(name, Module()) for name in ("model.language_model.layers.0.self_attn.q_proj",
                "model.language_model.layers.0.self_attn.v_proj", "model.language_model.layers.1.self_attn.q_proj",
                "model.language_model.layers.1.self_attn.v_proj", "model.visual.blocks.0.attn.q_proj")]
    targets = inventory_targets(FakeModel(), {"text_config": {"num_hidden_layers": 2,
            "layer_types": ["sliding_attention", "full_attention"], "attention_k_eq_v": True}})
    require(len(targets) == 4 and not any("visual" in x for x in targets), "qwen30_synthetic_targets_failed")
    return {"ok": True, "checks": 8, "pixels_read": 0, "bank_video_labels_read": 0,
            "checks_scope": "stdlib_metadata_masks_and_projection_inventory_not_gpu_training"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=("preflight", "smoke", "train", "synthetic-check"), default="preflight")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "synthetic-check":
            emit(synthetic_check())
            return 0
        require(args.config is not None, "qwen30_config_required")
        set_offline()
        config = resolve_config(args.config)
        checked = preflight(config)
        if args.mode == "preflight":
            emit({"ok": True, "state": "preflight_passed_not_trained", "public_data": checked["stats"],
                  "manifest_sha256": config["expected_manifest_sha256"]})
        else:
            emit(train(config, checked, smoke=args.mode == "smoke", resume=args.resume))
        return 0
    except Exception as error:
        result = {"ok": False, "state": "failed", "error_type": type(error).__name__}
        if isinstance(error, TrainingGateError):
            result["error_code"] = str(error)
        emit(result)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
