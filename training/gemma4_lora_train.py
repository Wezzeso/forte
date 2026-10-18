"""Offline Gemma 4 public-object QLoRA; never reads bank-video labels.

The recipe is fixed before evaluation. Checkpoints are selected by the declared
step/time budget, never by Forte's 40-video scores. Stdout contains JSON metrics
and fixed error codes only; arbitrary library output is suppressed. Images are
decoded only inside this local training process and never saved or displayed.

python -m training.gemma4_lora_train --config CONFIG --mode preflight
python -m training.gemma4_lora_train --config CONFIG --mode smoke
python -m training.gemma4_lora_train --config CONFIG --mode train
python -m training.gemma4_lora_train --config CONFIG --mode train --resume CHECKPOINT
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

from training.data import completion_labels
from training.guards import TrainingGateError, canonical_sha256, file_sha256, require

MODEL_ID = "google/gemma-4-31B-it"
REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
MODEL_CLASS = "Gemma4ForConditionalGeneration"
CONFIG_SHA256 = "e967dd38bc5cfd38bd09a995a7bf4a754075df2b46aba68f7fbb5a791e6d8dd1"
PINS = {"torch": "2.13.0+cu129", "transformers": "5.17.0", "peft": "0.21.2",
        "accelerate": "1.15.0", "bitsandbytes": "0.50.2", "pillow": "12.3.0"}
CLASSES = frozenset({"banknote", "id_card", "other", "unclear"})
SOURCE_DATASETS = frozenset({"dny3dgvvw8", "MIDV-Holo"})
TARGET_RE = re.compile(r"model\.language_model\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)$")
SCHEMA = "forte.gemma4-public-lora.run.v1"
PATH_FIELDS = ("model_dir", "model_download_receipt", "model_weight_manifest", "manifest",
               "image_root", "output_dir")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        raise TrainingGateError("gemma4_invalid_or_missing_json") from None


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
    require(isinstance(config, dict), "gemma4_config_object_required")
    for key in PATH_FIELDS:
        value = config.get(key)
        require(isinstance(value, str) and bool(value) and "://" not in value,
                "gemma4_local_paths_required")
        config[key] = str((path.resolve().parent / value).resolve())
    return config


def validate_config(config: dict) -> None:
    require(config.get("schema_version") == "forte.gemma4-public-lora.config.v1",
            "gemma4_config_schema_invalid")
    require(config.get("model_id") == MODEL_ID and config.get("revision") == REVISION,
            "gemma4_base_identity_mismatch")
    require(config.get("runtime_versions") == PINS, "gemma4_runtime_pins_mismatch")
    require(config.get("training_data_policy") == "public_published_object_labels_only"
            and config.get("bank_video_labels_used") is False
            and config.get("production_enabled") is False, "gemma4_public_only_staging_required")
    require(isinstance(config.get("recipe_principle"), str) and bool(config["recipe_principle"]),
            "gemma4_visual_principle_required")
    for key, low, high in (("optimizer_steps", 1, 200), ("grad_accum_steps", 1, 16),
        ("max_length", 128, 4096), ("max_soft_tokens", 70, 1120), ("seed", 0, 2**31-1),
        ("lora_rank", 1, 32), ("lora_alpha", 1, 64), ("checkpoint_every", 1, 100),
        ("pilot_steps", 1, 100), ("pilot_max_seconds", 10, 1000),
        ("warmup_steps", 0, 100), ("max_training_seconds", 30, 3600)):
        require(type(config.get(key)) is int and low <= config[key] <= high,
                "gemma4_training_parameter_invalid")
    for key, low, high in (("learning_rate", 1e-7, .001), ("lora_dropout", 0, .2),
        ("weight_decay", 0, 1), ("max_grad_norm", .01, 10),
        ("minimum_free_cuda_gib", 1, 80), ("gpu_memory_budget_gib", 10, 79)):
        value = config.get(key)
        require(type(value) in {int, float} and math.isfinite(value) and low <= value <= high,
                "gemma4_training_parameter_invalid")
    require(config.get("quantization") == "nf4" and type(config.get("verify_weight_hashes")) is bool,
            "gemma4_nf4_recipe_required")
    require(config["warmup_steps"] < config["optimizer_steps"], "gemma4_warmup_exceeds_steps")
    require(re.fullmatch(r"[0-9a-f]{64}", str(config.get("expected_manifest_sha256"))) is not None,
            "gemma4_manifest_hash_pin_required")
    stop = deadline(config)
    require(stop.tzinfo is not None and stop.utcoffset().total_seconds() == 0,
            "gemma4_utc_deadline_required")


def deadline(config: dict) -> datetime:
    try:
        return datetime.fromisoformat(config["stop_at_utc"].replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        raise TrainingGateError("gemma4_utc_deadline_required") from None


def safe_image_path(root: Path, value: str) -> Path:
    require(isinstance(value, str) and bool(value) and "://" not in value,
            "gemma4_image_path_invalid")
    path = (root / value).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file()
            and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"},
            "gemma4_image_outside_public_root")
    return path


def validate_rows(rows: list[dict]) -> dict:
    require(bool(rows), "gemma4_manifest_empty")
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    hashes: dict[str, set[str]] = defaultdict(set)
    prompts, counts = set(), Counter()
    for row in rows:
        require(isinstance(row, dict) and row.get("split") in {"train", "validation"},
                "gemma4_train_validation_rows_only")
        require(row.get("source_dataset") in SOURCE_DATASETS
                and isinstance(row.get("source_group"), str) and bool(row["source_group"]),
                "gemma4_published_source_required")
        require(row.get("origin", "public") == "public", "gemma4_nonpublic_source_forbidden")
        target = row.get("target")
        require(isinstance(target, dict) and set(target) == {"object_class"}
                and target["object_class"] in CLASSES, "gemma4_closed_object_target_required")
        prompt = row.get("prompt")
        require(isinstance(prompt, str) and 10 <= len(prompt) <= 2048
                and "<|" not in prompt and "://" not in prompt,
                "gemma4_fixed_question_required")
        prompts.add(prompt)
        for key in ("source_group", "template_group", "physical_document_group", "scene_group", "source_image_group"):
            if row.get(key):
                require(isinstance(row[key], str), "gemma4_public_group_invalid")
                groups[(key, row[key])].add(row["split"])
        for key in ("content_sha256", "image_sha256", "source_content_sha256"):
            if row.get(key):
                require(re.fullmatch(r"[0-9a-f]{64}", row[key]) is not None,
                        "gemma4_source_hash_invalid")
                hashes[row[key]].add(row["split"])
        box = row.get("crop_box_xyxy")
        require(isinstance(box, list) and len(box) == 4 and all(type(x) is int for x in box)
                and 0 <= box[0] < box[2] and 0 <= box[1] < box[3], "gemma4_crop_box_invalid")
        transform = row.get("transform", {})
        require(isinstance(transform, dict) and set(transform) <= {"rotation_degrees", "short_side", "brightness"},
                "gemma4_augmentation_invalid")
        for key, low, high, default in (("rotation_degrees", -12, 12, 0),
            ("short_side", 0, 1024, 0), ("brightness", .7, 1.3, 1)):
            value = transform.get(key, default)
            require(type(value) in {int, float} and math.isfinite(value) and low <= value <= high,
                    "gemma4_augmentation_invalid")
        counts[(row["split"], target["object_class"])] += 1
    require(len(prompts) == 1, "gemma4_per_sample_questions_forbidden")
    require(all(len(splits) == 1 for splits in groups.values())
            and all(len(splits) == 1 for splits in hashes.values()), "gemma4_source_group_split_leakage")
    require(all(counts[(split, name)] for split in ("train", "validation")
                for name in ("banknote", "id_card")), "gemma4_both_objects_each_split_required")
    return {"rows": len(rows), "source_groups": len({r["source_group"] for r in rows}),
            "lineage_groups": len(groups),
            "class_counts": {f"{split}/{name}": count for (split, name), count in sorted(counts.items())}}


def load_rows(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except (OSError, ValueError):
        raise TrainingGateError("gemma4_manifest_invalid") from None


def preflight(config: dict) -> dict:
    validate_config(config)
    manifest_path = Path(config["manifest"])
    require(file_sha256(manifest_path) == config["expected_manifest_sha256"], "gemma4_manifest_hash_mismatch")
    rows = load_rows(manifest_path)
    stats = validate_rows(rows)
    root = Path(config["image_root"])
    seen = set()
    for row in rows:
        path = safe_image_path(root, row["image_path"])
        digest = row.get("image_sha256", row.get("content_sha256"))
        require(isinstance(digest, str), "gemma4_public_image_hash_required")
        if (path, digest) not in seen:
            require(file_sha256(path) == digest, "gemma4_public_image_hash_mismatch")
            seen.add((path, digest))
    model_root = Path(config["model_dir"])
    model_config = read_json(model_root / "config.json")
    require(file_sha256(model_root / "config.json") == CONFIG_SHA256
            and model_config.get("model_type") == "gemma4"
            and model_config.get("architectures") == [MODEL_CLASS]
            and not model_config.get("auto_map") and not model_config.get("quantization_config"),
            "gemma4_pinned_architecture_mismatch")
    receipt, weights = read_json(Path(config["model_download_receipt"])), read_json(Path(config["model_weight_manifest"]))
    require(receipt.get("repo") == MODEL_ID and receipt.get("revision") == REVISION
            and receipt.get("license") == "apache-2.0" and receipt.get("complete") is True
            and receipt.get("config_sha256") == CONFIG_SHA256
            and receipt.get("weight_manifest_sha256") == canonical_sha256(weights), "gemma4_weight_receipt_mismatch")
    require(isinstance(weights, dict) and bool(weights) and all(name == Path(name).name
            and name.endswith(".safetensors") and re.fullmatch(r"[0-9a-f]{64}", digest)
            for name, digest in weights.items()), "gemma4_weight_inventory_invalid")
    require(set(weights) == {p.name for p in model_root.glob("*.safetensors")}, "gemma4_weight_inventory_mismatch")
    if config["verify_weight_hashes"]:
        for name, digest in weights.items():
            require(file_sha256(model_root / name) == digest, "gemma4_weight_hash_mismatch")
    metadata_files = {p.name: file_sha256(p) for p in model_root.iterdir()
                      if p.is_file() and not p.name.endswith(".safetensors")}
    return {"rows": rows, "stats": stats, "model_config": model_config, "weight_receipt": receipt,
            "model_metadata_files": metadata_files, "model_metadata_sha256": canonical_sha256(metadata_files),
            "source_sha256": file_sha256(Path(__file__))}


class Gemma4Collator:
    """Single public crop; prompt and multimodal tokens receive no loss."""
    def __init__(self, processor: Any, image_root: Path, *, max_length: int = 1024,
                 max_soft_tokens: int = 280, image_token_id: int = 258880,
                 visual_ids: set[int] | None = None):
        self.processor, self.image_root = processor, Path(image_root)
        self.max_length, self.max_soft_tokens = max_length, max_soft_tokens
        self.image_token_id = image_token_id
        self.visual_ids = visual_ids or {258880, 255999, 258882, 258881, 256000, 258883}

    def _crop(self, row: dict):
        from PIL import Image, ImageEnhance
        with Image.open(safe_image_path(self.image_root, row["image_path"])) as source:
            box = row["crop_box_xyxy"]
            require(box[2] <= source.width and box[3] <= source.height, "gemma4_crop_outside_image")
            crop = source.convert("RGB").crop(tuple(box))
        transform = row.get("transform", {})
        angle = transform.get("rotation_degrees", 0)
        if angle:
            crop = crop.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=(127, 127, 127))
        side = int(transform.get("short_side", 0))
        if side and min(crop.size) > side:
            ratio = side / min(crop.size)
            size = (max(1, round(crop.width * ratio)), max(1, round(crop.height * ratio)))
            crop = crop.resize(size, Image.Resampling.BILINEAR)
        brightness = transform.get("brightness", 1)
        if brightness != 1:
            crop = ImageEnhance.Brightness(crop).enhance(brightness)
        return crop

    def _texts(self, row: dict, *, prompt_only: bool = False) -> tuple[str, str]:
        user = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": row["prompt"]}]}]
        prompt_text = self.processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True,
                                                        enable_thinking=False)
        if prompt_only:
            return prompt_text, ""
        answer = json.dumps(row["target"], sort_keys=True, separators=(",", ":"), allow_nan=False)
        full = [*user, {"role": "assistant", "content": [{"type": "text", "text": answer}]}]
        full_text = self.processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False,
                                                      enable_thinking=False)
        require(full_text.startswith(prompt_text), "gemma4_chat_template_prefix_mismatch")
        return prompt_text, full_text

    def _process(self, text: str, crop: Any) -> dict:
        batch = dict(self.processor(text=[text], images=[[crop]], padding=False, truncation=False,
                                   return_tensors="pt", images_kwargs={"max_soft_tokens": self.max_soft_tokens}))
        required = {"input_ids", "attention_mask", "pixel_values", "image_position_ids"}
        require(required <= set(batch) and not any("audio" in k or "video" in k for k in batch),
                "gemma4_image_only_processor_required")
        ids = batch["input_ids"][0].tolist()
        require(len(ids) <= self.max_length and 0 < ids.count(self.image_token_id) <= self.max_soft_tokens,
                "gemma4_visual_or_sequence_budget_exceeded")
        return batch

    def prompt_inputs(self, row: dict) -> dict:
        prompt, _ = self._texts(row, prompt_only=True)
        return self._process(prompt, self._crop(row))

    def __call__(self, rows: list[dict]) -> dict:
        require(len(rows) == 1, "gemma4_one_image_microbatch_required")
        import torch
        row, crop = rows[0], self._crop(rows[0])
        prompt_text, full_text = self._texts(row)
        prompt, batch = self._process(prompt_text, crop), self._process(full_text, crop)
        labels = completion_labels(batch["input_ids"][0].tolist(), prompt["input_ids"][0].tolist(),
            self.visual_ids, attention_mask=batch["attention_mask"][0].tolist(),
            prompt_attention_mask=prompt["attention_mask"][0].tolist())
        require(all(key in prompt and batch[key].equal(prompt[key])
                    for key in ("pixel_values", "image_position_ids")), "gemma4_prompt_pixels_mismatch")
        batch["labels"] = torch.tensor([labels], dtype=torch.long)
        return batch


def inventory_targets(model: Any, model_config: dict) -> list[str]:
    targets = sorted(name for name, module in model.named_modules()
                     if TARGET_RE.fullmatch(name) and hasattr(module, "weight"))
    text = model_config["text_config"]
    layers = text["num_hidden_layers"]
    types = text["layer_types"]
    kv_start = layers - text.get("num_kv_shared_layers", 0)
    expected = {f"model.language_model.layers.{i}.self_attn.q_proj" for i in range(layers)}
    expected |= {f"model.language_model.layers.{i}.self_attn.v_proj" for i in range(kv_start)
                 if not text.get("attention_k_eq_v", False) or types[i] == "sliding_attention"}
    require(set(targets) == expected, "gemma4_language_projection_inventory_mismatch")
    return targets


def completion_loss(model: Any, batch: dict):
    """Compute LM logits only for answer tokens, saving a large vocabulary tensor."""
    import torch
    import torch.nn.functional as F
    labels = batch["labels"]
    positions = torch.nonzero(labels[0] != -100, as_tuple=False).flatten()
    require(positions.numel() > 0 and positions[0].item() > 0
            and positions.tolist() == list(range(positions[0].item(), labels.shape[-1])),
            "gemma4_completion_suffix_required")
    count = positions.numel()
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    logits = model(**inputs, use_cache=False, logits_to_keep=count + 1).logits[:, :-1, :]
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels[:, -count:].reshape(-1))


def balanced_indices(rows: list[dict], count: int, seed: int) -> list[int]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        buckets[row["target"]["object_class"]].append(index)
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
        require(parameter.is_contiguous(), "gemma4_frozen_parameter_noncontiguous")
        raw = parameter.detach().view(torch.uint8).reshape(-1)
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
            "gemma4_runtime_version_mismatch")
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Gemma4ForConditionalGeneration
    from bitsandbytes.nn import Linear4bit
    from peft import LoraConfig, PeftModel, get_peft_model
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1
            and torch.cuda.is_bf16_supported(), "gemma4_one_cuda_bf16_device_required")
    free, total = torch.cuda.mem_get_info(0)
    require(free >= config["minimum_free_cuda_gib"] * 1024**3, "gemma4_insufficient_free_cuda")
    budget = config["gpu_memory_budget_gib"] * 1024**3
    require(budget <= total, "gemma4_gpu_budget_exceeds_device")
    torch.cuda.set_per_process_memory_fraction(budget / total, 0)
    torch.cuda.reset_peak_memory_stats(0)
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              llm_int8_skip_modules=["vision_tower", "embed_vision", "lm_head"])
    model = Gemma4ForConditionalGeneration.from_pretrained(config["model_dir"], local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, quantization_config=quant, device_map={"": 0},
        attn_implementation="sdpa")
    require(getattr(model, "is_loaded_in_4bit", False) and model.__class__.__name__ == MODEL_CLASS
            and all(p.device.type == "cuda" for p in model.parameters()), "gemma4_quantized_cuda_model_required")
    model.requires_grad_(False)
    require(not any(isinstance(module, Linear4bit) for module in model.model.vision_tower.modules()),
            "gemma4_vision_quantization_forbidden")
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
            for name, _ in trainable), "gemma4_unexpected_trainable_parameter")
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    # Avoid prepare_model_for_kbit_training's blanket FP32 cast of frozen vision
    # and giant token embeddings. Non-reentrant checkpointing supports frozen inputs.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    processor = AutoProcessor.from_pretrained(config["model_dir"], local_files_only=True, trust_remote_code=False)
    require(processor.__class__.__name__ == "Gemma4Processor", "gemma4_native_processor_required")
    return model, processor, trainable, targets


def save_checkpoint(model: Any, optimizer: Any, output: Path, state: dict, config_hash: str,
                    manifest_hash: str) -> Path:
    import torch
    target = output / f"checkpoint-{state['optimizer_steps']:04d}"
    require(not target.exists(), "gemma4_checkpoint_already_exists")
    staging = target.with_name(target.name + ".partial")
    require(not staging.exists(), "gemma4_checkpoint_partial_exists")
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
    write_json(staging / "checkpoint_complete.json", {"schema_version": "forte.gemma4-lora.checkpoint.v1",
        "complete": True, "files": {path.name: file_sha256(path) for path in staging.iterdir() if path.is_file()}})
    staging.rename(target)
    return target


def validate_checkpoint(path: Path) -> dict:
    receipt = read_json(path / "checkpoint_complete.json")
    require(receipt.get("schema_version") == "forte.gemma4-lora.checkpoint.v1" and receipt.get("complete") is True,
            "gemma4_resume_checkpoint_incomplete")
    files = receipt.get("files")
    require(isinstance(files, dict) and {"adapter_config.json", "adapter_model.safetensors", "optimizer.pt",
            "trainer_state.json"} <= set(files), "gemma4_resume_checkpoint_files_missing")
    require(set(files) | {"checkpoint_complete.json"} == {p.name for p in path.iterdir() if p.is_file()},
            "gemma4_resume_checkpoint_inventory_mismatch")
    for name, digest in files.items():
        require(name == Path(name).name and file_sha256(path / name) == digest,
                "gemma4_resume_checkpoint_hash_mismatch")
    state = read_json(path / "trainer_state.json")
    require(state.get("schema_version") == SCHEMA and type(state.get("optimizer_steps")) is int
            and state["optimizer_steps"] > 0, "gemma4_resume_state_invalid")
    return state


def train(config: dict, checked: dict, *, smoke: bool = False, resume: Path | None = None) -> dict:
    output = Path(config["output_dir"])
    if smoke:
        output = output.with_name(output.name + "-smoke")
    require(resume is not None or not output.exists(), "gemma4_output_exists_use_explicit_resume")
    output.mkdir(parents=True, exist_ok=True)
    config_hash = canonical_sha256(config)
    previous = None
    if resume:
        require(resume.resolve().is_relative_to(output.resolve()), "gemma4_resume_outside_run")
        previous = validate_checkpoint(resume)
        require(previous.get("config_sha256") == config_hash
                and previous.get("manifest_sha256") == config["expected_manifest_sha256"]
                and previous.get("source_sha256") == checked["source_sha256"]
                and previous.get("weight_manifest_sha256") == checked["weight_receipt"]["weight_manifest_sha256"]
                and previous.get("model_metadata_sha256") == checked["model_metadata_sha256"],
                "gemma4_resume_provenance_mismatch")
    run = {"schema_version": SCHEMA, "state": "loading", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID, "revision": REVISION, "config_sha256": config_hash,
        "manifest_sha256": config["expected_manifest_sha256"], "source_sha256": checked["source_sha256"],
        "weight_manifest_sha256": checked["weight_receipt"]["weight_manifest_sha256"],
        "model_metadata_sha256": checked["model_metadata_sha256"],
        "model_metadata_files": checked["model_metadata_files"],
        "public_data": checked["stats"], "bank_video_labels_used": False, "production_enabled": False,
        "qualification_state": "not_evaluated", "mode": "smoke" if smoke else "train", "optimizer_steps": 0,
        "steps": [], "runtime_versions": PINS, "quantization": "nf4",
        "checkpoint_selection": "predeclared_final_step_or_time_budget_no_bank_scores"}
    run["training_hyperparameters_predeclared"] = ["optimizer_steps", "grad_accum_steps", "learning_rate",
        "warmup_steps", "weight_decay", "max_grad_norm", "lora_rank", "lora_alpha", "lora_dropout",
        "max_length", "max_soft_tokens", "seed"]
    run["training_hyperparameter_count"] = 12
    run["training_parameters_tuned_to_bank_labels"] = 0
    run["pilot"] = {"steps": config["pilot_steps"], "maximum_seconds": config["pilot_max_seconds"],
                    "scope": "optimizer_loop_only_excludes_model_load_and_parameter_digest", "state": "pending"}
    write_json(output / "run.json", run)
    model = optimizer = None
    try:
        require(datetime.now(timezone.utc) < deadline(config), "gemma4_deadline_already_passed")
        emit({"phase": "model_load", "mode": run["mode"]})
        with quiet_libraries():
            model, processor, trainable, targets = load_training_model(config, checked, resume)
        import torch
        collator = Gemma4Collator(processor, Path(config["image_root"]), max_length=config["max_length"],
            max_soft_tokens=config["max_soft_tokens"], image_token_id=checked["model_config"]["image_token_id"])
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
                    "gemma4_resume_frozen_parameter_mismatch")
        rows = [r for r in checked["rows"] if r["split"] == "train"]
        target_steps = 1 if smoke else config["optimizer_steps"]
        order = balanced_indices(rows, target_steps * config["grad_accum_steps"], config["seed"])
        start = time.monotonic()
        previous_seconds = previous.get("cumulative_training_seconds", 0) if previous else 0
        model.train()
        for name, module in model.named_modules():
            if name.endswith(".vision_tower"):
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
                    require(bool(torch.isfinite(loss).item()), "gemma4_nonfinite_loss")
                    losses.append(float(loss.detach().item()))
                    (loss / config["grad_accum_steps"]).backward()
                del loss, batch
            if interrupted:
                optimizer.zero_grad(set_to_none=True)
                break
            gradients = [p.grad for _, p in trainable if p.grad is not None]
            require(bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients)
                    and any(bool(torch.count_nonzero(g)) for g in gradients), "gemma4_invalid_or_zero_gradients")
            norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in trainable], config["max_grad_norm"]).item())
            require(math.isfinite(norm), "gemma4_nonfinite_gradient_norm")
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
            require(all(bool(torch.isfinite(p).all()) for _, p in trainable), "gemma4_nonfinite_adapter")
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
        require(run["optimizer_steps"] > 0, "gemma4_no_completed_optimizer_step")
        optimizer.zero_grad(set_to_none=True)
        run["final_adapter_sha256"] = adapter_digest(model)
        require(run["final_adapter_sha256"] != run["initial_adapter_sha256"], "gemma4_adapter_weights_unchanged")
        emit({"phase": "frozen_parameter_digest_final"})
        run["frozen_after_sha256"] = frozen_digest(model)
        require(run["frozen_after_sha256"] == run["frozen_before_sha256"], "gemma4_frozen_parameters_changed")
        last_checkpoint = output / f"checkpoint-{run['optimizer_steps']:04d}"
        if not last_checkpoint.exists():
            with quiet_libraries():
                last_checkpoint = save_checkpoint(model, optimizer, output, run, config_hash, config["expected_manifest_sha256"])
        run["last_checkpoint"] = last_checkpoint.name
        final = output / "adapter"
        if final.exists():
            require(previous is not None, "gemma4_final_adapter_already_exists")
            archived = output / f"adapter-before-resume-{previous['optimizer_steps']:04d}"
            require(not archived.exists(), "gemma4_resume_adapter_archive_exists")
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
    base = {"source_dataset": "dny3dgvvw8", "source_group": "public-a", "origin": "public",
        "prompt": 'Classify the visible object; answer one JSON object.', "crop_box_xyxy": [0, 0, 8, 8]}
    rows = [dict(base, source_group=f"g-{split}-{name}", split=split, target={"object_class": name})
            for split in ("train", "validation") for name in ("banknote", "id_card")]
    require(validate_rows(rows)["rows"] == 4, "gemma4_synthetic_rows_failed")
    bad = [*rows, dict(rows[0], split="validation")]
    try:
        validate_rows(bad)
        raise AssertionError("split leakage accepted")
    except TrainingGateError as error:
        require(str(error) == "gemma4_source_group_split_leakage", "gemma4_synthetic_wrong_leakage_gate")
    for target in ({"object_class": "banknote", "verdict": "violation"}, {"description": "blocked"}):
        try:
            validate_rows([dict(rows[0], target=target), *rows[1:]])
            raise AssertionError("free-form or verdict target accepted")
        except TrainingGateError:
            pass
    indices = balanced_indices(rows[:2], 20, 0)
    require(Counter(indices) == {0: 10, 1: 10}, "gemma4_synthetic_balance_failed")
    labels = completion_labels([2, 258880, 7, 8, 9, 106], [2, 258880, 7], {258880})
    require(labels == [-100, -100, -100, 8, 9, 106], "gemma4_synthetic_mask_failed")
    class Module:
        weight = True
    class FakeModel:
        def named_modules(self):
            return [(name, Module()) for name in ("model.language_model.layers.0.self_attn.q_proj",
                "model.language_model.layers.0.self_attn.v_proj", "model.language_model.layers.1.self_attn.q_proj",
                "model.vision_tower.layers.0.self_attn.q_proj")]
    targets = inventory_targets(FakeModel(), {"text_config": {"num_hidden_layers": 2,
            "layer_types": ["sliding_attention", "full_attention"], "attention_k_eq_v": True}})
    require(len(targets) == 3 and not any("vision" in x for x in targets), "gemma4_synthetic_targets_failed")
    return {"ok": True, "checks": 7, "pixels_read": 0, "bank_video_labels_read": 0,
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
        require(args.config is not None, "gemma4_config_required")
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
