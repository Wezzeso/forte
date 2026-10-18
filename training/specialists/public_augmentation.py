"""Predeclared public-only sampling and synthetic training zoom-out.

The pinned RF-DETR 1.11.2 contract was checked against its official tag:
https://github.com/roboflow/rf-detr/blob/1.11.2/src/rfdetr/training/module_data.py
https://github.com/roboflow/rf-detr/blob/1.11.2/src/rfdetr/datasets/transforms.py

Native torchvision transforms return ImageNet-normalized CHW float images and
normalized cxcywh boxes. Only the training loader is replaced. Native setup,
COCO metadata, validation/test loaders, and collate/target packing stay native.
Torch is imported lazily so policy preflight does not need a training runtime.
"""
from __future__ import annotations

from collections import Counter
import copy
from pathlib import PurePosixPath

from .common import canonical_sha256, number, require

SCHEMA = "forte.specialists.public-augmentation.v1"
GROUPS = ("single_banknote", "id_card", "coin_background")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _training_rows(lock: dict) -> list[dict]:
    require(lock.get("schema_version") == "forte.specialists.public-dataset-lock.v1"
            and lock.get("origin") == "public" and isinstance(lock.get("records"), list),
            "public_augmentation_public_lock_required")
    rows = [row for row in lock["records"] if row.get("split") == "train" and row.get("included") is True]
    require(bool(rows), "public_augmentation_training_rows_missing")
    require(all(type(row.get("image_id")) is int and row["image_id"] > 0
                and row.get("duplicate_of") is None and isinstance(row.get("sample_id"), str)
                and bool(row["sample_id"]) for row in rows), "public_augmentation_training_identity_invalid")
    require(len({row["image_id"] for row in rows}) == len(rows)
            and len({row["sample_id"] for row in rows}) == len(rows)
            and len({row.get("content_sha256") for row in rows}) == len(rows),
            "public_augmentation_duplicate_training_image")
    return rows


def _group(row: dict) -> str:
    objects = row.get("objects")
    require(isinstance(objects, list) and all(isinstance(obj, dict) for obj in objects),
            "public_augmentation_objects_invalid")
    names = {obj.get("class_name") for obj in objects}
    if names:
        require(len(names) == 1 and names <= set(GROUPS[:2]), "public_augmentation_mixed_or_unknown_classes")
        return next(iter(names))
    source = row.get("source_record", {})
    require(row.get("kind", source.get("kind")) == "coin", "public_augmentation_unverified_background")
    return "coin_background"


def validate_policy(policy: dict, lock: dict, *, seed: int | None = None) -> dict:
    """Validate metadata only; return support counts, never inflated draw counts.

    Combined public training requires banknote positives, ID positives and coin
    negatives. A standalone banknote dataset may use its explicit two-group
    policy. Existing public-data verification must precede this helper.
    """
    require(isinstance(policy, dict) and set(policy) == {
        "schema_version", "seed", "class_balanced_sampling", "small_object_zoomout"}
            and policy.get("schema_version") == SCHEMA, "public_augmentation_policy_schema")
    require(type(policy.get("seed")) is int and policy["seed"] >= 0
            and (seed is None or policy["seed"] == seed), "public_augmentation_seed_mismatch")
    require(policy.get("class_balanced_sampling") is True, "public_augmentation_balancing_required")
    zoom = policy.get("small_object_zoomout")
    require(isinstance(zoom, dict) and set(zoom) == {
        "probability", "reference_resolution", "longest_box_min_px", "longest_box_max_px"}
            and number(zoom.get("probability")) and zoom["probability"] == 0.8
            and type(zoom.get("reference_resolution")) is int and zoom["reference_resolution"] == 512
            and type(zoom.get("longest_box_min_px")) is int and type(zoom.get("longest_box_max_px")) is int
            and zoom.get("longest_box_min_px") == 12 and zoom.get("longest_box_max_px") == 96,
            "public_augmentation_zoomout_recipe_mismatch")
    rows = _training_rows(lock)
    counts = Counter(_group(row) for row in rows)
    expected = {"single_banknote", "id_card", "coin_background"}
    if lock.get("dataset_kind") != "combined_public":
        require(lock.get("classes") == ["single_banknote"], "public_augmentation_standalone_scope_invalid")
        expected = {"single_banknote", "coin_background"}
    require(set(counts) == expected and set(lock.get("classes", [])) == expected - {"coin_background"},
            "public_augmentation_group_coverage_missing")
    require(lock.get("class_names") == {str(i): name for i, name in enumerate(lock["classes"])},
            "public_augmentation_class_mapping_mismatch")
    return {"schema_version": SCHEMA, "scope": "included_public_train_only",
            "policy": copy.deepcopy(policy), "policy_sha256": canonical_sha256(policy),
            "original_unique_training_images": len(rows),
            "original_unique_training_images_by_group": dict(sorted(counts.items())),
            "original_source_groups": len({row["source_group"] for row in rows if row.get("source_group")}),
            "original_template_groups": len({row["template_group"] for row in rows if row.get("template_group")}),
            "expected_sampling_mass_by_group": {name: 1.0 / len(counts) for name in sorted(counts)},
            "sampler": "seeded_WeightedRandomSampler_with_replacement",
            "support_semantics": "Retained unique images; replacement draws and synthetic variants add no independent support. Physical note/document and scene independence is not established by image counts.",
            "synthetic_rows_added": 0, "heldout_modified": False,
            "zoomout": {"application": "on_the_fly_after_native_train_normalization_only",
                        "requested_longest_box_distribution": "uniform_12_to_96_px_at_512_long_side_reference",
                        "native_smaller_objects": "never_upscaled",
                        "raster_size_rounding": "nearest_integer",
                        "background": "plain_neutral_RGB_uniform_intensity_0.2_to_0.8",
                        "empty_coin_images": "unchanged",
                        "image_shape": "unchanged",
                        "rng": "torch_worker_rng_seeded_by_training_seed",
                        "evidence_scope": "synthetic_training_only_not_evaluation_or_evidence"}}


class SmallObjectZoomOut:
    """Shrink an entire normalized image onto a plain same-size canvas.

    Geometry follows actual rounded raster dimensions; every box survives, and
    no crop, cutout, new object, inferred pixels or evaluation row is created.
    """

    def __init__(self, *, probability: float = 0.8, reference_resolution: int = 512,
                 longest_box_min_px: float = 12, longest_box_max_px: float = 96):
        require(number(probability, 0, 1) and type(reference_resolution) is int
                and reference_resolution > 0 and number(longest_box_min_px, 1)
                and number(longest_box_max_px, longest_box_min_px), "public_augmentation_transform_parameters")
        self.probability = probability
        self.reference_resolution = reference_resolution
        self.minimum = longest_box_min_px
        self.maximum = longest_box_max_px

    def __call__(self, image, target):
        import torch
        import torch.nn.functional as functional

        require(torch.is_tensor(image) and image.ndim == 3 and image.shape[0] == 3
                and image.is_floating_point() and image.device.type == "cpu"
                and min(image.shape[-2:]) > 0 and bool(torch.isfinite(image).all()),
                "public_augmentation_native_image_contract")
        require(isinstance(target, dict) and not {"masks", "keypoints"} & target.keys(),
                "public_augmentation_masks_or_keypoints_unsupported")
        boxes, labels = target.get("boxes"), target.get("labels")
        require(torch.is_tensor(boxes) and boxes.ndim == 2 and boxes.shape[1] == 4
                and boxes.is_floating_point() and boxes.device.type == "cpu"
                and bool(torch.isfinite(boxes).all())
                and torch.is_tensor(labels) and labels.ndim == 1 and labels.shape[0] == boxes.shape[0],
                "public_augmentation_native_target_contract")
        require(bool((boxes[:, 2:] > 0).all()) and bool((boxes >= 0).all())
                and bool((boxes[:, :2] - boxes[:, 2:] / 2 >= -1e-6).all())
                and bool((boxes[:, :2] + boxes[:, 2:] / 2 <= 1 + 1e-6).all()),
                "public_augmentation_normalized_cxcywh_required")
        area = target.get("area")
        require(torch.is_tensor(area) and area.ndim == 1 and len(area) == len(boxes)
                and bool(torch.isfinite(area).all()) and bool((area >= 0).all()),
                "public_augmentation_area_contract")
        if not len(boxes) or torch.rand(()).item() >= self.probability:
            return image, target
        height, width = image.shape[-2:]
        longest = float((boxes[:, 2:] * boxes.new_tensor([width, height])).max())
        desired = (self.minimum + torch.rand(()).item() * (self.maximum - self.minimum))
        desired *= max(height, width) / self.reference_resolution
        factor = min(1.0, desired / longest)
        new_height, new_width = max(1, round(height * factor)), max(1, round(width * factor))
        if (new_height, new_width) == (height, width):
            return image, target
        offset_y = int(torch.randint(height - new_height + 1, ()).item())
        offset_x = int(torch.randint(width - new_width + 1, ()).item())
        # Linear interpolation commutes with channel normalization. New plain
        # pixels are normalized explicitly with the native ImageNet statistics.
        grey = 0.2 + torch.rand(()).item() * 0.6
        mean = image.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = image.new_tensor(IMAGENET_STD).view(3, 1, 1)
        canvas = ((image.new_full((3, 1, 1), grey) - mean) / std).expand_as(image).clone()
        resized = functional.interpolate(image.unsqueeze(0), size=(new_height, new_width),
                                         mode="bilinear", align_corners=False, antialias=True)[0]
        canvas[:, offset_y:offset_y + new_height, offset_x:offset_x + new_width] = resized
        result = target.copy()
        result["boxes"] = boxes * boxes.new_tensor([new_width / width, new_height / height,
                                                    new_width / width, new_height / height])
        result["boxes"][:, :2] += boxes.new_tensor([offset_x / width, offset_y / height])
        result["area"] = area * (new_width / width) * (new_height / height)
        return canvas, result


class _TrainingZoomOutDataset:
    def __init__(self, dataset, rows: list[dict], classes: list[str], transform: SmallObjectZoomOut):
        self.dataset = dataset
        self.rows = rows
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        import torch
        image, target = self.dataset[index]
        row = self.rows[index]
        image_id = target.get("image_id")
        require(torch.is_tensor(image_id) and image_id.numel() == 1 and int(image_id.item()) == row["image_id"],
                "public_augmentation_sample_identity_mismatch")
        expected_label = self.classes.index(_group(row)) if row["objects"] else None
        labels = target.get("labels")
        require(torch.is_tensor(labels) and (bool((labels == expected_label).all()) if expected_label is not None
                                             else labels.numel() == 0), "public_augmentation_sample_label_mismatch")
        return self.transform(image, target)


def _native_rows(dataset, lock: dict) -> list[dict]:
    """Resolve weights by native COCO ids without decoding or indexing images."""
    rows = _training_rows(lock)
    by_id = {row["image_id"]: row for row in rows}
    ids = getattr(dataset, "ids", None)
    require(isinstance(ids, list) and len(ids) == len(rows) and len(set(ids)) == len(ids)
            and set(ids) == set(by_id) and len(dataset) == len(rows), "public_augmentation_native_inventory_mismatch")
    require(getattr(dataset, "cat2label", None) == {i + 1: i for i in range(len(lock["classes"]))},
            "public_augmentation_native_class_mapping_mismatch")
    coco = getattr(dataset, "coco", None)
    images = getattr(coco, "imgs", {})
    require(all(images.get(image_id, {}).get("file_name") == PurePosixPath(by_id[image_id]["export_path"]).name
                for image_id in ids), "public_augmentation_native_image_name_mismatch")
    transforms = getattr(getattr(dataset, "_transforms", None), "transforms", [])
    normalize = transforms[-1] if transforms else None
    native_normalize = getattr(normalize, "_normalize", None)
    require(sum(type(item).__name__ == "Normalize" for item in transforms) == 1
            and type(normalize).__name__ == "Normalize"
            and type(normalize).__module__ == "rfdetr.datasets.transforms"
            and tuple(getattr(native_normalize, "mean", ())) == IMAGENET_MEAN
            and tuple(getattr(native_normalize, "std", ())) == IMAGENET_STD,
            "public_augmentation_native_normalization_mismatch")
    return [by_id[image_id] for image_id in ids]


def balanced_public_datamodule(base_class, model_config, train_config, lock: dict, policy: dict):
    """Return ``(native_subclass_instance, metadata)`` without constructing splits.

    Instantiate before the baseline; inherited ``setup('validate')`` and native
    evaluation loaders are unchanged. Sampling/augmentation start only when
    training requests ``train_dataloader()``. This bounded route is single GPU,
    microbatch one and zero workers, matching the public training runner.
    """
    require(type(train_config.seed) is int and train_config.seed >= 0, "public_augmentation_seed_mismatch")
    metadata = validate_policy(policy, lock, seed=train_config.seed)
    require(train_config.batch_size == 1 and train_config.num_workers == 0,
            "public_augmentation_bounded_loader_required")
    multi_scale = getattr(train_config, "multi_scale", None)
    # The native config stores MultiScale.OFF even for a False input.
    multi_scale = getattr(multi_scale, "value", multi_scale)
    require(getattr(train_config, "augmentation_backend", None) in {"torchvision", "tv"}
            and getattr(train_config, "scale_jitter", None) is False
            and (multi_scale is False or multi_scale == "off")
            and not getattr(model_config, "segmentation_head", False)
            and not getattr(model_config, "use_grouppose_keypoints", False),
            "public_augmentation_native_detection_pipeline_required")
    lock_snapshot, policy_snapshot = copy.deepcopy(lock), copy.deepcopy(policy)

    class BalancedPublicDataModule(base_class):
        def train_dataloader(self):
            import torch
            from torch.utils.data import DataLoader, WeightedRandomSampler

            trainer = self.trainer
            require(getattr(trainer, "world_size", 1) == 1, "public_augmentation_single_process_required")
            native = super().train_dataloader()
            dataset = self._dataset_train
            rows = _native_rows(dataset, lock_snapshot)
            require(native.batch_size == 1 and native.num_workers == 0
                    and native.collate_fn is self._collate_fn_train,
                    "public_augmentation_native_loader_contract")
            counts = metadata["original_unique_training_images_by_group"]
            weights = [1.0 / counts[_group(row)] for row in rows]
            accumulation = self._effective_grad_accum_steps()
            require(type(accumulation) is int and accumulation > 0, "public_augmentation_accumulation_invalid")
            minimum_draws = max(len(rows), accumulation * 5)
            epoch_draws = ((minimum_draws + accumulation - 1) // accumulation) * accumulation
            if not hasattr(self, "_public_sampler_generator"):
                self._public_sampler_generator = torch.Generator().manual_seed(policy_snapshot["seed"])
                self._public_loader_generator = torch.Generator().manual_seed(policy_snapshot["seed"])
            sampler = WeightedRandomSampler(weights, epoch_draws, replacement=True,
                                             generator=self._public_sampler_generator)
            wrapped = _TrainingZoomOutDataset(dataset, rows, lock_snapshot["classes"],
                                             SmallObjectZoomOut(**policy_snapshot["small_object_zoomout"]))
            metadata["epoch_sampling_draws"] = epoch_draws
            metadata["draw_count_semantics"] = "Replacement optimizer exposure; never evaluation support."
            return DataLoader(wrapped, batch_size=1, sampler=sampler, drop_last=True,
                              collate_fn=native.collate_fn, num_workers=0, pin_memory=native.pin_memory,
                              persistent_workers=native.persistent_workers, prefetch_factor=native.prefetch_factor,
                              worker_init_fn=native.worker_init_fn, generator=self._public_loader_generator)

    return BalancedPublicDataModule(model_config, train_config), metadata
