"""Synthetic metadata/tensors only: no public or organizer payload is opened."""
from __future__ import annotations

import builtins
import copy
from enum import Enum
from types import SimpleNamespace

import pytest

from training.guards import TrainingGateError
from training.specialists.public_augmentation import (IMAGENET_MEAN, IMAGENET_STD, SCHEMA,
                                                     SmallObjectZoomOut, balanced_public_datamodule,
                                                     validate_policy)


@pytest.fixture
def policy():
    return {"schema_version": SCHEMA, "seed": 17, "class_balanced_sampling": True,
            "small_object_zoomout": {"probability": 0.8, "reference_resolution": 512,
                                     "longest_box_min_px": 12, "longest_box_max_px": 96}}


@pytest.fixture
def lock():
    rows = []
    for image_id, group in enumerate(["single_banknote"] * 6 + ["id_card"] * 2 + ["coin_background"], 1):
        rows.append({"image_id": image_id, "sample_id": f"synthetic-{image_id}",
                     "content_sha256": f"{image_id:064x}", "split": "train", "included": True,
                     "duplicate_of": None, "export_path": f"train/{image_id:064x}.jpg",
                     "width": 320, "height": 192, "source_group": f"source-{image_id % 2}",
                     "template_group": "template-1", "source_record": {"kind": "coin" if group == "coin_background" else "banknote"},
                     "objects": [] if group == "coin_background" else [{"class_name": group, "bbox": [80, 80, 80, 48]}]})
    # Held-out and quarantined records must have no effect on train exposure.
    for split in ("validation", "sealed_test", "quarantine"):
        rows.append({"image_id": 1, "sample_id": split, "split": split, "included": split != "quarantine",
                     "objects": [{"class_name": "id_card"}], "content_sha256": "f" * 64})
    return {"schema_version": "forte.specialists.public-dataset-lock.v1", "origin": "public",
            "dataset_kind": "combined_public", "classes": ["single_banknote", "id_card"],
            "class_names": {"0": "single_banknote", "1": "id_card"}, "records": rows}


def test_policy_preflight_is_stdlib_only_and_counts_original_support(monkeypatch, lock, policy):
    original_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("metadata preflight must not import torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    metadata = validate_policy(policy, lock, seed=17)
    assert metadata["original_unique_training_images"] == 9
    assert metadata["original_unique_training_images_by_group"] == {"single_banknote": 6, "id_card": 2, "coin_background": 1}
    assert metadata["expected_sampling_mass_by_group"] == dict.fromkeys(("single_banknote", "id_card", "coin_background"), 1 / 3)
    assert metadata["synthetic_rows_added"] == 0 and metadata["heldout_modified"] is False
    assert "independence is not established" in metadata["support_semantics"]


@pytest.mark.parametrize("mutation,code", [
    (lambda p, d: p.update(seed=18), "seed_mismatch"),
    (lambda p, d: p["small_object_zoomout"].update(probability=0.9), "zoomout_recipe_mismatch"),
    (lambda p, d: p.update(unregistered_option=True), "policy_schema"),
    (lambda p, d: d.update(origin="organizer"), "public_lock_required"),
    (lambda p, d: d["records"][0].update(content_sha256=d["records"][1]["content_sha256"]), "duplicate_training_image"),
    (lambda p, d: d.update(classes=["single_banknote"]), "group_coverage_missing"),
])
def test_unregistered_or_misrepresented_policy_fails_closed(lock, policy, mutation, code):
    mutation(policy, lock)
    with pytest.raises(TrainingGateError, match=code):
        validate_policy(policy, lock, seed=17)


def test_combined_scope_requires_both_positive_classes_and_known_coin_negatives(lock, policy):
    lock["records"] = [row for row in lock["records"] if not row.get("source_record", {}).get("kind") == "coin"]
    with pytest.raises(TrainingGateError, match="group_coverage_missing"):
        validate_policy(policy, lock)


def test_standalone_banknotes_explicitly_use_two_groups(lock, policy):
    lock["dataset_kind"], lock["classes"], lock["class_names"] = "kz_bd_v1", ["single_banknote"], {"0": "single_banknote"}
    lock["records"] = [row for row in lock["records"] if not row.get("objects") or row["objects"][0]["class_name"] != "id_card"]
    metadata = validate_policy(policy, lock)
    assert metadata["expected_sampling_mass_by_group"] == {"single_banknote": .5, "coin_background": .5}


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def sample(torch, *, empty=False):
    # A black image with a white marker, far from each border, permits an
    # independent measurement of the resized image's footprint on the canvas.
    image = torch.zeros((3, 192, 320))
    image[:, 90:138, 80:160] = 1
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = (image - mean) / std
    boxes = torch.empty((0, 4)) if empty else torch.tensor([[.375, .59375, .25, .25]])
    return image, {"boxes": boxes, "labels": torch.empty((0,), dtype=torch.long) if empty else torch.tensor([0]),
                   "area": torch.empty((0,)) if empty else torch.tensor([80. * 48.]),
                   "image_id": torch.tensor([3]), "size": torch.tensor([192, 320]),
                   "orig_size": torch.tensor([768, 1280]), "iscrowd": torch.empty((0,), dtype=torch.long) if empty else torch.tensor([0])}


def test_zoomout_aligns_asymmetric_box_with_pixels_and_area_without_input_mutation(torch):
    image, target = sample(torch)
    original_image, original_target = image.clone(), copy.deepcopy(target)
    torch.manual_seed(5)
    out, updated = SmallObjectZoomOut(probability=1, longest_box_min_px=48, longest_box_max_px=48)(image, target)
    rgb = out * torch.tensor(IMAGENET_STD).view(3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    y, x = (rgb[0] < .05).nonzero(as_tuple=True)
    top, left, height, width = int(y.min()), int(x.min()), int(y.max() - y.min() + 1), int(x.max() - x.min() + 1)
    assert (height, width) == (72, 120)
    expected = torch.tensor([[(120 * width / 320 + left) / 320,
                              (114 * height / 192 + top) / 192,
                              (80 * width / 320) / 320, (48 * height / 192) / 192]])
    torch.testing.assert_close(updated["boxes"], expected)
    assert updated["area"].item() == pytest.approx(80 * 48 * (120 / 320) * (72 / 192))
    assert float((updated["boxes"][:, 2:] * torch.tensor([320, 192])).max()) * 512 / 320 == pytest.approx(48)
    assert out.shape == image.shape and torch.equal(image, original_image)
    for name in target:
        assert torch.equal(target[name], original_target[name])
    assert torch.equal(updated["orig_size"], target["orig_size"])
    assert torch.equal(updated["size"], target["size"])
    assert torch.equal(updated["image_id"], target["image_id"])
    assert -1e-6 <= rgb.min() and rgb.max() <= 1 + 1e-6


def test_zoomout_replays_with_torch_seed_and_advances_between_draws(torch):
    image, target = sample(torch)
    transform = SmallObjectZoomOut(probability=1)
    torch.manual_seed(71)
    first = transform(image, target)
    second = transform(image, target)
    torch.manual_seed(71)
    replay = transform(image, target)
    assert torch.equal(first[0], replay[0]) and torch.equal(first[1]["boxes"], replay[1]["boxes"])
    assert not torch.equal(first[1]["boxes"], second[1]["boxes"])


def test_small_source_objects_are_never_enlarged_and_empty_coins_stay_empty(torch):
    image, target = sample(torch)
    target["boxes"] = torch.tensor([[.5, .5, 1 / 320, 1 / 192]])
    out, result = SmallObjectZoomOut(probability=1)(image, target)
    assert out is image and result is target
    empty_image, empty = sample(torch, empty=True)
    out, result = SmallObjectZoomOut(probability=1)(empty_image, empty)
    assert out is empty_image and result is empty and result["boxes"].shape == (0, 4)


@pytest.mark.parametrize("field", ["masks", "keypoints"])
def test_unsupported_dense_targets_are_rejected_even_with_probability_zero(torch, field):
    image, target = sample(torch)
    target[field] = torch.empty((0,))
    with pytest.raises(TrainingGateError, match="masks_or_keypoints_unsupported"):
        SmallObjectZoomOut(probability=0)(image, target)


def test_absolute_or_invalid_box_coordinates_cannot_be_silently_transformed(torch):
    image, target = sample(torch)
    target["boxes"] = torch.tensor([[80., 90., 160., 138.]])
    with pytest.raises(TrainingGateError, match="normalized_cxcywh_required"):
        SmallObjectZoomOut(probability=1)(image, target)


def fake_base(torch, lock):
    normalize_type = type("Normalize", (), {"__module__": "rfdetr.datasets.transforms"})

    class Dataset:
        def __init__(self):
            self.ids = [row["image_id"] for row in reversed(lock["records"]) if row["split"] == "train"]
            self.cat2label = {1: 0, 2: 1}
            rows = {row["image_id"]: row for row in lock["records"] if row["split"] == "train"}
            self.coco = SimpleNamespace(imgs={i: {"file_name": rows[i]["export_path"].split("/")[-1]} for i in self.ids})
            normalize = normalize_type()
            normalize._normalize = SimpleNamespace(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            self._transforms = SimpleNamespace(transforms=[normalize])
            self.reads = 0

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, index):
            self.reads += 1
            image_id = self.ids[index]
            row = next(row for row in lock["records"] if row["split"] == "train" and row["image_id"] == image_id)
            image, target = sample(torch, empty=not row["objects"])
            target["image_id"] = torch.tensor([image_id])
            if row["objects"]:
                target["labels"] = torch.tensor([lock["classes"].index(row["objects"][0]["class_name"])])
            return image, target

    class DataModule:
        def __init__(self, model_config, train_config):
            self.train_config = train_config
            self._dataset_train = None
            self._dataset_val = object()
            self._dataset_test = object()
            self._collate_fn_train = lambda batch: batch
            self.trainer = None

        def setup(self, stage):
            if stage == "fit" and self._dataset_train is None:
                self._dataset_train = Dataset()

        def _effective_grad_accum_steps(self):
            return self.train_config.grad_accum_steps

        def train_dataloader(self):
            return torch.utils.data.DataLoader(self._dataset_train, batch_size=1, num_workers=0,
                                               collate_fn=self._collate_fn_train, pin_memory=False)

        def val_dataloader(self):
            return self._dataset_val

        def test_dataloader(self):
            return self._dataset_test

    return DataModule


def configs():
    # RF-DETR converts False into this string enum in its real TrainConfig.
    class MultiScale(str, Enum):
        OFF = "off"
    return SimpleNamespace(segmentation_head=False, use_grouppose_keypoints=False), SimpleNamespace(
        seed=17, batch_size=1, num_workers=0, grad_accum_steps=8,
        augmentation_backend="torchvision", scale_jitter=False, multi_scale=MultiScale.OFF)


def test_balanced_loader_uses_native_id_order_collate_and_unchanged_eval(torch, lock, policy):
    snapshot = copy.deepcopy(lock)
    model, train = configs()
    base = fake_base(torch, lock)
    data, metadata = balanced_public_datamodule(base, model, train, lock, policy)
    assert data._dataset_train is None  # Baseline setup needs no training split.
    val, test = data.val_dataloader(), data.test_dataloader()
    assert type(data).setup is base.setup
    assert type(data).val_dataloader is base.val_dataloader and type(data).test_dataloader is base.test_dataloader
    data.setup("fit")
    original_train = data._dataset_train
    loader = data.train_dataloader()
    assert original_train is data._dataset_train and original_train.reads == 0
    assert val is data.val_dataloader() and test is data.test_dataloader()
    assert loader.collate_fn is data._collate_fn_train and loader.num_workers == 0 and loader.batch_size == 1
    assert isinstance(loader.sampler, torch.utils.data.WeightedRandomSampler)
    assert len(loader.sampler) == 40 and len(loader.sampler) % train.grad_accum_steps == 0
    # Native order is coin, ID, ID, six notes; each bucket has equal total mass.
    weights = loader.sampler.weights
    total = float(weights.sum())
    assert [float(weights[0] / total), float(weights[1:3].sum() / total), float(weights[3:].sum() / total)] == pytest.approx([1 / 3] * 3)
    assert metadata["original_unique_training_images"] == 9 and metadata["epoch_sampling_draws"] == 40
    assert metadata["synthetic_rows_added"] == 0 and lock == snapshot
    next(iter(loader))
    assert original_train.reads == 1


def test_sampler_seed_replays_and_does_not_restart_for_rebuilt_loader(torch, lock, policy):
    model, train = configs()
    base = fake_base(torch, lock)
    first, _ = balanced_public_datamodule(base, model, train, lock, policy)
    replay, _ = balanced_public_datamodule(base, model, train, lock, policy)
    first.setup("fit")
    replay.setup("fit")
    initial = list(first.train_dataloader().sampler)
    assert initial == list(replay.train_dataloader().sampler)
    assert initial != list(first.train_dataloader().sampler)


def test_native_inventory_and_normalization_must_match_lock(torch, lock, policy):
    model, train = configs()
    base = fake_base(torch, lock)
    data, _ = balanced_public_datamodule(base, model, train, lock, policy)
    data.setup("fit")
    data._dataset_train.ids.pop()
    with pytest.raises(TrainingGateError, match="native_inventory_mismatch"):
        data.train_dataloader()
    data, _ = balanced_public_datamodule(base, model, train, lock, policy)
    data.setup("fit")
    data._dataset_train._transforms.transforms[-1]._normalize.mean = (0., 0., 0.)
    with pytest.raises(TrainingGateError, match="native_normalization_mismatch"):
        data.train_dataloader()


@pytest.mark.parametrize("field,value,code", [("batch_size", 2, "bounded_loader_required"),
                                              ("num_workers", 1, "bounded_loader_required"),
                                              ("seed", 42, "seed_mismatch"),
                                              ("seed", None, "seed_mismatch"),
                                              ("augmentation_backend", "kornia", "native_detection_pipeline_required"),
                                              ("multi_scale", "per-batch", "native_detection_pipeline_required")])
def test_loader_recipe_cannot_change_silently(lock, policy, field, value, code):
    model, train = configs()
    setattr(train, field, value)
    with pytest.raises(TrainingGateError, match=code):
        balanced_public_datamodule(object, model, train, lock, policy)
