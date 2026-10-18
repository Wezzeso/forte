"""Fitting validates train/dev bytes while sealed images remain unopened."""
from pathlib import Path

from PIL import Image
import pytest

from training.guards import TrainingGateError, canonical_sha256, file_sha256
from training.specialists.common import write_json
from training.specialists.public_bundle import verify_fit_bundle


def bundle(tmp_path: Path):
    files, records = {}, []
    for index, (role, directory) in enumerate((("train", "train"), ("validation", "valid"), ("sealed_test", "sealed_test"))):
        part = tmp_path / directory
        part.mkdir()
        path = part / f"{index}.jpg"
        if role == "sealed_test":
            path.write_bytes(b"intentionally-not-a-decodable-image-until-authorized")
        else:
            Image.new("RGB", (8, 8), (index * 80, 2, 3)).save(path)
        files[f"{directory}/{path.name}"] = file_sha256(path)
        row = {"sample_id": f"s-{index}", "image_id": 1, "included": True, "split": role,
               "source_group": f"source-{index}", "template_group": f"template-{index}",
               "content_sha256": file_sha256(path), "export_path": f"{directory}/{path.name}",
               "width": 8, "height": 8, "objects": []}
        records.append(row)
        annotation = part / "_annotations.coco.json"
        write_json(annotation, {"categories": [{"id": 1, "name": "single_banknote"}],
                   "images": [{"id": 1, "file_name": path.name, "width": 8, "height": 8}], "annotations": []})
        files[f"{directory}/_annotations.coco.json"] = file_sha256(annotation)
    lock = {"schema_version": "forte.specialists.public-dataset-lock.v1", "state": "frozen", "origin": "public",
            "classes": ["single_banknote"], "class_names": {"0": "single_banknote"},
            "prediction_class_ids": {"single_banknote": 0}, "coco_category_ids": {"single_banknote": 1},
            "files": files, "records": records}
    lock["dataset_sha256"] = canonical_sha256(lock)
    write_json(tmp_path / "dataset-lock.json", lock)
    return lock, file_sha256(tmp_path / "dataset-lock.json")


def test_sealed_image_pixels_are_not_opened_during_fit_verification(tmp_path, monkeypatch):
    lock, expected = bundle(tmp_path)
    original = Image.open
    opened = []
    def inspect(path, *args, **kwargs):
        opened.append(Path(path).parent.name)
        assert Path(path).parent.name != "sealed_test"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Image, "open", inspect)
    assert verify_fit_bundle(tmp_path, expected) == lock
    assert set(opened) == {"train", "valid"}


def test_training_image_modification_is_rejected(tmp_path):
    _, expected = bundle(tmp_path)
    (tmp_path / "train/0.jpg").write_bytes(b"changed")
    with pytest.raises(TrainingGateError, match="file_changed"):
        verify_fit_bundle(tmp_path, expected)


def test_own_data_schema_cannot_enter_public_fit_bundle(tmp_path):
    lock, _ = bundle(tmp_path)
    lock["schema_version"] = "forte.specialists.dataset-lock.v1"
    write_json(tmp_path / "dataset-lock.json", lock)
    with pytest.raises(TrainingGateError, match="requires_public_data"):
        verify_fit_bundle(tmp_path, file_sha256(tmp_path / "dataset-lock.json"))
