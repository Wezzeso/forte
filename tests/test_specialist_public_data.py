"""Public importer contracts, exercised only with generated images and metadata."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import zipfile

from PIL import Image
import pytest

from training.guards import TrainingGateError, canonical_sha256, file_sha256
from training.specialists.common import write_json
from training.specialists.public_data import (
    dhash256, prepare_public_dataset, verify_public_dataset,
)


BANKNOTES = (500, 1000, 2000, 5000, 10000, 20000)
COINS = (1, 2, 5, 10, 20, 50, 100, 200)
DENOMINATIONS = BANKNOTES + COINS
TRAIN_DENOMINATIONS = (500, 1000, 2000, 5000, 1, 2, 5, 10)
VALIDATION_DENOMINATIONS = (10000, 20, 50)
TEST_DENOMINATIONS = (20000, 100, 200)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _repack(fixture: dict) -> None:
    """Keep archive binding valid when a test intentionally changes annotations."""
    archive = fixture["archive"]
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as target:
        for source in sorted(fixture["layout"].rglob("*")):
            if source.is_file():
                target.write(source, source.relative_to(fixture["extracted"]).as_posix())
    fixture["config"]["source_archive"].update(
        sha256=file_sha256(archive), size_bytes=archive.stat().st_size,
    )
    write_json(fixture["config_path"], fixture["config"])


@pytest.fixture
def public_source(tmp_path):
    approved = tmp_path / "approved"
    extracted = approved / "extracted"
    layout = extracted / "Kazakhstan_Banknotes_Dataset"
    generator = random.Random(20261018)
    classes = {}
    paths = {}
    for denomination in DENOMINATIONS:
        class_id = 1 if denomination == 100 else 0
        classes[str(denomination)] = {
            "kind": "banknote" if denomination in BANKNOTES else "coin",
            "source_class_ids": [class_id],
        }
        for upstream, count in (("training", 4), ("validation", 1)):
            folder = layout / upstream / f"{denomination}_tenge"
            folder.mkdir(parents=True)
            for index in range(count):
                image_path = folder / f"{denomination}_{index}.jpg"
                image = Image.frombytes("RGB", (64, 48), generator.randbytes(64 * 48 * 3))
                image.save(image_path, quality=95)
                image_path.with_suffix(".txt").write_text(
                    f"{class_id} 0.5 0.5 0.5 0.5\n", encoding="utf-8",
                )
                paths[upstream, denomination, index] = image_path
    license_evidence = approved / "license.txt"
    license_evidence.write_text("Synthetic test evidence: CC BY 4.0", encoding="utf-8")
    publication_evidence = approved / "annotation-method.txt"
    publication_evidence.write_text("Synthetic test evidence: manually annotated with LabelImg", encoding="utf-8")
    inventory = approved / "organizer-inventory.json"
    write_json(inventory, [{"content_sha256": "a" * 64}])
    archive = approved / "synthetic-source.zip"
    config = {
        "schema_version": "forte.specialists.public-data.v1", "frozen": True,
        "approved_root": str(approved), "extracted_root": str(extracted),
        "source_archive": {
            "path": str(archive), "sha256": "0" * 64, "size_bytes": 0,
            "url": "https://data.mendeley.com/datasets/dny3dgvvw8/1",
            "dataset_id": "dny3dgvvw8", "version": 1,
        },
        "license": {
            "spdx": "CC-BY-4.0", "url": "https://creativecommons.org/licenses/by/4.0/",
            "attribution": "Synthetic test fixture, not downloaded public data",
            "evidence": {"path": str(license_evidence), "sha256": file_sha256(license_evidence)},
        },
        "annotation_publication": {
            "url": "https://data.mendeley.com/datasets/dny3dgvvw8/1",
            "method": "manual_labelimg",
            "evidence": {"path": str(publication_evidence), "sha256": file_sha256(publication_evidence)},
        },
        "organizer_inventory": {"path": str(inventory), "sha256": file_sha256(inventory)},
        "denomination_classes": classes,
        "split_policy": {
            "mode": "denomination_holdout", "seed": 19, "near_duplicate_hamming": 4,
            "declared_before_import": True, "policy_reference": "20261018/public-group-policy-v1",
            "train_denominations": list(map(str, TRAIN_DENOMINATIONS)),
            "validation_denominations": list(map(str, VALIDATION_DENOMINATIONS)),
            "sealed_test_denominations": list(map(str, TEST_DENOMINATIONS)),
        },
    }
    result = {
        "config": config, "config_path": approved / "import.json", "archive": archive,
        "extracted": extracted, "layout": layout, "paths": paths,
        "output": tmp_path / "dataset", "base": tmp_path,
    }
    _repack(result)
    return result


def _prepare(fixture: dict):
    summary = prepare_public_dataset(fixture["config_path"], fixture["output"])
    return summary, verify_public_dataset(fixture["output"], summary["dataset_lock_sha256"])


def test_public_export_is_deterministic_with_physically_sealed_denomination_test(public_source):
    summary, lock = _prepare(public_source)
    second = prepare_public_dataset(public_source["config_path"], public_source["base"] / "second")
    assert second["dataset_sha256"] == summary["dataset_sha256"] == lock["dataset_sha256"]
    assert len(lock["records"]) == 70
    assert {row["upstream_split"] for row in lock["records"]} == {"training", "validation"}
    assert {row["denomination"] for row in lock["records"]} == set(map(str, DENOMINATIONS))
    sealed = [row for row in lock["records"] if row["denomination"] in set(map(str, TEST_DENOMINATIONS))]
    assert len(sealed) == 15
    assert all(row["split"] == "sealed_test" and row["export_path"].startswith("sealed_test/") for row in sealed)
    assert {row["split"] for row in lock["records"]} == {"train", "validation", "sealed_test", "quarantine"}
    for role, denominations, count in (("train", TRAIN_DENOMINATIONS, 32),
                                       ("validation", VALIDATION_DENOMINATIONS, 15),
                                       ("sealed_test", TEST_DENOMINATIONS, 15)):
        rows = [row for row in lock["records"] if row["split"] == role]
        assert len(rows) == count
        assert {row["denomination"] for row in rows} == set(map(str, denominations))
    quarantine = [row for row in lock["records"] if row["split"] == "quarantine"]
    assert len(quarantine) == 8 and all(row["upstream_split"] == "validation" for row in quarantine)
    assert all(row["denomination"] in set(map(str, TRAIN_DENOMINATIONS)) and not row["included"] for row in quarantine)
    assert not ({row["template_group"] for row in lock["records"] if row["split"] == "train"}
                & {row["template_group"] for row in lock["records"] if row["split"] in {"validation", "sealed_test"}})
    test = _read(public_source["output"] / "test/_annotations.coco.json")
    assert test["images"] == [] and test["annotations"] == []
    assert not list((public_source["output"] / "test").glob("*.jpg"))
    sealed_hashes = {row["content_sha256"] for row in sealed}
    exported_train = list((public_source["output"] / "train").glob("*.jpg"))
    exported_valid = list((public_source["output"] / "valid").glob("*.jpg"))
    assert exported_train and exported_valid
    assert not sealed_hashes & {file_sha256(path) for path in exported_train + exported_valid}
    assert all(len(row["dhash256"]) == 64 for row in lock["records"])
    assert all(row["source_group"].startswith("public:") and row["scene_group"] is None for row in lock["records"])


def test_public_banknotes_have_boxes_and_coins_remain_hard_negatives(public_source):
    _, lock = _prepare(public_source)
    for row in lock["records"]:
        if row["kind"] == "coin":
            assert row["objects"] == []
        else:
            assert row["objects"] == [{"class_name": "single_banknote", "bbox": [16.0, 12.0, 32.0, 24.0]}]
        assert row["template_group"] == "denomination:" + row["denomination"]
    for folder in ("train", "valid", "sealed_test"):
        coco = _read(public_source["output"] / folder / "_annotations.coco.json")
        assert {category["name"] for category in coco["categories"]} == {"single_banknote"}
        assert len(coco["annotations"]) < len(coco["images"])


def test_contained_feature_boxes_are_preserved_in_source_audit_not_detector_targets(public_source):
    source = public_source["paths"]["training", 500, 0]
    source.with_suffix(".txt").write_text(
        "0 0.5 0.5 0.5 0.5\n0 0.5 0.5 0.1 0.1\n", encoding="utf-8",
    )
    _repack(public_source)
    _, lock = _prepare(public_source)
    row = next(row for row in lock["records"] if row["content_sha256"] == file_sha256(source))
    assert row["included"] is True
    assert row["objects"] == [{"class_name": "single_banknote", "bbox": [16.0, 12.0, 32.0, 24.0]}]
    assert row["label_derivation"]["discarded_contained_feature_boxes"] == 1
    exported_label = public_source["output"] / row["annotation_export_path"]
    assert exported_label.read_bytes() == source.with_suffix(".txt").read_bytes()


def test_disjoint_same_class_boxes_are_quarantined_as_ambiguous_source_annotations(public_source):
    source = public_source["paths"]["training", 500, 0]
    source.with_suffix(".txt").write_text(
        "0 0.2 0.5 0.2 0.5\n0 0.8 0.5 0.2 0.5\n", encoding="utf-8",
    )
    _repack(public_source)
    _, lock = _prepare(public_source)
    row = next(row for row in lock["records"] if row["content_sha256"] == file_sha256(source))
    assert row["split"] == "quarantine" and row["included"] is False
    assert row["objects"] == []
    assert row["label_derivation"]["ambiguous"] is True


@pytest.mark.parametrize("label", [
    "1 0.5 0.5 0.5 0.5\n",  # Valid class for the 100-tenge coin, wrong here.
    "0 0.95 0.5 0.5 0.5\n",  # Center in range, box extends beyond the image.
    "0 0.5 0.5 0 0.5\n", "0 nan 0.5 0.5 0.5\n",
    "0 0.5 0.5 0.5\n", "0 0.5 0.5 0.5 0.5 extra\n", "",
])
def test_public_yolo_rejects_bad_labels(public_source, label):
    public_source["paths"]["training", 500, 0].with_suffix(".txt").write_text(label, encoding="utf-8")
    _repack(public_source)
    with pytest.raises(TrainingGateError):
        prepare_public_dataset(public_source["config_path"], public_source["output"])


@pytest.mark.parametrize("change", [
    lambda c: c.update(schema_version="forte.specialists.data.v1"),
    lambda c: c.update(frozen=False),
    lambda c: c["source_archive"].update(sha256="f" * 64),
    lambda c: c["source_archive"].update(size_bytes=1),
    lambda c: c["source_archive"].update(dataset_id="different_dataset"),
    lambda c: c["source_archive"].update(version=2),
    lambda c: c["source_archive"].update(url="http://example.invalid/source.zip"),
    lambda c: c["license"].update(spdx="CC-BY-NC-4.0"),
    lambda c: c["license"]["evidence"].update(sha256="f" * 64),
    lambda c: c["annotation_publication"].update(method="unreviewed_machine"),
    lambda c: c["annotation_publication"]["evidence"].update(sha256="f" * 64),
    lambda c: c["organizer_inventory"].update(sha256="f" * 64),
    lambda c: c["split_policy"].update(declared_before_import=False),
    lambda c: c["split_policy"].update(mode="random_frame_split"),
    lambda c: c["split_policy"].update(validation_denominations=["500", "20", "50"]),
    lambda c: c["split_policy"].update(near_duplicate_hamming=-1),
    lambda c: c["denomination_classes"]["1000"].update(kind="coin"),
    lambda c: c["denomination_classes"].pop("20000"),
])
def test_public_provenance_and_predeclared_split_guards(public_source, change):
    config = copy.deepcopy(public_source["config"])
    change(config)
    write_json(public_source["config_path"], config)
    with pytest.raises(TrainingGateError):
        prepare_public_dataset(public_source["config_path"], public_source["output"])


def test_public_archive_must_match_extracted_media(public_source):
    path = public_source["paths"]["training", 500, 0]
    path.write_bytes(path.read_bytes() + b"tampered-after-extraction")
    with pytest.raises(TrainingGateError):
        prepare_public_dataset(public_source["config_path"], public_source["output"])


@pytest.mark.parametrize("member", ["../escape.jpg", "/absolute.jpg", "C:/absolute.jpg", "nested/../../escape.jpg"])
def test_public_archive_paths_cannot_escape_approved_root(public_source, member):
    with zipfile.ZipFile(public_source["archive"], "a") as archive:
        archive.writestr(member, b"synthetic-invalid-image")
    public_source["config"]["source_archive"].update(
        sha256=file_sha256(public_source["archive"]), size_bytes=public_source["archive"].stat().st_size,
    )
    write_json(public_source["config_path"], public_source["config"])
    with pytest.raises(TrainingGateError):
        prepare_public_dataset(public_source["config_path"], public_source["output"])
    assert not (public_source["base"] / "escape.jpg").exists()


def test_public_extracted_root_must_stay_inside_approved_root(public_source):
    public_source["config"]["approved_root"] = str(public_source["base"] / "unapproved")
    write_json(public_source["config_path"], public_source["config"])
    with pytest.raises(TrainingGateError):
        prepare_public_dataset(public_source["config_path"], public_source["output"])


@pytest.mark.parametrize("section", ["license", "annotation_publication"])
def test_evidence_files_must_substantiate_the_declared_claim(public_source, section):
    evidence = public_source["config"][section]["evidence"]
    path = Path(evidence["path"])
    path.write_text("Synthetic unrelated document with no relevant claim.", encoding="utf-8")
    evidence["sha256"] = file_sha256(path)
    write_json(public_source["config_path"], public_source["config"])
    with pytest.raises(TrainingGateError):
        prepare_public_dataset(public_source["config_path"], public_source["output"])


def test_public_inventory_excludes_rebranded_organizer_images(public_source):
    path = public_source["paths"]["training", 500, 0]
    inventory = Path(public_source["config"]["organizer_inventory"]["path"])
    write_json(inventory, [{"content_sha256": file_sha256(path)}])
    public_source["config"]["organizer_inventory"]["sha256"] = file_sha256(inventory)
    write_json(public_source["config_path"], public_source["config"])
    with pytest.raises(TrainingGateError):
        prepare_public_dataset(public_source["config_path"], public_source["output"])


def test_cross_role_near_duplicates_are_quarantined(public_source):
    training = public_source["paths"]["training", 500, 0]
    sealed = public_source["paths"]["validation", 20000, 0]
    sealed.write_bytes(training.read_bytes() + b"same-pixels-distinct-encoding")
    assert file_sha256(training) != file_sha256(sealed)
    assert dhash256(training) == dhash256(sealed)
    _repack(public_source)
    _, lock = _prepare(public_source)
    hashes = {file_sha256(training), file_sha256(sealed)}
    rows = [row for row in lock["records"] if row["content_sha256"] in hashes]
    assert len(rows) == 2
    assert len({row["source_group"] for row in rows}) == 1
    assert all(row["split"] == "quarantine" and row["included"] is False for row in rows)
    assert all(row["export_path"].startswith("quarantine/") for row in rows)
    for folder in ("train", "valid", "sealed_test", "test"):
        assert not hashes & {file_sha256(path) for path in (public_source["output"] / folder).glob("*.jpg")}


def test_cross_upstream_duplicates_of_one_heldout_denomination_keep_the_heldout_role(public_source):
    upstream_training = public_source["paths"]["training", 20000, 0]
    upstream_validation = public_source["paths"]["validation", 20000, 0]
    upstream_validation.write_bytes(upstream_training.read_bytes() + b"same-heldout-template")
    _repack(public_source)
    _, lock = _prepare(public_source)
    hashes = {file_sha256(upstream_training), file_sha256(upstream_validation)}
    rows = [row for row in lock["records"] if row["content_sha256"] in hashes]
    assert len(rows) == 2 and len({row["source_group"] for row in rows}) == 1
    assert all(row["split"] == "sealed_test" for row in rows)


def test_same_role_same_kind_near_duplicates_across_denominations_share_one_group(public_source):
    original = public_source["paths"]["training", 500, 0]
    related = public_source["paths"]["training", 1000, 0]
    related.write_bytes(original.read_bytes() + b"same-kind-near-duplicate-audit")
    hashes = {file_sha256(original), file_sha256(related)}
    assert len(hashes) == 2 and dhash256(original) == dhash256(related)
    _repack(public_source)
    _, lock = _prepare(public_source)
    rows = [row for row in lock["records"] if row["source_content_sha256"] in hashes]
    assert len(rows) == 2 and all(row["split"] == "train" and row["included"] for row in rows)
    assert {row["denomination"] for row in rows} == {"500", "1000"}
    assert len({row["source_group"] for row in rows}) == 1
    assert len({row["template_group"] for row in rows}) == 2
    assert all(row["quarantine_reasons"] == [] for row in rows)


def test_same_content_across_denominations_is_quarantined_as_conflicting_metadata(public_source):
    original = public_source["paths"]["training", 500, 0]
    conflicting = public_source["paths"]["training", 1000, 0]
    conflicting.write_bytes(original.read_bytes())
    _repack(public_source)
    _, lock = _prepare(public_source)
    rows = [row for row in lock["records"] if row["source_content_sha256"] == file_sha256(original)]
    assert len(rows) == 2 and all(row["split"] == "quarantine" and not row["included"] for row in rows)
    assert {row["denomination"] for row in rows} == {"500", "1000"}
    assert len({row["source_group"] for row in rows}) == 1
    assert all("conflicting_exact_content_denominations" in row["quarantine_reasons"] for row in rows)


def test_cross_kind_near_duplicates_are_quarantined_even_within_the_same_role(public_source):
    banknote = public_source["paths"]["training", 500, 0]
    coin = public_source["paths"]["training", 1, 0]
    coin.write_bytes(banknote.read_bytes() + b"cross-kind-near-duplicate-audit")
    hashes = {file_sha256(banknote), file_sha256(coin)}
    assert len(hashes) == 2 and dhash256(banknote) == dhash256(coin)
    _repack(public_source)
    _, lock = _prepare(public_source)
    rows = [row for row in lock["records"] if row["source_content_sha256"] in hashes]
    assert len(rows) == 2 and all(row["split"] == "quarantine" and not row["included"] for row in rows)
    assert {row["kind"] for row in rows} == {"banknote", "coin"}
    assert len({row["source_group"] for row in rows}) == 1
    assert all("conflicting_source_kinds" in row["quarantine_reasons"] for row in rows)


def test_same_upstream_exact_duplicates_have_one_canonical_export(public_source):
    original = public_source["paths"]["training", 500, 0]
    duplicate = original.with_name("500_99.jpg")
    duplicate.write_bytes(original.read_bytes())
    duplicate.with_suffix(".txt").write_bytes(original.with_suffix(".txt").read_bytes())
    _repack(public_source)
    _, lock = _prepare(public_source)
    rows = [row for row in lock["records"] if row["content_sha256"] == file_sha256(original)]
    assert len(rows) == 2
    assert sum(row["included"] is True for row in rows) == 1
    assert len({row["export_path"] for row in rows}) == 1
    assert len({row["source_group"] for row in rows}) == 1
    canonical = next(row for row in rows if row["included"])
    assert next(row for row in rows if not row["included"])["duplicate_of"] == canonical["sample_id"]


def test_downscale_preserves_original_identity_and_scales_detector_boxes(public_source):
    original = public_source["paths"]["training", 500, 0]
    generator = random.Random(16384)
    Image.frombytes("RGB", (1600, 1200), generator.randbytes(1600 * 1200 * 3)).save(original, quality=95)
    public_source["config"]["image_preparation"] = {"max_side": 800}
    _repack(public_source)
    _, lock = _prepare(public_source)
    source_hash = file_sha256(original)
    row = next(row for row in lock["records"] if row["source_content_sha256"] == source_hash)
    assert (row["source_width"], row["source_height"]) == (1600, 1200)
    assert (row["width"], row["height"]) == (800, 600)
    assert row["content_sha256"] != source_hash
    exported = public_source["output"] / row["export_path"]
    assert file_sha256(exported) == row["content_sha256"]
    assert row["source_target_objects"] == [{"class_name": "single_banknote", "bbox": [400.0, 300.0, 800.0, 600.0]}]
    assert row["objects"] == [{"class_name": "single_banknote", "bbox": [200.0, 150.0, 400.0, 300.0]}]
    assert row["image_derivation"]["method"] == "lanczos_downscale_jpeg"
    assert row["image_derivation"]["scale_x"] == row["image_derivation"]["scale_y"] == 0.5
    with Image.open(exported) as prepared:
        assert prepared.size == (800, 600)


def test_conflicting_exact_duplicate_annotations_keep_distinct_source_audit_in_quarantine(public_source):
    original = public_source["paths"]["training", 500, 0]
    duplicate = original.with_name("500_99.jpg")
    duplicate.write_bytes(original.read_bytes())
    duplicate.with_suffix(".txt").write_text("0 0.25 0.5 0.25 0.5\n", encoding="utf-8")
    _repack(public_source)
    _, lock = _prepare(public_source)
    source_hash = file_sha256(original)
    rows = [row for row in lock["records"] if row["source_content_sha256"] == source_hash]
    assert len(rows) == 2 and all(row["split"] == "quarantine" and not row["included"] for row in rows)
    assert all("conflicting_exact_content_annotations" in row["quarantine_reasons"] for row in rows)
    assert len({row["annotation_sha256"] for row in rows}) == 2
    assert len({row["annotation_export_path"] for row in rows}) == 2
    assert {tuple(row["source_target_objects"][0]["bbox"]) for row in rows} == {
        (16.0, 12.0, 32.0, 24.0), (8.0, 12.0, 16.0, 24.0),
    }
    assert len({canonical_sha256(row["source_objects"]) for row in rows}) == 2


def test_coco_semantics_are_recomputed_even_if_file_hash_and_lock_are_repinned(public_source):
    _, lock = _prepare(public_source)
    relative = "train/_annotations.coco.json"
    path = public_source["output"] / relative
    coco = _read(path)
    coco["annotations"][0]["bbox"][0] += 1
    write_json(path, coco)
    lock["files"][relative] = file_sha256(path)
    lock["dataset_sha256"] = canonical_sha256({key: value for key, value in lock.items() if key != "dataset_sha256"})
    lock_path = public_source["output"] / "dataset-lock.json"
    write_json(lock_path, lock)
    with pytest.raises(TrainingGateError, match="public_coco_annotation_mismatch"):
        verify_public_dataset(public_source["output"], file_sha256(lock_path))


def test_invalid_coin_bounds_are_quarantined_with_original_label_and_safe_error_code(public_source):
    source = public_source["paths"]["training", 1, 0]
    label = source.with_suffix(".txt")
    raw_label = b"0 0.99 0.5 0.5 0.5\n"
    label.write_bytes(raw_label)
    public_source["config"]["annotation_error_policy"] = "quarantine"
    _repack(public_source)
    _, lock = _prepare(public_source)
    row = next(row for row in lock["records"] if row["source_content_sha256"] == file_sha256(source))
    assert row["split"] == "quarantine" and not row["included"]
    assert row["source_objects"] == row["source_target_objects"] == row["objects"] == []
    assert row["label_derivation"]["ambiguous"] is True
    assert row["label_derivation"]["validation_error"] == "public_yolo_coordinates_invalid"
    assert "invalid_source_annotation" in row["quarantine_reasons"]
    exported_label = public_source["output"] / row["annotation_export_path"]
    assert exported_label.read_bytes() == raw_label
    assert row["annotation_sha256"] == file_sha256(label) == file_sha256(exported_label)


def test_coin_exif_rotation_is_applied_to_clean_negative_export(public_source):
    source = public_source["paths"]["training", 1, 0]
    with Image.open(source) as original:
        pixels = original.copy()
    exif = Image.Exif()
    exif[274] = 6
    pixels.save(source, quality=95, exif=exif)
    _repack(public_source)
    _, lock = _prepare(public_source)
    row = next(row for row in lock["records"] if row["source_content_sha256"] == file_sha256(source))
    assert row["split"] == "train" and row["included"]
    assert row["source_exif_orientation"] == 6 and row["source_has_exif"] is True
    assert (row["source_width"], row["source_height"]) == (64, 48)
    assert (row["width"], row["height"]) == (48, 64)
    assert row["objects"] == row["source_target_objects"] == []
    assert row["image_derivation"]["exif_transposed"] is True
    assert row["image_derivation"]["metadata_stripped"] is True
    assert row["content_sha256"] != row["source_content_sha256"]
    with Image.open(public_source["output"] / row["export_path"]) as exported:
        assert exported.size == (48, 64) and not exported.getexif()


def test_positive_paper_exif_rotation_is_quarantined_without_rotating_boxes(public_source):
    source = public_source["paths"]["training", 500, 0]
    with Image.open(source) as original:
        pixels = original.copy()
    exif = Image.Exif()
    exif[274] = 6
    pixels.save(source, quality=95, exif=exif)
    _repack(public_source)
    _, lock = _prepare(public_source)
    row = next(row for row in lock["records"] if row["source_content_sha256"] == file_sha256(source))
    assert row["split"] == "quarantine" and not row["included"]
    assert row["source_exif_orientation"] == 6
    assert (row["width"], row["height"]) == (64, 48)
    assert row["objects"] == row["source_target_objects"] == [
        {"class_name": "single_banknote", "bbox": [16.0, 12.0, 32.0, 24.0]},
    ]
    assert row["image_derivation"]["exif_transposed"] is False
    with Image.open(public_source["output"] / row["export_path"]) as exported:
        assert exported.size == (64, 48) and not exported.getexif()


def test_low_resolution_orientation_one_exif_metadata_is_stripped(public_source):
    source = public_source["paths"]["training", 500, 0]
    with Image.open(source) as original:
        pixels = original.copy()
    exif = Image.Exif()
    exif[274] = 1
    exif[315] = "synthetic-exif-marker-not-for-export"
    exif[34853] = {0: b"\x02\x03\x00\x00"}  # Synthetic GPS-version tag, no coordinates.
    pixels.save(source, quality=95, exif=exif)
    _repack(public_source)
    _, lock = _prepare(public_source)
    row = next(row for row in lock["records"] if row["source_content_sha256"] == file_sha256(source))
    assert row["split"] == "train" and row["included"]
    assert row["source_exif_orientation"] == 1
    assert row["source_has_exif"] is True and row["source_gps_tag_present"] is True
    assert row["image_derivation"]["method"] == "metadata_strip_jpeg"
    assert row["image_derivation"]["exif_transposed"] is False
    assert (row["width"], row["height"]) == (64, 48)
    assert row["content_sha256"] != row["source_content_sha256"]
    assert "synthetic-exif-marker-not-for-export" not in json.dumps(lock)
    with Image.open(public_source["output"] / row["export_path"]) as exported:
        assert exported.size == (64, 48) and not exported.getexif()


@pytest.mark.parametrize("tamper", ["coco", "image", "lock"])
def test_public_verification_detects_dataset_tampering(public_source, tamper):
    summary, lock = _prepare(public_source)
    if tamper == "coco":
        (public_source["output"] / "train/_annotations.coco.json").write_text("{}", encoding="utf-8")
    elif tamper == "image":
        path = public_source["output"] / next(row["export_path"] for row in lock["records"] if row["split"] == "train")
        path.write_bytes(path.read_bytes() + b"tamper")
    else:
        path = public_source["output"] / "dataset-lock.json"
        lock["records"][0]["split"] = "train"
        lock["state"] = "changed"
        write_json(path, lock)
    with pytest.raises(TrainingGateError):
        verify_public_dataset(public_source["output"], summary["dataset_lock_sha256"])


def test_public_lock_cannot_be_relabelled_as_reviewed_own_recordings(public_source):
    summary, lock = _prepare(public_source)
    from training.specialists.prepare import verify_dataset

    with pytest.raises(TrainingGateError):
        verify_dataset(public_source["output"], summary["dataset_lock_sha256"])
    path = public_source["output"] / "dataset-lock.json"
    lock["schema_version"] = "forte.specialists.dataset-lock.v1"
    write_json(path, lock)
    with pytest.raises(TrainingGateError):
        verify_public_dataset(public_source["output"], file_sha256(path))
