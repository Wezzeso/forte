"""Synthetic MIDV-Holo import guards; fixtures are not real publisher imagery."""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import random
import tarfile

from PIL import Image
import pytest

from training.guards import TrainingGateError
from training.specialists.common import canonical_sha256, file_sha256, write_json
from training.specialists.public_holo import (_assign, _quad, prepare_public_holo_dataset,
                                             verify_public_holo_dataset)


@pytest.fixture
def holo_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    scene = source / "scene_subset"
    scene.mkdir()
    policy_path = source / "template-group-splits.json"
    write_json(policy_path, {"created_at_utc": "2026-10-18T00:00:00Z",
        "policy": "Synthetic fixture declared before fixture pixels.",
        "groups": {f"uto.id.type{i}": "train" if i <= 6 else "dev" if i <= 8 else "heldout"
                   for i in range(1, 11)}})
    license_path = source / "license.txt"
    license_path.write_text("creativecommons.org/licenses/by-sa/2.5/ Generated Photos https://generated.photos/")
    readme_path = source / "README.md"
    readme_path.write_text("Utopia artificial documents; manually selected document boundary quadrangle.")
    inventory = source / "organizer.json"
    write_json(inventory, [{"content_sha256": "a" * 64}])
    receipt = {"dataset": "MIDV-Holo", "archive_uri": "ftp://smartengines.com/midv-holo/images.tar",
        "archive_size": 14166804480, "whole_archive_sha256": None,
        "predeclared_split_sha256": file_sha256(policy_path),
        "selection_policy_declared_before_payloads": True, "files": []}
    archive_path = source / "markup.tar"
    with tarfile.open(archive_path, "w") as archive:
        for n, type_id in enumerate((1, 7, 9)):
            path = f"images/origins/ID/id{type_id:02d}_01_01/img_0001.jpg"
            destination = scene / path
            destination.parent.mkdir(parents=True)
            rng = random.Random(42 + n)
            image = Image.frombytes("RGB", (48, 64), bytes(rng.randrange(256) for _ in range(48 * 64 * 3)))
            image.save(destination)
            annotation = {"document": {"document_type": f"uto.id.type{type_id}", "templates": {
                f"uto.id.type{type_id}:front": {"template_quad": [[4, 8], [40, 6], [42, 52], [2, 56]]}}}, "props": {}}
            raw = json.dumps(annotation).encode()
            annotation_path = path.replace("images/", "markup/", 1) + ".json"
            member = tarfile.TarInfo(annotation_path)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
            receipt["files"].append({"source_path": path, "sha256": file_sha256(destination),
                "size_bytes": destination.stat().st_size, "header_offset": 1048576 * (n + 1),
                "data_offset": 1048576 * (n + 1) + 512, "annotation_path": annotation_path,
                "annotation_sha256": hashlib.sha256(raw).hexdigest()})
    receipt_path = source / "acquisition-manifest.json"
    write_json(receipt_path, receipt)
    config = {"schema_version": "forte.specialists.public-holo-data.v1", "frozen": True,
              "approved_root": str(source), "scene_root": str(scene)}
    for field, path in {"split_policy": policy_path, "source_license": license_path,
        "source_readme": readme_path, "organizer_inventory": inventory,
        "markup_archive": archive_path, "acquisition_manifest": receipt_path}.items():
        config[field] = {"path": str(path), "sha256": file_sha256(path)}
    config_path = source / "import.json"
    write_json(config_path, config)
    return {"config": config, "config_path": config_path, "source": source,
            "output": tmp_path / "dataset", "receipt": receipt, "receipt_path": receipt_path}


def test_import_retains_publisher_quad_license_and_template_disjointness(holo_source):
    result = prepare_public_holo_dataset(holo_source["config_path"], holo_source["output"])
    lock = verify_public_holo_dataset(holo_source["output"], result["dataset_lock_sha256"])
    assert lock["classes"] == ["id_card"] and lock["class_names"] == {"0": "id_card"}
    assert lock["coco_category_ids"] == {"id_card": 1}
    assert lock["summary"]["included_split_counts"] == {"train": 1, "validation": 1, "sealed_test": 1}
    assert all(row["objects"] == [{"class_name": "id_card", "bbox": [2, 6, 40, 50]}] for row in lock["records"])
    assert lock["license"]["spdx"] == "CC-BY-SA-2.5"
    assert any("Generated Photos" in text for text in lock["license"]["attribution"])
    assert lock["source_provenance"]["whole_archive_sha256"] is None
    assert not any("human_reviewed" in row or "participants_consented" in row for row in lock["records"])
    placeholder = json.loads((holo_source["output"] / "test/_annotations.coco.json").read_text())
    assert placeholder["images"] == [] and placeholder["annotations"] == []


def test_receipt_image_hash_is_verified(holo_source):
    image = holo_source["source"] / "scene_subset" / holo_source["receipt"]["files"][0]["source_path"]
    image.write_bytes(b"changed")
    with pytest.raises(TrainingGateError, match="scene_payload_mismatch"):
        prepare_public_holo_dataset(holo_source["config_path"], holo_source["output"])


def test_passport_never_becomes_id_card_positive():
    annotation = {"document": {"document_type": "uto.passport.type1", "templates": {
        "uto.passport.type1:main": {"template_quad": [[0, 0], [20, 0], [20, 20], [0, 20]]}}}}
    with pytest.raises(TrainingGateError, match="passport_or_document_type_mismatch"):
        _quad(annotation, "images/origins/ID/id01_01_01/img_0001.jpg", 32, 32)


def test_partial_publisher_document_box_clips_to_actual_scene():
    annotation = {"document": {"document_type": "uto.id.type1", "templates": {
        "uto.id.type1:front": {"template_quad": [[-4, 2], [40, 2], [40, 20], [-4, 20]]}}}}
    _type, points, box = _quad(annotation, "images/origins/ID/id01_01_01/img_0001.jpg", 32, 32)
    assert points[0] == [-4, 2] and box == [0, 2, 32, 18]


def test_predeclared_split_cannot_be_changed(holo_source):
    path = Path(holo_source["config"]["split_policy"]["path"])
    policy = json.loads(path.read_text())
    policy["groups"]["uto.id.type9"] = "train"
    write_json(path, policy)
    holo_source["config"]["split_policy"]["sha256"] = file_sha256(path)
    write_json(holo_source["config_path"], holo_source["config"])
    with pytest.raises(TrainingGateError, match="template_split_policy_mismatch"):
        prepare_public_holo_dataset(holo_source["config_path"], holo_source["output"])


def test_forged_coco_labels_are_recomputed_from_publisher_quad(holo_source):
    result = prepare_public_holo_dataset(holo_source["config_path"], holo_source["output"])
    root = holo_source["output"]
    path = root / "train/_annotations.coco.json"
    coco = json.loads(path.read_text())
    coco["annotations"][0]["bbox"] = [1, 1, 2, 2]
    write_json(path, coco)
    lock_path = root / "dataset-lock.json"
    lock = json.loads(lock_path.read_text())
    lock["files"]["train/_annotations.coco.json"] = file_sha256(path)
    lock["dataset_sha256"] = canonical_sha256({k: v for k, v in lock.items() if k != "dataset_sha256"})
    write_json(lock_path, lock)
    with pytest.raises(TrainingGateError, match="coco_publisher_recompute_mismatch"):
        verify_public_holo_dataset(root, file_sha256(lock_path))


def test_lock_does_not_accept_changed_file_or_missing_attribution(holo_source):
    result = prepare_public_holo_dataset(holo_source["config_path"], holo_source["output"])
    license_path = holo_source["output"] / "provenance/source_license.txt"
    license_path.write_text("CC-BY-SA-2.5")
    with pytest.raises(TrainingGateError, match="dataset_file_changed"):
        verify_public_holo_dataset(holo_source["output"], result["dataset_lock_sha256"])


def _duplicate_row(sample, digest, dhash, role="train", bbox=None):
    return {"sample_id": sample, "content_sha256": digest, "dhash256": f"{dhash:064x}",
            "declared_split": role, "objects": [{"class_name": "id_card", "bbox": bbox or [1, 1, 4, 3]}]}


def test_exact_duplicate_included_only_once_without_fabricating_new_group():
    rows = [_duplicate_row("a", "a" * 64, 0), _duplicate_row("b", "a" * 64, 0)]
    _assign(rows)
    assert [r["included"] for r in rows] == [True, False]
    assert rows[0]["source_group"] == rows[1]["source_group"]
    assert rows[1]["duplicate_of"] == "a"


@pytest.mark.parametrize("roles,boxes", [
    (("train", "validation"), ([1, 1, 4, 3], [1, 1, 4, 3])),
    (("train", "train"), ([1, 1, 4, 3], [2, 2, 4, 3])),
])
def test_duplicate_split_and_annotation_conflicts_quarantine_entire_component(roles, boxes):
    rows = [_duplicate_row("a", "a" * 64, 0, roles[0], boxes[0]),
            _duplicate_row("b", "a" * 64, 0, roles[1], boxes[1])]
    _assign(rows)
    assert all(r["split"] == "quarantine" and not r["included"] for r in rows)


def test_near_duplicate_radius_four_transitive_closure_blocks_cross_split_training():
    rows = [_duplicate_row("a", "a" * 64, 0), _duplicate_row("b", "b" * 64, 15),
            _duplicate_row("c", "c" * 64, 255, "sealed_test")]
    _assign(rows)
    assert len({r["source_group"] for r in rows}) == 1
    assert all(r["split"] == "quarantine" for r in rows)


@pytest.mark.parametrize("field", ["summary", "export_path"])
def test_rehashed_metadata_cannot_forge_counts_or_export_locations(holo_source, field):
    prepare_public_holo_dataset(holo_source["config_path"], holo_source["output"])
    lock_path = holo_source["output"] / "dataset-lock.json"
    lock = json.loads(lock_path.read_text())
    if field == "summary":
        lock["summary"]["physical_documents"] = 500
        code = "summary_recompute_mismatch"
    else:
        lock["records"][0]["export_path"] = "valid/wrong.jpg"
        code = "export_path_not_canonical"
    lock["dataset_sha256"] = canonical_sha256({k: v for k, v in lock.items() if k != "dataset_sha256"})
    write_json(lock_path, lock)
    with pytest.raises(TrainingGateError, match=code):
        verify_public_holo_dataset(holo_source["output"], file_sha256(lock_path))
