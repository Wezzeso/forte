"""Verify a frozen public training bundle without decoding sealed images.

The source importers perform their full provenance/geometry audit at data
freeze. Fitting pins that audited lock and rechecks every training/development
payload plus provenance bytes. Sealed image inventory is metadata until a
registered final-evaluation participant claims access.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path, PurePosixPath

from .common import canonical_sha256, file_sha256, load_json, pinned_json, relative_file, require, sha

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def verify_fit_bundle(root: Path, expected_lock_sha256: str) -> dict:
    root = root.resolve()
    lock = pinned_json(root / "dataset-lock.json", expected_lock_sha256)
    require(lock.get("schema_version") == "forte.specialists.public-dataset-lock.v1"
            and lock.get("state") == "frozen" and lock.get("origin") == "public",
            "public_training_requires_public_data")
    require(lock.get("dataset_sha256") == canonical_sha256({key: value for key, value in lock.items()
                                                            if key != "dataset_sha256"}),
            "public_fit_bundle_digest_mismatch")
    classes = lock.get("classes")
    require(isinstance(classes, list) and bool(classes) and len(set(classes)) == len(classes)
            and set(classes) <= {"single_banknote", "id_card"}
            and lock.get("class_names") == {str(i): name for i, name in enumerate(classes)}
            and lock.get("prediction_class_ids") == {name: i for i, name in enumerate(classes)}
            and lock.get("coco_category_ids") == {name: i + 1 for i, name in enumerate(classes)},
            "public_fit_bundle_class_mapping")
    files = lock.get("files")
    require(isinstance(files, dict) and bool(files), "public_fit_bundle_inventory_missing")
    for relative, expected in files.items():
        path = PurePosixPath(relative)
        require(isinstance(relative, str) and not path.is_absolute() and ".." not in path.parts
                and "\\" not in relative and sha(expected), "public_fit_bundle_inventory_invalid")
        image = path.suffix.lower() in IMAGE_SUFFIXES
        if not image or path.parts[0] in {"train", "valid"}:
            require(file_sha256(relative_file(root, relative)) == expected, "public_fit_bundle_file_changed")
        # No stat/open/Image.verify/dhash of heldout/quarantine/source images.
    require(all(part + "/_annotations.coco.json" in files for part in ("train", "valid", "sealed_test")),
            "public_fit_bundle_coco_missing")
    rows = lock.get("records")
    require(isinstance(rows, list) and bool(rows), "public_fit_bundle_records_missing")
    groups = {name: defaultdict(set) for name in ("source_group", "template_group", "content_sha256", "source_content_sha256")}
    active = []
    for row in rows:
        require(isinstance(row, dict) and row.get("split") in {"train", "validation", "sealed_test", "quarantine"}
                and type(row.get("included")) is bool, "public_fit_bundle_record_invalid")
        if not row["included"]:
            continue
        role = row["split"]
        require(role != "quarantine", "public_fit_bundle_quarantine_included")
        relative = row.get("export_path")
        expected_part = {"train": "train", "validation": "valid", "sealed_test": "sealed_test"}[role]
        require(isinstance(relative, str) and relative.startswith(expected_part + "/")
                and sha(row.get("content_sha256")) and files.get(relative) == row["content_sha256"],
                "public_fit_bundle_record_inventory_mismatch")
        for field, assignments in groups.items():
            value = row.get(field)
            if value is not None:
                require(isinstance(value, str) and value, "public_fit_bundle_group_invalid")
                assignments[value].add(role)
        active.append(row)
        if role in {"train", "validation"}:
            from PIL import Image
            try:
                with Image.open(relative_file(root, relative)) as image:
                    require(image.size == (row.get("width"), row.get("height")), "public_fit_bundle_dimensions_changed")
                    image.verify()
            except (OSError, ValueError):
                require(False, "public_fit_bundle_image_decode_failed")
    require(all(len(roles) == 1 for assignments in groups.values() for roles in assignments.values()),
            "public_fit_bundle_group_leakage")
    require({"train", "validation", "sealed_test"} <= {row["split"] for row in active},
            "public_fit_bundle_partition_coverage")
    for role, directory in (("train", "train"), ("validation", "valid")):
        coco = load_json(relative_file(root, directory + "/_annotations.coco.json"))
        role_rows = [row for row in active if row["split"] == role]
        require(isinstance(coco.get("images"), list) and len(coco["images"]) == len(role_rows),
                "public_fit_bundle_coco_count")
        indexed = {row["image_id"]: row for row in role_rows}
        require(len(indexed) == len(role_rows), "public_fit_bundle_image_id_duplicate")
        for image in coco["images"]:
            row = indexed.get(image.get("id"))
            require(row is not None and image.get("file_name") == PurePosixPath(row["export_path"]).name
                    and image.get("width") == row["width"] and image.get("height") == row["height"],
                    "public_fit_bundle_coco_image_mismatch")
    return lock
