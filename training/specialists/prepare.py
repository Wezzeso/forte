"""Validate reviewed own/synthetic frames and export immutable COCO partitions."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from .common import (canonical_sha256, cli_error, file_sha256, load_json, number,
                     relative_file, require, sha, write_json)

ROLES = {"train": "train", "validation": "valid", "holdout": "holdout",
         "organizer_eval": "organizer_eval"}


def organizer_hashes(path: Path, expected: str) -> set[str]:
    require(sha(expected), "specialist_organizer_inventory_hash_required")
    require(path.is_file() and file_sha256(path) == expected,
            "specialist_organizer_inventory_hash_mismatch")
    try:
        text = path.read_text(encoding="utf-8-sig")
        rows = json.loads(text) if path.suffix == ".json" else [
            json.loads(line) for line in text.splitlines() if line.strip()]
    except (ValueError, OSError):
        require(False, "specialist_organizer_inventory_invalid")
    if isinstance(rows, dict):
        rows = rows.get("records")
    require(isinstance(rows, list) and bool(rows), "specialist_organizer_inventory_empty")
    hashes = {row.get("content_sha256") for row in rows if isinstance(row, dict)}
    require(len(hashes) > 0 and all(sha(h) for h in hashes),
            "specialist_organizer_inventory_invalid")
    return hashes


def validate_manifest(manifest: dict, base: Path) -> tuple[list[dict], list[str], dict]:
    require(manifest.get("schema_version") == "forte.specialists.data.v1", "specialist_data_schema")
    require(manifest.get("frozen") is True and manifest.get("reviewed_by")
            and manifest.get("reviewed_at"), "specialist_data_review_required")
    classes = manifest.get("classes")
    require(isinstance(classes, list) and bool(classes)
            and all(isinstance(c, str) and c and c.isascii() and c.replace("_", "").isalnum() for c in classes)
            and len(classes) == len(set(classes)), "specialist_classes_invalid")
    require({"id_card", "key_in_lock"} <= set(classes), "specialist_required_classes_missing")
    inventory = manifest.get("organizer_inventory", {})
    inventory_path = (base / inventory.get("path", "")).resolve()
    banned = organizer_hashes(inventory_path, inventory.get("sha256", ""))
    media_root = (base / manifest.get("media_root", "")).resolve()
    require(media_root.is_dir(), "specialist_media_root_missing")
    rows = manifest.get("records")
    require(isinstance(rows, list) and bool(rows), "specialist_records_empty")
    by_id, group_roles, recording_roles, content_roles = {}, {}, {}, {}
    source_files: dict[str, str] = {}
    camera_roles: dict[str, set[str]] = {}
    verified = []
    from PIL import Image
    for raw in sorted(rows, key=lambda row: str(row.get("sample_id", ""))):
        require(isinstance(raw, dict), "specialist_record_invalid")
        row = dict(raw)
        sample, role, origin = row.get("sample_id"), row.get("split"), row.get("origin")
        require(isinstance(sample, str) and sample and sample not in by_id, "specialist_sample_id_invalid")
        require(role in ROLES and origin in {"own", "synthetic", "organizer"}, "specialist_origin_or_split_invalid")
        require((origin == "organizer") == (role == "organizer_eval"), "specialist_organizer_training_forbidden")
        source_hash, content_hash = row.get("source_recording_sha256"), row.get("content_sha256")
        require(sha(source_hash) and sha(content_hash), "specialist_content_hash_required")
        require(origin == "organizer" or source_hash not in banned, "specialist_organizer_source_rebranded")
        require(origin != "organizer" or source_hash in banned, "specialist_organizer_source_unknown")
        if origin != "organizer":
            source_path = relative_file(media_root, row.get("source_recording_path"))
            if str(source_path) not in source_files:
                source_files[str(source_path)] = file_sha256(source_path)
            require(source_files[str(source_path)] == source_hash, "specialist_source_recording_hash_mismatch")
        require(row.get("labels_reviewed") is True and row.get("exhaustive_labels") is True
                and row.get("label_reviewer") and row.get("label_reviewed_at"),
                "specialist_exhaustive_human_labels_required")
        provenance = row.get("provenance", {})
        require(isinstance(provenance, dict) and provenance.get("reference")
                and provenance.get("commercial_use_allowed") is True, "specialist_provenance_required")
        if origin == "own":
            require(provenance.get("participants_consented") is True
                    and provenance.get("recording_authorized") is True, "specialist_recording_consent_required")
        if origin == "synthetic":
            require(provenance.get("synthetic_inputs_reviewed") is True
                    and provenance.get("generator") and isinstance(provenance.get("input_source_hashes"), list)
                    and all(sha(h) for h in provenance["input_source_hashes"]),
                    "specialist_synthetic_provenance_required")
            require(not set(provenance["input_source_hashes"]) & banned,
                    "specialist_organizer_synthetic_input_forbidden")
        require(origin == "own" or role != "holdout", "specialist_holdout_requires_own_recording")
        require(content_hash not in content_roles, "specialist_duplicate_image")
        for field, assignments in (("scenario_group", group_roles),
                                   ("source_recording_sha256", recording_roles),
                                   ("content_sha256", content_roles)):
            value = row.get(field)
            require(isinstance(value, str) and value, "specialist_group_missing")
            require(assignments.get(value, role) == role, "specialist_partition_leakage")
            assignments[value] = role
        camera = row.get("camera_id")
        require(isinstance(camera, str) and camera, "specialist_camera_missing")
        if origin == "own":
            camera_roles.setdefault(camera, set()).add(role)
        path = relative_file(media_root, row.get("path"))
        require(path.suffix.lower() in {".png", ".jpg", ".jpeg"}, "specialist_image_format_invalid")
        require(file_sha256(path) == content_hash, "specialist_image_hash_mismatch")
        try:
            with Image.open(path) as image:
                require(image.size == (row.get("width"), row.get("height")), "specialist_dimensions_mismatch")
                image.verify()
        except (OSError, ValueError):
            require(False, "specialist_image_decode_failed")
        width, height = row.get("width"), row.get("height")
        require(type(width) is int and type(height) is int and width > 0 and height > 0,
                "specialist_dimensions_invalid")
        objects = row.get("objects")
        require(isinstance(objects, list), "specialist_objects_required")
        for obj in objects:
            require(isinstance(obj, dict) and obj.get("class_name") in classes, "specialist_object_class_invalid")
            box = obj.get("bbox")
            require(isinstance(box, list) and len(box) == 4 and all(number(v) for v in box),
                    "specialist_bbox_invalid")
            x, y, w, h = box
            require(w > 0 and h > 0 and x + w <= width and y + h <= height,
                    "specialist_bbox_outside_image")
        row["_resolved_path"] = path
        by_id[sample] = row
        verified.append(row)
    require({"train", "validation", "holdout"} <= {r["split"] for r in verified},
            "specialist_partition_coverage_missing")
    for roles in camera_roles.values():
        require("holdout" not in roles or roles == {"holdout"}, "specialist_holdout_camera_leakage")
    for row in verified:
        if row["origin"] == "synthetic":
            for content in row["provenance"]["input_source_hashes"]:
                require(recording_roles.get(content, row["split"]) == row["split"]
                        and content_roles.get(content, row["split"]) == row["split"],
                        "specialist_synthetic_input_leakage")
        parent = row.get("derived_from_sample_id")
        if parent:
            require(parent in by_id and parent != row["sample_id"], "specialist_derivation_parent_missing")
            for field in ("split", "origin", "source_recording_sha256", "scenario_group", "camera_id"):
                require(row[field] == by_id[parent][field], "specialist_derivation_leakage")
        seen, cursor = set(), row
        while cursor.get("derived_from_sample_id"):
            require(cursor["sample_id"] not in seen, "specialist_derivation_cycle")
            seen.add(cursor["sample_id"])
            cursor = by_id[cursor["derived_from_sample_id"]]
    summary = {"records": len(verified), "split_counts": dict(Counter(r["split"] for r in verified)),
               "organizer_inventory_sha256": inventory["sha256"],
               "holdout_cameras": sorted(c for c, roles in camera_roles.items() if roles == {"holdout"})}
    return verified, classes, summary


def prepare(manifest_path: Path, output: Path) -> dict:
    manifest = load_json(manifest_path)
    records, classes, summary = validate_manifest(manifest, manifest_path.resolve().parent)
    require(not output.exists(), "specialist_output_already_exists")
    output.mkdir(parents=True)
    categories = [{"id": i + 1, "name": name, "supercategory": "none"} for i, name in enumerate(classes)]
    files, exported_records = {}, []
    # RF-DETR sees train/valid only; test is an empty schema placeholder. The
    # camera holdout is physically separate and never passed to its train API.
    for role, directory in [*ROLES.items(), ("unused_test", "test")]:
        part = output / directory
        part.mkdir()
        images, annotations = [], []
        for row in (r for r in records if r["split"] == role):
            source = row["_resolved_path"]
            name = row["content_sha256"] + source.suffix.lower()
            target = part / name
            if not target.exists():
                shutil.copyfile(source, target)
            require(file_sha256(target) == row["content_sha256"], "specialist_copy_hash_mismatch")
            files[f"{directory}/{name}"] = row["content_sha256"]
            image_id = len(images) + 1
            images.append({"id": image_id, "file_name": name, "width": row["width"], "height": row["height"]})
            for obj in row["objects"]:
                annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                    "category_id": classes.index(obj["class_name"]) + 1,
                                    "bbox": obj["bbox"], "area": obj["bbox"][2] * obj["bbox"][3], "iscrowd": 0})
            exported_records.append({k: v for k, v in row.items() if k != "_resolved_path"} |
                                    {"export_path": f"{directory}/{name}", "image_id": image_id})
        annotation_path = part / "_annotations.coco.json"
        write_json(annotation_path, {"images": images, "annotations": annotations, "categories": categories})
        files[f"{directory}/_annotations.coco.json"] = file_sha256(annotation_path)
    lock = {"schema_version": "forte.specialists.dataset-lock.v1", "state": "frozen",
            "source_manifest_sha256": file_sha256(manifest_path), "classes": classes,
            "class_names": {str(i): name for i, name in enumerate(classes)},
            "coco_category_ids": {name: i + 1 for i, name in enumerate(classes)},
            "prediction_class_ids": {name: i for i, name in enumerate(classes)},
            "records": exported_records, "files": files, "summary": summary}
    lock["dataset_sha256"] = canonical_sha256(lock)
    write_json(output / "dataset-lock.json", lock)
    return {"ok": True, "records": len(records), "dataset_sha256": lock["dataset_sha256"],
            "dataset_lock_sha256": file_sha256(output / "dataset-lock.json"), "split_counts": summary["split_counts"]}


def verify_dataset(root: Path, expected_lock: str) -> dict:
    from .common import pinned_json
    lock = pinned_json(root / "dataset-lock.json", expected_lock)
    require(lock.get("schema_version") == "forte.specialists.dataset-lock.v1" and lock.get("state") == "frozen",
            "specialist_dataset_lock_invalid")
    require(lock.get("dataset_sha256") == canonical_sha256({k: v for k, v in lock.items() if k != "dataset_sha256"}),
            "specialist_dataset_digest_mismatch")
    require(isinstance(lock.get("files"), dict) and bool(lock["files"]), "specialist_dataset_inventory_missing")
    for relative, expected in lock["files"].items():
        require(file_sha256(relative_file(root, relative)) == expected, "specialist_dataset_file_changed")
    return lock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.manifest, args.output)))
        return 0
    except Exception as exc:
        return cli_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
