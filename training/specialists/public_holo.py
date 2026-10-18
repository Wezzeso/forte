"""Import publisher-annotated MIDV-Holo ID scene frames into a public-only lock.

Fictional Utopia ID cards use publisher document quadrangles, never passport or
hologram masks. Templates, physical documents, clips and duplicates retain their
predeclared partition. This module performs no downloads or training.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile

from .common import (canonical_sha256, cli_error, file_sha256, load_json, number,
                     pinned_json, relative_file, require, sha, write_json)
from .prepare import organizer_hashes
from .public_data import _components, _under, dhash256

SCHEMA = "forte.specialists.public-holo-data.v1"
LOCK_SCHEMA = "forte.specialists.public-dataset-lock.v1"
DATASET_ID = "MIDV-Holo"
CLASSES = ["id_card"]
DIRECTORIES = {"train": "train", "validation": "valid", "sealed_test": "sealed_test",
               "quarantine": "quarantine"}
TYPE_ROLES = {f"uto.id.type{i}": "train" if i <= 6 else "validation" if i <= 8 else "sealed_test"
              for i in range(1, 11)}
SOURCE_RE = re.compile(r"images/(?:origins|fraud/(?:photo_holo_copy|photo_replacement|copy_without_holo|pseudo_holo_copy))/ID/(id(0[1-9]|10)_([0-9]{2})_([0-9]{2}))/img_([0-9]+)\.jpg$")
ATTRIBUTION = [
    "Koliaskina et al. (2023), MIDV-Holo: A Dataset for ID Document Hologram Detection in a Video Stream. DOI: 10.1007/978-3-031-41682-8_30.",
    "Images of faces: Generated Photos (https://generated.photos/), retained per the publisher's attribution notice.",
]
LIMITATIONS = [
    "Small public scene subset of fictional Utopia documents; no real bank-camera accuracy claim.",
    "Publisher labels document boundaries only; absence of unrelated objects is not exhaustively verified.",
    "Complete images.tar was not downloaded: member payload hashes and source TAR offsets are pinned; the whole-archive SHA-256 is unavailable.",
]


def _descriptor(config: dict, key: str, approved: Path, base: Path) -> Path:
    descriptor = config.get(key, {})
    require(isinstance(descriptor, dict) and sha(descriptor.get("sha256")), "holo_source_descriptor_invalid")
    path = _under(approved, base, descriptor.get("path"))
    require(file_sha256(path) == descriptor["sha256"], "holo_source_hash_mismatch")
    return path


def _split_policy(path: Path) -> dict:
    policy = load_json(path)
    require(isinstance(policy, dict) and policy.get("created_at_utc")
            and isinstance(policy.get("policy"), str), "holo_predeclared_split_missing")
    aliases = {"train": "train", "dev": "validation", "heldout": "sealed_test",
               "validation": "validation", "sealed_test": "sealed_test"}
    groups = policy.get("groups", {})
    require(isinstance(groups, dict) and {k: aliases.get(v) for k, v in groups.items()} == TYPE_ROLES,
            "holo_template_split_policy_mismatch")
    return policy


def _license(path: Path, readme: Path) -> dict:
    text = path.read_text(encoding="utf-8-sig")
    description = readme.read_text(encoding="utf-8-sig")
    require("creativecommons.org/licenses/by-sa/2.5/" in text
            and "Generated Photos" in text and "generated.photos" in text,
            "holo_license_or_attribution_missing")
    require("Utopia" in description and "artificial" in description.lower()
            and "manually selected document boundary" in description.lower(),
            "holo_publisher_annotation_evidence_missing")
    return {"spdx": "CC-BY-SA-2.5", "url": "https://creativecommons.org/licenses/by-sa/2.5/",
            "attribution": ATTRIBUTION, "share_alike_notice": "Retain this license and attribution for derivative dataset artifacts.",
            "source_license_sha256": file_sha256(path), "source_readme_sha256": file_sha256(readme)}


def _receipt(path: Path, split_sha: str) -> tuple[dict, list[dict]]:
    receipt = load_json(path)
    require(isinstance(receipt, dict) and receipt.get("dataset") == DATASET_ID,
            "holo_acquisition_identity_invalid")
    require(receipt.get("archive_uri") == "ftp://smartengines.com/midv-holo/images.tar"
            and receipt.get("archive_size") == 14166804480
            and receipt.get("whole_archive_sha256") is None,
            "holo_archive_source_invalid")
    require(receipt.get("predeclared_split_sha256") == split_sha
            and receipt.get("selection_policy_declared_before_payloads") is True,
            "holo_predeclared_acquisition_policy_required")
    rows = receipt.get("files")
    require(isinstance(rows, list) and bool(rows), "holo_scene_files_missing")
    names = set()
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get("source_path"), str)
                and SOURCE_RE.fullmatch(row["source_path"]) and row["source_path"] not in names,
                "holo_scene_path_invalid_or_duplicate")
        names.add(row["source_path"])
        require(sha(row.get("sha256")) and type(row.get("size_bytes")) is int and row["size_bytes"] > 0,
                "holo_payload_hash_required")
        offset = row.get("header_offset")
        require(type(offset) is int and offset >= 0 and offset % 512 == 0
                and row.get("data_offset") == offset + 512
                and offset + 512 + row["size_bytes"] <= receipt["archive_size"],
                "holo_tar_offset_invalid")
        require(row.get("annotation_path") == row["source_path"].replace("images/", "markup/", 1) + ".json"
                and sha(row.get("annotation_sha256")), "holo_markup_link_invalid")
    return receipt, rows


def _quad(annotation: dict, path: str, width: int, height: int) -> tuple[str, list, list]:
    match = SOURCE_RE.fullmatch(path)
    require(match is not None, "holo_scene_path_invalid")
    document_type = annotation.get("document", {}).get("document_type")
    require(document_type == f"uto.id.type{int(match.group(2))}" and document_type in TYPE_ROLES,
            "holo_passport_or_document_type_mismatch")
    templates = annotation.get("document", {}).get("templates", {})
    require(isinstance(templates, dict) and set(templates) == {document_type + ":front"},
            "holo_document_front_quad_required")
    points = templates[document_type + ":front"].get("template_quad")
    require(isinstance(points, list) and len(points) == 4
            and all(isinstance(p, list) and len(p) == 2 and all(number(v, -100000, 100000) for v in p)
                    for p in points), "holo_quad_invalid")
    area = abs(sum(points[i][0] * points[(i + 1) % 4][1] - points[(i + 1) % 4][0] * points[i][1]
                   for i in range(4))) / 2
    require(area > 0, "holo_quad_degenerate")
    left, top = max(0.0, min(p[0] for p in points)), max(0.0, min(p[1] for p in points))
    right, bottom = min(float(width), max(p[0] for p in points)), min(float(height), max(p[1] for p in points))
    require(left < right and top < bottom, "holo_document_outside_frame")
    return document_type, points, [left, top, right - left, bottom - top]


def _assign(rows: list[dict]) -> None:
    for indices in _components(rows):
        members = [rows[i] for i in indices]
        group = "public:" + canonical_sha256(sorted({r["content_sha256"] for r in members}))
        roles = {r["declared_split"] for r in members}
        content_labels: dict[str, set[str]] = defaultdict(set)
        for row in members:
            content_labels[row["content_sha256"]].add(canonical_sha256(row["objects"]))
        conflict = len(roles) > 1 or any(len(v) > 1 for v in content_labels.values())
        for row in members:
            row.update(source_group=group, split="quarantine" if conflict else row["declared_split"],
                       quarantine_reasons=["cross_split_or_conflicting_duplicate_component"] if conflict else [])
    retained = {}
    for row in rows:
        key = row["split"], row["content_sha256"]
        row["duplicate_of"] = retained.get(key)
        row["included"] = row["split"] != "quarantine" and key not in retained
        retained.setdefault(key, row["sample_id"])


def _coco(rows: list[dict], role: str, directory: str) -> dict:
    images, annotations = [], []
    for row in rows:
        if row["split"] != role or not row["included"]:
            continue
        image_id = len(images) + 1
        images.append({"id": image_id, "file_name": Path(row["export_path"]).name,
                       "width": row["width"], "height": row["height"]})
        for obj in row["objects"]:
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": 1,
                "bbox": obj["bbox"], "area": obj["bbox"][2] * obj["bbox"][3], "iscrowd": 0})
    return {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "id_card", "supercategory": "none"}]}


def prepare_public_holo_dataset(config_path: Path, output: Path) -> dict:
    config_path, output = config_path.resolve(), output.resolve()
    config = load_json(config_path)
    require(isinstance(config, dict) and config.get("schema_version") == SCHEMA
            and config.get("frozen") is True, "holo_config_schema_or_freeze_invalid")
    base = config_path.parent
    approved = (base / config.get("approved_root", "")).resolve()
    require(approved.is_dir() and config_path.is_relative_to(approved), "holo_approved_root_invalid")
    source_root = _under(approved, base, config.get("scene_root"), directory=True)
    paths = {k: _descriptor(config, k, approved, base)
             for k in ("acquisition_manifest", "markup_archive", "split_policy", "source_license", "source_readme", "organizer_inventory")}
    policy = _split_policy(paths["split_policy"])
    license_info = _license(paths["source_license"], paths["source_readme"])
    receipt, members = _receipt(paths["acquisition_manifest"], file_sha256(paths["split_policy"]))
    banned = organizer_hashes(paths["organizer_inventory"], config["organizer_inventory"]["sha256"])
    from PIL import Image
    rows = []
    with tarfile.open(paths["markup_archive"], "r:") as archive:
        annotations = {m.name: m for m in archive.getmembers() if m.isfile()}
        require(len(annotations) == sum(m.isfile() for m in archive.getmembers()), "holo_duplicate_archive_member")
        for member in sorted(members, key=lambda r: r["source_path"]):
            image_path = relative_file(source_root, member["source_path"])
            require(image_path.stat().st_size == member["size_bytes"]
                    and file_sha256(image_path) == member["sha256"] and member["sha256"] not in banned,
                    "holo_scene_payload_mismatch")
            require(member["annotation_path"] in annotations, "holo_publisher_annotation_missing")
            stream = archive.extractfile(annotations[member["annotation_path"]])
            require(stream is not None, "holo_publisher_annotation_missing")
            with stream:
                annotation_bytes = stream.read()
            require(hashlib.sha256(annotation_bytes).hexdigest() == member["annotation_sha256"]
                    and member["annotation_sha256"] not in banned, "holo_publisher_annotation_hash_mismatch")
            with Image.open(image_path) as image:
                width, height = image.size
                require(image.getexif().get(274, 1) == 1, "holo_rotated_exif_unsupported")
                image.verify()
            document_type, points, box = _quad(json.loads(annotation_bytes), member["source_path"], width, height)
            match = SOURCE_RE.fullmatch(member["source_path"])
            assert match is not None
            clip = match.group(1)
            rows.append({"sample_id": "midv-holo:" + member["source_path"], "dataset_id": DATASET_ID,
                "origin": "public", "source_path": member["source_path"], "source_annotation_path": member["annotation_path"],
                "content_sha256": member["sha256"], "annotation_sha256": member["annotation_sha256"],
                "dhash256": dhash256(image_path), "width": width, "height": height,
                "document_type": document_type, "template_group": "midv-holo:" + document_type,
                "physical_document_group": "midv-holo:" + "_".join(clip.split("_")[:2]),
                "scene_group": "midv-holo:" + str(Path(member["source_path"]).parent).replace("\\", "/"),
                "declared_split": TYPE_ROLES[document_type], "source_quad": points,
                "objects": [{"class_name": "id_card", "bbox": box}],
                "annotation_origin": "publisher_manual_document_quadrangle",
                "source_tar_header_offset": member["header_offset"], "source_tar_data_offset": member["data_offset"],
                "_image_path": image_path, "_annotation_bytes": annotation_bytes})
    _assign(rows)
    require({r["split"] for r in rows if r["included"]} >= {"train", "validation", "sealed_test"},
            "holo_partition_coverage_missing")
    require(not output.exists() and not output.is_relative_to(source_root), "holo_output_exists_or_inside_source")
    output.mkdir(parents=True)
    files, provenance_files = {}, {}
    for key, path in {"config": config_path, **paths}.items():
        relative = "provenance/" + key + path.suffix.lower()
        target = output / relative
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, target)
        files[relative] = file_sha256(target)
        provenance_files[key] = relative
    for role, directory in DIRECTORIES.items():
        (output / directory).mkdir()
    (output / "test").mkdir()
    for row in rows:
        relative = DIRECTORIES[row["split"]] + "/" + row["content_sha256"] + ".jpg"
        if relative not in files:
            shutil.copyfile(row["_image_path"], output / relative)
            files[relative] = row["content_sha256"]
        row["export_path"] = relative
        annotation_relative = "source_annotations/" + row["annotation_sha256"] + ".json"
        target = output / annotation_relative
        if annotation_relative not in files:
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(row["_annotation_bytes"])
            files[annotation_relative] = row["annotation_sha256"]
        row["annotation_export_path"] = annotation_relative
    for role, directory in [*DIRECTORIES.items(), ("unused_test", "test")]:
        relative = directory + "/_annotations.coco.json"
        write_json(output / relative, _coco(rows, role, directory))
        files[relative] = file_sha256(output / relative)
    exported = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    lock = {"schema_version": LOCK_SCHEMA, "dataset_kind": "midv_holo", "dataset_id": DATASET_ID,
        "state": "frozen", "origin": "public", "classes": CLASSES, "class_names": {"0": "id_card"},
        "coco_category_ids": {"id_card": 1}, "prediction_class_ids": {"id_card": 0},
        "source_config_sha256": file_sha256(config_path), "license": license_info,
        "annotation_publication": {"url": "https://github.com/SmartEngines/midv-holo", "method": "publisher_manual_document_quadrangle"},
        "source_provenance": {"dataset_id": DATASET_ID, "author_repository": "https://github.com/SmartEngines/midv-holo",
            "archive_uri": receipt["archive_uri"], "archive_size": receipt["archive_size"], "whole_archive_sha256": None,
            "markup_archive_sha256": file_sha256(paths["markup_archive"]), "acquisition_manifest_sha256": file_sha256(paths["acquisition_manifest"])},
        "provenance_files": provenance_files, "split_policy": policy,
        "annotation_scope": "publisher_document_quadrangles_only", "scene_group_status": "publisher_clip_and_physical_document_ids",
        "sealed_test_policy": "predeclared_id09_id10_templates_never_used_for_training_or_tuning",
        "limitations": LIMITATIONS, "records": exported, "files": files,
        "summary": {"records": len(rows), "split_counts": dict(Counter(r["split"] for r in rows)),
            "included_split_counts": dict(Counter(r["split"] for r in rows if r["included"])),
            "templates": sorted({r["document_type"] for r in rows}),
            "physical_documents": len({r["physical_document_group"] for r in rows}),
            "scene_clips": len({r["scene_group"] for r in rows}), "organizer_inventory_sha256": config["organizer_inventory"]["sha256"]}}
    lock["dataset_sha256"] = canonical_sha256(lock)
    write_json(output / "dataset-lock.json", lock)
    verify_public_holo_dataset(output, file_sha256(output / "dataset-lock.json"))
    return {"ok": True, "records": len(rows), "dataset_sha256": lock["dataset_sha256"],
            "dataset_lock_sha256": file_sha256(output / "dataset-lock.json"), "split_counts": lock["summary"]["split_counts"]}


def verify_public_holo_dataset(root: Path, expected_lock_sha256: str) -> dict:
    lock = pinned_json(root / "dataset-lock.json", expected_lock_sha256)
    require(lock.get("schema_version") == LOCK_SCHEMA and lock.get("dataset_kind") == "midv_holo"
            and lock.get("state") == "frozen" and lock.get("origin") == "public", "holo_dataset_lock_invalid")
    require(lock.get("dataset_sha256") == canonical_sha256({k: v for k, v in lock.items() if k != "dataset_sha256"}),
            "holo_dataset_digest_mismatch")
    require(lock.get("classes") == CLASSES and lock.get("class_names") == {"0": "id_card"}
            and lock.get("coco_category_ids") == {"id_card": 1} and lock.get("prediction_class_ids") == {"id_card": 0},
            "holo_class_mapping_invalid")
    files = lock.get("files", {})
    require(isinstance(files, dict) and bool(files), "holo_dataset_inventory_missing")
    for relative, expected in files.items():
        require(sha(expected) and file_sha256(relative_file(root, relative)) == expected, "holo_dataset_file_changed")
    provenance = lock.get("provenance_files", {})
    require(set(provenance) == {"config", "acquisition_manifest", "markup_archive", "split_policy", "source_license", "source_readme", "organizer_inventory"},
            "holo_provenance_inventory_invalid")
    paths = {key: relative_file(root, relative) for key, relative in provenance.items()}
    config = load_json(paths["config"])
    require(config.get("schema_version") == SCHEMA and config.get("frozen") is True
            and lock.get("source_config_sha256") == file_sha256(paths["config"]),
            "holo_preserved_config_mismatch")
    for key in set(provenance) - {"config"}:
        require(config.get(key, {}).get("sha256") == file_sha256(paths[key])
                and files.get(provenance[key]) == config[key]["sha256"], "holo_config_source_pin_mismatch")
    policy = _split_policy(paths["split_policy"])
    require(lock.get("split_policy") == policy and lock.get("license") == _license(paths["source_license"], paths["source_readme"]),
            "holo_preserved_license_or_policy_mismatch")
    receipt, members = _receipt(paths["acquisition_manifest"], file_sha256(paths["split_policy"]))
    expected_provenance = {"dataset_id": DATASET_ID, "author_repository": "https://github.com/SmartEngines/midv-holo",
        "archive_uri": receipt["archive_uri"], "archive_size": receipt["archive_size"], "whole_archive_sha256": None,
        "markup_archive_sha256": file_sha256(paths["markup_archive"]),
        "acquisition_manifest_sha256": file_sha256(paths["acquisition_manifest"])}
    require(lock.get("source_provenance") == expected_provenance
            and lock.get("annotation_publication") == {"url": "https://github.com/SmartEngines/midv-holo",
                                                       "method": "publisher_manual_document_quadrangle"}
            and lock.get("limitations") == LIMITATIONS
            and lock.get("annotation_scope") == "publisher_document_quadrangles_only",
            "holo_publisher_provenance_metadata_mismatch")
    by_name = {r["source_path"]: r for r in members}
    banned = organizer_hashes(paths["organizer_inventory"], files[provenance["organizer_inventory"]])
    rows = lock.get("records")
    require(isinstance(rows, list) and len(rows) == len(members), "holo_record_coverage_mismatch")
    seen, recomputed = set(), []
    from PIL import Image
    with tarfile.open(paths["markup_archive"], "r:") as archive:
        for raw in rows:
            row = dict(raw)
            member = by_name.get(row.get("source_path"))
            require(member is not None and row.get("sample_id") == "midv-holo:" + row["source_path"]
                    and row["sample_id"] not in seen, "holo_record_identity_invalid")
            seen.add(row["sample_id"])
            require(row.get("content_sha256") == member["sha256"] and row["content_sha256"] not in banned
                    and row.get("annotation_sha256") == member["annotation_sha256"]
                    and row["annotation_sha256"] not in banned, "holo_record_source_hash_mismatch")
            require(row.get("split") in DIRECTORIES
                    and row.get("export_path") == DIRECTORIES[row["split"]] + "/" + row["content_sha256"] + ".jpg"
                    and row.get("annotation_export_path") == "source_annotations/" + row["annotation_sha256"] + ".json",
                    "holo_record_export_path_not_canonical")
            image_path = relative_file(root, row["export_path"])
            require(files.get(row["export_path"]) == row["content_sha256"]
                    and image_path.stat().st_size == member["size_bytes"], "holo_record_export_mismatch")
            with Image.open(image_path) as image:
                width, height = image.size
            require((width, height) == (row.get("width"), row.get("height"))
                    and row.get("dhash256") == dhash256(image_path), "holo_record_image_metadata_mismatch")
            annotation_path = relative_file(root, row["annotation_export_path"])
            require(files.get(row["annotation_export_path"]) == row["annotation_sha256"], "holo_record_annotation_inventory_mismatch")
            original = archive.extractfile(member["annotation_path"])
            require(original is not None, "holo_publisher_annotation_missing")
            with original:
                require(hashlib.sha256(original.read()).hexdigest() == row["annotation_sha256"], "holo_publisher_annotation_hash_mismatch")
            document_type, points, box = _quad(load_json(annotation_path), row["source_path"], width, height)
            clip = SOURCE_RE.fullmatch(row["source_path"]).group(1)
            expected = {"dataset_id": DATASET_ID, "origin": "public", "document_type": document_type,
                "template_group": "midv-holo:" + document_type,
                "physical_document_group": "midv-holo:" + "_".join(clip.split("_")[:2]),
                "scene_group": "midv-holo:" + str(Path(row["source_path"]).parent).replace("\\", "/"),
                "declared_split": TYPE_ROLES[document_type], "source_quad": points,
                "objects": [{"class_name": "id_card", "bbox": box}],
                "source_annotation_path": member["annotation_path"],
                "annotation_origin": "publisher_manual_document_quadrangle",
                "source_tar_header_offset": member["header_offset"], "source_tar_data_offset": member["data_offset"]}
            require(all(row.get(k) == v for k, v in expected.items()), "holo_record_publisher_metadata_mismatch")
            recomputed.append(dict(row))
    _assign(recomputed)
    require(recomputed == rows, "holo_split_or_duplicate_assignment_mismatch")
    expected_summary = {"records": len(rows), "split_counts": dict(Counter(r["split"] for r in rows)),
        "included_split_counts": dict(Counter(r["split"] for r in rows if r["included"])),
        "templates": sorted({r["document_type"] for r in rows}),
        "physical_documents": len({r["physical_document_group"] for r in rows}),
        "scene_clips": len({r["scene_group"] for r in rows}),
        "organizer_inventory_sha256": files[provenance["organizer_inventory"]]}
    require(lock.get("summary") == expected_summary, "holo_summary_recompute_mismatch")
    for key in ("template_group", "physical_document_group", "scene_group", "source_group"):
        groups: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            if row["included"]:
                groups[row[key]].add(row["split"])
        require(all(len(roles) == 1 for roles in groups.values()), "holo_partition_group_leakage")
    for role, directory in [*DIRECTORIES.items(), ("unused_test", "test")]:
        relative = directory + "/_annotations.coco.json"
        require(relative in files and load_json(root / relative) == _coco(rows, role, directory),
                "holo_coco_publisher_recompute_mismatch")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(actual == set(files) | {"dataset-lock.json"}, "holo_uninventoried_dataset_file")
    return lock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare_public_holo_dataset(args.config, args.output)))
        return 0
    except Exception as exc:
        return cli_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
