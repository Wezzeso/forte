"""Merge two pinned public exports; retain sealed groups and every source audit.

No footage is downloaded, labelled, human-reviewed, or reclassified here. The
source verifiers remain authoritative for publisher annotation derivations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path, PurePosixPath
import shutil

from .common import (canonical_sha256, cli_error, file_sha256, load_json, number,
                     pinned_json, relative_file, require, sha, write_json)
from .public_data import LOCK_SCHEMA, _components, dhash256

SCHEMA = "forte.specialists.public-merge.v1"
CLASSES = ["single_banknote", "id_card"]
DIRECTORIES = {"train": "train", "validation": "valid", "sealed_test": "sealed_test",
               "quarantine": "quarantine"}
SOURCES = {"kazakhstan_banknotes": ("dny3dgvvw8", ["single_banknote"]),
           "midv_holo": ("MIDV-Holo", ["id_card"])}
POLICY = {
    "method": "preserve_source_roles_and_quarantine_cross_role_components",
    "near_duplicate_hamming": 4,
    "transitive": True,
    "group_fields": ["source_group", "template_group", "scene_group", "physical_document_group"],
    "sealed_test": "never_promote_to_development; conflicting_nonsealed_members_quarantined",
    "conflicting_exact_annotations": "exclude_entire_component_from_coco; preserve_sealed_role",
}
LIMITATION = (
    "Each publisher annotates its own target class only. Combining the exports "
    "does not establish exhaustive annotation of both classes in every image, "
    "own-footage consent, human review, or performance on banking footage. "
    "Source scene, physical-document and template limitations remain applicable."
)


def _relative(value: object) -> str:
    """Inventory paths must be portable relative paths, never shell/path aliases."""
    require(isinstance(value, str) and bool(value) and "\\" not in value and ":" not in value,
            "public_merge_relative_path_invalid")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == value,
            "public_merge_relative_path_invalid")
    return value


def _verify_source(root: Path, expected: str, kind: str) -> dict:
    if kind == "kazakhstan_banknotes":
        from .public_data import verify_public_dataset
        lock = verify_public_dataset(root, expected)
    else:
        require(kind == "midv_holo", "public_merge_source_kind_invalid")
        from .public_holo import verify_public_holo_dataset
        lock = verify_public_holo_dataset(root, expected)
    dataset_id, classes = SOURCES[kind]
    require(lock.get("schema_version") == LOCK_SCHEMA and lock.get("origin") == "public"
            and lock.get("classes") == classes, "public_merge_source_schema_invalid")
    actual_id = lock.get("dataset_id", lock.get("source_provenance", {}).get("dataset_id"))
    require(actual_id == dataset_id, "public_merge_source_identity_invalid")
    return lock


def _descriptors(value: object, base: Path, *, embedded: bool) -> list[dict]:
    require(isinstance(value, list) and len(value) == 2, "public_merge_two_sources_required")
    descriptors = []
    for source in value:
        require(isinstance(source, dict), "public_merge_source_descriptor_invalid")
        kind = source.get("dataset_kind")
        kind = "kazakhstan_banknotes" if kind == "banknotes" else kind
        require(isinstance(kind, str) and kind in SOURCES, "public_merge_source_kind_invalid")
        dataset_id = SOURCES[kind][0]
        require(source.get("dataset_id") == dataset_id, "public_merge_source_identity_invalid")
        root_value = source.get("root")
        require(isinstance(root_value, str) and bool(root_value) and "://" not in root_value,
                "public_merge_source_root_invalid")
        if embedded:
            require(root_value == "source_datasets/" + dataset_id, "public_merge_embedded_root_invalid")
            _relative(root_value)
        root = (base / root_value).resolve()
        require(root.is_dir() and (not embedded or root.is_relative_to(base.resolve())),
                "public_merge_source_root_invalid")
        expected = source.get("lock_sha256" if embedded else "expected_lock_sha256")
        require(sha(expected), "public_merge_source_pin_required")
        descriptors.append({"dataset_kind": kind, "dataset_id": dataset_id, "root": root,
                            "lock_sha256": expected})
    require({source["dataset_kind"] for source in descriptors} == set(SOURCES),
            "public_merge_source_kind_coverage_invalid")
    return sorted(descriptors, key=lambda source: source["dataset_id"])


def _source_metadata(descriptors: list[dict], locks: dict[str, dict]) -> list[dict]:
    # Preserve all source metadata, including GeneratedPhotos attribution and
    # evidence hashes, without guessing the source export's attribution keys.
    return [{"dataset_id": source["dataset_id"], "dataset_kind": source["dataset_kind"],
             "root": "source_datasets/" + source["dataset_id"], "lock_sha256": source["lock_sha256"],
             "dataset_sha256": locks[source["dataset_id"]]["dataset_sha256"],
             "metadata": {key: value for key, value in locks[source["dataset_id"]].items()
                          if key not in {"records", "files", "dataset_sha256"}}}
            for source in descriptors]


def _attribution(provenance: list[dict]) -> str:
    parts = ["# Public dataset attribution", "", LIMITATION, ""]
    for source in provenance:
        parts += ["## " + source["dataset_id"], "",
                  "Source dataset-lock SHA-256: " + source["lock_sha256"], "",
                  "Publisher licensing, attribution, evidence and limitations (verbatim metadata):", "",
                  "```json", json.dumps(source["metadata"], ensure_ascii=False, sort_keys=True, indent=2),
                  "```", ""]
    return "\n".join(parts)


def _record(source: dict, original: dict) -> dict:
    dataset_id = source["dataset_id"]
    require(isinstance(original, dict) and isinstance(original.get("sample_id"), str)
            and bool(original["sample_id"]), "public_merge_source_record_invalid")
    require(original.get("split") in DIRECTORIES and type(original.get("included")) is bool,
            "public_merge_source_split_invalid")
    require(all(sha(original.get(field)) for field in
                ("content_sha256", "annotation_sha256", "dhash256")),
            "public_merge_source_record_hash_invalid")
    image_path = relative_file(source["root"], _relative(original.get("export_path")))
    annotation = _relative(original.get("annotation_export_path"))
    require(file_sha256(image_path) == original["content_sha256"], "public_merge_source_image_changed")
    require(file_sha256(relative_file(source["root"], annotation)) == original["annotation_sha256"],
            "public_merge_source_annotation_changed")
    width, height = original.get("width"), original.get("height")
    require(type(width) is int and type(height) is int and width > 0 and height > 0,
            "public_merge_source_dimensions_invalid")
    objects = original.get("objects")
    require(isinstance(objects, list), "public_merge_source_objects_invalid")
    for obj in objects:
        require(isinstance(obj, dict) and obj.get("class_name") in SOURCES[source["dataset_kind"]][1]
                and isinstance(obj.get("bbox"), list) and len(obj["bbox"]) == 4,
                "public_merge_source_objects_invalid")
        x, y, w, h = obj["bbox"]
        require(all(number(value) for value in (x, y, w, h)) and w > 0 and h > 0
                and x + w <= width + 1e-6 and y + h <= height + 1e-6,
                "public_merge_source_box_invalid")
    for field in POLICY["group_fields"]:
        require(original.get(field) is None or (isinstance(original[field], str) and bool(original[field])),
                "public_merge_source_group_invalid")
    require(isinstance(original.get("source_group"), str) and isinstance(original.get("template_group"), str),
            "public_merge_source_group_invalid")
    copied_root = "source_datasets/" + dataset_id + "/"
    return {"sample_id": dataset_id + "::" + original["sample_id"], "source_sample_id": original["sample_id"],
            "dataset_id": dataset_id, "dataset_kind": source["dataset_kind"],
            "split": original["split"], "included": original["included"],
            "source_split": original["split"], "source_included": original["included"],
            "source_group": original.get("source_group"), "template_group": original.get("template_group"),
            "scene_group": original.get("scene_group"),
            "physical_document_group": original.get("physical_document_group"),
            "content_sha256": original["content_sha256"], "annotation_sha256": original["annotation_sha256"],
            "source_content_sha256": original.get("source_content_sha256", original["content_sha256"]),
            "dhash256": dhash256(image_path), "source_dhash256": original["dhash256"],
            "width": width, "height": height, "objects": copy.deepcopy(objects),
            "annotation_export_path": copied_root + annotation,
            "source_export_path": copied_root + original["export_path"],
            "export_path": copied_root + original["export_path"], "image_id": None,
            "combined_group": None, "quarantine_reasons": [], "duplicate_of": None,
            "source_record": copy.deepcopy(original)}


def _assign(rows: list[dict]) -> None:
    """Enforce group/duplicate closure across both sources without moving tests."""
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)

    # All representations share one index: a raw image in one source may match
    # only another source's prepared image. Separate indexes miss that edge.
    representations = [(index, {"dhash256": row[hash_field],
                                "content_sha256": row["content_sha256"]})
                       for index, row in enumerate(rows)
                       for hash_field in ("dhash256", "source_dhash256")]
    for component in _components([probe for _owner, probe in representations]):
        first = representations[component[0]][0]
        for representation in component[1:]:
            union(first, representations[representation][0])
    groups: dict[tuple[str, str, str], int] = {}
    contents: dict[str, int] = {}
    for i, row in enumerate(rows):
        for content in {row["content_sha256"], row["source_content_sha256"]}:
            require(sha(content), "public_merge_source_content_hash_invalid")
            if content in contents:
                union(i, contents[content])
            contents[content] = i
        for field in POLICY["group_fields"]:
            value = row[field]
            if value is not None:
                key = row["dataset_id"], field, value
                if key in groups:
                    union(i, groups[key])
                groups[key] = i
    components: dict[int, list[dict]] = defaultdict(list)
    for i, row in enumerate(rows):
        components[find(i)].append(row)
    for members in components.values():
        combined_group = "combined_public:" + canonical_sha256(sorted(row["sample_id"] for row in members))
        active_roles = {row["source_split"] for row in members if row["source_included"]}
        by_content: dict[str, set[str]] = defaultdict(set)
        for row in members:
            by_content[row["content_sha256"]].add(canonical_sha256(row["objects"]))
        annotation_conflict = any(len(values) > 1 for values in by_content.values())
        reasons = []
        if len(active_roles) > 1:
            reasons.append("cross_source_role_group_or_duplicate_component")
        if annotation_conflict:
            reasons.append("conflicting_exact_content_target_annotations")
        for row in members:
            row["combined_group"] = combined_group
            row["quarantine_reasons"] = list(reasons)
            if reasons and row["source_split"] != "sealed_test":
                row["split"] = "quarantine"
                row["included"] = False
            if annotation_conflict:
                row["included"] = False
    retained: dict[tuple[str, str], str] = {}
    image_ids: dict[str, int] = defaultdict(int)
    for row in rows:
        if not row["included"]:
            continue
        key = row["split"], row["content_sha256"]
        if key in retained:
            row["included"] = False
            row["duplicate_of"] = retained[key]
            continue
        retained[key] = row["sample_id"]
        image_ids[row["split"]] += 1
        row["image_id"] = image_ids[row["split"]]
        suffix = PurePosixPath(row["source_export_path"]).suffix.lower()
        row["export_path"] = DIRECTORIES[row["split"]] + "/" + row["content_sha256"] + suffix


def _records(descriptors: list[dict], locks: dict[str, dict]) -> list[dict]:
    rows = [_record(source, row) for source in descriptors
            for row in locks[source["dataset_id"]]["records"]]
    rows.sort(key=lambda row: row["sample_id"])
    require(len({row["sample_id"] for row in rows}) == len(rows), "public_merge_duplicate_sample_id")
    _assign(rows)
    return rows


def _coco(rows: list[dict], role: str) -> dict:
    images, annotations = [], []
    for row in rows:
        if row["split"] != role or not row["included"]:
            continue
        images.append({"id": row["image_id"], "file_name": PurePosixPath(row["export_path"]).name,
                       "width": row["width"], "height": row["height"]})
        for obj in row["objects"]:
            annotations.append({"id": len(annotations) + 1, "image_id": row["image_id"],
                                "category_id": CLASSES.index(obj["class_name"]) + 1,
                                "bbox": obj["bbox"], "area": obj["bbox"][2] * obj["bbox"][3], "iscrowd": 0})
    return {"images": images, "annotations": annotations,
            "categories": [{"id": i + 1, "name": name, "supercategory": "none"}
                           for i, name in enumerate(CLASSES)]}


def _summary(rows: list[dict]) -> dict:
    return {"records": len(rows), "split_counts": dict(Counter(row["split"] for row in rows)),
            "included_split_counts": dict(Counter(row["split"] for row in rows if row["included"])),
            "included_class_split_counts": {name: dict(Counter(
                row["split"] for row in rows if row["included"]
                for obj in row["objects"] if obj["class_name"] == name)) for name in CLASSES},
            "quarantine_reason_counts": dict(Counter(reason for row in rows for reason in row["quarantine_reasons"])),
            "deduplicated_records": sum(row["duplicate_of"] is not None for row in rows)}


def prepare_combined_public_dataset(config_path: Path, output: Path) -> dict:
    config_path = config_path.resolve()
    config = load_json(config_path)
    require(isinstance(config, dict) and config.get("schema_version") == SCHEMA and config.get("frozen") is True,
            "public_merge_config_schema_or_freeze_invalid")
    sources = _descriptors(config.get("source_datasets"), config_path.parent, embedded=False)
    require(not output.exists(), "public_merge_output_already_exists")
    require(all(not output.resolve().is_relative_to(source["root"]) for source in sources),
            "public_merge_output_inside_source_forbidden")
    locks = {source["dataset_id"]: _verify_source(source["root"], source["lock_sha256"], source["dataset_kind"])
             for source in sources}
    rows = _records(sources, locks)
    provenance = _source_metadata(sources, locks)
    output.mkdir(parents=True)
    files: dict[str, str] = {}

    def copy_file(source_path: Path, relative: str, expected: str) -> None:
        _relative(relative)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)
        require(file_sha256(target) == expected, "public_merge_copy_hash_mismatch")
        files[relative] = expected

    copy_file(config_path, "provenance/merge-config.json", file_sha256(config_path))
    for source in sources:
        prefix = "source_datasets/" + source["dataset_id"] + "/"
        inventory = {**locks[source["dataset_id"]]["files"], "dataset-lock.json": source["lock_sha256"]}
        for relative, expected in sorted(inventory.items()):
            copy_file(relative_file(source["root"], _relative(relative)), prefix + relative, expected)
    for row in rows:
        if row["included"]:
            copy_file(relative_file(output, row["source_export_path"]), row["export_path"], row["content_sha256"])
    for role, directory in [*DIRECTORIES.items(), ("unused_test", "test")]:
        path = output / directory / "_annotations.coco.json"
        write_json(path, _coco(rows, role))
        files[path.relative_to(output).as_posix()] = file_sha256(path)
    attribution = output / "ATTRIBUTION.md"
    attribution.write_text(_attribution(provenance), encoding="utf-8")
    files["ATTRIBUTION.md"] = file_sha256(attribution)
    lock = {"schema_version": LOCK_SCHEMA, "state": "frozen", "origin": "public", "dataset_kind": "combined_public",
            "source_config_sha256": file_sha256(config_path), "classes": CLASSES,
            "class_names": {str(i): name for i, name in enumerate(CLASSES)},
            "prediction_class_ids": {name: i for i, name in enumerate(CLASSES)},
            "coco_category_ids": {name: i + 1 for i, name in enumerate(CLASSES)},
            "source_datasets": [{key: source[key] for key in ("dataset_id", "dataset_kind", "root", "lock_sha256")}
                                for source in provenance], "source_provenance": provenance,
            "split_policy": POLICY, "annotation_scope": "source_specific_publisher_classes_only",
            "limitations": [LIMITATION] + [item for source in provenance
                                          for item in source["metadata"].get("limitations", [])],
            "sealed_test_policy": POLICY["sealed_test"], "records": rows, "files": files, "summary": _summary(rows)}
    lock["dataset_sha256"] = canonical_sha256(lock)
    write_json(output / "dataset-lock.json", lock)
    # Verify the portable copies too; the original source roots need not remain.
    verify_combined_public_dataset(output, file_sha256(output / "dataset-lock.json"))
    return {"ok": True, "records": len(rows), "dataset_sha256": lock["dataset_sha256"],
            "dataset_lock_sha256": file_sha256(output / "dataset-lock.json"), "summary": lock["summary"]}


def verify_combined_public_dataset(root: Path, expected_lock_sha256: str) -> dict:
    root = root.resolve()
    lock = pinned_json(root / "dataset-lock.json", expected_lock_sha256)
    require(lock.get("schema_version") == LOCK_SCHEMA and lock.get("state") == "frozen"
            and lock.get("origin") == "public" and lock.get("dataset_kind") == "combined_public",
            "public_merge_lock_invalid")
    require(lock.get("dataset_sha256") == canonical_sha256({key: value for key, value in lock.items()
                                                            if key != "dataset_sha256"}),
            "public_merge_dataset_digest_mismatch")
    require(lock.get("classes") == CLASSES and lock.get("class_names") == {str(i): name for i, name in enumerate(CLASSES)}
            and lock.get("prediction_class_ids") == {name: i for i, name in enumerate(CLASSES)}
            and lock.get("coco_category_ids") == {name: i + 1 for i, name in enumerate(CLASSES)},
            "public_merge_class_mapping_invalid")
    require(lock.get("split_policy") == POLICY and lock.get("sealed_test_policy") == POLICY["sealed_test"]
            and lock.get("annotation_scope") == "source_specific_publisher_classes_only",
            "public_merge_policy_invalid")
    files = lock.get("files")
    require(isinstance(files, dict) and bool(files), "public_merge_inventory_missing")
    for relative, expected in files.items():
        require(sha(expected) and file_sha256(relative_file(root, _relative(relative))) == expected,
                "public_merge_file_changed")
    require(not any(path.is_symlink() for path in root.rglob("*")), "public_merge_symlink_forbidden")
    require({path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
            == set(files) | {"dataset-lock.json"}, "public_merge_unlisted_file")
    require(files.get("provenance/merge-config.json") == lock.get("source_config_sha256"),
            "public_merge_config_inventory_mismatch")
    config = load_json(root / "provenance/merge-config.json")
    require(config.get("schema_version") == SCHEMA and config.get("frozen") is True,
            "public_merge_config_schema_or_freeze_invalid")
    sources = _descriptors(lock.get("source_datasets"), root, embedded=True)
    # Pins in the copied config are part of the immutable chain; source paths
    # can be obsolete after moving the self-contained combined export.
    configured = config.get("source_datasets")
    require(isinstance(configured, list) and len(configured) == 2
            and {(source.get("dataset_id"), source.get("expected_lock_sha256")) for source in configured
                 if isinstance(source, dict)} == {(source["dataset_id"], source["lock_sha256"]) for source in sources},
            "public_merge_config_source_pin_mismatch")
    locks = {source["dataset_id"]: _verify_source(source["root"], source["lock_sha256"], source["dataset_kind"])
             for source in sources}
    provenance = _source_metadata(sources, locks)
    require(lock.get("source_provenance") == provenance, "public_merge_source_metadata_mismatch")
    require(lock.get("limitations") == [LIMITATION] + [item for source in provenance
                for item in source["metadata"].get("limitations", [])], "public_merge_limitations_mismatch")
    require((root / "ATTRIBUTION.md").read_text(encoding="utf-8") == _attribution(provenance),
            "public_merge_attribution_mismatch")
    rows = _records(sources, locks)
    require(lock.get("records") == rows, "public_merge_record_derivation_mismatch")
    require(lock.get("summary") == _summary(rows), "public_merge_summary_mismatch")
    expected_files = {"provenance/merge-config.json", "ATTRIBUTION.md"}
    for source in sources:
        prefix = "source_datasets/" + source["dataset_id"] + "/"
        expected_files |= {prefix + _relative(relative) for relative in locks[source["dataset_id"]]["files"]}
        expected_files.add(prefix + "dataset-lock.json")
    for row in rows:
        if row["included"]:
            expected_files.add(row["export_path"])
            require(files.get(row["export_path"]) == row["content_sha256"], "public_merge_image_inventory_mismatch")
    for role, directory in [*DIRECTORIES.items(), ("unused_test", "test")]:
        relative = directory + "/_annotations.coco.json"
        expected_files.add(relative)
        require(relative in files and load_json(root / relative) == _coco(rows, role),
                "public_merge_coco_derivation_mismatch")
    require(set(files) == expected_files, "public_merge_inventory_derivation_mismatch")
    return lock


# Stable short names for callers that dispatch on dataset_kind.
prepare_combined_dataset = prepare_combined_public_dataset
verify_combined_dataset = verify_combined_public_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare_combined_public_dataset(args.config, args.output)))
        return 0
    except Exception as exc:
        return cli_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
