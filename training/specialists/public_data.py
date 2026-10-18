"""Import a pinned, licensed public YOLO dataset without claiming own-footage review.

Complete denominations are held out before metrics. Publisher validation for
training denominations is excluded; related or ambiguously labelled images are
kept out of training. This does not use the own/synthetic schema.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from urllib.parse import urlparse
import zipfile

from .common import (TrainingGateError, canonical_sha256, cli_error, file_sha256, load_json, number,
                     pinned_json, relative_file, require, sha, write_json)
from .prepare import organizer_hashes

SCHEMA = "forte.specialists.public-data.v1"
LOCK_SCHEMA = "forte.specialists.public-dataset-lock.v1"
PAPER = {"500", "1000", "2000", "5000", "10000", "20000"}
COINS = {"1", "2", "5", "10", "20", "50", "100", "200"}
CLASSES = ["single_banknote"]
DIRECTORIES = {"train": "train", "validation": "valid", "sealed_test": "sealed_test",
               "quarantine": "quarantine"}
DUPLICATE_POLICY = {
    "algorithm": "dhash256_hamming4_transitive_components",
    "cross_final_role": "quarantine_component",
    "same_role_same_kind_cross_denomination": "retain_one_related_component_not_independent_samples",
    "same_content_conflicting_denominations_or_annotations": "quarantine_component",
    "cross_kind": "quarantine_component",
    "rationale": "Frozen before metrics: equal target scope may share a role; exact metadata conflicts and evaluation leakage may not.",
}
LIMITATION = ("Publisher does not provide scene/recording group IDs. Denomination is retained as "
              "template metadata; denomination holdout and image deduplication do not establish "
              "unseen-scene or physical-note independence. Exhaustiveness is limited to the "
              "publisher's YOLO object annotations; other object classes are not labelled.")


def _https(value: object) -> bool:
    return isinstance(value, str) and urlparse(value).scheme == "https" and bool(urlparse(value).netloc)


def _under(root: Path, base: Path, value: object, *, directory: bool = False) -> Path:
    require(isinstance(value, str) and bool(value) and "://" not in value, "public_path_invalid")
    path = (base / value).resolve()
    require(path.is_relative_to(root.resolve()), "public_path_outside_approved_root")
    require(path.is_dir() if directory else path.is_file(), "public_source_missing")
    return path


def _transpose_pixels(image, orientation: int):
    """Apply only the orientation tag; never decode or reserialize GPS IFDs."""
    from PIL import Image
    methods = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
               4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
               6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE, 8: Image.Transpose.ROTATE_90}
    method = methods.get(orientation)
    return image.transpose(method) if method is not None else image


def dhash256(path: Path) -> str:
    """256 horizontal luminance comparisons; no enhancement or inferred detail."""
    from PIL import Image, ImageOps
    try:
        with Image.open(path) as source:
            canonical = _transpose_pixels(source, source.getexif().get(274, 1))
            image = ImageOps.grayscale(canonical).resize((17, 16), Image.Resampling.LANCZOS)
            pixels = image.tobytes()
            bits = 0
            for y in range(16):
                for x in range(16):
                    bits = (bits << 1) | (pixels[y * 17 + x] > pixels[y * 17 + x + 1])
        return f"{bits:064x}"
    except (OSError, ValueError):
        require(False, "public_image_decode_failed")
    raise AssertionError("unreachable")


def _components(rows: list[dict]) -> list[list[int]]:
    """Exact dHash radius-4 components using five pigeonhole index chunks."""
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

    hashes: list[int] = []
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    exact: dict[str, int] = {}
    # Five chunks partition all 256 bits. <=4 differences leave one exact chunk.
    chunks = [(0, 52), (52, 51), (103, 51), (154, 51), (205, 51)]
    for i, row in enumerate(rows):
        value = int(row["dhash256"], 16)
        keys = [(n, (value >> start) & ((1 << size) - 1))
                for n, (start, size) in enumerate(chunks)]
        candidates = {j for key in keys for j in buckets[key]}
        for j in candidates:
            if (value ^ hashes[j]).bit_count() <= 4:
                union(i, j)
        content = row["content_sha256"]
        if content in exact:
            union(i, exact[content])
        exact[content] = i
        hashes.append(value)
        for key in keys:
            buckets[key].append(i)
    result: dict[int, list[int]] = defaultdict(list)
    for i in range(len(rows)):
        result[find(i)].append(i)
    return list(result.values())


def _evidence(config: dict, approved: Path, base: Path) -> dict[str, Path]:
    license_info = config.get("license", {})
    require(isinstance(license_info, dict) and license_info.get("spdx") == "CC-BY-4.0"
            and _https(license_info.get("url")) and isinstance(license_info.get("attribution"), str)
            and bool(license_info["attribution"].strip()), "public_license_evidence_required")
    publication = config.get("annotation_publication", {})
    require(isinstance(publication, dict) and _https(publication.get("url"))
            and publication.get("method") == "manual_labelimg", "public_annotation_evidence_required")
    paths = {}
    for key, descriptor in (("license", license_info), ("annotation_publication", publication)):
        evidence = descriptor.get("evidence", {})
        require(isinstance(evidence, dict) and sha(evidence.get("sha256")), "public_evidence_hash_required")
        path = _under(approved, base, evidence.get("path"))
        require(file_sha256(path) == evidence["sha256"], "public_evidence_hash_mismatch")
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, ValueError):
            require(False, "public_evidence_text_required")
        if key == "license":
            require(bool(re.search(r"CC[\s_-]*BY[\s_-]*4\.0|creativecommons\.org/licenses/by/4\.0", text, re.I)),
                    "public_license_text_missing")
        else:
            require("labelimg" in text.lower(), "public_annotation_method_missing")
        paths[key] = path
    return paths


def _source_class_ids(descriptor: dict) -> list[int]:
    values = descriptor.get("source_class_ids", [descriptor.get("source_class_id")])
    require(isinstance(values, list) and bool(values) and len(values) == len(set(values))
            and all(type(value) is int and value >= 0 for value in values), "public_source_class_mapping_invalid")
    return sorted(values)


def _annotation(path: Path, source_classes: list[int], kind: str, width: int, height: int) -> tuple[list[dict], list[dict], dict]:
    try:
        lines = [line.split() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except (OSError, ValueError):
        require(False, "public_annotation_invalid")
    require(bool(lines), "public_source_annotation_empty")
    source_objects, targets = [], []
    for fields in lines:
        require(len(fields) == 5 and re.fullmatch(r"0|[1-9][0-9]*", fields[0]) is not None,
                "public_yolo_fields_invalid")
        source_class = int(fields[0])
        require(source_class in source_classes, "public_yolo_denomination_class_mismatch")
        try:
            x, y, w, h = [float(value) for value in fields[1:]]
        except ValueError:
            require(False, "public_yolo_coordinates_invalid")
        require(all(number(v, 0, 1) for v in (x, y, w, h)) and w > 0 and h > 0
                and x - w / 2 >= -1e-9 and y - h / 2 >= -1e-9
                and x + w / 2 <= 1 + 1e-9 and y + h / 2 <= 1 + 1e-9,
                "public_yolo_coordinates_invalid")
        normalized = [x, y, w, h]
        source_objects.append({"source_class_id": source_class, "normalized_bbox": normalized})
    # The publication distinguishes an outer currency box from nested security
    # feature boxes but reuses IDs. IDs therefore cannot identify object scope.
    def edges(obj: dict) -> tuple[float, float, float, float]:
        x, y, w, h = obj["normalized_bbox"]
        return x - w / 2, y - h / 2, x + w / 2, y + h / 2

    candidates = []
    for i, obj in enumerate(source_objects):
        left, top, right, bottom = edges(obj)
        if all(left <= a + 1e-9 and top <= b + 1e-9 and right + 1e-9 >= c and bottom + 1e-9 >= d
               for a, b, c, d in (edges(other) for other in source_objects)):
            candidates.append(i)
    ambiguous = len(candidates) != 1
    if not ambiguous and kind == "banknote":
        x, y, w, h = source_objects[candidates[0]]["normalized_bbox"]
        left, top = max(0.0, x - w / 2), max(0.0, y - h / 2)
        targets.append({"class_name": "single_banknote", "bbox": [
            left * width, top * height, min(w, 1 - left) * width, min(h, 1 - top) * height]})
    derivation = {"method": "unique_enclosing_source_box", "source_box_count": len(source_objects),
                  "selected_source_box_index": candidates[0] if not ambiguous else None,
                  "discarded_contained_feature_boxes": len(source_objects) - 1 if not ambiguous else 0,
                  "ambiguous": ambiguous}
    return source_objects, targets, derivation


def _audited_annotation(path: Path, classes: list[int], kind: str, width: int, height: int,
                        error_policy: str) -> tuple[list[dict], list[dict], dict]:
    try:
        return _annotation(path, classes, kind, width, height)
    except TrainingGateError as exc:
        allowed = {"public_yolo_coordinates_invalid", "public_yolo_fields_invalid",
                   "public_source_annotation_empty", "public_annotation_invalid"}
        if error_policy != "quarantine" or str(exc) not in allowed:
            raise
        # The exact original label bytes remain in source_annotations/. An
        # invalid annotation never becomes a supposedly reviewed negative.
        return [], [], {"method": "unique_enclosing_source_box", "source_box_count": 0,
                        "selected_source_box_index": None, "discarded_contained_feature_boxes": 0,
                        "ambiguous": True, "validation_error": str(exc)}


def _member_hash(archive: zipfile.ZipFile, member: zipfile.ZipInfo) -> str:
    digest = hashlib.sha256()
    with archive.open(member) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_source(config: dict, approved: Path, base: Path, banned: set[str]) -> tuple[list[dict], dict]:
    source = config.get("source_archive", {})
    require(isinstance(source, dict) and source.get("dataset_id") == "dny3dgvvw8"
            and source.get("version") == 1 and _https(source.get("url")), "public_dataset_identity_invalid")
    archive_path = _under(approved, base, source.get("path"))
    require(sha(source.get("sha256")) and type(source.get("size_bytes")) is int
            and source["size_bytes"] > 0, "public_archive_pin_required")
    require(archive_path.stat().st_size == source["size_bytes"]
            and file_sha256(archive_path) == source["sha256"], "public_archive_pin_mismatch")
    require(source["sha256"] not in banned, "public_organizer_content_forbidden")
    extracted = _under(approved, base, config.get("extracted_root"), directory=True)
    layout = config.get("source_layout", "Kazakhstan_Banknotes_Dataset")
    require(isinstance(layout, str) and re.fullmatch(r"[A-Za-z0-9_-]+", layout) is not None,
            "public_source_layout_invalid")
    mapping = config.get("denomination_classes")
    require(isinstance(mapping, dict) and set(mapping) == PAPER | COINS, "public_denomination_coverage_invalid")
    for denomination, value in mapping.items():
        require(isinstance(value, dict) and value.get("kind") == (
                    "banknote" if denomination in PAPER else "coin"), "public_source_class_mapping_invalid")
        _source_class_ids(value)
    from PIL import Image
    rows = []
    pattern = re.compile(rf"{re.escape(layout)}/(training|validation)/([0-9]+)_tenge/([0-9]+_[0-9]+)\.(jpg|jpeg|png)", re.I)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            require(len({entry.filename for entry in infos}) == len(infos), "public_archive_duplicate_member")
            for entry in infos:
                name = entry.filename
                require(not name.startswith(("/", "\\")) and "\\" not in name
                        and ":" not in name and ".." not in PurePosixPath(name).parts,
                        "public_archive_member_outside_root")
            by_name = {entry.filename: entry for entry in infos}
            image_names = sorted(name for name in by_name if pattern.fullmatch(name))
            require(bool(image_names), "public_source_images_missing")
            for name in image_names:
                match = pattern.fullmatch(name)
                assert match is not None
                upstream, denomination, stem, _extension = match.groups()
                require(denomination in mapping and stem.split("_")[0] == denomination,
                        "public_source_denomination_invalid")
                label_name = name.rsplit(".", 1)[0] + ".txt"
                require(label_name in by_name, "public_source_annotation_missing")
                image_path = relative_file(extracted, name)
                annotation_path = relative_file(extracted, label_name)
                content = file_sha256(image_path)
                annotation_hash = file_sha256(annotation_path)
                require(content == _member_hash(archive, by_name[name])
                        and annotation_hash == _member_hash(archive, by_name[label_name]),
                        "public_extracted_archive_mismatch")
                require(content not in banned and annotation_hash not in banned, "public_organizer_content_forbidden")
                try:
                    with Image.open(image_path) as image:
                        width, height = image.size
                        exif = image.getexif()
                        orientation = exif.get(274, 1)
                        has_exif, has_gps_tag = bool(exif), 34853 in exif
                        require(type(orientation) is int and 1 <= orientation <= 8, "public_exif_orientation_invalid")
                        require(type(width) is int and type(height) is int and width > 0 and height > 0,
                                "public_dimensions_invalid")
                        image.verify()
                except (OSError, ValueError):
                    require(False, "public_image_decode_failed")
                descriptor = mapping[denomination]
                class_ids = _source_class_ids(descriptor)
                objects, targets, derivation = _audited_annotation(
                    annotation_path, class_ids, descriptor["kind"], width, height, config.get("annotation_error_policy", "reject"))
                rows.append({"sample_id": f"kzbd:{upstream}/{denomination}/{stem}", "upstream_split": upstream,
                             "source_path": name, "source_annotation_path": label_name,
                             "content_sha256": content, "annotation_sha256": annotation_hash,
                             "dhash256": dhash256(image_path), "width": width, "height": height,
                             "source_exif_orientation": orientation,
                             "source_has_exif": has_exif, "source_gps_tag_present": has_gps_tag,
                             "denomination": denomination, "source_class_ids": class_ids,
                             "source_class_id": objects[0]["source_class_id"] if objects else None, "label_derivation": derivation,
                             "kind": descriptor["kind"], "source_objects": objects, "objects": targets,
                             "template_group": "denomination:" + denomination, "scene_group": None,
                             "_image_path": image_path, "_annotation_path": annotation_path})
            # Every publisher image/label under the source split directories must
            # match the declared layout; no silently dropped denomination files.
            relevant = {entry.filename for entry in infos if not entry.is_dir()
                        and entry.filename.startswith((layout + "/training/", layout + "/validation/"))
                        and Path(entry.filename).suffix.lower() in {".jpg", ".jpeg", ".png", ".txt"}}
            expected = set(image_names) | {name.rsplit(".", 1)[0] + ".txt" for name in image_names}
            require(relevant == expected, "public_unmatched_source_file")
    except (zipfile.BadZipFile, RuntimeError, OSError):
        require(False, "public_archive_invalid")
    counts = dict(Counter(row["upstream_split"] for row in rows))
    expected_counts = config.get("expected_upstream_counts")
    require(expected_counts is None or expected_counts == counts, "public_upstream_count_mismatch")
    for upstream in ("training", "validation"):
        require({row["denomination"] for row in rows if row["upstream_split"] == upstream} == PAPER | COINS,
                "public_upstream_denomination_coverage_missing")
    observed_classes = {}
    for denomination in sorted(PAPER | COINS, key=int):
        observed = sorted({obj["source_class_id"] for row in rows if row["denomination"] == denomination
                           for obj in row["source_objects"]})
        require(observed == _source_class_ids(mapping[denomination]), "public_declared_class_inventory_mismatch")
        observed_classes[denomination] = observed
    return rows, {"dataset_id": source["dataset_id"], "version": source["version"], "url": source["url"],
                  "archive_sha256": source["sha256"], "archive_size_bytes": source["size_bytes"],
                  "upstream_counts": counts, "observed_source_class_ids_by_denomination": observed_classes}


def _assign(rows: list[dict], policy: dict) -> None:
    require(isinstance(policy, dict) and policy.get("declared_before_import") is True
            and type(policy.get("seed")) is int and policy.get("near_duplicate_hamming") == 4
            and policy.get("mode") == "denomination_holdout"
            and policy.get("policy_reference") == "20261018/public-group-policy-v1",
            "public_frozen_split_policy_required")
    expected = {"train_denominations": {"500", "1000", "2000", "5000", "1", "2", "5", "10"},
                "validation_denominations": {"10000", "20", "50"},
                "sealed_test_denominations": {"20000", "100", "200"}}
    for field, values in expected.items():
        declared = policy.get(field)
        require(isinstance(declared, list) and len(declared) == len(values) and set(declared) == values,
                "public_frozen_denomination_policy_invalid")
    for row in rows:
        denomination = row["denomination"]
        if denomination in expected["train_denominations"]:
            row["split"] = "train" if row["upstream_split"] == "training" else "quarantine"
        elif denomination in expected["validation_denominations"]:
            row["split"] = "validation"
        else:
            row["split"] = "sealed_test"
        row["quarantine_reasons"] = []
        if row["split"] == "quarantine":
            row["quarantine_reasons"].append("publisher_validation_excluded")
        if row["label_derivation"]["ambiguous"]:
            row["quarantine_reasons"].append("invalid_source_annotation" if row["label_derivation"].get("validation_error")
                                             else "ambiguous_source_annotation")
            row["split"] = "quarantine"
        if row["kind"] == "banknote" and row["source_exif_orientation"] != 1:
            row["quarantine_reasons"].append("paper_exif_coordinate_ambiguity")
            row["split"] = "quarantine"
    for indices in _components(rows):
        members = [rows[i] for i in indices]
        group = "public:" + canonical_sha256(sorted({row["content_sha256"] for row in members}))
        reasons = sorted({reason for row in members for reason in row["quarantine_reasons"]})
        if len({row["split"] for row in members}) > 1:
            reasons.append("cross_final_split_duplicate_component")
        if len({row["kind"] for row in members}) > 1:
            reasons.append("conflicting_source_kinds")
        by_content: dict[str, set[str]] = defaultdict(set)
        content_denominations: dict[str, set[str]] = defaultdict(set)
        for row in members:
            by_content[row["content_sha256"]].add(canonical_sha256(row["source_objects"]))
            content_denominations[row["content_sha256"]].add(row["denomination"])
        if any(len(values) > 1 for values in content_denominations.values()):
            reasons.append("conflicting_exact_content_denominations")
        if any(len(values) > 1 for values in by_content.values()):
            reasons.append("conflicting_exact_content_annotations")
        split = "quarantine" if reasons else members[0]["split"]
        for row in members:
            row.update(source_group=group, split=split, quarantine_reasons=reasons)
    retained: dict[tuple[str, str], str] = {}
    for row in rows:
        key = row["split"], row["content_sha256"]
        row["included"] = row["split"] != "quarantine" and key not in retained
        row["duplicate_of"] = retained.get(key)
        retained.setdefault(key, row["sample_id"])
    for split in ("train", "validation", "sealed_test"):
        require(any(row["split"] == split and row["included"] and row["objects"] for row in rows),
                "public_positive_partition_coverage_missing")
    require(any(row["split"] == "train" and row["included"] and row["kind"] == "coin" for row in rows),
            "public_training_coin_background_missing")


def _copy_prepared_image(row: dict, part: Path, max_side: int | None) -> tuple[str, str]:
    """Bound disk and detector input size; retain original hashes and label boxes."""
    source = row["_image_path"]
    row["source_content_sha256"] = row["content_sha256"]
    row["source_width"], row["source_height"] = row["width"], row["height"]
    row["source_target_objects"] = row["objects"]
    transpose = row["kind"] == "coin" and row["source_exif_orientation"] != 1
    oriented_width, oriented_height = row["width"], row["height"]
    if transpose and row["source_exif_orientation"] in {5, 6, 7, 8}:
        oriented_width, oriented_height = oriented_height, oriented_width
    resize = max_side is not None and max(oriented_width, oriented_height) > max_side
    strip_metadata = row.get("source_has_exif", False)
    if resize or transpose or strip_metadata:
        from PIL import Image
        factor = min(1.0, max_side / max(oriented_width, oriented_height)) if max_side is not None else 1.0
        width, height = max(1, round(oriented_width * factor)), max(1, round(oriented_height * factor))
        # Each content key is processed at most once per final role.
        temporary = part / (row["content_sha256"] + ".prepared.tmp")
        with Image.open(source) as image:
            if transpose:
                image = _transpose_pixels(image, row["source_exif_orientation"])
            image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS).save(
                temporary, format="JPEG", quality=90, optimize=False, progressive=False)
        prepared_hash = file_sha256(temporary)
        name = prepared_hash + ".jpg"
        temporary.replace(part / name)
        scale_x, scale_y = width / oriented_width, height / oriented_height
        row["objects"] = [{"class_name": obj["class_name"], "bbox": [
            obj["bbox"][0] * scale_x, obj["bbox"][1] * scale_y,
            obj["bbox"][2] * scale_x, obj["bbox"][3] * scale_y]} for obj in row["objects"]]
        row["width"], row["height"] = width, height
        row["content_sha256"] = prepared_hash
        method = "exif_transpose_lanczos_jpeg" if transpose else ("lanczos_downscale_jpeg" if resize else "metadata_strip_jpeg")
        row["image_derivation"] = {"method": method,
                                   "max_side": max_side, "quality": 90, "scale_x": scale_x, "scale_y": scale_y,
                                   "oriented_width": oriented_width, "oriented_height": oriented_height,
                                   "exif_transposed": transpose, "metadata_stripped": True}
    else:
        name = row["content_sha256"] + source.suffix.lower()
        shutil.copyfile(source, part / name)
        row["image_derivation"] = {"method": "byte_copy"}
    require(file_sha256(part / name) == row["content_sha256"], "public_copy_hash_mismatch")
    return name, row["content_sha256"]


def prepare_public_dataset(config_path: Path, output: Path) -> dict:
    config_path = config_path.resolve()
    config = load_json(config_path)
    require(isinstance(config, dict) and config.get("schema_version") == SCHEMA
            and config.get("frozen") is True, "public_config_schema_or_freeze_invalid")
    base = config_path.parent
    require(isinstance(config.get("approved_root"), str) and bool(config["approved_root"]), "public_approved_root_required")
    approved = (base / config["approved_root"]).resolve()
    require(approved.is_dir() and config_path.is_relative_to(approved), "public_approved_root_invalid")
    evidence = _evidence(config, approved, base)
    extra_evidence = config.get("source_annotation_audit")
    if extra_evidence is not None:
        require(isinstance(extra_evidence, dict) and sha(extra_evidence.get("sha256")), "public_annotation_audit_pin_required")
        audit_path = _under(approved, base, extra_evidence.get("path"))
        require(file_sha256(audit_path) == extra_evidence["sha256"], "public_annotation_audit_pin_mismatch")
        evidence["source_annotation_audit"] = audit_path
    inventory = config.get("organizer_inventory", {})
    require(isinstance(inventory, dict), "public_organizer_inventory_required")
    inventory_path = _under(approved, base, inventory.get("path"))
    banned = organizer_hashes(inventory_path, inventory.get("sha256", ""))
    annotation_error_policy = config.get("annotation_error_policy", "reject")
    require(annotation_error_policy in {"reject", "quarantine"}, "public_annotation_error_policy_invalid")
    rows, source = _read_source(config, approved, base, banned)
    policy = config.get("split_policy", {})
    _assign(rows, policy)
    resize_policy = dict(config.get("image_preparation", {"max_side": 800}))
    resize_policy.setdefault("coin_exif", "transpose")
    require(isinstance(resize_policy, dict) and type(resize_policy.get("max_side")) is int
            and 64 <= resize_policy["max_side"] <= 800 and resize_policy["coin_exif"] == "transpose", "public_image_preparation_invalid")
    require(not output.exists(), "public_output_already_exists")
    require(not output.resolve().is_relative_to((base / config["extracted_root"]).resolve()),
            "public_output_inside_source_forbidden")
    output.mkdir(parents=True)
    files: dict[str, str] = {}
    provenance_paths = {"config": config_path, "organizer_inventory": inventory_path, **evidence}
    for name, path in provenance_paths.items():
        target = output / "provenance" / (name + path.suffix.lower())
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, target)
        files[target.relative_to(output).as_posix()] = file_sha256(target)
    categories = [{"id": 1, "name": "single_banknote", "supercategory": "none"}]
    for role, directory in [*DIRECTORIES.items(), ("unused_test", "test")]:
        part = output / directory
        part.mkdir()
        images, annotations = [], []
        image_ids: dict[str, int] = {}
        prepared_by_source: dict[str, dict] = {}
        for row in (row for row in rows if row["split"] == role):
            source_hash = row["content_sha256"]
            if source_hash in prepared_by_source:
                original = prepared_by_source[source_hash]
                row["source_target_objects"] = row["objects"]
                for field in ("source_content_sha256", "source_width", "source_height",
                              "width", "height", "content_sha256", "image_derivation"):
                    row[field] = original[field]
                if row["image_derivation"]["method"] != "byte_copy":
                    sx, sy = row["image_derivation"]["scale_x"], row["image_derivation"]["scale_y"]
                    row["objects"] = [{"class_name": obj["class_name"], "bbox": [
                        obj["bbox"][0] * sx, obj["bbox"][1] * sy, obj["bbox"][2] * sx, obj["bbox"][3] * sy]}
                        for obj in row["source_target_objects"]]
                name = Path(original["export_path"]).name
            else:
                name, _digest = _copy_prepared_image(row, part, resize_policy["max_side"])
            relative = directory + "/" + name
            if relative not in files:
                files[relative] = row["content_sha256"]
                image_id = len(images) + 1
                image_ids[relative] = image_id
                images.append({"id": image_id, "file_name": name, "width": row["width"], "height": row["height"]})
                for obj in row["objects"]:
                    annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": 1,
                                        "bbox": obj["bbox"], "area": obj["bbox"][2] * obj["bbox"][3], "iscrowd": 0})
            row["export_path"], row["image_id"] = relative, image_ids[relative]
            prepared_by_source[source_hash] = row
            annotation_relative = "source_annotations/" + row["annotation_sha256"] + ".txt"
            if annotation_relative not in files:
                destination = output / annotation_relative
                destination.parent.mkdir(exist_ok=True)
                shutil.copyfile(row["_annotation_path"], destination)
                files[annotation_relative] = row["annotation_sha256"]
            row["annotation_export_path"] = annotation_relative
        annotation_path = part / "_annotations.coco.json"
        write_json(annotation_path, {"images": images, "annotations": annotations, "categories": categories})
        files[annotation_path.relative_to(output).as_posix()] = file_sha256(annotation_path)
    exported = [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]
    summary = {"records": len(rows), "split_counts": dict(Counter(row["split"] for row in rows)),
               "included_split_counts": dict(Counter(row["split"] for row in rows if row["included"])),
               "coin_negative_records": sum(row["kind"] == "coin" and row["included"] for row in rows),
               "quarantine_reason_counts": dict(Counter(reason for row in rows for reason in row["quarantine_reasons"])),
               "deduplicated_records": sum(row["duplicate_of"] is not None for row in rows),
               "source_annotation_boxes": sum(len(row["source_objects"]) for row in rows),
               "discarded_contained_feature_boxes": sum(row["label_derivation"]["discarded_contained_feature_boxes"] for row in rows),
               "ambiguous_annotation_records": sum(row["label_derivation"]["ambiguous"] for row in rows),
               "invalid_annotation_records": sum(bool(row["label_derivation"].get("validation_error")) for row in rows),
               "source_exif_orientation_counts": dict(Counter(str(row["source_exif_orientation"]) for row in rows)),
               "coin_exif_transposed_records": sum(row["image_derivation"].get("exif_transposed", False) for row in rows),
               "selected_banknote_boxes": sum(len(row["objects"]) for row in rows if row["included"]),
               "organizer_inventory_sha256": inventory["sha256"]}
    lock = {"schema_version": LOCK_SCHEMA, "state": "frozen", "origin": "public", "dataset_kind": "kz_bd_v1",
            "source_config_sha256": file_sha256(config_path), "classes": CLASSES,
            "class_names": {"0": "single_banknote"}, "coco_category_ids": {"single_banknote": 1},
            "prediction_class_ids": {"single_banknote": 0}, "source_provenance": source,
            "license": config["license"], "annotation_publication": config["annotation_publication"],
            "denomination_classes": config["denomination_classes"], "split_policy": policy,
            "duplicate_policy": DUPLICATE_POLICY,
            "image_preparation": resize_policy,
            "annotation_error_policy": annotation_error_policy,
            "source_annotation_audit": extra_evidence,
            "annotation_scope": "publisher_yolo_objects_only", "scene_group_status": "publisher_unavailable",
            "limitations": [LIMITATION], "sealed_test_policy": "entire_predeclared_denominations_never_used_for_training_or_tuning",
            "records": exported, "files": files, "summary": summary}
    lock["dataset_sha256"] = canonical_sha256(lock)
    write_json(output / "dataset-lock.json", lock)
    return {"ok": True, "records": len(rows), "dataset_sha256": lock["dataset_sha256"],
            "dataset_lock_sha256": file_sha256(output / "dataset-lock.json"), "split_counts": summary["split_counts"]}


def verify_public_dataset(root: Path, expected_lock_sha256: str) -> dict:
    """Verify a public export independently of the own/synthetic dataset verifier."""
    lock = pinned_json(root / "dataset-lock.json", expected_lock_sha256)
    require(lock.get("schema_version") == LOCK_SCHEMA and lock.get("state") == "frozen"
            and lock.get("origin") == "public", "public_dataset_lock_invalid")
    require(lock.get("dataset_sha256") == canonical_sha256({key: value for key, value in lock.items()
                                                            if key != "dataset_sha256"}), "public_dataset_digest_mismatch")
    require(lock.get("classes") == CLASSES and lock.get("class_names") == {"0": "single_banknote"}
            and lock.get("coco_category_ids") == {"single_banknote": 1}
            and lock.get("prediction_class_ids") == {"single_banknote": 0}, "public_class_mapping_invalid")
    require(lock.get("duplicate_policy") == DUPLICATE_POLICY, "public_duplicate_policy_invalid")
    files = lock.get("files")
    require(isinstance(files, dict) and bool(files) and all(sha(value) for value in files.values()),
            "public_dataset_inventory_missing")
    for relative, expected in files.items():
        require(file_sha256(relative_file(root, relative)) == expected, "public_dataset_file_changed")
    require({path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
            == set(files) | {"dataset-lock.json"}, "public_dataset_unlisted_file")
    for key in ("license", "annotation_publication"):
        descriptor = lock.get(key, {})
        prefix = "provenance/" + key + "."
        evidence_files = [name for name in files if name.startswith(prefix)]
        require(isinstance(descriptor, dict) and len(evidence_files) == 1
                and files[evidence_files[0]] == descriptor.get("evidence", {}).get("sha256")
                and _https(descriptor.get("url")), "public_evidence_lock_mismatch")
        evidence_text = (root / evidence_files[0]).read_text(encoding="utf-8-sig")
        if key == "license":
            require(descriptor.get("spdx") == "CC-BY-4.0" and bool(descriptor.get("attribution"))
                    and bool(re.search(r"CC[\s_-]*BY[\s_-]*4\.0|creativecommons\.org/licenses/by/4\.0", evidence_text, re.I)),
                    "public_license_evidence_required")
        else:
            require(descriptor.get("method") == "manual_labelimg" and "labelimg" in evidence_text.lower(),
                    "public_annotation_evidence_required")
    configs = [name for name in files if name.startswith("provenance/config.")]
    inventories = [name for name in files if name.startswith("provenance/organizer_inventory.")]
    require(len(configs) == 1 and files[configs[0]] == lock.get("source_config_sha256")
            and len(inventories) == 1 and files[inventories[0]] == lock.get("summary", {}).get("organizer_inventory_sha256"),
            "public_provenance_inventory_mismatch")
    banned = organizer_hashes(root / inventories[0], files[inventories[0]])
    rows = lock.get("records")
    require(isinstance(rows, list) and bool(rows), "public_dataset_records_missing")
    groups: dict[str, set[str]] = defaultdict(set)
    contents: dict[str, set[str]] = defaultdict(set)
    samples = set()
    retained: set[tuple[str, str]] = set()
    indexed = {row.get("sample_id"): row for row in rows if isinstance(row, dict)}
    policy = lock.get("split_policy", {})
    require(policy.get("mode") == "denomination_holdout" and policy.get("declared_before_import") is True
            and policy.get("near_duplicate_hamming") == 4 and type(policy.get("seed")) is int
            and policy.get("policy_reference") == "20261018/public-group-policy-v1", "public_frozen_split_policy_required")
    policy_sets = {"train": {"500", "1000", "2000", "5000", "1", "2", "5", "10"},
                   "validation": {"10000", "20", "50"}, "sealed_test": {"20000", "100", "200"}}
    for role, denominations in policy_sets.items():
        declared = policy.get(role + "_denominations")
        require(isinstance(declared, list) and len(declared) == len(denominations)
                and set(declared) == denominations, "public_frozen_denomination_policy_invalid")
    for row in rows:
        require(isinstance(row, dict) and row.get("sample_id") not in samples
                and isinstance(row.get("sample_id"), str), "public_record_invalid")
        samples.add(row["sample_id"])
        role = row.get("split")
        require(role in DIRECTORIES and row.get("upstream_split") in {"training", "validation"}, "public_record_split_invalid")
        denomination = row.get("denomination")
        require(denomination in PAPER | COINS and (role == "quarantine" or denomination in policy_sets[role])
                and (role != "train" or row["upstream_split"] == "training"), "public_sealed_split_violation")
        content, annotation = row.get("content_sha256"), row.get("annotation_sha256")
        require(sha(content) and sha(annotation) and sha(row.get("dhash256"))
                and isinstance(row.get("source_group"), str) and row["source_group"].startswith("public:"), "public_record_hash_invalid")
        groups[row["source_group"]].add(role)
        contents[content].add(role)
        require(sha(row.get("source_content_sha256")), "public_source_image_hash_missing")
        require(content not in banned and row["source_content_sha256"] not in banned and annotation not in banned,
                "public_organizer_content_forbidden")
        contents[row["source_content_sha256"]].add(role)
        export, label = row.get("export_path"), row.get("annotation_export_path")
        require(isinstance(export, str) and export.startswith(DIRECTORIES[role] + "/")
                and files.get(export) == content and files.get(label) == annotation, "public_record_inventory_mismatch")
        require(row.get("scene_group") is None and row.get("template_group") == "denomination:" + str(row.get("denomination")),
                "public_group_metadata_invalid")
        require(row.get("kind") in {"banknote", "coin"} and isinstance(row.get("objects"), list)
                and (row["kind"] != "coin" or not row["objects"]), "public_coin_reinterpreted_as_banknote")
        require(all(obj.get("class_name") == "single_banknote" for obj in row["objects"]), "public_target_class_invalid")
        require(type(row.get("included")) is bool and (role != "quarantine" or not row["included"]), "public_quarantine_included")
        key = role, content
        if row["included"]:
            require(key not in retained and row.get("duplicate_of") is None, "public_exact_duplicate_included")
            retained.add(key)
        duplicate = row.get("duplicate_of")
        if duplicate is not None:
            parent = indexed.get(duplicate)
            require(isinstance(parent, dict) and parent["sample_id"] != row["sample_id"]
                    and parent.get("split") == role and parent.get("content_sha256") == content
                    and row["included"] is False, "public_duplicate_reference_invalid")
        descriptor = lock.get("denomination_classes", {}).get(denomination, {})
        source_objects, target_objects, derivation = _audited_annotation(
            relative_file(root, label), _source_class_ids(descriptor), row["kind"], row["source_width"], row["source_height"],
            lock.get("annotation_error_policy", "reject"))
        require(source_objects == row.get("source_objects") and target_objects == row.get("source_target_objects")
                and derivation == row.get("label_derivation"), "public_label_derivation_mismatch")
        require(not derivation["ambiguous"] or role == "quarantine", "public_ambiguous_annotation_included")
        orientation = row.get("source_exif_orientation")
        require(type(orientation) is int and 1 <= orientation <= 8
                and (row["kind"] != "banknote" or orientation == 1 or role == "quarantine"),
                "public_paper_exif_coordinate_ambiguity")
        image_derivation = row.get("image_derivation", {})
        if image_derivation.get("method") == "byte_copy":
            require(content == row["source_content_sha256"]
                    and (row["width"], row["height"]) == (row["source_width"], row["source_height"])
                    and row["objects"] == target_objects, "public_image_derivation_mismatch")
        else:
            require(image_derivation.get("method") in {"lanczos_downscale_jpeg", "exif_transpose_lanczos_jpeg", "metadata_strip_jpeg"}
                    and max(row["width"], row["height"]) <= lock.get("image_preparation", {}).get("max_side", 0),
                    "public_image_derivation_mismatch")
            transpose = image_derivation.get("exif_transposed") is True
            require(not transpose or row["kind"] == "coin", "public_paper_exif_transform_forbidden")
            ow, oh = row["source_width"], row["source_height"]
            if transpose and orientation in {5, 6, 7, 8}:
                ow, oh = oh, ow
            require(image_derivation.get("oriented_width") == ow and image_derivation.get("oriented_height") == oh,
                    "public_image_derivation_mismatch")
            sx, sy = row["width"] / ow, row["height"] / oh
            expected_objects = [{"class_name": obj["class_name"], "bbox": [obj["bbox"][0] * sx, obj["bbox"][1] * sy,
                                obj["bbox"][2] * sx, obj["bbox"][3] * sy]} for obj in target_objects]
            require(row["objects"] == expected_objects and image_derivation.get("scale_x") == sx
                    and image_derivation.get("scale_y") == sy, "public_image_derivation_mismatch")
        from PIL import Image
        try:
            with Image.open(relative_file(root, export)) as image:
                require(image.size == (row["width"], row["height"]), "public_dimensions_mismatch")
                require(not image.getexif(), "public_export_metadata_not_stripped")
                image.verify()
        except (OSError, ValueError):
            require(False, "public_image_decode_failed")
    require(all(len(roles) == 1 for roles in groups.values()) and all(len(roles) == 1 for roles in contents.values()),
            "public_partition_leakage")
    for indices in _components(rows):
        members = [rows[i] for i in indices]
        expected_group = "public:" + canonical_sha256(sorted({row["source_content_sha256"] for row in members}))
        require({row["source_group"] for row in members} == {expected_group}
                and len({row["split"] for row in members}) == 1, "public_duplicate_component_leakage")
    for role, directory in [*DIRECTORIES.items(), ("unused_test", "test")]:
        path = directory + "/_annotations.coco.json"
        require(path in files, "public_coco_inventory_missing")
        coco = load_json(root / path)
        require(coco.get("categories") == [{"id": 1, "name": "single_banknote", "supercategory": "none"}]
                and isinstance(coco.get("images"), list) and isinstance(coco.get("annotations"), list), "public_coco_invalid")
        expected = {}
        for row in rows:
            if row["split"] == role:
                expected.setdefault(row["export_path"], row)
        require(len(coco["images"]) == len(expected), "public_coco_image_count_mismatch")
        for image in coco["images"]:
            row = expected.get(directory + "/" + image.get("file_name", ""))
            require(row is not None and image.get("id") == row["image_id"]
                    and (image.get("width"), image.get("height")) == (row["width"], row["height"]), "public_coco_record_mismatch")
        expected_annotations = []
        for image in coco["images"]:
            row = expected[directory + "/" + image["file_name"]]
            expected_annotations.extend({"id": len(expected_annotations) + i + 1, "image_id": image["id"], "category_id": 1,
                                         "bbox": obj["bbox"], "area": obj["bbox"][2] * obj["bbox"][3], "iscrowd": 0}
                                        for i, obj in enumerate(row["objects"]))
        require(coco["annotations"] == expected_annotations, "public_coco_annotation_mismatch")
        if role == "unused_test":
            require(not coco["images"] and not coco["annotations"], "public_test_placeholder_not_empty")
    return lock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare_public_dataset(args.config, args.output)))
        return 0
    except Exception as exc:
        return cli_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
