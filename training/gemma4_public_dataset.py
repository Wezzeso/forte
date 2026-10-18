"""Prepare a deterministic, public-only crop manifest without decoding images.

Only publisher object labels are supervised.  This module never reads Forte's
video/label inventory, never contacts a model service, and never opens an image
decoder.  The on-host trainer/evaluator crops the authenticated original image
at runtime.  Existing public validation is reused development data, not a fresh
lockbox or evidence of banking-video performance.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tarfile
from typing import Any


SCHEMA = "forte.gemma4.public-crop-manifest.v1"
QUESTION = (
    "Classify the visible object crop by its visual appearance. Return exactly "
    'one JSON object with one field: {"object_class":"id_card"}, '
    '{"object_class":"banknote"}, {"object_class":"other"}, or '
    '{"object_class":"unclear"}. Do not read text, identify people, estimate '
    "amounts, or explain the answer."
)
SOURCE_CLASSES = {"id_card": "id_card", "single_banknote": "banknote"}
SOURCE_LICENSES = {"MIDV-Holo": "CC-BY-SA-2.5", "dny3dgvvw8": "CC-BY-4.0"}
GROUP_FIELDS = (
    "combined_group", "source_group", "template_group",
    "physical_document_group", "scene_group",
)
TRAIN_VIEWS = [
    {"rotation_degrees": angle, "short_side": side, "brightness": 1.0}
    for side in (0, 32, 64, 128, 256) for angle in (-10, 0, 10)
]
VALIDATION_VIEWS = [
    {"rotation_degrees": 0, "short_side": side, "brightness": 1.0}
    for side in (0, 32, 64, 128)
]


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_file(root: Path, name: str) -> Path:
    require(isinstance(name, str) and bool(name), "missing_relative_path")
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts
            and ":" not in name and "\\" not in name, "unsafe_relative_path")
    result = (root / relative).resolve()
    require(result.is_relative_to(root.resolve()) and result.is_file(),
            "source_file_missing_or_outside_public_root")
    return result


def load_lock(root: Path) -> tuple[dict, str]:
    path = root / "dataset-lock.json"
    lock = json.loads(path.read_text(encoding="utf-8"))
    require(lock.get("schema_version") == "forte.specialists.public-dataset-lock.v1"
            and lock.get("origin") == "public" and lock.get("state") == "frozen"
            and lock.get("dataset_kind") == "combined_public", "frozen_public_lock_required")
    require(lock.get("dataset_sha256") == canonical_sha256(
        {key: value for key, value in lock.items() if key != "dataset_sha256"}),
        "public_lock_canonical_digest_mismatch")
    require(lock.get("classes") == ["single_banknote", "id_card"],
            "publisher_class_mapping_mismatch")
    provenance = lock.get("source_provenance", [])
    require({source.get("dataset_id") for source in provenance} == set(SOURCE_LICENSES),
            "unexpected_public_sources")
    for source in provenance:
        require(source.get("metadata", {}).get("license", {}).get("spdx")
                == SOURCE_LICENSES[source["dataset_id"]], "source_license_mismatch")

    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    hashes: dict[str, set[str]] = defaultdict(set)
    identifiers = set()
    for row in lock.get("records", []):
        require(row.get("dataset_id") in SOURCE_LICENSES
                and row.get("sample_id") not in identifiers,
                "public_record_identity_invalid")
        identifiers.add(row["sample_id"])
        if not row.get("included"):
            continue
        require(row.get("split") in {"train", "validation", "sealed_test"},
                "included_quarantine_or_unknown_split")
        require(row.get("source_group") and row.get("template_group"),
                "publisher_source_groups_required")
        require(lock.get("files", {}).get(row.get("export_path"))
                == row.get("content_sha256"), "public_inventory_hash_mismatch")
        for key in GROUP_FIELDS:
            if row.get(key):
                groups[(key, row[key])].add(row["split"])
        for key in ("content_sha256", "source_content_sha256"):
            if row.get(key):
                hashes[row[key]].add(row["split"])
    require(identifiers and all(len(value) == 1 for value in groups.values())
            and all(len(value) == 1 for value in hashes.values()),
            "public_source_or_content_group_leakage")
    return lock, file_sha256(path)


def publisher_crops(lock: dict, split: str, seed: int) -> dict[str, list[dict]]:
    require(split in {"train", "validation"}, "sealed_test_is_not_accessible")
    buckets: dict[str, list[dict]] = {name: [] for name in SOURCE_CLASSES.values()}
    for record in lock["records"]:
        if not record.get("included") or record.get("split") != split:
            continue
        width, height = record.get("width"), record.get("height")
        require(type(width) is int and type(height) is int and width > 0 and height > 0,
                "publisher_dimensions_invalid")
        for index, annotation in enumerate(record.get("objects", [])):
            name, box = annotation.get("class_name"), annotation.get("bbox")
            require(name in SOURCE_CLASSES and isinstance(box, list) and len(box) == 4
                    and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                            and math.isfinite(v) for v in box), "publisher_box_invalid")
            x, y, box_width, box_height = box
            require(x >= 0 and y >= 0 and box_width > 0 and box_height > 0
                    and x + box_width <= width + .01 and y + box_height <= height + .01,
                    "publisher_box_outside_original")
            identity = record["sample_id"] + f":object:{index}"
            # All paths, class names and lineage are metadata, never model inputs.
            crop = {
                "source_object_id": hashlib.sha256(identity.encode()).hexdigest(),
                "source_dataset": record["dataset_id"],
                "source_group": record["source_group"],
                "source_image_group": record["content_sha256"],
                "template_group": record["template_group"],
                "physical_document_group": record.get("physical_document_group"),
                "scene_group": record.get("scene_group"),
                "source_export_path": record["export_path"],
                "image_path": "images/" + record["content_sha256"] + ".jpg",
                "image_sha256": record["content_sha256"],
                "crop_box_xyxy": [math.floor(x), math.floor(y),
                                  min(width, math.ceil(x + box_width)),
                                  min(height, math.ceil(y + box_height))],
                "source_image_size": [width, height],
                "prompt": QUESTION,
                "target": {"object_class": SOURCE_CLASSES[name]},
                "split": split,
                "license_spdx": SOURCE_LICENSES[record["dataset_id"]],
                "label_source": "published_object_class_and_box",
            }
            buckets[SOURCE_CLASSES[name]].append(crop)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: hashlib.sha256(
            f"{seed}:{row['source_object_id']}".encode()).hexdigest())
    require(all(buckets.values()), "both_public_classes_required")
    return buckets


def with_view(crop: dict, view: dict) -> dict:
    result = dict(crop)
    result["transform"] = dict(view)
    result["example_id"] = canonical_sha256(
        {"source_object_id": crop["source_object_id"], "transform": view})
    return result


def select_rows(lock: dict, seed: int) -> tuple[list[dict], list[dict]]:
    train_buckets = publisher_crops(lock, "train", seed)
    # Each of the scarce ID crops gets the same fixed geometry curriculum.
    # Banknotes use more unique originals with exactly matched view counts.
    id_crops = train_buckets["id_card"]
    banknotes = train_buckets["banknote"]
    banknote_count = min(len(banknotes), len(id_crops) * len(TRAIN_VIEWS))
    banknote_count -= banknote_count % len(TRAIN_VIEWS)
    require(banknote_count >= len(TRAIN_VIEWS), "too_few_banknote_crops")
    id_count = banknote_count // len(TRAIN_VIEWS)
    train = [with_view(crop, view) for crop in id_crops[:id_count] for view in TRAIN_VIEWS]
    train += [with_view(crop, TRAIN_VIEWS[index % len(TRAIN_VIEWS)])
              for index, crop in enumerate(banknotes[:banknote_count])]
    train.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['example_id']}".encode()).hexdigest())

    validation_buckets = publisher_crops(lock, "validation", seed)
    validation = [with_view(crop, view) for name, count in (("id_card", 16), ("banknote", 32))
                  for crop in validation_buckets[name][:count] for view in VALIDATION_VIEWS]
    validation.sort(key=lambda row: row["example_id"])
    require(not ({row["example_id"] for row in train}
                 & {row["example_id"] for row in validation}), "example_split_overlap")
    for key in GROUP_FIELDS + ("source_image_group", "source_object_id"):
        require(not ({row.get(key) for row in train if row.get(key)}
                     & {row.get(key) for row in validation if row.get(key)}),
                "selected_public_group_overlap:" + key)
    return train, validation


def row_summary(rows: list[dict]) -> dict:
    return {
        "examples": len(rows),
        "unique_object_crops": len({row["source_object_id"] for row in rows}),
        "unique_original_images": len({row["image_sha256"] for row in rows}),
        "target_counts": dict(sorted(Counter(row["target"]["object_class"] for row in rows).items())),
        "source_counts": dict(sorted(Counter(row["source_dataset"] for row in rows).items())),
        "unique_objects_by_class": {name: len({row["source_object_id"] for row in rows
                                               if row["target"]["object_class"] == name})
                                    for name in SOURCE_CLASSES.values()},
        "group_counts": {key: len({row.get(key) for row in rows if row.get(key)})
                         for key in GROUP_FIELDS if key != "combined_group"},
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8")


def prepare(root: Path, output: Path, seed: int, bundle: Path | None = None) -> dict:
    root, output = root.resolve(), output.resolve()
    require(not output.exists(), "output_must_be_fresh_for_immutable_manifest")
    require(not output.is_relative_to(root), "output_must_not_modify_public_source")
    lock, lock_hash = load_lock(root)
    train, validation = select_rows(lock, seed)
    selected = train + validation
    # Verify exactly the selected public originals, with no image decoding.
    originals: dict[str, Path] = {}
    for row in selected:
        if row["image_path"] not in originals:
            source = relative_file(root, row["source_export_path"])
            require(file_sha256(source) == row["image_sha256"], "selected_public_image_hash_mismatch")
            originals[row["image_path"]] = source
    attribution = relative_file(root, "ATTRIBUTION.md")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    for relative, original in sorted(originals.items()):
        destination = output / relative
        try:
            os.link(original, destination)
        except OSError:
            shutil.copyfile(original, destination)
    shutil.copyfile(attribution, output / "ATTRIBUTION.md")

    manifests = {}
    for name, rows in (("train", train), ("validation", validation)):
        path = output / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=False) + "\n")
        manifests[path.name] = file_sha256(path)

    provenance = {
        "schema_version": SCHEMA,
        "origin": "public",
        "dataset_lock_sha256": lock_hash,
        "dataset_content_sha256": lock["dataset_sha256"],
        "sources": lock["source_provenance"],
        "split_policy": lock["split_policy"],
        "preserved_source_splits": True,
        "excluded_splits": ["sealed_test", "quarantine"],
        "validation_exposure": "reused_public_development_split_exposed_in_prior_experiments",
        "supplied_bank_videos_or_labels_used": False,
        "frame_or_scene_description_targets": False,
        "face_or_identity_or_ocr_targets": False,
        "new_derivation": "publisher_box_native_crop_on_host_then_declared_geometry_and_downsampling",
        "runtime_transform_order": ["native_crop", "rotate_expand", "short_side_downsample_only", "brightness"],
        "model_inputs": ["runtime_image_crop", "fixed_generic_prompt"],
        "model_excluded_inputs": ["image_path", "example_id", "source_dataset", "all_lineage_metadata"],
        "closed_output_enum": ["id_card", "banknote", "other", "unclear"],
        "supervised_output_enum": ["id_card", "banknote"],
        "unsupervised_output_limit": "other/unclear have no fabricated supervision or measured negative-room specificity",
        "seed": seed,
        "train_views": TRAIN_VIEWS,
        "validation_views": VALIDATION_VIEWS,
        "selection_parameters": {"train_banknote_max": 540, "train_id_max": 36,
                                 "validation_banknote_max": 32, "validation_id_max": 16},
        "selection_tuned_to_any_model_metric": False,
        "license_notice": "Original dataset terms/attribution retained; adapter release rights require separate review.",
        "images": {name: {"sha256": name.split("/")[-1].split(".")[0],
                           "bytes": path.stat().st_size} for name, path in sorted(originals.items())},
    }
    write_json(output / "provenance.json", provenance)
    receipt = {
        "schema_version": SCHEMA,
        "state": "prepared_public_only_untrained",
        "dataset_lock_sha256": lock_hash,
        "manifest_sha256": manifests,
        "provenance_sha256": file_sha256(output / "provenance.json"),
        "attribution_sha256": file_sha256(output / "ATTRIBUTION.md"),
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "seed": seed,
        "train": row_summary(train),
        "validation": row_summary(validation),
        "selected_original_bytes": sum(path.stat().st_size for path in originals.values()),
        "image_decoding_performed": False,
        "sealed_images_read_or_bundled": False,
        "organizer_media_or_labels_read_or_bundled": False,
        "limitations": [
            "Repeated geometry views are not independent examples.",
            "Only 36 public ID training crops; limited scene/template diversity.",
            "Existing public validation has prior experimental exposure and is not a fresh lockbox.",
            "Positive object crops do not prove omission detection or banking-video verdict accuracy.",
            "No benchmark score or false-alarm claim follows from this preparation.",
        ],
    }
    write_json(output / "receipt.json", receipt)
    if bundle is not None:
        bundle = bundle.resolve()
        require(not bundle.exists() and not bundle.is_relative_to(output), "fresh_external_bundle_required")
        # Stable tar headers make the same inputs produce the same artifact.
        with tarfile.open(bundle, "w") as archive:
            for path in sorted(output.rglob("*")):
                if not path.is_file():
                    continue
                info = archive.gettarinfo(str(path), arcname=path.relative_to(output).as_posix())
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                info.mode = 0o644
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
        # Bundle hash lives beside the immutable bundle, not inside its own hash.
        write_json(bundle.with_suffix(bundle.suffix + ".receipt.json"), {
            "bundle_sha256": file_sha256(bundle), "bundle_bytes": bundle.stat().st_size,
            "manifest_receipt_sha256": file_sha256(output / "receipt.json"),
            "contains_only_selected_public_images_and_provenance": True,
        })
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("training/private/public_datasets/combined-banknote-id-v1"))
    parser.add_argument("--output", type=Path, default=Path("training/private/gemma4-public-v1"))
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--desk-from-existing", type=Path,
                        help="Prepare metadata-only DESK composites referencing an existing public bundle")
    args = parser.parse_args()
    require(0 <= args.seed < 2**31, "invalid_seed")
    if args.desk_from_existing:
        result = prepare_desk(args.dataset_root, args.desk_from_existing, args.output, args.seed)
    else:
        result = prepare(args.dataset_root, args.output, args.seed, args.bundle)
    print(json.dumps(result, sort_keys=True))


def prepare_desk(root: Path, existing: Path, output: Path, seed: int) -> dict:
    """Metadata-only, partially supervised DESK adaptation; no image decoding.

    KZ-BD supplies boxes, not segmentation.  Its rectangular cutouts cannot
    certify absence of other objects.  Only known cash/id VALUE tokens are
    supervised, with all unsupported values explicit loss-masked placeholders.
    """
    root, existing, output = root.resolve(), existing.resolve(), output.resolve()
    require(not output.exists(), "fresh_desk_output_required")
    lock, lock_hash = load_lock(root)
    frozen_receipt = json.loads((existing / "receipt.json").read_text(encoding="utf-8"))
    require(frozen_receipt["dataset_lock_sha256"] == lock_hash, "public_bundle_source_lock_mismatch")
    base_rows = {}
    for split in ("train", "validation"):
        path = existing / f"{split}.jsonl"
        require(file_sha256(path) == frozen_receipt["manifest_sha256"][path.name],
                "public_bundle_manifest_hash_mismatch")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        unique = {row["source_object_id"]: row for row in rows}
        base_rows[split] = {name: sorted((row for row in unique.values()
                                         if row["target"]["object_class"] == name),
                                        key=lambda row: row["source_object_id"])
                            for name in ("id_card", "banknote")}
    originals = {r["content_sha256"]: r for r in lock["records"] if r.get("included")
                 and r["split"] in {"train", "validation"}}
    prompts_path = Path(__file__).resolve().parents[1] / "src/forte/fast/prompts.py"
    tree = ast.parse(prompts_path.read_text(encoding="utf-8"))
    prompt_values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"SYSTEM", "DESK_Q", "DESK_SCHEMA"}:
                    prompt_values[target.id] = ast.literal_eval(node.value)
    require(set(prompt_values) == {"SYSTEM", "DESK_Q", "DESK_SCHEMA"}, "desk_prompt_contract_missing")

    def component(row: dict, condition: str, index: int, side: int) -> dict:
        source = originals[row["image_sha256"]]
        kind = row["target"]["object_class"]
        quad = source.get("source_record", {}).get("source_quad") if kind == "id_card" else None
        if kind == "id_card":
            require(isinstance(quad, list) and len(quad) == 4
                    and all(isinstance(point, list) and len(point) == 2 for point in quad),
                    "publisher_id_quadrangle_required")
        x1, y1, x2, y2 = row["crop_box_xyxy"]
        ratio = (x2 - x1) / (y2 - y1)
        height = side if ratio >= 1 else max(1, round(side / ratio))
        width = max(1, round(height * ratio))
        # Separate halves leave no component overlap in co-presence examples.
        x = 20 + (index * 17 % 40) if kind == "banknote" else 244 + (index * 11 % 16)
        y = 80 + (index * 19 % 180)
        width, height = min(width, 168), min(height, 168)
        if x + width >= 448:
            x = 447 - width
        return ({key: row.get(key) for key in (
            "image_path", "image_sha256", "crop_box_xyxy", "source_dataset", "source_group",
            "source_image_group", "template_group", "physical_document_group", "scene_group")}
        | {"source_quad": quad, "object_class": kind, "x": x, "y": y,
           "width": width, "height": height,
           "mask_kind": "publisher_quadrangle" if quad else "bbox_unknown_background"})

    all_rows = []
    summaries = {}
    for split, per_condition in (("train", 256), ("validation", 32)):
        rows = []
        ids, notes = base_rows[split]["id_card"], base_rows[split]["banknote"]
        require(ids and len(notes) >= per_condition, "insufficient_public_desk_objects")
        for condition in ("note_only", "id_only", "both", "blank"):
            for index in range(per_condition):
                material = f"{seed}:{split}:{condition}:{index}"
                digest = hashlib.sha256(material.encode()).hexdigest()
                gray = 90 + int(digest[:2], 16) % 111
                side = (16, 32, 64, 96)[index % 4]
                components = []
                if condition in {"note_only", "both"}:
                    components.append(component(notes[index % len(notes)], condition, index, side))
                if condition in {"id_only", "both"}:
                    components.append(component(ids[index % len(ids)], condition, index, side))
                target = {key: False for key in prompt_values["DESK_SCHEMA"]["required"]}
                target["cash"] = condition in {"note_only", "both"}
                target["id_or_document"] = condition in {"id_only", "both"}
                supervision = ["cash"] if condition == "note_only" else ["cash", "id_or_document"]
                rows.append({
                    "example_id": digest,
                    "origin": "public",
                    "split": split,
                    "source_dataset": "public_composition",
                    "source_group": "composition:" + digest,
                    "condition": condition,
                    "system": prompt_values["SYSTEM"],
                    "prompt": prompt_values["DESK_Q"],
                    "json_schema": prompt_values["DESK_SCHEMA"],
                    "target": target,
                    "supervision_fields": supervision,
                    "label_source": "published_object_annotation_and_generated_empty_canvas",
                    "unsupported_values_are_loss_masked_placeholders": True,
                    "composition": {"width": 448, "height": 448, "background": [gray, gray, gray],
                                    "components": components, "blank_seed_group": "generated:" + digest},
                })
        rows.sort(key=lambda row: row["example_id"])
        all_rows.extend(rows)
        summaries[split] = {"examples": len(rows), "condition_counts": dict(Counter(
            row["condition"] for row in rows)), "known_field_tokens": sum(
            len(row["supervision_fields"]) for row in rows),
            "unique_id_images": len({c["image_sha256"] for r in rows
                                      for c in r["composition"]["components"] if c["object_class"] == "id_card"}),
            "unique_banknote_images": len({c["image_sha256"] for r in rows
                                            for c in r["composition"]["components"] if c["object_class"] == "banknote"})}
    for key in ("source_group", "source_image_group", "template_group", "physical_document_group", "scene_group"):
        split_values = {split: {component[key] for row in all_rows if row["split"] == split
                               for component in row["composition"]["components"] if component.get(key)}
                        for split in ("train", "validation")}
        require(not split_values["train"] & split_values["validation"], "desk_component_group_overlap:" + key)

    output.mkdir(parents=True)
    hashes = {}
    for split in ("train", "validation", "manifest"):
        path = output / (split + ".jsonl")
        rows = all_rows if split == "manifest" else [r for r in all_rows if r["split"] == split]
        path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                                for row in rows), encoding="utf-8", newline="\n")
        hashes[path.name] = file_sha256(path)
    receipt = {
        "schema_version": "forte.q30.public-desk-partial.v1",
        "state": "prepared_public_only_untrained",
        "origin": "public",
        "dataset_lock_sha256": lock_hash,
        "base_public_receipt_sha256": file_sha256(existing / "receipt.json"),
        "manifest_sha256": hashes,
        "prompt_source_sha256": file_sha256(prompts_path),
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "prompt_contract": prompt_values,
        "counts": summaries,
        "supervised_fields": ["cash", "id_or_document"],
        "loss_masked_fields": ["cash_bag", "coins_or_valuables", "drink", "coat_or_handbag"],
        "unknown_note_only_field": "id_or_document",
        "synthetic_render_policy": "generated_neutral_canvas; ID publisher quadrangle alpha mask; KZ bbox unknown background retained with unsupported fields loss-masked",
        "image_root_relative_to_existing_bundle": ".",
        "original_image_bundle_sha256": "b68bb1c09c21f3915c59ac3c3b95029e8387351a7ebaed4a0e20fa5899102abf",
        "image_decoding_performed": False,
        "organizer_data_used": False,
        "validation_exposure": "reused_public_development_not_fresh_lockbox",
        "sources": lock["source_provenance"],
        "seed": seed,
    }
    write_json(output / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    main()
