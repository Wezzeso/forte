"""Synthetic-only contracts for a portable two-class public training export."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import shutil

from PIL import Image
import pytest

from training.guards import TrainingGateError, canonical_sha256, file_sha256
from training.specialists.common import pinned_json, require, write_json
from training.specialists import public_merge as merge

# Reuse the independent, generated publisher-layout fixtures to exercise the
# real source verifiers together once, without accessing public payloads.
from test_public_holo import holo_source  # noqa: F401
from test_specialist_public_data import public_source  # noqa: F401


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _repin(root: Path, change=None):
    lock = _read(root / "dataset-lock.json")
    if change:
        change(lock)
    lock["dataset_sha256"] = canonical_sha256({k: v for k, v in lock.items() if k != "dataset_sha256"})
    write_json(root / "dataset-lock.json", lock)
    return file_sha256(root / "dataset-lock.json")


@pytest.fixture
def public_pair(tmp_path, monkeypatch):
    """Source contracts are stubbed; source importer tests own license/box proof."""
    generator = random.Random(20261018)
    config = {"schema_version": merge.SCHEMA, "frozen": True, "source_datasets": []}
    roots = {}
    for kind, (dataset_id, classes) in merge.SOURCES.items():
        root = tmp_path / dataset_id
        root.mkdir()
        roots[dataset_id] = root
        files, rows = {}, []
        for index, role in enumerate(("train", "validation", "sealed_test", "quarantine")):
            relative = merge.DIRECTORIES[role] + f"/sample-{index}.png"
            image_path = root / relative
            image_path.parent.mkdir()
            Image.frombytes("RGB", (64, 48), generator.randbytes(64 * 48 * 3)).save(image_path)
            annotation = f"source_annotations/label-{index}.json"
            write_json(root / annotation, {"synthetic": True, "box": [4, 6, 30, 20]})
            objects = [{"class_name": classes[0], "bbox": [4, 6, 30, 20]}]
            rows.append({"sample_id": f"synthetic:{index}", "split": role, "included": role != "quarantine",
                         "source_group": "source:" + str(index), "template_group": "template:" + str(index),
                         "scene_group": "scene:" + str(index) if kind == "midv_holo" else None,
                         "physical_document_group": "document:" + str(index) if kind == "midv_holo" else None,
                         "content_sha256": file_sha256(image_path), "annotation_sha256": file_sha256(root / annotation),
                         "dhash256": merge.dhash256(image_path), "width": 64, "height": 48,
                         "objects": objects, "export_path": relative, "annotation_export_path": annotation})
            files[relative] = file_sha256(image_path)
            files[annotation] = file_sha256(root / annotation)
        evidence = root / "provenance/LICENSE.txt"
        evidence.parent.mkdir()
        evidence.write_text("Synthetic license evidence, not downloaded data", encoding="utf-8")
        files["provenance/LICENSE.txt"] = file_sha256(evidence)
        lock = {"schema_version": merge.LOCK_SCHEMA, "state": "frozen", "origin": "public", "dataset_kind": kind,
                "dataset_id": dataset_id, "classes": classes, "records": rows, "files": files,
                "license": {"spdx": "CC-BY-SA-2.5" if kind == "midv_holo" else "CC-BY-4.0",
                            "attribution": ["Synthetic publisher", "GeneratedPhotos (synthetic attribution test)"]},
                "limitations": ["Synthetic source: no scene independence established"],
                "annotation_scope": "publisher_document_quadrangles_only" if kind == "midv_holo" else "publisher_yolo_objects_only"}
        lock["dataset_sha256"] = canonical_sha256(lock)
        write_json(root / "dataset-lock.json", lock)
        config["source_datasets"].append({"dataset_id": dataset_id, "dataset_kind": kind,
                                         "root": str(root), "expected_lock_sha256": file_sha256(root / "dataset-lock.json")})
    calls = []

    def verified_source(root, expected, kind):
        calls.append((root, kind))
        lock = pinned_json(root / "dataset-lock.json", expected)
        require(lock["dataset_sha256"] == canonical_sha256({k: v for k, v in lock.items() if k != "dataset_sha256"}),
                "synthetic_source_digest_changed")
        for relative, digest in lock["files"].items():
            require(file_sha256(root / relative) == digest, "synthetic_source_file_changed")
        return lock

    monkeypatch.setattr(merge, "_verify_source", verified_source)
    path = tmp_path / "merge.json"
    write_json(path, config)
    return {"path": path, "config": config, "roots": roots, "output": tmp_path / "combined", "calls": calls}


def _prepare(fixture):
    summary = merge.prepare_combined_public_dataset(fixture["path"], fixture["output"])
    return summary, merge.verify_combined_public_dataset(fixture["output"], summary["dataset_lock_sha256"])


def test_merge_is_portable_deterministic_and_maps_runtime_and_coco_classes(public_pair, tmp_path):
    result, lock = _prepare(public_pair)
    second = merge.prepare_combined_public_dataset(public_pair["path"], tmp_path / "second")
    assert result["dataset_sha256"] == second["dataset_sha256"]
    assert lock["dataset_kind"] == "combined_public" and lock["origin"] == "public"
    assert lock["classes"] == ["single_banknote", "id_card"]
    assert lock["prediction_class_ids"] == {"single_banknote": 0, "id_card": 1}
    assert lock["coco_category_ids"] == {"single_banknote": 1, "id_card": 2}
    for source in lock["source_datasets"]:
        original = public_pair["roots"][source["dataset_id"]]
        copied = public_pair["output"] / source["root"]
        original_lock = _read(original / "dataset-lock.json")
        assert (copied / "dataset-lock.json").read_bytes() == (original / "dataset-lock.json").read_bytes()
        for relative in original_lock["files"]:
            assert (copied / relative).read_bytes() == (original / relative).read_bytes()
        shutil.rmtree(original)
    # Everything necessary to verify lives in the export, not old source paths.
    assert merge.verify_combined_public_dataset(public_pair["output"], result["dataset_lock_sha256"])["records"] == lock["records"]
    for role, directory in merge.DIRECTORIES.items():
        coco = _read(public_pair["output"] / directory / "_annotations.coco.json")
        assert len(coco["images"]) == (0 if role == "quarantine" else 2)
        assert {obj["category_id"] for obj in coco["annotations"]} == (set() if role == "quarantine" else {1, 2})
    assert _read(public_pair["output"] / "test/_annotations.coco.json")["images"] == []


def test_source_groups_and_license_evidence_are_retained_without_review_claims(public_pair):
    _, lock = _prepare(public_pair)
    attribution = (public_pair["output"] / "ATTRIBUTION.md").read_text(encoding="utf-8")
    assert "GeneratedPhotos" in attribution and "CC-BY-SA-2.5" in attribution and "CC-BY-4.0" in attribution
    assert all("training_authorized" not in row and "human_reviewed" not in row for row in lock["records"])
    for row in lock["records"]:
        assert row["source_group"] == row["source_record"]["source_group"]
        assert row["template_group"] == row["source_record"]["template_group"]
        assert row["scene_group"] == row["source_record"]["scene_group"]
        assert row["source_split"] == row["source_record"]["split"]
    assert any("no scene independence" in text for text in lock["limitations"])


def test_real_importer_contracts_merge_without_source_verifier_stubs(public_source, holo_source, tmp_path):
    from training.specialists.public_data import prepare_public_dataset
    from training.specialists.public_holo import prepare_public_holo_dataset
    config = {"schema_version": merge.SCHEMA, "frozen": True, "source_datasets": []}
    for kind, (fixture, prepare) in {
        "kazakhstan_banknotes": (public_source, prepare_public_dataset),
        "midv_holo": (holo_source, prepare_public_holo_dataset),
    }.items():
        output = tmp_path / (kind + "-export")
        result = prepare(fixture["config_path"], output)
        config["source_datasets"].append({"dataset_kind": kind, "dataset_id": merge.SOURCES[kind][0],
                                         "root": str(output), "expected_lock_sha256": result["dataset_lock_sha256"]})
    path = tmp_path / "real-contracts-merge.json"
    write_json(path, config)
    output = tmp_path / "real-contracts-combined"
    result = merge.prepare_combined_public_dataset(path, output)
    lock = merge.verify_combined_public_dataset(output, result["dataset_lock_sha256"])
    assert len(lock["records"]) == 73
    assert "Generated Photos" in (output / "ATTRIBUTION.md").read_text(encoding="utf-8")
    for field in merge.POLICY["group_fields"]:
        groups = {}
        for row in lock["records"]:
            if row["included"] and row[field] is not None:
                key = row["dataset_id"], row[field]
                assert groups.setdefault(key, row["split"]) == row["split"]
    for role in ("train", "validation", "sealed_test"):
        assert lock["summary"]["included_class_split_counts"]["single_banknote"][role] > 0
        assert lock["summary"]["included_class_split_counts"]["id_card"][role] == 1


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(frozen=False),
    lambda c: c.update(schema_version="forte.specialists.data.v1"),
    lambda c: c.update(source_datasets=c["source_datasets"][:1]),
    lambda c: c["source_datasets"][0].update(expected_lock_sha256="0" * 64),
    lambda c: c["source_datasets"][0].update(dataset_id="unverified-source"),
    lambda c: c["source_datasets"][0].update(dataset_kind="combined_public"),
    lambda c: c["source_datasets"].__setitem__(1, copy.deepcopy(c["source_datasets"][0])),
])
def test_merge_requires_two_frozen_pinned_public_sources(public_pair, mutation):
    mutation(public_pair["config"])
    write_json(public_pair["path"], public_pair["config"])
    with pytest.raises(TrainingGateError):
        merge.prepare_combined_public_dataset(public_pair["path"], public_pair["output"])
    assert not public_pair["output"].exists()


def test_source_tampering_is_rejected_before_any_export(public_pair):
    root = public_pair["roots"]["MIDV-Holo"]
    (root / "train/sample-0.png").write_bytes(b"changed public source")
    with pytest.raises(TrainingGateError, match="synthetic_source_file_changed"):
        merge.prepare_combined_public_dataset(public_pair["path"], public_pair["output"])
    assert not public_pair["output"].exists()


@pytest.mark.parametrize("relative", ["../outside", "/absolute", "C:/absolute", "train\\bad", "./train/image.png"])
def test_merge_inventory_paths_are_strictly_relative(relative):
    with pytest.raises(TrainingGateError, match="relative_path_invalid"):
        merge._relative(relative)


@pytest.mark.parametrize("field,value", [
    ("classes", ["id_card", "single_banknote"]),
    ("prediction_class_ids", {"single_banknote": 1, "id_card": 2}),
    ("coco_category_ids", {"single_banknote": 0, "id_card": 1}),
    ("split_policy", {"near_duplicate_hamming": 0}),
    ("limitations", []),
    ("source_provenance", []),
])
def test_rehashing_lock_cannot_bypass_mapping_policy_or_license_guards(public_pair, field, value):
    _prepare(public_pair)
    expected = _repin(public_pair["output"], lambda lock: lock.update({field: value}))
    with pytest.raises(TrainingGateError):
        merge.verify_combined_public_dataset(public_pair["output"], expected)


def test_rehashing_coco_cannot_invent_labels(public_pair):
    _prepare(public_pair)
    root = public_pair["output"]
    relative = "train/_annotations.coco.json"
    coco = _read(root / relative)
    coco["annotations"][0]["bbox"] = [1, 1, 2, 2]
    write_json(root / relative, coco)
    expected = _repin(root, lambda lock: lock["files"].update({relative: file_sha256(root / relative)}))
    with pytest.raises(TrainingGateError, match="coco_derivation_mismatch"):
        merge.verify_combined_public_dataset(root, expected)


def test_rehashing_rows_cannot_promote_sealed_test_or_invent_manual_labels(public_pair):
    _prepare(public_pair)
    root = public_pair["output"]

    def change(lock):
        row = next(row for row in lock["records"] if row["split"] == "sealed_test")
        row.update(split="train", included=True)
        row["objects"][0]["bbox"] = [1, 1, 2, 2]

    expected = _repin(root, change)
    with pytest.raises(TrainingGateError, match="record_derivation_mismatch"):
        merge.verify_combined_public_dataset(root, expected)


def test_rehashed_attribution_cannot_drop_generatedphotos(public_pair):
    _prepare(public_pair)
    root = public_pair["output"]
    relative = "ATTRIBUTION.md"
    path = root / relative
    path.write_text("Incomplete synthetic attribution", encoding="utf-8")
    expected = _repin(root, lambda lock: lock["files"].update({relative: file_sha256(path)}))
    with pytest.raises(TrainingGateError, match="attribution_mismatch"):
        merge.verify_combined_public_dataset(root, expected)


def test_even_rehashed_extra_training_image_is_forbidden(public_pair):
    _prepare(public_pair)
    root = public_pair["output"]
    extra = root / "train/unlisted.png"
    extra.write_bytes(b"synthetic")
    expected = _repin(root, lambda lock: lock["files"].update({"train/unlisted.png": file_sha256(extra)}))
    with pytest.raises(TrainingGateError, match="inventory_derivation_mismatch"):
        merge.verify_combined_public_dataset(root, expected)


def _row(index, role, dhash, *, dataset="source", template=None, objects=None):
    return {"sample_id": str(index), "dataset_id": dataset, "split": role, "source_split": role,
            "included": role != "quarantine", "source_included": role != "quarantine",
            "source_group": f"group:{index}", "template_group": template or f"template:{index}",
            "scene_group": None, "physical_document_group": None,
            "content_sha256": canonical_sha256(["content", index]),
            "source_content_sha256": canonical_sha256(["original", index]),
            "dhash256": f"{dhash:064x}", "source_dhash256": f"{dhash:064x}",
            "objects": objects if objects is not None else [{"class_name": "id_card", "bbox": [0, 0, 2, 2]}],
            "width": 64, "height": 48,
            "source_export_path": f"source_datasets/source/{role}/{index}.png",
            "export_path": f"source_datasets/source/{role}/{index}.png", "image_id": None,
            "duplicate_of": None}


def test_radius_four_duplicate_components_are_transitive_and_keep_sealed_role():
    rows = [_row(0, "train", 0), _row(1, "validation", 15), _row(2, "sealed_test", 255)]
    assert (0 ^ 255).bit_count() == 8
    merge._assign(rows)
    assert len({row["combined_group"] for row in rows}) == 1
    assert [row["split"] for row in rows] == ["quarantine", "quarantine", "sealed_test"]
    assert [row["included"] for row in rows] == [False, False, True]


def test_mixed_original_and_prepared_near_duplicate_closes_across_representations():
    rows = [_row(0, "train", (1 << 128) - 1),
            _row(1, "sealed_test", 15, dataset="other")]
    rows[0]["source_dhash256"] = f"{0:064x}"
    rows[1]["source_dhash256"] = f"{((1 << 256) - 1) ^ ((1 << 128) - 1):064x}"
    prepared_a, prepared_b = (int(row["dhash256"], 16) for row in rows)
    original_a, original_b = (int(row["source_dhash256"], 16) for row in rows)
    assert (prepared_a ^ prepared_b).bit_count() > 4
    assert (original_a ^ original_b).bit_count() > 4
    assert (prepared_a ^ original_b).bit_count() > 4
    assert (original_a ^ prepared_b).bit_count() == 4
    merge._assign(rows)
    assert len({row["combined_group"] for row in rows}) == 1
    assert rows[0]["split"] == "quarantine" and not rows[0]["included"]
    assert rows[1]["split"] == "sealed_test" and rows[1]["included"]


def test_group_closure_quarantines_whole_training_template_after_cross_source_duplicate():
    rows = [_row(0, "train", 0, template="same-template"),
            _row(1, "train", (1 << 128) - 1, template="same-template"),
            _row(2, "sealed_test", 15, dataset="other")]
    merge._assign(rows)
    assert [row["split"] for row in rows] == ["quarantine", "quarantine", "sealed_test"]
    assert len({row["combined_group"] for row in rows}) == 1


def test_original_quarantine_can_share_template_without_disabling_clean_training():
    rows = [_row(0, "train", 0, template="publisher-denomination"),
            _row(1, "quarantine", (1 << 128) - 1, template="publisher-denomination")]
    merge._assign(rows)
    assert rows[0]["split"] == "train" and rows[0]["included"]
    assert rows[1]["split"] == "quarantine" and not rows[1]["included"]


def test_exact_duplicate_with_conflicting_class_targets_is_excluded_even_from_sealed_coco():
    rows = [_row(0, "train", 0, objects=[{"class_name": "single_banknote", "bbox": [0, 0, 2, 2]}]),
            _row(1, "sealed_test", 0, dataset="other")]
    rows[1]["content_sha256"] = rows[0]["content_sha256"]
    merge._assign(rows)
    assert rows[0]["split"] == "quarantine" and rows[1]["split"] == "sealed_test"
    assert not any(row["included"] for row in rows)
    assert all("conflicting_exact_content_target_annotations" in row["quarantine_reasons"] for row in rows)
    assert merge._coco(rows, "sealed_test")["images"] == []


def test_consistent_same_role_exact_duplicates_are_deduplicated_globally():
    rows = [_row(0, "train", 0), _row(1, "train", 0, dataset="other")]
    rows[1]["content_sha256"] = rows[0]["content_sha256"]
    merge._assign(rows)
    assert rows[0]["included"] and not rows[1]["included"]
    assert rows[1]["duplicate_of"] == rows[0]["sample_id"]
    assert len(merge._coco(rows, "train")["images"]) == 1
