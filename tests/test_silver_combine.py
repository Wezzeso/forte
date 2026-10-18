"""Synthetic append-only export, two-root media and compact alias checks."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from training.guards import TrainingGateError, canonical_sha256, file_sha256
from training.silver_combine import combine_exports, compact_training_aliases, compile_media, resolve_original_frame


def combine_fixture(tmp_path):
    identity = {"model_id": "synthetic/private", "model_revision": "a" * 40,
                "processor_revision": "a" * 40, "model_manifest_sha256": "b" * 64}
    exports = []
    for index in range(2):
        provenance = {**identity, "prompt_sha256": str(index) * 64, "source_proposal_sha256": str(index + 2) * 64}
        row = {"content_sha256": str(index + 1) * 64, "split": "train", "human_reviewed": False,
               "source_pts_identity_checked": True, "provenance": provenance,
               "windows": [{"provenance": provenance}]}
        path = tmp_path / f"source-{index}.jsonl"
        path.write_text(json.dumps(row, separators=(",", ":")) + "\n", encoding="utf-8")
        exports.append({"path": str(path), "sha256": file_sha256(path)})
    schema = tmp_path / "synthetic-schema.json"
    schema.write_text('{"type":"object"}', encoding="utf-8")
    return {"input_exports": exports, "expected_teacher": identity,
        "output": str(tmp_path / "new-export.jsonl"), "proof_output": str(tmp_path / "proof.json"),
        "machine_schema": str(schema), "expected_machine_schema_sha256": file_sha256(schema)}


def test_combination_preserves_both_raw_exports_exactly_with_distinct_prompts(tmp_path):
    config = combine_fixture(tmp_path)
    old, new = [Path(item["path"]).read_bytes() for item in config["input_exports"]]
    result = combine_exports(config)
    assert Path(config["output"]).read_bytes() == old + new
    assert result["windows"] == 2 and result["records"] == 2
    assert result["old_export_preserved_as_exact_byte_prefix"] is True
    assert Path(config["input_exports"][0]["path"]).read_bytes() == old
    with pytest.raises(TrainingGateError, match="output_exists"):
        combine_exports(config)


@pytest.mark.parametrize("mutation", ["identity", "duplicate", "role", "bytes"])
def test_invalid_combination_fails_before_writing(tmp_path, mutation):
    config = combine_fixture(tmp_path)
    item = config["input_exports"][1]
    path = Path(item["path"])
    row = json.loads(path.read_text())
    if mutation == "identity":
        row["provenance"]["model_revision"] = "c" * 40
    elif mutation == "duplicate":
        row["content_sha256"] = "1" * 64
    elif mutation == "role":
        row["split"] = "final_check"
    else:
        path.write_bytes(b"altered")
    if mutation != "bytes":
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        item["sha256"] = file_sha256(path)
    with pytest.raises(TrainingGateError):
        combine_exports(config)
    assert not Path(config["output"]).exists()


def test_windows_and_posix_original_frames_are_copied_without_reencoding(tmp_path):
    from forte.video.batches import source_frame_id
    roots = [tmp_path / "legacy", tmp_path / "new"]
    rows = []
    for index, root in enumerate(roots):
        root.mkdir()
        source = root / "frame.jpg"
        source.write_bytes(b"synthetic-original-jpeg-" + str(index).encode())
        content = str(index + 1) * 64
        frame = {"frame_id": source_frame_id(content, 5, 1, 10), "source_pts": 5, "pts_sec": .5,
                 "path": r"C:\old\frame.jpg" if index == 0 else "/old-linux/frame.jpg",
                 "jpeg_sha256": file_sha256(source)}
        rows.append({"content_sha256": content, "split": "train", "human_reviewed": False,
            "partition_binding": {"assignments_sha256": "f" * 64}, "source_time_base": "1/10",
            "source_start_sec": 0, "frames": [frame]})
    annotations = tmp_path / "combined.jsonl"
    annotations.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    config = {"annotations": str(annotations), "expected_annotations_sha256": file_sha256(annotations),
        "media_root": str(tmp_path / "media"), "manifest_output": str(tmp_path / "media-manifest.json"),
        "assignments_sha256": "f" * 64, "source_mappings": [
            {"path_style": "windows", "original_prefix": r"C:\old", "actual_root": str(roots[0])},
            {"path_style": "posix", "original_prefix": "/old-linux", "actual_root": str(roots[1])}]}
    result = compile_media(config)
    assert result["frames"] == result["contents"] == 2
    manifest = json.loads(Path(config["manifest_output"]).read_text())
    for entry in manifest["frames"]:
        assert file_sha256(Path(config["media_root"]) / entry["path"]) == entry["sha256"]
    assert annotations.read_text().count("old-linux") == 1


def test_frame_relocation_cannot_escape_or_match_ambiguous_roots(tmp_path):
    mappings = [{"path_style": "posix", "original_prefix": "/old", "actual_root": str(tmp_path)}]
    with pytest.raises(TrainingGateError, match="frame_path_escape"):
        resolve_original_frame("/old/../outside.jpg", mappings)
    file = tmp_path / "frame.jpg"
    file.write_bytes(b"synthetic")
    with pytest.raises(TrainingGateError, match="exactly_one_root_mapping"):
        resolve_original_frame("/old/frame.jpg", mappings * 2)


def test_verified_legacy_content_addressed_cache_resolves_without_raw_path_rewrite(tmp_path):
    content, fid = "a" * 64, "b" * 64
    source = tmp_path / content / (fid + ".jpg")
    source.parent.mkdir()
    source.write_bytes(b"original-verified-jpeg")
    mappings = [{"path_style": "windows", "original_prefix": r"C:\old\proposals",
                 "actual_root": str(tmp_path), "content_addressed": True}]
    assert resolve_original_frame(r"C:\old\proposals\job\frames\000001.jpg", mappings,
                                  content=content, frame_id=fid) == source


def test_compact_aliases_match_serving_and_leave_canonical_teachers_unchanged():
    first, second = "a" * 64, "b" * 64
    row = {"sample_id": "sample", "split": "train", "target_scope": "exact_anchor_frame",
        "focal_frame_id": second, "frames": [{"frame_id": first, "path": "one.jpg"}, {"frame_id": second, "path": "two.jpg"}],
        "prompt": "Evaluate focal " + second, "target": {"facts": [{"fact_id": "CASH_VISIBLE",
        "state": "present", "supporting_frame_ids": [second]}], "candidates": []}}
    original = deepcopy(row)
    rows, proof = compact_training_aliases([row])
    assert row == original
    assert rows[0]["focal_frame_id"] == "f01"
    assert rows[0]["target"]["facts"][0]["supporting_frame_ids"] == ["f01"]
    assert second not in rows[0]["prompt"] and "f01" in rows[0]["prompt"]
    assert rows[0]["canonical_focal_frame_id"] == second
    assert proof["balance_must_precede_aliasing"] is True
    assert proof["teacher_bytes_modified"] is False
