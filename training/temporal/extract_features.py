"""Prepare private hash-named sources and extract complete native-frame features.

Preparation uses the existing CPU FFprobe. Extraction needs only PyAV and the
offline encoder runtime; it consumes no organizer answers or source filenames.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import tarfile
import time

from training.guards import (TrainingGateError, canonical_sha256, file_sha256,
                             load_json, require)

SCHEMA = "forte.private-feature-inputs.v1"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    require(not path.exists() and not temporary.exists(), "feature_output_already_exists")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.rename(path)


def prepare_sources(worklist_path: Path, split_path: Path, output: Path, archive: Path, *,
                    expected_worklist_sha256: str, expected_split_sha256: str,
                    source_mount: str = "/private/sources") -> dict:
    from forte.video.probe import probe_video

    require(file_sha256(worklist_path) == expected_worklist_sha256
            and file_sha256(split_path) == expected_split_sha256, "feature_preparation_input_pin_mismatch")
    worklist, split = load_json(worklist_path), load_json(split_path)
    require(worklist.get("schema_version") == "forte.autonomous-annotation-worklist.v1"
            and worklist.get("split_file_sha256") == expected_split_sha256,
            "feature_worklist_split_binding_invalid")
    assignments = {row["content_hash"]: row for row in split["records"]}
    expected = {content for content, row in assignments.items() if row["split"] in {"train", "validation"}}
    rows = worklist.get("records", [])
    require(len(rows) == len(expected) and {row.get("content_sha256") for row in rows} == expected,
            "feature_worklist_population_mismatch")
    mount = PurePosixPath(source_mount)
    require(mount.is_absolute() and str(mount) == "/private/sources", "feature_source_mount_invalid")
    require(not archive.exists() and not output.exists(), "feature_output_already_exists")
    archive.parent.mkdir(parents=True, exist_ok=True)
    prepared, source_paths = [], []
    for row in sorted(rows, key=lambda item: item["content_sha256"]):
        content = row["content_sha256"]
        assignment = assignments[content]
        require(row["split"] == assignment["split"]
                and row["scenario_group_id"] == assignment["scenario_group_id"],
                "feature_worklist_assignment_mismatch")
        source = Path(row["source_path"]).resolve(strict=True)
        require(source.is_file(), "feature_source_missing")
        asset = probe_video(source, job_id="feature-" + content[:24])
        require(asset.content_hash == content, "feature_source_digest_mismatch")
        member = content + ".mov"
        data = asset.model_dump(mode="json")
        data["owned_path"] = str(mount / member)
        # Probe diagnostics are unnecessary model inputs; retain only contracts.
        data["metadata"] = {}
        prepared.append({"content_sha256": content, "sample_id": row["sample_id"],
            "group_id": assignment["scenario_group_id"], "split": assignment["split"],
            "source_file": member, "source_bytes": source.stat().st_size, "asset": data})
        source_paths.append((source, member))
    with tarfile.open(archive, mode="x", format=tarfile.USTAR_FORMAT) as bundle:
        for source, member in source_paths:
            info = bundle.gettarinfo(str(source), arcname=member)
            require(info.isreg(), "feature_source_archive_member_invalid")
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o400
            info.mtime = 0
            with source.open("rb") as stream:
                bundle.addfile(info, stream)
    payload = {"schema_version": SCHEMA, "worklist_sha256": expected_worklist_sha256,
        "split_manifest_sha256": expected_split_sha256,
        "assignments_sha256": worklist["assignments_sha256"],
        "source_archive_sha256": file_sha256(archive), "source_archive_bytes": archive.stat().st_size,
        "source_mount": str(mount), "records": prepared, "source_count": len(prepared),
        "final_check_excluded": True, "organizer_labels_in_model_inputs": False,
        "sampling_policy": "all_native_frames_with_original_pts_and_tail"}
    payload["payload_sha256"] = canonical_sha256(payload)
    validate_prepared_inputs(payload)
    _write_json(output, payload)
    return {"status": "prepared", "source_count": len(prepared),
        "split_counts": dict(Counter(row["split"] for row in prepared)),
        "source_bytes": sum(row["source_bytes"] for row in prepared),
        "input_manifest_sha256": file_sha256(output),
        "source_archive_sha256": payload["source_archive_sha256"],
        "source_archive_bytes": payload["source_archive_bytes"], "gpu_used": False}


def validate_prepared_inputs(payload: dict) -> list[dict]:
    from forte.shared.schemas import VideoAsset

    require(payload.get("schema_version") == SCHEMA and payload.get("payload_sha256") == canonical_sha256(
        {key: value for key, value in payload.items() if key != "payload_sha256"}),
        "feature_input_manifest_invalid")
    require(payload.get("final_check_excluded") is True and payload.get("organizer_labels_in_model_inputs") is False
            and payload.get("source_mount") == "/private/sources", "feature_input_scope_invalid")
    rows = payload.get("records", [])
    require(isinstance(rows, list) and 0 < len(rows) <= 1000 and payload.get("source_count") == len(rows)
            and len({row.get("content_sha256") for row in rows}) == len(rows), "feature_input_population_invalid")
    groups = {}
    for row in rows:
        asset = VideoAsset.model_validate(row["asset"])
        content = row["content_sha256"]
        require(asset.content_hash == content and row["source_file"] == content + ".mov"
                and asset.owned_path == "/private/sources/" + content + ".mov",
                "feature_input_source_binding_invalid")
        require(row["split"] in {"train", "validation"} and bool(row.get("group_id"))
                and groups.get(row["group_id"], row["split"]) == row["split"],
                "feature_input_assignment_invalid")
        groups[row["group_id"]] = row["split"]
        require(type(row.get("source_bytes")) is int and 0 < row["source_bytes"] <= 256 * 1024 * 1024,
                "feature_input_source_size_invalid")
    return rows


def extract_sources(inputs: Path, destination: Path, *, expected_inputs_sha256: str,
                    snapshot: Path, encoder_manifest: Path, encoder_manifest_sha256: str,
                    batch_size: int = 32, per_source_timeout_sec: float = 120.) -> dict:
    require(file_sha256(inputs) == expected_inputs_sha256, "feature_input_file_pin_mismatch")
    payload = load_json(inputs)
    rows = validate_prepared_inputs(payload)
    require(0 < per_source_timeout_sec <= 600, "feature_extract_timeout_invalid")
    destination = destination.resolve()
    require(not (destination / "index.json").exists(), "feature_cache_index_already_exists")
    # Preflight every source before GPU loading, decoding or partial cache writes.
    for row in rows:
        source = Path(row["asset"]["owned_path"])
        require(not source.is_symlink() and source.is_file() and source.stat().st_size == row["source_bytes"]
                and file_sha256(source) == row["content_sha256"], "feature_staged_source_integrity_failed")
    from forte.shared.schemas import VideoAsset
    from forte.video.batches import iter_rgb_batches, rgb_transform_recipe
    from serving.fast_encoder import FastEncoder, FastEncoderConfig
    from serving.feature_stream import iter_encoded_batches
    from training.temporal.cache import write_feature_cache

    decoder_recipe = rgb_transform_recipe(target_size=(256, 256), aspect_policy="letterbox")
    sampling_recipe = {"mode": "native", "all_native_frames": True, "tail_required": True,
                       "time": "original_integer_pts_and_canonical_rational_base"}
    crop_recipe = {"views": "global_only", "source_box_xyxy": [0., 0., 1., 1.]}
    encoder = FastEncoder(FastEncoderConfig(snapshot, encoder_manifest, encoder_manifest_sha256,
        batch_size=batch_size, preprocessing_mode="decoder_rgb_256",
        input_transform_sha256=decoder_recipe["sha256"]))
    started = time.monotonic()
    verification = encoder.load()
    entries = []
    for row in rows:
        asset = VideoAsset.model_validate(row["asset"])
        deadline = time.monotonic() + per_source_timeout_sec

        def source_factory(cancel, current_asset=asset, current_deadline=deadline):
            return iter_rgb_batches(current_asset, sample_fps=None, cancel=cancel,
                verify_source_hash=True, batch_size=batch_size, target_size=(256, 256),
                aspect_policy="letterbox", deadline_monotonic=current_deadline,
                max_runtime_sec=per_source_timeout_sec, max_source_frames=10000,
                max_output_frames=10000, use_subprocess=True)

        stream = iter_encoded_batches(source_factory, encoder, deadline=deadline)
        cache = destination / row["content_sha256"]
        if cache.exists():
            # A restart may reuse only an already complete identical source/recipe.
            from training.temporal.cache import load_feature_cache
            seq, _, existing = load_feature_cache(cache / "manifest.json")
            require(seq.content_sha256 == row["content_sha256"] and seq.group_id == row["group_id"]
                    and seq.split == row["split"]
                    and existing["feature_spec"]["encoder_recipe_sha256"] == encoder.recipe_sha256,
                    "feature_existing_cache_binding_mismatch")
            entry = {"manifest_path": str(cache / "manifest.json"),
                "manifest_sha256": file_sha256(cache / "manifest.json"),
                "content_sha256": seq.content_sha256, "group_id": seq.group_id, "split": seq.split,
                "feature_spec_sha256": existing["feature_spec_sha256"],
                "encoder_manifest_sha256": seq.encoder_manifest_sha256,
                "frame_count": len(seq.frames), "cache_complete": True}
        else:
            entry = write_feature_cache(cache, asset, stream, encoder.recipe, decoder_recipe,
                sampling_recipe, crop_recipe, row["group_id"], row["split"])
        entries.append(entry)
    index = {"schema_version": "forte.private-feature-cache-index.v1",
        "input_manifest_sha256": expected_inputs_sha256, "assignments_sha256": payload["assignments_sha256"],
        "entries": entries, "all_native_frames": True, "final_check_excluded": True,
        "encoder_manifest_sha256": encoder_manifest_sha256}
    _write_json(destination / "index.json", index)
    return {"status": "done", "sources": len(entries), "frames": sum(row["frame_count"] for row in entries),
        "split_counts": dict(Counter(row["split"] for row in entries)),
        "wall_seconds": time.monotonic() - started, "gpu_used": True,
        "index_sha256": file_sha256(destination / "index.json"),
        "model_files_verified": verification.get("verified", verification.get("local_model_loaded", False))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--worklist", required=True, type=Path)
    prepare.add_argument("--split", required=True, type=Path)
    prepare.add_argument("--expected-worklist-sha256", required=True)
    prepare.add_argument("--expected-split-sha256", required=True)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--archive", required=True, type=Path)
    extract = commands.add_parser("extract")
    extract.add_argument("--inputs", required=True, type=Path)
    extract.add_argument("--expected-inputs-sha256", required=True)
    extract.add_argument("--destination", required=True, type=Path)
    extract.add_argument("--snapshot", required=True, type=Path)
    extract.add_argument("--encoder-manifest", required=True, type=Path)
    extract.add_argument("--encoder-manifest-sha256", required=True)
    extract.add_argument("--batch-size", type=int, default=32)
    extract.add_argument("--per-source-timeout-sec", type=float, default=120.)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare_sources(args.worklist, args.split, args.output, args.archive,
                expected_worklist_sha256=args.expected_worklist_sha256,
                expected_split_sha256=args.expected_split_sha256)
        else:
            result = extract_sources(args.inputs, args.destination, expected_inputs_sha256=args.expected_inputs_sha256,
                snapshot=args.snapshot, encoder_manifest=args.encoder_manifest,
                encoder_manifest_sha256=args.encoder_manifest_sha256, batch_size=args.batch_size,
                per_source_timeout_sec=args.per_source_timeout_sec)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        # Never print a decoder/ML exception containing private source paths.
        code = str(exc) if isinstance(exc, TrainingGateError) else "feature_extract_or_prepare_failed"
        print(json.dumps({"status": "failed", "reason_code": code, "error_type": type(exc).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
