"""Private, source-bound frozen features; imports no ML runtime until requested.

A cache is one complete native-frame recording, never a positive label copied
to chunks. Labels and filenames are deliberately absent from tensor inputs.
"""
from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Iterable

from training.guards import (TrainingGateError, canonical_sha256, file_sha256,
                             load_json, require)
from training.temporal.feature_spec import build_feature_spec
from training.temporal.targets import FeatureFrame, FeatureSequence

SCHEMA = "forte.private-feature-cache.v1"
MAX_FRAMES = 10000
MAX_MANIFEST_BYTES = 32 * 1024 * 1024
MAX_TENSOR_BYTES = MAX_FRAMES * 768 * 2 + 1024 * 1024


def sequence_from_source_frames(asset: Any, frames: list[Any], *, group_id: str,
                                split: str, encoder_recipe: dict,
                                decoder_recipe: dict, sampling_recipe: dict,
                                crop_recipe: dict) -> tuple[FeatureSequence, dict, str]:
    """Validate every native PTS and global view without loading tensor packages."""
    require(0 < len(frames) <= MAX_FRAMES, "cache_frame_budget_exceeded")
    require(isinstance(group_id, str) and bool(group_id), "cache_group_missing")
    base = Fraction(asset.time_base)
    origin = frames[0].source_origin_pts
    require(type(origin) is int and abs(float(base * origin) - asset.source_start_sec) <= 1e-6,
            "cache_source_origin_mismatch")
    for index, frame in enumerate(frames):
        require(frame.content_hash == asset.content_hash and frame.job_id == asset.job_id
                and frame.stream_index == asset.stream_index, "cache_source_binding_mismatch")
        require(frame.time_base_num == base.numerator and frame.time_base_den == base.denominator
                and frame.source_origin_pts == origin, "cache_source_time_base_mismatch")
        require(math.isfinite(frame.pts_sec) and frame.pts_sec < asset.duration_sec,
                "cache_source_time_outside_recording")
        if index + 1 < len(frames):
            require(frame.next_source_pts == frames[index + 1].source_pts,
                    "cache_native_frame_gap")
        else:
            require(frame.next_source_pts is None and frame.next_pts_sec is None
                    and frame.source_end_bound_sec is not None
                    and abs(frame.source_end_bound_sec - asset.duration_sec) <= 1e-6,
                    "cache_original_tail_missing")
    decoder_sha, sampling_sha, crop_sha = map(canonical_sha256,
                                              (decoder_recipe, sampling_recipe, crop_recipe))
    spec, spec_sha = build_feature_spec(encoder_recipe, decoder_recipe_sha256=decoder_sha,
        sampling_recipe_sha256=sampling_sha, crop_recipe_sha256=crop_sha)
    sequence = FeatureSequence(asset.content_hash, group_id, split, base.numerator,
        base.denominator, origin, 768, encoder_recipe["encoder_manifest_sha256"],
        decoder_sha, sampling_sha, crop_sha,
        tuple(FeatureFrame.from_source_identity(frame, view_mask=(True,),
            view_roles=("global",), normalized_boxes=((0.0, 0.0, 1.0, 1.0),)) for frame in frames))
    sequence.validate()
    return sequence, spec, spec_sha


def _sequence_from_dict(raw: dict) -> FeatureSequence:
    require(isinstance(raw, dict) and isinstance(raw.get("frames"), (list, tuple))
            and len(raw["frames"]) <= MAX_FRAMES, "cache_sequence_invalid")
    data = dict(raw)
    frames = []
    for row in data.pop("frames"):
        item = dict(row)
        for key in ("view_mask", "view_roles"):
            item[key] = tuple(item[key])
        item["normalized_boxes"] = tuple(tuple(box) for box in item["normalized_boxes"])
        frames.append(FeatureFrame(**item))
    result = FeatureSequence(**data, frames=tuple(frames))
    result.validate()
    return result


def _verify_manifest(payload: dict) -> FeatureSequence:
    require(isinstance(payload, dict) and payload.get("schema_version") == SCHEMA,
            "cache_schema_invalid")
    unhashed = {key: value for key, value in payload.items() if key != "payload_sha256"}
    require(payload.get("payload_sha256") == canonical_sha256(unhashed), "cache_manifest_digest_mismatch")
    require(payload.get("tensor_file") == "features.safetensors", "cache_tensor_path_invalid")
    require(payload.get("feature_spec_sha256") == canonical_sha256(payload.get("feature_spec")),
            "cache_feature_spec_digest_mismatch")
    sequence = _sequence_from_dict(payload["sequence"])
    require(payload.get("tensor_shape") == [len(sequence.frames), 1, 768]
            and payload.get("tensor_dtype") == "float16", "cache_tensor_contract_invalid")
    spec = payload["feature_spec"]
    require(spec.get("encoder_recipe_sha256") == canonical_sha256(spec.get("encoder_recipe"))
            and spec["encoder_recipe"].get("encoder_manifest_sha256") == sequence.encoder_manifest_sha256
            and spec.get("decoder_recipe_sha256") == sequence.decoder_config_sha256
            and spec.get("sampling_recipe_sha256") == sequence.sampling_config_sha256
            and spec.get("crop_recipe_sha256") == sequence.crop_config_sha256,
            "cache_sequence_recipe_mismatch")
    coverage = payload.get("coverage", {})
    require(coverage.get("source_complete") is True and coverage.get("tail_included") is True
            and coverage.get("selection_mode") == "native"
            and coverage.get("decoded_source_frame_count") == len(sequence.frames)
            and coverage.get("delivered_count") == len(sequence.frames), "cache_coverage_incomplete")
    require(payload.get("content_sha256") == sequence.content_sha256
            and payload.get("group_id") == sequence.group_id and payload.get("split") == sequence.split,
            "cache_top_level_identity_mismatch")
    return sequence


def write_feature_cache(destination: Path, asset: Any, encoded_batches: Iterable[Any],
                        encoder_recipe: dict, decoder_recipe: dict, sampling_recipe: dict,
                        crop_recipe: dict, group_id: str, split: str) -> dict:
    """Consume A15 EncodedSourceBatch objects; atomically publish a complete cache.

    Only one global view is supported by this first runtime. The explicit limit
    is a failure, never truncation. A partial stream cannot publish a manifest.
    """
    import torch
    from safetensors.torch import save_file

    destination = Path(destination).resolve()
    require(not destination.exists(), "cache_destination_already_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_recipe = canonical_sha256(encoder_recipe)
    parts, frames, transforms = [], [], []
    final = None
    segment = None
    iterator = iter(encoded_batches)
    try:
        for batch in iterator:
            require(final is None or not final.source_complete, "cache_data_after_source_end")
            require(batch.views_per_frame == 1 and batch.selection_mode == "native",
                    "cache_native_global_view_required")
            require(batch.encoder_recipe_sha256 == expected_recipe, "cache_encoder_recipe_mismatch")
            require(segment is None or segment == batch.segment_id, "cache_segment_mismatch")
            segment = batch.segment_id
            count = len(batch.frames)
            require(count > 0 and len(batch.transforms) == count, "cache_batch_source_shape_invalid")
            require(len(frames) + count <= MAX_FRAMES, "cache_frame_budget_exceeded")
            values = batch.features
            require(isinstance(values, torch.Tensor) and tuple(values.shape) == (count, 768)
                    and values.dtype == torch.float16 and values.device.type == "cpu"
                    and bool(torch.isfinite(values).all()), "cache_feature_tensor_invalid")
            require(all(tuple(transform.source_box_xyxy) == (0.0, 0.0, 1.0, 1.0)
                        and transform.view_id == "global" for transform in batch.transforms),
                    "cache_global_geometry_invalid")
            frames.extend(batch.frames)
            transforms.extend(asdict(transform) for transform in batch.transforms)
            parts.append(values.detach().clone())
            require(batch.delivered_count == len(frames), "cache_delivered_count_mismatch")
            final = batch
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
    require(final is not None and final.source_complete and final.segment_complete
            and final.tail_included and final.decoded_source_frame_count == len(frames),
            "cache_coverage_incomplete")
    sequence, spec, spec_sha = sequence_from_source_frames(asset, frames, group_id=group_id,
        split=split, encoder_recipe=encoder_recipe, decoder_recipe=decoder_recipe,
        sampling_recipe=sampling_recipe, crop_recipe=crop_recipe)
    features = torch.cat(parts, dim=0).unsqueeze(1).contiguous()
    temporary = Path(tempfile.mkdtemp(prefix=".forte-cache-", dir=destination.parent)).resolve()
    require(temporary.parent == destination.parent and temporary.name.startswith(".forte-cache-"),
            "cache_temporary_root_invalid")
    tensor = temporary / "features.safetensors"
    manifest_path = temporary / "manifest.json"
    try:
        save_file({"features": features}, str(tensor), metadata={"schema_version": SCHEMA,
                  "feature_spec_sha256": spec_sha})
        payload = {"schema_version": SCHEMA, "content_sha256": sequence.content_sha256,
            "group_id": group_id, "split": split, "tensor_file": tensor.name,
            "tensor_sha256": file_sha256(tensor), "tensor_bytes": tensor.stat().st_size,
            "tensor_shape": list(features.shape), "tensor_dtype": "float16",
            "sequence": asdict(sequence), "transforms": transforms,
            "feature_spec": spec, "feature_spec_sha256": spec_sha,
            "coverage": {"source_complete": True, "tail_included": True, "selection_mode": "native",
                "decoded_source_frame_count": final.decoded_source_frame_count,
                "delivered_count": len(frames)},
            "feature_coverage_implies_visibility": False, "human_verified": False,
            "source_bytes_reverified_by_cache_writer": False}
        payload["payload_sha256"] = canonical_sha256(payload)
        _verify_manifest(payload)
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
        require(len(raw) <= MAX_MANIFEST_BYTES and payload["tensor_bytes"] <= MAX_TENSOR_BYTES,
                "cache_serialized_budget_exceeded")
        with manifest_path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            import os
            os.fsync(stream.fileno())
        # rename refuses an existing directory; completed caches are immutable.
        temporary.rename(destination)
        return {"manifest_path": str(destination / manifest_path.name),
            "manifest_sha256": file_sha256(destination / manifest_path.name),
            "content_sha256": sequence.content_sha256, "group_id": group_id, "split": split,
            "feature_spec_sha256": spec_sha, "encoder_manifest_sha256": sequence.encoder_manifest_sha256,
            "frame_count": len(frames), "tensor_shape": list(features.shape), "cache_complete": True}
    finally:
        # These are the only two files created in the verified temporary folder.
        if temporary.exists():
            tensor.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
            temporary.rmdir()


def load_feature_cache(manifest_path: Path) -> tuple[FeatureSequence, Any, dict]:
    """Hash/validate private metadata before loading finite CPU tensors."""
    manifest_path = Path(manifest_path).resolve()
    require(manifest_path.is_file() and manifest_path.stat().st_size <= MAX_MANIFEST_BYTES,
            "cache_manifest_missing_or_oversized")
    payload = load_json(manifest_path)
    sequence = _verify_manifest(payload)
    tensor = manifest_path.parent / "features.safetensors"
    require(not tensor.is_symlink() and tensor.is_file()
            and tensor.stat().st_size == payload.get("tensor_bytes")
            and tensor.stat().st_size <= MAX_TENSOR_BYTES, "cache_tensor_missing_or_oversized")
    require(file_sha256(tensor) == payload.get("tensor_sha256"), "cache_tensor_digest_mismatch")
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(str(tensor), framework="pt", device="cpu") as handle:
        require(list(handle.keys()) == ["features"] and handle.metadata() == {
            "schema_version": SCHEMA, "feature_spec_sha256": payload["feature_spec_sha256"]},
            "cache_tensor_embedded_identity_mismatch")
    # Own the small feature tensor rather than retain a mapped-file handle.
    # This also permits controlled corruption checks/cleanup on Windows.
    values = load_file(str(tensor), device="cpu")["features"].clone()
    require(tuple(values.shape) == tuple(payload["tensor_shape"]) and values.dtype == torch.float16
            and values.device.type == "cpu" and bool(torch.isfinite(values).all()),
            "cache_loaded_tensor_invalid")
    return sequence, values, payload
