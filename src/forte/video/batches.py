"""Optional, bounded RGB batches with exact source identities and no JPEG cache.

This module is additive: the established JPEG decoder/evidence contracts retain
their behavior. Feature pixels stay in local memory. Callers must exhaust or
close readers; ``contextlib.closing`` also cancels a partly consumed reader.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import itertools
import json
import math
import multiprocessing
from pathlib import Path
import re
import time
from typing import Any, Literal

from forte.shared.schemas import FrameRef, VideoAsset
from forte.video.decode import decode_frames, frame_identity
from forte.video.probe import (
    CancelSignal, VideoCancelledError, VideoDeadlineError, VideoDecodeError,
    VideoError, VideoLimitError, check_cancelled, content_sha256,
    local_media_path, operation_slot,
)


def source_frame_id(content_sha256: str, source_pts: int, time_base_num: int, time_base_den: int) -> str:
    """Pure canonical identity, identical to ``decode.frame_identity``.

    No fabricated VideoAsset or JPEG path is needed at a feature-cache boundary.
    Reduced rational units are required; this function never rescales PTS.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", content_sha256):
        raise ValueError("content_sha256 must be a lowercase SHA-256 digest")
    if type(source_pts) is not int or type(time_base_num) is not int or type(time_base_den) is not int:
        raise ValueError("Source PTS and time-base components must be integers")
    if time_base_num <= 0 or time_base_den <= 0 or math.gcd(time_base_num, time_base_den) != 1:
        raise ValueError("Time base must be positive and reduced")
    value = f"{content_sha256}:{source_pts}:{time_base_num}/{time_base_den}"
    return hashlib.sha256(value.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class SourceFrameIdentity:
    frame_id: str
    content_hash: str
    job_id: str
    stream_index: int
    source_pts: int
    time_base: str
    time_base_num: int
    time_base_den: int
    source_origin_pts: int
    pts_sec: float
    next_source_pts: int | None
    next_pts_sec: float | None
    source_end_bound_sec: float | None
    width: int
    height: int
    applied_rotation_deg: int
    key_frame: bool

    @property
    def observation_interval_sec(self) -> float:
        end = self.next_pts_sec if self.next_pts_sec is not None else self.source_end_bound_sec
        if end is None:
            raise VideoDecodeError("Source frame lacks an explicit observation end bound")
        return max(0.0, end - self.pts_sec)


@dataclass(frozen=True, slots=True)
class FrameTransform:
    source_width: int
    source_height: int
    resized_width: int
    resized_height: int
    target_width: int
    target_height: int
    pad_left: int
    pad_top: int
    applied_rotation_deg: int
    source_box_xyxy: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    view_id: str = "global"


@dataclass(frozen=True, slots=True)
class RGBFrameBatch:
    frames: tuple[SourceFrameIdentity, ...]
    pixels: bytes = field(repr=False)
    width: int
    height: int
    aspect_policy: Literal["letterbox", "stretch"]
    transform_sha256: str
    transforms: tuple[FrameTransform, ...]
    selection_mode: Literal["native", "rate", "boundary"]
    requested_count: int | None
    delivered_count: int
    total_delivered_count: int
    decoded_source_frame_count: int
    segment_id: str
    request_ids: tuple[str, ...]
    segment_complete: bool
    source_complete: bool
    tail_included: bool

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return len(self.frames), self.height, self.width, 3

    def frame_pixels(self, index: int) -> memoryview:
        if not 0 <= index < len(self.frames):
            raise IndexError(index)
        stride = self.width * self.height * 3
        return memoryview(self.pixels)[index * stride:(index + 1) * stride]


@dataclass(frozen=True, slots=True)
class BoundaryReadRequest:
    request_id: str
    lower_source_pts: int
    upper_source_pts: int
    required_fact_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BoundaryReadGroup:
    group_id: str
    lower_source_pts: int
    upper_source_pts: int
    request_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Config:
    batch_size: int
    target_size: tuple[int, int]
    aspect_policy: Literal["letterbox", "stretch"]
    deadline_monotonic: float
    max_batch_bytes: int
    max_source_frames: int
    max_output_frames: int
    max_pixels: int
    thread_count: int


@dataclass(slots=True)
class _Progress:
    decoded: int = 0
    delivered: int = 0


def rgb_transform_recipe(*, target_size: tuple[int, int] = (256, 256),
                         aspect_policy: str = "letterbox") -> dict:
    """Version the external RGB geometry separately from tower normalization."""
    import av
    from PIL import __version__ as pillow_version
    if len(target_size) != 2 or any(type(value) is not int or value <= 0 for value in target_size):
        raise ValueError("target_size must contain two positive integers")
    if aspect_policy not in ("letterbox", "stretch"):
        raise ValueError("Unknown RGB aspect policy")
    recipe = {
        "schema_version": "1.0", "implementation": "forte.video.batches.rgb_geometry.v1",
        "target_width": target_size[0], "target_height": target_size[1],
        "aspect_policy": aspect_policy, "interpolation": "libswscale.BILINEAR",
        "rgb_conversion": "libswscale/source_color_metadata", "letterbox_rgb": [0, 0, 0],
        "rotation": "source_orthogonal_rotation_applied", "layout": "NHWC_RGB_uint8",
        "normalization": "none; encoder must apply its versioned rescale/mean/std",
        "pyav": av.__version__, "libswscale": list(av.library_versions.get("libswscale", ())),
        "pillow": pillow_version,
    }
    recipe["sha256"] = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return recipe


def _asset_base(asset: VideoAsset) -> Fraction:
    try:
        base = Fraction(asset.time_base)
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        raise VideoDecodeError("Video asset has no valid source time base") from exc
    if base <= 0 or asset.time_base != f"{base.numerator}/{base.denominator}":
        raise VideoDecodeError("Video asset time base must retain its canonical source units")
    return base


def validate_source_identity(asset: VideoAsset, source: SourceFrameIdentity) -> None:
    base = _asset_base(asset)
    integers = (source.source_pts, source.source_origin_pts, source.stream_index,
                source.time_base_num, source.time_base_den)
    if any(type(value) is not int for value in integers) or (source.next_source_pts is not None and type(source.next_source_pts) is not int):
        raise VideoDecodeError("In-memory frame source ticks must be integers")
    if (source.content_hash, source.job_id, source.stream_index, source.time_base) != (
        asset.content_hash, asset.job_id, asset.stream_index, asset.time_base
    ):
        raise VideoDecodeError("In-memory frame belongs to another source or job")
    if (source.time_base_num, source.time_base_den) != (base.numerator, base.denominator):
        raise VideoDecodeError("In-memory frame time base was rescaled")
    if source.frame_id != frame_identity(asset, source.source_pts):
        raise VideoDecodeError("In-memory frame identity does not match its source PTS")
    if (source.width, source.height, source.applied_rotation_deg) != (
        asset.width, asset.height, asset.rotation_deg
    ):
        raise VideoDecodeError("In-memory frame dimensions or rotation do not match its source")
    if not math.isclose(float(source.source_origin_pts * base), asset.source_start_sec, abs_tol=1e-7, rel_tol=0):
        raise VideoDecodeError("In-memory frame source origin does not match the video")
    expected = float((source.source_pts - source.source_origin_pts) * base)
    if not math.isfinite(source.pts_sec) or not math.isclose(source.pts_sec, expected, abs_tol=1e-7, rel_tol=0):
        raise VideoDecodeError("In-memory frame normalized time does not match its source PTS")
    if not 0 <= expected <= asset.duration_sec + 1e-7:
        raise VideoDecodeError("In-memory frame lies outside the source timeline")
    if source.next_source_pts is not None:
        expected_next = float((source.next_source_pts - source.source_origin_pts) * base)
        if (source.next_source_pts <= source.source_pts or source.next_pts_sec is None
            or not math.isclose(source.next_pts_sec, expected_next, abs_tol=1e-7, rel_tol=0)
            or source.source_end_bound_sec is not None or expected_next > asset.duration_sec + 1e-7):
            raise VideoDecodeError("In-memory frame has an invalid native next-PTS bound")
    elif (source.next_pts_sec is not None or source.source_end_bound_sec is None
          or not math.isclose(source.source_end_bound_sec, asset.duration_sec, abs_tol=1e-7, rel_tol=0)):
        raise VideoDecodeError("Final source frame needs a distinct video-end bound")


def _configuration(
    *, batch_size: int = 32, target_size: tuple[int, int] = (256, 256),
    aspect_policy: Literal["letterbox", "stretch"] = "letterbox",
    deadline_monotonic: float | None = None, max_runtime_sec: float = 30.0,
    max_batch_bytes: int = 32 * 1024 * 1024, max_source_frames: int = 100_000,
    max_output_frames: int = 20_000, max_pixels: int = 33_554_432, thread_count: int = 4,
) -> _Config:
    check_cancelled(deadline_monotonic, None)
    if not math.isfinite(max_runtime_sec) or max_runtime_sec <= 0:
        raise ValueError("max_runtime_sec must be positive and finite")
    integers = (batch_size, max_batch_bytes, max_source_frames, max_output_frames, max_pixels, thread_count, *target_size)
    if any(type(value) is not int or value <= 0 for value in integers):
        raise ValueError("Batch dimensions and limits must be positive integers")
    if len(target_size) != 2 or batch_size > 256 or thread_count > 8:
        raise ValueError("Unsupported batch dimensions or codec thread count")
    if aspect_policy not in ("letterbox", "stretch"):
        raise ValueError("aspect_policy must be letterbox or stretch")
    if batch_size * target_size[0] * target_size[1] * 3 > max_batch_bytes:
        raise VideoLimitError("RGB batch bytes exceed the configured memory limit")
    deadline = min(deadline_monotonic if deadline_monotonic is not None else math.inf,
                   time.monotonic() + max_runtime_sec)
    return _Config(batch_size, target_size, aspect_policy, deadline, max_batch_bytes,
                   max_source_frames, max_output_frames, max_pixels, thread_count)


def _open_stream(container: Any, asset: VideoAsset, config: _Config) -> tuple[Any, int, Fraction]:
    if not container.streams.video:
        raise VideoDecodeError("Media has no video stream")
    stream = container.streams.video[0]
    base = _asset_base(asset)
    if stream.index != asset.stream_index or not stream.time_base:
        raise VideoDecodeError("Probed video stream does not match the decoded stream")
    if Fraction(stream.time_base.numerator, stream.time_base.denominator) != base:
        raise VideoDecodeError("Source stream time base changed")
    if stream.start_time is None:
        raise VideoDecodeError("Source stream has no integer presentation origin")
    origin = int(stream.start_time)
    if not math.isclose(float(origin * base), asset.source_start_sec, abs_tol=1e-7, rel_tol=0):
        raise VideoDecodeError("Source stream presentation origin changed")
    stream.codec_context.thread_count = config.thread_count
    stream.codec_context.thread_type = "FRAME"
    return stream, origin, base


def _pairs(container: Any, stream: Any, asset: VideoAsset, config: _Config,
           progress: _Progress, cancel: CancelSignal) -> Iterator[tuple[Any, int | None]]:
    base = _asset_base(asset)
    previous = None
    raw_dimensions = (asset.height, asset.width) if asset.rotation_deg in (90, 270) else (asset.width, asset.height)
    for frame in container.decode(stream):
        check_cancelled(config.deadline_monotonic, cancel)
        if frame.pts is None or not frame.time_base or frame.is_corrupt:
            raise VideoDecodeError("Decoded source frame lacks trustworthy presentation metadata")
        if Fraction(frame.time_base.numerator, frame.time_base.denominator) != base:
            raise VideoDecodeError("Decoded source frame changed time-base units")
        if (frame.width, frame.height) != raw_dimensions:
            raise VideoDecodeError("Decoded source dimensions changed")
        if frame.width * frame.height > config.max_pixels:
            raise VideoLimitError("Decoded source dimensions exceed configured limits")
        if previous is not None and frame.pts <= previous.pts:
            raise VideoDecodeError("Decoded source PTS are not strictly increasing")
        progress.decoded += 1
        if progress.decoded > config.max_source_frames:
            raise VideoLimitError("Decoded source frame count exceeds configured limits")
        if previous is not None:
            yield previous, int(frame.pts)
        previous = frame
    if previous is None:
        raise VideoDecodeError("Source contains no decodable video frames")
    yield previous, None


def _identity(frame: Any, next_pts: int | None, asset: VideoAsset, origin: int, base: Fraction) -> SourceFrameIdentity:
    source = SourceFrameIdentity(
        frame_id=frame_identity(asset, int(frame.pts)), content_hash=asset.content_hash,
        job_id=asset.job_id, stream_index=asset.stream_index, source_pts=int(frame.pts),
        time_base=asset.time_base, time_base_num=base.numerator, time_base_den=base.denominator,
        source_origin_pts=origin, pts_sec=float((frame.pts - origin) * base),
        next_source_pts=next_pts,
        next_pts_sec=float((next_pts - origin) * base) if next_pts is not None else None,
        source_end_bound_sec=asset.duration_sec if next_pts is None else None,
        width=asset.width, height=asset.height, applied_rotation_deg=asset.rotation_deg,
        key_frame=bool(frame.key_frame),
    )
    validate_source_identity(asset, source)
    return source


def _full_records(container: Any, stream: Any, asset: VideoAsset, origin: int, base: Fraction,
                  config: _Config, progress: _Progress, cancel: CancelSignal,
                  sample_fps: Fraction | None) -> Iterator[tuple[Any, SourceFrameIdentity]]:
    pairs = _pairs(container, stream, asset, config, progress, cancel)
    previous = None
    previous_meta = None
    target_index = 0
    last_emitted = None
    target_count = math.ceil(Fraction(str(asset.duration_sec)) * sample_fps) if sample_fps else 0
    if sample_fps and target_count + 1 > config.max_output_frames:
        raise VideoLimitError("Requested RGB rate samples exceed configured output limits")
    first = True
    for frame, next_pts in pairs:
        source = _identity(frame, next_pts, asset, origin, base)
        if first and frame.pts != origin:
            raise VideoDecodeError("Full-source decode did not begin at the presentation origin")
        first = False
        if sample_fps is None:
            yield frame, source
        else:
            current_seconds = Fraction(frame.pts - origin) * base
            while target_index < target_count and Fraction(target_index, 1) / sample_fps <= current_seconds:
                target = Fraction(target_index, 1) / sample_fps
                chosen, chosen_meta = frame, source
                if previous is not None:
                    previous_seconds = Fraction(previous.pts - origin) * base
                    if target - previous_seconds <= current_seconds - target:
                        chosen, chosen_meta = previous, previous_meta
                if chosen_meta.frame_id != last_emitted:
                    yield chosen, chosen_meta
                    last_emitted = chosen_meta.frame_id
                target_index += 1
            previous, previous_meta = frame, source
    if asset.frame_count is not None and progress.decoded != asset.frame_count:
        raise VideoDecodeError("Full-source decode does not match its declared native frame count")
    if sample_fps is not None and previous_meta is not None and previous_meta.frame_id != last_emitted:
        # The actual final native frame is always selected, independent of rate.
        yield previous, previous_meta


def _pixels(frame: Any, source: SourceFrameIdentity, config: _Config) -> tuple[bytes, FrameTransform]:
    from PIL import Image
    target_width, target_height = config.target_size
    if config.aspect_policy == "letterbox":
        scale = min(target_width / source.width, target_height / source.height)
        resized_width = max(1, min(target_width, round(source.width * scale)))
        resized_height = max(1, min(target_height, round(source.height * scale)))
    else:
        resized_width, resized_height = target_width, target_height
    coded_size = ((resized_height, resized_width) if source.applied_rotation_deg in (90, 270)
                  else (resized_width, resized_height))
    resized = frame.reformat(width=coded_size[0], height=coded_size[1], format="rgb24", threads=1)
    left, top = (target_width - resized_width) // 2, (target_height - resized_height) // 2
    transform = FrameTransform(source.width, source.height, resized_width, resized_height,
                               target_width, target_height, left, top, source.applied_rotation_deg)
    plane = resized.planes[0]
    if (not source.applied_rotation_deg and (resized_width, resized_height) == config.target_size
        and plane.line_size == target_width * 3):
        # No Pillow roundtrip or padded-row copy for a tight native RGB surface.
        return bytes(memoryview(plane)[:target_width * target_height * 3]), transform
    image = resized.to_image()
    try:
        if source.applied_rotation_deg:
            operations = {90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180,
                          270: Image.Transpose.ROTATE_270}
            upright = image.transpose(operations[source.applied_rotation_deg])
            image.close()
            image = upright
        if image.size == config.target_size:
            data = image.tobytes()
        else:
            with Image.new("RGB", config.target_size, (0, 0, 0)) as canvas:
                canvas.paste(image, (left, top))
                data = canvas.tobytes()
    finally:
        image.close()
    return data, transform


def _batches(records: Iterator[tuple[Any, SourceFrameIdentity]], config: _Config,
             progress: _Progress, cancel: CancelSignal, *, mode: str,
             requested_count: int | None, segment_id: str, request_ids: tuple[str, ...],
             full_source: bool) -> Iterator[RGBFrameBatch]:
    transform_sha = rgb_transform_recipe(target_size=config.target_size, aspect_policy=config.aspect_policy)["sha256"]
    segment_start_count = progress.delivered
    pending = next(records, None)
    if pending is None:
        raise VideoDecodeError("Requested source segment contains no frames")
    while pending is not None:
        metadata, chunks, transforms = [], [], []
        while pending is not None and len(metadata) < config.batch_size:
            check_cancelled(config.deadline_monotonic, cancel)
            frame, source = pending
            pixels, transform = _pixels(frame, source, config)
            metadata.append(source)
            chunks.append(pixels)
            transforms.append(transform)
            progress.delivered += 1
            if progress.delivered > config.max_output_frames:
                raise VideoLimitError("Selected RGB frame count exceeds configured limits")
            # Drive one-record lookahead before declaring a batch complete. This
            # also runs segment's exact-PTS checks before its final batch is sent.
            pending = next(records, None)
        final = pending is None
        data = b"".join(chunks)
        if len(data) > config.max_batch_bytes:
            raise VideoLimitError("RGB batch exceeds configured memory limits")
        yield RGBFrameBatch(
            frames=tuple(metadata), pixels=data, width=config.target_size[0], height=config.target_size[1],
            aspect_policy=config.aspect_policy, transform_sha256=transform_sha,
            transforms=tuple(transforms), selection_mode=mode,
            requested_count=requested_count, delivered_count=progress.delivered - segment_start_count,
            total_delivered_count=progress.delivered,
            decoded_source_frame_count=progress.decoded, segment_id=segment_id,
            request_ids=request_ids, segment_complete=final,
            source_complete=final and full_source,
            tail_included=final and metadata[-1].next_source_pts is None,
        )


def group_boundary_requests(asset: VideoAsset, requests: Sequence[BoundaryReadRequest], *,
                            max_groups: int = 64, max_refine_span_sec: float = 30.0) -> tuple[BoundaryReadGroup, ...]:
    base = _asset_base(asset)
    if max_groups < 1 or not math.isfinite(max_refine_span_sec) or max_refine_span_sec <= 0:
        raise ValueError("Boundary limits must be positive")
    if len(requests) > 4096:
        raise VideoLimitError("Boundary request count exceeds configured limits")
    if len({request.request_id for request in requests}) != len(requests):
        raise ValueError("Boundary request IDs must be unique")
    origin = round(asset.source_start_sec / float(base))
    end = origin + Fraction(str(asset.duration_sec)) / base
    merged: list[tuple[int, int, list[str]]] = []
    for request in sorted(requests, key=lambda item: (item.lower_source_pts, item.upper_source_pts)):
        lower, upper = request.lower_source_pts, request.upper_source_pts
        if (not request.request_id or len(request.request_id) > 256 or type(lower) is not int
            or type(upper) is not int or lower > upper or lower < origin or upper > end + 1e-6):
            raise ValueError("Boundary request must use source ticks inside the video")
        if merged and lower <= merged[-1][1]:
            prior_lower, prior_upper, ids = merged[-1]
            merged[-1] = prior_lower, max(upper, prior_upper), [*ids, request.request_id]
        else:
            merged.append((lower, upper, [request.request_id]))
    if len(merged) > max_groups or sum(float((upper - lower) * base) for lower, upper, _ in merged) > max_refine_span_sec:
        raise VideoLimitError("Complete boundary refinement exceeds configured limits")
    return tuple(BoundaryReadGroup(
        hashlib.sha256(f"{asset.content_hash}:{lower}:{upper}".encode("ascii")).hexdigest(),
        lower, upper, tuple(ids),
    ) for lower, upper, ids in merged)


def _source_index(container: Any, stream: Any, asset: VideoAsset, config: _Config,
                  cancel: CancelSignal, source_index: Sequence[SourceFrameIdentity] | None) -> tuple[list[int], list[int]]:
    if source_index is not None:
        if not source_index or len(source_index) > config.max_source_frames:
            raise VideoLimitError("Native source index is empty or exceeds configured limits")
        for index, source in enumerate(source_index):
            validate_source_identity(asset, source)
            if index + 1 < len(source_index) and source.next_source_pts != source_index[index + 1].source_pts:
                raise VideoDecodeError("Boundary source index is not a complete native-PTS chain")
        if source_index[-1].next_source_pts is not None:
            raise VideoDecodeError("Boundary source index does not include the actual source tail")
        if source_index[0].source_pts != source_index[0].source_origin_pts:
            raise VideoDecodeError("Boundary source index does not include the actual source start")
        if asset.frame_count is not None and len(source_index) != asset.frame_count:
            raise VideoDecodeError("Boundary source index does not match the declared native frame count")
        return [source.source_pts for source in source_index], [source.source_pts for source in source_index if source.key_frame]
    pts, keyframes = [], []
    # Metadata-only scan of compressed video packets, not another pixel decode.
    for packet in container.demux(stream):
        check_cancelled(config.deadline_monotonic, cancel)
        if not packet.size:
            continue
        if packet.pts is None:
            raise VideoDecodeError("Boundary packet index has missing presentation timestamps")
        pts.append(int(packet.pts))
        if packet.is_keyframe:
            keyframes.append(int(packet.pts))
        if len(pts) > config.max_source_frames:
            raise VideoLimitError("Boundary packet index exceeds configured limits")
    if not pts or len(set(pts)) != len(pts):
        raise VideoDecodeError("Boundary packet index lacks unique source timestamps")
    return sorted(pts), sorted(keyframes)


def _boundary_records(container: Any, stream: Any, asset: VideoAsset, origin: int, base: Fraction,
                      config: _Config, progress: _Progress, cancel: CancelSignal,
                      group: BoundaryReadGroup, index: list[int], keyframes: list[int]) -> Iterator[tuple[Any, SourceFrameIdentity]]:
    expected = [pts for pts in index if group.lower_source_pts <= pts <= group.upper_source_pts]
    if not expected:
        raise VideoDecodeError("Boundary bracket contains no native source frame")
    anchor = max((pts for pts in keyframes if pts <= group.lower_source_pts), default=origin)
    records = None
    first = None
    for target in dict.fromkeys((anchor, origin)):
        container.seek(target, stream=stream, backward=True, any_frame=False)
        records = _pairs(container, stream, asset, config, progress, cancel)
        first = next(records, None)
        if first is not None and first[0].pts <= expected[0]:
            break
        records.close()
        first = None
    if first is None or records is None:
        raise VideoDecodeError("Keyframe seek skipped a required source frame")
    observed = 0
    for frame, next_pts in itertools.chain((first,), records):
        if frame.pts < group.lower_source_pts:
            continue
        if frame.pts > group.upper_source_pts:
            break
        if observed >= len(expected) or frame.pts != expected[observed]:
            raise VideoDecodeError("Boundary decode did not reproduce the exact native PTS set")
        observed += 1
        yield frame, _identity(frame, next_pts, asset, origin, base)
    if observed != len(expected):
        raise VideoDecodeError("Boundary decode omitted a required native source frame")


def _local_batches(asset: VideoAsset, config: _Config, cancel: CancelSignal, *,
                   sample_fps: Fraction | None, groups: tuple[BoundaryReadGroup, ...] | None,
                   source_index: Sequence[SourceFrameIdentity] | None) -> Iterator[RGBFrameBatch]:
    import av
    progress = _Progress()
    try:
        with av.open(str(asset.owned_path), options={"protocol_whitelist": "file,pipe"}) as container:
            stream, origin, base = _open_stream(container, asset, config)
            if groups is None:
                requested = asset.frame_count if sample_fps is None else math.ceil(Fraction(str(asset.duration_sec)) * sample_fps) + 1
                records = _full_records(container, stream, asset, origin, base, config, progress, cancel, sample_fps)
                yield from _batches(records, config, progress, cancel,
                                    mode="native" if sample_fps is None else "rate", requested_count=requested,
                                    segment_id="full-source", request_ids=(), full_source=True)
            else:
                pts, keyframes = _source_index(container, stream, asset, config, cancel, source_index)
                for group in groups:
                    records = _boundary_records(container, stream, asset, origin, base, config, progress,
                                                cancel, group, pts, keyframes)
                    requested = sum(group.lower_source_pts <= value <= group.upper_source_pts for value in pts)
                    yield from _batches(records, config, progress, cancel, mode="boundary", requested_count=requested,
                                        segment_id=group.group_id, request_ids=group.request_ids, full_source=False)
    except (VideoError, ValueError):
        raise
    except Exception as exc:
        raise VideoDecodeError("RGB batch source decoding failed") from exc


def _batch_process_entry(connection: Any, asset: VideoAsset, config: _Config, arguments: dict) -> None:
    def send(message: tuple) -> bool:
        try:
            connection.send(message)
            return True
        except (BrokenPipeError, EOFError, OSError):
            # Closing a partly consumed reader deliberately closes this pipe.
            return False
    try:
        for batch in _local_batches(asset, config, None, **arguments):
            if not send(("batch", batch)):
                return
        send(("done", None))
    except VideoError as exc:
        send((type(exc).__name__, str(exc)))
    except Exception:
        send(("VideoDecodeError", "RGB batch subprocess failed"))
    finally:
        connection.close()


def _read_batches(asset: VideoAsset, config: _Config, cancel: CancelSignal, *, use_subprocess: bool,
                  verify_source_hash: bool, sample_fps: Fraction | None,
                  groups: tuple[BoundaryReadGroup, ...] | None,
                  source_index: Sequence[SourceFrameIdentity] | None) -> Iterator[RGBFrameBatch]:
    check_cancelled(config.deadline_monotonic, cancel)
    _asset_base(asset)
    path = local_media_path(asset.owned_path)
    if verify_source_hash and content_sha256(path, deadline_monotonic=config.deadline_monotonic, cancel=cancel) != asset.content_hash:
        raise VideoDecodeError("Owned video content has changed")
    arguments = dict(sample_fps=sample_fps, groups=groups, source_index=source_index)
    with operation_slot(config.deadline_monotonic, cancel):
        if not use_subprocess:
            yield from _local_batches(asset, config, cancel, **arguments)
            return
        context = multiprocessing.get_context("spawn")
        receiving, sending = context.Pipe(duplex=False)
        process = context.Process(target=_batch_process_entry, args=(sending, asset, config, arguments), daemon=True)
        try:
            process.start()
            sending.close()
            while True:
                while not receiving.poll(0.05):
                    check_cancelled(config.deadline_monotonic, cancel)
                    if not process.is_alive():
                        raise VideoDecodeError("RGB batch subprocess exited without completion")
                check_cancelled(config.deadline_monotonic, cancel)
                kind, value = receiving.recv()
                if kind == "done":
                    break
                if kind != "batch":
                    error_type = {"VideoDeadlineError": VideoDeadlineError, "VideoCancelledError": VideoCancelledError,
                                  "VideoLimitError": VideoLimitError}.get(kind, VideoDecodeError)
                    raise error_type(value)
                if not isinstance(value, RGBFrameBatch) or len(value.pixels) > config.max_batch_bytes:
                    raise VideoDecodeError("RGB batch subprocess returned invalid bounded data")
                yield value
        except EOFError as exc:
            raise VideoDecodeError("RGB batch subprocess failed") from exc
        finally:
            receiving.close()
            sending.close()
            if process.pid is not None:
                process.join(timeout=0.1)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1)
                process.close()


def iter_rgb_batches(
    asset: VideoAsset, *, sample_fps: float | Fraction | None = None,
    cancel: CancelSignal = None, use_subprocess: bool = True, verify_source_hash: bool = False,
    **options: Any,
) -> Iterator[RGBFrameBatch]:
    """Read native or nearest-rate RGB batches; always include the actual tail.

    Pixel bytes are tight RGB uint8 with ``batch.shape == (B,H,W,3)``. Rotation
    is already applied. Source identities carry no imaginary file paths. Default
    process isolation permits hard cancellation; an outer supervised decode
    process may use ``use_subprocess=False`` to avoid pipe copies. Close readers
    that are not exhausted. Options are the explicit bounds in ``_configuration``.
    """
    config = _configuration(**options)
    rate = None
    if sample_fps is not None:
        try:
            rate = sample_fps if isinstance(sample_fps, Fraction) else Fraction(str(sample_fps))
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError("sample_fps must be positive and finite") from exc
        if rate <= 0:
            raise ValueError("sample_fps must be positive and finite")
    return _read_batches(asset, config, cancel, use_subprocess=use_subprocess,
                         verify_source_hash=verify_source_hash, sample_fps=rate,
                         groups=None, source_index=None)


def iter_boundary_batches(
    asset: VideoAsset, requests: Sequence[BoundaryReadRequest], *,
    source_index: Sequence[SourceFrameIdentity] | None = None,
    max_groups: int = 64, max_refine_span_sec: float = 30.0,
    cancel: CancelSignal = None, use_subprocess: bool = True, verify_source_hash: bool = False,
    **options: Any,
) -> Iterator[RGBFrameBatch]:
    """Merge overlapping brackets and read every native frame from keyframes.

    A supplied index must contain the full native chain. Otherwise a bounded
    metadata-only compressed-packet scan verifies exact boundary PTS coverage.
    Boundary completion does not imply full-video or visual/procedure coverage.
    """
    config = _configuration(**options)
    groups = group_boundary_requests(asset, requests, max_groups=max_groups,
                                      max_refine_span_sec=max_refine_span_sec)
    if not groups:
        return iter(())
    return _read_batches(asset, config, cancel, use_subprocess=use_subprocess,
                         verify_source_hash=verify_source_hash, sample_fps=None,
                         groups=groups, source_index=source_index)


def materialize_evidence_frames(
    asset: VideoAsset, sources: Sequence[SourceFrameIdentity], output_dir: str | Path,
    *, deadline_monotonic: float | None = None, cancel: CancelSignal = None,
    max_runtime_sec: float = 120.0, **decode_options: Any,
) -> list[FrameRef]:
    """Materialize selected identities through the established exact JPEG path.

    No path-bearing FrameRef is fabricated. Source identity/hash are validated,
    then the existing decoder creates actual owned JPEGs and exact IDs are
    required. Batch identities across events once. This conservative adapter
    retains one full forward source decode; grouped JPEG seeks are a later
    optimization, separate from the existing evidence contract.
    """
    for source in sources:
        validate_source_identity(asset, source)
    if not sources:
        return []
    if len(sources) > 4096:
        raise VideoLimitError("Evidence identity count exceeds configured limits")
    if not math.isfinite(max_runtime_sec) or max_runtime_sec <= 0:
        raise ValueError("max_runtime_sec must be positive and finite")
    check_cancelled(deadline_monotonic, cancel)
    deadline = min(deadline_monotonic if deadline_monotonic is not None else math.inf,
                   time.monotonic() + max_runtime_sec)
    path = local_media_path(asset.owned_path)
    if content_sha256(path, deadline_monotonic=deadline, cancel=cancel) != asset.content_hash:
        raise VideoDecodeError("Owned video content has changed")
    refs = decode_frames(asset, output_dir, timestamps_sec=sorted({source.pts_sec for source in sources}),
                         deadline_monotonic=deadline, cancel=cancel,
                         max_runtime_sec=max_runtime_sec, **decode_options)
    if {ref.frame_id for ref in refs} != {source.frame_id for source in sources}:
        raise VideoDecodeError("Evidence decoder did not reproduce exact source identities")
    return refs


__all__ = ["SourceFrameIdentity", "FrameTransform", "RGBFrameBatch", "BoundaryReadRequest",
           "BoundaryReadGroup", "source_frame_id", "validate_source_identity", "iter_rgb_batches",
           "group_boundary_requests", "iter_boundary_batches", "materialize_evidence_frames", "rgb_transform_recipe"]
