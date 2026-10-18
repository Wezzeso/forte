"""Timestamp-correct, bounded local decoding and exact-source evidence.

PyAV reads video only. Sampling uses actual presentation timestamps, never the
frame index or nominal FPS. JPEGs contain upright original frames without crops,
overlays, temporal interpolation, or content-dependent modification.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
import hashlib
import math
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any
import uuid

from forte.shared.schemas import FrameRef, VideoAsset
from forte.video.probe import (
    CancelSignal, VideoCancelledError, VideoDeadlineError, VideoDecodeError,
    VideoError, VideoLimitError, VideoProbeError, check_cancelled,
    content_sha256, local_media_path, operation_slot, probe_video,
)


def frame_identity(asset: VideoAsset, source_pts: int) -> str:
    """Bind a source frame to immutable file identity and the stream time base."""
    value = f"{asset.content_hash}:{source_pts}:{asset.time_base}"
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def validate_frame_ref(asset: VideoAsset, frame: FrameRef) -> None:
    """Reject cross-video/job refs and timestamps that do not match source PTS."""
    if frame.source_pts is None or not asset.time_base:
        raise VideoDecodeError("Evidence reference lacks exact source timestamps")
    if frame.frame_id != frame_identity(asset, frame.source_pts):
        raise VideoDecodeError("Evidence frame does not belong to this video")
    if frame.job_id is not None and frame.job_id != asset.job_id:
        raise VideoDecodeError("Evidence frame does not belong to this job")
    if (frame.width, frame.height, frame.rotation_deg) != (asset.width, asset.height, asset.rotation_deg):
        raise VideoDecodeError("Evidence dimensions or rotation differ from the source")
    expected = float(frame.source_pts * Fraction(asset.time_base)) - asset.source_start_sec
    tolerance = max(1e-7, 4 * math.ulp(abs(expected)))
    if not math.isclose(frame.pts_sec, expected, rel_tol=0, abs_tol=tolerance):
        raise VideoDecodeError("Evidence timestamp does not match the source PTS")
    if not 0 <= frame.pts_sec <= asset.duration_sec + tolerance:
        raise VideoDecodeError("Evidence timestamp lies outside the video timeline")


def _target_seconds(
    asset: VideoAsset, timestamps_sec: Sequence[float] | None,
    sample_fps: float, max_frames: int,
) -> list[float]:
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if timestamps_sec is None:
        if not math.isfinite(sample_fps) or sample_fps <= 0:
            raise ValueError("sample_fps must be positive and finite")
        requested = asset.duration_sec * sample_fps
        if requested > max_frames:
            raise VideoLimitError("Requested frame count exceeds the configured limit")
        count = max(1, math.ceil(requested))
        if count > max_frames:
            raise VideoLimitError("Requested frame count exceeds the configured limit")
        values = [index / sample_fps for index in range(count)]
    else:
        if len(timestamps_sec) > max_frames:
            raise VideoLimitError("Requested frame count exceeds the configured limit")
        values = [float(value) for value in timestamps_sec]
    for value in values:
        if not math.isfinite(value) or value < 0 or value > asset.duration_sec + 1e-7:
            raise ValueError("Requested timestamps must lie inside the video timeline")
    return sorted(set(min(value, asset.duration_sec) for value in values))


def _write_frame(frame: Any, asset: VideoAsset, directory: Path, quality: int) -> FrameRef:
    pts = int(frame.pts)
    timestamp = float(pts * Fraction(asset.time_base)) - asset.source_start_sec
    if timestamp < -1e-7 or timestamp > asset.duration_sec + 1e-7:
        raise VideoDecodeError("Decoded presentation timestamp lies outside the video timeline")
    image = frame.to_image()
    if asset.rotation_deg:
        # Orthogonal rotations preserve every source pixel; no resampling.
        from PIL import Image
        transpose = {90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180,
                     270: Image.Transpose.ROTATE_270}
        image = image.transpose(transpose[asset.rotation_deg])
    identity = frame_identity(asset, pts)
    destination = directory / f"{identity}.jpg"
    temporary = directory / f".{identity}.{uuid.uuid4().hex}.tmp"
    try:
        image.save(temporary, format="JPEG", quality=quality, subsampling=0)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
        image.close()
    return FrameRef(
        frame_id=identity, pts_sec=max(0.0, timestamp), path=str(destination),
        width=asset.width, height=asset.height, job_id=asset.job_id,
        rotation_deg=asset.rotation_deg, source_pts=pts,
        jpeg_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
    )


def _decode_local(
    asset: VideoAsset, directory: Path, targets: list[float], *,
    expected_ids: set[str] | None, deadline_monotonic: float,
    cancel: CancelSignal, thread_count: int, max_pixels: int, jpeg_quality: int,
) -> list[FrameRef]:
    try:
        import av
    except ImportError as exc:
        raise VideoDecodeError("PyAV is required for video decoding") from exc
    refs: dict[str, FrameRef] = {}
    position = 0
    previous: Any | None = None
    previous_time: float | None = None
    time_base = Fraction(asset.time_base)
    directory.mkdir(parents=True, exist_ok=True)

    def retain(frame: Any) -> None:
        identity = frame_identity(asset, int(frame.pts))
        if expected_ids is not None and identity not in expected_ids:
            raise VideoDecodeError("Exact evidence frame was not found at its timestamp")
        if identity not in refs:
            refs[identity] = _write_frame(frame, asset, directory, jpeg_quality)

    try:
        with av.open(str(asset.owned_path), mode="r", options={"protocol_whitelist": "file,pipe"}) as container:
            if not container.streams.video:
                raise VideoDecodeError("Media has no video stream")
            stream = container.streams.video[0]
            if stream.index != asset.stream_index or not stream.time_base:
                raise VideoDecodeError("Probed video stream does not match the decoded stream")
            if Fraction(stream.time_base.numerator, stream.time_base.denominator) != time_base:
                raise VideoDecodeError("Source stream time base changed")
            if stream.start_time is None or not math.isclose(
                float(stream.start_time * time_base), asset.source_start_sec,
                abs_tol=1e-7, rel_tol=0,
            ):
                raise VideoDecodeError("Source stream presentation origin changed")
            stream.codec_context.thread_count = thread_count
            stream.codec_context.thread_type = "SLICE"
            for frame in container.decode(stream):
                check_cancelled(deadline_monotonic, cancel)
                if frame.pts is None or not frame.time_base or frame.is_corrupt:
                    raise VideoDecodeError("Decoded frame lacks trustworthy presentation metadata")
                actual_base = Fraction(frame.time_base.numerator, frame.time_base.denominator)
                if actual_base != time_base:
                    raise VideoDecodeError("Decoded time base differs from probed video stream")
                if frame.width * frame.height > max_pixels:
                    raise VideoLimitError("Decoded frame dimensions exceed configured limits")
                if (frame.width, frame.height) != (
                    (asset.height, asset.width) if asset.rotation_deg in (90, 270)
                    else (asset.width, asset.height)
                ):
                    raise VideoDecodeError("Video dimensions changed during decoding")
                timestamp = float(frame.pts * actual_base) - asset.source_start_sec
                if previous_time is not None and timestamp <= previous_time:
                    raise VideoDecodeError("Video presentation timestamps are not strictly increasing")
                if timestamp < -1e-7:
                    raise VideoDecodeError("Video PTS precedes its probed presentation start")
                while position < len(targets) and targets[position] <= timestamp:
                    chosen = frame
                    if previous is not None and targets[position] - previous_time <= timestamp - targets[position]:
                        chosen = previous
                    retain(chosen)
                    position += 1
                previous, previous_time = frame, timestamp
                if position == len(targets):
                    break
            if previous is None:
                raise VideoDecodeError("Video contains no decodable frames")
            # A request at duration/end belongs to the last real frame's PTS.
            while position < len(targets):
                retain(previous)
                position += 1
    except VideoError:
        raise
    except Exception as exc:
        raise VideoDecodeError("Video frame decoding failed") from exc
    if expected_ids is not None and set(refs) != expected_ids:
        raise VideoDecodeError("One or more exact evidence frames are missing")
    return sorted(refs.values(), key=lambda reference: reference.pts_sec)


def _decode_process_entry(connection: Any, payload: dict) -> None:
    """Spawn entry point. Only metadata crosses the pipe, never image bytes."""
    try:
        asset = VideoAsset.model_validate(payload.pop("asset"))
        directory = Path(payload.pop("directory"))
        refs = _decode_local(asset, directory, cancel=None, **payload)
        connection.send(("ok", [ref.model_dump(mode="json") for ref in refs]))
    except VideoError as exc:
        connection.send((type(exc).__name__, str(exc)))
    except Exception:
        connection.send(("VideoDecodeError", "Video decoder subprocess failed"))
    finally:
        connection.close()


def _execute_decode(
    asset: VideoAsset, directory: Path, targets: list[float], *, expected_ids: set[str] | None,
    deadline_monotonic: float, cancel: CancelSignal, thread_count: int, max_pixels: int,
    jpeg_quality: int, use_subprocess: bool,
) -> list[FrameRef]:
    arguments = dict(
        targets=targets, expected_ids=expected_ids, deadline_monotonic=deadline_monotonic,
        thread_count=thread_count, max_pixels=max_pixels, jpeg_quality=jpeg_quality,
    )
    with operation_slot(deadline_monotonic, cancel):
        if not use_subprocess:
            return _decode_local(asset, directory, cancel=cancel, **arguments)
        context = multiprocessing.get_context("spawn")
        receiving, sending = context.Pipe(duplex=False)
        process = context.Process(
            target=_decode_process_entry,
            args=(sending, dict(asset=asset.model_dump(mode="json"), directory=str(directory), **arguments)),
            daemon=True,
        )
        try:
            process.start()
            sending.close()
            while not receiving.poll(0.05):
                check_cancelled(deadline_monotonic, cancel)
                if not process.is_alive():
                    raise VideoDecodeError("Video decoder subprocess exited without a result")
            check_cancelled(deadline_monotonic, cancel)
            status, result = receiving.recv()
            if status != "ok":
                error_type = {
                    "VideoCancelledError": VideoCancelledError, "VideoDeadlineError": VideoDeadlineError,
                    "VideoLimitError": VideoLimitError,
                }.get(status, VideoDecodeError)
                raise error_type(result)
            return [FrameRef.model_validate(ref) for ref in result]
        except EOFError as exc:
            raise VideoDecodeError("Video decoder subprocess failed") from exc
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


def decode_frames(
    video: str | os.PathLike[str] | VideoAsset,
    output_dir: str | os.PathLike[str],
    *,
    timestamps_sec: Sequence[float] | None = None,
    sample_fps: float = 1.0,
    job_id: str | None = None,
    deadline_monotonic: float | None = None,
    cancel: CancelSignal = None,
    max_runtime_sec: float = 120.0,
    max_frames: int = 4096,
    max_pixels: int = 33_554_432,
    thread_count: int = 2,
    jpeg_quality: int = 95,
    use_subprocess: bool = True,
) -> list[FrameRef]:
    """Decode sorted sparse samples in one forward pass and write owned JPEGs.

    Requests resolve to the nearest actual presentation frame; ties prefer the
    earlier frame. Repeated resolutions are deduplicated. The endpoint timestamp
    resolves to the final frame. Returned timestamps are actual normalized PTS,
    not requested times. At most two operations per process run concurrently.

    By default a child process enforces a hard wall-clock deadline, including a
    blocked codec. ``use_subprocess=False`` enables cooperative checks between
    frames for deployments whose outer worker already supplies process isolation.
    The caller owns/cleans the output directory, including any cancelled outputs.
    """
    if not math.isfinite(max_runtime_sec) or max_runtime_sec <= 0:
        raise ValueError("max_runtime_sec must be positive and finite")
    if not 1 <= thread_count <= 8 or not 1 <= jpeg_quality <= 100 or max_pixels <= 0:
        raise ValueError("Invalid decoder resource or JPEG settings")
    check_cancelled(deadline_monotonic, cancel)
    deadline = min(deadline_monotonic if deadline_monotonic is not None else math.inf,
                   time.monotonic() + max_runtime_sec)
    check_cancelled(deadline, cancel)
    asset = video if isinstance(video, VideoAsset) else probe_video(
        video, job_id=job_id, deadline_monotonic=deadline, cancel=cancel, max_pixels=max_pixels,
    )
    if job_id is not None and job_id != asset.job_id:
        raise VideoDecodeError("Video asset belongs to a different job")
    targets = _target_seconds(asset, timestamps_sec, sample_fps, max_frames)
    if not targets:
        return []
    directory = Path(output_dir).expanduser().resolve()
    if directory == Path(asset.owned_path).resolve():
        raise VideoDecodeError("Frame output directory overlaps the source file")
    return _execute_decode(
        asset, directory, targets, expected_ids=None, deadline_monotonic=deadline,
        cancel=cancel, thread_count=thread_count, max_pixels=max_pixels,
        jpeg_quality=jpeg_quality, use_subprocess=use_subprocess,
    )


def extract_evidence_frames(
    asset: VideoAsset,
    frame_refs: Sequence[FrameRef],
    output_dir: str | os.PathLike[str],
    *,
    deadline_monotonic: float | None = None,
    cancel: CancelSignal = None,
    max_runtime_sec: float = 120.0,
    thread_count: int = 2,
    max_pixels: int = 33_554_432,
    jpeg_quality: int = 95,
    use_subprocess: bool = True,
) -> list[FrameRef]:
    """Revalidate source identity and batch-retrieve exact original evidence.

    Frame paths from the request are never read. Regenerated JPEGs must match
    the requested source identities and timestamps; nearest substitutes fail.
    """
    if not math.isfinite(max_runtime_sec) or max_runtime_sec <= 0:
        raise ValueError("max_runtime_sec must be positive and finite")
    if not 1 <= thread_count <= 8 or not 1 <= jpeg_quality <= 100 or max_pixels <= 0:
        raise ValueError("Invalid decoder resource or JPEG settings")
    if len(frame_refs) > 4096:
        raise VideoLimitError("Evidence frame count exceeds the configured limit")
    check_cancelled(deadline_monotonic, cancel)
    deadline = min(deadline_monotonic if deadline_monotonic is not None else math.inf,
                   time.monotonic() + max_runtime_sec)
    check_cancelled(deadline, cancel)
    for frame in frame_refs:
        validate_frame_ref(asset, frame)
    if not frame_refs:
        return []
    path = local_media_path(asset.owned_path)
    if content_sha256(path, deadline_monotonic=deadline, cancel=cancel) != asset.content_hash:
        raise VideoDecodeError("Owned video content has changed")
    directory = Path(output_dir).expanduser().resolve()
    if directory == path:
        raise VideoDecodeError("Frame output directory overlaps the source file")
    targets = sorted(set(ref.pts_sec for ref in frame_refs))
    return _execute_decode(
        asset, directory, targets, expected_ids={ref.frame_id for ref in frame_refs},
        deadline_monotonic=deadline, cancel=cancel, thread_count=thread_count,
        max_pixels=max_pixels, jpeg_quality=jpeg_quality, use_subprocess=use_subprocess,
    )
