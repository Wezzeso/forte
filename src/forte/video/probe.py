"""Bounded, local video metadata discovery using FFprobe.

All public timestamps are seconds from the selected video's presentation start.
No audio packets, image pixels, or media-dependent descriptions are returned.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from fractions import Fraction
from typing import Callable, Iterator

from forte.shared.schemas import VideoAsset


class VideoError(RuntimeError):
    """A media operation failed; never interpret this as a negative verdict."""


class VideoProbeError(VideoError):
    pass


class VideoDecodeError(VideoError):
    pass


class VideoCancelledError(VideoError):
    pass


class VideoDeadlineError(VideoError):
    pass


class VideoLimitError(VideoError):
    pass


CancelSignal = Callable[[], bool] | threading.Event | None
_OPERATION_SLOTS = threading.BoundedSemaphore(2)


def check_cancelled(deadline_monotonic: float | None, cancel: CancelSignal) -> None:
    """Check an absolute monotonic deadline and a callback or Event."""
    if deadline_monotonic is not None and not math.isfinite(deadline_monotonic):
        raise ValueError("deadline_monotonic must be finite")
    if cancel is not None:
        cancelled = cancel() if callable(cancel) else cancel.is_set()
        if cancelled:
            raise VideoCancelledError("Video operation cancelled")
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise VideoDeadlineError("Video operation exceeded its deadline")


@contextmanager
def operation_slot(deadline_monotonic: float | None, cancel: CancelSignal) -> Iterator[None]:
    """Bound subprocess/decode concurrency per application worker process."""
    while not _OPERATION_SLOTS.acquire(timeout=0.05):
        check_cancelled(deadline_monotonic, cancel)
    try:
        check_cancelled(deadline_monotonic, cancel)
        yield
    finally:
        _OPERATION_SLOTS.release()


def local_media_path(path: str | os.PathLike[str]) -> Path:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise VideoProbeError("Media path must be a local regular file")
    return resolved


def content_sha256(
    path: Path, *, deadline_monotonic: float | None = None, cancel: CancelSignal = None
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            check_cancelled(deadline_monotonic, cancel)
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _positive_rate(value: object) -> float | None:
    try:
        rate = float(Fraction(str(value)))
        return rate if math.isfinite(rate) and rate > 0 else None
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _rotation(stream: dict) -> int:
    # FFmpeg display-matrix rotation is counterclockwise, as is Pillow.rotate.
    value = next(
        (entry["rotation"] for entry in stream.get("side_data_list", []) if "rotation" in entry),
        stream.get("tags", {}).get("rotate", 0),
    )
    angle = _finite_number(value)
    if angle is None or not math.isclose(angle % 90, 0, abs_tol=0.01):
        raise VideoProbeError("Unsupported non-orthogonal display rotation")
    return int(round(angle)) % 360


def probe_video(
    path: str | os.PathLike[str],
    *,
    job_id: str | None = None,
    deadline_monotonic: float | None = None,
    cancel: CancelSignal = None,
    timeout_sec: float = 15.0,
    max_pixels: int = 33_554_432,
    ffprobe_bin: str | None = None,
) -> VideoAsset:
    """Probe the first video stream without decoding or reading audio.

    ``duration_sec`` prefers the video stream's duration over audio/container
    duration. ``source_start_sec`` preserves a nonzero source PTS origin. The
    returned dimensions describe upright decoded images. Missing timestamps or
    unknown duration fail explicitly rather than inventing an FPS timeline.
    """
    if not math.isfinite(timeout_sec) or timeout_sec <= 0:
        raise ValueError("timeout_sec must be positive and finite")
    if max_pixels <= 0:
        raise ValueError("max_pixels must be positive")
    check_cancelled(deadline_monotonic, cancel)
    deadline = min(deadline_monotonic if deadline_monotonic is not None else math.inf,
                   time.monotonic() + timeout_sec)
    check_cancelled(deadline, cancel)
    try:
        owned_path = local_media_path(path)
    except (OSError, ValueError) as exc:
        raise VideoProbeError("Media file is unavailable") from exc
    executable = ffprobe_bin or os.environ.get("FORTE_FFPROBE") or shutil.which("ffprobe")
    if not executable:
        raise VideoProbeError("FFprobe is required but unavailable")
    command = [
        executable, "-v", "error", "-protocol_whitelist", "file,pipe",
        "-select_streams", "v:0", "-show_entries",
        "stream=index,codec_name,width,height,avg_frame_rate,time_base,start_time,"
        "start_pts,duration,duration_ts,nb_frames:stream_tags=rotate:"
        "stream_side_data=rotation:format=duration,start_time,format_name",
        "-of", "json", str(owned_path),
    ]
    source_stat = owned_path.stat()
    with operation_slot(deadline, cancel), tempfile.TemporaryFile() as captured:
        try:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=captured,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise VideoProbeError("FFprobe could not be started") from exc
        try:
            while True:
                check_cancelled(deadline, cancel)
                if os.fstat(captured.fileno()).st_size > 1_048_576:
                    raise VideoLimitError("FFprobe metadata exceeds configured limits")
                try:
                    process.wait(timeout=0.05)
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            process.kill()
            process.wait()
            raise
        captured.seek(0)
        stdout = captured.read(1_048_577)
    if process.returncode != 0 or len(stdout) > 1_048_576:
        raise VideoProbeError("FFprobe rejected the media")
    try:
        metadata = json.loads(stdout)
        stream = metadata["streams"][0]
        container = metadata.get("format", {})
        width, height = int(stream["width"]), int(stream["height"])
        if width <= 0 or height <= 0 or width * height > max_pixels:
            raise VideoLimitError("Video dimensions exceed configured decode limits")
        time_base = Fraction(stream["time_base"])
        if time_base <= 0:
            raise VideoProbeError("Video stream has no usable time base")
        # Integer ticks avoid FFprobe's six-decimal rounded time strings.
        start = None
        if stream.get("start_pts") is not None:
            start = float(int(stream["start_pts"]) * time_base)
        if start is None:
            start = _finite_number(stream.get("start_time"))
        if start is None:
            raise VideoProbeError("Video stream has no presentation start timestamp")
        duration = None
        if stream.get("duration_ts") is not None:
            duration = float(int(stream["duration_ts"]) * time_base)
        if duration is None:
            duration = _finite_number(stream.get("duration"))
        if duration is None:
            # FFmpeg's Matroska demuxer reports a timestamp endpoint as format
            # duration for nonzero starts. Other demuxers report elapsed time.
            container_duration = _finite_number(container.get("duration"))
            container_start = _finite_number(container.get("start_time"))
            if container_duration is not None and container_start is not None:
                if "matroska" in container.get("format_name", "").split(","):
                    duration = container_duration - start
                else:
                    duration = container_duration - max(0.0, start - container_start)
        if duration is None or duration <= 0:
            raise VideoProbeError("Video stream has no usable duration")
        rotation = _rotation(stream)
        if rotation in (90, 270):
            width, height = height, width
        frames = stream.get("nb_frames")
        frame_count = int(frames) if frames is not None and str(frames).isdigit() and int(frames) > 0 else None
        digest = content_sha256(owned_path, deadline_monotonic=deadline, cancel=cancel)
        final_stat = owned_path.stat()
        if (source_stat.st_size, source_stat.st_mtime_ns, source_stat.st_ino) != (
            final_stat.st_size, final_stat.st_mtime_ns, final_stat.st_ino
        ):
            raise VideoProbeError("Media changed while it was being probed")
        return VideoAsset(
            job_id=job_id or digest[:24], owned_path=str(owned_path), content_hash=digest,
            duration_sec=duration, width=width, height=height,
            fps=_positive_rate(stream.get("avg_frame_rate")),
            time_base=f"{time_base.numerator}/{time_base.denominator}",
            source_start_sec=start, rotation_deg=rotation,
            codec=stream.get("codec_name"), frame_count=frame_count,
            stream_index=int(stream.get("index", 0)),
            metadata={"duration_source": "video_stream" if stream.get("duration") is not None
                      or stream.get("duration_ts") is not None else "container"},
        )
    except (VideoError,):
        raise
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise VideoProbeError("Invalid or missing video-stream metadata") from exc
