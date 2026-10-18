"""Single-pass video decoding into timestamped PIL frames (visual stream only)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Callable

import av
from PIL import Image


@dataclass
class Frame:
    t: float            # seconds from the start of the original video
    image: Image.Image  # full-resolution, upright RGB


@dataclass
class DecodedVideo:
    duration: float
    fps: float
    width: int
    height: int
    frames: list[Frame]          # sampled at `sample_fps`

    def nearest(self, t: float) -> Frame:
        return min(self.frames, key=lambda f: abs(f.t - t))

    def between(self, start: float, end: float) -> list[Frame]:
        return [f for f in self.frames if start - 1e-6 <= f.t <= end + 1e-6]


def _rotation(stream: av.video.stream.VideoStream) -> int:
    try:
        rotate = int(stream.metadata.get("rotate", 0))
    except (TypeError, ValueError):
        rotate = 0
    return rotate % 360


def decode(path: str, sample_fps: float = 2.0, max_frames: int = 600, *,
           on_frame: Callable[[Frame], object] | None = None,
           on_start: Callable[[float], object] | None = None,
           max_memory_bytes: int = 2 * 1024 ** 3,
           deadline: float | None = None,
           cancel_event: threading.Event | None = None) -> DecodedVideo:
    """Decode once; optionally process every native frame while retaining samples.

    The callback must retain only metadata, not the image. Limits raise instead
    of silently analysing a truncated video. Retained images are original pixels.
    """
    if not math.isfinite(sample_fps) or not 0 < sample_fps <= 120:
        raise ValueError("fast_decode_sample_fps_invalid")
    if max_frames < 1 or max_memory_bytes < 1:
        raise ValueError("fast_decode_limits_invalid")
    frames: list[Frame] = []
    memory_bytes = 0

    def check_budget():
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("analysis_cancelled")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("analysis_deadline_exceeded")

    def keep(item: Frame):
        nonlocal memory_bytes
        size = item.image.width * item.image.height * 3
        if len(frames) >= max_frames or memory_bytes + size > max_memory_bytes:
            item.image.close()
            raise ValueError("fast_decode_retained_frame_limit_exceeded")
        frames.append(item)
        memory_bytes += size

    def upright(frame, rotate):
        image = frame.to_image().convert("RGB")
        angle = int(getattr(frame, "rotation", 0) or 0)
        if not angle and rotate:
            angle = -rotate
        if angle % 360:
            original = image
            image = original.rotate(angle % 360, expand=True)
            original.close()
        return image

    check_budget()
    try:
        return _decode(path, sample_fps, on_frame, on_start, frames, check_budget, keep, upright)
    except BaseException:
        for item in frames:
            item.image.close()
        raise


def _decode(path, sample_fps, on_frame, on_start, frames, check_budget, keep, upright):
    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError("no decodable video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate or stream.guessed_rate or 30)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fast_decode_frame_rate_invalid")
        if on_start is not None:
            on_start(fps)
        tb = float(stream.time_base) if stream.time_base else 1.0 / fps
        duration = None
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = container.duration / 1_000_000
        rotate = _rotation(stream)
        start = float(stream.start_time * stream.time_base) if stream.start_time else 0.0
        step = 1.0 / sample_fps
        next_t = 0.0
        last_t = 0.0
        last_frame = None
        for frame in container.decode(stream):
            check_budget()
            if frame.pts is None:
                raise ValueError("fast_decode_source_timestamp_missing")
            t = float(frame.pts) * float(frame.time_base or tb) - start
            if not math.isfinite(t) or t < -1e-3 or (last_frame is not None and t < last_t - 1e-6):
                raise ValueError("fast_decode_source_timestamp_invalid")
            last_t = max(last_t, t)
            last_frame = frame
            retain = t + 1e-6 >= next_t
            if retain or on_frame is not None:
                item = Frame(t=round(max(t, 0.0), 6), image=upright(frame, rotate))
                try:
                    if on_frame is not None:
                        on_frame(item)
                    if retain:
                        keep(item)
                finally:
                    if not retain or not frames or frames[-1] is not item:
                        item.image.close()
                if retain:
                    next_t += step
                    while next_t <= t:
                        next_t += step
        check_budget()
        if not frames:
            raise ValueError("no decodable video frames")
        if last_t > frames[-1].t + 1e-6:
            keep(Frame(t=round(last_t, 6), image=upright(last_frame, rotate)))
        duration = duration or (last_t + 1.0 / fps)
        width, height = frames[0].image.size
        return DecodedVideo(duration=duration, fps=fps, width=width, height=height, frames=frames)
