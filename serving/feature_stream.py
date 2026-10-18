"""Bounded CPU decode/GPU encode overlap, preserving source metadata privately.

No media, model, ML library or decoder is opened/imported at module import.
The caller supplies A05's iterator factory and an explicitly loaded FastEncoder.
This bridge performs no label lookup, split selection or cache publication.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
import math
import queue
import re
import threading
import time
from typing import Any


class FeatureStreamError(RuntimeError):
    """Static diagnostics only; never wrap decoder exception text or pixels."""


@dataclass(frozen=True)
class EncodedSourceBatch:
    features: Any = field(repr=False)
    frames: tuple[Any, ...] = field(repr=False)
    transforms: tuple[Any, ...] = field(repr=False)
    encoder_recipe_sha256: str
    metrics: dict[str, float]
    source_complete: bool
    segment_complete: bool
    tail_included: bool
    decoded_source_frame_count: int
    delivered_count: int
    selection_mode: str
    segment_id: str = field(repr=False)
    views_per_frame: int = 1


@dataclass(frozen=True)
class _Failure:
    code: str


class _CombinedCancel:
    def __init__(self, internal: threading.Event, external: Any):
        self.internal, self.external = internal, external

    def is_set(self) -> bool:
        return self.internal.is_set() or (self.external is not None and self.external.is_set())


def _checkpoint(deadline: float, cancel: _CombinedCancel) -> None:
    if cancel.is_set():
        raise FeatureStreamError("feature_stream_cancelled")
    if time.monotonic() >= deadline:
        raise FeatureStreamError("feature_stream_deadline_exceeded")


def _check_metadata(batch: Any, previous: Any, *, require_native: bool) -> tuple[Any, int]:
    # Import at execution only; the decoder owns the canonical identity protocol.
    from forte.video.batches import source_frame_id
    frames, transforms = tuple(batch.frames), tuple(batch.transforms)
    if not frames or len(batch.shape) != 4 or batch.shape[0] % len(frames):
        raise FeatureStreamError("feature_stream_frame_view_shape_invalid")
    views = batch.shape[0] // len(frames)
    if views not in {1, 2} or len(transforms) != len(frames) * views:
        raise FeatureStreamError("feature_stream_view_budget_invalid")
    if require_native and batch.selection_mode != "native":
        raise FeatureStreamError("feature_stream_native_frames_required")
    if any(type(value) is not bool for value in (batch.source_complete, batch.segment_complete, batch.tail_included)):
        raise FeatureStreamError("feature_stream_coverage_flags_invalid")
    for index, frame in enumerate(frames):
        numerator, denominator = frame.time_base_num, frame.time_base_den
        if (not re.fullmatch(r"[0-9a-f]{64}", frame.content_hash)
                or type(frame.source_origin_pts) is not int
                or type(frame.source_pts) is not int or type(numerator) is not int or type(denominator) is not int
                or numerator <= 0 or denominator <= 0 or math.gcd(numerator, denominator) != 1):
            raise FeatureStreamError("feature_stream_source_timebase_invalid")
        expected_id = source_frame_id(frame.content_hash, frame.source_pts, numerator, denominator)
        expected_time = (frame.source_pts - frame.source_origin_pts) * numerator / denominator
        if (frame.frame_id != expected_id or not math.isfinite(frame.pts_sec)
                or abs(frame.pts_sec - expected_time) > 1e-6):
            raise FeatureStreamError("feature_stream_source_identity_invalid")
        if previous is not None:
            if (frame.content_hash != previous.content_hash
                    or (numerator, denominator, frame.source_origin_pts) != (
                        previous.time_base_num, previous.time_base_den, previous.source_origin_pts)
                    or frame.source_pts <= previous.source_pts
                    or (require_native and previous.next_source_pts != frame.source_pts)):
                raise FeatureStreamError("feature_stream_source_gap_or_mismatch")
        current_views = transforms[index * views:(index + 1) * views]
        if current_views[0].view_id != "global":
            raise FeatureStreamError("feature_stream_global_view_first_required")
        globals_count = 0
        for transform in current_views:
            box = tuple(transform.source_box_xyxy)
            if (len(box) != 4 or not all(math.isfinite(value) and 0 <= value <= 1 for value in box)
                    or box[0] >= box[2] or box[1] >= box[3]):
                raise FeatureStreamError("feature_stream_view_box_invalid")
            if transform.view_id == "global":
                globals_count += 1
                if box != (0, 0, 1, 1):
                    raise FeatureStreamError("feature_stream_global_view_cropped")
        if globals_count != 1:
            raise FeatureStreamError("feature_stream_global_view_required")
        previous = frame
    if batch.source_complete and (not batch.tail_included or frames[-1].next_source_pts is not None):
        raise FeatureStreamError("feature_stream_false_eof_or_tail")
    return previous, views


def iter_encoded_batches(
    source_factory: Callable[[Any], Iterator[Any]], encoder: Any, *, deadline: float,
    cancel_event: Any = None, prefetch_batches: int = 2,
    require_native: bool = True, require_complete: bool = True,
    max_source_frames: int = 10000, close_timeout_sec: float = 2.0,
) -> Iterator[EncodedSourceBatch]:
    """Overlap A05 decode with one GPU consumer; close on exhaustion/early exit.

    ``source_factory(combined_cancel)`` must create the decoder in its producer
    thread, normally with A05 subprocess supervision enabled. At most two
    lookahead batches (including the one being decoded) plus one active encoding
    batch are retained. The producer alone closes its iterator. Callers should
    use ``contextlib.closing`` if they may stop before exhaustion.
    """
    if (type(prefetch_batches) is not int or not 1 <= prefetch_batches <= 2
            or not math.isfinite(deadline) or deadline <= time.monotonic()
            or type(max_source_frames) is not int or not 1 <= max_source_frames <= 10000
            or not math.isfinite(close_timeout_sec) or not 0 < close_timeout_sec <= 5):
        raise FeatureStreamError("feature_stream_limits_invalid")
    # Fail before spawning a decoder when the explicit encoder load has not run.
    recipe_sha256 = encoder.recipe_sha256
    stop, producer_done = threading.Event(), threading.Event()
    combined = _CombinedCancel(stop, cancel_event)
    pending: queue.Queue = queue.Queue(maxsize=prefetch_batches)
    slots = threading.Semaphore(prefetch_batches)
    end = object()

    def put(value: Any) -> bool:
        while not combined.is_set() and time.monotonic() < deadline:
            try:
                pending.put(value, timeout=.05)
                return True
            except queue.Full:
                pass
        return False

    def producer() -> None:
        reader = None
        terminal: Any = end
        try:
            reader = iter(source_factory(combined))
            while not combined.is_set() and time.monotonic() < deadline:
                if not slots.acquire(timeout=.05):
                    continue
                try:
                    value = next(reader)
                except StopIteration:
                    slots.release()
                    break
                except BaseException:
                    slots.release()
                    raise
                if not put(value):
                    slots.release()
                    return
        except BaseException:
            terminal = _Failure("feature_stream_decoder_failed")
        finally:
            try:
                if reader is not None and hasattr(reader, "close"):
                    reader.close()
            except BaseException:
                terminal = _Failure("feature_stream_decoder_close_failed")
            put(terminal)
            producer_done.set()

    worker = threading.Thread(target=producer, name="forte-feature-decode", daemon=True)
    worker.start()
    previous = None
    delivered = 0
    last_complete = last_tail = False
    accumulated_queue_wait = 0.0
    try:
        while True:
            _checkpoint(deadline, combined)
            waiting = time.perf_counter()
            try:
                value = pending.get(timeout=min(.1, max(.001, deadline - time.monotonic())))
            except queue.Empty:
                accumulated_queue_wait += time.perf_counter() - waiting
                if producer_done.is_set():
                    raise FeatureStreamError("feature_stream_decoder_ended_without_eof")
                continue
            queue_wait = accumulated_queue_wait + time.perf_counter() - waiting
            accumulated_queue_wait = 0.0
            if value is end:
                if require_complete and not (last_complete and last_tail):
                    raise FeatureStreamError("feature_stream_incomplete_source")
                return
            if isinstance(value, _Failure):
                raise FeatureStreamError(value.code)
            slots.release()
            previous, views = _check_metadata(value, previous, require_native=require_native)
            delivered += len(value.frames)
            if delivered > max_source_frames:
                raise FeatureStreamError("feature_stream_frame_limit_exceeded")
            if (getattr(value, "total_delivered_count", value.delivered_count) != delivered
                    or value.decoded_source_frame_count < delivered):
                raise FeatureStreamError("feature_stream_coverage_counts_mismatch")
            result = encoder.encode_frame_batch(value, deadline=deadline, cancel_event=combined)
            if result.shape != (value.shape[0], 768) or result.encoder_recipe_sha256 != recipe_sha256:
                raise FeatureStreamError("feature_stream_encoder_identity_or_shape_mismatch")
            last_complete, last_tail = bool(value.source_complete), bool(value.tail_included)
            yield EncodedSourceBatch(result.features, tuple(value.frames), tuple(value.transforms),
                recipe_sha256, {**result.metrics, "decode_queue_wait_sec": queue_wait},
                last_complete, bool(value.segment_complete), last_tail,
                value.decoded_source_frame_count, value.delivered_count,
                value.selection_mode, value.segment_id, views)
            del value, result
    finally:
        stop.set()
        worker.join(timeout=close_timeout_sec)
        while True:
            try:
                pending.get_nowait()
            except queue.Empty:
                break
        if worker.is_alive():
            raise FeatureStreamError("feature_stream_decoder_cleanup_not_drained")
