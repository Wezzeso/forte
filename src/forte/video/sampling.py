"""Bounded sparse video coverage with explicit completion accounting.

Windows describe the timeline queried by the model, not continuous observation.
Successful inspection records retain the actual decoded presentation timestamps.
Motion cues may add refinement requests; they never replace baseline coverage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Iterable, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from forte.shared.schemas import ClipRequest, FactDefinition, FrameRef, RuleRecord


class CoverageError(ValueError):
    """Coverage cannot be established from the supplied data."""


class CoverageBudgetExceeded(CoverageError):
    """A complete request exceeds a declared budget; no tail was discarded."""


@dataclass(frozen=True, slots=True)
class CoverageWindow:
    index: int
    start_sec: float
    end_sec: float
    timestamps_sec: tuple[float, ...]
    purpose: str = "coverage"


@dataclass(frozen=True, slots=True)
class ProcessorBudget:
    """Limits checked against actual processor measurements when available."""

    max_frames: int = 16
    max_processed_pixels: int | None = None
    max_visual_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_frames", "max_processed_pixels", "max_visual_tokens"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise CoverageError(f"{name} must be a positive integer")

    def check(
        self,
        frame_count: int,
        *,
        processed_pixels: int | None = None,
        visual_tokens: int | None = None,
    ) -> None:
        for label, actual, limit in (
            ("frames", frame_count, self.max_frames),
            ("processed pixels", processed_pixels, self.max_processed_pixels),
            ("visual tokens", visual_tokens, self.max_visual_tokens),
        ):
            if limit is not None and actual is None:
                raise CoverageError(f"Actual {label} measurement is required")
            if actual is not None and actual < 0:
                raise CoverageError(f"Actual {label} cannot be negative")
            if limit is not None and actual is not None and actual > limit:
                raise CoverageBudgetExceeded(f"{label} budget exceeded: {actual} > {limit}")


@dataclass(frozen=True, slots=True)
class InspectedInterval:
    window_index: int
    start_sec: float
    end_sec: float
    frame_ids: tuple[str, ...]
    frame_pts_sec: tuple[float, ...]
    actual_start_sec: float
    actual_end_sec: float


@dataclass(frozen=True, slots=True)
class CoverageReport:
    duration_sec: float
    complete: bool
    planned_intervals: tuple[tuple[float, float], ...]
    inspected_intervals: tuple[InspectedInterval, ...]
    gaps: tuple[tuple[float, float], ...]
    missing_window_indices: tuple[int, ...]
    policy: str = "overlapping_sparse_windows"

    def to_dict(self) -> dict:
        return asdict(self)


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise CoverageError(f"{name} must be finite and positive")
    return value


def _timestamps(start: float, end: float, fps: float, budget: int) -> tuple[float, ...]:
    # End-inclusive target requests are resolved by the decoder to real PTS.
    # Sixteen samples / eight seconds is approximately two visible frames/s.
    # Even a sub-half-second clip requests both boundaries: a single sample at
    # the end would omit its opening. A one-frame source is deduplicated later.
    count = max(2, math.ceil((end - start) * fps - 1e-10))
    if count > budget:
        raise CoverageBudgetExceeded(f"Window requires {count} frames; budget is {budget}")
    return tuple(start + (end - start) * i / (count - 1) for i in range(count))


def coverage_windows(
    duration_sec: float,
    *,
    window_sec: float = 8.0,
    stride_sec: float = 4.0,
    sample_fps: float = 2.0,
    max_frames_per_window: int = 16,
    max_windows: int | None = None,
    max_total_frames: int | None = None,
) -> list[CoverageWindow]:
    """Plan complete overlapping coverage, including an end-aligned final window.

    ``max_total_frames`` counts model-visible frame uses including overlap, not
    unique decoded images. Insufficient budgets raise before any plan is returned.
    A duration shorter than one window still creates one inspection request.
    """
    duration = _positive(duration_sec, "duration_sec")
    window = _positive(window_sec, "window_sec")
    stride = _positive(stride_sec, "stride_sec")
    fps = _positive(sample_fps, "sample_fps")
    if stride > window:
        raise CoverageError("stride_sec cannot exceed window_sec: this would leave gaps")
    if isinstance(max_frames_per_window, bool) or not isinstance(max_frames_per_window, int) or max_frames_per_window < 1:
        raise CoverageError("max_frames_per_window must be positive")
    for name, value in (("max_windows", max_windows), ("max_total_frames", max_total_frames)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise CoverageError(f"{name} must be positive")
    tail_start = max(0.0, duration - window)
    # Compute the size before allocation so hostile durations cannot exhaust RAM.
    regular_count = math.floor(tail_start / stride + 1e-10) + 1
    has_tail = not math.isclose((regular_count - 1) * stride, tail_start, abs_tol=1e-8)
    count = regular_count + int(has_tail)
    if count > (max_windows if max_windows is not None else 100_000):
        raise CoverageBudgetExceeded(f"Complete coverage needs {count} windows")
    starts = [index * stride for index in range(regular_count)]
    if has_tail:
        starts.append(tail_start)
    windows = [
        CoverageWindow(index, start, min(duration, start + window),
                       _timestamps(start, min(duration, start + window), fps, max_frames_per_window))
        for index, start in enumerate(starts)
    ]
    total = sum(len(item.timestamps_sec) for item in windows)
    if max_total_frames is not None and total > max_total_frames:
        raise CoverageBudgetExceeded(f"Complete coverage needs {total} frame uses; budget is {max_total_frames}")
    return windows


def context_window(
    duration_sec: float,
    start_sec: float,
    end_sec: float,
    *,
    context_sec: float = 2.0,
    sample_fps: float = 2.0,
    max_frames: int = 64,
    index: int = 0,
) -> CoverageWindow:
    """Request longer procedural/refinement context with an explicit frame budget."""
    duration = _positive(duration_sec, "duration_sec")
    if not all(math.isfinite(v) for v in (start_sec, end_sec, context_sec)):
        raise CoverageError("Context interval must be finite")
    if not 0 <= start_sec < end_sec <= duration or context_sec < 0:
        raise CoverageError("Invalid context interval")
    start = max(0.0, start_sec - context_sec)
    end = min(duration, end_sec + context_sec)
    return CoverageWindow(index, start, end, _timestamps(start, end, _positive(sample_fps, "sample_fps"), max_frames), "refinement")


def sampling_timestamps(windows: Iterable[CoverageWindow]) -> list[float]:
    """Deduplicate decoder targets while retaining source time precision."""
    return sorted({time for window in windows for time in window.timestamps_sec})


def semantic_clip_id(window: CoverageWindow, frames: Sequence[FrameRef]) -> str:
    """Bind a clip to its source observations rather than upload ownership.

    Canonical frame IDs already bind original content SHA-256, native PTS and
    time-base units. Job IDs, private JPEG paths and orchestration indices must
    stay out of this identity: fresh uploads of identical source observations
    should yield identical candidate projections. Ownership remains separately
    enforced by ClipRequest and the model adapter's job-directory boundary.
    """
    identity = {
        "schema_version": "forte.semantic-clip.v1",
        "purpose": window.purpose,
        "start_sec": float(window.start_sec), "end_sec": float(window.end_sec),
        "frames": [{"frame_id": frame.frame_id, "source_pts": frame.source_pts,
                    "pts_sec": float(frame.pts_sec), "width": frame.width, "height": frame.height,
                    "rotation_deg": frame.rotation_deg} for frame in frames],
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode("utf-8")).hexdigest()
    return f"clip-{window.purpose}-{digest}"


def make_clip_requests(
    job_id: str,
    frames: Sequence[FrameRef],
    windows: Sequence[CoverageWindow],
    *,
    fact_definitions: Sequence[FactDefinition] = (),
    rule_definitions: Sequence[RuleRecord] = (),
    analysis_version: str = "1.0.0",
    budget: ProcessorBudget | None = None,
    max_target_distance_sec: float | None = None,
) -> list[ClipRequest]:
    """Bind real decoded PTS to requests; never fabricate or interpolate frames.

    Only frames inside a window are eligible. Decode the deduplicated targets
    once before calling this helper. Repeated source frames are deduplicated.
    """
    from forte.shared.schemas import ClipRequest

    budget = budget or ProcessorBudget()
    if max_target_distance_sec is not None:
        max_target_distance_sec = _positive(max_target_distance_sec, "max_target_distance_sec")
    ordered = sorted(frames, key=lambda frame: (frame.pts_sec, frame.frame_id))
    if len({frame.frame_id for frame in ordered}) != len(ordered):
        raise CoverageError("Decoded frame IDs must be unique")
    requests: list[ClipRequest] = []
    for window in windows:
        eligible = [frame for frame in ordered if window.start_sec <= frame.pts_sec <= window.end_sec]
        if not eligible:
            raise CoverageError(f"No real source frames for window {window.index}")
        selected = {}
        step = max((b - a for a, b in zip(window.timestamps_sec, window.timestamps_sec[1:])), default=0.5)
        allowed_distance = max_target_distance_sec if max_target_distance_sec is not None else max(0.5, step)
        for target in window.timestamps_sec:
            frame = min(eligible, key=lambda item: (abs(item.pts_sec - target), item.pts_sec))
            if abs(frame.pts_sec - target) > allowed_distance + 1e-8:
                raise CoverageError(f"Decoded frames do not support requested sampling in window {window.index}")
            selected[frame.frame_id] = frame
        refs = sorted(selected.values(), key=lambda frame: (frame.pts_sec, frame.frame_id))
        # Pixel/token budgets are checked after the actual processor runs.
        if len(refs) > budget.max_frames:
            raise CoverageBudgetExceeded(f"Window {window.index} exceeds frame budget")
        requests.append(ClipRequest(job_id=job_id, clip_id=semantic_clip_id(window, refs),
                                    frame_refs=refs, fact_definitions=list(fact_definitions),
                                    rule_definitions=list(rule_definitions), analysis_version=analysis_version,
                                    start_sec=window.start_sec, end_sec=window.end_sec))
    return requests


def record_inspection(window: CoverageWindow, clip: ClipRequest) -> InspectedInterval:
    """Call only after successful model inspection of this exact clip request."""
    if clip.start_sec != window.start_sec or clip.end_sec != window.end_sec:
        raise CoverageError("Inspected clip interval does not match its planned window")
    if not clip.frame_refs:
        raise CoverageError("An empty request cannot establish inspection")
    points = tuple(frame.pts_sec for frame in clip.frame_refs)
    if any(not window.start_sec <= point <= window.end_sec for point in points):
        raise CoverageError("Inspected frame lies outside its interval")
    return InspectedInterval(window.index, window.start_sec, window.end_sec,
                             tuple(frame.frame_id for frame in clip.frame_refs), points, min(points), max(points))


def _gaps(duration: float, intervals: Iterable[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    cursor = 0.0
    gaps = []
    for start, end in sorted(intervals):
        if start > cursor + 1e-8:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration - 1e-8:
        gaps.append((cursor, duration))
    return tuple(gaps)


def coverage_report(
    duration_sec: float,
    windows: Sequence[CoverageWindow],
    inspections: Sequence[InspectedInterval],
) -> CoverageReport:
    """Distinguish planned coverage, successful sparse inspection, and missing work."""
    duration = _positive(duration_sec, "duration_sec")
    planned = {window.index: window for window in windows if window.purpose == "coverage"}
    if len(planned) != len([w for w in windows if w.purpose == "coverage"]):
        raise CoverageError("Coverage window indices must be unique")
    successful: dict[int, InspectedInterval] = {}
    for record in inspections:
        window = planned.get(record.window_index)
        if window is None:
            raise CoverageError("Inspection references an unknown coverage window")
        if (record.start_sec, record.end_sec) != (window.start_sec, window.end_sec):
            raise CoverageError("Inspection interval differs from the planned coverage")
        if not record.frame_ids or len(record.frame_ids) != len(record.frame_pts_sec):
            raise CoverageError("Inspection must identify its actual source frames")
        if len(set(record.frame_ids)) != len(record.frame_ids):
            raise CoverageError("Inspection source frame IDs must be unique")
        if any(not window.start_sec <= point <= window.end_sec for point in record.frame_pts_sec):
            raise CoverageError("Inspection source timestamp lies outside its interval")
        if (record.actual_start_sec, record.actual_end_sec) != (min(record.frame_pts_sec), max(record.frame_pts_sec)):
            raise CoverageError("Actual inspection bounds must match its source timestamps")
        successful[record.window_index] = record
    planned_intervals = tuple((window.start_sec, window.end_sec) for window in planned.values())
    if any(not 0 <= start < end <= duration for start, end in planned_intervals):
        raise CoverageError("Planned coverage interval is outside the video")
    gaps = _gaps(duration, ((item.start_sec, item.end_sec) for item in successful.values()))
    missing = tuple(sorted(set(planned) - set(successful)))
    return CoverageReport(duration, bool(planned) and not gaps and not missing,
                          planned_intervals, tuple(successful[index] for index in sorted(successful)), gaps, missing)


def require_complete_coverage(report: CoverageReport) -> None:
    if not report.complete:
        raise CoverageError("Analysis is incomplete; successful full coverage is required")
