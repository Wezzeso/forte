"""Synthetic-only timeline and image stress cases; no supplied media is opened."""

from dataclasses import replace
import math

from PIL import Image, ImageDraw
import pytest

from forte.shared.schemas import FrameRef
from forte.video.crops import (
    FULL_SCENE,
    NormalizedBox,
    adaptive_crops,
    detect_motion_cues,
    materialize_crops,
)
from forte.video.sampling import (
    CoverageBudgetExceeded,
    CoverageError,
    ProcessorBudget,
    context_window,
    coverage_report,
    coverage_windows,
    make_clip_requests,
    record_inspection,
    require_complete_coverage,
    sampling_timestamps,
)


def synthetic_refs(points, *, path="synthetic.jpg", width=320, height=180):
    return [FrameRef(frame_id=f"synthetic-{index}", pts_sec=float(point), path=str(path),
                     width=width, height=height, job_id="synthetic-job")
            for index, point in enumerate(points)]


@pytest.mark.parametrize("duration", [0.01, 0.5, 1.0, 7.99, 8.0, 8.01, 12.0, 13.7, 145.0])
def test_full_plan_includes_tail_and_has_no_timeline_gaps(duration):
    windows = coverage_windows(duration)
    assert windows[0].start_sec == 0
    assert windows[-1].end_sec == duration
    assert windows[-1].timestamps_sec[-1] == pytest.approx(duration)
    assert all(a.end_sec >= b.start_sec for a, b in zip(windows, windows[1:]))
    assert all(1 <= len(window.timestamps_sec) <= 16 for window in windows)
    assert sampling_timestamps(windows) == sorted(set(sampling_timestamps(windows)))


def test_boundary_and_tail_short_event_fixtures_are_queried():
    windows = coverage_windows(13.7)
    for start, end in [(3.9, 4.1), (7.9, 8.1), (13.5, 13.7)]:
        assert any(window.start_sec <= start and end <= window.end_sec for window in windows)
    # Sparse coverage cannot guarantee events shorter than the sample spacing.
    assert max(b - a for w in windows for a, b in zip(w.timestamps_sec, w.timestamps_sec[1:])) <= 0.534


def test_very_short_clip_requests_opening_and_tail():
    window = coverage_windows(0.25)[0]
    assert window.timestamps_sec == (0.0, 0.25)


@pytest.mark.parametrize("kwargs", [
    {"max_frames_per_window": 15}, {"max_windows": 2}, {"max_total_frames": 47}
])
def test_budget_exhaustion_fails_instead_of_losing_tail(kwargs):
    with pytest.raises(CoverageBudgetExceeded):
        coverage_windows(20.0, **kwargs)


@pytest.mark.parametrize("duration", [0, -1, math.inf, math.nan])
def test_invalid_duration(duration):
    with pytest.raises(CoverageError):
        coverage_windows(duration)


def test_stride_cannot_create_gaps():
    with pytest.raises(CoverageError):
        coverage_windows(20, window_sec=8, stride_sec=9)


def test_inspection_uses_real_pts_and_keeps_completion_separate():
    windows = coverage_windows(13.7)
    refs = synthetic_refs(sorted(set(round(t * 30) / 30 for t in sampling_timestamps(windows))))
    requests = make_clip_requests("synthetic-job", refs, windows)
    assert len(requests) == len(windows)
    assert all(frame in refs for request in requests for frame in request.frame_refs)
    records = [record_inspection(window, clip) for window, clip in zip(windows, requests)]
    report = coverage_report(13.7, windows, records)
    require_complete_coverage(report)
    assert report.complete
    assert report.gaps == ()
    assert report.to_dict()["inspected_intervals"][0]["frame_pts_sec"]
    assert not coverage_report(13.7, windows, []).complete
    incomplete = coverage_report(13.7, windows, records[:-1])
    assert not incomplete.complete
    assert incomplete.missing_window_indices == (windows[-1].index,)
    with pytest.raises(CoverageError):
        require_complete_coverage(incomplete)


def test_missing_overlap_still_requires_every_planned_window():
    windows = coverage_windows(20)
    refs = synthetic_refs(sampling_timestamps(windows))
    requests = make_clip_requests("synthetic-job", refs, windows)
    records = [record_inspection(w, c) for w, c in zip(windows, requests)]
    incomplete = coverage_report(20, windows, records[:1] + records[2:])
    assert not incomplete.complete
    assert incomplete.gaps == ()
    assert incomplete.missing_window_indices == (1,)


def test_empty_or_inadequate_sparse_decode_fails():
    windows = coverage_windows(8)
    for refs in ([], synthetic_refs([4])):
        with pytest.raises(CoverageError):
            make_clip_requests("synthetic-job", refs, windows)


def test_inconsistent_inspection_bounds_rejected():
    windows = coverage_windows(8)
    refs = synthetic_refs(sampling_timestamps(windows))
    request = make_clip_requests("synthetic-job", refs, windows)[0]
    record = record_inspection(windows[0], request)
    with pytest.raises(CoverageError):
        coverage_report(8, windows, [replace(record, actual_end_sec=2)])


def test_longer_procedure_context_requires_explicit_budget():
    longer = context_window(40, 5, 25, context_sec=3)
    assert (longer.start_sec, longer.end_sec, longer.purpose) == (2, 28, "refinement")
    assert len(longer.timestamps_sec) == 52
    with pytest.raises(CoverageBudgetExceeded):
        context_window(40, 5, 25, context_sec=3, max_frames=16)


def test_processor_budget_uses_actual_measurements():
    budget = ProcessorBudget(max_frames=16, max_processed_pixels=1000, max_visual_tokens=200)
    budget.check(16, processed_pixels=1000, visual_tokens=200)
    with pytest.raises(CoverageError):
        budget.check(16)
    with pytest.raises(CoverageBudgetExceeded):
        budget.check(16, processed_pixels=1001, visual_tokens=100)


def test_crops_keep_global_context_and_adapt_to_region():
    left = NormalizedBox(0.01, 0.4, 0.02, 0.5)
    right = NormalizedBox(0.98, 0.4, 0.99, 0.5)
    boxes = adaptive_crops([left, right], min_area_fraction=0.25)
    assert boxes[0] == FULL_SCENE
    assert len(boxes) == 3
    assert boxes[1].area >= 0.25 - 1e-9
    assert boxes[1].left <= left.left < left.right <= boxes[1].right
    assert boxes[2].left <= right.left < right.right <= boxes[2].right
    assert boxes[1] != boxes[2]
    assert adaptive_crops([left], max_crops=0) == [FULL_SCENE]


def test_materialized_crops_preserve_source_timestamp_and_original(tmp_path):
    path = tmp_path / "synthetic.jpg"
    Image.new("RGB", (320, 180), (70, 130, 190)).save(path)
    original = path.read_bytes()
    frame = synthetic_refs([3.25], path=path)[0]
    views = materialize_crops(frame, tmp_path / "views", [NormalizedBox(0.3, 0.4, 0.4, 0.5)])
    assert views[0].frame_ref == frame
    assert views[0].is_full_scene
    assert path.read_bytes() == original
    assert len(views) == 2
    assert views[1].frame_ref.pts_sec == 3.25
    assert views[1].source_frame_id == frame.frame_id
    assert views[1].frame_ref.frame_id != frame.frame_id
    with Image.open(views[1].frame_ref.path) as crop:
        assert crop.format == "JPEG"
        assert crop.size == (views[1].frame_ref.width, views[1].frame_ref.height)


def test_motion_is_optional_and_uses_synthetic_local_images(tmp_path):
    paths = []
    for index in range(3):
        image = Image.new("RGB", (320, 180), "black")
        if index:
            ImageDraw.Draw(image).rectangle((160, 0, 319, 179), fill="white")
        path = tmp_path / f"synthetic-{index}.jpg"
        image.save(path)
        paths.append(path)
    frames = [synthetic_refs([index], path=path)[0].model_copy(update={"frame_id": f"motion-{index}"})
              for index, path in enumerate(paths)]
    cues = detect_motion_cues(frames)
    assert len(cues) == 1
    assert (cues[0].start_sec, cues[0].end_sec) == (0, 1)
    assert 0 <= cues[0].score <= 1
    assert cues[0].region.left > 0.4
    assert detect_motion_cues(frames, max_cues=0) == []
