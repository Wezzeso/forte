"""Generated numeric geometry only, with frozen-denominator validation."""
import pytest

from benchmarks.performance.qwen38_latency_feasibility import assess
from benchmarks.performance.repeatability import RepeatabilityError


def plan():
    return {"worker": {"window_seconds": 8., "stride_seconds": 4., "sample_fps": 2.,
        "max_frames_per_window": 16, "max_refinement_passes": 8, "job_timeout_seconds": 285.,
        "max_concurrent_jobs": 1, "max_concurrent_clips": 1, "max_inflight_per_job": 1, "staged_screening": False},
        "cases": [{"duration_sec": 17.3, "coarse_windows": 4}, {"duration_sec": 78.733021, "coarse_windows": 19}],
        "minimum_initial_generation_requests": 23}


def test_projection_preserves_boundary_and_no_measurement_claim():
    result = assess(plan(), reference_seconds=69.554)
    assert result["reference_exceeds_provider_request_timeout"] is False
    assert result["conditional_jobs_exceeding_deadline_before_other_work"] == 1
    assert result["cases"][0]["conditional_seconds_left_for_all_nonmodel_and_refinement"] == pytest.approx(6.784)
    assert result["cases"][1]["maximum_average_initial_request_seconds_with_reserve"] == pytest.approx(270 / 19)
    assert result["cases"][1]["maximum_average_request_seconds_with_all_refinements_and_reserve"] == 10.
    assert result["deterministic_infeasibility_proved"] is False and result["media_or_labels_read"] is False
    assert assess(plan(), reference_seconds=69.554, provider_timeout_seconds=60.)["reference_exceeds_provider_request_timeout"] is True


def test_plan_geometry_or_denominator_mismatch_is_rejected():
    changed = plan()
    changed["cases"][0]["coarse_windows"] = 3
    with pytest.raises(RepeatabilityError, match="geometry_mismatch"):
        assess(changed, reference_seconds=69.554)
    changed = plan()
    changed["minimum_initial_generation_requests"] = 22
    with pytest.raises(RepeatabilityError, match="initial_count_mismatch"):
        assess(changed, reference_seconds=69.554)
