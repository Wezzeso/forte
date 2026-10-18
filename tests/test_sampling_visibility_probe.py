"""Executable generated-pixel visibility counterexample; no learned model."""

from qa.sampling_visibility_probe import run_probe


def test_sparse_coverage_metadata_does_not_imply_visibility_of_brief_action():
    result = run_probe()
    sparse, denser = result["profiles"]
    assert result["synthetic_only"] and result["model_calls"] == 0
    assert result["native_marker_frame_count"] == 4
    assert sparse["interval_coverage_complete_if_requests_succeed"]
    assert sparse["marker_selected_frames"] == 0
    assert sparse["sampled_motion_cues"] == 0
    assert denser["marker_selected_frames"] > 0
    assert sparse["maximum_frames_per_request"] <= 16
    assert denser["maximum_frames_per_request"] <= 16
    assert denser["requests"] > sparse["requests"]
    assert result["original_bytes_unchanged"]
