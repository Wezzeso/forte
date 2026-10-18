"""Fresh upload ownership must not change semantic clip/model input identity."""

from dataclasses import replace
from pathlib import Path
import time

import pytest

from forte.decision import DecisionConfig, build_prompt, validate_decisions
from forte.decision.aliasing import compact_frame_request
from forte.model import LocalModelConfig, UnsafeMedia
from forte.model.adapter import _prepare_media
from forte.shared.schemas import FactDefinition
from forte.video.decode import decode_frames
from forte.video.probe import probe_video
from forte.video.sampling import CoverageWindow, coverage_windows, make_clip_requests, sampling_timestamps
from test_video_batches import make_fixture


def _request_pair(tmp_path: Path):
    source = make_fixture(tmp_path / "generated.mov")
    windows = coverage_windows(probe_video(source).duration_sec)
    requests = []
    for job in ("fresh_job_one", "fresh_job_two"):
        asset = probe_video(source, job_id=job)
        refs = decode_frames(asset, tmp_path / "media" / job / "frames",
                             timestamps_sec=sampling_timestamps(windows), use_subprocess=False)
        requests.append(make_clip_requests(job, refs, windows,
            fact_definitions=[FactDefinition(fact_id="VISIBLE", description="A synthetic marker is visible.")])[0])
    return source, windows, requests


def test_same_bytes_fresh_jobs_have_same_clip_prompt_schema_and_wire_images(tmp_path: Path):
    _, _, (first, second) = _request_pair(tmp_path)
    assert first.job_id != second.job_id
    assert first.frame_refs[0].path != second.frame_refs[0].path
    assert first.clip_id == second.clip_id
    for compact in (False, True):
        config = DecisionConfig(compact_frame_ids=compact)
        left = compact_frame_request(first).request if compact else first
        right = compact_frame_request(second).request if compact else second
        bundles = [build_prompt(request, reviewed_event_ids=("synthetic_event",), config=config)
                   for request in (left, right)]
        assert bundles[0].prompt == bundles[1].prompt
        assert bundles[0].prompt_hash == bundles[1].prompt_hash
        assert bundles[0].response_schema == bundles[1].response_schema
        assert first.job_id not in bundles[0].prompt and second.job_id not in bundles[0].prompt
        media_config = LocalModelConfig(media_root=tmp_path / "media", require_readiness=False)
        packets = [_prepare_media(request, media_config, time.monotonic() + 10, None)
                   for request in (left, right)]
        assert packets[0] == packets[1]
    payload = {"facts": [{"fact_id": "VISIBLE", "state": "present",
                           "supporting_frame_ids": [first.frame_refs[0].frame_id], "notes": None}],
               "candidates": [{"candidate_id": "observed", "event_type_id": "synthetic_event",
                   "start_frame_id": first.frame_refs[0].frame_id, "end_frame_id": first.frame_refs[0].frame_id,
                   "supporting_fact_ids": ["VISIBLE"], "supporting_frame_ids": [first.frame_refs[0].frame_id],
                   "visibility_notes": []}]}
    candidates = [validate_decisions(payload, request, reviewed_event_ids=("synthetic_event",)).candidates
                  for request in (first, second)]
    assert candidates[0] == candidates[1]


def test_clip_identity_changes_for_source_or_bounds_but_not_orchestration_index(tmp_path: Path):
    source, windows, (first, _) = _request_pair(tmp_path)
    reindexed = replace(windows[0], index=99)
    again = make_clip_requests(first.job_id, first.frame_refs, [reindexed])[0]
    assert first.clip_id == again.clip_id
    refined = replace(windows[0], purpose="refinement")
    assert first.clip_id != make_clip_requests(first.job_id, first.frame_refs, [refined])[0].clip_id
    shifted = CoverageWindow(0, 0.0, windows[0].end_sec + 0.01, windows[0].timestamps_sec)
    assert first.clip_id != make_clip_requests(first.job_id, first.frame_refs, [shifted])[0].clip_id
    # Appending an ignorable container suffix changes immutable source bytes
    # without changing decoded pixels; source-bound frame IDs still differ.
    changed = tmp_path / "changed.mov"
    changed.write_bytes(source.read_bytes() + b"synthetic-new-content")
    asset = probe_video(changed, job_id=first.job_id)
    refs = decode_frames(asset, tmp_path / "media" / first.job_id / "changed",
                         timestamps_sec=sampling_timestamps(windows), use_subprocess=False)
    assert make_clip_requests(first.job_id, refs, windows)[0].clip_id != first.clip_id


def test_stable_clip_identity_never_grants_cross_job_image_access(tmp_path: Path):
    _, _, (first, second) = _request_pair(tmp_path)
    forged = second.model_copy(deep=True)
    forged.frame_refs[0].path = first.frame_refs[0].path
    assert first.clip_id == forged.clip_id
    config = LocalModelConfig(media_root=tmp_path / "media", require_readiness=False)
    with pytest.raises(UnsafeMedia):
        _prepare_media(forged, config, time.monotonic() + 10, None)
