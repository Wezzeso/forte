"""Synthetic pipeline orchestration regressions; no real inference or media."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import pytest

import forte.pipeline as pipeline_module
from forte.pipeline import AnalysisPipeline
from forte.shared.schemas import CandidateEvent, FactDecision, FrameRef, VideoAsset


@dataclass(frozen=True)
class _Batch:
    request: object
    candidates: tuple = ()
    uncertainty: bool = False
    facts: tuple = ()
    prompt_version: str = "synthetic-test-only"
    config_version: str = "synthetic-test-only"
    prompt_hash: str = "0" * 64

    @property
    def has_uncertainty(self):
        return self.uncertainty

    @property
    def start_sec(self):
        return self.request.start_sec

    @property
    def end_sec(self):
        return self.request.end_sec


@dataclass(frozen=True)
class _Stats:
    fixture: bool = True


def _isolated_pipeline(monkeypatch, tmp_path, *, uncertainty=False, duration=20.0):
    """Keep real planning/binding/accounting and replace expensive I/O boundaries."""
    pipeline = AnalysisPipeline.__new__(AnalysisPipeline)
    pipeline.settings = SimpleNamespace(
        work_dir=tmp_path / "work", window_seconds=8.0, stride_seconds=4.0,
        sample_fps=2.0, max_frames_per_window=16, max_concurrent_clips=2,
        max_refinement_passes=8, model_name="synthetic-test-only",
        compact_frame_ids=False, concise_observations=False,
        staged_screening=False, max_inflight_per_job=1,
    )
    pipeline.catalog = SimpleNamespace(version="synthetic-test-only", rules={}, bindings={
        "synthetic-rule": {"event_type_id": "synthetic-action", "is_omission": False,
                           "required_fact_states": {"synthetic-fact": "present"}},
    })
    pipeline.facts, pipeline.rules, pipeline.event_config = [], [], None
    job = SimpleNamespace(job_id="synthetic-job", attempt=1, owned_path=str(tmp_path / "fixture.mov"),
                          content_hash="0" * 64)
    asset = VideoAsset(job_id=job.job_id, owned_path=job.owned_path, content_hash=job.content_hash,
                       duration_sec=duration, width=64, height=48)
    decoded_targets = []
    observed_requests = []
    coarse_candidates = []

    def probe(*args, **kwargs):
        return asset

    def decode(video, destination, *, timestamps_sec, **kwargs):
        decoded_targets.append(tuple(timestamps_sec))
        return [FrameRef(frame_id=f"synthetic-{point:.8f}", pts_sec=point,
                         path=str(Path(destination) / f"frame-{index}.jpg"), width=64, height=48,
                         job_id=job.job_id)
                for index, point in enumerate(timestamps_sec)]

    def candidate(request, start, end, identity):
        first = min(request.frame_refs, key=lambda frame: abs(frame.pts_sec - start))
        last = min(request.frame_refs, key=lambda frame: abs(frame.pts_sec - end))
        support = list(dict.fromkeys((first.frame_id, last.frame_id)))
        fact = FactDecision(fact_id="synthetic-fact", state="present", supporting_frame_ids=support)
        return CandidateEvent(candidate_id=identity, event_type_id="synthetic-action",
                              start_sec=min(first.pts_sec, last.pts_sec),
                              end_sec=max(first.pts_sec, last.pts_sec),
                              supporting_facts=[fact], supporting_frame_ids=support,
                              clip_id=request.clip_id)

    async def infer(request, deadline, cancelled):
        observed_requests.append(request)
        if "-coverage-" in request.clip_id:
            if uncertainty and request.start_sec == 0.0:
                return _Batch(request, uncertainty=True)
            if not uncertainty and request.start_sec == 0.0:
                proposed = candidate(request, 2.0, 3.0, "synthetic-opening")
                coarse_candidates.append(proposed)
                return _Batch(request, candidates=(proposed,), facts=tuple(proposed.supporting_facts))
            if not uncertainty and request.end_sec == duration:
                proposed = candidate(request, duration - 3.0, duration - 2.0, "synthetic-tail")
                coarse_candidates.append(proposed)
                return _Batch(request, candidates=(proposed,), facts=tuple(proposed.supporting_facts))
        else:
            # Return matching source-bound observations, so this regression
            # exercises one successful confirmation round rather than retries
            # for a deliberately unresolved/disappearing proposal.
            confirmed = tuple(candidate(request, proposed.start_sec, proposed.end_sec,
                                        proposed.candidate_id)
                              for proposed in coarse_candidates
                              if request.start_sec <= proposed.start_sec <= proposed.end_sec <= request.end_sec)
            return _Batch(request, candidates=confirmed,
                          facts=tuple(fact for proposed in confirmed for fact in proposed.supporting_facts))
        return _Batch(request)

    def events(*args, **kwargs):
        return SimpleNamespace(can_publish=True, verified_events=(), limitations=(), refinement_requests=())

    def assemble(**kwargs):
        return SimpleNamespace(json_bytes=b"{}", stats=_Stats())

    monkeypatch.setattr(pipeline_module, "probe_video", probe)
    monkeypatch.setattr(pipeline_module, "decode_frames", decode)
    monkeypatch.setattr(pipeline_module, "build_events", events)
    monkeypatch.setattr(pipeline_module, "assemble_result", assemble)
    pipeline._infer = infer
    return pipeline, job, decoded_targets, observed_requests


async def _stage(_):
    pass


@pytest.mark.asyncio
async def test_multiple_refinements_share_one_forward_decode(monkeypatch, tmp_path):
    pipeline, job, decodes, requests = _isolated_pipeline(monkeypatch, tmp_path)
    await pipeline.analyze(job, stage=_stage, deadline=time.monotonic() + 30,
                           cancel_event=threading.Event())
    refinements = [request for request in requests if "-refinement-" in request.clip_id]
    assert len(refinements) >= 2
    assert len(decodes) <= 2, "Decode baseline once and the union of refinement targets once"
    assert all(len(request.frame_refs) <= 16 for request in requests)


@pytest.mark.asyncio
async def test_uncertain_window_receives_new_temporal_context(monkeypatch, tmp_path):
    pipeline, job, _, requests = _isolated_pipeline(monkeypatch, tmp_path, uncertainty=True)
    await pipeline.analyze(job, stage=_stage, deadline=time.monotonic() + 30,
                           cancel_event=threading.Event())
    coarse = next(request for request in requests if "-coverage-" in request.clip_id and request.start_sec == 0.0)
    refinements = [request for request in requests if "-refinement-" in request.clip_id]
    assert refinements
    assert any(request.start_sec < coarse.start_sec or request.end_sec > coarse.end_sec
               for request in refinements), "An uncertain procedure needs context beyond the same coarse window"
    coarse_pts = {frame.pts_sec for frame in coarse.frame_refs}
    assert any({frame.pts_sec for frame in request.frame_refs} != coarse_pts for request in refinements)
    assert all(len(request.frame_refs) <= 16 for request in refinements)
