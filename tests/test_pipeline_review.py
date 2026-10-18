"""Peer-review regressions over generated videos and explicit fake observations.

These scenarios contain no supplied recording, annotation, or learned detector.
"""

import asyncio
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import shutil
import threading
import time
from types import SimpleNamespace

import av
from PIL import Image
import pytest

from forte.config import WorkerSettings
from forte.pipeline import AnalysisIncomplete, AnalysisPipeline, _reconcile_batches
from forte.decision import DecisionBatch
from forte.shared.schemas import CandidateEvent, ClipRequest, FactDecision, FrameRef, RuleRecord
from rules.catalog import load_catalog


def _video(path: Path, seconds: int) -> str:
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=4)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for index in range(seconds * 4):
            with Image.new("RGB", (64, 48), (index % 200, 50, 100)) as image:
                frame = av.VideoFrame.from_image(image)
                frame.pts, frame.time_base = index, Fraction(1, 4)
                for packet in stream.encode(frame):
                    output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return sha256(path.read_bytes()).hexdigest()


def _settings(tmp_path: Path):
    return WorkerSettings(
        database_url="sqlite://", data_dir=tmp_path / "runtime", model_base_url="http://127.0.0.1:1/v1",
        model_name="synthetic-review-only", model_api_key=None, worker_id="synthetic-review",
        heartbeat_file=tmp_path / "heartbeat", rules_path=Path("rules/catalog.json"),
        taxonomy_path=Path("annotations/taxonomy.v1.json"), model_manifest_path=Path("models/manifest.json"),
        trusted_scene_context="bank_cash_workstation",
    )


class _Provider:
    def __init__(self, catalog, mode):
        self.catalog, self.mode, self.intervals = catalog, mode, []
        self.binding = catalog.bindings[
            "RK_NBK231_P19_2_RECEIPT_COUNT" if mode.startswith("long_context") else "RK_NBK231_P22_RECEIPT_DEPARTURE"
        ]

    async def infer(self, clip, **kwargs):
        if self.mode == "provider_failure":
            raise RuntimeError("synthetic review provider failure")
        refined = "-refinement-" in clip.clip_id
        self.intervals.append((clip.start_sec, clip.end_sec, refined))
        frames = list(clip.frame_refs)
        support = [frame.frame_id for frame in frames]
        states = dict(self.binding["required_fact_states"])
        if self.mode.startswith("long_context") and refined and clip.end_sec - clip.start_sec < (16 if self.mode.endswith("16") else 8):
            states["PROCEDURE_CONTEXT_COMPLETE"] = "absent"
        if self.mode == "explicit_refutation" and refined:
            states = {}
        if self.mode == "facts_then_refute" and refined:
            states = {}
        if self.mode == "irrelevant_unclear":
            states = {}
        facts = [{"fact_id": definition.fact_id, "state": states.get(definition.fact_id, "absent"),
                  "supporting_frame_ids": support, "notes": None}
                 for definition in clip.fact_definitions]
        if self.mode == "irrelevant_unclear":
            for observed in facts:
                if observed["fact_id"] == "CASH_COUNT_VISIBLE":
                    observed.update(state="unclear", supporting_frame_ids=[])
        candidates = []
        # Only the first coverage window proposes an event. Later coarse
        # windows remain complete so the regression concerns refinement alone.
        missing_modes = {"positive_facts_no_candidate", "facts_then_refute", "facts_then_candidate", "irrelevant_unclear"}
        if ((not refined and clip.start_sec == 0.0 and self.mode not in missing_modes)
                or (refined and (self.mode.startswith("long_context") or self.mode == "facts_then_candidate"))):
            target = 5.0 if clip.end_sec >= 5 else clip.end_sec / 2
            selected = min(frames, key=lambda frame: abs(frame.pts_sec - target))
            candidates = [{"candidate_id": "review-candidate", "event_type_id": self.binding["event_type_id"],
                           "start_frame_id": selected.frame_id, "end_frame_id": selected.frame_id,
                           "supporting_fact_ids": list(self.binding["required_fact_states"]),
                           "supporting_frame_ids": [selected.frame_id], "visibility_notes": []}]
        return SimpleNamespace(data={"facts": facts, "candidates": candidates}, metrics={}, finish_reason="stop")


async def _run(tmp_path, mode, seconds):
    source = tmp_path / "generated.mov"
    digest = _video(source, seconds)
    catalog = load_catalog()
    provider = _Provider(catalog, mode)
    pipeline = AnalysisPipeline(_settings(tmp_path), provider=provider, catalog=catalog)
    job = SimpleNamespace(job_id="synthetic-review", attempt=1, owned_path=source, content_hash=digest)

    async def stage(_):
        pass

    try:
        output = await pipeline.analyze(job, stage=stage, deadline=time.monotonic() + 60,
                                        cancel_event=threading.Event())
        return json.loads(output.json_bytes), provider
    finally:
        await pipeline.aclose()


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="Synthetic pipeline requires FFprobe")
def test_refinement_cannot_silently_erase_a_supported_coarse_occurrence(tmp_path):
    with pytest.raises(AnalysisIncomplete, match="insufficient_visible_evidence") as rejected:
        asyncio.run(_run(tmp_path, "candidate_disappears", 1))
    audit = json.loads((_settings(tmp_path).work_dir / "synthetic-review/attempt-1/analysis.private.json").read_text(encoding="utf-8"))
    assert audit["schema_version"] == "forte.local-proposal.v1"
    assert audit["phase"] == "failed" and not audit["publication_ready"]
    assert audit["candidate_events"] and audit["window_decisions"] and audit["decisions"]
    assert audit["provenance"]["model_revision"] is None
    assert audit["provenance"]["actual_snapshot_verified"] is None
    assert Path(audit["asset"]["owned_path"]).is_absolute()
    assert all(Path(frame["path"]).is_absolute() for frame in audit["frames"])
    assert audit["timings"]["model_requests"]["request_record_count"] == len(audit["window_decisions"])
    assert rejected.value.forte_timings["model_requests"]["request_record_count"] > 0


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="Synthetic pipeline requires FFprobe")
def test_omission_refinement_requests_complete_procedure_context(tmp_path):
    result, provider = asyncio.run(_run(tmp_path, "long_context", 12))
    assert result["verdict"] == "violation"
    assert any(refined and end - start >= 8 for start, end, refined in provider.intervals)


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="Synthetic pipeline requires FFprobe")
def test_additional_rule_refinement_requests_are_executed(tmp_path):
    result, provider = asyncio.run(_run(tmp_path, "long_context_16", 20))
    assert result["verdict"] == "violation"
    assert sum(refined for _, _, refined in provider.intervals) >= 2
    assert any(refined and end - start >= 16 for start, end, refined in provider.intervals)


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="Synthetic pipeline requires FFprobe")
def test_full_context_explicit_refutation_can_clear_a_false_candidate(tmp_path):
    result, _ = asyncio.run(_run(tmp_path, "explicit_refutation", 1))
    assert result["verdict"] == "no_violation"


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="Synthetic pipeline requires FFprobe")
def test_provider_failure_retains_owned_source_audit_without_a_negative(tmp_path):
    with pytest.raises(RuntimeError, match="synthetic review provider failure"):
        asyncio.run(_run(tmp_path, "provider_failure", 1))
    audit = json.loads((_settings(tmp_path).work_dir / "synthetic-review/attempt-1/analysis.private.json").read_text(encoding="utf-8"))
    assert audit["phase"] == "failed" and not audit["publication_ready"]
    assert audit["error_code"] == "coarse_model_incomplete"
    assert not audit["coverage"]["complete"] and audit["frames"]
    assert not audit["decisions"] and not audit["verified_events"]


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="Synthetic pipeline requires FFprobe")
def test_supported_original_facts_without_any_candidate_cannot_become_normal(tmp_path):
    with pytest.raises(AnalysisIncomplete, match="insufficient_visible_evidence"):
        asyncio.run(_run(tmp_path, "positive_facts_no_candidate", 1))
    audit = json.loads((_settings(tmp_path).work_dir / "synthetic-review/attempt-1/analysis.private.json").read_text(encoding="utf-8"))
    assert "rule_facts_without_candidate" in audit["limitations"]


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="Synthetic pipeline requires FFprobe")
def test_later_explicit_full_context_refutation_can_resolve_missing_proposal(tmp_path):
    result, _ = asyncio.run(_run(tmp_path, "facts_then_refute", 1))
    assert result["verdict"] == "no_violation"


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="Synthetic pipeline requires FFprobe")
def test_later_actual_candidate_can_resolve_missing_original_proposal(tmp_path):
    result, _ = asyncio.run(_run(tmp_path, "facts_then_candidate", 1))
    assert result["verdict"] == "violation"


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="Synthetic pipeline requires FFprobe")
def test_irrelevant_unclear_can_finish_normal_without_spending_refinement_budget(tmp_path):
    result, provider = asyncio.run(_run(tmp_path, "irrelevant_unclear", 1))
    assert result["verdict"] == "no_violation"
    assert len(provider.intervals) == 1
    audit = json.loads((_settings(tmp_path).work_dir / "synthetic-review/attempt-1/analysis.private.json").read_text(encoding="utf-8"))
    assert "unclear_visual_context_irrelevant_to_refuted_rules" in audit["limitations"]
    assert audit["publication_ready"] and audit["phase"] == "done"


def _batch(clip_id, start, end, points=(), *, state="present", unclear=False, kind="TYPE"):
    refs = [FrameRef(frame_id=f"f{value}", pts_sec=float(value), path=f"synthetic/f{value}.jpg",
                     width=32, height=32, job_id="synthetic") for value in range(21)
            if start <= value <= end]
    observed = FactDecision(fact_id="ACTION", state="unclear" if unclear else state,
                            supporting_frame_ids=[] if unclear else [frame.frame_id for frame in refs])
    request = ClipRequest(job_id="synthetic", clip_id=clip_id, frame_refs=refs,
                          start_sec=float(start), end_sec=float(end))
    candidates = tuple(CandidateEvent(candidate_id=f"c{index}", event_type_id=kind,
                        start_sec=float(point), end_sec=float(point), supporting_facts=[observed],
                        supporting_frame_ids=[f"f{point}"], clip_id=clip_id)
                       for index, point in enumerate(points))
    return DecisionBatch(request=request, facts=(observed,), candidates=candidates)


def _catalog():
    record = RuleRecord(rule_id="RULE", title="Synthetic fixture", source_url="https://example.invalid/rule",
                         applicability="Synthetic only", required_facts=["ACTION"], enabled=True,
                         applicability_verified=True, verification_state="verified")
    return SimpleNamespace(rules={"RULE": record},
                           bindings={"RULE": {"event_type_id": "TYPE", "required_fact_states": {"ACTION": "present"}}})


def test_narrow_opposite_state_does_not_refute_a_complete_parent_procedure():
    coarse = _batch("coarse", 0, 8, [3])
    narrow = _batch("narrow", 2, 4, state="absent")
    _, pending, requests = _reconcile_batches([coarse], [[narrow]], _catalog())
    assert len(pending) == 1 and requests
    wide = _batch("wide", 0, 12, state="absent")
    _, pending, _ = _reconcile_batches([coarse], [[narrow], [wide]], _catalog())
    assert not pending


def test_two_distinct_candidates_cannot_be_resolved_by_one_refined_occurrence():
    coarse = _batch("coarse", 0, 8, [3, 5])
    refined = _batch("refined", 0, 12, [3])
    final, pending, _ = _reconcile_batches([coarse], [[refined]], _catalog())
    assert len(pending) == 1 and pending[0].start_sec == 5
    assert sum(len(batch.candidates) for batch in final) == 1


def test_overlap_duplicates_from_different_coarse_clips_can_share_confirmation():
    one = _batch("coarse1", 0, 8, [5])
    two = _batch("coarse2", 4, 12, [5])
    refined = _batch("refined", 0, 12, [5])
    final, pending, _ = _reconcile_batches([one, two], [[refined]], _catalog())
    assert not pending
    assert sum(len(batch.candidates) for batch in final) == 1


def test_empty_refinement_with_identical_supporting_facts_retains_pending():
    coarse = _batch("coarse", 0, 8, [3])
    refined = _batch("refined", 0, 12)
    _, pending, _ = _reconcile_batches([coarse], [[refined]], _catalog())
    assert any(item.reason == "unresolved_coarse_candidate" for item in pending)
    assert any(item.reason == "rule_facts_without_candidate" for item in pending)


def test_a_different_event_type_never_confirms_the_original_type():
    coarse = _batch("coarse", 0, 8, [3])
    refined = _batch("refined", 0, 12, [3], kind="OTHER_TYPE")
    _, pending, _ = _reconcile_batches([coarse], [[refined]], _catalog())
    assert any(item.reason == "unresolved_coarse_candidate" and item.event_type_id == "TYPE" for item in pending)


def test_only_full_containing_clear_context_supersedes_uncertainty():
    coarse = _batch("coarse", 0, 8, unclear=True)
    narrow = _batch("narrow", 2, 4)
    final, _, _ = _reconcile_batches([coarse], [[narrow]], _catalog())
    assert any(batch.has_uncertainty for batch in final)
    wide = _batch("wide", 0, 12)
    final, _, _ = _reconcile_batches([coarse], [[narrow], [wide]], _catalog())
    assert all(not batch.has_uncertainty for batch in final)


def test_refutation_must_exclude_every_applicable_binding_for_the_event_type():
    catalog = _catalog()
    other = catalog.rules["RULE"].model_copy(update={"rule_id": "OTHER", "required_facts": ["OTHER_ACTION"]})
    catalog.rules["OTHER"] = other
    catalog.bindings["OTHER"] = {"event_type_id": "TYPE", "required_fact_states": {"OTHER_ACTION": "present"}}
    coarse = _batch("coarse", 0, 8, [3])
    refined = _batch("refined", 0, 12, state="absent")
    _, pending, _ = _reconcile_batches([coarse], [[refined]], catalog)
    assert pending  # ACTION absent does not establish OTHER_ACTION absent.
