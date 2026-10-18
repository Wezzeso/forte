"""Explicit synthetic observations; no video model or deployed GPU is called."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

import forte.pipeline as module
from forte.config import WorkerSettings
from forte.pipeline import AnalysisIncomplete, AnalysisPipeline, SCREENING_FACT_IDS
from forte.shared.schemas import FrameRef, VideoAsset
from rules.catalog import load_catalog


@dataclass(frozen=True)
class _Stats:
    synthetic: bool = True


class _Provider:
    def __init__(self, catalog, mode):
        self.catalog, self.mode, self.calls = catalog, mode, []
        self.binding = catalog.bindings["RK_NBK231_P22_RECEIPT_DEPARTURE"]

    async def infer(self, clip, **kwargs):
        screening = clip.clip_id.endswith("-screen")
        refinement = "-refinement-" in clip.clip_id
        phase = "screen" if screening else "refinement" if refinement else "full_primary"
        self.calls.append((phase, clip))
        frame_ids = [frame.frame_id for frame in clip.frame_refs]
        requested = [fact.fact_id for fact in clip.fact_definitions]
        if screening:
            assert set(requested) == set(SCREENING_FACT_IDS)
            assert not clip.rule_definitions
            assert kwargs["response_schema"]["properties"]["candidates"]["maxItems"] == 0
        if self.mode == "provider_failure" and not screening:
            raise RuntimeError("synthetic primary failure")
        states = {key: "present" if screening else "absent" for key in requested}
        if screening and self.mode in {"refuted", "partial_scope"}:
            states = {key: "absent" for key in requested}
        if screening and self.mode == "selective" and clip.start_sec > 0:
            states = {key: "absent" for key in requested}
        if screening and self.mode == "physical_backstop":
            states = {key: "absent" for key in requested}
            states["CASH_VISIBLE"] = "unclear"
        if self.mode == "all_unclear":
            states = {key: "unclear" for key in requested}
        if not screening and self.mode in {"positive", "empty_full"}:
            states.update(self.binding["required_fact_states"])
        if not screening and self.mode == "full_partial_scope":
            for key in ("WORK_AREA_CONTINUOUSLY_VISIBLE", "PROCEDURE_CONTEXT_COMPLETE",
                        "ALL_COUNTING_PATHS_VISIBLE", "PREVIOUSLY_ACCEPTED_CASH_TRACKED", "PREVIOUS_RECEIPT_COMPLETE"):
                if key in states:
                    states[key] = "present"
        partial = (screening and self.mode == "partial_scope") or (not screening and self.mode == "full_partial_scope")
        support = frame_ids[:1] if partial else frame_ids
        facts = [{"fact_id": key, "state": state,
                  "supporting_frame_ids": [] if state == "unclear" else support}
                 for key, state in states.items()]
        if screening and self.mode == "missing_fact":
            facts.pop()
        candidates = []
        if (not screening and self.mode == "positive") or (screening and self.mode == "illegal_screen_candidate"):
            candidates.append({"candidate_id": "synthetic-event", "event_type_id": self.binding["event_type_id"],
                               "start_frame_id": frame_ids[0], "end_frame_id": frame_ids[-1],
                               "supporting_fact_ids": list(states), "supporting_frame_ids": frame_ids,
                               "visibility_notes": []})
        return SimpleNamespace(data={"facts": facts, "candidates": candidates}, metrics={}, finish_reason="stop")


def _setup(monkeypatch, tmp_path, mode, *, duration=1.0, staged=True, budget=8, concise=False, compact=False):
    source = tmp_path / "synthetic.bin"
    source.write_bytes(b"synthetic-source-only")
    digest = sha256(source.read_bytes()).hexdigest()
    settings = WorkerSettings(database_url="sqlite://", data_dir=tmp_path / "runtime",
        model_base_url="http://127.0.0.1:1/v1", model_name="synthetic-staged-provider", model_api_key=None,
        worker_id="synthetic", heartbeat_file=tmp_path / "heartbeat", rules_path=Path("rules/catalog.json"),
        taxonomy_path=Path("annotations/taxonomy.v1.json"), model_manifest_path=Path("models/manifest.json"),
        trusted_scene_context="bank_cash_workstation", staged_screening=staged,
        concise_observations=concise, compact_frame_ids=compact, max_refinement_passes=budget)
    catalog = load_catalog()
    provider = _Provider(catalog, mode)
    pipeline = AnalysisPipeline(settings, provider=provider, catalog=catalog)
    asset = VideoAsset(job_id="synthetic", owned_path=source, content_hash=digest,
                       duration_sec=duration, width=64, height=48)
    decodes = []
    assembled_events = []

    def decode(video, destination, *, timestamps_sec, **kwargs):
        decodes.append(tuple(timestamps_sec))
        return [FrameRef(frame_id=sha256(f"{digest}:{point:.9f}".encode()).hexdigest(), pts_sec=point,
                         path=str(Path(destination) / f"synthetic-{index}.jpg"), width=64, height=48,
                         job_id=asset.job_id) for index, point in enumerate(timestamps_sec)]

    def assemble(**kwargs):
        assembled_events.extend(kwargs["events"])
        return SimpleNamespace(json_bytes=json.dumps({"verdict": "violation" if kwargs["events"] else "no_violation"}).encode(),
                               stats=_Stats())

    monkeypatch.setattr(module, "probe_video", lambda *args, **kwargs: asset)
    monkeypatch.setattr(module, "decode_frames", decode)
    monkeypatch.setattr(module, "assemble_result", assemble)
    job = SimpleNamespace(job_id=asset.job_id, attempt=1, owned_path=source, content_hash=digest)
    return pipeline, provider, job, decodes, assembled_events


async def _run(pipeline, job):
    async def stage(_):
        pass
    try:
        return await pipeline.analyze(job, stage=stage, deadline=time.monotonic() + 30,
                                      cancel_event=threading.Event())
    finally:
        await pipeline.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["refuted", "physical_backstop"])
async def test_refuted_physical_screen_finishes_without_full_pass(monkeypatch, tmp_path, mode):
    pipeline, provider, job, decodes, events = _setup(monkeypatch, tmp_path, mode)
    result = await _run(pipeline, job)
    assert json.loads(result.json_bytes)["verdict"] == "no_violation"
    assert len(provider.calls) == 1 and provider.calls[0][0] == "screen"
    assert len(decodes) == 1 and not events
    audit = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert audit["primary_required"] == audit["primary_completed"] == 0
    assert set(audit["window_decisions"][0]["requested_fact_ids"]) == set(SCREENING_FACT_IDS)
    if mode == "physical_backstop":
        assert audit["limitations"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["full_refute", "partial_scope"])
async def test_unresolved_screen_requires_full_facts_on_same_original_frames(monkeypatch, tmp_path, mode):
    pipeline, provider, job, decodes, _ = _setup(monkeypatch, tmp_path, mode)
    result = await _run(pipeline, job)
    assert json.loads(result.json_bytes)["verdict"] == "no_violation"
    assert [phase for phase, _ in provider.calls] == ["screen", "full_primary"]
    screen, full = [clip for _, clip in provider.calls]
    assert full.frame_refs == screen.frame_refs
    assert (full.start_sec, full.end_sec) == (screen.start_sec, screen.end_sec)
    assert len(full.fact_definitions) > 4 and full.rule_definitions
    assert len(decodes) == 1
    audit = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert [window["phase"] for window in audit["window_decisions"]] == ["screen", "full_primary"]
    assert len(audit["window_decisions"][0]["requested_fact_ids"]) == 4
    assert len(audit["window_decisions"][1]["requested_fact_ids"]) > 4


@pytest.mark.asyncio
async def test_more_than_eight_full_primary_windows_do_not_use_optional_budget(monkeypatch, tmp_path):
    pipeline, provider, job, decodes, _ = _setup(monkeypatch, tmp_path, "full_refute", duration=44.0, budget=1)
    result = await _run(pipeline, job)
    screen = [clip for phase, clip in provider.calls if phase == "screen"]
    full = [clip for phase, clip in provider.calls if phase == "full_primary"]
    assert len(screen) == len(full) > 8
    assert not any(phase == "refinement" for phase, _ in provider.calls)
    assert json.loads(result.json_bytes)["verdict"] == "no_violation"
    assert len(decodes) == 1


@pytest.mark.asyncio
async def test_only_unresolved_screen_windows_receive_full_primary_analysis(monkeypatch, tmp_path):
    pipeline, provider, job, decodes, _ = _setup(monkeypatch, tmp_path, "selective", duration=16.0)
    result = await _run(pipeline, job)
    screens = [clip for phase, clip in provider.calls if phase == "screen"]
    full = [clip for phase, clip in provider.calls if phase == "full_primary"]
    assert len(screens) == 3 and len(full) == 1 and full[0].start_sec == 0
    assert json.loads(result.json_bytes)["verdict"] == "no_violation" and len(decodes) == 1
    audit = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert audit["coverage"]["complete"]
    assert audit["primary_required"] == audit["primary_completed"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["empty_full", "all_unclear", "full_partial_scope"])
async def test_empty_unknown_or_insufficiently_scoped_full_pass_cannot_erase_screen(monkeypatch, tmp_path, mode):
    pipeline, provider, job, _, _ = _setup(monkeypatch, tmp_path, mode)
    with pytest.raises(AnalysisIncomplete, match="insufficient_visible_evidence"):
        await _run(pipeline, job)
    assert any(phase == "full_primary" for phase, _ in provider.calls)
    audit = json.loads((pipeline.settings.work_dir / "synthetic/attempt-1/analysis.private.json").read_text(encoding="utf-8"))
    assert not audit["publication_ready"] and audit["phase"] == "failed"
    assert "unresolved_screen_branch" in audit["limitations"]


@pytest.mark.asyncio
async def test_positive_event_comes_only_from_full_fact_observations(monkeypatch, tmp_path):
    pipeline, provider, job, _, events = _setup(monkeypatch, tmp_path, "positive")
    result = await _run(pipeline, job)
    assert json.loads(result.json_bytes)["verdict"] == "violation" and events
    audit = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert all(not window["candidate_events"] for window in audit["window_decisions"] if window["phase"] == "screen")
    assert any(window["candidate_events"] for window in audit["window_decisions"] if window["phase"] != "screen")
    assert all(set(pipeline.catalog.rules[event.rule_id].required_facts)
               <= {fact.fact_id for fact in event.observed_facts} for event in events)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["missing_fact", "illegal_screen_candidate", "provider_failure"])
async def test_screen_contract_and_primary_technical_failures_remain_failures(monkeypatch, tmp_path, mode):
    from forte.decision import DecisionOutputError
    pipeline, provider, job, _, _ = _setup(monkeypatch, tmp_path, mode)
    with pytest.raises((DecisionOutputError, RuntimeError)):
        await _run(pipeline, job)


@pytest.mark.asyncio
async def test_disabled_flag_preserves_full_fact_baseline_and_both_engines_share_concise_setting(monkeypatch, tmp_path):
    pipeline, provider, job, _, _ = _setup(monkeypatch, tmp_path, "full_refute", staged=False, concise=True)
    assert pipeline.engine.config.concise_observations
    assert pipeline.screening_engine.config.concise_observations
    result = await _run(pipeline, job)
    assert json.loads(result.json_bytes)["verdict"] == "no_violation"
    assert [phase for phase, _ in provider.calls] == ["full_primary"]


@pytest.mark.asyncio
async def test_compact_transport_ids_are_restored_in_both_stage_audits(monkeypatch, tmp_path):
    pipeline, provider, job, _, _ = _setup(monkeypatch, tmp_path, "full_refute", compact=True, concise=True)
    result = await _run(pipeline, job)
    assert all(len(frame.frame_id) < 10 for _, clip in provider.calls for frame in clip.frame_refs)
    audit = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert all(len(frame_id) == 64 for window in audit["window_decisions"] for frame_id in window["frame_ids"])
    assert [window["phase"] for window in audit["window_decisions"]] == ["screen", "full_primary"]
