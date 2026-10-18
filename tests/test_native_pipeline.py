"""Synthetic orchestration regressions for the opt-in native head worker.

No supplied recording, trained model, teacher, labels or GPU is opened. The
source/probe/client/materializer below are explicit test doubles; generated JPEGs
exercise the ordinary evidence bytes and deterministic legal builder.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import httpx
from PIL import Image
import pytest

from forte.config import WorkerSettings
from forte.decision.head_contract import TemporalHeadOutput
from forte.decision.temporal_http import HttpNativeHeadClient
from forte.decision.temporal_native import NativeHeadResult, NativeSourceContext
from forte.decision.temporal_provider import (
    DECODER_SHA256, TemporalProviderError, load_physical_fact_run_policy,
    physical_policy_observation_definitions,
)
from forte.native_pipeline import NativeAnalysisPipeline
from forte.pipeline import AnalysisIncomplete
from forte.shared.schemas import FrameRef, VideoAsset
from forte.video.batches import source_frame_id
from rules.catalog import load_catalog
from test_machine_calibration_contract import SyntheticVerifier, digest, manifest


def _settings(tmp_path):
    return WorkerSettings(
        database_url="sqlite://", data_dir=tmp_path / "runtime",
        model_base_url="http://127.0.0.1:1/v1", model_name="synthetic-native-only",
        model_api_key=None, worker_id="synthetic-native-worker",
        heartbeat_file=tmp_path / "heartbeat", rules_path=Path("rules/catalog.json"),
        taxonomy_path=Path("annotations/taxonomy.v1.json"),
        model_manifest_path=Path("models/manifest.json"),
        trusted_scene_context="bank_cash_workstation",
    )


def _fixture(tmp_path, *, occurrence_indices=(4, 12)):
    """Actual source IDs, nonzero original PTS, calibrated synthetic outputs."""
    catalog = load_catalog()
    baseline = manifest()
    facts = tuple(fact.fact_id for fact in catalog.fact_definitions())
    events = tuple(sorted(catalog.event_types))
    identity = baseline.identity.model_copy(update={"fact_ids": facts, "event_type_ids": events,
        "taxonomy_sha256": sha256(Path("annotations/taxonomy.v1.json").read_bytes()).hexdigest()})
    fact_record = next(iter(baseline.fact_calibration.values()))
    event_record = next(iter(baseline.event_calibration.values()))
    calibration = baseline.model_copy(update={
        "identity": identity, "decoder_sha256": DECODER_SHA256,
        "fact_calibration": {key: fact_record for key in facts},
        "event_calibration": {key: event_record for key in events},
    })
    source = tmp_path / "arbitrary upload name not in any training split.mov"
    source.write_bytes(b"Explicit synthetic source double, not a video or supplied media.")
    job = SimpleNamespace(job_id="synthetic-native-worker", attempt=1,
                          owned_path=source, content_hash=sha256(source.read_bytes()).hexdigest())
    asset = VideoAsset(job_id=job.job_id, owned_path=str(source.resolve()),
        content_hash=job.content_hash, duration_sec=4.0, width=32, height=16,
        fps=4.0, time_base="1/4", source_start_sec=25.0, frame_count=16)
    frames = [{
        "frame_id": source_frame_id(asset.content_hash, 100 + index, 1, 4),
        "source_pts": 100 + index,
        "next_source_pts": 101 + index if index < 15 else None,
        "pts_sec": index / 4,
        "next_pts_sec": (index + 1) / 4 if index < 15 else None,
        "source_end_bound_sec": 4.0 if index == 15 else None,
        "width": 32, "height": 16, "applied_rotation_deg": 0,
        "key_frame": index == 0, "view_mask": [True], "view_roles": ["global"],
        "normalized_boxes": [[0.0, 0.0, 1.0, 1.0]],
    } for index in range(16)]
    context = NativeSourceContext.model_validate_json(json.dumps({
        "job_id": job.job_id, "clip_id": "synthetic-native-clip",
        "content_sha256": asset.content_hash, "duration_sec": 4.0,
        "time_base_num": 1, "time_base_den": 4, "origin_pts": 100, "stream_index": 0,
        "feature_spec_sha256": identity.feature_spec_sha256,
        "encoder_manifest_sha256": identity.encoder_artifacts_sha256,
        "source_complete": True, "tail_included": True, "frames": frames,
    }))
    target = catalog.bindings["RK_NBK231_P20_PREVIOUS_CASH_POCKET"]["event_type_id"]
    boundary = [[[-4.0, -4.0] for _ in events] for _ in frames]
    for index in occurrence_indices:
        boundary[index][events.index(target)] = [4.0, 4.0]
    localized = [[[0.0, 4.0, -4.0] for _ in facts] for _ in frames]
    for fact_id in ("CASH_PLACED_IN_CLOTHING_POCKET", "CASH_PLACED_IN_PERSONAL_CONTAINER"):
        axis = facts.index(fact_id)
        for index in range(len(frames)):
            if index not in occurrence_indices:
                localized[index][axis] = [-4.0, 0.0, 4.0]
    output = TemporalHeadOutput.model_validate_json(json.dumps({
        "identity": identity.model_dump(mode="json"), "job_id": asset.job_id,
        "clip_id": context.clip_id,
        "frames": [{key: frame[key] for key in ("frame_id", "pts_sec", "source_pts", "next_source_pts")}
                   for frame in frames],
        "window_logits": [[0.0, 4.0, -4.0] for _ in facts],
        "frame_logits": localized,
        "support_logits": [[4.0 for _ in facts] for _ in frames],
        "boundary_logits": boundary, "frame_valid": [True for _ in frames],
        "global_view_available": [True for _ in frames], "feature_coverage_complete": True,
        "native_source_continuity_verified": True,
        "trained_components": {"window_fact_states": True, "frame_fact_states": True,
                               "frame_support": True, "event_boundaries": True},
    }))
    result = NativeHeadResult(context=context, output=output, video_logit=-100.0,
        metrics={"head_elapsed_sec": 0.01, "native_frames": 16.0})
    return SimpleNamespace(catalog=catalog, manifest=calibration, job=job, asset=asset, result=result)


class _Client:
    def __init__(self, result, *, error=None):
        self.result, self.error = result, error
        self.identity = result.output.identity
        self.calls, self.closed = [], False

    async def check_ready(self, *, deadline):
        return {"ready": True, "head_sha256": self.identity.head_sha256,
                "feature_spec_sha256": self.identity.feature_spec_sha256}

    async def predict_native(self, asset, *, clip_id, deadline, cancel_event=None):
        self.calls.append((asset, clip_id, deadline, cancel_event))
        if self.error is not None:
            raise self.error
        # The real service echoes the arbitrary upload's requested clip identity.
        context = self.result.context.model_copy(update={"clip_id": clip_id})
        output = self.result.output.model_copy(update={"clip_id": clip_id})
        return self.result.model_copy(update={"context": context, "output": output})

    async def aclose(self):
        self.closed = True


class _Materializer:
    def __init__(self, *, tamper=None, error=None):
        self.calls, self.tamper, self.error = [], tamper, error

    def __call__(self, asset, selected_sources, output_dir, *, source_index, **kwargs):
        selected_sources = tuple(selected_sources)
        self.calls.append((asset, selected_sources, tuple(source_index), kwargs))
        assert 0 < len(selected_sources) <= 3
        assert set(source.frame_id for source in selected_sources) <= set(source.frame_id for source in source_index)
        if self.error is not None:
            raise self.error
        directory = Path(output_dir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        refs = []
        for source in selected_sources:
            path = directory / f"{source.frame_id}.jpg"
            Image.new("RGB", (source.width, source.height), (10, 30, 50)).save(path)
            refs.append(FrameRef(frame_id=source.frame_id, source_pts=source.source_pts,
                pts_sec=source.pts_sec, path=str(path), width=source.width, height=source.height,
                job_id=asset.job_id, jpeg_sha256=sha256(path.read_bytes()).hexdigest()))
        if self.tamper == "source_pts":
            refs[0] = refs[0].model_copy(update={"source_pts": refs[0].source_pts + 1})
        elif self.tamper == "jpeg_hash":
            refs[0] = refs[0].model_copy(update={"jpeg_sha256": digest("wrong-jpeg")})
        elif self.tamper == "frame_id":
            refs[0] = refs[0].model_copy(update={"frame_id": digest("wrong-source-frame")})
        return refs


def _pipeline(tmp_path, fixture, *, client=None, materializer=None, probe=None, verifier=None,
              absence_context_policy=None, trusted_scene_context="bank_cash_workstation"):
    client = client or _Client(fixture.result)
    materializer = materializer or _Materializer()
    probes = []

    def fake_probe(path, **kwargs):
        probes.append((path, kwargs))
        return fixture.asset

    settings = replace(_settings(tmp_path), trusted_scene_context=trusted_scene_context)
    pipeline = NativeAnalysisPipeline(settings, client=client,
        manifest=fixture.manifest, verifier=verifier or SyntheticVerifier(),
        catalog=fixture.catalog, probe=probe or fake_probe, materializer=materializer,
        absence_context_policy=absence_context_policy)
    return pipeline, client, materializer, probes


async def _analyze(pipeline, job, *, cancel_event=None):
    stages = []

    async def stage(value):
        stages.append(value)

    result = await pipeline.analyze(job, stage=stage, deadline=time.monotonic() + 10,
                                    cancel_event=cancel_event or threading.Event())
    return result, stages


def _audit(tmp_path, job):
    path = _settings(tmp_path).work_dir / job.job_id / f"attempt-{job.attempt}" / "analysis.private.json"
    assert path.is_file()
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_native_arbitrary_upload_keeps_two_source_occurrences_and_bounded_original_evidence(tmp_path):
    fixture = _fixture(tmp_path)
    pipeline, client, materializer, probes = _pipeline(tmp_path, fixture)
    try:
        result, stages = await _analyze(pipeline, fixture.job)
    finally:
        await pipeline.aclose()
    public = json.loads(result.json_bytes)
    assert public["verdict"] == "violation"
    assert len(public["violations"]) == 2
    assert [event["timecode_sec"] for event in public["violations"]] == [1.0, 3.0]
    assert len(client.calls) == 1 and client.calls[0][0].owned_path == fixture.asset.owned_path
    assert Path(probes[0][0]).resolve() == fixture.job.owned_path.resolve()
    assert stages and client.closed
    assert materializer.calls
    selected = [source.frame_id for _, sources, _, _ in materializer.calls for source in sources]
    assert len(selected) == len(set(selected))
    assert set(selected) <= {source.frame_id for source in fixture.result.context.frames}
    assert all(len(sources) <= 3 for _, sources, _, _ in materializer.calls)
    assert result.metadata_path.is_file()
    audit = _audit(tmp_path, fixture.job)
    assert audit["phase"] == "done" and audit["publication_ready"]
    assert "arbitrary upload" not in result.json_bytes.decode()
    assert "-100.0" not in result.json_bytes.decode()  # A negative MIL logit never suppresses real source events.


@pytest.mark.asyncio
async def test_native_boundary_occurrences_are_not_duplicated_by_broad_physical_fact_runs(tmp_path):
    fixture = _fixture(tmp_path)
    output = fixture.result.output
    # A weakly localized physical run spans the full video while the trained
    # calibrated boundary decoder identifies two distinct original occurrences.
    output = output.model_copy(update={"frame_logits": tuple(
        tuple((0.0, 4.0, -4.0) for _ in output.identity.fact_ids) for _ in output.frames)})
    fixture.result = fixture.result.model_copy(update={"output": output})
    pipeline, _, _, _ = _pipeline(tmp_path, fixture)
    try:
        result, _ = await _analyze(pipeline, fixture.job)
    finally:
        await pipeline.aclose()
    public = json.loads(result.json_bytes)
    assert [event["timecode_sec"] for event in public["violations"]] == [1.0, 3.0]
    assert len(_audit(tmp_path, fixture.job)["candidate_events"]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["unclear", "untrained", "invalid_frame", "missing_global"])
async def test_native_unknown_untrained_or_uninspected_source_never_becomes_clean_negative(tmp_path, mutation):
    fixture = _fixture(tmp_path, occurrence_indices=())
    _normal_observations(fixture)
    output = fixture.result.output
    if mutation == "unclear":
        output = output.model_copy(update={"window_logits": tuple((-4.0, 0.0, 4.0) for _ in output.window_logits)})
    elif mutation == "untrained":
        output = output.model_copy(update={"trained_components": {**output.trained_components, "frame_support": False}})
    elif mutation == "invalid_frame":
        output = output.model_copy(update={"frame_valid": (False, *output.frame_valid[1:])})
    else:
        output = output.model_copy(update={"global_view_available": (False, *output.global_view_available[1:])})
    fixture.result = fixture.result.model_copy(update={"output": output})
    pipeline, _, _, _ = _pipeline(tmp_path, fixture, absence_context_policy=_absence_policy(fixture))
    try:
        with pytest.raises((AnalysisIncomplete, TemporalProviderError)):
            await _analyze(pipeline, fixture.job)
    finally:
        await pipeline.aclose()
    audit = _audit(tmp_path, fixture.job)
    assert audit["phase"] == "failed" and not audit["publication_ready"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["content_hash", "job_id", "source_pts", "clip_id", "head_identity", "coverage"])
async def test_native_service_identity_or_source_tampering_fails_before_evidence(tmp_path, mutation):
    fixture = _fixture(tmp_path)
    result = fixture.result
    if mutation in {"content_hash", "job_id"}:
        field = "content_sha256" if mutation == "content_hash" else "job_id"
        result = result.model_copy(update={"context": result.context.model_copy(update={field: digest("other")})})
    elif mutation == "source_pts":
        frames = list(result.output.frames)
        frames[0] = frames[0].model_copy(update={"source_pts": 99})
        result = result.model_copy(update={"output": result.output.model_copy(update={"frames": tuple(frames)})})
    elif mutation == "head_identity":
        identity = result.output.identity.model_copy(update={"head_sha256": digest("other-head")})
        result = result.model_copy(update={"output": result.output.model_copy(update={"identity": identity})})
    elif mutation == "coverage":
        result = result.model_copy(update={"output": result.output.model_copy(update={"feature_coverage_complete": False})})
    client = _Client(result)
    client.identity = fixture.manifest.identity
    if mutation == "clip_id":
        async def wrong_clip(asset, *, clip_id, **kwargs):
            return result.model_copy(update={
                "context": result.context.model_copy(update={"clip_id": "unrequested-clip"}),
                "output": result.output.model_copy(update={"clip_id": "unrequested-clip"}),
            })
        client.predict_native = wrong_clip
    pipeline, _, materializer, _ = _pipeline(tmp_path, fixture, client=client)
    try:
        with pytest.raises((AnalysisIncomplete, TemporalProviderError, ValueError)):
            await _analyze(pipeline, fixture.job)
    finally:
        await pipeline.aclose()
    assert not materializer.calls
    audit = _audit(tmp_path, fixture.job)
    assert audit["phase"] == "failed" and not audit["publication_ready"]


def _normal_observations(fixture):
    """Synthetic supported operation absence refutes every complete rule branch."""
    output = fixture.result.output
    windows = list(output.window_logits)
    localized = [[(0.0, 4.0, -4.0) for _ in output.identity.fact_ids] for _ in output.frames]
    for fact_id in ("CASH_RECEIPT_OPERATION", "CASH_ISSUE_OPERATION"):
        axis = output.identity.fact_ids.index(fact_id)
        windows[axis] = (4.0, 0.0, -4.0)
        for frame in localized:
            frame[axis] = (4.0, 0.0, -4.0)
    fixture.result = fixture.result.model_copy(update={"output": output.model_copy(update={
        "window_logits": tuple(windows), "frame_logits": tuple(tuple(frame) for frame in localized),
    })})


def _absence_policy(fixture):
    path = Path("rules/catalog.json")
    physical = load_physical_fact_run_policy(identity=fixture.manifest.identity,
        expected_catalog_sha256=sha256(path.read_bytes()).hexdigest(), catalog_path=path)
    _, required = physical_policy_observation_definitions(physical)
    contexts = {key: list(value) for key, value in required.items()}
    for fact_id in ("CASH_RECEIPT_OPERATION", "CASH_ISSUE_OPERATION"):
        contexts[fact_id] = ["WORK_AREA_CONTINUOUSLY_VISIBLE", "PROCEDURE_CONTEXT_COMPLETE"]
    return {"schema_version": "forte.fact-absence-context-policy.v1",
        "taxonomy_sha256": fixture.manifest.identity.taxonomy_sha256,
        "head_sha256": fixture.manifest.identity.head_sha256, "context_fact_ids": contexts}


@pytest.mark.asyncio
async def test_native_negative_requires_supported_absence_policy_and_complete_source(tmp_path):
    fixture = _fixture(tmp_path, occurrence_indices=())
    _normal_observations(fixture)
    pipeline, client, materializer, _ = _pipeline(tmp_path, fixture,
        absence_context_policy=_absence_policy(fixture))
    try:
        result, _ = await _analyze(pipeline, fixture.job)
    finally:
        await pipeline.aclose()
    public = json.loads(result.json_bytes)
    assert public["verdict"] == "no_violation" and public["violations"] == []
    assert len(client.calls) == 1
    assert all(len(sources) <= 3 for _, sources, _, _ in materializer.calls)
    audit = _audit(tmp_path, fixture.job)
    assert audit["publication_ready"] and audit["coverage"]["complete"]


@pytest.mark.asyncio
async def test_native_unmapped_absence_does_not_promote_a_normal_result(tmp_path):
    fixture = _fixture(tmp_path, occurrence_indices=())
    _normal_observations(fixture)
    pipeline, _, _, _ = _pipeline(tmp_path, fixture)
    try:
        with pytest.raises((AnalysisIncomplete, TemporalProviderError)):
            await _analyze(pipeline, fixture.job)
    finally:
        await pipeline.aclose()
    audit = _audit(tmp_path, fixture.job)
    assert not audit["publication_ready"]


@pytest.mark.asyncio
async def test_native_scene_context_is_required_even_with_high_bank_logits(tmp_path):
    fixture = _fixture(tmp_path)
    pipeline, _, _, _ = _pipeline(tmp_path, fixture, trusted_scene_context=None)
    try:
        with pytest.raises((AnalysisIncomplete, TemporalProviderError)):
            await _analyze(pipeline, fixture.job)
    finally:
        await pipeline.aclose()
    assert not _audit(tmp_path, fixture.job)["publication_ready"]


@pytest.mark.asyncio
async def test_native_machine_evidence_failure_cannot_publish_even_with_high_logits(tmp_path):
    fixture = _fixture(tmp_path)
    pipeline, _, materializer, _ = _pipeline(tmp_path, fixture,
        verifier=SyntheticVerifier(threshold_selection_verified=False))
    try:
        with pytest.raises((AnalysisIncomplete, TemporalProviderError)):
            await _analyze(pipeline, fixture.job)
    finally:
        await pipeline.aclose()
    assert not materializer.calls
    assert not _audit(tmp_path, fixture.job)["publication_ready"]


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", ["source_pts", "jpeg_hash", "frame_id"])
async def test_native_materialized_evidence_must_match_exact_source_and_jpeg_hash(tmp_path, tamper):
    fixture = _fixture(tmp_path)
    pipeline, _, materializer, _ = _pipeline(tmp_path, fixture, materializer=_Materializer(tamper=tamper))
    try:
        with pytest.raises((AnalysisIncomplete, TemporalProviderError, ValueError)):
            await _analyze(pipeline, fixture.job)
    finally:
        await pipeline.aclose()
    assert materializer.calls
    audit = _audit(tmp_path, fixture.job)
    assert audit["phase"] == "failed" and not audit["publication_ready"]


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["client", "materializer"])
async def test_native_failure_keeps_private_audit_without_echoing_private_error(tmp_path, where):
    fixture = _fixture(tmp_path)
    error = RuntimeError("SYNTHETIC_PRIVATE_SENTINEL")
    client = _Client(fixture.result, error=error if where == "client" else None)
    materializer = _Materializer(error=error if where == "materializer" else None)
    pipeline, _, _, _ = _pipeline(tmp_path, fixture, client=client, materializer=materializer)
    try:
        with pytest.raises(Exception) as rejected:
            await _analyze(pipeline, fixture.job)
    finally:
        await pipeline.aclose()
    assert "SYNTHETIC_PRIVATE_SENTINEL" not in str(rejected.value)
    audit = _audit(tmp_path, fixture.job)
    assert audit["phase"] == "failed" and not audit["publication_ready"]
    assert "SYNTHETIC_PRIVATE_SENTINEL" not in json.dumps(audit)


@pytest.mark.asyncio
async def test_native_task_cancellation_propagates_and_drains_active_client(tmp_path):
    fixture = _fixture(tmp_path)
    started, drained = asyncio.Event(), asyncio.Event()
    client = _Client(fixture.result)

    async def blocked(asset, **kwargs):
        started.set()
        try:
            await asyncio.Future()
        finally:
            await asyncio.sleep(0)
            drained.set()

    client.predict_native = blocked
    pipeline, _, materializer, _ = _pipeline(tmp_path, fixture, client=client)
    task = asyncio.create_task(_analyze(pipeline, fixture.job))
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
    finally:
        await pipeline.aclose()
    assert drained.is_set() and client.closed and not materializer.calls
    audit = _audit(tmp_path, fixture.job)
    assert audit["phase"] == "failed" and not audit["publication_ready"]


@pytest.mark.asyncio
async def test_native_job_cancel_signal_reaches_client_and_prevents_materialization(tmp_path):
    fixture = _fixture(tmp_path)
    started, drained = asyncio.Event(), asyncio.Event()
    signal = threading.Event()
    client = _Client(fixture.result)

    async def waits_for_signal(asset, *, cancel_event, **kwargs):
        started.set()
        try:
            while not cancel_event.is_set():
                await asyncio.sleep(0.001)
            raise asyncio.CancelledError
        finally:
            drained.set()

    client.predict_native = waits_for_signal
    pipeline, _, materializer, _ = _pipeline(tmp_path, fixture, client=client)
    task = asyncio.create_task(_analyze(pipeline, fixture.job, cancel_event=signal))
    await asyncio.wait_for(started.wait(), timeout=2)
    signal.set()
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
    finally:
        await pipeline.aclose()
    assert drained.is_set() and not materializer.calls
    assert not _audit(tmp_path, fixture.job)["publication_ready"]


@pytest.mark.asyncio
async def test_native_cancellation_waits_for_blocking_evidence_worker_to_stop(tmp_path):
    fixture = _fixture(tmp_path)
    started, drained = threading.Event(), threading.Event()

    def blocked_materializer(asset, selected_sources, output_dir, *, cancel, **kwargs):
        started.set()
        try:
            while not cancel.is_set():
                time.sleep(0.001)
            time.sleep(0.01)  # A decoder cleanup that must finish before this job ends.
            raise RuntimeError("SYNTHETIC_PRIVATE_DECODER_SENTINEL")
        finally:
            drained.set()

    pipeline, client, _, _ = _pipeline(tmp_path, fixture, materializer=blocked_materializer)
    task = asyncio.create_task(_analyze(pipeline, fixture.job))
    assert await asyncio.to_thread(started.wait, 2)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
    finally:
        await pipeline.aclose()
    assert drained.is_set() and client.closed
    audit = _audit(tmp_path, fixture.job)
    assert not audit["publication_ready"]
    assert "SYNTHETIC_PRIVATE_DECODER_SENTINEL" not in json.dumps(audit)


@pytest.mark.asyncio
async def test_native_deadline_stops_and_drains_client_before_returning(tmp_path):
    fixture = _fixture(tmp_path)
    started, drained = asyncio.Event(), asyncio.Event()
    client = _Client(fixture.result)

    async def blocked(asset, **kwargs):
        started.set()
        try:
            await asyncio.Future()
        finally:
            await asyncio.sleep(0)
            drained.set()

    async def stage(value):
        pass

    client.predict_native = blocked
    pipeline, _, materializer, _ = _pipeline(tmp_path, fixture, client=client)
    task = asyncio.create_task(pipeline.analyze(fixture.job, stage=stage,
        deadline=time.monotonic() + 0.3, cancel_event=threading.Event()))
    await asyncio.wait_for(started.wait(), timeout=2)
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(task, timeout=2)
    finally:
        await pipeline.aclose()
    assert drained.is_set() and not materializer.calls
    assert not _audit(tmp_path, fixture.job)["publication_ready"]


@pytest.mark.parametrize("tamper", ["taxonomy", "head", "weaken", "unknown_axis", "self_context"])
def test_native_explicit_absence_policy_cannot_weaken_or_escape_its_pins(tmp_path, tamper):
    fixture = _fixture(tmp_path)
    policy = _absence_policy(fixture)
    if tamper == "taxonomy":
        policy["taxonomy_sha256"] = digest("other-taxonomy")
    elif tamper == "head":
        policy["head_sha256"] = digest("other-head")
    elif tamper == "weaken":
        policy["context_fact_ids"]["CASH_HANDLER_PRESENT"] = ["WORK_AREA_CONTINUOUSLY_VISIBLE"]
    elif tamper == "unknown_axis":
        policy["context_fact_ids"]["CASH_RECEIPT_OPERATION"] = ["UNREVIEWED_CONTEXT"]
    else:
        policy["context_fact_ids"]["CASH_RECEIPT_OPERATION"] = ["CASH_RECEIPT_OPERATION"]
    with pytest.raises(AnalysisIncomplete, match="native_absence_policy_invalid"):
        _pipeline(tmp_path, fixture, absence_context_policy=policy)


@pytest.mark.asyncio
@pytest.mark.parametrize("pins", [None, "execution_only", "source_only"])
async def test_native_http_constructor_requires_both_execution_and_source_code_pins(tmp_path, pins):
    fixture = _fixture(tmp_path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(500))) as transport:
        client = HttpNativeHeadClient("http://127.0.0.1:9100", identity=fixture.manifest.identity,
            client=transport,
            expected_execution_manifest_sha256=digest("execution") if pins == "execution_only" else None,
            expected_source_code_bundle_sha256=digest("source-code") if pins == "source_only" else None)
        with pytest.raises(AnalysisIncomplete, match="native_execution_and_code_pins_required"):
            _pipeline(tmp_path, fixture, client=client)


@pytest.mark.asyncio
@pytest.mark.parametrize("response_kind", ["not_ready", "wrong_execution", "malformed"])
async def test_native_http_readiness_failure_is_static_and_keeps_private_audit(tmp_path, response_kind):
    fixture = _fixture(tmp_path)
    requests = []

    def handler(request):
        requests.append(request.url.path)
        if response_kind == "malformed":
            return httpx.Response(200, content=b"SYNTHETIC_PRIVATE_READY_SENTINEL")
        return httpx.Response(200, json={"ready": response_kind != "not_ready",
            "head_sha256": fixture.manifest.identity.head_sha256,
            "feature_spec_sha256": fixture.manifest.identity.feature_spec_sha256,
            "execution_manifest_sha256": digest("wrong-execution"),
            "source_code_bundle_sha256": digest("source-code"),
            "private": "SYNTHETIC_PRIVATE_READY_SENTINEL"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        client = HttpNativeHeadClient("http://127.0.0.1:9100", identity=fixture.manifest.identity,
            client=transport, expected_execution_manifest_sha256=digest("execution"),
            expected_source_code_bundle_sha256=digest("source-code"))
        pipeline, _, materializer, _ = _pipeline(tmp_path, fixture, client=client)
        try:
            with pytest.raises((AnalysisIncomplete, TemporalProviderError)) as rejected:
                await _analyze(pipeline, fixture.job)
        finally:
            await pipeline.aclose()
    assert requests == ["/ready"] and not materializer.calls
    assert "SYNTHETIC_PRIVATE_READY_SENTINEL" not in str(rejected.value)
    audit = _audit(tmp_path, fixture.job)
    assert not audit["publication_ready"]
    assert "SYNTHETIC_PRIVATE_READY_SENTINEL" not in json.dumps(audit)


@pytest.mark.asyncio
async def test_native_loaded_absence_map_mutation_fails_before_any_client_call(tmp_path):
    fixture = _fixture(tmp_path)
    pipeline, client, materializer, probes = _pipeline(tmp_path, fixture)
    ready_calls = []

    async def ready(*, deadline):
        ready_calls.append(deadline)
        return {"ready": True}

    client.check_ready = ready
    pipeline.absence_context_fact_ids["CASH_RECEIPT_OPERATION"] = ("WORK_AREA_CONTINUOUSLY_VISIBLE",)
    try:
        with pytest.raises(AnalysisIncomplete, match="native_loaded_binding_changed"):
            await _analyze(pipeline, fixture.job)
    finally:
        await pipeline.aclose()
    assert not client.calls and not ready_calls and not materializer.calls and not probes
    audit = _audit(tmp_path, fixture.job)
    assert audit["phase"] == "failed" and not audit["publication_ready"]
    assert audit["error_code"] == "native_loaded_binding_changed"


@pytest.mark.asyncio
async def test_native_aclose_cancels_and_drains_active_blocking_evidence_worker(tmp_path):
    fixture = _fixture(tmp_path)
    started, drained = threading.Event(), threading.Event()

    def blocked_materializer(asset, selected_sources, output_dir, *, cancel, **kwargs):
        started.set()
        try:
            while not cancel.is_set():
                time.sleep(0.001)
            time.sleep(0.01)
            raise RuntimeError("SYNTHETIC_PRIVATE_SHUTDOWN_SENTINEL")
        finally:
            drained.set()

    pipeline, client, _, _ = _pipeline(tmp_path, fixture, materializer=blocked_materializer)
    task = asyncio.create_task(_analyze(pipeline, fixture.job))
    assert await asyncio.to_thread(started.wait, 2)
    await asyncio.wait_for(pipeline.aclose(), timeout=2)
    assert drained.is_set() and client.closed
    with pytest.raises((asyncio.CancelledError, AnalysisIncomplete)):
        await asyncio.wait_for(task, timeout=2)
    audit = _audit(tmp_path, fixture.job)
    assert audit["phase"] == "failed" and not audit["publication_ready"]
    assert "SYNTHETIC_PRIVATE_SHUTDOWN_SENTINEL" not in json.dumps(audit)
