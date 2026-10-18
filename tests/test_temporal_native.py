"""Synthetic native binding/HTTP/projection tests; never execute a head/encoder."""
from hashlib import sha256
import json
from pathlib import Path
import time

import httpx
from PIL import Image
import pytest
from pydantic import ValidationError

from forte.decision.head_contract import TemporalHeadOutput
from forte.decision.temporal_http import HttpNativeHeadClient, private_native_endpoint
from forte.decision.temporal_native import NativeSourceContext, NativeHeadResult
from forte.decision.temporal_observations import map_native_observations
from forte.decision.temporal_provider import TemporalProviderError
from forte.shared.schemas import FactDefinition, FrameRef, VideoAsset
from forte.video.batches import source_frame_id
from test_machine_calibration_contract import manifest, SyntheticVerifier, digest


def fixture():
    calibration = manifest()
    identity = calibration.identity
    asset = VideoAsset(job_id="synthetic-native-job", owned_path="/synthetic/uploads/fixture.mov",
        content_hash=digest("synthetic-native-source"), duration_sec=0.2, width=32, height=16,
        fps=25.0, time_base="1/25", source_start_sec=4.0, frame_count=5)
    frames = [{"frame_id": source_frame_id(asset.content_hash, 100 + index, 1, 25),
        "source_pts": 100 + index, "next_source_pts": 101 + index if index < 4 else None,
        "pts_sec": index / 25, "next_pts_sec": (index + 1) / 25 if index < 4 else None,
        "source_end_bound_sec": 0.2 if index == 4 else None, "width": 32, "height": 16,
        "applied_rotation_deg": 0, "key_frame": index == 0, "view_mask": [True],
        "view_roles": ["global"], "normalized_boxes": [[0.0, 0.0, 1.0, 1.0]]} for index in range(5)]
    context = NativeSourceContext.model_validate_json(json.dumps({"job_id": asset.job_id,
        "clip_id": "synthetic-native-clip", "content_sha256": asset.content_hash,
        "duration_sec": asset.duration_sec, "time_base_num": 1, "time_base_den": 25,
        "origin_pts": 100, "stream_index": 0, "feature_spec_sha256": identity.feature_spec_sha256,
        "encoder_manifest_sha256": identity.encoder_artifacts_sha256, "source_complete": True,
        "tail_included": True, "frames": frames}))
    output = TemporalHeadOutput.model_validate_json(json.dumps({"identity": identity.model_dump(mode="json"),
        "job_id": asset.job_id, "clip_id": context.clip_id,
        "frames": [{key: frame[key] for key in ("frame_id", "pts_sec", "source_pts", "next_source_pts")} for frame in frames],
        "window_logits": [[0.0, 4.0, -4.0]] * 19,
        "frame_logits": [[[0.0, 4.0, -4.0]] * 19] * 5, "support_logits": [[4.0] * 19] * 5,
        "frame_valid": [True] * 5, "global_view_available": [True] * 5,
        "feature_coverage_complete": True, "native_source_continuity_verified": True,
        "trained_components": {"window_fact_states": True, "frame_fact_states": True, "frame_support": True}}))
    result = NativeHeadResult(context=context, output=output, video_logit=0.0,
        metrics={"head_elapsed_sec": 0.1, "native_frames": 5.0})
    return asset, result, calibration


def test_complete_native_nonzero_original_pts_bind_without_any_training_split():
    asset, result, calibration = fixture()
    result.context.validate_asset(asset, calibration.identity)
    index = result.context.source_index()
    assert len(index) == 5 and index[0].source_pts == 100 and index[0].pts_sec == 0.0
    assert index[-1].source_end_bound_sec == 0.2 and index[0].key_frame
    assert "split" not in result.context.model_dump()


@pytest.mark.parametrize("mutation", ["id", "pts", "chain", "tail", "global", "start"])
def test_native_metadata_rejects_fabricated_or_incomplete_source_context(mutation):
    _, result, _ = fixture()
    raw = result.context.model_dump(mode="json")
    if mutation == "id": raw["frames"][0]["frame_id"] = digest("invented")
    if mutation == "pts": raw["frames"][1]["pts_sec"] = 0.041
    if mutation == "chain": raw["frames"][0]["next_source_pts"] = 102
    if mutation == "tail": raw["tail_included"] = False
    if mutation == "global": raw["frames"][0]["view_roles"] = ["roi"]
    if mutation == "start": raw["origin_pts"] = 99
    with pytest.raises(ValidationError):
        NativeSourceContext.model_validate_json(json.dumps(raw))


def test_native_asset_and_model_identity_binding_is_exact():
    asset, result, calibration = fixture()
    with pytest.raises(TemporalProviderError):
        result.context.validate_asset(asset.model_copy(update={"content_hash": digest("other")}), calibration.identity)
    with pytest.raises(TemporalProviderError):
        result.context.validate_asset(asset, calibration.identity.model_copy(update={"feature_spec_sha256": digest("other")}))
    raw = result.model_dump(mode="json")
    raw["output"]["frames"][0]["source_pts"] = 99
    with pytest.raises(ValidationError):
        NativeHeadResult.model_validate_json(json.dumps(raw))


def test_native_plan_projects_only_actual_original_jpegs_into_existing_contract(tmp_path):
    asset, result, calibration = fixture()
    facts = [FactDefinition(fact_id=fact_id, description="Synthetic fixture only") for fact_id in calibration.identity.fact_ids]
    plan = map_native_observations(result, asset, calibration, verifier=SyntheticVerifier(),
        fact_definitions=facts, event_definitions={}, absence_context_fact_ids={})
    assert 0 < len(plan.requested_frame_ids) <= 3
    source = {frame.frame_id: frame for frame in result.context.frames}
    actual = []
    for index, frame_id in enumerate(plan.requested_frame_ids):
        path = tmp_path / f"synthetic-{index}.jpg"
        Image.new("RGB", (32, 16), (255, 0, 0)).save(path)
        frame = source[frame_id]
        actual.append(FrameRef(frame_id=frame_id, pts_sec=frame.pts_sec, source_pts=frame.source_pts,
            path=str(path), width=32, height=16, job_id=asset.job_id,
            jpeg_sha256=sha256(path.read_bytes()).hexdigest()))
    batch = plan.bind_evidence_frames(actual)
    assert len(batch.facts) == 19 and len(batch.request.frame_refs) == len(plan.requested_frame_ids)
    assert all(fact.state == "present" for fact in batch.facts)
    assert batch.metrics["serving_attestation"]["provisional"]
    actual[0] = actual[0].model_copy(update={"source_pts": 999})
    with pytest.raises(TemporalProviderError):
        plan.bind_evidence_frames(actual)


@pytest.mark.parametrize("endpoint", ["https://example.com:443", "http://8.8.8.8:9100",
    "http://127.0.0.1:9100/private-path", "http://user:secret@127.0.0.1:9100", "http://127.0.0.1"])
def test_native_client_refuses_external_or_ambiguous_endpoints(endpoint):
    with pytest.raises(TemporalProviderError):
        private_native_endpoint(endpoint)


@pytest.mark.asyncio
async def test_private_native_http_protocol_is_bound_and_omits_arbitrary_metadata():
    asset, result, calibration = fixture()
    observed = []
    def handler(request):
        observed.append(request)
        if request.url.path == "/ready":
            return httpx.Response(200, json={"ready": True, "head_sha256": calibration.identity.head_sha256,
                "feature_spec_sha256": calibration.identity.feature_spec_sha256})
        body = json.loads(request.content)
        assert body["asset"]["metadata"] == {} and body["deadline_budget_sec"] > 0
        return httpx.Response(200, json=result.model_dump(mode="json"))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        client = HttpNativeHeadClient("http://127.0.0.1:9100", identity=calibration.identity, client=transport)
        ready = await client.check_ready(deadline=time.monotonic() + 2)
        assert not ready["candidate_calibration_verified"]
        returned = await client.predict_native(asset.model_copy(update={"metadata": {"ignore": "SYNTHETIC_PRIVATE_SENTINEL"}}),
            clip_id=result.context.clip_id, deadline=time.monotonic() + 2)
        assert returned.context.content_sha256 == asset.content_hash and len(observed) == 2


@pytest.mark.asyncio
async def test_http_result_rejects_wrong_original_source_and_never_echoes_private_body():
    asset, result, calibration = fixture()
    raw = result.model_dump(mode="json")
    raw["context"]["content_sha256"] = digest("other-source")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=raw))) as transport:
        client = HttpNativeHeadClient("http://127.0.0.1:9100", identity=calibration.identity, client=transport)
        with pytest.raises(TemporalProviderError) as error:
            await client.predict_native(asset, clip_id=result.context.clip_id, deadline=time.monotonic() + 2)
        assert str(error.value) == "optional_temporal_provider_incomplete"


@pytest.mark.asyncio
async def test_cpu_client_and_actual_asgi_service_share_native_serialization_contract():
    from serving.fast_service import NativeServiceApp
    asset, result, calibration = fixture()

    class SyntheticBackend:
        def check_ready(self):
            return {"ready": True, "head_sha256": calibration.identity.head_sha256,
                "feature_spec_sha256": calibration.identity.feature_spec_sha256,
                "candidate_calibration_verified": False}

        def predict(self, actual_asset, *, clip_id, deadline, cancel_event):
            assert actual_asset.content_hash == asset.content_hash and clip_id == result.context.clip_id
            assert not cancel_event.is_set()
            return result

    app = NativeServiceApp(SyntheticBackend())
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as transport:
            client = HttpNativeHeadClient("http://127.0.0.1:9100", identity=calibration.identity, client=transport)
            assert (await client.check_ready(deadline=time.monotonic() + 2))["ready"]
            output = await client.predict_native(asset, clip_id=result.context.clip_id, deadline=time.monotonic() + 2)
            assert output.output.identity == calibration.identity
            assert len(output.context.source_index()) == len(result.context.frames)
    finally:
        app._executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_reproducibility_pin_mismatch_blocks_prediction_before_post():
    asset, result, calibration = fixture()
    methods = []
    def handler(request):
        methods.append(request.method)
        return httpx.Response(200, json={"ready": True, "head_sha256": calibration.identity.head_sha256,
            "feature_spec_sha256": calibration.identity.feature_spec_sha256,
            "source_code_bundle_sha256": digest("different-code")})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        client = HttpNativeHeadClient("http://127.0.0.1:9100", identity=calibration.identity, client=transport,
            expected_source_code_bundle_sha256=digest("frozen-code"))
        with pytest.raises(TemporalProviderError) as error:
            await client.predict_native(asset, clip_id=result.context.clip_id, deadline=time.monotonic() + 2)
        assert error.value.subcode == "native_sidecar_execution_pin_mismatch"
    assert methods == ["GET"]
