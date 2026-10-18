"""Synthetic native-frame, confirmation and failure boundaries; no model quality claim."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import av
from jsonschema import ValidationError
from PIL import Image
import pytest

from forte.fast.analyzer import Candidate, analyze, extract_states
from forte.fast.service import _await_owned_thread, decode_for_analysis, load_config
from forte.fast.video import DecodedVideo, Frame, decode
from forte.fast.vlm import VLMClient, parse_json


def generated_video(path):
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 32, 48, "yuv420p"
        for i in range(20):
            image = Image.new("RGB", (32, 48), (i * 8, 30, 80))
            frame = av.VideoFrame.from_image(image)
            image.close()
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return path


def test_native_callback_and_retained_tail_share_one_decode(tmp_path):
    native = []
    video = decode(str(generated_video(tmp_path / "generated.mov")), 2, on_frame=lambda f: native.append(f.t))
    assert len(native) == 20
    assert native == sorted(set(native))
    assert [f.t for f in video.frames] == pytest.approx([0, .5, 1, 1.5, 1.9])
    assert video.duration == pytest.approx(2)
    assert all(f.image.size == (32, 48) for f in video.frames)


def test_decoder_limits_and_cancellation_fail_instead_of_truncating(tmp_path):
    path = str(generated_video(tmp_path / "generated.mov"))
    with pytest.raises(ValueError, match="frame_limit"):
        decode(path, 5, max_frames=2)
    with pytest.raises(ValueError, match="frame_limit"):
        decode(path, 5, max_memory_bytes=1)
    with pytest.raises(TimeoutError):
        decode(path, deadline=time.monotonic() - 1)
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        decode(path, cancel_event=cancelled)


def test_decode_streams_specialist_and_retains_dense_verification(tmp_path):
    class Runtime:
        def __init__(self):
            self.times = []
            self.aborted = False
        def start_video(self, fps):
            self.fps = fps
        def process_frame(self, frame):
            self.times.append(frame.t)
        def finish_video(self):
            return {"observed": len(self.times)}
        def abort_video(self):
            self.aborted = True
    runtime = Runtime()
    video, observations = decode_for_analysis(str(generated_video(tmp_path / "generated.mov")), {}, runtime)
    assert len(runtime.times) == 20 and observations == {"observed": 20}
    assert runtime.aborted and runtime.fps == 10 and len(video.frames) >= 10


@pytest.mark.asyncio
async def test_failed_scene_queries_are_not_a_clean_verdict():
    client = SimpleNamespace(chat_json=AsyncMock(side_effect=RuntimeError("unavailable")))
    video = DecodedVideo(1, 2, 16, 16, [Frame(0, Image.new("RGB", (16, 16)))])
    with pytest.raises(RuntimeError, match="observations_unavailable"):
        await extract_states(client, video, {"state_fps": 1, "state_max_side": 64}, {})


@pytest.mark.asyncio
async def test_untrained_protected_type_cannot_be_enabled_by_config():
    video = DecodedVideo(1, 2, 16, 16, [Frame(0, Image.new("RGB", (16, 16)))])
    with (patch("forte.fast.analyzer.extract_states", AsyncMock(return_value=[{"t": 0}])),
          patch("forte.fast.analyzer.rule_candidates", return_value=([Candidate("NO_CLIENT_ID", 0, "rule")], []))):
        result = await analyze(None, video, {"use_crops": False, "disabled_types": []})
    assert not result.events and result.audit["suppressed"][0]["reason"] == "type_not_promoted"


@pytest.mark.asyncio
@pytest.mark.parametrize("visible,expected", [(True, 1), (False, 0), ("yes", 0)])
async def test_specialist_requires_vlm_confirmation_and_keeps_disagreement(visible, expected):
    video = DecodedVideo(1, 2, 16, 16, [Frame(0, Image.new("RGB", (16, 16)))])
    specialist = SimpleNamespace(approved_types=("KEY_LEFT_IN_SAFE",), provenance={"synthetic_only": True, "evaluation_only": True})
    proposal = SimpleNamespace(type_id="KEY_LEFT_IN_SAFE", t=0, note="synthetic key", score=.99, evidence={})
    verification = {"visible": visible, "confidence": .95, "t_sec": 0, "observation_ru": "Синтетический пример."}
    with (patch("forte.fast.analyzer.extract_states", AsyncMock(return_value=[{"t": 0}])),
          patch("forte.fast.analyzer.rule_candidates", return_value=([], [])),
          patch("forte.fast.temporal_events.specialist_candidates", return_value=([proposal], {})),
          patch("forte.fast.analyzer.verify", AsyncMock(return_value=verification)) as confirm):
        result = await analyze(None, video, {"use_crops": False, "use_verify": False}, specialist_analysis=specialist)
    assert len(result.events) == expected
    confirm.assert_awaited_once()
    assert expected or result.audit["suppressed"][0]["reason"] == "vlm_disagrees_or_unclear"


@pytest.mark.asyncio
async def test_verification_transport_failure_fails_analysis():
    video = DecodedVideo(1, 2, 16, 16, [Frame(0, Image.new("RGB", (16, 16)))])
    with (patch("forte.fast.analyzer.extract_states", AsyncMock(return_value=[{"t": 0}])),
          patch("forte.fast.analyzer.rule_candidates", return_value=([Candidate("PERSONAL_ITEMS", 0, "rule")], [])),
          patch("forte.fast.analyzer.verify", AsyncMock(return_value={"error": "timeout"}))):
        with pytest.raises(RuntimeError, match="verification_unavailable"):
            await analyze(None, video, {"use_crops": False, "require_corroboration": True})


def test_bad_override_config_does_not_silently_select_defaults(tmp_path):
    (tmp_path / "fast_config.json").write_text("invalid", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(tmp_path)
    (tmp_path / "fast_config.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(tmp_path)


@pytest.mark.asyncio
async def test_vlm_json_is_schema_checked_and_readiness_obeys_deadline():
    client = VLMClient("http://127.0.0.1:1/v1", "synthetic")
    client.chat = AsyncMock(return_value=json.dumps({"visible": "yes"}))
    try:
        with pytest.raises(ValidationError):
            await client.chat_json([], json_schema={"type": "object", "properties": {"visible": {"type": "boolean"}}})
        with pytest.raises(TimeoutError):
            await client.check_ready(deadline=time.monotonic() - 1)
        client.ready = AsyncMock(return_value=False)
        with pytest.raises(RuntimeError, match="not_ready"):
            await client.check_ready(deadline=time.monotonic() + 1)
    finally:
        await client.aclose()


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_json_is_a_failure_not_a_negative(value):
    with pytest.raises(ValueError):
        parse_json('{"visible":true,"confidence":' + value + '}')


def test_promotion_policy_binds_settings_but_not_artifact_locations():
    from forte.fast.policy import analyzer_policy_sha256

    base = {"specialists": {"enabled": True, "checkpoint": "a", "gate_path": "class"}, "verify_threshold": .8}
    moved = {**base, "specialists": {**base["specialists"], "checkpoint": "b", "gate_path": "video"}}
    assert analyzer_policy_sha256(base) == analyzer_policy_sha256(moved)
    assert analyzer_policy_sha256(base) != analyzer_policy_sha256({**base, "verify_threshold": .1})


@pytest.mark.asyncio
async def test_thread_ownership_survives_repeated_cancellation():
    release = threading.Event()
    started = threading.Event()
    cancelled = threading.Event()
    owned = asyncio.Lock()

    def load():
        started.set()
        release.wait(timeout=2)

    async def owner():
        async with owned:
            task = asyncio.create_task(asyncio.to_thread(load))
            await _await_owned_thread(task, cancelled)

    task = asyncio.create_task(owner())
    while not started.is_set():
        await asyncio.sleep(.001)
    task.cancel()
    await asyncio.sleep(.005)
    task.cancel()
    await asyncio.sleep(.005)
    assert owned.locked() and cancelled.is_set() and not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not owned.locked()


@pytest.mark.asyncio
async def test_changed_promoted_policy_is_rejected_before_model_queries():
    from forte.fast.policy import analyzer_policy_sha256

    video = DecodedVideo(1, 2, 16, 16, [Frame(0, Image.new("RGB", (16, 16)))])
    specialist = SimpleNamespace(approved_types=("KEY_LEFT_IN_SAFE",),
                                 provenance={"analyzer_policy_sha256": analyzer_policy_sha256({"verify_threshold": .8})})
    with pytest.raises(ValueError, match="promoted_analyzer_policy_mismatch"):
        await analyze(None, video, {"verify_threshold": .1}, specialist_analysis=specialist)


@pytest.mark.asyncio
async def test_key_verification_includes_original_before_and_after_crops():
    from forte.fast.analyzer import DEFAULT_CONFIG, verify

    video = DecodedVideo(2, 2, 32, 48, [Frame(t, Image.new("RGB", (32, 48))) for t in (0, 1, 2)])
    client = SimpleNamespace(chat_json=AsyncMock(return_value={"visible": False, "confidence": 0, "t_sec": 1}))
    candidate = Candidate("KEY_LEFT_IN_SAFE", 1, "specialist", evidence={"lock_snapshots": [
        {"t": 0, "box": [.1, .1, .2, .2]}, {"t": 2, "box": [.1, .1, .2, .2]}]})
    await verify(client, video, candidate, DEFAULT_CONFIG)
    parts = client.chat_json.call_args.args[0]
    assert sum(p.get("type") == "image_url" for p in parts) == 5
    assert sum("Original lock close-up" in p.get("text", "") for p in parts) == 2
