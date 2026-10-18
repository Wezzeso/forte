"""Synthetic transport/media fixtures only; never reads the supplied dataset."""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from PIL import Image

from forte.model import (
    InvalidModelOutput, LocalModelConfig, LocalModelProvider, ManifestError,
    ModelCancelled, ModelDeadlineExceeded, ModelUnavailable, UnsafeMedia,
    load_manifest, verify_local_snapshot,
)
from forte.model.adapter import strict_json_object, validate_private_endpoint
from forte.model.staging import _download, stage_local_snapshot
from forte.shared.schemas import ClipRequest, FrameRef


SCHEMA = {"type": "object", "properties": {"state": {"enum": ["unclear"]}},
          "required": ["state"], "additionalProperties": False}


def envelope(content='{"state":"unclear"}', finish_reason="stop", **updates):
    value = {"model": "Qwen3-VL-8B-Instruct", "choices": [{"finish_reason": finish_reason,
             "message": {"role": "assistant", "content": content}}],
             "usage": {"prompt_tokens": 32, "completion_tokens": 8, "total_tokens": 40}}
    value.update(updates)
    return value


class ModelAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="forte-model-test-")
        self.root = Path(self.temp.name)
        self.job = self.root / "job_synthetic"
        self.job.mkdir()
        self.image = self.job / "synthetic_source.jpg"
        Image.new("RGB", (16, 12), color=(8, 20, 30)).save(self.image, "JPEG")
        self.frame = FrameRef(frame_id="frame_opaque", pts_sec=0.0, path=self.image,
                              width=16, height=12, job_id="job_synthetic")
        self.clip = ClipRequest(job_id="job_synthetic", clip_id="clip_opaque", frame_refs=[self.frame])
        self.config = LocalModelConfig(media_root=self.root, base_url="http://127.0.0.1:8000/v1",
                                       require_readiness=False)

    def tearDown(self):
        self.temp.cleanup()

    async def run_request(self, handler, **kwargs):
        async with LocalModelProvider(self.config, transport=httpx.MockTransport(handler)) as provider:
            return await provider.infer(self.clip, prompt="Return only requested synthetic observations.",
                response_schema=SCHEMA, deadline=time.monotonic() + 2, **kwargs)

    async def test_structured_body_private_media_and_metrics(self):
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(request.url.path, "/v1/chat/completions")
            self.assertEqual(body["structured_outputs"], {"json": SCHEMA})
            self.assertEqual(body["messages"][0]["role"], "system")
            self.assertNotIn("synthetic_source", request.content.decode())
            self.assertNotIn(str(self.root), request.content.decode())
            self.assertTrue(body["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
            self.assertEqual(body["temperature"], 0.0)
            return httpx.Response(200, json=envelope())
        result = await self.run_request(handler)
        self.assertEqual(result.data, {"state": "unclear"})
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.metrics.input_dimensions, ((16, 12),))
        self.assertEqual(result.metrics.generated_tokens, 8)
        self.assertIsNone(result.metrics.visual_tokens)
        self.assertIsNone(result.metrics.processor_output_dimensions)
        self.assertNotIn("unclear", repr(result))

    async def test_truncation_refused_even_complete_json(self):
        with self.assertRaisesRegex(InvalidModelOutput, "incomplete"):
            await self.run_request(lambda _: httpx.Response(200, json=envelope(finish_reason="length")))

    async def test_server_metrics_preserve_only_actual_numeric_fields(self):
        reported = {"time_to_first_token_ms": 12.5, "generation_time_ms": 80,
                    "queue_time_ms": -1, "mean_itl_ms": None, "tokens_per_second": True,
                    "unknown_private_field": "never copied"}
        result = await self.run_request(lambda _: httpx.Response(200, json=envelope(metrics=reported)))
        self.assertEqual(result.metrics.server_metrics, {"time_to_first_token_ms": 12.5,
            "generation_time_ms": 80.0, "queue_time_ms": None, "mean_itl_ms": None,
            "tokens_per_second": None})

    async def test_synthetic_visual_gate_rejects_valid_all_absent_output(self):
        from forte.model.smoke import synthetic_smoke
        class TestOnlyProvider:
            def __init__(self, config):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def infer(self, clip, **kwargs):
                return SimpleNamespace(data={"facts": [{"fact_id": "synthetic_marker_visible",
                    "state": "absent", "supporting_frame_ids": [clip.frame_refs[0].frame_id]}],
                    "candidates": []}, finish_reason="stop", metrics=None, model="synthetic_test_only")
        with patch("forte.model.smoke.LocalModelProvider", TestOnlyProvider):
            with self.assertRaisesRegex(InvalidModelOutput, "visual_sanity"):
                await synthetic_smoke(base_url="http://127.0.0.1:8000/v1", media_root=self.root,
                                      require_readiness=False)

    async def test_tool_calls_refused(self):
        value = envelope()
        value["choices"][0]["message"]["tool_calls"] = [{"id": "synthetic"}]
        with self.assertRaises(InvalidModelOutput):
            await self.run_request(lambda _: httpx.Response(200, json=value))

    async def test_schema_invalid_output_refused(self):
        with self.assertRaisesRegex(InvalidModelOutput, "schema"):
            await self.run_request(lambda _: httpx.Response(200, json=envelope('{"state":"not_allowed"}')))

    async def test_identity_mismatch_refused(self):
        with self.assertRaisesRegex(InvalidModelOutput, "identity"):
            await self.run_request(lambda _: httpx.Response(200, json=envelope(model="wrong-model")))

    async def test_retry_only_pre_admission_failure_once(self):
        for failure in (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout):
            with self.subTest(failure=failure.__name__):
                calls = []
                def handler(request):
                    calls.append(1)
                    if len(calls) == 1:
                        raise failure("synthetic-private-error", request=request)
                    return httpx.Response(200, json=envelope())
                result = await self.run_request(handler)
                self.assertEqual(result.metrics.attempts, 2)

    async def test_no_retry_ambiguous_transport_failure(self):
        for failure in (httpx.ReadTimeout, httpx.ReadError, httpx.WriteTimeout,
                        httpx.WriteError, httpx.RemoteProtocolError):
            with self.subTest(failure=failure.__name__):
                calls = []
                def handler(request):
                    calls.append(1)
                    raise failure("synthetic-private-error", request=request)
                with self.assertRaises(ModelUnavailable) as error:
                    await self.run_request(handler)
                self.assertEqual(len(calls), 1)
                self.assertNotIn("synthetic-private", str(error.exception))

    async def test_no_replay_after_http_response(self):
        for status in (429, 502, 503, 504):
            with self.subTest(status=status):
                calls = []
                def handler(_):
                    calls.append(1)
                    return httpx.Response(status, text="synthetic-private-error")
                with self.assertRaises(ModelUnavailable):
                    await self.run_request(handler)
                self.assertEqual(len(calls), 1)

    async def test_no_retry_bad_request_no_error_body_leak(self):
        calls = []
        def handler(_):
            calls.append(1)
            return httpx.Response(400, text="synthetic-sensitive-data")
        with self.assertRaises(ModelUnavailable) as error:
            await self.run_request(handler)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("synthetic-sensitive", str(error.exception))

    async def test_retry_count_bounded(self):
        calls = []
        def handler(request):
            calls.append(1)
            raise httpx.ConnectError("synthetic connect failure", request=request)
        with self.assertRaises(ModelUnavailable):
            await self.run_request(handler)
        self.assertEqual(len(calls), 2)

    async def test_proxy_and_redirects_disabled(self):
        calls = []
        def handler(_):
            calls.append(1)
            return httpx.Response(307, headers={"location": "https://public.example/v1/chat/completions"})
        with self.assertRaises(ModelUnavailable):
            await self.run_request(handler)
        self.assertEqual(len(calls), 1)

    async def test_deadline_stops_inflight(self):
        closed = asyncio.Event()
        entered = asyncio.Event()
        async def handler(_):
            entered.set()
            try:
                await asyncio.sleep(3)
            finally:
                closed.set()
        async with LocalModelProvider(self.config, transport=httpx.MockTransport(handler)) as provider:
            with self.assertRaises(ModelDeadlineExceeded):
                await provider.infer(self.clip, prompt="synthetic", response_schema=SCHEMA,
                                      deadline=time.monotonic() + 0.5)
        # Under a heavily loaded host the deadline may expire before HTTP starts;
        # either case must fail, and an entered HTTP handler must be cancelled.
        if entered.is_set():
            self.assertTrue(closed.is_set())

    async def test_thread_event_cancels_inflight(self):
        cancel = threading.Event()
        async def handler(_):
            cancel.set()
            await asyncio.sleep(3)
        with self.assertRaises(ModelCancelled):
            await self.run_request(handler, cancel_event=cancel)

    async def test_cancelled_before_start_no_request(self):
        cancel = asyncio.Event()
        cancel.set()
        with self.assertRaises(ModelCancelled):
            await self.run_request(lambda _: self.fail("should not request"), cancel_event=cancel)

    async def test_repeated_cancel_preserves_slow_response_close(self):
        entered, closing, release, closed = (asyncio.Event() for _ in range(4))
        close_calls = []
        class SlowStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                entered.set()
                yield b'{"synthetic":'
                await asyncio.Event().wait()
            async def aclose(self):
                close_calls.append(1)
                closing.set()
                await release.wait()
                closed.set()
        async with LocalModelProvider(self.config, transport=httpx.MockTransport(
                lambda _: httpx.Response(200, stream=SlowStream()))) as provider:
            task = asyncio.create_task(provider.infer(self.clip, prompt="synthetic",
                response_schema=SCHEMA, deadline=time.monotonic()+2))
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            await asyncio.wait_for(closing.wait(), 1)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(closed.is_set())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(closed.is_set())
            self.assertEqual(len(close_calls), 1)
            self.assertEqual(provider.pending_cleanup_count, 0)

    async def test_repeated_cancel_preserves_preheaders_transport_cleanup(self):
        entered, closing, release, closed = (asyncio.Event() for _ in range(4))
        async def handler(_):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                closing.set()
                await release.wait()
                closed.set()
        async with LocalModelProvider(self.config, transport=httpx.MockTransport(handler)) as provider:
            task = asyncio.create_task(provider.infer(self.clip, prompt="synthetic",
                response_schema=SCHEMA, deadline=time.monotonic()+2))
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            await asyncio.wait_for(closing.wait(), 1)
            task.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(closed.is_set())

    async def test_cancel_during_eof_close_is_shielded(self):
        closing, release, closed = (asyncio.Event() for _ in range(3))
        class SlowStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield json.dumps(envelope()).encode()
            async def aclose(self):
                closing.set()
                await release.wait()
                closed.set()
        async with LocalModelProvider(self.config, transport=httpx.MockTransport(
                lambda _: httpx.Response(200, stream=SlowStream()))) as provider:
            task = asyncio.create_task(provider.infer(self.clip, prompt="synthetic",
                response_schema=SCHEMA, deadline=time.monotonic()+2))
            await asyncio.wait_for(closing.wait(), 1)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(closed.is_set())

    async def test_cleanup_grace_retains_and_eventually_settles_close(self):
        entered, closing, release, closed = (asyncio.Event() for _ in range(4))
        class SlowStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                entered.set()
                yield b'{"synthetic":'
                await asyncio.Event().wait()
            async def aclose(self):
                closing.set()
                await release.wait()
                closed.set()
        config = replace(self.config, cleanup_grace_sec=0.03)
        async with LocalModelProvider(config, transport=httpx.MockTransport(
                lambda _: httpx.Response(200, stream=SlowStream()))) as provider:
            task = asyncio.create_task(provider.infer(self.clip, prompt="synthetic",
                response_schema=SCHEMA, deadline=time.monotonic()+2))
            await asyncio.wait_for(entered.wait(), 1)
            started = time.monotonic()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertLess(time.monotonic()-started, 0.5)
            self.assertTrue(closing.is_set())
            self.assertGreater(provider.pending_cleanup_count, 0)
            release.set()
            await asyncio.wait_for(closed.wait(), 1)
            for _ in range(10):
                if not provider.pending_cleanup_count:
                    break
                await asyncio.sleep(0)
            self.assertEqual(provider.pending_cleanup_count, 0)

    async def test_accepted_post_read_failure_not_retried_and_closed(self):
        calls, closes = [], []
        class FailedStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'{"synthetic":'
                raise httpx.ReadTimeout("synthetic-private-error")
            async def aclose(self):
                closes.append(1)
        def handler(_):
            calls.append(1)
            return httpx.Response(200, stream=FailedStream())
        with self.assertRaises(ModelUnavailable) as error:
            await self.run_request(handler)
        self.assertEqual((len(calls), len(closes)), (1, 1))
        self.assertNotIn("synthetic-private", str(error.exception))

    async def test_cross_job_file_rejected(self):
        other = self.root / "other_job"
        other.mkdir()
        target = other / "frame.jpg"
        target.write_bytes(self.image.read_bytes())
        self.clip.frame_refs = [self.frame.model_copy(update={"path": str(target)})]
        with self.assertRaises(UnsafeMedia):
            await self.run_request(lambda _: self.fail("should not request"))

    async def test_remote_media_rejected(self):
        self.clip.frame_refs = [self.frame.model_copy(update={"path": "https://example.com/media.jpg"})]
        with self.assertRaises(UnsafeMedia):
            await self.run_request(lambda _: self.fail("should not request"))

    async def test_checksum_change_rejected(self):
        self.clip.frame_refs = [self.frame.model_copy(update={"jpeg_sha256": "0" * 64})]
        with self.assertRaisesRegex(UnsafeMedia, "integrity"):
            await self.run_request(lambda _: self.fail("should not request"))

    async def test_frame_dimensions_verified(self):
        self.clip.frame_refs = [self.frame.model_copy(update={"width": 99})]
        with self.assertRaises(UnsafeMedia):
            await self.run_request(lambda _: self.fail("should not request"))

    async def test_frame_budget_never_silently_drops(self):
        self.clip.frame_refs = [self.frame.model_copy(update={"frame_id": f"f{i}"}) for i in range(17)]
        with self.assertRaisesRegex(UnsafeMedia, "budget"):
            await self.run_request(lambda _: self.fail("should not request"))

    async def test_response_byte_budget(self):
        self.config = replace(self.config, max_output_bytes=16)
        with self.assertRaisesRegex(InvalidModelOutput, "budget"):
            await self.run_request(lambda _: httpx.Response(200, json=envelope()))

    async def test_response_identity_encoding_required(self):
        def handler(request):
            self.assertEqual(request.headers["accept-encoding"], "identity")
            return httpx.Response(200, headers={"content-encoding": "synthetic-encoding"},
                                  content=json.dumps(envelope()).encode())
        with self.assertRaisesRegex(InvalidModelOutput, "encoding"):
            await self.run_request(handler)

    async def test_remote_schema_ref_never_fetches(self):
        async with LocalModelProvider(self.config) as provider:
            with self.assertRaisesRegex(ValueError, "local references"):
                await provider.infer(self.clip, prompt="synthetic", deadline=time.monotonic()+2,
                                      response_schema={"$ref": "https://public.example/schema.json"})

    async def test_availability_checks_real_model_identity(self):
        async with LocalModelProvider(self.config, transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"data": [{"id": self.config.model}]}))) as provider:
            await provider.check_available(deadline=time.monotonic()+2)
        async with LocalModelProvider(self.config, transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"data": [{"id": "other"}]}))) as provider:
            with self.assertRaises(ModelUnavailable):
                await provider.check_available(deadline=time.monotonic()+2)

    async def test_ready_requires_hash_verified_revision_and_warmup(self):
        calls = []
        def handler(request):
            calls.append(request.url.path)
            if request.url.path == "/health":
                return httpx.Response(200)
            if request.url.path == "/readyz":
                return httpx.Response(200, json={"ready": True, "model_files_verified": True,
                    "model_revision": self.config.expected_revision,
                    "processor_revision": self.config.expected_revision,
                    "model_manifest_sha256": "a"*64,
                    "vllm_version": self.config.expected_vllm_version})
            return httpx.Response(200, json={"data": [{"id": self.config.model}]})
        async with LocalModelProvider(self.config, transport=httpx.MockTransport(handler)) as provider:
            await provider.check_ready(deadline=time.monotonic()+2)
        self.assertEqual(calls, ["/health", "/readyz", "/v1/models"])

    async def test_actual_ready_attestation_propagated_without_config_guess(self):
        self.config = replace(self.config, require_readiness=True)
        def handler(request):
            if request.url.path == "/health":
                return httpx.Response(200)
            if request.url.path == "/readyz":
                return httpx.Response(200, json={"ready": True, "model_files_verified": True,
                    "model_revision": self.config.expected_revision,
                    "processor_revision": self.config.expected_revision,
                    "model_manifest_sha256": "a"*64,
                    "vllm_version": self.config.expected_vllm_version})
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": self.config.model}]})
            return httpx.Response(200, json=envelope())
        async with LocalModelProvider(self.config, transport=httpx.MockTransport(handler)) as provider:
            self.assertIsNone(provider.serving_attestation)
            result = await provider.infer(self.clip, prompt="synthetic", response_schema=SCHEMA,
                                          deadline=time.monotonic()+2)
            self.assertEqual(result.metrics.serving_attestation["model_manifest_sha256"], "a"*64)
            external = provider.serving_attestation
            external["model_revision"] = "wrong"
            self.assertEqual(provider.serving_attestation["model_revision"], self.config.expected_revision)

    async def test_ready_rejects_alias_without_revision_attestation(self):
        async with LocalModelProvider(self.config, transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"ready": True}))) as provider:
            with self.assertRaisesRegex(ModelUnavailable, "identity"):
                await provider.check_ready(deadline=time.monotonic()+2)

    async def test_cancel_racing_ready_lock_acquire_does_not_poison_provider(self):
        cancelled = asyncio.Event()
        async with LocalModelProvider(self.config, transport=httpx.MockTransport(
            lambda request: httpx.Response(503))) as provider:
            asyncio.get_running_loop().call_soon(cancelled.set)
            with self.assertRaises(ModelCancelled):
                await provider.check_ready(deadline=time.monotonic()+2, cancel_event=cancelled)
            self.assertFalse(provider._readiness_lock.locked())
            with self.assertRaises(ModelUnavailable):
                await provider.check_ready(deadline=time.monotonic()+2)

    async def test_production_infer_does_not_send_media_before_readiness(self):
        self.config = replace(self.config, require_readiness=True)
        calls = []
        def handler(request):
            calls.append(request.method)
            return httpx.Response(503)
        with self.assertRaises(ModelUnavailable):
            await self.run_request(handler)
        self.assertEqual(calls, ["GET", "GET"])


class PureModelSecurityTests(unittest.TestCase):
    def test_private_endpoints_only(self):
        for url in ["http://127.0.0.1:8000/v1", "http://[::1]:8000", "http://10.1.2.3/v1",
                    "http://172.16.1.2/v1", "https://192.168.1.2:8000/v1", "http://model:8000/v1"]:
            validate_private_endpoint(url, ("model",))
        for url in ["https://api.openai.com/v1", "https://example.com/v1", "http://8.8.8.8/v1",
                    "http://169.254.169.254/v1", "http://0.0.0.0/v1", "file:///local/path",
                    "http://model:8000/v1?redirect=1", "http://user:secret@127.0.0.1/v1",
                    "http://127.0.0.1/redirect", "http://192.0.2.1/v1"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_private_endpoint(url, ("model",))

    def test_strict_json_rejects_ambiguous_and_wrapped_outputs(self):
        for raw in ['{"a":1,"a":2}', '{"v":NaN}', '```json\n{"a":1}\n```', '[]', '{} trailing']:
            with self.subTest(raw=raw), self.assertRaises(InvalidModelOutput):
                strict_json_object(raw)

    def test_secret_not_in_config_repr(self):
        config = LocalModelConfig(media_root=Path("synthetic"), api_key="synthetic-secret-token")
        self.assertNotIn("synthetic-secret-token", repr(config))


class ModelManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="forte-manifest-test-")
        self.root = Path(self.temp.name)
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        files = []
        for name in ["config.json", "preprocessor_config.json", "tokenizer_config.json", "model.safetensors"]:
            raw = b"synthetic-not-a-model-file"
            (self.snapshot / name).write_bytes(raw)
            files.append({"path": name, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        self.manifest = {"model_id": "synthetic/Qwen-fixture", "revision": "a"*40,
                         "processor_revision": "a"*40, "license": "apache-2.0",
                         "license_verified": True, "selected_adapter": None, "files": files}
        self.path = self.root / "manifest.json"
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        self.path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_exact_snapshot_checksums(self):
        result = verify_local_snapshot(self.snapshot, self.path)
        self.assertEqual(result.files_verified, 4)

    def test_missing_snapshot_fails_clearly(self):
        with self.assertRaisesRegex(ManifestError, "snapshot_missing"):
            verify_local_snapshot(self.root / "unavailable", self.path)

    def test_tampered_weight_same_size_fails(self):
        (self.snapshot / "model.safetensors").write_bytes(b"x"*len(b"synthetic-not-a-model-file"))
        with self.assertRaisesRegex(ManifestError, "checksum"):
            verify_local_snapshot(self.snapshot, self.path)

    def test_processor_revision_mismatch_fails(self):
        self.manifest["processor_revision"] = "b"*40
        self.save()
        with self.assertRaises(ManifestError):
            load_manifest(self.path)

    def test_unknown_license_fails(self):
        self.manifest["license"] = "unknown"
        self.save()
        with self.assertRaises(ManifestError):
            load_manifest(self.path)

    def test_manifest_path_traversal_rejected(self):
        self.manifest["files"][0]["path"] = "../config.json"
        self.save()
        with self.assertRaises(ManifestError):
            load_manifest(self.path)

    def test_staging_already_verified_files_never_use_network(self):
        with patch("forte.model.staging.build_opener", side_effect=AssertionError("network forbidden")):
            result = stage_local_snapshot(self.snapshot, self.path)
        self.assertTrue(result["verified"])
        self.assertEqual(result["bytes_downloaded"], 0)
        self.assertFalse(result["local_inference_verified"])

    def test_staging_resume_checks_range_and_checksum(self):
        raw = b"synthetic-weight-content"
        entry = {"path": "new.safetensors", "size_bytes": len(raw),
                 "sha256": hashlib.sha256(raw).hexdigest()}
        (self.snapshot / "new.safetensors.part").write_bytes(raw[:3])
        class Response(BytesIO):
            headers = {"Content-Range": f"bytes 3-{len(raw)-1}/{len(raw)}"}
            def getcode(self):
                return 206
        class Opener:
            def open(inner, request, **kwargs):
                self.assertEqual(request.get_header("Range"), "bytes=3-")
                return Response(raw[3:])
        with patch("forte.model.staging.build_opener", return_value=Opener()):
            count = _download(self.snapshot, entry, "synthetic/model", "a"*40, time.monotonic()+2)
        self.assertEqual(count, len(raw)-3)
        self.assertEqual((self.snapshot / "new.safetensors").read_bytes(), raw)
        self.assertFalse((self.snapshot / "new.safetensors.part").exists())

    def test_staging_corrupt_download_never_published(self):
        raw = b"synthetic-weight-content"
        entry = {"path": "new.safetensors", "size_bytes": len(raw), "sha256": "0"*64}
        class Response(BytesIO):
            headers = {}
            def getcode(self):
                return 200
        class Opener:
            def open(inner, *args, **kwargs):
                return Response(raw)
        with patch("forte.model.staging.build_opener", return_value=Opener()):
            with self.assertRaisesRegex(ManifestError, "checksum"):
                _download(self.snapshot, entry, "synthetic/model", "a"*40, time.monotonic()+2)
        self.assertFalse((self.snapshot / "new.safetensors").exists())
        self.assertFalse((self.snapshot / "new.safetensors.part").exists())


if __name__ == "__main__":
    unittest.main()
