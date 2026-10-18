"""CPU synthetic service boundaries; no decoder, head, weights or GPU load."""
import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from serving.fast_service import (FastServiceConfig, FastServiceError, NativeFeatureBackend, NativeServiceApp,
                                  source_code_bundle, verify_source)


ROOT = Path(__file__).resolve().parents[2]


def config(root, **changes):
    result = FastServiceConfig(Path(root), Path("snapshot"), Path("manifest"), "1"*64,
        Path("feature-spec"), "2"*64, Path("head-identity"), "3"*64, Path("checkpoint"), "4"*64, "5"*64)
    return replace(result, **changes)


def asset(root, data=b"synthetic-video-placeholder", **changes):
    value = dict(job_id="generated-job", owned_path=str(Path(root) / "generated-job.mov"),
        content_hash=hashlib.sha256(data).hexdigest(), duration_sec=1., width=16, height=16,
        fps=30., time_base="1/30", source_start_sec=0., rotation_deg=0, codec="h264",
        frame_count=30, stream_index=0, metadata={})
    value.update(changes)
    return value


class SourceBoundaryTests(unittest.TestCase):
    def test_module_and_help_import_no_ml_or_decoder_without_site_packages(self):
        result = subprocess.run([sys.executable, "-S", "-c",
            "import json,sys; import serving.fast_service; print(json.dumps([x for x in "
            "('torch','transformers','numpy','av','forte.video.batches','fastapi','uvicorn') if x in sys.modules]))"],
            cwd=ROOT, text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(result.stdout), [])
        result = subprocess.run([sys.executable, "-S", "-m", "serving.fast_service", "--help"],
            cwd=ROOT, text=True, capture_output=True, check=True)
        self.assertIn("--checkpoint-manifest-sha256", result.stdout)

    def test_exact_job_owned_source_and_actual_digest_are_required(self):
        with TemporaryDirectory() as directory:
            value = asset(directory)
            source = Path(value["owned_path"])
            source.write_bytes(b"synthetic-video-placeholder")
            path, fingerprint = verify_source(SimpleNamespace(**value), config(directory),
                deadline=time.monotonic()+1, cancel_event=threading.Event())
            self.assertEqual(path, source)
            self.assertEqual(fingerprint[2], len(b"synthetic-video-placeholder"))
            for changes in ({"content_hash": "9"*64}, {"job_id": "other-job"},
                            {"owned_path": str(source.parent / ".." / source.name)}):
                with self.subTest(changes=changes), self.assertRaises(FastServiceError):
                    verify_source(SimpleNamespace(**{**value, **changes}), config(directory),
                        deadline=time.monotonic()+1, cancel_event=threading.Event())

    def test_source_limits_and_cancel_before_file_read(self):
        with TemporaryDirectory() as directory:
            value = asset(directory)
            Path(value["owned_path"]).write_bytes(b"synthetic-video-placeholder")
            for changes in ({"duration_sec": 143.}, {"frame_count": 10001}, {"job_id": "../escape"},
                            {"width": 100000, "height": 100000}):
                with self.subTest(changes=changes), self.assertRaises(FastServiceError):
                    verify_source(SimpleNamespace(**{**value, **changes}), config(directory),
                        deadline=time.monotonic()+1, cancel_event=threading.Event())
            event = threading.Event()
            event.set()
            with self.assertRaisesRegex(FastServiceError, "native_request_cancelled"):
                verify_source(SimpleNamespace(**value), config(directory),
                    deadline=time.monotonic()+1, cancel_event=event)
            with self.assertRaisesRegex(FastServiceError, "native_source_size_invalid"):
                verify_source(SimpleNamespace(**value), config(directory, max_source_bytes=2),
                    deadline=time.monotonic()+1, cancel_event=threading.Event())

    def test_configuration_bounds_are_checked_without_runtime(self):
        for changes in ({"batch_size": 129}, {"device": "cpu"}, {"max_source_frames": 10001},
                        {"max_budget_sec": float("inf")}, {"max_duration_sec": 144.}):
            with self.subTest(changes=changes), self.assertRaises(FastServiceError):
                config(".", **changes)

    def test_code_bundle_hashes_packaged_code_without_private_artifacts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for group in ("serving", "src/forte", "training"):
                (root / group).mkdir(parents=True)
                (root / group / "module.py").write_text("# synthetic code\n")
            (root / "training/private").mkdir()
            (root / "training/private/untrusted.py").write_text("# private artifact excluded\n")
            bundle = source_code_bundle(root)
            self.assertEqual(len(bundle["files"]), 3)
            self.assertNotIn("training/private/untrusted.py", bundle["files"])
            before = dict(bundle["files"])
            (root / "serving/module.py").write_text("# changed code\n")
            self.assertNotEqual(source_code_bundle(root)["files"], before)


class StartupBindingTests(unittest.TestCase):
    def fixture(self, directory, *, tamper_spec=False, tamper_identity=False):
        from training.guards import canonical_sha256
        from training.temporal.feature_spec import build_feature_spec
        decoder = {"sha256": "d"*64, "synthetic_only": True}
        recipe = {"encoder_manifest_sha256": "1"*64, "synthetic_only": True}
        sampling = {"mode": "native", "all_native_frames": True, "tail_required": True,
                    "time": "original_integer_pts_and_canonical_rational_base"}
        crop = {"views": "global_only", "source_box_xyxy": [0., 0., 1., 1.]}
        spec, spec_sha = build_feature_spec(recipe, decoder_recipe_sha256=canonical_sha256(decoder),
            sampling_recipe_sha256=canonical_sha256(sampling), crop_recipe_sha256=canonical_sha256(crop))
        identity = {"encoder_id": "synthetic-encoder", "encoder_revision": "synthetic-revision",
            "encoder_artifacts_sha256": "1"*64, "head_sha256": "a"*64,
            "feature_spec_sha256": spec_sha, "taxonomy_sha256": "c"*64,
            "fact_ids": [f"fact-{index}" for index in range(19)], "event_type_ids": []}
        if tamper_spec:
            spec["encoder_recipe"]["changed"] = True
        if tamper_identity:
            identity["encoder_artifacts_sha256"] = "9"*64
        root = Path(directory)
        spec_path, identity_path = root / "feature-spec.json", root / "identity.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        identity_path.write_text(json.dumps(identity), encoding="utf-8")
        cfg = config(root, feature_spec=spec_path, feature_spec_sha256=spec_sha,
            head_identity=identity_path, head_identity_sha256=hashlib.sha256(identity_path.read_bytes()).hexdigest())
        cfg = replace(cfg, source_code_bundle_sha256=canonical_sha256(source_code_bundle()))
        return cfg, recipe, decoder

    def test_spec_or_encoder_identity_tampering_fails_before_model_creation(self):
        for changes in ({"tamper_spec": True}, {"tamper_identity": True}):
            with TemporaryDirectory() as directory, self.subTest(changes=changes):
                cfg, _, _ = self.fixture(directory, **changes)
                with patch("serving.fast_encoder.FastEncoder") as encoder, self.assertRaises(FastServiceError):
                    NativeFeatureBackend(cfg).load()
                encoder.assert_not_called()

    def test_executed_recipe_mismatch_closes_encoder_and_never_readies_head(self):
        with TemporaryDirectory() as directory:
            cfg, recipe, decoder = self.fixture(directory)
            with (patch("forte.video.batches.rgb_transform_recipe", return_value=decoder),
                  patch("serving.fast_encoder.FastEncoder") as encoder,
                  patch("forte.decision.temporal_native.NativeHeadRunner") as runner):
                encoder.return_value.recipe = {**recipe, "runtime_changed": True}
                backend = NativeFeatureBackend(cfg)
                with self.assertRaisesRegex(FastServiceError, "native_pinned_runtime_load_failed"):
                    backend.load()
                encoder.return_value.close.assert_called_once()
                runner.assert_not_called()
                self.assertFalse(backend.check_ready()["ready"])

    def test_matching_pins_load_once_and_keep_calibration_explicitly_unverified(self):
        with TemporaryDirectory() as directory:
            cfg, recipe, decoder = self.fixture(directory)
            with (patch("forte.video.batches.rgb_transform_recipe", return_value=decoder),
                  patch("serving.fast_encoder.FastEncoder") as encoder,
                  patch("forte.decision.temporal_native.NativeHeadRunner") as runner):
                encoder.return_value.recipe = recipe
                encoder.return_value.load.return_value = {"gpu_name": "synthetic-only", "gpu_compute_capability": [8, 0]}
                runner.return_value.load.return_value = {"ready": True}
                runner.return_value.check_ready.return_value = {"ready": True}
                backend = NativeFeatureBackend(cfg)
                ready = backend.load()
                self.assertTrue(ready["ready"])
                self.assertFalse(ready["candidate_calibration_verified"])
                self.assertEqual(ready["feature_spec_sha256"], cfg.feature_spec_sha256)
                self.assertEqual(backend.execution_manifest["batch_size"], 32)
                self.assertEqual(backend.execution_manifest["decoder_threads"], 4)
                self.assertEqual(backend.execution_manifest["prefetch_batches"], 2)
                self.assertEqual(ready["source_code_bundle_sha256"], cfg.source_code_bundle_sha256)
                self.assertEqual(len(ready["execution_manifest_sha256"]), 64)
                self.assertEqual(backend.load(), ready)
                encoder.assert_called_once()
                runner.assert_called_once()
                backend.close()
                self.assertFalse(backend.check_ready()["ready"])


class FakeBackend:
    def __init__(self):
        self.ready = True
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancel_seen = threading.Event()
        self.block = self.error = False
        self.calls = 0
        self.closed = False

    def check_ready(self):
        return {"ready": self.ready, "head_sha256": "a"*64,
            "feature_spec_sha256": "b"*64, "candidate_calibration_verified": False,
            "execution_manifest_sha256": "e"*64, "source_code_bundle_sha256": "f"*64}

    def load(self):
        return self.check_ready()

    def close(self):
        self.closed = True

    def predict(self, value, *, clip_id, deadline, cancel_event):
        self.calls += 1
        self.started.set()
        while self.block and not self.release.wait(.005):
            if cancel_event.is_set():
                self.cancel_seen.set()
        if self.error:
            raise RuntimeError("private generated decoder path and observations")
        return {"schema_version": "synthetic-only", "metrics": {"native_frames": 30.}}


class FastServiceHttpTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.backend = FakeBackend()
        self.app = NativeServiceApp(self.backend)

    async def asyncTearDown(self):
        self.backend.release.set()
        if self.app._active:
            await asyncio.wait(tuple(self.app._active), timeout=1)
        self.app._executor.shutdown(wait=True)

    async def request(self, payload=None, *, raw=None, method="POST", path="/v1/temporal/predict",
                      disconnect=None):
        body = raw if raw is not None else json.dumps(payload or {
            "asset": asset(Path.cwd()), "deadline_budget_sec": 1., "clip_id": "generated-clip"}).encode()
        messages = asyncio.Queue()
        await messages.put({"type": "http.request", "body": body})
        sent = []

        async def receive():
            if not messages.empty():
                return await messages.get()
            if disconnect is not None:
                await disconnect.wait()
                return {"type": "http.disconnect"}
            return await messages.get()

        async def send(message):
            sent.append(message)

        await self.app({"type": "http", "method": method, "path": path}, receive, send)
        return sent

    @staticmethod
    def response(messages):
        return messages[0]["status"], json.loads(messages[1]["body"])

    async def wait_until(self, predicate):
        async with asyncio.timeout(1):
            while not predicate():
                await asyncio.sleep(.002)

    async def test_candidate_readiness_is_explicit_and_no_model_is_loaded(self):
        status, body = self.response(await self.request(method="GET", path="/ready"))
        self.assertEqual(status, 200)
        self.assertFalse(body["candidate_calibration_verified"])
        self.assertEqual(self.backend.calls, 0)
        self.backend.ready = False
        self.assertEqual(self.response(await self.request(method="GET", path="/ready"))[0], 503)

    async def test_real_cpu_http_client_checks_service_readiness_and_execution_pins(self):
        import httpx
        from forte.decision.head_contract import HeadIdentity
        from forte.decision.temporal_http import HttpNativeHeadClient
        from forte.decision.temporal_provider import TemporalProviderError
        identity = HeadIdentity(encoder_id="synthetic-encoder", encoder_revision="synthetic-revision",
            encoder_artifacts_sha256="1"*64, head_sha256="a"*64, feature_spec_sha256="b"*64,
            taxonomy_sha256="c"*64, fact_ids=tuple(f"fact-{index}" for index in range(19)))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app)) as transport:
            client = HttpNativeHeadClient("http://127.0.0.1:8010", identity=identity, client=transport,
                expected_execution_manifest_sha256="e"*64, expected_source_code_bundle_sha256="f"*64)
            ready = await client.check_ready(deadline=time.monotonic()+1)
            self.assertTrue(ready["ready"])
            self.assertEqual(ready["execution_manifest_sha256"], "e"*64)
            client.expected_execution_manifest_sha256 = "d"*64
            with self.assertRaises(TemporalProviderError):
                await client.check_ready(deadline=time.monotonic()+1)
        self.assertEqual(self.backend.calls, 0)

    async def test_valid_bounded_request_calls_native_backend_once(self):
        status, body = self.response(await self.request())
        self.assertEqual((status, body["metrics"]["native_frames"]), (200, 30.))
        self.assertEqual(self.backend.calls, 1)

    async def test_unknown_fields_invalid_json_and_budgets_never_start_backend(self):
        for payload in ({"asset": asset(Path.cwd()), "deadline_budget_sec": True},
                        {"asset": asset(Path.cwd()), "deadline_budget_sec": 121.},
                        {"asset": asset(Path.cwd()), "deadline_budget_sec": 1., "labels": [1]},
                        {"asset": asset(Path.cwd()), "deadline_budget_sec": 1., "clip_id": "../secret"}):
            with self.subTest(payload_keys=tuple(payload)):
                self.assertEqual(self.response(await self.request(payload))[0], 400)
        self.assertEqual(self.response(await self.request(raw=b"private invalid JSON"))[0], 400)
        self.assertEqual(self.backend.calls, 0)

    async def test_large_request_is_rejected_before_model_and_errors_hide_exception_text(self):
        self.assertEqual(self.response(await self.request(raw=b"x"*(64*1024+1)))[0], 413)
        self.assertEqual(self.backend.calls, 0)
        self.backend.error = True
        status, body = self.response(await self.request())
        self.assertEqual(status, 422)
        self.assertEqual(body, {"error": "native_prediction_failed"})

    async def test_timeout_retains_gpu_admission_until_actual_worker_drains(self):
        self.backend.block = True
        task = asyncio.create_task(self.request({"asset": asset(Path.cwd()), "deadline_budget_sec": .05}))
        await self.wait_until(self.backend.started.is_set)
        status, _ = self.response(await task)
        self.assertEqual(status, 504)
        await self.wait_until(self.backend.cancel_seen.is_set)
        self.assertTrue(self.app._busy)
        self.assertEqual(self.response(await self.request())[0], 503)
        self.assertEqual(self.backend.calls, 1)
        self.backend.release.set()
        await self.wait_until(lambda: not self.app._busy)
        self.assertEqual(self.response(await self.request())[0], 200)
        self.assertEqual(self.backend.calls, 2)

    async def test_disconnect_signals_cancel_and_does_not_release_worker_early(self):
        self.backend.block = True
        disconnect = asyncio.Event()
        task = asyncio.create_task(self.request(disconnect=disconnect))
        await self.wait_until(self.backend.started.is_set)
        disconnect.set()
        self.assertEqual(await task, [])
        await self.wait_until(self.backend.cancel_seen.is_set)
        self.assertTrue(self.app._busy)
        self.backend.release.set()
        await self.wait_until(lambda: not self.app._busy)

    async def test_outer_cancellation_is_also_retained_until_drain(self):
        self.backend.block = True
        task = asyncio.create_task(self.request())
        await self.wait_until(self.backend.started.is_set)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await self.wait_until(self.backend.cancel_seen.is_set)
        self.assertTrue(self.app._busy)
        self.backend.release.set()
        await self.wait_until(lambda: not self.app._busy)

    async def test_response_limit_and_missing_ready_never_return_large_private_body(self):
        self.app.max_response_bytes = 2
        self.assertEqual(self.response(await self.request())[0], 422)
        self.backend.ready = False
        self.assertEqual(self.response(await self.request())[0], 503)

    async def test_lifespan_closes_only_after_work_drained(self):
        messages = iter(({"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}))
        sent = []
        async def receive():
            return next(messages)
        async def send(value):
            sent.append(value)
        await self.app({"type": "lifespan"}, receive, send)
        self.assertEqual([row["type"] for row in sent],
            ["lifespan.startup.complete", "lifespan.shutdown.complete"])
        self.assertTrue(self.backend.closed)


if __name__ == "__main__":
    unittest.main()
