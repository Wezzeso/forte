"""Synthetic execution tests. These make no GPU/model-capacity assertion."""

import asyncio
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

from serving.launch import build_command
from serving.readiness import is_ready, publish_ready
from serving.scheduler import FairScheduler, QueueFull, SchedulerClosed
from serving.security import DataOnlyMiddleware


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.deadline = asyncio.get_running_loop().time() + 5

    async def test_round_robin_prevents_a_video_backlog_starving_another(self):
        started = asyncio.Event()
        release = asyncio.Event()
        order = []

        async def run(label):
            order.append(label)
            if label == "a0":
                started.set()
                await release.wait()
            return label

        async with FairScheduler(concurrency=1) as scheduler:
            first = asyncio.create_task(scheduler.submit("a", lambda: run("a0"), deadline=self.deadline))
            await started.wait()
            tasks = [asyncio.create_task(scheduler.submit("a", lambda i=i: run(f"a{i}"),
                      deadline=self.deadline)) for i in range(1, 4)]
            other = asyncio.create_task(scheduler.submit("b", lambda: run("b0"), deadline=self.deadline))
            await asyncio.sleep(0)
            release.set()
            await asyncio.gather(first, other, *tasks)
        self.assertEqual(order, ["a0", "a1", "b0", "a2", "a3"])

    async def test_inflight_per_job_leaves_capacity_for_another_job(self):
        a_started, b_started, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        active = 0
        max_active = 0

        async def run(job):
            nonlocal active, max_active
            active += 1
            max_active = max(active, max_active)
            (a_started if job == "a" else b_started).set()
            await release.wait()
            active -= 1
            return job

        async with FairScheduler(concurrency=2, max_inflight_per_job=1) as scheduler:
            tasks = [asyncio.create_task(scheduler.submit("a", lambda: run("a"), deadline=self.deadline))
                     for _ in range(3)]
            await a_started.wait()
            tasks.append(asyncio.create_task(scheduler.submit("b", lambda: run("b"), deadline=self.deadline)))
            await asyncio.wait_for(b_started.wait(), 1)
            self.assertEqual(scheduler.inflight, 2)
            release.set()
            await asyncio.gather(*tasks)
        self.assertEqual(max_active, 2)

    async def test_capacity_is_bounded_and_cancelled_entries_are_pruned(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def blocker():
            started.set()
            await release.wait()

        async with FairScheduler(concurrency=1, max_pending=2, max_pending_per_job=1) as scheduler:
            running = asyncio.create_task(scheduler.submit("busy", blocker, deadline=self.deadline))
            await started.wait()
            queued = asyncio.create_task(scheduler.submit("x", blocker, deadline=self.deadline))
            await asyncio.sleep(0)
            with self.assertRaises(QueueFull):
                await scheduler.submit("x", blocker, deadline=self.deadline)
            for _ in range(20):
                queued.cancel()
                await asyncio.gather(queued, return_exceptions=True)
                queued = asyncio.create_task(scheduler.submit("x", blocker, deadline=self.deadline))
                await asyncio.sleep(0)
                self.assertLessEqual(len(scheduler._requests), 2)
            queued.cancel()
            await asyncio.gather(queued, return_exceptions=True)
            release.set()
            await running

    async def test_expired_queued_work_never_reaches_provider(self):
        started, release = asyncio.Event(), asyncio.Event()
        called = False

        async def blocker():
            started.set()
            await release.wait()

        async def expired():
            nonlocal called
            called = True

        async with FairScheduler(concurrency=1) as scheduler:
            running = asyncio.create_task(scheduler.submit("a", blocker, deadline=self.deadline))
            await started.wait()
            with self.assertRaises(TimeoutError):
                await scheduler.submit("b", expired, deadline=asyncio.get_running_loop().time() + 0.02)
            release.set()
            await running
        self.assertFalse(called)

    async def test_running_deadline_and_cancellation_release_capacity(self):
        stopped = asyncio.Event()

        async def forever():
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        async with FairScheduler(concurrency=1) as scheduler:
            with self.assertRaises(TimeoutError):
                await scheduler.submit("a", forever, deadline=asyncio.get_running_loop().time() + 0.02)
            await asyncio.wait_for(stopped.wait(), 1)
            stopped.clear()
            cancel_event = asyncio.Event()
            task = asyncio.create_task(scheduler.submit("b", forever, deadline=self.deadline,
                                                       cancel_event=cancel_event))
            await asyncio.sleep(0.01)
            cancel_event.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(stopped.wait(), 1)

    async def test_lease_loss_cancels_all_clip_calls_and_provider_errors_propagate(self):
        started = asyncio.Event()

        async def forever():
            started.set()
            await asyncio.Event().wait()

        async def oom():
            raise MemoryError("synthetic provider OOM")

        async with FairScheduler(concurrency=1) as scheduler:
            running = asyncio.create_task(scheduler.submit("a", forever, deadline=self.deadline))
            await started.wait()
            queued = asyncio.create_task(scheduler.submit("a", forever, deadline=self.deadline))
            await asyncio.sleep(0)
            self.assertEqual(await scheduler.cancel_job("a"), 2)
            results = await asyncio.gather(running, queued, return_exceptions=True)
            self.assertTrue(all(isinstance(result, asyncio.CancelledError) for result in results))
            with self.assertRaises(MemoryError):
                await scheduler.submit("c", oom, deadline=self.deadline)
        with self.assertRaises(SchedulerClosed):
            await scheduler.submit("z", oom, deadline=self.deadline)

    async def test_worker_threading_signal_cancels_without_blocking_event_loop(self):
        cancelled = threading.Event()
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def model():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        async with FairScheduler(concurrency=1) as scheduler:
            call = asyncio.create_task(scheduler.submit("a", model, deadline=self.deadline,
                                                        cancel_event=cancelled))
            await started.wait()
            cancelled.set()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(call, .5)
            await asyncio.wait_for(stopped.wait(), .5)


class ServingSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_ready_route_exposes_only_verified_public_startup_metadata(self):
        config_path = Path("configs/serving/baseline.json").resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        verification = {"processor_revision": config["model_revision"], "manifest_sha256": "0" * 64}
        response = []
        async def app(*_):
            self.fail("readiness must terminate at private middleware")
        async def receive():
            return {"type": "http.request", "body": b""}
        async def send(item):
            response.append(item)
        with TemporaryDirectory() as root:
            state = Path(root) / "ready.json"
            state.write_text(json.dumps({"snapshot_verification": verification}))
            with patch.dict(os.environ, {"FORTE_SERVING_CONFIG": str(config_path), "FORTE_SERVING_STATE": str(state)}), \
                    patch("serving.readiness.is_ready", return_value=True):
                await DataOnlyMiddleware(app)({"type": "http", "method": "GET", "path": "/readyz"}, receive, send)
        self.assertEqual(response[0]["status"], 200)
        metadata = json.loads(response[1]["body"])
        self.assertEqual(metadata, {"ready": True, "model_revision": config["model_revision"],
                                    "processor_revision": config["model_revision"], "model_manifest_sha256": "0" * 64,
                                    "vllm_version": "0.31.0", "model_files_verified": True})

    async def test_model_side_media_fetching_is_rejected(self):
        accepted = []

        async def app(scope, receive, send):
            accepted.append(await receive())

        middleware = DataOnlyMiddleware(app, max_body_bytes=256)
        for address in ("https://example.invalid/x.jpg", "file:///videos/x.jpg", "/videos/x.jpg"):
            response = []
            async def receive():
                return {"type": "http.request", "body": json.dumps({"image_url": {"url": address}}).encode()}
            async def send(item):
                response.append(item)
            await middleware({"type": "http", "method": "POST"}, receive, send)
            self.assertEqual(response[0]["status"], 400)
        self.assertFalse(accepted)
        self.assertTrue(DataOnlyMiddleware._safe({"image_url": {"url": "data:image/jpeg;base64,/9j/"}}))
        self.assertFalse(DataOnlyMiddleware._safe({"type": "input_audio", "data": "fake"}))
        self.assertFalse(DataOnlyMiddleware._safe({"type": "image_embeds", "data": "fake"}))

    async def test_private_readiness_requires_matching_warmup_and_live_health(self):
        config = json.loads(Path("configs/serving/baseline.json").read_text(encoding="utf-8"))
        with TemporaryDirectory() as root:
            state = Path(root) / "ready.json"
            self.assertFalse(is_ready(config, state))
            verification = {"verified": True, "revision": config["model_revision"],
                            "processor_revision": config["model_revision"], "model_id": config["model_id"],
                            "manifest_sha256": "0" * 64}
            smoke = {"synthetic_only": True, "real_local_inference": True,
                     "schema_and_semantic_validation_passed": True}
            publish_ready(config, state, 123, [{"synthetic": True}],
                          verification=verification, schema_smoke=smoke)
            with patch("serving.readiness.os.kill"), patch("serving.readiness.request_local"):
                self.assertTrue(is_ready(config, state))
                self.assertFalse(is_ready({**config, "max_num_seqs": 3}, state))
            with patch("serving.readiness.os.kill"), patch("serving.readiness.request_local", side_effect=OSError):
                self.assertFalse(is_ready(config, state))

    async def test_launch_uses_only_local_pinned_model_and_closed_media_policy(self):
        config = json.loads(Path("configs/serving/baseline.json").read_text(encoding="utf-8"))
        command = build_command(config)
        self.assertEqual(command[:3], ["vllm", "serve", "/models/Qwen3-VL-8B-Instruct"])
        for flag in ("--no-enable-log-requests", "--no-enable-prefix-caching", "--no-skip-mm-profiling",
                     "--no-trust-request-mm-kwargs", "--no-trust-request-chat-template"):
            self.assertIn(flag, command)
        self.assertNotIn("--trust-remote-code", command)
        self.assertNotIn("--allowed-local-media-path", command)


if __name__ == "__main__":
    unittest.main()
