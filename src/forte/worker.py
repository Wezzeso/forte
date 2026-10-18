"""Independent, crash-recoverable job worker. Run with ``python -m forte.worker``."""

from __future__ import annotations

import os

import argparse
import asyncio
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import signal
import threading
import time
from typing import Any

from forte.config import WorkerSettings
from forte.decision import DecisionConfigurationError, DecisionOutputError
from forte.decision.structured import DECISION_OUTPUT_SUBCODES
from forte.jobs import JobStore, LeaseLost
from forte.observability import StageTimer, sanitize_model_metrics
from forte.observability.timing import STAGES
from forte.pipeline import AnalysisPipeline, atomic_private_json


def safe_error_code(error: BaseException) -> str:
    """Static codes only. Never log model notes, paths or exception messages."""
    if isinstance(error, TimeoutError):
        return "deadline_exceeded"
    if isinstance(error, DecisionOutputError):
        return "invalid_model_output"
    if isinstance(error, DecisionConfigurationError):
        return "decision_configuration_error"
    allowed = {
        "analysis_cancelled", "analysis_incomplete", "no_enabled_verified_rules",
        "accepted_source_hash_changed", "refinement_budget_insufficient",
        "insufficient_visible_evidence", "model_unavailable", "model_cancelled",
        "model_deadline_exceeded", "invalid_model_output",
    }
    code = getattr(error, "code", "")
    return code if code in allowed else "processing_failed"


def safe_stage_timings(value: Any) -> dict[str, Any] | None:
    """Expose only fixed numeric timings, never arbitrary exception attributes."""
    if not isinstance(value, dict) or value.get("clock") != "process_monotonic":
        return None
    stages = value.get("stages_ms")
    if not isinstance(stages, dict):
        return None
    if any(key not in STAGES or isinstance(duration, bool)
           or not isinstance(duration, (int, float))
           or not math.isfinite(duration) or duration < 0
           for key, duration in stages.items()):
        return None
    result = {"clock": "process_monotonic", "stages_ms": dict(stages)}
    model_requests = sanitize_model_metrics(value.get("model_requests"))
    if model_requests is not None:
        result["model_requests"] = model_requests
    return result


class Worker:
    def __init__(self, settings: WorkerSettings, *, store: JobStore | None = None,
                 pipeline: Any = None):
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.work_dir.mkdir(parents=True, exist_ok=True)
        self.store = store or JobStore(settings.database_url, settings.result_dir,
                                      lease_seconds=settings.lease_seconds,
                                      job_timeout_seconds=settings.job_timeout_seconds)
        if pipeline is not None:
            self.pipeline = pipeline
        elif os.environ.get("FORTE_ANALYZER", "").strip().lower() == "fast":
            from forte.fast.service import FastPipeline
            self.pipeline = FastPipeline(settings)
        elif settings.model_native_launch_config_path is not None:
            from forte.native_dispatch import make_native_worker_pipeline
            self.pipeline = make_native_worker_pipeline(settings)
        else:
            self.pipeline = AnalysisPipeline(settings)
        self.stop = asyncio.Event()
        self._owned_store = store is None

    async def _lease_keeper(self, job: Any, cancelled: threading.Event) -> None:
        try:
            while not cancelled.is_set():
                await asyncio.sleep(self.settings.heartbeat_seconds)
                if cancelled.is_set():
                    return
                await asyncio.to_thread(self.store.heartbeat, job.job_id,
                                        self.settings.worker_id, job.attempt)
        except asyncio.CancelledError:
            raise
        except Exception:
            # An unavailable DB is not permission to publish without ownership.
            cancelled.set()

    async def process_claimed(self, job: Any) -> dict[str, Any]:
        cancelled = threading.Event()
        keeper = asyncio.create_task(self._lease_keeper(job, cancelled))
        timer = StageTimer()
        timer.mark("T3")
        deadline = time.monotonic() + job.remaining_seconds()
        summary: dict[str, Any] = {"job_id": job.job_id, "attempt": job.attempt}

        async def stage(name: str) -> None:
            if cancelled.is_set():
                raise LeaseLost("worker ownership unavailable")
            await asyncio.to_thread(self.store.set_stage, job.job_id,
                                    self.settings.worker_id, job.attempt, name)

        try:
            async with asyncio.timeout(max(.001, deadline - time.monotonic())):
                with timer.stage("analysis"):
                    result = await self.pipeline.analyze(job, stage=stage, deadline=deadline,
                                                         cancel_event=cancelled)
                if cancelled.is_set():
                    raise LeaseLost("worker ownership unavailable")
                with timer.stage("publication"):
                    await asyncio.to_thread(self.store.publish, job.job_id,
                                            self.settings.worker_id, job.attempt, result.json_bytes)
                timer.mark("T4")
                summary.update(status="done", timings=timer.snapshot(),
                               pipeline_timings=safe_stage_timings(result.timings))
        except LeaseLost:
            summary.update(status="ownership_lost")
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                # Leave shutdown attempts leased for crash recovery.
                raise
            if cancelled.is_set():
                # A lost lease cancels the scheduler's future without calling
                # Task.cancel on this task. Do not stop unrelated jobs.
                summary.update(status="ownership_lost")
            else:
                try:
                    await asyncio.to_thread(self.store.fail, job.job_id, self.settings.worker_id,
                                            job.attempt, "model_cancelled", error_code="model_cancelled")
                except LeaseLost:
                    pass
                summary.update(status="failed", error_code="model_cancelled")
        except Exception as error:
            code = safe_error_code(error)
            try:
                await asyncio.to_thread(self.store.fail, job.job_id, self.settings.worker_id,
                                        job.attempt, code, error_code=code, retryable=False)
            except LeaseLost:
                pass
            summary.update(status="failed", error_code=code)
            if isinstance(error, DecisionOutputError):
                subcode = getattr(error, "forte_decision_subcode", None)
                if isinstance(subcode, str) and subcode in DECISION_OUTPUT_SUBCODES:
                    summary["decision_error_subcode"] = subcode
                rejected_metrics = sanitize_model_metrics(getattr(error, "forte_model_metrics", None))
                if rejected_metrics is not None:
                    summary["rejected_model_request_metrics"] = rejected_metrics
            partial = safe_stage_timings(getattr(error, "forte_timings", None))
            if partial is not None:
                summary["pipeline_timings"] = partial
        finally:
            cancelled.set()
            keeper.cancel()
            await asyncio.gather(keeper, return_exceptions=True)
        summary.setdefault("timings", timer.snapshot())
        # Only static diagnostics and timing values enter the aggregate worker log.
        return summary

    async def run_once(self) -> dict[str, Any] | None:
        await asyncio.to_thread(self.store.recover)
        job = await asyncio.to_thread(self.store.claim, self.settings.worker_id)
        return await self.process_claimed(job) if job is not None else None

    async def _heartbeat(self) -> None:
        while not self.stop.is_set():
            record = {
                "updated_at": datetime.now(UTC).isoformat(), "worker_id": self.settings.worker_id,
            }
            if self.settings.model_native_launch_config_path is not None:
                binding = getattr(self.pipeline, "native_consumer_binding", None)
                if not isinstance(binding, dict):
                    raise RuntimeError("native worker pipeline binding is unavailable")
                record["native_consumer"] = binding
            await asyncio.to_thread(atomic_private_json, self.settings.heartbeat_file, record)
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=self.settings.heartbeat_seconds)
            except TimeoutError:
                pass

    async def run(self, *, once: bool = False) -> None:
        if once:
            result = await self.run_once()
            print(json.dumps(result or {"status": "idle"}, ensure_ascii=False))
            return
        beat = asyncio.create_task(self._heartbeat())
        active: set[asyncio.Task] = set()
        try:
            while not self.stop.is_set():
                await asyncio.to_thread(self.store.recover)
                while len(active) < self.settings.max_concurrent_jobs and not self.stop.is_set():
                    job = await asyncio.to_thread(self.store.claim, self.settings.worker_id)
                    if job is None:
                        break
                    active.add(asyncio.create_task(self.process_claimed(job)))
                if active:
                    done, pending = await asyncio.wait(active, timeout=self.settings.poll_seconds,
                                                       return_when=asyncio.FIRST_COMPLETED)
                    active = set(pending)
                    for task in done:
                        if not task.cancelled():
                            print(json.dumps(task.result(), ensure_ascii=False), flush=True)
                else:
                    try:
                        await asyncio.wait_for(self.stop.wait(), timeout=self.settings.poll_seconds)
                    except TimeoutError:
                        pass
        finally:
            self.stop.set()
            for task in active:
                task.cancel()
            await asyncio.gather(*active, return_exceptions=True)
            beat.cancel()
            await asyncio.gather(beat, return_exceptions=True)

    async def aclose(self) -> None:
        close = getattr(self.pipeline, "aclose", None)
        if close is not None:
            await close()
        if self._owned_store:
            await asyncio.to_thread(self.store.close)


async def _main(once: bool, check: bool) -> int:
    worker = Worker(WorkerSettings.from_env())
    try:
        if check:
            ready = getattr(worker.pipeline.provider, "check_ready", None)
            if ready is None:
                raise RuntimeError("model_readiness_check_unavailable")
            await ready(deadline=time.monotonic() + 20)
            print(json.dumps({"ready": True, "rules": len(worker.pipeline.rules)}))
            return 0
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, worker.stop.set)
            except (NotImplementedError, RuntimeError):
                pass
        await worker.run(once=once)
        return 0
    finally:
        await worker.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the independent local analysis worker")
    parser.add_argument("--once", action="store_true", help="claim at most one queued job")
    parser.add_argument("--check", action="store_true", help="check local model and catalog readiness")
    args = parser.parse_args()
    try:
        code = asyncio.run(_main(args.once, args.check))
    except KeyboardInterrupt:
        code = 130
    except Exception as error:
        print(json.dumps({"ready": False, "error_code": safe_error_code(error)}))
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
