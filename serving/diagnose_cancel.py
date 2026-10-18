"""Two synthetic-only cancellation probes; emit aggregate lifecycle timings only.

This utility never opens supplied media. Run only in an exclusive coordinated
GPU window. It does not change serving configuration or infer successful abort
from a client exception alone.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time

import httpx

from forte.decision import DecisionConfig, DecisionEngine, build_prompt
from forte.model import LocalModelConfig, LocalModelProvider
from forte.model.adapter import ModelError
from rules.catalog import load_catalog
from serving.bench_synthetic import make_clips
from serving.scheduler import FairScheduler, ScheduledProvider


async def gauges(client: httpx.AsyncClient, url: str) -> dict[str, float]:
    async with client.stream("GET", url, timeout=3) as response:
        if response.status_code != 200:
            raise RuntimeError("private_metrics_unavailable")
        raw = bytearray()
        async for chunk in response.aiter_bytes():
            raw.extend(chunk)
            if len(raw) > 2 * 1024 * 1024:
                raise RuntimeError("private_metrics_byte_limit")
    selected = {"vllm:num_requests_running": 0.0, "vllm:num_requests_waiting": 0.0}
    found = set()
    for line in raw.decode().splitlines():
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in selected:
            selected[name] += float(line.rsplit(" ", 1)[1])
            found.add(name)
    if found != set(selected):
        raise RuntimeError("private_metrics_required_gauges_missing")
    return {"running": selected["vllm:num_requests_running"],
            "waiting": selected["vllm:num_requests_waiting"]}


async def probe(args, *, condition: str, seed: int) -> dict:
    started = time.monotonic()
    row = {"condition": condition, "synthetic_only": True,
           "input_frames": 16, "image_side": 672, "fact_count": 19,
           "stream": False, "generated_tokens": None}
    async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as metrics:
        before = await gauges(metrics, args.metrics_url)
        row["before"] = before
        if before["running"] or before["waiting"]:
            raise RuntimeError("exclusive_gpu_not_idle")
        with tempfile.TemporaryDirectory(prefix="forte-cancel-synthetic-") as temporary:
            root = Path(temporary)
            facts = load_catalog().fact_definitions()
            if len(facts) != 19:
                raise RuntimeError("production_fact_count_changed")
            clip = make_clips(root, seed=seed, count=1, frames=16, side=672, facts=facts)[0]
            # The diagnostic must never exceed the authorized two POST calls.
            async with LocalModelProvider(LocalModelConfig(media_root=root, base_url=args.base_url,
                                                          max_attempts=1)) as raw:
                await raw.check_ready(deadline=time.monotonic() + 20)
                config = DecisionConfig(compact_frame_ids=False)
                bundle = build_prompt(clip, reviewed_event_ids=(), config=config)
                cancelled = asyncio.Event()
                scheduler = FairScheduler(concurrency=2, max_inflight_per_job=2)
                await scheduler.start()
                request_started = time.monotonic()
                deadline = request_started + (8 if condition == "nested_shared_deadline" else 120)

                async def infer():
                    if condition == "direct_provider_signal":
                        return await raw.infer(clip, prompt=bundle.prompt,
                            response_schema=bundle.response_schema, deadline=deadline,
                            cancel_event=cancelled)
                    engine = DecisionEngine(ScheduledProvider(raw, scheduler),
                        reviewed_event_ids=(), config=config)
                    # Same nesting as worker -> decision -> scheduler -> adapter.
                    async with asyncio.timeout_at(deadline):
                        return await engine.analyze(clip, deadline=deadline, cancel_event=cancelled)

                task = asyncio.create_task(infer())
                admitted_at = None
                peak = {"running": 0.0, "waiting": 0.0}
                while admitted_at is None and not task.done():
                    sample = await gauges(metrics, args.metrics_url)
                    peak = {key: max(peak[key], sample[key]) for key in peak}
                    if sample["running"] > 0:
                        admitted_at = time.monotonic()
                        break
                    if time.monotonic() - request_started > 15:
                        task.cancel()
                        break
                    await asyncio.sleep(.1)
                row["admission_observed"] = admitted_at is not None
                row["admission_after_start_sec"] = (admitted_at - request_started) if admitted_at else None
                if condition == "direct_provider_signal" and admitted_at is not None:
                    await asyncio.sleep(5)
                    cancelled.set()
                    cancel_at = time.monotonic()
                else:
                    cancel_at = deadline
                row["cancellation_after_admission_sec"] = (cancel_at - admitted_at) if admitted_at else None
                row["cancellation_window_5_to_8_sec"] = bool(admitted_at is not None
                    and 4.95 <= cancel_at - admitted_at <= 8.05)
                try:
                    result = await task
                    row["caller_outcome"] = "completed_before_cancellation"
                    details = getattr(result, "metrics", None)
                    row["generated_tokens"] = getattr(details, "generated_tokens", None)
                except BaseException as error:
                    row["caller_outcome"] = type(error).__name__
                    if isinstance(error, ModelError):
                        row["static_model_error"] = str(error)
                terminal_at = time.monotonic()
                row["caller_terminal_after_start_sec"] = terminal_at - request_started
                row["caller_terminal_after_cancellation_sec"] = terminal_at - cancel_at
                row["scheduler_inflight_at_caller_terminal"] = scheduler.inflight
                idle_at = None
                while time.monotonic() - max(cancel_at, terminal_at) <= args.idle_timeout:
                    sample = await gauges(metrics, args.metrics_url)
                    peak = {key: max(peak[key], sample[key]) for key in peak}
                    if not sample["running"] and not sample["waiting"]:
                        idle_at = time.monotonic()
                        break
                    await asyncio.sleep(.2)
                row["peak_gauges"] = peak
                row["idle_observed"] = idle_at is not None
                row["idle_after_cancellation_sec"] = idle_at - cancel_at if idle_at else None
                row["idle_after_caller_terminal_sec"] = idle_at - terminal_at if idle_at else None
                row["after"] = sample
                row["scheduler_inflight_at_idle"] = scheduler.inflight
                await scheduler.close()
    row["probe_elapsed_sec"] = time.monotonic() - started
    return row


async def main_async(args) -> dict:
    rows = []
    for condition, seed in (("direct_provider_signal", 8710), ("nested_shared_deadline", 8720)):
        row = await probe(args, condition=condition, seed=seed)
        rows.append(row)
        # Do not add another request to an engine that did not actually drain.
        if not row["idle_observed"]:
            break
    return {"recorded_at": datetime.now(timezone.utc).isoformat(),
            "synthetic_only": True, "running_service_config_changed": False,
            "maximum_model_calls": 2, "max_attempts_per_call": 1, "records": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://model:8000/v1")
    parser.add_argument("--metrics-url", default="http://model:8000/metrics")
    parser.add_argument("--idle-timeout", type=float, default=180)
    args = parser.parse_args()
    if not 0 < args.idle_timeout <= 180:
        parser.error("idle timeout must be positive and at most180 seconds")
    try:
        print(json.dumps(asyncio.run(main_async(args))))
        return 0
    except Exception as error:
        print(json.dumps({"synthetic_only": True, "probe_completed": False,
                          "static_error_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
