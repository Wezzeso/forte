"""Compare per-job concurrency on generated images after genuine warm readiness.

Never opens supplied video or accepts existing media. JSON output contains only
configuration, timings, token totals and validation booleans. This is a synthetic
serving benchmark, not event accuracy or whole-video latency evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import time

from PIL import Image, ImageDraw
from pydantic import ValidationError

from forte.decision import DecisionConfig, DecisionEngine
from forte.model import LocalModelConfig, LocalModelProvider
from forte.shared.schemas import ClipRequest, FactDefinition, FrameRef
from serving.scheduler import FairScheduler, ScheduledProvider


def make_clips(root: Path, *, seed: int, count: int, frames: int, side: int,
               facts: list[FactDefinition]) -> list[ClipRequest]:
    job_id = f"synthetic-{seed}"
    directory = root / job_id
    directory.mkdir()
    clips = []
    for index in range(count):
        references = []
        for frame_index in range(frames):
            image = Image.new("RGB", (side, side), ((seed * 17 + index * 53 + frame_index) % 256, 100, 180))
            ImageDraw.Draw(image).rectangle((50 + frame_index * 3, 50 + index,
                                             200 + frame_index * 3, 160 + index), fill=(220, 10, 10))
            path = directory / f"c{index}-f{frame_index}.jpg"
            image.save(path, format="JPEG", quality=95)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            # Frame identity includes source position even for equal pixels.
            identity = hashlib.sha256(f"{job_id}:{index}:{frame_index}".encode()).hexdigest()
            references.append(FrameRef(frame_id=identity, job_id=job_id,
                pts_sec=float(frame_index), path=path, width=side, height=side,
                jpeg_sha256=digest))
        clips.append(ClipRequest(job_id=job_id, clip_id=f"c{index}", frame_refs=references,
            fact_definitions=facts))
    return clips


async def benchmark(args) -> dict:
    records = []
    root = Path(args.media_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.timeout
    if args.production_facts:
        from rules.catalog import load_catalog
        facts = load_catalog().fact_definitions()
    else:
        facts = [FactDefinition(fact_id=f"SYNTHETIC_RED_MARKER_{index}",
                    description="A red rectangular marker is visibly present in the supplied synthetic images.")
                 for index in range(args.facts)]
    compact_modes = (False, True) if args.compact_mode == "both" else (args.compact_mode == "compact",)
    with tempfile.TemporaryDirectory(dir=root, prefix="synthetic_concurrency_") as temporary:
        owned = Path(temporary)
        async with LocalModelProvider(LocalModelConfig(media_root=owned, base_url=args.base_url)) as raw:
            await raw.check_ready(deadline=min(deadline, time.monotonic() + 20))
            for trial in range(args.trials):
                for compact in compact_modes:
                    for per_job in (1, 2):
                        clips = make_clips(owned, seed=trial * 100 + per_job * 10 + int(compact),
                            count=args.requests, frames=args.frames, side=args.side, facts=facts)
                        async with FairScheduler(concurrency=2, max_inflight_per_job=per_job) as scheduler:
                            provider = ScheduledProvider(raw, scheduler)
                            engine = DecisionEngine(provider, reviewed_event_ids=(),
                                config=DecisionConfig(compact_frame_ids=compact))
                            durations = []
                            tokens = []

                            async def one(clip):
                                started = time.perf_counter()
                                batch = await engine.analyze(clip, deadline=deadline)
                                durations.append(time.perf_counter() - started)
                                tokens.append((batch.metrics.get("prompt_tokens"), batch.metrics.get("generated_tokens")))

                            started = time.perf_counter()
                            results = await asyncio.gather(*(one(clip) for clip in clips), return_exceptions=True)
                            elapsed = time.perf_counter() - started
                            failures = [value for value in results if isinstance(value, BaseException)]
                            completed = len(results) - len(failures)
                            records.append({"trial": trial, "compact_frame_ids": compact,
                                "max_inflight_per_job": per_job,
                                "requests": args.requests, "completed_requests": completed,
                                "failure_types": sorted({type(value).__name__ for value in failures}),
                                "elapsed_seconds": elapsed,
                                "requests_per_second": completed / elapsed,
                                "request_latency_seconds": durations,
                                "prompt_tokens_total": sum(pair[0] or 0 for pair in tokens),
                                "generated_tokens_total": sum(pair[1] or 0 for pair in tokens),
                                "all_schema_and_semantic_validation_passed": not failures})
    medians = {f"compact={compact},inflight={limit}": statistics.median(row["requests_per_second"] for row in records
                  if row["max_inflight_per_job"] == limit and row["compact_frame_ids"] == compact)
               for compact in compact_modes for limit in (1, 2)}
    return {"synthetic_only": True, "whole_video_latency_measured": False,
            "event_accuracy_measured": False, "frames_per_request": args.frames,
            "fact_count": len(facts), "production_fact_definitions": args.production_facts,
            "image_side": args.side, "replica_concurrency": 2,
            "records": records, "median_requests_per_second": medians}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--media-root", default="/tmp/forte-model-benchmark")
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--side", type=int, default=672)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--facts", type=int, default=19)
    parser.add_argument("--production-facts", action="store_true")
    parser.add_argument("--compact-mode", choices=("both", "raw", "compact"), default="both")
    args = parser.parse_args()
    if not (1 <= args.requests <= 8 and 1 <= args.frames <= 16 and 32 <= args.side <= 1024
            and 1 <= args.trials <= 3 and 1 <= args.facts <= 32 and 0 < args.timeout <= 600):
        parser.error("synthetic benchmark limits are invalid")
    try:
        print(json.dumps(asyncio.run(benchmark(args))))
        return 0
    except Exception as error:
        diagnostic = {"synthetic_only": True, "benchmark_completed": False,
                      "error_type": type(error).__name__}
        if isinstance(error, ValidationError):
            diagnostic["validation_failures"] = [
                {"location": list(item["loc"]), "type": item["type"]}
                for item in error.errors(include_input=False, include_context=False)[:5]
            ]
        print(json.dumps(diagnostic))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
